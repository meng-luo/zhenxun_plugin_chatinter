"""Group plugin retrieval, routing, execution and observation flow."""

from __future__ import annotations

import time
from typing import Any
import uuid

from nonebot.adapters import Bot, Event

from zhenxun.services import logger

from .agents.core import PluginCommandRequest
from .agents.plugin_command_agent import PluginCommandAgent
from .chat_handler import (
    artifacts_from_send_observations,
    messages_summary_from_send_observations,
    reroute_to_plugin_with_result,
)
from .config import NATIVE_REROUTE_TIMEOUT_SECONDS
from .event_runtime import event_adapter_name, event_is_private, event_type_name
from .execution_observer import (
    EXECUTION_REASON_ERROR,
    EXECUTION_REASON_INVALID_COMMAND,
    EXECUTION_REASON_MISSING_PARAMS,
    EXECUTION_REASON_REROUTE_FAILED,
    EXECUTION_REASON_ROUTE_SUCCESS,
    start_execution_observation,
)
from .feedback import FeedbackStore
from .feedback_keys import (
    FEEDBACK_REASON_REROUTE_FAILED as _FEEDBACK_REASON_REROUTE_FAILED,
)
from .feedback_keys import (
    FEEDBACK_REASON_ROUTE_SUCCESS as _FEEDBACK_REASON_ROUTE_SUCCESS,
)
from .feedback_keys import (
    FEEDBACK_REASON_TARGET_REQUIRED as _FEEDBACK_REASON_TARGET_REQUIRED,
)
from .intent_classifier import classify_message_intent
from .main_request_models import MainRequestResult
from .memory import _chat_memory
from .middleware import TurnMiddlewareState
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .native_executor import NativeToolExecutionResult, NativeValidatedRoute
from .native_route import NativeRouteDecision, NativeRouteReport
from .pipeline_stages import (
    _build_agent_stage_hooks,
    _prepare_current_message_context,
    _route_report_observer_kwargs,
    _set_agent_stage_result,
    _tag_execution_observation,
)
from .plugin_registry import (
    PluginRegistry,
    PluginSelectionContext,
    get_user_plugin_knowledge,
)
from .route_execution import (
    build_invalid_route_observation,
    build_reply_image_segments_for_reroute,
    build_route_message_with_explicit_context,
    build_route_observation,
    build_target_modules,
    collect_target_capable_command_heads,
    extract_at_tokens,
    extract_image_tokens,
    extract_reply_sender_id,
    prepare_route_execution_plan,
    select_adapter_policy_for_message,
)
from .route_text import (
    is_usage_question,
    normalize_message_text,
    should_force_knowledge_refresh,
)
from .runtime_result import _finalize_result
from .target_context import (
    append_mention_context_xml,
    build_mention_name_map,
    needs_target_for_route,
)
from .target_resolver import resolve_execution_target, resolve_pre_route_target
from .trace import StageTrace
from .turn_frame import PipelineStage, TurnFrame
from .turn_runtime import TurnBudgetController

_KNOWLEDGE_REFRESH_COOLDOWN = 30.0

_last_knowledge_refresh_ts = 0.0

_STRUCTURAL_REROUTE_FAILURE_ERRORS = {
    "reroute timeout",
    "unresolved image placeholder",
}

_PLUGIN_PARAM_REJECTION_MARKERS = {
    "文字数量不符": "text",
    "文本数量不符": "text",
    "图片数量不符": "image",
    "参数数量不符": "params",
}

_VISIBLE_OUTPUT_REQUIRED_MODES = {"image", "file"}

_VISIBLE_OUTPUT_REQUIRED_SIDE_EFFECTS = {"query", "send", "mutate"}

def _candidate_tool_snapshot(
    validated: NativeValidatedRoute,
) -> CommandToolSnapshot | None:
    candidate = getattr(validated, "candidate", None)
    snapshot = getattr(candidate, "tool", None) if candidate is not None else None
    return snapshot if isinstance(snapshot, CommandToolSnapshot) else None

def _route_requires_visible_output(validated: NativeValidatedRoute) -> bool:
    snapshot = _candidate_tool_snapshot(validated)
    if snapshot is None:
        return False
    if not bool(getattr(snapshot, "requires_real_tool", True)):
        return False
    role = normalize_message_text(str(getattr(snapshot, "command_role", "") or ""))
    if role in {"helper", "usage", "catalog"}:
        return False
    output_mode = normalize_message_text(
        str(getattr(snapshot, "output_mode", "") or "")
    )
    side_effect = normalize_message_text(
        str(getattr(snapshot, "side_effect", "") or "")
    )
    if output_mode == "plugin_output":
        return True
    return (
        output_mode in _VISIBLE_OUTPUT_REQUIRED_MODES
        or side_effect in _VISIBLE_OUTPUT_REQUIRED_SIDE_EFFECTS
    )

def _plugin_param_failure_fields(text: str) -> list[str]:
    normalized = normalize_message_text(str(text or ""))
    fields: list[str] = []
    for marker, field in _PLUGIN_PARAM_REJECTION_MARKERS.items():
        if marker in normalized and field not in fields:
            fields.append(field)
    return fields

def _plugin_param_rejection_error(output_texts: list[str]) -> str:
    for item in output_texts:
        text = normalize_message_text(str(item or ""))
        if text and _plugin_param_failure_fields(text):
            return text[:260]
    return ""

def _reroute_success_from_observation(
    *,
    validated: NativeValidatedRoute,
    reroute_success: bool,
    output_texts: list[str],
    output_artifacts: list[dict[str, object]],
) -> tuple[bool, str, bool]:
    if not reroute_success:
        return False, "", True
    param_rejection = _plugin_param_rejection_error(output_texts)
    if param_rejection:
        return False, param_rejection, False
    if not _route_requires_visible_output(validated):
        return True, "", False
    if output_texts or output_artifacts:
        return True, "", False
    return False, "plugin_completed_without_visible_output", True

def _execution_failure_reason(
    *,
    reroute_result,
    observation_error: str,
) -> str:
    error = normalize_message_text(
        observation_error or getattr(reroute_result, "error", "") or ""
    )
    if error in _STRUCTURAL_REROUTE_FAILURE_ERRORS:
        return EXECUTION_REASON_REROUTE_FAILED
    if error == EXECUTION_REASON_MISSING_PARAMS or _plugin_param_failure_fields(error):
        return EXECUTION_REASON_MISSING_PARAMS
    if error == "plugin_completed_without_visible_output":
        return EXECUTION_REASON_INVALID_COMMAND
    if getattr(reroute_result, "timed_out", False):
        return EXECUTION_REASON_REROUTE_FAILED
    if error:
        return EXECUTION_REASON_ERROR
    return EXECUTION_REASON_REROUTE_FAILED

async def _execute_native_tool_route(
    *,
    bot: Bot,
    event: Event,
    trace: StageTrace,
    validated: NativeValidatedRoute,
    knowledge_plugins,
    current_message: str,
    user_id: str,
    group_id: str | None,
    session_id: str | None,
    has_reply: bool,
    extra_image_segments: list | None,
    route_report: NativeRouteReport,
    mention_profiles: dict[str, dict[str, str]] | None = None,
) -> NativeToolExecutionResult:
    route_result = validated.route_result
    if route_result is None:
        return NativeToolExecutionResult(
            success=False,
            route_result=None,
            output=build_invalid_route_observation(
                decision=validated.decision,
                task_text=validated.task_frame.effective_text
                if validated.task_frame is not None
                else normalize_message_text(validated.decision.command or ""),
                ambient_message=current_message,
                error="工具调用没有生成有效插件路由。",
                retryable=True,
            ),
            reason="invalid route",
        )

    task_frame = validated.task_frame
    task_message = (
        task_frame.effective_text
        if task_frame is not None and task_frame.effective_text
        else normalize_message_text(route_result.decision.command or "")
    )
    target_resolution = await resolve_execution_target(
        group_id=group_id,
        bot=bot,
        bot_id=getattr(bot, "self_id", None),
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        task_message=task_message,
        ambient_message=current_message,
        target_hint=task_frame.target_hint if task_frame is not None else "",
        mention_profiles=mention_profiles,
    )
    if target_resolution.blocked:
        return NativeToolExecutionResult(
            success=False,
            route_result=route_result,
            route_command=route_result.decision.command,
            output=build_route_observation(
                route_result=route_result,
                ok=False,
                route_command=route_result.decision.command,
                task_text=task_message,
                ambient_message=current_message,
                error=target_resolution.prompt,
                missing=["target"],
                retryable=True,
            ),
            display_text=target_resolution.prompt,
            reason=_FEEDBACK_REASON_TARGET_REQUIRED,
        )
    if target_resolution.resolved:
        task_message = target_resolution.message_text
    decision = route_result.decision
    target_modules = build_target_modules(route_result, knowledge_plugins)
    execution_plan = prepare_route_execution_plan(
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        current_message=task_message,
        ambient_message=current_message,
        user_id=user_id,
    )
    if execution_plan.need_followup:
        return NativeToolExecutionResult(
            success=False,
            route_result=route_result,
            route_command=execution_plan.command or decision.command,
            output=build_route_observation(
                route_result=route_result,
                ok=False,
                route_command=execution_plan.command or decision.command,
                task_text=task_message,
                ambient_message=current_message,
                error=execution_plan.followup_message or "缺少必要参数或上下文。",
                missing=list(route_result.missing),
                retryable=True,
            ),
            display_text=execution_plan.followup_message or "",
            reason=execution_plan.feedback_reason or "",
        )

    route_command = execution_plan.command or decision.command
    execution_frame = start_execution_observation(
        action="execute",
        plugin_module=decision.plugin_module,
        plugin_name=decision.plugin_name,
        command_id=route_result.command_id,
        command=route_command,
        route_stage=route_result.stage,
        session_id=session_id,
        message_preview=task_message,
        selected_rank=route_result.selected_rank,
        selected_score=route_result.selected_score,
        selected_reason=route_result.selected_reason,
        **_route_report_observer_kwargs(route_report),
    )
    reroute_result = await reroute_to_plugin_with_result(
        bot,
        event,
        route_command,
        target_modules=target_modules,
        extra_image_segments=extra_image_segments,
        trace_id=f"ci-{uuid.uuid4().hex}",
        wait=True,
        timeout=float(NATIVE_REROUTE_TIMEOUT_SECONDS),
    )
    output_texts = messages_summary_from_send_observations(reroute_result.outputs)
    output_artifacts = artifacts_from_send_observations(
        reroute_result.outputs,
        trace_id=reroute_result.trace_id,
    )
    observed_success, observation_error, retryable = _reroute_success_from_observation(
        validated=validated,
        reroute_success=reroute_result.success,
        output_texts=output_texts,
        output_artifacts=output_artifacts,
    )
    if observed_success:
        observation = execution_frame.finish(
            success=True,
            reason=EXECUTION_REASON_ROUTE_SUCCESS,
        )
        feedback_reason = _FEEDBACK_REASON_ROUTE_SUCCESS
    else:
        failure_reason = _execution_failure_reason(
            reroute_result=reroute_result,
            observation_error=observation_error,
        )
        observation = execution_frame.finish(
            success=False,
            reason=failure_reason,
        )
        feedback_reason = (
            _FEEDBACK_REASON_REROUTE_FAILED
            if failure_reason == EXECUTION_REASON_REROUTE_FAILED
            else failure_reason
        )
    param_failure_fields = _plugin_param_failure_fields(observation_error)
    _tag_execution_observation(trace, observation)
    await FeedbackStore.record_plugin_outcome(
        session_id=session_id,
        message_text=task_message,
        route_result=route_result,
        modules=target_modules,
        route_command=route_command,
        success=observed_success,
        reason=feedback_reason,
        image_missing=int("image" in param_failure_fields),
        text_missing=int("text" in param_failure_fields),
    )

    error_text = observation_error or reroute_result.error or ""
    payload = build_route_observation(
        route_result=route_result,
        ok=observed_success,
        route_command=route_command,
        messages_sent=output_texts,
        artifacts=output_artifacts,
        task_text=task_message,
        ambient_message=current_message,
        trace_id=reroute_result.trace_id,
        error=error_text,
        missing=param_failure_fields or None,
        retryable=bool(retryable or reroute_result.timed_out or reroute_result.error),
    )
    display_text = (
        "插件执行完成，结果已发送。"
        if observed_success and (output_texts or output_artifacts)
        else "插件执行完成。"
        if observed_success
        else error_text or "插件执行失败。"
    )
    return NativeToolExecutionResult(
        success=observed_success,
        route_result=route_result,
        route_command=route_command,
        output=payload,
        display_text=display_text,
        reason=observation.reason,
    )

async def stage_route_media_context(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    """Collect only media needed by plugin routing."""

    await _prepare_lightweight_media_reply_context(frame=frame, bot=bot, event=event)
    frame.stage(PipelineStage.MEDIA)

async def _prepare_lightweight_media_reply_context(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    """Prepare current/reply media signals for routing without chat memory."""

    current_image_count = _current_image_count(frame)
    if not frame.reply_images_data:
        source = (
            frame.raw_message
            if frame.turn_messages
            else frame.uni_msg or frame.raw_message
        )
        try:
            _lines, reply_images = await _chat_memory._build_current_message_layers(
                frame.group_id,
                source,
                frame.nickname,
                frame.bot_id,
                bot,
                event,
            )
        except Exception as exc:
            logger.debug(f"[ChatInter] route media context skipped: {exc}")
            reply_images = []
        frame.reply_images_data = list(reply_images or [])
    frame.reply_image_count = len(frame.reply_images_data)
    frame.reply_image_segments_for_reroute = build_reply_image_segments_for_reroute(
        frame.reply_images_data
    )
    frame.router_context.update(
        {
            "current_image_count": current_image_count,
            "reply_image_count": frame.reply_image_count,
        }
    )
    frame.update_tags(
        current_image_count=float(current_image_count),
        reply_image_count=float(frame.reply_image_count),
    )

def _current_image_count(frame: TurnFrame) -> int:
    event_images = getattr(getattr(frame, "event_context", None), "images", []) or []
    return max(len(event_images), len(extract_image_tokens(frame.current_message)))

async def _prepare_capability_route_context(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    knowledge_base = frame.knowledge_base
    if knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")

    command_heads = collect_target_capable_command_heads(knowledge_base)
    event_context = frame.event_context
    reply_sender_id = (
        event_context.reply.sender_id
        if event_context is not None and event_context.reply is not None
        else extract_reply_sender_id(event)
    )
    reply_image_count = len(frame.reply_images_data or [])
    current_image_count = _current_image_count(frame)
    frame.reply_sender_id = reply_sender_id
    frame.reply_image_count = reply_image_count
    frame.has_reply = bool(reply_sender_id) or reply_image_count > 0
    if reply_image_count > 0:
        logger.debug(f"回复上下文包含 {reply_image_count} 张图片")
    if not frame.reply_image_segments_for_reroute:
        frame.reply_image_segments_for_reroute = build_reply_image_segments_for_reroute(
            frame.reply_images_data
        )
    pre_route_target_policy = select_adapter_policy_for_message(
        frame.current_message,
        knowledge_base,
    )
    route_message_base = build_route_message_with_explicit_context(
        message_text=frame.current_message,
        user_id=frame.user_id,
        reply_image_count=reply_image_count,
        reply_sender_id=reply_sender_id,
        target_policy=pre_route_target_policy,
        command_heads=command_heads,
    )
    target_resolution = await resolve_pre_route_target(
        group_id=frame.group_id,
        bot=bot,
        original_message=frame.current_message,
        route_message=route_message_base,
        mention_profiles=frame.mention_profiles,
        target_policy=pre_route_target_policy,
        command_heads=command_heads,
    )
    route_message = target_resolution.message_text
    mention_profiles = target_resolution.mention_profiles
    fuzzy_prompt = target_resolution.prompt
    frame.set_tag("target_resolution", target_resolution.status)
    frame.route_message = route_message
    frame.mention_profiles = mention_profiles
    frame.mention_name_map = build_mention_name_map(mention_profiles)
    if frame.mention_name_map or mention_profiles:
        frame.context_xml = append_mention_context_xml(
            frame.context_xml,
            frame.mention_name_map,
            mention_profiles,
        )
        logger.debug(
            "已解析 @ 目标映射: "
            + ", ".join(
                (
                    f"{mapped_user_id}->{profile.get('display_name')}"
                    + (f"(uid:{profile.get('uid')})" if profile.get("uid") else "")
                )
                for mapped_user_id, profile in mention_profiles.items()
            )
        )
    if fuzzy_prompt:
        frame.set_tag("target_context", "ambiguous")
        logger.debug(f"目标解析需要澄清: {fuzzy_prompt}")

    if needs_target_for_route(
        frame.current_message,
        route_message,
        target_policy=pre_route_target_policy,
    ):
        frame.set_tag("target_context", "required")
        logger.debug(
            "插件调用缺少目标: "
            f"{pre_route_target_policy.target_missing_message or '-'}"
        )

    if route_message != frame.current_message:
        logger.debug(
            "ChatInter 路由消息重写: "
            f"before='{frame.current_message}' -> after='{route_message}'"
        )
    frame.router_context = {
        "has_reply": frame.has_reply,
        "has_image": bool(extract_image_tokens(route_message))
        or current_image_count > 0,
        "has_at": bool(extract_at_tokens(route_message)),
        "current_image_count": current_image_count,
        "reply_image_count": reply_image_count,
        "target_resolution": target_resolution.status,
    }
    frame.stage(PipelineStage.ROUTE_PREPARE)

async def _select_capability_route(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    global _last_knowledge_refresh_ts

    knowledge_base = frame.knowledge_base
    if knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")

    frame.sync_to_middleware(
        middleware_state,
        phase=PipelineStage.ROUTE_SELECTION.value,
        route_message=frame.route_message,
    )
    frame.stage(PipelineStage.ROUTE_SELECTION)
    await middleware.dispatch("before_route", middleware_state)
    route_message = middleware_state.route_message or frame.route_message
    frame.route_message = route_message
    current_image_count = int(frame.router_context.get("current_image_count", 0) or 0)
    has_image = bool(extract_image_tokens(route_message)) or current_image_count > 0
    has_at = bool(extract_at_tokens(route_message))
    frame.router_context.update(
        {
            "has_reply": frame.has_reply,
            "has_image": has_image,
            "has_at": has_at,
            "current_image_count": current_image_count,
            "reply_image_count": frame.reply_image_count,
        }
    )
    selection_context = PluginSelectionContext(
        query=route_message,
        session_id=frame.session_key,
        user_id=frame.user_id,
        group_id=frame.group_id,
        is_superuser=frame.is_superuser,
        event_type=event_type_name(event),
        adapter=event_adapter_name(bot),
        is_private=event_is_private(event),
        has_image=has_image,
        has_at=has_at,
        has_reply=frame.has_reply,
        addressee_user_id=frame.addressee_result.target_user_id
        if frame.addressee_result
        else None,
        addressee_source=frame.addressee_result.source
        if frame.addressee_result
        else "",
        thread_id=frame.thread_context.thread_id if frame.thread_context else "",
        intervention_action=frame.intervention_decision.action
        if frame.intervention_decision
        else "",
    )
    frame.selection_context = selection_context
    knowledge_base = PluginRegistry.filter_knowledge_base(
        knowledge_base,
        selection_context=selection_context,
    )
    frame.knowledge_base = knowledge_base

    if should_force_knowledge_refresh(route_message, knowledge_base):
        now = time.monotonic()
        if now - _last_knowledge_refresh_ts >= _KNOWLEDGE_REFRESH_COOLDOWN:
            _last_knowledge_refresh_ts = now
            refreshed_knowledge = await get_user_plugin_knowledge(force_refresh=True)
            filtered_knowledge = PluginRegistry.filter_knowledge_base(
                refreshed_knowledge,
                selection_context=selection_context,
            )
            if len(filtered_knowledge.plugins) > len(knowledge_base.plugins):
                knowledge_base = filtered_knowledge
                frame.knowledge_base = knowledge_base
                logger.info(
                    f"插件知识库已刷新，候选数: {len(knowledge_base.plugins)}"
                )

    command_tools = (
        PluginRegistry.build_command_tool_snapshots(
            knowledge_base,
            selection_context=selection_context,
        )
        if frame.allow_plugin_tools
        else []
    )
    frame.command_tools = command_tools
    frame.chat_tool_exposure_state = (
        "plugin_tools_exposed" if frame.allow_plugin_tools and command_tools else "none"
    )

    intent_profile = classify_message_intent(route_message, knowledge_base)
    frame.intent_profile = intent_profile
    frame.update_tags(
        intent_kind=intent_profile.kind,
        intent_reason=intent_profile.reason,
    )
    logger.debug(
        "ChatInter intent classify: "
        f"kind={intent_profile.kind} "
        f"reason={intent_profile.reason} "
        f"explicit={intent_profile.explicit_command} "
        f"command={intent_profile.command_head or '-'} "
        f"chat_subkind={getattr(intent_profile, 'chat_subkind', 'general_chat')} "
        f"confidence={intent_profile.confidence:.2f}"
    )
    middleware_state.intent = intent_profile
    middleware_state.route_message = route_message
    middleware_state.metadata = {
        "phase": "after_intent",
        "intent_kind": intent_profile.kind,
        "intent_reason": intent_profile.reason,
    }
    await middleware.dispatch("after_intent", middleware_state)
    frame.apply_prompt_state(middleware_state)
    route_message = middleware_state.route_message or route_message
    frame.route_message = route_message
    frame.stage(PipelineStage.INTENT)

    route_report = NativeRouteReport(helper_mode=is_usage_question(route_message))
    route_report.note_candidate_policy(
        reason="main_request_pending",
        limit=len(command_tools),
    )
    route_report.candidate_total = max(route_report.candidate_total, len(command_tools))
    frame.set_native_route(
        native_decision=NativeRouteDecision(
            action="chat",
            confidence=0.0,
            reason="main_request_pending",
        ),
        route_result=None,
        route_report=route_report,
    )
    frame.update_tags(
        route_reason=route_report.final_reason,
        route_candidates=route_report.candidate_total,
    )

async def _run_plugin_command_agent_turn(
    *,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None,
    route_executor: Any,
    route_completed_hook: Any | None,
    reply_hook: Any | None,
    router_context: dict[str, object] | None,
) -> MainRequestResult:
    normalized_message = normalize_message_text(message_text)
    report = NativeRouteReport(helper_mode=is_usage_question(normalized_message))
    started = time.perf_counter()
    try:
        result = (
            await PluginCommandAgent().run(
                PluginCommandRequest(
                    message_text=normalized_message,
                    knowledge_base=knowledge_base,
                    session_key=session_key,
                    budget_controller=budget_controller,
                    has_reply=has_reply,
                    command_tools=command_tools,
                    route_executor=route_executor,
                    router_context=router_context,
                    report=report,
                )
            )
        ).to_main_result()
        return await _finalize_result(
            result,
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )
    finally:
        if budget_controller is not None:
            budget_controller.record_stage(
                "main_request",
                time.perf_counter() - started,
            )

async def stage_plugin_run(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    knowledge_base = frame.knowledge_base
    if knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")
    frame.stage(PipelineStage.AGENT_RUN)
    route_completed_hook, reply_hook = _build_agent_stage_hooks(
        frame=frame,
        middleware_state=middleware_state,
        middleware=middleware,
    )

    async def execute_native_route(
        validated: NativeValidatedRoute,
        report: NativeRouteReport,
    ) -> NativeToolExecutionResult:
        return await _execute_native_tool_route(
            bot=bot,
            event=event,
            trace=frame.trace,
            validated=validated,
            knowledge_plugins=knowledge_base.plugins,
            current_message=frame.route_message or frame.current_message,
            user_id=frame.user_id,
            group_id=frame.group_id,
            session_id=frame.session_key,
            has_reply=frame.has_reply,
            extra_image_segments=frame.reply_image_segments_for_reroute,
            route_report=report,
            mention_profiles=frame.mention_profiles,
        )

    main_result = await _run_plugin_command_agent_turn(
        message_text=frame.route_message or frame.current_message,
        knowledge_base=knowledge_base,
        session_key=frame.session_key,
        budget_controller=frame.budget_controller,
        has_reply=frame.has_reply,
        command_tools=frame.command_tools,
        route_executor=execute_native_route,
        route_completed_hook=route_completed_hook,
        reply_hook=reply_hook,
        router_context=frame.router_context,
    )
    _set_agent_stage_result(frame=frame, main_result=main_result)
async def stage_group_capability_hint(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
    cached_plain_text: str | None = None,
) -> None:
    frame.knowledge_base = await get_user_plugin_knowledge()
    frame.stage(PipelineStage.KNOWLEDGE)
    await _prepare_current_message_context(
        frame=frame,
        middleware_state=middleware_state,
        middleware=middleware,
        cached_plain_text=cached_plain_text,
    )
    await _prepare_capability_route_context(frame=frame, bot=bot, event=event)
    await _select_capability_route(
        frame=frame,
        bot=bot,
        event=event,
        middleware_state=middleware_state,
        middleware=middleware,
    )
    frame.stage(PipelineStage.CAPABILITY_HINT)


__all__ = [
    "stage_group_capability_hint",
    "stage_plugin_run",
    "stage_route_media_context",
]
