"""Group plugin retrieval, routing, execution and observation flow."""

from __future__ import annotations

import uuid

from nonebot.adapters import Bot, Event

from zhenxun.services import logger

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
    EXECUTION_REASON_TIMEOUT,
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
from .memory import _chat_memory
from .models.pydantic_models import CommandToolSnapshot
from .native_executor import NativeToolExecutionResult, NativeValidatedRoute
from .native_route import NativeRouteDecision, NativeRouteReport
from .pipeline_stages import (
    _prepare_current_message_context,
    _route_report_observer_kwargs,
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
    build_route_observation,
    build_target_modules,
    extract_image_tokens,
    extract_reply_sender_id,
    prepare_route_execution_plan,
)
from .route_text import (
    is_usage_question,
    normalize_message_text,
)
from .target_context import (
    build_mention_name_map,
)
from .target_resolver import VerifiedActionTarget, resolve_execution_target
from .trace import StageTrace
from .turn_frame import PipelineStage, TurnFrame

_STRUCTURAL_REROUTE_FAILURE_ERRORS = {
    "reroute timeout",
    "unresolved image placeholder",
}

_VISIBLE_OUTPUT_REQUIRED_MODES = {"image", "file"}

_VISIBLE_OUTPUT_REQUIRED_SIDE_EFFECTS = {"query", "send", "mutate"}

_NONREPEATABLE_SIDE_EFFECTS = {"send", "mutate"}


def _missing_error_text(missing: list[str]) -> str:
    """Return a human-readable error message describing which input is missing."""
    if "image" in missing:
        return "该命令需要提供图片才能执行"
    if "at" in missing or "target" in missing:
        return "该命令需要@目标用户才能执行"
    if "reply" in missing:
        return "该命令需要回复某条消息才能执行"
    if "text" in missing:
        return "该命令需要提供文字内容才能执行"
    if missing:
        return "该命令还需要一些额外信息才能执行"
    return ""


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


def _final_side_effect_action_key(
    validated: NativeValidatedRoute,
    route_command: str,
) -> str:
    snapshot = _candidate_tool_snapshot(validated)
    side_effect = normalize_message_text(
        str(getattr(snapshot, "side_effect", "") or "")
    ).casefold()
    if side_effect not in _NONREPEATABLE_SIDE_EFFECTS:
        return ""
    route_result = validated.route_result
    module = normalize_message_text(
        route_result.decision.plugin_module if route_result is not None else ""
    ).casefold()
    command = normalize_message_text(route_command)
    return f"{module}\0{command}" if module and command else ""


def _duplicate_final_action_result(
    *,
    validated: NativeValidatedRoute,
    route_command: str,
    task_message: str,
    ambient_message: str,
) -> NativeToolExecutionResult:
    route_result = validated.route_result
    payload = build_route_observation(
        route_result=route_result,
        ok=False,
        route_command=route_command,
        task_text=task_message,
        ambient_message=ambient_message,
        error="本轮已提交相同的副作用操作，已阻止重复执行。",
        retryable=False,
    )
    payload["status"] = "blocked"
    payload["plugin_execution"] = False
    payload["executed"] = False
    payload["duplicate_blocked"] = True
    return NativeToolExecutionResult(
        success=False,
        route_result=route_result,
        route_command=route_command,
        output=payload,
        display_text="本轮相同操作已提交，已阻止重复执行。",
        reason="duplicate_execution_blocked",
    )


def _reroute_success_from_observation(
    *,
    validated: NativeValidatedRoute,
    reroute_success: bool,
    timed_out: bool = False,
    output_texts: list[str],
    output_artifacts: list[dict[str, object]],
) -> tuple[bool, str, bool]:
    if not reroute_success:
        return False, "", False
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
    if getattr(reroute_result, "timed_out", False):
        return EXECUTION_REASON_TIMEOUT
    if error in _STRUCTURAL_REROUTE_FAILURE_ERRORS:
        return EXECUTION_REASON_REROUTE_FAILED
    if error == EXECUTION_REASON_MISSING_PARAMS:
        return EXECUTION_REASON_MISSING_PARAMS
    if error == "plugin_completed_without_visible_output":
        return EXECUTION_REASON_INVALID_COMMAND
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
    reply_image_count: int,
    route_report: NativeRouteReport,
    mention_profiles: dict[str, dict[str, str]] | None = None,
    submitted_action_keys: set[str] | None = None,
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
        trusted_target_ids=(
            task_frame.trusted_target_ids if task_frame is not None else ()
        ),
        mention_profiles=mention_profiles,
        use_ambient_target_context=bool(
            task_frame is not None and not task_frame.effective_text
        ),
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
                error=_missing_error_text(["target"]),
                missing=["target"],
                retryable=False,
            ),
            display_text="",
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
        ambient_message=target_resolution.message_text,
        user_id=user_id,
        reply_image_count=reply_image_count,
    )
    if execution_plan.blocked:
        missing = list(route_result.missing)
        if execution_plan.image_missing > 0 and "image" not in missing:
            missing.append("image")
        if execution_plan.text_missing > 0 and "text" not in missing:
            missing.append("text")
        if execution_plan.feedback_reason == _FEEDBACK_REASON_TARGET_REQUIRED:
            if "target" not in missing:
                missing.append("target")
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
                error=_missing_error_text(missing),
                missing=missing,
                retryable=False,
            ),
            display_text="",
            reason=execution_plan.feedback_reason or "",
        )

    route_command = execution_plan.command or decision.command
    action_key = _final_side_effect_action_key(validated, route_command)
    if action_key and submitted_action_keys is not None:
        if action_key in submitted_action_keys:
            return _duplicate_final_action_result(
                validated=validated,
                route_command=route_command,
                task_message=task_message,
                ambient_message=current_message,
            )
        submitted_action_keys.add(action_key)
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
        timed_out=reroute_result.timed_out,
        output_texts=output_texts,
        output_artifacts=output_artifacts,
    )
    execution_uncertain = bool(
        reroute_result.execution_uncertain
        or (reroute_result.execution_started and not observed_success)
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
    _tag_execution_observation(trace, observation)
    if not execution_uncertain:
        await FeedbackStore.record_plugin_outcome(
            session_id=session_id,
            message_text=task_message,
            route_result=route_result,
            modules=target_modules,
            route_command=route_command,
            success=observed_success,
            reason=feedback_reason,
            image_missing=0,
            text_missing=0,
        )

    error_text = (
        "插件执行超时，结果不确定，不得重复执行同一命令。"
        if reroute_result.timed_out
        else "插件执行已开始，但未能确认完整结果，不得自动重复执行。"
        if execution_uncertain and not reroute_result.error
        else observation_error or reroute_result.error or ""
    )
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
        retryable=retryable,
    )
    payload["plugin_execution"] = bool(reroute_result.execution_started)
    payload["executed"] = bool(observed_success)
    if not observed_success:
        payload["failure_stage"] = "native_reroute"
        payload["native_reroute_reason"] = observation.reason
    if target_resolution.resolved_target_ids:
        payload["resolved_target"] = [
            {"user_id": user_id}
            for user_id in target_resolution.resolved_target_ids
        ]
    if execution_uncertain:
        payload["status"] = "uncertain"
        payload["execution_uncertain"] = True
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
        execution_started=bool(reroute_result.execution_started),
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
        source = frame.uni_msg or frame.raw_message
        try:
            _lines, reply_images = await _chat_memory._build_current_message_layers(
                frame.group_id,
                source,
                frame.nickname,
                frame.bot_id,
                bot,
                event,
                reply_context=(
                    frame.event_context.reply
                    if frame.event_context is not None
                    else None
                ),
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
    if frame.knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")

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
    verified_target = getattr(frame, "verified_action_target", None)
    if (
        reply_image_count > 0
        and str(getattr(verified_target, "source", "") or "") == "reply"
    ):
        frame.verified_action_target = VerifiedActionTarget()
    if reply_image_count > 0:
        logger.debug(f"回复上下文包含 {reply_image_count} 张图片")
    if not frame.reply_image_segments_for_reroute:
        frame.reply_image_segments_for_reroute = build_reply_image_segments_for_reroute(
            frame.reply_images_data
        )
    route_message = frame.current_message
    frame.route_message = route_message
    frame.mention_name_map = build_mention_name_map(frame.mention_profiles)
    excluded_target_ids = {
        str(getattr(event_context, "user_id", "") or ""),
        str(getattr(event_context, "bot_id", "") or ""),
    }
    mention_user_ids = tuple(
        dict.fromkeys(
            str(getattr(item, "user_id", "") or "")
            for item in getattr(event_context, "mentions", ()) or ()
            if str(getattr(item, "user_id", "") or "")
            not in excluded_target_ids
        )
    )
    has_at = bool(mention_user_ids)
    has_image = current_image_count > 0 or reply_image_count > 0
    frame.router_context = {
        "has_reply": frame.has_reply,
        "has_image": has_image,
        "has_at": has_at,
        "current_image_count": current_image_count,
        "reply_image_count": reply_image_count,
        "mention_user_ids": mention_user_ids,
        "reply_sender_id": reply_sender_id or "",
    }
    frame.stage(PipelineStage.ROUTE_PREPARE)


async def _select_capability_route(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    knowledge_base = frame.knowledge_base
    if knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")

    frame.stage(PipelineStage.ROUTE_SELECTION)
    route_message = frame.current_message
    frame.route_message = route_message
    current_image_count = int(frame.router_context.get("current_image_count", 0) or 0)
    has_image = bool(frame.router_context.get("has_image", False))
    has_at = bool(frame.router_context.get("has_at", False))
    verified_target = getattr(frame, "verified_action_target", None)
    has_verified_target = bool(getattr(verified_target, "is_resolved", False))
    verified_target_source = str(
        getattr(verified_target, "source", "") or ""
    )
    frame.router_context.update(
        {
            "has_reply": frame.has_reply,
            "has_image": has_image,
            "has_at": has_at,
            "has_verified_target": has_verified_target,
            "verified_target_source": verified_target_source,
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
        has_verified_target=has_verified_target,
        verified_target_source=verified_target_source,
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

    frame.intent_profile = None
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


async def stage_group_capability_hint(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    cached_plain_text: str | None = None,
) -> None:
    frame.knowledge_base = await get_user_plugin_knowledge()
    frame.stage(PipelineStage.KNOWLEDGE)
    await _prepare_current_message_context(
        frame=frame,
        cached_plain_text=cached_plain_text,
    )
    await _prepare_capability_route_context(frame=frame, bot=bot, event=event)
    await _select_capability_route(
        frame=frame,
        bot=bot,
        event=event,
    )
    frame.stage(PipelineStage.CAPABILITY_HINT)


__all__ = [
    "stage_group_capability_hint",
    "stage_route_media_context",
]
