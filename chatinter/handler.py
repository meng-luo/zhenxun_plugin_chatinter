"""
ChatInter - 主处理器

实现消息处理流程，支持多模态输入（图片识别）。
使用 UniMessage 统一处理消息。
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
import uuid

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger
from zhenxun.utils.message import MessageUtils

from .addressee_resolver import AddresseeResult, resolve_addressee
from .chat_dialogue_planner import plan_chat_dialogue
from .chat_handler import (
    normalize_ai_reply_text,
    replace_mention_ids_with_names,
    reroute_to_plugin_with_result,
)
from .config import get_config_value, get_model_name
from .context_packer import DialogueContextPack
from .event_context import ChatInterEventContext, build_event_context
from .event_runtime import (
    apply_runtime_plugin_overrides,
    event_adapter_name,
    event_is_private,
    event_type_name,
    get_nickname,
    is_already_handled,
    mark_as_handled,
    resolve_superuser,
)
from .execution_observer import (
    EXECUTION_REASON_CANCELLED,
    EXECUTION_REASON_ERROR,
    EXECUTION_REASON_REROUTE_FAILED,
    EXECUTION_REASON_ROUTE_SUCCESS,
    ExecutionObservation,
    record_execution_observation,
    start_execution_observation,
)
from .feedback import FeedbackStore
from .feedback_keys import (
    FEEDBACK_REASON_MISSING_PARAMS as _FEEDBACK_REASON_MISSING_PARAMS,
)
from .feedback_keys import (
    FEEDBACK_REASON_REROUTE_FAILED as _FEEDBACK_REASON_REROUTE_FAILED,
)
from .feedback_keys import (
    FEEDBACK_REASON_ROUTE_SUCCESS as _FEEDBACK_REASON_ROUTE_SUCCESS,
)
from .intent_classifier import classify_message_intent
from .intervention_router import InterventionDecision, decide_intervention
from .main_request import MainRequestResult, run_chatinter_main_request
from .memory import _chat_memory
from .memory_writer import MemoryWriteContext, MemoryWriter
from .middleware import TurnMiddlewareState, get_middleware_manager
from .native_executor import NativeToolExecutionResult, NativeValidatedRoute
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
)
from .person_registry import (
    PersonProfile,
    get_person_profile,
    resolve_relevant_people,
    upsert_seen_person,
)
from .plugin_registry import (
    PluginRegistry,
    PluginSelectionContext,
    get_user_plugin_knowledge,
)
from .route_execution import (
    RouteExecutionPlan,
    apply_command_plan_to_route_result,
    build_planner_followup_message,
    build_reply_image_segments_for_reroute,
    build_route_message_with_explicit_context,
    build_target_modules,
    collect_target_capable_command_heads,
    extract_at_tokens,
    extract_image_tokens,
    extract_reply_sender_id,
    plan_route_command,
    planner_missing_contains,
    prepare_route_execution_plan,
    select_adapter_policy_for_message,
)
from .route_text import (
    ROUTE_ACTION_WORDS,
    contains_any,
    is_usage_question,
    normalize_message_text,
    should_force_knowledge_refresh,
)
from .target_context import (
    append_mention_context_xml,
    build_mention_name_map,
    build_mention_profiles,
    enrich_route_message_with_fuzzy_target,
    extract_pending_entities,
    needs_target_for_route,
    remember_target_resolution,
)
from .thread_resolver import ThreadContext, resolve_thread_context
from .thread_store import record_thread_message
from .trace import StageTrace
from .turn_frame import PipelineStage, TurnFrame
from .turn_metrics import (
    build_turn_metrics_snapshot,
    emit_turn_metrics,
    record_route_observation,
)
from .turn_output import ChannelName, TurnChannelEnvelope, log_turn_channels
from .turn_runtime import TurnBudgetController
from .utils.multimodal import extract_images_from_message
from .utils.unimsg_utils import remove_reply_segment, uni_to_text_with_tags

if TYPE_CHECKING:
    pass

_INTENT_REFRESH_PUNCTUATION = ("。", "！", "？", "；", ";")
_KNOWLEDGE_REFRESH_COOLDOWN = 30.0
_last_knowledge_refresh_ts = 0.0
_SOFT_INVOKE_PREFIXES = (
    "请你",
    "麻烦你",
    "请帮我",
    "麻烦帮我",
    "请给我",
    "麻烦给我",
    "能不能帮我",
    "能否帮我",
    "可以帮我",
    "你帮我",
    "帮我",
    "给我",
    "替我",
)
_EXECUTION_INTENT_HINTS = (
    "帮我",
    "帮忙",
    "请",
    "麻烦",
    "执行",
    "调用",
    "使用",
    "打开",
    "关闭",
    "开启",
    "禁用",
    "设置",
    "查看",
    "看看",
    "看下",
    "查询",
    "生成",
    "制作",
    "发送",
    "来个",
    "来一个",
    "来一张",
    "做个",
    "做一个",
    "做一张",
    "再来个",
    "再来一个",
    "再来一张",
)
_ROUTE_META_CHAT_HINTS = (
    "刚有人说",
    "有人说了",
    "我觉得挺有意思",
    "只是提到",
    "不是在让你执行",
    "不是让你执行",
)
_GROUP_MEMBER_PROFILE_CACHE_TTL = 90.0
_GROUP_MEMBER_PROFILE_CACHE_MAX = 256
_GROUP_MEMBER_PROFILE_CACHE: dict[
    str, tuple[float, list[dict[str, str | tuple[str, ...]]]]
] = {}
_GROUP_ACTIVE_RANK_CACHE_TTL = 30.0
_GROUP_ACTIVE_RANK_CACHE_MAX = 256
_GROUP_ACTIVE_RANK_CACHE: dict[str, tuple[float, dict[str, float]]] = {}
_NICKNAME_RESOLUTION_MEMORY_TTL = 12 * 3600.0
_NICKNAME_RESOLUTION_MEMORY_MAX = 2048
_NICKNAME_RESOLUTION_MEMORY: dict[str, tuple[float, str]] = {}




async def _persist_message_timeline(
    *,
    main_result: MainRequestResult,
    user_id: str,
    group_id: str | None,
    nickname: str,
    user_message,
    bot_id: str | None,
    event_context: ChatInterEventContext | None = None,
    thread_context: ThreadContext | None = None,
) -> None:
    timeline = [item.to_dict() for item in main_result.timeline]
    if not timeline:
        return
    user_text = uni_to_text_with_tags(user_message)
    response_summary = (
        main_result.output.memory_text or main_result.output.final_text or ""
    ).strip()
    dialog = await _chat_memory.add_timeline(
        user_id=user_id,
        group_id=group_id,
        nickname=nickname,
        user_message=user_message,
        response_summary=response_summary,
        timeline=timeline,
        bot_id=bot_id,
    )
    if event_context is not None and thread_context is not None:
        pending_entities = tuple(
            dict.fromkeys(
                (
                    *thread_context.pending_entities,
                    *extract_pending_entities(user_text),
                )
            )
        )
        await record_thread_message(
            thread_id=thread_context.thread_id,
            group_id=group_id,
            message_id=event_context.event_id,
            dialog_id=int(getattr(dialog, "id", 0) or 0) if dialog else None,
            user_id=user_id,
            participants=thread_context.participants,
            topic_key=thread_context.topic_key,
            source=thread_context.source,
            confidence=thread_context.confidence,
            message_text=user_text,
            pending_entities=pending_entities,
            entity_hints=thread_context.entity_hints,
        )
    await MemoryWriter.write_from_dialog(
        MemoryWriteContext(
            session_id=_chat_memory.get_session_id(user_id, group_id),
            user_id=str(user_id),
            group_id=str(group_id) if group_id else None,
            message_text=user_text,
            response_text=response_summary,
            source_dialog_id=int(getattr(dialog, "id", 0) or 0) if dialog else None,
            thread_id=thread_context.thread_id if thread_context is not None else None,
            topic_key=thread_context.topic_key if thread_context is not None else "",
            participants=thread_context.participants
            if thread_context is not None
            else (),
        )
    )


async def _build_dialogue_context_pack(
    *,
    event_context: ChatInterEventContext,
    mention_profiles: dict[str, dict[str, str]] | None = None,
) -> tuple[
    DialogueContextPack,
    PersonProfile | None,
    AddresseeResult,
    ThreadContext,
    InterventionDecision,
]:
    speaker_profile = await get_person_profile(
        user_id=event_context.user_id,
        group_id=event_context.group_id,
        fallback_name=event_context.nickname,
    )
    await upsert_seen_person(
        user_id=event_context.user_id,
        group_id=event_context.group_id,
        nickname=event_context.nickname,
    )
    addressee = await resolve_addressee(
        event_context=event_context,
        bot_names=(BotConfig.self_nickname or "",),
        mention_profiles=mention_profiles,
        speaker_profile=speaker_profile,
    )
    thread = await resolve_thread_context(
        event_context=event_context,
        addressee=addressee,
    )
    relevant_people = await resolve_relevant_people(
        group_id=event_context.group_id,
        message_text=event_context.message_text_with_tags,
        speaker_profile=speaker_profile,
        bot_id=event_context.bot_id,
        mention_user_ids=tuple(event_context.mentioned_user_ids),
        reply_sender_id=event_context.reply.sender_id
        if event_context.reply is not None
        else None,
        thread_user_ids=thread.participants,
        entity_hints=thread.pending_entities,
    )
    current_pending_entities = extract_pending_entities(
        event_context.message_text_with_tags
    )
    if current_pending_entities:
        thread = ThreadContext(
            thread_id=thread.thread_id,
            source=thread.source,
            confidence=thread.confidence,
            related_user_ids=thread.related_user_ids,
            topic_key=thread.topic_key,
            pending_entities=tuple(
                dict.fromkeys((*thread.pending_entities, *current_pending_entities))
            )[:8],
            entity_hints=tuple(dict.fromkeys((*thread.entity_hints, "identity_query")))[
                :8
            ],
        )
    route_signal = (
        contains_any(event_context.normalized_text, ROUTE_ACTION_WORDS)
        or bool(event_context.mentions)
        or bool(event_context.images)
    )
    intervention = decide_intervention(
        event_context=event_context,
        addressee=addressee,
        route_signal=route_signal,
    )
    pack = DialogueContextPack(
        event_context=event_context,
        speaker_profile=speaker_profile,
        addressee=addressee,
        thread=thread,
        relevant_people=relevant_people,
    )
    return pack, speaker_profile, addressee, thread, intervention




def _finish_trace(
    *,
    trace: StageTrace,
    user_id: str,
    group_id: str | None,
    message_preview: str,
    route_report: NativeRouteReport | None,
    budget_controller: TurnBudgetController | None = None,
) -> None:
    total_seconds = trace.finish()
    emit_turn_metrics(
        build_turn_metrics_snapshot(
            trace=trace,
            total_seconds=total_seconds,
            route_report=route_report,
            budget_controller=budget_controller,
        )
    )
    record_route_observation(
        user_id=user_id,
        group_id=group_id,
        message_preview=message_preview,
        trace_tags=dict(trace.tags),
        route_report=route_report,
    )


def _tag_execution_observation(
    trace: StageTrace,
    observation: ExecutionObservation,
) -> None:
    trace.update_tags(
        exec_action=observation.action,
        exec_success=int(observation.success),
        exec_reason=observation.reason,
        exec_latency_ms=observation.latency_ms,
    )


def _route_report_value(
    route_report: NativeRouteReport | None,
    name: str,
    default: object = 0,
):
    if route_report is None:
        return default
    return getattr(route_report, name, default)


def _route_report_observer_kwargs(
    route_report: NativeRouteReport | None,
) -> dict[str, object]:
    return {
        "candidate_total": _route_report_value(route_report, "candidate_total", 0),
        "tool_candidates": _route_report_value(route_report, "tool_candidates", 0),
    }


def _append_route_notice(context_xml: str, notice: str) -> str:
    text = normalize_message_text(notice)
    if not text:
        return context_xml
    section = f"<route_notice>{text}</route_notice>"
    if section in str(context_xml or ""):
        return context_xml
    return f"{context_xml}\n{section}".strip()




async def _execute_native_tool_route(
    *,
    bot: Bot,
    event: Event,
    trace: StageTrace,
    validated: NativeValidatedRoute,
    knowledge_plugins,
    current_message: str,
    user_id: str,
    session_id: str | None,
    has_reply: bool,
    extra_image_segments: list | None,
    route_report: NativeRouteReport,
) -> NativeToolExecutionResult:
    route_result = validated.route_result
    if route_result is None:
        return NativeToolExecutionResult(
            success=False,
            route_result=None,
            output={
                "ok": False,
                "status": "failed",
                "error_type": "InvalidRoute",
                "message": "工具调用没有生成有效插件路由。",
                "is_retryable": True,
            },
            reason="invalid route",
        )

    task_frame = validated.task_frame
    task_message = (
        task_frame.effective_text
        if task_frame is not None and task_frame.effective_text
        else current_message
    )
    task_image_tokens = extract_image_tokens(task_message)
    planned_image_count = len(task_image_tokens)
    if task_message != current_message:
        for token in extract_image_tokens(current_message):
            planned_image_count += 0 if token in task_image_tokens else 1
    if extra_image_segments:
        planned_image_count += len(extra_image_segments)
    command_plan = plan_route_command(
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        current_message=task_message,
        ambient_message=current_message,
        has_reply=has_reply,
        image_count=planned_image_count,
    )
    route_result = apply_command_plan_to_route_result(route_result, command_plan)
    decision = route_result.decision
    target_modules = build_target_modules(route_result, knowledge_plugins)
    execution_plan = prepare_route_execution_plan(
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        current_message=task_message,
        ambient_message=current_message,
        user_id=user_id,
    )
    if not execution_plan.need_followup and command_plan.action == "clarify":
        execution_plan = RouteExecutionPlan(
            command=command_plan.final_command or decision.command,
            need_followup=True,
            followup_message=build_planner_followup_message(command_plan.missing),
            feedback_reason=_FEEDBACK_REASON_MISSING_PARAMS,
            image_missing=1
            if planner_missing_contains(command_plan.missing, {"image", "图片"})
            else 0,
            text_missing=1
            if planner_missing_contains(
                command_plan.missing,
                {"text", "文本", "文字", "参数", "内容"},
            )
            else 0,
        )
    if execution_plan.need_followup:
        return NativeToolExecutionResult(
            success=False,
            route_result=route_result,
            route_command=execution_plan.command or decision.command,
            output={
                "ok": False,
                "status": "failed",
                "error_type": "MissingContext",
                "message": execution_plan.followup_message or "缺少必要参数或上下文。",
                "missing": list(route_result.missing),
                "is_retryable": True,
            },
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
        timeout=float(get_config_value("NATIVE_REROUTE_TIMEOUT", 10) or 10),
    )
    if reroute_result.success:
        observation = execution_frame.finish(
            success=True,
            reason=EXECUTION_REASON_ROUTE_SUCCESS,
        )
        feedback_reason = _FEEDBACK_REASON_ROUTE_SUCCESS
    else:
        observation = execution_frame.finish(
            success=False,
            reason=EXECUTION_REASON_REROUTE_FAILED,
        )
        feedback_reason = _FEEDBACK_REASON_REROUTE_FAILED
    _tag_execution_observation(trace, observation)
    await FeedbackStore.record_plugin_outcome(
        session_id=session_id,
        message_text=task_message,
        route_result=route_result,
        modules=target_modules,
        route_command=route_command,
        success=reroute_result.success,
        reason=feedback_reason,
    )

    output_texts = [item.text for item in reroute_result.outputs if item.text]
    payload = {
        "ok": reroute_result.success,
        "status": "success" if reroute_result.success else "failed",
        "plugin": decision.plugin_name,
        "plugin_module": decision.plugin_module,
        "command": route_command,
        "command_id": route_result.command_id,
        "trace_id": reroute_result.trace_id,
        "observed_output": bool(output_texts),
        "outputs": output_texts[:6],
        "message": reroute_result.error,
        "is_retryable": bool(reroute_result.timed_out or reroute_result.error),
    }
    display_text = (
        "插件已执行，已观测到发送输出。"
        if reroute_result.success and output_texts
        else "插件已执行。"
        if reroute_result.success
        else reroute_result.error or "插件执行失败。"
    )
    return NativeToolExecutionResult(
        success=reroute_result.success,
        route_result=route_result,
        route_command=route_command,
        output=payload,
        display_text=display_text,
        reason=observation.reason,
    )


async def _stage_load_knowledge(
    *,
    frame: TurnFrame,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    await middleware.dispatch("pre_gate", middleware_state)
    FeedbackStore.inspect_user_followup(
        session_id=frame.session_key,
        message_text=frame.raw_message,
    )
    await apply_runtime_plugin_overrides(
        event=event,
        session_key=frame.session_key,
        group_id=frame.group_id,
    )
    frame.knowledge_base = await get_user_plugin_knowledge()
    frame.stage(PipelineStage.KNOWLEDGE)


async def _stage_build_event_dialogue_context(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    session: Uninfo,
    message=None,
    cached_plain_text: str | None = None,
) -> None:
    try:
        event_message = event.get_message()
    except Exception:
        event_message = None
    frame.event_message = event_message

    uni_msg = None
    if message:
        try:
            uni_msg = UniMessage.of(message)
        except Exception:
            pass
    frame.uni_msg = uni_msg

    event_context = build_event_context(
        bot=bot,
        event=event,
        session=session,
        raw_message=frame.raw_message,
        nickname=frame.nickname,
        event_message=event_message,
        uni_msg=uni_msg,
        cached_plain_text=cached_plain_text,
    )
    frame.event_context = event_context
    mention_profiles = await build_mention_profiles(
        frame.group_id,
        event_context.message_text_with_tags,
        bot_id=frame.bot_id,
    )
    frame.mention_profiles = mention_profiles
    (
        dialogue_context_pack,
        _speaker_profile,
        addressee_result,
        thread_context,
        intervention_decision,
    ) = await _build_dialogue_context_pack(
        event_context=event_context,
        mention_profiles=mention_profiles,
    )
    frame.dialogue_context_pack = dialogue_context_pack
    frame.addressee_result = addressee_result
    frame.thread_context = thread_context
    frame.intervention_decision = intervention_decision
    frame.update_tags(
        addressee_source=addressee_result.source,
        addressee_confidence=f"{addressee_result.confidence:.2f}",
        thread_id=thread_context.thread_id,
        intervention=intervention_decision.action,
        intervention_reason=intervention_decision.reason,
    )
    frame.stage(PipelineStage.EVENT_CONTEXT)


async def _stage_build_memory_context(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    (
        chat_system_prompt,
        context_xml,
        reply_images_data,
        history_messages,
    ) = await _chat_memory.build_full_context(
        frame.user_id,
        frame.group_id,
        frame.nickname,
        frame.uni_msg or frame.raw_message,
        bot,
        frame.bot_id,
        event,
        frame.dialogue_context_pack,
    )
    frame.set_context(
        system_prompt=chat_system_prompt,
        context_xml=context_xml,
        reply_images_data=reply_images_data,
        history_messages=history_messages,
    )
    frame.stage(PipelineStage.CONTEXT)


async def _stage_prepare_intent_context(
    *,
    frame: TurnFrame,
    middleware_state: TurnMiddlewareState,
    middleware,
    cached_plain_text: str | None = None,
) -> None:
    if frame.event_message is not None:
        current_message = uni_to_text_with_tags(frame.event_message)
    elif frame.uni_msg:
        current_msg = remove_reply_segment(frame.uni_msg)
        current_message = uni_to_text_with_tags(current_msg)
    elif cached_plain_text:
        current_message = cached_plain_text.strip()
    else:
        current_message = frame.raw_message.strip()
    frame.current_message = current_message

    frame.sync_to_middleware(
        middleware_state,
        phase="intent_routing",
    )
    await middleware.dispatch("before_intent", middleware_state)
    frame.apply_prompt_state(middleware_state)
    frame.stage(PipelineStage.INTENT_BUDGET)


async def _stage_prepare_route_context(
    *,
    frame: TurnFrame,
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
    frame.reply_sender_id = reply_sender_id
    frame.reply_image_count = reply_image_count
    frame.has_reply = bool(reply_sender_id) or reply_image_count > 0
    if reply_image_count > 0:
        logger.debug(f"Reply ?????? {reply_image_count} ?????????")
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
    )
    (
        route_message,
        mention_profiles,
        fuzzy_prompt,
    ) = await enrich_route_message_with_fuzzy_target(
        group_id=frame.group_id,
        original_message=frame.current_message,
        route_message=route_message_base,
        mention_profiles=frame.mention_profiles,
        target_policy=pre_route_target_policy,
        command_heads=command_heads,
    )
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
            "???@????: "
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
        frame.context_xml = _append_route_notice(frame.context_xml, fuzzy_prompt)
        logger.debug("???????????????????????" f"{fuzzy_prompt}")

    if needs_target_for_route(
        frame.current_message,
        route_message,
        target_policy=pre_route_target_policy,
    ):
        frame.set_tag("target_context", "required")
        target_notice = (
            pre_route_target_policy.target_missing_message
            or "需要明确目标后才能调用对应插件。"
        )
        frame.context_xml = _append_route_notice(frame.context_xml, target_notice)
        logger.debug(
            "?????????????????????"
            f"{pre_route_target_policy.target_missing_message or '-'}"
        )

    if route_message != frame.current_message:
        logger.debug(
            "ChatInter ????????"
            f"before='{frame.current_message}' -> after='{route_message}'"
        )
    frame.stage(PipelineStage.ROUTE_PREPARE)


async def _stage_select_route(
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
    selection_context = PluginSelectionContext(
        query=route_message,
        session_id=frame.session_key,
        user_id=frame.user_id,
        group_id=frame.group_id,
        is_superuser=frame.is_superuser,
        event_type=event_type_name(event),
        adapter=event_adapter_name(bot),
        is_private=event_is_private(event),
        has_image=bool(extract_image_tokens(route_message)),
        has_at=bool(extract_at_tokens(route_message)),
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
                    "???????????????????????" f"{len(knowledge_base.plugins)} ???"
                )

    command_tools = PluginRegistry.build_command_tool_snapshots(
        knowledge_base,
        selection_context=selection_context,
    )
    frame.command_tools = command_tools

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


async def _stage_prepare_main_request(
    *,
    frame: TurnFrame,
    message=None,
) -> None:
    source_for_media = (
        frame.event_message or frame.uni_msg or message or frame.raw_message
    )
    image_parts = await extract_images_from_message(source_for_media)
    frame.image_parts = image_parts
    if image_parts:
        logger.debug(f"??????? {len(image_parts)} ???")

    if frame.reply_images_data:
        from .utils.multimodal import _process_image_segment

        for img_seg in frame.reply_images_data:
            image_part = await _process_image_segment(img_seg)
            if image_part:
                image_parts.append(image_part)
        frame.image_parts = image_parts
        if frame.reply_images_data:
            logger.debug(f"?????? {len(frame.reply_images_data)} ???")
    frame.stage(PipelineStage.MEDIA)
    frame.enriched_context_xml = frame.context_xml
    if frame.intent_profile is None:
        raise RuntimeError("missing intent profile for main request")
    dialogue_plan = plan_chat_dialogue(
        message_text=frame.current_message,
        intent=frame.intent_profile,
        has_images=bool(image_parts),
        has_reply=frame.has_reply,
    )
    frame.dialogue_plan = dialogue_plan
    frame.update_tags(
        chat_kind=dialogue_plan.kind,
        chat_style=dialogue_plan.style,
        chat_reason=dialogue_plan.reason,
    )


async def _stage_run_main_request(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
    finalize_callback,
) -> None:
    if frame.dialogue_plan is None:
        raise RuntimeError("missing dialogue plan for main request")

    knowledge_base = frame.knowledge_base
    if knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")

    frame.context_xml = frame.enriched_context_xml
    frame.sync_to_middleware(
        middleware_state,
        phase=PipelineStage.MAIN_REQUEST.value,
    )
    await middleware.dispatch("before_chat", middleware_state)
    frame.apply_prompt_state(middleware_state)
    frame.enriched_context_xml = frame.context_xml
    frame.stage(PipelineStage.MAIN_REQUEST)

    async def _execute_native_route_callback(
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
            session_id=frame.session_key,
            has_reply=frame.has_reply,
            extra_image_segments=frame.reply_image_segments_for_reroute,
            route_report=report,
        )

    async def _route_completed_callback(main_result) -> None:
        frame.set_native_route(
            native_decision=main_result.decision,
            route_result=main_result.route_result,
            route_report=main_result.report,
        )
        frame.update_tags(
            native_action=main_result.decision.action,
            native_confidence=f"{main_result.decision.confidence:.2f}",
            native_reason=main_result.decision.reason or "",
            native_plugin=main_result.route_result.decision.plugin_module
            if main_result.route_result
            else "",
            native_command=main_result.route_result.decision.command
            if main_result.route_result
            else "",
            route_reason=main_result.report.final_reason,
            route_candidates=main_result.report.candidate_total,
            route_attempts=main_result.report.attempts,
            route_tool_candidates=main_result.report.tool_candidates,
            route_tool_choices=main_result.report.tool_choice_count,
        )
        middleware_state.metadata = {
            "phase": "route_completed",
            "native_action": main_result.decision.action,
            "route_reason": main_result.report.final_reason,
        }
        await middleware.dispatch("after_route", middleware_state)

    async def _reply_hook(reply_text: str) -> str:
        finalized_text = normalize_ai_reply_text(reply_text)
        finalized_text = replace_mention_ids_with_names(
            finalized_text,
            frame.mention_name_map,
        )
        middleware_state.response_text = finalized_text
        await middleware.dispatch("after_chat", middleware_state)
        finalized_text = normalize_ai_reply_text(
            middleware_state.response_text or finalized_text
        )
        return replace_mention_ids_with_names(finalized_text, frame.mention_name_map)

    main_result = await run_chatinter_main_request(
        frame.route_message or frame.current_message,
        knowledge_base,
        session_key=frame.session_key,
        budget_controller=frame.budget_controller,
        has_reply=frame.has_reply,
        command_tools=frame.command_tools,
        system_prompt=frame.system_prompt,
        context_xml=frame.context_xml,
        history_messages=frame.history_messages,
        image_parts=frame.image_parts,
        dialogue_plan=frame.dialogue_plan,
        route_executor=_execute_native_route_callback,
        route_completed_hook=_route_completed_callback,
        reply_hook=_reply_hook,
    )

    envelope = TurnChannelEnvelope()
    frame.update_tags(
        path="main_request",
        outcome=main_result.output.outcome,
    )
    envelope.add(ChannelName.ANALYSIS, main_result.output.analysis)
    if not main_result.output.should_send:
        log_turn_channels(envelope)
        await _persist_message_timeline(
            main_result=main_result,
            user_id=frame.user_id,
            group_id=frame.group_id,
            nickname=frame.nickname,
            user_message=frame.uni_msg or frame.current_message,
            bot_id=frame.bot_id,
            event_context=frame.event_context,
            thread_context=frame.thread_context,
        )
        frame.stage(PipelineStage.PERSIST)
        await finalize_callback(phase="post_gate:main_request")
        _finish_trace(
            trace=frame.trace,
            user_id=frame.user_id,
            group_id=frame.group_id,
            message_preview=frame.current_message,
            route_report=frame.route_report,
            budget_controller=frame.budget_controller,
        )
        return

    reply_text = main_result.output.final_text
    if not normalize_message_text(reply_text):
        reply_text = "我暂时没想好怎么回答你。"

    chat_execution_frame = start_execution_observation(
        action="chat",
        route_stage="main_request",
        session_id=frame.session_key,
        message_preview=frame.current_message,
        **_route_report_observer_kwargs(frame.route_report),
    )
    envelope.add(ChannelName.FINAL, reply_text)
    log_turn_channels(envelope)
    await _persist_message_timeline(
        main_result=main_result,
        user_id=frame.user_id,
        group_id=frame.group_id,
        nickname=frame.nickname,
        user_message=frame.uni_msg or frame.current_message,
        bot_id=frame.bot_id,
        event_context=frame.event_context,
        thread_context=frame.thread_context,
    )
    frame.stage(PipelineStage.PERSIST)
    await MessageUtils.build_message(envelope.final).send()
    frame.stage(PipelineStage.SEND)
    await finalize_callback(
        response_text=envelope.final,
        phase="post_gate:main_request",
    )
    if main_result.output.record_chat_feedback:
        FeedbackStore.record_chat(
            session_id=frame.session_key,
            kind=main_result.output.feedback_kind,
            message_text=frame.current_message,
            reply_text=envelope.final,
            weight=0.2,
        )
    _tag_execution_observation(
        frame.trace,
        chat_execution_frame.finish(
            success=True,
            reason=main_result.output.observation_reason,
        ),
    )
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )


async def handle_fallback(
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message=None,
    route_modules: set[str] | None = None,
    cached_plain_text: str | None = None,
) -> None:
    """?????

    ??????????????? AI ??????????

    ??:
        bot: Bot ??
        event: ????
        session: Uninfo ????
        raw_message: ??????
        message: ??????????

    ??:
        bool: ??????
    """
    if not get_config_value("ENABLE_FALLBACK", True):
        logger.debug("ChatInter ?????")
        return

    if is_already_handled(event):
        logger.debug("?????????")
        return

    if route_modules:
        logger.debug("??????????? ChatInter fallback")
        return

    user_id = str(session.user.id)
    group_id = str(session.group.id) if session.group else None
    frame = TurnFrame.create(
        raw_message=raw_message,
        user_id=user_id,
        group_id=group_id,
        nickname=get_nickname(session),
        bot_id=str(bot.self_id) if hasattr(bot, "self_id") else None,
        model_name=get_model_name(),
        is_superuser=resolve_superuser(bot, user_id),
        message_id=str(getattr(event, "message_id", "")),
    )
    mark_as_handled(event)
    middleware = get_middleware_manager()
    middleware_state = frame.create_middleware_state()

    async def _dispatch_post_gate(
        *,
        response_text: str | None = None,
        phase: str = "post_gate",
    ) -> None:
        if frame.post_gate_dispatched:
            return
        if response_text is not None:
            middleware_state.response_text = response_text
        middleware_state.metadata = {
            **middleware_state.metadata,
            "phase": phase,
        }
        await middleware.dispatch("post_gate", middleware_state)
        frame.post_gate_dispatched = True

    try:
        await _stage_load_knowledge(
            frame=frame,
            event=event,
            middleware_state=middleware_state,
            middleware=middleware,
        )
        await _stage_build_event_dialogue_context(
            frame=frame,
            bot=bot,
            event=event,
            session=session,
            message=message,
            cached_plain_text=cached_plain_text,
        )
        await _stage_build_memory_context(frame=frame, bot=bot, event=event)
        await _stage_prepare_intent_context(
            frame=frame,
            middleware_state=middleware_state,
            middleware=middleware,
            cached_plain_text=cached_plain_text,
        )
        await _stage_prepare_route_context(frame=frame, event=event)
        await _stage_select_route(
            frame=frame,
            bot=bot,
            event=event,
            middleware_state=middleware_state,
            middleware=middleware,
        )
        await _stage_prepare_main_request(frame=frame, message=message)
        await _stage_run_main_request(
            frame=frame,
            bot=bot,
            event=event,
            middleware_state=middleware_state,
            middleware=middleware,
            finalize_callback=_dispatch_post_gate,
        )
        return

    except asyncio.CancelledError:
        frame.update_tags(path="cancelled", outcome="cancelled")
        _tag_execution_observation(
            frame.trace,
            record_execution_observation(
                action="chat",
                success=False,
                reason=EXECUTION_REASON_CANCELLED,
                session_id=frame.session_key,
                message_preview=frame.current_message,
            ),
        )
        group_name = frame.group_id or "private"
        logger.debug(f"ChatInter ?????????: user={frame.user_id}, group={group_name}")
        await _dispatch_post_gate(phase="post_gate:cancelled")
        _finish_trace(
            trace=frame.trace,
            user_id=frame.user_id,
            group_id=frame.group_id,
            message_preview=frame.current_message,
            route_report=frame.route_report,
            budget_controller=frame.budget_controller,
        )
        return
    except Exception as e:
        frame.update_tags(path="error", outcome="error")
        _tag_execution_observation(
            frame.trace,
            record_execution_observation(
                action="chat",
                success=False,
                reason=EXECUTION_REASON_ERROR,
                session_id=frame.session_key,
                message_preview=frame.current_message,
            ),
        )
        middleware_state.message_text = frame.current_message
        middleware_state.system_prompt = frame.system_prompt
        middleware_state.context_xml = frame.enriched_context_xml or frame.context_xml
        middleware_state.metadata = {"phase": "error", "error": str(e)}
        await middleware.dispatch("on_error", middleware_state)
        logger.error(f"ChatInter ?????{e}")
        await MessageUtils.build_failure_message().send()
        frame.stage(PipelineStage.ERROR)
        await _dispatch_post_gate(phase="post_gate:error")
        _finish_trace(
            trace=frame.trace,
            user_id=frame.user_id,
            group_id=frame.group_id,
            message_preview=frame.current_message,
            route_report=frame.route_report,
            budget_controller=frame.budget_controller,
        )
        return


__all__ = [
    "handle_fallback",
    "remember_target_resolution",
]
