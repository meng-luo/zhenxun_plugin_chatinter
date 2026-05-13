"""
ChatInter - 主处理器

实现消息处理流程，支持多模态输入（图片识别）。
使用 UniMessage 统一处理消息。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger
from zhenxun.utils.message import MessageUtils

from .addressee_resolver import AddresseeResult, resolve_addressee
from .agent_gate import decide_agent_gate
from .agent_runner import run_chatinter_agent
from .chat_dialogue_planner import ChatDialoguePlan, plan_chat_dialogue
from .chat_handler import (
    handle_chat_message,
    normalize_ai_reply_text,
    replace_mention_ids_with_names,
    reroute_to_plugin_with_result,
)
from .chat_quality_guard import refine_chat_reply
from .config import get_config_value, get_mcp_endpoints, get_model_name
from .context_packer import DialogueContextPack
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
from .event_context import ChatInterEventContext, build_event_context
from .execution_observer import (
    EXECUTION_REASON_CANCELLED,
    EXECUTION_REASON_CHAT_COMPLETED,
    EXECUTION_REASON_CHAT_REWRITTEN,
    EXECUTION_REASON_ERROR,
    EXECUTION_REASON_REROUTE_FAILED,
    EXECUTION_REASON_ROUTE_SUCCESS,
    ExecutionObservation,
    record_execution_observation,
    start_execution_observation,
)
from .feedback_keys import (
    FEEDBACK_REASON_MISSING_PARAMS as _FEEDBACK_REASON_MISSING_PARAMS,
)
from .feedback_keys import (
    FEEDBACK_REASON_REROUTE_FAILED as _FEEDBACK_REASON_REROUTE_FAILED,
)
from .feedback_keys import (
    FEEDBACK_REASON_ROUTE_SUCCESS as _FEEDBACK_REASON_ROUTE_SUCCESS,
)
from .feedback_keys import (
    FEEDBACK_REASON_TARGET_REQUIRED as _FEEDBACK_REASON_TARGET_REQUIRED,
)
from .feedback import FeedbackStore
from .intent_classifier import classify_message_intent
from .intervention_router import InterventionDecision, decide_intervention
from .memory import _chat_memory
from .memory_writer import MemoryWriteContext, MemoryWriter
from .middleware import TurnMiddlewareState, get_middleware_manager
from .models.pydantic_models import PluginKnowledgeBase
from .native_executor import NativeToolExecutionResult, NativeValidatedRoute
from .native_tool_loop import run_native_tool_loop
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
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
)
from .route_text import (
    ROUTE_ACTION_WORDS,
    contains_any,
    is_usage_question,
    normalize_message_text,
    should_force_knowledge_refresh,
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
    planner_missing_contains,
    plan_route_command,
    prepare_route_execution_plan,
    select_adapter_policy_for_message,
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
    from zhenxun.services.llm import LLMMessage

_INTENT_REFRESH_PUNCTUATION = ("。", "！", "？", "；", ";")
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




async def _persist_final_only_dialog(
    *,
    envelope: TurnChannelEnvelope,
    user_id: str,
    group_id: str | None,
    nickname: str,
    user_message,
    bot_id: str | None,
    event_context: ChatInterEventContext | None = None,
    thread_context: ThreadContext | None = None,
) -> None:
    final_text = str(envelope.final or "").strip()
    if not final_text:
        return
    dialog = await _chat_memory.add_dialog(
        user_id=user_id,
        group_id=group_id,
        nickname=nickname,
        user_message=user_message,
        ai_response=final_text,
        bot_id=bot_id,
    )
    if event_context is not None and thread_context is not None:
        pending_entities = tuple(
            dict.fromkeys(
                (
                    *thread_context.pending_entities,
                    *extract_pending_entities(uni_to_text_with_tags(user_message)),
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
            message_text=uni_to_text_with_tags(user_message),
            pending_entities=pending_entities,
            entity_hints=thread_context.entity_hints,
        )
    await MemoryWriter.write_from_dialog(
        MemoryWriteContext(
            session_id=_chat_memory.get_session_id(user_id, group_id),
            user_id=str(user_id),
            group_id=str(group_id) if group_id else None,
            message_text=uni_to_text_with_tags(user_message),
            response_text=final_text,
            source_dialog_id=int(getattr(dialog, "id", 0) or 0) if dialog else None,
            thread_id=thread_context.thread_id if thread_context is not None else None,
            topic_key=thread_context.topic_key if thread_context is not None else "",
            participants=thread_context.participants
            if thread_context is not None
            else (),
        )
    )


async def _handle_chat_dialogue_special_case(
    *,
    plan: ChatDialoguePlan,
    trace: StageTrace,
    user_id: str,
    group_id: str | None,
    nickname: str,
    user_message,
    bot_id: str | None,
    session_key: str,
    current_message: str,
    route_report: NativeRouteReport | None,
    budget_controller: TurnBudgetController | None,
    finalize_callback,
    event_context: ChatInterEventContext | None = None,
    thread_context: ThreadContext | None = None,
) -> bool:
    if plan.kind != "recap":
        return False
    recap = await _chat_memory.build_recent_conversation_recap(user_id, group_id)
    envelope = TurnChannelEnvelope()
    trace.update_tags(path="chat", outcome="chat_recap")
    envelope.add(ChannelName.ANALYSIS, "chat dialogue special: recap")
    envelope.add(ChannelName.FINAL, recap)
    log_turn_channels(envelope)
    await _persist_final_only_dialog(
        envelope=envelope,
        user_id=user_id,
        group_id=group_id,
        nickname=nickname,
        user_message=user_message,
        bot_id=bot_id,
        event_context=event_context,
        thread_context=thread_context,
    )
    trace.stage("persist")
    await MessageUtils.build_message(envelope.final).send()
    trace.stage("send")
    FeedbackStore.record_chat(
        session_id=session_key,
        kind="chat_completed",
        message_text=current_message,
        reply_text=envelope.final,
        weight=0.2,
    )
    if finalize_callback is not None:
        await finalize_callback(
            response_text=envelope.final,
            phase="post_gate:chat_recap",
        )
    _tag_execution_observation(
        trace,
        record_execution_observation(
            action="chat",
            success=True,
            reason=EXECUTION_REASON_CHAT_COMPLETED,
            session_id=session_key,
            route_stage="chat_recap",
            message_preview=current_message,
        ),
    )
    _finish_trace(
        trace=trace,
        user_id=str(user_id),
        group_id=group_id,
        message_preview=current_message,
        route_report=route_report,
        budget_controller=budget_controller,
    )
    return True


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

    planned_image_count = len(extract_image_tokens(current_message))
    if extra_image_segments:
        planned_image_count += len(extra_image_segments)
    command_plan = plan_route_command(
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        current_message=current_message,
        has_reply=has_reply,
        image_count=planned_image_count,
    )
    route_result = apply_command_plan_to_route_result(route_result, command_plan)
    decision = route_result.decision
    target_modules = build_target_modules(route_result, knowledge_plugins)
    execution_plan = prepare_route_execution_plan(
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        current_message=current_message,
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
        message_preview=current_message,
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
        message_text=current_message,
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


async def _build_chat_fallback_reply(
    *,
    bot: Bot,
    event: Event,
    user_id: str,
    group_id: str | None,
    nickname: str,
    model_name: str | None,
    mention_name_map: dict[str, str],
    session_key: str,
    current_message: str,
    middleware_state: TurnMiddlewareState,
    dialogue_plan: ChatDialoguePlan,
    image_parts,
    budget_controller: TurnBudgetController,
    native_force_pure_chat: bool,
    history_messages: list[LLMMessage] | None = None,
) -> tuple[str, bool, bool]:
    intent_profile = middleware_state.intent
    if intent_profile is None:
        raise RuntimeError("missing intent profile for chat fallback")
    intent_timeout = int(get_config_value("INTENT_TIMEOUT", 20) or 20)
    agent_gate = decide_agent_gate(
        config_enabled=bool(get_config_value("ENABLE_AGENT_MODE", True)),
        intent=intent_profile,
        message_text=middleware_state.message_text,
        has_images=bool(image_parts),
        has_mcp_endpoints=bool(get_mcp_endpoints()),
    )
    agent_enabled = False if native_force_pure_chat else agent_gate.enabled
    logger.debug(
        f"ChatInter agent gate: enabled={agent_enabled} reason={agent_gate.reason}"
    )
    reply: str | UniMessage | None = None
    if agent_enabled:
        middleware_state.metadata = {"phase": "agent_fallback"}
        await get_middleware_manager().dispatch("before_agent", middleware_state)
        try:
            agent_response = await run_chatinter_agent(
                bot=bot,
                event=event,
                user_id=str(user_id),
                group_id=str(group_id) if group_id else None,
                model=model_name,
                timeout=max(intent_timeout, 5),
                system_prompt=middleware_state.system_prompt,
                context_xml=middleware_state.context_xml,
                history_messages=history_messages,
                message_text=middleware_state.message_text,
                image_parts=image_parts or None,
                budget_controller=budget_controller,
            )
            if agent_response and str(agent_response.text or "").strip():
                reply = str(agent_response.text)
            usage = (
                agent_response.usage_info
                if agent_response and isinstance(agent_response.usage_info, dict)
                else {}
            )
            logger.debug(
                "chatinter agent reply ready: "
                f"prompt_tokens={usage.get('prompt_tokens', 0)} "
                f"completion_tokens={usage.get('completion_tokens', 0)} "
                f"total_tokens={usage.get('total_tokens', 0)}"
            )
        except Exception as exc:
            logger.warning(f"ChatInter agent 执行失败，降级普通对话: {exc}")
    if reply is None:
        reply = await handle_chat_message(
            message=middleware_state.message_text,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            mention_name_map=mention_name_map,
            session_key=session_key,
            budget_controller=budget_controller,
            dialogue_plan=dialogue_plan,
            context_xml=middleware_state.context_xml,
            history_messages=history_messages,
        )

    reply_text = (
        str(reply)
        if reply is not None and str(reply).strip()
        else "我暂时没想好怎么回答你。"
    )
    middleware_state.response_text = reply_text
    if agent_enabled:
        await get_middleware_manager().dispatch("after_agent", middleware_state)
    await get_middleware_manager().dispatch("after_chat", middleware_state)
    reply_text = (
        middleware_state.response_text
        if middleware_state.response_text is not None
        else reply_text
    )
    reply_text = normalize_ai_reply_text(reply_text or "")
    refined_reply_text = await refine_chat_reply(
        plan=dialogue_plan,
        user_message=current_message,
        reply_text=reply_text,
        context_xml=middleware_state.context_xml,
        budget_controller=budget_controller,
    )
    rewritten = refined_reply_text != reply_text
    reply_text = replace_mention_ids_with_names(refined_reply_text, mention_name_map)
    return reply_text, rewritten, agent_enabled


def _log_middleware_budget_report(label: str, metadata: dict) -> None:
    budget_report = metadata.get("budget_report")
    if not isinstance(budget_report, dict):
        return
    logger.debug(
        f"ChatInter {label} budget: "
        f"before={budget_report.get('before_tokens')} "
        f"after={budget_report.get('after_tokens')} "
        f"budget={budget_report.get('budget')} "
        f"ratio={budget_report.get('ratio')}"
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
    _log_middleware_budget_report("intent", middleware_state.metadata)
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
        frame.completion_disabled_force_chat = True
        frame.set_tag("outcome", "target_clarify_fallback_chat")
        logger.debug("???????????????????????" f"{fuzzy_prompt}")

    if needs_target_for_route(
        frame.current_message,
        route_message,
        target_policy=pre_route_target_policy,
    ):
        frame.completion_disabled_force_chat = True
        frame.set_tag("outcome", "target_required_fallback_chat")
        logger.debug(
            "?????????????????????"
            f"{pre_route_target_policy.target_missing_message or '-'}"
        )

    if route_message != frame.current_message:
        logger.debug(
            "ChatInter ????????"
            f"before='{frame.current_message}' -> after='{route_message}'"
        )
    if frame.completion_disabled_force_chat:
        frame.knowledge_base = PluginKnowledgeBase(
            plugins=[],
            user_role=knowledge_base.user_role,
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
            current_message=route_message,
            user_id=frame.user_id,
            session_id=frame.session_key,
            has_reply=frame.has_reply,
            extra_image_segments=frame.reply_image_segments_for_reroute,
            route_report=report,
        )

    native_result = await run_native_tool_loop(
        route_message,
        knowledge_base,
        session_key=frame.session_key,
        budget_controller=frame.budget_controller,
        has_reply=frame.has_reply,
        command_tools=command_tools,
        system_prompt=frame.system_prompt,
        context_xml=frame.context_xml,
        history_messages=frame.history_messages,
        route_executor=_execute_native_route_callback,
    )
    if native_result is not None:
        native_decision = native_result.decision
        route_result = native_result.route_result
        route_report = native_result.report
        frame.native_direct_reply = native_result.direct_reply
    else:
        route_report = NativeRouteReport(helper_mode=is_usage_question(route_message))
        route_report.finalize(
            reason="native_tool_loop_unavailable",
            stage="native_tool_loop",
        )
        native_decision = NativeRouteDecision(
            action="chat",
            confidence=0.0,
            reason="native_tool_loop_unavailable",
        )
        route_result = None
        frame.native_direct_reply = ""
    frame.set_native_route(
        native_decision=native_decision,
        route_result=route_result,
        route_report=route_report,
    )
    frame.update_tags(
        native_action=native_decision.action,
        native_confidence=f"{native_decision.confidence:.2f}",
        native_reason=native_decision.reason or "",
        native_plugin=route_result.decision.plugin_module if route_result else "",
        native_command=route_result.decision.command if route_result else "",
    )
    if route_report is not None:
        frame.update_tags(
            route_reason=route_report.final_reason,
            route_candidates=route_report.candidate_total,
            route_attempts=route_report.attempts,
            route_tool_candidates=route_report.tool_candidates,
        )
    logger.debug(
        "ChatInter native route result: "
        f"action={native_decision.action} "
        f"confidence={native_decision.confidence:.2f} "
        f"reason={native_decision.reason or '-'} "
        f"module={route_result.decision.plugin_module if route_result else '-'} "
        f"command={route_result.decision.command if route_result else '-'}"
    )
    middleware_state.metadata = {
        "phase": "route_completed",
        "native_action": native_decision.action,
        "route_reason": route_report.final_reason if route_report else "",
    }
    await middleware.dispatch("after_route", middleware_state)


async def _stage_prepare_chat_fallback(
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
        raise RuntimeError("missing intent profile for chat fallback")
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


async def _stage_run_chat_fallback(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
    finalize_callback,
) -> None:
    if frame.dialogue_plan is None:
        raise RuntimeError("missing dialogue plan for chat fallback")

    if await _handle_chat_dialogue_special_case(
        plan=frame.dialogue_plan,
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        nickname=frame.nickname,
        user_message=frame.uni_msg or frame.current_message,
        bot_id=frame.bot_id,
        session_key=frame.session_key,
        current_message=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
        event_context=frame.event_context,
        thread_context=frame.thread_context,
        finalize_callback=finalize_callback,
    ):
        return

    frame.context_xml = frame.enriched_context_xml
    frame.sync_to_middleware(
        middleware_state,
        phase=PipelineStage.CHAT_FALLBACK.value,
    )
    await middleware.dispatch("before_chat", middleware_state)
    frame.apply_prompt_state(middleware_state)
    frame.enriched_context_xml = frame.context_xml
    _log_middleware_budget_report("agent", middleware_state.metadata)
    frame.stage(PipelineStage.AGENT_BUDGET)

    chat_execution_frame = start_execution_observation(
        action="chat",
        route_stage="chat",
        session_id=frame.session_key,
        message_preview=frame.current_message,
        **_route_report_observer_kwargs(frame.route_report),
    )
    reply_text, rewritten, agent_enabled = await _build_chat_fallback_reply(
        bot=bot,
        event=event,
        user_id=frame.user_id,
        group_id=frame.group_id,
        nickname=frame.nickname,
        model_name=frame.model_name,
        mention_name_map=frame.mention_name_map,
        session_key=frame.session_key,
        current_message=frame.current_message,
        middleware_state=middleware_state,
        dialogue_plan=frame.dialogue_plan,
        image_parts=frame.image_parts,
        budget_controller=frame.budget_controller,
        native_force_pure_chat=frame.native_force_pure_chat,
        history_messages=frame.history_messages,
    )
    frame.update_tags(
        agent_enabled=int(agent_enabled),
        chat_rewritten=int(rewritten),
    )
    frame.stage(PipelineStage.CHAT_FALLBACK)
    envelope = TurnChannelEnvelope()
    frame.update_tags(path="chat", outcome="chat_fallback")
    envelope.add(ChannelName.ANALYSIS, "chat fallback")
    envelope.add(ChannelName.FINAL, reply_text)
    log_turn_channels(envelope)
    await _persist_final_only_dialog(
        envelope=envelope,
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
        phase="post_gate:chat_fallback",
    )
    FeedbackStore.record_chat(
        session_id=frame.session_key,
        kind="chat_rewritten" if rewritten else "chat_completed",
        message_text=frame.current_message,
        reply_text=envelope.final,
        weight=0.35 if rewritten else 0.2,
    )
    _tag_execution_observation(
        frame.trace,
        chat_execution_frame.finish(
            success=True,
            reason=EXECUTION_REASON_CHAT_REWRITTEN
            if rewritten
            else EXECUTION_REASON_CHAT_COMPLETED,
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


async def _stage_send_native_direct_chat(
    *,
    frame: TurnFrame,
    finalize_callback,
) -> bool:
    reply_text = normalize_ai_reply_text(frame.native_direct_reply)
    reply_text = replace_mention_ids_with_names(reply_text, frame.mention_name_map)
    if not normalize_message_text(str(reply_text or "")):
        return False

    chat_execution_frame = start_execution_observation(
        action="chat",
        route_stage="native_tools",
        session_id=frame.session_key,
        message_preview=frame.current_message,
        **_route_report_observer_kwargs(frame.route_report),
    )
    envelope = TurnChannelEnvelope()
    frame.update_tags(path="chat", outcome="native_tools_chat")
    envelope.add(ChannelName.ANALYSIS, "native tools direct chat")
    envelope.add(ChannelName.FINAL, reply_text)
    log_turn_channels(envelope)
    await _persist_final_only_dialog(
        envelope=envelope,
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
        phase="post_gate:native_tools_chat",
    )
    FeedbackStore.record_chat(
        session_id=frame.session_key,
        kind="chat_completed",
        message_text=frame.current_message,
        reply_text=envelope.final,
        weight=0.2,
    )
    _tag_execution_observation(
        frame.trace,
        chat_execution_frame.finish(
            success=True,
            reason=EXECUTION_REASON_CHAT_COMPLETED,
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
    return True


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

        if frame.native_decision is None:
            raise RuntimeError("missing native tool decision")
        knowledge_base = frame.knowledge_base
        if knowledge_base is None:
            raise RuntimeError("missing plugin knowledge base")

        if (
            frame.native_decision.action == "chat"
            and frame.native_direct_reply
            and await _stage_send_native_direct_chat(
                frame=frame,
                finalize_callback=_dispatch_post_gate,
            )
        ):
            return


        if frame.native_decision.action == "clarify":
            frame.update_tags(path="chat", outcome="native_clarify_fallback_chat")
            logger.debug(
                "Native tools need clarification, fallback to chat: "
                f"missing={','.join(frame.native_decision.missing)} "
                f"reason={frame.native_decision.reason or '-'}"
            )


        frame.native_force_pure_chat = True
        await _stage_prepare_chat_fallback(frame=frame, message=message)
        await _stage_run_chat_fallback(
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
