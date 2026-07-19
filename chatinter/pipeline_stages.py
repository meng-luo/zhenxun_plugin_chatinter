"""
ChatInter - pipeline stages

实现消息处理流程，支持多模态输入（图片识别）。
使用 UniMessage 统一处理消息。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from html import escape as _xml_escape
import time
from typing import Any, cast

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger
from zhenxun.utils.message import MessageUtils

from .addressee_resolver import AddresseeResult, resolve_addressee
from .agents.chat_reply_agent import ChatReplyAgent
from .agents.core import PrivateChatRequest
from .chat_handler import (
    normalize_ai_reply_text,
    replace_mention_ids_with_names,
)
from .chat_runtime import ChatRuntime
from .config import INTENT_TIMEOUT_SECONDS, get_fallback_models
from .context_packer import DialogueContextPack
from .event_context import ChatInterEventContext, build_event_context
from .event_runtime import (
    apply_runtime_plugin_overrides,
    event_adapter_name,
    event_is_private,
    event_type_name,
)
from .execution_observer import (
    EXECUTION_REASON_CANCELLED,
    EXECUTION_REASON_ERROR,
    ExecutionObservation,
    record_execution_observation,
    start_execution_observation,
)
from .feedback import FeedbackKind, FeedbackStore
from .group_turn_context import clear_group_turn_context, consume_group_turn_context
from .intent_classifier import classify_message_intent
from .intervention_router import InterventionDecision, decide_intervention
from .llm_compat import LLMContentPart, LLMMessage
from .main_request_models import MainRequestResult
from .memory import _chat_memory
from .memory_writer import MemoryWriteContext, MemoryWriter
from .middleware import TurnMiddlewareState
from .models.chat_history import ChatInterChatHistory
from .models.pydantic_models import PluginKnowledgeBase
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
    PluginSelectionContext,
)
from .route_execution import (
    build_reply_image_segments_for_reroute,
    extract_at_tokens,
    extract_image_tokens,
    extract_reply_sender_id,
)
from .route_text import (
    ROUTE_ACTION_WORDS,
    contains_any,
    is_usage_question,
    normalize_message_text,
    normalize_reply_text,
)
from .runtime_result import _fallback_result, _finalize_result, _user_timeline_item
from .target_context import (
    build_mention_profiles,
    extract_pending_entities,
    remember_target_resolution,
)
from .thread_resolver import ThreadContext, resolve_thread_context
from .thread_store import record_thread_message
from .trace import StageTrace
from .trajectory_eval import schedule_response_quality_eval
from .turn_frame import PipelineStage, TurnFrame
from .turn_metrics import (
    build_turn_metrics_snapshot,
    emit_turn_metrics,
    record_route_observation,
)
from .turn_output import ChannelName, TurnChannelEnvelope, log_turn_channels
from .turn_runtime import TurnBudgetController, estimate_text_tokens
from .utils.multimodal import (
    MAX_CHAT_IMAGE_PARTS,
    ChatImageExtraction,
    build_labeled_image_user_content,
    extract_chat_images_from_message,
    extract_chat_images_from_reply_chain,
    route_images_for_chat,
)
from .utils.unimsg_utils import remove_reply_segment, uni_to_text_with_tags

_MAIN_REQUEST_STAGE = "main_request"
_PERSONA_PROMPT_MARKER = "当前人格设定（来自配置，优先遵循）："


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
    session_id: str | None = None,
) -> None:
    if main_result.output.outcome != "chat_completed":
        return
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
        session_id=session_id,
    )
    if dialog is not None and int(getattr(dialog, "id", 0) or 0) % 32 == 0:
        await ChatInterChatHistory.prune_old_dialogs(
            session_id or _chat_memory.get_session_id(user_id, group_id),
            200,
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
            session_id=session_id or _chat_memory.get_session_id(user_id, group_id),
            user_id=str(user_id),
            group_id=str(group_id) if group_id else None,
            message_text=user_text,
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
















def _append_context_section(context_xml: str, section: str) -> str:
    text = str(section or "").strip()
    if not text:
        return context_xml
    if text in str(context_xml or ""):
        return context_xml
    return f"{context_xml}\n{text}".strip()




async def stage_identity(
    *,
    frame: TurnFrame,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    await middleware.dispatch("pre_gate", middleware_state)
    if frame.allow_plugin_tools:
        FeedbackStore.inspect_user_followup(
            session_id=frame.session_key,
            message_text=frame.raw_message,
        )
        await apply_runtime_plugin_overrides(
            event=event,
            session_key=frame.session_key,
            group_id=frame.group_id,
        )
    frame.stage(PipelineStage.IDENTITY)


async def stage_event_context(
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
    if event_context.turn_messages:
        frame.update_tags(
            turn_messages=len(event_context.turn_messages),
            turn_priority=event_context.turn_priority,
            pending_updates=len(event_context.pending_human_updates),
        )
    mention_profiles = await build_mention_profiles(
        frame.group_id,
        event_context.message_text_with_tags,
        bot_id=frame.bot_id,
        bot=bot,
    )
    frame.mention_profiles = mention_profiles
    if frame.turn_messages:
        current_message = frame.raw_message.strip()
    elif frame.event_message is not None:
        current_message = uni_to_text_with_tags(frame.event_message)
    elif frame.uni_msg:
        current_msg = remove_reply_segment(frame.uni_msg)
        current_message = uni_to_text_with_tags(current_msg)
    elif cached_plain_text:
        current_message = cached_plain_text.strip()
    else:
        current_message = frame.raw_message.strip()
    frame.current_message = current_message
    frame.stage(PipelineStage.EVENT_CONTEXT)


async def stage_thread_context(
    *,
    frame: TurnFrame,
) -> None:
    if frame.event_context is None:
        raise RuntimeError("missing event context")
    (
        dialogue_context_pack,
        _speaker_profile,
        addressee_result,
        thread_context,
        intervention_decision,
    ) = await _build_dialogue_context_pack(
        event_context=frame.event_context,
        mention_profiles=frame.mention_profiles,
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
    frame.stage(PipelineStage.THREAD_CONTEXT)


async def stage_memory(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    profile = frame.chat_runtime_profile
    if profile is None:
        raise RuntimeError("missing chat runtime profile")
    dialogue_state = frame.dialogue_state
    memory_message = (
        frame.raw_message if frame.turn_messages else frame.uni_msg or frame.raw_message
    )
    memory_dialogue_state = ChatRuntime.memory_dialogue_state(frame)
    (
        chat_system_prompt,
        context_xml,
        reply_images_data,
        history_messages,
    ) = await _chat_memory.build_full_context(
        frame.user_id,
        frame.group_id,
        frame.nickname,
        memory_message,
        bot,
        frame.bot_id,
        event,
        frame.dialogue_context_pack,
        memory_dialogue_state,
        frame.scenario,
        persona_selection=profile.persona_selection if profile is not None else None,
        session_id=frame.session_key,
        legacy_session_id=frame.legacy_session_key,
    )
    frame.chat_memory_layered = getattr(
        frame.dialogue_context_pack,
        "layered_memory",
        None,
    )
    frame.dialogue_state = dialogue_state
    frame.set_context(
        system_prompt=chat_system_prompt,
        context_xml=context_xml,
        reply_images_data=reply_images_data,
        history_messages=history_messages,
    )
    frame.stage(PipelineStage.MEMORY)








async def stage_dialogue_state(*, frame: TurnFrame) -> None:
    profile = ChatRuntime.attach_profile(frame)
    if profile is not None:
        frame.chat_runtime_profile = profile
        frame.dialogue_plan = profile.dialogue_plan
        frame.dialogue_state = profile.dialogue_state
        frame.previous_dialogue_state = profile.previous_state
        frame.update_tags(
            dialogue_tone=profile.dialogue_state.tone,
            dialogue_emotion=profile.dialogue_state.user_emotion,
            dialogue_purpose=profile.dialogue_state.dialogue_purpose,
            reply_posture=profile.dialogue_state.reply_posture,
            group_atmosphere=profile.dialogue_state.group_atmosphere,
        )
    frame.stage(PipelineStage.DIALOGUE_STATE)


async def _prepare_current_message_context(
    *,
    frame: TurnFrame,
    middleware_state: TurnMiddlewareState,
    middleware,
    cached_plain_text: str | None = None,
) -> None:
    if frame.turn_messages:
        current_message = frame.raw_message.strip()
    elif frame.event_message is not None:
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






async def stage_chat_capability_hint(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
    cached_plain_text: str | None = None,
) -> None:
    frame.knowledge_base = _empty_plugin_knowledge_base()
    frame.stage(PipelineStage.KNOWLEDGE)
    await _prepare_current_message_context(
        frame=frame,
        middleware_state=middleware_state,
        middleware=middleware,
        cached_plain_text=cached_plain_text,
    )
    await _prepare_lightweight_chat_route(
        frame=frame,
        bot=bot,
        event=event,
        middleware_state=middleware_state,
        middleware=middleware,
    )
    frame.stage(PipelineStage.CAPABILITY_HINT)


def _empty_plugin_knowledge_base() -> PluginKnowledgeBase:
    return PluginKnowledgeBase(plugins=[], user_role="普通用户")


async def _prepare_lightweight_chat_route(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
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
    frame.reply_image_segments_for_reroute = build_reply_image_segments_for_reroute(
        frame.reply_images_data
    )
    frame.route_message = frame.current_message
    frame.sync_to_middleware(
        middleware_state,
        phase=PipelineStage.ROUTE_SELECTION.value,
        route_message=frame.route_message,
    )
    frame.stage(PipelineStage.ROUTE_SELECTION)
    await middleware.dispatch("before_route", middleware_state)
    route_message = middleware_state.route_message or frame.route_message
    frame.route_message = route_message
    frame.selection_context = PluginSelectionContext(
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
        thread_id=frame.thread_context.thread_id if frame.thread_context else "",
        intervention_action=frame.intervention_decision.action
        if frame.intervention_decision
        else "",
    )
    frame.command_tools = []
    frame.chat_tool_exposure_state = "none"
    intent_profile = classify_message_intent(
        route_message,
        frame.knowledge_base or _empty_plugin_knowledge_base(),
    )
    frame.intent_profile = intent_profile
    frame.update_tags(
        intent_kind=intent_profile.kind,
        intent_reason=intent_profile.reason,
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
    route_report = NativeRouteReport(helper_mode=is_usage_question(frame.route_message))
    frame.set_native_route(
        native_decision=NativeRouteDecision(
            action="chat",
            confidence=0.0,
            reason="chat_only_no_plugin_tools",
        ),
        route_result=None,
        route_report=route_report,
    )


def _build_system_prompt(frame: TurnFrame) -> str:
    return normalize_message_text(frame.system_prompt)


def _text_section_stats(text: str) -> dict[str, int]:
    content = str(text or "")
    return {"chars": len(content), "tokens": estimate_text_tokens(content)}


def _message_text(content: str | list[LLMContentPart]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.text or part.thought_text or "")
        for part in content
        if str(part.type or "") == "text" or part.text or part.thought_text
    )


def _messages_text(messages: list[LLMMessage] | None) -> str:
    return "\n".join(_message_text(message.content) for message in messages or [])


def _extract_tag_blocks(source: str, tag: str) -> tuple[str, str]:
    text = str(source or "")
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    blocks: list[str] = []
    rest: list[str] = []
    cursor = 0
    while True:
        start = text.find(start_tag, cursor)
        if start < 0:
            rest.append(text[cursor:])
            break
        end = text.find(end_tag, start + len(start_tag))
        if end < 0:
            rest.append(text[cursor:])
            break
        end += len(end_tag)
        rest.append(text[cursor:start])
        blocks.append(text[start:end])
        cursor = end
    return "\n".join(blocks), "".join(rest)


def _extract_persona_prompt(system_prompt: str) -> tuple[str, str]:
    text = str(system_prompt or "")
    if _PERSONA_PROMPT_MARKER not in text:
        return "", text
    marker = text.find(_PERSONA_PROMPT_MARKER)
    boundary = max(text.rfind("\n\n你是"), text.rfind(" 你是"))
    if boundary <= marker:
        return "", text
    return text[:boundary].strip(), text[boundary:].strip()


def _record_prompt_section_stats(
    frame: TurnFrame,
    *,
    system_prompt: str,
    context_xml: str,
    history_messages: list[LLMMessage] | None,
    current_user_text: str,
) -> None:
    persona_text, system_without_persona = _extract_persona_prompt(system_prompt)
    memory_text, context_without_memory = _extract_tag_blocks(
        context_xml,
        "long_term_memory",
    )
    image_context_text, context_without_media = _extract_tag_blocks(
        context_without_memory,
        "image_context",
    )
    image_label_text = "\n".join(
        f"Image {index}:"
        for index, _part in enumerate(
            (frame.image_parts or [])[:MAX_CHAT_IMAGE_PARTS],
            1,
        )
    )
    image_text = "\n".join(
        part for part in (image_context_text, image_label_text) if part
    )
    sections = {
        "system": system_without_persona,
        "persona": persona_text,
        "context": context_without_media,
        "history": _messages_text(history_messages),
        "memory": memory_text,
        "image": image_text,
        "current_user": current_user_text,
    }
    tags: dict[str, str | float | None] = {
        "prompt_image_count": float(len(frame.image_parts or [])),
    }
    for name, text in sections.items():
        stats = _text_section_stats(text)
        tags[f"prompt_{name}_chars"] = float(stats["chars"])
        tags[f"prompt_{name}_tokens"] = float(stats["tokens"])
    frame.update_tags(**tags)


def _build_agent_messages(frame: TurnFrame) -> list[LLMMessage]:
    system_prompt = _build_system_prompt(frame)
    current_text = frame.route_message or frame.current_message
    user_text = current_text
    turn_xml = _build_turn_queue_context(frame)
    if turn_xml:
        user_text = turn_xml
        if len(frame.turn_messages or []) <= 1:
            user_text = f"{turn_xml}\n\n{_xml_escape(current_text, quote=False)}"
    current_user_section = user_text
    if frame.context_xml:
        current_user_payload = (
            user_text if turn_xml else _xml_escape(user_text, quote=False)
        )
        current_user_section = (
            f"<current_user_message>{current_user_payload}</current_user_message>"
        )
        user_text = (
            f"{frame.context_xml}\n\n"
            f"{current_user_section}"
        )
    user_content: str | list[LLMContentPart]
    if frame.image_parts:
        user_content = build_labeled_image_user_content(user_text, frame.image_parts)
    else:
        user_content = user_text
    _record_prompt_section_stats(
        frame,
        system_prompt=system_prompt,
        context_xml=frame.context_xml,
        history_messages=frame.history_messages,
        current_user_text=current_user_section,
    )
    return [
        LLMMessage.system(system_prompt),
        *list(frame.history_messages or []),
        LLMMessage.user(user_content),
    ]


def _should_build_agent_messages(frame: TurnFrame) -> bool:

    return not (
        frame.scenario == "group_plugin_selector"
        and frame.chat_tool_exposure_state in {"unknown", "plugin_tools_exposed"}
    )


def _build_turn_queue_context(frame: TurnFrame) -> str:
    sections: list[str] = []
    if frame.turn_messages and len(frame.turn_messages) > 1:
        lines = "\n".join(
            f'<message index="{index}">'
            f"{_xml_escape(normalize_message_text(text), quote=False)}"
            "</message>"
            for index, text in enumerate(frame.turn_messages, 1)
            if normalize_message_text(text)
        )
        if lines:
            sections.append(f"<merged_turn_messages>\n{lines}\n</merged_turn_messages>")
    if frame.pending_human_updates:
        lines = "\n".join(
            f'<update index="{index}">'
            f"{_xml_escape(normalize_message_text(text), quote=False)}"
            "</update>"
            for index, text in enumerate(frame.pending_human_updates, 1)
            if normalize_message_text(text)
        )
        if lines:
            sections.append(
                "<pending_human_updates>\n" f"{lines}\n" "</pending_human_updates>"
            )
    return "\n".join(sections)


async def _prepare_chat_multimodal(frame: TurnFrame) -> None:
    image_parts = list(getattr(frame, "image_parts", []) or [])
    if not image_parts:
        return
    timeout = float(INTENT_TIMEOUT_SECONDS)
    routing = await route_images_for_chat(
        image_parts,
        text=frame.route_message or frame.current_message,
        model_name=frame.model_name,
        fallback_models=get_fallback_models(frame.model_name),
        timeout=min(max(timeout, 5.0), 30.0),
    )
    frame.image_parts = routing.image_parts
    if routing.context_xml:
        frame.context_xml = _append_context_section(
            frame.context_xml,
            routing.context_xml,
        )
        frame.enriched_context_xml = _append_context_section(
            frame.enriched_context_xml or frame.context_xml,
            routing.context_xml,
        )
    if routing.mode != "none":
        frame.update_tags(
            image_mode=routing.mode,
            image_count=routing.original_count,
            image_passed=len(routing.image_parts),
        )


async def stage_current_user(
    *,
    frame: TurnFrame,
    message=None,
) -> None:
    image_extraction = await _extract_current_turn_images(frame=frame, message=message)
    image_parts = list(image_extraction.image_parts)
    frame.image_parts = image_parts
    if image_extraction.context_xml:
        frame.context_xml = _append_context_section(
            frame.context_xml,
            image_extraction.context_xml,
        )
    if image_parts:
        logger.debug(f"当前消息提取到 {len(image_parts)} 张图片")

    if frame.reply_images_data:
        reply_extraction = await extract_chat_images_from_reply_chain(
            frame.reply_images_data
        )
        image_parts.extend(reply_extraction.image_parts)
        frame.image_parts = image_parts
        if reply_extraction.context_xml:
            frame.context_xml = _append_context_section(
                frame.context_xml,
                reply_extraction.context_xml,
            )
        if frame.reply_images_data:
            logger.debug(f"回复链提取到 {len(frame.reply_images_data)} 张图片")
    frame.stage(PipelineStage.CURRENT_USER)
    frame.enriched_context_xml = frame.context_xml
    if frame.intent_profile is None:
        raise RuntimeError("missing intent profile for main request")
    dialogue_plan = frame.dialogue_plan
    dialogue_state = frame.dialogue_state
    if dialogue_plan is None or dialogue_state is None:
        raise RuntimeError("missing chat runtime profile")
    chat_prompt = ChatRuntime.build_prompt_context(
        frame,
        base_context_xml=frame.enriched_context_xml,
    )
    frame.enriched_context_xml = chat_prompt.context_xml
    if chat_prompt.tags:
        frame.update_tags(**chat_prompt.tags)
    frame.update_tags(
        chat_kind=dialogue_plan.kind,
        chat_style=dialogue_plan.style,
        chat_reason=dialogue_plan.reason,
        dialogue_tone=dialogue_state.tone,
        dialogue_emotion=dialogue_state.user_emotion,
        dialogue_purpose=dialogue_state.dialogue_purpose,
        reply_posture=dialogue_state.reply_posture,
        group_atmosphere=dialogue_state.group_atmosphere,
    )


async def _extract_current_turn_images(
    *,
    frame: TurnFrame,
    message=None,
) -> ChatImageExtraction:
    source_for_media = (
        frame.event_message or frame.uni_msg or message or frame.raw_message
    )
    sources = frame.turn_message_sources or [source_for_media]
    extractions = [
        await extract_chat_images_from_message(source) for source in sources
    ]
    contexts = list(
        dict.fromkeys(item.context_xml for item in extractions if item.context_xml)
    )
    return ChatImageExtraction(
        image_parts=[part for item in extractions for part in item.image_parts],
        context_xml="\n".join(contexts),
        original_count=sum(item.original_count for item in extractions),
        skipped_oversized=sum(item.skipped_oversized for item in extractions),
    )


async def stage_scratchpad(
    *,
    frame: TurnFrame,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    if frame.dialogue_plan is None:
        raise RuntimeError("missing dialogue plan for main request")
    frame.context_xml = frame.enriched_context_xml
    frame.sync_to_middleware(
        middleware_state,
        phase=PipelineStage.SCRATCHPAD.value,
    )
    await middleware.dispatch("before_chat", middleware_state)
    frame.apply_prompt_state(middleware_state)
    frame.enriched_context_xml = frame.context_xml
    if _should_build_agent_messages(frame):
        await _prepare_chat_multimodal(frame)
    frame.agent_messages = (
        _build_agent_messages(frame) if _should_build_agent_messages(frame) else []
    )
    frame.stage(PipelineStage.SCRATCHPAD)




async def _run_chat_reply_agent_turn(
    *,
    message_text: str,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    messages: list[LLMMessage],
    route_completed_hook: Any | None,
    reply_hook: Any | None,
) -> MainRequestResult:
    normalized_message = normalize_message_text(message_text)
    report = NativeRouteReport(helper_mode=is_usage_question(normalized_message))

    async def run_agent() -> MainRequestResult:
        return (
            await ChatReplyAgent().run(
                PrivateChatRequest(
                    message_text=normalized_message,
                    session_key=session_key,
                    budget_controller=budget_controller,
                    messages=messages,
                    report=report,
                )
            )
        ).to_main_result()

    return await _run_direct_agent_turn(
        message_text=normalized_message,
        report=report,
        budget_controller=budget_controller,
        route_completed_hook=route_completed_hook,
        reply_hook=reply_hook,
        run_agent=run_agent,
    )


async def _run_direct_agent_turn(
    *,
    message_text: str,
    report: NativeRouteReport,
    budget_controller: TurnBudgetController | None,
    route_completed_hook: Any | None,
    reply_hook: Any | None,
    run_agent,
) -> MainRequestResult:
    started = time.perf_counter()
    try:
        try:
            result = await run_agent()
        except Exception as exc:
            logger.error(f"ChatInter agent failed: {exc}")
            result = _fallback_result(
                report=report,
                reason=f"agent_error:{type(exc).__name__}",
                reply="抱歉，我刚刚处理失败了。",
                timeline=[_user_timeline_item(message_text)],
            )
        return await _finalize_result(
            result,
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )
    finally:
        if budget_controller is not None:
            budget_controller.record_stage(
                _MAIN_REQUEST_STAGE,
                time.perf_counter() - started,
            )


def _build_agent_stage_hooks(
    *,
    frame: TurnFrame,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> tuple[Any, Any]:
    async def _route_completed_callback(main_result: MainRequestResult) -> None:
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

    return _route_completed_callback, _reply_hook


def _set_agent_stage_result(
    *,
    frame: TurnFrame,
    main_result: MainRequestResult,
) -> None:
    frame.main_result = main_result
    envelope = TurnChannelEnvelope()
    frame.update_tags(
        path="main_request",
        outcome=main_result.output.outcome,
    )
    envelope.add(ChannelName.ANALYSIS, main_result.output.analysis)
    if main_result.output.should_send:
        reply_text = main_result.output.final_text
        if not normalize_message_text(reply_text):
            reply_text = "我暂时没想好怎么回答你。"
        quality_enabled = ChatRuntime.isolation_for_frame(frame).allow_quality_judge
        quality = ChatRuntime.judge_final_reply(
            frame=frame,
            final_text=reply_text,
        )
        frame.response_quality_result = quality
        if quality.reason:
            frame.update_tags(
                response_quality=quality.reason,
                response_quality_action=quality.action,
            )
        if quality_enabled and quality.revised_text:
            reply_text = normalize_reply_text(quality.revised_text) or reply_text
            main_result = replace(
                main_result,
                output=replace(main_result.output, final_text=reply_text),
            )
            frame.main_result = main_result
        if quality_enabled:
            schedule_response_quality_eval(
                quality_result=quality,
                final_text=reply_text,
                original_message=frame.current_message,
                scenario=frame.scenario,
                session_key=frame.session_key,
                trace_id=str(frame.trace.tags.get("message_id", "")),
            )
        envelope.add(ChannelName.FINAL, reply_text)
    frame.final_envelope = envelope




async def stage_chat_run(
    *,
    frame: TurnFrame,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    if frame.dialogue_plan is None:
        raise RuntimeError("missing dialogue plan for main request")
    if not frame.agent_messages:
        await _prepare_chat_multimodal(frame)
        frame.agent_messages = _build_agent_messages(frame)
    frame.stage(PipelineStage.AGENT_RUN)
    route_completed_hook, reply_hook = _build_agent_stage_hooks(
        frame=frame,
        middleware_state=middleware_state,
        middleware=middleware,
    )
    progress_task = (
        asyncio.create_task(_send_delayed_reply_status(frame))
        if frame.scenario == "private_chat"
        else None
    )
    try:
        main_result = await _run_chat_reply_agent_turn(
            message_text=frame.route_message or frame.current_message,
            session_key=frame.session_key,
            budget_controller=frame.budget_controller,
            messages=frame.agent_messages,
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )
    finally:
        if progress_task is not None:
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
    _set_agent_stage_result(frame=frame, main_result=main_result)


async def _send_delayed_reply_status(frame: TurnFrame) -> None:
    await asyncio.sleep(15.0)
    if _frame_is_current(frame) and not frame.turn_finished:
        await MessageUtils.build_message("正在回复...").send()


async def prepare_plugin_fallback_chat_context(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    frame.command_tools = []
    frame.chat_tool_exposure_state = "none"
    frame.route_message = frame.current_message
    frame.update_tags(plugin_router_fallback="chat")
    await stage_dialogue_state(frame=frame)
    await stage_memory(frame=frame, bot=bot, event=event)
    await stage_current_user(frame=frame, message=frame.message)
    await stage_scratchpad(
        frame=frame,
        middleware_state=middleware_state,
        middleware=middleware,
    )


async def stage_persist(frame: TurnFrame) -> None:
    main_result = frame.main_result
    if main_result is None:
        raise RuntimeError("missing main request result")
    envelope = frame.final_envelope or TurnChannelEnvelope()
    if not _frame_is_current(frame) and not bool(
        getattr(frame, "delivery_succeeded", False)
    ):
        frame.update_tags(path="superseded", outcome="superseded")
        frame.turn_finished = True
        return
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
            session_id=getattr(frame, "session_key", None),
        )
        if main_result.output.outcome in {"tool_completed", "tool_failed"}:
            _mark_group_context_answered(frame)
        frame.stage(PipelineStage.PERSIST)
        if frame.post_gate_callback is not None:
            await frame.post_gate_callback(phase="post_gate:main_request")
        _finish_trace(
            trace=frame.trace,
            user_id=frame.user_id,
            group_id=frame.group_id,
            message_preview=frame.current_message,
            route_report=frame.route_report,
            budget_controller=frame.budget_controller,
        )
        frame.turn_finished = True
        return

    if not bool(getattr(frame, "delivery_succeeded", False)):
        raise RuntimeError("assistant reply was not delivered")
    try:
        await _persist_message_timeline(
            main_result=main_result,
            user_id=frame.user_id,
            group_id=frame.group_id,
            nickname=frame.nickname,
            user_message=frame.uni_msg or frame.current_message,
            bot_id=frame.bot_id,
            event_context=frame.event_context,
            thread_context=frame.thread_context,
            session_id=frame.session_key,
        )
    except Exception as exc:
        logger.warning(f"ChatInter delivered reply persistence failed: {exc}")
        frame.update_tags(persistence_error=type(exc).__name__)
    frame.stage(PipelineStage.PERSIST)
    if frame.post_gate_callback is not None:
        await frame.post_gate_callback(
            response_text=envelope.final,
            phase="post_gate:main_request",
        )
    if main_result.output.record_chat_feedback:
        FeedbackStore.record_chat(
            session_id=frame.session_key,
            kind=cast(FeedbackKind, main_result.output.feedback_kind),
            message_text=frame.current_message,
            reply_text=envelope.final,
            weight=0.2,
        )
    _persist_plain_chat_dialogue_state(frame=frame, reply_text=envelope.final)
    _tag_execution_observation(
        frame.trace,
        frame.chat_execution_frame.finish(
            success=True,
            reason=main_result.output.observation_reason,
        )
        if frame.chat_execution_frame is not None
        else record_execution_observation(
            action="chat",
            success=True,
            reason=main_result.output.observation_reason,
            session_id=frame.session_key,
            message_preview=frame.current_message,
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
    frame.turn_finished = True


async def stage_send(frame: TurnFrame) -> None:
    if frame.turn_finished:
        return
    main_result = frame.main_result
    if main_result is None:
        raise RuntimeError("missing main request result")
    envelope = frame.final_envelope
    if envelope is None:
        raise RuntimeError("missing final envelope")
    if not main_result.output.should_send:
        return
    if not _frame_is_current(frame):
        frame.update_tags(path="superseded", outcome="superseded")
        frame.turn_finished = True
        return
    log_turn_channels(envelope)
    frame.chat_execution_frame = start_execution_observation(
        action="chat",
        route_stage="main_request",
        session_id=frame.session_key,
        message_preview=frame.current_message,
        **_route_report_observer_kwargs(frame.route_report),
    )
    await MessageUtils.build_message(envelope.final).send()
    frame.delivery_succeeded = True
    _mark_group_context_answered(frame)
    frame.stage(PipelineStage.SEND)


def _persist_plain_chat_dialogue_state(
    *,
    frame: TurnFrame,
    reply_text: str,
) -> None:
    try:
        if ChatRuntime.persist_dialogue_state(frame=frame, reply_text=reply_text):
            frame.update_tags(dialogue_state_persisted="1")
    except Exception:
        return


def _mark_group_context_answered(frame: TurnFrame) -> None:
    if not frame.group_id:
        return
    message_id = normalize_message_text(
        getattr(getattr(frame, "event_context", None), "event_id", "")
        or frame.trace.tags.get("message_id", "")
        or getattr(getattr(frame, "event", None), "message_id", "")
        or getattr(getattr(frame, "event", None), "event_id", "")
    )
    if message_id:
        consume_group_turn_context(frame.group_id, message_id)
    else:
        clear_group_turn_context(frame.group_id)


async def handle_pipeline_cancelled(frame: TurnFrame) -> None:
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
    logger.debug(f"ChatInter 处理被取消: user={frame.user_id}, group={group_name}")
    if frame.post_gate_callback is not None:
        await frame.post_gate_callback(phase="post_gate:cancelled")
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )
    frame.turn_finished = True


async def handle_pipeline_error(frame: TurnFrame, error: Exception) -> None:
    frame.update_tags(path="error", outcome="error")
    plugin_error = (
        frame.scenario == "group_plugin_selector"
        and frame.chat_tool_exposure_state == "plugin_tools_exposed"
    )
    _tag_execution_observation(
        frame.trace,
        record_execution_observation(
            action="execute" if plugin_error else "chat",
            success=False,
            reason=EXECUTION_REASON_ERROR,
            session_id=frame.session_key,
            message_preview=frame.current_message,
        ),
    )
    if frame.middleware_state is not None and frame.middleware is not None:
        frame.middleware_state.message_text = frame.current_message
        frame.middleware_state.system_prompt = frame.system_prompt
        frame.middleware_state.context_xml = (
            frame.enriched_context_xml or frame.context_xml
        )
        frame.middleware_state.metadata = {"phase": "error", "error": str(error)}
        await frame.middleware.dispatch("on_error", frame.middleware_state)
    logger.error(f"ChatInter 处理失败: {error}")
    if (
        not bool(getattr(frame, "delivery_succeeded", False))
        and _frame_is_current(frame)
    ):
        await MessageUtils.build_failure_message().send()
    _mark_group_context_answered(frame)
    frame.stage(PipelineStage.ERROR)
    if frame.post_gate_callback is not None:
        await frame.post_gate_callback(phase="post_gate:error")
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )
    frame.turn_finished = True


def _frame_is_current(frame: TurnFrame) -> bool:
    check = getattr(frame, "is_current_turn", None)
    return bool(check()) if callable(check) else True


__all__ = [
    "handle_pipeline_cancelled",
    "handle_pipeline_error",
    "prepare_plugin_fallback_chat_context",
    "remember_target_resolution",
    "stage_chat_capability_hint",
    "stage_chat_run",
    "stage_current_user",
    "stage_dialogue_state",
    "stage_event_context",
    "stage_identity",
    "stage_memory",
    "stage_persist",
    "stage_scratchpad",
    "stage_send",
    "stage_thread_context",
]
