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
from .chat_handler import (
    normalize_ai_reply_text,
    replace_mention_ids_with_names,
)
from .chat_runtime import ChatRuntime, strip_dialogue_state_context
from .config import (
    get_chat_history_limit,
    get_reply_delivery_interval_settings,
    get_reply_delivery_settings,
    reply_to_trigger_message_enabled,
)
from .context_budget import ChatContextBundle, ChatContextSection
from .context_packer import DialogueContextPack
from .event_context import ChatInterEventContext, build_event_context
from .event_runtime import (
    apply_runtime_plugin_overrides,
    event_adapter_name,
    event_is_private,
    event_type_name,
)
from .event_signals import get_event_signal
from .execution_observer import (
    EXECUTION_REASON_CANCELLED,
    EXECUTION_REASON_ERROR,
    ExecutionObservation,
    record_execution_observation,
    start_execution_observation,
)
from .feedback import FeedbackKind, FeedbackStore
from .group_turn_context import (
    consume_group_turn_context,
    snapshot_group_turn_records,
)
from .history_policy import get_durable_history_summary_cursor
from .intent_classifier import classify_message_intent
from .intervention_router import InterventionDecision, decide_intervention
from .llm_compat import LLMContentPart, LLMMessage
from .main_request_models import (
    MainRequestOutput,
    MainRequestResult,
    MainRequestTimelineItem,
)
from .memory import _chat_memory
from .memory_writer import MemoryWriteContext, MemoryWriter
from .models.chat_history import ChatInterChatHistory
from .models.pydantic_models import PluginKnowledgeBase
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
)
from .person_candidates import (
    TurnPersonCandidateLedger,
    retrieve_person_candidates,
)
from .person_registry import (
    PersonProfile,
    RelevantPerson,
    get_person_profile,
    resolve_relevant_people,
    upsert_seen_person,
)
from .plugin_registry import (
    PluginSelectionContext,
)
from .reaction_delivery import reaction_message, validated_reaction_path
from .reaction_models import ReactionAction
from .reply_delivery import (
    DeliveryReceipt,
    build_reply_delivery_plan,
    conversational_send_interval,
)
from .response_defaults import AGENT_ERROR_REPLY_TEXT, EMPTY_REPLY_TEXT
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
from .runtime_result import (
    _fallback_result,
    _finalize_result,
    _is_model_chat_result,
    _sync_visible_chat_result,
    _timeline_memory_text,
    _user_timeline_item,
)
from .target_context import (
    build_mention_profiles,
    extract_pending_entities,
    get_current_group_member_profiles_for_target,
)
from .target_resolver import (
    VerifiedActionTarget,
    resolve_verified_action_target,
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
)
from .utils.unimsg_utils import remove_reply_segment, uni_to_text_with_tags

_MAIN_REQUEST_STAGE = "main_request"
_PERSONA_PROMPT_MARKER = "当前人格设定（来自配置）："
_PERSONA_PROMPT_TAG = "persona_config"


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
    if main_result.output.outcome not in {
        "chat_completed",
        "tool_completed",
        "tool_failed",
    }:
        return
    timeline = [item.to_dict() for item in main_result.timeline]
    if not timeline:
        return
    user_text = uni_to_text_with_tags(user_message)
    response_summary = str(main_result.output.memory_text or "").strip()
    if _is_model_chat_result(main_result):
        response_summary = str(
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
        active_session_id = session_id or _chat_memory.get_session_id(user_id, group_id)
        await ChatInterChatHistory.prune_old_dialogs(
            active_session_id,
            get_chat_history_limit(),
            through_dialog_id=get_durable_history_summary_cursor(active_session_id),
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
    bot: Bot | None = None,
) -> tuple[
    DialogueContextPack,
    PersonProfile | None,
    AddresseeResult,
    VerifiedActionTarget,
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
        alias_candidates=[],
    )
    thread = await resolve_thread_context(
        event_context=event_context,
        addressee=addressee,
    )
    recent_user_ids = _recent_group_participant_ids(event_context)
    roster_profiles = await get_current_group_member_profiles_for_target(
        event_context.group_id,
        bot=bot,
    )
    person_candidate_set = await retrieve_person_candidates(
        group_id=event_context.group_id,
        message_text=event_context.message_text_with_tags,
        roster_profiles=roster_profiles,
        current_user_id=event_context.user_id,
        bot_id=event_context.bot_id,
        mention_user_ids=tuple(event_context.mentioned_user_ids),
        reply_sender_id=(
            event_context.reply.sender_id if event_context.reply is not None else None
        ),
        thread_user_ids=thread.participants,
        recent_user_ids=recent_user_ids,
        current_speaker_profile=speaker_profile,
    )
    person_candidate_ledger = TurnPersonCandidateLedger(person_candidate_set)
    relevant_people = await resolve_relevant_people(
        group_id=event_context.group_id,
        message_text="",
        speaker_profile=speaker_profile,
        bot_id=event_context.bot_id,
        mention_user_ids=tuple(event_context.mentioned_user_ids),
        reply_sender_id=event_context.reply.sender_id
        if event_context.reply is not None
        else None,
        thread_user_ids=thread.participants,
        recent_user_ids=recent_user_ids,
        entity_hints=(),
        alias_candidates=[],
    )
    relevant_people = _merge_person_candidates(
        person_candidate_set.candidates,
        relevant_people,
    )
    verified_action_target = resolve_verified_action_target(
        event_context=event_context,
        addressee=addressee,
        speaker_profile=speaker_profile,
        reply_has_image=bool(
            event_context.reply and extract_image_tokens(event_context.reply.text)
        ),
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
            entity_hints=thread.entity_hints,
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
        person_candidate_ledger=person_candidate_ledger,
    )
    return (
        pack,
        speaker_profile,
        addressee,
        verified_action_target,
        thread,
        intervention,
    )


def _merge_person_candidates(
    candidates,
    relevant_people: tuple[RelevantPerson, ...],
) -> tuple[RelevantPerson, ...]:
    merged: list[RelevantPerson] = []
    seen: set[str] = set()
    for candidate in candidates:
        user_id = str(candidate.profile.user_id or "").strip()
        if not user_id or user_id in seen:
            continue
        seen.add(user_id)
        merged.append(
            RelevantPerson(
                profile=candidate.profile,
                reason="person_candidate",
                confidence=candidate.score,
                matched_alias=candidate.matched_alias,
            )
        )
    for person in relevant_people:
        user_id = str(person.profile.user_id or "").strip()
        if not user_id or user_id in seen:
            continue
        seen.add(user_id)
        merged.append(person)
    return tuple(merged[:8])


def _recent_group_participant_ids(
    event_context: ChatInterEventContext,
) -> tuple[str, ...]:
    records = snapshot_group_turn_records(
        group_id=event_context.group_id,
        current_user_id=event_context.user_id,
        current_message_text=event_context.message_text_with_tags,
        current_message_id=str(event_context.event_id or ""),
        limit=12,
    )
    excluded = {
        str(event_context.user_id or "").strip(),
        str(event_context.bot_id or "").strip(),
    }
    recent: list[str] = []
    for record in reversed(records):
        user_id = str(record.user_id or "").strip()
        if not user_id or user_id in excluded or user_id in recent:
            continue
        recent.append(user_id)
        if len(recent) >= 4:
            break
    return tuple(recent)


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
        prompt_tokens=(
            budget_controller.prompt_tokens if budget_controller is not None else 0
        ),
        cached_prompt_tokens=(
            budget_controller.cached_prompt_tokens
            if budget_controller is not None
            else 0
        ),
        cache_observed_prompt_tokens=(
            budget_controller.cache_observed_prompt_tokens
            if budget_controller is not None
            else 0
        ),
        cache_unknown_prompt_tokens=(
            budget_controller.cache_unknown_prompt_tokens
            if budget_controller is not None
            else 0
        ),
        cache_observed_model_calls=(
            budget_controller.cache_observed_model_calls
            if budget_controller is not None
            else 0
        ),
        cache_unknown_model_calls=(
            budget_controller.cache_unknown_model_calls
            if budget_controller is not None
            else 0
        ),
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


def _append_frame_context_section(
    frame: TurnFrame,
    *,
    name: str,
    section: str,
) -> None:
    bundle = frame.context_bundle or ChatContextBundle()
    updated = bundle.with_text(name, section)
    if updated == bundle:
        return
    frame.context_bundle = updated
    rendered = updated.render()
    frame.context_xml = rendered
    frame.enriched_context_xml = rendered


async def stage_identity(
    *,
    frame: TurnFrame,
    event: Event,
) -> None:
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
    if event_context.turn_priority:
        frame.update_tags(
            turn_priority=event_context.turn_priority,
        )
    mention_profiles = await build_mention_profiles(
        frame.group_id,
        event_context.message_text_with_tags,
        bot_id=frame.bot_id,
        bot=bot,
    )
    frame.mention_profiles = mention_profiles
    if frame.event_message is not None:
        current_message = uni_to_text_with_tags(
            remove_reply_segment(frame.event_message)
        )
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
    bot: Bot | None = None,
) -> None:
    if frame.event_context is None:
        raise RuntimeError("missing event context")
    (
        dialogue_context_pack,
        _speaker_profile,
        addressee_result,
        verified_action_target,
        thread_context,
        intervention_decision,
    ) = await _build_dialogue_context_pack(
        event_context=frame.event_context,
        mention_profiles=frame.mention_profiles,
        bot=bot,
    )
    frame.dialogue_context_pack = dialogue_context_pack
    frame.person_candidate_ledger = dialogue_context_pack.person_candidate_ledger
    frame.addressee_result = addressee_result
    frame.verified_action_target = verified_action_target
    frame.thread_context = thread_context
    frame.intervention_decision = intervention_decision
    frame.update_tags(
        addressee_source=addressee_result.source,
        addressee_confidence=f"{addressee_result.confidence:.2f}",
        verified_target_source=verified_action_target.source,
        verified_target_confidence=f"{verified_action_target.confidence:.2f}",
        thread_id=thread_context.thread_id,
        intervention=intervention_decision.action,
        intervention_reason=intervention_decision.reason,
        **(
            dialogue_context_pack.person_candidate_ledger.snapshot()
            if dialogue_context_pack.person_candidate_ledger is not None
            else {}
        ),
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
    memory_message = frame.uni_msg or frame.raw_message
    memory_dialogue_state = ChatRuntime.memory_dialogue_state(frame)
    context_sections: list[ChatContextSection] = []
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
        reply_context=(
            frame.event_context.reply if frame.event_context is not None else None
        ),
        context_sections_out=context_sections,
        recent_reactions_out=frame.recent_reactions,
    )
    frame.chat_memory_layered = getattr(
        frame.dialogue_context_pack,
        "layered_memory",
        None,
    )
    frame.dialogue_state = dialogue_state
    frame.context_bundle = ChatContextBundle(tuple(context_sections))
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
    cached_plain_text: str | None = None,
) -> None:
    if frame.event_message is not None:
        current_message = uni_to_text_with_tags(
            remove_reply_segment(frame.event_message)
        )
    elif frame.uni_msg:
        current_msg = remove_reply_segment(frame.uni_msg)
        current_message = uni_to_text_with_tags(current_msg)
    elif cached_plain_text:
        current_message = cached_plain_text.strip()
    else:
        current_message = frame.raw_message.strip()
    frame.current_message = current_message
    frame.stage(PipelineStage.INTENT_BUDGET)


async def stage_chat_capability_hint(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    cached_plain_text: str | None = None,
) -> None:
    frame.knowledge_base = _empty_plugin_knowledge_base()
    frame.stage(PipelineStage.KNOWLEDGE)
    await _prepare_current_message_context(
        frame=frame,
        cached_plain_text=cached_plain_text,
    )
    await _prepare_lightweight_chat_route(
        frame=frame,
        bot=bot,
        event=event,
    )
    frame.stage(PipelineStage.CAPABILITY_HINT)


def _empty_plugin_knowledge_base() -> PluginKnowledgeBase:
    return PluginKnowledgeBase(plugins=[], user_role="普通用户")


async def _prepare_lightweight_chat_route(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
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
    frame.stage(PipelineStage.ROUTE_SELECTION)
    route_message = frame.route_message
    frame.route_message = route_message
    verified_target = getattr(frame, "verified_action_target", None)
    has_verified_target = bool(getattr(verified_target, "is_resolved", False))
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
        has_verified_target=has_verified_target,
        verified_target_source=(
            str(getattr(verified_target, "source", "") or "")
            if has_verified_target
            else ""
        ),
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
    return str(frame.system_prompt or "").strip()


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
    persona, rest = _extract_tag_blocks(text, _PERSONA_PROMPT_TAG)
    if persona:
        return persona.strip(), rest.strip()
    if _PERSONA_PROMPT_MARKER not in text:
        return "", text
    marker = text.find(_PERSONA_PROMPT_MARKER)
    boundary = text.rfind("\n\n")
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
    current_user_section = user_text
    if frame.context_xml:
        current_user_payload = _xml_escape(user_text, quote=False)
        current_user_section = (
            f"<current_user_message>{current_user_payload}</current_user_message>"
        )
        user_text = f"{frame.context_xml}\n\n{current_user_section}"
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
    del frame
    return True


async def _prepare_chat_multimodal(frame: TurnFrame) -> None:
    image_parts = list(getattr(frame, "image_parts", []) or [])
    if not image_parts:
        return
    limited = image_parts[:MAX_CHAT_IMAGE_PARTS]
    frame.image_parts = limited
    if len(image_parts) > len(limited):
        _append_frame_context_section(
            frame,
            name="current_media",
            section=(
                "<image_context>"
                f"当前消息包含 {len(image_parts)} 张图片，"
                f"已只传入前 {len(limited)} 张。"
                "</image_context>"
            ),
        )
    frame.update_tags(
        image_mode="candidate_routed",
        image_count=len(image_parts),
        image_passed=len(limited),
    )


async def stage_current_user(
    *,
    frame: TurnFrame,
    message=None,
) -> None:
    image_extraction = await _extract_current_turn_images(frame=frame, message=message)
    image_parts = list(image_extraction.image_parts)
    frame.current_image_parts = list(image_parts)
    frame.image_parts = image_parts
    if image_extraction.context_xml:
        _append_frame_context_section(
            frame,
            name="current_media",
            section=image_extraction.context_xml,
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
            _append_frame_context_section(
                frame,
                name="current_media",
                section=reply_extraction.context_xml,
            )
        if frame.reply_images_data:
            logger.debug(f"回复链提取到 {len(frame.reply_images_data)} 张图片")
    frame.stage(PipelineStage.CURRENT_USER)
    frame.enriched_context_xml = frame.context_xml
    if frame.intent_profile is None and not frame.allow_plugin_tools:
        raise RuntimeError("missing intent profile for main request")
    dialogue_plan = frame.dialogue_plan
    dialogue_state = frame.dialogue_state
    if dialogue_plan is None or dialogue_state is None:
        raise RuntimeError("missing chat runtime profile")
    if frame.context_bundle is not None:
        frame.context_bundle = frame.context_bundle.transform_text(
            strip_dialogue_state_context
        )
        frame.context_xml = frame.context_bundle.render()
        frame.enriched_context_xml = frame.context_xml
    chat_prompt = ChatRuntime.build_prompt_context(
        frame,
        base_context_xml=frame.enriched_context_xml,
    )
    if chat_prompt.context_sections:
        for section in chat_prompt.context_sections:
            _append_frame_context_section(
                frame,
                name="guidance",
                section=section,
            )
        frame.enriched_context_xml = frame.context_xml
    else:
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
    extraction = await extract_chat_images_from_message(source_for_media)
    return ChatImageExtraction(
        image_parts=list(extraction.image_parts),
        context_xml=extraction.context_xml,
        original_count=extraction.original_count,
        skipped_oversized=extraction.skipped_oversized,
    )


async def stage_scratchpad(
    *,
    frame: TurnFrame,
) -> None:
    if frame.dialogue_plan is None:
        raise RuntimeError("missing dialogue plan for main request")
    frame.context_xml = frame.enriched_context_xml
    frame.enriched_context_xml = frame.context_xml
    if _should_build_agent_messages(frame):
        await _prepare_chat_multimodal(frame)
    frame.agent_messages = (
        _build_agent_messages(frame) if _should_build_agent_messages(frame) else []
    )
    frame.stage(PipelineStage.SCRATCHPAD)


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
                reply=AGENT_ERROR_REPLY_TEXT,
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

    async def _reply_hook(reply_text: str) -> str:
        finalized_text = normalize_ai_reply_text(reply_text)
        finalized_text = replace_mention_ids_with_names(
            finalized_text,
            frame.mention_name_map,
        )
        return finalized_text

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
        if (
            not normalize_message_text(reply_text)
            and not main_result.output.nontext_delivery
        ):
            reply_text = EMPTY_REPLY_TEXT
        model_chat_completed = _is_model_chat_result(main_result)
        quality_enabled = bool(
            model_chat_completed
            and ChatRuntime.isolation_for_frame(frame).allow_quality_judge
        )
        if quality_enabled:
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
            if quality.revised_text:
                reply_text = normalize_reply_text(quality.revised_text) or reply_text
            schedule_response_quality_eval(
                quality_result=quality,
                final_text=reply_text,
                original_message=frame.current_message,
                scenario=frame.scenario,
                session_key=frame.session_key,
                trace_id=str(frame.trace.tags.get("message_id", "")),
            )
        if model_chat_completed:
            main_result = _sync_visible_chat_result(
                main_result,
                final_text=reply_text,
            )
            frame.main_result = main_result
        if normalize_message_text(reply_text):
            envelope.add(ChannelName.FINAL, reply_text)
    frame.final_envelope = envelope


async def _send_delayed_reply_status(frame: TurnFrame) -> None:
    await asyncio.sleep(15.0)
    if _frame_is_current(frame) and not frame.turn_finished:
        await _send_frame_message(
            frame,
            MessageUtils.build_message("正在回复..."),
        )


async def _send_frame_message(
    frame: TurnFrame,
    message: Any,
    *,
    reply_to: bool = False,
) -> None:
    if frame.event is None or frame.bot is None:
        raise RuntimeError("missing pipeline send target")
    kwargs: dict[str, Any] = {
        "target": frame.event,
        "bot": frame.bot,
    }
    if reply_to:
        kwargs["reply_to"] = True
    await message.send(**kwargs)


def _delivery_receipt(frame: TurnFrame) -> DeliveryReceipt | None:
    receipt = getattr(frame, "delivery_receipt", None)
    return receipt if isinstance(receipt, DeliveryReceipt) else None


def _should_segment_conversational_reply(main_result: MainRequestResult) -> bool:
    return bool(
        main_result.output.outcome == "chat_completed"
        and not main_result.executions
        and not main_result.tool_results
    )


def _should_quote_first_segment(frame: TurnFrame) -> bool:
    return reply_to_trigger_message_enabled()


async def _persist_partial_delivery(frame: TurnFrame) -> bool:
    receipt = _delivery_receipt(frame)
    if (
        receipt is None
        or receipt.complete
        or receipt.delivered_count <= 0
        or bool(getattr(frame, "delivery_persisted", False))
    ):
        return False
    main_result = getattr(frame, "main_result", None)
    if not isinstance(main_result, MainRequestResult):
        return False
    delivered_text = receipt.delivered_text
    if not delivered_text or main_result.output.outcome != "chat_completed":
        return False
    visible_result = _sync_visible_chat_result(
        main_result,
        final_text=delivered_text,
    )
    try:
        await _persist_message_timeline(
            main_result=visible_result,
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
        logger.warning(f"ChatInter partial reply persistence failed: {exc}")
        frame.update_tags(partial_persistence_error=type(exc).__name__)
        return False
    frame.delivery_persisted = True
    frame.update_tags(
        delivery_state="partial",
        delivery_segments=float(receipt.delivered_count),
    )
    return True


async def stage_persist(frame: TurnFrame) -> None:
    main_result = frame.main_result
    if main_result is None:
        raise RuntimeError("missing main request result")
    envelope = frame.final_envelope or TurnChannelEnvelope()
    receipt = _delivery_receipt(frame)
    if receipt is not None and receipt.delivered_count > 0 and not receipt.complete:
        await _persist_partial_delivery(frame)
        frame.update_tags(path="partial_delivery", outcome="partial_delivery")
        frame.stage(PipelineStage.PERSIST)
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
    if not _frame_is_current(frame) and not bool(
        getattr(frame, "delivery_succeeded", False)
    ):
        frame.update_tags(path="superseded", outcome="superseded")
        frame.turn_finished = True
        return
    if not main_result.output.should_send:
        log_turn_channels(envelope)
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
                session_id=getattr(frame, "session_key", None),
            )
        except Exception as exc:
            logger.warning(f"ChatInter plugin result persistence failed: {exc}")
            frame.update_tags(persistence_error=type(exc).__name__)
        frame.delivery_persisted = True
        if main_result.output.outcome in {"tool_completed", "tool_failed"}:
            _mark_group_context_answered(frame)
        frame.stage(PipelineStage.PERSIST)
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
    frame.delivery_persisted = True
    frame.stage(PipelineStage.PERSIST)
    persisted_reply_text = (
        main_result.output.memory_text or envelope.final
        if main_result.output.nontext_delivery
        else envelope.final
    )
    if main_result.output.record_chat_feedback:
        FeedbackStore.record_chat(
            session_id=frame.session_key,
            kind=cast(FeedbackKind, main_result.output.feedback_kind),
            message_text=frame.current_message,
            reply_text=persisted_reply_text,
            weight=0.2,
        )
    _persist_plain_chat_dialogue_state(
        frame=frame,
        reply_text=persisted_reply_text,
    )
    _record_delivered_result_observation(frame, main_result)
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )
    frame.turn_finished = True


def _record_delivered_result_observation(
    frame: TurnFrame,
    main_result: MainRequestResult,
) -> None:
    tool_outcome = normalize_message_text(main_result.output.tool_outcome).casefold()
    if tool_outcome:
        _tag_execution_observation(
            frame.trace,
            record_execution_observation(
                action="execute",
                success=tool_outcome == "executed",
                reason=tool_outcome,
                session_id=frame.session_key,
                message_preview=frame.current_message,
            ),
        )
        frame.update_tags(fallback_chat_delivered="1")
        return
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
    candidate_reaction = getattr(frame, "reaction_action", None)
    reaction_action = (
        candidate_reaction if isinstance(candidate_reaction, ReactionAction) else None
    )
    reaction_path = (
        await validated_reaction_path(reaction_action)
        if reaction_action is not None
        else None
    )
    if reaction_action is not None and reaction_path is None:
        frame.update_tags(
            reaction_delivery="invalid",
            reaction_delivery_result="validation_failed",
            reaction_abstain_stage="validation_failed",
        )
        if not normalize_message_text(envelope.final):
            main_result = _apply_reaction_text_fallback(
                frame=frame,
                main_result=main_result,
                envelope=envelope,
                action=reaction_action,
            )
        else:
            main_result = replace(
                main_result,
                timeline=_without_reaction_history(main_result.timeline),
                output=replace(
                    main_result.output,
                    memory_text=envelope.final,
                    nontext_delivery=False,
                ),
            )
            frame.main_result = main_result
        reaction_action = None
        frame.reaction_action = None
    delivery_mode, max_chars, max_segments = get_reply_delivery_settings()
    plan = build_reply_delivery_plan(
        envelope.final,
        conversational=(
            delivery_mode == "streaming"
            and _should_segment_conversational_reply(main_result)
        ),
        hard_limit=max_chars,
        max_segments=max_segments,
        attachment_count=1 if reaction_action is not None else 0,
    )
    frame.delivery_plan = plan
    interval_settings = (
        get_reply_delivery_interval_settings() if plan.conversational else None
    )
    delivered: list[str] = []
    delivered_attachments = 0
    frame.delivery_receipt = DeliveryReceipt.from_plan(plan)
    quote_first = _should_quote_first_segment(frame)
    for index, segment in enumerate(plan.segments):
        if not _frame_is_current(frame):
            break
        message = MessageUtils.build_message(segment)
        if quote_first and index == 0:
            await _send_frame_message(frame, message, reply_to=True)
        else:
            await _send_frame_message(frame, message)
        delivered.append(segment)
        frame.delivery_receipt = DeliveryReceipt.from_plan(
            plan,
            delivered,
            delivered_attachments=delivered_attachments,
        )
        if (
            plan.conversational
            and index + 1 < len(plan.segments)
            and _frame_is_current(frame)
            and interval_settings is not None
        ):
            interval_method, interval, log_base = interval_settings
            delay = conversational_send_interval(
                plan.segments[index + 1],
                method=interval_method,
                interval=interval,
                log_base=log_base,
            )
            if delay > 0:
                await asyncio.sleep(delay)
    if (
        reaction_action is not None
        and reaction_path is not None
        and _frame_is_current(frame)
    ):
        try:
            await _send_frame_message(
                frame,
                reaction_message(reaction_path),
                reply_to=quote_first and not delivered,
            )
            delivered_attachments = 1
            frame.update_tags(
                reaction_delivery="complete",
                reaction_delivery_result="complete",
                reaction_abstain_stage="",
            )
            main_result = replace(
                main_result,
                output=replace(
                    main_result.output,
                    memory_text=reaction_action.memory_text,
                    nontext_delivery=True,
                ),
            )
            frame.main_result = main_result
            frame.delivery_receipt = DeliveryReceipt.from_plan(
                plan,
                delivered,
                delivered_attachments=delivered_attachments,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"ChatInter reaction delivery failed: {exc}")
            frame.update_tags(
                reaction_delivery="failed",
                reaction_delivery_result="failed",
                reaction_abstain_stage="delivery_failed",
            )
            if not delivered:
                main_result = _apply_reaction_text_fallback(
                    frame=frame,
                    main_result=main_result,
                    envelope=envelope,
                    action=reaction_action,
                )
                fallback_plan = build_reply_delivery_plan(
                    envelope.final,
                    conversational=False,
                    hard_limit=max_chars,
                    max_segments=max_segments,
                )
                fallback_message = MessageUtils.build_message(envelope.final)
                await _send_frame_message(
                    frame,
                    fallback_message,
                    reply_to=quote_first,
                )
                delivered = [envelope.final]
                plan = fallback_plan
                frame.delivery_plan = fallback_plan
                frame.delivery_receipt = DeliveryReceipt.from_plan(
                    fallback_plan,
                    delivered,
                )
            else:
                main_result = replace(
                    main_result,
                    timeline=_without_reaction_history(main_result.timeline),
                    output=replace(
                        main_result.output,
                        memory_text=envelope.final,
                        nontext_delivery=False,
                    ),
                )
                frame.main_result = main_result
                frame.reaction_action = None
    receipt = _delivery_receipt(frame)
    frame.delivery_succeeded = bool(receipt and receipt.complete)
    frame.update_tags(
        delivery_state="complete" if frame.delivery_succeeded else "partial",
        delivery_planned=float(len(plan.segments) + plan.planned_attachments),
        delivery_segments=float(receipt.delivered_count if receipt else 0),
        delivery_conversational="1" if plan.conversational else "0",
    )
    if receipt and receipt.delivered_count > 0:
        frame.stage(PipelineStage.SEND)
    if frame.delivery_succeeded:
        _mark_group_context_answered(frame)


def _apply_reaction_text_fallback(
    *,
    frame: TurnFrame,
    main_result: MainRequestResult,
    envelope: TurnChannelEnvelope,
    action: ReactionAction,
) -> MainRequestResult:
    fallback = normalize_reply_text(action.fallback_text) or EMPTY_REPLY_TEXT
    envelope.add(ChannelName.FINAL, fallback)
    timeline = list(_without_reaction_history(main_result.timeline))
    if not any(
        item.role == "assistant"
        and item.kind == "final_output"
        and normalize_message_text(item.content) == normalize_message_text(fallback)
        for item in timeline
    ):
        timeline.append(
            MainRequestTimelineItem(
                role="assistant",
                kind="final_output",
                content=fallback,
                metadata={"assistant_history": True},
            )
        )
    updated = replace(
        main_result,
        timeline=tuple(timeline),
        output=replace(
            main_result.output,
            final_text=fallback,
            memory_text=fallback,
            should_send=True,
            nontext_delivery=False,
        ),
    )
    frame.main_result = updated
    frame.reaction_action = None
    return updated


def _without_reaction_history(
    timeline: tuple[MainRequestTimelineItem, ...],
) -> tuple[MainRequestTimelineItem, ...]:
    return tuple(
        item
        for item in timeline
        if not (item.role == "assistant" and item.kind == "reaction_output")
    )


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
        get_event_signal(
            getattr(frame, "event", None),
            "_chatinter_group_context_record_id",
            "",
        )
        or getattr(getattr(frame, "event_context", None), "event_id", "")
        or frame.trace.tags.get("message_id", "")
        or getattr(getattr(frame, "event", None), "message_id", "")
        or getattr(getattr(frame, "event", None), "event_id", "")
    )
    if message_id:
        consume_group_turn_context(frame.group_id, message_id)


async def handle_pipeline_cancelled(frame: TurnFrame) -> None:
    await _persist_partial_delivery(frame)
    reroute_receipt = dict(getattr(frame, "cancelled_reroute_receipt", None) or {})
    execution_uncertain = bool(reroute_receipt.get("execution_uncertain"))
    if execution_uncertain:
        await _persist_cancelled_uncertain_execution(frame, reroute_receipt)
    _mark_group_context_answered(frame)
    frame.update_tags(
        path="cancelled",
        outcome="execution_uncertain" if execution_uncertain else "cancelled",
        plugin_outcome="uncertain" if execution_uncertain else None,
    )
    _tag_execution_observation(
        frame.trace,
        record_execution_observation(
            action="execute" if execution_uncertain else "chat",
            success=False,
            reason=(
                "execution_uncertain"
                if execution_uncertain
                else EXECUTION_REASON_CANCELLED
            ),
            plugin_module=str(reroute_receipt.get("plugin_module", "") or ""),
            plugin_name=str(reroute_receipt.get("plugin_name", "") or ""),
            command_id=str(reroute_receipt.get("command_id", "") or ""),
            command=str(reroute_receipt.get("command", "") or ""),
            session_id=frame.session_key,
            message_preview=frame.current_message,
        ),
    )
    group_name = frame.group_id or "private"
    logger.debug(f"ChatInter 处理被取消: user={frame.user_id}, group={group_name}")
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
    await _persist_partial_delivery(frame)
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
    logger.error(f"ChatInter 处理失败: {error}")
    receipt = _delivery_receipt(frame)
    delivered_any = bool(receipt and receipt.delivered_count > 0)
    if (
        not delivered_any
        and not bool(getattr(frame, "delivery_succeeded", False))
        and _frame_is_current(frame)
    ):
        try:
            await _send_frame_message(frame, MessageUtils.build_failure_message())
        except Exception as send_error:
            logger.warning(f"ChatInter failure reply delivery failed: {send_error}")
            frame.update_tags(failure_delivery_error=type(send_error).__name__)
    _mark_group_context_answered(frame)
    frame.stage(PipelineStage.ERROR)
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )
    frame.turn_finished = True


def complete_suppressed_turn(frame: TurnFrame, *, reason: str) -> None:
    frame.update_tags(path="suppressed", outcome=reason)
    _mark_group_context_answered(frame)
    frame.stage(PipelineStage.ROUTE)
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )
    frame.turn_finished = True


async def _persist_cancelled_uncertain_execution(
    frame: TurnFrame,
    receipt: dict[str, Any],
) -> None:
    command = normalize_message_text(str(receipt.get("command", "") or ""))
    command_id = normalize_message_text(str(receipt.get("command_id", "") or ""))
    plugin_name = normalize_message_text(str(receipt.get("plugin_name", "") or ""))
    plugin_module = normalize_message_text(str(receipt.get("plugin_module", "") or ""))
    task_text = normalize_message_text(
        str(receipt.get("task_text", "") or frame.current_message)
    )
    output = {
        "status": "uncertain",
        "ok": False,
        "execution_uncertain": True,
        "plugin_execution": True,
        "executed": False,
        "command_id": command_id,
        "rendered_command": command,
        "matched_plugin": plugin_name or plugin_module,
        "task_text": task_text,
        "messages_sent": [],
        "error": "reroute_cancelled",
        "retryable": False,
        "need_continue": False,
    }
    execution = {
        "tool_kind": "native_command",
        "status": "uncertain",
        "plugin_outcome": "uncertain",
        "command_id": command_id,
    }
    timeline = (
        _user_timeline_item(frame.current_message),
        MainRequestTimelineItem(
            role="tool",
            kind="tool_result",
            metadata={"output": output, "execution": execution},
        ),
    )
    result = MainRequestResult(
        decision=NativeRouteDecision(
            action="execute",
            confidence=1.0,
            reason="reroute_cancelled_uncertain",
            plugin_name=plugin_name,
            plugin_module=plugin_module,
            command_id=command_id,
            command=command,
        ),
        route_result=None,
        report=frame.route_report or NativeRouteReport(helper_mode=False),
        timeline=timeline,
        output=MainRequestOutput(
            final_text="",
            memory_text=_timeline_memory_text(timeline),
            should_send=False,
            outcome="tool_failed",
            feedback_kind="tool_failed",
            record_chat_feedback=False,
            observation_reason="execution_uncertain",
            tool_outcome="uncertain",
        ),
    )
    try:
        await _persist_message_timeline(
            main_result=result,
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
        logger.warning(f"ChatInter uncertain cancellation persistence failed: {exc}")
        frame.update_tags(uncertain_persistence_error=type(exc).__name__)


def _frame_is_current(frame: TurnFrame) -> bool:
    check = getattr(frame, "is_current_turn", None)
    return bool(check()) if callable(check) else True


__all__ = [
    "complete_suppressed_turn",
    "handle_pipeline_cancelled",
    "handle_pipeline_error",
    "stage_chat_capability_hint",
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
