"""Unified pipeline stage: one agent turn for chat and plugin invocation.

Replaces the old two-path shape (structured plugin router + separate chat
degrade call).  Both group and private turns run the same UnifiedChatAgent;
plugin tools are attached when the scenario allows them and the command
snapshot is non-empty.
"""

from __future__ import annotations

import asyncio
import time

from nonebot.adapters import Bot, Event

from zhenxun.utils.utils import get_entity_ids

from .agents.core import AgentResult, UnifiedChatRequest
from .agents.unified_chat_agent import UnifiedChatAgent
from .candidate_exposure import CandidateExposureLedger
from .chat_handler import (
    RerouteExecutionResult,
    consume_reroute_cancellation_receipt,
)
from .group_plugin_flow import _execute_native_tool_route
from .gscore_adapter import get_gscore_adapter
from .mixed_tool_catalog import build_mixed_tool_catalog
from .native_executor import (
    NativeCommandExecutionContext,
    NativeToolExecutionResult,
    NativeValidatedRoute,
)
from .native_route import NativeRouteReport
from .pipeline_stages import (
    _build_agent_stage_hooks,
    _run_direct_agent_turn,
    _send_delayed_reply_status,
    _set_agent_stage_result,
)
from .plugin_registry import PluginRegistry
from .plugin_skill_index import build_plugin_skill_index, log_skill_debug_once
from .reaction_tools import build_reaction_tools
from .route_text import is_usage_question, normalize_message_text
from .strict_identity import resolve_strict_command_candidates
from .turn_frame import PipelineStage, TurnFrame


async def stage_unified_run(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    route_completed_hook, reply_hook = _build_agent_stage_hooks(
        frame=frame,
    )
    message_text = normalize_message_text(frame.route_message or frame.current_message)
    report = NativeRouteReport(helper_mode=is_usage_question(message_text))
    command_context: NativeCommandExecutionContext | None = None
    tools = None
    tool_catalog = None
    command_candidate_text = ""
    available_snapshots = list(frame.command_tools or [])
    candidates = []
    exposure_ledger = CandidateExposureLedger()
    reaction_state, reaction_tools = await build_reaction_tools(
        session_id=frame.session_key,
        recent_reactions=tuple(getattr(frame, "recent_reactions", ()) or ()),
    )
    frame.reaction_turn_state = reaction_state
    if reaction_tools:
        tools = dict(reaction_tools)
        frame.update_tags(reaction_tools=float(len(reaction_tools)))
    gscore_tool_started = time.perf_counter()
    gscore_route_result = getattr(frame, "gscore_route_result", None)
    gscore_tools = (
        await get_gscore_adapter().build_tools(
            frame,
            route_result=gscore_route_result,
            exposure_ledger=exposure_ledger,
        )
        if gscore_route_result is not None
        else {}
    )
    frame.update_tags(
        gscore_tool_build_ms=(time.perf_counter() - gscore_tool_started) * 1000,
    )
    if gscore_tools:
        tools = {**(tools or {}), **gscore_tools}
        frame.update_tags(
            gscore_capabilities=float(
                sum(tool.capability_count for tool in gscore_tools.values())
            ),
            gscore_skills=float(len(gscore_tools)),
        )
    knowledge_base = frame.knowledge_base
    if (
        frame.allow_plugin_tools
        and knowledge_base is not None
        and knowledge_base.plugins
    ):
        skill_snapshots = PluginRegistry.build_command_tool_snapshots(
            knowledge_base,
            selection_context=None,
        )
        skill_index = build_plugin_skill_index(knowledge_base, skill_snapshots)
    else:
        skill_snapshots = []
        skill_index = None
    if skill_index is not None and skill_index.skills:
        submitted_action_keys: set[str] = set()
        person_candidate_ledger = getattr(frame, "person_candidate_ledger", None)
        candidates = list(
            resolve_strict_command_candidates(
                message_text,
                available_snapshots,
                trusted_person_spans=(
                    person_candidate_ledger.trusted_identity_spans()
                    if person_candidate_ledger is not None
                    else ()
                ),
            )
        )
        report.lexical_candidates = len(candidates)
        report.note_tool_pool(len(candidates))
        report.note_prompt_exposure(candidates)

        async def execute_native_route(
            validated: NativeValidatedRoute,
            route_report: NativeRouteReport,
        ) -> NativeToolExecutionResult:
            try:
                return await _execute_native_tool_route(
                    bot=bot,
                    event=event,
                    trace=frame.trace,
                    validated=validated,
                    knowledge_plugins=knowledge_base.plugins,
                    current_message=message_text,
                    user_id=frame.user_id,
                    group_id=frame.group_id,
                    session_id=frame.session_key,
                    has_reply=frame.has_reply,
                    extra_image_segments=frame.reply_image_segments_for_reroute,
                    reply_image_count=frame.reply_image_count,
                    route_report=route_report,
                    mention_profiles=frame.mention_profiles,
                    submitted_action_keys=submitted_action_keys,
                )
            except asyncio.CancelledError:
                receipt = consume_reroute_cancellation_receipt()
                if receipt is not None and receipt.execution_uncertain:
                    _project_cancelled_reroute_receipt(
                        frame=frame,
                        receipt=receipt,
                        validated=validated,
                        message_text=message_text,
                    )
                raise

        dialogue_context_pack = getattr(frame, "dialogue_context_pack", None)
        command_context = NativeCommandExecutionContext(
            candidates=candidates,
            has_reply=frame.has_reply,
            report=report,
            route_executor=execute_native_route,
            message_text=message_text,
            event_target_hint=_event_target_hint(frame=frame, bot=bot),
            event_target_ids=_event_target_ids(frame=frame, bot=bot),
            target_refs=(
                dialogue_context_pack.action_target_refs()
                if dialogue_context_pack is not None
                else {}
            ),
            person_candidate_ledger=getattr(
                frame,
                "person_candidate_ledger",
                None,
            ),
            retrieval_context=_retrieval_context(frame),
        )
        frame.chat_tool_exposure_state = "plugin_tools_exposed"
        log_skill_debug_once(skill_index)
        frame.update_tags(
            skill_commands=float(skill_index.command_count),
            skill_count=float(len(skill_index.skills)),
            strict_identity_match_count=float(len(candidates)),
        )

    if tools or (skill_index is not None and skill_index.skills):
        local_tools_enabled = skill_index is not None and bool(skill_index.skills)
        tool_catalog = build_mixed_tool_catalog(
            skill_index=skill_index if local_tools_enabled else None,
            known_commands=skill_snapshots if local_tools_enabled else [],
            available_commands=available_snapshots if local_tools_enabled else [],
            initial_candidates=candidates if local_tools_enabled else [],
            knowledge_base=knowledge_base if local_tools_enabled else None,
            session_id=frame.session_key,
            command_context=command_context if local_tools_enabled else None,
            exposure_ledger=exposure_ledger,
        )
        frame.chat_tool_exposure_state = "plugin_tools_exposed"

    history_scope = _bound_history_scope(frame)
    request = UnifiedChatRequest(
        message_text=message_text,
        session_key=frame.session_key,
        budget_controller=frame.budget_controller,
        messages=list(frame.agent_messages or []),
        report=report,
        scenario=frame.scenario,
        user_id=history_scope["user_id"],
        group_id=history_scope["group_id"],
        bot_id=history_scope["bot_id"],
        platform=history_scope["platform"],
        channel_id=history_scope["channel_id"],
        command_candidate_text=command_candidate_text,
        tools=tools,
        tool_catalog=tool_catalog,
        command_context=command_context,
        context_bundle=frame.context_bundle,
        context_xml=frame.context_xml,
    )

    agent_result_holder: list[AgentResult] = []

    async def run_agent():
        agent_result = await UnifiedChatAgent().run(request)
        agent_result_holder.append(agent_result)
        return agent_result.to_main_result()

    progress_task = (
        asyncio.create_task(_send_delayed_reply_status(frame))
        if frame.scenario == "private_chat"
        else None
    )
    try:
        main_result = await _run_direct_agent_turn(
            message_text=message_text,
            report=report,
            budget_controller=frame.budget_controller,
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
            run_agent=run_agent,
        )
    finally:
        if progress_task is not None:
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
    if reaction_state is not None:
        frame.reaction_action = reaction_state.action
    _set_agent_stage_result(frame=frame, main_result=main_result)
    if agent_result_holder:
        frame.agent_observations = list(
            getattr(agent_result_holder[-1], "observations", ())
        )
        _tag_agent_observations(frame)
    person_candidate_ledger = getattr(frame, "person_candidate_ledger", None)
    if person_candidate_ledger is not None:
        frame.update_tags(**person_candidate_ledger.snapshot())
    frame.stage(PipelineStage.AGENT_RUN)


def _project_cancelled_reroute_receipt(
    *,
    frame: TurnFrame,
    receipt: RerouteExecutionResult,
    validated: NativeValidatedRoute,
    message_text: str,
) -> None:
    route_result = validated.route_result
    frame.cancelled_reroute_receipt = {
        "status": "uncertain",
        "execution_uncertain": True,
        "execution_started": bool(receipt.execution_started),
        "task_stopped": bool(receipt.task_stopped),
        "trace_id": receipt.trace_id,
        "command": receipt.command,
        "command_id": route_result.command_id if route_result is not None else "",
        "plugin_name": (
            route_result.decision.plugin_name if route_result is not None else ""
        ),
        "plugin_module": (
            route_result.decision.plugin_module if route_result is not None else ""
        ),
        "task_text": (
            validated.task_frame.effective_text
            if validated.task_frame is not None
            else message_text
        ),
    }


def _tag_agent_observations(frame: TurnFrame) -> None:
    if not frame.agent_observations:
        return
    metadata = getattr(frame.agent_observations[-1], "metadata", None)
    if not isinstance(metadata, dict):
        return
    executions = tuple(metadata.get("tool_executions") or ())
    model_requests = tuple(metadata.get("model_requests") or ())
    selected_commands = tuple(
        dict.fromkeys(
            str(item.get("command_id", "") or "")
            for item in executions
            if isinstance(item, dict) and str(item.get("command_id", "") or "")
        )
    )
    exact_identity_ids = tuple(
        str(item or "")
        for item in metadata.get("exact_identity_ids", ())
        if str(item or "")
    )
    available_commands: tuple[str, ...] = ()
    if model_requests and isinstance(model_requests[-1], dict):
        available_commands = tuple(
            str(item or "")
            for key in ("native_command_ids", "indexed_command_ids")
            for item in model_requests[-1].get(key, ())
            if str(item or "")
        )
    frame.update_tags(
        agent_model_requests=float(len(model_requests)),
        agent_tool_executions=float(len(executions)),
        plugin_outcome=str(metadata.get("plugin_outcome", "") or ""),
        plugin_outcome_reason=str(metadata.get("plugin_outcome_reason", "") or ""),
        missing_input_fields="|".join(
            dict.fromkeys(
                str(item or "").strip()
                for item in metadata.get("missing_input_fields", ())
                if str(item or "").strip()
            )
        ),
        failure_layer=str(metadata.get("failure_layer", "") or ""),
        exact_identity_ids="|".join(exact_identity_ids),
        strict_identity_match_modes="|".join(
            str(item or "")
            for item in metadata.get("strict_identity_match_modes", ())
            if str(item or "")
        ),
        exposed_command_ids="|".join(
            str(item or "")
            for item in metadata.get("exposed_command_ids", ())
            if str(item or "")
        ),
        available_command_count=float(len(dict.fromkeys(available_commands))),
        selected_command_ids="|".join(selected_commands),
        selected_skill=str(metadata.get("selected_skill", "") or ""),
        discovery_source=str(metadata.get("discovery_source", "") or ""),
        retrieval_query_count=float(metadata.get("retrieval_query_count", 0) or 0),
        candidate_count=float(metadata.get("candidate_count", 0) or 0),
        candidate_displayed=float(metadata.get("candidate_displayed", 0) or 0),
        candidate_omitted=float(metadata.get("candidate_omitted", 0) or 0),
        candidate_exposure_count=float(
            metadata.get("candidate_exposure_count", 0) or 0
        ),
        selected_command_id=str(metadata.get("selected_command_id", "") or ""),
        selected_capability_id=str(metadata.get("selected_capability_id", "") or ""),
        execution_validation_reason=str(
            metadata.get("execution_validation_reason", "") or ""
        ),
        native_validation_reason=str(
            metadata.get("native_validation_reason", "") or ""
        ),
        argument_validation_error=str(
            metadata.get("argument_validation_error", "") or ""
        ),
        argument_validation_field=str(
            metadata.get("argument_validation_field", "") or ""
        ),
        protocol_argument_retries=float(
            metadata.get("protocol_argument_retries", 0) or 0
        ),
        protocol_format_retries=float(metadata.get("protocol_format_retries", 0) or 0),
        protocol_text_only_retries=float(
            metadata.get("protocol_text_only_retries", 0) or 0
        ),
        protocol_text_suppressed=float(
            metadata.get("protocol_text_suppressed", 0) or 0
        ),
        tool_argument_envelope_repairs=float(
            metadata.get("tool_argument_envelope_repairs", 0) or 0
        ),
        protocol_tool_name_count=float(
            metadata.get("protocol_tool_name_count", 0) or 0
        ),
        web_search_used=float(bool(metadata.get("web_search_used"))),
        reaction_search_exposed=float(bool(metadata.get("reaction_search_exposed"))),
        reaction_search_called=float(bool(metadata.get("reaction_search_called"))),
        reaction_candidate_count=float(
            metadata.get("reaction_candidate_count", 0) or 0
        ),
        reaction_selected=str(metadata.get("reaction_selected", "") or ""),
        reaction_mode=str(metadata.get("reaction_mode", "") or ""),
        reaction_recent_count=float(metadata.get("reaction_recent_count", 0) or 0),
        reaction_delivery_result=str(
            metadata.get("reaction_delivery_result", "") or ""
        ),
        reaction_abstain_stage=str(metadata.get("reaction_abstain_stage", "") or ""),
        client_web_search_calls=float(metadata.get("client_web_search_calls", 0) or 0),
        web_citation_count=float(metadata.get("web_citation_count", 0) or 0),
        native_web_search_exposed=float(
            any(
                bool(item.get("native_web_search_exposed"))
                for item in model_requests
                if isinstance(item, dict)
            )
        ),
        client_web_search_exposed=float(
            any(
                bool(item.get("client_web_search_exposed"))
                for item in model_requests
                if isinstance(item, dict)
            )
        ),
        tool_schema_omitted_count=float(
            max(
                (
                    int(item.get("tool_schema_omitted_count", 0) or 0)
                    for item in model_requests
                    if isinstance(item, dict)
                ),
                default=0,
            )
        ),
        skill_schema_omitted_count=float(
            max(
                (
                    int(item.get("skill_schema_omitted_count", 0) or 0)
                    for item in model_requests
                    if isinstance(item, dict)
                ),
                default=0,
            )
        ),
        plugin_capacity_degraded=float(bool(metadata.get("plugin_capacity_degraded"))),
    )


def _retrieval_context(frame: TurnFrame) -> dict[str, bool | int | str]:
    selection = frame.selection_context
    target = getattr(frame, "verified_action_target", None)
    has_verified_target = bool(getattr(target, "is_resolved", False))
    return {
        "has_reply": bool(frame.has_reply),
        "has_at": bool(getattr(selection, "has_at", False)),
        "has_image": bool(getattr(selection, "has_image", False)),
        "reply_image_count": len(frame.reply_image_segments_for_reroute or []),
        "has_verified_target": has_verified_target,
        "verified_target_source": (
            str(getattr(target, "source", "") or "") if has_verified_target else ""
        ),
    }


def _event_target_hint(*, frame: TurnFrame, bot: Bot) -> str:
    target_user_id = _event_target_id(frame=frame, bot=bot)
    return f"[@{target_user_id}]" if target_user_id else ""


def _event_target_ids(*, frame: TurnFrame, bot: Bot) -> tuple[str, ...]:
    target_user_id = _event_target_id(frame=frame, bot=bot)
    return (target_user_id,) if target_user_id else ()


def _event_target_id(*, frame: TurnFrame, bot: Bot) -> str:
    target = getattr(frame, "verified_action_target", None)
    if target is None or not bool(getattr(target, "is_resolved", False)):
        return ""
    if str(getattr(target, "source", "") or "") not in {
        "at",
        "reply",
        "alias",
        "self_nickname",
    }:
        return ""
    target_user_id = normalize_message_text(str(getattr(target, "user_id", "") or ""))
    if not target_user_id or target_user_id == str(getattr(bot, "self_id", "") or ""):
        return ""
    return target_user_id


def _bound_history_scope(frame: TurnFrame) -> dict[str, str | None]:
    session = getattr(frame, "session", None)
    user_id = str(getattr(frame, "user_id", "") or "")
    group_id = str(getattr(frame, "group_id", "") or "") or None
    bot_id = str(getattr(frame, "bot_id", "") or "") or None
    platform = None
    channel_id = None
    if session is not None:
        try:
            entity = get_entity_ids(session)
            user_id = str(entity.user_id or user_id)
            group_id = str(entity.group_id or "") or None
            channel_id = str(entity.channel_id or "") or None
        except Exception:
            pass
        platform = str(getattr(session, "platform", "") or "") or None
    return {
        "user_id": user_id,
        "group_id": group_id,
        "bot_id": bot_id,
        "platform": platform,
        "channel_id": channel_id,
    }


__all__ = ["stage_unified_run"]
