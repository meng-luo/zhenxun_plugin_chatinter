"""Unified ChatInter main request runner.

This module prepares the turn prompt and command tools, then delegates the
model/tool loop to ``agent_runtime``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from inspect import isawaitable
import time
from typing import Any, cast
import uuid

from zhenxun.services import logger
from zhenxun.services.llm import LLMMessage
from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types.models import ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .agent_runtime import AgentRuntime
from .agent_state import AgentRunState, AgentRuntimeResult, AgentRuntimeTimelineItem
from .command_catalog_tool import (
    COMMAND_CATALOG_TOOL_NAME,
    CommandCatalogState,
    CommandCatalogTool,
)
from .config import build_reasoning_generation_config, get_config_value, get_model_name
from .models.pydantic_models import PluginKnowledgeBase
from .native_executor import (
    ExecuteNativeRoute,
    NativeCommandExecutionContext,
    NativeToolExecutionResult,
)
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
    NativeRouteResult,
)
from .route_text import is_usage_question, normalize_message_text
from .soft_tool_policy import filter_soft_candidates
from .superuser_agent import build_superuser_agent_tools
from .tool_intent_gate import ToolIntentGate, ToolIntentGateResult
from .tool_retriever import CommandToolRetriever
from .turn_runtime import TurnBudgetController

_MAIN_STAGE = "main_request"
_TOOL_INTENT_GATE_STAGE = "tool_intent_gate"
_MAX_REQUEST_TOOL_COUNT = 120
MainRequestRouteHook = Callable[["MainRequestResult"], Awaitable[None] | None]
MainRequestReplyHook = Callable[[str], Awaitable[str] | str]


@dataclass(frozen=True)
class MainRequestOutput:
    analysis: str = "main request"
    final_text: str = ""
    memory_text: str = ""
    should_send: bool = True
    outcome: str = "chat_completed"
    feedback_kind: str = "chat_completed"
    record_chat_feedback: bool = True
    observation_reason: str = "chat_completed"


@dataclass(frozen=True)
class MainRequestTimelineItem:
    role: str
    kind: str
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "kind": self.kind,
        }
        if self.content:
            payload["content"] = self.content
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class MainRequestResult:
    decision: NativeRouteDecision
    route_result: NativeRouteResult | None
    report: NativeRouteReport
    executions: tuple[NativeToolExecutionResult, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    timeline: tuple[MainRequestTimelineItem, ...] = ()
    output: MainRequestOutput = field(default_factory=MainRequestOutput)

    @property
    def handled_by_tools(self) -> bool:
        return any(item.route_result is not None for item in self.executions)


@dataclass(frozen=True)
class ToolObligationDecision:
    obligation: str
    reason: str
    required_tool_names: tuple[str, ...] = ()
    gate_result: ToolIntentGateResult | None = None


async def run_chatinter_main_request(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None,
    messages: list[LLMMessage],
    route_executor: ExecuteNativeRoute,
    route_completed_hook: MainRequestRouteHook | None = None,
    reply_hook: MainRequestReplyHook | None = None,
    enable_plugin_tools: bool = True,
    initial_command_exposure: bool = False,
    enable_agent_tools: bool = False,
) -> MainRequestResult:
    normalized_message = normalize_message_text(message_text)
    report = NativeRouteReport(helper_mode=is_usage_question(normalized_message))

    if budget_controller is not None and not budget_controller.allow_classifier(
        _MAIN_STAGE
    ):
        return await _finalize_result(
            _fallback_result(
                report=report,
                reason="main_request_budget_exhausted",
                reply="我现在有点忙，稍后再试试吧。",
                timeline=[_user_timeline_item(normalized_message)],
            ),
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )

    started = time.perf_counter()
    try:
        result = await _run_main_request(
            normalized_message,
            knowledge_base,
            session_key=session_key,
            budget_controller=budget_controller,
            has_reply=has_reply,
            command_tools=command_tools,
            messages=messages,
            route_executor=route_executor,
            report=report,
            enable_plugin_tools=enable_plugin_tools,
            initial_command_exposure=initial_command_exposure,
            enable_agent_tools=enable_agent_tools,
        )
        return await _finalize_result(
            result,
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )
    except Exception as exc:
        logger.error(f"ChatInter main request failed: {exc}")
        return await _finalize_result(
            _fallback_result(
                report=report,
                reason=f"main_request_error:{type(exc).__name__}",
                reply="抱歉，我刚刚处理失败了。",
                timeline=[_user_timeline_item(normalized_message)],
            ),
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )
    finally:
        if budget_controller is not None:
            budget_controller.record_classifier(
                _MAIN_STAGE,
                time.perf_counter() - started,
            )


async def _run_main_request(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None,
    messages: list[LLMMessage],
    route_executor: ExecuteNativeRoute,
    report: NativeRouteReport,
    enable_plugin_tools: bool,
    initial_command_exposure: bool,
    enable_agent_tools: bool,
) -> MainRequestResult:
    retriever = CommandToolRetriever(
        knowledge_base,
        session_id=session_key,
        tools=cast(Any, command_tools),
    )
    agent_tools = build_superuser_agent_tools() if enable_agent_tools else {}
    base_tool_count = len(agent_tools) + (1 if enable_plugin_tools else 0)
    command_tool_capacity = max(1, _MAX_REQUEST_TOOL_COUNT - base_tool_count)
    catalog_state = CommandCatalogState(
        retriever=retriever,
        max_command_tools=command_tool_capacity,
    )

    report.note_candidate_policy(
        reason="main_request_catalog_retrieval"
        if enable_plugin_tools
        else "plugin_tools_disabled",
        limit=retriever.total_commands if enable_plugin_tools else 0,
    )
    if enable_plugin_tools:
        report.candidate_total = max(report.candidate_total, retriever.total_commands)
    report.note_tool_pool(1 if enable_plugin_tools else 0)

    tool_map: dict[str, ToolExecutable] = {}
    if enable_plugin_tools:
        catalog_tool = CommandCatalogTool(catalog_state)
        tool_map[COMMAND_CATALOG_TOOL_NAME] = cast(ToolExecutable, catalog_tool)
    tool_map.update(agent_tools)
    if tool_map:
        report.note_tool_pool(len(tool_map))
    command_context = NativeCommandExecutionContext(
        candidates=[],
        has_reply=has_reply,
        report=report,
        route_executor=route_executor,
        message_text=message_text,
    )
    if enable_plugin_tools and initial_command_exposure:
        initial_result = retriever.initial_command_exposure(
            message_text,
            max_total=command_tool_capacity,
        )
        catalog_state.inject(list(initial_result.candidates))
        command_context.candidates = catalog_state.candidates
        tool_map.update(catalog_state.tool_map)
        report.note_prompt_exposure(catalog_state.candidates)
        report.tool_candidates = max(
            report.tool_candidates,
            catalog_state.injected_count,
        )
        report.note_tool_pool(len(tool_map))
        report.note_candidate_policy(
            reason="initial_grouped_command_exposure",
            limit=catalog_state.injected_count,
        )
    trace_id = uuid.uuid4().hex[:12]
    obligation_decision = await _resolve_tool_obligation(
        message_text=message_text,
        enable_plugin_tools=enable_plugin_tools,
        enable_agent_tools=enable_agent_tools,
        candidates=catalog_state.candidates,
        tool_map=tool_map,
        trace_id=trace_id,
        model_name=get_model_name(),
        generation_config=build_reasoning_generation_config(),
        timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
        budget_controller=budget_controller,
    )
    run_context = RunContext(
        session_id=session_key,
        extra={
            "native_command_context": command_context,
            "command_catalog_state": catalog_state,
            "actor_user_id": session_key or "",
            "agent_mode": "superuser_agent" if enable_agent_tools else "chatinter",
            "enable_agent_tools": enable_agent_tools,
        },
    )
    state = AgentRunState.create(
        trace_id=trace_id,
        session_key=session_key,
        messages=messages,
        tool_map=tool_map,
        current_message=message_text,
        max_steps=8 if enable_agent_tools else 5,
        budget_controller=budget_controller,
        tool_obligation=obligation_decision.obligation,
        tool_obligation_reason=obligation_decision.reason,
        required_tool_names=obligation_decision.required_tool_names,
    )
    state.append_timeline(
        role="system",
        kind="tool_intent_gate",
        metadata=_tool_obligation_metadata(obligation_decision),
    )
    runtime = AgentRuntime(
        state=state,
        run_context=run_context,
        message_text=message_text,
        model_name=get_model_name(),
        generation_config=build_reasoning_generation_config(),
        timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
        budget_controller=budget_controller,
    )
    agent_result = await runtime.run()
    timeline = _convert_runtime_timeline(agent_result.timeline)
    if catalog_state.candidates:
        report.note_prompt_exposure(catalog_state.candidates)
        report.tool_candidates = max(
            report.tool_candidates,
            catalog_state.injected_count,
        )
    report.tool_choice_count += sum(
        1 for item in agent_result.timeline if item.kind == "tool_call"
    )
    return _result_from_agent_runtime(
        report=report,
        executions=command_context.executions,
        agent_result=agent_result,
        timeline=timeline,
    )


def _fallback_result(
    *,
    report: NativeRouteReport,
    reason: str,
    reply: str,
    timeline: list[MainRequestTimelineItem] | None = None,
) -> MainRequestResult:
    decision = NativeRouteDecision(action="chat", confidence=0.0, reason=reason)
    report.finalize(reason=reason, stage=_MAIN_STAGE)
    return MainRequestResult(
        decision=decision,
        route_result=None,
        report=report,
        timeline=(
            *(timeline or []),
            MainRequestTimelineItem(
                role="system",
                kind="fallback",
                content=reason,
            ),
        ),
        output=MainRequestOutput(final_text=reply, memory_text=reply),
    )


def _result_from_agent_runtime(
    *,
    report: NativeRouteReport,
    executions: list[NativeToolExecutionResult],
    agent_result: AgentRuntimeResult,
    timeline: list[MainRequestTimelineItem],
) -> MainRequestResult:
    stop_reason = agent_result.stop_reason
    reason = f"main_request:{stop_reason}"
    if report.final_reason == "init":
        first_route = _first_route(executions)
        report.finalize(
            reason=reason,
            stage=first_route.stage if first_route is not None else _MAIN_STAGE,
            plugin_name=first_route.decision.plugin_name
            if first_route is not None
            else None,
            plugin_module=first_route.decision.plugin_module
            if first_route is not None
            else None,
            command=first_route.decision.command if first_route is not None else None,
        )
    command_tool_results = [
        result
        for result in agent_result.tool_results
        if not _is_catalog_tool_result(result)
    ]
    final_text = normalize_message_text(agent_result.final_text)
    should_send = bool(final_text)
    memory_text = _timeline_memory_text(timeline, fallback=final_text)
    handled_by_tools = bool(executions or command_tool_results)
    return MainRequestResult(
        decision=NativeRouteDecision(
            action="chat",
            confidence=0.9 if handled_by_tools else 0.84,
            reason=reason,
        ),
        route_result=_first_route(executions),
        report=report,
        executions=tuple(executions),
        tool_results=tuple(command_tool_results),
        timeline=tuple(timeline),
        output=MainRequestOutput(
            final_text=final_text,
            memory_text=memory_text,
            should_send=should_send,
            outcome="tool_completed" if handled_by_tools else "chat_completed",
            feedback_kind="tool_completed" if handled_by_tools else "chat_completed",
            record_chat_feedback=not handled_by_tools,
            observation_reason="route_success"
            if any(item.success for item in executions)
            else "reroute_failed"
            if handled_by_tools
            else "chat_completed",
        ),
    )


def _is_catalog_tool_result(result: ToolResult) -> bool:
    output = result.output if isinstance(result.output, dict) else {}
    return output.get("status") in {
        "retrieved",
        "capability_candidates_retrieved",
    }


async def _finalize_result(
    result: MainRequestResult,
    *,
    route_completed_hook: MainRequestRouteHook | None,
    reply_hook: MainRequestReplyHook | None,
) -> MainRequestResult:
    if route_completed_hook is not None:
        maybe_awaitable = route_completed_hook(result)
        if maybe_awaitable is not None:
            await maybe_awaitable

    output = result.output
    if not output.should_send:
        return result

    final_text = normalize_message_text(output.final_text)
    if not final_text:
        final_text = (
            _fallback_final_reply(list(result.executions)) or "我暂时没想好怎么回答你。"
        )
    if reply_hook is not None:
        maybe_reply = reply_hook(final_text)
        final_text = (
            await maybe_reply if isawaitable(maybe_reply) else str(maybe_reply or "")
        )
    final_text = normalize_message_text(final_text)
    if not final_text:
        final_text = "我暂时没想好怎么回答你。"
    final_timeline = _with_final_timeline(
        result.timeline,
        final_text=final_text,
        should_send=True,
    )
    memory_text = normalize_message_text(output.memory_text) or _timeline_memory_text(
        list(final_timeline),
        fallback=final_text,
    )
    return replace(
        result,
        timeline=final_timeline,
        output=replace(
            output,
            final_text=final_text,
            memory_text=memory_text,
            should_send=True,
        ),
    )


def _first_route(
    executions: list[NativeToolExecutionResult],
) -> NativeRouteResult | None:
    for execution in executions:
        if execution.route_result is not None:
            return execution.route_result
    return None


def _fallback_final_reply(executions: list[NativeToolExecutionResult]) -> str:
    if not executions:
        return ""
    success_count = sum(1 for item in executions if item.success)
    latest = executions[-1]
    if latest.display_text:
        return latest.display_text
    if success_count:
        return "处理好了。"
    message = str(latest.output.get("error", "") or latest.reason or "").strip()
    return message or "这个暂时没处理成功。"


def _timeline_memory_text(
    timeline: list[MainRequestTimelineItem] | tuple[MainRequestTimelineItem, ...],
    *,
    fallback: str = "",
) -> str:
    lines: list[str] = []
    for item in timeline:
        text = _timeline_item_summary(item)
        if text:
            lines.append(text)
    if fallback:
        lines.append(normalize_message_text(f"assistant: {fallback}"))
    return "\n".join(dict.fromkeys(line for line in lines if line))[:4000]


def _timeline_item_summary(item: MainRequestTimelineItem) -> str:
    role = normalize_message_text(item.role)
    kind = normalize_message_text(item.kind)
    prefix = f"{role}/{kind}".strip("/")
    if item.tool_name:
        prefix = f"{prefix}:{normalize_message_text(item.tool_name)}"
    content = normalize_message_text(item.content)
    if not content:
        output = item.metadata.get("output") if isinstance(item.metadata, dict) else None
        content = _compact_output_summary(output)
    if not content:
        arguments = (
            item.metadata.get("arguments") if isinstance(item.metadata, dict) else None
        )
        content = _compact_output_summary(arguments)
    if not content:
        return ""
    return f"{prefix}: {content}"[:800]


def _compact_output_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return normalize_message_text(str(value or ""))[:500]
    parts: list[str] = []
    for key in (
        "status",
        "ok",
        "command_id",
        "rendered_command",
        "matched_plugin",
        "task_text",
        "error",
        "remaining_task_hint",
    ):
        item = value.get(key)
        if item not in ("", [], {}, None):
            parts.append(f"{key}={normalize_message_text(str(item))}")
    messages = value.get("messages_sent")
    if isinstance(messages, list) and messages:
        parts.append(
            "messages_sent="
            + " | ".join(
                normalize_message_text(str(message or ""))
                for message in messages[:3]
                if normalize_message_text(str(message or ""))
            )
        )
    artifacts = value.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        summaries = [
            normalize_message_text(str(item.get("summary", "") or ""))
            for item in artifacts[:3]
            if isinstance(item, dict)
            and normalize_message_text(str(item.get("summary", "") or ""))
        ]
        if summaries:
            parts.append("artifacts=" + " | ".join(summaries))
    return "；".join(parts)[:500]


def _user_timeline_item(message_text: str) -> MainRequestTimelineItem:
    return MainRequestTimelineItem(
        role="user",
        kind="current_user",
        content=message_text,
    )


def _convert_runtime_timeline(
    items: tuple[AgentRuntimeTimelineItem, ...],
) -> list[MainRequestTimelineItem]:
    return [
        MainRequestTimelineItem(
            role=item.role,
            kind=item.kind,
            content=item.content,
            tool_name=item.tool_name,
            metadata=dict(item.metadata),
        )
        for item in items
    ]


def _with_final_timeline(
    timeline: tuple[MainRequestTimelineItem, ...],
    *,
    final_text: str,
    should_send: bool,
) -> tuple[MainRequestTimelineItem, ...]:
    if not final_text and not should_send:
        return timeline
    return (
        *timeline,
        MainRequestTimelineItem(
            role="assistant",
            kind="final_output",
            content=final_text,
            metadata={"sent_by_chatinter": should_send},
        ),
    )


async def _resolve_tool_obligation(
    *,
    message_text: str,
    enable_plugin_tools: bool,
    enable_agent_tools: bool,
    candidates: list[Any],
    tool_map: dict[str, ToolExecutable],
    trace_id: str,
    model_name: str | None,
    generation_config: Any,
    timeout: float,
    budget_controller: TurnBudgetController | None,
) -> ToolObligationDecision:
    if not tool_map:
        return ToolObligationDecision(obligation="none", reason="no_tools")
    if enable_agent_tools:
        return ToolObligationDecision(
            obligation="auto",
            reason="superuser_agent_tools_available",
        )
    if not enable_plugin_tools:
        return ToolObligationDecision(
            obligation="none",
            reason="plugin_tools_disabled",
        )
    command_tools = _command_tool_names(tool_map)
    if not command_tools:
        return ToolObligationDecision(obligation="auto", reason="catalog_only")
    if not candidates:
        return ToolObligationDecision(
            obligation="none",
            reason="no_command_candidates",
        )

    if budget_controller is not None and not budget_controller.allow_classifier(
        _TOOL_INTENT_GATE_STAGE
    ):
        return ToolObligationDecision(
            obligation="auto",
            reason="tool_intent_gate_budget_exhausted",
        )

    started = time.perf_counter()
    gate = ToolIntentGate(
        trace_id=trace_id,
        model_name=model_name,
        generation_config=generation_config,
        timeout=timeout,
    )
    result = await gate.judge(
        message_text=message_text,
        candidates=candidates,
        command_tool_count=len(command_tools),
    )
    if budget_controller is not None:
        budget_controller.record_classifier(
            _TOOL_INTENT_GATE_STAGE,
            time.perf_counter() - started,
        )

    required_tool_names: tuple[str, ...] = ()
    if result.intent == "chat":
        obligation = "none"
    elif result.intent == "plugin_required":
        command_id_filter = {
            normalize_message_text(str(command_id or ""))
            for command_id in (
                result.required_command_ids or result.allowed_command_ids
            )
            if normalize_message_text(str(command_id or ""))
        }
        if not command_id_filter:
            allowed_soft_candidates = filter_soft_candidates(message_text, candidates)
            command_id_filter = {
                normalize_message_text(
                    str(getattr(getattr(candidate, "schema", None), "command_id", ""))
                )
                for candidate in allowed_soft_candidates
                if normalize_message_text(
                    str(getattr(getattr(candidate, "schema", None), "command_id", ""))
                )
            }
        obligation = "required"
        required_tool_names = _tool_names_for_command_ids(
            tool_map,
            tuple(command_id_filter),
        )
        if not required_tool_names:
            obligation = "auto"
    else:
        obligation = "auto"

    return ToolObligationDecision(
        obligation=obligation,
        reason=_gate_obligation_reason(result),
        required_tool_names=required_tool_names,
        gate_result=result,
    )


def _command_tool_names(tool_map: dict[str, ToolExecutable]) -> tuple[str, ...]:
    names: list[str] = []
    for name, tool in tool_map.items():
        binding = getattr(tool, "binding", None)
        command_id = normalize_message_text(str(getattr(binding, "command_id", "")))
        if command_id:
            names.append(normalize_message_text(name))
    return tuple(name for name in names if name)


def _tool_names_for_command_ids(
    tool_map: dict[str, ToolExecutable],
    command_ids: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    wanted = {
        normalize_message_text(str(command_id or ""))
        for command_id in command_ids
        if normalize_message_text(str(command_id or ""))
    }
    if not wanted:
        return ()
    names: list[str] = []
    for name, tool in tool_map.items():
        binding = getattr(tool, "binding", None)
        command_id = normalize_message_text(str(getattr(binding, "command_id", "")))
        if command_id and command_id in wanted:
            names.append(normalize_message_text(name))
    return tuple(name for name in names if name)


def _gate_obligation_reason(result: ToolIntentGateResult) -> str:
    return normalize_message_text(
        "tool_intent_gate:"
        f"{result.intent}:confidence={float(result.confidence or 0.0):.2f}:"
        f"{result.reason or 'no_reason'}"
    )


def _tool_obligation_metadata(
    decision: ToolObligationDecision,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tool_obligation": decision.obligation,
        "tool_obligation_reason": decision.reason,
        "required_tool_names": list(decision.required_tool_names),
    }
    result = decision.gate_result
    if result is not None:
        metadata.update(
            {
                "gate_intent": result.intent,
                "gate_confidence": result.confidence,
                "gate_reason": result.reason,
                "required_command_ids": list(result.required_command_ids),
                "allowed_command_ids": list(result.allowed_command_ids),
                "needs_real_execution": result.needs_real_execution,
            }
        )
    return metadata


__all__ = [
    "MainRequestOutput",
    "MainRequestResult",
    "MainRequestTimelineItem",
    "run_chatinter_main_request",
]
