"""Result construction helpers for ChatInter main requests."""

from __future__ import annotations

from dataclasses import replace
from inspect import isawaitable
from typing import Any

from zhenxun.services.llm.types.models import ToolResult

from .agent_state import AgentRuntimeResult, AgentRuntimeTimelineItem
from .main_request_models import (
    MainRequestOutput,
    MainRequestReplyHook,
    MainRequestResult,
    MainRequestRouteHook,
    MainRequestTimelineItem,
)
from .native_executor import NativeCommandExecutionContext, NativeToolExecutionResult
from .native_route import NativeRouteDecision, NativeRouteReport, NativeRouteResult
from .route_text import normalize_message_text
from .task_coverage import TaskCoverageReport

_MAIN_STAGE = "main_request"

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

def _result_from_task_execution_queue(
    *,
    message_text: str,
    report: NativeRouteReport,
    command_context: NativeCommandExecutionContext,
    task_router_payload: dict[str, Any],
    task_queue_payload: dict[str, Any],
    task_coverage_report: TaskCoverageReport,
    tool_results: list[ToolResult],
    final_text: str,
) -> MainRequestResult:
    executions = list(command_context.executions)
    reason = "main_request:task_execution_queue"
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
    timeline: tuple[MainRequestTimelineItem, ...] = (
        _user_timeline_item(message_text),
        MainRequestTimelineItem(
            role="system",
            kind="task_router",
            metadata=task_router_payload,
        ),
        MainRequestTimelineItem(
            role="system",
            kind="task_execution_queue",
            metadata=task_queue_payload,
        ),
        MainRequestTimelineItem(
            role="system",
            kind="task_coverage",
            metadata=task_coverage_report.to_payload(),
        ),
    )
    reply = normalize_message_text(final_text) or _fallback_final_reply(executions)
    if not reply:
        reply = "没有可执行的明确任务。"
    return MainRequestResult(
        decision=NativeRouteDecision(
            action="execute",
            confidence=0.94,
            reason=reason,
        ),
        route_result=_first_route(executions),
        report=report,
        executions=tuple(executions),
        tool_results=tuple(tool_results),
        timeline=timeline,
        output=MainRequestOutput(
            final_text=reply,
            memory_text=_timeline_memory_text(timeline, fallback=reply),
            should_send=bool(reply),
            outcome="tool_completed"
            if task_coverage_report.all_completed
            else "tool_failed",
            feedback_kind="tool_completed"
            if task_coverage_report.all_completed
            else "tool_failed",
            record_chat_feedback=False,
            observation_reason="route_success"
            if task_coverage_report.all_completed
            else "reroute_failed",
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
        output = (
            item.metadata.get("output") if isinstance(item.metadata, dict) else None
        )
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

__all__ = [
    "_convert_runtime_timeline",
    "_fallback_final_reply",
    "_fallback_result",
    "_finalize_result",
    "_result_from_agent_runtime",
    "_result_from_task_execution_queue",
    "_timeline_memory_text",
    "_user_timeline_item",
]
