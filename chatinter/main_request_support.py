"""Support routines behind ChatInter scenario agents.

This module owns legacy runtime plumbing, local direct command execution,
tool-obligation policy, result construction and finalization.  The public
``main_request`` module should stay a thin scenario dispatcher.
"""

from __future__ import annotations

import time
from typing import Any

from zhenxun.services.llm.types.models import ToolResult

from .local_direct_command import (
    LocalDirectCommandPlan,
    plan_local_direct_command,
    plan_local_direct_command_batch,
)
from .main_request_models import (
    MainRequestOutput,
    MainRequestResult,
    MainRequestTimelineItem,
)
from .native_executor import (
    ExecuteNativeRoute,
    NativeCommandExecutionContext,
    NativeToolExecutionResult,
)
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
)
from .plugin_command_support import (
    _apply_tool_exposure_policy,
    _candidates_for_task_routes,
    _resolve_tool_obligation,
)
from .runtime_result import (
    _fallback_final_reply,
    _fallback_result,
    _finalize_result,
    _result_from_task_execution_queue,
    _timeline_memory_text,
    _user_timeline_item,
)
from .tool_retriever import CommandToolRetriever
from .turn_runtime import TurnBudgetController


async def _try_local_direct_command(
    *,
    message_text: str,
    retriever: CommandToolRetriever,
    command_context: NativeCommandExecutionContext,
    report: NativeRouteReport,
    route_executor: ExecuteNativeRoute,
    budget_controller: TurnBudgetController | None,
) -> MainRequestResult | None:
    del route_executor
    started = time.perf_counter()
    candidates = retriever.retrieve(message_text, limit=32).candidates
    batch_plan = plan_local_direct_command_batch(
        message_text=message_text,
        candidates=list(candidates),
        tool_map={},
    )
    if batch_plan is not None:
        return await _execute_local_direct_plans(
            message_text=message_text,
            plans=batch_plan.steps,
            candidates=list(candidates),
            command_context=command_context,
            report=report,
            budget_controller=budget_controller,
            started=started,
            reason=batch_plan.reason,
        )
    plan = plan_local_direct_command(
        message_text=message_text,
        candidates=list(candidates),
        tool_map={},
    )
    if plan is None:
        return None

    return await _execute_local_direct_plans(
        message_text=message_text,
        plans=[plan],
        candidates=list(candidates),
        command_context=command_context,
        report=report,
        budget_controller=budget_controller,
        started=started,
        reason=plan.reason,
    )


async def _execute_local_direct_plans(
    *,
    message_text: str,
    plans: list[LocalDirectCommandPlan],
    candidates: list[Any],
    command_context: NativeCommandExecutionContext,
    report: NativeRouteReport,
    budget_controller: TurnBudgetController | None,
    started: float,
    reason: str,
) -> MainRequestResult | None:
    if not plans:
        return None
    from .native_command_tools import build_native_command_tools

    tools = build_native_command_tools([plan.candidate for plan in plans])
    if not tools:
        return None
    tools_by_command = {tool.binding.command_id: tool for tool in tools}
    command_context.candidates = candidates
    tool_results: list[ToolResult] = []
    timeline_items: list[MainRequestTimelineItem] = [_user_timeline_item(message_text)]
    latest_execution: NativeToolExecutionResult | None = None
    for plan in plans:
        tool = tools_by_command.get(plan.candidate.schema.command_id)
        if tool is None:
            continue
        result = await command_context.execute_tool(
            binding=tool.binding,
            raw_slots=dict(plan.raw_slots),
        )
        tool_results.append(result)
        execution = (
            command_context.executions[-1] if command_context.executions else None
        )
        if execution is None:
            execution = NativeToolExecutionResult(
                success=False,
                route_result=None,
                output=result.output if isinstance(result.output, dict) else {},
                display_text=result.display_content or "",
                reason="local_direct_no_execution",
            )
        latest_execution = execution
        timeline_items.append(
            MainRequestTimelineItem(
                role="tool",
                kind="local_direct_command",
                content=execution.display_text or execution.reason,
                tool_name=tool.binding.tool_name,
                metadata=execution.output,
            )
        )
    if budget_controller is not None:
        budget_controller.record_tool_batch(
            batch_kind="local_direct_command",
            duration=time.perf_counter() - started,
        )
    if not tool_results:
        return None
    if latest_execution is None:
        return None
    timeline = tuple(timeline_items)
    final_text = _fallback_final_reply(command_context.executions)
    all_completed = bool(command_context.executions) and all(
        item.success for item in command_context.executions
    )
    return MainRequestResult(
        decision=NativeRouteDecision(
            action="execute",
            confidence=0.92,
            reason=reason,
        ),
        route_result=latest_execution.route_result,
        report=report,
        executions=tuple(command_context.executions),
        tool_results=tuple(tool_results),
        timeline=timeline,
        output=MainRequestOutput(
            final_text=final_text,
            memory_text=_timeline_memory_text(timeline, fallback=final_text),
            should_send=bool(final_text),
            outcome="tool_completed" if all_completed else "tool_failed",
            feedback_kind="tool_completed" if all_completed else "tool_failed",
            record_chat_feedback=False,
            observation_reason="route_success" if all_completed else "reroute_failed",
        ),
    )


def _should_hide_delegate_task(
    *,
    enable_agent_tools: bool,
    complexity_mode: str,
) -> bool:
    """Keep sub-agent delegation out of fast paths.

    ``delegate_task`` is useful for long superuser engineering tasks, but it is
    too expensive and uncertain for normal plugin routing or light private
    commands.  The scenario router already prevents group plugin calls from
    registering superuser tools; this guard keeps the superuser fast path small.
    """

    return bool(enable_agent_tools and complexity_mode != "complex_pev")


__all__ = [
    "_apply_tool_exposure_policy",
    "_candidates_for_task_routes",
    "_fallback_result",
    "_finalize_result",
    "_resolve_tool_obligation",
    "_result_from_task_execution_queue",
    "_try_local_direct_command",
    "_user_timeline_item",
]
