"""Lightweight plugin command router.

Group plugin calls are routed by ToolRouter and executed by TaskExecutionQueue.
No superuser tools, MCP, approval, delegate_task or AgentRuntime fallback live here.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import TYPE_CHECKING, Any
import uuid

from ..config import build_reasoning_generation_config, get_config_value, get_model_name
from ..native_executor import NativeCommandExecutionContext
from ..native_route import NativeRouteReport
from ..route_text import is_usage_question, normalize_message_text
from ..task_coverage import build_task_coverage_report, synthesize_task_coverage_reply
from ..task_execution_queue import TaskExecutionQueue
from ..task_planner_lite import TaskItem, plan_task_items
from ..task_router import TaskRouter
from ..tool_retriever import CommandToolRetriever
from .core import (
    PLUGIN_COMMAND_TOOL_SCOPE,
    AgentObservation,
    AgentRequest,
    AgentResult,
)

if TYPE_CHECKING:
    from ..main_request_models import MainRequestResult

CandidatesForTaskRoutes = Callable[..., list[Any]]
ResultFromTaskExecutionQueue = Callable[..., "MainRequestResult"]
TryLocalDirectCommand = Callable[..., Any]


class PluginCommandAgent:
    """Boundary for group plugin command routing."""

    def __init__(
        self,
        *,
        candidates_for_task_routes: CandidatesForTaskRoutes,
        result_from_task_execution_queue: ResultFromTaskExecutionQueue,
        try_local_direct_command: TryLocalDirectCommand,
    ) -> None:
        self._candidates_for_task_routes = candidates_for_task_routes
        self._result_from_task_execution_queue = result_from_task_execution_queue
        self._try_local_direct_command = try_local_direct_command

    async def run(self, request: AgentRequest) -> AgentResult:
        started = time.perf_counter()
        message_text = normalize_message_text(request.message_text)
        report = request.report or NativeRouteReport(
            helper_mode=is_usage_question(message_text)
        )
        retriever = CommandToolRetriever(
            request.knowledge_base,
            session_id=request.session_key,
            tools=request.command_tools,
        )
        report.note_candidate_policy(
            reason="plugin_router_retrieval"
            if retriever.total_commands > 0
            else "plugin_tools_disabled",
            limit=retriever.total_commands,
        )
        report.candidate_total = max(report.candidate_total, retriever.total_commands)

        command_context = NativeCommandExecutionContext(
            candidates=[],
            has_reply=request.has_reply,
            report=report,
            route_executor=request.route_executor,
            message_text=message_text,
        )

        direct_result = await self._try_local_direct_command(
            message_text=message_text,
            retriever=retriever,
            command_context=command_context,
            report=report,
            route_executor=request.route_executor,
            budget_controller=request.budget_controller,
        )
        if direct_result is not None:
            return _agent_result(
                direct_result,
                started=started,
                observation="local_direct_command",
            )

        trace_id = uuid.uuid4().hex[:12]
        task_router_result = await TaskRouter(
            retriever=retriever,
            trace_id=trace_id,
            model_name=get_model_name(),
            generation_config=build_reasoning_generation_config(),
            timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
        ).route_tasks(_router_tasks(message_text, plan_task_items(message_text)))
        task_queue_candidates = self._candidates_for_task_routes(
            retriever=retriever,
            routes=list(task_router_result.routes),
        )
        command_context.candidates = task_queue_candidates
        if task_queue_candidates:
            report.note_prompt_exposure(task_queue_candidates)
            report.note_tool_pool(
                len(task_queue_candidates),
                choice_count=task_router_result.selected_count,
            )
            report.tool_candidates = max(
                report.tool_candidates,
                len(task_queue_candidates),
            )
        task_queue_result = await TaskExecutionQueue(
            command_context=command_context,
            candidates=task_queue_candidates,
        ).execute(task_router_result)
        task_coverage_report = build_task_coverage_report(
            task_router_result,
            task_queue_result,
        )
        return _agent_result(
            self._result_from_task_execution_queue(
                message_text=message_text,
                report=report,
                command_context=command_context,
                task_router_payload=task_router_result.to_payload(),
                task_queue_payload=task_queue_result.to_payload(),
                task_coverage_report=task_coverage_report,
                tool_results=list(task_queue_result.tool_results),
                final_text=synthesize_task_coverage_reply(task_coverage_report),
            ),
            started=started,
            observation="task_execution_queue",
        )


def _router_tasks(
    message_text: str,
    planned: tuple[TaskItem, ...],
) -> tuple[TaskItem, ...]:
    if planned:
        return planned
    text = normalize_message_text(message_text)
    if not text:
        return ()
    return (TaskItem(task_id="task_1", text=text, order=1),)


def _agent_result(
    result: "MainRequestResult",
    *,
    started: float,
    observation: str,
) -> AgentResult:
    return AgentResult(
        agent_kind="plugin_command",
        main_result=result,
        observations=(AgentObservation(kind=observation, status="ok"),),
        tool_scope=PLUGIN_COMMAND_TOOL_SCOPE,
        elapsed_ms=max(int((time.perf_counter() - started) * 1000), 0),
    )


__all__ = ["PluginCommandAgent"]
