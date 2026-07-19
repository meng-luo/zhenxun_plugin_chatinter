"""Lightweight plugin command router.

Group plugin calls are routed by ToolRouter and executed by TaskExecutionQueue.
No superuser tools, MCP, approval, delegate_task or AgentRuntime fallback live here.
"""

from __future__ import annotations

import time
from typing import Any
import uuid

from zhenxun.services.ai.core.engine.token_counter import parse_usage_info

from ..config import (
    INTENT_TIMEOUT_SECONDS,
    build_agent_generation_config,
    get_agent_model,
)
from ..main_request_models import (
    MainRequestOutput,
    MainRequestResult,
    MainRequestTimelineItem,
)
from ..native_executor import NativeCommandExecutionContext
from ..native_route import NativeRouteDecision, NativeRouteReport
from ..route_text import is_usage_question, normalize_message_text
from ..runtime_result import _result_from_task_execution_queue
from ..task_coverage import build_task_coverage_report
from ..task_execution_queue import TaskExecutionQueue
from ..task_planner_lite import TaskItem, plan_task_items
from ..task_router import TaskRouter
from ..tool_retriever import CommandToolRetriever
from .core import (
    PLUGIN_COMMAND_TOOL_SCOPE,
    AgentObservation,
    AgentResult,
    PluginCommandRequest,
)


class PluginCommandAgent:
    """Boundary for group plugin command routing."""

    async def run(self, request: PluginCommandRequest) -> AgentResult:
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

        trace_id = uuid.uuid4().hex[:12]

        def _record_usage(usage_info: dict[str, Any] | None) -> None:
            if request.budget_controller is None:
                return
            usage = parse_usage_info(usage_info)
            request.budget_controller.record_model_usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )

        task_router_result = await TaskRouter(
            retriever=retriever,
            trace_id=trace_id,
            model_name=get_agent_model("plugin"),
            generation_config=build_agent_generation_config("plugin"),
            timeout=float(INTENT_TIMEOUT_SECONDS),
            usage_callback=(
                _record_usage if request.budget_controller is not None else None
            ),
        ).route_tasks(
            _router_tasks(message_text, plan_task_items(message_text)),
            router_context=request.router_context,
        )
        if task_router_result.selected_count == 0:
            return _agent_result(
                _no_selection_result(
                    message_text=message_text,
                    report=report,
                    task_router_payload=task_router_result.to_payload(),
                ),
                started=started,
                observation="no_plugin_selection",
            )

        task_queue_candidates = _candidates_for_task_routes(
            candidates=task_router_result.candidates,
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
            _result_from_task_execution_queue(
                message_text=message_text,
                report=report,
                command_context=command_context,
                task_router_payload=task_router_result.to_payload(),
                task_queue_payload=task_queue_result.to_payload(),
                task_coverage_report=task_coverage_report,
                tool_results=list(task_queue_result.tool_results),
                final_text="",
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


def _candidates_for_task_routes(
    *,
    candidates: tuple[Any, ...],
    routes: list[Any],
) -> list[Any]:
    candidates_by_id = {
        normalize_message_text(candidate.schema.command_id): candidate
        for candidate in candidates
        if normalize_message_text(candidate.schema.command_id)
    }
    selected: list[Any] = []
    for route in routes:
        command_id = normalize_message_text(str(getattr(route, "command_id", "") or ""))
        if not command_id or getattr(route, "status", "") != "selected":
            continue
        candidate = candidates_by_id.get(command_id)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    return selected


def _no_selection_result(
    *,
    message_text: str,
    report: NativeRouteReport,
    task_router_payload: dict[str, Any],
) -> MainRequestResult:
    reason = "plugin_router:no_selection"
    report.finalize(reason=reason, stage="plugin_command_agent")
    return MainRequestResult(
        decision=NativeRouteDecision(
            action="chat",
            confidence=0.0,
            reason=reason,
        ),
        route_result=None,
        report=report,
        timeline=(
            MainRequestTimelineItem(
                role="user",
                kind="current_user",
                content=message_text,
            ),
            MainRequestTimelineItem(
                role="system",
                kind="task_router",
                metadata=task_router_payload,
            ),
        ),
        output=MainRequestOutput(
            final_text="",
            should_send=False,
            outcome="plugin_no_selection",
            feedback_kind="plugin_no_selection",
            record_chat_feedback=False,
            observation_reason=reason,
        ),
    )


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
