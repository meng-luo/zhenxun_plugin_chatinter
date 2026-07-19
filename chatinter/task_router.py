"""Task-level command router for obvious multi-command turns.

This module binds TaskPlannerLite items to candidate command tools.  It does
not execute commands; execution order is handled by later stages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .command_index import CommandCandidate
from .native_command_tools import build_native_command_tools
from .route_text import normalize_message_text
from .task_planner_lite import TaskItem
from .tool_retriever import CommandToolRetriever
from .tool_router import ToolRouter, ToolRouterBatchDecision, ToolRouterSelection

TaskRouteStatus = Literal["selected", "unsupported"]

_DEFAULT_TASK_RETRIEVAL_LIMIT = 18
_MIN_SELECT_CONFIDENCE = 0.35


@dataclass(frozen=True, slots=True)
class TaskRouteResult:
    task_id: str
    text: str
    order: int
    status: TaskRouteStatus
    command_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    candidate_count: int = 0

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "text": self.text,
            "order": self.order,
            "status": self.status,
            "command_id": self.command_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "candidate_count": self.candidate_count,
        }
        return {
            key: value for key, value in payload.items() if value not in ("", {}, None)
        }


@dataclass(frozen=True, slots=True)
class TaskRouterResult:
    routes: tuple[TaskRouteResult, ...] = ()
    reason: str = ""
    candidates: tuple[CommandCandidate, ...] = ()

    @property
    def selected_count(self) -> int:
        return sum(1 for route in self.routes if route.status == "selected")

    @property
    def uncertain_count(self) -> int:
        return sum(1 for route in self.routes if route.status != "selected")

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": "task_router",
            "reason": self.reason,
            "selected_count": self.selected_count,
            "uncertain_count": self.uncertain_count,
            "routes": [route.to_payload() for route in self.routes],
        }


class TaskRouter:
    """Route TaskItems through one retriever pass and one ToolRouter call."""

    def __init__(
        self,
        *,
        retriever: CommandToolRetriever,
        trace_id: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
        retrieval_limit: int = _DEFAULT_TASK_RETRIEVAL_LIMIT,
        router: Any | None = None,
        usage_callback: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.retriever = retriever
        self._router = router
        self._trace_id = trace_id
        self._model_name = model_name
        self._generation_config = generation_config
        self._timeout = timeout
        self._usage_callback = usage_callback
        self.retrieval_limit = max(1, int(retrieval_limit or 1))

    async def route_tasks(
        self,
        tasks: tuple[TaskItem, ...],
        *,
        router_context: dict[str, Any] | None = None,
    ) -> TaskRouterResult:
        if not tasks:
            return TaskRouterResult(reason="no_tasks")

        candidates_by_task = [
            (
                task,
                list(
                    self.retriever.retrieve(
                        task.text,
                        limit=self.retrieval_limit,
                        context=router_context,
                    ).candidates
                ),
            )
            for task in tasks
        ]
        candidates = _dedupe_candidates(
            [
                candidate
                for _, task_candidates in candidates_by_task
                for candidate in task_candidates
            ],
            limit=self.retrieval_limit,
        )
        if not candidates:
            return TaskRouterResult(
                routes=tuple(_unsupported(task, "no_candidates") for task in tasks),
                reason="task_router:no_candidates",
            )

        tool_names_by_command_id = _tool_names_by_command_id(candidates)
        decision = await self._get_router().route_tasks(
            tasks=[task.to_payload() for task in tasks],
            candidates=candidates,
            tool_names_by_command_id=tool_names_by_command_id,
            router_context=router_context,
        )
        routes = _results_from_batch_decision(
            tasks,
            decision=decision,
            candidate_count=len(candidates),
        )
        if not routes:
            return TaskRouterResult(reason="no_tasks", candidates=tuple(candidates))
        if any(route.status != "selected" for route in routes):
            reason = "task_router:partial_or_uncertain"
        else:
            reason = "task_router:all_selected"
        return TaskRouterResult(
            routes=tuple(routes),
            reason=reason,
            candidates=tuple(candidates),
        )

    def _get_router(self) -> Any:
        if self._router is None:
            self._router = ToolRouter(
                trace_id=f"{self._trace_id}:task-router",
                model_name=self._model_name,
                generation_config=self._generation_config,
                timeout=self._timeout,
                usage_callback=self._usage_callback,
            )
        return self._router


def _tool_names_by_command_id(candidates: list[CommandCandidate]) -> dict[str, str]:
    result: dict[str, str] = {}
    for tool in build_native_command_tools(candidates):
        command_id = normalize_message_text(tool.binding.command_id)
        tool_name = normalize_message_text(tool.binding.tool_name)
        if command_id and tool_name:
            result[command_id] = tool_name
    return result


def _dedupe_candidates(
    candidates: list[CommandCandidate],
    *,
    limit: int,
) -> list[CommandCandidate]:
    by_id: dict[str, CommandCandidate] = {}
    for candidate in candidates:
        command_id = normalize_message_text(candidate.schema.command_id)
        if not command_id:
            continue
        previous = by_id.get(command_id)
        if previous is None or _candidate_rank(candidate) > _candidate_rank(previous):
            by_id[command_id] = candidate
    return sorted(by_id.values(), key=_candidate_rank, reverse=True)[
        : max(1, int(limit or 1))
    ]


def _candidate_rank(candidate: CommandCandidate) -> tuple[int, float, str]:
    return (
        1 if candidate.exact_protected else 0,
        float(candidate.score or 0.0),
        normalize_message_text(candidate.schema.command_id),
    )


def _results_from_batch_decision(
    tasks: tuple[TaskItem, ...],
    *,
    decision: ToolRouterBatchDecision,
    candidate_count: int,
) -> list[TaskRouteResult]:
    selections = {
        normalize_message_text(selection.task_id): selection
        for selection in decision.selections
        if normalize_message_text(selection.task_id)
    }
    routes: list[TaskRouteResult] = []
    for task in tasks:
        selection = selections.get(normalize_message_text(task.task_id))
        if selection is None:
            routes.append(
                TaskRouteResult(
                    task_id=task.task_id,
                    text=task.text,
                    order=task.order,
                    status="unsupported",
                    reason=normalize_message_text(decision.reason)
                    or "task_router:no_match",
                    candidate_count=candidate_count,
                )
            )
            continue
        routes.append(
            _result_from_selection(
                task,
                selection=selection,
                candidate_count=candidate_count,
            )
        )
    return routes


def _result_from_selection(
    task: TaskItem,
    *,
    selection: ToolRouterSelection,
    candidate_count: int,
) -> TaskRouteResult:
    command_id = normalize_message_text(selection.command_id)
    tool_name = normalize_message_text(selection.tool_name)
    confidence = _confidence(selection.confidence)
    if command_id and tool_name and confidence >= _MIN_SELECT_CONFIDENCE:
        return TaskRouteResult(
            task_id=task.task_id,
            text=task.text,
            order=task.order,
            status="selected",
            command_id=command_id,
            tool_name=tool_name,
            arguments=dict(selection.arguments),
            confidence=confidence,
            reason="task_router:selected",
            candidate_count=candidate_count,
        )
    return TaskRouteResult(
        task_id=task.task_id,
        text=task.text,
        order=task.order,
        status="unsupported",
        confidence=confidence,
        reason="select_below_confidence_or_missing_binding",
        candidate_count=candidate_count,
    )


def _unsupported(task: TaskItem, reason: str) -> TaskRouteResult:
    return TaskRouteResult(
        task_id=task.task_id,
        text=task.text,
        order=task.order,
        status="unsupported",
        reason=reason,
    )


def _confidence(value: Any) -> float:
    try:
        confidence = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


__all__ = [
    "TaskRouteResult",
    "TaskRouteStatus",
    "TaskRouter",
    "TaskRouterResult",
]
