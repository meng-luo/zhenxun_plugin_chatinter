"""Task-level command router for obvious multi-command turns.

This module binds TaskPlannerLite items to candidate command tools.  It does
not execute commands; execution order is handled by later stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .command_index import CommandCandidate
from .native_command_tools import build_native_command_tools
from .route_text import normalize_message_text
from .task_planner_lite import TaskItem
from .tool_retriever import CommandToolRetriever
from .tool_router import ToolRouter, ToolRouterDecision

TaskRouteStatus = Literal["selected", "clarify", "unsupported"]

_DEFAULT_TASK_RETRIEVAL_LIMIT = 24
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
    clarification_question: str = ""
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
            "clarification_question": self.clarification_question,
            "candidate_count": self.candidate_count,
        }
        return {
            key: value for key, value in payload.items() if value not in ("", {}, None)
        }


@dataclass(frozen=True, slots=True)
class TaskRouterResult:
    routes: tuple[TaskRouteResult, ...] = ()
    reason: str = ""

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
    """Route each TaskItem independently through retriever + ToolRouter."""

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
    ) -> None:
        self.retriever = retriever
        self._router = router
        self._trace_id = trace_id
        self._model_name = model_name
        self._generation_config = generation_config
        self._timeout = timeout
        self.retrieval_limit = max(1, int(retrieval_limit or 1))

    async def route_tasks(self, tasks: tuple[TaskItem, ...]) -> TaskRouterResult:
        routes: list[TaskRouteResult] = []
        for task in tasks:
            routes.append(await self._route_one(task))
        if not routes:
            return TaskRouterResult(reason="no_tasks")
        if any(route.status != "selected" for route in routes):
            reason = "task_router:partial_or_uncertain"
        else:
            reason = "task_router:all_selected"
        return TaskRouterResult(routes=tuple(routes), reason=reason)

    async def _route_one(self, task: TaskItem) -> TaskRouteResult:
        retrieval = self.retriever.retrieve(
            task.text,
            limit=self.retrieval_limit,
        )
        candidates = list(retrieval.candidates)
        if not candidates:
            return _unsupported(task, "no_candidates")

        tool_names_by_command_id = _tool_names_by_command_id(candidates)
        decision = await self._get_router().route(
            message_text=task.text,
            candidates=candidates,
            tool_names_by_command_id=tool_names_by_command_id,
        )
        return _result_from_decision(
            task,
            decision=decision,
            candidate_count=len(candidates),
        )

    def _get_router(self) -> Any:
        if self._router is None:
            self._router = ToolRouter(
                trace_id=f"{self._trace_id}:task-router",
                model_name=self._model_name,
                generation_config=self._generation_config,
                timeout=self._timeout,
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


def _result_from_decision(
    task: TaskItem,
    *,
    decision: ToolRouterDecision,
    candidate_count: int,
) -> TaskRouteResult:
    reason = normalize_message_text(decision.reason)
    if decision.action == "select":
        command_id = normalize_message_text(decision.command_id)
        tool_name = normalize_message_text(decision.tool_name)
        confidence = _confidence(decision.confidence)
        if command_id and tool_name and confidence >= _MIN_SELECT_CONFIDENCE:
            return TaskRouteResult(
                task_id=task.task_id,
                text=task.text,
                order=task.order,
                status="selected",
                command_id=command_id,
                tool_name=tool_name,
                arguments=dict(decision.arguments),
                confidence=confidence,
                reason=reason or "task_router:selected",
                candidate_count=candidate_count,
            )
        return TaskRouteResult(
            task_id=task.task_id,
            text=task.text,
            order=task.order,
            status="unsupported",
            confidence=confidence,
            reason=reason or "select_below_confidence_or_missing_binding",
            candidate_count=candidate_count,
        )
    if decision.action == "clarify" or decision.needs_clarification:
        return TaskRouteResult(
            task_id=task.task_id,
            text=task.text,
            order=task.order,
            status="clarify",
            confidence=_confidence(decision.confidence),
            reason=reason or "task_router:needs_clarification",
            clarification_question=normalize_message_text(
                decision.clarification_question,
            ),
            candidate_count=candidate_count,
        )
    return TaskRouteResult(
        task_id=task.task_id,
        text=task.text,
        order=task.order,
        status="unsupported",
        confidence=_confidence(decision.confidence),
        reason=reason or "task_router:no_match",
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
