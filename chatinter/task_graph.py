"""Explicit task graph for superuser Agent runs.

The graph is deliberately small and JSON-friendly.  It is not a local planner
that decides which tool to call; it is a durable contract: what needs to be
done, which tools may help, and what observations/artifacts prove completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid

from .route_text import normalize_message_text

TASK_STATUS_PENDING = "pending"
TASK_STATUS_IN_PROGRESS = "in_progress"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_BLOCKED = "blocked"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_SKIPPED = "skipped"

TERMINAL_TASK_STATUSES = {
    TASK_STATUS_COMPLETED,
    TASK_STATUS_BLOCKED,
    TASK_STATUS_FAILED,
    TASK_STATUS_SKIPPED,
}


@dataclass
class TaskGraphTask:
    task_id: str
    goal: str
    required_tools: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    status: str = TASK_STATUS_PENDING
    observations: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def add_observation(self, payload: dict[str, Any]) -> None:
        compact = _compact_observation(payload)
        if compact:
            self.observations.append(compact)
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        if isinstance(artifacts, list | tuple):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                artifact_id = normalize_message_text(str(item.get("artifact_id") or ""))
                if not artifact_id:
                    continue
                if any(
                    normalize_message_text(str(old.get("artifact_id") or ""))
                    == artifact_id
                    for old in self.artifacts
                ):
                    continue
                self.artifacts.append(
                    {
                        "artifact_id": artifact_id,
                        "type": normalize_message_text(str(item.get("type", "") or "")),
                        "summary": normalize_message_text(
                            str(item.get("summary", "") or "")
                        )[:300],
                    }
                )


@dataclass
class TaskGraph:
    graph_id: str
    original_goal: str
    tasks: list[TaskGraphTask] = field(default_factory=list)
    status: str = TASK_STATUS_PENDING
    reason: str = ""

    @classmethod
    def create(
        cls,
        *,
        original_goal: str,
        tasks: list[TaskGraphTask] | None = None,
        graph_id: str | None = None,
    ) -> "TaskGraph":
        graph = cls(
            graph_id=normalize_message_text(graph_id or uuid.uuid4().hex[:10]),
            original_goal=normalize_message_text(original_goal),
            tasks=list(tasks or []),
        )
        graph.refresh_status()
        return graph

    @property
    def incomplete_tasks(self) -> list[TaskGraphTask]:
        return [
            task for task in self.tasks if task.status not in TERMINAL_TASK_STATUSES
        ]

    @property
    def completed_tasks(self) -> list[TaskGraphTask]:
        return [task for task in self.tasks if task.status == TASK_STATUS_COMPLETED]

    def add_task(self, task: TaskGraphTask) -> None:
        if not task.goal:
            return
        if any(_same_goal(old.goal, task.goal) for old in self.tasks):
            return
        self.tasks.append(task)
        self.refresh_status()

    def task_by_id(self, task_id: str) -> TaskGraphTask | None:
        normalized = normalize_message_text(task_id)
        if not normalized:
            return None
        for task in self.tasks:
            if task.task_id == normalized:
                return task
        return None

    def refresh_status(self) -> str:
        if not self.tasks:
            self.status = TASK_STATUS_COMPLETED
            return self.status
        if all(task.status == TASK_STATUS_COMPLETED for task in self.tasks):
            self.status = TASK_STATUS_COMPLETED
        elif any(
            task.status in {TASK_STATUS_PENDING, TASK_STATUS_IN_PROGRESS}
            for task in self.tasks
        ):
            self.status = TASK_STATUS_IN_PROGRESS
        elif any(task.status == TASK_STATUS_FAILED for task in self.tasks):
            self.status = TASK_STATUS_FAILED
        elif any(task.status == TASK_STATUS_BLOCKED for task in self.tasks):
            self.status = TASK_STATUS_BLOCKED
        else:
            self.status = TASK_STATUS_SKIPPED
        return self.status

    def to_public_payload(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "original_goal": self.original_goal,
            "status": self.status,
            "reason": self.reason,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "required_tools": list(task.required_tools),
                    "acceptance_criteria": list(task.acceptance_criteria),
                    "status": task.status,
                    "reason": task.reason,
                    "observation_count": len(task.observations),
                    "artifacts": list(task.artifacts[-6:]),
                }
                for task in self.tasks
            ],
        }


def fallback_task_graph(
    *,
    original_goal: str,
    available_tools: list[dict[str, Any]] | None = None,
) -> TaskGraph:
    """Create a conservative graph when LLM planning is unavailable."""

    tools = [
        normalize_message_text(str(item.get("tool", "") or ""))
        for item in (available_tools or [])[:8]
        if isinstance(item, dict)
        and normalize_message_text(str(item.get("tool", "") or ""))
    ]
    goal = normalize_message_text(original_goal)
    if not goal:
        return TaskGraph.create(original_goal=original_goal)
    return TaskGraph.create(
        original_goal=goal,
        tasks=[
            TaskGraphTask(
                task_id="task_1",
                goal=goal,
                required_tools=tools,
                acceptance_criteria=[
                    "Use observations from real tools for factual claims.",
                    "Summarize result honestly if no tool is needed or available.",
                ],
            )
        ],
    )


def task_graph_from_payload(payload: Any) -> TaskGraph | None:
    if not isinstance(payload, dict):
        return None
    tasks: list[TaskGraphTask] = []
    for item in payload.get("tasks", []) or []:
        if not isinstance(item, dict):
            continue
        goal = normalize_message_text(str(item.get("goal", "") or ""))
        task_id = normalize_message_text(str(item.get("task_id", "") or ""))
        if not goal or not task_id:
            continue
        tasks.append(
            TaskGraphTask(
                task_id=task_id,
                goal=goal,
                required_tools=_text_list(item.get("required_tools")),
                acceptance_criteria=_text_list(item.get("acceptance_criteria")),
                status=_normalize_status(str(item.get("status", "") or "")),
                observations=[
                    dict(row)
                    for row in item.get("observations", []) or []
                    if isinstance(row, dict)
                ],
                artifacts=[
                    dict(row)
                    for row in item.get("artifacts", []) or []
                    if isinstance(row, dict)
                ],
                reason=normalize_message_text(str(item.get("reason", "") or "")),
            )
        )
    graph = TaskGraph.create(
        graph_id=normalize_message_text(str(payload.get("graph_id", "") or "")) or None,
        original_goal=normalize_message_text(
            str(payload.get("original_goal", "") or "")
        ),
        tasks=tasks,
    )
    status = _normalize_status(str(payload.get("status", "") or ""))
    if status:
        graph.status = status
    graph.reason = normalize_message_text(str(payload.get("reason", "") or ""))
    return graph


def task_graph_to_payload(graph: TaskGraph | None) -> dict[str, Any] | None:
    if graph is None:
        return None
    return {
        "graph_id": graph.graph_id,
        "original_goal": graph.original_goal,
        "status": graph.status,
        "reason": graph.reason,
        "tasks": [
            {
                "task_id": task.task_id,
                "goal": task.goal,
                "required_tools": list(task.required_tools),
                "acceptance_criteria": list(task.acceptance_criteria),
                "status": task.status,
                "observations": list(task.observations[-12:]),
                "artifacts": list(task.artifacts[-12:]),
                "reason": task.reason,
            }
            for task in graph.tasks
        ],
    }


def _compact_observation(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result = {
        "ok": bool(payload.get("ok")),
        "status": normalize_message_text(str(payload.get("status", "") or "")),
        "tool_name": normalize_message_text(str(payload.get("tool_name", "") or "")),
        "command_id": normalize_message_text(str(payload.get("command_id", "") or "")),
        "task_text": normalize_message_text(str(payload.get("task_text", "") or "")),
        "error": normalize_message_text(str(payload.get("error", "") or ""))[:300],
    }
    return {key: value for key, value in result.items() if value not in {"", False}}


def _same_goal(left: str, right: str) -> bool:
    first = normalize_message_text(left)
    second = normalize_message_text(right)
    return bool(
        first and second and (first == second or first in second or second in first)
    )


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[str] = []
    for item in value:
        text = normalize_message_text(str(item or ""))
        if text and text not in result:
            result.append(text)
    return result[:12]


def _normalize_status(status: str) -> str:
    normalized = normalize_message_text(status)
    if normalized in {
        TASK_STATUS_PENDING,
        TASK_STATUS_IN_PROGRESS,
        TASK_STATUS_COMPLETED,
        TASK_STATUS_BLOCKED,
        TASK_STATUS_FAILED,
        TASK_STATUS_SKIPPED,
    }:
        return normalized
    return TASK_STATUS_PENDING


__all__ = [
    "TASK_STATUS_BLOCKED",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_FAILED",
    "TASK_STATUS_IN_PROGRESS",
    "TASK_STATUS_PENDING",
    "TASK_STATUS_SKIPPED",
    "TERMINAL_TASK_STATUSES",
    "TaskGraph",
    "TaskGraphTask",
    "fallback_task_graph",
    "task_graph_from_payload",
    "task_graph_to_payload",
]
