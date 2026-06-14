"""Observation-backed coverage for task-routed command execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .route_text import normalize_message_text
from .task_execution_queue import TaskExecutionQueueResult, TaskObservation
from .task_router import TaskRouteResult, TaskRouterResult

TaskCoverageStatus = Literal[
    "completed",
    "failed",
    "unsupported",
    "clarify",
    "missing",
]


@dataclass(frozen=True, slots=True)
class TaskCoverageItem:
    """Coverage result for one planner task.

    The source of truth is the router task plus its concrete observation.  A
    final LLM reply is intentionally not part of this decision.
    """

    task_id: str
    text: str
    status: TaskCoverageStatus
    command_id: str = ""
    reason: str = ""
    observation_index: int = -1
    output_summary: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "text": self.text,
            "status": self.status,
            "command_id": self.command_id,
            "reason": self.reason,
            "observation_index": self.observation_index,
            "output_summary": self.output_summary,
        }
        return {
            key: value
            for key, value in payload.items()
            if value not in ("", None) and value != -1
        }


@dataclass(frozen=True, slots=True)
class TaskCoverageReport:
    items: tuple[TaskCoverageItem, ...] = ()
    reason: str = ""
    orphan_observation_count: int = 0

    @property
    def covered(self) -> bool:
        """Whether every routed task has a concrete observation record."""

        return bool(self.items) and all(item.status != "missing" for item in self.items)

    @property
    def all_completed(self) -> bool:
        return bool(self.items) and all(
            item.status == "completed" for item in self.items
        )

    @property
    def completed(self) -> tuple[TaskCoverageItem, ...]:
        return tuple(item for item in self.items if item.status == "completed")

    @property
    def failed(self) -> tuple[TaskCoverageItem, ...]:
        return tuple(item for item in self.items if item.status == "failed")

    @property
    def unsupported(self) -> tuple[TaskCoverageItem, ...]:
        return tuple(item for item in self.items if item.status == "unsupported")

    @property
    def clarify(self) -> tuple[TaskCoverageItem, ...]:
        return tuple(item for item in self.items if item.status == "clarify")

    @property
    def missing(self) -> tuple[TaskCoverageItem, ...]:
        return tuple(item for item in self.items if item.status == "missing")

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": "task_coverage",
            "reason": self.reason,
            "covered": self.covered,
            "all_completed": self.all_completed,
            "completed_count": len(self.completed),
            "failed_count": len(self.failed),
            "unsupported_count": len(self.unsupported),
            "clarify_count": len(self.clarify),
            "missing_count": len(self.missing),
            "orphan_observation_count": self.orphan_observation_count,
            "items": [item.to_payload() for item in self.items],
        }


def build_task_coverage_report(
    route_result: TaskRouterResult,
    queue_result: TaskExecutionQueueResult,
) -> TaskCoverageReport:
    observations = tuple(queue_result.observations)
    observations_by_task: dict[str, tuple[int, TaskObservation]] = {}
    for index, observation in enumerate(observations):
        task_id = normalize_message_text(observation.task_id)
        if task_id and task_id not in observations_by_task:
            observations_by_task[task_id] = (index, observation)

    items = tuple(
        _coverage_item_for_route(route, observations_by_task.get(route.task_id))
        for route in sorted(route_result.routes, key=lambda item: item.order)
    )
    routed_task_ids = {
        normalize_message_text(route.task_id) for route in route_result.routes
    }
    orphan_count = sum(
        1
        for observation in observations
        if normalize_message_text(observation.task_id) not in routed_task_ids
    )
    if not items:
        reason = "task_coverage:no_routes"
    elif all(item.status == "completed" for item in items):
        reason = "task_coverage:all_completed"
    elif any(item.status == "missing" for item in items):
        reason = "task_coverage:missing_observation"
    else:
        reason = "task_coverage:partial_or_failed"
    return TaskCoverageReport(
        items=items,
        reason=reason,
        orphan_observation_count=orphan_count,
    )


def synthesize_task_coverage_reply(report: TaskCoverageReport) -> str:
    """Build the final reply from coverage facts only.

    This keeps the LLM out of completion judgment: completed means a task has
    an ok observation, while every other state is explicitly listed as not done.
    """

    if not report.items:
        return "没有可执行的明确任务。"

    completed = [_format_completed_item(item) for item in report.completed]
    unfinished = [
        _format_unfinished_item(item)
        for item in report.items
        if item.status != "completed"
    ]
    parts: list[str] = []
    if completed:
        parts.append("已完成：" + "；".join(completed))
    if unfinished:
        parts.append("未完成：" + "；".join(unfinished))
    return "；".join(parts)


def _coverage_item_for_route(
    route: TaskRouteResult,
    observation_pair: tuple[int, TaskObservation] | None,
) -> TaskCoverageItem:
    if observation_pair is None:
        return TaskCoverageItem(
            task_id=route.task_id,
            text=route.text,
            status="missing",
            command_id=route.command_id,
            reason="没有执行结果",
        )

    observation_index, observation = observation_pair
    command_id = observation.command_id or route.command_id
    if route.status == "clarify":
        return TaskCoverageItem(
            task_id=route.task_id,
            text=route.text,
            status="clarify",
            command_id=command_id,
            reason=_route_reason(route, observation, fallback="需要澄清"),
            observation_index=observation_index,
            output_summary=_output_summary(observation.output),
        )
    if route.status == "unsupported":
        return TaskCoverageItem(
            task_id=route.task_id,
            text=route.text,
            status="unsupported",
            command_id=command_id,
            reason=_route_reason(route, observation, fallback="未找到可用命令"),
            observation_index=observation_index,
            output_summary=_output_summary(observation.output),
        )
    if observation.ok:
        return TaskCoverageItem(
            task_id=route.task_id,
            text=route.text,
            status="completed",
            command_id=command_id,
            reason="observation_ok",
            observation_index=observation_index,
            output_summary=_output_summary(observation.output),
        )
    return TaskCoverageItem(
        task_id=route.task_id,
        text=route.text,
        status="failed",
        command_id=command_id,
        reason=_route_reason(route, observation, fallback="执行失败"),
        observation_index=observation_index,
        output_summary=_output_summary(observation.output),
    )


def _route_reason(
    route: TaskRouteResult,
    observation: TaskObservation,
    *,
    fallback: str,
) -> str:
    for value in (
        observation.error,
        route.clarification_question,
        route.reason,
        _output_error(observation.output),
    ):
        text = normalize_message_text(str(value or ""))
        if text:
            return text
    return fallback


def _format_completed_item(item: TaskCoverageItem) -> str:
    label = _item_label(item)
    if item.output_summary:
        return f"{label}（{item.output_summary}）"
    return label


def _format_unfinished_item(item: TaskCoverageItem) -> str:
    reason = _friendly_reason(item.status, item.reason)
    return f"{_item_label(item)}（{reason}）"


def _friendly_reason(status: TaskCoverageStatus, reason: str) -> str:
    text = normalize_message_text(reason)
    if status == "unsupported":
        return "未找到可用命令" if _is_internal_unsupported_reason(text) else text
    if status == "clarify":
        return f"需要澄清：{text}" if text else "需要澄清"
    if status == "missing":
        return text or "没有执行结果"
    if status == "failed":
        return f"执行失败：{text}" if text else "执行失败"
    return text or status


def _is_internal_unsupported_reason(reason: str) -> bool:
    if not reason:
        return True
    return reason in {
        "no_candidates",
        "no_match",
        "unsupported",
        "task_router:no_match",
        "select_below_confidence_or_missing_binding",
    }


def _item_label(item: TaskCoverageItem) -> str:
    return (
        normalize_message_text(item.text)
        or normalize_message_text(item.task_id)
        or "任务"
    )


def _output_error(output: dict[str, Any]) -> str:
    value = output.get("error") if isinstance(output, dict) else ""
    return normalize_message_text(str(value or ""))


def _output_summary(output: dict[str, Any]) -> str:
    if not isinstance(output, dict):
        return ""
    for key in ("messages_sent_summary", "visible_output"):
        value = output.get(key)
        if isinstance(value, str) and value:
            return normalize_message_text(value)[:180]
    messages = output.get("messages_sent")
    if isinstance(messages, list):
        joined = " ".join(normalize_message_text(str(item or "")) for item in messages)
        if joined.strip():
            return joined.strip()[:180]
    return ""


__all__ = [
    "TaskCoverageItem",
    "TaskCoverageReport",
    "TaskCoverageStatus",
    "build_task_coverage_report",
    "synthesize_task_coverage_reply",
]
