"""Sequential executor for task-routed plugin commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .command_observation import build_command_observation
from .native_command_tools import NativeCommandTool, build_native_command_tools
from .native_executor import NativeCommandExecutionContext, NativeToolExecutionResult
from .route_text import normalize_message_text
from .task_router import TaskRouteResult, TaskRouterResult


@dataclass(frozen=True, slots=True)
class TaskObservation:
    task_id: str
    command_id: str = ""
    ok: bool = False
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "command_id": self.command_id,
            "ok": self.ok,
            "output": dict(self.output),
            "error": self.error,
        }
        return {
            key: value for key, value in payload.items() if value not in ("", {}, None)
        }


@dataclass(frozen=True, slots=True)
class TaskExecutionQueueResult:
    observations: tuple[TaskObservation, ...] = ()
    tool_results: tuple[Any, ...] = ()
    reason: str = ""

    @property
    def ok_count(self) -> int:
        return sum(1 for item in self.observations if item.ok)

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.observations if not item.ok)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": "task_execution_queue",
            "reason": self.reason,
            "ok_count": self.ok_count,
            "failed_count": self.failed_count,
            "observations": [item.to_payload() for item in self.observations],
        }


class TaskExecutionQueue:
    """Execute selected task routes sequentially for message-order safety."""

    def __init__(
        self,
        *,
        command_context: NativeCommandExecutionContext,
        candidates: list[Any],
    ) -> None:
        self.command_context = command_context
        self.candidates = list(candidates)
        self.tools_by_command = _tools_by_command(self.candidates)

    async def execute(self, route_result: TaskRouterResult) -> TaskExecutionQueueResult:
        observations: list[TaskObservation] = []
        tool_results: list[Any] = []
        for route in sorted(route_result.routes, key=lambda item: item.order):
            observation, tool_result = await self._execute_route(route)
            observations.append(observation)
            if tool_result is not None:
                tool_results.append(tool_result)
        reason = (
            "task_execution_queue:completed"
            if observations and all(item.ok for item in observations)
            else "task_execution_queue:partial_or_failed"
            if observations
            else "task_execution_queue:no_observations"
        )
        return TaskExecutionQueueResult(
            observations=tuple(observations),
            tool_results=tuple(tool_results),
            reason=reason,
        )

    async def _execute_route(
        self,
        route: TaskRouteResult,
    ) -> tuple[TaskObservation, Any | None]:
        if route.status != "selected":
            return _skipped_observation(route), None
        tool = self.tools_by_command.get(route.command_id)
        if tool is None:
            return _missing_tool_observation(route), None
        raw_slots = dict(route.arguments)
        raw_slots.setdefault("task_text", route.text)
        execution_count_before = len(self.command_context.executions)
        tool_result = await self.command_context.execute_tool(
            binding=tool.binding,
            raw_slots=raw_slots,
        )
        execution = (
            self.command_context.executions[-1]
            if len(self.command_context.executions) > execution_count_before
            else None
        )
        return _observation_from_execution(route, execution, tool_result), tool_result


def synthesize_task_observation_reply(
    observations: tuple[TaskObservation, ...] | list[TaskObservation],
) -> str:
    """Legacy observation-only summary.

    New multi-task completion judgment must use ``task_coverage`` because route
    status is required to distinguish failed, unsupported and missing tasks.
    """

    if not observations:
        return ""
    parts: list[str] = []
    for observation in observations:
        label = _task_label(observation)
        if observation.ok:
            summary = _output_summary(observation.output)
            parts.append(f"{label}完成" + (f"：{summary}" if summary else ""))
        else:
            error = normalize_message_text(observation.error) or "未完成"
            parts.append(f"{label}失败：{error}")
    return "；".join(parts)


def _task_label(observation: TaskObservation) -> str:
    task_text = ""
    if isinstance(observation.output, dict):
        task_text = normalize_message_text(str(observation.output.get("task_text", "")))
    return task_text or normalize_message_text(observation.task_id) or "任务"


def _tools_by_command(candidates: list[Any]) -> dict[str, NativeCommandTool]:
    result: dict[str, NativeCommandTool] = {}
    for tool in build_native_command_tools(candidates):
        command_id = normalize_message_text(tool.binding.command_id)
        if command_id:
            result[command_id] = tool
    return result


def _skipped_observation(route: TaskRouteResult) -> TaskObservation:
    reason = (
        route.clarification_question
        if route.status == "clarify"
        else route.reason or "unsupported"
    )
    return TaskObservation(
        task_id=route.task_id,
        command_id=route.command_id,
        ok=False,
        error=normalize_message_text(reason),
        output=build_command_observation(
            ok=False,
            command_id=route.command_id,
            rendered_command="",
            matched_plugin="",
            task_text=route.text,
            error=reason,
            retryable=route.status == "clarify",
        ),
    )


def _missing_tool_observation(route: TaskRouteResult) -> TaskObservation:
    error = "selected command binding is missing"
    return TaskObservation(
        task_id=route.task_id,
        command_id=route.command_id,
        ok=False,
        error=error,
        output=build_command_observation(
            ok=False,
            command_id=route.command_id,
            rendered_command="",
            matched_plugin="",
            task_text=route.text,
            error=error,
            retryable=True,
        ),
    )


def _observation_from_execution(
    route: TaskRouteResult,
    execution: NativeToolExecutionResult | None,
    tool_result: Any,
) -> TaskObservation:
    output = getattr(tool_result, "output", None)
    if not isinstance(output, dict):
        output = {}
    ok = bool(execution.success if execution is not None else output.get("ok", False))
    error = normalize_message_text(
        str(output.get("error", "") or getattr(execution, "reason", "") or "")
    )
    return TaskObservation(
        task_id=route.task_id,
        command_id=route.command_id,
        ok=ok,
        output=dict(output),
        error=error,
    )


def _output_summary(output: dict[str, Any]) -> str:
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
    "TaskExecutionQueue",
    "TaskExecutionQueueResult",
    "TaskObservation",
    "synthesize_task_observation_reply",
]
