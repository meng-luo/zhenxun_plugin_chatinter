from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from zhenxun.services.llm.types import LLMToolCall

from .tool_registry import ChatInterToolRegistry
from .turn_runtime import TurnBudgetController


class ToolExecutionMode(str, Enum):
    READONLY = "readonly"
    WRITE = "write"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolBatch:
    mode: ToolExecutionMode
    concurrency_safe: bool
    tool_calls: list[LLMToolCall]


@dataclass(frozen=True)
class ToolExecutionPlan:
    batches: list[ToolBatch]


def build_tool_execution_plan(tool_calls: list[LLMToolCall]) -> ToolExecutionPlan:
    batches: list[ToolBatch] = []
    for call in tool_calls:
        tool_name = str(getattr(call.function, "name", "") or "").strip()
        mode = _resolve_tool_mode(tool_name)
        concurrency_safe = (
            mode == ToolExecutionMode.READONLY
            and ChatInterToolRegistry.is_concurrency_safe(tool_name)
        )
        if (
            batches
            and batches[-1].mode == mode
            and batches[-1].concurrency_safe == concurrency_safe
        ):
            batches[-1].tool_calls.append(call)
            continue
        batches.append(
            ToolBatch(
                mode=mode,
                concurrency_safe=concurrency_safe,
                tool_calls=[call],
            )
        )
    return ToolExecutionPlan(batches=batches)


def allow_tool_batch(
    controller: TurnBudgetController | None,
    batch: ToolBatch,
) -> bool:
    if controller is None:
        return True
    return controller.allow_tool_batch(
        call_count=len(batch.tool_calls),
        batch_kind=batch.mode.value,
    )


def record_tool_batch(
    controller: TurnBudgetController | None,
    *,
    batch: ToolBatch,
    duration: float,
) -> None:
    if controller is None:
        return
    controller.record_tool_batch(
        batch_kind=batch.mode.value,
        duration=duration,
    )


def _resolve_tool_mode(tool_name: str) -> ToolExecutionMode:
    raw_mode = ChatInterToolRegistry.get_access_mode(tool_name)
    if raw_mode == "readonly":
        return ToolExecutionMode.READONLY
    if raw_mode == "write":
        return ToolExecutionMode.WRITE
    return ToolExecutionMode.UNKNOWN


__all__ = [
    "ToolBatch",
    "ToolExecutionMode",
    "ToolExecutionPlan",
    "allow_tool_batch",
    "build_tool_execution_plan",
    "record_tool_batch",
]
