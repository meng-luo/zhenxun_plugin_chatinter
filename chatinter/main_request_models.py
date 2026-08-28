"""Stable result types for ChatInter main request dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .llm_compat import ToolResult
from .native_executor import NativeToolExecutionResult
from .native_route import NativeRouteDecision, NativeRouteReport, NativeRouteResult


@dataclass(frozen=True)
class MainRequestOutput:
    analysis: str = "main request"
    final_text: str = ""
    memory_text: str = ""
    should_send: bool = True
    outcome: str = "chat_completed"
    feedback_kind: str = "chat_completed"
    record_chat_feedback: bool = True
    observation_reason: str = "chat_completed"
    tool_outcome: str = ""
    nontext_delivery: bool = False


@dataclass(frozen=True)
class MainRequestTimelineItem:
    role: str
    kind: str
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "kind": self.kind,
        }
        if self.content:
            payload["content"] = self.content
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class MainRequestResult:
    decision: NativeRouteDecision
    route_result: NativeRouteResult | None
    report: NativeRouteReport
    executions: tuple[NativeToolExecutionResult, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    timeline: tuple[MainRequestTimelineItem, ...] = ()
    output: MainRequestOutput = field(default_factory=MainRequestOutput)

    @property
    def handled_by_tools(self) -> bool:
        return any(item.route_result is not None for item in self.executions)


MainRequestRouteHook = Callable[[MainRequestResult], Awaitable[None] | None]
MainRequestReplyHook = Callable[[str], Awaitable[str] | str]


__all__ = [
    "MainRequestOutput",
    "MainRequestReplyHook",
    "MainRequestResult",
    "MainRequestRouteHook",
    "MainRequestTimelineItem",
]
