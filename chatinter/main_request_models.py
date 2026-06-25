"""Stable result types for ChatInter main request dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zhenxun.services.llm.types.models import ToolResult

from .native_executor import NativeToolExecutionResult
from .native_route import NativeRouteDecision, NativeRouteReport, NativeRouteResult

if TYPE_CHECKING:
    from .tool_intent_gate import ToolIntentGateResult


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


@dataclass(frozen=True)
class ToolObligationDecision:
    obligation: str
    reason: str
    required_tool_names: tuple[str, ...] = ()
    gate_result: "ToolIntentGateResult | None" = None


@dataclass(frozen=True)
class CandidateObligationEvaluation:
    candidate: Any
    score: float
    request_strength: Any
    capability_factor: float
    recall_factor: float
    reliability_factor: float
    schema_factor: float
    requires_real_tool: bool
    real_output_factor: float
    reason: str


MainRequestRouteHook = Callable[[MainRequestResult], Awaitable[None] | None]
MainRequestReplyHook = Callable[[str], Awaitable[str] | str]


__all__ = [
    "CandidateObligationEvaluation",
    "MainRequestOutput",
    "MainRequestReplyHook",
    "MainRequestResult",
    "MainRequestRouteHook",
    "MainRequestTimelineItem",
    "ToolObligationDecision",
]
