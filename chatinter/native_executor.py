"""Execution bridge for native ChatInter plugin command tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from zhenxun.services.llm.types.models import ToolResult

from .command_index import CommandCandidate
from .models.pydantic_models import CommandSlotSpec
from .native_command_tools import NativeCommandToolBinding
from .native_route import (
    NativeCommandSelection,
    NativeRouteDecision,
    NativeRouteReport,
    NativeRouteResult,
    NativeSlotValue,
    candidate_selection_to_native_route,
)
from .route_text import normalize_message_text

_NATIVE_EXECUTION_STAGE = "main_request"


@dataclass(frozen=True)
class NativeValidatedRoute:
    decision: NativeRouteDecision
    route_result: NativeRouteResult | None
    reason: str


@dataclass(frozen=True)
class NativeToolExecutionResult:
    success: bool
    route_result: NativeRouteResult | None
    route_command: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    display_text: str = ""
    reason: str = ""


ExecuteNativeRoute = Callable[
    [NativeValidatedRoute, NativeRouteReport],
    Awaitable[NativeToolExecutionResult],
]


@dataclass
class NativeCommandExecutionContext:
    candidates: list[CommandCandidate]
    has_reply: bool
    report: NativeRouteReport
    route_executor: ExecuteNativeRoute
    message_text: str
    executions: list[NativeToolExecutionResult] = field(default_factory=list)

    async def execute_tool(
        self,
        *,
        binding: NativeCommandToolBinding,
        raw_slots: dict[str, Any],
    ) -> ToolResult:
        validated = self._validate_tool_call(binding=binding, raw_slots=raw_slots)
        if validated is None:
            return ToolResult(
                output={
                    "ok": False,
                    "status": "failed",
                    "error_type": "InvalidToolCall",
                    "message": "工具调用未通过本地校验，请重新选择候选工具或直接聊天。",
                    "is_retryable": True,
                },
                display_content="工具调用校验失败",
            )

        execution = await self.route_executor(validated, self.report)
        self.executions.append(execution)
        self._finalize_report(validated=validated, execution=execution)
        return ToolResult(
            output=execution.output,
            display_content=execution.display_text or execution.reason,
        )

    def _validate_tool_call(
        self,
        *,
        binding: NativeCommandToolBinding,
        raw_slots: dict[str, Any],
    ) -> NativeValidatedRoute | None:
        candidate = binding.candidate
        slots = normalize_native_tool_slots(candidate.schema.slots, raw_slots)
        selection = NativeCommandSelection(
            action="execute",
            command_id=binding.command_id,
            slots=[
                NativeSlotValue(name=name, value=str(value))
                for name, value in slots.items()
                if normalize_message_text(name)
            ],
            confidence=0.9,
            reason=f"native_tool_call:{binding.tool_name};validated",
        )
        route = candidate_selection_to_native_route(
            selection=selection,
            candidates=self.candidates,
            message_text=self.message_text,
            stage=_NATIVE_EXECUTION_STAGE,
            has_reply=self.has_reply,
        )
        if route is None:
            return None
        decision, route_result = route
        return NativeValidatedRoute(
            decision=decision,
            route_result=route_result,
            reason=selection.reason,
        )

    def _finalize_report(
        self,
        *,
        validated: NativeValidatedRoute,
        execution: NativeToolExecutionResult,
    ) -> None:
        if execution.route_result is None or self.report.final_reason != "init":
            return
        self.report.finalize(
            reason=validated.decision.reason or validated.reason,
            stage=execution.route_result.stage,
            plugin_name=execution.route_result.decision.plugin_name,
            plugin_module=execution.route_result.decision.plugin_module,
            command=execution.route_result.decision.command,
        )


def normalize_native_tool_slots(
    slot_specs: list[CommandSlotSpec],
    raw_slots: dict[str, Any],
) -> dict[str, str]:
    slot_by_key: dict[str, CommandSlotSpec] = {}
    for slot in slot_specs:
        keys = [slot.name, *slot.aliases]
        for key in keys:
            normalized = normalize_message_text(str(key or ""))
            if normalized:
                slot_by_key[normalized] = slot

    normalized_slots: dict[str, str] = {}
    for key, value in raw_slots.items():
        slot = slot_by_key.get(normalize_message_text(str(key or "")))
        if slot is None:
            continue
        coerced = _coerce_slot_value(slot, value)
        if coerced is None:
            continue
        normalized_slots[slot.name] = coerced
    return normalized_slots


def _coerce_slot_value(slot: CommandSlotSpec, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = normalize_message_text(value)
        if not text or text.lower() in {"null", "none", "undefined"}:
            return None
    else:
        text = str(value)

    if slot.type == "bool":
        lowered = text.strip().lower()
        if lowered in {"1", "true", "yes", "on", "是", "开启"}:
            return "true"
        if lowered in {"0", "false", "no", "off", "否", "关闭"}:
            return "false"
        return None

    if slot.type == "int":
        try:
            return str(int(float(text)))
        except (TypeError, ValueError):
            return None

    if slot.type == "float":
        try:
            return str(float(text))
        except (TypeError, ValueError):
            return None

    return normalize_message_text(text)


__all__ = [
    "ExecuteNativeRoute",
    "NativeCommandExecutionContext",
    "NativeToolExecutionResult",
    "NativeValidatedRoute",
    "normalize_native_tool_slots",
]
