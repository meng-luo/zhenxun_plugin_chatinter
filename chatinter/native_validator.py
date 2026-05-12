"""Local validation and fallback execution for native ChatInter tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .command_index import CommandCandidate
from .models.pydantic_models import CommandSlotSpec
from .native_command_tools import (
    NativeCommandToolBinding,
    parse_native_tool_arguments,
)
from .route_engine import (
    LLMCommandSelection,
    LLMRouterDecision,
    LLMSlotValue,
    RouteResolveResult,
    _candidate_selection_to_route_result,
    resolve_local_candidate_selection,
)
from .route_text import normalize_message_text


@dataclass(frozen=True)
class NativeValidatedRoute:
    decision: LLMRouterDecision
    route_result: RouteResolveResult | None
    reason: str


def validate_native_tool_call_route(
    *,
    tool_call: Any,
    bindings: dict[str, NativeCommandToolBinding],
    candidates: list[CommandCandidate],
    message_text: str,
    has_reply: bool,
) -> NativeValidatedRoute | None:
    function = getattr(tool_call, "function", None)
    tool_name = normalize_message_text(getattr(function, "name", "") or "")
    if not tool_name or tool_name not in bindings:
        return None

    binding = bindings[tool_name]
    candidate = binding.candidate
    raw_slots = parse_native_tool_arguments(tool_call)
    slots = normalize_native_tool_slots(candidate.schema.slots, raw_slots)
    selection = LLMCommandSelection(
        action="execute",
        command_id=binding.command_id,
        slots=[
            LLMSlotValue(name=name, value=str(value))
            for name, value in slots.items()
            if normalize_message_text(name)
        ],
        confidence=0.9,
        reason=f"native_tool_call:{tool_name};validated",
    )
    route = _candidate_selection_to_route_result(
        selection=selection,
        candidates=candidates,
        message_text=message_text,
        stage="native_tools",
        has_reply=has_reply,
    )
    if route is None:
        return None
    decision, route_result = route
    return NativeValidatedRoute(
        decision=decision,
        route_result=route_result,
        reason=selection.reason,
    )


def resolve_local_native_fallback(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    has_reply: bool,
    reason: str,
) -> NativeValidatedRoute | None:
    selection = resolve_local_candidate_selection(
        message_text=message_text,
        candidates=candidates,
        has_reply=has_reply,
    )
    if selection is None:
        return None
    selection.reason = f"{reason};{selection.reason}"
    route = _candidate_selection_to_route_result(
        selection=selection,
        candidates=candidates,
        message_text=message_text,
        stage="native_local",
        has_reply=has_reply,
    )
    if route is None:
        return None
    decision, route_result = route
    return NativeValidatedRoute(
        decision=decision,
        route_result=route_result,
        reason=selection.reason,
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
    "NativeValidatedRoute",
    "normalize_native_tool_slots",
    "resolve_local_native_fallback",
    "validate_native_tool_call_route",
]
