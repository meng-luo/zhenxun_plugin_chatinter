"""Execution bridge for native ChatInter plugin command tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
import shlex
from typing import Any

from .command_index import CommandCandidate
from .command_observation import build_command_observation
from .llm_compat import ToolResult
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
from .route_text import (
    collect_placeholders,
    normalize_message_text,
    parse_command_with_head,
    sanitize_template_tail,
)

_PAYLOAD_EXPLANATION_MARKERS = (
    "作为",
    "将",
    "把",
    "直接",
    "输入",
    "填写",
    "使用",
    "文案",
    "文本",
    "内容",
    "目标",
    "无需",
    "需要",
    "指定",
    "说明",
    "参数",
    "：",
    ":",
    "@",
)
from .task_frame import TaskFrame, pop_task_context, pop_task_text

_NATIVE_EXECUTION_STAGE = "main_request"


@dataclass(frozen=True)
class NativeValidatedRoute:
    decision: NativeRouteDecision
    route_result: NativeRouteResult | None
    reason: str
    task_frame: TaskFrame | None = None
    candidate: CommandCandidate | None = None


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
    task_count: int = 0

    async def execute_tool(
        self,
        *,
        binding: NativeCommandToolBinding,
        raw_slots: dict[str, Any],
    ) -> ToolResult:
        validated = self._validate_tool_call(binding=binding, raw_slots=raw_slots)
        if validated is None:
            task_text, _slots = pop_task_text(raw_slots)
            return _failure_tool_result(
                binding=binding,
                task_text=task_text,
                ambient_message=self.message_text,
                error="工具调用未通过本地校验，请重新选择候选工具或直接聊天。",
                display_content="工具调用校验失败",
                retryable=True,
            )

        try:
            execution = await self.route_executor(validated, self.report)
        except Exception as exc:
            execution = _execution_failure_from_exception(
                binding=binding,
                validated=validated,
                ambient_message=self.message_text,
                exc=exc,
            )
        execution = _ensure_execution_observation(
            binding=binding,
            execution=execution,
            validated=validated,
            ambient_message=self.message_text,
        )
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
        task_text, target_hint, payload_hint, plugin_raw_slots = pop_task_context(
            raw_slots
        )
        target_hint = _normalize_target_hint_for_schema(
            candidate.schema,
            target_hint,
        )
        slots = normalize_native_tool_slots(candidate.schema.slots, plugin_raw_slots)
        slots = _fill_missing_text_slots_from_task(
            candidate.schema,
            slots,
            task_text=task_text,
        )
        slots = _fill_missing_slots_from_payload_hint(
            candidate.schema.slots,
            slots,
            payload_hint=payload_hint,
            target_hint=target_hint,
            task_text=task_text,
        )
        shortcut_command = _shortcut_command_for_task(
            candidate,
            task_text=task_text,
            ambient_message=self.message_text,
        )
        self.task_count += 1
        task_frame = TaskFrame(
            task_index=self.task_count,
            command_id=binding.command_id,
            plugin_module=candidate.plugin_module,
            task_text=task_text,
            fallback_text="",
            slots=dict(slots),
            target_hint=target_hint,
            payload_hint=payload_hint,
            ambient_message=self.message_text,
        )
        route_message_text = _merge_ambient_context_tokens(
            _merge_task_frame_hints(task_frame),
            self.message_text,
        )
        if shortcut_command:
            route_message_text = shortcut_command
            slots = _slots_from_shortcut_command(
                candidate.schema, shortcut_command, slots
            )
            task_frame = replace(task_frame, slots=dict(slots))
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
            candidate=candidate,
            message_text=route_message_text,
            stage=_NATIVE_EXECUTION_STAGE,
            has_reply=self.has_reply,
        )
        if route is None:
            return None
        decision, route_result = route
        if shortcut_command and route_result is not None:
            decision.command = shortcut_command
            route_result = replace(
                route_result,
                decision=replace(route_result.decision, command=shortcut_command),
            )
        return NativeValidatedRoute(
            decision=decision,
            route_result=route_result,
            reason=selection.reason,
            task_frame=task_frame,
            candidate=candidate,
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


def _fill_missing_slots_from_payload_hint(
    slot_specs: list[CommandSlotSpec],
    slots: dict[str, str],
    *,
    payload_hint: str,
    target_hint: str,
    task_text: str = "",
) -> dict[str, str]:
    """Use explicit generic hints only when a command slot is otherwise empty."""

    if not slot_specs:
        return slots
    filled = dict(slots)
    for slot in slot_specs:
        if slot.name in filled:
            continue
        slot_type = str(slot.type or "text")
        if slot_type == "text":
            continue
        value = target_hint if slot_type == "at" else payload_hint
        if not value:
            continue
        coerced = _coerce_slot_value(slot, value)
        if coerced is not None:
            filled[slot.name] = coerced
    return filled


def _fill_missing_text_slots_from_task(
    schema: Any,
    slots: dict[str, str],
    *,
    task_text: str,
) -> dict[str, str]:
    if normalize_message_text(getattr(schema, "payload_policy", "")) != "text":
        return slots
    required = [
        slot
        for slot in list(getattr(schema, "slots", []) or [])
        if bool(getattr(slot, "required", False))
        and str(getattr(slot, "type", "") or "") == "text"
    ]
    if not required or any(slot.name in slots for slot in required):
        return slots
    parsed = parse_command_with_head(
        task_text,
        normalize_message_text(getattr(schema, "head", "")),
        allow_sticky=True,
    )
    if parsed is None:
        return slots
    tail = sanitize_template_tail(parsed.payload_text)
    if not tail:
        return slots
    try:
        values = shlex.split(tail)
    except ValueError:
        values = tail.split()
    if len(required) == 1:
        values = [" ".join(values)]
    if len(values) != len(required):
        return slots
    filled = dict(slots)
    for slot, value in zip(required, values, strict=True):
        coerced = _coerce_slot_value(slot, value)
        if coerced is None:
            return slots
        filled[slot.name] = coerced
    return filled


def _safe_text_payload_hint(payload_hint: str, *, task_text: str = "") -> str:
    text = normalize_message_text(payload_hint)
    if not text:
        return ""
    lowered = text.casefold()
    if any(marker in text for marker in _PAYLOAD_EXPLANATION_MARKERS):
        return ""

    if task_text and len(text) > max(len(normalize_message_text(task_text)) + 8, 48):
        return ""
    if lowered in {"null", "none", "undefined", "n/a"}:
        return ""
    return text


def _schema_accepts_target(schema: Any) -> bool:
    target_requirement = normalize_message_text(
        str(getattr(schema, "target_requirement", "") or "")
    )
    if target_requirement in {"required", "optional"}:
        return True
    if bool(getattr(schema, "allow_at", False)):
        return True
    requires = dict(getattr(schema, "requires", {}) or {})
    if bool(requires.get("at")) or bool(requires.get("image")):
        return True
    for slot in getattr(schema, "slots", []) or []:
        if str(getattr(slot, "type", "") or "") == "at":
            return True
    return False


def _normalize_target_hint_for_schema(schema: Any, target_hint: str) -> str:
    text = normalize_message_text(target_hint)
    if not text:
        return ""
    lowered = text.casefold()
    no_target_values = {
        "null",
        "none",
        "undefined",
        "n/a",
        "无",
        "无目标",
        "无特定目标",
        "无需目标",
        "无需指定目标",
        "不需要目标",
        "没有目标",
    }
    if lowered in no_target_values:
        return ""
    if any(marker in text for marker in ("无特定目标", "无需指定目标", "不需要目标")):
        return ""
    if not _schema_accepts_target(schema):
        return ""
    return text


def _merge_ambient_context_tokens(task_text: str, ambient_text: str) -> str:
    """Expose media/@ context to validators without leaking other task text."""

    text = normalize_message_text(task_text)
    placeholders = [
        token
        for token in collect_placeholders(ambient_text)
        if token not in collect_placeholders(text)
    ]
    if not placeholders:
        return text
    return normalize_message_text(f"{text} {' '.join(placeholders)}")


def _merge_task_frame_hints(task_frame: TaskFrame) -> str:
    """Expose explicit LLM hints to validators without reusing full ambient tails."""

    parts = [
        task_frame.effective_text,
        task_frame.target_hint,
        _safe_text_payload_hint(
            task_frame.payload_hint,
            task_text=task_frame.effective_text,
        ),
    ]
    return normalize_message_text(" ".join(part for part in parts if part))


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

    normalized = normalize_message_text(text)
    choices = [
        normalize_message_text(str(choice or ""))
        for choice in getattr(slot, "choices", []) or []
        if normalize_message_text(str(choice or ""))
    ]
    if choices and normalized not in choices:
        return None
    return normalized


def _shortcut_command_for_task(
    candidate: CommandCandidate,
    *,
    task_text: str,
    ambient_message: str,
) -> str:
    schema = candidate.schema
    shortcuts = getattr(schema, "shortcut_renders", None)
    if not shortcuts:
        meta = (
            getattr(candidate.tool, "meta", None)
            if candidate.tool is not None
            else None
        )
        shortcuts = meta.get("shortcut_renders") if isinstance(meta, dict) else None
    if not isinstance(shortcuts, list):
        return ""
    haystack = normalize_message_text(" ".join([task_text, ambient_message]))
    if not haystack:
        return ""
    best_alias = ""
    best_render = ""
    for item in shortcuts:
        if not isinstance(item, dict):
            continue
        alias = normalize_message_text(str(item.get("alias") or ""))
        render = normalize_message_text(str(item.get("render") or ""))
        if not alias or not render:
            continue
        if alias and alias in haystack and len(alias) > len(best_alias):
            best_alias = alias
            best_render = render
    return best_render


def _slots_from_shortcut_command(
    schema: Any,
    shortcut_command: str,
    existing_slots: dict[str, str],
) -> dict[str, str]:
    parts = normalize_message_text(shortcut_command).split()
    if len(parts) <= 1:
        return existing_slots
    slots = dict(existing_slots)
    values = parts[1:]
    slot_specs = list(getattr(schema, "slots", []) or [])
    for slot, value in zip(slot_specs, values, strict=False):
        coerced = _coerce_slot_value(slot, value)
        if coerced is not None:
            slots[slot.name] = coerced
    return slots


def _failure_tool_result(
    *,
    binding: NativeCommandToolBinding,
    task_text: str,
    ambient_message: str,
    error: str,
    display_content: str,
    retryable: bool,
    missing: list[str] | tuple[str, ...] | None = None,
) -> ToolResult:
    return ToolResult(
        output=build_command_observation(
            ok=False,
            command_id=binding.command_id,
            rendered_command=binding.candidate.schema.head,
            matched_plugin=binding.candidate.plugin_name,
            task_text=task_text,
            ambient_message=ambient_message,
            error=error,
            missing=missing,
            retryable=retryable,
            plugin_module=binding.candidate.plugin_module,
        ),
        display_content=display_content,
    )


def _execution_failure_from_exception(
    *,
    binding: NativeCommandToolBinding,
    validated: NativeValidatedRoute,
    ambient_message: str,
    exc: Exception,
) -> NativeToolExecutionResult:
    task_text = (
        validated.task_frame.effective_text if validated.task_frame is not None else ""
    )
    route_result = validated.route_result
    rendered = (
        route_result.decision.command
        if route_result is not None
        else binding.candidate.schema.head
    )
    return NativeToolExecutionResult(
        success=False,
        route_result=route_result,
        route_command=rendered,
        output=build_command_observation(
            ok=False,
            command_id=binding.command_id,
            rendered_command=rendered,
            matched_plugin=binding.candidate.plugin_name,
            task_text=task_text,
            ambient_message=ambient_message,
            error=f"插件执行链路异常：{type(exc).__name__}: {exc}",
            slots=route_result.slots if route_result is not None else {},
            retryable=False,
            plugin_module=binding.candidate.plugin_module,
        ),
        display_text="插件执行链路异常。",
        reason="route_executor_exception",
    )


def _ensure_execution_observation(
    *,
    binding: NativeCommandToolBinding,
    execution: NativeToolExecutionResult,
    validated: NativeValidatedRoute,
    ambient_message: str,
) -> NativeToolExecutionResult:
    if _is_standard_observation(execution.output):
        return execution
    route_result = execution.route_result
    task_text = (
        validated.task_frame.effective_text if validated.task_frame is not None else ""
    )
    rendered = (
        execution.route_command
        or (route_result.decision.command if route_result is not None else "")
        or binding.candidate.schema.head
    )
    plugin_name = (
        route_result.decision.plugin_name
        if route_result is not None
        else binding.candidate.plugin_name
    )
    plugin_module = (
        route_result.decision.plugin_module
        if route_result is not None
        else binding.candidate.plugin_module
    )
    error = ""
    if not execution.success:
        output = execution.output if isinstance(execution.output, dict) else {}
        error = normalize_message_text(
            str(output.get("error", ""))
            or execution.display_text
            or execution.reason
            or "插件执行失败。"
        )
    return replace(
        execution,
        output=build_command_observation(
            ok=execution.success,
            command_id=binding.command_id,
            rendered_command=rendered,
            matched_plugin=plugin_name,
            task_text=task_text,
            ambient_message=ambient_message,
            error=error,
            slots=route_result.slots if route_result is not None else {},
            retryable=not execution.success,
            plugin_module=plugin_module,
        ),
    )


def _is_standard_observation(output: dict[str, Any]) -> bool:
    required_keys = {
        "ok",
        "command_id",
        "rendered_command",
        "matched_plugin",
        "task_text",
        "messages_sent",
        "error",
        "retryable",
        "need_continue",
    }
    return isinstance(output, dict) and required_keys.issubset(output.keys())


__all__ = [
    "ExecuteNativeRoute",
    "NativeCommandExecutionContext",
    "NativeToolExecutionResult",
    "NativeValidatedRoute",
    "normalize_native_tool_slots",
]
