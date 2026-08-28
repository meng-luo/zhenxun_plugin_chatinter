"""Execution bridge for native ChatInter plugin command tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
import shlex
from typing import Any, Literal

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
from .person_candidates import TurnPersonCandidateLedger
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
from .task_frame import TaskFrame, pop_task_context

_NATIVE_EXECUTION_STAGE = "main_request"


@dataclass(frozen=True)
class NativeValidatedRoute:
    decision: NativeRouteDecision
    route_result: NativeRouteResult | None
    reason: str
    task_frame: TaskFrame | None = None
    candidate: CommandCandidate | None = None


NativeValidationReason = Literal[
    "untrusted_target",
    "unknown_target_ref",
    "command_identity_mismatch",
    "route_validation_failed",
]


@dataclass(frozen=True)
class NativeValidationFailure:
    reason: NativeValidationReason
    task_text: str
    target_ref: str = ""


@dataclass(frozen=True)
class NativeToolExecutionResult:
    success: bool
    route_result: NativeRouteResult | None
    route_command: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    display_text: str = ""
    reason: str = ""
    execution_started: bool = False


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
    event_target_hint: str = ""
    event_target_ids: tuple[str, ...] = ()
    target_refs: dict[str, str] = field(default_factory=dict)
    person_candidate_ledger: TurnPersonCandidateLedger | None = None
    retrieval_context: dict[str, bool | int | str] = field(default_factory=dict)
    executions: list[NativeToolExecutionResult] = field(default_factory=list)
    task_count: int = 0
    execution_receipts: dict[str, NativeToolExecutionResult] = field(
        default_factory=dict
    )

    async def execute_tool(
        self,
        *,
        binding: NativeCommandToolBinding,
        raw_slots: dict[str, Any],
    ) -> ToolResult:
        self.report.note_tool_choice()
        validation = self._validate_tool_call(binding=binding, raw_slots=raw_slots)
        if isinstance(validation, NativeValidationFailure):
            if self.person_candidate_ledger is not None:
                self.person_candidate_ledger.note_validation(validation.reason)
            return _validation_failure_tool_result(
                binding=binding,
                failure=validation,
                ambient_message=self.message_text,
            )
        validated = validation

        execution_key = _native_execution_key(binding=binding, validated=validated)
        previous_execution = self.execution_receipts.get(execution_key)
        if previous_execution is not None and (
            bool(previous_execution.output.get("execution_uncertain"))
            or _has_nonrepeatable_side_effect(binding)
        ):
            return _duplicate_execution_tool_result(
                binding=binding,
                validated=validated,
                ambient_message=self.message_text,
                previous_execution=previous_execution,
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
        if (
            execution.execution_started
            or execution.success
            or bool(execution.output.get("execution_uncertain"))
        ):
            self.execution_receipts[execution_key] = execution
        self.executions.append(execution)
        self._finalize_report(validated=validated, execution=execution)
        return ToolResult(
            output=execution.output,
            display_content=execution.display_text or execution.reason,
            is_error=not execution.success,
            is_retryable=bool(execution.output.get("retryable", False)),
        )

    def _validate_tool_call(
        self,
        *,
        binding: NativeCommandToolBinding,
        raw_slots: dict[str, Any],
    ) -> NativeValidatedRoute | NativeValidationFailure:
        candidate = binding.candidate
        (
            task_text,
            target_hint,
            target_refs,
            payload_hint,
            plugin_raw_slots,
        ) = pop_task_context(raw_slots)
        primary_target_ref = target_refs[0] if target_refs else ""
        if _has_untrusted_task_target(task_text, self.message_text):
            return NativeValidationFailure(
                "untrusted_target", task_text, primary_target_ref
            )
        command_id = normalize_message_text(binding.command_id)
        if not command_id or not any(
            normalize_message_text(item.schema.command_id) == command_id
            for item in self.candidates
        ):
            return NativeValidationFailure(
                "command_identity_mismatch",
                task_text,
                primary_target_ref,
            )
        trusted_target_ids: tuple[str, ...] = ()
        if self.event_target_hint:
            target_hint = self.event_target_hint
            hinted_ids = set(_target_hint_ids(target_hint))
            trusted_target_ids = tuple(
                user_id
                for user_id in dict.fromkeys(self.event_target_ids)
                if user_id in hinted_ids
            )
        elif target_refs and _schema_accepts_target(candidate.schema):
            resolved_target_ids = (
                self.person_candidate_ledger.validate_many(target_refs)
                if self.person_candidate_ledger is not None
                else _resolve_target_refs(target_refs, self.target_refs)
            )
            if not resolved_target_ids:
                return NativeValidationFailure(
                    "unknown_target_ref",
                    task_text,
                    _first_unknown_target_ref(target_refs, self.target_refs),
                )
            trusted_target_ids = tuple(dict.fromkeys(resolved_target_ids))
            target_hint = " ".join(
                f"[@{target_user_id}]" for target_user_id in trusted_target_ids
            )
        target_hint = _normalize_target_hint_for_schema(
            candidate.schema,
            target_hint,
        )
        if not target_hint:
            trusted_target_ids = ()
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
            target_refs=target_refs,
            trusted_target_ids=trusted_target_ids,
        )
        route_message_text = _merge_ambient_context_tokens(
            _merge_task_frame_hints(task_frame),
            self.message_text,
        )
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
            if self.person_candidate_ledger is not None and target_refs:
                self.person_candidate_ledger.note_validation(
                    "route_validation_failed"
                )
            return NativeValidationFailure(
                "route_validation_failed",
                task_text,
                primary_target_ref,
            )
        decision, route_result = route
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
        if slot is None or slot.type == "at":
            continue
        coerced = _coerce_slot_value(slot, value)
        if coerced is None:
            continue
        normalized_slots[slot.name] = coerced
    return normalized_slots


def _native_execution_key(
    *,
    binding: NativeCommandToolBinding,
    validated: NativeValidatedRoute,
) -> str:
    route_result = validated.route_result
    module = normalize_message_text(
        (
            route_result.decision.plugin_module
            if route_result is not None
            else ""
        )
        or binding.candidate.plugin_module
        or (
            route_result.decision.plugin_name
            if route_result is not None
            else ""
        )
        or binding.candidate.plugin_name
    ).casefold()
    command = normalize_message_text(
        validated.decision.command
        or (
            route_result.decision.command
            if route_result is not None
            else ""
        )
    )
    task_frame = validated.task_frame
    task_identity = "\0".join(
        normalize_message_text(value)
        for value in (
            task_frame.effective_text if task_frame is not None else "",
            task_frame.target_hint if task_frame is not None else "",
        )
    )
    return f"{module}\0{command}\0{task_identity}"


def _has_nonrepeatable_side_effect(binding: NativeCommandToolBinding) -> bool:
    snapshot = binding.candidate.tool
    side_effect = normalize_message_text(
        str(getattr(snapshot, "side_effect", "") or "")
    ).casefold()
    return side_effect in {"send", "mutate"}


def _duplicate_execution_tool_result(
    *,
    binding: NativeCommandToolBinding,
    validated: NativeValidatedRoute,
    ambient_message: str,
    previous_execution: NativeToolExecutionResult,
) -> ToolResult:
    route_result = validated.route_result
    task_text = (
        validated.task_frame.effective_text if validated.task_frame is not None else ""
    )
    rendered = validated.decision.command or binding.candidate.schema.head
    payload = build_command_observation(
        ok=False,
        command_id=binding.command_id,
        rendered_command=rendered,
        matched_plugin=binding.candidate.plugin_name,
        task_text=task_text,
        ambient_message=ambient_message,
        error="本轮已提交相同的副作用操作，已阻止重复执行。",
        slots=route_result.slots if route_result is not None else {},
        retryable=False,
        plugin_module=binding.candidate.plugin_module,
    )
    payload["status"] = "blocked"
    if bool(previous_execution.output.get("execution_uncertain")):
        payload["prior_execution_uncertain"] = True
    payload["duplicate_blocked"] = True
    return ToolResult(
        output=payload,
        display_content="本轮相同操作已提交，已阻止重复执行。",
        is_error=True,
        is_retryable=False,
    )


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


def _target_hint_ids(target_hint: str) -> tuple[str, ...]:
    ids: list[str] = []
    for token in collect_placeholders(target_hint):
        if token.startswith("[@") and token.endswith("]"):
            user_id = token[2:-1].strip()
            if user_id:
                ids.append(user_id)
    return tuple(dict.fromkeys(ids))


def _resolve_target_refs(
    target_refs: tuple[str, ...],
    available_refs: dict[str, str],
) -> tuple[str, ...] | None:
    resolved = tuple(
        available_refs.get(normalize_message_text(target_ref).casefold(), "")
        for target_ref in target_refs
    )
    return None if any(not user_id for user_id in resolved) else resolved


def _first_unknown_target_ref(
    target_refs: tuple[str, ...],
    available_refs: dict[str, str],
) -> str:
    return next(
        (
            target_ref
            for target_ref in target_refs
            if not available_refs.get(normalize_message_text(target_ref).casefold())
        ),
        target_refs[0] if target_refs else "",
    )


def _has_untrusted_task_target(task_text: str, ambient_text: str) -> bool:
    ambient_targets = {
        token
        for token in collect_placeholders(ambient_text)
        if token.startswith("[@")
    }
    return any(
        token.startswith("[@") and token not in ambient_targets
        for token in collect_placeholders(task_text)
    )


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


def _validation_failure_tool_result(
    *,
    binding: NativeCommandToolBinding,
    failure: NativeValidationFailure,
    ambient_message: str,
) -> ToolResult:
    output = build_command_observation(
        ok=False,
        command_id=binding.command_id,
        rendered_command=binding.candidate.schema.head,
        matched_plugin=binding.candidate.plugin_name,
        task_text=failure.task_text,
        ambient_message=ambient_message,
        error="工具参数未通过本轮执行边界校验。",
        retryable=False,
        plugin_module=binding.candidate.plugin_module,
    )
    output.update(
        status="invalid_tool_arguments",
        validation_reason=failure.reason,
    )
    if failure.target_ref:
        output["target_ref"] = failure.target_ref[:64]
    return ToolResult(
        output=output,
        display_content="",
        is_error=True,
        is_retryable=False,
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
    "NativeValidationFailure",
    "NativeValidationReason",
    "normalize_native_tool_slots",
]
