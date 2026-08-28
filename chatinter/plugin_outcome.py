"""Canonical outcome projection for mixed-chat plugin calls."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from .llm_compat import ToolResult
from .response_defaults import (
    PLUGIN_FAILURE_REPLY_TEXT,
    PLUGIN_SELECTION_REPLY_TEXT,
    PLUGIN_SUCCESS_REPLY_TEXT,
)
from .route_text import normalize_message_text

PluginOutcomeKind = Literal[
    "executed",
    "partial",
    "needs_input",
    "not_executed",
    "uncertain",
]


@dataclass(frozen=True, slots=True)
class PluginOutcome:
    kind: PluginOutcomeKind
    reason: str = ""

    @property
    def executed(self) -> bool:
        return self.kind == "executed"

    @property
    def executed_any(self) -> bool:
        return self.kind in {"executed", "partial"}


def classify_plugin_result(result: ToolResult) -> PluginOutcome:
    output = result.output if isinstance(result.output, dict) else {}
    status = normalize_message_text(str(output.get("status", "") or "")).casefold()
    if status == "uncertain" or bool(output.get("execution_uncertain")):
        return PluginOutcome("uncertain", reason=status or "execution_uncertain")

    missing = tuple(
        normalize_message_text(str(item or ""))
        for item in output.get("missing", ())
        if normalize_message_text(str(item or ""))
    )
    if status in {"ambiguous", "selection_required"}:
        return PluginOutcome(
            "needs_input",
            reason=status,
        )
    if missing:
        return PluginOutcome("not_executed", reason="missing_input")

    failure_stage = normalize_message_text(
        str(output.get("failure_stage", "") or "")
    ).casefold()
    if failure_stage == "native_reroute":
        return PluginOutcome("not_executed", reason="native_reroute")

    if output.get("plugin_execution") is False:
        return PluginOutcome("not_executed", reason=status or "not_executed")
    if result.is_error or output.get("ok") is False:
        return PluginOutcome("not_executed", reason=status or "failed")
    return PluginOutcome("executed", reason=status or "completed")


def aggregate_plugin_outcomes(
    outcomes: list[PluginOutcome] | tuple[PluginOutcome, ...],
) -> PluginOutcome:
    if not outcomes:
        return PluginOutcome("not_executed", reason="no_plugin_result")
    if any(item.kind == "uncertain" for item in outcomes):
        return PluginOutcome("uncertain", reason="execution_uncertain")
    executed_count = sum(item.kind == "executed" for item in outcomes)
    if executed_count == len(outcomes):
        return PluginOutcome("executed", reason="plugin_executed")
    if executed_count:
        return PluginOutcome("partial", reason="plugin_partial")
    for item in outcomes:
        if item.kind == "needs_input":
            return item
    return PluginOutcome("not_executed", reason=outcomes[-1].reason)


def plugin_results_have_visible_output(
    results: list[ToolResult] | tuple[ToolResult, ...],
) -> bool:
    return any(_result_has_visible_output(result) for result in results)


def plugin_results_own_delivery(
    results: list[ToolResult] | tuple[ToolResult, ...],
) -> bool:
    return any(_result_owns_delivery(result) for result in results)


def plugin_terminal_reply(
    outcome: PluginOutcome,
    results: list[ToolResult] | tuple[ToolResult, ...],
) -> str:
    if outcome.kind == "needs_input":
        return PLUGIN_SELECTION_REPLY_TEXT
    if outcome.kind == "uncertain":
        return "这个操作可能已经执行，但结果暂时无法确认。为避免重复，我没有再次执行。"
    if outcome.kind == "partial":
        return _latest_user_display(results) or "部分操作已完成，其余操作未能执行。"
    if outcome.reason == "missing_input":
        for result in reversed(results):
            out = result.output if isinstance(result.output, dict) else {}
            msg = normalize_message_text(str(out.get("error", "") or ""))
            if msg:
                return msg
        return PLUGIN_FAILURE_REPLY_TEXT

    display = _latest_user_display(results)
    if outcome.kind == "executed":
        return display or PLUGIN_SUCCESS_REPLY_TEXT

    statuses = {
        normalize_message_text(
            str(
                (result.output if isinstance(result.output, dict) else {}).get(
                    "status", ""
                )
                or ""
            )
        ).casefold()
        for result in results
    }
    if statuses & {
        "invalid",
        "invalid_arguments",
        "invalid_tool_arguments",
        "argument_error",
        "validation_failed",
    }:
        return PLUGIN_FAILURE_REPLY_TEXT
    if statuses & {
        "not_found",
        "tool_not_found",
        "no_result",
        "no_results",
        "empty",
        "web_search_empty",
    }:
        return "我没找到可执行的对应功能，请换种更具体的说法。"
    if statuses & {"unavailable", "unavailable_in_context", "disabled"}:
        return "这个功能当前不可用。"
    if statuses & {"blocked", "duplicate_blocked"}:
        return "这个操作没有执行，以免重复处理。"
    return display or PLUGIN_FAILURE_REPLY_TEXT


def plugin_failure_layer(
    outcome: str,
    execution_records: list[dict[str, object]],
) -> str:
    if not outcome or outcome == "executed":
        return ""
    if outcome == "uncertain":
        return "execution"
    reasons = {
        normalize_message_text(str(item.get("outcome_reason", "") or "")).casefold()
        for item in execution_records
    }
    if reasons & {"invalid_tool_arguments", "missing_input"}:
        return "argument_validation"
    if reasons & {
        "ambiguous",
        "invalid",
        "not_found",
        "selection_required",
        "tool_not_found",
        "unavailable_in_context",
    }:
        return "selection"
    if "native_reroute" in reasons:
        return "execution"
    return "execution"


def plugin_input_rejected(outcome: PluginOutcome) -> bool:
    return outcome.kind == "not_executed" and outcome.reason in {
        "argument_error",
        "invalid_arguments",
        "invalid_tool_arguments",
        "missing_input",
        "validation_failed",
    }


def _result_has_visible_output(result: ToolResult) -> bool:
    output = result.output if isinstance(result.output, dict) else {}
    return output.get("delivery_observed") is True


def _result_owns_delivery(result: ToolResult) -> bool:
    output = result.output if isinstance(result.output, dict) else {}
    delivery_state = normalize_message_text(
        str(output.get("delivery_state", "") or "")
    ).casefold()
    delivery_observed = output.get("delivery_observed") is True or delivery_state in {
        "complete",
        "completed",
        "delivered",
        "observed",
        "sent",
    }
    return bool(
        delivery_observed
        and (output.get("delivery_owner") or output.get("external_delivery"))
    )


def _latest_user_display(results: list[ToolResult] | tuple[ToolResult, ...]) -> str:
    for result in reversed(results):
        display = normalize_message_text(str(result.display_content or ""))
        if display and not _looks_internal_display(display):
            return display[:500]
    return ""


# Matches snake_case internal tokens: all-lowercase, at least one underscore,
# e.g. "plugin_completed_without_visible_output", "plugin_not_observed".
# Short plain words like "ok", "done", "error" are NOT matched (no underscore).
_INTERNAL_SNAKE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")


def _looks_internal_display(text: str) -> bool:
    lowered = text.strip().casefold()
    if lowered.startswith("ci_skill_"):
        return True
    if lowered in {
        "ambiguous",
        "failed",
        "missing",
        "not_executed",
    }:
        return True
    if _INTERNAL_SNAKE_TOKEN_RE.match(lowered):
        return True
    return False


__all__ = [
    "PluginOutcome",
    "PluginOutcomeKind",
    "aggregate_plugin_outcomes",
    "classify_plugin_result",
    "plugin_failure_layer",
    "plugin_input_rejected",
    "plugin_results_have_visible_output",
    "plugin_results_own_delivery",
    "plugin_terminal_reply",
]
