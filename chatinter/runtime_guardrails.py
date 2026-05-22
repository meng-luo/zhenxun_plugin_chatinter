"""Runtime guardrails for the ChatInter agent loop.

These checks stop empty tool loops with concrete observations instead of hoping
the prompt will convince the model to recover by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Literal

from zhenxun.services.llm.types.models import LLMToolCall, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .command_catalog_tool import COMMAND_CATALOG_TOOL_NAME
from .feedback import FeedbackStore
from .route_text import normalize_message_text
from .task_frame import TASK_TEXT_FIELD

GuardrailSeverity = Literal["light", "medium", "heavy"]
GuardrailAction = Literal["observe", "block_tool", "stop"]

_CATALOG_NO_RESULT_LIMIT = 2
_CATALOG_NO_RESULT_STOP_LIMIT = 3
_CATALOG_RETRIEVE_STOP_LIMIT = 2
_REPEAT_FAILURE_BLOCK_LIMIT = 2
_REPEAT_FAILURE_STOP_LIMIT = 3
_NO_VISIBLE_OUTPUT_OBSERVE_LIMIT = 2
_NO_VISIBLE_OUTPUT_STOP_LIMIT = 3
_NO_OUTPUT_TASK_BLOCK_LIMIT = 2
_NO_OUTPUT_TASK_STOP_LIMIT = 3
_EXCESSIVE_CATALOG_TOOLS = 72
_EXCESSIVE_CATALOG_CALLS = 2
_REPEAT_SUCCESS_STOP_LIMIT = 2


@dataclass(frozen=True)
class RuntimeGuardrailDecision:
    reason: str
    message: str
    severity: GuardrailSeverity = "light"
    action: GuardrailAction = "observe"
    tool_name: str = ""
    command_id: str = ""
    task_text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        return self.action == "stop" or self.severity == "heavy"

    @property
    def should_block_tool(self) -> bool:
        return self.action == "block_tool" and bool(self.tool_name)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "status": "runtime_guardrail",
            "guardrail_reason": self.reason,
            "severity": self.severity,
            "action": self.action,
            "message": self.message,
            "tool_name": self.tool_name,
            "command_id": self.command_id,
            "task_text": self.task_text,
            "messages_sent": [],
            "artifacts": [],
            "need_continue": not self.should_stop,
            "remaining_task_hint": self.message if not self.should_stop else "",
            "error": self.message,
            "retryable": not self.should_stop,
        }
        payload.update(self.payload)
        return {
            key: value
            for key, value in payload.items()
            if value not in ("", [], {}, None)
        }


class RuntimeGuardrails:
    """Bounded stateful loop protection for one agent run."""

    def __init__(
        self,
        *,
        session_id: str | None,
        max_elapsed_seconds: float,
    ) -> None:
        self.session_id = normalize_message_text(session_id or "")
        self.max_elapsed_seconds = max(float(max_elapsed_seconds or 0), 15.0)
        self.started_at = time.monotonic()
        self.blocked_tool_names: set[str] = set()
        self.catalog_no_result_streak = 0
        self.catalog_retrieve_count = 0
        self.catalog_active_tools = 0
        self.command_execution_count = 0
        self.failed_call_counts: dict[str, int] = {}
        self.successful_call_signatures: set[str] = set()
        self.repeated_success_counts: dict[str, int] = {}
        self.validation_failed_signatures: set[str] = set()
        self.no_visible_output_streak = 0
        self.no_output_task_commands: dict[str, set[str]] = {}
        self.blocked_call_counts: dict[str, int] = {}

    def filter_tool_map(
        self,
        tool_map: dict[str, ToolExecutable],
    ) -> dict[str, ToolExecutable]:
        if not self.blocked_tool_names:
            return dict(tool_map)
        return {
            name: tool
            for name, tool in tool_map.items()
            if name not in self.blocked_tool_names
        }

    def block_tool(self, tool_name: str) -> None:
        normalized = normalize_message_text(tool_name)
        if normalized:
            self.blocked_tool_names.add(normalized)

    def before_model_request(self, state: Any) -> RuntimeGuardrailDecision | None:
        _ = state
        elapsed = time.monotonic() - self.started_at
        if elapsed > self.max_elapsed_seconds:
            return self._decision(
                reason="time_budget_exhausted",
                message="本轮处理已经超过时间预算，请停止工具调用并基于现有结果回复。",
                severity="heavy",
                action="stop",
                payload={"elapsed_seconds": round(elapsed, 2)},
            )

        if (
            self.catalog_active_tools >= _EXCESSIVE_CATALOG_TOOLS
            and self.catalog_retrieve_count >= _EXCESSIVE_CATALOG_CALLS
            and self.command_execution_count <= 0
        ):
            return self._decision(
                reason="catalog_over_injected_without_execution",
                message=(
                    "已注入大量命令 schema 但还没有执行任何插件。请选择已有工具，"
                    "或直接说明没有合适插件，不要继续扩大检索。"
                ),
                severity="medium",
                action="block_tool",
                tool_name=COMMAND_CATALOG_TOOL_NAME,
                payload={
                    "active_command_tools": self.catalog_active_tools,
                    "catalog_retrieve_count": self.catalog_retrieve_count,
                },
            )
        return None

    def before_tool_call(
        self,
        *,
        tool_call: LLMToolCall,
        tool_map: dict[str, ToolExecutable],
    ) -> RuntimeGuardrailDecision | None:
        tool_name = _tool_name(tool_call)
        if tool_name in self.blocked_tool_names:
            count = self.blocked_call_counts.get(tool_name, 0) + 1
            self.blocked_call_counts[tool_name] = count
            return self._decision(
                reason="blocked_tool_recalled",
                message="这个工具刚刚已被运行时禁用，请换一个已注入工具或直接回复。",
                severity="heavy" if count >= 2 else "medium",
                action="stop" if count >= 2 else "block_tool",
                tool_name=tool_name,
            )

        executable = tool_map.get(tool_name)
        if executable is None:
            return self._decision(
                reason="unknown_tool_called",
                message="模型调用了当前工具列表里不存在的工具，请重新选择可用工具。",
                severity="medium",
                action="block_tool",
                tool_name=tool_name,
            )

        signature = _tool_call_signature(tool_call, executable)
        if signature in self.successful_call_signatures:
            duplicate_count = self.repeated_success_counts.get(signature, 0) + 1
            self.repeated_success_counts[signature] = duplicate_count
            command_id = _command_id_from_executable(executable) or tool_name
            task_text = _task_text_from_tool_call(tool_call)
            severity: GuardrailSeverity = (
                "heavy" if duplicate_count >= _REPEAT_SUCCESS_STOP_LIMIT else "medium"
            )
            return self._decision(
                reason="repeated_successful_tool_call",
                message=(
                    "这个子任务对应的工具和参数已经成功执行过。不要重复调用；"
                    "请继续处理剩余子任务，或在全部完成后给最终回复。"
                ),
                severity=severity,
                action="stop" if severity == "heavy" else "observe",
                tool_name=tool_name,
                command_id=command_id,
                task_text=task_text,
                payload={"duplicate_count": duplicate_count},
            )
        failed_count = self.failed_call_counts.get(signature, 0)
        if failed_count <= 0:
            return None

        command_id = _command_id_from_executable(executable) or tool_name
        task_text = _task_text_from_tool_call(tool_call)
        if signature in self.validation_failed_signatures:
            severity: GuardrailSeverity = "heavy" if failed_count >= 2 else "medium"
            return self._decision(
                reason="repeated_validation_failed_call",
                message=(
                    "同一个工具参数刚刚没有通过本地校验。请修改参数、换工具，"
                    "或直接说明缺少信息。不要原样再次调用。"
                ),
                severity=severity,
                action="stop" if severity == "heavy" else "block_tool",
                tool_name=tool_name,
                command_id=command_id,
                task_text=task_text,
                payload={"failed_count": failed_count},
            )

        if failed_count >= _REPEAT_FAILURE_BLOCK_LIMIT:
            severity = (
                "heavy"
                if failed_count >= _REPEAT_FAILURE_STOP_LIMIT
                else "medium"
            )
            return self._decision(
                reason="repeated_failed_tool_call",
                message="同一命令同一参数已经连续失败，请换工具、换参数或停止工具调用。",
                severity=severity,
                action="stop" if severity == "heavy" else "block_tool",
                tool_name=tool_name,
                command_id=command_id,
                task_text=task_text,
                payload={"failed_count": failed_count},
            )
        return None

    def after_tool_result(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
        tool_map: dict[str, ToolExecutable],
    ) -> RuntimeGuardrailDecision | None:
        output = tool_result.output if isinstance(tool_result.output, dict) else {}
        tool_name = _tool_name(tool_call)
        executable = tool_map.get(tool_name)

        if _is_catalog_result(output):
            return self._after_catalog_result(output)

        command_id = normalize_message_text(
            str(output.get("command_id") or _command_id_from_executable(executable))
        )
        if command_id:
            self.command_execution_count += 1

        ok = bool(output.get("ok"))
        task_text = normalize_message_text(
            str(output.get("task_text") or _task_text_from_tool_call(tool_call))
        )
        has_output = _has_visible_output(output)
        signature = _tool_call_signature(tool_call, executable)
        no_visible_output_decision = self._record_visible_output_state(
            tool_name=tool_name,
            command_id=command_id,
            task_text=task_text,
            has_output=has_output,
        )

        if not ok:
            count = self.failed_call_counts.get(signature, 0) + 1
            self.failed_call_counts[signature] = count
            error = normalize_message_text(str(output.get("error", "")))
            if _looks_like_validation_failure(error):
                self.validation_failed_signatures.add(signature)
            if count >= _REPEAT_FAILURE_BLOCK_LIMIT:
                severity: GuardrailSeverity = (
                    "heavy" if count >= _REPEAT_FAILURE_STOP_LIMIT else "medium"
                )
                return self._decision(
                    reason="repeated_failed_tool_call",
                    message=(
                        "同一命令同一参数连续失败。请不要继续原样调用，"
                        "改参数、换工具或直接说明失败原因。"
                    ),
                    severity=severity,
                    action="stop" if severity == "heavy" else "block_tool",
                    tool_name=tool_name,
                    command_id=command_id,
                    task_text=task_text,
                    payload={"failed_count": count, "last_error": error},
                )
            return no_visible_output_decision

        if command_id:
            self.successful_call_signatures.add(signature)

        if command_id and task_text and not has_output:
            bucket = self.no_output_task_commands.setdefault(task_text, set())
            bucket.add(command_id)
            if len(bucket) >= _NO_OUTPUT_TASK_BLOCK_LIMIT:
                severity = (
                    "heavy"
                    if len(bucket) >= _NO_OUTPUT_TASK_STOP_LIMIT
                    else "medium"
                )
                return self._decision(
                    reason="same_task_no_output_loop",
                    message=(
                        "同一个子任务已经被多个工具处理但没有可见输出。"
                        "请换成最终回复，或解释插件没有产生结果。"
                    ),
                    severity=severity,
                    action="stop" if severity == "heavy" else "block_tool",
                    tool_name=tool_name,
                    command_id=command_id,
                    task_text=task_text,
                    payload={"tried_commands": sorted(bucket)},
                )
        return no_visible_output_decision

    def on_budget_exhausted(self, *, call_count: int) -> RuntimeGuardrailDecision:
        return self._decision(
            reason="tool_budget_exhausted",
            message="本轮工具调用预算已耗尽，请停止调用工具并基于现有结果回复。",
            severity="heavy",
            action="stop",
            payload={"requested_tool_calls": max(int(call_count or 0), 0)},
        )

    def on_max_steps(self) -> RuntimeGuardrailDecision:
        return self._decision(
            reason="max_agent_steps_reached",
            message="Agent 已达到最大步骤数，请停止工具调用并给出简短最终回复。",
            severity="heavy",
            action="stop",
        )

    def tool_result_for_decision(
        self,
        *,
        tool_call: LLMToolCall,
        decision: RuntimeGuardrailDecision,
    ) -> ToolResult:
        output = decision.to_payload()
        if "task_text" not in output:
            task_text = _task_text_from_tool_call(tool_call)
            if task_text:
                output["task_text"] = task_text
        return ToolResult(output=output, display_content=decision.message)

    def _after_catalog_result(
        self,
        output: dict[str, Any],
    ) -> RuntimeGuardrailDecision | None:
        retrieved = _safe_int(output.get("retrieved"))
        active_tools = _safe_int(output.get("active_command_tools"))
        injected_tools = _safe_int(output.get("injected_command_tools"))
        self.catalog_retrieve_count += 1
        self.catalog_active_tools = max(
            self.catalog_active_tools,
            active_tools,
            injected_tools,
        )
        if retrieved <= 0:
            self.catalog_no_result_streak += 1
            if self.catalog_no_result_streak >= _CATALOG_NO_RESULT_LIMIT:
                severity: GuardrailSeverity = (
                    "heavy"
                    if self.catalog_no_result_streak >= _CATALOG_NO_RESULT_STOP_LIMIT
                    else "medium"
                )
                return self._decision(
                    reason="catalog_no_result_loop",
                    message=(
                        "连续检索都没有找到命令。请换查询词一次，或直接回复没有"
                        "合适插件；不要继续重复检索。"
                    ),
                    severity=severity,
                    action="stop" if severity == "heavy" else "block_tool",
                    tool_name=COMMAND_CATALOG_TOOL_NAME,
                    payload={"no_result_streak": self.catalog_no_result_streak},
                )
            return self._decision(
                reason="catalog_no_result",
                message="本次命令检索没有结果。请换查询词、换表达，或直接聊天回复。",
                severity="light",
                action="observe",
                tool_name=COMMAND_CATALOG_TOOL_NAME,
            )

        self.catalog_no_result_streak = 0
        if (
            self.catalog_active_tools >= _EXCESSIVE_CATALOG_TOOLS
            and self.catalog_retrieve_count >= _EXCESSIVE_CATALOG_CALLS
            and self.command_execution_count <= 0
        ):
            return self._decision(
                reason="catalog_over_injected_without_execution",
                message=(
                    "已经注入很多命令 schema，但没有执行任何插件。请选择现有工具，"
                    "或直接说明没有匹配插件。"
                ),
                severity="medium",
                action="block_tool",
                tool_name=COMMAND_CATALOG_TOOL_NAME,
                payload={"active_command_tools": self.catalog_active_tools},
            )
        if self.catalog_retrieve_count >= _CATALOG_RETRIEVE_STOP_LIMIT:
            return self._decision(
                reason="catalog_retrieve_budget_exhausted",
                message=(
                    "本轮已经检索过插件能力。请从已注入的命令工具中选择，"
                    "或直接说明没有合适插件，不要继续检索。"
                ),
                severity="medium",
                action="block_tool",
                tool_name=COMMAND_CATALOG_TOOL_NAME,
                payload={"catalog_retrieve_count": self.catalog_retrieve_count},
            )
        return None

    def _record_visible_output_state(
        self,
        *,
        tool_name: str,
        command_id: str,
        task_text: str,
        has_output: bool,
    ) -> RuntimeGuardrailDecision | None:
        if not command_id:
            return None
        if has_output:
            self.no_visible_output_streak = 0
            return None
        self.no_visible_output_streak += 1
        if self.no_visible_output_streak < _NO_VISIBLE_OUTPUT_OBSERVE_LIMIT:
            return None
        severity: GuardrailSeverity = (
            "heavy"
            if self.no_visible_output_streak >= _NO_VISIBLE_OUTPUT_STOP_LIMIT
            else "medium"
        )
        return self._decision(
            reason="no_visible_output_loop",
            message=(
                "连续多个工具调用没有产生可见发送输出。请停止继续试探工具，"
                "基于已知观察结果给用户最终回复，或说明插件没有产生结果。"
            ),
            severity=severity,
            action="stop" if severity == "heavy" else "observe",
            tool_name=tool_name,
            command_id=command_id,
            task_text=task_text,
            payload={"no_visible_output_streak": self.no_visible_output_streak},
        )

    def _decision(
        self,
        *,
        reason: str,
        message: str,
        severity: GuardrailSeverity,
        action: GuardrailAction,
        tool_name: str = "",
        command_id: str = "",
        task_text: str = "",
        payload: dict[str, Any] | None = None,
    ) -> RuntimeGuardrailDecision:
        decision = RuntimeGuardrailDecision(
            reason=normalize_message_text(reason),
            message=normalize_message_text(message),
            severity=severity,
            action=action,
            tool_name=normalize_message_text(tool_name),
            command_id=normalize_message_text(command_id),
            task_text=normalize_message_text(task_text),
            payload=dict(payload or {}),
        )
        FeedbackStore.record_runtime_guardrail(
            session_id=self.session_id,
            reason=decision.reason,
            severity=decision.severity,
            message_text=decision.message,
            command_id=decision.command_id,
            tool_name=decision.tool_name,
            weight=_guardrail_weight(decision.severity),
        )
        return decision


def _tool_name(tool_call: LLMToolCall) -> str:
    return normalize_message_text(str(getattr(tool_call.function, "name", "") or ""))


def _tool_call_signature(
    tool_call: LLMToolCall,
    executable: ToolExecutable | None,
) -> str:
    command_id = _command_id_from_executable(executable) or _tool_name(tool_call)
    arguments = _parse_arguments(str(getattr(tool_call.function, "arguments", "") or ""))
    return "|".join([command_id, _normalized_argument_fingerprint(arguments)])


def _parse_arguments(arguments: str) -> dict[str, Any]:
    text = str(arguments or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return {"raw": normalize_message_text(text)}
    return value if isinstance(value, dict) else {"value": value}


def _normalized_argument_fingerprint(arguments: dict[str, Any]) -> str:
    normalized: dict[str, Any] = {}
    for key, value in arguments.items():
        name = normalize_message_text(str(key or ""))
        if not name or name in {"target_hint", "payload_hint"}:
            continue
        if isinstance(value, str):
            normalized[name] = normalize_message_text(value)
        else:
            normalized[name] = value
    try:
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(sorted((key, str(value)) for key, value in normalized.items()))


def _command_id_from_executable(executable: ToolExecutable | None) -> str:
    binding = getattr(executable, "binding", None)
    return normalize_message_text(str(getattr(binding, "command_id", "") or ""))


def _task_text_from_tool_call(tool_call: LLMToolCall) -> str:
    arguments = _parse_arguments(str(getattr(tool_call.function, "arguments", "") or ""))
    return normalize_message_text(str(arguments.get(TASK_TEXT_FIELD, "") or ""))


def _is_catalog_result(output: dict[str, Any]) -> bool:
    return output.get("status") in {
        "retrieved",
        "capability_candidates_retrieved",
    }


def _has_visible_output(output: dict[str, Any]) -> bool:
    if bool(output.get("visible_output")):
        return True
    summary = normalize_message_text(str(output.get("messages_sent_summary", "") or ""))
    if summary:
        return True
    messages = output.get("messages_sent")
    if isinstance(messages, list) and any(
        normalize_message_text(str(item or "")) for item in messages
    ):
        return True
    artifacts = output.get("artifacts")
    return isinstance(artifacts, list) and any(
        isinstance(item, dict) and item.get("artifact_id") for item in artifacts
    )


def _looks_like_validation_failure(error: str) -> bool:
    normalized = normalize_message_text(error).casefold()
    return any(
        token in normalized
        for token in (
            "校验",
            "validation",
            "invalid route",
            "invalid",
            "缺少执行上下文",
            "没有生成有效插件路由",
        )
    )


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _guardrail_weight(severity: GuardrailSeverity) -> float:
    if severity == "heavy":
        return -1.0
    if severity == "medium":
        return -0.45
    return -0.12


__all__ = [
    "RuntimeGuardrailDecision",
    "RuntimeGuardrails",
]
