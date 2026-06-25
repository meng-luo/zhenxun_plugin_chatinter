"""Superuser Agent tool retry guardrail."""

from __future__ import annotations

import json
from typing import Any

from zhenxun.services.llm.types.models import LLMToolCall, ToolResult

from ..route_text import normalize_message_text

SUPERUSER_AGENT_MODES = {"superuser_agent", "superuser_subagent"}
SUPERUSER_TOOL_FAILURE_HINT_LIMIT = 2
SUPERUSER_TOOL_FAILURE_STOP_LIMIT = 5


class SuperuserToolGuardrail:
    def __init__(self) -> None:
        self._failures: dict[tuple[str, str, str], int] = {}

    def after_result(
        self,
        *,
        agent_mode: str,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
    ) -> dict[str, Any] | None:
        if normalize_message_text(agent_mode) not in SUPERUSER_AGENT_MODES:
            return None
        failure = _tool_failure_label(tool_result)
        if not failure:
            self.clear(tool_call)
            return None

        tool_name = normalize_message_text(str(tool_call.function.name or ""))
        if not tool_name:
            return None
        signature = (
            tool_name,
            _normalized_tool_arguments(str(tool_call.function.arguments or "")),
            failure,
        )
        count = self._failures.get(signature, 0) + 1
        self._failures[signature] = count
        if count == SUPERUSER_TOOL_FAILURE_HINT_LIMIT:
            return _superuser_tool_guardrail_payload(
                tool_name=tool_name,
                failure=failure,
                count=count,
                stopped=False,
            )
        if count >= SUPERUSER_TOOL_FAILURE_STOP_LIMIT:
            return _superuser_tool_guardrail_payload(
                tool_name=tool_name,
                failure=failure,
                count=count,
                stopped=True,
            )
        return None

    def clear(self, tool_call: LLMToolCall) -> None:
        if not self._failures:
            return
        tool_name = normalize_message_text(str(tool_call.function.name or ""))
        normalized_args = _normalized_tool_arguments(
            str(tool_call.function.arguments or "")
        )
        for signature in list(self._failures):
            if signature[:2] == (tool_name, normalized_args):
                self._failures.pop(signature, None)


def _parse_tool_arguments(arguments: str) -> dict[str, Any] | str:
    text = str(arguments or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return text
    return value if isinstance(value, dict) else {"value": value}


def _normalized_tool_arguments(arguments: str) -> str:
    value = _parse_tool_arguments(arguments)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def _tool_failure_label(tool_result: ToolResult) -> str:
    output = tool_result.output if isinstance(tool_result.output, dict) else {}
    if not output:
        return ""
    ok_value = output.get("ok")
    if ok_value is True:
        return ""
    status = normalize_message_text(str(output.get("status") or ""))
    error = normalize_message_text(
        str(
            output.get("error")
            or output.get("message")
            or tool_result.display_content
            or ""
        )
    )
    if ok_value is False:
        return error or status or "failed"
    if error:
        return error
    if _looks_like_tool_failure_status(status):
        return status
    return ""


def _looks_like_tool_failure_status(status: str) -> bool:
    normalized = normalize_message_text(status).casefold()
    return any(
        token in normalized
        for token in (
            "失败",
            "异常",
            "错误",
            "拒绝",
            "超时",
            "不存在",
            "denied",
            "error",
            "exception",
            "fail",
            "invalid",
            "not_found",
            "timeout",
        )
    )


def _superuser_tool_guardrail_payload(
    *,
    tool_name: str,
    failure: str,
    count: int,
    stopped: bool,
) -> dict[str, Any]:
    if stopped:
        message = (
            f"超级用户工具 {tool_name} 使用相同参数连续失败 {count} 次，"
            "本轮已停止这个工具。请换工具、换参数，或基于现有结果回复。"
        )
    else:
        message = (
            f"超级用户工具 {tool_name} 使用相同参数已失败 {count} 次。"
            "不要原样重试；请修改参数、换工具，或说明失败原因。"
        )
    return {
        "ok": False,
        "status": "runtime_guardrail",
        "guardrail_reason": "superuser_tool_repeated_failure",
        "severity": "medium",
        "action": "block_tool" if stopped else "observe",
        "tool_name": tool_name,
        "message": message,
        "error": message,
        "last_error": failure,
        "failed_count": count,
        "messages_sent": [],
        "artifacts": [],
        "need_continue": True,
        "retryable": not stopped,
    }


__all__ = ["SUPERUSER_AGENT_MODES", "SuperuserToolGuardrail"]
