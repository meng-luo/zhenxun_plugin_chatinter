"""Tool execution helpers for ChatInter agent runtime."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ...llm_compat import (
    LLMToolCall,
    RunContext,
    ToolInvoker,
    ToolResult,
    validate_tool_call_arguments,
)
from ...route_text import normalize_message_text

_TASK_TEXT_FIELD = "task_text"
_SUPERUSER_TOOL_DEFAULT_DEADLINE_SECONDS = 125.0
_SUPERUSER_TOOL_MAX_DEADLINE_SECONDS = 1800.0
_SUPERUSER_TOOL_DEADLINE_GRACE_SECONDS = 5.0


async def execute_tool_call(
    invoker: ToolInvoker,
    tool_call: LLMToolCall,
    tool_map: dict[str, Any],
    run_context: RunContext,
) -> tuple[LLMToolCall, ToolResult]:
    coro = invoker.execute_tool_call(tool_call, tool_map, run_context)
    timeout = superuser_tool_deadline_seconds(tool_call, run_context)
    if timeout is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return tool_call, tool_deadline_result(
            tool_call,
            timeout,
        )


async def validate_superuser_tool_call(
    tool_call: LLMToolCall,
    tool_map: dict[str, Any],
) -> ToolResult | None:
    _tool, _arguments, error = await validate_tool_call_arguments(
        tool_call,
        tool_map,
    )
    return error


def superuser_tool_deadline_seconds(
    tool_call: LLMToolCall,
    run_context: RunContext,
) -> float:
    del run_context
    arguments = _parse_tool_arguments(str(tool_call.function.arguments or ""))
    if isinstance(arguments, dict) and "timeout_seconds" in arguments:
        try:
            return (
                min(
                    max(float(arguments["timeout_seconds"]), 1.0),
                    _SUPERUSER_TOOL_MAX_DEADLINE_SECONDS,
                )
                + _SUPERUSER_TOOL_DEADLINE_GRACE_SECONDS
            )
        except (TypeError, ValueError):
            pass
    return _SUPERUSER_TOOL_DEFAULT_DEADLINE_SECONDS


def exception_tool_result(
    tool_call: LLMToolCall,
    exc: Exception,
) -> ToolResult:
    task_text = _task_text(tool_call)
    output = {
        "ok": False,
        "rendered_command": str(tool_call.function.name or ""),
        "task_text": task_text,
        "error": f"工具执行异常：{type(exc).__name__}: {exc}",
        "artifacts": [],
        "need_continue": True,
        "remaining_task_hint": task_text,
        "retryable": False,
        "status": "tool_execution_exception",
    }
    return ToolResult(
        output=output,
        display_content=normalize_message_text(str(output.get("error", ""))),
    )


def tool_deadline_result(
    tool_call: LLMToolCall,
    timeout: float,
) -> ToolResult:
    task_text = _task_text(tool_call)
    message = f"超级用户工具执行超时（>{timeout:.0f}s）"
    output = {
        "ok": False,
        "rendered_command": str(tool_call.function.name or ""),
        "task_text": task_text,
        "error": message,
        "artifacts": [],
        "need_continue": True,
        "remaining_task_hint": task_text,
        "retryable": False,
        "status": "tool_execution_timeout",
        "timeout_seconds": timeout,
    }
    return ToolResult(output=output, display_content=message)


def _task_text(tool_call: LLMToolCall) -> str:
    arguments = _parse_tool_arguments(str(tool_call.function.arguments or ""))
    if not isinstance(arguments, dict):
        return ""
    return normalize_message_text(str(arguments.get(_TASK_TEXT_FIELD) or ""))


def _parse_tool_arguments(arguments: str) -> dict[str, Any] | str:
    text = str(arguments or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return text
    return value if isinstance(value, dict) else {"value": value}
