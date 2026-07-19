"""Tool execution helpers for ChatInter agent runtime."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ...llm_compat import LLMToolCall, RunContext, ToolInvoker, ToolResult
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
    tool_name = str(tool_call.function.name or "")
    tool = tool_map.get(tool_name)
    if tool is None:
        return _tool_validation_result(
            tool_name,
            status="tool_not_found",
            error=f"未知工具：{tool_name or '<empty>'}",
        )

    raw_arguments = str(tool_call.function.arguments or "").strip()
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, ValueError) as exc:
        return _tool_validation_result(
            tool_name,
            validation_error="invalid_json",
            error=f"工具参数不是有效 JSON：{exc}",
        )
    if not isinstance(arguments, dict):
        return _tool_validation_result(
            tool_name,
            validation_error="arguments_not_object",
            error="工具参数必须是 JSON object。",
        )

    definition = await tool.get_definition()
    schema = definition.parameters if isinstance(definition.parameters, dict) else {}
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required")
    required = required if isinstance(required, list | tuple) else ()
    missing = [str(name) for name in required if str(name) not in arguments]
    if missing:
        return _tool_validation_result(
            tool_name,
            validation_error="missing_required",
            error="缺少必填参数：" + ", ".join(missing),
            missing_fields=missing,
        )

    if schema.get("additionalProperties") is False:
        unexpected = sorted(str(name) for name in arguments if name not in properties)
        if unexpected:
            return _tool_validation_result(
                tool_name,
                validation_error="unexpected_arguments",
                error="包含未定义参数：" + ", ".join(unexpected),
                unexpected_fields=unexpected,
            )

    for name, value in arguments.items():
        property_schema = properties.get(name)
        if not isinstance(property_schema, dict):
            continue
        expected = property_schema.get("type")
        expected_types = [expected] if isinstance(expected, str) else expected
        if not isinstance(expected_types, list | tuple):
            continue
        if any(_matches_json_type(value, item) for item in expected_types):
            continue
        return _tool_validation_result(
            tool_name,
            validation_error="type_mismatch",
            error=f"参数 {name} 类型无效，期望：{', '.join(map(str, expected_types))}",
            field=name,
            expected_types=list(expected_types),
        )
    return None


def _matches_json_type(value: Any, expected: Any) -> bool:
    return {
        "array": lambda: isinstance(value, list),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "null": lambda: value is None,
        "number": lambda: isinstance(value, int | float)
        and not isinstance(value, bool),
        "object": lambda: isinstance(value, dict),
        "string": lambda: isinstance(value, str),
    }.get(str(expected), lambda: False)()


def _tool_validation_result(
    tool_name: str,
    *,
    status: str = "invalid_tool_arguments",
    validation_error: str = "",
    error: str,
    **details: Any,
) -> ToolResult:
    return ToolResult(
        output={
            "ok": False,
            "status": status,
            "tool_name": tool_name,
            "validation_error": validation_error,
            "error": error,
            "retryable": True,
            "need_continue": True,
            **details,
        },
        display_content=error,
        is_error=True,
        is_retryable=True,
    )


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
