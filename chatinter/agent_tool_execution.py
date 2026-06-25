"""Tool execution helpers for ChatInter agent runtime."""

from __future__ import annotations

import asyncio
from typing import Any

from zhenxun.services.llm.tools import RunContext, ToolInvoker
from zhenxun.services.llm.types.models import LLMToolCall, ToolResult

from .command_observation import build_command_observation
from .route_text import normalize_message_text
from .superuser_agent.tool_guardrail import SUPERUSER_AGENT_MODES, _parse_tool_arguments
from .task_frame import TASK_TEXT_FIELD

_SUPERUSER_TOOL_DEFAULT_DEADLINE_SECONDS = 120.0
_SUPERUSER_TOOL_MAX_DEADLINE_SECONDS = 180.0
_SUPERUSER_TOOL_DEADLINE_GRACE_SECONDS = 5.0


async def execute_tool_call(
    invoker: ToolInvoker,
    tool_call: LLMToolCall,
    tool_map: dict[str, Any],
    run_context: RunContext,
    *,
    message_text: str,
    trace_id: str,
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
            tool_map=tool_map,
            message_text=message_text,
            trace_id=trace_id,
        )


def superuser_tool_deadline_seconds(
    tool_call: LLMToolCall,
    run_context: RunContext,
) -> float | None:
    extra = getattr(run_context, "extra", None)
    agent_mode = (
        normalize_message_text(str(extra.get("agent_mode", "") or ""))
        if isinstance(extra, dict)
        else ""
    )
    if agent_mode not in SUPERUSER_AGENT_MODES:
        return None
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
    *,
    tool_map: dict[str, Any],
    message_text: str,
    trace_id: str,
) -> ToolResult:
    executable = tool_map.get(str(tool_call.function.name or ""))
    binding = getattr(executable, "binding", None)
    candidate = getattr(binding, "candidate", None)
    task_text = _task_text(tool_call)
    if candidate is None:
        output = {
            "ok": False,
            "command_id": "",
            "rendered_command": "",
            "matched_plugin": "",
            "task_text": task_text,
            "error": f"工具执行异常：{type(exc).__name__}: {exc}",
            "messages_sent": [],
            "artifacts": [],
            "need_continue": True,
            "remaining_task_hint": task_text,
            "retryable": False,
            "status": "tool_execution_exception",
        }
    else:
        output = build_command_observation(
            ok=False,
            command_id=getattr(binding, "command_id", ""),
            rendered_command=getattr(candidate.schema, "head", ""),
            matched_plugin=getattr(candidate, "plugin_name", ""),
            task_text=task_text,
            ambient_message=message_text,
            trace_id=trace_id,
            error=f"工具执行异常：{type(exc).__name__}: {exc}",
            retryable=False,
            plugin_module=getattr(candidate, "plugin_module", ""),
        )
        output["status"] = "tool_execution_exception"
    return ToolResult(
        output=output,
        display_content=normalize_message_text(str(output.get("error", ""))),
    )


def tool_deadline_result(
    tool_call: LLMToolCall,
    timeout: float,
    *,
    tool_map: dict[str, Any],
    message_text: str,
    trace_id: str,
) -> ToolResult:
    executable = tool_map.get(str(tool_call.function.name or ""))
    binding = getattr(executable, "binding", None)
    candidate = getattr(binding, "candidate", None)
    task_text = _task_text(tool_call)
    message = f"超级用户工具执行超时（>{timeout:.0f}s）"
    if candidate is None:
        output = {
            "ok": False,
            "command_id": "",
            "rendered_command": str(tool_call.function.name or ""),
            "matched_plugin": str(tool_call.function.name or ""),
            "task_text": task_text,
            "error": message,
            "messages_sent": [],
            "artifacts": [],
            "need_continue": True,
            "remaining_task_hint": task_text,
            "retryable": False,
            "status": "tool_execution_timeout",
            "timeout_seconds": timeout,
        }
    else:
        output = build_command_observation(
            ok=False,
            command_id=getattr(binding, "command_id", ""),
            rendered_command=getattr(candidate.schema, "head", ""),
            matched_plugin=getattr(candidate, "plugin_name", ""),
            task_text=task_text,
            ambient_message=message_text,
            trace_id=trace_id,
            error=message,
            retryable=False,
            plugin_module=getattr(candidate, "plugin_module", ""),
        )
        output["status"] = "tool_execution_timeout"
        output["timeout_seconds"] = timeout
    return ToolResult(output=output, display_content=message)


def _task_text(tool_call: LLMToolCall) -> str:
    arguments = _parse_tool_arguments(str(tool_call.function.arguments or ""))
    if not isinstance(arguments, dict):
        return ""
    return normalize_message_text(str(arguments.get(TASK_TEXT_FIELD) or ""))
