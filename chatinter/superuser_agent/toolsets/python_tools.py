"""Python execution tools for the superuser private Agent scenario."""

from __future__ import annotations

import asyncio
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..audit_log import record_audit_event
from ..permission_policy import decide_python
from ..registry import register_superuser_tool
from ..workspace_isolation import resolve_cwd
from .common import (
    actor_from_context,
    approval_required_result,
    audited_error_result,
    coerce_timeout,
    decode,
    permission_denied_result,
    project_root,
    tool_result,
    worktree_id_from_context,
)


class PythonExecTool:
    name = "python_exec"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：使用项目 Python 解释器执行一段临时代码。"
                "适合快速检查、脚本化分析和维护任务；执行前经过 python.allow/ask/deny。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码。"},
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "工作目录，留空使用当前项目目录。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么需要执行这段代码。",
                    },
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "description": "超时时间，默认 20 秒，最大 120 秒。",
                    },
                },
                "required": ["code", "cwd", "reason", "timeout_seconds"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        code = str(kwargs.get("code", "") or "")
        cwd = str(kwargs.get("cwd", "") or "").strip() or None
        reason = str(kwargs.get("reason", "") or "")
        timeout_seconds = coerce_timeout(kwargs.get("timeout_seconds"))
        actor = actor_from_context(context)
        cwd, isolation = resolve_cwd(
            cwd,
            actor=actor,
            worktree_id=worktree_id_from_context(context),
        )
        if isolation.get("invalid_worktree") or isolation.get("escaped_worktree"):
            return tool_result(
                False, "worktree_resolution_failed", cwd=cwd, isolation=isolation
            )
        if not code.strip():
            return tool_result(False, "python_empty_code")
        decision = decide_python("python_exec " + code[:240].replace("\n", " "))
        payload = {
            "code": code,
            "cwd": cwd,
            "reason": reason,
            "timeout_seconds": timeout_seconds,
            "isolation": isolation,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="python_exec",
                payload={**payload, "code": code[:240]},
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="python_exec",
                payload=payload,
                permission=decision,
            )
        return await run_python_code(
            code=code,
            cwd=cwd,
            actor=actor,
            timeout_seconds=timeout_seconds,
            isolation=isolation,
        )


class PythonModuleTool:
    name = "python_module"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：执行 python -m <module>。适合 pytest、py_compile、"
                "脚本模块等；执行前经过 python.allow/ask/deny。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "模块名，例如 pytest。",
                    },
                    "args": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "传给模块的参数数组。",
                    },
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "工作目录，留空使用当前项目目录。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么需要执行该模块。",
                    },
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "description": "超时时间，默认 20 秒，最大 120 秒。",
                    },
                },
                "required": ["module", "args", "cwd", "reason", "timeout_seconds"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        module = str(kwargs.get("module", "") or "").strip()
        args = [str(item) for item in (kwargs.get("args") or [])]
        cwd = str(kwargs.get("cwd", "") or "").strip() or None
        reason = str(kwargs.get("reason", "") or "")
        timeout_seconds = coerce_timeout(kwargs.get("timeout_seconds"))
        actor = actor_from_context(context)
        cwd, isolation = resolve_cwd(
            cwd,
            actor=actor,
            worktree_id=worktree_id_from_context(context),
        )
        if isolation.get("invalid_worktree") or isolation.get("escaped_worktree"):
            return tool_result(
                False, "worktree_resolution_failed", cwd=cwd, isolation=isolation
            )
        if not module:
            return tool_result(False, "python_empty_module")
        command_preview = "python -m " + " ".join([module, *args])
        decision = decide_python(command_preview)
        payload = {
            "module": module,
            "args": args,
            "cwd": cwd,
            "reason": reason,
            "timeout_seconds": timeout_seconds,
            "isolation": isolation,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="python_module",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="python_module",
                payload=payload,
                permission=decision,
            )
        return await run_python_module(
            module=module,
            args=args,
            cwd=cwd,
            actor=actor,
            timeout_seconds=timeout_seconds,
            isolation=isolation,
        )


async def run_python_code(
    *,
    code: str,
    cwd: str | None,
    actor: dict[str, str],
    approval_id: str | None = None,
    timeout_seconds: float | None = None,
    isolation: dict[str, Any] | None = None,
) -> ToolResult:
    return await _run_python_process(
        args=["-"],
        stdin=code.encode("utf-8"),
        cwd=cwd,
        actor=actor,
        action="python_exec",
        approval_id=approval_id,
        timeout_seconds=timeout_seconds,
        payload={"code_preview": code[:240], "isolation": isolation or {}},
    )


async def run_python_module(
    *,
    module: str,
    args: list[str],
    cwd: str | None,
    actor: dict[str, str],
    approval_id: str | None = None,
    timeout_seconds: float | None = None,
    isolation: dict[str, Any] | None = None,
) -> ToolResult:
    return await _run_python_process(
        args=["-m", module, *args],
        stdin=None,
        cwd=cwd,
        actor=actor,
        action="python_module",
        approval_id=approval_id,
        timeout_seconds=timeout_seconds,
        payload={"module": module, "args": args, "isolation": isolation or {}},
    )


async def _run_python_process(
    *,
    args: list[str],
    stdin: bytes | None,
    cwd: str | None,
    actor: dict[str, str],
    action: str,
    approval_id: str | None,
    timeout_seconds: float | None,
    payload: dict[str, Any],
) -> ToolResult:
    timeout = coerce_timeout(timeout_seconds)
    executable = _python_executable()
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin),
            timeout=timeout,
        )
        ok = process.returncode == 0
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=action,
            payload={**payload, "cwd": cwd, "approval_id": approval_id},
            result={"ok": ok, "returncode": process.returncode},
        )
        return tool_result(
            ok,
            f"{action}_completed",
            executable=str(executable),
            args=args,
            cwd=cwd,
            approval_id=approval_id,
            isolation=dict(payload.get("isolation") or {}),
            returncode=process.returncode,
            stdout=decode(stdout),
            stderr=decode(stderr),
        )
    except asyncio.TimeoutError:
        return audited_error_result(
            actor=actor,
            action=action,
            payload={**payload, "cwd": cwd, "approval_id": approval_id},
            status="timeout",
        )
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action=action,
            payload={**payload, "cwd": cwd, "approval_id": approval_id},
            status="execution_error",
            error=str(exc),
        )


def _python_executable() -> str:
    root = project_root()
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "python"


register_superuser_tool(PythonExecTool)
register_superuser_tool(PythonModuleTool)

__all__ = [
    "PythonExecTool",
    "PythonModuleTool",
    "run_python_code",
    "run_python_module",
]
