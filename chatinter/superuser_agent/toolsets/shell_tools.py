"""Shell execution tools for the superuser private Agent scenario."""

from __future__ import annotations

import asyncio
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..audit_log import record_audit_event
from ..permission_policy import decide_shell
from ..registry import register_superuser_tool
from ..workspace_isolation import resolve_cwd
from .common import (
    actor_from_context,
    approval_required_result,
    audited_error_result,
    coerce_timeout,
    decode,
    permission_denied_result,
    tool_result,
    worktree_id_from_context,
)

SHELL_TIMEOUT_SECONDS = 20.0


class ShellCommandTool:
    name = "shell_command"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：执行系统 shell 命令。适合项目维护、服务检查、"
                "git 状态查看等。uv/python 优先使用专用工具；所有命令执行前都会"
                "经过 shell.allow/ask/deny 权限策略。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的完整 shell 命令。",
                    },
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "工作目录，留空使用当前项目目录。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么需要执行该命令。",
                    },
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "description": "超时时间，默认 20 秒，最大 120 秒。",
                    },
                },
                "required": ["command", "cwd", "reason", "timeout_seconds"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", "") or "").strip()
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
        if not command:
            return tool_result(False, "shell_empty_command", command=command)
        decision = decide_shell(command)
        payload = {
            "command": command,
            "cwd": cwd,
            "reason": reason,
            "timeout_seconds": timeout_seconds,
            "isolation": isolation,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="shell_command",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="shell_command",
                payload=payload,
                permission=decision,
            )
        return await run_shell_command(
            command=command,
            cwd=cwd,
            actor=actor,
            approval_id=None,
            timeout_seconds=timeout_seconds,
            isolation=isolation,
        )


async def run_shell_command(
    *,
    command: str,
    cwd: str | None,
    actor: dict[str, str],
    approval_id: str | None = None,
    timeout_seconds: float | None = None,
    action: str = "shell_command",
    isolation: dict[str, Any] | None = None,
) -> ToolResult:
    timeout = coerce_timeout(timeout_seconds, default=SHELL_TIMEOUT_SECONDS)
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
        output = {
            "command": command,
            "cwd": cwd,
            "approval_id": approval_id,
            "isolation": isolation or {},
            "returncode": process.returncode,
            "stdout": decode(stdout),
            "stderr": decode(stderr),
        }
        ok = process.returncode == 0
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=action,
            payload={"command": command, "cwd": cwd, "approval_id": approval_id},
            result={"ok": ok, "returncode": process.returncode},
        )
        status = (
            "shell_completed" if action == "shell_command" else f"{action}_completed"
        )
        return tool_result(ok, status, **output)
    except asyncio.TimeoutError:
        return audited_error_result(
            actor=actor,
            action=action,
            payload={"command": command, "cwd": cwd, "approval_id": approval_id},
            status="timeout",
        )
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action=action,
            payload={"command": command, "cwd": cwd, "approval_id": approval_id},
            status="execution_error",
            error=str(exc),
        )


register_superuser_tool(ShellCommandTool)

__all__ = ["ShellCommandTool", "run_shell_command"]
