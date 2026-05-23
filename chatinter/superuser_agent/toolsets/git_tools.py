"""Git tools for the superuser private Agent scenario."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..permission_policy import decide_git
from ..registry import register_superuser_tool
from ..workspace_isolation import resolve_cwd
from .common import (
    actor_from_context,
    approval_required_result,
    coerce_timeout,
    permission_denied_result,
    tool_result,
    worktree_id_from_context,
)
from .shell_tools import run_shell_command


class GitCommandTool:
    name = "git_command"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：执行 git 操作。只传 git 后面的参数，例如 "
                "'status --short'、'diff'、'add path'、'commit -m msg'、'push origin main'。"
                "执行前经过 git.allow/ask/deny；不要用 shell_command 执行 git。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "git 后面的完整参数，或以 git 开头的完整命令。",
                    },
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "仓库目录，留空使用当前项目目录。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么需要执行该 git 操作。",
                    },
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "description": "超时时间，默认 20 秒，最大 120 秒。",
                    },
                },
                "required": ["args", "cwd", "reason", "timeout_seconds"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        args = str(kwargs.get("args", "") or "").strip()
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
            return tool_result(False, "worktree_resolution_failed", cwd=cwd, isolation=isolation)
        if not args:
            return tool_result(False, "git_empty_args")
        command = git_command_from_args(args)
        decision = decide_git(command)
        payload = {
            "args": args,
            "command": command,
            "cwd": cwd,
            "reason": reason,
            "timeout_seconds": timeout_seconds,
            "isolation": isolation,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="git_command",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="git_command",
                payload=payload,
                permission=decision,
            )
        return await run_git_command(
            args=args,
            cwd=cwd,
            actor=actor,
            approval_id=None,
            timeout_seconds=timeout_seconds,
            isolation=isolation,
        )


async def run_git_command(
    *,
    args: str,
    cwd: str | None,
    actor: dict[str, str],
    approval_id: str | None = None,
    timeout_seconds: float | None = None,
    isolation: dict[str, Any] | None = None,
) -> ToolResult:
    command = git_command_from_args(args)
    return await run_shell_command(
        command=command,
        cwd=cwd,
        actor=actor,
        approval_id=approval_id,
        timeout_seconds=timeout_seconds,
        action="git_command",
        isolation=isolation,
    )


def git_command_from_args(args: str) -> str:
    args = str(args or "").strip()
    if args.startswith("git "):
        return args
    return f"git {args}"


register_superuser_tool(GitCommandTool)

__all__ = ["GitCommandTool", "git_command_from_args", "run_git_command"]
