"""Background task tools for the superuser private Agent scenario."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..background_tasks import (
    cancel_background_task,
    get_background_task,
    list_background_tasks,
    start_background_command,
)
from ..permission_policy import (
    PermissionResult,
    decide_background,
    decide_git,
    decide_python,
    decide_server,
    decide_shell,
    decide_uv,
)
from ..registry import register_superuser_tool
from .common import (
    actor_from_context,
    approval_required_result,
    permission_denied_result,
    tool_result,
)

_COMMAND_TYPES = {"shell", "git", "uv", "python_module", "server"}


class BackgroundTaskStartTool:
    name = "background_task_start"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：把较慢的命令放到后台执行，立即返回 task_id。"
                "适合长时间 uv、git、服务维护、脚本检查等任务。启动前经过 background "
                "和具体命令域的 allow/ask/deny 策略。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command_type": {
                        "type": "string",
                        "enum": sorted(_COMMAND_TYPES),
                        "description": "命令类型：shell/git/uv/python_module/server。",
                    },
                    "command": {
                        "type": "string",
                        "description": "命令文本。git/uv 可传参数或完整命令；python_module 传模块名。",
                    },
                    "args": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "python_module 的参数；其他类型通常传 null。",
                    },
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "工作目录，留空使用当前项目目录。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么需要后台执行。",
                    },
                },
                "required": ["command_type", "command", "args", "cwd", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        payload = _normalize_start_payload(kwargs)
        if payload["command_type"] not in _COMMAND_TYPES:
            return tool_result(False, "background_invalid_command_type", **payload)
        command = _render_command(payload)
        if not command:
            return tool_result(False, "background_empty_command", **payload)
        payload["rendered_command"] = command
        background_decision = decide_background("background_task_start " + payload["command_type"])
        command_decision = _decide_command(payload["command_type"], command)
        for decision in (background_decision, command_decision):
            if decision.decision == "deny":
                return permission_denied_result(
                    actor=actor,
                    action="background_task_start",
                    payload=payload,
                    permission=decision,
                )
        ask_decision = _first_ask(background_decision, command_decision)
        if ask_decision is not None:
            return approval_required_result(
                actor=actor,
                action="background_task_start",
                payload=payload,
                permission=ask_decision,
            )
        return await start_background_task(actor=actor, approval_id=None, **payload)


class BackgroundTaskStatusTool:
    name = "background_task_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：查看后台任务状态、返回码和输出摘要。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": ["string", "null"],
                        "description": "任务 ID；为空则列出当前会话任务。",
                    },
                    "include_output": {
                        "type": ["boolean", "null"],
                        "description": "是否返回 stdout/stderr，默认 true。",
                    },
                    "include_finished": {
                        "type": ["boolean", "null"],
                        "description": "列出任务时是否包含已结束任务，默认 true。",
                    },
                },
                "required": ["task_id", "include_output", "include_finished"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        task_id = str(kwargs.get("task_id", "") or "").strip()
        include_output = True if kwargs.get("include_output") is None else bool(kwargs.get("include_output"))
        include_finished = True if kwargs.get("include_finished") is None else bool(kwargs.get("include_finished"))
        if task_id:
            task = get_background_task(
                task_id=task_id,
                user_id=actor["user_id"],
                session_key=actor["session_key"],
            )
            if task is None:
                return tool_result(False, "background_task_not_found", task_id=task_id)
            return tool_result(
                True,
                "background_task_status",
                task=task.public_payload(include_output=include_output),
            )
        tasks = list_background_tasks(
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            include_finished=include_finished,
        )
        return tool_result(
            True,
            "background_task_listed",
            tasks=[task.public_payload(include_output=include_output) for task in tasks[:50]],
            count=len(tasks),
        )


class BackgroundTaskCancelTool:
    name = "background_task_cancel"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：取消后台任务。默认需要确认，确认后会 terminate/kill 兜底。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "要取消的后台任务 ID。"},
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么要取消该任务。",
                    },
                },
                "required": ["task_id", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        task_id = str(kwargs.get("task_id", "") or "").strip()
        reason = str(kwargs.get("reason", "") or "")
        if not task_id:
            return tool_result(False, "background_task_id_required")
        decision = decide_background("background_task_cancel " + task_id)
        payload = {"task_id": task_id, "reason": reason}
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="background_task_cancel",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="background_task_cancel",
                payload=payload,
                permission=decision,
            )
        return await cancel_task(actor=actor, task_id=task_id)


async def start_background_task(
    *,
    actor: dict[str, str],
    command_type: str,
    command: str,
    args: list[str],
    cwd: str | None,
    reason: str,
    rendered_command: str,
    approval_id: str | None = None,
) -> ToolResult:
    task = start_background_command(
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action="background_task_start:" + command_type,
        command=rendered_command,
        cwd=cwd,
        reason=reason,
        approval_id=approval_id,
    )
    return tool_result(
        True,
        "background_task_started",
        task=task.public_payload(include_output=False),
        instruction="稍后调用 background_task_status 查看进度和输出。",
    )


async def cancel_task(*, actor: dict[str, str], task_id: str) -> ToolResult:
    task = await cancel_background_task(
        task_id=task_id,
        user_id=actor["user_id"],
        session_key=actor["session_key"],
    )
    if task is None:
        return tool_result(False, "background_task_not_found", task_id=task_id)
    return tool_result(
        True,
        "background_task_cancelled",
        task=task.public_payload(include_output=True),
    )


def _normalize_start_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_type": str(kwargs.get("command_type", "") or "").strip(),
        "command": str(kwargs.get("command", "") or "").strip(),
        "args": [str(item) for item in (kwargs.get("args") or [])],
        "cwd": str(kwargs.get("cwd", "") or "").strip() or None,
        "reason": str(kwargs.get("reason", "") or ""),
    }


def _render_command(payload: dict[str, Any]) -> str:
    command_type = payload["command_type"]
    command = str(payload["command"] or "").strip()
    args = [str(item) for item in payload.get("args", [])]
    if command_type == "git":
        return command if command.startswith("git ") else f"git {command}"
    if command_type == "uv":
        return command if command.startswith(("uv ", "uvx ")) else f"uv {command}"
    if command_type == "python_module":
        if not command:
            return ""
        return " ".join(["python", "-m", command, *args])
    return command


def _decide_command(command_type: str, command: str) -> PermissionResult:
    if command_type == "git":
        return decide_git(command)
    if command_type == "uv":
        return decide_uv(command)
    if command_type == "python_module":
        return decide_python(command)
    if command_type == "server":
        return decide_server(command)
    return decide_shell(command)


def _first_ask(*decisions: PermissionResult) -> PermissionResult | None:
    for decision in decisions:
        if decision.decision == "ask":
            return decision
    return None


register_superuser_tool(BackgroundTaskStartTool)
register_superuser_tool(BackgroundTaskStatusTool)
register_superuser_tool(BackgroundTaskCancelTool)

__all__ = [
    "BackgroundTaskCancelTool",
    "BackgroundTaskStartTool",
    "BackgroundTaskStatusTool",
    "cancel_task",
    "start_background_task",
]
