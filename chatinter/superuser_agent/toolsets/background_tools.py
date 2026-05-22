"""Background task tools for the superuser private Agent scenario."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..background_tasks import (
    cancel_background_task,
    get_background_task,
    list_observation_events,
    list_background_tasks,
    start_background_command,
    wait_for_observation_event,
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
                    "tail_only": {
                        "type": ["boolean", "null"],
                        "description": "是否只返回 stdout/stderr 尾部，默认 true。",
                    },
                },
                "required": [
                    "task_id",
                    "include_output",
                    "include_finished",
                    "tail_only",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        task_id = str(kwargs.get("task_id", "") or "").strip()
        include_output = True if kwargs.get("include_output") is None else bool(kwargs.get("include_output"))
        include_finished = True if kwargs.get("include_finished") is None else bool(kwargs.get("include_finished"))
        tail_only = True if kwargs.get("tail_only") is None else bool(kwargs.get("tail_only"))
        if task_id:
            task = get_background_task(
                task_id=task_id,
                user_id=actor["user_id"],
                session_key=actor["session_key"],
            )
            if task is None:
                return tool_result(False, "background_task_not_found", task_id=task_id)
            payload = task.public_payload(include_output=include_output)
            if include_output and tail_only:
                payload.pop("stdout", None)
                payload.pop("stderr", None)
            return tool_result(
                True,
                "background_task_status",
                task=payload,
                observation_events=[
                    event.public_payload()
                    for event in list_observation_events(
                        task_id=task_id,
                        user_id=actor["user_id"],
                        session_key=actor["session_key"],
                        limit=8,
                    )
                ],
            )
        tasks = list_background_tasks(
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            include_finished=include_finished,
        )
        return tool_result(
            True,
            "background_task_listed",
            tasks=[
                _task_payload(
                    task,
                    include_output=include_output,
                    tail_only=tail_only,
                )
                for task in tasks[:50]
            ],
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


class BackgroundObservationWaitTool:
    name = "background_observation_wait"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：等待后台任务 ObservationEvent。适合 AgentRun "
                "resume 后继续长任务；输出会带 artifact 引用而不是塞入完整日志。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "后台任务 ID。"},
                    "after_event_id": {
                        "type": ["string", "null"],
                        "description": "只等待此 event_id 之后的新事件。",
                    },
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "description": "最多等待秒数，默认 8，最大 30。",
                    },
                    "terminal_only": {
                        "type": ["boolean", "null"],
                        "description": "是否只等待 terminal 事件，默认 true。",
                    },
                },
                "required": [
                    "task_id",
                    "after_event_id",
                    "timeout_seconds",
                    "terminal_only",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        task_id = str(kwargs.get("task_id", "") or "").strip()
        if not task_id:
            return tool_result(False, "background_task_id_required")
        event = await wait_for_observation_event(
            task_id=task_id,
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            after_event_id=str(kwargs.get("after_event_id", "") or "").strip(),
            timeout=_coerce_wait_timeout(kwargs.get("timeout_seconds")),
            terminal_only=True
            if kwargs.get("terminal_only") is None
            else bool(kwargs.get("terminal_only")),
        )
        if event is None:
            return tool_result(
                False,
                "background_observation_timeout",
                task_id=task_id,
                retryable=True,
                need_continue=True,
                instruction="稍后再次调用 background_observation_wait 或 background_task_status。",
            )
        return tool_result(
            event.status == "completed",
            "background_observation_event",
            event=event.public_payload(),
            task_id=event.task_id,
            event_id=event.event_id,
            artifacts=list(event.artifacts),
            retryable=event.status not in {"completed", "failed", "cancelled", "error"},
            need_continue=event.status not in {"completed", "failed", "cancelled", "error"},
        )


class BackgroundObservationListTool:
    name = "background_observation_list"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：列出后台任务 ObservationEvent 流。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": ["string", "null"],
                        "description": "任务 ID；为空则列出当前会话最近事件。",
                    },
                    "after_event_id": {
                        "type": ["string", "null"],
                        "description": "只列出此 event_id 之后的事件。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "返回事件数，默认 20，最大 100。",
                    },
                    "terminal_only": {
                        "type": ["boolean", "null"],
                        "description": "是否只列出 terminal 事件，默认 false。",
                    },
                },
                "required": ["task_id", "after_event_id", "limit", "terminal_only"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        events = list_observation_events(
            task_id=str(kwargs.get("task_id", "") or "").strip(),
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            after_event_id=str(kwargs.get("after_event_id", "") or "").strip(),
            limit=_coerce_limit(kwargs.get("limit")),
            terminal_only=bool(kwargs.get("terminal_only") or False),
        )
        return tool_result(
            True,
            "background_observation_listed",
            events=[event.public_payload() for event in events],
            count=len(events),
        )


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
        observation_event=getattr(
            getattr(task, "last_observation_event", None),
            "public_payload",
            lambda: {},
        )(),
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


def _coerce_wait_timeout(value: Any) -> float:
    try:
        return max(0.5, min(float(value or 8.0), 30.0))
    except (TypeError, ValueError):
        return 8.0


def _coerce_limit(value: Any) -> int:
    try:
        return max(1, min(int(value or 20), 100))
    except (TypeError, ValueError):
        return 20


def _task_payload(task: Any, *, include_output: bool, tail_only: bool) -> dict[str, Any]:
    payload = task.public_payload(include_output=include_output)
    if include_output and tail_only:
        payload.pop("stdout", None)
        payload.pop("stderr", None)
    return payload


register_superuser_tool(BackgroundTaskStartTool)
register_superuser_tool(BackgroundTaskStatusTool)
register_superuser_tool(BackgroundTaskCancelTool)
register_superuser_tool(BackgroundObservationWaitTool)
register_superuser_tool(BackgroundObservationListTool)

__all__ = [
    "BackgroundObservationListTool",
    "BackgroundObservationWaitTool",
    "BackgroundTaskCancelTool",
    "BackgroundTaskStartTool",
    "BackgroundTaskStatusTool",
    "cancel_task",
    "start_background_task",
]
