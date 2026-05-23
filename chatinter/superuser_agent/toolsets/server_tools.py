"""Server maintenance tools for the superuser private Agent scenario."""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import socket
from pathlib import Path
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..audit_log import record_audit_event
from ..permission_policy import decide_server
from ..registry import register_superuser_tool
from ..workspace_isolation import resolve_cwd, resolve_working_path
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
from .shell_tools import run_shell_command


class ServerStatusTool:
    name = "server_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查看宿主机基础状态，包括系统、Python、磁盘、"
                "可选 psutil 内存/CPU 摘要。不会执行外部命令。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": ["string", "null"],
                        "description": "统计磁盘空间的路径，留空使用项目目录。",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        decision = decide_server("server_status")
        path = str(kwargs.get("path", "") or "").strip() or str(project_root())
        path, isolation = resolve_working_path(
            path,
            actor=actor,
            worktree_id=worktree_id_from_context(context),
        )
        if isolation.get("invalid_worktree") or isolation.get("escaped_worktree"):
            return tool_result(False, "worktree_resolution_failed", path=path, isolation=isolation)
        payload = {"path": path, "isolation": isolation}
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="server_status",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="server_status",
                payload=payload,
                permission=decision,
            )
        return await server_status(path=path, actor=actor, isolation=isolation)


class ProcessListTool:
    name = "process_list"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查看进程列表，可按关键字过滤。优先使用 psutil，"
                "否则使用系统只读进程命令。受 server.allow/ask/deny 控制。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": ["string", "null"],
                        "description": "可选过滤关键字，例如 python、uvicorn、zhenxun。",
                    },
                    "max_results": {
                        "type": ["integer", "null"],
                        "description": "最多返回进程数，默认 40。",
                    },
                },
                "required": ["query", "max_results"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        query = str(kwargs.get("query", "") or "").strip()
        try:
            max_results = max(1, min(int(kwargs.get("max_results") or 40), 120))
        except (TypeError, ValueError):
            max_results = 40
        decision = decide_server("process_list " + query)
        payload = {"query": query, "max_results": max_results}
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="process_list",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="process_list",
                payload=payload,
                permission=decision,
            )
        return await process_list(query=query, max_results=max_results, actor=actor)


class ServerCommandTool:
    name = "server_command"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：执行服务器维护命令，例如 systemctl、service、pm2、"
                "screen、taskkill、kill、netstat、ss、df、free、uptime。执行前经过 "
                "server.allow/ask/deny；不要用 shell_command 执行服务器维护命令。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "完整服务器维护命令。",
                    },
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "工作目录，留空使用当前项目目录。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么需要执行该维护命令。",
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
            return tool_result(False, "worktree_resolution_failed", cwd=cwd, isolation=isolation)
        if not command:
            return tool_result(False, "server_empty_command")
        decision = decide_server(command)
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
                action="server_command",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="server_command",
                payload=payload,
                permission=decision,
            )
        return await run_server_command(
            command=command,
            cwd=cwd,
            actor=actor,
            approval_id=None,
            timeout_seconds=timeout_seconds,
            isolation=isolation,
        )


async def server_status(
    *,
    path: str,
    actor: dict[str, str],
    isolation: dict[str, Any] | None = None,
    approval_id: str | None = None,
) -> ToolResult:
    try:
        disk_path = Path(path or project_root())
        usage = shutil.disk_usage(disk_path)
        payload: dict[str, Any] = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "isolation": isolation or {},
            "disk": {
                "path": str(disk_path),
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
            },
            "approval_id": approval_id,
        }
        payload.update(_psutil_status())
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="server_status",
            payload={"path": path, "approval_id": approval_id},
            result={"ok": True},
        )
        return tool_result(True, "server_status", **payload)
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="server_status",
            payload={"path": path, "approval_id": approval_id},
            status="server_status_error",
            error=str(exc),
        )


async def process_list(
    *,
    query: str,
    max_results: int,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    try:
        processes = _processes_from_psutil(query=query, max_results=max_results)
        source = "psutil"
        if processes is None:
            source = "system_command"
            processes = await _processes_from_command(query=query, max_results=max_results)
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="process_list",
            payload={"query": query, "max_results": max_results, "approval_id": approval_id},
            result={"ok": True, "count": len(processes), "source": source},
        )
        return tool_result(
            True,
            "process_listed",
            query=query,
            source=source,
            approval_id=approval_id,
            processes=processes,
            truncated=len(processes) >= max_results,
        )
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="process_list",
            payload={"query": query, "max_results": max_results, "approval_id": approval_id},
            status="process_list_error",
            error=str(exc),
        )


async def run_server_command(
    *,
    command: str,
    cwd: str | None,
    actor: dict[str, str],
    approval_id: str | None = None,
    timeout_seconds: float | None = None,
    isolation: dict[str, Any] | None = None,
) -> ToolResult:
    return await run_shell_command(
        command=command,
        cwd=cwd,
        actor=actor,
        approval_id=approval_id,
        timeout_seconds=timeout_seconds,
        action="server_command",
        isolation=isolation,
    )


def _psutil_status() -> dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "cpu_count": psutil.cpu_count(),
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
            },
        }
    except Exception:
        return {"cpu_count": os.cpu_count()}


def _processes_from_psutil(
    *,
    query: str,
    max_results: int,
) -> list[dict[str, Any]] | None:
    try:
        import psutil
    except Exception:
        return None
    normalized_query = query.lower()
    processes: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "username"]):
        if len(processes) >= max_results:
            break
        try:
            info = proc.info
            cmdline = " ".join(info.get("cmdline") or [])
            haystack = f"{info.get('name') or ''} {cmdline}".lower()
            if normalized_query and normalized_query not in haystack:
                continue
            processes.append(
                {
                    "pid": info.get("pid"),
                    "ppid": info.get("ppid"),
                    "name": info.get("name"),
                    "username": info.get("username"),
                    "cmdline": cmdline[:500],
                }
            )
        except Exception:
            continue
    return processes


async def _processes_from_command(*, query: str, max_results: int) -> list[dict[str, Any]]:
    command = "tasklist /FO CSV /NH" if os.name == "nt" else "ps -eo pid,ppid,pcpu,pmem,comm,args"
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
    lines = decode(stdout, max_chars=12000).splitlines()
    normalized_query = query.lower()
    results: list[dict[str, Any]] = []
    for line in lines:
        if len(results) >= max_results:
            break
        if normalized_query and normalized_query not in line.lower():
            continue
        results.append({"line": line[:700]})
    return results


register_superuser_tool(ServerStatusTool)
register_superuser_tool(ProcessListTool)
register_superuser_tool(ServerCommandTool)

__all__ = [
    "ProcessListTool",
    "ServerCommandTool",
    "ServerStatusTool",
    "process_list",
    "run_server_command",
    "server_status",
]
