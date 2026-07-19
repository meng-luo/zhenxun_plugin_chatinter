"""Shell execution tools for the superuser private Agent scenario."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from ...artifact_store import get_artifact_store
from ...llm_compat import ToolDefinition, ToolResult
from ..audit_log import record_audit_event
from ..permission_policy import decide_shell
from ..process_control import (
    attach_process_tree,
    release_process_tree,
    subprocess_group_kwargs,
    terminate_process_tree,
)
from .common import (
    MAX_TIMEOUT_SECONDS,
    actor_from_context,
    approval_required_result,
    audited_error_result,
    coerce_timeout,
    permission_denied_result,
    tool_result,
)

SHELL_TIMEOUT_SECONDS = 120.0
_STREAM_READ_BYTES = 64 * 1024
_STREAM_PREVIEW_BYTES = 8_000
class _StreamCapture:
    def __init__(self, name: str) -> None:
        self.name = name
        self.head = bytearray()
        self.tail = bytearray()
        self.total_bytes = 0
        handle = tempfile.NamedTemporaryFile(
            prefix="chatinter-shell-",
            suffix=".log",
            delete=False,
        )
        self.path = Path(handle.name)
        self._handle = handle

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._handle.write(chunk)
        self.total_bytes += len(chunk)
        head_limit = _STREAM_PREVIEW_BYTES // 2
        if len(self.head) < head_limit:
            consumed = min(head_limit - len(self.head), len(chunk))
            self.head.extend(chunk[:consumed])
            chunk = chunk[consumed:]
        if chunk:
            self.tail.extend(chunk)
            tail_limit = _STREAM_PREVIEW_BYTES - head_limit
            if len(self.tail) > tail_limit:
                del self.tail[: len(self.tail) - tail_limit]

    def preview(self) -> str:
        omitted = max(self.total_bytes - len(self.head) - len(self.tail), 0)
        marker = f"\n[... {omitted} bytes omitted ...]\n" if omitted else ""
        return (
            self.head.decode("utf-8", errors="replace")
            + marker
            + self.tail.decode("utf-8", errors="replace")
        )

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def discard(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)


class ShellCommandTool:
    name = "shell_command"
    read_only = False

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "执行系统 shell 命令，适合项目维护、测试、Git、Python 和服务操作。"
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
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "description": (
                            f"超时时间，默认 {SHELL_TIMEOUT_SECONDS:.0f} 秒，"
                            f"最大 {MAX_TIMEOUT_SECONDS:.0f} 秒。"
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", "") or "").strip()
        cwd = str(kwargs.get("cwd", "") or "").strip() or None
        reason = str(kwargs.get("reason", "") or "")
        timeout_seconds = coerce_timeout(
            kwargs.get("timeout_seconds"),
            default=SHELL_TIMEOUT_SECONDS,
        )
        actor = actor_from_context(context)
        cwd = str(Path(cwd).resolve()) if cwd else None
        if not command:
            return tool_result(False, "shell_empty_command", command=command)
        decision = decide_shell(command, cwd=cwd)
        payload = {
            "command": command,
            "cwd": cwd,
            "reason": reason,
            "timeout_seconds": timeout_seconds,
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
        )


async def run_shell_command(
    *,
    command: str,
    cwd: str | None,
    actor: dict[str, str],
    approval_id: str | None = None,
    timeout_seconds: float | None = None,
    action: str = "shell_command",
) -> ToolResult:
    timeout = coerce_timeout(timeout_seconds, default=SHELL_TIMEOUT_SECONDS)
    try:
        shell_command, shell_name, shell_script = _local_shell_command(command)
    except OSError as exc:
        return audited_error_result(
            actor=actor,
            action=action,
            payload={
                "command": command,
                "cwd": cwd,
                "approval_id": approval_id,
                "execution_backend": "local",
            },
            status="shell_unavailable",
            error=f"本机 Shell 初始化失败，命令未执行：{exc}",
        )
    if not shell_command:
        return audited_error_result(
            actor=actor,
            action=action,
            payload={
                "command": command,
                "cwd": cwd,
                "approval_id": approval_id,
                "execution_backend": "local",
            },
            status="shell_unavailable",
            error="未找到可用的本机 Shell，命令未执行。",
        )
    process: asyncio.subprocess.Process | None = None
    captures: dict[str, _StreamCapture] = {}
    reader_tasks: list[asyncio.Task[None]] = []
    try:
        process = await asyncio.create_subprocess_exec(
            *shell_command,
            cwd=str(Path(cwd).resolve() if cwd else Path.cwd().resolve()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **subprocess_group_kwargs(),
        )
        attach_process_tree(process)
        captures = {
            "stdout": _StreamCapture("stdout"),
            "stderr": _StreamCapture("stderr"),
        }
        reader_tasks = [
            asyncio.create_task(_consume_stream(process.stdout, captures["stdout"])),
            asyncio.create_task(_consume_stream(process.stderr, captures["stderr"])),
        ]
        await asyncio.wait_for(process.wait(), timeout=timeout)
        await asyncio.gather(*reader_tasks)
        release_process_tree(process)
        output = {
            "command": command,
            "cwd": cwd,
            "approval_id": approval_id,
            "execution_backend": "local",
            "shell": shell_name,
            "returncode": process.returncode,
            **_finalize_captures(
                captures,
                actor=actor,
                command=command,
                action=action,
            ),
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
        if process is not None:
            await terminate_process_tree(process)
        await _finish_readers(reader_tasks)
        return audited_error_result(
            actor=actor,
            action=action,
            payload={
                "command": command,
                "cwd": cwd,
                "approval_id": approval_id,
                "execution_backend": "local",
                "shell": shell_name,
                "returncode": process.returncode if process is not None else None,
                **_finalize_captures(
                    captures,
                    actor=actor,
                    command=command,
                    action=action,
                ),
                "timed_out": True,
            },
            status="timeout",
        )
    except asyncio.CancelledError:
        if process is not None:
            await terminate_process_tree(process)
        await _finish_readers(reader_tasks)
        return audited_error_result(
            actor=actor,
            action=action,
            payload={
                "command": command,
                "cwd": cwd,
                "approval_id": approval_id,
                "execution_backend": "local",
                "shell": shell_name,
                "returncode": process.returncode if process is not None else None,
                **_finalize_captures(
                    captures,
                    actor=actor,
                    command=command,
                    action=action,
                ),
                "cancelled": True,
            },
            status="cancelled",
        )
    except Exception as exc:
        if process is not None and process.returncode is None:
            await terminate_process_tree(process)
        await _finish_readers(reader_tasks)
        return audited_error_result(
            actor=actor,
            action=action,
            payload={
                "command": command,
                "cwd": cwd,
                "approval_id": approval_id,
                "execution_backend": "local",
                "shell": shell_name,
                **_finalize_captures(
                    captures,
                    actor=actor,
                    command=command,
                    action=action,
                ),
            },
            status="execution_error",
            error=str(exc),
        )
    finally:
        if shell_script is not None:
            shell_script.unlink(missing_ok=True)


def _local_shell_command(command: str) -> tuple[tuple[str, ...], str, Path | None]:
    shell = _resolve_shell()
    if not shell:
        return (), "", None
    windows = os.name == "nt"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\r\n" if windows else "\n",
        suffix=".cmd" if windows else ".sh",
        prefix="chatinter-shell-",
        delete=False,
    ) as handle:
        if windows:
            handle.write("@echo off\nchcp 65001 >nul\n")
        handle.write(command)
        handle.write("\n")
        script_path = Path(handle.name)
    argv = (
        (shell, "/d", "/s", "/c", str(script_path))
        if windows
        else (shell, str(script_path))
    )
    return argv, Path(shell).name, script_path


def _resolve_shell() -> str:
    configured = os.environ.get("COMSPEC" if os.name == "nt" else "SHELL", "").strip()
    fallback = "cmd.exe" if os.name == "nt" else "/bin/sh"
    candidate = configured or fallback
    if Path(candidate).is_file():
        return str(Path(candidate))
    return str(shutil.which(candidate) or "")


async def _consume_stream(
    stream: asyncio.StreamReader | None,
    capture: _StreamCapture,
) -> None:
    if stream is None:
        return
    while chunk := await stream.read(_STREAM_READ_BYTES):
        capture.write(chunk)


async def _finish_readers(tasks: list[asyncio.Task[None]]) -> None:
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _finalize_captures(
    captures: dict[str, _StreamCapture],
    *,
    actor: dict[str, str],
    command: str,
    action: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {"stdout": "", "stderr": "", "artifacts": []}
    for name, capture in captures.items():
        capture.close()
        output[name] = capture.preview()
        ref = get_artifact_store().store_file(
            capture.path,
            artifact_type="log",
            trace_id=str(actor.get("trace_id", "") or ""),
            source=f"{action}:{name}",
            summary=f"{name} from {command[:120]}",
            mime_type="text/plain",
            move=True,
        )
        if ref is not None:
            output["artifacts"].append(ref.to_dict())
        else:
            capture.discard()
    return output


__all__ = ["ShellCommandTool", "run_shell_command"]
