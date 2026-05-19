"""Durable background task registry for superuser Agent operations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
import uuid
from typing import Any

from ..persistence import read_json, state_path, write_json
from .audit_log import record_audit_event

_MAX_OUTPUT_CHARS = 8000
_TASKS_PATH = state_path("background_tasks.json")
_TASKS: dict[str, "BackgroundTask"] = {}
_LOADED = False


@dataclass
class BackgroundTask:
    task_id: str
    user_id: str
    session_key: str
    action: str
    command: str
    cwd: str | None
    reason: str = ""
    approval_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "running"
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    process: asyncio.subprocess.Process | None = None
    runner: asyncio.Task[None] | None = None

    def public_payload(self, *, include_output: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "session_key": self.session_key,
            "action": self.action,
            "command": self.command,
            "cwd": self.cwd,
            "reason": self.reason,
            "approval_id": self.approval_id,
            "created_at": int(self.created_at),
            "updated_at": int(self.updated_at),
            "age_seconds": int(max(0, time.time() - self.created_at)),
            "status": self.status,
            "returncode": self.returncode,
            "error": self.error,
        }
        if include_output:
            payload["stdout"] = self.stdout
            payload["stderr"] = self.stderr
        return payload

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "session_key": self.session_key,
            "action": self.action,
            "command": self.command,
            "cwd": self.cwd,
            "reason": self.reason,
            "approval_id": self.approval_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


def start_background_command(
    *,
    user_id: str,
    session_key: str,
    action: str,
    command: str,
    cwd: str | None,
    reason: str = "",
    approval_id: str | None = None,
) -> BackgroundTask:
    _ensure_loaded()
    task = BackgroundTask(
        task_id=uuid.uuid4().hex[:10],
        user_id=str(user_id or ""),
        session_key=str(session_key or ""),
        action=str(action or "background_task"),
        command=str(command or ""),
        cwd=cwd or None,
        reason=str(reason or ""),
        approval_id=approval_id,
    )
    _TASKS[task.task_id] = task
    _save_tasks()
    task.runner = asyncio.create_task(_run_task(task))
    record_audit_event(
        event="background_started",
        user_id=task.user_id,
        session_key=task.session_key,
        action=task.action,
        payload={
            "task_id": task.task_id,
            "command": task.command,
            "cwd": task.cwd,
            "approval_id": task.approval_id,
        },
    )
    return task


def get_background_task(
    *,
    task_id: str,
    user_id: str,
    session_key: str,
) -> BackgroundTask | None:
    _ensure_loaded()
    task = _TASKS.get(str(task_id or "").strip())
    if task is None:
        return None
    if task.user_id != str(user_id or ""):
        return None
    if task.session_key != str(session_key or ""):
        return None
    return task


def list_background_tasks(
    *,
    user_id: str,
    session_key: str,
    include_finished: bool = True,
) -> list[BackgroundTask]:
    _ensure_loaded()
    tasks = [
        task
        for task in _TASKS.values()
        if task.user_id == str(user_id or "")
        and task.session_key == str(session_key or "")
    ]
    if not include_finished:
        tasks = [task for task in tasks if task.status == "running"]
    return sorted(tasks, key=lambda item: item.created_at, reverse=True)


async def cancel_background_task(
    *,
    task_id: str,
    user_id: str,
    session_key: str,
) -> BackgroundTask | None:
    task = get_background_task(
        task_id=task_id,
        user_id=user_id,
        session_key=session_key,
    )
    if task is None:
        return None
    if task.status != "running":
        return task
    task.status = "cancelling"
    task.updated_at = time.time()
    _save_tasks()
    process = task.process
    if process is not None and process.returncode is None:
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    if task.runner is not None and not task.runner.done():
        task.runner.cancel()
    task.status = "cancelled"
    task.updated_at = time.time()
    _save_tasks()
    record_audit_event(
        event="background_cancelled",
        user_id=task.user_id,
        session_key=task.session_key,
        action=task.action,
        payload={"task_id": task.task_id, "command": task.command},
    )
    return task


async def _run_task(task: BackgroundTask) -> None:
    try:
        process = await asyncio.create_subprocess_shell(
            task.command,
            cwd=task.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        task.process = process
        task.updated_at = time.time()
        _save_tasks()
        stdout, stderr = await process.communicate()
        if task.status == "cancelled":
            return
        task.returncode = process.returncode
        task.stdout = _decode(stdout)
        task.stderr = _decode(stderr)
        task.status = "completed" if process.returncode == 0 else "failed"
        task.updated_at = time.time()
        _save_tasks()
        record_audit_event(
            event="background_finished",
            user_id=task.user_id,
            session_key=task.session_key,
            action=task.action,
            payload={"task_id": task.task_id, "command": task.command, "cwd": task.cwd},
            result={"status": task.status, "returncode": process.returncode},
        )
    except asyncio.CancelledError:
        task.status = "cancelled"
        task.updated_at = time.time()
        _save_tasks()
        raise
    except Exception as exc:
        task.status = "error"
        task.error = str(exc)
        task.updated_at = time.time()
        _save_tasks()
        record_audit_event(
            event="background_failed",
            user_id=task.user_id,
            session_key=task.session_key,
            action=task.action,
            payload={"task_id": task.task_id, "command": task.command, "cwd": task.cwd},
            result={"error": str(exc)},
        )


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    raw = read_json(_TASKS_PATH, {})
    if not isinstance(raw, dict):
        return
    changed = False
    for task_id, payload in raw.items():
        task = _task_from_payload(task_id, payload)
        if task is None:
            continue
        if task.status in {"running", "cancelling"}:
            task.status = "interrupted_after_restart"
            task.error = task.error or "Bot restarted before this background task completed."
            task.updated_at = time.time()
            changed = True
        _TASKS[task.task_id] = task
    if changed:
        _save_tasks()


def _task_from_payload(task_id: object, payload: object) -> BackgroundTask | None:
    if not isinstance(payload, dict):
        return None
    data = dict(payload)
    data["task_id"] = str(data.get("task_id") or task_id or "")
    if not data["task_id"]:
        return None
    try:
        return BackgroundTask(
            task_id=str(data["task_id"]),
            user_id=str(data.get("user_id", "") or ""),
            session_key=str(data.get("session_key", "") or ""),
            action=str(data.get("action", "") or ""),
            command=str(data.get("command", "") or ""),
            cwd=str(data.get("cwd", "") or "") or None,
            reason=str(data.get("reason", "") or ""),
            approval_id=str(data.get("approval_id", "") or "") or None,
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(
                data.get("updated_at") or data.get("created_at") or time.time()
            ),
            status=str(data.get("status", "") or "unknown"),
            returncode=data.get("returncode"),
            stdout=str(data.get("stdout", "") or ""),
            stderr=str(data.get("stderr", "") or ""),
            error=str(data.get("error", "") or ""),
        )
    except Exception:
        return None


def _save_tasks() -> None:
    write_json(
        _TASKS_PATH,
        {
            task_id: task.to_record()
            for task_id, task in sorted(_TASKS.items())
        },
    )


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]


__all__ = [
    "BackgroundTask",
    "cancel_background_task",
    "get_background_task",
    "list_background_tasks",
    "start_background_command",
]
