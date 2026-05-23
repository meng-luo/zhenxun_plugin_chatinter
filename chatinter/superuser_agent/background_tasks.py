"""Durable background task registry for superuser Agent operations."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import time
from typing import Any
import uuid

from ..artifact_store import get_artifact_store, summarize_artifact_text
from ..persistence import read_json, state_path, write_json
from ..runtime_events import emit_runtime_event
from .audit_log import record_audit_event

_MAX_OUTPUT_CHARS = 8000
_STREAM_TAIL_CHARS = 4000
_EVENT_TAIL_CHARS = 1600
_EVENT_WAIT_TIMEOUT_SECONDS = 8.0
_STREAM_EVENT_MIN_INTERVAL_SECONDS = 1.2
_MAX_STREAM_EVENTS_PER_TASK = 80
_TASKS_PATH = state_path("background_tasks.json")
_EVENTS_PATH = state_path("background_observation_events.json")
_TASKS: dict[str, "BackgroundTask"] = {}
_OBSERVATION_EVENTS: dict[str, "ObservationEvent"] = {}
_EVENT_WATCHERS: dict[str, list[tuple[asyncio.Future["ObservationEvent"], bool]]] = {}
_LOADED = False
_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "error",
    "interrupted_after_restart",
}


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
    output_tail: str = ""
    stderr_tail: str = ""
    error: str = ""
    last_stream_event_at: float = 0.0
    stream_event_count: int = 0
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
            "stream_event_count": self.stream_event_count,
        }
        if include_output:
            payload["stdout"] = self.stdout
            payload["stderr"] = self.stderr
            payload["output_tail"] = self.output_tail
            payload["stderr_tail"] = self.stderr_tail
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
            "output_tail": self.output_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
            "last_stream_event_at": self.last_stream_event_at,
            "stream_event_count": self.stream_event_count,
        }


@dataclass(frozen=True)
class ObservationEvent:
    event_id: str
    task_id: str
    user_id: str
    session_key: str
    action: str
    kind: str
    status: str
    command: str
    cwd: str | None = None
    returncode: int | None = None
    output_tail: str = ""
    stderr_tail: str = ""
    error: str = ""
    artifacts: tuple[dict[str, Any], ...] = ()
    created_at: float = field(default_factory=time.time)

    def public_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = int(self.created_at)
        payload["artifacts"] = list(self.artifacts)
        return payload


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
    event = _emit_observation_event(task, kind="background_started")
    setattr(task, "last_observation_event", event)
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
    _emit_observation_event(task, kind="background_cancelled")
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
        await asyncio.gather(
            _stream_output(task, process.stdout, stdout=True),
            _stream_output(task, process.stderr, stdout=False),
        )
        await process.wait()
        if task.status == "cancelled":
            return
        task.returncode = process.returncode
        task.output_tail = _tail(task.stdout)
        task.stderr_tail = _tail(task.stderr)
        task.status = "completed" if process.returncode == 0 else "failed"
        task.updated_at = time.time()
        _save_tasks()
        _emit_observation_event(task, kind="background_finished")
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
        _emit_observation_event(task, kind="background_cancelled")
        raise
    except Exception as exc:
        task.status = "error"
        task.error = str(exc)
        task.updated_at = time.time()
        _save_tasks()
        _emit_observation_event(task, kind="background_error")
        record_audit_event(
            event="background_failed",
            user_id=task.user_id,
            session_key=task.session_key,
            action=task.action,
            payload={"task_id": task.task_id, "command": task.command, "cwd": task.cwd},
            result={"error": str(exc)},
        )


async def _stream_output(
    task: BackgroundTask,
    stream: asyncio.StreamReader | None,
    *,
    stdout: bool,
) -> None:
    if stream is None:
        return
    while True:
        data = await stream.read(4096)
        if not data:
            return
        text = _decode_chunk(data)
        if stdout:
            task.stdout = _append_limited(task.stdout, text)
            task.output_tail = _tail(task.stdout)
        else:
            task.stderr = _append_limited(task.stderr, text)
            task.stderr_tail = _tail(task.stderr)
        task.updated_at = time.time()
        _save_tasks()
        _maybe_emit_stream_event(task, stdout=stdout)


async def wait_for_observation_event(
    *,
    task_id: str,
    user_id: str,
    session_key: str,
    after_event_id: str = "",
    timeout: float = _EVENT_WAIT_TIMEOUT_SECONDS,
    terminal_only: bool = True,
) -> ObservationEvent | None:
    _ensure_loaded()
    immediate = get_latest_observation_event(
        task_id=task_id,
        user_id=user_id,
        session_key=session_key,
        after_event_id=after_event_id,
        terminal_only=terminal_only,
    )
    if immediate is not None:
        return immediate
    loop = asyncio.get_running_loop()
    future: asyncio.Future[ObservationEvent] = loop.create_future()
    key = _event_key(task_id, user_id, session_key)
    _EVENT_WATCHERS.setdefault(key, []).append((future, terminal_only))
    try:
        return await asyncio.wait_for(future, timeout=max(0.1, float(timeout or 0.1)))
    except asyncio.TimeoutError:
        return None
    finally:
        watchers = _EVENT_WATCHERS.get(key, [])
        for item in list(watchers):
            if item[0] is future:
                watchers.remove(item)
                break
        if not watchers:
            _EVENT_WATCHERS.pop(key, None)


def get_latest_observation_event(
    *,
    task_id: str,
    user_id: str,
    session_key: str,
    after_event_id: str = "",
    terminal_only: bool = True,
) -> ObservationEvent | None:
    _ensure_loaded()
    normalized_task = str(task_id or "").strip()
    normalized_user = str(user_id or "")
    normalized_session = str(session_key or "")
    after_seen = not bool(after_event_id)
    rows = sorted(_OBSERVATION_EVENTS.values(), key=lambda item: item.created_at)
    for event in rows:
        if event.event_id == after_event_id:
            after_seen = True
            continue
        if not after_seen:
            continue
        if event.task_id != normalized_task:
            continue
        if event.user_id != normalized_user or event.session_key != normalized_session:
            continue
        if terminal_only and event.status not in _TERMINAL_STATUSES:
            continue
        return event
    return None


def list_observation_events(
    *,
    task_id: str = "",
    user_id: str,
    session_key: str,
    after_event_id: str = "",
    limit: int = 20,
    terminal_only: bool = False,
) -> list[ObservationEvent]:
    """Return persisted ObservationEvents for a task/session.

    This is the public observation bus query surface used by superuser Agent
    tools.  Runtime waiting can use `wait_for_observation_event`; inspection
    and recovery should use this bounded listing instead of reading state files.
    """

    _ensure_loaded()
    normalized_task = str(task_id or "").strip()
    normalized_user = str(user_id or "")
    normalized_session = str(session_key or "")
    after_seen = not bool(after_event_id)
    result: list[ObservationEvent] = []
    for event in sorted(_OBSERVATION_EVENTS.values(), key=lambda item: item.created_at):
        if event.event_id == after_event_id:
            after_seen = True
            continue
        if not after_seen:
            continue
        if normalized_task and event.task_id != normalized_task:
            continue
        if event.user_id != normalized_user or event.session_key != normalized_session:
            continue
        if terminal_only and event.status not in _TERMINAL_STATUSES:
            continue
        result.append(event)
    return result[-max(1, min(int(limit or 20), 100)) :]


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
    raw_events = read_json(_EVENTS_PATH, {})
    if isinstance(raw_events, dict):
        for event_id, payload in raw_events.items():
            event = _event_from_payload(event_id, payload)
            if event is not None:
                _OBSERVATION_EVENTS[event.event_id] = event


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
            output_tail=str(
                data.get("output_tail", "")
                or _tail(str(data.get("stdout", "") or ""))
            ),
            stderr_tail=str(
                data.get("stderr_tail", "")
                or _tail(str(data.get("stderr", "") or ""))
            ),
            error=str(data.get("error", "") or ""),
            last_stream_event_at=float(data.get("last_stream_event_at") or 0.0),
            stream_event_count=int(data.get("stream_event_count", 0) or 0),
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


def _save_events() -> None:
    write_json(
        _EVENTS_PATH,
        {
            event_id: event.public_payload()
            for event_id, event in sorted(
                _OBSERVATION_EVENTS.items(),
                key=lambda item: item[1].created_at,
            )[-500:]
        },
    )


def _emit_observation_event(task: BackgroundTask, *, kind: str) -> ObservationEvent:
    _ensure_loaded()
    event = ObservationEvent(
        event_id=uuid.uuid4().hex[:12],
        task_id=task.task_id,
        user_id=task.user_id,
        session_key=task.session_key,
        action=task.action,
        kind=str(kind or "background_event"),
        status=task.status,
        command=task.command,
        cwd=task.cwd,
        returncode=task.returncode,
        output_tail=_tail(task.output_tail or task.stdout, max_chars=_EVENT_TAIL_CHARS),
        stderr_tail=_tail(task.stderr_tail or task.stderr, max_chars=_EVENT_TAIL_CHARS),
        error=task.error,
        artifacts=tuple(_store_task_artifacts(task)),
    )
    _OBSERVATION_EVENTS[event.event_id] = event
    _save_events()
    _emit_background_runtime_event(event)
    _notify_watchers(event)
    return event


def _maybe_emit_stream_event(task: BackgroundTask, *, stdout: bool) -> None:
    now = time.time()
    if task.stream_event_count >= _MAX_STREAM_EVENTS_PER_TASK:
        return
    if now - float(task.last_stream_event_at or 0.0) < _STREAM_EVENT_MIN_INTERVAL_SECONDS:
        return
    task.last_stream_event_at = now
    task.stream_event_count += 1
    _save_tasks()
    _emit_observation_event(
        task,
        kind="background_stdout" if stdout else "background_stderr",
    )


def _notify_watchers(event: ObservationEvent) -> None:
    key = _event_key(event.task_id, event.user_id, event.session_key)
    watchers = list(_EVENT_WATCHERS.get(key, []))
    remaining: list[tuple[asyncio.Future[ObservationEvent], bool]] = []
    for future, terminal_only in watchers:
        if terminal_only and event.status not in _TERMINAL_STATUSES:
            remaining.append((future, terminal_only))
            continue
        if not future.done():
            future.set_result(event)
    if remaining:
        _EVENT_WATCHERS[key] = remaining
    else:
        _EVENT_WATCHERS.pop(key, None)


def _event_key(task_id: str, user_id: str, session_key: str) -> str:
    return "|".join([str(task_id or ""), str(user_id or ""), str(session_key or "")])


def _store_task_artifacts(task: BackgroundTask) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    store = get_artifact_store()
    if task.stdout and len(task.stdout) > _EVENT_TAIL_CHARS:
        ref = store.store_text(
            task.stdout,
            artifact_type="log",
            trace_id=task.task_id,
            source=f"background_task:{task.task_id}:stdout",
            force_file=True,
        )
        if ref is not None:
            artifacts.append(ref.to_dict())
    if task.stderr and len(task.stderr) > _EVENT_TAIL_CHARS:
        ref = store.store_text(
            task.stderr,
            artifact_type="log",
            trace_id=task.task_id,
            source=f"background_task:{task.task_id}:stderr",
            force_file=True,
        )
        if ref is not None:
            artifacts.append(ref.to_dict())
    summary = {
        "task_id": task.task_id,
        "status": task.status,
        "returncode": task.returncode,
        "stdout_tail": summarize_artifact_text(task.output_tail, limit=360),
        "stderr_tail": summarize_artifact_text(task.stderr_tail, limit=360),
        "error": task.error,
    }
    ref = store.store_json(
        summary,
        artifact_type="plugin_output",
        trace_id=task.task_id,
        source=f"background_task:{task.task_id}:summary",
    )
    if ref is not None:
        artifacts.append(ref.to_dict())
    return artifacts


def _event_from_payload(
    event_id: object,
    payload: object,
) -> ObservationEvent | None:
    if not isinstance(payload, dict):
        return None
    data = dict(payload)
    data["event_id"] = str(data.get("event_id") or event_id or "")
    if not data["event_id"]:
        return None
    try:
        return ObservationEvent(
            event_id=str(data["event_id"]),
            task_id=str(data.get("task_id", "") or ""),
            user_id=str(data.get("user_id", "") or ""),
            session_key=str(data.get("session_key", "") or ""),
            action=str(data.get("action", "") or ""),
            kind=str(data.get("kind", "") or "background_event"),
            status=str(data.get("status", "") or ""),
            command=str(data.get("command", "") or ""),
            cwd=str(data.get("cwd", "") or "") or None,
            returncode=data.get("returncode"),
            output_tail=str(data.get("output_tail", "") or ""),
            stderr_tail=str(data.get("stderr_tail", "") or ""),
            error=str(data.get("error", "") or ""),
            artifacts=tuple(
                dict(item)
                for item in data.get("artifacts", []) or []
                if isinstance(item, dict)
            ),
            created_at=float(data.get("created_at") or time.time()),
        )
    except Exception:
        return None


def _emit_background_runtime_event(event: ObservationEvent) -> None:
    emit_runtime_event(
        kind="background_job",
        status=_runtime_status_from_background(event.status),
        source=f"background:{event.kind}",
        session_key=event.session_key,
        user_id=event.user_id,
        summary=f"{event.kind}:{event.task_id}:{event.status}",
        payload=event.public_payload(),
        artifacts=list(event.artifacts),
        related_ids={
            "background_task_id": event.task_id,
            "observation_event_id": event.event_id,
        },
    )


def _runtime_status_from_background(status: str) -> str:
    normalized = str(status or "")
    if normalized == "running":
        return "progress"
    if normalized == "completed":
        return "completed"
    if normalized in {"cancelled", "interrupted_after_restart"}:
        return "cancelled"
    if normalized in {"failed", "error"}:
        return "failed"
    return "info"


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]


def _decode_chunk(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


def _append_limited(old: str, new: str, *, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    text = f"{old or ''}{new or ''}"
    return text[-max(1, max_chars) :]


def _tail(value: str, *, max_chars: int = _STREAM_TAIL_CHARS) -> str:
    text = str(value or "")
    return text[-max(1, max_chars) :]


__all__ = [
    "BackgroundTask",
    "ObservationEvent",
    "cancel_background_task",
    "get_background_task",
    "get_latest_observation_event",
    "list_background_tasks",
    "list_observation_events",
    "start_background_command",
    "wait_for_observation_event",
]
