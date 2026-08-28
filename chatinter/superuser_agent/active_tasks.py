"""Persistent active-task records and SchedulerManager integration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import inspect
from pathlib import Path
import re
import threading
import time
from typing import Any, Literal, cast
import uuid

from pydantic import BaseModel

from ..persistence import read_json_strict, state_path, utc_now_iso, write_json

ActiveTaskKind = Literal["agent", "script", "notify"]
ActiveTaskTrigger = Literal["date", "cron", "interval", "webhook"]
ActiveTaskCallback = Callable[
    ["ActiveTask", Any | None, Any | None], Awaitable[None] | None
]

_STORE_VERSION = 2
_SCHEDULER_PLUGIN_NAME = "chatinter_active_tasks"
_SCHEDULER_SOURCE = "CHATINTER_ACTIVE_TASK"
_TASK_KINDS = frozenset({"agent", "script", "notify"})
_TRIGGER_TYPES = frozenset({"date", "cron", "interval", "webhook"})
_CONTROL_ACTIONS = frozenset({"pause", "resume", "delete", "run_now"})
_MAX_NAME_LENGTH = 160
_MAX_INSTRUCTION_LENGTH = 20_000
_MAX_IDENTIFIER_LENGTH = 255
_MAX_ARGS = 64
_MAX_ARG_LENGTH = 1_024
_MAX_ARGS_TOTAL_LENGTH = 8_192
_MAX_ERROR_LENGTH = 4_000
_MAX_PATH_LENGTH = 4_096
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_STATUS_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_INCOMPLETE_RUN_STATUSES = frozenset({"claimed", "queued", "started"})
_INCOMPLETE_DELIVERY_STATUSES = frozenset({"executed"})
_DATE_MISFIRE_GRACE_SECONDS = 300
_WEBHOOK_EVENT_RECEIPT_TTL_SECONDS = 10 * 60.0
_WEBHOOK_EVENT_RECEIPT_LIMIT = 256
_CRON_FIELDS = frozenset(
    {
        "year",
        "month",
        "day",
        "week",
        "day_of_week",
        "hour",
        "minute",
        "second",
    }
)
_COMMON_TRIGGER_FIELDS = frozenset(
    {"start_date", "end_date", "timezone", "jitter"}
)
_INTERVAL_FIELDS = frozenset(
    {"weeks", "days", "hours", "minutes", "seconds"}
)

_SINGLETON_LOCK = threading.RLock()
_SCHEDULE_LOCK = asyncio.Lock()
_DEFAULT_STORE: ActiveTaskStore | None = None
_DEFAULT_ADAPTER: ActiveTaskSchedulerAdapter | None = None
_DISPATCH_STORE: ActiveTaskStore | None = None
_DISPATCH_CALLBACK: ActiveTaskCallback | None = None


class _ActiveTaskScheduleParams(BaseModel):
    task_id: str


@dataclass(frozen=True, slots=True)
class ScriptIdentity:
    entrypoint: str
    cwd: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ActiveTask:
    task_id: str
    owner: str
    session_key: str
    user_id: str
    bot_id: str
    conversation_id: str
    name: str
    kind: ActiveTaskKind
    instruction: str
    entrypoint: str | None
    cwd: str | None
    args: tuple[str, ...]
    entrypoint_sha256: str | None
    trigger_type: ActiveTaskTrigger
    trigger_config: dict[str, Any]
    schedule_id: int | None
    webhook_token_hash: str | None
    webhook_event_receipts: dict[str, float]
    enabled: bool
    created_at: str
    updated_at: str
    last_run_at: str | None
    last_status: str
    last_error: str
    run_count: int
    last_execution_status: str
    last_delivery_status: str
    allow_network: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "owner": self.owner,
            "session_key": self.session_key,
            "user_id": self.user_id,
            "bot_id": self.bot_id,
            "conversation_id": self.conversation_id,
            "name": self.name,
            "kind": self.kind,
            "instruction": self.instruction,
            "entrypoint": self.entrypoint,
            "cwd": self.cwd,
            "args": list(self.args),
            "entrypoint_sha256": self.entrypoint_sha256,
            "trigger_type": self.trigger_type,
            "trigger_config": dict(self.trigger_config),
            "schedule_id": self.schedule_id,
            "webhook_token_hash": self.webhook_token_hash,
            "webhook_event_receipts": dict(self.webhook_event_receipts),
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "run_count": self.run_count,
            "last_execution_status": self.last_execution_status,
            "last_delivery_status": self.last_delivery_status,
            "allow_network": self.allow_network,
        }


class ActiveTaskStore:
    """Thread-safe JSON storage with owner-isolated public operations."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else state_path("active_tasks.json")
        self._lock = threading.RLock()

    def create(
        self,
        *,
        session_key: str,
        user_id: str,
        bot_id: str,
        conversation_id: str,
        name: str,
        kind: str,
        instruction: str,
        trigger_type: str,
        trigger_config: Mapping[str, Any] | None = None,
        entrypoint: str | Path | None = None,
        cwd: str | Path | None = None,
        args: Sequence[str] | None = None,
        expected_entrypoint_sha256: str | None = None,
        webhook_token_hash: str | None = None,
        allow_network: bool | None = None,
        enabled: bool = True,
    ) -> ActiveTask:
        owner = _normalize_identifier(session_key, "session_key")
        normalized_kind = _normalize_choice(kind, _TASK_KINDS, "kind")
        normalized_trigger = _normalize_choice(
            trigger_type, _TRIGGER_TYPES, "trigger_type"
        )
        normalized_instruction = _normalize_instruction(instruction)
        if normalized_kind in {"agent", "notify"} and not normalized_instruction:
            raise ValueError(f"{normalized_kind} task requires instruction")
        normalized_args = _normalize_args(args)
        script_identity = _prepare_script_identity(
            normalized_kind,
            entrypoint=entrypoint,
            cwd=cwd,
            args=normalized_args,
            expected_sha256=expected_entrypoint_sha256,
        )
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        if allow_network is None:
            allow_network = normalized_kind == "agent"
        if not isinstance(allow_network, bool):
            raise TypeError("allow_network must be a boolean")
        if normalized_kind != "agent" and allow_network:
            raise ValueError("allow_network is only valid for agent tasks")
        now = utc_now_iso()
        config_value = {} if trigger_config is None else dict(trigger_config)
        if normalized_trigger == "interval" and "start_date" not in config_value:
            config_value["start_date"] = now
        task = ActiveTask(
            task_id=uuid.uuid4().hex,
            owner=owner,
            session_key=owner,
            user_id=_normalize_identifier(user_id, "user_id"),
            bot_id=_normalize_identifier(bot_id, "bot_id"),
            conversation_id=_normalize_identifier(
                conversation_id, "conversation_id"
            ),
            name=_normalize_name(name),
            kind=cast(ActiveTaskKind, normalized_kind),
            instruction=normalized_instruction,
            entrypoint=script_identity.entrypoint if script_identity else None,
            cwd=script_identity.cwd if script_identity else None,
            args=normalized_args,
            entrypoint_sha256=script_identity.sha256 if script_identity else None,
            trigger_type=cast(ActiveTaskTrigger, normalized_trigger),
            trigger_config=_normalize_trigger_config(
                normalized_trigger, config_value
            ),
            schedule_id=None,
            webhook_token_hash=_normalize_webhook_hash(
                normalized_trigger, webhook_token_hash
            ),
            webhook_event_receipts={},
            enabled=enabled,
            created_at=now,
            updated_at=now,
            last_run_at=None,
            last_status="pending",
            last_error="",
            run_count=0,
            last_execution_status="pending",
            last_delivery_status="pending",
            allow_network=allow_network,
        )
        if _date_task_is_expired(task):
            raise ValueError("date trigger is outside the delivery window")
        with self._lock:
            raw_tasks = self._read_raw()
            if task.task_id in raw_tasks:
                raise RuntimeError("active task UUID collision")
            raw_tasks[task.task_id] = task.to_dict()
            self._write_raw(raw_tasks)
        return task

    def list(self, session_key: str) -> list[ActiveTask]:
        owner = _normalize_identifier(session_key, "session_key")
        with self._lock:
            tasks = [
                task
                for task in self._read_valid().values()
                if _task_is_owned_by(task, owner)
            ]
        return sorted(tasks, key=lambda task: (task.created_at, task.task_id))

    def get(
        self,
        task_id: str,
        session_key: str | None = None,
    ) -> ActiveTask | None:
        normalized_id = _normalize_task_id(task_id)
        owner = (
            _normalize_identifier(session_key, "session_key")
            if session_key is not None
            else None
        )
        with self._lock:
            task = self._read_valid().get(normalized_id)
        if task is None or (owner is not None and not _task_is_owned_by(task, owner)):
            return None
        return task

    def delete(self, task_id: str, session_key: str) -> ActiveTask | None:
        owner = _normalize_identifier(session_key, "session_key")
        normalized_id = _normalize_task_id(task_id)
        with self._lock:
            raw_tasks = self._read_raw()
            task = _task_from_payload(raw_tasks.get(normalized_id))
            if task is None or not _task_is_owned_by(task, owner):
                return None
            raw_tasks.pop(normalized_id, None)
            self._write_raw(raw_tasks)
            return task

    def update_status(
        self,
        task_id: str,
        *,
        session_key: str,
        status: str,
        error: str = "",
        ran_at: str | None = None,
        increment_run_count: bool = True,
        execution_status: str | None = None,
        delivery_status: str | None = None,
        touch_run_at: bool = True,
    ) -> ActiveTask | None:
        if not isinstance(increment_run_count, bool):
            raise TypeError("increment_run_count must be a boolean")
        owner = _normalize_identifier(session_key, "session_key")
        with self._lock:
            task = self._owned(task_id, owner)
            if task is None:
                return None
            updated = replace(
                task,
                last_run_at=(
                    _normalize_timestamp(ran_at or utc_now_iso(), "ran_at")
                    if touch_run_at
                    else task.last_run_at
                ),
                last_status=_normalize_status(status),
                last_error=_clip_error(error),
                run_count=task.run_count + (1 if increment_run_count else 0),
                last_execution_status=(
                    task.last_execution_status
                    if execution_status is None
                    else _normalize_status(execution_status)
                ),
                last_delivery_status=(
                    task.last_delivery_status
                    if delivery_status is None
                    else _normalize_status(delivery_status)
                ),
                updated_at=utc_now_iso(),
            )
            self._put(updated)
            return updated

    def set_enabled(
        self,
        task_id: str,
        session_key: str,
        enabled: bool,
    ) -> ActiveTask:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        return self._replace_owned(task_id, session_key, enabled=enabled)

    def set_schedule_id(
        self,
        task_id: str,
        session_key: str,
        schedule_id: int | None,
    ) -> ActiveTask:
        normalized_id = _normalize_schedule_id(schedule_id)
        return self._replace_owned(task_id, session_key, schedule_id=normalized_id)

    def set_webhook_token_hash(
        self,
        task_id: str,
        session_key: str,
        token_hash: str,
    ) -> ActiveTask:
        normalized = _normalize_webhook_hash("webhook", token_hash)
        current = self.get(task_id, session_key)
        if current is None:
            raise KeyError("active task not found")
        if current.trigger_type != "webhook":
            raise ValueError("webhook credential is only valid for webhook tasks")
        return self._replace_owned(
            task_id,
            session_key,
            webhook_token_hash=normalized,
            webhook_event_receipts={},
        )

    def record_schedule_error(
        self,
        task_id: str,
        session_key: str,
        error: Exception | str,
    ) -> ActiveTask | None:
        owner = _normalize_identifier(session_key, "session_key")
        with self._lock:
            task = self._owned(task_id, owner)
            if task is None:
                return None
            updated = replace(
                task,
                last_status="schedule_error",
                last_error=_clip_error(str(error) or type(error).__name__),
                updated_at=utc_now_iso(),
            )
            self._put(updated)
            return updated

    def claim_scheduled(self, task_id: str) -> ActiveTask | None:
        with self._lock:
            task = self._read_valid().get(_normalize_task_id(task_id))
            if task is None or not task.enabled:
                return None
            if task.trigger_type == "date":
                claimed_at = utc_now_iso()
                self._put(
                    replace(
                        task,
                        enabled=False,
                        schedule_id=None,
                        last_run_at=claimed_at,
                        last_status="claimed",
                        last_error="",
                        updated_at=claimed_at,
                    )
                )
            return task

    def all_tasks(self) -> list[ActiveTask]:
        with self._lock:
            return list(self._read_valid().values())

    def webhook_event_seen(
        self,
        task_id: str,
        event_digest: str,
        *,
        now: float | None = None,
    ) -> bool:
        digest = _normalize_optional_sha256(event_digest, "event_digest")
        if digest is None:
            return False
        with self._lock:
            task = self._read_valid().get(_normalize_task_id(task_id))
            if task is None or task.trigger_type != "webhook":
                return False
            receipts = _current_webhook_receipts(
                task.webhook_event_receipts,
                now=time.time() if now is None else float(now),
            )
            return digest in receipts

    def reserve_webhook_event(
        self,
        task_id: str,
        event_digest: str,
        *,
        now: float | None = None,
    ) -> bool:
        digest = _normalize_optional_sha256(event_digest, "event_digest")
        if digest is None:
            raise ValueError("event_digest is required")
        timestamp = time.time() if now is None else float(now)
        with self._lock:
            task = self._read_valid().get(_normalize_task_id(task_id))
            if (
                task is None
                or task.trigger_type != "webhook"
                or not task.enabled
            ):
                return False
            receipts = _current_webhook_receipts(
                task.webhook_event_receipts,
                now=timestamp,
            )
            if digest in receipts:
                return False
            receipts[digest] = timestamp
            receipts = dict(
                sorted(
                    receipts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:_WEBHOOK_EVENT_RECEIPT_LIMIT]
            )
            self._put(
                replace(
                    task,
                    webhook_event_receipts=receipts,
                    last_status="queued",
                    last_error="webhook",
                    updated_at=utc_now_iso(),
                )
            )
            return True

    def _replace_owned(
        self,
        task_id: str,
        session_key: str,
        **changes: Any,
    ) -> ActiveTask:
        owner = _normalize_identifier(session_key, "session_key")
        with self._lock:
            task = self._owned(task_id, owner)
            if task is None:
                raise KeyError("active task not found")
            updated = replace(task, updated_at=utc_now_iso(), **changes)
            self._put(updated)
            return updated

    def _owned(self, task_id: str, owner: str) -> ActiveTask | None:
        task = self._read_valid().get(_normalize_task_id(task_id))
        if task is None or not _task_is_owned_by(task, owner):
            return None
        return task

    def _put(self, task: ActiveTask) -> None:
        raw_tasks = self._read_raw()
        raw_tasks[task.task_id] = task.to_dict()
        self._write_raw(raw_tasks)

    def replace_definition(
        self,
        task: ActiveTask,
        session_key: str,
    ) -> ActiveTask:
        owner = _normalize_identifier(session_key, "session_key")
        with self._lock:
            current = self._owned(task.task_id, owner)
            if current is None or not _task_is_owned_by(task, owner):
                raise KeyError("active task not found")
            self._put(task)
            return task

    def _read_valid(self) -> dict[str, ActiveTask]:
        tasks: dict[str, ActiveTask] = {}
        for task_id, payload in self._read_raw().items():
            task = _task_from_payload(payload)
            if task is None or task.task_id != task_id:
                raise ValueError(f"invalid active task record: {task_id}")
            tasks[task_id] = task
        return tasks

    def _read_raw(self) -> dict[str, Any]:
        payload = read_json_strict(self.path, {})
        if not isinstance(payload, dict):
            raise ValueError("active task store must be a JSON object")
        raw_tasks = payload.get("tasks")
        if isinstance(raw_tasks, list):
            return {
                str(item.get("task_id")): item
                for item in raw_tasks
                if isinstance(item, dict) and item.get("task_id")
            }
        if isinstance(raw_tasks, dict):
            return {str(key): value for key, value in raw_tasks.items()}
        if payload and all(isinstance(value, dict) for value in payload.values()):
            return {str(key): value for key, value in payload.items()}
        if not payload:
            return {}
        raise ValueError("active task store has an unsupported schema")

    def _write_raw(self, tasks: Mapping[str, Any]) -> None:
        write_json(
            self.path,
            {
                "version": _STORE_VERSION,
                "tasks": {task_id: tasks[task_id] for task_id in sorted(tasks)},
            },
        )


class ActiveTaskSchedulerAdapter:
    """Narrow adapter over the existing persistent SchedulerManager."""

    def __init__(self, manager: Any, callback: ActiveTaskCallback):
        if manager is None:
            raise TypeError("manager is required")
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.manager = manager
        self.callback = callback

    def register_dispatch_bridge(self, store: ActiveTaskStore) -> None:
        register_active_task_dispatch_callback(self.callback, store=store)
        if _SCHEDULER_PLUGIN_NAME in self.manager.get_registered_plugins():
            return
        self.manager.register(
            _SCHEDULER_PLUGIN_NAME,
            params_model=_ActiveTaskScheduleParams,
            default_permission=9,
        )(_scheduled_active_task_dispatch)

    async def schedule(self, task: ActiveTask) -> int:
        if task.trigger_type == "webhook":
            raise ValueError("webhook tasks do not use SchedulerManager")
        self.register_dispatch_bridge(_DISPATCH_STORE or get_active_task_store())
        from zhenxun.services.scheduler import Trigger
        from zhenxun.services.scheduler.types import ExecutionOptions, JobConfig

        trigger = getattr(Trigger, task.trigger_type)(**task.trigger_config)
        schedule = await self.manager.add_schedule(
            plugin_name=_SCHEDULER_PLUGIN_NAME,
            target_type="GLOBAL",
            target_identifier=task.task_id,
            config=JobConfig(
                trigger=trigger,
                job_kwargs={"task_id": task.task_id},
                bot_id=task.bot_id,
                name=task.name,
                created_by=task.owner,
                required_permission=9,
                source=_SCHEDULER_SOURCE,
                is_one_off=task.trigger_type == "date",
                execution_options=ExecutionOptions(concurrency_policy="SKIP"),
            ),
        )
        if schedule is None or not isinstance(schedule.id, int):
            raise RuntimeError(
                "SchedulerManager did not create the active task schedule"
            )
        return schedule.id

    async def pause(self, task: ActiveTask) -> None:
        if task.trigger_type == "webhook":
            return
        await self._target(task).pause()

    async def resume(self, task: ActiveTask) -> int:
        return await self.schedule(task)

    async def delete(self, task: ActiveTask) -> None:
        if task.trigger_type == "webhook" and task.schedule_id is None:
            return
        if task.schedule_id is not None:
            schedule = await self.manager.get_schedule_by_id(task.schedule_id)
            if schedule is not None and _schedule_matches_task(schedule, task):
                await self.manager.target(id=task.schedule_id).remove()
        await self._target(task).remove()

    async def run_now(self, task: ActiveTask) -> int | None:
        if not task.enabled:
            raise RuntimeError("cannot run a paused active task")
        if task.trigger_type == "webhook":
            await _invoke_callback(self.callback, task, None, None)
            return None
        schedule_id = await self.schedule(task)
        success, message = await self.manager.trigger_now(schedule_id)
        if not success:
            raise RuntimeError(message)
        return schedule_id

    async def remove_orphaned_schedules(self, valid_task_ids: set[str]) -> int:
        schedules, _total = await self.manager.get_schedules(
            plugin_name=_SCHEDULER_PLUGIN_NAME
        )
        removed = 0
        for schedule in schedules:
            target_identifier = str(
                getattr(schedule, "target_identifier", "") or ""
            )
            if target_identifier in valid_task_ids:
                continue
            schedule_id = getattr(schedule, "id", None)
            if not isinstance(schedule_id, int):
                continue
            count, _message = await self.manager.target(id=schedule_id).remove()
            removed += int(count or 0)
        return removed

    def _target(self, task: ActiveTask) -> Any:
        return self.manager.target(
            plugin_name=_SCHEDULER_PLUGIN_NAME,
            target_type="GLOBAL",
            target_identifier=task.task_id,
        )


def get_active_task_store() -> ActiveTaskStore:
    """Return the process-wide active-task JSON store."""

    global _DEFAULT_STORE
    with _SINGLETON_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = ActiveTaskStore()
        return _DEFAULT_STORE


def register_active_task_dispatch_callback(
    callback: ActiveTaskCallback,
    *,
    store: ActiveTaskStore | None = None,
) -> None:
    """Register the callback and store used by the fixed scheduler bridge."""

    if not callable(callback):
        raise TypeError("callback must be callable")
    global _DISPATCH_CALLBACK, _DISPATCH_STORE
    _DISPATCH_CALLBACK = callback
    _DISPATCH_STORE = store or get_active_task_store()


async def initialize_active_task_schedules(
    callback: ActiveTaskCallback,
    *,
    manager: Any | None = None,
    store: ActiveTaskStore | None = None,
) -> tuple[int, int]:
    """Register the fixed bridge and reconcile persisted scheduler rows."""

    active_store = store or get_active_task_store()
    if manager is None:
        from zhenxun.services.scheduler import scheduler_manager

        manager = scheduler_manager
    adapter = ActiveTaskSchedulerAdapter(manager, callback)
    registered = 0
    failed = 0
    async with _SCHEDULE_LOCK:
        adapter.register_dispatch_bridge(active_store)
        global _DEFAULT_ADAPTER
        _DEFAULT_ADAPTER = adapter
        tasks = active_store.all_tasks()
        try:
            await adapter.remove_orphaned_schedules(
                {task.task_id for task in tasks}
            )
        except Exception:
            failed += 1
        for task in tasks:
            try:
                if task.last_status in _INCOMPLETE_RUN_STATUSES:
                    recovered = active_store.update_status(
                        task.task_id,
                        session_key=task.owner,
                        status="uncertain",
                        error="execution interrupted by process restart",
                        ran_at=task.last_run_at or utc_now_iso(),
                        increment_run_count=False,
                        execution_status="uncertain",
                        delivery_status="not_started",
                    )
                    if recovered is not None and recovered.enabled:
                        recovered = active_store.set_enabled(
                            recovered.task_id,
                            recovered.owner,
                            False,
                        )
                    task = recovered or task
                elif task.last_status in _INCOMPLETE_DELIVERY_STATUSES:
                    task = active_store.update_status(
                        task.task_id,
                        session_key=task.owner,
                        status="delivery_uncertain",
                        error="delivery interrupted by process restart",
                        ran_at=task.last_run_at or utc_now_iso(),
                        increment_run_count=False,
                        delivery_status="uncertain",
                    ) or task
                if task.trigger_type == "date" and _date_task_is_expired(task):
                    task = active_store.update_status(
                        task.task_id,
                        session_key=task.owner,
                        status="missed",
                        error="scheduled time passed outside the delivery window",
                        increment_run_count=False,
                        execution_status="not_started",
                        delivery_status="not_started",
                        touch_run_at=False,
                    ) or task
                    if task.enabled:
                        task = active_store.set_enabled(
                            task.task_id,
                            task.owner,
                            False,
                        )
                if task.trigger_type == "webhook":
                    if task.schedule_id is not None:
                        await adapter.delete(task)
                        active_store.set_schedule_id(task.task_id, task.owner, None)
                elif task.enabled:
                    schedule_id = await adapter.schedule(task)
                    active_store.set_schedule_id(
                        task.task_id, task.owner, schedule_id
                    )
                    registered += 1
                else:
                    await adapter.pause(task)
            except Exception as exc:
                failed += 1
                active_store.record_schedule_error(task.task_id, task.owner, exc)
    return registered, failed


async def register_scheduled_tasks(
    callback: ActiveTaskCallback,
) -> tuple[int, int]:
    """Compatibility entry point for active-task startup registration."""

    return await initialize_active_task_schedules(callback)


async def create_active_task(**kwargs: Any) -> ActiveTask:
    """Create a record and attach its persistent schedule when required."""

    store = get_active_task_store()
    task = store.create(**kwargs)
    if not task.enabled or task.trigger_type == "webhook":
        return task
    adapter = _require_default_adapter()
    async with _SCHEDULE_LOCK:
        try:
            schedule_id = await adapter.schedule(task)
            return store.set_schedule_id(task.task_id, task.owner, schedule_id)
        except Exception:
            try:
                await adapter.delete(task)
            finally:
                store.delete(task.task_id, task.owner)
            raise


def list_active_tasks(session_key: str) -> list[ActiveTask]:
    return get_active_task_store().list(session_key)


def get_active_task(
    task_id: str,
    session_key: str | None = None,
) -> ActiveTask | None:
    return get_active_task_store().get(task_id, session_key)


def update_active_task_status(
    task_id: str,
    *,
    session_key: str,
    status: str,
    error: str = "",
    ran_at: str | None = None,
    increment_run_count: bool = True,
    execution_status: str | None = None,
    delivery_status: str | None = None,
    touch_run_at: bool = True,
) -> ActiveTask | None:
    return get_active_task_store().update_status(
        task_id,
        session_key=session_key,
        status=status,
        error=error,
        ran_at=ran_at,
        increment_run_count=increment_run_count,
        execution_status=execution_status,
        delivery_status=delivery_status,
        touch_run_at=touch_run_at,
    )


async def ensure_active_task_schedule(
    task_id: str,
    session_key: str,
) -> ActiveTask:
    store = get_active_task_store()
    task = _require_owned_task(store, task_id, session_key)
    if task.trigger_type == "webhook" or not task.enabled:
        return task
    async with _SCHEDULE_LOCK:
        schedule_id = await _require_default_adapter().schedule(task)
        return store.set_schedule_id(task.task_id, task.owner, schedule_id)


async def unregister_active_task_schedule(
    task_id: str,
    session_key: str,
) -> ActiveTask:
    store = get_active_task_store()
    task = _require_owned_task(store, task_id, session_key)
    async with _SCHEDULE_LOCK:
        if task.trigger_type != "webhook" or task.schedule_id is not None:
            await _require_default_adapter().delete(task)
        return store.set_schedule_id(task.task_id, task.owner, None)


async def control_active_task(
    task_id: str,
    action: str,
    session_key: str,
) -> ActiveTask | None:
    store = get_active_task_store()
    task = _require_owned_task(store, task_id, session_key)
    normalized_action = _normalize_choice(action, _CONTROL_ACTIONS, "action")
    if normalized_action == "run_now":
        schedule_id = await _require_default_adapter().run_now(task)
        if schedule_id is not None:
            store.set_schedule_id(task.task_id, task.owner, schedule_id)
        return store.get(task.task_id, task.owner)

    async with _SCHEDULE_LOCK:
        if normalized_action == "pause":
            task = store.set_enabled(task.task_id, task.owner, False)
            if task.trigger_type != "webhook":
                await _require_default_adapter().pause(task)
            return store.get(task.task_id, task.owner)
        if normalized_action == "resume":
            if task.trigger_type == "date" and _date_task_is_expired(
                replace(task, enabled=True)
            ):
                store.update_status(
                    task.task_id,
                    session_key=task.owner,
                    status="missed",
                    error="scheduled time passed outside the delivery window",
                    increment_run_count=False,
                    execution_status="not_started",
                    delivery_status="not_started",
                    touch_run_at=False,
                )
                raise RuntimeError("date task is outside the delivery window")
            task = store.set_enabled(task.task_id, task.owner, True)
            if task.trigger_type == "webhook":
                return task
            try:
                schedule_id = await _require_default_adapter().resume(task)
                return store.set_schedule_id(task.task_id, task.owner, schedule_id)
            except Exception:
                store.set_enabled(task.task_id, task.owner, False)
                raise
        deleted = store.delete(task.task_id, task.owner)
        if deleted is None:
            raise KeyError("active task not found")
        if deleted.trigger_type != "webhook" or deleted.schedule_id is not None:
            await _require_default_adapter().delete(deleted)
        return None


async def pause_active_tasks_for_conversation(
    session_key: str,
    conversation_id: str,
) -> int:
    target = str(conversation_id or "").strip()
    if not target:
        return 0
    tasks = [
        task
        for task in list_active_tasks(session_key)
        if task.conversation_id == target and task.enabled
    ]
    paused = 0
    for task in tasks:
        await control_active_task(task.task_id, "pause", session_key)
        update_active_task_status(
            task.task_id,
            session_key=session_key,
            status="paused",
            error="bound conversation archived",
            increment_run_count=False,
            touch_run_at=False,
        )
        paused += 1
    return paused


async def delete_active_tasks_for_conversation(
    session_key: str,
    conversation_id: str,
) -> int:
    target = str(conversation_id or "").strip()
    if not target:
        return 0
    tasks = [
        task
        for task in list_active_tasks(session_key)
        if task.conversation_id == target
    ]
    deleted = 0
    for task in tasks:
        await control_active_task(task.task_id, "delete", session_key)
        deleted += 1
    return deleted


async def update_active_task(
    task_id: str,
    session_key: str,
    changes: Mapping[str, Any],
) -> ActiveTask:
    store = get_active_task_store()
    original = _require_owned_task(store, task_id, session_key)
    updated = _updated_task_definition(original, changes)
    adapter = _require_default_adapter()
    async with _SCHEDULE_LOCK:
        removed_old_schedule = False
        try:
            if original.trigger_type != "webhook" and original.schedule_id is not None:
                await adapter.delete(original)
                removed_old_schedule = True
            updated = store.replace_definition(updated, session_key)
            if updated.enabled and updated.trigger_type != "webhook":
                schedule_id = await adapter.schedule(updated)
                updated = store.set_schedule_id(
                    updated.task_id,
                    updated.owner,
                    schedule_id,
                )
            return updated
        except Exception:
            try:
                if updated.trigger_type != "webhook":
                    await adapter.delete(updated)
            except Exception:
                pass
            try:
                restored_definition = (
                    replace(original, schedule_id=None)
                    if removed_old_schedule
                    else original
                )
                restored = store.replace_definition(
                    restored_definition,
                    session_key,
                )
                if removed_old_schedule and restored.enabled:
                    schedule_id = await adapter.schedule(restored)
                    store.set_schedule_id(
                        restored.task_id,
                        restored.owner,
                        schedule_id,
                    )
            except Exception:
                pass
            raise


def rotate_active_task_webhook_token(
    task_id: str,
    session_key: str,
    token_hash: str,
) -> ActiveTask:
    return get_active_task_store().set_webhook_token_hash(
        task_id,
        session_key,
        token_hash,
    )


def _updated_task_definition(
    task: ActiveTask,
    changes: Mapping[str, Any],
) -> ActiveTask:
    allowed = {
        "name",
        "instruction",
        "trigger_type",
        "trigger_config",
        "entrypoint",
        "cwd",
        "args",
        "expected_entrypoint_sha256",
        "allow_network",
    }
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"unsupported active task changes: {sorted(unknown)}")
    if not changes:
        raise ValueError("active task update requires at least one change")

    name = _normalize_name(changes.get("name", task.name))
    instruction = _normalize_instruction(
        changes.get("instruction", task.instruction)
    )
    if task.kind in {"agent", "notify"} and not instruction:
        raise ValueError(f"{task.kind} task requires instruction")
    allow_network = changes.get("allow_network", task.allow_network)
    if not isinstance(allow_network, bool):
        raise TypeError("allow_network must be a boolean")
    if task.kind != "agent" and allow_network:
        raise ValueError("allow_network is only valid for agent tasks")

    trigger_type = _normalize_choice(
        changes.get("trigger_type", task.trigger_type),
        _TRIGGER_TYPES,
        "trigger_type",
    )
    if (trigger_type == "webhook") != (task.trigger_type == "webhook"):
        raise ValueError(
            "changing between webhook and scheduled triggers is unsupported"
        )
    if "trigger_type" in changes and "trigger_config" not in changes:
        raise ValueError("changing trigger_type requires trigger_config")
    trigger_config_value = changes.get("trigger_config", task.trigger_config)
    if trigger_type == "interval" and isinstance(trigger_config_value, Mapping):
        trigger_config_value = dict(trigger_config_value)
        trigger_config_value.setdefault("start_date", utc_now_iso())
    trigger_config = _normalize_trigger_config(
        trigger_type,
        trigger_config_value,
    )

    args = _normalize_args(changes.get("args", task.args))
    script_identity = _prepare_script_identity(
        task.kind,
        entrypoint=changes.get("entrypoint", task.entrypoint),
        cwd=changes.get("cwd", task.cwd),
        args=args,
        expected_sha256=changes.get(
            "expected_entrypoint_sha256",
            task.entrypoint_sha256,
        ),
    )
    updated = replace(
        task,
        name=name,
        instruction=instruction,
        trigger_type=cast(ActiveTaskTrigger, trigger_type),
        trigger_config=trigger_config,
        entrypoint=script_identity.entrypoint if script_identity else None,
        cwd=script_identity.cwd if script_identity else None,
        args=args,
        entrypoint_sha256=script_identity.sha256 if script_identity else None,
        allow_network=allow_network,
        schedule_id=None if trigger_type != "webhook" else task.schedule_id,
        updated_at=utc_now_iso(),
    )
    if _date_task_is_expired(updated):
        raise ValueError("date trigger is outside the delivery window")
    return updated


def build_script_identity(
    entrypoint: str | Path,
    cwd: str | Path | None = None,
) -> ScriptIdentity:
    """Resolve a Python entrypoint and bind it to its current content hash."""

    resolved_entrypoint, resolved_cwd = _resolve_script_paths(
        entrypoint, cwd or Path.cwd()
    )
    return ScriptIdentity(
        entrypoint=str(resolved_entrypoint),
        cwd=str(resolved_cwd),
        sha256=_sha256_file(resolved_entrypoint),
    )


def verify_script_identity(identity: ScriptIdentity | ActiveTask) -> bool:
    """Reject a moved, escaped, missing, or modified Python entrypoint."""

    if isinstance(identity, ActiveTask):
        if identity.kind != "script":
            raise ValueError("script identity is only available for script tasks")
        if (
            not identity.entrypoint
            or not identity.cwd
            or not identity.entrypoint_sha256
        ):
            raise ValueError("script task has incomplete identity")
        script_identity = ScriptIdentity(
            entrypoint=identity.entrypoint,
            cwd=identity.cwd,
            sha256=identity.entrypoint_sha256,
        )
    elif isinstance(identity, ScriptIdentity):
        script_identity = identity
    else:
        raise TypeError("identity must be ScriptIdentity or ActiveTask")
    entrypoint, _ = _resolve_script_paths(
        script_identity.entrypoint, script_identity.cwd
    )
    if not hmac.compare_digest(_sha256_file(entrypoint), script_identity.sha256):
        raise ValueError("script entrypoint content has changed")
    return True


def verify_task_entrypoint(task: ActiveTask) -> bool:
    return verify_script_identity(task)


def normalize_active_task_trigger(
    trigger_type: str,
    trigger_config: Mapping[str, Any] | None,
) -> tuple[ActiveTaskTrigger, dict[str, Any]]:
    """Validate and normalize a model-supplied trigger without persisting it."""

    normalized_type = _normalize_choice(
        trigger_type,
        _TRIGGER_TYPES,
        "trigger_type",
    )
    return (
        cast(ActiveTaskTrigger, normalized_type),
        _normalize_trigger_config(normalized_type, trigger_config or {}),
    )


async def _scheduled_active_task_dispatch(
    bot: Any,
    context: Any,
    params: _ActiveTaskScheduleParams,
) -> None:
    callback = _DISPATCH_CALLBACK
    store = _DISPATCH_STORE
    if callback is None or store is None:
        raise RuntimeError("active task dispatch is not initialized")
    task = store.claim_scheduled(params.task_id)
    if task is not None:
        await _invoke_callback(callback, task, bot, context)


async def _invoke_callback(
    callback: ActiveTaskCallback,
    task: ActiveTask,
    bot: Any | None,
    context: Any | None,
) -> None:
    result = callback(task, bot, context)
    if inspect.isawaitable(result):
        await result


def _require_default_adapter() -> ActiveTaskSchedulerAdapter:
    if _DEFAULT_ADAPTER is None:
        raise RuntimeError("active task scheduler has not been initialized")
    return _DEFAULT_ADAPTER


def _require_owned_task(
    store: ActiveTaskStore,
    task_id: str,
    session_key: str,
) -> ActiveTask:
    task = store.get(task_id, session_key)
    if task is None:
        raise KeyError("active task not found")
    return task


def _schedule_matches_task(schedule: Any, task: ActiveTask) -> bool:
    return (
        str(getattr(schedule, "plugin_name", "")) == _SCHEDULER_PLUGIN_NAME
        and str(getattr(schedule, "target_type", "")) == "GLOBAL"
        and str(getattr(schedule, "target_identifier", "")) == task.task_id
    )


def _task_is_owned_by(task: ActiveTask, owner: str) -> bool:
    return hmac.compare_digest(task.owner, owner)


def _task_from_payload(payload: Any) -> ActiveTask | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        task_id = _normalize_task_id(payload.get("task_id"))
        session_key = _normalize_identifier(payload.get("session_key"), "session_key")
        owner = _normalize_identifier(payload.get("owner") or session_key, "owner")
        if not hmac.compare_digest(owner, session_key):
            raise ValueError("owner and session_key must match")
        kind = _normalize_choice(payload.get("kind"), _TASK_KINDS, "kind")
        trigger_type = _normalize_choice(
            payload.get("trigger_type"), _TRIGGER_TYPES, "trigger_type"
        )
        args = _normalize_args(payload.get("args"))
        entrypoint = _normalize_optional_path(payload.get("entrypoint"), "entrypoint")
        cwd = _normalize_optional_path(payload.get("cwd"), "cwd")
        entrypoint_sha256 = _normalize_optional_sha256(
            payload.get("entrypoint_sha256"), "entrypoint_sha256"
        )
        if kind == "script":
            if not entrypoint or not cwd or not entrypoint_sha256:
                raise ValueError("script task has incomplete identity")
        elif entrypoint or cwd or args or entrypoint_sha256:
            raise ValueError("non-script task contains script fields")
        instruction = _normalize_instruction(payload.get("instruction", ""))
        if kind in {"agent", "notify"} and not instruction:
            raise ValueError(f"{kind} task requires instruction")
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        run_count = payload.get("run_count", 0)
        if isinstance(run_count, bool) or not isinstance(run_count, int):
            raise TypeError("run_count must be an integer")
        if run_count < 0:
            raise ValueError("run_count cannot be negative")
        last_status = str(payload.get("last_status", "") or "").strip().lower()
        if last_status:
            last_status = _normalize_status(last_status)
        created_at = _normalize_timestamp(payload.get("created_at"), "created_at")
        trigger_config_value = payload.get("trigger_config")
        if trigger_type == "interval" and isinstance(
            trigger_config_value,
            Mapping,
        ):
            trigger_config_value = dict(trigger_config_value)
            trigger_config_value.setdefault("start_date", created_at)
        allow_network = payload.get("allow_network", kind == "agent")
        if not isinstance(allow_network, bool):
            raise TypeError("allow_network must be a boolean")
        if kind != "agent" and allow_network:
            raise ValueError("allow_network is only valid for agent tasks")
        execution_status = str(
            payload.get("last_execution_status", "") or ""
        ).strip().lower()
        delivery_status = str(
            payload.get("last_delivery_status", "") or ""
        ).strip().lower()
        return ActiveTask(
            task_id=task_id,
            owner=owner,
            session_key=session_key,
            user_id=_normalize_identifier(payload.get("user_id"), "user_id"),
            bot_id=_normalize_identifier(payload.get("bot_id"), "bot_id"),
            conversation_id=_normalize_identifier(
                payload.get("conversation_id"), "conversation_id"
            ),
            name=_normalize_name(payload.get("name")),
            kind=cast(ActiveTaskKind, kind),
            instruction=instruction,
            entrypoint=entrypoint,
            cwd=cwd,
            args=args,
            entrypoint_sha256=entrypoint_sha256,
            trigger_type=cast(ActiveTaskTrigger, trigger_type),
            trigger_config=_normalize_trigger_config(
                trigger_type, trigger_config_value
            ),
            schedule_id=_normalize_schedule_id(payload.get("schedule_id")),
            webhook_token_hash=_normalize_webhook_hash(
                trigger_type, payload.get("webhook_token_hash")
            ),
            webhook_event_receipts=_normalize_webhook_event_receipts(
                trigger_type,
                payload.get("webhook_event_receipts"),
            ),
            enabled=enabled,
            created_at=created_at,
            updated_at=_normalize_timestamp(payload.get("updated_at"), "updated_at"),
            last_run_at=_normalize_optional_timestamp(payload.get("last_run_at")),
            last_status=last_status,
            last_error=_clip_error(payload.get("last_error", "")),
            run_count=run_count,
            last_execution_status=(
                _normalize_status(execution_status)
                if execution_status
                else _legacy_execution_status(last_status)
            ),
            last_delivery_status=(
                _normalize_status(delivery_status)
                if delivery_status
                else _legacy_delivery_status(last_status)
            ),
            allow_network=allow_network,
        )
    except (TypeError, ValueError):
        return None


def _normalize_trigger_config(
    trigger_type: str,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("trigger_config must be an object")
    config = dict(value)
    if any(not isinstance(key, str) for key in config):
        raise TypeError("trigger_config keys must be strings")
    if trigger_type == "webhook":
        if config:
            raise ValueError("webhook trigger_config must be empty")
        return {}
    if trigger_type == "date":
        normalized = _normalize_date_trigger(config)
    elif trigger_type == "cron":
        normalized = _normalize_cron_trigger(config)
    elif trigger_type == "interval":
        normalized = _normalize_interval_trigger(config)
    else:
        raise ValueError(f"unsupported trigger_type: {trigger_type}")
    _validate_scheduler_trigger(trigger_type, normalized)
    return normalized


def _normalize_date_trigger(config: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_keys(config, {"run_date", "timezone"})
    if "run_date" not in config:
        raise ValueError("date trigger requires run_date")
    normalized = {"run_date": _normalize_datetime_value(config["run_date"])}
    if "timezone" in config:
        normalized["timezone"] = _normalize_timezone(config["timezone"])
    return normalized


def _normalize_cron_trigger(config: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_keys(config, _CRON_FIELDS | _COMMON_TRIGGER_FIELDS)
    if not _CRON_FIELDS.intersection(config):
        raise ValueError("cron trigger requires at least one cron field")
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if key in _CRON_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int | str):
                raise TypeError(f"cron field {key} must be an integer or string")
            if isinstance(value, str):
                value = value.strip()
                if not value or len(value) > 128:
                    raise ValueError(f"invalid cron field {key}")
            normalized[key] = value
        elif key in {"start_date", "end_date"}:
            normalized[key] = _normalize_datetime_value(value)
        elif key == "timezone":
            normalized[key] = _normalize_timezone(value)
        else:
            normalized[key] = _normalize_nonnegative_int(value, key)
    return normalized


def _normalize_interval_trigger(config: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_keys(config, _INTERVAL_FIELDS | _COMMON_TRIGGER_FIELDS)
    normalized: dict[str, Any] = {}
    total = 0
    for key, value in config.items():
        if key in _INTERVAL_FIELDS:
            interval = _normalize_nonnegative_int(value, key)
            normalized[key] = interval
            total += interval
        elif key in {"start_date", "end_date"}:
            normalized[key] = _normalize_datetime_value(value)
        elif key == "timezone":
            normalized[key] = _normalize_timezone(value)
        else:
            normalized[key] = _normalize_nonnegative_int(value, key)
    if total <= 0:
        raise ValueError("interval trigger requires a positive interval")
    return normalized


def active_task_next_run_time(task: ActiveTask) -> str | None:
    if not task.enabled or task.trigger_type == "webhook":
        return None
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    trigger_class = {
        "date": DateTrigger,
        "cron": CronTrigger,
        "interval": IntervalTrigger,
    }[task.trigger_type]
    trigger = trigger_class(**task.trigger_config)
    now = datetime.now(timezone.utc)
    next_run = trigger.get_next_fire_time(None, now)
    return next_run.isoformat() if next_run is not None else None


def _date_task_is_expired(task: ActiveTask) -> bool:
    if task.trigger_type != "date" or not task.enabled:
        return False
    from apscheduler.triggers.date import DateTrigger

    run_date = DateTrigger(**task.trigger_config).run_date
    return run_date + timedelta(seconds=_DATE_MISFIRE_GRACE_SECONDS) < datetime.now(
        timezone.utc
    )


def _legacy_execution_status(last_status: str) -> str:
    if last_status == "success":
        return "succeeded"
    if last_status in {"failed", "timed_out"}:
        return last_status
    if last_status == "started":
        return "running"
    return "pending"


def _legacy_delivery_status(last_status: str) -> str:
    if last_status == "success":
        return "delivered"
    return "pending"


def _validate_scheduler_trigger(trigger_type: str, config: dict[str, Any]) -> None:
    if trigger_type == "date":
        from apscheduler.triggers.date import DateTrigger

        DateTrigger(**config)
    elif trigger_type == "cron":
        from apscheduler.triggers.cron import CronTrigger

        CronTrigger(**config)
    elif trigger_type == "interval":
        from apscheduler.triggers.interval import IntervalTrigger

        IntervalTrigger(**config)


def _reject_unknown_keys(
    config: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
) -> None:
    unknown = sorted(set(config).difference(allowed))
    if unknown:
        raise ValueError(f"unknown trigger_config fields: {', '.join(unknown)}")


def _prepare_script_identity(
    kind: str,
    *,
    entrypoint: str | Path | None,
    cwd: str | Path | None,
    args: tuple[str, ...],
    expected_sha256: str | None = None,
) -> ScriptIdentity | None:
    if kind != "script":
        if entrypoint is not None or cwd is not None or args or expected_sha256:
            raise ValueError("entrypoint, cwd and args are only valid for script tasks")
        return None
    if entrypoint is None:
        raise ValueError("script task requires entrypoint")
    identity = build_script_identity(entrypoint, cwd)
    expected = _normalize_optional_sha256(
        expected_sha256,
        "expected_entrypoint_sha256",
    )
    if expected is not None and not hmac.compare_digest(identity.sha256, expected):
        raise ValueError("script entrypoint content changed after approval")
    return identity


def _resolve_script_paths(
    entrypoint: str | Path,
    cwd: str | Path,
) -> tuple[Path, Path]:
    cwd_path = Path(cwd).expanduser().resolve(strict=True)
    if not cwd_path.is_dir():
        raise ValueError("script cwd must be a directory")
    entrypoint_path = Path(entrypoint).expanduser()
    if not entrypoint_path.is_absolute():
        entrypoint_path = cwd_path / entrypoint_path
    entrypoint_path = entrypoint_path.resolve(strict=True)
    if not entrypoint_path.is_file():
        raise ValueError("script entrypoint must be a file")
    if entrypoint_path.suffix.casefold() != ".py":
        raise ValueError("script entrypoint must be a .py file")
    try:
        entrypoint_path.relative_to(cwd_path)
    except ValueError as exc:
        raise ValueError("script entrypoint must be inside cwd") from exc
    return entrypoint_path, cwd_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(128 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_args(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise TypeError("args must be an array of strings")
    if len(value) > _MAX_ARGS:
        raise ValueError(f"args cannot contain more than {_MAX_ARGS} items")
    normalized: list[str] = []
    total = 0
    for item in value:
        if not isinstance(item, str):
            raise TypeError("args must contain only strings")
        if "\x00" in item:
            raise ValueError("args cannot contain NUL bytes")
        if len(item) > _MAX_ARG_LENGTH:
            raise ValueError(f"each arg is limited to {_MAX_ARG_LENGTH} characters")
        total += len(item)
        if total > _MAX_ARGS_TOTAL_LENGTH:
            raise ValueError("combined args are too long")
        normalized.append(item)
    return tuple(normalized)


def _normalize_name(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("name must be a string")
    name = " ".join(value.split())
    if not name:
        raise ValueError("name cannot be empty")
    if len(name) > _MAX_NAME_LENGTH:
        raise ValueError(f"name is limited to {_MAX_NAME_LENGTH} characters")
    return name


def _normalize_instruction(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("instruction must be a string")
    instruction = value.strip()
    if "\x00" in instruction:
        raise ValueError("instruction cannot contain NUL bytes")
    if len(instruction) > _MAX_INSTRUCTION_LENGTH:
        raise ValueError(
            f"instruction is limited to {_MAX_INSTRUCTION_LENGTH} characters"
        )
    return instruction


def _normalize_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str | int) or isinstance(value, bool):
        raise TypeError(f"{field} must be a string or integer")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    if len(normalized) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{field} is too long")
    if any(ord(char) < 32 for char in normalized):
        raise ValueError(f"{field} contains control characters")
    return normalized


def _normalize_task_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("task_id must be a string")
    task_id = value.strip().lower()
    if not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError("task_id must be a UUID hex string")
    return task_id


def _normalize_choice(value: Any, choices: frozenset[str], field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"unsupported {field}: {value}")
    return normalized


def _normalize_webhook_hash(trigger_type: str, value: Any) -> str | None:
    if trigger_type != "webhook":
        if value is not None and value != "":
            raise ValueError("webhook_token_hash is only valid for webhook tasks")
        return None
    normalized = _normalize_optional_sha256(value, "webhook_token_hash")
    if normalized is None:
        raise ValueError("webhook task requires webhook_token_hash")
    return normalized


def _normalize_webhook_event_receipts(
    trigger_type: str,
    value: Any,
) -> dict[str, float]:
    if trigger_type != "webhook":
        if value not in (None, {}, []):
            raise ValueError(
                "webhook_event_receipts are only valid for webhook tasks"
            )
        return {}
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("webhook_event_receipts must be an object")
    receipts: dict[str, float] = {}
    for raw_digest, raw_timestamp in value.items():
        digest = _normalize_optional_sha256(raw_digest, "webhook event digest")
        if digest is None or isinstance(raw_timestamp, bool):
            raise ValueError("invalid webhook event receipt")
        timestamp = float(raw_timestamp)
        if timestamp <= 0 or timestamp != timestamp or timestamp == float("inf"):
            raise ValueError("invalid webhook event receipt timestamp")
        receipts[digest] = timestamp
    return dict(
        sorted(receipts.items(), key=lambda item: item[1], reverse=True)[
            :_WEBHOOK_EVENT_RECEIPT_LIMIT
        ]
    )


def _current_webhook_receipts(
    value: Mapping[str, float],
    *,
    now: float,
) -> dict[str, float]:
    cutoff = now - _WEBHOOK_EVENT_RECEIPT_TTL_SECONDS
    return {
        digest: timestamp
        for digest, timestamp in value.items()
        if timestamp > cutoff
    }


def _normalize_optional_sha256(value: Any, field: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return normalized


def _normalize_optional_path(value: Any, field: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized or "\x00" in normalized or len(normalized) > _MAX_PATH_LENGTH:
        raise ValueError(f"invalid {field}")
    if not Path(normalized).is_absolute():
        raise ValueError(f"stored {field} must be absolute")
    return normalized


def _normalize_schedule_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("schedule_id must be a positive integer")
    return value


def _normalize_status(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("status must be a string")
    normalized = value.strip().lower()
    if not _STATUS_RE.fullmatch(normalized):
        raise ValueError("invalid active task status")
    return normalized


def _normalize_timestamp(value: Any, field: str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO datetime string")
    normalized = value.strip()
    if not normalized or len(normalized) > 64:
        raise ValueError(f"invalid {field}")
    try:
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {field}") from exc
    return normalized


def _normalize_optional_timestamp(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return _normalize_timestamp(value, "last_run_at")


def _normalize_datetime_value(value: Any) -> str:
    return _normalize_timestamp(value, "trigger datetime")


def _normalize_timezone(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("timezone must be a string")
    timezone = value.strip()
    if not timezone or len(timezone) > 128:
        raise ValueError("invalid timezone")
    return timezone


def _normalize_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} cannot be negative")
    return value


def _clip_error(value: Any) -> str:
    error = str(value or "").strip()
    if len(error) <= _MAX_ERROR_LENGTH:
        return error
    return error[: _MAX_ERROR_LENGTH - 3] + "..."


__all__ = [
    "ActiveTask",
    "ActiveTaskCallback",
    "ActiveTaskSchedulerAdapter",
    "ActiveTaskStore",
    "ScriptIdentity",
    "active_task_next_run_time",
    "build_script_identity",
    "control_active_task",
    "create_active_task",
    "delete_active_tasks_for_conversation",
    "ensure_active_task_schedule",
    "get_active_task",
    "get_active_task_store",
    "initialize_active_task_schedules",
    "list_active_tasks",
    "normalize_active_task_trigger",
    "pause_active_tasks_for_conversation",
    "register_active_task_dispatch_callback",
    "register_scheduled_tasks",
    "rotate_active_task_webhook_token",
    "unregister_active_task_schedule",
    "update_active_task",
    "update_active_task_status",
    "verify_script_identity",
    "verify_task_entrypoint",
]
