import asyncio
from collections.abc import Awaitable, Callable
import contextlib
from dataclasses import dataclass, field
import time
from typing import TypeVar

from zhenxun.services.log import logger

_MODEL_CONCURRENCY_LIMIT = 4
_QUEUE_WAIT_LOG_THRESHOLD_MS = 80.0
_METRIC_LOG_INTERVAL = 60.0
_METRIC_LOG_STEP = 40

T = TypeVar("T")


@dataclass
class RuntimeMetrics:
    total_runs: int = 0
    completed_runs: int = 0
    cancelled_runs: int = 0
    cancel_requests: int = 0
    wait_events: int = 0
    total_wait_ms: float = 0.0
    max_wait_ms: float = 0.0
    model_runs: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, float | int | dict[str, int]]:
        avg_wait = self.total_wait_ms / self.total_runs if self.total_runs else 0.0
        return {
            "total_runs": self.total_runs,
            "completed_runs": self.completed_runs,
            "cancelled_runs": self.cancelled_runs,
            "cancel_requests": self.cancel_requests,
            "wait_events": self.wait_events,
            "avg_wait_ms": round(avg_wait, 2),
            "max_wait_ms": round(self.max_wait_ms, 2),
            "model_runs": dict(self.model_runs),
        }


class ChatInterRuntimeScheduler:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._model_semaphores: dict[str, asyncio.Semaphore] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._metrics = RuntimeMetrics()
        self._last_metrics_log = 0.0

    async def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        async with self._lock:
            lock = self._session_locks.get(session_key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_key] = lock
            return lock

    async def _get_model_semaphore(self, model_key: str) -> asyncio.Semaphore:
        async with self._lock:
            semaphore = self._model_semaphores.get(model_key)
            if semaphore is None:
                semaphore = asyncio.Semaphore(_MODEL_CONCURRENCY_LIMIT)
                self._model_semaphores[model_key] = semaphore
            return semaphore

    async def cancel_session(self, session_key: str) -> None:
        async with self._lock:
            task = self._active_tasks.get(session_key)
            self._metrics.cancel_requests += 1
        if not task or task.done():
            return
        current = asyncio.current_task()
        if task is current:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, timeout=0.25)

    async def _set_active(self, session_key: str, task: asyncio.Task | None) -> None:
        if task is None:
            return
        async with self._lock:
            self._active_tasks[session_key] = task

    async def _clear_active(self, session_key: str, task: asyncio.Task | None) -> None:
        async with self._lock:
            current = self._active_tasks.get(session_key)
            if current is task:
                self._active_tasks.pop(session_key, None)

    async def _record_wait(self, *, waited_ms: float, model_key: str) -> None:
        async with self._lock:
            self._metrics.total_runs += 1
            self._metrics.total_wait_ms += waited_ms
            self._metrics.max_wait_ms = max(self._metrics.max_wait_ms, waited_ms)
            self._metrics.model_runs[model_key] = (
                self._metrics.model_runs.get(model_key, 0) + 1
            )
            if waited_ms >= _QUEUE_WAIT_LOG_THRESHOLD_MS:
                self._metrics.wait_events += 1

    async def _record_completed(self) -> None:
        async with self._lock:
            self._metrics.completed_runs += 1

    async def _record_cancelled(self) -> None:
        async with self._lock:
            self._metrics.cancelled_runs += 1

    async def _maybe_log_metrics(self) -> None:
        now = time.monotonic()
        async with self._lock:
            total_runs = self._metrics.total_runs
            need_log = (
                total_runs % _METRIC_LOG_STEP == 0
                or now - self._last_metrics_log >= _METRIC_LOG_INTERVAL
            )
            if not need_log or total_runs == 0:
                return
            self._last_metrics_log = now
            snapshot = self._metrics.snapshot()
        logger.debug("chatinter scheduler metrics: %s", snapshot)

    async def get_metrics_snapshot(self) -> dict[str, float | int | dict[str, int]]:
        async with self._lock:
            return self._metrics.snapshot()

    async def run(
        self,
        *,
        session_key: str,
        model_key: str,
        runner: Callable[[], Awaitable[T]],
        interrupt_previous: bool = False,
    ) -> T:
        session_key = session_key or "global"
        model_key = model_key or "default"

        if interrupt_previous:
            await self.cancel_session(session_key)

        started = time.perf_counter()
        session_lock = await self._get_session_lock(session_key)
        model_semaphore = await self._get_model_semaphore(model_key)

        async with session_lock:
            async with model_semaphore:
                waited_ms = (time.perf_counter() - started) * 1000
                await self._record_wait(waited_ms=waited_ms, model_key=model_key)
                if waited_ms >= _QUEUE_WAIT_LOG_THRESHOLD_MS:
                    logger.debug(
                        "chatinter queue waited %.1fms session=%s model=%s",
                        waited_ms,
                        session_key,
                        model_key,
                    )
                current_task = asyncio.current_task()
                await self._set_active(session_key, current_task)
                try:
                    result = await runner()
                    await self._record_completed()
                    return result
                except asyncio.CancelledError:
                    await self._record_cancelled()
                    raise
                finally:
                    await self._clear_active(session_key, current_task)
                    await self._maybe_log_metrics()


_scheduler: ChatInterRuntimeScheduler | None = None


def get_runtime_scheduler() -> ChatInterRuntimeScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ChatInterRuntimeScheduler()
    return _scheduler


async def get_runtime_metrics_snapshot() -> dict[str, float | int | dict[str, int]]:
    return await get_runtime_scheduler().get_metrics_snapshot()
