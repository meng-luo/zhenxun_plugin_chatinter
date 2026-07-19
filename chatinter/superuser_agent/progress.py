"""Progress ping helper for ChatInter agent runtime."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..route_text import normalize_message_text

_PROGRESS_FIRST_DELAY = 15.0
_PROGRESS_UPDATE_INTERVAL = 30.0
_SHELL_HEARTBEAT_INTERVAL = 90.0


class AgentProgressReporter:
    """Small throttled progress reporter; no runtime policy lives here."""

    def __init__(self, hook: Any | None) -> None:
        self._hook = hook
        self._started_monotonic = 0.0
        self._current_phase = ""
        self._last_sent_phase: str | None = None
        self._last_sent_monotonic = 0.0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._shell_heartbeat_task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        self._started_monotonic = time.monotonic()
        self._current_phase = ""
        self._last_sent_phase = None
        self._last_sent_monotonic = 0.0
        if self._hook is not None:
            self._schedule(self._send_initial())

    async def stop(self) -> None:
        self._shell_heartbeat_task = None
        tasks = tuple(self._tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def emit(
        self,
        *,
        observations: list[Any],
    ) -> None:
        self.tool_started(_last_tool_name(observations))

    def tool_started(self, tool_name: str) -> None:
        hook = self._hook
        if hook is None:
            return
        phase = progress_phase(tool_name)
        self._current_phase = phase
        if tool_name == "shell_command" and (
            self._shell_heartbeat_task is None
            or self._shell_heartbeat_task.done()
        ):
            self._shell_heartbeat_task = self._schedule(self._shell_heartbeat())
        now = time.monotonic()
        if now - self._started_monotonic < _PROGRESS_FIRST_DELAY:
            return
        if self._last_sent_phase == phase:
            return
        if now - self._last_sent_monotonic < _PROGRESS_UPDATE_INTERVAL:
            return
        self._last_sent_phase = phase
        self._last_sent_monotonic = now

        self._schedule(self._safe_send(_progress_text(phase)))

    async def tool_finished(self, tool_name: str) -> None:
        if tool_name != "shell_command" or self._shell_heartbeat_task is None:
            return
        task = self._shell_heartbeat_task
        self._shell_heartbeat_task = None
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _schedule(self, awaitable: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _send_initial(self) -> None:
        await asyncio.sleep(_PROGRESS_FIRST_DELAY)
        if self._last_sent_phase is not None:
            return
        self._last_sent_phase = self._current_phase
        self._last_sent_monotonic = time.monotonic()
        await self._safe_send(_progress_text(self._current_phase))

    async def _shell_heartbeat(self) -> None:
        while True:
            await asyncio.sleep(_SHELL_HEARTBEAT_INTERVAL)
            self._last_sent_monotonic = time.monotonic()
            await self._safe_send(_progress_text("正在执行命令"))

    async def _safe_send(self, text: str) -> None:
        hook = self._hook
        if hook is None:
            return
        try:
            await hook(text)
        except Exception:
            pass


def progress_phase(tool_name: str) -> str:
    if not tool_name:
        return ""
    if tool_name in {"read_file", "list_dir", "search_files", "artifact_read"}:
        return "正在读取文件"
    if tool_name in {"write_file", "replace_in_file"}:
        return "正在应用修改"
    if tool_name == "shell_command":
        return "正在执行命令"
    return "正在处理任务"


def _progress_text(phase: str) -> str:
    return "正在执行" + (f" · {phase}" if phase else "")


def _last_tool_name(observations: list[Any]) -> str:
    for observation in reversed(observations):
        tool_name = normalize_message_text(str(getattr(observation, "tool_name", "")))
        if tool_name:
            return tool_name
    return ""
