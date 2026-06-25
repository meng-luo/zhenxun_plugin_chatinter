"""Progress ping helper for ChatInter agent runtime."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .route_text import normalize_message_text

_PROGRESS_TASKS: set[asyncio.Task[Any]] = set()
_PROGRESS_FIRST_DELAY = 15.0
_PROGRESS_MIN_INTERVAL = 25.0
_PROGRESS_SKIP_TOOLS = {
    "approve_pending_action",
    "list_pending_approvals",
    "reject_pending_action",
    "revoke_pending_approval",
    "tool_registry_status",
}


class AgentProgressReporter:
    """Small throttled progress reporter; no runtime policy lives here."""

    def __init__(self, hook: Any | None) -> None:
        self._hook = hook
        self._started_monotonic = 0.0
        self._last_sent = 0.0

    def start(self) -> None:
        self._started_monotonic = time.monotonic()
        self._last_sent = 0.0

    def emit(
        self,
        *,
        step: int,
        max_steps: int,
        observations: list[Any],
    ) -> None:
        hook = self._hook
        if hook is None:
            return
        now = time.monotonic()
        if now - self._started_monotonic < _PROGRESS_FIRST_DELAY:
            return
        if now - self._last_sent < _PROGRESS_MIN_INTERVAL:
            return
        self._last_sent = now
        last_tool = _last_tool_name(observations)
        if last_tool in _PROGRESS_SKIP_TOOLS:
            return
        phase = progress_phase(last_tool)
        text = f"⏳ 执行中 第{step}/{max_steps}步" + (
            f" · {phase}" if phase else ""
        )

        async def _safe_send() -> None:
            try:
                await hook(text)
            except Exception:
                pass

        task = asyncio.ensure_future(_safe_send())
        _PROGRESS_TASKS.add(task)
        task.add_done_callback(_PROGRESS_TASKS.discard)


def progress_phase(tool_name: str) -> str:
    if not tool_name:
        return ""
    if tool_name in {"read_file", "list_dir", "search_files", "artifact_read"}:
        return "正在读取文件"
    if tool_name.startswith(("patch_", "write_", "append_", "replace_")):
        return "正在应用修改"
    if tool_name.startswith(("engineering_eval_", "uv_", "python_")):
        return "正在跑验证"
    if tool_name.endswith("_command") or tool_name in {
        "shell_command",
        "git_command",
        "server_command",
    }:
        return "正在执行命令"
    return "正在处理任务"


def _last_tool_name(observations: list[Any]) -> str:
    for observation in reversed(observations):
        tool_name = normalize_message_text(str(getattr(observation, "tool_name", "")))
        if tool_name:
            return tool_name
    return ""
