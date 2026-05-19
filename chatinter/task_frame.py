"""Per-tool task isolation for ChatInter native command execution.

A single user turn can contain several tool calls.  Each call must carry its own
task text so later command rendering does not accidentally consume another
task's natural-language tail from the full user message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .route_text import normalize_message_text

TASK_TEXT_FIELD = "task_text"
_MAX_TASK_TEXT_LEN = 240


@dataclass(frozen=True)
class TaskFrame:
    """Isolated context for one native tool call."""

    task_index: int
    command_id: str
    plugin_module: str = ""
    task_text: str = ""
    fallback_text: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    media_refs: tuple[str, ...] = ()
    target_refs: tuple[str, ...] = ()
    ambient_message: str = ""

    @property
    def effective_text(self) -> str:
        return self.task_text


def normalize_task_text(task_text: str) -> str:
    """Normalize only the task text explicitly provided by the tool call."""

    return normalize_message_text(task_text)


def pop_task_text(raw_slots: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Remove ChatInter-only task text from raw tool arguments."""

    copied = dict(raw_slots or {})
    raw_task_text = copied.pop(TASK_TEXT_FIELD, None)
    task_text = normalize_task_text(str(raw_task_text or ""))
    if len(task_text) > _MAX_TASK_TEXT_LEN:
        task_text = task_text[:_MAX_TASK_TEXT_LEN].rstrip()
    return task_text, copied


__all__ = ["TASK_TEXT_FIELD", "TaskFrame", "normalize_task_text", "pop_task_text"]
