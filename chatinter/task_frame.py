"""Per-tool task isolation for ChatInter native command execution.

A single user turn can contain several tool calls.  Each call must carry its own
task text so later command rendering does not accidentally consume another
task's natural-language tail from the full user message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .route_text import normalize_message_text, strip_invoke_prefix

TASK_TEXT_FIELD = "task_text"
_MAX_TASK_TEXT_LEN = 240
_TRAILING_TASK_PATTERN = re.compile(
    r"(?:，|,|。|；|;|\s)+(?:然后|接着|再|最后|顺便|并且|以及|还有|同时)"
    r".*$",
    re.DOTALL,
)
_TASK_SPLIT_PATTERN = re.compile(
    r"(?:，|,|。|；|;|\s)+(?:然后|接着|再|最后|顺便|并且|以及|还有|同时)\s*"
)
_LEADING_TASK_CONNECTOR_PATTERN = re.compile(
    r"^(?:然后|接着|再|最后|顺便|并且|以及|还有|同时)+"
)


def _strip_task_leading_words(text: str) -> str:
    stripped = normalize_message_text(text)
    while stripped:
        next_text = normalize_message_text(
            strip_invoke_prefix(_LEADING_TASK_CONNECTOR_PATTERN.sub("", stripped))
        )
        if next_text == stripped:
            return stripped
        stripped = next_text
    return ""


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
        return self.task_text or self.fallback_text


def isolate_task_text(task_text: str, *, command_text: str = "") -> str:
    """Keep a tool call from swallowing later tasks in the same user turn."""

    normalized = normalize_message_text(task_text)
    if not normalized:
        return ""
    normalized = _strip_task_leading_words(normalized)
    if not normalized:
        return ""
    command = normalize_message_text(command_text)
    if command:
        for part in _TASK_SPLIT_PATTERN.split(normalized):
            candidate = _strip_task_leading_words(part)
            if command in candidate:
                return candidate
        # Without an explicit per-tool fragment, falling back to the command head is
        # safer than giving a later renderer the whole multi-task user turn.
        return command
    isolated = normalize_message_text(
        _strip_task_leading_words(_TRAILING_TASK_PATTERN.sub("", normalized))
    )
    if isolated:
        return isolated

    return normalized


def pop_task_text(raw_slots: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Remove ChatInter-only task text from raw tool arguments."""

    copied = dict(raw_slots or {})
    raw_task_text = copied.pop(TASK_TEXT_FIELD, None)
    task_text = isolate_task_text(str(raw_task_text or ""))
    if len(task_text) > _MAX_TASK_TEXT_LEN:
        task_text = task_text[:_MAX_TASK_TEXT_LEN].rstrip()
    return task_text, copied


__all__ = ["TASK_TEXT_FIELD", "TaskFrame", "isolate_task_text", "pop_task_text"]
