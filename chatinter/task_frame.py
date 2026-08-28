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
TARGET_HINT_FIELD = "target_hint"
TARGET_REF_FIELD = "target_ref"
TARGET_REFS_FIELD = "target_refs"
TARGET_REF_SCHEMA_DESCRIPTION = (
    "可选的受限操作目标。仅填写 <turn_identity> 的 "
    "current_speaker_target_ref 或 <relevant_people> 中本轮已有的 target_ref；"
    "不得填写昵称或用户 ID、历史引用，否则填写 null。"
)
TARGET_REFS_SCHEMA_DESCRIPTION = (
    "可选的多个受限操作目标，按用户请求中的人物顺序填写 2 到 4 个本轮已有的 "
    "target_ref。不得填写昵称、用户 ID 或历史引用；单个目标请使用 target_ref。"
)
PAYLOAD_HINT_FIELD = "payload_hint"
_MAX_TASK_TEXT_LEN = 240
_MAX_HINT_LEN = 180


@dataclass(frozen=True)
class TaskFrame:
    """Isolated context for one native tool call."""

    task_index: int
    command_id: str
    plugin_module: str = ""
    task_text: str = ""
    fallback_text: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    target_hint: str = ""
    payload_hint: str = ""
    media_refs: tuple[str, ...] = ()
    target_refs: tuple[str, ...] = ()
    trusted_target_ids: tuple[str, ...] = ()
    ambient_message: str = ""

    @property
    def effective_text(self) -> str:
        """Only the LLM-provided task text is executable task context."""

        return self.task_text


def normalize_task_text(task_text: str) -> str:
    """Normalize only the task text explicitly provided by the tool call."""

    return normalize_message_text(task_text)


def pop_task_text(raw_slots: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Remove ChatInter-only task text from raw tool arguments."""

    task_text, _target_hint, _target_refs, _payload_hint, slots = pop_task_context(
        raw_slots
    )
    return task_text, slots


def pop_task_context(
    raw_slots: dict[str, Any],
) -> tuple[str, str, tuple[str, ...], str, dict[str, Any]]:
    """Remove ChatInter-only execution hints from raw tool arguments."""

    copied = dict(raw_slots or {})
    raw_task_text = copied.pop(TASK_TEXT_FIELD, None)
    raw_target_hint = copied.pop(TARGET_HINT_FIELD, None)
    raw_target_ref = copied.pop(TARGET_REF_FIELD, None)
    raw_target_refs = copied.pop(TARGET_REFS_FIELD, None)
    raw_payload_hint = copied.pop(PAYLOAD_HINT_FIELD, None)
    task_text = normalize_task_text(str(raw_task_text or ""))
    if len(task_text) > _MAX_TASK_TEXT_LEN:
        task_text = task_text[:_MAX_TASK_TEXT_LEN].rstrip()
    target_hint = normalize_task_text(str(raw_target_hint or ""))
    target_refs: list[str] = []
    for raw_ref in (
        raw_target_ref,
        *(raw_target_refs if isinstance(raw_target_refs, list | tuple) else ()),
    ):
        target_ref = normalize_task_text(str(raw_ref or ""))
        if len(target_ref) > _MAX_HINT_LEN:
            target_ref = target_ref[:_MAX_HINT_LEN].rstrip()
        if target_ref and target_ref.casefold() not in {
            item.casefold() for item in target_refs
        }:
            target_refs.append(target_ref)
        if len(target_refs) >= 4:
            break
    payload_hint = normalize_task_text(str(raw_payload_hint or ""))
    if len(target_hint) > _MAX_HINT_LEN:
        target_hint = target_hint[:_MAX_HINT_LEN].rstrip()
    if len(payload_hint) > _MAX_HINT_LEN:
        payload_hint = payload_hint[:_MAX_HINT_LEN].rstrip()
    return task_text, target_hint, tuple(target_refs), payload_hint, copied


__all__ = [
    "PAYLOAD_HINT_FIELD",
    "TARGET_HINT_FIELD",
    "TARGET_REFS_FIELD",
    "TARGET_REFS_SCHEMA_DESCRIPTION",
    "TARGET_REF_FIELD",
    "TARGET_REF_SCHEMA_DESCRIPTION",
    "TASK_TEXT_FIELD",
    "TaskFrame",
    "normalize_task_text",
    "pop_task_context",
    "pop_task_text",
]
