from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from typing import Any

from .route_text import normalize_message_text

_SNAPSHOT_MARKER = "<agent_turn_state>"


def build_agent_turn_state_snapshot(state: Any) -> tuple[str, str]:
    payload = {
        "run_id": str(getattr(state, "run_id", "") or ""),
        "status": str(getattr(state, "status", "") or ""),
        "step": int(getattr(state, "step", 0) or 0),
        "agent_complexity_mode": str(getattr(state, "agent_complexity_mode", "") or ""),
        "tool_obligation": str(getattr(state, "tool_obligation", "") or ""),
        "pending_tasks": [
            _public_item(item) for item in _items(state, "pending_tasks")
        ],
        "completed_tasks": [
            _public_item(item) for item in _items(state, "completed_tasks")
        ],
        "observations": [_public_item(item) for item in _items(state, "observations")],
        "waiting_approval_ids": list(_items(state, "waiting_approval_ids")),
        "background_task_ids": list(_items(state, "background_task_ids")),
        "observation_event_ids": list(_items(state, "observation_event_ids")),
    }
    text = "\n".join(
        [
            _SNAPSHOT_MARKER,
            "rule=Use this snapshot as current agent turn state.",
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
            "</agent_turn_state>",
        ]
    )
    fingerprint = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return fingerprint, text


def append_agent_turn_state_snapshot(
    state: Any,
    *,
    message_factory: Callable[[str], Any],
) -> bool:
    fingerprint, text = build_agent_turn_state_snapshot(state)
    if fingerprint == getattr(state, "turn_state_snapshot_fingerprint", ""):
        return False

    messages = getattr(state, "messages", None)
    if messages is None:
        messages = []
        setattr(state, "messages", messages)

    rendered = message_factory(text)
    snapshot_indexes = [
        index
        for index, message in enumerate(messages)
        if _SNAPSHOT_MARKER in str(message)
    ]
    if snapshot_indexes:
        messages[snapshot_indexes[-1]] = rendered
        for index in reversed(snapshot_indexes[:-1]):
            del messages[index]
    else:
        messages.append(rendered)

    setattr(state, "turn_state_snapshot_fingerprint", fingerprint)
    append_timeline = getattr(state, "append_timeline", None)
    if callable(append_timeline):
        append_timeline(kind="agent_turn_state_snapshot", fingerprint=fingerprint)
    return True


def _items(state: Any, name: str) -> list[Any]:
    value = getattr(state, name, None)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _public_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        source = item
    else:
        source = getattr(item, "__dict__", {})
    result: dict[str, Any] = {}
    for key in (
        "task_id",
        "text",
        "goal",
        "source",
        "command_id",
        "tool_name",
        "task_text",
        "ok",
        "need_continue",
        "remaining_task_hint",
        "error",
        "status",
    ):
        if key not in source:
            continue
        value = source.get(key)
        if isinstance(value, str):
            value = normalize_message_text(value)
        if value not in ("", None, [], {}):
            result[key] = value
    return result


__all__ = [
    "append_agent_turn_state_snapshot",
    "build_agent_turn_state_snapshot",
]
