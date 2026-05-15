"""Model-facing observations for native command execution."""

from __future__ import annotations

import re
from typing import Any

from .route_text import normalize_message_text, strip_invoke_prefix

_TASK_SPLIT_PATTERN = re.compile(
    r"(?:，|,|。|；|;|\s)+(?:然后|接着|再|最后|顺便|并且|以及|还有|同时)\s*"
)
_LEADING_CONNECTOR_PATTERN = re.compile(
    r"^(?:然后|接着|再|最后|顺便|并且|以及|还有|同时)+"
)


def build_command_observation(
    *,
    ok: bool,
    command_id: str | None,
    rendered_command: str | None,
    matched_plugin: str | None,
    messages_sent: list[str] | tuple[str, ...] | None = None,
    task_text: str = "",
    ambient_message: str = "",
    trace_id: str = "",
    error: str = "",
    missing: list[str] | tuple[str, ...] | None = None,
    retryable: bool = False,
    plugin_module: str = "",
) -> dict[str, Any]:
    """Build the only payload command tools return to the model."""

    sent = [
        normalize_message_text(str(item or ""))
        for item in (messages_sent or [])
        if normalize_message_text(str(item or ""))
    ]
    remaining = infer_remaining_task_hint(
        ambient_message=ambient_message,
        task_text=task_text,
        rendered_command=rendered_command or "",
    )
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "command_id": normalize_message_text(command_id or ""),
        "rendered_command": normalize_message_text(rendered_command or ""),
        "matched_plugin": normalize_message_text(matched_plugin or plugin_module or ""),
        "messages_sent": sent[:8],
        "need_continue": bool(remaining),
        "remaining_task_hint": remaining,
    }
    if plugin_module:
        payload["plugin_module"] = normalize_message_text(plugin_module)
    if trace_id:
        payload["trace_id"] = normalize_message_text(trace_id)
    if error:
        payload["error"] = normalize_message_text(error)
    if missing:
        payload["missing"] = [
            normalize_message_text(str(item or ""))
            for item in missing
            if normalize_message_text(str(item or ""))
        ]
    if retryable:
        payload["retryable"] = True
    return payload


def infer_remaining_task_hint(
    *,
    ambient_message: str,
    task_text: str,
    rendered_command: str = "",
) -> str:
    ambient = _strip_leading_connector(strip_invoke_prefix(ambient_message))
    task = _strip_leading_connector(task_text)
    command = normalize_message_text(rendered_command)
    if not ambient or not task or ambient == task:
        return ""

    parts = [
        _strip_leading_connector(part)
        for part in _TASK_SPLIT_PATTERN.split(ambient)
        if _strip_leading_connector(part)
    ]
    if len(parts) <= 1:
        return _remaining_after_substring(ambient, task)

    matched_index = -1
    for index, part in enumerate(parts):
        if _task_matches_part(part, task=task, command=command):
            matched_index = index
            break
    if matched_index < 0:
        return _remaining_after_substring(ambient, task)

    return normalize_message_text("，".join(parts[matched_index + 1 :]))


def _task_matches_part(part: str, *, task: str, command: str) -> bool:
    normalized_part = normalize_message_text(part)
    normalized_task = normalize_message_text(task)
    if normalized_task and (
        normalized_task in normalized_part or normalized_part in normalized_task
    ):
        return True
    return bool(
        command
        and (command in normalized_part or _is_subsequence(command, normalized_part))
    )


def _is_subsequence(needle: str, haystack: str) -> bool:
    needle = normalize_message_text(needle)
    haystack = normalize_message_text(haystack)
    if not needle or not haystack:
        return False
    cursor = 0
    for char in needle:
        cursor = haystack.find(char, cursor)
        if cursor < 0:
            return False
        cursor += 1
    return True


def _remaining_after_substring(ambient: str, task: str) -> str:
    if task and task in ambient:
        _before, _sep, after = ambient.partition(task)
        return _strip_leading_connector(after)
    return ""


def _strip_leading_connector(text: str) -> str:
    normalized = normalize_message_text(text)
    while normalized:
        next_text = normalize_message_text(
            strip_invoke_prefix(_LEADING_CONNECTOR_PATTERN.sub("", normalized))
        )
        if next_text == normalized:
            return normalized
        normalized = next_text
    return ""


__all__ = [
    "build_command_observation",
    "infer_remaining_task_hint",
]
