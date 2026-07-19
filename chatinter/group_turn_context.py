from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

from .route_text import normalize_message_text

_MAX_GROUPS = 256
_MAX_RECORDS_PER_GROUP = 80
_RECORD_TTL_SECONDS = 15 * 60
_LINE_TEXT_LIMIT = 180


@dataclass(frozen=True)
class GroupTurnRecord:
    user_id: str
    nickname: str
    message_id: str
    text: str
    created_at: float


_records: dict[str, deque[GroupTurnRecord]] = {}


def record_group_turn_message(
    *,
    group_id: str | None,
    user_id: str,
    nickname: str,
    text: str,
    message_id: str = "",
) -> None:
    group_key = normalize_message_text(group_id or "")
    normalized_text = normalize_message_text(text)
    if not group_key or not normalized_text:
        return
    _prune_all()
    records = _records.setdefault(group_key, deque())
    normalized_id = normalize_message_text(message_id)
    if normalized_id and any(item.message_id == normalized_id for item in records):
        return
    nickname_text = normalize_message_text(nickname) or normalize_message_text(user_id)
    records.append(
        GroupTurnRecord(
            user_id=normalize_message_text(user_id),
            nickname=nickname_text,
            message_id=normalized_id,
            text=normalized_text[:_LINE_TEXT_LIMIT],
            created_at=time.time(),
        )
    )
    while len(records) > _MAX_RECORDS_PER_GROUP:
        records.popleft()


def snapshot_group_turn_context(
    *,
    group_id: str | None,
    current_user_id: str = "",
    current_message_text: str = "",
    current_message_id: str = "",
    limit: int = 16,
) -> list[str]:
    group_key = normalize_message_text(group_id or "")
    if not group_key:
        return []
    _prune_group(group_key)
    records = _records.get(group_key)
    if not records:
        return []
    current_user = normalize_message_text(current_user_id)
    current_text = normalize_message_text(current_message_text)
    current_id = normalize_message_text(current_message_id)
    lines: list[str] = []
    for item in records:
        if current_id and item.message_id == current_id:
            continue
        if current_text and item.user_id == current_user and item.text == current_text:
            continue
        timestamp = time.strftime("%H:%M:%S", time.localtime(item.created_at))
        lines.append(f"[{timestamp}] {item.nickname}: {item.text}")
    return lines[-max(int(limit or 0), 0) :]


def clear_group_turn_context(group_id: str | None) -> None:
    group_key = normalize_message_text(group_id or "")
    if group_key:
        _records.pop(group_key, None)


def consume_group_turn_context(
    group_id: str | None,
    through_message_id: str | None,
) -> None:
    group_key = normalize_message_text(group_id or "")
    through_id = normalize_message_text(through_message_id or "")
    if not group_key or not through_id:
        return
    _prune_group(group_key)
    records = _records.get(group_key)
    if not records:
        return
    consume_count = 0
    for item in records:
        consume_count += 1
        if item.message_id == through_id:
            break
    else:
        return
    for _ in range(consume_count):
        records.popleft()
    if not records:
        _records.pop(group_key, None)


def _prune_all() -> None:
    while len(_records) > _MAX_GROUPS:
        _records.pop(next(iter(_records)), None)
    for group_key in list(_records):
        _prune_group(group_key)


def _prune_group(group_key: str) -> None:
    records = _records.get(group_key)
    if not records:
        _records.pop(group_key, None)
        return
    cutoff = time.time() - _RECORD_TTL_SECONDS
    while records and records[0].created_at < cutoff:
        records.popleft()
    if not records:
        _records.pop(group_key, None)


__all__ = [
    "clear_group_turn_context",
    "consume_group_turn_context",
    "record_group_turn_message",
    "snapshot_group_turn_context",
]
