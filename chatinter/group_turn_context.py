from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count
import time

from .config import get_chat_history_limit
from .route_text import normalize_message_text

_MAX_GROUPS = 256
_RECORD_TTL_SECONDS = 15 * 60
_LINE_TEXT_LIMIT = 180


@dataclass(frozen=True)
class GroupTurnRecord:
    record_id: str
    user_id: str
    nickname: str
    message_id: str
    text: str
    created_at: float


_records: dict[str, deque[GroupTurnRecord]] = {}
_recent_records: dict[str, deque[GroupTurnRecord]] = {}
_record_sequence = count(1)


def record_group_turn_message(
    *,
    group_id: str | None,
    user_id: str,
    nickname: str,
    text: str,
    message_id: str = "",
    record_id: str = "",
) -> str:
    group_key = normalize_message_text(group_id or "")
    normalized_text = normalize_message_text(text)
    if not group_key or not normalized_text:
        return ""
    _prune_all()
    records = _records.setdefault(group_key, deque())
    recent_records = _recent_records.setdefault(group_key, deque())
    normalized_id = normalize_message_text(message_id)
    if normalized_id:
        existing = next(
            (item for item in recent_records if item.message_id == normalized_id),
            None,
        )
        if existing is not None:
            return existing.record_id
    normalized_record_id = normalize_message_text(record_id)
    if not normalized_record_id:
        normalized_record_id = normalized_id or f"local:{next(_record_sequence)}"
    nickname_text = normalize_message_text(nickname) or normalize_message_text(user_id)
    record = GroupTurnRecord(
        record_id=normalized_record_id,
        user_id=normalize_message_text(user_id),
        nickname=nickname_text,
        message_id=normalized_id,
        text=normalized_text[:_LINE_TEXT_LIMIT],
        created_at=time.time(),
    )
    records.append(record)
    recent_records.append(record)
    while len(records) > get_chat_history_limit():
        records.popleft()
    while len(recent_records) > get_chat_history_limit():
        recent_records.popleft()
    return normalized_record_id


def snapshot_group_turn_context(
    *,
    group_id: str | None,
    current_user_id: str = "",
    current_message_text: str = "",
    current_message_id: str = "",
    limit: int = 16,
) -> list[str]:
    group_key = normalize_message_text(group_id or "")
    _prune_group(group_key)
    visible_records = _snapshot_records(
        _records.get(group_key),
        current_user_id=current_user_id,
        current_message_text=current_message_text,
        current_message_id=current_message_id,
        limit=limit,
    )
    lines: list[str] = []
    for item in visible_records:
        timestamp = time.strftime("%H:%M", time.localtime(item.created_at))
        lines.append(f"[{timestamp}] {item.nickname}: {item.text}")
    return lines


def snapshot_group_turn_records(
    *,
    group_id: str | None,
    current_user_id: str = "",
    current_message_text: str = "",
    current_message_id: str = "",
    limit: int = 16,
) -> tuple[GroupTurnRecord, ...]:
    group_key = normalize_message_text(group_id or "")
    if not group_key:
        return ()
    _prune_group(group_key)
    return _snapshot_records(
        _recent_records.get(group_key),
        current_user_id=current_user_id,
        current_message_text=current_message_text,
        current_message_id=current_message_id,
        limit=limit,
    )


def _snapshot_records(
    records: deque[GroupTurnRecord] | None,
    *,
    current_user_id: str,
    current_message_text: str,
    current_message_id: str,
    limit: int,
) -> tuple[GroupTurnRecord, ...]:
    if not records:
        return ()
    current_user = normalize_message_text(current_user_id)
    current_text = normalize_message_text(current_message_text)
    current_id = normalize_message_text(current_message_id)
    visible_records = list(records)
    boundary = _current_record_index(
        visible_records,
        current_user=current_user,
        current_text=current_text,
        current_id=current_id,
    )
    if boundary is not None:
        visible_records = visible_records[:boundary]
    if boundary is None:
        visible_records = [
            item
            for item in visible_records
            if not (
                (current_id and item.message_id == current_id)
                or (
                    current_text
                    and item.user_id == current_user
                    and item.text == current_text
                )
            )
        ]
    return tuple(visible_records[-max(int(limit or 0), 0) :])


def clear_group_turn_context(group_id: str | None) -> None:
    group_key = normalize_message_text(group_id or "")
    if group_key:
        _records.pop(group_key, None)
        _recent_records.pop(group_key, None)


def remove_group_turn_message(
    group_id: str | None,
    record_id: str | None,
) -> bool:
    group_key = normalize_message_text(group_id or "")
    target_id = normalize_message_text(record_id or "")
    if not group_key or not target_id:
        return False
    _prune_group(group_key)
    records = _records.get(group_key)
    recent_records = _recent_records.get(group_key)
    if not records and not recent_records:
        return False
    kept = deque(item for item in records or () if item.record_id != target_id)
    recent_kept = deque(
        item for item in recent_records or () if item.record_id != target_id
    )
    removed = len(kept) != len(records or ()) or len(recent_kept) != len(
        recent_records or ()
    )
    if not removed:
        return False
    if kept:
        _records[group_key] = kept
    else:
        _records.pop(group_key, None)
    if recent_kept:
        _recent_records[group_key] = recent_kept
    else:
        _recent_records.pop(group_key, None)
    return True


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
        if item.message_id == through_id or item.record_id == through_id:
            break
    else:
        return
    for _ in range(consume_count):
        records.popleft()
    if not records:
        _records.pop(group_key, None)


def _prune_all() -> None:
    group_keys = list(dict.fromkeys((*_records, *_recent_records)))
    while len(group_keys) > _MAX_GROUPS:
        group_key = group_keys.pop(0)
        _records.pop(group_key, None)
        _recent_records.pop(group_key, None)
    for group_key in group_keys:
        _prune_group(group_key)


def _current_record_index(
    records: list[GroupTurnRecord],
    *,
    current_user: str,
    current_text: str,
    current_id: str,
) -> int | None:
    if current_id:
        for index, item in enumerate(records):
            if item.record_id == current_id or item.message_id == current_id:
                return index
    if current_text:
        for index, item in enumerate(records):
            if item.user_id == current_user and item.text == current_text:
                return index
    return None


def _prune_group(group_key: str) -> None:
    cutoff = time.time() - _RECORD_TTL_SECONDS
    for store in (_records, _recent_records):
        records = store.get(group_key)
        if not records:
            store.pop(group_key, None)
            continue
        while records and records[0].created_at < cutoff:
            records.popleft()
        if not records:
            store.pop(group_key, None)


__all__ = [
    "clear_group_turn_context",
    "consume_group_turn_context",
    "record_group_turn_message",
    "remove_group_turn_message",
    "snapshot_group_turn_context",
    "snapshot_group_turn_records",
]
