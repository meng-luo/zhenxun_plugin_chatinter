from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any

from .route_text import normalize_message_text

_THREAD_DIALOG_LIMIT_DEFAULT = 8
_MESSAGE_PREVIEW_LIMIT = 160


@dataclass(frozen=True)
class StoredThreadSnapshot:
    thread_id: str
    group_id: str | None
    participants: tuple[str, ...]
    topic_key: str = ""
    source: str = ""
    confidence: float = 0.0
    pending_entities: tuple[str, ...] = ()
    entity_hints: tuple[str, ...] = ()
    last_active: datetime | None = None


async def get_thread_by_message(
    *,
    group_id: str | None,
    message_id: str | None,
) -> StoredThreadSnapshot | None:
    if not message_id:
        return None
    model = _get_thread_message_model()
    thread_model = _get_thread_model()
    if model is None or thread_model is None:
        return None
    try:
        mapping = (
            await model.filter(
                group_id=group_id,
                message_id=str(message_id),
            )
            .order_by("-id")
            .first()
        )
        if mapping is None:
            return None
        row = (
            await thread_model.filter(thread_id=mapping.thread_id, group_id=group_id)
            .order_by("-id")
            .first()
        )
        if row is None:
            return None
        return _snapshot_from_row(row)
    except Exception:
        return None


async def get_recent_thread_dialog_ids(
    *,
    thread_id: str | None,
    group_id: str | None,
    limit: int = _THREAD_DIALOG_LIMIT_DEFAULT,
) -> list[int]:
    if not thread_id:
        return []
    model = _get_thread_message_model()
    if model is None:
        return []
    try:
        rows = (
            await model.filter(thread_id=thread_id, group_id=group_id)
            .exclude(dialog_id=None)
            .order_by("-create_time", "-id")
            .limit(max(int(limit or 0), 1))
        )
    except Exception:
        return []
    ids = [int(row.dialog_id) for row in rows if row.dialog_id]
    return list(reversed(ids))


async def find_recent_thread(
    *,
    group_id: str | None,
    topic_key: str,
    participants: tuple[str, ...],
) -> StoredThreadSnapshot | None:
    thread_model = _get_thread_model()
    if thread_model is None or not topic_key:
        return None
    try:
        rows = (
            await thread_model.filter(
                group_id=group_id,
                topic_key=topic_key,
                archived=False,
            )
            .order_by("-last_active", "-id")
            .limit(6)
        )
    except Exception:
        return None
    participant_set = set(participants)
    for row in rows:
        stored = _split_participants(str(row.participants or ""))
        if not participant_set or participant_set.intersection(stored):
            return _snapshot_from_row(row)
    return None


async def find_recent_pending_thread(
    *,
    group_id: str | None,
    participants: tuple[str, ...],
) -> StoredThreadSnapshot | None:
    thread_model = _get_thread_model()
    if thread_model is None or not group_id:
        return None
    try:
        rows = (
            await thread_model.filter(
                group_id=group_id,
                archived=False,
            )
            .exclude(pending_entities="")
            .order_by("-last_active", "-id")
            .limit(8)
        )
    except Exception:
        return None
    participant_set = set(participants)
    for row in rows:
        stored = _split_participants(str(row.participants or ""))
        if not participant_set or participant_set.intersection(stored):
            return _snapshot_from_row(row)
    return None


async def record_thread_message(
    *,
    thread_id: str,
    group_id: str | None,
    message_id: str | None,
    dialog_id: int | None,
    user_id: str,
    participants: tuple[str, ...],
    topic_key: str,
    source: str,
    confidence: float,
    message_text: str,
    pending_entities: tuple[str, ...] = (),
    entity_hints: tuple[str, ...] = (),
) -> None:
    thread_model = _get_thread_model()
    message_model = _get_thread_message_model()
    if thread_model is None or message_model is None or not thread_id:
        return
    preview = normalize_message_text(message_text)[:_MESSAGE_PREVIEW_LIMIT]
    participants_text = ",".join(item for item in dict.fromkeys(participants) if item)[
        :1024
    ]
    pending_text = _dump_items(pending_entities)
    hints_text = _dump_items(entity_hints)
    try:
        row = (
            await thread_model.filter(thread_id=thread_id, group_id=group_id)
            .order_by("-id")
            .first()
        )
        if row is None:
            await thread_model.create(
                thread_id=thread_id,
                group_id=group_id,
                participants=participants_text,
                topic_key=topic_key,
                topic_summary=topic_key,
                pending_entities=pending_text,
                entity_hints=hints_text,
                last_message=preview,
                source=source,
                confidence=float(confidence or 0.0),
            )
        else:
            merged_participants = tuple(
                dict.fromkeys((*_split_participants(row.participants), *participants))
            )
            row.participants = ",".join(merged_participants)[:1024]
            row.topic_key = topic_key or row.topic_key
            row.topic_summary = row.topic_summary or topic_key
            if pending_text:
                row.pending_entities = _dump_items(
                    (*_load_items(row.pending_entities), *pending_entities)
                )
            if hints_text:
                row.entity_hints = _dump_items(
                    (*_load_items(row.entity_hints), *entity_hints)
                )
            row.last_message = preview
            row.source = source or row.source
            row.confidence = max(float(row.confidence or 0.0), float(confidence or 0.0))
            row.archived = False
            await row.save()
        if message_id or dialog_id:
            await message_model.create(
                thread_id=thread_id,
                group_id=group_id,
                message_id=str(message_id) if message_id else None,
                dialog_id=dialog_id,
                user_id=str(user_id or ""),
                message_preview=preview,
            )
    except Exception:
        return


def _snapshot_from_row(row: Any) -> StoredThreadSnapshot:
    return StoredThreadSnapshot(
        thread_id=str(getattr(row, "thread_id", "") or ""),
        group_id=getattr(row, "group_id", None),
        participants=_split_participants(str(getattr(row, "participants", "") or "")),
        topic_key=str(getattr(row, "topic_key", "") or ""),
        source=str(getattr(row, "source", "") or ""),
        confidence=float(getattr(row, "confidence", 0.0) or 0.0),
        pending_entities=_load_items(getattr(row, "pending_entities", "") or ""),
        entity_hints=_load_items(getattr(row, "entity_hints", "") or ""),
        last_active=getattr(row, "last_active", None),
    )


def _split_participants(raw: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip() for item in str(raw or "").split(",") if item.strip()
        )
    )


def _load_items(raw: object) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    try:
        value = json.loads(text)
    except Exception:
        value = None
    if isinstance(value, list):
        items = value
    else:
        items = _split_items_fallback(text)
    return tuple(
        dict.fromkeys(
            normalize_message_text(str(item))
            for item in items
            if normalize_message_text(str(item))
        )
    )[:8]


def _split_items_fallback(text: str) -> list[str]:
    return [
        item.strip()
        for item in text.replace("，", ",").replace("、", ",").split(",")
        if item.strip()
    ]


def _dump_items(items: tuple[str, ...]) -> str:
    normalized = [
        item
        for item in dict.fromkeys(normalize_message_text(value) for value in items)
        if item
    ][:8]
    if not normalized:
        return ""
    return json.dumps(normalized, ensure_ascii=False)[:512]


def _get_thread_model() -> Any | None:
    try:
        from .models.chat_history import ChatInterThread

        return ChatInterThread
    except Exception:
        return None


def _get_thread_message_model() -> Any | None:
    try:
        from .models.chat_history import ChatInterThreadMessage

        return ChatInterThreadMessage
    except Exception:
        return None


__all__ = [
    "StoredThreadSnapshot",
    "find_recent_pending_thread",
    "find_recent_thread",
    "get_recent_thread_dialog_ids",
    "get_thread_by_message",
    "record_thread_message",
]
