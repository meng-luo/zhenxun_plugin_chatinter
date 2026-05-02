from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

from .route_text import normalize_message_text

_MEMORY_LIMIT = 8
_MEMORY_VALUE_MAX_LEN = 80
_MEMORY_CONFIDENCE_DEFAULT = 0.72
_RECENT_WRITE_CACHE_TTL = 60.0
_RECENT_WRITE_CACHE_MAX = 512
_recent_writes: dict[str, float] = {}


def _debug(message: str) -> None:
    try:
        from zhenxun.services.log import logger

        logger.debug(message)
    except Exception:
        pass


def _get_memory_model() -> Any | None:
    try:
        from .models.chat_history import ChatInterMemory

        return ChatInterMemory
    except Exception:
        return None


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: str
    key: str
    value: str
    confidence: float = _MEMORY_CONFIDENCE_DEFAULT


_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "nickname",
        "self_nickname",
        re.compile(r"(?:以后)?(?:叫我|喊我|称呼我)\s*([^\s，。,.!！?？]{1,24})"),
    ),
    (
        "preference",
        "like",
        re.compile(r"我(?:很|比较|最|超)?喜欢\s*([^\n，。,.!！?？]{1,40})"),
    ),
    (
        "preference",
        "dislike",
        re.compile(r"我(?:很|比较|最|超)?不喜欢\s*([^\n，。,.!！?？]{1,40})"),
    ),
    (
        "correction",
        "nickname_correction",
        re.compile(r"(?:别|不要)叫我\s*([^\s，。,.!！?？]{1,24})"),
    ),
)


def _clean_memory_value(value: str) -> str:
    normalized = normalize_message_text(value)
    return normalized[:_MEMORY_VALUE_MAX_LEN]


def extract_memory_candidates(message_text: str) -> list[MemoryCandidate]:
    text = normalize_message_text(message_text or "")
    if not text or text.endswith(("吗", "么", "嘛", "？", "?")):
        return []
    candidates: list[MemoryCandidate] = []
    for memory_type, key, pattern in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = _clean_memory_value(match.group(1))
        if not value:
            continue
        if value in {"一下", "这个", "那个", "它", "他", "她"}:
            continue
        candidates.append(
            MemoryCandidate(
                memory_type=memory_type,
                key=key,
                value=value,
                confidence=0.82 if memory_type == "nickname" else 0.72,
            )
        )
    return candidates


def _write_cache_key(
    *,
    session_id: str,
    user_id: str,
    candidate: MemoryCandidate,
) -> str:
    return "|".join(
        (
            normalize_message_text(session_id),
            normalize_message_text(user_id),
            candidate.memory_type,
            candidate.key,
            candidate.value,
        )
    )


def _remember_recent_write(key: str) -> bool:
    now = time.monotonic()
    expired = [item for item, deadline in _recent_writes.items() if deadline <= now]
    for item in expired:
        _recent_writes.pop(item, None)
    if key in _recent_writes:
        return False
    if len(_recent_writes) >= _RECENT_WRITE_CACHE_MAX:
        for item in list(_recent_writes)[:64]:
            _recent_writes.pop(item, None)
    _recent_writes[key] = now + _RECENT_WRITE_CACHE_TTL
    return True


class ChatMemoryStore:
    @staticmethod
    async def record_from_dialog(
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        message_text: str,
        source_dialog_id: int | None = None,
    ) -> int:
        memory_model = _get_memory_model()
        if memory_model is None:
            return 0
        candidates = extract_memory_candidates(message_text)
        if not candidates:
            return 0

        written = 0
        for candidate in candidates:
            cache_key = _write_cache_key(
                session_id=session_id,
                user_id=user_id,
                candidate=candidate,
            )
            if not _remember_recent_write(cache_key):
                continue
            try:
                await memory_model.upsert_memory(
                    session_id=session_id,
                    user_id=user_id,
                    group_id=group_id,
                    memory_type=candidate.memory_type,
                    key=candidate.key,
                    value=candidate.value,
                    confidence=candidate.confidence,
                    source_dialog_id=source_dialog_id,
                    source_message=message_text,
                )
                written += 1
            except Exception as exc:
                _debug(f"chatinter memory write skipped: {exc}")
        return written

    @staticmethod
    async def recall(
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        query: str,
        limit: int = _MEMORY_LIMIT,
    ) -> list[str]:
        memory_model = _get_memory_model()
        if memory_model is None:
            return []
        try:
            memories = await memory_model.recall_memories(
                session_id=session_id,
                user_id=user_id,
                group_id=group_id,
                query=query,
                limit=limit,
            )
        except Exception as exc:
            _debug(f"chatinter memory recall skipped: {exc}")
            return []
        lines: list[str] = []
        for memory in memories:
            memory_type = normalize_message_text(getattr(memory, "memory_type", ""))
            key = normalize_message_text(getattr(memory, "key", ""))
            value = normalize_message_text(getattr(memory, "value", ""))
            if not value:
                continue
            label = f"{memory_type}:{key}".strip(":")
            lines.append(f"{label}={value}" if label else value)
        return lines[: max(int(limit or 0), 0)]


__all__ = [
    "ChatMemoryStore",
    "MemoryCandidate",
    "extract_memory_candidates",
]
