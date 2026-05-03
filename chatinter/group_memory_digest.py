from __future__ import annotations

from dataclasses import dataclass
import time

from .route_text import normalize_message_text

_DIGEST_CACHE_TTL = 8 * 60.0
_DIGEST_CACHE_MAX = 512
_DIGEST_VALUE_MAX_LEN = 140
_recent_digests: dict[str, float] = {}


@dataclass(frozen=True)
class GroupMemoryDigest:
    session_id: str
    user_id: str
    group_id: str
    thread_id: str
    key: str
    value: str
    confidence: float = 0.58
    participants: tuple[str, ...] = ()
    source_message: str = ""


def build_group_memory_digest(
    *,
    session_id: str,
    user_id: str,
    group_id: str | None,
    thread_id: str | None,
    topic_key: str = "",
    participants: tuple[str, ...] = (),
    message_text: str,
    response_text: str = "",
) -> GroupMemoryDigest | None:
    normalized_group = normalize_message_text(group_id or "")
    normalized_thread = normalize_message_text(thread_id or "")
    if not normalized_group or not normalized_thread:
        return None
    message = _clip_digest_text(message_text)
    response = _clip_digest_text(response_text)
    if not message:
        return None
    key = (
        normalize_message_text(topic_key or normalized_thread)[:96] or normalized_thread
    )
    if response:
        value = f"用户提到：{message}；回复要点：{response}"
    else:
        value = f"用户提到：{message}"
    value = value[:_DIGEST_VALUE_MAX_LEN]
    digest = GroupMemoryDigest(
        session_id=normalize_message_text(session_id),
        user_id=normalize_message_text(user_id),
        group_id=normalized_group,
        thread_id=normalized_thread,
        key=key,
        value=value,
        participants=participants,
        source_message=message,
    )
    if not _remember_digest(digest):
        return None
    return digest


def _clip_digest_text(value: str) -> str:
    text = normalize_message_text(value or "")
    if not text:
        return ""
    return text[:72]


def _remember_digest(digest: GroupMemoryDigest) -> bool:
    now = time.monotonic()
    expired = [key for key, deadline in _recent_digests.items() if deadline <= now]
    for key in expired:
        _recent_digests.pop(key, None)
    key = "|".join((digest.group_id, digest.thread_id, digest.key, digest.value))
    if key in _recent_digests:
        return False
    if len(_recent_digests) >= _DIGEST_CACHE_MAX:
        for item in list(_recent_digests)[:64]:
            _recent_digests.pop(item, None)
    _recent_digests[key] = now + _DIGEST_CACHE_TTL
    return True


__all__ = [
    "GroupMemoryDigest",
    "build_group_memory_digest",
]
