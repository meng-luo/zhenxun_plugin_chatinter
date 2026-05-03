from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

from .addressee_resolver import AddresseeResult
from .event_context import ChatInterEventContext
from .route_text import normalize_message_text
from .thread_store import find_recent_thread, get_thread_by_message

_THREAD_TTL = 20 * 60.0
_THREAD_CACHE_MAX = 512
_thread_cache: dict[str, tuple[float, "ThreadContext"]] = {}


@dataclass(frozen=True)
class ThreadContext:
    thread_id: str
    source: str
    confidence: float
    related_user_ids: tuple[str, ...] = ()
    topic_key: str = ""

    @property
    def participants(self) -> tuple[str, ...]:
        return self.related_user_ids


async def resolve_thread_context(
    *,
    event_context: ChatInterEventContext,
    addressee: AddresseeResult,
) -> ThreadContext:
    now = time.monotonic()
    _trim_cache(now)
    group_key = event_context.group_id or f"private:{event_context.user_id}"
    participants = _participants(event_context, addressee)

    if event_context.reply and event_context.reply.message_id:
        stored = await get_thread_by_message(
            group_id=event_context.group_id,
            message_id=event_context.reply.message_id,
        )
        if stored is not None and stored.thread_id:
            ctx = ThreadContext(
                thread_id=stored.thread_id,
                source="reply_store",
                confidence=max(stored.confidence, 0.96),
                related_user_ids=stored.participants or participants,
                topic_key=stored.topic_key,
            )
            _thread_cache[ctx.thread_id] = (now, ctx)
            return ctx
        seed = f"reply:{group_key}:{event_context.reply.message_id}"
        ctx = ThreadContext(
            thread_id=_stable_thread_id(seed),
            source="reply",
            confidence=0.95,
            related_user_ids=participants,
            topic_key=_topic_key(event_context.normalized_text),
        )
        _thread_cache[ctx.thread_id] = (now, ctx)
        return ctx

    topic_key = _topic_key(event_context.normalized_text)
    stored = await find_recent_thread(
        group_id=event_context.group_id,
        topic_key=topic_key,
        participants=participants,
    )
    if stored is not None and stored.thread_id:
        ctx = ThreadContext(
            thread_id=stored.thread_id,
            source="topic_store",
            confidence=max(stored.confidence, 0.62),
            related_user_ids=stored.participants or participants,
            topic_key=stored.topic_key or topic_key,
        )
        _thread_cache[ctx.thread_id] = (now, ctx)
        return ctx

    target_id = addressee.target_user_id or "broadcast"
    seed = f"topic:{group_key}:{target_id}:{topic_key}"
    ctx = ThreadContext(
        thread_id=_stable_thread_id(seed),
        source="topic",
        confidence=0.58 if topic_key else 0.35,
        related_user_ids=participants,
        topic_key=topic_key,
    )
    _thread_cache[ctx.thread_id] = (now, ctx)
    return ctx


def format_thread_xml(thread: ThreadContext) -> list[str]:
    lines = ["<thread>"]
    lines.append(f"thread_id={thread.thread_id}")
    lines.append(f"source={thread.source}")
    lines.append(f"confidence={thread.confidence:.2f}")
    if thread.topic_key:
        lines.append(f"topic_key={_xml_escape(thread.topic_key)}")
    if thread.related_user_ids:
        lines.append(f"related_user_ids={','.join(thread.related_user_ids)}")
    lines.append("</thread>")
    return lines


def _participants(
    event_context: ChatInterEventContext,
    addressee: AddresseeResult,
) -> tuple[str, ...]:
    values = [event_context.user_id]
    if addressee.target_user_id:
        values.append(addressee.target_user_id)
    if event_context.reply and event_context.reply.sender_id:
        values.append(event_context.reply.sender_id)
    return tuple(item for item in dict.fromkeys(values) if item)


def _topic_key(text: str) -> str:
    normalized = normalize_message_text(text)
    if not normalized:
        return ""
    tokens = [token for token in normalized.split() if token]
    if tokens:
        return " ".join(tokens[:8])[:120]
    return normalized[:24]


def _stable_thread_id(seed: str) -> str:
    return hashlib.blake2s(seed.encode("utf-8"), digest_size=6).hexdigest()


def _trim_cache(now: float) -> None:
    expired = [key for key, (ts, _) in _thread_cache.items() if now - ts > _THREAD_TTL]
    for key in expired:
        _thread_cache.pop(key, None)
    if len(_thread_cache) <= _THREAD_CACHE_MAX:
        return
    evict_count = len(_thread_cache) - _THREAD_CACHE_MAX
    for key in sorted(
        _thread_cache,
        key=lambda item: _thread_cache[item][0],
    )[:evict_count]:
        _thread_cache.pop(key, None)


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


__all__ = ["ThreadContext", "format_thread_xml", "resolve_thread_context"]
