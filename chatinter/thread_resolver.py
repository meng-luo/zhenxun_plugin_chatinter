from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .addressee_resolver import AddresseeResult
from .event_context import ChatInterEventContext
from .route_text import contains_any, normalize_message_text
from .thread_store import (
    find_recent_pending_thread,
    find_recent_thread,
    get_thread_by_message,
)

_PENDING_FOLLOWUP_HINTS = (
    "那个",
    "那位",
    "这个人",
    "那个人",
    "群里那个",
    "群里的",
    "刚才那个",
    "刚说的",
    "上面那个",
    "他",
    "她",
    "ta",
    "TA",
)


@dataclass(frozen=True)
class ThreadContext:
    thread_id: str
    source: str
    confidence: float
    related_user_ids: tuple[str, ...] = ()
    topic_key: str = ""
    pending_entities: tuple[str, ...] = ()
    entity_hints: tuple[str, ...] = ()

    @property
    def participants(self) -> tuple[str, ...]:
        return self.related_user_ids


async def resolve_thread_context(
    *,
    event_context: ChatInterEventContext,
    addressee: AddresseeResult,
) -> ThreadContext:
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
                pending_entities=stored.pending_entities,
                entity_hints=stored.entity_hints,
            )
            return ctx
        seed = f"reply:{group_key}:{event_context.reply.message_id}"
        ctx = ThreadContext(
            thread_id=_stable_thread_id(seed),
            source="reply",
            confidence=0.95,
            related_user_ids=participants,
            topic_key=_topic_key(event_context.normalized_text),
        )
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
            pending_entities=stored.pending_entities,
            entity_hints=stored.entity_hints,
        )
        return ctx

    if _is_pending_entity_followup(event_context.normalized_text):
        stored = await find_recent_pending_thread(
            group_id=event_context.group_id,
            participants=participants,
        )
        if stored is not None and stored.thread_id:
            ctx = ThreadContext(
                thread_id=stored.thread_id,
                source="pending_entity_store",
                confidence=max(stored.confidence, 0.7),
                related_user_ids=stored.participants or participants,
                topic_key=stored.topic_key or topic_key,
                pending_entities=stored.pending_entities,
                entity_hints=stored.entity_hints,
            )
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
    return ctx


def format_thread_xml(thread: ThreadContext) -> list[str]:
    lines = ["<thread>"]
    if thread.topic_key and thread.source not in {"topic", "reply"}:
        lines.append(f"topic_key={_xml_escape(thread.topic_key)}")
    if thread.related_user_ids:
        lines.append(f"related_user_ids={','.join(thread.related_user_ids)}")
    if thread.pending_entities:
        lines.append(
            f"pending_entities={_xml_escape('、'.join(thread.pending_entities))}"
        )
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


def _is_pending_entity_followup(text: str) -> bool:
    normalized = normalize_message_text(text)
    if not normalized:
        return False
    return contains_any(normalized, _PENDING_FOLLOWUP_HINTS)


def _stable_thread_id(seed: str) -> str:
    return hashlib.blake2s(seed.encode("utf-8"), digest_size=6).hexdigest()


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


__all__ = ["ThreadContext", "format_thread_xml", "resolve_thread_context"]
