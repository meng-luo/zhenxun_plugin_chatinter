from __future__ import annotations

from dataclasses import dataclass

from .route_text import normalize_message_text


@dataclass(frozen=True)
class MemoryRecallContext:
    session_id: str
    user_id: str
    group_id: str | None = None
    thread_id: str | None = None
    topic_key: str = ""
    participants: tuple[str, ...] = ()
    addressee_user_id: str | None = None
    query: str = ""
    intent_kind: str = ""

    @classmethod
    def build(
        cls,
        *,
        session_id: str,
        user_id: str,
        group_id: str | None = None,
        thread_id: str | None = None,
        topic_key: str = "",
        participants: tuple[str, ...] = (),
        addressee_user_id: str | None = None,
        query: str = "",
        intent_kind: str = "",
    ) -> "MemoryRecallContext":
        return cls(
            session_id=normalize_message_text(session_id),
            user_id=normalize_message_text(user_id),
            group_id=normalize_message_text(group_id or "") or None,
            thread_id=normalize_message_text(thread_id or "") or None,
            topic_key=normalize_message_text(topic_key),
            participants=tuple(
                dict.fromkeys(
                    normalize_message_text(item)
                    for item in participants
                    if normalize_message_text(item)
                )
            ),
            addressee_user_id=normalize_message_text(addressee_user_id or "") or None,
            query=normalize_message_text(query),
            intent_kind=normalize_message_text(intent_kind),
        )


def split_memory_participants(raw: str | None) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip() for item in str(raw or "").split(",") if item.strip()
        )
    )


def join_memory_participants(participants: tuple[str, ...] | list[str]) -> str:
    return ",".join(
        item
        for item in dict.fromkeys(normalize_message_text(v) for v in participants)
        if item
    )[:1024]


__all__ = [
    "MemoryRecallContext",
    "join_memory_participants",
    "split_memory_participants",
]
