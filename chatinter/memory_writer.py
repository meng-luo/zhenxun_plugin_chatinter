from __future__ import annotations

from dataclasses import dataclass

from .chat_memory_store import ChatMemoryStore, extract_memory_candidates
from .group_memory_digest import build_group_memory_digest
from .memory_policy import MemoryPolicyDecision, decide_memory_policy
from .reflection_observer import record_reflection_observation
from .route_text import normalize_message_text


@dataclass(frozen=True)
class MemoryWriteContext:
    session_id: str
    user_id: str
    group_id: str | None
    message_text: str
    response_text: str = ""
    source_dialog_id: int | None = None
    thread_id: str | None = None
    topic_key: str = ""
    participants: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryWriteResult:
    written: int
    policy: MemoryPolicyDecision
    candidate_count: int = 0


class MemoryWriter:
    @classmethod
    async def write_from_dialog(cls, context: MemoryWriteContext) -> MemoryWriteResult:
        message_text = normalize_message_text(context.message_text)
        response_text = normalize_message_text(context.response_text)
        candidates = extract_memory_candidates(message_text)
        policy = decide_memory_policy(
            message_text=message_text,
            response_text=response_text,
            group_id=context.group_id,
            thread_id=context.thread_id,
            memory_candidate_count=len(candidates),
        )
        if not policy.should_write:
            _record_reflection(
                context,
                action="memory_skip",
                policy=policy,
                written=0,
                candidate_count=len(candidates),
            )
            return MemoryWriteResult(
                written=0,
                policy=policy,
                candidate_count=len(candidates),
            )

        written = await ChatMemoryStore.record_candidates(
            session_id=context.session_id,
            user_id=context.user_id,
            group_id=context.group_id,
            candidates=candidates,
            source_dialog_id=context.source_dialog_id,
            source_message=message_text,
            scope="user",
            thread_id=context.thread_id,
            topic_key=context.topic_key,
            participants=context.participants,
        )
        action = "memory_write"
        if policy.action == "digest":
            digest = build_group_memory_digest(
                session_id=context.session_id,
                user_id=context.user_id,
                group_id=context.group_id,
                thread_id=context.thread_id,
                topic_key=context.topic_key,
                participants=context.participants,
                message_text=message_text,
                response_text=response_text,
            )
            if digest is not None:
                written += await ChatMemoryStore.record_group_digest(digest)
                action = "memory_digest"

        _record_reflection(
            context,
            action=action,
            policy=policy,
            written=written,
            candidate_count=len(candidates),
        )
        return MemoryWriteResult(
            written=written,
            policy=policy,
            candidate_count=len(candidates),
        )


def _record_reflection(
    context: MemoryWriteContext,
    *,
    action: str,
    policy: MemoryPolicyDecision,
    written: int,
    candidate_count: int,
) -> None:
    record_reflection_observation(
        action=action,
        session_id=context.session_id,
        user_id=context.user_id,
        group_id=context.group_id,
        thread_id=context.thread_id,
        reason=policy.reason,
        written=written,
        candidate_count=candidate_count,
        message_text=context.message_text,
    )


__all__ = [
    "MemoryWriteContext",
    "MemoryWriteResult",
    "MemoryWriter",
]
