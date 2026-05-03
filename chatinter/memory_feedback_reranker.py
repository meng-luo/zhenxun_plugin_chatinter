from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import ClassVar

from .route_text import normalize_message_text

_RECALL_TTL = 10 * 60.0
_FEEDBACK_TTL = 6 * 60 * 60.0
_MAX_GLOBAL_ITEMS = 2048
_MAX_SESSION_ITEMS = 256
_MAX_LAST_RECALL_SESSIONS = 512


@dataclass(frozen=True)
class MemoryRecallRecord:
    timestamp: float
    session_id: str
    memory_ids: tuple[int, ...]
    query: str = ""
    thread_id: str = ""
    topic_key: str = ""


class MemoryFeedbackReranker:
    """In-memory feedback signal for long-term memory recall.

    The signal is intentionally short-lived and bounded: it only reranks memories
    that already passed structured/vector filtering, so feedback cannot pull
    unrelated memories across groups or threads.
    """

    _last_recall: ClassVar[dict[str, MemoryRecallRecord]] = {}
    _memory_feedback: ClassVar[dict[int, float]] = {}
    _memory_feedback_time: ClassVar[dict[int, float]] = {}
    _session_memory_feedback: ClassVar[dict[str, dict[int, float]]] = defaultdict(dict)
    _session_feedback_time: ClassVar[dict[str, float]] = {}

    @classmethod
    def remember_recall(
        cls,
        *,
        session_id: str | None,
        memory_ids: list[int] | tuple[int, ...],
        query: str = "",
        thread_id: str | None = None,
        topic_key: str = "",
    ) -> None:
        normalized_session = normalize_message_text(session_id or "")
        if not normalized_session:
            return
        deduped_ids = tuple(
            dict.fromkeys(int(memory_id) for memory_id in memory_ids if memory_id > 0)
        )
        if not deduped_ids:
            return
        now = time.monotonic()
        cls._last_recall[normalized_session] = MemoryRecallRecord(
            timestamp=now,
            session_id=normalized_session,
            memory_ids=deduped_ids,
            query=normalize_message_text(query)[:160],
            thread_id=normalize_message_text(thread_id or ""),
            topic_key=normalize_message_text(topic_key)[:120],
        )
        cls._prune(now)

    @classmethod
    def record_feedback(
        cls,
        *,
        session_id: str | None,
        kind: str,
        weight: float = 0.0,
    ) -> None:
        normalized_session = normalize_message_text(session_id or "")
        normalized_kind = normalize_message_text(kind).lower()
        if not normalized_session or not normalized_kind:
            return
        now = time.monotonic()
        cls._prune(now)
        recall = cls._last_recall.get(normalized_session)
        if recall is None or now - recall.timestamp > _RECALL_TTL:
            return
        delta = cls._feedback_delta(normalized_kind, weight)
        if not delta:
            return
        session_bucket = cls._session_memory_feedback[normalized_session]
        for rank, memory_id in enumerate(recall.memory_ids):
            ranked_delta = delta * cls._rank_weight(normalized_kind, rank)
            if not ranked_delta:
                continue
            cls._memory_feedback[memory_id] = _clamp_feedback(
                cls._memory_feedback.get(memory_id, 0.0) + ranked_delta * 0.35
            )
            cls._memory_feedback_time[memory_id] = now
            session_bucket[memory_id] = _clamp_feedback(
                session_bucket.get(memory_id, 0.0) + ranked_delta
            )
        cls._session_feedback_time[normalized_session] = now
        cls._prune(now)

    @classmethod
    def score_memory(
        cls,
        *,
        memory_id: int,
        session_id: str | None,
    ) -> float:
        if memory_id <= 0:
            return 0.0
        now = time.monotonic()
        cls._prune(now)
        normalized_session = normalize_message_text(session_id or "")
        global_score = cls._memory_feedback.get(memory_id, 0.0)
        session_score = 0.0
        if normalized_session:
            session_score = cls._session_memory_feedback.get(
                normalized_session, {}
            ).get(memory_id, 0.0)
        # Keep feedback as a reranker, not a bypass for context.
        return max(min(global_score * 0.15 + session_score * 0.35, 0.42), -0.62)

    @classmethod
    def clear(cls) -> None:
        cls._last_recall.clear()
        cls._memory_feedback.clear()
        cls._memory_feedback_time.clear()
        cls._session_memory_feedback.clear()
        cls._session_feedback_time.clear()

    @classmethod
    def _feedback_delta(cls, kind: str, weight: float) -> float:
        if kind == "user_thanks":
            return 0.42 + max(float(weight or 0.0), 0.0) * 0.25
        if kind == "user_corrected":
            return -0.78 + min(float(weight or 0.0), 0.0) * 0.20
        if kind == "followup_same_topic":
            return 0.12
        return 0.0

    @classmethod
    def _rank_weight(cls, kind: str, rank: int) -> float:
        if rank < 0:
            return 0.0
        if kind == "user_corrected":
            return (1.0, 0.25, 0.1)[rank] if rank < 3 else 0.0
        if kind == "user_thanks":
            return (1.0, 0.6, 0.35, 0.2)[rank] if rank < 4 else 0.0
        if kind == "followup_same_topic":
            return (0.55, 0.35, 0.2)[rank] if rank < 3 else 0.0
        return 0.0

    @classmethod
    def _prune(cls, now: float) -> None:
        expired_recall_sessions = [
            session_id
            for session_id, record in cls._last_recall.items()
            if now - record.timestamp > _RECALL_TTL
        ]
        for session_id in expired_recall_sessions:
            cls._last_recall.pop(session_id, None)
        if len(cls._last_recall) > _MAX_LAST_RECALL_SESSIONS:
            oldest = sorted(
                cls._last_recall,
                key=lambda session_id: cls._last_recall[session_id].timestamp,
            )[:64]
            for session_id in oldest:
                cls._last_recall.pop(session_id, None)

        expired_memory_ids = [
            memory_id
            for memory_id, updated_at in cls._memory_feedback_time.items()
            if now - updated_at > _FEEDBACK_TTL
        ]
        for memory_id in expired_memory_ids:
            cls._memory_feedback.pop(memory_id, None)
            cls._memory_feedback_time.pop(memory_id, None)
        if len(cls._memory_feedback) > _MAX_GLOBAL_ITEMS:
            weakest = sorted(
                cls._memory_feedback,
                key=lambda memory_id: (
                    abs(cls._memory_feedback[memory_id]),
                    cls._memory_feedback_time.get(memory_id, 0.0),
                ),
            )[:128]
            for memory_id in weakest:
                cls._memory_feedback.pop(memory_id, None)
                cls._memory_feedback_time.pop(memory_id, None)

        expired_sessions = [
            session_id
            for session_id, updated_at in cls._session_feedback_time.items()
            if now - updated_at > _FEEDBACK_TTL
        ]
        for session_id in expired_sessions:
            cls._session_memory_feedback.pop(session_id, None)
            cls._session_feedback_time.pop(session_id, None)
        for session_id, bucket in list(cls._session_memory_feedback.items()):
            if len(bucket) <= _MAX_SESSION_ITEMS:
                continue
            weakest = sorted(bucket, key=lambda memory_id: abs(bucket[memory_id]))[:64]
            for memory_id in weakest:
                bucket.pop(memory_id, None)
            if not bucket:
                cls._session_memory_feedback.pop(session_id, None)
                cls._session_feedback_time.pop(session_id, None)


def _clamp_feedback(value: float) -> float:
    return max(min(float(value or 0.0), 1.2), -1.6)


__all__ = [
    "MemoryFeedbackReranker",
    "MemoryRecallRecord",
]
