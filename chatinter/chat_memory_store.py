from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

from .group_memory_digest import GroupMemoryDigest
from .memory_feedback_reranker import MemoryFeedbackReranker
from .memory_recall_context import (
    MemoryRecallContext,
    join_memory_participants,
)
from .memory_vector_index import (
    MemoryVectorIndex,
    MemoryVectorMetadata,
    build_memory_vector_text,
)
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


async def _upsert_vector_if_needed(
    *,
    row: Any,
    memory_type: str,
    key: str,
    value: str,
    session_id: str,
    user_id: str,
    group_id: str | None,
    scope: str,
    thread_id: str | None,
    topic_key: str,
    participants: tuple[str, ...],
    confidence: float,
) -> None:
    if not MemoryVectorIndex.is_indexable_type(memory_type):
        return
    memory_id = int(getattr(row, "id", 0) or 0)
    if memory_id <= 0:
        return
    metadata = MemoryVectorMetadata(
        memory_id=memory_id,
        session_id=normalize_message_text(session_id),
        user_id=normalize_message_text(user_id),
        group_id=normalize_message_text(group_id or "") or None,
        memory_type=normalize_message_text(memory_type),
        scope=normalize_message_text(scope),
        thread_id=normalize_message_text(thread_id or "") or None,
        topic_key=normalize_message_text(topic_key),
        participants=tuple(
            dict.fromkeys(
                normalize_message_text(item)
                for item in participants
                if normalize_message_text(item)
            )
        ),
        confidence=float(confidence or 0.0),
    )
    text = build_memory_vector_text(
        memory_type=memory_type,
        key=key,
        value=value,
        metadata=metadata,
    )
    try:
        await MemoryVectorIndex.upsert_memory_vector(
            memory_id=memory_id,
            text=text,
            metadata=metadata,
        )
    except Exception as exc:
        _debug(f"chatinter memory vector upsert skipped: {exc}")


async def _merge_vector_memories(
    *,
    memory_model: Any,
    structured_memories: list[Any],
    vector_results: list[Any],
    recall_context: MemoryRecallContext,
    limit: int,
) -> list[Any]:
    selected_limit = max(int(limit or 0), 0)
    if not vector_results:
        selected = _rerank_with_feedback(
            structured_memories,
            recall_context=recall_context,
            base_scores={
                int(getattr(memory, "id", 0) or 0): max(1.2 - index * 0.04, 0.0)
                for index, memory in enumerate(structured_memories)
            },
            limit=selected_limit,
        )
        _remember_selected_recall(selected, recall_context=recall_context)
        return selected
    by_id: dict[int, Any] = {}
    order_scores: dict[int, float] = {}
    for index, memory in enumerate(structured_memories):
        memory_id = int(getattr(memory, "id", 0) or 0)
        if memory_id <= 0:
            continue
        by_id[memory_id] = memory
        order_scores[memory_id] = max(1.2 - index * 0.04, 0.0)

    missing_ids = [
        item.memory_id
        for item in vector_results
        if int(item.memory_id or 0) > 0 and item.memory_id not in by_id
    ]
    if missing_ids:
        try:
            rows = await memory_model.filter(id__in=missing_ids, expired=False).all()
        except Exception:
            rows = []
        for row in rows:
            memory_id = int(getattr(row, "id", 0) or 0)
            if memory_id > 0:
                by_id[memory_id] = row

    for item in vector_results:
        memory_id = int(item.memory_id or 0)
        if memory_id <= 0 or memory_id not in by_id:
            continue
        order_scores[memory_id] = max(
            order_scores.get(memory_id, 0.0),
            float(item.score or 0.0) + 0.16,
        )
        setattr(by_id[memory_id], "_chatinter_vector_score", float(item.score or 0.0))
        setattr(by_id[memory_id], "_chatinter_vector_type", item.vector_type)

    selected = _rerank_with_feedback(
        list(by_id.values()),
        recall_context=recall_context,
        base_scores=order_scores,
        limit=selected_limit,
    )
    try:
        await memory_model.mark_recalled(
            [int(getattr(row, "id", 0) or 0) for row in selected]
        )
    except Exception:
        pass
    _remember_selected_recall(selected, recall_context=recall_context)
    return selected


def _rerank_with_feedback(
    rows: list[Any],
    *,
    recall_context: MemoryRecallContext,
    base_scores: dict[int, float],
    limit: int,
) -> list[Any]:
    if limit <= 0 or not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            _memory_rank_score(
                row,
                recall_context=recall_context,
                base_scores=base_scores,
            ),
            float(getattr(row, "confidence", 0.0) or 0.0),
            int(getattr(row, "id", 0) or 0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _memory_rank_score(
    row: Any,
    *,
    recall_context: MemoryRecallContext,
    base_scores: dict[int, float],
) -> float:
    memory_id = int(getattr(row, "id", 0) or 0)
    feedback_score = MemoryFeedbackReranker.score_memory(
        memory_id=memory_id,
        session_id=recall_context.session_id,
    )
    if feedback_score:
        setattr(row, "_chatinter_feedback_score", feedback_score)
    return base_scores.get(memory_id, 0.0) + feedback_score


def _remember_selected_recall(
    selected: list[Any],
    *,
    recall_context: MemoryRecallContext,
) -> None:
    memory_ids = [int(getattr(row, "id", 0) or 0) for row in selected]
    MemoryFeedbackReranker.remember_recall(
        session_id=recall_context.session_id,
        memory_ids=memory_ids,
        query=recall_context.query,
        thread_id=recall_context.thread_id,
        topic_key=recall_context.topic_key,
    )


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
        candidates = extract_memory_candidates(message_text)
        if not candidates:
            return 0
        return await ChatMemoryStore.record_candidates(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            candidates=candidates,
            source_dialog_id=source_dialog_id,
            source_message=message_text,
        )

    @staticmethod
    async def record_candidates(
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        candidates: list[MemoryCandidate],
        source_dialog_id: int | None = None,
        source_message: str | None = None,
        scope: str = "user",
        thread_id: str | None = None,
        topic_key: str = "",
        participants: tuple[str, ...] = (),
    ) -> int:
        memory_model = _get_memory_model()
        if memory_model is None:
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
                row = await memory_model.upsert_memory(
                    session_id=session_id,
                    user_id=user_id,
                    group_id=group_id,
                    memory_type=candidate.memory_type,
                    key=candidate.key,
                    value=candidate.value,
                    confidence=candidate.confidence,
                    scope=scope,
                    thread_id=thread_id,
                    topic_key=topic_key,
                    participants=join_memory_participants(participants),
                    source_dialog_id=source_dialog_id,
                    source_message=source_message,
                )
                await _upsert_vector_if_needed(
                    row=row,
                    memory_type=candidate.memory_type,
                    key=candidate.key,
                    value=candidate.value,
                    session_id=session_id,
                    user_id=user_id,
                    group_id=group_id,
                    scope=scope,
                    thread_id=thread_id,
                    topic_key=topic_key,
                    participants=participants,
                    confidence=candidate.confidence,
                )
                written += 1
            except Exception as exc:
                _debug(f"chatinter memory write skipped: {exc}")
        return written

    @staticmethod
    async def record_group_digest(digest: GroupMemoryDigest) -> int:
        memory_model = _get_memory_model()
        if memory_model is None:
            return 0
        candidate = MemoryCandidate(
            memory_type="group_digest",
            key=digest.key,
            value=digest.value,
            confidence=digest.confidence,
        )
        cache_key = _write_cache_key(
            session_id=digest.session_id,
            user_id=digest.user_id,
            candidate=candidate,
        )
        if not _remember_recent_write(cache_key):
            return 0
        try:
            row = await memory_model.upsert_memory(
                session_id=digest.session_id,
                user_id=digest.user_id,
                group_id=digest.group_id,
                memory_type=candidate.memory_type,
                key=candidate.key,
                value=candidate.value,
                confidence=candidate.confidence,
                scope="thread",
                thread_id=digest.thread_id,
                topic_key=digest.key,
                participants=join_memory_participants(digest.participants),
                source_dialog_id=None,
                source_message=digest.source_message,
            )
            await _upsert_vector_if_needed(
                row=row,
                memory_type=candidate.memory_type,
                key=candidate.key,
                value=candidate.value,
                session_id=digest.session_id,
                user_id=digest.user_id,
                group_id=digest.group_id,
                scope="thread",
                thread_id=digest.thread_id,
                topic_key=digest.key,
                participants=digest.participants,
                confidence=candidate.confidence,
            )
            return 1
        except Exception as exc:
            _debug(f"chatinter group memory digest skipped: {exc}")
            return 0

    @staticmethod
    async def recall(
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        query: str,
        limit: int = _MEMORY_LIMIT,
        recall_context: MemoryRecallContext | None = None,
    ) -> list[str]:
        memory_model = _get_memory_model()
        if memory_model is None:
            return []
        resolved_context = recall_context or MemoryRecallContext.build(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            query=query,
        )
        try:
            memories = await memory_model.recall_memories(
                session_id=resolved_context.session_id,
                user_id=resolved_context.user_id,
                group_id=resolved_context.group_id,
                query=resolved_context.query or query,
                limit=limit,
                thread_id=resolved_context.thread_id,
                topic_key=resolved_context.topic_key,
                participants=resolved_context.participants,
                addressee_user_id=resolved_context.addressee_user_id,
            )
            vector_results = await MemoryVectorIndex.search_memory_vectors(
                query=resolved_context.query or query,
                recall_context=resolved_context,
                top_k=limit,
            )
            memories = await _merge_vector_memories(
                memory_model=memory_model,
                structured_memories=memories,
                vector_results=vector_results,
                recall_context=resolved_context,
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
