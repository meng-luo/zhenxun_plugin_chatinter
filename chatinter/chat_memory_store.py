from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any

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

_MEMORY_LIMIT = 4
_MEMORY_CONFIDENCE_DEFAULT = 0.72
_PRIVATE_RECALL_THRESHOLD = 0.24
_GROUP_RECALL_THRESHOLD = 0.38
_GROUP_CONTEXT_RECALL_THRESHOLD = 0.26
_RECENT_WRITE_CACHE_TTL = 60.0
_RECENT_WRITE_CACHE_MAX = 512
_recent_writes: dict[str, float] = {}
_TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_PROFILE_MEMORY_TYPES = {
    "nickname",
    "correction",
    "person_profile_summary",
    "preference",
    "relationship",
}
_TYPE_TEXT = {
    "nickname": "昵称 称呼 叫我",
    "correction": "称呼 更正 别叫我",
    "person_profile_summary": "个人 信息",
    "preference": "偏好 喜欢 不喜欢",
    "relationship": "关系",
    "group_digest": "群聊 总结 之前",
    "thread_digest": "话题 总结 之前",
    "thread_fact": "话题 事实",
    "recent_thread_fact": "话题 事实 刚才",
}


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
    supersedes: bool = False


@dataclass(frozen=True)
class LayeredMemoryRecall:
    person_facts: tuple[str, ...] = ()
    relationship_facts: tuple[str, ...] = ()
    preference_facts: tuple[str, ...] = ()
    recent_thread_facts: tuple[str, ...] = ()
    other_facts: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.person_facts,
                self.relationship_facts,
                self.preference_facts,
                self.recent_thread_facts,
                self.other_facts,
            )
        )

    def to_xml_lines(self) -> list[str]:
        sections = (
            ("person_facts", self.person_facts),
            ("relationship_facts", self.relationship_facts),
            ("preference_facts", self.preference_facts),
            ("recent_thread_facts", self.recent_thread_facts),
            ("other_facts", self.other_facts),
        )
        lines: list[str] = []
        for tag, values in sections:
            if not values:
                continue
            lines.append(f"<{tag}>")
            lines.extend(values)
            lines.append(f"</{tag}>")
        return lines

    def flatten(self) -> list[str]:
        return [
            *self.person_facts,
            *self.relationship_facts,
            *self.preference_facts,
            *self.recent_thread_facts,
            *self.other_facts,
        ]

@dataclass
class _LayerBuckets:
    person_facts: list[str] = field(default_factory=list)
    relationship_facts: list[str] = field(default_factory=list)
    preference_facts: list[str] = field(default_factory=list)
    recent_thread_facts: list[str] = field(default_factory=list)
    other_facts: list[str] = field(default_factory=list)

    def freeze(self) -> LayeredMemoryRecall:
        return LayeredMemoryRecall(
            person_facts=tuple(self.person_facts),
            relationship_facts=tuple(self.relationship_facts),
            preference_facts=tuple(self.preference_facts),
            recent_thread_facts=tuple(self.recent_thread_facts),
            other_facts=tuple(self.other_facts),
        )


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
    relevant_rows: list[Any] = []
    for row in rows:
        relevance_score = _memory_relevance_score(row, recall_context=recall_context)
        if relevance_score < _memory_relevance_threshold(
            row,
            recall_context=recall_context,
        ):
            continue
        setattr(row, "_chatinter_relevance_score", relevance_score)
        relevant_rows.append(row)
    if not relevant_rows:
        return []
    prompt_limit = min(limit, 3 if recall_context.group_id else 4)
    ranked = sorted(
        relevant_rows,
        key=lambda row: (
            float(getattr(row, "_chatinter_relevance_score", 0.0) or 0.0),
            _memory_rank_score(
                row,
                recall_context=recall_context,
                base_scores=base_scores,
            )
            * 0.02,
            float(getattr(row, "confidence", 0.0) or 0.0),
            _memory_recency(row),
            int(getattr(row, "id", 0) or 0),
        ),
        reverse=True,
    )
    return ranked[:prompt_limit]


def _memory_relevance_threshold(
    row: Any,
    *,
    recall_context: MemoryRecallContext,
) -> float:
    if not recall_context.group_id:
        return _PRIVATE_RECALL_THRESHOLD
    if (
        normalize_message_text(getattr(row, "scope", "")) == "user"
        and not normalize_message_text(getattr(row, "group_id", "") or "")
        and _looks_like_personal_memory_question(recall_context.query)
    ):
        return _PRIVATE_RECALL_THRESHOLD
    if _has_thread_context_match(row, recall_context=recall_context):
        return _GROUP_CONTEXT_RECALL_THRESHOLD
    return _GROUP_RECALL_THRESHOLD


def _memory_relevance_score(
    row: Any,
    *,
    recall_context: MemoryRecallContext,
) -> float:
    vector_score = max(
        min(float(getattr(row, "_chatinter_vector_score", 0.0) or 0.0), 1.0),
        0.0,
    )
    vector_type = str(getattr(row, "_chatinter_vector_type", "") or "")
    semantic_score = vector_score if vector_type == "embedding" else 0.0
    lexical_score = _lexical_overlap_score(
        recall_context.query,
        _memory_search_text(row),
    )
    if semantic_score > 0:
        score = semantic_score * 0.62 + lexical_score * 0.18
    else:
        score = lexical_score * 0.56
    score += _scope_relevance(row, recall_context=recall_context)
    score += _context_relevance(row, recall_context=recall_context)
    score += _type_prior(row, query=recall_context.query)
    return max(min(score, 1.0), 0.0)


def _memory_search_text(row: Any) -> str:
    memory_type = normalize_message_text(getattr(row, "memory_type", ""))
    parts = [
        _TYPE_TEXT.get(memory_type, memory_type),
        getattr(row, "key", ""),
        getattr(row, "value", ""),
        getattr(row, "topic_key", ""),
        getattr(row, "source_message", ""),
    ]
    return " ".join(normalize_message_text(str(part)) for part in parts if part)


def _scope_relevance(
    row: Any,
    *,
    recall_context: MemoryRecallContext,
) -> float:
    score = 0.0
    if normalize_message_text(getattr(row, "user_id", "")) == recall_context.user_id:
        score += 0.08
    if (
        normalize_message_text(getattr(row, "session_id", ""))
        == recall_context.session_id
    ):
        score += 0.06
    row_group_id = normalize_message_text(getattr(row, "group_id", "") or "") or None
    if recall_context.group_id and row_group_id == recall_context.group_id:
        score += 0.08
    if recall_context.group_id and row_group_id is None:
        score += 0.04
    return score


def _context_relevance(
    row: Any,
    *,
    recall_context: MemoryRecallContext,
) -> float:
    score = 0.0
    row_thread_id = normalize_message_text(getattr(row, "thread_id", "") or "")
    row_topic_key = normalize_message_text(getattr(row, "topic_key", "") or "")
    row_participants = {
        item for item in str(getattr(row, "participants", "") or "").split(",") if item
    }
    if recall_context.thread_id and row_thread_id == recall_context.thread_id:
        score += 0.22
    elif recall_context.thread_id and row_thread_id:
        score -= 0.12
    if recall_context.topic_key and row_topic_key == recall_context.topic_key:
        score += 0.1
    if recall_context.participants and row_participants:
        overlap = len(set(recall_context.participants) & row_participants)
        if overlap:
            score += min(overlap, 2) * 0.05
    if (
        recall_context.addressee_user_id
        and recall_context.addressee_user_id in row_participants
    ):
        score += 0.08
    row_scope = normalize_message_text(getattr(row, "scope", ""))
    if recall_context.group_id and row_scope == "thread" and not score:
        score -= 0.08
    return score


def _type_prior(row: Any, *, query: str) -> float:
    memory_type = normalize_message_text(getattr(row, "memory_type", ""))
    if memory_type in _PROFILE_MEMORY_TYPES:
        return 0.12 if _looks_like_personal_memory_question(query) else 0.04
    if memory_type in {"group_digest", "thread_digest", "thread_fact"}:
        return 0.03
    return 0.0


def _has_thread_context_match(
    row: Any,
    *,
    recall_context: MemoryRecallContext,
) -> bool:
    row_thread_id = normalize_message_text(getattr(row, "thread_id", "") or "")
    row_topic_key = normalize_message_text(getattr(row, "topic_key", "") or "")
    if recall_context.thread_id and row_thread_id == recall_context.thread_id:
        return True
    if recall_context.topic_key and row_topic_key == recall_context.topic_key:
        return True
    row_participants = {
        item for item in str(getattr(row, "participants", "") or "").split(",") if item
    }
    return bool(
        row_participants
        and (
            set(recall_context.participants) & row_participants
            or recall_context.addressee_user_id in row_participants
        )
    )


def _looks_like_personal_memory_question(query: str) -> bool:
    normalized = normalize_message_text(query)
    if "我" not in normalized:
        return False
    return normalized.endswith(("?", "？", "吗", "么", "嘛")) or any(
        marker in normalized
        for marker in ("什么", "哪个", "哪种", "谁", "多少", "来着")
    )


def _lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = _recall_text_tokens(query)
    text_tokens = _recall_text_tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    if not overlap:
        compact_query = "".join(query_tokens)
        compact_text = "".join(text_tokens)
        if len(compact_query) >= 2 and compact_query in compact_text:
            return 0.45
        if len(compact_text) >= 2 and compact_text in compact_query:
            return 0.45
        return 0.0
    return min(len(overlap) / max(min(len(query_tokens), len(text_tokens)), 1), 1.0)


def _recall_text_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in _TEXT_TOKEN_PATTERN.findall(normalize_message_text(text)):
        token = raw_token.casefold()
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.update(_cjk_ngrams(token))
        elif len(token) >= 2:
            tokens.add(token)
    return tokens


def _cjk_ngrams(token: str) -> set[str]:
    if len(token) <= 2:
        return {token}
    result: set[str] = set()
    for size in (2, 3):
        result.update(
            token[index : index + size]
            for index in range(len(token) - size + 1)
        )
    return result


def _memory_recency(row: Any) -> float:
    value = getattr(row, "update_time", None) or getattr(row, "create_time", None)
    if value is None:
        return 0.0
    try:
        return float(value.timestamp())
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0


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
                    replace_existing=candidate.supersedes,
                )
                if row is None:
                    continue
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
    async def recall(
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        query: str,
        limit: int = _MEMORY_LIMIT,
        recall_context: MemoryRecallContext | None = None,
    ) -> list[str]:
        memories = await ChatMemoryStore._recall_rows(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            query=query,
            limit=limit,
            recall_context=recall_context,
        )
        lines: list[str] = []
        for memory in memories:
            line = _format_memory_line(memory)
            if line:
                lines.append(line)
        return lines[: max(int(limit or 0), 0)]

    @staticmethod
    async def recall_layered(
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        query: str,
        limit: int = _MEMORY_LIMIT,
        recall_context: MemoryRecallContext | None = None,
    ) -> LayeredMemoryRecall:
        memories = await ChatMemoryStore._recall_rows(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            query=query,
            limit=limit,
            recall_context=recall_context,
        )
        buckets = _LayerBuckets()
        for memory in memories:
            line = _format_memory_line(memory)
            if not line:
                continue
            _append_layered_memory(buckets, memory=memory, line=line)
        return buckets.freeze()

    @staticmethod
    async def _recall_rows(
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        query: str,
        limit: int,
        recall_context: MemoryRecallContext | None,
    ) -> list[Any]:
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
            return await _merge_vector_memories(
                memory_model=memory_model,
                structured_memories=memories,
                vector_results=vector_results,
                recall_context=resolved_context,
                limit=limit,
            )
        except Exception as exc:
            _debug(f"chatinter memory recall skipped: {exc}")
            return []


def _format_memory_line(memory: Any) -> str:
    memory_type = normalize_message_text(getattr(memory, "memory_type", ""))
    key = normalize_message_text(getattr(memory, "key", ""))
    value = normalize_message_text(getattr(memory, "value", ""))
    if not value:
        return ""
    if memory_type == "preference":
        if key == "dislike":
            return f"用户不喜欢：{value}"
        return f"用户偏好：{value}"
    if memory_type == "nickname":
        return f"用户称呼：{value}"
    if memory_type == "correction":
        return f"称呼更正：{value}"
    if memory_type == "relationship":
        relation = value.replace("=", "是", 1)
        return f"关系信息：{relation}"
    if memory_type in {"person_profile_summary", "profile"}:
        return f"用户信息：{value}"
    if memory_type in {"group_digest", "thread_digest", "thread_fact"}:
        return f"话题摘要：{value}"
    label = f"{memory_type}:{key}".strip(":")
    return f"{label}={value}" if label else value


def _append_layered_memory(
    buckets: _LayerBuckets,
    *,
    memory: Any,
    line: str,
) -> None:
    memory_type = normalize_message_text(getattr(memory, "memory_type", ""))
    key = normalize_message_text(getattr(memory, "key", ""))
    scope = normalize_message_text(getattr(memory, "scope", ""))
    target: list[str]
    if memory_type in {
        "nickname",
        "correction",
        "person_profile_summary",
    }:
        target = buckets.person_facts
    elif memory_type == "relationship" or key.startswith("relationship"):
        target = buckets.relationship_facts
    elif memory_type == "preference":
        target = buckets.preference_facts
    elif (
        memory_type
        in {
            "group_digest",
            "thread_digest",
            "thread_fact",
            "recent_thread_fact",
        }
        or scope == "thread"
    ):
        target = buckets.recent_thread_facts
    else:
        target = buckets.other_facts
    if line not in target:
        target.append(line)


__all__ = [
    "ChatMemoryStore",
    "LayeredMemoryRecall",
    "MemoryCandidate",
]
