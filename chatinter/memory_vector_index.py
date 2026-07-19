from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, ClassVar, cast

from .config import MEMORY_VECTOR_MAX_ITEMS
from .memory_recall_context import MemoryRecallContext, split_memory_participants
from .route_text import normalize_message_text
from .vector_store import VectorDocument, VectorStore, create_vector_store

_INDEX_PATH = Path("data/cache/chatinter/memory_vector_index.json")
_FALLBACK_DIM = 384
_EMBEDDING_COOLDOWN = 120.0
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_INDEXABLE_TYPES = {
    "nickname",
    "correction",
    "preference",
    "relationship",
    "person_profile_summary",
    "thread_digest",
    "group_digest",
}
_SEARCH_LIMIT = 16


@dataclass(frozen=True)
class MemoryVectorMetadata:
    memory_id: int
    session_id: str
    user_id: str
    group_id: str | None
    memory_type: str
    scope: str
    thread_id: str | None = None
    topic_key: str = ""
    participants: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class MemoryVectorSearchResult:
    memory_id: int
    score: float
    vector_type: str


@dataclass
class _MemoryVectorDoc:
    memory_id: int
    text_hash: str
    vector: list[float]
    vector_type: str
    metadata: MemoryVectorMetadata
    updated_at: float = 0.0
    accessed_at: float = 0.0
    importance: float = 0.0


class MemoryVectorIndex:
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _loaded: ClassVar[bool] = False
    _docs: ClassVar[dict[int, _MemoryVectorDoc]] = {}
    _store: ClassVar[VectorStore] = create_vector_store("memory")
    _embedding_supported: ClassVar[bool | None] = None
    _embedding_disabled_until: ClassVar[float] = 0.0

    @classmethod
    def is_indexable_type(cls, memory_type: str) -> bool:
        return normalize_message_text(memory_type) in _INDEXABLE_TYPES

    @classmethod
    async def upsert_memory_vector(
        cls,
        *,
        memory_id: int,
        text: str,
        metadata: MemoryVectorMetadata,
    ) -> bool:
        if memory_id <= 0 or not cls.is_indexable_type(metadata.memory_type):
            return False
        normalized_text = _build_index_text(text=text, metadata=metadata)
        if not normalized_text:
            return False
        text_hash = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
        async with cls._lock:
            await cls._load()
            existing = cls._docs.get(memory_id)
            if existing and existing.text_hash == text_hash:
                return False
            vectors, vector_type = await cls._embed_documents([normalized_text])
            if not vectors:
                return False
            cls._docs[memory_id] = _MemoryVectorDoc(
                memory_id=memory_id,
                text_hash=text_hash,
                vector=vectors[0],
                vector_type=vector_type,
                metadata=metadata,
                updated_at=time.time(),
                accessed_at=time.time(),
                importance=_memory_importance(metadata),
            )
            cls._store.upsert(_doc_to_vector_document(cls._docs[memory_id]))
            cls._prune_nolock()
            await cls._save()
            return True

    @classmethod
    async def search_memory_vectors(
        cls,
        *,
        query: str,
        recall_context: MemoryRecallContext,
        top_k: int = _SEARCH_LIMIT,
    ) -> list[MemoryVectorSearchResult]:
        normalized_query = normalize_message_text(query or recall_context.query)
        if not normalized_query:
            return []
        async with cls._lock:
            await cls._load()
            if not cls._docs:
                return []
            query_vector = await cls._embed_query(normalized_query)
            if not query_vector:
                return []
            limit = max(int(top_k or 0), 0)
            store_results = cls._store.search(
                query_vector,
                top_k=limit,
                filter_fn=lambda doc: _matches_context(
                    _metadata_for_store_doc(cls._docs, doc),
                    recall_context,
                ),
                boost_fn=lambda doc: _structured_boost(
                    _metadata_for_store_doc(cls._docs, doc),
                    recall_context,
                ),
            )
            now = time.time()
            ranked: list[MemoryVectorSearchResult] = []
            for result in store_results:
                try:
                    memory_id = int(result.doc_id)
                except (TypeError, ValueError):
                    continue
                doc = cls._docs.get(memory_id)
                if doc is None:
                    continue
                doc.accessed_at = now
                ranked.append(
                    MemoryVectorSearchResult(
                        memory_id=memory_id,
                        score=result.score,
                        vector_type=result.vector_type,
                    )
                )
            return ranked

    @classmethod
    async def delete_memory_vector(cls, memory_id: int) -> None:
        async with cls._lock:
            await cls._load()
            if cls._docs.pop(int(memory_id), None) is not None:
                cls._store.delete(str(memory_id))
                await cls._save()

    @classmethod
    async def _load(cls) -> None:
        if cls._loaded:
            return

        def _read() -> dict:
            if not _INDEX_PATH.exists():
                return {}
            try:
                payload = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
            return payload if isinstance(payload, dict) else {}

        payload = await asyncio.to_thread(_read)
        docs = payload.get("docs", {})
        if isinstance(docs, dict):
            for raw_id, item in docs.items():
                if not isinstance(item, dict):
                    continue
                try:
                    memory_id = int(raw_id)
                    metadata = _metadata_from_payload(item.get("metadata", {}))
                    if not cls.is_indexable_type(metadata.memory_type):
                        continue
                    vector = item.get("vector", [])
                    if not isinstance(vector, list) or not vector:
                        continue
                    cls._docs[memory_id] = _MemoryVectorDoc(
                        memory_id=memory_id,
                        text_hash=str(item.get("text_hash", "")),
                        vector=[float(v) for v in vector],
                        vector_type=str(item.get("vector_type", "fallback")),
                        metadata=metadata,
                        updated_at=float(item.get("updated_at", 0.0) or 0.0),
                        accessed_at=float(item.get("accessed_at", 0.0) or 0.0),
                        importance=float(
                            item.get("importance", _memory_importance(metadata)) or 0.0
                        ),
                    )
                except Exception:
                    continue
        cls._prune_nolock()
        cls._rebuild_store_nolock()
        cls._loaded = True

    @classmethod
    async def _save(cls) -> None:
        payload = {
            "version": 1,
            "docs": {
                str(memory_id): {
                    "text_hash": doc.text_hash,
                    "vector": doc.vector,
                    "vector_type": doc.vector_type,
                    "metadata": _metadata_to_payload(doc.metadata),
                    "updated_at": doc.updated_at,
                    "accessed_at": doc.accessed_at,
                    "importance": doc.importance,
                }
                for memory_id, doc in cls._docs.items()
            },
        }

        def _write() -> None:
            _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            _INDEX_PATH.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

        await asyncio.to_thread(_write)

    @classmethod
    async def _embed_documents(cls, texts: list[str]) -> tuple[list[list[float]], str]:
        if not texts:
            return [], "fallback"
        if cls._embedding_supported is None:
            cls._embedding_supported = _has_embedding_models()
        if (
            not cls._embedding_supported
            or time.monotonic() < cls._embedding_disabled_until
        ):
            return [_fallback_vector(text) for text in texts], "fallback"
        try:
            embed_documents = _get_llm_func("embed_documents")
            if embed_documents is None:
                raise RuntimeError("embedding api unavailable")
            vectors = await embed_documents(texts)
            if not vectors or len(vectors) != len(texts):
                raise RuntimeError("memory embedding result invalid")
            return (
                [_normalize_vector([float(v) for v in vector]) for vector in vectors],
                "embedding",
            )
        except Exception as exc:
            cls._embedding_disabled_until = time.monotonic() + _EMBEDDING_COOLDOWN
            _debug(f"chatinter memory embedding fallback: {exc}")
            return [_fallback_vector(text) for text in texts], "fallback"

    @classmethod
    async def _embed_query(cls, text: str) -> list[float]:
        if not text.strip():
            return []
        if cls._embedding_supported is None:
            cls._embedding_supported = _has_embedding_models()
        if (
            not cls._embedding_supported
            or time.monotonic() < cls._embedding_disabled_until
        ):
            return _fallback_vector(text)
        try:
            embed_query = _get_llm_func("embed_query")
            if embed_query is None:
                raise RuntimeError("embedding api unavailable")
            vector = await embed_query(text)
            return _normalize_vector([float(v) for v in vector])
        except Exception:
            cls._embedding_disabled_until = time.monotonic() + _EMBEDDING_COOLDOWN
            return _fallback_vector(text)

    @classmethod
    def _prune_nolock(cls) -> None:
        max_docs = max(int(MEMORY_VECTOR_MAX_ITEMS or 0), 0)
        if max_docs <= 0 or len(cls._docs) <= max_docs:
            return
        overflow = len(cls._docs) - max_docs
        oldest_ids = sorted(
            cls._docs,
            key=lambda memory_id: (
                cls._docs[memory_id].importance,
                cls._docs[memory_id].accessed_at or cls._docs[memory_id].updated_at,
                cls._docs[memory_id].updated_at,
                memory_id,
            ),
        )[:overflow]
        for memory_id in oldest_ids:
            cls._docs.pop(memory_id, None)
            cls._store.delete(str(memory_id))

    @classmethod
    def _rebuild_store_nolock(cls) -> None:
        cls._store.load_documents(
            _doc_to_vector_document(doc) for doc in cls._docs.values()
        )


def build_memory_vector_text(
    *,
    memory_type: str,
    key: str,
    value: str,
    metadata: MemoryVectorMetadata,
) -> str:
    lines = [
        f"类型: {memory_type}",
        f"话题: {key or metadata.topic_key}",
        f"内容: {value}",
    ]
    if metadata.group_id:
        lines.append(f"群: {metadata.group_id}")
    if metadata.thread_id:
        lines.append(f"线程: {metadata.thread_id}")
    if metadata.participants:
        lines.append("参与者: " + ",".join(metadata.participants))
    return "\n".join(lines)


def _build_index_text(*, text: str, metadata: MemoryVectorMetadata) -> str:
    return normalize_message_text(text)[:1200]


def _metadata_to_payload(metadata: MemoryVectorMetadata) -> dict:
    payload = asdict(metadata)
    payload["participants"] = list(metadata.participants)
    return payload


def _metadata_from_payload(payload: object) -> MemoryVectorMetadata:
    data = payload if isinstance(payload, dict) else {}
    participants = data.get("participants", ())
    if isinstance(participants, str):
        parsed_participants = split_memory_participants(participants)
    elif isinstance(participants, list):
        parsed_participants = tuple(str(item) for item in participants if str(item))
    else:
        parsed_participants = ()
    return MemoryVectorMetadata(
        memory_id=int(data.get("memory_id", 0) or 0),
        session_id=str(data.get("session_id", "") or ""),
        user_id=str(data.get("user_id", "") or ""),
        group_id=str(data.get("group_id") or "") or None,
        memory_type=str(data.get("memory_type", "") or ""),
        scope=str(data.get("scope", "") or ""),
        thread_id=str(data.get("thread_id") or "") or None,
        topic_key=str(data.get("topic_key", "") or ""),
        participants=parsed_participants,
        confidence=float(data.get("confidence", 0.0) or 0.0),
    )


def _matches_context(
    metadata: MemoryVectorMetadata,
    context: MemoryRecallContext,
) -> bool:
    if metadata.scope == "user":
        return metadata.user_id == context.user_id and (
            metadata.group_id is None or metadata.group_id == context.group_id
        )
    if metadata.scope in {"group", "thread"}:
        return bool(context.group_id and metadata.group_id == context.group_id)
    return False


def _structured_boost(
    metadata: MemoryVectorMetadata,
    context: MemoryRecallContext,
) -> float:
    score = 0.0
    if context.thread_id and metadata.thread_id == context.thread_id:
        score += 0.28
    elif context.thread_id and metadata.thread_id:
        score -= 0.18
    if context.topic_key and metadata.topic_key == context.topic_key:
        score += 0.12
    participant_set = set(context.participants)
    metadata_participants = set(metadata.participants)
    if participant_set and metadata_participants:
        overlap = len(participant_set & metadata_participants)
        if overlap:
            score += min(overlap, 3) * 0.08
    if context.addressee_user_id and context.addressee_user_id in metadata_participants:
        score += 0.12
    score += min(max(metadata.confidence, 0.0), 1.0) * 0.05
    return score


def _memory_importance(metadata: MemoryVectorMetadata) -> float:
    type_weight = {
        "person_profile_summary": 0.35,
        "thread_digest": 0.25,
        "group_digest": 0.15,
    }.get(normalize_message_text(metadata.memory_type), 0.0)
    scope_weight = 0.1 if metadata.scope in {"user", "thread"} else 0.0
    confidence_weight = min(max(metadata.confidence, 0.0), 1.0) * 0.25
    participant_weight = min(len(metadata.participants), 4) * 0.02
    return type_weight + scope_weight + confidence_weight + participant_weight


def _doc_to_vector_document(doc: _MemoryVectorDoc) -> VectorDocument:
    return VectorDocument(
        doc_id=str(doc.memory_id),
        vector=doc.vector,
        vector_type=doc.vector_type,
        metadata=_metadata_to_payload(doc.metadata),
        updated_at=doc.updated_at,
        accessed_at=doc.accessed_at or doc.updated_at,
        importance=doc.importance,
    )


def _metadata_for_store_doc(
    docs: dict[int, _MemoryVectorDoc],
    store_doc: VectorDocument,
) -> MemoryVectorMetadata:
    try:
        memory_id = int(store_doc.doc_id)
    except (TypeError, ValueError):
        memory_id = 0
    doc = docs.get(memory_id)
    if doc is not None:
        return doc.metadata
    return _metadata_from_payload(store_doc.metadata)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(text or ""):
        normalized = token.lower()
        if normalized:
            tokens.append(normalized)
            tokens.extend(_chinese_ngrams(normalized))
    return tokens


def _chinese_ngrams(token: str) -> list[str]:
    chars = [char for char in token if "\u4e00" <= char <= "\u9fff"]
    if len(chars) < 2:
        return []
    text = "".join(chars)
    grams: list[str] = []
    for size in (2, 3):
        if len(text) < size:
            continue
        grams.extend(
            text[index : index + size] for index in range(len(text) - size + 1)
        )
    return grams


def _normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return values
    return [v / norm for v in values]


def _cosine_score(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=False))


def _fallback_vector(text: str, dim: int = _FALLBACK_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big", signed=False) % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return _normalize_vector(vector)


def _get_llm_func(name: str) -> Any | None:
    try:
        from .llm_compat import __dict__ as _llm_api

        return _llm_api.get(name)
    except Exception as exc:
        _debug(f"chatinter memory embedding api unavailable: {exc}")
        return None


def _has_embedding_models() -> bool:
    list_models = _get_llm_func("list_embedding_models")
    if list_models is None:
        return False
    try:
        return bool(cast(Any, list_models)())
    except Exception as exc:
        _debug(f"chatinter memory embedding model check failed: {exc}")
        return False


def _debug(message: str) -> None:
    try:
        from zhenxun.services.log import logger

        logger.debug(message)
    except Exception:
        pass


__all__ = [
    "MemoryVectorIndex",
    "MemoryVectorMetadata",
    "MemoryVectorSearchResult",
    "build_memory_vector_text",
]
