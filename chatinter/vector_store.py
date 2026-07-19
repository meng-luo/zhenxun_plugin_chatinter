from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import math
import time
from typing import Any, Protocol


@dataclass(slots=True)
class VectorDocument:
    """Backend-neutral vector document used by memory and plugin RAG."""

    doc_id: str
    vector: list[float]
    vector_type: str = "fallback"
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0
    accessed_at: float = 0.0
    importance: float = 0.0


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    doc_id: str
    score: float
    vector_type: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def load_documents(self, documents: Iterable[VectorDocument]) -> None: ...

    def upsert(self, document: VectorDocument) -> None: ...

    def delete(self, doc_id: str) -> None: ...

    def get(self, doc_id: str) -> VectorDocument | None: ...

    def documents(self) -> tuple[VectorDocument, ...]: ...

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filter_fn: Callable[[VectorDocument], bool] | None = None,
        boost_fn: Callable[[VectorDocument], float] | None = None,
    ) -> list[VectorSearchResult]: ...


class InMemoryVectorStore:
    """Small local fallback store.

    The abstraction keeps callers independent from the physical backend.  A
    future sqlite-vec/FAISS implementation only needs to implement VectorStore.
    """

    def __init__(self) -> None:
        self._docs: dict[str, VectorDocument] = {}

    def load_documents(self, documents: Iterable[VectorDocument]) -> None:
        self._docs = {document.doc_id: document for document in documents}

    def upsert(self, document: VectorDocument) -> None:
        self._docs[document.doc_id] = document

    def delete(self, doc_id: str) -> None:
        self._docs.pop(str(doc_id), None)

    def get(self, doc_id: str) -> VectorDocument | None:
        return self._docs.get(str(doc_id))

    def documents(self) -> tuple[VectorDocument, ...]:
        return tuple(self._docs.values())

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filter_fn: Callable[[VectorDocument], bool] | None = None,
        boost_fn: Callable[[VectorDocument], float] | None = None,
    ) -> list[VectorSearchResult]:
        if not query_vector or top_k <= 0:
            return []
        ranked: list[VectorSearchResult] = []
        now = time.time()
        for document in self._docs.values():
            if filter_fn is not None and not filter_fn(document):
                continue
            score = cosine_score(query_vector, document.vector)
            if boost_fn is not None:
                score += float(boost_fn(document) or 0.0)
            if score <= 0:
                continue
            document.accessed_at = now
            ranked.append(
                VectorSearchResult(
                    doc_id=document.doc_id,
                    score=score,
                    vector_type=document.vector_type,
                    metadata=dict(document.metadata),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]


class BM25LexicalIndex:
    """Lightweight sparse index for local RAG reranking."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._term_freqs: dict[str, Counter[str]] = {}
        self._doc_freqs: Counter[str] = Counter()
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_len: float = 0.0

    def rebuild(self, documents: Mapping[str, Iterable[str]]) -> None:
        self._term_freqs.clear()
        self._doc_freqs.clear()
        self._doc_lengths.clear()
        total_length = 0
        for doc_id, tokens_iter in documents.items():
            tokens = [token for token in tokens_iter if token]
            counter = Counter(tokens)
            if not counter:
                continue
            self._term_freqs[str(doc_id)] = counter
            length = sum(counter.values())
            self._doc_lengths[str(doc_id)] = length
            total_length += length
            for token in counter:
                self._doc_freqs[token] += 1
        doc_count = max(len(self._term_freqs), 1)
        self._avg_doc_len = total_length / doc_count if total_length else 0.0

    def score_all(
        self,
        query_tokens: Iterable[str],
        *,
        filter_ids: set[str] | None = None,
    ) -> dict[str, float]:
        query_counter = Counter(token for token in query_tokens if token)
        if not query_counter or not self._term_freqs:
            return {}
        scores: dict[str, float] = defaultdict(float)
        doc_count = len(self._term_freqs)
        avg_len = self._avg_doc_len or 1.0
        target_ids = filter_ids if filter_ids is not None else set(self._term_freqs)
        if not target_ids:
            return {}
        for token, query_weight in query_counter.items():
            doc_freq = self._doc_freqs.get(token, 0)
            if doc_freq <= 0:
                continue
            idf = math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            for doc_id in target_ids:
                term_freq = self._term_freqs.get(doc_id, {}).get(token, 0)
                if term_freq <= 0:
                    continue
                doc_len = self._doc_lengths.get(doc_id, 0) or avg_len
                denom = term_freq + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
                scores[doc_id] += (
                    float(query_weight)
                    * idf
                    * (term_freq * (self.k1 + 1.0))
                    / max(denom, 1e-9)
                )
        return dict(scores)


def cosine_score(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=False))


def sqlite_vec_available() -> bool:
    """Whether the optional ``sqlite_vec`` disk-KNN backend can be loaded."""
    import importlib.util

    return importlib.util.find_spec("sqlite_vec") is not None


class SqliteVecStore:
    """Optional disk-backed KNN store.

    Activated only when the ``sqlite_vec`` extension is installed; it implements
    the same ``VectorStore`` surface as :class:`InMemoryVectorStore` but keeps
    vectors in a ``vec0`` virtual table so the working set is not fully resident
    in process memory and survives restarts without a JSON re-parse.

    Vectors of differing dimensionality (a real embedding vs. the 384-dim hash
    fallback) cannot share one ``vec0`` table; this store binds to the first
    dimension it sees and skips mismatched documents (logged), so callers that
    mix embedding/fallback vectors should prefer the in-memory store until a
    real embedding backend is guaranteed.

    NOTE: the ``vec0`` path is exercised only when ``sqlite_vec`` is present in
    the environment; in environments without it, ``create_vector_store`` returns
    :class:`InMemoryVectorStore` and this class is never instantiated.
    """

    def __init__(self, db_path: str, *, dim: int | None = None) -> None:
        import sqlite3

        import sqlite_vec  # type: ignore[import-not-found]

        self._db_path = db_path
        self._dim = dim
        self._meta: dict[str, VectorDocument] = {}
        self._conn = sqlite3.connect(db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_meta("
            "doc_id TEXT PRIMARY KEY, vector_type TEXT, text TEXT, "
            "metadata TEXT, updated_at REAL, accessed_at REAL, importance REAL)"
        )
        if dim:
            self._ensure_table(dim)

    def _ensure_table(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items "
            f"USING vec0(doc_id TEXT PRIMARY KEY, embedding float[{self._dim}])"
        )

    def load_documents(self, documents: Iterable[VectorDocument]) -> None:
        for document in documents:
            self.upsert(document)

    def upsert(self, document: VectorDocument) -> None:
        import json as _json

        vector = list(document.vector or [])
        if not vector:
            return
        if self._dim is None:
            self._ensure_table(len(vector))
        if len(vector) != self._dim:
            from zhenxun.services.log import logger

            logger.warning(
                f"SqliteVecStore 跳过维度不匹配文档 {document.doc_id}: "
                f"{len(vector)} != {self._dim}",
                "ChatInterVectorStore",
            )
            return
        self._meta[document.doc_id] = document
        self._conn.execute("DELETE FROM vec_items WHERE doc_id = ?", (document.doc_id,))
        self._conn.execute(
            "INSERT INTO vec_items(doc_id, embedding) VALUES (?, ?)",
            (document.doc_id, _json.dumps(vector)),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO vec_meta VALUES (?,?,?,?,?,?,?)",
            (
                document.doc_id,
                document.vector_type,
                document.text,
                _json.dumps(document.metadata, ensure_ascii=False),
                document.updated_at,
                document.accessed_at,
                document.importance,
            ),
        )
        self._conn.commit()

    def delete(self, doc_id: str) -> None:
        self._meta.pop(str(doc_id), None)
        self._conn.execute("DELETE FROM vec_items WHERE doc_id = ?", (str(doc_id),))
        self._conn.execute("DELETE FROM vec_meta WHERE doc_id = ?", (str(doc_id),))
        self._conn.commit()

    def get(self, doc_id: str) -> VectorDocument | None:
        return self._meta.get(str(doc_id))

    def documents(self) -> tuple[VectorDocument, ...]:
        return tuple(self._meta.values())

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filter_fn: Callable[[VectorDocument], bool] | None = None,
        boost_fn: Callable[[VectorDocument], float] | None = None,
    ) -> list[VectorSearchResult]:
        import json as _json

        if not query_vector or top_k <= 0 or self._dim is None:
            return []
        if len(query_vector) != self._dim:
            return []

        fetch = top_k * 4 if (filter_fn or boost_fn) else top_k
        rows = self._conn.execute(
            "SELECT doc_id, distance FROM vec_items "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (_json.dumps(list(query_vector)), fetch),
        ).fetchall()
        results: list[VectorSearchResult] = []
        for doc_id, distance in rows:
            document = self._meta.get(str(doc_id))
            if document is None:
                continue
            if filter_fn is not None and not filter_fn(document):
                continue
            score = 1.0 - float(distance)
            if boost_fn is not None:
                score += float(boost_fn(document) or 0.0)
            if score <= 0:
                continue
            results.append(
                VectorSearchResult(
                    doc_id=str(doc_id),
                    score=score,
                    vector_type=document.vector_type,
                    metadata=dict(document.metadata),
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


def create_vector_store(
    namespace: str = "default", *, dim: int | None = None
) -> VectorStore:
    """Return the best available vector backend.

    Prefers the disk-backed ``SqliteVecStore`` when ``sqlite_vec`` is installed
    and the ``CHATINTER_VECTOR_BACKEND`` config is ``auto``/``sqlite``; otherwise
    falls back to the in-memory store.
    """
    backend = "auto"
    try:
        from zhenxun.configs.config import Config

        backend = str(
            Config.get_config("chatinter", "VECTOR_BACKEND", "auto") or "auto"
        ).strip().lower()
    except Exception:
        backend = "auto"

    if backend in {"memory", "inmemory"}:
        return InMemoryVectorStore()
    if backend in {"auto", "sqlite", "sqlite_vec"} and sqlite_vec_available():
        try:
            from pathlib import Path

            db_dir = Path("data/cache/chatinter/vec")
            db_dir.mkdir(parents=True, exist_ok=True)
            return SqliteVecStore(str(db_dir / f"{namespace}.db"), dim=dim)
        except Exception:
            from zhenxun.services.log import logger

            logger.warning(
                "SqliteVecStore 初始化失败,回退内存向量存储", "ChatInterVectorStore"
            )
    return InMemoryVectorStore()


def normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max((abs(value) for value in scores.values()), default=0.0)
    if max_score <= 1e-12:
        return {key: 0.0 for key in scores}
    return {key: float(value) / max_score for key, value in scores.items()}


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Iterable[str]],
    *,
    k: float = 60.0,
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, 1):
            if not doc_id:
                continue
            scores[str(doc_id)] += 1.0 / (k + rank)
    return dict(scores)


__all__ = [
    "BM25LexicalIndex",
    "InMemoryVectorStore",
    "SqliteVecStore",
    "VectorDocument",
    "VectorSearchResult",
    "VectorStore",
    "cosine_score",
    "create_vector_store",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "sqlite_vec_available",
]
