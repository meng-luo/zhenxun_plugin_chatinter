"""Embedding-free sparse retrieval for the local reaction-image library."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import ClassVar

from .reaction_models import ReactionRecord
from .route_text import normalize_message_text
from .sparse_retrieval import (
    BM25FIndex,
    fuse_sparse_rankings,
    normalize_retrieval_queries,
    normalize_sparse_scores,
)

_FIELD_WEIGHTS = {
    "reply_intents": 2.4,
    "usage_scenarios": 2.0,
    "tones": 1.4,
    "actions": 1.3,
    "target_relation": 1.0,
    "caption": 1.5,
    "tags": 2.1,
    "visible_text": 2.3,
    "category_description": 1.2,
    "category": 1.5,
    "filename": 0.8,
}
_INDEX_CACHE_LIMIT = 8


@dataclass(slots=True)
class _ReactionSparseState:
    signature: str
    index: BM25FIndex


class ReactionSearchIndex:
    _indexes: ClassVar[dict[str, _ReactionSparseState]] = {}

    @classmethod
    async def search(
        cls,
        root: Path,
        records: Sequence[ReactionRecord],
        query: str,
        *,
        semantic_enabled: bool,
        retrieval_queries: object = None,
        category_hints: object = None,
        top_k: int = 8,
        min_score: float = 0.22,
    ) -> list[tuple[ReactionRecord, float]]:
        searchable = tuple(
            record
            for record in records
            if record.status != "rejected" and record.semantic_text
        )
        queries = normalize_retrieval_queries(
            query,
            retrieval_queries if semantic_enabled else None,
        )
        if not queries or not searchable:
            return []
        state = cls._ensure_index(root, searchable)
        records_by_id = {record.reaction_id: record for record in searchable}
        rankings: list[list[str]] = []
        normalized_scores: list[dict[str, float]] = []
        exact_ids: set[str] = set()
        for item_query in queries:
            scores = state.index.score_all(item_query)
            normalized = normalize_sparse_scores(scores)
            normalized_scores.append(normalized)
            rankings.append(
                sorted(
                    scores,
                    key=lambda reaction_id: (
                        -scores[reaction_id],
                        reaction_id,
                    ),
                )
            )
            exact_ids.update(
                record.reaction_id
                for record in searchable
                if _has_exact_identity(record, item_query)
            )
        valid_categories = {record.category for record in searchable if record.category}
        raw_hints = category_hints if isinstance(category_hints, list | tuple) else ()
        hints = tuple(
            dict.fromkeys(
                str(value or "").strip()
                for value in raw_hints
                if str(value or "").strip() in valid_categories
            )
        )[:3]
        hinted_ids: set[str] = set()
        for category in hints:
            category_records = [
                record for record in searchable if record.category == category
            ]
            category_records.sort(
                key=lambda record: (
                    -max(
                        (
                            scores.get(record.reaction_id, 0.0)
                            for scores in normalized_scores
                        ),
                        default=0.0,
                    ),
                    -record.has_full_semantics,
                    record.reaction_id,
                )
            )
            ranking = [record.reaction_id for record in category_records]
            rankings.append(ranking)
            hinted_ids.update(ranking)
        fusion_queries = [*queries, *(f"category:{hint}" for hint in hints)]
        fused = fuse_sparse_rankings(
            fusion_queries,
            rankings,
            exact_ids=exact_ids,
        )
        fused_scores = normalize_sparse_scores(fused.scores)
        ranked: list[tuple[ReactionRecord, float]] = []
        for reaction_id in fused.ranked_ids:
            record = records_by_id.get(reaction_id)
            if record is None:
                continue
            strength = max(
                (scores.get(reaction_id, 0.0) for scores in normalized_scores),
                default=0.0,
            )
            if (
                reaction_id not in exact_ids
                and reaction_id not in hinted_ids
                and strength < max(float(min_score), 0.0)
            ):
                continue
            score = (
                1.0
                if reaction_id in exact_ids
                else min(
                    strength * 0.78 + fused_scores.get(reaction_id, 0.0) * 0.22,
                    1.0,
                )
            )
            ranked.append((record, score))
            if len(ranked) >= max(min(int(top_k), 12), 1):
                break
        return ranked

    @classmethod
    def _ensure_index(
        cls,
        root: Path,
        records: Sequence[ReactionRecord],
    ) -> _ReactionSparseState:
        key = str(root.resolve()).casefold()
        signature = _records_signature(records)
        cached = cls._indexes.get(key)
        if cached is not None and cached.signature == signature:
            return cached
        index = BM25FIndex(field_weights=_FIELD_WEIGHTS)
        index.rebuild(
            {
                record.reaction_id: {
                    "caption": record.caption,
                    "tags": " ".join(record.tags),
                    "visible_text": record.visible_text,
                    "reply_intents": " ".join(record.reply_intents),
                    "usage_scenarios": " ".join(record.usage_scenarios),
                    "tones": " ".join(record.tones),
                    "actions": " ".join(record.actions),
                    "target_relation": record.target_relation,
                    "category_description": record.category_description,
                    "category": record.category,
                    "filename": Path(record.relative_path)
                    .stem.replace("_", " ")
                    .replace("-", " "),
                }
                for record in records
            }
        )
        state = _ReactionSparseState(signature=signature, index=index)
        if len(cls._indexes) >= _INDEX_CACHE_LIMIT and key not in cls._indexes:
            cls._indexes.pop(next(iter(cls._indexes)))
        cls._indexes[key] = state
        return state


def _records_signature(records: Sequence[ReactionRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.reaction_id):
        values = (
            record.content_sha256,
            record.status,
            record.caption,
            "\0".join(record.tags),
            record.visible_text,
            "\0".join(record.reply_intents),
            "\0".join(record.usage_scenarios),
            "\0".join(record.tones),
            "\0".join(record.actions),
            record.target_relation,
            str(record.semantic_version),
            record.category_description,
            record.category,
            record.relative_path,
        )
        digest.update("\0".join(values).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _has_exact_identity(record: ReactionRecord, query: str) -> bool:
    identity = normalize_message_text(query).casefold()
    if not identity:
        return False
    values = (
        record.category,
        record.caption,
        record.visible_text,
        record.target_relation,
        Path(record.relative_path).stem.replace("_", " ").replace("-", " "),
        *record.tags,
        *record.reply_intents,
        *record.usage_scenarios,
        *record.tones,
        *record.actions,
    )
    return any(
        normalize_message_text(str(value or "")).casefold() == identity
        for value in values
        if value
    )


__all__ = ["ReactionSearchIndex"]
