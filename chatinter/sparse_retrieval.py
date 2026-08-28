"""Shared multi-query sparse retrieval helpers for mixed-chat Skills."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import re

from .route_text import normalize_message_text

_MAX_QUERY_COUNT = 6
_MAX_QUERY_CHARS = 256
_MAX_QUERY_TOTAL_CHARS = 1_200
_RRF_K = 60.0
_SPARSE_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SparseFusionResult:
    queries: tuple[str, ...]
    ranked_ids: tuple[str, ...]
    scores: dict[str, float]


class BM25FIndex:
    """Small in-memory field-aware sparse index for ChatInter metadata."""

    def __init__(
        self,
        *,
        field_weights: Mapping[str, float],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.field_weights = {
            str(name): max(float(weight), 0.0)
            for name, weight in field_weights.items()
            if str(name) and float(weight) > 0
        }
        self.k1 = max(float(k1), 0.1)
        self.b = min(max(float(b), 0.0), 1.0)
        self._field_terms: dict[str, dict[str, Counter[str]]] = {}
        self._field_lengths: dict[str, dict[str, int]] = {}
        self._average_lengths: dict[str, float] = {}
        self._document_frequencies: Counter[str] = Counter()

    def rebuild(self, documents: Mapping[str, Mapping[str, object]]) -> None:
        self._field_terms.clear()
        self._field_lengths.clear()
        self._average_lengths.clear()
        self._document_frequencies.clear()
        field_totals: Counter[str] = Counter()
        field_counts: Counter[str] = Counter()
        for raw_id, fields in documents.items():
            document_id = str(raw_id)
            seen_terms: set[str] = set()
            term_fields: dict[str, Counter[str]] = {}
            length_fields: dict[str, int] = {}
            for field_name in self.field_weights:
                tokens = sparse_text_tokens(str(fields.get(field_name, "") or ""))
                if not tokens:
                    continue
                counter = Counter(tokens)
                term_fields[field_name] = counter
                length = sum(counter.values())
                length_fields[field_name] = length
                field_totals[field_name] += length
                field_counts[field_name] += 1
                seen_terms.update(counter)
            if not term_fields:
                continue
            self._field_terms[document_id] = term_fields
            self._field_lengths[document_id] = length_fields
            self._document_frequencies.update(seen_terms)
        self._average_lengths = {
            field_name: field_totals[field_name] / max(field_counts[field_name], 1)
            for field_name in self.field_weights
        }

    def score_all(self, query: str) -> dict[str, float]:
        query_terms = Counter(sparse_text_tokens(query))
        if not query_terms or not self._field_terms:
            return {}
        document_count = len(self._field_terms)
        scores: dict[str, float] = {}
        for document_id, fields in self._field_terms.items():
            score = 0.0
            lengths = self._field_lengths.get(document_id, {})
            for term, query_frequency in query_terms.items():
                document_frequency = self._document_frequencies.get(term, 0)
                if document_frequency <= 0:
                    continue
                weighted_frequency = 0.0
                for field_name, weight in self.field_weights.items():
                    frequency = fields.get(field_name, {}).get(term, 0)
                    if frequency <= 0:
                        continue
                    average_length = self._average_lengths.get(field_name, 0.0) or 1.0
                    field_length = lengths.get(field_name, 0) or average_length
                    normalization = (
                        1.0 - self.b + self.b * field_length / average_length
                    )
                    weighted_frequency += weight * frequency / max(normalization, 1e-9)
                if weighted_frequency <= 0:
                    continue
                inverse_document_frequency = math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                score += (
                    min(query_frequency, 3)
                    * inverse_document_frequency
                    * weighted_frequency
                    * (self.k1 + 1.0)
                    / (weighted_frequency + self.k1)
                )
            if score > 0:
                scores[document_id] = score
        return scores


def sparse_text_tokens(value: str) -> tuple[str, ...]:
    """Tokenize Latin words and CJK text without a language model."""
    tokens: list[str] = []
    for raw_token in _SPARSE_TOKEN_PATTERN.findall(str(value or "").casefold()):
        if not raw_token:
            continue
        tokens.append(raw_token)
        if not re.fullmatch(r"[\u3400-\u9fff]+", raw_token):
            continue
        for size in (2, 3):
            if len(raw_token) < size:
                continue
            tokens.extend(
                raw_token[index : index + size]
                for index in range(len(raw_token) - size + 1)
            )
    return tuple(tokens)


def normalize_sparse_scores(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    maximum = max((float(value) for value in scores.values()), default=0.0)
    if maximum <= 1e-12:
        return {}
    return {
        str(item_id): max(float(value), 0.0) / maximum
        for item_id, value in scores.items()
        if float(value) > 0
    }


def normalize_retrieval_queries(
    task_text: str,
    retrieval_queries: object = None,
) -> tuple[str, ...]:
    """Return the original task plus bounded, stable model-provided rewrites."""
    original = normalize_message_text(str(task_text or ""))
    if not original:
        return ()
    source: Iterable[object]
    if isinstance(retrieval_queries, list | tuple):
        source = retrieval_queries
    else:
        source = ()
    result = [original[:_MAX_QUERY_CHARS]]
    seen = {result[0].casefold()}
    used_chars = len(result[0])
    for value in source:
        query = normalize_message_text(str(value or ""))[:_MAX_QUERY_CHARS]
        key = query.casefold()
        if not query or key in seen:
            continue
        if (
            len(result) >= _MAX_QUERY_COUNT
            or used_chars + len(query) > _MAX_QUERY_TOTAL_CHARS
        ):
            break
        result.append(query)
        seen.add(key)
        used_chars += len(query)
    return tuple(result)


def fuse_sparse_rankings(
    queries: Sequence[str],
    ranked_ids: Sequence[Sequence[str]],
    *,
    exact_ids: Iterable[str] = (),
) -> SparseFusionResult:
    """Fuse sparse rankings while protecting real exact command identities."""
    scores: dict[str, float] = defaultdict(float)
    first_positions: dict[str, int] = {}
    original_positions: dict[str, int] = {}
    for query_index, ranking in enumerate(ranked_ids):
        seen: set[str] = set()
        for rank, raw_id in enumerate(ranking, 1):
            item_id = normalize_message_text(str(raw_id or ""))
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] += 1.0 / (_RRF_K + rank)
            first_positions[item_id] = min(first_positions.get(item_id, rank), rank)
            if query_index == 0:
                original_positions[item_id] = rank
    protected = {
        normalize_message_text(str(item_id or ""))
        for item_id in exact_ids
        if normalize_message_text(str(item_id or ""))
    }
    ordered = sorted(
        scores,
        key=lambda item_id: (
            item_id not in protected,
            -scores[item_id],
            original_positions.get(item_id, 1 << 30),
            first_positions.get(item_id, 1 << 30),
            item_id.casefold(),
            item_id,
        ),
    )
    return SparseFusionResult(
        queries=tuple(queries),
        ranked_ids=tuple(ordered),
        scores=dict(scores),
    )


__all__ = [
    "BM25FIndex",
    "SparseFusionResult",
    "fuse_sparse_rankings",
    "normalize_retrieval_queries",
    "normalize_sparse_scores",
    "sparse_text_tokens",
]
