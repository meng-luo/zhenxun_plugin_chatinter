"""Capability retriever for ChatInter command tools.

The retriever keeps every known command discoverable without exposing every
command as a native function tool in the first model request.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

from .capability_registry import CapabilityRegistry
from .command_index import CommandCandidate, build_command_candidates
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .route_text import match_command_head, normalize_message_text, strip_invoke_prefix
from .soft_tool_policy import (
    filter_soft_candidates,
    is_high_reliability_candidate,
    should_catalog_only_candidate,
    sort_exposure_candidates,
)

_DEFAULT_RETRIEVAL_LIMIT = 24
_MAX_RETRIEVAL_LIMIT = 64
_DEFAULT_INITIAL_TOOL_CAP = 24
_DEFAULT_LARGE_PLUGIN_COMMAND_CAP = 8
_DEFAULT_TWO_STAGE_PLUGIN_CAP = 8
_DEFAULT_TWO_STAGE_THRESHOLD = 96
_BM25_CACHE_MAX = 8
_BM25_PREFILTER_MIN = 96
_BM25_K1 = 1.5
_BM25_B = 0.75
_ASCII_TOKEN_PATTERN = re.compile(r"[0-9a-z][0-9a-z_.:/-]*", re.IGNORECASE)
_CJK_CHUNK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
_BM25_INDEX_CACHE: dict[str, "_Bm25CommandIndex"] = {}
_BM25_INDEX_CACHE_ORDER: list[str] = []


@dataclass(frozen=True)
class CommandRetrievalResult:
    query: str
    candidates: tuple[CommandCandidate, ...]
    total_commands: int
    capability_payloads: tuple[dict[str, Any], ...] = ()


class CommandToolRetriever:
    """Local retriever that turns a natural-language query into command candidates."""

    def __init__(
        self,
        knowledge_base: PluginKnowledgeBase,
        *,
        session_id: str | None,
        tools: list[Any] | None = None,
    ) -> None:
        self.registry = CapabilityRegistry.from_knowledge_base(
            knowledge_base,
            session_id=session_id,
            tools=tools,
        )

    @property
    def total_commands(self) -> int:
        return self.registry.total_commands

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> CommandRetrievalResult:
        normalized_query = normalize_message_text(query)
        retrieval_limit = _coerce_limit(limit)
        candidate_tools = _get_bm25_index(self.registry.tools).select(
            normalized_query,
            limit=retrieval_limit,
        )
        if candidate_tools:
            candidates = build_command_candidates(
                self.registry.knowledge_base,
                normalized_query,
                limit=retrieval_limit,
                session_id=self.registry.session_id,
                diversify=True,
                tools=candidate_tools,
                include_unscored=False,
            )
            candidates = [
                _mark_bm25_recall(candidate)
                for candidate in candidates
                if self.registry.record_for(candidate.schema.command_id) is not None
            ]
        else:
            candidates = self.registry.recall(
                normalized_query,
                limit=retrieval_limit,
                diversify=True,
            )
        return CommandRetrievalResult(
            query=normalized_query,
            candidates=tuple(candidates),
            total_commands=self.total_commands,
            capability_payloads=tuple(
                self.registry.candidate_payload(candidate, index=index)
                for index, candidate in enumerate(candidates, 1)
            ),
        )

    def initial_command_exposure(
        self,
        query: str,
        *,
        max_total: int = _DEFAULT_INITIAL_TOOL_CAP,
        large_plugin_command_cap: int = _DEFAULT_LARGE_PLUGIN_COMMAND_CAP,
    ) -> CommandRetrievalResult:
        """Build first-turn executable command tools.

        Policy:
        - plugins with <= ``large_plugin_command_cap`` commands expose all commands;
        - larger plugins expose only the top related commands for this query.

        This restores broad command-level choice without exceeding provider tool
        limits when a plugin contributes many command schemas.
        Low-reliability commands stay catalog-discoverable but are not exposed
        automatically unless they are exact/high-confidence choices.
        """

        normalized_query = normalize_message_text(query)
        two_stage = self.should_use_two_stage()
        if two_stage:
            max_total = min(
                max(int(max_total or _DEFAULT_INITIAL_TOOL_CAP), 1),
                _config_int("COMMAND_INITIAL_EXPOSURE_CAP", _DEFAULT_INITIAL_TOOL_CAP),
            )
            large_plugin_command_cap = min(
                max(
                    int(large_plugin_command_cap or _DEFAULT_LARGE_PLUGIN_COMMAND_CAP),
                    1,
                ),
                _config_int(
                    "COMMAND_TWO_STAGE_PLUGIN_CAP",
                    _DEFAULT_TWO_STAGE_PLUGIN_CAP,
                ),
            )
        grouped = self.registry.command_tools_by_plugin()
        selected: list[CommandCandidate] = []
        for plugin_tools in grouped.values():
            if len(plugin_tools) <= large_plugin_command_cap and not two_stage:
                plugin_candidates = build_command_candidates(
                    self.registry.knowledge_base,
                    normalized_query,
                    limit=None,
                    session_id=self.registry.session_id,
                    diversify=False,
                    tools=plugin_tools,
                    include_unscored=True,
                )
                selected.extend(
                    _initial_exposure_candidates(
                        normalized_query,
                        plugin_candidates,
                    )
                )
                continue
            plugin_candidates = build_command_candidates(
                self.registry.knowledge_base,
                normalized_query,
                limit=large_plugin_command_cap,
                session_id=self.registry.session_id,
                diversify=False,
                tools=plugin_tools,
                include_unscored=False,
            )
            selected.extend(
                _initial_exposure_candidates(
                    normalized_query,
                    plugin_candidates,
                    two_stage=two_stage,
                )
            )

        selected = _dedupe_and_trim_candidates(
            sort_exposure_candidates(normalized_query, selected),
            max_total=max_total,
        )
        return CommandRetrievalResult(
            query=normalized_query,
            candidates=tuple(selected),
            total_commands=self.total_commands,
            capability_payloads=tuple(
                self.registry.candidate_payload(candidate, index=index)
                for index, candidate in enumerate(selected, 1)
            ),
        )

    def should_use_two_stage(self) -> bool:
        return self.total_commands > _config_int(
            "COMMAND_TWO_STAGE_THRESHOLD",
            _DEFAULT_TWO_STAGE_THRESHOLD,
        )


@dataclass(frozen=True)
class _Bm25Document:
    tool: CommandToolSnapshot
    weights: dict[str, float]
    length: float


@dataclass(frozen=True)
class _Bm25CommandIndex:
    fingerprint: str
    tools: tuple[CommandToolSnapshot, ...]
    documents: tuple[_Bm25Document, ...]
    postings: dict[str, tuple[tuple[int, float], ...]]
    idf: dict[str, float]
    avgdl: float

    def select(
        self,
        query: str,
        *,
        limit: int | None,
    ) -> list[CommandToolSnapshot] | None:
        terms = _bm25_query_terms(query)
        if not terms or not self.tools:
            return None
        scores: dict[int, float] = defaultdict(float)
        query_weights = _bm25_query_weights(terms)
        for term, query_weight in query_weights.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, term_frequency in self.postings.get(term, ()):
                document = self.documents[index]
                denominator = term_frequency + _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * document.length / max(self.avgdl, 1.0)
                )
                scores[index] += (
                    idf
                    * (term_frequency * (_BM25_K1 + 1.0) / denominator)
                    * query_weight
                )
        if not scores:
            return None

        query_variants = _query_variants(query)
        for index in list(scores):
            scores[index] += _exact_invocation_boost(
                query_variants,
                self.tools[index],
            )

        max_items = _bm25_prefilter_limit(limit=limit, total=len(self.tools))
        if max_items >= len(self.tools):
            return None
        ranked_indexes = sorted(
            scores,
            key=lambda index: (
                scores[index],
                self.tools[index].schema_quality,
                self.tools[index].reliability,
                -index,
            ),
            reverse=True,
        )[:max_items]
        return [self.tools[index] for index in ranked_indexes]


def _get_bm25_index(tools: list[CommandToolSnapshot]) -> _Bm25CommandIndex:
    fingerprint = _bm25_fingerprint(tools)
    cached = _BM25_INDEX_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    documents = tuple(_bm25_document(tool) for tool in tools)
    postings_mutable: dict[str, list[tuple[int, float]]] = defaultdict(list)
    document_frequency: dict[str, int] = defaultdict(int)
    for index, document in enumerate(documents):
        for term, term_frequency in document.weights.items():
            postings_mutable[term].append((index, term_frequency))
            document_frequency[term] += 1

    total = max(len(documents), 1)
    idf = {
        term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    avgdl = (
        sum(document.length for document in documents) / len(documents)
        if documents
        else 1.0
    )
    index = _Bm25CommandIndex(
        fingerprint=fingerprint,
        tools=tuple(tools),
        documents=documents,
        postings={
            term: tuple(items) for term, items in postings_mutable.items() if items
        },
        idf=idf,
        avgdl=max(avgdl, 1.0),
    )
    _BM25_INDEX_CACHE[fingerprint] = index
    _BM25_INDEX_CACHE_ORDER.append(fingerprint)
    while len(_BM25_INDEX_CACHE_ORDER) > _BM25_CACHE_MAX:
        old = _BM25_INDEX_CACHE_ORDER.pop(0)
        _BM25_INDEX_CACHE.pop(old, None)
    return index


def _bm25_document(tool: CommandToolSnapshot) -> _Bm25Document:
    weights: dict[str, float] = defaultdict(float)

    def add(value: object, weight: float) -> None:
        text = normalize_message_text(str(value or ""))
        if not text:
            return
        for token in _bm25_terms(text):
            weights[token] += weight

    add(tool.command_id, 4.0)
    add(tool.head, 7.0)
    add(tool.plugin_name, 2.0)
    add(tool.plugin_module, 1.0)
    add(tool.family, 1.0)
    add(tool.render, 4.0)
    add(tool.description, 2.6)
    add(tool.usage or "", 1.6)
    add(tool.capability_text, 3.0)
    add(tool.output_mode, 0.8)
    add(tool.side_effect, 0.8)
    add(tool.source_of_truth, 0.6)
    add(tool.command_role, 1.2)
    add(tool.payload_policy, 1.0)
    for value in tool.aliases:
        add(value, 6.0)
    for value in tool.retrieval_phrases:
        add(value, 3.2)
    for value in tool.examples:
        add(value, 2.2)
    for value in tool.task_verbs:
        add(value, 2.0)
    for value in tool.input_requirements:
        add(value, 1.8)
    for value in tool.use_cases:
        add(value, 2.2)
    for value in tool.anti_use_cases:
        add(value, 0.8)
    for value in tool.intent_types:
        add(value, 1.4)
    for item in _tool_shortcut_renders(tool):
        if not isinstance(item, dict):
            continue
        add(item.get("alias"), 6.0)
        add(item.get("render"), 4.0)
    for slot in tool.slots:
        add(slot.name, 2.4)
        add(slot.type, 1.0)
        add(slot.description, 2.0)
        for alias in slot.aliases:
            add(alias, 2.2)

    length = sum(weights.values())
    return _Bm25Document(
        tool=tool,
        weights=dict(weights),
        length=max(length, 1.0),
    )


def _bm25_fingerprint(tools: list[CommandToolSnapshot]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for tool in tools:
        parts = [
            tool.command_id,
            tool.plugin_module,
            tool.plugin_name,
            tool.head,
            " ".join(tool.aliases),
            tool.render,
            tool.description,
            tool.capability_text,
            " ".join(tool.retrieval_phrases),
            " ".join(tool.task_verbs),
            " ".join(tool.input_requirements),
            " ".join(tool.use_cases),
            _shortcut_fingerprint(tool),
            str(tool.source_signature or ""),
        ]
        for part in parts:
            digest.update(str(part or "").encode("utf-8", "ignore"))
            digest.update(b"\x00")
    return digest.hexdigest()


def _bm25_query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for variant in _query_variants(query):
        terms.extend(_bm25_terms(variant))
    return terms


def _query_variants(query: str) -> tuple[str, str]:
    normalized = normalize_message_text(query)
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if not stripped or stripped == normalized:
        return (normalized, "")
    return normalized, stripped


def _bm25_query_weights(terms: list[str]) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(float)
    for term in terms:
        weights[term] += 1.0
    return {term: min(weight, 2.5) for term, weight in weights.items()}


def _bm25_terms(text: str) -> list[str]:
    normalized = normalize_message_text(text).casefold()
    if not normalized:
        return []
    terms: list[str] = []
    for token in _ASCII_TOKEN_PATTERN.findall(normalized):
        terms.append(token)
        for part in re.split(r"[_.:/-]+", token):
            if part:
                terms.append(part)
    for chunk in _CJK_CHUNK_PATTERN.findall(normalized):
        if len(chunk) == 1:
            terms.append(chunk)
            continue
        terms.append(chunk)
        max_size = min(len(chunk), 4)
        for size in range(2, max_size + 1):
            for start in range(0, len(chunk) - size + 1):
                terms.append(chunk[start : start + size])
    return [term for term in terms if term]


def _exact_invocation_boost(
    query_variants: tuple[str, str],
    tool: CommandToolSnapshot,
) -> float:
    boost = 0.0
    phrases = [tool.head, *list(tool.aliases)]
    for item in _tool_shortcut_renders(tool):
        if not isinstance(item, dict):
            continue
        phrases.append(str(item.get("alias") or ""))
        phrases.append(str(item.get("render") or ""))
    for phrase in phrases:
        normalized_phrase = normalize_message_text(str(phrase or ""))
        if not normalized_phrase:
            continue
        for text in query_variants:
            if not text:
                continue
            if match_command_head(text, normalized_phrase):
                boost = max(boost, 18.0)
            elif normalized_phrase in text:
                boost = max(boost, 7.0)
    return boost


def _bm25_prefilter_limit(*, limit: int | None, total: int) -> int:
    if total <= 0:
        return 0
    requested = _coerce_limit(limit)
    return min(total, max(requested * 4, _BM25_PREFILTER_MIN))


def _tool_shortcut_renders(tool: CommandToolSnapshot) -> list[dict[str, Any]]:
    value = tool.meta.get("shortcut_renders") if isinstance(tool.meta, dict) else None
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _shortcut_fingerprint(tool: CommandToolSnapshot) -> str:
    parts: list[str] = []
    for item in _tool_shortcut_renders(tool):
        parts.append(str(item.get("alias") or ""))
        parts.append(str(item.get("render") or ""))
    return " ".join(parts)


def _mark_bm25_recall(candidate: CommandCandidate) -> CommandCandidate:
    reasons = tuple(
        reason if reason.startswith("recall:") else f"recall:{reason}"
        for reason in candidate.reasons
    )
    if "recall:bm25" not in reasons:
        reasons = (*reasons, "recall:bm25")
    return CommandCandidate(
        plugin_module=candidate.plugin_module,
        plugin_name=candidate.plugin_name,
        schema=candidate.schema,
        score=candidate.score,
        reason=",".join(reasons) or "recall:bm25",
        family=candidate.family,
        tool=candidate.tool,
        reasons=reasons,
        exact_protected=candidate.exact_protected,
        features=candidate.features,
    )


def _config_int(name: str, default: int) -> int:
    try:
        from . import config as config_module

        return int(getattr(config_module, name))
    except Exception:
        return int(default)


def _coerce_limit(limit: int | None) -> int:
    if limit is None:
        return _DEFAULT_RETRIEVAL_LIMIT
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = _DEFAULT_RETRIEVAL_LIMIT
    return max(1, min(value, _MAX_RETRIEVAL_LIMIT))


def _dedupe_and_trim_candidates(
    candidates: list[CommandCandidate],
    *,
    max_total: int,
) -> list[CommandCandidate]:
    by_id: dict[str, CommandCandidate] = {}
    for candidate in candidates:
        command_id = normalize_message_text(candidate.schema.command_id)
        if not command_id:
            continue
        previous = by_id.get(command_id)
        if previous is None or _candidate_rank_key(candidate) > _candidate_rank_key(
            previous
        ):
            by_id[command_id] = candidate

    ordered = sorted(
        by_id.values(),
        key=_candidate_rank_key,
        reverse=True,
    )
    cap = max(1, min(int(max_total or _DEFAULT_INITIAL_TOOL_CAP), 120))
    return ordered[:cap]


def _initial_exposure_candidates(
    message_text: str,
    candidates: list[CommandCandidate],
    *,
    two_stage: bool = False,
) -> list[CommandCandidate]:
    filtered = filter_soft_candidates(message_text, candidates)
    result: list[CommandCandidate] = []
    deferred: list[CommandCandidate] = []
    for candidate in filtered:
        if should_catalog_only_candidate(candidate, message_text=message_text):
            deferred.append(candidate)
            continue
        result.append(candidate)
    if two_stage:
        focused = [
            candidate for candidate in result if _two_stage_initial_candidate(candidate)
        ]
        if focused:
            return sort_exposure_candidates(message_text, focused)
        exact_or_reliable = [
            candidate
            for candidate in filtered
            if candidate.exact_protected or is_high_reliability_candidate(candidate)
        ]
        return sort_exposure_candidates(message_text, exact_or_reliable)
    if result:
        return sort_exposure_candidates(message_text, result)
    exact_or_reliable = [
        candidate
        for candidate in filtered
        if candidate.exact_protected or is_high_reliability_candidate(candidate)
    ]
    if exact_or_reliable:
        return sort_exposure_candidates(message_text, exact_or_reliable)
    return []


def _two_stage_initial_candidate(candidate: CommandCandidate) -> bool:
    if candidate.exact_protected or is_high_reliability_candidate(candidate):
        return True
    score = float(candidate.score or 0.0)
    features = candidate.features
    exact = float(getattr(features, "exact_score", 0.0) or 0.0)
    lexical = float(getattr(features, "lexical_score", 0.0) or 0.0)
    semantic = float(getattr(features, "semantic_score", 0.0) or 0.0)
    context = float(getattr(features, "context_score", 0.0) or 0.0)
    schema = float(getattr(features, "schema_score", 0.0) or 0.0)
    return bool(
        score > 0 and (exact > 0 or lexical > 0 or semantic + context + schema >= 3.0)
    )


def _candidate_rank_key(
    candidate: CommandCandidate,
) -> tuple[int, int, int, float, float, float, float, float, float, float, int, str]:
    exact = 1 if candidate.exact_protected else 0
    tool = candidate.tool
    soft = 0 if bool(getattr(tool, "soft_tool", False)) else 1
    concrete = 1 if _is_concrete_tool(tool) else 0
    score = float(candidate.score or 0.0)
    context = float(getattr(candidate.features, "context_score", 0.0) or 0.0)
    schema = float(getattr(candidate.features, "schema_score", 0.0) or 0.0)
    reliability = float(getattr(candidate.features, "reliability_score", 0.0) or 0.0)
    false_trigger = float(
        getattr(candidate.features, "false_trigger_score", 0.0) or 0.0
    )
    param_failure = float(
        getattr(candidate.features, "param_failure_score", 0.0) or 0.0
    )
    latency = float(getattr(candidate.features, "latency_score", 0.0) or 0.0)
    non_empty = 1 if score > 0 else 0
    return (
        exact,
        concrete,
        soft,
        reliability + false_trigger + param_failure + latency,
        reliability,
        schema,
        score,
        context,
        param_failure,
        latency,
        non_empty,
        normalize_message_text(candidate.schema.command_id),
    )


def _is_concrete_tool(tool: object | None) -> bool:
    if tool is None:
        return False
    output_mode = normalize_message_text(str(getattr(tool, "output_mode", "") or ""))
    side_effect = normalize_message_text(str(getattr(tool, "side_effect", "") or ""))
    return output_mode in {"image", "file", "action"} or side_effect in {
        "query",
        "send",
        "mutate",
    }


__all__ = [
    "CommandRetrievalResult",
    "CommandToolRetriever",
]
