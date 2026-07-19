"""Recall-only command candidate index for ChatInter.

This module ranks candidates to keep the prompt small.  It must not be treated
as the final command selector; executable schema choice belongs to the LLM.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

from .capability_graph import build_capability_graph_snapshot
from .feedback import (
    get_command_feedback_profile,
    get_command_success_examples,
    get_contextual_command_feedback_profile,
)
from .models.pydantic_models import (
    CommandCandidateFeatures,
    CommandCandidateSnapshot,
    CommandToolSnapshot,
    PluginCommandSchema,
    PluginKnowledgeBase,
)
from .plugin_reference import build_command_tool_snapshots
from .route_text import (
    invoke_prefix_variants,
    match_command_head,
    normalize_action_phrases,
    normalize_message_text,
)

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)
_ASCII_TERM_PATTERN = re.compile(r"[0-9a-z][0-9a-z_.:/-]*", re.IGNORECASE)
_IMAGE_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_AT_PATTERN = re.compile(r"\[@(?:[^\]\s]+|所有人)\]|@\d{5,20}", re.IGNORECASE)
_EXACT_BOOST = 180.0
_ALIAS_BOOST = 160.0
_RRF_K = 60.0
_FAMILY_SOFT_CAP = 6
_PLUGIN_SOFT_CAP = 8
_EXACT_KEEP_LIMIT = 8
_INDEX_PREFILTER_MIN_TOOLS = 64
_INDEX_PREFILTER_LIMIT = 160
_STATIC_BM25_CAP = 8
_STATIC_BM25_CUT_RATIO = 0.38
_STATIC_BM25_K1 = 1.2
_STATIC_BM25_B = 0.55
_STATIC_BM25_AVGDL = 24.0
_CJK_COMMAND_BOUNDARY_CHARS = frozenset(" ，,。.!！？?；;：:/|）)]】}《<>")
_STATIC_BM25_STOP_CHARS = frozenset(
    "我你他她它的了呢啊呀吧吗嘛请用这张这个那个一下一个一条给看查帮寻真"
)
_STATIC_BM25_WEAK_WORDS = frozenset({"怎么", "怎样", "怎么样", "觉得", "认为", "感觉"})
_STATIC_BM25_CONTEXT_ONLY_TERMS = frozenset(
    {"图片", "照片", "这张图", "这图", "张图", "图像"}
)
_STATIC_BM25_FIELD_WEIGHTS = {
    "head": 5.0,
    "alias": 4.8,
    "shortcut": 4.5,
    "example": 3.5,
    "usage": 3.0,
    "description": 2.8,
    "slot": 1.6,
}
_EXPANDED_RELEVANCE_CONTEXT_KEYS = ("has_reply", "multi_task", "is_multi_task")
_MEDIA_CONTEXT_TERMS = (
    "表情",
    "表情包",
    "梗图",
    "头像",
    "图片",
    "照片",
    "这张图",
    "这图",
    "图",
)
_MEDIA_ACTION_TERMS = (
    "做",
    "制作",
    "生成",
    "整",
    "弄",
    "来",
    "发",
    "画",
    "转",
    "识别",
    "搜",
)


@dataclass(frozen=True)
class CommandCandidate:
    plugin_module: str
    plugin_name: str
    schema: PluginCommandSchema
    score: float
    reason: str
    family: str = "general"
    tool: CommandToolSnapshot | None = None
    reasons: tuple[str, ...] = ()
    exact_protected: bool = False
    features: CommandCandidateFeatures | None = None


@dataclass(frozen=True)
class _ScoredCandidate:
    tool: CommandToolSnapshot
    schema: PluginCommandSchema
    score: float
    reasons: tuple[str, ...]
    exact_protected: bool
    features: CommandCandidateFeatures


@dataclass
class _CommandInvertedIndex:
    fingerprint: str
    tools: tuple[CommandToolSnapshot, ...]
    postings: dict[str, dict[int, float]]

    def select(
        self,
        query: str,
        *,
        limit: int | None,
    ) -> list[CommandToolSnapshot] | None:
        terms = _query_index_terms(query)
        if not terms:
            return None
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            for index, weight in self.postings.get(term, {}).items():
                scores[index] += weight
        if not scores:
            return None
        max_items = _prefilter_limit(limit=limit, total=len(self.tools))
        if len(scores) >= len(self.tools) * 0.85:
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
        if not ranked_indexes:
            return None
        return [self.tools[index] for index in ranked_indexes]


@dataclass(frozen=True)
class _StaticBm25Document:
    tool: CommandToolSnapshot
    schema: PluginCommandSchema
    field_terms: dict[str, list[str]]
    field_counts: dict[str, dict[str, int]]
    field_lengths: dict[str, int]
    shortcut_texts: tuple[str, ...]


@dataclass(frozen=True)
class _StaticBm25Index:
    fingerprint: str
    documents: tuple[_StaticBm25Document, ...]
    idf: dict[str, float]
    postings: dict[str, frozenset[int]]
    exact_identity: dict[str, frozenset[int]]
    single_cjk_prefix: dict[str, frozenset[int]]
    head_prefixes: dict[str, tuple[tuple[str, int], ...]]
    long_identity: tuple[tuple[str, int], ...]
    short_cjk_identity: tuple[tuple[str, int], ...]
    expanded_shortcut_aliases_by_module: dict[str, frozenset[str]]


_INDEX_CACHE: dict[str, _CommandInvertedIndex] = {}
_INDEX_CACHE_ORDER: list[str] = []
_INDEX_CACHE_MAX = 8
_STATIC_BM25_CACHE: dict[tuple[Any, ...], _StaticBm25Index] = {}
_STATIC_BM25_CACHE_ORDER: list[tuple[Any, ...]] = []
_STATIC_BM25_CACHE_MAX = 8


def _empty_features() -> CommandCandidateFeatures:
    return CommandCandidateFeatures()


def _reason_feature_deltas(reason: str) -> dict[str, float]:
    if reason in {"exact_head", "exact_alias", "exact_shortcut"}:
        return {"exact_score": _EXACT_BOOST if reason == "exact_head" else _ALIAS_BOOST}
    if reason in {
        "head_prefix",
        "head",
        "alias_prefix",
        "alias",
        "shortcut_prefix",
        "shortcut",
        "short_cjk_fuzzy",
        "ascii_token",
        "cjk_ngram",
        "plugin",
    }:
        return {"lexical_score": 1.0}
    if reason in {
        "retrieval_phrase",
        "direct_retrieval_phrase",
        "description_phrase",
        "usage_example",
        "catalog",
        "helper",
        "helper_langs",
        "random",
        "template",
    }:
        return {"semantic_score": 1.0}
    if reason in {"slot", "required_context"}:
        return {"slot_score": 1.0}
    if reason in {
        "image_signal",
        "image_policy",
        "at_signal",
        "reply_signal",
        "payload_context",
    }:
        return {"context_score": 1.0}
    if reason in {"feedback", "success_example_history"}:
        return {"feedback_score": 1.0}
    if reason == "schema_quality":
        return {"schema_score": 1.0}
    if reason == "reliable_history":
        return {"reliability_score": 1.0}
    if reason in {"false_trigger_history", "low_reliability_history"}:
        return {"false_trigger_score": 1.0}
    if reason.endswith("_penalty") or "penalty" in reason:
        return {"negative_score": 1.0}
    return {}


def _build_candidate_features(
    *,
    score: float,
    reasons: tuple[str, ...],
    feedback_score: float = 0.0,
    schema_score: float = 0.0,
    reliability_score: float = 0.0,
    false_trigger_score: float = 0.0,
    param_failure_score: float = 0.0,
    latency_score: float = 0.0,
) -> CommandCandidateFeatures:
    values = {
        "lexical_score": 0.0,
        "exact_score": 0.0,
        "semantic_score": 0.0,
        "slot_score": 0.0,
        "context_score": 0.0,
        "feedback_score": float(feedback_score),
        "schema_score": float(schema_score),
        "reliability_score": float(reliability_score),
        "false_trigger_score": float(false_trigger_score),
        "param_failure_score": float(param_failure_score),
        "latency_score": float(latency_score),
        "negative_score": 0.0,
    }
    for reason in reasons:
        for key, delta in _reason_feature_deltas(reason).items():
            values[key] += float(delta)
    if values["lexical_score"]:
        values["lexical_score"] = min(values["lexical_score"] * 45.0, score)
    if values["semantic_score"]:
        values["semantic_score"] = min(values["semantic_score"] * 32.0, score)
    if values["slot_score"]:
        values["slot_score"] = min(values["slot_score"] * 28.0, score)
    if values["context_score"]:
        values["context_score"] = min(values["context_score"] * 28.0, score)
    if values["negative_score"]:
        values["negative_score"] = -abs(values["negative_score"] * 48.0)
    return CommandCandidateFeatures(**values)


def _tokens(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(normalize_message_text(text))
        if token
    }
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", normalize_message_text(text)):
        max_size = min(len(chunk), 4)
        for size in range(2, max_size + 1):
            for start in range(0, len(chunk) - size + 1):
                tokens.add(chunk[start : start + size].casefold())
    return tokens


def _ascii_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _ASCII_TERM_PATTERN.findall(normalize_message_text(text).casefold()):
        if not token:
            continue
        terms.add(token)
        for part in re.split(r"[_.:/-]+", token):
            if part:
                terms.add(part)
    return terms


def _cjk_terms(text: str) -> set[str]:
    normalized = normalize_message_text(text).casefold()
    terms = _cjk_ngrams_for_index(normalized)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        terms.add(chunk)
    return terms


def _score_text_overlap(
    query: str,
    texts: list[str],
    *,
    ascii_weight: float,
    cjk_weight: float,
    phrase_weight: float,
    cap: float,
) -> float:
    normalized_query = normalize_message_text(query).casefold()
    haystack = normalize_message_text(" ".join(texts)).casefold()
    if not normalized_query or not haystack:
        return 0.0
    score = 0.0
    ascii_overlap = len(_ascii_terms(normalized_query) & _ascii_terms(haystack))
    cjk_overlap = len(_cjk_terms(normalized_query) & _cjk_terms(haystack))
    if ascii_overlap:
        score += ascii_overlap * ascii_weight
    if cjk_overlap:
        score += cjk_overlap * cjk_weight
    for text in texts:
        phrase = normalize_message_text(str(text or "")).casefold()
        if len(phrase) >= 2 and phrase in normalized_query:
            score += min(phrase_weight + len(phrase) * 2.0, phrase_weight * 2.0)
            break
    return min(score, cap)


def _query_index_terms(query: str) -> set[str]:
    normalized, stripped = _query_variants(query)
    terms = _tokens(normalized) | _tokens(stripped)
    for text in (normalized, stripped):
        lowered = text.casefold()
        if not lowered:
            continue
        for part in re.split(r"\s+", lowered):
            if part:
                terms.add(part)
        terms.update(_cjk_ngrams_for_index(lowered))
    return {term for term in terms if term}


def _cjk_ngrams_for_index(text: str) -> set[str]:
    chars = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
    if len(chars) < 2:
        return set()
    result: set[str] = set()
    max_size = min(len(chars), 4)
    for size in range(2, max_size + 1):
        for start in range(0, len(chars) - size + 1):
            result.add(chars[start : start + size].casefold())
    return result


def _static_bm25_terms(text: str) -> list[str]:
    normalized = normalize_message_text(text).casefold()
    for word in _STATIC_BM25_WEAK_WORDS:
        normalized = normalized.replace(word, "")
    terms: list[str] = []
    terms.extend(_mixed_ascii_cjk_terms(normalized))
    for token in _ASCII_TERM_PATTERN.findall(normalized):
        if len(token) > 1:
            terms.append(token.casefold())
    for chunk in re.findall(r"[\u4e00-\u9fff]+", normalized):
        chars = [char for char in chunk if char not in _STATIC_BM25_STOP_CHARS]
        max_size = min(len(chars), 4)
        for size in range(2, max_size + 1):
            for start in range(0, len(chars) - size + 1):
                term = "".join(chars[start : start + size]).casefold()
                if term and term not in _STATIC_BM25_WEAK_WORDS:
                    terms.append(term)
    return terms


def _mixed_ascii_cjk_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"([a-z][0-9a-z_.:/-]*)([\u4e00-\u9fff]{1,4})", text):
        ascii_part = match.group(1)
        cjk_part = "".join(
            char for char in match.group(2) if char not in _STATIC_BM25_STOP_CHARS
        )
        if not cjk_part:
            continue
        for end in range(1, len(cjk_part) + 1):
            term = f"{ascii_part}{cjk_part[:end]}".casefold()
            if len(term) >= 2:
                terms.append(term)
    return terms


def _query_terms_no_single(text: str) -> set[str]:
    """Query evidence for static plugin recall: ASCII tokens + CJK 2-4 grams."""

    lexical_text = _AT_PATTERN.sub(" ", _IMAGE_PATTERN.sub(" ", text))
    return {
        term
        for term in _static_bm25_terms(lexical_text)
        if term not in _STATIC_BM25_CONTEXT_ONLY_TERMS
    }


def _static_bm25_field_texts(
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
    *,
    expanded_shortcut_aliases: set[str] | None = None,
) -> dict[str, list[str]]:
    shortcut_texts = _shortcut_texts(
        schema,
        ignored_aliases=expanded_shortcut_aliases,
    )
    return {
        "head": [schema.head],
        "alias": list(schema.aliases),
        "shortcut": shortcut_texts,
        "example": list(tool.examples or []),
        "usage": [tool.usage or ""],
        "description": [schema.description],
        "slot": _slot_texts(schema),
    }


def _static_bm25_field_terms(
    field_texts: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        field: [
            term
            for text in texts
            for term in _static_bm25_terms(text)
        ]
        for field, texts in field_texts.items()
    }


def _static_bm25_idf(
    all_field_terms: list[dict[str, list[str]]],
) -> dict[str, float]:
    df: dict[str, int] = defaultdict(int)
    for field_terms in all_field_terms:
        seen = {
            term
            for terms in field_terms.values()
            for term in terms
            if term
        }
        for term in seen:
            df[term] += 1
    total = max(len(all_field_terms), 1)
    return {
        term: math.log((total + 1.0) / (count + 1.0)) + 1.0
        for term, count in df.items()
    }


def _static_bm25_cache_key(tools: list[CommandToolSnapshot]) -> tuple[Any, ...]:
    return tuple(
        (
            tool.command_id,
            tool.plugin_module,
            tool.head,
            tuple(tool.aliases),
            tool.usage,
            tuple(tool.examples),
            tool.description,
            tuple(
                (
                    slot.name,
                    slot.type,
                    slot.required,
                    repr(slot.default),
                    tuple(slot.aliases),
                    slot.description,
                    tuple(slot.choices),
                )
                for slot in tool.slots
            ),
            tuple(sorted(tool.requires.items())),
            tool.allow_at,
            tool.actor_scope,
            tool.target_requirement,
            tuple(tool.target_sources),
            tool.render,
            tool.payload_policy,
            tool.extra_text_policy,
            tool.command_role,
            tool.source,
            tool.confidence,
            tool.matcher_key,
            tuple(tool.retrieval_phrases),
            tuple(
                (
                    normalize_message_text(str(item.get("alias") or "")),
                    normalize_message_text(str(item.get("render") or "")),
                )
                for item in (
                    tool.meta.get("shortcut_renders", [])
                    if isinstance(tool.meta, dict)
                    else []
                )
                if isinstance(item, dict)
            ),
        )
        for tool in tools
    )


def _get_static_bm25_index(
    tools: list[CommandToolSnapshot],
) -> _StaticBm25Index:
    cache_key = _static_bm25_cache_key(tools)
    cached = _STATIC_BM25_CACHE.get(cache_key)
    if cached is not None:
        return cached

    expanded = _expanded_shortcut_aliases_by_module(tools)
    documents: list[_StaticBm25Document] = []
    postings: dict[str, set[int]] = defaultdict(set)
    exact_identity: dict[str, set[int]] = defaultdict(set)
    single_cjk_prefix: dict[str, set[int]] = defaultdict(set)
    head_prefixes: dict[str, list[tuple[str, int]]] = defaultdict(list)
    long_identity: list[tuple[str, int]] = []
    short_cjk_identity: list[tuple[str, int]] = []
    for tool in tools:
        document_index = len(documents)
        schema = _schema_from_tool_snapshot(tool)
        ignored_aliases = expanded.get(tool.plugin_module)
        field_terms = _static_bm25_field_terms(
            _static_bm25_field_texts(
                tool,
                schema,
                expanded_shortcut_aliases=ignored_aliases,
            )
        )
        field_counts: dict[str, dict[str, int]] = {}
        field_lengths: dict[str, int] = {}
        for terms in field_terms.values():
            for term in terms:
                postings[term].add(document_index)
        for field, terms in field_terms.items():
            counts: dict[str, int] = defaultdict(int)
            for term in terms:
                counts[term] += 1
            field_counts[field] = dict(counts)
            field_lengths[field] = max(sum(counts.values()), 1)
        head_aliases = [schema.head, *schema.aliases]
        normalized_head = normalize_message_text(schema.head).casefold()
        if normalized_head:
            head_prefixes[normalized_head[0]].append(
                (normalized_head, document_index)
            )
        identity_phrases = [
            *head_aliases,
            *_shortcut_texts(schema, ignored_aliases=ignored_aliases),
        ]
        for phrase in identity_phrases:
            normalized_phrase = normalize_message_text(phrase).casefold()
            if normalized_phrase:
                exact_identity[normalized_phrase].add(document_index)
        for phrase in head_aliases:
            normalized_phrase = normalize_message_text(phrase).casefold()
            cjk_phrase = re.sub(r"[^\u4e00-\u9fff]+", "", normalized_phrase)
            if len(normalized_phrase) == 1 and len(cjk_phrase) == 1:
                single_cjk_prefix[normalized_phrase].add(document_index)
            if 2 <= len(cjk_phrase) <= 4:
                short_cjk_identity.append((cjk_phrase, document_index))
            if len(cjk_phrase) >= 4:
                long_identity.append((normalized_phrase, document_index))
        documents.append(
            _StaticBm25Document(
                tool=tool,
                schema=schema,
                field_terms=field_terms,
                field_counts=field_counts,
                field_lengths=field_lengths,
                shortcut_texts=tuple(
                    _shortcut_texts(schema, ignored_aliases=ignored_aliases)
                ),
            )
        )
    digest = hashlib.blake2b(repr(cache_key).encode("utf-8", "ignore"), digest_size=16)
    index = _StaticBm25Index(
        fingerprint=digest.hexdigest(),
        documents=tuple(documents),
        idf=_static_bm25_idf([document.field_terms for document in documents]),
        postings={term: frozenset(indexes) for term, indexes in postings.items()},
        exact_identity={
            phrase: frozenset(indexes) for phrase, indexes in exact_identity.items()
        },
        single_cjk_prefix={
            phrase: frozenset(indexes)
            for phrase, indexes in single_cjk_prefix.items()
        },
        head_prefixes={
            first: tuple(sorted(items, key=lambda item: len(item[0]), reverse=True))
            for first, items in head_prefixes.items()
        },
        long_identity=tuple(long_identity),
        short_cjk_identity=tuple(short_cjk_identity),
        expanded_shortcut_aliases_by_module={
            module: frozenset(aliases) for module, aliases in expanded.items()
        },
    )
    _STATIC_BM25_CACHE[cache_key] = index
    _STATIC_BM25_CACHE_ORDER.append(cache_key)
    while len(_STATIC_BM25_CACHE_ORDER) > _STATIC_BM25_CACHE_MAX:
        old = _STATIC_BM25_CACHE_ORDER.pop(0)
        _STATIC_BM25_CACHE.pop(old, None)
    return index


def _static_identity_indexes(
    index: _StaticBm25Index,
    *,
    variants: tuple[str, ...],
    router_context: dict[str, Any] | None,
) -> tuple[set[int], set[int]]:
    matched: set[int] = set()
    longest_prefix_matches: set[int] = set()
    texts = [text.casefold() for text in variants if text]
    for text in texts:
        matched.update(index.exact_identity.get(text, ()))
        if " " in text:
            matched.update(index.exact_identity.get(text.split(" ", 1)[0], ()))
        if len(text) > 1:
            matched.update(index.single_cjk_prefix.get(text[0], ()))
        prefix_matches = [
            (phrase, document_index)
            for phrase, document_index in index.head_prefixes.get(text[:1], ())
            if text.startswith(phrase)
            and len(text) > len(phrase)
            and _allows_sticky_identity_tail(
                index.documents[document_index].schema,
                router_context=router_context,
            )
        ]
        if prefix_matches:
            longest = len(prefix_matches[0][0])
            for phrase, document_index in prefix_matches:
                if len(phrase) != longest:
                    break
                matched.add(document_index)
                longest_prefix_matches.add(document_index)
    for phrase, document_index in index.long_identity:
        if any(phrase in text for text in texts):
            matched.add(document_index)

    cjk_query = re.sub(r"[^\u4e00-\u9fff]+", "", variants[0] if variants else "")
    for phrase, document_index in index.short_cjk_identity:
        if phrase in cjk_query:
            continue
        start = 0
        positions: list[int] = []
        for char in phrase:
            position = cjk_query.find(char, start)
            if position < 0:
                break
            positions.append(position)
            start = position + 1
        if (
            len(positions) == len(phrase)
            and positions[-1] - positions[0] <= len(phrase) + 1
        ):
            matched.add(document_index)
    return matched, longest_prefix_matches


def _allows_sticky_identity_tail(
    schema: PluginCommandSchema,
    *,
    router_context: dict[str, Any] | None,
) -> bool:
    context = router_context or {}
    return bool(
        context.get("has_image")
        or context.get("has_at")
        or context.get("has_reply")
        or schema.payload_policy != "none"
    )


def _static_bm25_reason(field: str, terms: set[str]) -> str:
    if field == "example":
        return "usage_example"
    if field == "description":
        return "description_phrase"
    return field


def _index_terms_for_schema(
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
) -> dict[str, float]:
    weighted: dict[str, float] = {}

    def add(text: str, weight: float) -> None:
        normalized = normalize_message_text(text).casefold()
        if not normalized:
            return
        weighted[normalized] = max(weighted.get(normalized, 0.0), weight)
        for term in _tokens(normalized):
            weighted[term] = max(weighted.get(term, 0.0), weight)
        for term in _cjk_ngrams_for_index(normalized):
            weighted[term] = max(weighted.get(term, 0.0), weight * 0.8)

    add(schema.command_id, 3.0)
    add(schema.head, 4.0)
    for alias in schema.aliases:
        add(alias, 3.6)
    for phrase in schema.retrieval_phrases:
        add(phrase, 2.2)
    add(schema.description, 1.5)
    add(tool.plugin_name, 1.3)
    add(tool.plugin_module, 1.0)
    add(tool.capability_text, 1.4)
    for text in [
        *list(tool.task_verbs or []),
        *list(tool.input_requirements or []),
        *list(getattr(tool, "use_cases", []) or []),
        *list(getattr(tool, "intent_types", []) or []),
    ]:
        add(text, 1.2)
    return weighted


def _command_index_fingerprint(tools: list[CommandToolSnapshot]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for tool in tools:
        parts = [
            str(tool.command_id or ""),
            str(tool.source_signature or ""),
            str(tool.plugin_module or ""),
            str(tool.plugin_name or ""),
            str(tool.head or ""),
            " ".join(tool.aliases or []),
            " ".join(tool.retrieval_phrases or []),
            str(tool.description or ""),
            str(tool.capability_text or ""),
        ]
        for part in parts:
            digest.update(part.encode("utf-8", "ignore"))
            digest.update(b"\x00")
    return digest.hexdigest()


def _get_command_inverted_index(
    tools: list[CommandToolSnapshot],
) -> _CommandInvertedIndex:
    fingerprint = _command_index_fingerprint(tools)
    cached = _INDEX_CACHE.get(fingerprint)
    if cached is not None:
        return cached
    postings: dict[str, dict[int, float]] = defaultdict(dict)
    for index, tool in enumerate(tools):
        schema = _schema_from_tool_snapshot(tool)
        for term, weight in _index_terms_for_schema(tool, schema).items():
            if not term:
                continue
            postings[term][index] = max(postings[term].get(index, 0.0), weight)
    built = _CommandInvertedIndex(
        fingerprint=fingerprint,
        tools=tuple(tools),
        postings={term: dict(items) for term, items in postings.items()},
    )
    _INDEX_CACHE[fingerprint] = built
    _INDEX_CACHE_ORDER.append(fingerprint)
    while len(_INDEX_CACHE_ORDER) > _INDEX_CACHE_MAX:
        old = _INDEX_CACHE_ORDER.pop(0)
        _INDEX_CACHE.pop(old, None)
    return built


def _prefilter_limit(*, limit: int | None, total: int) -> int:
    if total <= 0:
        return 0
    requested = total if limit is None or int(limit or 0) <= 0 else int(limit)
    return min(total, max(requested * 6, _INDEX_PREFILTER_LIMIT))


def _schema_text(schema: PluginCommandSchema) -> str:
    slot_text = " ".join(
        " ".join([slot.name, slot.description, *slot.aliases]) for slot in schema.slots
    )
    return normalize_message_text(
        " ".join(
            [
                schema.command_id,
                schema.head,
                " ".join(schema.aliases),
                " ".join(schema.retrieval_phrases),
                schema.description,
                schema.command_role,
                schema.payload_policy,
                slot_text,
            ]
        )
    )


def _static_metadata_text(
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
) -> str:
    slot_text = " ".join(
        " ".join(
            [
                slot.name,
                slot.type,
                slot.description,
                " ".join(slot.aliases),
            ]
        )
        for slot in schema.slots
    )
    shortcut_text = " ".join(
        " ".join(
            [
                str(item.get("alias") or ""),
                str(item.get("render") or ""),
            ]
        )
        for item in schema.shortcut_renders
        if isinstance(item, dict)
    )
    return normalize_message_text(
        " ".join(
            [
                schema.head,
                " ".join(schema.aliases),
                shortcut_text,
                schema.description,
                tool.usage or "",
                " ".join(tool.examples),
                slot_text,
            ]
        )
    )


def _tool_text(tool: CommandToolSnapshot, schema: PluginCommandSchema) -> str:
    return normalize_message_text(
        " ".join(
            [
                _schema_text(schema),
                tool.capability_text,
                " ".join(tool.task_verbs),
                " ".join(tool.input_requirements),
                " ".join(getattr(tool, "use_cases", []) or []),
                " ".join(getattr(tool, "anti_use_cases", []) or []),
                " ".join(getattr(tool, "intent_types", []) or []),
                getattr(tool, "output_mode", ""),
                getattr(tool, "side_effect", ""),
                getattr(tool, "risk_level", ""),
                getattr(tool, "risk", ""),
                getattr(tool, "source_of_truth", ""),
                "requires_real_tool"
                if bool(getattr(tool, "requires_real_tool", False))
                else "",
                getattr(tool, "entity_scope", ""),
                "soft_tool" if bool(getattr(tool, "soft_tool", False)) else "",
                f"reliability {float(getattr(tool, 'reliability', 0.0) or 0.0):.2f}",
                "schema_quality "
                f"{float(getattr(tool, 'schema_quality', 0.0) or 0.0):.2f}",
                getattr(tool, "execution_policy", ""),
                " ".join(tool.retrieval_phrases),
            ]
        )
    )


def _schema_quality_score(
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
) -> float:
    score = 0.0
    if normalize_message_text(schema.command_id):
        score += 0.5
    if normalize_message_text(schema.head):
        score += 0.75
    if normalize_message_text(schema.description or tool.description):
        score += 0.7
    if normalize_message_text(schema.render or tool.render):
        score += 0.8
    if schema.aliases or tool.aliases:
        score += 0.35
    if schema.retrieval_phrases or tool.retrieval_phrases:
        score += 0.45
    if getattr(tool, "capability_text", ""):
        score += 0.45
    if getattr(tool, "use_cases", None):
        score += 0.35
    if getattr(tool, "anti_use_cases", None):
        score += 0.25

    slots = list(schema.slots or [])
    if slots:
        described = sum(
            1
            for slot in slots
            if normalize_message_text(slot.description) or slot.aliases
        )
        score += 0.45 + min(described / max(len(slots), 1), 1.0) * 0.75
        required_count = sum(1 for slot in slots if slot.required)
        if required_count and "{" in str(schema.render or ""):
            score += 0.35

    requires = schema.requires or {}
    if any(bool(requires.get(key)) for key in ("text", "image", "reply", "at")):
        score += 0.35
    if schema.payload_policy not in {"", "none"}:
        score += 0.35
    if schema.target_requirement != "none" or schema.target_sources:
        score += 0.35
    if schema.source in {"explicit", "override"}:
        score += 1.0
    elif schema.source == "metadata":
        score += 0.7
    elif schema.source == "matcher":
        score += 0.35
    elif schema.source == "fallback":
        score -= 0.35
    score += max(0.0, min(float(schema.confidence or 0.0), 1.0)) * 1.2
    return max(0.0, min(score, 7.5))


def _static_schema_quality_score(schema: PluginCommandSchema) -> float:
    score = 0.0
    if normalize_message_text(schema.command_id):
        score += 0.5
    if normalize_message_text(schema.head):
        score += 0.75
    if normalize_message_text(schema.description):
        score += 0.7
    if schema.aliases:
        score += 0.35
    if schema.shortcut_renders:
        score += 0.35
    if schema.retrieval_phrases:
        score += 0.45
    slots = list(schema.slots or [])
    if slots:
        described = sum(
            1
            for slot in slots
            if normalize_message_text(slot.description) or slot.aliases
        )
        score += 0.45 + min(described / max(len(slots), 1), 1.0) * 0.75
    requires = schema.requires or {}
    if any(bool(requires.get(key)) for key in ("text", "image", "reply", "at")):
        score += 0.35
    if schema.payload_policy not in {"", "none"}:
        score += 0.35
    if schema.command_role != "execute":
        score += 0.25
    return max(0.0, min(score, 5.5))


def _query_variants(query: str) -> tuple[str, str]:
    variants = _query_identity_variants(query)
    return variants[0], variants[-1]


def _query_identity_variants(query: str) -> tuple[str, ...]:
    normalized = normalize_message_text(normalize_action_phrases(query or ""))
    variants = tuple(
        variant
        for variant in invoke_prefix_variants(normalized)
        if normalize_message_text(variant)
    )
    return variants or (normalized,)


def _match_exact_or_alias_variants(
    variants: tuple[str, ...],
    schema: PluginCommandSchema,
) -> tuple[bool, bool]:
    return (
        any(match_command_head(text, schema.head) for text in variants),
        any(
            match_command_head(text, alias)
            for text in variants
            for alias in schema.aliases
        ),
    )


def _match_exact_or_alias(
    *,
    normalized: str,
    stripped: str,
    schema: PluginCommandSchema,
) -> tuple[bool, bool]:
    head = normalize_message_text(schema.head)
    aliases = [
        alias
        for alias in (normalize_message_text(item) for item in schema.aliases)
        if alias
    ]
    texts = [normalized, stripped]
    exact_head = any(match_command_head(text, head) for text in texts if text and head)
    exact_alias = any(
        match_command_head(text, alias)
        for text in texts
        for alias in aliases
        if text and alias
    )
    return exact_head, exact_alias


def _shortcut_texts(
    schema: PluginCommandSchema,
    *,
    ignored_aliases: set[str] | None = None,
) -> list[str]:
    texts: list[str] = []
    ignored = ignored_aliases or set()
    for item in schema.shortcut_renders:
        if not isinstance(item, dict):
            continue
        for key in ("alias", "render"):
            value = normalize_message_text(str(item.get(key) or ""))
            if key == "alias" and value.casefold() in ignored:
                continue
            if value:
                texts.append(value)
    return texts


def _expanded_shortcut_aliases_by_module(
    tools: list[CommandToolSnapshot],
) -> dict[str, set[str]]:
    heads_by_module: dict[str, set[str]] = defaultdict(set)
    for tool in tools:
        head = normalize_message_text(tool.head).casefold()
        if head:
            heads_by_module[tool.plugin_module].add(head)

    result: dict[str, set[str]] = defaultdict(set)
    for tool in tools:
        meta = tool.meta if isinstance(tool.meta, dict) else {}
        for item in meta.get("shortcut_renders", []) or []:
            if not isinstance(item, dict):
                continue
            alias = normalize_message_text(str(item.get("alias") or "")).casefold()
            head = normalize_message_text(tool.head).casefold()
            if alias and alias != head and alias in heads_by_module[tool.plugin_module]:
                result[tool.plugin_module].add(alias)
    return result


def _slot_texts(schema: PluginCommandSchema) -> list[str]:
    texts: list[str] = []
    for slot in schema.slots:
        texts.extend(
            [
                slot.name,
                slot.type,
                slot.description,
                " ".join(slot.aliases),
                " ".join(str(choice or "") for choice in slot.choices),
            ]
        )
    requires = schema.requires or {}
    texts.extend(key for key, enabled in requires.items() if enabled)
    texts.extend(schema.target_sources)
    if schema.target_requirement != "none":
        texts.append(schema.target_requirement)
    return [
        normalize_message_text(text)
        for text in texts
        if normalize_message_text(text)
    ]


def _has_cjk_command_boundary(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            return False
        end = index + len(phrase)
        if end >= len(text) or text[end] in _CJK_COMMAND_BOUNDARY_CHARS:
            return True
        start = index + 1


def _is_embedded_short_cjk_match(text: str, phrase: str) -> bool:
    normalized_text = normalize_message_text(text).casefold()
    normalized_phrase = normalize_message_text(phrase).casefold()
    if len(normalized_phrase) > 2:
        return False
    if not normalized_text or not normalized_phrase:
        return False
    if match_command_head(normalized_text, normalized_phrase):
        return False
    return not _has_cjk_command_boundary(normalized_text, normalized_phrase)


def _is_glued_short_cjk_prefix(text: str, phrase: str) -> bool:
    normalized_text = normalize_message_text(text).casefold()
    normalized_phrase = normalize_message_text(phrase).casefold()
    if len(normalized_phrase) > 2:
        return False
    if not normalized_text.startswith(normalized_phrase):
        return False
    if len(normalized_text) <= len(normalized_phrase):
        return False
    next_char = normalized_text[len(normalized_phrase)]
    return (
        "\u4e00" <= next_char <= "\u9fff"
        and next_char not in _CJK_COMMAND_BOUNDARY_CHARS
    )


def _is_single_cjk_prefix(text: str, phrase: str) -> bool:
    normalized_text = normalize_message_text(text).casefold()
    normalized_phrase = normalize_message_text(phrase).casefold()
    return (
        len(normalized_phrase) == 1
        and "\u4e00" <= normalized_phrase <= "\u9fff"
        and normalized_text.startswith(normalized_phrase)
        and len(normalized_text) > len(normalized_phrase)
    )


def _short_cjk_fuzzy_score(text: str, phrases: list[str]) -> float:
    normalized = re.sub(r"[^\u4e00-\u9fff]+", "", normalize_message_text(text))
    if not normalized:
        return 0.0
    best = 0.0
    for raw_phrase in phrases:
        phrase = re.sub(r"[^\u4e00-\u9fff]+", "", normalize_message_text(raw_phrase))
        if not 2 <= len(phrase) <= 4:
            continue
        if phrase in normalized:
            continue
        positions: list[int] = []
        start = 0
        for char in phrase:
            index = normalized.find(char, start)
            if index < 0:
                positions = []
                break
            positions.append(index)
            start = index + 1
        if len(positions) != len(phrase):
            continue
        span = positions[-1] - positions[0] + 1
        if span > len(phrase) + 2:
            continue
        best = max(best, 72.0 - (span - len(phrase)) * 10.0)
    return best


def _has_media_template_context(
    text: str,
    *,
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
    static_metadata_only: bool = False,
) -> bool:
    """Whether a short CJK head is likely a media/template action, not noise."""

    lowered = normalize_message_text(text).casefold()
    if not lowered:
        return False
    role = normalize_message_text(str(schema.command_role or ""))
    payload_policy = normalize_message_text(str(schema.payload_policy or ""))
    media_capability = role == "template" or payload_policy in {
        "image_only",
        "text_or_image",
    }
    if not static_metadata_only:
        output_mode = normalize_message_text(
            str(getattr(tool, "output_mode", "") or "")
        )
        intent_types = {
            normalize_message_text(str(intent or "")).lower()
            for intent in list(getattr(tool, "intent_types", []) or [])
            if normalize_message_text(str(intent or ""))
        }
        media_capability = (
            media_capability
            or output_mode == "image"
            or bool(intent_types & {"media", "generate", "transform"})
            or bool(getattr(tool, "generative", False))
        )
    if not media_capability:
        return False
    return any(term in lowered for term in _MEDIA_CONTEXT_TERMS) and any(
        term in lowered for term in _MEDIA_ACTION_TERMS
    )


def _direct_retrieval_phrase_score(
    *,
    normalized: str,
    schema: PluginCommandSchema,
) -> tuple[float, tuple[str, ...]]:
    """Score explicit schema phrases that token overlap often misses in CJK."""

    lowered = normalize_message_text(normalized).casefold()
    if not lowered:
        return 0.0, ()
    score = 0.0
    reasons: list[str] = []
    seen: set[str] = set()
    for raw_phrase in [
        *list(schema.retrieval_phrases or []),
        schema.description,
    ]:
        phrase = normalize_message_text(str(raw_phrase or "")).casefold()
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        if len(phrase) < 2:
            continue
        if phrase not in lowered:
            continue
        score += min(72.0 + len(phrase) * 8.0, 160.0)
        reasons.append("direct_retrieval_phrase")
    return score, tuple(dict.fromkeys(reasons))


def _success_example_score(*, normalized: str, command_id: str) -> float:
    """Boost commands that previously succeeded for similar user messages.

    This is generic feedback learning: examples are produced by real executions
    as message -> command_id -> slots -> rendered_command and are never tied to
    plugin names.
    """

    query = normalize_message_text(normalized)
    if not query:
        return 0.0
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0

    best = 0.0
    for example in get_command_success_examples(command_id=command_id, limit=8):
        if not isinstance(example, dict):
            continue
        example_text = normalize_message_text(str(example.get("message", "") or ""))
        rendered = normalize_message_text(
            str(example.get("rendered_command", "") or "")
        )
        slots = example.get("slots") or {}
        slot_text = ""
        if isinstance(slots, dict):
            slot_text = normalize_message_text(
                " ".join(str(value or "") for value in slots.values())
            )
        search_text = normalize_message_text(
            " ".join(part for part in (example_text, rendered, slot_text) if part)
        )
        if not search_text:
            continue
        if query == example_text:
            best = max(best, 42.0)
            continue
        tokens = _tokens(search_text)
        if not tokens:
            continue
        overlap = len(query_tokens & tokens)
        if overlap < 2:
            continue
        jaccard = overlap / max(len(query_tokens | tokens), 1)
        containment = overlap / max(min(len(query_tokens), len(tokens)), 1)
        if jaccard >= 0.28 or containment >= 0.55:
            best = max(best, min(12.0 + overlap * 6.0 + containment * 18.0, 42.0))
    return best


def _base_score_tool(
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
    query: str,
    *,
    plugin_name: str = "",
    plugin_module: str = "",
    session_id: str | None = None,
    use_feedback: bool = True,
    static_metadata_only: bool = False,
    router_context: dict[str, Any] | None = None,
    expanded_shortcut_aliases: set[str] | None = None,
) -> tuple[float, tuple[str, ...], bool, CommandCandidateFeatures]:
    normalized, stripped = _query_variants(query)
    lowered = normalized.casefold()
    stripped_lowered = stripped.casefold()
    if not normalized:
        return 0.0, ("empty",), False, _empty_features()

    score = 0.0
    reasons: list[str] = []
    head = normalize_message_text(schema.head).casefold()
    aliases = [
        normalize_message_text(alias).casefold()
        for alias in schema.aliases
        if normalize_message_text(alias)
    ]
    exact_head, exact_alias = _match_exact_or_alias(
        normalized=normalized,
        stripped=stripped,
        schema=schema,
    )
    shortcut_texts = _shortcut_texts(
        schema,
        ignored_aliases=expanded_shortcut_aliases,
    )
    exact_shortcut = any(
        match_command_head(text, shortcut)
        for text in (normalized, stripped)
        for shortcut in shortcut_texts
        if text and shortcut
    )
    if exact_head and any(
        _is_glued_short_cjk_prefix(text, head) for text in (normalized, stripped)
    ):
        exact_head = False
    if exact_alias and any(
        _is_glued_short_cjk_prefix(text, alias)
        for text in (normalized, stripped)
        for alias in aliases
    ):
        exact_alias = False
    exact_protected = exact_head or exact_alias or exact_shortcut
    if exact_head:
        score += _EXACT_BOOST
        reasons.append("exact_head")
    elif exact_alias:
        score += _ALIAS_BOOST
        reasons.append("exact_alias")
    elif exact_shortcut:
        score += _ALIAS_BOOST
        reasons.append("exact_shortcut")

    glued_head_prefix = bool(
        head
        and any(
            _is_glued_short_cjk_prefix(text, head) for text in (normalized, stripped)
        )
    )
    if head and lowered.startswith(head):
        score += 84.0 if glued_head_prefix else 260.0
        reasons.append("head_prefix")
    if (
        head
        and head in lowered
        and not glued_head_prefix
        and not _is_embedded_short_cjk_match(lowered, head)
    ):
        score += 120.0 + min(len(head), 8)
        reasons.append("head")
    elif (
        head
        and len(head) <= 2
        and head in lowered
        and _has_media_template_context(
            lowered,
            tool=tool,
            schema=schema,
            static_metadata_only=static_metadata_only,
        )
    ):
        score += 96.0 + min(len(head), 4) * 8.0
        reasons.append("short_media_template_head")
    for alias in aliases:
        if alias and lowered.startswith(alias):
            score += 240.0
            reasons.append("alias_prefix")
        elif (
            alias
            and alias in lowered
            and not _is_embedded_short_cjk_match(lowered, alias)
        ):
            score += 150.0 + min(len(alias), 12)
            reasons.append("alias")
    for shortcut in shortcut_texts:
        if shortcut and lowered.startswith(shortcut):
            score += 220.0
            reasons.append("shortcut_prefix")
        elif (
            shortcut
            and shortcut in lowered
            and not _is_embedded_short_cjk_match(lowered, shortcut)
        ):
            score += 132.0 + min(len(shortcut), 16)
            reasons.append("shortcut")
    fuzzy_score = _short_cjk_fuzzy_score(
        normalized,
        [head, *aliases, *shortcut_texts],
    )
    if fuzzy_score:
        score += fuzzy_score
        reasons.append("short_cjk_fuzzy")

    phrase_score, phrase_reasons = _direct_retrieval_phrase_score(
        normalized=normalized,
        schema=schema,
    )
    if phrase_score:
        score += phrase_score
        reasons.extend(phrase_reasons)

    usage_example_score = _score_text_overlap(
        normalized,
        [tool.usage or "", *list(tool.examples or [])],
        ascii_weight=10.0,
        cjk_weight=7.0,
        phrase_weight=48.0,
        cap=92.0,
    )
    if usage_example_score:
        score += usage_example_score
        reasons.append("usage_example")

    description_score = _score_text_overlap(
        normalized,
        [schema.description, *list(schema.retrieval_phrases or [])],
        ascii_weight=8.0,
        cjk_weight=6.0,
        phrase_weight=42.0,
        cap=82.0,
    )
    if description_score:
        score += description_score
        reasons.append("description_phrase")

    phrase_text = (
        _static_metadata_text(tool, schema)
        if static_metadata_only
        else _tool_text(tool, schema)
    )
    ascii_overlap = len(_ascii_terms(normalized) & _ascii_terms(phrase_text))
    cjk_overlap = len(_cjk_terms(normalized) & _cjk_terms(phrase_text))
    if ascii_overlap:
        score += min(ascii_overlap * 9.0, 54.0)
        reasons.append("ascii_token")
    if cjk_overlap:
        score += min(cjk_overlap * 5.0, 80.0)
        reasons.append("cjk_ngram")

    slot_score = _score_text_overlap(
        normalized,
        _slot_texts(schema),
        ascii_weight=8.0,
        cjk_weight=6.0,
        phrase_weight=28.0,
        cap=54.0,
    )
    if slot_score:
        score += slot_score
        reasons.append("slot")

    if not (ascii_overlap or cjk_overlap) and not (
        usage_example_score or description_score or slot_score
    ):
        overlap = len(_tokens(normalized) & _tokens(phrase_text))
        score += min(overlap * 8.0, 40.0)
        if overlap:
            reasons.append("retrieval_phrase")

    if not static_metadata_only:
        name_text = normalize_message_text(f"{plugin_name} {plugin_module}").casefold()
        name_overlap = len(_tokens(normalized) & _tokens(name_text))
        if name_overlap:
            score += min(name_overlap * 12.0, 48.0)
            reasons.append("plugin")

    requires = schema.requires or {}
    context = router_context or {}
    try:
        reply_image_count = int(context.get("reply_image_count", 0) or 0)
    except (TypeError, ValueError):
        reply_image_count = 0
    has_image = (
        bool(_IMAGE_PATTERN.search(normalized))
        or bool(context.get("has_image"))
        or reply_image_count > 0
    )
    mentions_image = any(
        term in lowered for term in _MEDIA_CONTEXT_TERMS if len(term) > 1
    )
    has_at = bool(_AT_PATTERN.search(normalized)) or bool(context.get("has_at"))
    has_reply = (
        "[reply:" in lowered
        or "[reply:" in stripped_lowered
        or bool(context.get("has_reply"))
    )
    required_image_slot = any(
        slot.type == "image" and slot.required for slot in schema.slots
    )
    image_required = (
        bool(requires.get("image"))
        or schema.payload_policy == "image_only"
        or required_image_slot
    )
    image_compatible = (
        image_required
        or schema.payload_policy == "text_or_image"
        or any(slot.type == "image" for slot in schema.slots)
    )
    text_compatible = schema.payload_policy in {
        "text",
        "slots",
        "text_or_image",
        "free_tail",
    } or any(slot.type == "text" for slot in schema.slots)
    if has_image and requires.get("image"):
        score += 42.0
        reasons.append("image_signal")
    elif has_image and schema.payload_policy in {"image_only", "text_or_image"}:
        score += 34.0
        reasons.append("image_policy")
    if has_at and requires.get("at"):
        score += 24.0
        reasons.append("at_signal")
    if has_reply:
        if requires.get("reply"):
            score += 32.0
            reasons.append("reply_signal")
    if has_image and image_compatible:
        score += 24.0
        reasons.append("payload_context")
    elif mentions_image and image_compatible and not image_required:
        score += 10.0
        reasons.append("payload_context")
    if text_compatible and any(
        mark in lowered for mark in ("：", ":", "说", "查", "搜")
    ):
        score += 12.0
        reasons.append("payload_context")
    missing_required_contexts = 0
    if image_required and not has_image:
        missing_required_contexts += 1
    if requires.get("reply") and not has_reply:
        missing_required_contexts += 1
    if _missing_required_target_context(
        schema,
        has_at=has_at,
        has_reply=has_reply,
    ):
        missing_required_contexts += 1
    if missing_required_contexts:
        score = max(score - 54.0 * missing_required_contexts, 1.0) if score > 0 else 0.0
        reasons.append("required_context_penalty")

    role = schema.command_role
    if role == "catalog" and any(
        token in lowered for token in ("列表", "有哪些", "打开", "查看", "头像表情")
    ):
        score += 80.0
        reasons.append("catalog")
    if role == "helper" and any(token in lowered for token in ("搜索", "找", "查找")):
        score += 70.0
        reasons.append("helper")
    if role == "helper" and any(
        token in lowered for token in ("支持哪些", "哪些语言", "语种", "支持什么语言")
    ):
        score += 220.0
        reasons.append("helper_langs")
    if role == "random" and "随机" in lowered:
        score += 100.0
        reasons.append("random")

    success_example_score = (
        _success_example_score(
            normalized=normalized,
            command_id=schema.command_id,
        )
        if use_feedback
        else 0.0
    )
    if success_example_score:
        score += success_example_score
        reasons.append("success_example_history")

    evidence_score = score
    if evidence_score <= 0 and not exact_protected:
        return 0.0, ("fallback",), False, _empty_features()

    schema_quality = (
        _static_schema_quality_score(schema)
        if static_metadata_only
        else _schema_quality_score(tool, schema)
    )
    schema_score = schema_quality * 3.0
    if schema_score:
        score += schema_score
        reasons.append("schema_quality")

    feedback_score = 0.0
    reliability_score = 0.0
    false_trigger_score = 0.0
    param_failure_score = 0.0
    latency_score = 0.0
    low_reliability = False
    high_reliability = False
    context_false_trigger = False
    context_param_failure = False
    if use_feedback:
        feedback_profile = get_command_feedback_profile(
            command_id=schema.command_id,
            session_id=session_id,
            plugin_module=plugin_module,
        )
        context_profile = get_contextual_command_feedback_profile(
            message_text=normalized,
            command_id=schema.command_id,
        )
        feedback_score = feedback_profile.feedback_score
        reliability_score = (
            feedback_profile.reliability_score
            + context_profile.reliability_score * 0.7
        )
        false_trigger_score = (
            feedback_profile.false_trigger_score
            + context_profile.false_trigger_score * 0.9
        )
        param_failure_score = (
            feedback_profile.param_failure_score
            + context_profile.param_failure_score * 0.8
        )
        latency_score = feedback_profile.latency_score
        low_reliability = (
            feedback_profile.low_reliability or context_profile.low_reliability
        )
        high_reliability = (
            feedback_profile.high_reliability or context_profile.high_reliability
        )
        context_false_trigger = bool(context_profile.false_trigger_score)
        context_param_failure = bool(context_profile.param_failure_score)
    if success_example_score:
        feedback_score += success_example_score
    if feedback_score:
        score += feedback_score
        reasons.append("feedback")
    if reliability_score:
        score += reliability_score
        if high_reliability:
            reasons.append("reliable_history")
    if false_trigger_score:
        score += false_trigger_score
        reasons.append(
            "context_false_trigger_history"
            if context_false_trigger
            else "false_trigger_history"
        )
    if param_failure_score:
        score += param_failure_score
        reasons.append(
            "context_param_failure_history"
            if context_param_failure
            else "param_failure_history"
        )
    if latency_score:
        score += latency_score
        reasons.append("latency_history")
    if low_reliability and not exact_protected:
        score = max(score - 24.0, 0.0)
        reasons.append("low_reliability_history")

    if (
        not static_metadata_only
        and bool(getattr(tool, "soft_tool", False))
        and not exact_protected
    ):
        score = max(score - 28.0, 1.0)
        reasons.append("soft_tool_penalty")

    deduped_reasons = tuple(dict.fromkeys(reasons)) or ("fallback",)
    features = _build_candidate_features(
        score=score,
        reasons=deduped_reasons,
        feedback_score=feedback_score,
        schema_score=schema_score,
        reliability_score=reliability_score,
        false_trigger_score=false_trigger_score,
        param_failure_score=param_failure_score,
        latency_score=latency_score,
    )
    return score, deduped_reasons, exact_protected, features


def _schema_from_tool_snapshot(tool: CommandToolSnapshot) -> PluginCommandSchema:
    schema = PluginCommandSchema(
        command_id=tool.command_id,
        head=tool.head,
        aliases=tool.aliases,
        description=tool.description,
        slots=tool.slots,
        render=tool.render or tool.head,
        requires=tool.requires,
        allow_at=tool.allow_at,
        actor_scope=tool.actor_scope,
        target_requirement=tool.target_requirement,
        target_sources=list(tool.target_sources),
        command_role=tool.command_role,
        payload_policy=tool.payload_policy,
        extra_text_policy=tool.extra_text_policy,
        source=tool.source,
        confidence=tool.confidence,
        matcher_key=tool.matcher_key,
        retrieval_phrases=tool.retrieval_phrases,
        shortcut_renders=list(
            tool.meta.get("shortcut_renders", []) if isinstance(tool.meta, dict) else []
        ),
    )
    return schema


def _unscored_features(
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
    *,
    session_id: str | None,
    use_feedback: bool = True,
) -> CommandCandidateFeatures:
    schema_score = _schema_quality_score(tool, schema) * 3.0
    if not use_feedback:
        return CommandCandidateFeatures(schema_score=schema_score)
    feedback_profile = get_command_feedback_profile(
        command_id=schema.command_id,
        session_id=session_id,
        plugin_module=tool.plugin_module,
    )
    return CommandCandidateFeatures(
        schema_score=schema_score,
        feedback_score=feedback_profile.feedback_score,
        reliability_score=feedback_profile.reliability_score,
        false_trigger_score=feedback_profile.false_trigger_score,
        param_failure_score=feedback_profile.param_failure_score,
        latency_score=feedback_profile.latency_score,
    )


def _is_scored_identity_match(candidate: _ScoredCandidate) -> bool:
    return candidate.exact_protected or "short_cjk_fuzzy" in candidate.reasons


def _score_static_metadata_bm25_tools(
    tools: list[CommandToolSnapshot],
    query: str,
    *,
    router_context: dict[str, Any] | None = None,
) -> list[_ScoredCandidate]:
    query_terms = _query_terms_no_single(query)
    identity_variants = _query_identity_variants(query)
    normalized, stripped = identity_variants[0], identity_variants[-1]
    if not query_terms and not normalize_message_text(stripped):
        return []

    index = _get_static_bm25_index(tools)
    identity_indexes, longest_prefix_matches = _static_identity_indexes(
        index,
        variants=identity_variants,
        router_context=router_context,
    )
    relevant_indexes = set(identity_indexes)
    for term in query_terms:
        relevant_indexes.update(index.postings.get(term, ()))
    prepared: list[tuple[int, _StaticBm25Document]] = []
    blocked_identity_match = False
    for document_index in sorted(relevant_indexes):
        document = index.documents[document_index]
        schema = document.schema
        if _missing_static_required_context(
            schema,
            query=normalized,
            router_context=router_context,
        ):
            if document_index in identity_indexes:
                blocked_identity_match = True
            continue
        prepared.append((document_index, document))

    scored: list[_ScoredCandidate] = []
    for document_index, document in prepared:
        tool = document.tool
        schema = document.schema
        field_terms = document.field_terms
        score = 0.0
        reasons: list[str] = []
        matched_terms: dict[str, set[str]] = {}
        strong_match = False
        ascii_match = False

        exact_head, exact_alias = _match_exact_or_alias_variants(
            identity_variants,
            schema,
        )
        exact_shortcut = any(
            match_command_head(text, shortcut)
            for text in (normalized, stripped)
            for shortcut in document.shortcut_texts
            if text and shortcut
        )
        exact_protected = exact_head or exact_alias or exact_shortcut
        if exact_head:
            score += _EXACT_BOOST
            reasons.append("exact_head")
        elif exact_alias:
            score += _ALIAS_BOOST
            reasons.append("exact_alias")
        elif exact_shortcut:
            score += _ALIAS_BOOST
            reasons.append("exact_shortcut")

        if document_index in longest_prefix_matches:
            score += 96.0 + min(len(schema.head), 4) * 8.0
            strong_match = True
            reasons.append("head_prefix")

        fuzzy_score = _short_cjk_fuzzy_score(
            normalized,
            [schema.head, *list(schema.aliases or [])],
        )
        if fuzzy_score:
            score += fuzzy_score
            strong_match = True
            reasons.append("short_cjk_fuzzy")

        for field, terms in field_terms.items():
            if not terms:
                continue
            counts = document.field_counts[field]
            doc_len = document.field_lengths[field]
            field_score = 0.0
            field_matches: set[str] = set()
            for term in query_terms:
                term_count = counts.get(term, 0)
                if term_count <= 0:
                    continue
                field_matches.add(term)
                field_score += index.idf.get(term, 1.0) * (
                    (term_count * (_STATIC_BM25_K1 + 1.0))
                    / (
                        term_count
                        + _STATIC_BM25_K1
                        * (
                            1.0
                            - _STATIC_BM25_B
                            + _STATIC_BM25_B * doc_len / _STATIC_BM25_AVGDL
                        )
                    )
                )
            if not field_score:
                continue
            score += field_score * _STATIC_BM25_FIELD_WEIGHTS.get(field, 1.0)
            matched_terms[field] = field_matches
            if field in {
                "head",
                "alias",
                "shortcut",
                "example",
                "usage",
                "description",
                "slot",
            }:
                strong_match = True
            if any(_ASCII_TERM_PATTERN.fullmatch(term) for term in field_matches):
                ascii_match = True
            reasons.append(_static_bm25_reason(field, field_matches))

        if ascii_match:
            reasons.append("ascii_token")
        if score <= 0 or not (exact_protected or strong_match):
            continue

        score, context_reasons = _apply_static_context_score(
            score,
            schema=schema,
            query=normalized,
            router_context=router_context,
        )
        reasons.extend(context_reasons)
        if score <= 0:
            continue

        deduped_reasons = tuple(dict.fromkeys(reasons))
        scored.append(
            _ScoredCandidate(
                tool=tool,
                schema=schema,
                score=score,
                reasons=deduped_reasons,
                exact_protected=exact_protected,
                features=_build_candidate_features(
                    score=score,
                    reasons=deduped_reasons,
                ),
            )
        )

    if blocked_identity_match and not any(
        _is_scored_identity_match(item) for item in scored
    ):
        return []
    return _dynamic_relevance_cut(scored, router_context=router_context)


def _static_context_state(
    query: str,
    router_context: dict[str, Any] | None,
) -> tuple[bool, bool, bool]:
    context = router_context or {}
    lowered = normalize_message_text(query).casefold()
    try:
        reply_image_count = int(context.get("reply_image_count", 0) or 0)
    except (TypeError, ValueError):
        reply_image_count = 0
    has_image = (
        bool(_IMAGE_PATTERN.search(lowered))
        or bool(context.get("has_image"))
        or reply_image_count > 0
    )
    has_reply = "[reply:" in lowered or bool(context.get("has_reply"))
    has_at = bool(_AT_PATTERN.search(lowered)) or bool(context.get("has_at"))
    return has_image, has_reply, has_at


def _missing_required_target_context(
    schema: PluginCommandSchema,
    *,
    has_at: bool,
    has_reply: bool,
) -> bool:
    if schema.target_requirement != "required":
        return False
    requires = schema.requires or {}
    sources = set(schema.target_sources or [])
    accepts_at = bool(
        sources & {"at", "nickname", "self"}
        or schema.allow_at is True
        or requires.get("at")
    )
    accepts_reply = "reply" in sources
    return not ((accepts_at and has_at) or (accepts_reply and has_reply))


def _missing_static_required_context(
    schema: PluginCommandSchema,
    *,
    query: str,
    router_context: dict[str, Any] | None,
) -> bool:
    has_image, has_reply, has_at = _static_context_state(query, router_context)
    requires = schema.requires or {}
    required_image_slot = any(
        slot.type == "image" and slot.required for slot in schema.slots
    )
    image_required = (
        bool(requires.get("image"))
        or schema.payload_policy == "image_only"
        or required_image_slot
    )
    if image_required and not (has_image or has_reply or has_at):
        return True

    if requires.get("reply") and not has_reply:
        return True
    return _missing_required_target_context(
        schema,
        has_at=has_at,
        has_reply=has_reply,
    )


def _apply_static_context_score(
    score: float,
    *,
    schema: PluginCommandSchema,
    query: str,
    router_context: dict[str, Any] | None,
) -> tuple[float, list[str]]:
    has_image, has_reply, has_at = _static_context_state(query, router_context)
    requires = schema.requires or {}
    required_image_slot = any(
        slot.type == "image" and slot.required for slot in schema.slots
    )
    image_required = (
        bool(requires.get("image"))
        or schema.payload_policy == "image_only"
        or required_image_slot
    )
    image_compatible = (
        image_required
        or schema.payload_policy == "text_or_image"
        or any(slot.type == "image" for slot in schema.slots)
    )
    reasons: list[str] = []
    if has_image and requires.get("image"):
        score += 42.0
        reasons.append("image_signal")
    elif has_image and schema.payload_policy in {"image_only", "text_or_image"}:
        score += 34.0
        reasons.append("image_policy")
    if has_image and image_compatible:
        score += 24.0
        reasons.append("payload_context")
    if has_at and requires.get("at"):
        score += 24.0
        reasons.append("at_signal")
    if has_reply and requires.get("reply"):
        score += 32.0
        reasons.append("reply_signal")

    missing_required_contexts = 0
    if image_required and not (has_image or has_reply or has_at):
        missing_required_contexts += 1
    if requires.get("reply") and not has_reply:
        missing_required_contexts += 1
    if _missing_required_target_context(
        schema,
        has_at=has_at,
        has_reply=has_reply,
    ):
        missing_required_contexts += 1
    if missing_required_contexts:
        score = max(score - 54.0 * missing_required_contexts, 1.0)
        reasons.append("required_context_penalty")
    return score, reasons


def _dynamic_relevance_cut(
    candidates: list[_ScoredCandidate],
    *,
    router_context: dict[str, Any] | None = None,
) -> list[_ScoredCandidate]:
    candidates = [item for item in candidates if item.score > 0]
    candidates.sort(
        key=lambda item: (
            item.exact_protected,
            item.score,
            item.schema.command_role in {"catalog", "helper", "random"},
            -len(item.schema.head),
            item.tool.plugin_module,
        ),
        reverse=True,
    )
    if not candidates:
        return []
    top_score = candidates[0].score
    if top_score <= 0:
        return []
    threshold = top_score * _STATIC_BM25_CUT_RATIO
    cap = _dynamic_relevance_cap(router_context)
    selected: list[_ScoredCandidate] = []
    previous = top_score
    for item in candidates[:cap]:
        if selected and item.score < threshold:
            break
        if len(selected) >= 4 and previous > 0 and item.score < previous * 0.55:
            break
        selected.append(item)
        previous = item.score
    return selected


def _dynamic_relevance_cap(router_context: dict[str, Any] | None) -> int:
    context = router_context or {}
    try:
        reply_image_count = int(context.get("reply_image_count", 0) or 0)
    except (TypeError, ValueError):
        reply_image_count = 0
    try:
        task_count = int(context.get("task_count", 0) or 0)
    except (TypeError, ValueError):
        task_count = 0
    if (
        any(bool(context.get(key)) for key in _EXPANDED_RELEVANCE_CONTEXT_KEYS)
        or reply_image_count > 1
        or task_count > 1
    ):
        return 12
    return _STATIC_BM25_CAP


def _score_all_tools(
    tools: list[CommandToolSnapshot],
    query: str,
    *,
    session_id: str | None,
    use_feedback: bool = True,
    static_metadata_only: bool = False,
    router_context: dict[str, Any] | None = None,
    expanded_shortcut_aliases_by_module: dict[str, set[str]] | None = None,
) -> list[_ScoredCandidate]:
    if static_metadata_only and not use_feedback:
        return _score_static_metadata_bm25_tools(
            tools,
            query,
            router_context=router_context,
        )

    scored: list[_ScoredCandidate] = []
    for tool in tools:
        schema = _schema_from_tool_snapshot(tool)
        score, reasons, exact, features = _base_score_tool(
            tool,
            schema,
            query,
            plugin_name=tool.plugin_name,
            plugin_module=tool.plugin_module,
            session_id=session_id,
            use_feedback=use_feedback,
            static_metadata_only=static_metadata_only,
            router_context=router_context,
            expanded_shortcut_aliases=(
                expanded_shortcut_aliases_by_module or {}
            ).get(tool.plugin_module),
        )
        if score <= 0:
            continue
        scored.append(
            _ScoredCandidate(
                tool=tool,
                schema=schema,
                score=score,
                reasons=reasons,
                exact_protected=exact,
                features=features,
            )
        )
    scored.sort(
        key=lambda item: (
            item.exact_protected,
            item.score,
            item.schema.command_role in {"catalog", "helper", "random"},
            -len(item.schema.head),
            item.tool.plugin_module,
        ),
        reverse=True,
    )
    return scored


def _merge_ranked_candidates(
    ranked: list[_ScoredCandidate],
) -> list[_ScoredCandidate]:
    by_id: dict[str, tuple[CommandToolSnapshot, PluginCommandSchema]] = {}
    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)
    exact: dict[str, bool] = defaultdict(bool)
    raw_score: dict[str, float] = defaultdict(float)
    features: dict[str, CommandCandidateFeatures] = defaultdict(_empty_features)

    for rank, candidate in enumerate(ranked, 1):
        command_id = candidate.schema.command_id
        by_id.setdefault(command_id, (candidate.tool, candidate.schema))
        scores[command_id] += 1.0 / (_RRF_K + rank)
        raw_score[command_id] = max(raw_score[command_id], candidate.score)
        if candidate.exact_protected:
            exact[command_id] = True
        current_features = features[command_id]
        features[command_id] = CommandCandidateFeatures(
            lexical_score=max(
                current_features.lexical_score,
                candidate.features.lexical_score,
            ),
            exact_score=max(
                current_features.exact_score,
                candidate.features.exact_score,
            ),
            semantic_score=max(
                current_features.semantic_score,
                candidate.features.semantic_score,
            ),
            slot_score=max(
                current_features.slot_score,
                candidate.features.slot_score,
            ),
            context_score=max(
                current_features.context_score,
                candidate.features.context_score,
            ),
            feedback_score=max(
                current_features.feedback_score,
                candidate.features.feedback_score,
            ),
            schema_score=max(
                current_features.schema_score,
                candidate.features.schema_score,
            ),
            reliability_score=max(
                current_features.reliability_score,
                candidate.features.reliability_score,
            ),
            false_trigger_score=min(
                current_features.false_trigger_score,
                candidate.features.false_trigger_score,
            ),
            param_failure_score=min(
                current_features.param_failure_score,
                candidate.features.param_failure_score,
            ),
            latency_score=_merge_latency_feature(
                current_features.latency_score,
                candidate.features.latency_score,
            ),
            negative_score=min(
                current_features.negative_score,
                candidate.features.negative_score,
            ),
        )
        for reason in candidate.reasons:
            if reason not in reasons[command_id]:
                reasons[command_id].append(reason)

    merged: list[_ScoredCandidate] = []
    for command_id, (tool, schema) in by_id.items():
        final_score = raw_score[command_id] + scores[command_id] * 1000.0
        if exact[command_id]:
            final_score += 500.0
        merged.append(
            _ScoredCandidate(
                tool=tool,
                schema=schema,
                score=final_score,
                reasons=tuple(reasons[command_id]),
                exact_protected=exact[command_id],
                features=features[command_id],
            )
        )
    merged.sort(
        key=lambda item: (
            item.exact_protected,
            item.score,
            item.schema.command_role in {"catalog", "helper", "random"},
            -len(item.schema.head),
            item.tool.plugin_module,
        ),
        reverse=True,
    )
    return merged


def _merge_latency_feature(current: float, incoming: float) -> float:
    if incoming < 0:
        return min(current, incoming)
    if current < 0:
        return current
    return max(current, incoming)


def _diversify_candidates(
    candidates: list[_ScoredCandidate],
    *,
    limit: int,
    diversify: bool,
) -> list[_ScoredCandidate]:
    max_items = max(int(limit or 0), 1)
    if not diversify or len(candidates) <= max_items:
        return candidates[:max_items]

    exact_items = [item for item in candidates if item.exact_protected][
        :_EXACT_KEEP_LIMIT
    ]
    selected: list[_ScoredCandidate] = []
    seen_ids: set[str] = set()
    family_counts: dict[str, int] = {}
    plugin_counts: dict[str, int] = {}

    for item in exact_items:
        selected.append(item)
        seen_ids.add(item.schema.command_id)
        family_counts[item.tool.family] = family_counts.get(item.tool.family, 0) + 1
        plugin_counts[item.tool.plugin_module] = (
            plugin_counts.get(item.tool.plugin_module, 0) + 1
        )

    for item in candidates:
        if len(selected) >= max_items:
            break
        if item.schema.command_id in seen_ids:
            continue
        family_count = family_counts.get(item.tool.family, 0)
        plugin_count = plugin_counts.get(item.tool.plugin_module, 0)
        if family_count >= _FAMILY_SOFT_CAP and len(selected) < max_items // 2:
            continue
        if plugin_count >= _PLUGIN_SOFT_CAP and len(selected) < max_items // 2:
            continue
        selected.append(item)
        seen_ids.add(item.schema.command_id)
        family_counts[item.tool.family] = family_count + 1
        plugin_counts[item.tool.plugin_module] = plugin_count + 1

    if len(selected) < max_items:
        for item in candidates:
            if item.schema.command_id in seen_ids:
                continue
            selected.append(item)
            seen_ids.add(item.schema.command_id)
            if len(selected) >= max_items:
                break
    return selected[:max_items]


def _command_candidate_from_scored(item: _ScoredCandidate) -> CommandCandidate:
    return CommandCandidate(
        plugin_module=item.tool.plugin_module,
        plugin_name=item.tool.plugin_name,
        schema=item.schema,
        score=item.score,
        reason=",".join(item.reasons),
        family=item.tool.family,
        tool=item.tool,
        reasons=item.reasons,
        exact_protected=item.exact_protected,
        features=item.features,
    )


def build_command_candidates(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    limit: int | None = 48,
    session_id: str | None = None,
    diversify: bool = True,
    tools: list[CommandToolSnapshot] | None = None,
    include_unscored: bool = False,
    use_feedback: bool = True,
    use_prefilter: bool = True,
    static_metadata_only: bool = False,
    router_context: dict[str, Any] | None = None,
) -> list[CommandCandidate]:
    if tools is None:
        graph = build_capability_graph_snapshot(knowledge_base)
        tools = build_command_tool_snapshots(graph)
    static_bm25 = static_metadata_only and not use_feedback
    expanded_shortcut_aliases = (
        None if static_bm25 else _expanded_shortcut_aliases_by_module(tools)
    )
    scoring_tools = tools
    if (
        len(tools) >= _INDEX_PREFILTER_MIN_TOOLS
        and not include_unscored
        and use_prefilter
        and query.strip()
    ):
        indexed_tools = _get_command_inverted_index(tools).select(
            query,
            limit=limit,
        )
        if indexed_tools:
            scoring_tools = indexed_tools
    ranked = _score_all_tools(
        scoring_tools,
        query,
        session_id=session_id,
        use_feedback=use_feedback,
        static_metadata_only=static_metadata_only,
        router_context=router_context,
        expanded_shortcut_aliases_by_module=expanded_shortcut_aliases,
    )
    merged = _merge_ranked_candidates(ranked)
    max_items = len(tools) if limit is None or int(limit or 0) <= 0 else int(limit)
    selected = _diversify_candidates(
        merged,
        limit=max_items,
        diversify=diversify,
    )
    if include_unscored and len(selected) < max_items:
        seen_ids = {item.schema.command_id for item in selected}
        for tool in tools:
            if len(selected) >= max_items:
                break
            if tool.command_id in seen_ids:
                continue
            schema = _schema_from_tool_snapshot(tool)
            selected.append(
                _ScoredCandidate(
                    tool=tool,
                    schema=schema,
                    score=0.0,
                    reasons=("full_exposure",),
                    exact_protected=False,
                    features=_unscored_features(
                        tool,
                        schema,
                        session_id=session_id,
                        use_feedback=use_feedback,
                    ),
                )
            )
            seen_ids.add(tool.command_id)
    return [_command_candidate_from_scored(item) for item in selected]


def retrieve_command_candidates(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    limit: int | None = 48,
    session_id: str | None = None,
    diversify: bool = True,
    tools: list[CommandToolSnapshot] | None = None,
    use_feedback: bool = True,
    use_prefilter: bool = True,
    static_metadata_only: bool = False,
    router_context: dict[str, Any] | None = None,
) -> list[CommandCandidate]:
    """Recall command candidates for a query; final execution is decided by LLM."""

    return build_command_candidates(
        knowledge_base,
        query,
        limit=limit,
        session_id=session_id,
        diversify=diversify,
        tools=tools,
        include_unscored=False,
        use_feedback=use_feedback,
        use_prefilter=use_prefilter,
        static_metadata_only=static_metadata_only,
        router_context=router_context,
    )


def group_candidates_by_module(
    candidates: list[CommandCandidate],
) -> dict[str, list[CommandCandidate]]:
    grouped: dict[str, list[CommandCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.plugin_module, []).append(candidate)
    return grouped


def dump_schema_for_prompt(schema: PluginCommandSchema) -> dict[str, object]:
    payload: dict[str, object] = {
        "command_id": schema.command_id,
        "head": schema.head,
        "role": schema.command_role,
        "payload_policy": schema.payload_policy,
        "extra_text_policy": schema.extra_text_policy,
        "actor_scope": schema.actor_scope,
        "target_requirement": schema.target_requirement,
        "target_sources": list(schema.target_sources),
        "allow_at": schema.allow_at,
        "source": schema.source,
        "confidence": schema.confidence,
    }
    if schema.aliases:
        payload["aliases"] = list(schema.aliases)
    if schema.description:
        payload["description"] = schema.description
    true_requires = {
        key: value for key, value in (schema.requires or {}).items() if value
    }
    if true_requires:
        payload["requires"] = true_requires
    if schema.slots:
        payload["slots"] = [
            {
                key: value
                for key, value in {
                    "name": slot.name,
                    "type": slot.type,
                    "required": slot.required or None,
                    "default": slot.default,
                    "aliases": list(slot.aliases) or None,
                    "description": slot.description or None,
                }.items()
                if value is not None
            }
            for slot in schema.slots
        ]
    if schema.render and schema.render != schema.head:
        payload["render"] = schema.render
    return payload


def dump_candidate_for_prompt(
    candidate: CommandCandidate,
    *,
    index: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rank": index,
        "score": round(candidate.score, 2),
        "family": candidate.family,
        "reason": candidate.reason,
        "exact_protected": candidate.exact_protected or None,
        "plugin_module": candidate.plugin_module,
        "plugin_name": candidate.plugin_name,
        "command_id": candidate.schema.command_id,
        "head": candidate.schema.head,
    }
    features = candidate.features or _empty_features()
    feature_payload = {
        key: value
        for key, value in {
            "lexical": round(features.lexical_score, 2),
            "exact": round(features.exact_score, 2),
            "semantic": round(features.semantic_score, 2),
            "slot": round(features.slot_score, 2),
            "context": round(features.context_score, 2),
            "feedback": round(features.feedback_score, 2),
            "schema": round(features.schema_score, 2),
            "reliability": round(features.reliability_score, 2),
            "false_trigger": round(features.false_trigger_score, 2),
            "param_failure": round(features.param_failure_score, 2),
            "latency": round(features.latency_score, 2),
            "negative": round(features.negative_score, 2),
        }.items()
        if value
    }
    if feature_payload:
        payload["features"] = feature_payload
    payload.update(dump_schema_for_prompt(candidate.schema))
    return payload


def build_candidate_snapshots(
    candidates: list[CommandCandidate],
) -> list[CommandCandidateSnapshot]:
    snapshots: list[CommandCandidateSnapshot] = []
    for index, candidate in enumerate(candidates, 1):
        schema = candidate.schema
        snapshots.append(
            CommandCandidateSnapshot(
                rank=index,
                score=candidate.score,
                reason=candidate.reason,
                exact_protected=candidate.exact_protected,
                plugin_module=candidate.plugin_module,
                plugin_name=candidate.plugin_name,
                family=candidate.family,
                command_id=schema.command_id,
                head=schema.head,
                aliases=list(schema.aliases),
                description=schema.description,
                requires=dict(schema.requires or {}),
                slots=list(schema.slots),
                render=schema.render,
                payload_policy=schema.payload_policy,
                command_role=schema.command_role,
                source=schema.source,
                confidence=schema.confidence,
                intent_types=list(getattr(candidate.tool, "intent_types", []) or []),
                requires_real_result=bool(
                    getattr(candidate.tool, "requires_real_result", True)
                ),
                generative=bool(getattr(candidate.tool, "generative", False)),
                execution_policy=str(
                    getattr(candidate.tool, "execution_policy", "normal") or "normal"
                ),
                source_of_truth=str(
                    getattr(candidate.tool, "source_of_truth", "plugin_runtime")
                    or "plugin_runtime"
                ),
                requires_real_tool=bool(
                    getattr(candidate.tool, "requires_real_tool", True)
                ),
                output_mode=str(
                    getattr(candidate.tool, "output_mode", "plugin_output")
                    or "plugin_output"
                ),
                entity_scope=str(
                    getattr(candidate.tool, "entity_scope", "global") or "global"
                ),
                risk=str(
                    getattr(candidate.tool, "risk", "")
                    or getattr(candidate.tool, "risk_level", "low")
                    or "low"
                ),
                reliability=float(getattr(candidate.tool, "reliability", 0.5) or 0.5),
                schema_quality=float(
                    getattr(candidate.tool, "schema_quality", 0.5) or 0.5
                ),
                soft_tool=bool(getattr(candidate.tool, "soft_tool", False)),
                features=candidate.features or _empty_features(),
            )
        )
    return snapshots


__all__ = [
    "CommandCandidate",
    "build_candidate_snapshots",
    "build_command_candidates",
    "dump_candidate_for_prompt",
    "dump_schema_for_prompt",
    "group_candidates_by_module",
    "retrieve_command_candidates",
]
