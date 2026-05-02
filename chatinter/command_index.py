"""Command-level candidate index for ChatInter routing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Literal

from .capability_graph import build_capability_graph_snapshot
from .command_schema import find_command_schema
from .execution_observer import get_command_feedback_score
from .models.pydantic_models import (
    CommandCandidateFeatures,
    CommandCandidateSnapshot,
    CommandToolSnapshot,
    PluginCommandSchema,
    PluginKnowledgeBase,
)
from .plugin_adapters import collect_score_hints, extract_adapter_slots
from .plugin_reference import build_command_tool_snapshots
from .route_text import (
    match_command_head,
    normalize_action_phrases,
    normalize_message_text,
    strip_invoke_prefix,
)

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)
_IMAGE_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_AT_PATTERN = re.compile(r"\[@(?:[^\]\s]+|所有人)\]|@\d{5,20}", re.IGNORECASE)
_EXACT_BOOST = 420.0
_ALIAS_BOOST = 380.0
_SLOT_BOOST = 190.0
_RRF_K = 60.0
_FAMILY_SOFT_CAP = 6
_PLUGIN_SOFT_CAP = 8
_EXACT_KEEP_LIMIT = 8


@dataclass(frozen=True)
class CommandCandidate:
    plugin_module: str
    plugin_name: str
    schema: PluginCommandSchema
    score: float
    reason: str
    family: str = "general"
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


def _empty_features() -> CommandCandidateFeatures:
    return CommandCandidateFeatures()


def _reason_feature_deltas(reason: str) -> dict[str, float]:
    if reason in {"exact_head", "exact_alias"}:
        return {"exact_score": _EXACT_BOOST if reason == "exact_head" else _ALIAS_BOOST}
    if reason in {"head_prefix", "head", "alias_prefix", "alias", "plugin"}:
        return {"lexical_score": 1.0}
    if reason in {
        "retrieval_phrase",
        "catalog",
        "helper",
        "helper_langs",
        "random",
        "template",
        "meme_search_intent",
        "meme_info_intent",
        "meme_catalog_intent",
        "meme_template_missing",
        "about_intent",
        "abbr_intent",
    }:
        return {"semantic_score": 1.0}
    if reason == "slot_signal":
        return {"slot_score": _SLOT_BOOST}
    if reason in {"image_signal", "image_policy", "at_signal", "reply_signal"}:
        return {"context_score": 1.0}
    if reason == "feedback":
        return {"feedback_score": 1.0}
    if reason.endswith("_penalty") or "penalty" in reason:
        return {"negative_score": 1.0}
    return {}


def _build_candidate_features(
    *,
    score: float,
    reasons: tuple[str, ...],
    feedback_score: float = 0.0,
) -> CommandCandidateFeatures:
    values = {
        "lexical_score": 0.0,
        "exact_score": 0.0,
        "semantic_score": 0.0,
        "slot_score": 0.0,
        "context_score": 0.0,
        "feedback_score": float(feedback_score),
        "negative_score": 0.0,
    }
    for reason in reasons:
        for key, delta in _reason_feature_deltas(reason).items():
            values[key] += float(delta)
    if values["lexical_score"]:
        values["lexical_score"] = min(values["lexical_score"] * 45.0, score)
    if values["semantic_score"]:
        values["semantic_score"] = min(values["semantic_score"] * 32.0, score)
    if values["context_score"]:
        values["context_score"] = min(values["context_score"] * 28.0, score)
    if values["negative_score"]:
        values["negative_score"] = -abs(values["negative_score"] * 48.0)
    return CommandCandidateFeatures(**values)


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(normalize_message_text(text))
        if token
    }


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


def _tool_text(tool: CommandToolSnapshot, schema: PluginCommandSchema) -> str:
    return normalize_message_text(
        " ".join(
            [
                _schema_text(schema),
                tool.capability_text,
                " ".join(tool.task_verbs),
                " ".join(tool.input_requirements),
                " ".join(tool.retrieval_phrases),
            ]
        )
    )


def _query_variants(query: str) -> tuple[str, str]:
    normalized = normalize_message_text(normalize_action_phrases(query or ""))
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    return normalized, stripped or normalized


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


def _base_score_tool(
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
    query: str,
    *,
    plugin_name: str = "",
    plugin_module: str = "",
    session_id: str | None = None,
) -> tuple[float, tuple[str, ...], bool, CommandCandidateFeatures]:
    normalized, stripped = _query_variants(query)
    lowered = normalized.casefold()
    stripped_lowered = stripped.casefold()
    if not normalized:
        return 0.0, ("empty",), False, _empty_features()

    score = 0.0
    reasons: list[str] = []
    exact_head, exact_alias = _match_exact_or_alias(
        normalized=normalized,
        stripped=stripped,
        schema=schema,
    )
    exact_protected = exact_head or exact_alias
    if exact_head:
        score += _EXACT_BOOST
        reasons.append("exact_head")
    elif exact_alias:
        score += _ALIAS_BOOST
        reasons.append("exact_alias")

    head = normalize_message_text(schema.head).casefold()
    aliases = [
        normalize_message_text(alias).casefold()
        for alias in schema.aliases
        if normalize_message_text(alias)
    ]
    if head and lowered.startswith(head):
        score += 260.0
        reasons.append("head_prefix")
    if head and head in lowered:
        score += 120.0 + min(len(head), 8)
        reasons.append("head")
    for alias in aliases:
        if alias and lowered.startswith(alias):
            score += 240.0
            reasons.append("alias_prefix")
        elif alias and alias in lowered:
            score += 150.0 + min(len(alias), 12)
            reasons.append("alias")

    phrase_text = _tool_text(tool, schema)
    overlap = len(_tokens(normalized) & _tokens(phrase_text))
    if overlap:
        score += min(overlap * 16.0, 112.0)
        reasons.append("retrieval_phrase")

    name_text = normalize_message_text(f"{plugin_name} {plugin_module}").casefold()
    name_overlap = len(_tokens(normalized) & _tokens(name_text))
    if name_overlap:
        score += min(name_overlap * 12.0, 48.0)
        reasons.append("plugin")

    requires = schema.requires or {}
    has_image = bool(_IMAGE_PATTERN.search(normalized))
    has_at = bool(_AT_PATTERN.search(normalized))
    if has_image and requires.get("image"):
        score += 42.0
        reasons.append("image_signal")
    elif has_image and schema.payload_policy in {"image_only", "text_or_image"}:
        score += 34.0
        reasons.append("image_policy")
    if has_at and requires.get("at"):
        score += 24.0
        reasons.append("at_signal")
    if "[reply:" in lowered or "[reply:" in stripped_lowered:
        if requires.get("reply"):
            score += 32.0
            reasons.append("reply_signal")

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
    for hint in collect_score_hints(
        schema,
        lowered_query=lowered,
        stripped_lowered_query=stripped_lowered,
        plugin_module=plugin_module,
    ):
        score += hint.score
        reasons.append(hint.reason)
    if extract_adapter_slots(schema.command_id, normalized):
        score += _SLOT_BOOST
        reasons.append("slot_signal")

    feedback_score = get_command_feedback_score(
        command_id=schema.command_id,
        session_id=session_id,
        plugin_module=plugin_module,
    )
    if feedback_score:
        score += feedback_score
        reasons.append("feedback")

    deduped_reasons = tuple(dict.fromkeys(reasons)) or ("fallback",)
    features = _build_candidate_features(
        score=score,
        reasons=deduped_reasons,
        feedback_score=feedback_score,
    )
    return score, deduped_reasons, exact_protected, features


def score_command_schema(
    schema: PluginCommandSchema,
    query: str,
    *,
    plugin_name: str = "",
    plugin_module: str = "",
    session_id: str | None = None,
) -> tuple[float, str]:
    tool = CommandToolSnapshot(
        command_id=schema.command_id,
        plugin_module=plugin_module,
        plugin_name=plugin_name,
        head=schema.head,
        aliases=list(schema.aliases),
        description=schema.description,
        slots=list(schema.slots),
        requires=dict(schema.requires or {}),
        render=schema.render,
        payload_policy=schema.payload_policy,
        extra_text_policy=schema.extra_text_policy,
        command_role=schema.command_role,
        retrieval_phrases=list(schema.retrieval_phrases),
    )
    score, reasons, _exact, _features = _base_score_tool(
        tool,
        schema,
        query,
        plugin_name=plugin_name,
        plugin_module=plugin_module,
        session_id=session_id,
    )
    return score, ",".join(reasons)


def _score_all_tools(
    tools: list[CommandToolSnapshot],
    query: str,
    *,
    session_id: str | None,
) -> list[_ScoredCandidate]:
    scored: list[_ScoredCandidate] = []
    for tool in tools:
        schema = PluginCommandSchema(
            command_id=tool.command_id,
            head=tool.head,
            aliases=tool.aliases,
            description=tool.description,
            slots=tool.slots,
            render=tool.render or tool.head,
            requires=tool.requires,
            command_role=tool.command_role,
            payload_policy=tool.payload_policy,
            extra_text_policy=tool.extra_text_policy,
            source=tool.source,
            confidence=tool.confidence,
            matcher_key=tool.matcher_key,
            retrieval_phrases=tool.retrieval_phrases,
        )
        score, reasons, exact, features = _base_score_tool(
            tool,
            schema,
            query,
            plugin_name=tool.plugin_name,
            plugin_module=tool.plugin_module,
            session_id=session_id,
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


def build_command_candidates(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    limit: int = 48,
    session_id: str | None = None,
    diversify: bool = True,
    tools: list[CommandToolSnapshot] | None = None,
) -> list[CommandCandidate]:
    if tools is None:
        graph = build_capability_graph_snapshot(knowledge_base)
        tools = build_command_tool_snapshots(graph)
    ranked = _score_all_tools(tools, query, session_id=session_id)
    merged = _merge_ranked_candidates(ranked)
    selected = _diversify_candidates(
        merged,
        limit=limit,
        diversify=diversify,
    )
    return [
        CommandCandidate(
            plugin_module=item.tool.plugin_module,
            plugin_name=item.tool.plugin_name,
            schema=item.schema,
            score=item.score,
            reason=",".join(item.reasons),
            family=item.tool.family,
            reasons=item.reasons,
            exact_protected=item.exact_protected,
            features=item.features,
        )
        for item in selected
    ]


def build_recovered_command_candidates(
    knowledge_base: PluginKnowledgeBase,
    *,
    original_query: str,
    capability_query: str,
    limit: int = 48,
    session_id: str | None = None,
    tools: list[CommandToolSnapshot] | None = None,
) -> list[CommandCandidate]:
    """Recall tools with a rewritten capability query, scoped to installed tools."""

    if tools is None:
        graph = build_capability_graph_snapshot(knowledge_base)
        tools = build_command_tool_snapshots(graph)
    recovered = build_command_candidates(
        knowledge_base,
        capability_query,
        limit=max(limit * 2, limit),
        session_id=session_id,
        diversify=False,
        tools=tools,
    )
    if not recovered:
        return []
    original_ranked = {
        candidate.schema.command_id: candidate
        for candidate in build_command_candidates(
            knowledge_base,
            original_query,
            limit=max(limit, 8),
            session_id=session_id,
            diversify=False,
            tools=tools,
        )
    }
    merged: list[CommandCandidate] = []
    for candidate in recovered:
        original = original_ranked.get(candidate.schema.command_id)
        score = candidate.score + (original.score * 0.25 if original else 0.0)
        reason = f"recovery:{candidate.reason}"
        if original is not None:
            reason = f"{reason};original:{original.reason}"
        merged.append(
            CommandCandidate(
                plugin_module=candidate.plugin_module,
                plugin_name=candidate.plugin_name,
                schema=candidate.schema,
                score=score,
                reason=reason,
                family=candidate.family,
                reasons=("recovery", *candidate.reasons),
                exact_protected=candidate.exact_protected,
                features=candidate.features,
            )
        )
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:limit]


def group_candidates_by_module(
    candidates: list[CommandCandidate],
) -> dict[str, list[CommandCandidate]]:
    grouped: dict[str, list[CommandCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.plugin_module, []).append(candidate)
    return grouped


def dump_schema_for_prompt(
    schema: PluginCommandSchema,
    *,
    compact: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "command_id": schema.command_id,
        "head": schema.head,
        "role": schema.command_role,
        "payload_policy": schema.payload_policy,
    }
    if schema.aliases:
        payload["aliases"] = schema.aliases[: 4 if compact else 10]
    if schema.description:
        payload["description"] = schema.description
    true_requires = {
        key: value for key, value in (schema.requires or {}).items() if value
    }
    if true_requires:
        payload["requires"] = true_requires
    if compact:
        return payload

    payload["extra_text_policy"] = schema.extra_text_policy
    payload["source"] = schema.source
    payload["confidence"] = schema.confidence
    if schema.slots:
        payload["slots"] = [
            {
                key: value
                for key, value in {
                    "name": slot.name,
                    "type": slot.type,
                    "required": slot.required or None,
                    "default": slot.default,
                    "aliases": slot.aliases[:5] or None,
                    "description": slot.description or None,
                }.items()
                if value is not None
            }
            for slot in schema.slots[:4]
        ]
    if schema.render and schema.render != schema.head:
        payload["render"] = schema.render
    return payload


def _prompt_level_for_candidate(
    index: int,
    candidate: CommandCandidate,
) -> Literal["full", "compact", "name_only"]:
    if candidate.exact_protected or index <= 8:
        return "full"
    if index <= 24:
        return "compact"
    return "name_only"


def dump_candidate_for_prompt(
    candidate: CommandCandidate,
    *,
    index: int,
) -> dict[str, object]:
    level = _prompt_level_for_candidate(index, candidate)
    payload: dict[str, object] = {
        "rank": index,
        "score": round(candidate.score, 2),
        "family": candidate.family,
        "reason": candidate.reason,
        "exact_protected": candidate.exact_protected or None,
        "prompt_level": level,
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
            "negative": round(features.negative_score, 2),
        }.items()
        if value
    }
    if feature_payload:
        payload["features"] = feature_payload
    if level == "name_only":
        if candidate.schema.description:
            payload["description"] = candidate.schema.description[:80]
        return payload
    payload.update(
        dump_schema_for_prompt(
            candidate.schema,
            compact=level == "compact",
        )
    )
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
                features=candidate.features or _empty_features(),
                prompt_level=_prompt_level_for_candidate(index, candidate),
            )
        )
    return snapshots


def find_schema_in_candidates(
    candidates: list[CommandCandidate],
    *,
    command_id: str | None = None,
    command: str | None = None,
) -> PluginCommandSchema | None:
    schemas = [candidate.schema for candidate in candidates]
    return find_command_schema(schemas, command_id=command_id, command=command)


__all__ = [
    "CommandCandidate",
    "build_candidate_snapshots",
    "build_command_candidates",
    "build_recovered_command_candidates",
    "dump_candidate_for_prompt",
    "dump_schema_for_prompt",
    "find_schema_in_candidates",
    "group_candidates_by_module",
    "score_command_schema",
]
