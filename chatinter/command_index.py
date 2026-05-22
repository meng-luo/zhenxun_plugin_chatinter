"""Recall-only command candidate index for ChatInter.

This module ranks candidates to keep the prompt small.  It must not be treated
as the final command selector; executable schema choice belongs to the LLM.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

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
    match_command_head,
    normalize_action_phrases,
    normalize_message_text,
    strip_invoke_prefix,
)

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)
_IMAGE_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_AT_PATTERN = re.compile(r"\[@(?:[^\]\s]+|所有人)\]|@\d{5,20}", re.IGNORECASE)
_EXACT_BOOST = 180.0
_ALIAS_BOOST = 160.0
_RRF_K = 60.0
_FAMILY_SOFT_CAP = 6
_PLUGIN_SOFT_CAP = 8
_EXACT_KEEP_LIMIT = 8
_CJK_COMMAND_BOUNDARY_CHARS = frozenset(" ，,。.!！？?；;：:/|）)]】}《<>")
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
    }:
        return {"semantic_score": 1.0}
    if reason in {"image_signal", "image_policy", "at_signal", "reply_signal"}:
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
                f"schema_quality {float(getattr(tool, 'schema_quality', 0.0) or 0.0):.2f}",
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
    if len(normalized_phrase) > 1:
        return False
    if not normalized_text or not normalized_phrase:
        return False
    if match_command_head(normalized_text, normalized_phrase):
        return False
    return not _has_cjk_command_boundary(normalized_text, normalized_phrase)


def _has_media_template_context(
    text: str,
    *,
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
) -> bool:
    """Whether a short CJK head is likely a media/template action, not noise."""

    lowered = normalize_message_text(text).casefold()
    if not lowered:
        return False
    output_mode = normalize_message_text(str(getattr(tool, "output_mode", "") or ""))
    intent_types = {
        normalize_message_text(str(intent or "")).lower()
        for intent in list(getattr(tool, "intent_types", []) or [])
        if normalize_message_text(str(intent or ""))
    }
    role = normalize_message_text(str(schema.command_role or ""))
    payload_policy = normalize_message_text(str(schema.payload_policy or ""))
    media_capability = (
        role == "template"
        or payload_policy in {"image_only", "text_or_image"}
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
    if head and head in lowered and not _is_embedded_short_cjk_match(lowered, head):
        score += 120.0 + min(len(head), 8)
        reasons.append("head")
    elif (
        head
        and len(head) <= 2
        and head in lowered
        and _has_media_template_context(lowered, tool=tool, schema=schema)
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

    phrase_score, phrase_reasons = _direct_retrieval_phrase_score(
        normalized=normalized,
        schema=schema,
    )
    if phrase_score:
        score += phrase_score
        reasons.extend(phrase_reasons)

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

    success_example_score = _success_example_score(
        normalized=normalized,
        command_id=schema.command_id,
    )
    if success_example_score:
        score += success_example_score
        reasons.append("success_example_history")

    evidence_score = score
    if evidence_score <= 0 and not exact_protected:
        return 0.0, ("fallback",), False, _empty_features()

    schema_quality = _schema_quality_score(tool, schema)
    schema_score = schema_quality * 3.0
    if schema_score:
        score += schema_score
        reasons.append("schema_quality")

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
    if success_example_score:
        feedback_score += success_example_score
    if feedback_score:
        score += feedback_score
        reasons.append("feedback")
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
    if reliability_score:
        score += reliability_score
        if feedback_profile.high_reliability or context_profile.high_reliability:
            reasons.append("reliable_history")
    if false_trigger_score:
        score += false_trigger_score
        reasons.append(
            "context_false_trigger_history"
            if context_profile.false_trigger_score
            else "false_trigger_history"
        )
    if param_failure_score:
        score += param_failure_score
        reasons.append(
            "context_param_failure_history"
            if context_profile.param_failure_score
            else "param_failure_history"
        )
    if latency_score:
        score += latency_score
        reasons.append("latency_history")
    if (
        feedback_profile.low_reliability or context_profile.low_reliability
    ) and not exact_protected:
        score = max(score - 24.0, 0.0)
        reasons.append("low_reliability_history")

    if bool(getattr(tool, "soft_tool", False)) and not exact_protected:
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
    return PluginCommandSchema(
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
    )


def _unscored_features(
    tool: CommandToolSnapshot,
    schema: PluginCommandSchema,
    *,
    session_id: str | None,
) -> CommandCandidateFeatures:
    schema_score = _schema_quality_score(tool, schema) * 3.0
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


def _score_all_tools(
    tools: list[CommandToolSnapshot],
    query: str,
    *,
    session_id: str | None,
) -> list[_ScoredCandidate]:
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
) -> list[CommandCandidate]:
    if tools is None:
        graph = build_capability_graph_snapshot(knowledge_base)
        tools = build_command_tool_snapshots(graph)
    ranked = _score_all_tools(tools, query, session_id=session_id)
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
