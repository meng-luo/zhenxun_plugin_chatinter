"""Command-level candidate index for ChatInter routing."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .capability_graph import build_capability_graph_snapshot
from .execution_observer import get_command_feedback_score
from .models.pydantic_models import PluginCommandSchema, PluginKnowledgeBase
from .plugin_reference import build_plugin_references
from .route_text import normalize_action_phrases, normalize_message_text
from .slot_extractors import extract_builtin_slots

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)
_IMAGE_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_AT_PATTERN = re.compile(r"\[@(?:[^\]\s]+|所有人)\]|@\d{5,20}", re.IGNORECASE)


@dataclass(frozen=True)
class CommandCandidate:
    plugin_module: str
    plugin_name: str
    schema: PluginCommandSchema
    score: float
    reason: str
    family: str = "general"
    reasons: tuple[str, ...] = ()


def _candidate_family(schema: PluginCommandSchema, *, plugin_module: str) -> str:
    command_id = normalize_message_text(schema.command_id).casefold()
    module = normalize_message_text(plugin_module).casefold()
    head = normalize_message_text(schema.head)
    if "meme" in module or command_id.startswith("memes."):
        return "meme"
    if "gold_redbag" in module or "红包" in head:
        return "gold_redbag"
    if "translate" in module or command_id.startswith("translate."):
        return "translate"
    if "roll" in module or command_id.startswith("roll."):
        return "choice"
    if schema.command_role in {"catalog", "helper", "usage"}:
        return schema.command_role
    return module.rsplit(".", 1)[-1] or "general"


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
                schema.description,
                schema.command_role,
                schema.payload_policy,
                slot_text,
            ]
        )
    )


def score_command_schema(
    schema: PluginCommandSchema,
    query: str,
    *,
    plugin_name: str = "",
    plugin_module: str = "",
    session_id: str | None = None,
) -> tuple[float, str]:
    normalized = normalize_message_text(normalize_action_phrases(query or ""))
    lowered = normalized.casefold()
    if not normalized:
        return 0.0, "empty"

    score = 0.0
    reasons: list[str] = []
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

    text = _schema_text(schema)
    overlap = len(_tokens(normalized) & _tokens(text))
    if overlap:
        score += min(overlap * 16.0, 96.0)
        reasons.append("tokens")

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
        reasons.append("image_fit")
    elif has_image and schema.payload_policy in {"image_only", "text_or_image"}:
        score += 34.0
        reasons.append("image_policy")
    if has_at and requires.get("at"):
        score += 24.0
        reasons.append("at_fit")

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
    if (
        schema.command_id == "about.info"
        and any(token in lowered for token in ("真寻", "小真寻", "bot", "机器人"))
        and any(token in lowered for token in ("信息", "介绍", "了解", "项目"))
    ):
        score += 120.0
        reasons.append("about_intent")
    if schema.command_id == "nbnhhsh.expand" and re.search(
        r"[0-9A-Za-z_]{2,}\s*(?:是)?(?:什么|啥|哪个)?缩写",
        lowered,
        re.IGNORECASE,
    ):
        score += 120.0
        reasons.append("abbr_intent")
    if role == "template" and any(
        token in lowered for token in ("表情", "表情包", "梗图", "头像")
    ):
        score += 38.0
        reasons.append("template")
    if schema.command_id == "memes.search" and any(
        token in lowered for token in ("相关表情", "找一下", "搜一下", "搜索")
    ):
        score += 260.0
        reasons.append("meme_search_intent")
    if schema.command_id == "memes.info" and any(
        token in lowered for token in ("怎么用", "用法", "详情")
    ):
        score += 220.0
        reasons.append("meme_info_intent")
    if schema.requires.get("text") and any(
        token in lowered for token in ("支持哪些", "哪些语言", "语种", "支持什么语言")
    ):
        score -= 360.0
        reasons.append("text_lang_penalty")
    if extract_builtin_slots(schema.command_id, normalized):
        score += 190.0
        reasons.append("slots")

    feedback_score = get_command_feedback_score(
        command_id=schema.command_id,
        session_id=session_id,
        plugin_module=plugin_module,
    )
    if feedback_score:
        score += feedback_score
        reasons.append("feedback")

    return score, ",".join(dict.fromkeys(reasons)) or "fallback"


def build_command_candidates(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    limit: int = 48,
    session_id: str | None = None,
    diversify: bool = True,
) -> list[CommandCandidate]:
    graph = build_capability_graph_snapshot(knowledge_base)
    references = build_plugin_references(graph)
    candidates: list[CommandCandidate] = []
    for reference in references:
        for schema in reference.command_schemas:
            score, reason = score_command_schema(
                schema,
                query,
                plugin_name=reference.name,
                plugin_module=reference.module,
                session_id=session_id,
            )
            if score <= 0:
                continue
            reasons = tuple(item for item in reason.split(",") if item)
            candidates.append(
                CommandCandidate(
                    plugin_module=reference.module,
                    plugin_name=reference.name,
                    schema=schema,
                    score=score,
                    reason=reason,
                    family=_candidate_family(
                        schema,
                        plugin_module=reference.module,
                    ),
                    reasons=reasons,
                )
            )
    candidates.sort(
        key=lambda item: (
            item.score,
            item.schema.command_role in {"catalog", "helper", "random"},
            -len(item.schema.head),
            item.plugin_module,
        ),
        reverse=True,
    )
    max_items = max(int(limit or 0), 1)
    if not diversify or len(candidates) <= max_items:
        return candidates[:max_items]

    # Keep the very top candidates, but avoid a single broad plugin family
    # monopolising the whole LLM shortlist. ToolRerank-style family diversity is
    # useful for fuzzy requests where several plugins share generic words.
    selected: list[CommandCandidate] = []
    family_counts: dict[str, int] = {}
    soft_family_cap = 6
    for candidate in candidates:
        count = family_counts.get(candidate.family, 0)
        if count >= soft_family_cap and len(selected) < max_items // 2:
            continue
        selected.append(candidate)
        family_counts[candidate.family] = count + 1
        if len(selected) >= max_items:
            break
    if len(selected) < max_items:
        seen_ids = {item.schema.command_id for item in selected}
        for candidate in candidates:
            if candidate.schema.command_id in seen_ids:
                continue
            selected.append(candidate)
            if len(selected) >= max_items:
                break
    return selected[:max_items]


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
        "source": schema.source,
        "confidence": schema.confidence,
    }
    if schema.aliases:
        payload["aliases"] = schema.aliases[:10]
    if schema.description:
        payload["description"] = schema.description
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
    true_requires = {
        key: value for key, value in (schema.requires or {}).items() if value
    }
    if true_requires:
        payload["requires"] = true_requires
    return payload


__all__ = [
    "CommandCandidate",
    "build_command_candidates",
    "dump_schema_for_prompt",
    "group_candidates_by_module",
    "score_command_schema",
]
