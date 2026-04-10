from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, cast
from typing import Literal

from pydantic import BaseModel, Field

from zhenxun.services import chat, generate_structured, logger
from zhenxun.services.llm.types import LLMToolCall

from .config import (
    ROUTE_TOOL_PLANNER_ENABLED,
    ROUTE_TOOL_PLANNER_MAX_TOOLS,
    get_config_value,
    get_model_name,
)
from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .prompt_guard import guard_prompt_sections
from .route_tool_planner import RoutePlannerToolSpec, build_route_planner_tools
from .route_text import (
    collect_placeholders,
    contains_any,
    is_usage_question,
    match_command_head_or_sticky,
    normalize_action_phrases,
    normalize_message_text,
    ROUTE_ACTION_WORDS,
    strip_invoke_prefix,
)
from .route_policy import is_route_action_compatible
from .route_policy import infer_message_action_role
from .route_policy import infer_route_action_role
from .schema_policy import resolve_command_target_policy
from .skill_registry import (
    SkillRouteDecision,
    get_skill_registry,
    infer_query_family,
    infer_query_families,
    render_skill_namespace,
    select_relevant_skills,
    skill_execute,
    skill_search,
)
from .turn_runtime import TurnBudgetController

_ROUTE_NAMESPACE_LIMIT = 12
_ROUTE_PLUGIN_SHORTLIST_LIMIT = 8
_ROUTE_TOOLS_PER_SKILL_LIMIT = 2
_SHORTLIST_ALIGNMENT_TOOLS_PER_SKILL_LIMIT = 4
_SHORTLIST_ALIGNMENT_INSTRUCTION = """
你是 ChatInter 的 shortlist 对齐器。
给你的不是全库，而是当前消息最可能命中的少量插件卡片。
你的任务只有三种：
1. route：明显是在调用插件，并且已经能确定插件与插件内命令。
2. usage：明显有插件调用意图，但参数、图片、目标或上下文不够，适合返回插件用法。
3. skip：更像普通聊天，或者仍然无法可靠判断。

严格遵守：
1. 只能从 skills_json 里选择 plugin_module 与 command。
2. command 只能填命令头，不要把数字、文本、[@...]、[image] 拼进去；本地会绑定显式参数。
3. 不要臆造当前消息里不存在的参数、目标或图片。
4. “怎么用/帮助/用法/参数/说明”优先 usage。
5. 对红包、查询、搜索、今日类命令，优先依据动作词和时间词选择插件内正确动作。
6. 无法可靠判断时返回 skip。
""".strip()
_TOOL_ROUTE_INSTRUCTION = """
你是 ChatInter 的工具式路由规划器。
给你的每个工具都已经绑定到一个明确的插件命令。
严格遵守：
1. 明显对应当前诉求的工具才调用，且只调用一个。
2. text 只填额外纯文本，不要重复命令头、[@...]、[image]、reply。
3. 命令不需要文本参数时不要传 text。
4. 像普通对话、闲聊、澄清或把握不足时，直接回复 SKIP。
5. “怎么用/用法/帮助/参数”类问题优先帮助型命令，不要误调用业务动作命令。
""".strip()
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_AT_PLACEHOLDER_PATTERN = re.compile(r"\[@\d{5,20}\]")
_AT_INLINE_PATTERN = re.compile(r"@\d{5,20}")
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_ROUTE_TRACE_SAMPLE_LIMIT = 12
_SHORTLIST_ENTITY_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_SHORTLIST_ENTITY_STOPWORDS = {
    "帮我",
    "请",
    "麻烦",
    "一下",
    "一个",
    "一张",
    "这个",
    "那个",
    "看看",
    "看下",
    "查看",
    "查询",
    "设置",
    "生成",
    "制作",
    "来个",
    "做个",
    "做一个",
    "做一张",
    "什么",
    "怎么",
    "为什么",
    "多少",
    "一下子",
    "可以",
}
_SHORTLIST_COMPACT_ROUTE_HINTS = ("抽", "找", "搜", "查", "查询", "来个", "做个")
_SHORTLIST_COMPACT_QUESTION_HINTS = ("什么", "怎么", "为什么", "多少", "吗", "嘛", "呢")


@dataclass(frozen=True)
class RouteResolveResult:
    decision: SkillRouteDecision
    stage: str
    report: "RouteAttemptReport | None" = None


@dataclass
class RouteAttemptReport:
    helper_mode: bool
    candidate_total: int = 0
    lexical_candidates: int = 0
    direct_candidates: int = 0
    vector_candidates: int = 0
    attempts: int = 0
    tool_attempts: int = 0
    tool_candidates: int = 0
    tool_choice_count: int = 0
    final_reason: str = "init"
    selected_stage: str | None = None
    selected_plugin: str | None = None
    selected_module: str | None = None
    selected_command: str | None = None
    attempt_modules: list[list[str]] = field(default_factory=list)

    def note_attempt(self, modules: list[str]) -> None:
        self.attempts += 1
        self.attempt_modules.append(modules[:_ROUTE_TRACE_SAMPLE_LIMIT])

    def note_tool_pool(self, tool_count: int, choice_count: int = 0) -> None:
        self.tool_attempts += 1
        self.tool_candidates = max(self.tool_candidates, max(tool_count, 0))
        self.tool_choice_count += max(choice_count, 0)

    def finalize(
        self,
        *,
        reason: str,
        stage: str | None = None,
        plugin_name: str | None = None,
        plugin_module: str | None = None,
        command: str | None = None,
    ) -> None:
        self.final_reason = reason
        if stage is not None:
            self.selected_stage = stage
        if plugin_name is not None:
            self.selected_plugin = plugin_name
        if plugin_module is not None:
            self.selected_module = plugin_module
        if command is not None:
            self.selected_command = command


class _LLMRouteDecision(BaseModel):
    action: Literal["route", "skip"] = Field(
        default="skip", description="route=调用插件；skip=不路由"
    )
    plugin_module: str | None = Field(
        default=None, description="目标插件模块，必须来自 skills_json"
    )
    plugin_name: str | None = Field(default=None, description="目标插件名称，可选")
    command: str | None = Field(default=None, description="最终可执行命令")


class _ShortlistAlignmentDecision(BaseModel):
    action: Literal["route", "usage", "skip"] = Field(
        default="skip",
        description="route=直接执行；usage=返回该插件用法；skip=按普通聊天处理",
    )
    plugin_module: str | None = Field(
        default=None,
        description="目标插件模块，必须来自 skills_json",
    )
    plugin_name: str | None = Field(default=None, description="目标插件名称，可选")
    command: str | None = Field(
        default=None,
        description="命令头，不要带参数、[@...]、[image]",
    )


def _resolve_target_plugin(
    decision: _LLMRouteDecision,
    knowledge_base: PluginKnowledgeBase,
) -> PluginInfo | None:
    module_text = normalize_message_text(decision.plugin_module or "")
    if module_text:
        module_fold = module_text.casefold()
        for plugin in knowledge_base.plugins:
            if normalize_message_text(plugin.module).casefold() == module_fold:
                return plugin
        tail_matches = [
            plugin
            for plugin in knowledge_base.plugins
            if normalize_message_text(plugin.module.rsplit(".", 1)[-1]).casefold()
            == module_fold
        ]
        if len(tail_matches) == 1:
            return tail_matches[0]

    name_text = normalize_message_text(decision.plugin_name or "")
    if name_text:
        name_fold = name_text.casefold()
        for plugin in knowledge_base.plugins:
            if normalize_message_text(plugin.name).casefold() == name_fold:
                return plugin
    return None


def _is_public_command_meta(meta: object) -> bool:
    return (
        normalize_message_text(str(getattr(meta, "access_level", "public") or "public"))
        .lower()
        == "public"
    )


def _collect_allowed_heads(plugin: PluginInfo) -> set[str]:
    allowed: set[str] = set()
    if plugin.command_meta:
        metas = plugin.command_meta
    else:
        metas = []
    for meta in metas:
        if not _is_public_command_meta(meta):
            continue
        command_head = normalize_message_text(getattr(meta, "command", ""))
        if command_head:
            allowed.add(command_head.casefold())
        for alias in getattr(meta, "aliases", None) or []:
            alias_text = normalize_message_text(alias)
            if alias_text:
                allowed.add(alias_text.casefold())
    if not allowed:
        for command in plugin.commands:
            normalized = normalize_message_text(command)
            if normalized:
                allowed.add(normalized.casefold())
    return allowed


def _collect_helper_heads(
    plugin_module: str,
    knowledge_base: PluginKnowledgeBase,
) -> set[str]:
    registry = get_skill_registry(knowledge_base)
    for skill in registry.skills:
        if skill.plugin_module != plugin_module:
            continue
        return {
            normalize_message_text(command).casefold()
            for command in skill.helper_commands
            if normalize_message_text(command)
        }
    return set()


def _resolve_command_schema(plugin: PluginInfo, command_head: str):
    normalized_head = normalize_message_text(command_head).casefold()
    if not normalized_head:
        return None
    for meta in plugin.command_meta:
        if not _is_public_command_meta(meta):
            continue
        head = normalize_message_text(getattr(meta, "command", "")).casefold()
        if head and head == normalized_head:
            return meta
        for alias in getattr(meta, "aliases", None) or []:
            alias_text = normalize_message_text(alias).casefold()
            if alias_text and alias_text == normalized_head:
                return meta
    return None


def _collect_placeholder_tokens(command: str) -> set[str]:
    tokens: set[str] = set()
    for token in collect_placeholders(command):
        normalized = normalize_message_text(token)
        if not normalized:
            continue
        if _AT_INLINE_PATTERN.fullmatch(normalized):
            normalized = f"[{normalized}]"
        tokens.add(normalized)
    return tokens


def _normalize_placeholder_tokens(
    command: str,
    *,
    include_at: bool = True,
    include_image: bool = True,
) -> tuple[list[str], list[str]]:
    at_tokens: list[str] = []
    image_tokens: list[str] = []
    for token in collect_placeholders(command):
        normalized = normalize_message_text(token)
        if not normalized:
            continue
        if _AT_INLINE_PATTERN.fullmatch(normalized):
            normalized = f"[{normalized}]"
        if include_at and _AT_PLACEHOLDER_PATTERN.fullmatch(normalized):
            if normalized not in at_tokens:
                at_tokens.append(normalized)
            continue
        is_image_token = _IMAGE_PLACEHOLDER_PATTERN.fullmatch(normalized) or (
            normalized.lower().startswith("[image")
        )
        if include_image and is_image_token:
            if normalized not in image_tokens:
                image_tokens.append(normalized)
    return at_tokens, image_tokens


def _is_explicit_helper_command_request(
    message_text: str,
    command_head: str,
) -> bool:
    normalized_message = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    normalized_head = normalize_message_text(command_head)
    if not normalized_message or not normalized_head:
        return False
    if normalized_message == normalized_head:
        return True
    return match_command_head_or_sticky(
        normalized_message,
        normalized_head,
        allow_sticky=True,
    )


def _sanitize_command_with_schema(
    plugin: PluginInfo,
    *,
    command: str,
) -> str:
    normalized = normalize_message_text(command)
    if not normalized:
        return ""
    parts = normalized.split(" ")
    head = normalize_message_text(parts[0] if parts else "")
    if not head:
        return ""

    schema = _resolve_command_schema(plugin, head)
    if schema is None:
        return normalized

    policy = resolve_command_target_policy(schema)
    image_max_raw = getattr(schema, "image_max", None)
    image_max: int | None = None
    if image_max_raw is not None:
        image_max = max(int(image_max_raw), 0)
    accepts_at_target = policy.allow_at and policy.target_requirement != "none"
    include_image = image_max is None or image_max > 0
    placeholder_tokens = _collect_placeholder_tokens(normalized)
    at_tokens, image_tokens = _normalize_placeholder_tokens(
        normalized,
        include_at=accepts_at_target,
        include_image=include_image,
    )

    text_tokens: list[str] = []
    for token in parts[1:]:
        token_text = normalize_message_text(token)
        if not token_text:
            continue
        if token_text in placeholder_tokens:
            continue
        text_tokens.append(token_text)

    if image_max is not None:
        image_tokens = image_tokens[:image_max]

    text_max = getattr(schema, "text_max", None)
    if text_max is not None:
        text_max = max(int(text_max), 0)
        if text_max == 0:
            text_tokens = []
        elif text_max == 1 and len(text_tokens) > 1:
            text_tokens = [" ".join(text_tokens)]
        elif len(text_tokens) > text_max:
            text_tokens = text_tokens[:text_max]

    payload = [*text_tokens, *at_tokens, *image_tokens]
    return normalize_message_text(" ".join([head, *payload]).strip())


def _to_route_result(
    decision: _LLMRouteDecision,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> RouteResolveResult | None:
    if decision.action != "route":
        return None

    plugin = _resolve_target_plugin(decision, knowledge_base)
    if plugin is None:
        return None

    command = normalize_message_text(decision.command or "")
    if not command:
        return None
    command_head = normalize_message_text(command.split(" ", 1)[0]).casefold()
    if not command_head:
        return None

    allowed_heads = _collect_allowed_heads(plugin)
    if command_head not in allowed_heads:
        for candidate_head in sorted(allowed_heads, key=len, reverse=True):
            if not candidate_head:
                continue
            if command.casefold().startswith(candidate_head):
                remainder = normalize_message_text(command[len(candidate_head) :])
                if not remainder:
                    continue
                command = normalize_message_text(f"{candidate_head} {remainder}")
                command_head = candidate_head
                break
    if command_head not in allowed_heads:
        return None

    command = _sanitize_command_with_schema(plugin, command=command)
    command_head = normalize_message_text(command.split(" ", 1)[0]).casefold()
    if not command_head or command_head not in allowed_heads:
        return None

    helper_heads = _collect_helper_heads(plugin.module, knowledge_base)
    if (
        helper_heads
        and command_head in helper_heads
        and not is_usage_question(message_text)
        and not _is_explicit_helper_command_request(message_text, command_head)
    ):
        return None

    route_decision = SkillRouteDecision(
        plugin_name=plugin.name,
        plugin_module=plugin.module,
        command=command,
        source="llm",
        skill_kind="llm",
    )
    return RouteResolveResult(decision=route_decision, stage="llm")


def _validate_existing_route_result(
    route_result: RouteResolveResult,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> RouteResolveResult | None:
    validated = _to_route_result(
        _LLMRouteDecision(
            action="route",
            plugin_module=route_result.decision.plugin_module,
            plugin_name=route_result.decision.plugin_name,
            command=route_result.decision.command,
        ),
        message_text,
        knowledge_base,
    )
    if validated is None:
        return None
    return RouteResolveResult(
        decision=SkillRouteDecision(
            plugin_name=validated.decision.plugin_name,
            plugin_module=validated.decision.plugin_module,
            command=validated.decision.command,
            source=route_result.decision.source,
            skill_kind=route_result.decision.skill_kind,
        ),
        stage=route_result.stage,
    )


def _extract_alignment_numbers(message_text: str) -> list[str]:
    normalized = normalize_message_text(message_text)
    if not normalized:
        return []
    values: list[str] = []
    for match in re.findall(r"\d+(?:\.\d+)?", normalized):
        if match not in values:
            values.append(match)
    return values[:6]


def _build_shortlist_alignment_prompt(
    *,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    local_hint: RouteResolveResult | None,
    entity_hint: RouteResolveResult | None,
) -> str:
    skills_json = render_skill_namespace(
        knowledge_base,
        query=message_text,
        limit=max(len(knowledge_base.plugins), 1),
        preferred_modules=[
            item
            for item in (
                local_hint.decision.plugin_module if local_hint is not None else "",
                entity_hint.decision.plugin_module if entity_hint is not None else "",
            )
            if item
        ],
        include_helpers=True,
        mask_module=False,
    )
    signal_payload = {
        "entity_tokens": list(_extract_shortlist_entity_tokens(message_text)),
        "numbers": _extract_alignment_numbers(message_text),
        "has_at": bool(_AT_PLACEHOLDER_PATTERN.search(message_text) or _AT_INLINE_PATTERN.search(message_text)),
        "has_image": bool(_IMAGE_PLACEHOLDER_PATTERN.search(message_text)),
        "is_usage_question": is_usage_question(message_text),
        "local_hint": {
            "plugin_module": local_hint.decision.plugin_module,
            "command": local_hint.decision.command,
            "source": local_hint.decision.source,
        }
        if local_hint is not None
        else None,
        "entity_hint": {
            "plugin_module": entity_hint.decision.plugin_module,
            "command": entity_hint.decision.command,
            "source": entity_hint.decision.source,
        }
        if entity_hint is not None
        else None,
    }
    return (
        f"用户消息:\n{message_text}\n\n"
        f"消息信号 JSON:\n{json.dumps(signal_payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "候选插件卡片 JSON:\n"
        f"{skills_json}\n\n"
        "只基于上述 shortlist 做对齐。"
    )


def _subset_knowledge_base_by_modules(
    knowledge_base: PluginKnowledgeBase,
    modules: list[str],
) -> PluginKnowledgeBase:
    if not modules:
        return knowledge_base
    normalized_modules = {
        normalize_message_text(module)
        for module in modules
        if normalize_message_text(module)
    }
    if not normalized_modules:
        return knowledge_base
    plugins = [
        plugin
        for plugin in knowledge_base.plugins
        if normalize_message_text(plugin.module) in normalized_modules
    ]
    if not plugins:
        return knowledge_base
    return PluginKnowledgeBase(plugins=plugins, user_role=knowledge_base.user_role)


def _extract_shortlist_entity_tokens(message_text: str) -> tuple[str, ...]:
    normalized = normalize_message_text(strip_invoke_prefix(message_text or "")).lower()
    if not normalized:
        return ()

    tokens: list[str] = []
    for token in _SHORTLIST_ENTITY_TOKEN_PATTERN.findall(normalized):
        value = token.strip()
        if not value or value in _SHORTLIST_ENTITY_STOPWORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", value):
            for size in range(min(3, len(value)), 1, -1):
                for start in range(0, len(value) - size + 1):
                    fragment = value[start : start + size]
                    if fragment in _SHORTLIST_ENTITY_STOPWORDS:
                        continue
                    if fragment not in tokens:
                        tokens.append(fragment)
            continue
        if len(value) >= 3 and value not in tokens:
            tokens.append(value)
    return tuple(tokens[:24])


def _looks_like_compact_route_query(message_text: str) -> bool:
    normalized = normalize_message_text(strip_invoke_prefix(message_text or "")).lower()
    if not normalized or is_usage_question(normalized):
        return False
    if any(marker in normalized for marker in _SHORTLIST_COMPACT_QUESTION_HINTS):
        return False
    compact = normalized.replace(" ", "")
    if len(compact) > 10:
        return False
    return contains_any(normalized, ROUTE_ACTION_WORDS) or any(
        hint in normalized for hint in _SHORTLIST_COMPACT_ROUTE_HINTS
    )


def _skill_matches_shortlist_entities(skill: Any, entity_tokens: tuple[str, ...]) -> bool:
    if not entity_tokens:
        return False
    haystack = normalize_message_text(
        " ".join(
            [
                str(getattr(skill, "plugin_name", "") or ""),
                str(getattr(skill, "plugin_module", "") or ""),
                " ".join(getattr(skill, "commands", ()) or ()),
                " ".join(getattr(skill, "aliases", ()) or ()),
                " ".join(getattr(skill, "examples", ()) or ()),
                str(getattr(skill, "usage", "") or ""),
            ]
        )
    ).lower()
    if not haystack:
        return False
    return any(token in haystack for token in entity_tokens)


def _build_shortlist_knowledge_base(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    limit: int,
) -> tuple[PluginKnowledgeBase, list[str]]:
    shortlist_limit = max(int(limit), 1)
    registry = get_skill_registry(knowledge_base)
    selected_skills = list(
        select_relevant_skills(
            registry,
            message_text,
            limit=shortlist_limit,
        )
    )
    query_families = infer_query_families(message_text)
    entity_tokens = _extract_shortlist_entity_tokens(message_text)
    enable_entity_enrichment = any(family != "general" for family in query_families) or (
        entity_tokens and _looks_like_compact_route_query(message_text)
    )
    if enable_entity_enrichment and entity_tokens:
        entity_skills = [
            skill
            for skill in registry.skills
            if _skill_matches_shortlist_entities(skill, entity_tokens)
        ]
        merged: list[Any] = []
        seen_modules: set[str] = set()
        for skill in [*entity_skills, *selected_skills]:
            module = normalize_message_text(getattr(skill, "plugin_module", ""))
            if not module or module in seen_modules:
                continue
            seen_modules.add(module)
            merged.append(skill)
            if len(merged) >= shortlist_limit:
                break
        selected_skills = merged

    selected_modules = [
        normalize_message_text(skill.plugin_module)
        for skill in selected_skills
        if normalize_message_text(skill.plugin_module)
    ]
    subset = _subset_knowledge_base_by_modules(knowledge_base, selected_modules)
    if not selected_modules:
        selected_modules = [
            normalize_message_text(plugin.module)
            for plugin in subset.plugins
            if normalize_message_text(plugin.module)
        ]
    return subset, selected_modules


def _resolve_shortlist_local_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> RouteResolveResult | None:
    search_result = skill_search(
        message_text,
        knowledge_base,
        include_usage=True,
        include_similarity=True,
    )
    decision = skill_execute(search_result, message_text, knowledge_base)
    if decision is None:
        return None
    return RouteResolveResult(decision=decision, stage="shortlist")


def _resolve_shortlist_entity_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> RouteResolveResult | None:
    entity_tokens = _extract_shortlist_entity_tokens(message_text)
    if not entity_tokens:
        return None

    query_family = infer_query_family(message_text)
    normalized = normalize_message_text(message_text).lower()
    registry = get_skill_registry(knowledge_base)

    best_skill: Any | None = None
    best_score = 0.0
    for skill in registry.skills:
        haystack = normalize_message_text(
            " ".join(
                [
                    skill.plugin_name,
                    skill.plugin_module,
                    " ".join(skill.commands),
                    " ".join(skill.aliases),
                    " ".join(skill.examples),
                    skill.usage or "",
                ]
            )
        ).lower()
        if not haystack:
            continue
        matched = [token for token in entity_tokens if token in haystack]
        if not matched:
            continue
        score = max(len(token) for token in matched) * 10.0 + len(matched)
        if query_family == "search":
            score += 4.0
        if score > best_score:
            best_score = score
            best_skill = skill

    if best_skill is None:
        return None

    command_pool: list[str] = []
    for command in [*best_skill.commands, *best_skill.aliases]:
        normalized_command = normalize_message_text(command)
        if normalized_command and normalized_command not in command_pool:
            command_pool.append(normalized_command)

    if not command_pool:
        return None

    chosen_command: str | None = None
    if query_family == "search":
        if "抽" in normalized:
            for candidate in command_pool:
                if any(token in candidate for token in ("今天", "今日", "本日", "当日")):
                    chosen_command = candidate
                    break
        if chosen_command is None and any(token in normalized for token in ("找", "搜", "查", "查询")):
            for candidate in command_pool:
                if any(token in candidate for token in ("找", "搜", "查")):
                    chosen_command = candidate
                    break

    if chosen_command is None:
        for candidate in command_pool:
            if any(token in candidate for token in entity_tokens):
                chosen_command = candidate
                break

    if chosen_command is None:
        chosen_command = command_pool[0]

    return RouteResolveResult(
        decision=SkillRouteDecision(
            plugin_name=best_skill.plugin_name,
            plugin_module=best_skill.plugin_module,
            command=chosen_command,
            source="shortlist_entity",
            skill_kind=best_skill.kind,
        ),
        stage="shortlist",
    )


def _pick_shortlist_fallback_route(
    message_text: str,
    local_hint: RouteResolveResult | None,
    entity_hint: RouteResolveResult | None,
) -> RouteResolveResult | None:
    if local_hint is None:
        return entity_hint
    if entity_hint is None:
        return local_hint

    message_role = infer_message_action_role(message_text)
    local_role = infer_route_action_role(local_hint.decision.command)
    entity_role = infer_route_action_role(entity_hint.decision.command)

    if message_role in {"create", "open", "return", "query"}:
        if local_role == "other" and entity_role != "other":
            return entity_hint
        if entity_role == message_role and local_role != message_role:
            return entity_hint
        if local_role == message_role and entity_role != message_role:
            return local_hint

    local_is_strong = local_hint.decision.source in {"fast", "rank_exact"}
    entity_is_strong = entity_hint.decision.source in {"fast", "rank_exact"}
    if local_is_strong and not entity_is_strong:
        return local_hint
    if entity_is_strong and not local_is_strong:
        return entity_hint

    local_has_payload = " " in normalize_message_text(local_hint.decision.command)
    entity_has_payload = " " in normalize_message_text(entity_hint.decision.command)
    if entity_has_payload and not local_has_payload:
        return entity_hint
    if local_has_payload and not entity_has_payload:
        return local_hint

    return local_hint


def probe_shortlist_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    shortlist_limit: int | None = None,
) -> tuple[RouteResolveResult | None, RouteAttemptReport]:
    normalized_message = normalize_message_text(message_text)
    report = RouteAttemptReport(helper_mode=is_usage_question(normalized_message))
    if not normalized_message or not knowledge_base.plugins:
        report.finalize(reason="empty_input")
        return None, report

    full_plugins = list(knowledge_base.plugins)
    report.candidate_total = len(full_plugins)
    effective_limit = max(
        int(
            shortlist_limit
            or get_config_value(
                "ROUTE_PLUGIN_SHORTLIST_LIMIT",
                _ROUTE_PLUGIN_SHORTLIST_LIMIT,
            )
            or _ROUTE_PLUGIN_SHORTLIST_LIMIT
        ),
        1,
    )
    attempt_kb, shortlisted_modules = _build_shortlist_knowledge_base(
        normalized_message,
        knowledge_base,
        limit=effective_limit,
    )
    report.lexical_candidates = len(attempt_kb.plugins)
    report.direct_candidates = len(attempt_kb.plugins)
    report.vector_candidates = 0

    attempt_modules = shortlisted_modules[:_ROUTE_TRACE_SAMPLE_LIMIT]
    if not attempt_modules:
        attempt_modules = [
            normalize_message_text(plugin.module)
            for plugin in attempt_kb.plugins[:_ROUTE_TRACE_SAMPLE_LIMIT]
        ]
    report.note_attempt(attempt_modules)

    entity_route_result = _resolve_shortlist_entity_route(
        normalized_message,
        attempt_kb,
    )
    shortlist_route_result = _resolve_shortlist_local_route(
        normalized_message,
        attempt_kb,
    )
    if entity_route_result is not None:
        if shortlist_route_result is None or shortlist_route_result.decision.source not in {
            "fast",
            "rank_exact",
        }:
            shortlist_route_result = entity_route_result
    if shortlist_route_result is None:
        report.finalize(reason="shortlist_probe_miss")
        return None, report

    validated_shortlist_result = _validate_existing_route_result(
        shortlist_route_result,
        normalized_message,
        attempt_kb,
    )
    if validated_shortlist_result is None:
        report.finalize(reason="shortlist_probe_invalid")
        return None, report

    report.finalize(
        reason="shortlist_route",
        stage=validated_shortlist_result.stage,
        plugin_name=validated_shortlist_result.decision.plugin_name,
        plugin_module=validated_shortlist_result.decision.plugin_module,
        command=validated_shortlist_result.decision.command,
    )
    return (
        RouteResolveResult(
            decision=validated_shortlist_result.decision,
            stage=validated_shortlist_result.stage,
            report=report,
        ),
        report,
    )


def _is_schema_parameter_error(exc: Exception) -> bool:
    text = normalize_message_text(str(exc or "")).lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "invalid_parameter",
            "invalid_argument",
            "response_schema",
            "responsejsonschema",
            "additionalproperties",
        )
    )


def _normalize_tool_call_arguments(raw_arguments: str) -> dict[str, Any]:
    raw_text = str(raw_arguments or "").strip()
    if not raw_text:
        return {}
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _tool_call_to_route_result(
    tool_call: LLMToolCall,
    spec_map: dict[str, RoutePlannerToolSpec],
) -> RouteResolveResult | None:
    spec = spec_map.get(str(tool_call.function.name or "").strip())
    if spec is None:
        return None

    args = _normalize_tool_call_arguments(tool_call.function.arguments)
    text_value = normalize_message_text(str(args.get("text", "") or ""))
    command = spec.command_head
    if text_value and spec.text_allowed:
        command = normalize_message_text(f"{command} {text_value}")

    decision = SkillRouteDecision(
        plugin_name=spec.plugin_name,
        plugin_module=spec.plugin_module,
        command=command,
        source="tool",
        skill_kind="tool",
    )
    return RouteResolveResult(decision=decision, stage="tool")


async def _request_tool_route_decision(
    *,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    timeout_value: float | None,
    session_key: str,
    budget_controller: TurnBudgetController | None,
) -> tuple[RouteResolveResult | None, int, int]:
    if not ROUTE_TOOL_PLANNER_ENABLED:
        return None, 0, 0

    max_tools = max(int(ROUTE_TOOL_PLANNER_MAX_TOOLS), 1)
    planner_tools, spec_map = build_route_planner_tools(
        message_text=message_text,
        knowledge_base=knowledge_base,
        max_tools=max_tools,
        tools_per_skill=_ROUTE_TOOLS_PER_SKILL_LIMIT,
        query_family="general",
        candidate_modules=tuple(
            normalize_message_text(plugin.module)
            for plugin in knowledge_base.plugins
            if normalize_message_text(plugin.module)
        ),
    )
    if not planner_tools or not spec_map:
        return None, len(planner_tools), 0

    trace_specs = [
        f"{spec.plugin_name}:{spec.command_head}"
        for spec in list(spec_map.values())[:_ROUTE_TRACE_SAMPLE_LIMIT]
    ]
    logger.debug(
        "ChatInter tool 路由尝试: "
        f"tools={len(planner_tools)} "
        f"choices={trace_specs}"
    )

    try:
        if budget_controller is not None and not budget_controller.allow_classifier(
            "route_tool"
        ):
            return None, len(planner_tools), 0
        started = time.perf_counter()
        guarded = guard_prompt_sections(
            session_key=session_key,
            stage="route_tool",
            system_prompt=_TOOL_ROUTE_INSTRUCTION,
            user_text=message_text,
            controller=budget_controller,
        )
        response = await chat(
            guarded.user_text,
            model=get_model_name(),
            instruction=guarded.system_prompt,
            tools=cast(list[Any], planner_tools),
            tool_choice="auto",
            timeout=timeout_value,
        )
        if budget_controller is not None:
            budget_controller.record_classifier(
                "route_tool",
                time.perf_counter() - started,
            )
    except Exception as exc:
        logger.debug(f"ChatInter tool 路由失败，回退 JSON 路由: {exc}")
        return None, len(planner_tools), 0

    raw_tool_calls = response.tool_calls or []
    if not raw_tool_calls:
        logger.debug("ChatInter tool 路由未选择任何工具")
        return None, len(planner_tools), 0

    normalized_calls: list[LLMToolCall] = []
    for item in raw_tool_calls:
        if isinstance(item, LLMToolCall):
            normalized_calls.append(item)
            continue
        if isinstance(item, dict):
            try:
                normalized_calls.append(LLMToolCall(**item))
            except Exception:
                continue
    if not normalized_calls:
        logger.debug("ChatInter tool 路由返回的 tool_calls 无法解析")
        return None, len(planner_tools), 0

    route_result = _tool_call_to_route_result(normalized_calls[0], spec_map)
    if route_result is None:
        logger.debug(
            "ChatInter tool 路由结果未映射到候选工具: "
            f"name={normalized_calls[0].function.name}"
        )
        return None, len(planner_tools), len(normalized_calls)
    logger.debug(
        "ChatInter tool 路由命中: "
        f"module={route_result.decision.plugin_module} "
        f"command={route_result.decision.command}"
    )
    return route_result, len(planner_tools), len(normalized_calls)


async def _request_shortlist_alignment_decision(
    *,
    prompt: str,
    timeout_value: float | None,
    session_key: str,
    budget_controller: TurnBudgetController | None,
) -> _ShortlistAlignmentDecision | None:
    try:
        if budget_controller is not None and not budget_controller.allow_classifier(
            "shortlist_align"
        ):
            return None
        started = time.perf_counter()
        guarded = guard_prompt_sections(
            session_key=session_key,
            stage="shortlist_align",
            system_prompt=_SHORTLIST_ALIGNMENT_INSTRUCTION,
            user_text=prompt,
            controller=budget_controller,
        )
        decision = await generate_structured(
            guarded.user_text,
            _ShortlistAlignmentDecision,
            model=get_model_name(),
            instruction=guarded.system_prompt,
            timeout=timeout_value,
        )
        if budget_controller is not None:
            budget_controller.record_classifier(
                "shortlist_align",
                time.perf_counter() - started,
            )
        return decision
    except Exception as exc:
        logger.debug(f"ChatInter shortlist 对齐失败: {exc}")
        return None


async def resolve_shortlist_alignment(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None = None,
    budget_controller: TurnBudgetController | None = None,
) -> tuple[_ShortlistAlignmentDecision | None, RouteResolveResult | None, RouteAttemptReport]:
    normalized_message = normalize_message_text(message_text)
    report = RouteAttemptReport(helper_mode=is_usage_question(normalized_message))
    if not normalized_message or not knowledge_base.plugins:
        report.finalize(reason="shortlist_alignment_empty")
        return None, None, report

    timeout = get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout_value = float(timeout) if timeout else None
    except (TypeError, ValueError):
        timeout_value = None

    shortlist_limit = max(
        int(
            get_config_value(
                "ROUTE_PLUGIN_SHORTLIST_LIMIT",
                _ROUTE_PLUGIN_SHORTLIST_LIMIT,
            )
            or _ROUTE_PLUGIN_SHORTLIST_LIMIT
        ),
        1,
    )
    attempt_kb, shortlisted_modules = _build_shortlist_knowledge_base(
        normalized_message,
        knowledge_base,
        limit=shortlist_limit,
    )
    report.candidate_total = len(knowledge_base.plugins)
    report.lexical_candidates = len(attempt_kb.plugins)
    report.direct_candidates = len(attempt_kb.plugins)
    report.vector_candidates = 0
    attempt_modules = shortlisted_modules[:_ROUTE_TRACE_SAMPLE_LIMIT]
    if not attempt_modules:
        attempt_modules = [
            normalize_message_text(plugin.module)
            for plugin in attempt_kb.plugins[:_ROUTE_TRACE_SAMPLE_LIMIT]
        ]
    report.note_attempt(attempt_modules)

    local_hint = _resolve_shortlist_local_route(normalized_message, attempt_kb)
    entity_hint = _resolve_shortlist_entity_route(normalized_message, attempt_kb)
    prompt = _build_shortlist_alignment_prompt(
        message_text=normalized_message,
        knowledge_base=attempt_kb,
        local_hint=local_hint,
        entity_hint=entity_hint,
    )
    decision = await _request_shortlist_alignment_decision(
        prompt=prompt,
        timeout_value=timeout_value,
        session_key=session_key or "global",
        budget_controller=budget_controller,
    )
    if decision is None:
        fallback = _pick_shortlist_fallback_route(
            normalized_message,
            local_hint,
            entity_hint,
        )
        if fallback is not None:
            if not is_route_action_compatible(
                normalized_message,
                fallback.decision.command,
            ):
                fallback = None
        if fallback is not None:
            validated_fallback = _validate_existing_route_result(
                fallback,
                normalized_message,
                attempt_kb,
            )
            if validated_fallback is not None:
                report.finalize(
                    reason="shortlist_alignment_fallback",
                    stage=validated_fallback.stage,
                    plugin_name=validated_fallback.decision.plugin_name,
                    plugin_module=validated_fallback.decision.plugin_module,
                    command=validated_fallback.decision.command,
                )
                return None, validated_fallback, report
        report.finalize(reason="shortlist_alignment_miss")
        return None, None, report

    if decision.action == "skip":
        report.finalize(reason="shortlist_alignment_skip")
        return decision, None, report

    route_result = _to_route_result(
        _LLMRouteDecision(
            action="route",
            plugin_module=decision.plugin_module,
            plugin_name=decision.plugin_name,
            command=decision.command,
        ),
        normalized_message,
        attempt_kb,
    )
    if route_result is None:
        report.finalize(reason="shortlist_alignment_invalid")
        return decision, None, report
    if not is_route_action_compatible(
        normalized_message,
        route_result.decision.command,
    ):
        fallback = _pick_shortlist_fallback_route(
            normalized_message,
            local_hint,
            entity_hint,
        )
        if fallback is not None and is_route_action_compatible(
            normalized_message,
            fallback.decision.command,
        ):
            validated_fallback = _validate_existing_route_result(
                fallback,
                normalized_message,
                attempt_kb,
            )
            if validated_fallback is not None:
                report.finalize(
                    reason="shortlist_alignment_fallback",
                    stage=validated_fallback.stage,
                    plugin_name=validated_fallback.decision.plugin_name,
                    plugin_module=validated_fallback.decision.plugin_module,
                    command=validated_fallback.decision.command,
                )
                return None, validated_fallback, report
        report.finalize(reason="shortlist_alignment_incompatible")
        return decision, None, report

    stage_name = "shortlist_align_usage" if decision.action == "usage" else "shortlist_align"
    report.finalize(
        reason=f"shortlist_alignment_{decision.action}",
        stage=stage_name,
        plugin_name=route_result.decision.plugin_name,
        plugin_module=route_result.decision.plugin_module,
        command=route_result.decision.command,
    )
    return (
        decision,
        RouteResolveResult(
            decision=SkillRouteDecision(
                plugin_name=route_result.decision.plugin_name,
                plugin_module=route_result.decision.plugin_module,
                command=route_result.decision.command,
                source="shortlist_align",
                skill_kind="shortlist_align",
            ),
            stage=stage_name,
        ),
        report,
    )


__all__ = [
    "RouteAttemptReport",
    "RouteResolveResult",
    "probe_shortlist_route",
    "resolve_shortlist_alignment",
]
