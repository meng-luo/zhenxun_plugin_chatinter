from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from zhenxun.services import generate_structured, logger

from .capability_graph import capability_from_plugin
from .command_index import (
    CommandCandidate,
    build_command_candidates,
    dump_schema_for_prompt,
)
from .command_schema import (
    build_command_schemas,
    complete_slots,
    render_command,
    select_command_schema,
)
from .config import (
    get_config_value,
    get_model_name,
)
from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .plugin_registry import PluginRegistry
from .prompt_guard import guard_prompt_sections
from .route_policy import (
    infer_message_action_role,
    infer_route_action_role,
    is_route_action_compatible,
)
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_placeholders,
    collect_weak_route_signals,
    contains_any,
    is_usage_question,
    match_command_head_or_sticky,
    normalize_action_phrases,
    normalize_message_text,
    parse_command_with_head,
    strip_invoke_prefix,
)
from .route_tool_planner import build_command_choice_tools
from .schema_policy import resolve_command_target_policy
from .skill_registry import (
    SkillRouteDecision,
    _extract_argument_around_head,
    _message_has_payload_signals,
    get_skill_registry,
    infer_query_family,
    render_skill_namespace,
    select_relevant_skills,
    skill_execute,
    skill_search,
)
from .speech_act import classify_speech_act
from .turn_runtime import TurnBudgetController

_ROUTE_NAMESPACE_LIMIT = 12
_ROUTE_PLUGIN_SHORTLIST_LIMIT = 30
_WEAK_SIGNAL_SHORTLIST_LIMIT = 5
_ROUTER_INSTRUCTION = """
判断用户消息是普通对话、插件执行、插件用法查询、还是需要澄清。
优先选择 plugin_cards.command_schemas 中存在的 command_id，并填写 slots。
不要自由发明命令；需要执行时给出 command_id 或合法命令头。
列表/有哪些/搜索类 helper 命令属于 execute，只有“怎么用/用法/教程/示例”才 usage。
图片模板命令只需要图片时不要把自然语言描述当作 text 参数。
只有必填槽位缺失时才 clarify；无必填槽位的命令可以直接 execute。
输出 JSON，不要输出额外文本。
""".strip()
_SHORTLIST_ALIGNMENT_INSTRUCTION = _ROUTER_INSTRUCTION
_WEAK_SIGNAL_ALIGNMENT_INSTRUCTION = _ROUTER_INSTRUCTION
_TOOL_ROUTE_INSTRUCTION = """
你是 ChatInter 的工具式路由规划器。
给你的每个工具都已经绑定到一个明确的插件命令。
严格遵守：
1. 明显对应当前诉求的工具才调用，且只调用一个。
2. text 只填额外纯文本，不要重复命令头、[@...]、[image]、reply。
3. 命令不需要文本参数时不要传 text。
4. 像普通对话、闲聊、澄清或把握不足时，直接回复 SKIP。
5. "怎么用/用法/帮助/参数"类问题优先帮助型命令，不要误调用业务动作命令。
""".strip()
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_AT_PLACEHOLDER_PATTERN = re.compile(r"\[@[^\]\s]+\]")
_AT_INLINE_PATTERN = re.compile(r"@\d{5,20}")
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_ROUTE_TRACE_SAMPLE_LIMIT = 12
_SHORTLIST_ENTITY_TOKEN_PATTERN = re.compile(
    r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE
)
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
    command_id: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()


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


class _WeakSignalDecision(BaseModel):
    action: Literal["chat", "usage", "execute", "ambiguous"] = Field(
        default="chat",
        description="chat=普通聊天；usage=插件用法；execute=直接执行；ambiguous=介于两者之间",
    )
    plugin_module: str | None = Field(
        default=None,
        description="目标插件模块，必须来自 cards_json",
    )
    plugin_name: str | None = Field(default=None, description="目标插件名称，可选")
    command: str | None = Field(
        default=None,
        description="命令头，不要带参数、[@...]、[image]",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = Field(default=None, description="简短原因")
    missing_context: list[str] = Field(
        default_factory=list,
        description="缺失的上下文，例如 at/image/reply/target/参数",
    )


class LLMRouterDecision(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(
        default="chat",
        description="chat=普通对话；execute=执行插件；usage=查询用法；clarify=需要补充信息",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    plugin_module: str | None = Field(default=None, description="必须来自 plugin_cards")
    plugin_name: str | None = Field(default=None, description="插件名称，可选")
    command_id: str | None = Field(default=None, description="优先使用的命令 schema ID")
    command: str | None = Field(default=None, description="插件命令头")
    slots: dict[str, Any] = Field(default_factory=dict, description="命令槽位")
    arguments_text: str = Field(default="", description="命令后的自然语言参数")
    missing: list[str] = Field(default_factory=list, description="缺失信息")
    reason: str | None = Field(default=None, description="简短原因")


class LLMCommandSelection(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(
        default="chat",
        description="chat=普通对话；execute=执行候选命令；usage=查询候选命令用法；clarify=需要补充信息",
    )
    command_id: str | None = Field(
        default=None,
        description="必须来自 candidates.command_id；chat 时为空",
    )
    slots: dict[str, Any] = Field(
        default_factory=dict,
        description="按候选 schema 填写的槽位，不要臆造 schema 之外的槽位",
    )
    missing: list[str] = Field(
        default_factory=list,
        description="缺失的必填槽位或上下文，例如 text/image/reply/target",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="简短选择理由")


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
        normalize_message_text(
            str(getattr(meta, "access_level", "public") or "public")
        ).lower()
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
    capability = capability_from_plugin(plugin)
    if capability is not None:
        for schema in build_command_schemas(plugin.module, capability.commands):
            head = normalize_message_text(schema.head)
            if head:
                allowed.add(head.casefold())
            for alias in schema.aliases:
                alias_text = normalize_message_text(alias)
                if alias_text:
                    allowed.add(alias_text.casefold())
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


def _strip_supported_command_prefixes(command: str, plugin: PluginInfo) -> str:
    normalized = normalize_message_text(command)
    if not normalized:
        return ""

    allowed_heads = _collect_allowed_heads(plugin)
    normalized_fold = normalized.casefold()
    command_head = normalize_message_text(normalized.split(" ", 1)[0]).casefold()
    if command_head in allowed_heads:
        return normalized

    prefixes: list[str] = []
    for meta in plugin.command_meta:
        if not _is_public_command_meta(meta):
            continue
        for prefix in getattr(meta, "prefixes", None) or []:
            prefix_text = normalize_message_text(str(prefix or ""))
            if prefix_text and prefix_text not in prefixes:
                prefixes.append(prefix_text)

    for prefix in sorted(prefixes, key=len, reverse=True):
        if normalized.startswith(prefix):
            stripped = normalize_message_text(normalized[len(prefix) :])
            stripped_head = normalize_message_text(
                stripped.split(" ", 1)[0] if stripped else ""
            ).casefold()
            if stripped_head in allowed_heads:
                return stripped
            for candidate_head in sorted(allowed_heads, key=len, reverse=True):
                if candidate_head and stripped.casefold().startswith(candidate_head):
                    return stripped

    if normalized_fold.startswith("/") and len(normalized) > 1:
        stripped = normalize_message_text(normalized[1:])
        stripped_head = normalize_message_text(
            stripped.split(" ", 1)[0] if stripped else ""
        ).casefold()
        if stripped_head in allowed_heads:
            return stripped
    return normalized


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


def _schema_accepts_text_payload(schema: Any | None) -> bool:
    if schema is None:
        return False
    text_min = max(int(getattr(schema, "text_min", 0) or 0), 0)
    if text_min > 0:
        return True
    text_max = getattr(schema, "text_max", None)
    if text_max is not None:
        try:
            return int(text_max) > 0
        except Exception:
            return False
    return bool(getattr(schema, "params", None))


def _rehydrate_command_payload_from_message(
    command: str,
    *,
    message_text: str,
    schema,
) -> str:
    normalized = normalize_message_text(command)
    if not normalized:
        return normalized
    if len(normalized.split(" ", 1)) > 1:
        return normalized

    has_payload_signal = _message_has_payload_signals(message_text)
    accepts_text_payload = _schema_accepts_text_payload(schema)
    if schema is None and not has_payload_signal:
        return normalized

    if not accepts_text_payload and schema is not None:
        return normalized
    if not accepts_text_payload and not has_payload_signal:
        return normalized

    command_head = normalize_message_text(normalized.split(" ", 1)[0])
    if not command_head:
        return normalized

    allow_sticky = (
        bool(getattr(schema, "allow_sticky_arg", False))
        if schema is not None
        else False
    )
    recovered = parse_command_with_head(
        message_text,
        command_head,
        allow_sticky=allow_sticky,
        max_prefix_len=16,
    )
    payload = normalize_message_text(
        (recovered.payload_text if recovered else "") or ""
    )
    if not payload:
        payload = normalize_message_text(
            _extract_argument_around_head(message_text, command_head)
        )
    if not payload:
        return normalized

    payload_tokens = [token for token in payload.split(" ") if token]
    if not payload_tokens:
        return normalized

    text_max = getattr(schema, "text_max", None) if schema is not None else None
    if text_max is not None:
        try:
            text_max_int = max(int(text_max), 0)
        except Exception:
            text_max_int = None
        if text_max_int is not None:
            if text_max_int == 0:
                return command_head
            if text_max_int == 1 and len(payload_tokens) > 1:
                payload_tokens = [" ".join(payload_tokens)]
            elif len(payload_tokens) > text_max_int:
                payload_tokens = payload_tokens[:text_max_int]

    return normalize_message_text(" ".join([command_head, *payload_tokens]).strip())


def _to_route_result(
    decision: _LLMRouteDecision,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    command_id: str | None = None,
    slots: dict[str, Any] | None = None,
    missing: list[str] | tuple[str, ...] = (),
) -> RouteResolveResult | None:
    if decision.action != "route":
        return None

    plugin = _resolve_target_plugin(decision, knowledge_base)
    if plugin is None:
        return None

    command = normalize_message_text(decision.command or "")
    if not command:
        return None
    command = _strip_supported_command_prefixes(command, plugin)
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
    schema = _resolve_command_schema(plugin, command_head)
    if schema is not None:
        command = _rehydrate_command_payload_from_message(
            command,
            message_text=message_text,
            schema=schema,
        )
    elif _message_has_payload_signals(message_text):
        command = _rehydrate_command_payload_from_message(
            command,
            message_text=message_text,
            schema=None,
        )
    command_head = normalize_message_text(command.split(" ", 1)[0]).casefold()
    if not command_head or command_head not in allowed_heads:
        return None

    helper_heads = _collect_helper_heads(plugin.module, knowledge_base)
    if (
        helper_heads
        and command_head in helper_heads
        and not command_id
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
    return RouteResolveResult(
        decision=route_decision,
        stage="llm",
        command_id=normalize_message_text(command_id or "") or None,
        slots=dict(slots or {}),
        missing=tuple(item for item in missing if item),
    )


def _to_plugin_route_result(
    *,
    plugin: PluginInfo,
    stage: str,
    source: str,
    skill_kind: str,
    command: str = "",
    command_id: str | None = None,
    slots: dict[str, Any] | None = None,
    missing: list[str] | tuple[str, ...] = (),
) -> RouteResolveResult:
    return RouteResolveResult(
        decision=SkillRouteDecision(
            plugin_name=plugin.name,
            plugin_module=plugin.module,
            command=normalize_message_text(command),
            source=source,
            skill_kind=skill_kind,
        ),
        stage=stage,
        command_id=normalize_message_text(command_id or "") or None,
        slots=dict(slots or {}),
        missing=tuple(item for item in missing if item),
    )


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
        command_id=route_result.command_id,
        slots=route_result.slots,
        missing=route_result.missing,
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
        command_id=validated.command_id or route_result.command_id,
        slots=dict(validated.slots or route_result.slots),
        missing=validated.missing or route_result.missing,
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
    has_reply: bool,
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
        "has_at": bool(
            _AT_PLACEHOLDER_PATTERN.search(message_text)
            or _AT_INLINE_PATTERN.search(message_text)
        ),
        "has_image": bool(_IMAGE_PLACEHOLDER_PATTERN.search(message_text)),
        "has_reply": has_reply,
        "is_usage_question": is_usage_question(message_text),
        "message_role": infer_message_action_role(message_text),
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
        "消息信号 JSON:\n"
        f"{json.dumps(signal_payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "候选插件卡片 JSON:\n"
        f"{skills_json}\n\n"
        "只基于上述 shortlist 做对齐。"
    )


def _build_weak_signal_alignment_prompt(
    *,
    message_text: str,
    original_message_text: str | None,
    knowledge_base: PluginKnowledgeBase,
    has_at: bool,
    has_image: bool,
    has_reply: bool,
    is_private: bool,
) -> str:
    normalized_message = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    weak_tags = list(collect_weak_route_signals(normalized_message))
    shortlist_limit = max(_WEAK_SIGNAL_SHORTLIST_LIMIT, 1)
    registry = get_skill_registry(knowledge_base)
    selected_skills = list(
        select_relevant_skills(
            registry,
            normalized_message,
            limit=shortlist_limit,
        )
    )
    preferred_modules = [
        normalize_message_text(skill.plugin_module)
        for skill in selected_skills
        if normalize_message_text(skill.plugin_module)
    ]
    cards_json = render_skill_namespace(
        knowledge_base,
        query=normalized_message,
        limit=shortlist_limit,
        preferred_modules=preferred_modules,
        include_helpers=True,
        mask_module=False,
    )
    signal_payload = {
        "original_text": original_message_text or message_text,
        "normalized_text": normalized_message,
        "weak_tags": weak_tags,
        "context": {
            "has_at": has_at,
            "has_image": has_image,
            "has_reply": has_reply,
            "is_private": is_private,
            "message_role": infer_message_action_role(normalized_message),
        },
        "shortlist_modules": preferred_modules,
    }
    return (
        f"用户原文:\n{original_message_text or message_text}\n\n"
        f"归一化文本:\n{normalized_message}\n\n"
        "弱词标签 JSON:\n"
        f"{json.dumps(signal_payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "候选插件卡片 JSON:\n"
        f"{cards_json}\n\n"
        "只基于上述 cards_json 判定 chat / usage / execute / ambiguous。"
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


def _has_strong_shortlist_entity_signal(entity_tokens: tuple[str, ...]) -> bool:
    if not entity_tokens:
        return False
    if len(entity_tokens) >= 2:
        return True
    token = next(iter(entity_tokens), "").strip()
    if not token:
        return False
    if any(char.isascii() for char in token):
        return len(token) >= 5
    return len(token) >= 4


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


def _skill_matches_shortlist_entities(
    skill: Any, entity_tokens: tuple[str, ...]
) -> bool:
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
    entity_tokens = _extract_shortlist_entity_tokens(message_text)
    enable_entity_enrichment = _has_strong_shortlist_entity_signal(entity_tokens)
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


def _build_router_prompt(
    *,
    message_text: str,
    cards: list[dict[str, object]],
    has_reply: bool,
) -> str:
    payload = {
        "message": message_text,
        "has_reply": has_reply,
        "plugin_cards": cards,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _message_context_flags(
    message_text: str, *, has_reply: bool = False
) -> dict[str, bool]:
    return {
        "has_image": bool(_IMAGE_PLACEHOLDER_PATTERN.search(message_text)),
        "has_at": bool(
            _AT_PLACEHOLDER_PATTERN.search(message_text)
            or _AT_INLINE_PATTERN.search(message_text)
        ),
        "has_reply": has_reply,
    }


def _resolve_command_index_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    has_reply: bool = False,
    session_key: str | None = None,
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    flags = _message_context_flags(message_text, has_reply=has_reply)
    speech_act = classify_speech_act(message_text, **flags)
    if speech_act == "discuss_command":
        return LLMRouterDecision(
            action="chat",
            confidence=0.82,
            reason=f"speech_act:{speech_act}",
        ), None

    candidates = build_command_candidates(
        knowledge_base,
        message_text,
        limit=8,
        session_id=session_key,
    )
    if not candidates:
        return None

    top = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = top.score - second_score
    if top.score < 190.0:
        return None
    if margin < 24.0 and top.schema.command_role not in {"catalog", "helper", "random"}:
        return None

    schema = top.schema
    route_slots, schema_missing = complete_slots(
        schema,
        slots={},
        message_text=message_text,
        arguments_text="",
    )
    rendered, schema_missing = render_command(
        schema,
        slots=route_slots,
        message_text=message_text,
        arguments_text="",
    )
    action: Literal["chat", "execute", "usage", "clarify"]
    if speech_act == "ask_usage":
        action = "usage"
    elif schema_missing:
        action = "clarify"
    else:
        action = "execute"
    decision_command = schema.head if action == "usage" else rendered or schema.head
    decision_slots = {} if action == "usage" else route_slots
    decision_missing = [] if action == "usage" else list(schema_missing)

    decision = LLMRouterDecision(
        action=action,
        confidence=min(max(top.score / 360.0, 0.0), 0.98),
        plugin_module=top.plugin_module,
        plugin_name=top.plugin_name,
        command_id=schema.command_id,
        command=decision_command,
        slots=decision_slots,
        missing=decision_missing,
        reason=f"command_index:{top.reason};score={top.score:.1f};margin={margin:.1f}",
    )
    if action == "usage":
        return decision, _to_plugin_route_result(
            plugin=_resolve_target_plugin(
                _LLMRouteDecision(
                    action="route",
                    plugin_module=top.plugin_module,
                    plugin_name=top.plugin_name,
                    command=schema.head,
                ),
                knowledge_base,
            )
            or PluginInfo(
                module=top.plugin_module,
                name=top.plugin_name,
                description="",
            ),
            stage="command_index_usage",
            source="command_index",
            skill_kind="command_index",
            command=decision_command,
            command_id=schema.command_id,
            slots=decision_slots,
            missing=decision_missing,
        )

    route_result = _to_route_result(
        _LLMRouteDecision(
            action="route",
            plugin_module=top.plugin_module,
            plugin_name=top.plugin_name,
            command=rendered or schema.head,
        ),
        message_text,
        knowledge_base,
        command_id=schema.command_id,
        slots=route_slots,
        missing=schema_missing,
    )
    if route_result is None:
        return None
    return decision, RouteResolveResult(
        decision=SkillRouteDecision(
            plugin_name=route_result.decision.plugin_name,
            plugin_module=route_result.decision.plugin_module,
            command=route_result.decision.command,
            source="command_index",
            skill_kind="command_index",
        ),
        stage="command_index",
        report=route_result.report,
        command_id=route_result.command_id,
        slots=route_result.slots,
        missing=route_result.missing,
    )


def _build_candidate_selection_prompt(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    has_reply: bool,
) -> str:
    _, tool_specs = build_command_choice_tools(
        candidates,
        max_tools=len(candidates),
    )
    tool_by_command_id = {
        spec.command_id: {
            "tool_name": spec.tool_name,
            "candidate_index": spec.candidate_index,
        }
        for spec in tool_specs.values()
    }
    payload = {
        "message": message_text,
        "has_reply": has_reply,
        "task": (
            "只在 candidates 中选择一个 command_id，或判断为 chat/clarify/usage。"
            "需要执行时同时填写 slots。不要自由发明命令。"
        ),
        "candidates": [
            {
                "rank": index,
                "score": round(candidate.score, 2),
                "family": candidate.family,
                "reason": candidate.reason,
                "plugin_module": candidate.plugin_module,
                "plugin_name": candidate.plugin_name,
                "tool": tool_by_command_id.get(candidate.schema.command_id, {}),
                **dump_schema_for_prompt(candidate.schema),
            }
            for index, candidate in enumerate(candidates, 1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _fallback_candidate_selection(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    has_reply: bool,
) -> LLMCommandSelection | None:
    if not candidates:
        return None
    flags = _message_context_flags(message_text, has_reply=has_reply)
    speech_act = classify_speech_act(message_text, **flags)
    if speech_act in {"casual_chat", "discuss_command"}:
        return None
    top = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = top.score - second_score
    strong_enough = top.score >= 150.0 and (
        margin >= 18.0 or top.schema.command_role in {"catalog", "helper", "random"}
    )
    if not strong_enough:
        return None
    if speech_act == "ask_usage":
        return LLMCommandSelection(
            action="usage",
            command_id=top.schema.command_id,
            confidence=min(max(top.score / 360.0, 0.0), 0.9),
            reason=f"local_fallback_usage:{top.reason}",
        )
    slots, missing = complete_slots(
        top.schema,
        slots={},
        message_text=message_text,
        arguments_text="",
    )
    return LLMCommandSelection(
        action="clarify" if missing else "execute",
        command_id=top.schema.command_id,
        slots=slots,
        missing=list(missing),
        confidence=min(max(top.score / 360.0, 0.0), 0.9),
        reason=f"local_fallback:{top.reason};score={top.score:.1f};margin={margin:.1f}",
    )


def _is_generic_meme_creation_request(
    message_text: str,
    candidates: list[CommandCandidate],
) -> bool:
    normalized = normalize_message_text(message_text).casefold()
    if not normalized:
        return False
    meme_candidates = [item for item in candidates if item.family == "meme"]
    if not meme_candidates:
        return False
    if "随机" in normalized:
        return False
    if not any(token in normalized for token in ("表情", "表情包", "梗图", "头像")):
        return False
    if not any(
        token in normalized
        for token in ("做", "制作", "生成", "整", "来个", "来一个", "来张", "来一张")
    ):
        return False
    for candidate in meme_candidates:
        schema = candidate.schema
        if schema.command_role not in {"template", "random"}:
            continue
        phrases = [schema.head, *schema.aliases]
        if any(
            normalize_message_text(phrase).casefold()
            and normalize_message_text(phrase).casefold() in normalized
            for phrase in phrases
        ):
            return False
    return True


def _pick_meme_clarify_candidate(
    candidates: list[CommandCandidate],
) -> CommandCandidate | None:
    meme_candidates = [item for item in candidates if item.family == "meme"]
    for candidate in meme_candidates:
        if candidate.schema.command_id == "memes.list":
            return candidate
    for candidate in meme_candidates:
        if candidate.schema.command_role in {"catalog", "helper"}:
            return candidate
    return meme_candidates[0] if meme_candidates else None


def _candidate_selection_to_route_result(
    *,
    selection: LLMCommandSelection,
    candidates: list[CommandCandidate],
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    stage: str,
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    if selection.action == "chat":
        return (
            LLMRouterDecision(
                action="chat",
                confidence=selection.confidence,
                reason=selection.reason or f"{stage}:chat",
            ),
            None,
        )

    command_id = normalize_message_text(selection.command_id or "")
    if not command_id:
        return None
    candidate = next(
        (
            item
            for item in candidates
            if normalize_message_text(item.schema.command_id) == command_id
        ),
        None,
    )
    if candidate is None:
        return None

    schema = candidate.schema
    route_slots = dict(selection.slots or {})
    if selection.action == "usage":
        rendered = schema.head
        schema_missing: list[str] | tuple[str, ...] = ()
        route_slots = {}
    else:
        route_slots, schema_missing = complete_slots(
            schema,
            slots=route_slots,
            message_text=message_text,
            arguments_text="",
        )
        rendered, schema_missing = render_command(
            schema,
            slots=route_slots,
            message_text=message_text,
            arguments_text="",
        )
    missing = [*selection.missing, *list(schema_missing)]
    action: Literal["chat", "execute", "usage", "clarify"] = selection.action
    if action == "execute" and missing:
        action = "clarify"

    command = schema.head if action == "usage" else rendered or schema.head
    decision = LLMRouterDecision(
        action=action,
        confidence=selection.confidence,
        plugin_module=candidate.plugin_module,
        plugin_name=candidate.plugin_name,
        command_id=schema.command_id,
        command=command,
        slots={} if action == "usage" else route_slots,
        missing=[] if action == "usage" else missing,
        reason=selection.reason or f"{stage}:{candidate.reason}",
    )
    route_result = _to_route_result(
        _LLMRouteDecision(
            action="route",
            plugin_module=candidate.plugin_module,
            plugin_name=candidate.plugin_name,
            command=command,
        ),
        message_text,
        knowledge_base,
        command_id=schema.command_id,
        slots=decision.slots,
        missing=decision.missing,
    )
    if route_result is None:
        return None

    route_result = RouteResolveResult(
        decision=SkillRouteDecision(
            plugin_name=route_result.decision.plugin_name,
            plugin_module=route_result.decision.plugin_module,
            command=route_result.decision.command,
            source=stage,
            skill_kind=stage,
        ),
        stage=stage,
        report=route_result.report,
        command_id=route_result.command_id,
        slots=route_result.slots,
        missing=route_result.missing,
    )
    return decision, route_result


async def _resolve_candidate_selection_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    report: RouteAttemptReport,
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    candidates = build_command_candidates(
        knowledge_base,
        message_text,
        limit=max(
            int(get_config_value("ROUTE_COMMAND_CANDIDATE_LIMIT", 32) or 32),
            8,
        ),
        session_id=session_key,
    )
    if not candidates:
        return None
    # Very weak candidates are better left to the existing chat fallback path.
    if candidates[0].score < 80.0:
        return None

    report.note_tool_pool(len(candidates))
    if _is_generic_meme_creation_request(message_text, candidates):
        clarify_candidate = _pick_meme_clarify_candidate(candidates)
        if clarify_candidate is not None:
            return _candidate_selection_to_route_result(
                selection=LLMCommandSelection(
                    action="clarify",
                    command_id=clarify_candidate.schema.command_id,
                    missing=["具体表情模板"],
                    confidence=0.86,
                    reason="generic_meme_template_missing",
                ),
                candidates=candidates,
                message_text=message_text,
                knowledge_base=knowledge_base,
                stage="command_selector",
            )
    prompt = _build_candidate_selection_prompt(
        message_text=message_text,
        candidates=candidates,
        has_reply=has_reply,
    )
    timeout = get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout_value = float(timeout) if timeout else None
    except (TypeError, ValueError):
        timeout_value = None

    candidate_ids = {
        normalize_message_text(candidate.schema.command_id) for candidate in candidates
    }

    async def _validate_selection(selection: LLMCommandSelection) -> None:
        if selection.action in {"execute", "usage", "clarify"}:
            selected_id = normalize_message_text(selection.command_id or "")
            if selected_id not in candidate_ids:
                raise ValueError(f"unknown command_id: {selected_id}")

    selection: LLMCommandSelection | None = None
    try:
        if budget_controller is not None and not budget_controller.allow_classifier(
            "command_selector"
        ):
            selection = _fallback_candidate_selection(
                message_text=message_text,
                candidates=candidates,
                has_reply=has_reply,
            )
        else:
            started = time.perf_counter()
            guarded = guard_prompt_sections(
                session_key=session_key or "global",
                stage="command_selector",
                system_prompt=(
                    "你是命令分类器。只能从 candidates 里选 command_id；"
                    "如果用户是在聊天、讨论概念、或请求不属于候选命令，返回 chat；"
                    "如果用户询问怎么用/用法，返回 usage；"
                    "如果命令明确但缺少必填参数，返回 clarify 并列出 missing；"
                    "如果执行意图明确，返回 execute 并填写 slots。"
                ),
                user_text=prompt,
                controller=budget_controller,
            )
            selection = await generate_structured(
                guarded.user_text,
                LLMCommandSelection,
                model=get_model_name(),
                instruction=guarded.system_prompt,
                timeout=timeout_value,
                validation_callback=_validate_selection,
            )
            if budget_controller is not None:
                budget_controller.record_classifier(
                    "command_selector",
                    time.perf_counter() - started,
                )
    except Exception as exc:
        logger.debug(f"ChatInter command selector 失败，尝试本地兜底: {exc}")
        selection = _fallback_candidate_selection(
            message_text=message_text,
            candidates=candidates,
            has_reply=has_reply,
        )

    if selection is None:
        return None
    result = _candidate_selection_to_route_result(
        selection=selection,
        candidates=candidates,
        message_text=message_text,
        knowledge_base=knowledge_base,
        stage="command_selector",
    )
    if result is not None:
        report.tool_choice_count += 1
    return result


def _pick_default_command(plugin: PluginInfo) -> str:
    for meta in plugin.command_meta:
        command = normalize_message_text(meta.command)
        if command:
            return command
    for command in plugin.commands:
        normalized = normalize_message_text(command)
        if normalized:
            return normalized
    return ""


def _router_decision_to_route_result(
    decision: LLMRouterDecision,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    allow_default_command: bool = False,
) -> RouteResolveResult | None:
    plugin = _resolve_target_plugin(
        _LLMRouteDecision(
            action="route",
            plugin_module=decision.plugin_module,
            plugin_name=decision.plugin_name,
            command=decision.command,
        ),
        knowledge_base,
    )
    if plugin is None:
        return None

    command = normalize_message_text(decision.command or "")
    route_command_id = normalize_message_text(decision.command_id or "") or None
    route_slots = dict(decision.slots or {})
    schemas = []
    capability = capability_from_plugin(plugin)
    if capability is not None:
        schemas = build_command_schemas(plugin.module, capability.commands)
    selection = None
    if not (decision.action == "usage" and not decision.command_id and not command):
        selection = select_command_schema(
            schemas,
            command_id=decision.command_id,
            command=command,
            message_text=message_text,
            arguments_text=decision.arguments_text,
            slots=route_slots,
            action=decision.action,
        )
    schema = selection.schema if selection is not None else None
    if schema is not None:
        route_command_id = schema.command_id
        route_slots, schema_missing = complete_slots(
            schema,
            slots=route_slots,
            message_text=message_text,
            arguments_text=decision.arguments_text,
        )
        rendered, schema_missing = render_command(
            schema,
            slots=route_slots,
            message_text=message_text,
            arguments_text=decision.arguments_text,
        )
        command = rendered or schema.head
        if schema_missing and decision.action == "execute":
            decision.action = "clarify"
            decision.missing = [*decision.missing, *schema_missing]

    if not command and allow_default_command:
        command = schemas[0].head if schemas else _pick_default_command(plugin)
    if not command:
        if decision.action == "usage":
            return _to_plugin_route_result(
                plugin=plugin,
                stage="llm",
                source="llm",
                skill_kind="llm",
                command_id=route_command_id,
                slots=route_slots,
                missing=decision.missing,
            )
        return None

    arguments = normalize_message_text(decision.arguments_text or "")
    if (
        decision.action == "execute"
        and schema is None
        and arguments
        and arguments not in command
    ):
        command = normalize_message_text(f"{command} {arguments}")

    return _to_route_result(
        _LLMRouteDecision(
            action="route",
            plugin_module=plugin.module,
            plugin_name=plugin.name,
            command=command,
        ),
        message_text,
        knowledge_base,
        command_id=route_command_id,
        slots=route_slots,
        missing=decision.missing,
    )


async def resolve_llm_router(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None = None,
    budget_controller: TurnBudgetController | None = None,
    has_reply: bool = False,
) -> tuple[LLMRouterDecision, RouteResolveResult | None, RouteAttemptReport]:
    normalized_message = normalize_message_text(message_text)
    report = RouteAttemptReport(helper_mode=is_usage_question(normalized_message))
    if not normalized_message:
        report.finalize(reason="router_empty")
        return LLMRouterDecision(action="chat", reason="empty"), None, report

    indexed = _resolve_command_index_route(
        normalized_message,
        knowledge_base,
        has_reply=has_reply,
        session_key=session_key,
    )
    if indexed is not None:
        decision, route_result = indexed
        report.finalize(
            reason=decision.reason or f"command_index_{decision.action}",
            stage=route_result.stage if route_result is not None else "speech_act",
            plugin_name=route_result.decision.plugin_name if route_result else None,
            plugin_module=route_result.decision.plugin_module if route_result else None,
            command=route_result.decision.command if route_result else None,
        )
        return decision, route_result, report

    candidate_route = await _resolve_candidate_selection_route(
        normalized_message,
        knowledge_base,
        session_key=session_key,
        budget_controller=budget_controller,
        has_reply=has_reply,
        report=report,
    )
    if candidate_route is not None:
        decision, route_result = candidate_route
        report.finalize(
            reason=decision.reason or f"command_selector_{decision.action}",
            stage=(
                route_result.stage
                if route_result is not None
                else "command_selector"
            ),
            plugin_name=route_result.decision.plugin_name if route_result else None,
            plugin_module=route_result.decision.plugin_module if route_result else None,
            command=route_result.decision.command if route_result else None,
        )
        return decision, route_result, report

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
    report.note_attempt(shortlisted_modules[:_ROUTE_TRACE_SAMPLE_LIMIT])

    cards = PluginRegistry.build_router_cards(
        attempt_kb,
        limit=shortlist_limit,
        query=normalized_message,
    )
    prompt = _build_router_prompt(
        message_text=normalized_message,
        cards=cards,
        has_reply=has_reply,
    )
    timeout = get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout_value = float(timeout) if timeout else None
    except (TypeError, ValueError):
        timeout_value = None

    try:
        if budget_controller is not None and not budget_controller.allow_classifier(
            "router"
        ):
            report.finalize(reason="router_budget_denied")
            return (
                LLMRouterDecision(action="chat", reason="budget_denied"),
                None,
                report,
            )
        started = time.perf_counter()
        guarded = guard_prompt_sections(
            session_key=session_key or "global",
            stage="router",
            system_prompt=_ROUTER_INSTRUCTION,
            user_text=prompt,
            controller=budget_controller,
        )
        decision = await generate_structured(
            guarded.user_text,
            LLMRouterDecision,
            model=get_model_name(),
            instruction=guarded.system_prompt,
            timeout=timeout_value,
        )
        if budget_controller is not None:
            budget_controller.record_classifier(
                "router",
                time.perf_counter() - started,
            )
    except Exception as exc:
        logger.debug(f"ChatInter Router 失败，降级为纯会话: {exc}")
        report.finalize(reason="router_error")
        return LLMRouterDecision(action="chat", reason="router_error"), None, report

    route_result: RouteResolveResult | None = None
    if decision.action in {"execute", "usage", "clarify"}:
        route_result = _router_decision_to_route_result(
            decision,
            normalized_message,
            attempt_kb,
            allow_default_command=decision.action == "clarify",
        )
        if route_result is None:
            report.finalize(reason="router_invalid_selection")
            return (
                LLMRouterDecision(
                    action="chat",
                    confidence=decision.confidence,
                    reason="invalid_selection",
                ),
                None,
                report,
            )

    report.finalize(
        reason=f"router_{decision.action}",
        stage="router",
        plugin_name=route_result.decision.plugin_name if route_result else None,
        plugin_module=route_result.decision.plugin_module if route_result else None,
        command=route_result.decision.command if route_result else None,
    )
    return decision, route_result, report


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
    if not _has_strong_shortlist_entity_signal(entity_tokens):
        return None

    query_family = infer_query_family(message_text)
    normalized = normalize_message_text(message_text).lower()
    registry = get_skill_registry(knowledge_base)

    best_skill: Any | None = None
    best_command: str | None = None
    best_score = 0.0
    for skill in registry.skills:
        command_pool: list[str] = []
        for command in [*skill.commands, *skill.aliases]:
            normalized_command = normalize_message_text(command)
            if normalized_command and normalized_command not in command_pool:
                command_pool.append(normalized_command)
        if not command_pool:
            continue

        skill_best_score = 0.0
        skill_best_command: str | None = None
        for candidate in command_pool:
            candidate_lower = candidate.lower()
            matched = [
                token for token in entity_tokens if token and token in candidate_lower
            ]
            if not matched:
                continue
            score = len(matched) * 10.0 + max(len(token) for token in matched)
            if candidate_lower == normalized:
                score += 24.0
            elif candidate_lower.startswith(normalized) or normalized.startswith(
                candidate_lower
            ):
                score += 10.0
            if len(candidate_lower) <= len(normalized):
                score += 2.0
            if query_family == "search":
                score += 2.0
            if score > skill_best_score:
                skill_best_score = score
                skill_best_command = candidate

        if skill_best_command is None:
            continue
        if skill_best_score > best_score:
            best_score = skill_best_score
            best_skill = skill
            best_command = skill_best_command

    if best_skill is None:
        return None
    if best_command is None:
        return None

    return RouteResolveResult(
        decision=SkillRouteDecision(
            plugin_name=best_skill.plugin_name,
            plugin_module=best_skill.plugin_module,
            command=best_command,
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
        if (
            shortlist_route_result is None
            or shortlist_route_result.decision.source
            not in {
                "fast",
                "rank_exact",
            }
        ):
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
            command_id=validated_shortlist_result.command_id,
            slots=dict(validated_shortlist_result.slots),
            missing=validated_shortlist_result.missing,
        ),
        report,
    )


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


async def _request_weak_signal_decision(
    *,
    prompt: str,
    timeout_value: float | None,
    session_key: str,
    budget_controller: TurnBudgetController | None,
) -> _WeakSignalDecision | None:
    try:
        if budget_controller is not None and not budget_controller.allow_classifier(
            "weak_signal_align"
        ):
            return None
        started = time.perf_counter()
        guarded = guard_prompt_sections(
            session_key=session_key,
            stage="weak_signal_align",
            system_prompt=_WEAK_SIGNAL_ALIGNMENT_INSTRUCTION,
            user_text=prompt,
            controller=budget_controller,
        )
        decision = await generate_structured(
            guarded.user_text,
            _WeakSignalDecision,
            model=get_model_name(),
            instruction=guarded.system_prompt,
            timeout=timeout_value,
        )
        if budget_controller is not None:
            budget_controller.record_classifier(
                "weak_signal_align",
                time.perf_counter() - started,
            )
        return decision
    except Exception as exc:
        logger.debug(f"ChatInter 弱词对齐失败: {exc}")
        return None


async def resolve_shortlist_alignment(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None = None,
    budget_controller: TurnBudgetController | None = None,
    has_reply: bool = False,
) -> tuple[
    _ShortlistAlignmentDecision | None, RouteResolveResult | None, RouteAttemptReport
]:
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
        has_reply=has_reply,
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

    stage_name = (
        "shortlist_align_usage" if decision.action == "usage" else "shortlist_align"
    )
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
            command_id=route_result.command_id,
            slots=dict(route_result.slots),
            missing=route_result.missing,
        ),
        report,
    )


async def resolve_weak_signal_intent(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    original_message_text: str | None = None,
    has_at: bool = False,
    has_image: bool = False,
    has_reply: bool = False,
    is_private: bool = False,
    session_key: str | None = None,
    budget_controller: TurnBudgetController | None = None,
) -> tuple[_WeakSignalDecision | None, RouteResolveResult | None, RouteAttemptReport]:
    normalized_message = normalize_message_text(message_text)
    weak_tags = collect_weak_route_signals(normalized_message)
    report = RouteAttemptReport(helper_mode=bool(weak_tags))
    if not normalized_message or not knowledge_base.plugins or not weak_tags:
        report.finalize(reason="weak_signal_empty")
        return None, None, report

    timeout = get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout_value = float(timeout) if timeout else None
    except (TypeError, ValueError):
        timeout_value = None

    shortlist_limit = max(_WEAK_SIGNAL_SHORTLIST_LIMIT, 1)
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

    prompt = _build_weak_signal_alignment_prompt(
        message_text=normalized_message,
        original_message_text=original_message_text,
        knowledge_base=attempt_kb,
        has_at=has_at,
        has_image=has_image,
        has_reply=has_reply,
        is_private=is_private,
    )
    decision = await _request_weak_signal_decision(
        prompt=prompt,
        timeout_value=timeout_value,
        session_key=session_key or "global",
        budget_controller=budget_controller,
    )
    if decision is None:
        report.finalize(reason="weak_signal_miss")
        return None, None, report

    if decision.action in {"chat", "ambiguous"}:
        report.finalize(
            reason=f"weak_signal_{decision.action}",
            stage="weak_signal_chat",
        )
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
        report.finalize(reason="weak_signal_invalid")
        return decision, None, report
    if not is_route_action_compatible(
        normalized_message,
        route_result.decision.command,
    ):
        report.finalize(reason="weak_signal_incompatible")
        return decision, None, report

    stage_name = (
        "weak_signal_usage" if decision.action == "usage" else "weak_signal_route"
    )
    report.finalize(
        reason=f"weak_signal_{decision.action}",
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
                source="weak_signal_llm",
                skill_kind="weak_signal_llm",
            ),
            stage=stage_name,
            command_id=route_result.command_id,
            slots=dict(route_result.slots),
            missing=route_result.missing,
        ),
        report,
    )


async def resolve_llm_align_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None = None,
    budget_controller: TurnBudgetController | None = None,
    has_reply: bool = False,
) -> tuple[
    _ShortlistAlignmentDecision | None, RouteResolveResult | None, RouteAttemptReport
]:
    """明确表达 ChatInter 的 LLM 对齐阶段。

    保留旧函数名 `resolve_shortlist_alignment` 作为兼容入口。
    direct -> llm_align -> validator -> route
    """
    return await resolve_shortlist_alignment(
        message_text,
        knowledge_base,
        session_key=session_key,
        budget_controller=budget_controller,
        has_reply=has_reply,
    )


__all__ = [
    "LLMCommandSelection",
    "LLMRouterDecision",
    "RouteAttemptReport",
    "RouteResolveResult",
    "resolve_llm_router",
]
