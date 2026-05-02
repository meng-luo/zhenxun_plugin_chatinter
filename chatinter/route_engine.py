from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from zhenxun.services import generate_structured, logger

from .capability_graph import capability_from_plugin
from .command_index import (
    CommandCandidate,
    build_candidate_snapshots,
    build_command_candidates,
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
from .no_hit_recovery import CapabilityRewriteResult, recover_no_hit_candidates
from .plugin_adapters import (
    get_adapter_target_policy_for_schema,
    resolve_adapter_clarify_route,
)
from .plugin_registry import PluginRegistry
from .prompt_guard import guard_prompt_sections
from .route_text import (
    STRONG_EXECUTE_WORDS,
    collect_placeholders,
    contains_any,
    has_chat_context_hint,
    has_negative_route_intent,
    is_usage_question,
    match_command_head_or_sticky,
    normalize_action_phrases,
    normalize_message_text,
    parse_command_with_head,
    strip_invoke_prefix,
)
from .schema_policy import resolve_command_target_policy
from .skill_registry import (
    SkillRouteDecision,
    _extract_argument_around_head,
    _message_has_payload_signals,
    get_skill_registry,
    select_relevant_skills,
)
from .speech_act import classify_speech_act
from .tool_reranker import ToolRerankDecision, request_tool_rerank
from .turn_runtime import TurnBudgetController

_ROUTE_NAMESPACE_LIMIT = 12
_ROUTE_PLUGIN_SHORTLIST_LIMIT = 30
_ROUTER_INSTRUCTION = """
判断用户消息是普通对话、插件执行、插件用法查询、还是需要澄清。
优先选择 plugin_cards.command_schemas 中存在的 command_id，并填写 slots。
不要自由发明命令；需要执行时给出 command_id 或合法命令头。
如果候选带 exact_protected=true，除非用户明显是在闲聊/讨论，否则不要改选其它命令。
列表/有哪些/搜索类 helper 命令属于 execute，只有“怎么用/用法/教程/示例”才 usage。
图片模板命令只需要图片时不要把自然语言描述当作 text 参数。
只有必填槽位缺失时才 clarify；无必填槽位的命令可以直接 execute。
输出 JSON，不要输出额外文本。
""".strip()
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


def _action_for_schema(
    *,
    schema: Any,
    speech_act: str,
    missing: list[str] | tuple[str, ...],
) -> Literal["chat", "execute", "usage", "clarify"]:
    role = normalize_message_text(getattr(schema, "command_role", "") or "").lower()
    if speech_act == "ask_usage" or role == "usage":
        return "usage"
    if role == "helper" and speech_act != "perform_command":
        return "usage"
    if missing:
        return "clarify"
    return "execute"


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


@dataclass(frozen=True)
class RouteResolveResult:
    decision: SkillRouteDecision
    stage: str
    report: "RouteAttemptReport | None" = None
    command_id: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    selected_rank: int = 0
    selected_score: float = 0.0
    selected_reason: str = ""


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
    prompt_full_candidates: int = 0
    prompt_compact_candidates: int = 0
    prompt_name_only_candidates: int = 0
    no_hit_recovery_attempts: int = 0
    no_hit_recovery_success: int = 0
    no_hit_recovery_query: str = ""
    no_hit_recovery_reason: str = ""
    rerank_attempts: int = 0
    rerank_success: int = 0
    rerank_no_available: int = 0
    rerank_stage: str = ""
    rerank_reason: str = ""
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

    def note_prompt_exposure(self, candidates: list[CommandCandidate]) -> None:
        snapshots = build_candidate_snapshots(candidates)
        self.prompt_full_candidates = max(
            self.prompt_full_candidates,
            sum(1 for item in snapshots if item.prompt_level == "full"),
        )
        self.prompt_compact_candidates = max(
            self.prompt_compact_candidates,
            sum(1 for item in snapshots if item.prompt_level == "compact"),
        )
        self.prompt_name_only_candidates = max(
            self.prompt_name_only_candidates,
            sum(1 for item in snapshots if item.prompt_level == "name_only"),
        )

    def note_no_hit_recovery(
        self,
        rewrite: CapabilityRewriteResult | None,
        *,
        success: bool,
    ) -> None:
        self.no_hit_recovery_attempts += 1
        if success:
            self.no_hit_recovery_success += 1
        if rewrite is not None:
            self.no_hit_recovery_query = normalize_message_text(
                rewrite.capability_query
            )[:160]
            self.no_hit_recovery_reason = normalize_message_text(
                rewrite.reason or rewrite.action
            )[:160]

    def note_rerank(
        self,
        *,
        stage: str,
        decision: ToolRerankDecision | None,
        success: bool,
    ) -> None:
        self.rerank_attempts += 1
        self.rerank_stage = normalize_message_text(stage)[:80]
        if success:
            self.rerank_success += 1
        if decision is None:
            return
        if decision.action == "no_available_tool":
            self.rerank_no_available += 1
        self.rerank_reason = normalize_message_text(
            decision.reason or decision.no_available_tool_reason or decision.action
        )[:160]

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


class LLMSlotValue(BaseModel):
    name: str = Field(description="槽位名称，必须来自候选 schema")
    value: str = Field(default="", description="槽位值，统一以字符串填写")


def _slots_to_items(value: Any) -> list[LLMSlotValue]:
    if not value:
        return []
    if isinstance(value, dict):
        return [
            LLMSlotValue(name=str(key), value=str(slot_value))
            for key, slot_value in value.items()
            if normalize_message_text(str(key or ""))
        ]
    items: list[LLMSlotValue] = []
    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, LLMSlotValue):
                name = item.name
                slot_value = item.value
            elif isinstance(item, dict):
                name = str(item.get("name", "") or "")
                slot_value = str(item.get("value", "") or "")
            else:
                continue
            if normalize_message_text(name):
                items.append(LLMSlotValue(name=name, value=slot_value))
    return items


def _slots_to_dict(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _slots_to_items(value):
        name = normalize_message_text(item.name)
        if name:
            result[name] = str(item.value or "")
    return result


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
    slots: list[LLMSlotValue] = Field(
        default_factory=list,
        description="命令槽位列表，格式为 [{name,value}]，不要使用任意对象键",
    )
    arguments_text: str = Field(default="", description="命令后的自然语言参数")
    missing: list[str] = Field(default_factory=list, description="缺失信息")
    reason: str | None = Field(default=None, description="简短原因")

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots(cls, value: Any) -> list[LLMSlotValue]:
        return _slots_to_items(value)


class LLMCommandSelection(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(
        default="chat",
        description="chat=普通对话；execute=执行候选命令；usage=查询候选命令用法；clarify=需要补充信息",
    )
    command_id: str | None = Field(
        default=None,
        description="必须来自 candidates.command_id；chat 时为空",
    )
    slots: list[LLMSlotValue] = Field(
        default_factory=list,
        description=(
            "按候选 schema 填写的槽位列表，格式为 [{name,value}]，"
            "不要臆造 schema 之外的槽位"
        ),
    )
    missing: list[str] = Field(
        default_factory=list,
        description="缺失的必填槽位或上下文，例如 text/image/reply/target",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="简短选择理由")

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots(cls, value: Any) -> list[LLMSlotValue]:
        return _slots_to_items(value)


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
    command_id: str | None = None,
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

    policy = resolve_command_target_policy(
        schema,
        adapter_policy=get_adapter_target_policy_for_schema(
            schema,
            plugin_module=plugin.module,
            plugin_name=plugin.name,
            command_id=command_id or "",
        ),
    )
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
    missing: list[str] | tuple[str, ...] = (),
) -> str:
    normalized = normalize_message_text(command)
    if not normalized:
        return normalized
    if missing:
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

    command = _sanitize_command_with_schema(
        plugin,
        command=command,
        command_id=command_id,
    )
    schema = _resolve_command_schema(plugin, command_head)
    if schema is not None:
        command = _rehydrate_command_payload_from_message(
            command,
            message_text=message_text,
            schema=schema,
            missing=missing,
        )
    elif _message_has_payload_signals(message_text):
        command = _rehydrate_command_payload_from_message(
            command,
            message_text=message_text,
            schema=None,
            missing=missing,
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
        selected_rank=0,
        selected_score=0.0,
        selected_reason="",
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
        selected_rank=route_result.selected_rank,
        selected_score=route_result.selected_score,
        selected_reason=route_result.selected_reason,
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


def _is_exact_command_candidate(candidate: CommandCandidate | None) -> bool:
    return bool(candidate and candidate.exact_protected)


def _should_force_chat_before_command_index(message_text: str, speech_act: str) -> bool:
    normalized = normalize_message_text(message_text)
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if has_negative_route_intent(normalized) or has_negative_route_intent(stripped):
        return True
    if speech_act == "discuss_command":
        return True
    if speech_act != "casual_chat":
        return False
    if not has_chat_context_hint(stripped):
        return False
    return not contains_any(stripped, STRONG_EXECUTE_WORDS)


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
    command_tools: list[Any] | None = None,
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    flags = _message_context_flags(message_text, has_reply=has_reply)
    speech_act = classify_speech_act(message_text, **flags)
    if _should_force_chat_before_command_index(message_text, speech_act):
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
        tools=command_tools,
    )
    if not candidates:
        return None

    top = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = top.score - second_score
    if top.score < 190.0 and not top.exact_protected:
        return None
    if (
        margin < 24.0
        and not top.exact_protected
        and top.schema.command_role not in {"catalog", "helper", "random"}
    ):
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
    action = _action_for_schema(
        schema=schema,
        speech_act=speech_act,
        missing=schema_missing,
    )
    clarify_route = resolve_adapter_clarify_route(message_text, candidates)
    if clarify_route is not None and schema.command_id == clarify_route.command_id:
        action = "clarify"
        schema_missing = list(clarify_route.missing)
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
        slots=_slots_to_items(decision_slots),
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
        selected_rank=1,
        selected_score=top.score,
        selected_reason=top.reason,
    )


def _rerank_to_command_selection(
    decision: ToolRerankDecision,
) -> LLMCommandSelection | None:
    if decision.action == "no_available_tool":
        return LLMCommandSelection(
            action="chat",
            confidence=decision.confidence,
            reason=(
                "no_available_tool:"
                + normalize_message_text(
                    decision.no_available_tool_reason or decision.reason
                )
            ),
        )
    if decision.action == "chat":
        return LLMCommandSelection(
            action="chat",
            confidence=decision.confidence,
            reason=decision.reason or "tool_rerank_chat",
        )
    command_id = normalize_message_text(decision.command_id or "")
    if not command_id:
        return None
    return LLMCommandSelection(
        action=decision.action,
        command_id=command_id,
        slots=[
            LLMSlotValue(name=item.name, value=item.value)
            for item in decision.slots
            if normalize_message_text(item.name)
        ],
        missing=list(decision.missing),
        confidence=decision.confidence,
        reason=decision.reason or f"tool_rerank:{decision.action}",
    )


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
    if _should_force_chat_before_command_index(message_text, speech_act):
        return LLMCommandSelection(
            action="chat",
            confidence=0.82,
            reason=f"speech_act:{speech_act}",
        )
    if speech_act in {"casual_chat", "discuss_command"}:
        if not _is_exact_command_candidate(candidates[0]):
            return None
    top = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = top.score - second_score
    strong_enough = top.exact_protected or (
        top.score >= 150.0
        and (
            margin >= 18.0 or top.schema.command_role in {"catalog", "helper", "random"}
        )
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
    action = _action_for_schema(
        schema=top.schema,
        speech_act=speech_act,
        missing=missing,
    )
    return LLMCommandSelection(
        action=action,
        command_id=top.schema.command_id,
        slots=_slots_to_items(slots),
        missing=[] if action == "usage" else list(missing),
        confidence=min(max(top.score / 360.0, 0.0), 0.9),
        reason=f"local_fallback:{top.reason};score={top.score:.1f};margin={margin:.1f}",
    )


def _selection_matches_command_context(
    *,
    selection: LLMCommandSelection,
    candidate: CommandCandidate,
    message_text: str,
    has_reply: bool,
) -> tuple[bool, str]:
    if selection.action == "usage":
        return True, ""
    requires = candidate.schema.requires or {}
    flags = _message_context_flags(message_text, has_reply=has_reply)
    image_satisfied = flags["has_image"] or (requires.get("at") and flags["has_at"])
    if requires.get("image") and not image_satisfied:
        return False, "missing image context"
    if requires.get("reply") and not flags["has_reply"]:
        return False, "missing reply context"
    if requires.get("at") and not flags["has_at"]:
        return False, "missing at context"
    return True, ""


def _candidate_selection_to_route_result(
    *,
    selection: LLMCommandSelection,
    candidates: list[CommandCandidate],
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    stage: str,
    has_reply: bool = False,
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
    context_ok, context_reason = _selection_matches_command_context(
        selection=selection,
        candidate=candidate,
        message_text=message_text,
        has_reply=has_reply,
    )
    if not context_ok:
        missing_name = context_reason.rsplit(" ", 1)[-1] or "context"
        selection = LLMCommandSelection(
            action="clarify",
            command_id=candidate.schema.command_id,
            slots=_slots_to_items(selection.slots),
            missing=[*selection.missing, missing_name],
            confidence=min(selection.confidence, 0.82),
            reason=f"{selection.reason};validator:{context_reason}",
        )

    schema = candidate.schema
    route_slots = _slots_to_dict(selection.slots)
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
    action = _action_for_schema(
        schema=schema,
        speech_act=classify_speech_act(
            message_text,
            **_message_context_flags(message_text, has_reply=has_reply),
        ),
        missing=missing,
    )
    if selection.action == "clarify":
        action = "clarify"
    elif selection.action == "usage":
        action = "usage"
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
        slots=[] if action == "usage" else _slots_to_items(route_slots),
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
        slots=_slots_to_dict(decision.slots),
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
        selected_rank=next(
            (
                index
                for index, item in enumerate(candidates, 1)
                if item.schema.command_id == schema.command_id
            ),
            0,
        ),
        selected_score=candidate.score,
        selected_reason=candidate.reason,
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
    command_tools: list[Any] | None = None,
    candidates: list[CommandCandidate] | None = None,
    stage: str = "command_selector",
    use_reranker: bool = False,
    recovery_query: str = "",
    recovery_reason: str = "",
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    if candidates is None:
        candidates = build_command_candidates(
            knowledge_base,
            message_text,
            limit=max(
                int(get_config_value("ROUTE_COMMAND_CANDIDATE_LIMIT", 32) or 32),
                8,
            ),
            session_id=session_key,
            tools=command_tools,
        )
    if not candidates:
        return None

    report.candidate_total = max(report.candidate_total, len(candidates))
    report.note_tool_pool(len(candidates))
    report.note_prompt_exposure(candidates)
    clarify_route = resolve_adapter_clarify_route(message_text, candidates)
    if clarify_route is not None:
        clarify_candidate = next(
            (
                item
                for item in candidates
                if item.schema.command_id == clarify_route.command_id
            ),
            None,
        )
        if clarify_candidate is not None:
            route_result = _to_plugin_route_result(
                plugin=_resolve_target_plugin(
                    _LLMRouteDecision(
                        action="route",
                        plugin_module=clarify_candidate.plugin_module,
                        plugin_name=clarify_candidate.plugin_name,
                        command=clarify_candidate.schema.head,
                    ),
                    knowledge_base,
                )
                or PluginInfo(
                    module=clarify_candidate.plugin_module,
                    name=clarify_candidate.plugin_name,
                    description="",
                ),
                stage=stage,
                source=stage,
                skill_kind=stage,
                command=clarify_candidate.schema.head,
                command_id=clarify_candidate.schema.command_id,
                missing=clarify_route.missing,
            )
            selection = LLMCommandSelection(
                action="clarify",
                command_id=clarify_candidate.schema.command_id,
                missing=list(clarify_route.missing),
                confidence=clarify_route.confidence,
                reason=clarify_route.reason,
            )
            if route_result is None:
                return _candidate_selection_to_route_result(
                    selection=selection,
                    candidates=candidates,
                    message_text=message_text,
                    knowledge_base=knowledge_base,
                    stage=stage,
                    has_reply=has_reply,
                )
            return (
                LLMRouterDecision(
                    action="clarify",
                    confidence=clarify_route.confidence,
                    plugin_module=clarify_candidate.plugin_module,
                    plugin_name=clarify_candidate.plugin_name,
                    command_id=clarify_candidate.schema.command_id,
                    command=clarify_candidate.schema.head,
                    missing=list(clarify_route.missing),
                    reason=clarify_route.reason,
                ),
                RouteResolveResult(
                    decision=route_result.decision,
                    stage=route_result.stage,
                    report=route_result.report,
                    command_id=route_result.command_id,
                    slots=route_result.slots,
                    missing=route_result.missing,
                    selected_rank=next(
                        (
                            index
                            for index, item in enumerate(candidates, 1)
                            if item.schema.command_id
                            == clarify_candidate.schema.command_id
                        ),
                        0,
                    ),
                    selected_score=clarify_candidate.score,
                    selected_reason=clarify_candidate.reason,
                ),
            )

    # Very weak candidates are better left to the existing chat fallback path.
    if candidates[0].score < 80.0 and not candidates[0].exact_protected:
        return None

    selection: LLMCommandSelection | None = None
    try:
        rerank_decision = await request_tool_rerank(
            message_text=message_text,
            candidates=candidates,
            has_reply=has_reply,
            stage=stage if use_reranker else "command_selector",
            session_key=session_key,
            budget_controller=budget_controller,
            recovery_query=recovery_query,
            recovery_reason=recovery_reason,
        )
        report.note_rerank(
            stage=stage,
            decision=rerank_decision,
            success=bool(
                rerank_decision
                and rerank_decision.action in {"execute", "usage", "clarify"}
            ),
        )
        if rerank_decision is not None:
            selection = _rerank_to_command_selection(rerank_decision)
        if selection is None:
            selection = _fallback_candidate_selection(
                message_text=message_text,
                candidates=candidates,
                has_reply=has_reply,
            )
    except Exception as exc:
        logger.debug(f"ChatInter tool rerank 失败，尝试本地兜底: {exc}")
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
        stage=stage,
        has_reply=has_reply,
    )
    if result is not None:
        report.tool_choice_count += 1
    return result


async def _resolve_no_hit_recovery_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    report: RouteAttemptReport,
    command_tools: list[Any] | None = None,
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    tools = list(command_tools or [])
    if not tools:
        return None
    limit = max(int(get_config_value("ROUTE_COMMAND_CANDIDATE_LIMIT", 32) or 32), 8)
    rewrite, candidates = await recover_no_hit_candidates(
        knowledge_base,
        message_text,
        tools=tools,
        session_id=session_key,
        has_reply=has_reply,
        limit=limit,
    )
    report.note_no_hit_recovery(rewrite, success=bool(candidates))
    if not candidates:
        return None
    result = await _resolve_candidate_selection_route(
        message_text,
        knowledge_base,
        session_key=session_key,
        budget_controller=budget_controller,
        has_reply=has_reply,
        report=report,
        command_tools=tools,
        candidates=candidates,
        stage="no_hit_recovery",
        use_reranker=True,
        recovery_query=rewrite.capability_query if rewrite is not None else "",
        recovery_reason=rewrite.reason if rewrite is not None else "",
    )
    if result is None:
        return None
    decision, route_result = result
    if rewrite is not None:
        decision.reason = normalize_message_text(
            f"{decision.reason};recovery_query:{rewrite.capability_query}"
        )
        if route_result is not None:
            route_result = RouteResolveResult(
                decision=route_result.decision,
                stage=route_result.stage,
                report=route_result.report,
                command_id=route_result.command_id,
                slots=route_result.slots,
                missing=route_result.missing,
                selected_rank=route_result.selected_rank,
                selected_score=route_result.selected_score,
                selected_reason=normalize_message_text(
                    f"{route_result.selected_reason};"
                    f"recovery_query:{rewrite.capability_query}"
                ),
            )
    return decision, route_result


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
    route_slots = _slots_to_dict(decision.slots)
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
        allowed_slots = {
            normalize_message_text(slot.name)
            for slot in schema.slots
            if normalize_message_text(slot.name)
        }
        if allowed_slots:
            route_slots = {
                key: value
                for key, value in route_slots.items()
                if normalize_message_text(str(key or "")) in allowed_slots
            }
        else:
            route_slots = {}
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
    command_tools: list[Any] | None = None,
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
        command_tools=command_tools,
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
        command_tools=command_tools,
    )
    if candidate_route is not None:
        decision, route_result = candidate_route
        report.finalize(
            reason=decision.reason or f"command_selector_{decision.action}",
            stage=(
                route_result.stage if route_result is not None else "command_selector"
            ),
            plugin_name=route_result.decision.plugin_name if route_result else None,
            plugin_module=route_result.decision.plugin_module if route_result else None,
            command=route_result.decision.command if route_result else None,
        )
        return decision, route_result, report

    recovery_route = await _resolve_no_hit_recovery_route(
        normalized_message,
        knowledge_base,
        session_key=session_key,
        budget_controller=budget_controller,
        has_reply=has_reply,
        report=report,
        command_tools=command_tools,
    )
    if recovery_route is not None:
        decision, route_result = recovery_route
        report.finalize(
            reason=decision.reason or f"no_hit_recovery_{decision.action}",
            stage=(
                route_result.stage if route_result is not None else "no_hit_recovery"
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


__all__ = [
    "LLMCommandSelection",
    "LLMRouterDecision",
    "RouteAttemptReport",
    "RouteResolveResult",
    "resolve_llm_router",
]
