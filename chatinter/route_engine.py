from dataclasses import dataclass
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from zhenxun.services import chat, generate_structured, logger

from .config import get_config_value
from .knowledge_rag import PluginRAGService
from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .retrieval import rank_route_candidates
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_placeholders,
    contains_any,
    is_usage_question,
    normalize_message_text,
)
from .schema_policy import resolve_command_target_policy
from .skill_registry import (
    SkillRouteDecision,
    get_skill_registry,
    render_skill_namespace,
    skill_execute,
    skill_search,
)

_ROUTE_NAMESPACE_LIMIT = 16
_ROUTE_VECTOR_TOP_K = 10
_ROUTE_VECTOR_FETCH_K = 24
_ROUTE_VECTOR_MAX_K = 20
_ROUTE_VECTOR_INCREMENT = 4
_ROUTE_MIN_NAMESPACE_LIMIT = 6
_ROUTE_CANDIDATE_INITIAL_LIMIT = 10
_ROUTE_CANDIDATE_MAX_LIMIT = 20
_ROUTE_CANDIDATE_EXPAND_STEP = 4
_ROUTE_CANDIDATE_MIN_SCORE = 0.35
_ROUTE_PROMPT_TOKEN_PATTERN = re.compile(
    r"[a-z0-9_]+|[\u4e00-\u9fff]",
    re.IGNORECASE,
)
_LLM_ROUTE_INSTRUCTION = """
你是 ChatInter 的技能路由器，只负责在给定技能命名空间内选择可执行命令。
必须严格遵守：
1. 只能从 skills_json 中选择 plugin_module 与命令头。
2. command 必须是一条可直接发送给插件的命令字符串。
3. 用户是执行诉求时优先 action_commands。
4. 当用户是在问“怎么用/用法/帮助/参数/说明”时：
   优先选择通用帮助命令“帮助 <插件名或命令词>”或“功能 <插件名或命令词>”
   （推荐口径等价于“真寻帮助<插件名>”）；
   并尽量从用户问题里抽取目标插件名或命令词作为参数，例如“识图怎么用”=>“帮助 识图”；
   除非用户明确点名某个 helper 命令，否则不要把“怎么用/帮助”问题路由到
   业务 helper_commands（如“表情详情”）。
5. 若 schema.text_max 为 0，不要附带额外文本参数；仅保留 [@...] 或 [image] 占位符。
6. 保留用户消息里的 [@123456]、[image] 占位符，不要改写成自然语言。
7. 如果无法确定，返回 action=skip。
""".strip()
_LLM_ROUTE_RELAXED_SUFFIX = """
请仅输出一个 JSON 对象，不要输出额外解释。
可接受字段：
{
  "action": "route" | "skip",
  "plugin_module": "...",
  "plugin_name": "...",
  "command": "..."
}
""".strip()
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_MAX_VECTOR_NOTE_LEN = 96
_AT_PLACEHOLDER_PATTERN = re.compile(r"\[@\d{5,20}\]")
_AT_INLINE_PATTERN = re.compile(r"@\d{5,20}")
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)


@dataclass(frozen=True)
class RouteResolveResult:
    decision: SkillRouteDecision
    stage: str


class _LLMRouteDecision(BaseModel):
    action: Literal["route", "skip"] = Field(
        default="skip", description="route=调用插件；skip=不路由"
    )
    plugin_module: str | None = Field(
        default=None, description="目标插件模块，必须来自 skills_json"
    )
    plugin_name: str | None = Field(default=None, description="目标插件名称，可选")
    command: str | None = Field(default=None, description="最终可执行命令")


def _extract_first_json_object(text: str) -> str | None:
    source = str(text or "").strip()
    if not source:
        return None
    for fenced in _JSON_FENCE_PATTERN.findall(source):
        candidate = _extract_first_json_object(fenced)
        if candidate:
            return candidate

    start = source.find("{")
    if start < 0:
        return None

    in_string = False
    escape = False
    depth = 0
    for idx, ch in enumerate(source[start:], start=start):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    return None


def _validate_llm_route_decision(payload: dict) -> _LLMRouteDecision | None:
    try:
        return _LLMRouteDecision.model_validate(payload)
    except Exception:
        return None


def _normalize_llm_route_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}

    data = dict(payload)
    nested_result = data.get("result")
    if isinstance(nested_result, dict):
        data = dict(nested_result)
    plugin_intent = data.get("plugin_intent")
    if isinstance(plugin_intent, dict):
        merged = dict(plugin_intent)
        merged.setdefault("action", "route")
        return merged

    action_text = normalize_message_text(str(data.get("action", "")).lower())
    if action_text in {"call_plugin", "plugin", "route_plugin"}:
        data["action"] = "route"
    elif action_text in {"chat", "none", "no_route"}:
        data["action"] = "skip"
    return data


def _parse_llm_route_decision_loose(text: str) -> _LLMRouteDecision | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        json_block = _extract_first_json_object(raw)
        if not json_block:
            return None
        try:
            parsed = json.loads(json_block)
        except Exception:
            return None

    normalized = _normalize_llm_route_payload(parsed)
    return _validate_llm_route_decision(normalized)


def _resolve_stage(
    *,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    include_usage: bool,
    include_similarity: bool,
    stage: str,
) -> RouteResolveResult | None:
    search_result = skill_search(
        message_text,
        knowledge_base,
        include_usage=include_usage,
        include_similarity=include_similarity,
    )
    decision = skill_execute(search_result, message_text, knowledge_base)
    if decision is None:
        return None
    return RouteResolveResult(decision=decision, stage=stage)


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


def _collect_allowed_heads(plugin: PluginInfo) -> set[str]:
    allowed: set[str] = set()
    if plugin.command_meta:
        metas = plugin.command_meta
    else:
        metas = []
    for meta in metas:
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


def _resolve_namespace_limit(
    message_text: str,
    *,
    helper_mode: bool,
) -> int:
    normalized = normalize_message_text(message_text)
    if helper_mode:
        return min(max(_ROUTE_NAMESPACE_LIMIT - 2, 8), _ROUTE_NAMESPACE_LIMIT)
    if contains_any(normalized, ROUTE_ACTION_WORDS):
        return max(_ROUTE_NAMESPACE_LIMIT - 4, 8)
    if "[image" in normalized or "[@" in normalized:
        return max(_ROUTE_NAMESPACE_LIMIT - 4, 8)
    return max(_ROUTE_NAMESPACE_LIMIT - 6, 8)


def _resolve_vector_retrieve_options(
    message_text: str,
    *,
    helper_mode: bool,
) -> dict[str, object]:
    normalized = normalize_message_text(message_text)
    has_at = "[@" in normalized
    has_image = "[image" in normalized
    has_action_intent = contains_any(normalized, ROUTE_ACTION_WORDS)

    metadata_filters: dict[str, bool] = {}
    if has_at:
        metadata_filters["target_capable"] = True
    if has_image:
        metadata_filters["image_capable"] = True

    if helper_mode:
        min_score = 0.01
    elif has_action_intent or has_at or has_image:
        min_score = 0.02
    else:
        min_score = 0.05

    return {
        "fetch_k": _ROUTE_VECTOR_FETCH_K,
        "min_score": min_score,
        "max_k": _ROUTE_VECTOR_MAX_K,
        "k_increment": _ROUTE_VECTOR_INCREMENT,
        "metadata_filters": metadata_filters or None,
        "rerank": True,
    }


def _find_global_help_plugin(
    knowledge_base: PluginKnowledgeBase,
) -> PluginInfo | None:
    for plugin in knowledge_base.plugins:
        heads = _collect_allowed_heads(plugin)
        if "功能" in heads or "帮助" in heads or "help" in heads:
            return plugin
    return None


def _resolve_command_schema(plugin: PluginInfo, command_head: str):
    normalized_head = normalize_message_text(command_head).casefold()
    if not normalized_head:
        return None
    for meta in plugin.command_meta:
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


def _build_route_prompt(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    preferred_modules: list[str] | None = None,
    vector_context: str = "",
    include_helpers: bool = True,
    namespace_limit: int = _ROUTE_NAMESPACE_LIMIT,
) -> str:
    skills_json = render_skill_namespace(
        knowledge_base,
        query=message_text,
        limit=max(namespace_limit, 1),
        preferred_modules=preferred_modules or (),
        include_helpers=include_helpers,
        mask_module=True,
    )
    rag_section = ""
    context_text = normalize_message_text(vector_context)
    if context_text:
        rag_section = f"向量召回参考:\n{context_text}\n\n"
    return (
        f"用户消息:\n{message_text}\n\n"
        f"{rag_section}"
        "可用技能命名空间 JSON:\n"
        f"{skills_json}\n\n"
        "请基于上述命名空间选择路由。"
    )


def _estimate_route_prompt_tokens(text: str) -> int:
    source = str(text or "")
    if not source:
        return 0
    token_hits = len(_ROUTE_PROMPT_TOKEN_PATTERN.findall(source))
    return max(1, int(token_hits * 0.9))


def _fit_route_prompt_budget(
    *,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    preferred_modules: list[str],
    vector_context: str,
    include_helpers: bool,
    namespace_limit: int,
    token_budget: int,
) -> tuple[str, int, str]:
    resolved_budget = max(int(token_budget or 0), 800)
    current_limit = max(
        int(namespace_limit or _ROUTE_NAMESPACE_LIMIT),
        _ROUTE_MIN_NAMESPACE_LIMIT,
    )
    context_text = vector_context

    prompt = _build_route_prompt(
        message_text,
        knowledge_base,
        preferred_modules=preferred_modules,
        vector_context=context_text,
        include_helpers=include_helpers,
        namespace_limit=current_limit,
    )
    while (
        _estimate_route_prompt_tokens(prompt) > resolved_budget
        and current_limit > _ROUTE_MIN_NAMESPACE_LIMIT
    ):
        current_limit = max(current_limit - 2, _ROUTE_MIN_NAMESPACE_LIMIT)
        prompt = _build_route_prompt(
            message_text,
            knowledge_base,
            preferred_modules=preferred_modules,
            vector_context=context_text,
            include_helpers=include_helpers,
            namespace_limit=current_limit,
        )

    if _estimate_route_prompt_tokens(prompt) > resolved_budget and context_text:
        context_text = ""
        prompt = _build_route_prompt(
            message_text,
            knowledge_base,
            preferred_modules=preferred_modules,
            vector_context=context_text,
            include_helpers=include_helpers,
            namespace_limit=current_limit,
        )

    if _estimate_route_prompt_tokens(prompt) > resolved_budget:
        clipped = max(int(len(prompt) * 0.75), 600)
        prompt = f"{prompt[:clipped].rstrip()}\n\n[truncated]"

    return prompt, current_limit, context_text


def _build_vector_context_from_candidates(candidates: list[PluginInfo]) -> str:
    if not candidates:
        return ""
    lines: list[str] = []
    for idx, plugin in enumerate(candidates[:5], start=1):
        commands = ", ".join(plugin.commands[:3]) if plugin.commands else "无命令"
        note = normalize_message_text(plugin.description or plugin.usage or "")
        if len(note) > _MAX_VECTOR_NOTE_LEN:
            note = f"{note[:_MAX_VECTOR_NOTE_LEN]}..."
        lines.append(
            f"{idx}. {plugin.name} | commands: {commands} | "
            f"note: {note or '暂无说明'}"
        )
    return "\n".join(lines)


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


def _slice_knowledge_base(
    knowledge_base: PluginKnowledgeBase,
    plugins: list[PluginInfo],
) -> PluginKnowledgeBase:
    return PluginKnowledgeBase(
        plugins=plugins,
        user_role=knowledge_base.user_role,
    )


def _build_candidate_attempt_sizes(
    *,
    candidate_count: int,
    initial_limit: int,
    max_limit: int,
    expand_step: int,
    deferred_enabled: bool,
) -> list[int]:
    if candidate_count <= 0:
        return []
    first = min(max(initial_limit, _ROUTE_MIN_NAMESPACE_LIMIT), candidate_count)
    sizes = [first]
    if deferred_enabled:
        ceiling = min(max(max_limit, first), candidate_count)
        current = first
        while current < ceiling:
            current = min(current + max(expand_step, 1), ceiling)
            if current not in sizes:
                sizes.append(current)
    if sizes[-1] < candidate_count:
        sizes.append(candidate_count)
    return sizes


async def _request_llm_route_decision(
    *,
    prompt: str,
    timeout_value: float | None,
) -> tuple[_LLMRouteDecision | None, bool]:
    try:
        decision = await generate_structured(
            prompt,
            _LLMRouteDecision,
            model=get_config_value("INTENT_MODEL", None),
            instruction=_LLM_ROUTE_INSTRUCTION,
            timeout=timeout_value,
        )
        return decision, False
    except Exception as exc:
        if _is_schema_parameter_error(exc):
            logger.debug(
                f"ChatInter LLM 严格路由参数错误，跳过宽松解析并回退规则路由: {exc}"
            )
            return None, True
        logger.debug(f"ChatInter LLM 严格路由失败，尝试宽松解析: {exc}")
        try:
            relaxed_instruction = (
                f"{_LLM_ROUTE_INSTRUCTION}\n\n{_LLM_ROUTE_RELAXED_SUFFIX}"
            )
            relaxed_response = await chat(
                prompt,
                model=get_config_value("INTENT_MODEL", None),
                instruction=relaxed_instruction,
                timeout=timeout_value,
            )
            decision = _parse_llm_route_decision_loose(
                relaxed_response.text if relaxed_response else ""
            )
            if decision is None:
                logger.debug("ChatInter LLM 宽松路由解析失败")
            else:
                logger.debug("ChatInter LLM 宽松路由解析成功")
            return decision, False
        except Exception as relaxed_exc:
            logger.debug(f"ChatInter LLM 宽松路由失败: {relaxed_exc}")
            return None, False


async def resolve_llm_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> tuple[RouteResolveResult | None, bool]:
    normalized_message = normalize_message_text(message_text)
    if not normalized_message or not knowledge_base.plugins:
        return None, True

    timeout = get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout_value = float(timeout) if timeout else None
    except (TypeError, ValueError):
        timeout_value = None

    helper_mode = is_usage_question(normalized_message)
    vector_options = _resolve_vector_retrieve_options(
        normalized_message,
        helper_mode=helper_mode,
    )
    namespace_limit = _resolve_namespace_limit(
        normalized_message,
        helper_mode=helper_mode,
    )
    route_prompt_budget = int(
        get_config_value("ROUTE_PROMPT_TOKEN_BUDGET", 2600) or 2600
    )
    deferred_namespace_enabled = bool(
        get_config_value("ROUTE_DEFERRED_NAMESPACE_ENABLED", True)
    )
    initial_candidate_limit = max(
        int(
            get_config_value(
                "ROUTE_CANDIDATE_INITIAL_LIMIT",
                _ROUTE_CANDIDATE_INITIAL_LIMIT,
            )
            or _ROUTE_CANDIDATE_INITIAL_LIMIT
        ),
        _ROUTE_MIN_NAMESPACE_LIMIT,
    )
    max_candidate_limit = max(
        int(
            get_config_value(
                "ROUTE_CANDIDATE_MAX_LIMIT",
                _ROUTE_CANDIDATE_MAX_LIMIT,
            )
            or _ROUTE_CANDIDATE_MAX_LIMIT
        ),
        initial_candidate_limit,
    )
    candidate_expand_step = max(
        int(
            get_config_value(
                "ROUTE_CANDIDATE_EXPAND_STEP",
                _ROUTE_CANDIDATE_EXPAND_STEP,
            )
            or _ROUTE_CANDIDATE_EXPAND_STEP
        ),
        1,
    )
    candidate_min_score = max(
        float(
            get_config_value(
                "ROUTE_CANDIDATE_MIN_SCORE",
                _ROUTE_CANDIDATE_MIN_SCORE,
            )
            or _ROUTE_CANDIDATE_MIN_SCORE
        ),
        0.0,
    )

    preferred_modules: list[str] = []
    vector_context = ""
    ranked_candidates: list[PluginInfo] = []
    try:
        vector_candidates = await PluginRAGService.retrieve(
            query=normalized_message,
            knowledge=knowledge_base,
            top_k=_ROUTE_VECTOR_TOP_K,
            context_text=normalized_message,
            preferred_modules=preferred_modules,
            fetch_k=int(vector_options["fetch_k"]),
            min_score=float(vector_options["min_score"]),
            max_k=int(vector_options["max_k"]),
            k_increment=int(vector_options["k_increment"]),
            metadata_filters=vector_options.get("metadata_filters"),
            rerank=bool(vector_options.get("rerank", True)),
        )
        preferred_modules = [
            plugin.module
            for plugin in vector_candidates
            if normalize_message_text(plugin.module)
        ]
        vector_context = _build_vector_context_from_candidates(vector_candidates)
    except Exception as exc:
        logger.debug(f"ChatInter 路由向量召回失败，降级为词法命名空间: {exc}")

    metadata_filters = vector_options.get("metadata_filters")
    if not isinstance(metadata_filters, dict):
        metadata_filters = None
    ranked_candidates = rank_route_candidates(
        knowledge_base,
        normalized_message,
        preferred_modules=preferred_modules,
        metadata_filters=metadata_filters,
        min_score=candidate_min_score,
    )
    if not ranked_candidates and preferred_modules:
        preferred_set = {
            normalize_message_text(module)
            for module in preferred_modules
            if normalize_message_text(module)
        }
        ranked_candidates = [
            plugin
            for plugin in knowledge_base.plugins
            if normalize_message_text(plugin.module) in preferred_set
        ]

    if helper_mode:
        help_plugin = _find_global_help_plugin(knowledge_base)
        if help_plugin:
            ranked_candidates = [
                plugin
                for plugin in ranked_candidates
                if plugin.module != help_plugin.module
            ]
            ranked_candidates.insert(0, help_plugin)

    if not ranked_candidates:
        logger.debug("ChatInter 路由候选检索未命中，跳过插件路由")
        return None, False

    candidate_ceiling = min(max_candidate_limit, len(ranked_candidates))
    attempt_sizes = _build_candidate_attempt_sizes(
        candidate_count=candidate_ceiling,
        initial_limit=initial_candidate_limit,
        max_limit=max_candidate_limit,
        expand_step=candidate_expand_step,
        deferred_enabled=deferred_namespace_enabled,
    )
    if not attempt_sizes:
        return None, True

    last_decision: _LLMRouteDecision | None = None
    for attempt_idx, candidate_size in enumerate(attempt_sizes, start=1):
        attempt_plugins = ranked_candidates[:candidate_size]
        attempt_kb = _slice_knowledge_base(knowledge_base, attempt_plugins)
        attempt_pref_modules = [
            plugin.module
            for plugin in attempt_plugins
            if normalize_message_text(plugin.module)
        ]
        prompt, _, _ = _fit_route_prompt_budget(
            message_text=normalized_message,
            knowledge_base=attempt_kb,
            preferred_modules=attempt_pref_modules,
            vector_context=vector_context,
            include_helpers=helper_mode,
            namespace_limit=min(namespace_limit, candidate_size),
            token_budget=route_prompt_budget,
        )
        logger.debug(
            "ChatInter LLM 路由尝试: "
            f"attempt={attempt_idx}/{len(attempt_sizes)} "
            f"candidates={candidate_size}"
        )
        decision, schema_error = await _request_llm_route_decision(
            prompt=prompt,
            timeout_value=timeout_value,
        )
        if schema_error:
            return None, True
        if decision is None:
            continue

        last_decision = decision
        route_result = _to_route_result(
            decision=decision,
            message_text=normalized_message,
            knowledge_base=attempt_kb,
        )
        if route_result is not None:
            return route_result, False
        if decision.action == "route":
            logger.debug(
                "ChatInter LLM 路由结果未通过校验: "
                f"module={decision.plugin_module}, command={decision.command}"
            )
            continue
        if decision.action == "skip" and attempt_idx < len(attempt_sizes):
            continue
        if decision.action == "skip":
            return None, False

    if last_decision and last_decision.action == "skip":
        return None, False
    return None, True


def resolve_pre_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> RouteResolveResult | None:
    return _resolve_stage(
        message_text=message_text,
        knowledge_base=knowledge_base,
        include_usage=False,
        include_similarity=False,
        stage="pre",
    )


def resolve_semantic_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> RouteResolveResult | None:
    return _resolve_stage(
        message_text=message_text,
        knowledge_base=knowledge_base,
        include_usage=True,
        include_similarity=True,
        stage="semantic",
    )


__all__ = [
    "RouteResolveResult",
    "resolve_llm_route",
    "resolve_pre_route",
    "resolve_semantic_route",
]
