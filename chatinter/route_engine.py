from dataclasses import dataclass
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from zhenxun.services import chat, generate_structured, logger

from .config import get_config_value
from .knowledge_rag import PluginRAGService
from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_placeholders,
    contains_any,
    is_usage_question,
    normalize_message_text,
)
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
_LLM_ROUTE_INSTRUCTION = """
你是 ChatInter 的技能路由器，只负责在给定技能命名空间内选择可执行命令。
必须严格遵守：
1. 只能从 skills_json 中选择 plugin_module 与命令头。
2. command 必须是一条可直接发送给插件的命令字符串。
3. 用户是执行诉求时优先 action_commands；
   仅在“怎么用/详情/帮助/参数”语义下才选 helper_commands。
4. 若 schema.text_max 为 0，不要附带额外文本参数；仅保留 [@...] 或 [image] 占位符。
5. 保留用户消息里的 [@123456]、[image] 占位符，不要改写成自然语言。
6. 如果无法确定，返回 action=skip。
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

    name_text = normalize_message_text(decision.plugin_name or "")
    if name_text:
        name_fold = name_text.casefold()
        for plugin in knowledge_base.plugins:
            if normalize_message_text(plugin.name).casefold() == name_fold:
                return plugin
    return None


def _collect_allowed_heads(plugin: PluginInfo) -> set[str]:
    allowed: set[str] = set()
    for command in plugin.commands:
        normalized = normalize_message_text(command)
        if normalized:
            allowed.add(normalized.casefold())
    for meta in plugin.command_meta:
        command_head = normalize_message_text(getattr(meta, "command", ""))
        if command_head:
            allowed.add(command_head.casefold())
        for alias in getattr(meta, "aliases", None) or []:
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
    if helper_mode:
        metadata_filters["has_helper"] = True
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


def _normalize_placeholder_tokens(command: str) -> tuple[list[str], list[str]]:
    at_tokens: list[str] = []
    image_tokens: list[str] = []
    for token in collect_placeholders(command):
        normalized = normalize_message_text(token)
        if not normalized:
            continue
        if _AT_INLINE_PATTERN.fullmatch(normalized):
            normalized = f"[{normalized}]"
        if _AT_PLACEHOLDER_PATTERN.fullmatch(normalized):
            if normalized not in at_tokens:
                at_tokens.append(normalized)
            continue
        is_image_token = _IMAGE_PLACEHOLDER_PATTERN.fullmatch(normalized) or (
            normalized.lower().startswith("[image")
        )
        if is_image_token:
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

    at_tokens, image_tokens = _normalize_placeholder_tokens(normalized)
    text_tokens: list[str] = []
    for token in parts[1:]:
        token_text = normalize_message_text(token)
        if not token_text:
            continue
        if token_text in at_tokens or token_text in image_tokens:
            continue
        text_tokens.append(token_text)

    allow_at = getattr(schema, "allow_at", None)
    if allow_at is False:
        at_tokens = []

    image_max = getattr(schema, "image_max", None)
    if image_max is not None:
        image_max = max(int(image_max), 0)
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
            f"{idx}. [{plugin.module}] {plugin.name} | commands: {commands} | "
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

    preferred_modules: list[str] = []
    vector_context = ""
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

    prompt = _build_route_prompt(
        normalized_message,
        knowledge_base,
        preferred_modules=preferred_modules,
        vector_context=vector_context,
        include_helpers=helper_mode,
        namespace_limit=namespace_limit,
    )
    decision: _LLMRouteDecision | None = None
    try:
        decision = await generate_structured(
            prompt,
            _LLMRouteDecision,
            model=get_config_value("INTENT_MODEL", None),
            instruction=_LLM_ROUTE_INSTRUCTION,
            timeout=timeout_value,
        )
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
                logger.debug("ChatInter LLM 宽松路由解析失败，回退规则路由")
                return None, True
            logger.debug("ChatInter LLM 宽松路由解析成功")
        except Exception as relaxed_exc:
            logger.debug(f"ChatInter LLM 宽松路由失败，回退规则路由: {relaxed_exc}")
            return None, True

    if decision is None:
        return None, True

    route_result = _to_route_result(
        decision=decision,
        message_text=normalized_message,
        knowledge_base=knowledge_base,
    )
    if route_result is None and decision.action == "route":
        logger.debug(
            "ChatInter LLM 路由结果未通过校验: "
            f"module={decision.plugin_module}, command={decision.command}"
        )
    if route_result is not None:
        return route_result, False
    # LLM 明确给出 skip 时，不再回退规则路由，避免误触发无关插件
    if decision.action == "skip":
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
