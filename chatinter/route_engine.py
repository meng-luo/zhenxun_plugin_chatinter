import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from zhenxun.services import chat, generate_structured, logger

from .config import get_config_value
from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .route_text import is_usage_question, normalize_message_text
from .skill_registry import (
    SkillRouteDecision,
    get_skill_registry,
    render_skill_namespace,
    skill_execute,
    skill_search,
)

_ROUTE_NAMESPACE_LIMIT = 20
_LLM_ROUTE_INSTRUCTION = """
你是 ChatInter 的技能路由器，只负责在给定技能命名空间内选择可执行命令。
必须严格遵守：
1. 只能从 skills_json 中选择 plugin_module 与命令头。
2. command 必须是一条可直接发送给插件的命令字符串。
3. 用户是执行诉求时优先 action_commands；仅在“怎么用/详情/帮助/参数”语义下才选 helper_commands。
4. 保留用户消息里的 [@123456]、[image] 占位符，不要改写成自然语言。
5. 如果无法确定，返回 action=skip。
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
) -> str:
    skills_json = render_skill_namespace(
        knowledge_base,
        query=message_text,
        limit=_ROUTE_NAMESPACE_LIMIT,
    )
    return (
        f"用户消息:\n{message_text}\n\n"
        "可用技能命名空间 JSON:\n"
        f"{skills_json}\n\n"
        "请基于上述命名空间选择路由。"
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

    prompt = _build_route_prompt(normalized_message, knowledge_base)
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
