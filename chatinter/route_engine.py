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
    is_usage_question,
    match_command_head_or_sticky,
    normalize_action_phrases,
    normalize_message_text,
    strip_invoke_prefix,
)
from .schema_policy import resolve_command_target_policy
from .skill_registry import (
    SkillRouteDecision,
    get_skill_registry,
    render_skill_namespace,
    select_relevant_skills,
    skill_execute,
    skill_search,
)
from .turn_runtime import TurnBudgetController

_ROUTE_NAMESPACE_LIMIT = 12
_ROUTE_PLUGIN_SHORTLIST_LIMIT = 8
_ROUTE_TOOLS_PER_SKILL_LIMIT = 2
_LLM_ROUTE_INSTRUCTION = """
你是 ChatInter 的技能路由器，只负责在给定技能命名空间内选择可执行命令。
必须严格遵守：
1. 只能从 skills_json 里选 plugin_module 与 command。
2. command 必须能直接发送给插件，不要重复拼写 [@...]、[image] 或 reply。
3. 执行诉求优先 action_commands；“怎么用/用法/帮助/参数/说明”才看 helper_commands。
4. 若 schema.text_max 为 0，不要附带纯文本参数。
5. 保留用户消息里的 [@123456]、[image] 占位符，不要改写成自然语言。
6. 无法确定时返回 action=skip。
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
        "候选技能 JSON:\n"
        f"{skills_json}\n\n"
        "只基于上述候选选择路由。"
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


def _build_shortlist_knowledge_base(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    limit: int,
) -> tuple[PluginKnowledgeBase, list[str]]:
    shortlist_limit = max(int(limit), 1)
    registry = get_skill_registry(knowledge_base)
    selected_skills = select_relevant_skills(
        registry,
        message_text,
        limit=shortlist_limit,
    )
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
        json_block = _extract_first_json_object(raw_text)
        if not json_block:
            return {}
        try:
            parsed = json.loads(json_block)
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


async def _request_llm_route_decision(
    *,
    prompt: str,
    timeout_value: float | None,
    session_key: str,
    budget_controller: TurnBudgetController | None,
) -> tuple[_LLMRouteDecision | None, bool]:
    try:
        if budget_controller is not None and not budget_controller.allow_classifier(
            "route_json"
        ):
            return None, False
        started = time.perf_counter()
        guarded = guard_prompt_sections(
            session_key=session_key,
            stage="route_json",
            system_prompt=_LLM_ROUTE_INSTRUCTION,
            user_text=prompt,
            controller=budget_controller,
        )
        decision = await generate_structured(
            guarded.user_text,
            _LLMRouteDecision,
            model=get_model_name(),
            instruction=guarded.system_prompt,
            timeout=timeout_value,
        )
        if budget_controller is not None:
            budget_controller.record_classifier(
                "route_json",
                time.perf_counter() - started,
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
            guarded = guard_prompt_sections(
                session_key=session_key,
                stage="route_json_relaxed",
                system_prompt=relaxed_instruction,
                user_text=prompt,
                controller=budget_controller,
            )
            relaxed_response = await chat(
                guarded.user_text,
                model=get_model_name(),
                instruction=guarded.system_prompt,
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
    *,
    session_key: str | None = None,
    budget_controller: TurnBudgetController | None = None,
) -> tuple[RouteResolveResult | None, bool, RouteAttemptReport]:
    normalized_message = normalize_message_text(message_text)
    report = RouteAttemptReport(helper_mode=is_usage_question(normalized_message))
    if not normalized_message or not knowledge_base.plugins:
        report.finalize(reason="empty_input")
        return None, True, report

    timeout = get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout_value = float(timeout) if timeout else None
    except (TypeError, ValueError):
        timeout_value = None

    full_plugins = list(knowledge_base.plugins)
    report.candidate_total = len(full_plugins)
    shortlist_limit = max(
        int(get_config_value("ROUTE_PLUGIN_SHORTLIST_LIMIT", _ROUTE_PLUGIN_SHORTLIST_LIMIT) or _ROUTE_PLUGIN_SHORTLIST_LIMIT),
        1,
    )
    attempt_kb, shortlisted_modules = _build_shortlist_knowledge_base(
        normalized_message,
        knowledge_base,
        limit=shortlist_limit,
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
    logger.debug(
        "ChatInter 路由短名单: "
        f"total={len(full_plugins)} "
        f"shortlist={len(attempt_kb.plugins)} "
        f"modules={attempt_modules}"
    )
    report.note_attempt(attempt_modules)

    shortlist_route_result = _resolve_shortlist_local_route(
        normalized_message,
        attempt_kb,
    )
    if shortlist_route_result is not None:
        validated_shortlist_result = _validate_existing_route_result(
            shortlist_route_result,
            normalized_message,
            attempt_kb,
        )
        if validated_shortlist_result is not None:
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
                False,
                report,
            )

    prompt = _build_route_prompt(
        normalized_message,
        attempt_kb,
        preferred_modules=[],
        vector_context="",
        include_helpers=True,
        namespace_limit=max(len(attempt_kb.plugins), _ROUTE_NAMESPACE_LIMIT),
    )
    logger.debug(
        "ChatInter LLM 路由尝试: "
        f"attempt=1/1 "
        f"candidates={len(attempt_kb.plugins)} "
        f"modules={attempt_modules}"
    )
    tool_route_result, tool_count, tool_choice_count = await _request_tool_route_decision(
        message_text=normalized_message,
        knowledge_base=attempt_kb,
        timeout_value=timeout_value,
        session_key=session_key or "global",
        budget_controller=budget_controller,
    )
    if tool_count > 0:
        report.note_tool_pool(tool_count, tool_choice_count)
    if tool_route_result is not None:
        validated_tool_result = _validate_existing_route_result(
            tool_route_result,
            normalized_message,
            attempt_kb,
        )
        if validated_tool_result is not None:
            report.finalize(
                reason="tool_route",
                stage=validated_tool_result.stage,
                plugin_name=validated_tool_result.decision.plugin_name,
                plugin_module=validated_tool_result.decision.plugin_module,
                command=validated_tool_result.decision.command,
            )
            return (
                RouteResolveResult(
                    decision=validated_tool_result.decision,
                    stage=validated_tool_result.stage,
                    report=report,
                ),
                False,
                report,
            )
        logger.debug(
            "ChatInter tool 路由结果未通过命令校验: "
            f"module={tool_route_result.decision.plugin_module} "
            f"command={tool_route_result.decision.command}"
        )

    decision, schema_error = await _request_llm_route_decision(
        prompt=prompt,
        timeout_value=timeout_value,
        session_key=session_key or "global",
        budget_controller=budget_controller,
    )
    if schema_error:
        report.finalize(reason="schema_error")
        return None, True, report
    if decision is None:
        report.finalize(reason="llm_no_decision")
        return None, False, report

    route_result = _to_route_result(
        decision=decision,
        message_text=normalized_message,
        knowledge_base=attempt_kb,
    )
    if route_result is not None:
        report.finalize(
            reason="json_route",
            stage=route_result.stage,
            plugin_name=route_result.decision.plugin_name,
            plugin_module=route_result.decision.plugin_module,
            command=route_result.decision.command,
        )
        return (
            RouteResolveResult(
                decision=route_result.decision,
                stage=route_result.stage,
                report=report,
            ),
            False,
            report,
        )
    if decision.action == "route":
        logger.debug(
            "ChatInter LLM 路由结果未通过校验: "
            f"module={decision.plugin_module}, command={decision.command}"
        )
        report.finalize(reason="llm_invalid_route")
        return None, False, report

    report.finalize(reason="llm_skip")
    return None, False, report


__all__ = [
    "RouteAttemptReport",
    "RouteResolveResult",
    "resolve_llm_route",
]
