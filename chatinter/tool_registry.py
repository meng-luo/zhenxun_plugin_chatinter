import asyncio
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import re
import time
from typing import Any, ClassVar

import httpx

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable
from zhenxun.services.log import logger

from .agent_tools import get_chatinter_tools
from .config import get_mcp_endpoints

_MCP_DISCOVER_PATH = "/v1/tools"
_MCP_CALL_PATH = "/v1/tools/{name}/call"
_TOOL_CACHE_TTL = 30
_DEFINITION_CACHE_TTL = 300
_QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)
_STOPWORDS = {
    "工具",
    "调用",
    "执行",
    "一下",
    "请",
    "帮我",
    "需要",
    "功能",
    "命令",
}
_MAX_SELECTED_TOOLS = 12


@dataclass(frozen=True)
class ToolSelectionContext:
    query: str = ""
    context_text: str = ""
    session_id: str | None = None
    user_id: str | None = None
    group_id: str | None = None
    is_superuser: bool = False


@dataclass(frozen=True)
class ToolPolicy:
    aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    min_score: float = 0.0
    selector: Callable[[ToolSelectionContext], bool] | None = None
    authorization: Callable[[ToolSelectionContext], bool] | None = None
    enabled_by_default: bool = True
    concurrency_safe: bool = False


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords if keyword)


_TOOL_POLICIES: dict[str, ToolPolicy] = {
    "chatinter_lookup_plugin": ToolPolicy(
        aliases=("插件检索", "功能检索", "命令检索", "查插件"),
        examples=(
            "查一下点歌命令",
            "有哪些表情命令",
            "查帮助命令",
        ),
        min_score=0.1,
        concurrency_safe=True,
        selector=lambda ctx: _contains_any(
            f"{ctx.query} {ctx.context_text}",
            ("插件", "命令", "功能", "帮助", "怎么用", "usage"),
        ),
    ),
    "chatinter_safe_eval": ToolPolicy(
        aliases=("数学计算", "表达式计算", "计算器"),
        examples=("1+1", "sqrt(9)", "sin(pi/2)"),
        min_score=0.2,
        concurrency_safe=True,
        selector=lambda ctx: _contains_any(
            f"{ctx.query} {ctx.context_text}",
            ("计算", "表达式", "算一下", "数学", "eval"),
        ),
    ),
    "chatinter_safe_shell": ToolPolicy(
        aliases=("终端命令", "shell", "命令行"),
        examples=("echo hello", "pwd", "ls"),
        min_score=0.3,
        selector=lambda ctx: _contains_any(
            f"{ctx.query} {ctx.context_text}",
            ("shell", "命令行", "终端", "bash", "执行命令", "cmd"),
        ),
        authorization=lambda ctx: bool(ctx.is_superuser),
    ),
}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = [token.lower() for token in _QUERY_TOKEN_PATTERN.findall(text)]
    return [token for token in tokens if token and token not in _STOPWORDS]


def _score_tool(
    *,
    query_tokens: set[str],
    context_tokens: set[str],
    tool_name: str,
    tool_desc: str,
    aliases: tuple[str, ...] = (),
    examples: tuple[str, ...] = (),
) -> float:
    if not query_tokens and not context_tokens:
        return 0.0
    haystack_name = tool_name.lower()
    haystack_desc = tool_desc.lower()
    score = 0.0

    for token in query_tokens:
        if token in haystack_name:
            score += 2.0
        if token in haystack_desc:
            score += 1.25

    for token in context_tokens:
        if token in haystack_name:
            score += 1.0
        if token in haystack_desc:
            score += 0.5

    for alias in aliases:
        alias_l = alias.lower()
        if not alias_l:
            continue
        for token in query_tokens:
            if token in alias_l:
                score += 1.4
        for token in context_tokens:
            if token in alias_l:
                score += 0.7

    for example in examples:
        example_l = example.lower()
        if not example_l:
            continue
        for token in query_tokens:
            if token in example_l:
                score += 0.8

    return score


def _standardize_tool_definition(
    tool_name: str,
    definition: ToolDefinition,
    policy: ToolPolicy,
) -> ToolDefinition:
    description_parts: list[str] = []
    description = str(definition.description or "").strip()
    if description:
        description_parts.append(description)
    if policy.aliases:
        description_parts.append("别名: " + ", ".join(policy.aliases))
    if policy.examples:
        description_parts.append("示例: " + " | ".join(policy.examples[:3]))

    parameters = definition.parameters or {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    else:
        parameters = dict(parameters)
    if parameters.get("type") != "object":
        parameters["type"] = "object"
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    parameters["properties"] = properties
    parameters.setdefault("additionalProperties", False)

    if properties:
        hints: list[str] = []
        for arg_name, arg_schema_obj in properties.items():
            if not isinstance(arg_schema_obj, dict):
                continue
            arg_type = str(arg_schema_obj.get("type", "")).strip()
            arg_desc = str(arg_schema_obj.get("description", "")).strip()
            if arg_type and arg_desc:
                hints.append(f"{arg_name}({arg_type}):{arg_desc}")
            elif arg_type:
                hints.append(f"{arg_name}({arg_type})")
            elif arg_desc:
                hints.append(f"{arg_name}:{arg_desc}")
            if len(hints) >= 6:
                break
        if hints:
            description_parts.append("参数: " + "; ".join(hints))

    standardized_description = "\n".join(
        part for part in description_parts if part
    ).strip()
    if not standardized_description:
        standardized_description = tool_name

    return ToolDefinition(
        name=definition.name or tool_name,
        description=standardized_description,
        parameters=parameters,
    )


class MCPRemoteTool(ToolExecutable):
    def __init__(
        self,
        *,
        endpoint: str,
        name: str,
        description: str,
        parameters: dict[str, Any],
        timeout: float = 8.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.name = name
        self.description = description
        self.parameters = parameters
        self.timeout = timeout

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters or {"type": "object", "properties": {}},
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        payload = {"arguments": kwargs}
        if context and getattr(context, "session_id", None):
            payload["session_id"] = context.session_id

        call_url = f"{self.endpoint}{_MCP_CALL_PATH.format(name=self.name)}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(call_url, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return ToolResult(
                output={
                    "error_type": "ExecutionError",
                    "message": f"MCP tool call failed: {exc}",
                    "is_retryable": True,
                },
                display_content=f"{self.name} 调用失败",
            )

        if isinstance(data, dict) and "result" in data:
            output = data["result"]
        else:
            output = data
        return ToolResult(output=output, display_content=str(output)[:200])


def _parse_mcp_endpoints() -> list[str]:
    return get_mcp_endpoints()


async def _discover_remote_tools(endpoint: str) -> dict[str, ToolExecutable]:
    discover_url = f"{endpoint}{_MCP_DISCOVER_PATH}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(discover_url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.warning(f"MCP discover failed: endpoint={endpoint}, error={exc}")
        return {}

    raw_tools = payload.get("tools", []) if isinstance(payload, dict) else []
    tools: dict[str, ToolExecutable] = {}
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        description = str(item.get("description", "")).strip() or f"MCP Tool: {name}"
        parameters = item.get("parameters", {})
        tools[name] = MCPRemoteTool(
            endpoint=endpoint,
            name=name,
            description=description,
            parameters=parameters if isinstance(parameters, dict) else {},
        )
    return tools


class ChatInterToolRegistry:
    _cache_tools: ClassVar[dict[str, ToolExecutable]] = {}
    _cache_deadline: ClassVar[float] = 0.0
    _definition_cache: ClassVar[dict[str, tuple[ToolDefinition, float]]] = {}
    _session_tool_overrides: ClassVar[dict[str, dict[str, bool]]] = {}
    _group_tool_overrides: ClassVar[dict[str, dict[str, bool]]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    def _get_policy(cls, tool_name: str) -> ToolPolicy:
        return _TOOL_POLICIES.get(tool_name, ToolPolicy())

    @classmethod
    def is_concurrency_safe(cls, tool_name: str) -> bool:
        policy = cls._get_policy(tool_name)
        return bool(policy.concurrency_safe)

    @classmethod
    def _is_authorized(
        cls,
        tool_name: str,
        selection_context: ToolSelectionContext | None,
    ) -> bool:
        policy = cls._get_policy(tool_name)
        if policy.authorization is None:
            return True
        if selection_context is None:
            return False
        try:
            return bool(policy.authorization(selection_context))
        except Exception as exc:
            logger.debug(f"tool authorization failed: {tool_name}, error={exc}")
            return False

    @classmethod
    def _is_selected(
        cls,
        tool_name: str,
        selection_context: ToolSelectionContext | None,
    ) -> bool:
        policy = cls._get_policy(tool_name)
        if policy.selector is None or selection_context is None:
            return True
        try:
            return bool(policy.selector(selection_context))
        except Exception as exc:
            logger.debug(f"tool selector failed: {tool_name}, error={exc}")
            return True

    @classmethod
    def _is_enabled(
        cls,
        tool_name: str,
        selection_context: ToolSelectionContext | None,
    ) -> bool:
        policy = cls._get_policy(tool_name)
        enabled = policy.enabled_by_default
        if selection_context is None:
            return enabled
        session_id = str(selection_context.session_id or "").strip()
        group_id = str(selection_context.group_id or "").strip()
        if session_id:
            session_override = cls._session_tool_overrides.get(session_id, {})
            if tool_name in session_override:
                return session_override[tool_name]
        if group_id:
            group_override = cls._group_tool_overrides.get(group_id, {})
            if tool_name in group_override:
                return group_override[tool_name]
        return enabled

    @classmethod
    async def set_tool_enabled(
        cls,
        *,
        tool_name: str,
        enabled: bool,
        session_id: str | None = None,
        group_id: str | None = None,
    ) -> None:
        if not session_id and not group_id:
            return
        async with cls._lock:
            if session_id:
                session_key = str(session_id).strip()
                if session_key:
                    overrides = cls._session_tool_overrides.setdefault(session_key, {})
                    overrides[tool_name] = enabled
            if group_id:
                group_key = str(group_id).strip()
                if group_key:
                    overrides = cls._group_tool_overrides.setdefault(group_key, {})
                    overrides[tool_name] = enabled

    @classmethod
    async def reset_dynamic_overrides(
        cls,
        *,
        session_id: str | None = None,
        group_id: str | None = None,
    ) -> None:
        async with cls._lock:
            if session_id:
                cls._session_tool_overrides.pop(str(session_id).strip(), None)
            if group_id:
                cls._group_tool_overrides.pop(str(group_id).strip(), None)

    @classmethod
    def _apply_policy_filters(
        cls,
        tools: dict[str, ToolExecutable],
        *,
        selection_context: ToolSelectionContext | None = None,
    ) -> dict[str, ToolExecutable]:
        if not tools:
            return {}
        selected: dict[str, ToolExecutable] = {}
        for name, executable in tools.items():
            if not cls._is_enabled(name, selection_context):
                continue
            if not cls._is_authorized(name, selection_context):
                continue
            if not cls._is_selected(name, selection_context):
                continue
            selected[name] = executable
        return selected

    @classmethod
    async def _build_tool_map(cls) -> dict[str, ToolExecutable]:
        local_tools = await get_chatinter_tools()
        all_tools: dict[str, ToolExecutable] = dict(local_tools)

        endpoints = _parse_mcp_endpoints()
        if endpoints:
            discovered = await asyncio.gather(
                *(_discover_remote_tools(endpoint) for endpoint in endpoints),
                return_exceptions=True,
            )
            for result in discovered:
                if isinstance(result, dict):
                    for name, executable in result.items():
                        if name in all_tools:
                            logger.warning(
                                f"Tool name conflict: {name}, keep local implementation"
                            )
                            continue
                        all_tools[name] = executable
        return all_tools

    @classmethod
    async def get_tools(
        cls,
        preferred_names: set[str] | None = None,
        selection_context: ToolSelectionContext | None = None,
    ) -> dict[str, ToolExecutable]:
        now = time.monotonic()
        async with cls._lock:
            if now >= cls._cache_deadline or not cls._cache_tools:
                cls._cache_tools = await cls._build_tool_map()
                cls._cache_deadline = now + _TOOL_CACHE_TTL

            selected = dict(cls._cache_tools)
            if preferred_names:
                selected = {
                    name: executable
                    for name, executable in selected.items()
                    if name in preferred_names
                } or dict(cls._cache_tools)
        return cls._apply_policy_filters(
            selected,
            selection_context=selection_context,
        )

    @classmethod
    async def _get_tool_definition_cached(
        cls,
        tool_name: str,
        executable: ToolExecutable,
    ) -> ToolDefinition | None:
        now = time.monotonic()
        cached = cls._definition_cache.get(tool_name)
        if cached and cached[1] > now:
            return cached[0]
        try:
            definition = await executable.get_definition()
        except Exception as exc:
            logger.debug(f"tool definition resolve failed: {tool_name}, error={exc}")
            return None
        policy = cls._get_policy(tool_name)
        definition = _standardize_tool_definition(tool_name, definition, policy)
        cls._definition_cache[tool_name] = (definition, now + _DEFINITION_CACHE_TTL)
        return definition

    @classmethod
    async def get_tools_for_query(
        cls,
        *,
        query: str,
        context_text: str = "",
        preferred_names: set[str] | None = None,
        max_tools: int = _MAX_SELECTED_TOOLS,
        allow_fallback: bool = True,
        selection_context: ToolSelectionContext | None = None,
    ) -> dict[str, ToolExecutable]:
        if selection_context is None:
            selection_context = ToolSelectionContext(
                query=query,
                context_text=context_text,
            )
        else:
            selection_context = ToolSelectionContext(
                query=query,
                context_text=context_text,
                session_id=selection_context.session_id,
                user_id=selection_context.user_id,
                group_id=selection_context.group_id,
                is_superuser=selection_context.is_superuser,
            )
        base_tools = await cls.get_tools(
            preferred_names=preferred_names,
            selection_context=selection_context,
        )
        if not base_tools:
            return {}

        query_tokens = set(_tokenize(query))
        context_token_counter = Counter(_tokenize(context_text))
        context_tokens = {
            token for token, _ in context_token_counter.most_common(24) if token
        }

        if not query_tokens and not context_tokens:
            if allow_fallback:
                return dict(list(base_tools.items())[:max(max_tools, 1)])
            return {}

        scored: list[tuple[str, float]] = []
        for name, executable in base_tools.items():
            definition = await cls._get_tool_definition_cached(name, executable)
            description = (
                definition.description
                if definition and isinstance(definition.description, str)
                else ""
            )
            policy = cls._get_policy(name)
            score = _score_tool(
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                tool_name=name,
                tool_desc=description,
                aliases=policy.aliases,
                examples=policy.examples,
            )
            if score > 0 and score >= policy.min_score:
                scored.append((name, score))

        if not scored:
            if allow_fallback:
                return dict(list(base_tools.items())[:max(max_tools, 1)])
            return {}

        scored.sort(key=lambda item: item[1], reverse=True)
        selected_names = {name for name, _ in scored[: max(max_tools, 1)]}
        return {
            name: executable
            for name, executable in base_tools.items()
            if name in selected_names
        }

    @classmethod
    async def preload(cls) -> None:
        await cls.get_tools()
