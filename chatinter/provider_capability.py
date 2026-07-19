"""Provider/model capability adapter for ChatInter Agent requests.

This layer keeps ChatInter from assuming every model/provider accepts the same
tool count, tool_choice shape, multimodal input, or JSON schema dialect.
It is intentionally conservative: unknown OpenAI-compatible gateways get safe
limits instead of optimistic "send everything" behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
import copy
from dataclasses import dataclass
import re
from typing import Any, Literal

from zhenxun.services.ai.llm.system.capabilities import (
    ModelModality,
    get_model_capabilities,
)

from .config import COMMAND_TWO_STAGE_THRESHOLD, build_tool_generation_config
from .llm_compat import (
    LLMContentPart,
    LLMMessage,
    ToolDefinition,
    ToolExecutable,
    ToolResult,
)
from .native_command_tools import compact_command_tool_view
from .provider_protocol import (
    MCPToolProtocolProfile,
    ProviderProtocolProfile,
    SchemaDialect,
    ToolResultMessageFormat,
    adapt_tool_choice_for_policy,
    adapt_tool_result_payload_for_protocol,
    load_provider_protocol_profile,
    sanitize_external_tool_name_for_protocol,
)
from .route_text import normalize_message_text

ProviderFamily = Literal["openai", "gemini", "anthropic", "custom"]
ToolSchemaMode = Literal["full", "compact", "light"]

_AUTO_FULL_SCHEMA_TOOL_CAP = 8
_MAX_TOOL_DESCRIPTION_CHARS = 1800
_MAX_PARAM_DESCRIPTION_CHARS = 700
_DEFAULT_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "$schema",
        "$id",
        "examples",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)


@dataclass(frozen=True)
class ProviderCapabilityProfile:
    model_name: str
    family: ProviderFamily
    schema_dialect: SchemaDialect
    max_tools: int
    supports_tools: bool
    supports_image_input: bool
    supports_parallel_tool_calls: bool
    supports_required_tool_choice: bool
    supports_named_tool_choice: bool
    prefers_compact_command_schema: bool
    full_schema_tool_cap: int
    auto_command_tool_cap: int
    required_command_tool_cap: int
    tool_result_message_format: ToolResultMessageFormat
    mcp: MCPToolProtocolProfile
    protocol: ProviderProtocolProfile | None = None

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "family": self.family,
            "schema_dialect": self.schema_dialect,
            "max_tools": self.max_tools,
            "supports_tools": self.supports_tools,
            "supports_image_input": self.supports_image_input,
            "supports_parallel_tool_calls": self.supports_parallel_tool_calls,
            "supports_required_tool_choice": self.supports_required_tool_choice,
            "supports_named_tool_choice": self.supports_named_tool_choice,
            "prefers_compact_command_schema": self.prefers_compact_command_schema,
            "full_schema_tool_cap": self.full_schema_tool_cap,
            "auto_command_tool_cap": self.auto_command_tool_cap,
            "required_command_tool_cap": self.required_command_tool_cap,
            "tool_result_message_format": self.tool_result_message_format,
            "mcp": self.mcp.to_metadata(),
        }
        if self.protocol is not None:
            payload["protocol"] = self.protocol.to_metadata()
        return payload


@dataclass(frozen=True)
class ProviderToolSchemaPlan:
    """Schema exposure decision for one model request."""

    use_compact_schema: bool
    full_schema_names: frozenset[str]
    schema_modes: dict[str, ToolSchemaMode]
    reason: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "use_compact_schema": self.use_compact_schema,
            "full_schema_names": sorted(self.full_schema_names),
            "schema_modes": dict(self.schema_modes),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProviderPreparedRequest:
    """Final provider-safe request payload for the LLM service."""

    messages: list[LLMMessage]
    tools: dict[str, ToolExecutable] | None
    tool_choice: str | dict[str, Any] | None
    generation_config: Any | None
    metadata: dict[str, Any]


class ProviderAdjustedTool:
    """Tool view that sanitizes the definition for the target provider."""

    def __init__(
        self,
        *,
        executable: ToolExecutable,
        adapter: "ProviderCapabilityAdapter",
        schema_mode: ToolSchemaMode = "full",
    ) -> None:
        self.executable = executable
        self.adapter = adapter
        self.chatinter_schema_mode = schema_mode
        self.chatinter_provider_family = adapter.profile.family

    def __getattr__(self, name: str) -> Any:
        return getattr(self.executable, name)

    async def get_definition(self) -> ToolDefinition:
        definition = await self.executable.get_definition()
        if self.chatinter_schema_mode == "light":
            definition = _light_tool_definition(definition)
        return self.adapter.sanitize_tool_definition(definition)

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        return await self.executable.execute(context=context, **kwargs)


class ProviderCapabilityAdapter:
    """Adapts ChatInter request shape to model/provider capabilities."""

    def __init__(self, profile: ProviderCapabilityProfile) -> None:
        self.profile = profile

    @classmethod
    def for_model(cls, model_name: str | None) -> "ProviderCapabilityAdapter":
        name = normalize_message_text(str(model_name or "")) or "unknown"
        capabilities = get_model_capabilities(name)
        supports_tools = bool(capabilities.supports_tool_calling)
        supports_image = ModelModality.IMAGE in capabilities.input_modalities
        protocol = load_provider_protocol_profile(
            name,
            supports_tools=supports_tools,
            supports_image_input=supports_image,
        )
        profile = ProviderCapabilityProfile(
            model_name=name,
            family=protocol.family,
            schema_dialect=protocol.schema.dialect,
            max_tools=max(0, int(protocol.max_tools or 0)),
            supports_tools=protocol.tool_choice.supports_tools,
            supports_image_input=protocol.supports_image_input,
            supports_parallel_tool_calls=protocol.tool_choice.supports_parallel,
            supports_required_tool_choice=protocol.tool_choice.supports_required,
            supports_named_tool_choice=protocol.tool_choice.supports_named,
            prefers_compact_command_schema=protocol.schema_exposure.prefers_compact,
            full_schema_tool_cap=max(
                1,
                int(protocol.schema_exposure.full_schema_tool_cap or 1),
            ),
            auto_command_tool_cap=max(
                1,
                int(protocol.schema_exposure.auto_command_tool_cap or 1),
            ),
            required_command_tool_cap=max(
                1,
                int(protocol.schema_exposure.required_command_tool_cap or 1),
            ),
            tool_result_message_format=protocol.tool_result_message_format,
            mcp=protocol.mcp,
            protocol=protocol,
        )
        return cls(profile)

    @property
    def max_tools(self) -> int:
        return max(0, int(self.profile.max_tools or 0))

    def command_tool_capacity(self, *, reserved_tools: int = 0) -> int:
        if not self.profile.supports_tools:
            return 0
        return max(0, self.max_tools - max(int(reserved_tools or 0), 0))

    def command_exposure_cap(
        self,
        *,
        obligation: str,
        required_tool_count: int = 0,
        command_tool_capacity: int | None = None,
    ) -> int:
        """Return first-turn command schema exposure cap for this provider."""

        hard_cap = (
            self.command_tool_capacity()
            if command_tool_capacity is None
            else max(0, int(command_tool_capacity or 0))
        )
        if hard_cap <= 0:
            return 0
        policy_cap = (
            self.profile.required_command_tool_cap
            if obligation == "required"
            else self.profile.auto_command_tool_cap
        )
        if required_tool_count:
            policy_cap = max(policy_cap, int(required_tool_count) + 8)
        return max(1, min(hard_cap, policy_cap))

    def adapt_tool_choice(
        self,
        tool_choice: str | dict[str, Any] | None,
        *,
        has_tools: bool,
    ) -> str | dict[str, Any] | None:
        policy = (
            self.profile.protocol.tool_choice
            if self.profile.protocol is not None
            else None
        )
        if policy is not None:
            return adapt_tool_choice_for_policy(
                tool_choice,
                has_tools=has_tools,
                policy=policy,
            )
        return None if not has_tools or tool_choice in {None, "none"} else "auto"

    def limit_tool_map(
        self,
        tools: dict[str, ToolExecutable],
        *,
        required_tool_names: Iterable[str] = (),
    ) -> dict[str, ToolExecutable]:
        if not tools or not self.profile.supports_tools or self.max_tools <= 0:
            return {}
        if len(tools) <= self.max_tools:
            return dict(tools)

        required = {
            normalize_message_text(str(name or ""))
            for name in required_tool_names
            if normalize_message_text(str(name or ""))
        }
        selected: dict[str, ToolExecutable] = {}

        def add(name: str, tool: ToolExecutable) -> bool:
            if len(selected) >= self.max_tools:
                return False
            if name not in selected:
                selected[name] = tool
            return True

        for name, tool in tools.items():
            if normalize_message_text(name) in required:
                add(name, tool)
        for name, tool in tools.items():
            if not _is_command_tool(tool):
                add(name, tool)
        for name, tool in tools.items():
            if _is_command_tool(tool):
                add(name, tool)
            if len(selected) >= self.max_tools:
                break
        return selected

    def prepare_tool_map_for_request(
        self,
        tools: dict[str, ToolExecutable] | None,
        *,
        required_tool_names: Iterable[str] = (),
        schema_modes: dict[str, ToolSchemaMode] | None = None,
    ) -> dict[str, ToolExecutable] | None:
        if not tools or not self.profile.supports_tools:
            return None
        limited = self.limit_tool_map(tools, required_tool_names=required_tool_names)
        if not limited:
            return None
        schema_modes = schema_modes or {}
        return {
            name: ProviderAdjustedTool(
                executable=tool,
                adapter=self,
                schema_mode=schema_modes.get(name, "full"),
            )
            for name, tool in limited.items()
        }

    def prepare_chatinter_tools_for_request(
        self,
        tools: dict[str, ToolExecutable] | None,
        *,
        tool_choice: str | dict[str, Any] | None,
        required_tool_names: Iterable[str] = (),
        tool_obligation: str = "auto",
        has_command_observation: bool = False,
    ) -> dict[str, ToolExecutable] | None:
        """Prepare turn tools with provider-specific schema exposure policy."""

        if not tools or not self.profile.supports_tools:
            return None
        if tool_obligation == "none" and tool_choice is None:
            return None

        sorted_tools = self.sort_tool_map(
            tools,
            required_tool_names=required_tool_names,
        )
        schema_plan = self.command_schema_plan(
            sorted_tools,
            tool_choice=tool_choice,
            required_tool_names=required_tool_names,
            tool_obligation=tool_obligation,
            has_command_observation=has_command_observation,
        )
        request_tools = {
            name: compact_command_tool_view(tool)
            if _is_command_tool(tool)
            and schema_plan.schema_modes.get(name) == "compact"
            else tool
            for name, tool in sorted_tools.items()
        }
        return self.prepare_tool_map_for_request(
            request_tools,
            required_tool_names=required_tool_names,
            schema_modes=schema_plan.schema_modes,
        )

    def prepare_model_request(
        self,
        *,
        messages: list[LLMMessage],
        tools: dict[str, ToolExecutable] | None,
        tool_choice: str | dict[str, Any] | None,
        required_tool_names: Iterable[str] = (),
        schema_modes: dict[str, ToolSchemaMode] | None = None,
        generation_config: Any | None = None,
    ) -> ProviderPreparedRequest:
        """Build the provider-safe request shape consumed by AI.generate_internal."""

        request_tools: dict[str, ToolExecutable] | None
        if tools and all(
            str(getattr(tool, "chatinter_schema_mode", "") or "")
            in {"full", "compact", "light"}
            for tool in tools.values()
        ):
            request_tools = self.limit_tool_map(
                tools,
                required_tool_names=required_tool_names,
            )
        else:
            request_tools = self.prepare_tool_map_for_request(
                tools,
                required_tool_names=required_tool_names,
                schema_modes=schema_modes,
            )
        adapted_tool_choice = self.adapt_tool_choice(
            tool_choice,
            has_tools=bool(request_tools),
        )
        return ProviderPreparedRequest(
            messages=self.adapt_messages(messages),
            tools=request_tools,
            tool_choice=adapted_tool_choice,
            generation_config=build_tool_generation_config(
                tool_choice=adapted_tool_choice,
                base=generation_config,
            ),
            metadata={
                "provider": self.profile.to_metadata(),
                "tool_count": len(request_tools or {}),
                "tool_choice": adapted_tool_choice,
                "schema_modes": dict(schema_modes or {}),
            },
        )

    def sort_tool_map(
        self,
        tools: dict[str, ToolExecutable],
        *,
        required_tool_names: Iterable[str] = (),
    ) -> dict[str, ToolExecutable]:
        required = {
            normalize_message_text(str(name or ""))
            for name in required_tool_names
            if normalize_message_text(str(name or ""))
        }
        return {
            name: tools[name]
            for name in sorted(
                tools,
                key=lambda name: self._tool_sort_key(
                    name,
                    tools[name],
                    required_tool_names=required,
                ),
            )
        }

    def command_schema_plan(
        self,
        tools: dict[str, ToolExecutable],
        *,
        tool_choice: str | dict[str, Any] | None,
        required_tool_names: Iterable[str] = (),
        tool_obligation: str = "auto",
        has_command_observation: bool = False,
    ) -> ProviderToolSchemaPlan:
        required = {
            normalize_message_text(str(name or ""))
            for name in required_tool_names
            if normalize_message_text(str(name or ""))
        }
        command_names = [name for name, tool in tools.items() if _is_command_tool(tool)]
        if not command_names:
            return ProviderToolSchemaPlan(
                use_compact_schema=False,
                full_schema_names=frozenset(),
                schema_modes={name: "full" for name in tools},
                reason="no_command_tools",
            )
        if tool_choice == "required" or tool_obligation == "required":
            full = frozenset(command_names)
            return ProviderToolSchemaPlan(
                use_compact_schema=False,
                full_schema_names=full,
                schema_modes={name: "full" for name in tools},
                reason="required_tool_choice_full_schema",
            )
        if has_command_observation:
            return ProviderToolSchemaPlan(
                use_compact_schema=False,
                full_schema_names=frozenset(command_names),
                schema_modes={name: "full" for name in tools},
                reason="after_command_observation_full_schema",
            )

        two_stage = (
            len(command_names) > COMMAND_TWO_STAGE_THRESHOLD
            and tool_obligation != "required"
            and tool_choice != "required"
        )
        if two_stage:
            full_names = set(required)
        else:
            full_names = self._full_schema_tool_names(
                tools,
                required_tool_names=required,
            )
        use_compact = self._should_use_compact_command_schema(
            tools,
            full_schema_names=full_names,
            force=two_stage,
        )
        schema_modes: dict[str, ToolSchemaMode] = {}
        for name, tool in tools.items():
            if _is_command_tool(tool) and use_compact and name not in full_names:
                schema_modes[name] = "compact"
            else:
                schema_modes[name] = "full"
        if two_stage and use_compact:
            reason = "skills_like_two_stage_compact"
        elif use_compact:
            reason = "provider_compact_schema_policy"
        else:
            reason = "provider_full_schema_policy"
        return ProviderToolSchemaPlan(
            use_compact_schema=use_compact,
            full_schema_names=frozenset(full_names),
            schema_modes=schema_modes,
            reason=reason,
        )

    def adapt_messages(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        host_messages = list(messages)
        if self.profile.supports_image_input:
            return host_messages
        changed = False
        adapted: list[LLMMessage] = []
        for message in host_messages:
            if not isinstance(message.content, list):
                adapted.append(message)
                continue
            parts: list[LLMContentPart] = []
            for part in message.content:
                if part.type == "text":
                    parts.append(part)
                    continue
                changed = True
                parts.append(
                    LLMContentPart.text_part(
                        f"[{part.type or 'media'} omitted: current model does not "
                        "support this input modality]"
                    )
                )
            adapted.append(message.model_copy(update={"content": parts}))
        return adapted if changed else host_messages

    def sanitize_tool_definition(self, definition: ToolDefinition) -> ToolDefinition:
        protocol = self.profile.protocol
        schema_policy = protocol.schema if protocol is not None else None
        name = _sanitize_tool_name(definition.name, mcp=self.profile.mcp)
        description = _clip_text(
            definition.description,
            int(
                getattr(
                    schema_policy,
                    "max_tool_description_chars",
                    _MAX_TOOL_DESCRIPTION_CHARS,
                )
                or _MAX_TOOL_DESCRIPTION_CHARS
            ),
        )
        parameters = sanitize_json_schema(
            definition.parameters or {},
            dialect=self.profile.schema_dialect,
            protocol=protocol,
        )
        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )

    def should_use_compact_schema(self, *, tool_count: int) -> bool:
        return (
            self.profile.prefers_compact_command_schema
            or tool_count > self.profile.full_schema_tool_cap
        )

    def tool_calls_for_execution(self, tool_calls: list[Any]) -> list[Any]:
        if self.profile.supports_parallel_tool_calls:
            return list(tool_calls)
        if len(tool_calls) <= 1:
            return list(tool_calls)
        return list(tool_calls[:1])

    def parallel_tool_call_notice(
        self,
        *,
        original_count: int,
        executed_count: int,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "reason": "provider_parallel_tool_calls_disabled",
            "provider_family": self.profile.family,
            "original_tool_calls": int(original_count),
            "executed_this_step": int(executed_count),
            "instruction": (
                "The current provider is configured for sequential tool calls. "
                "Continue with remaining tasks after this observation."
            ),
        }

    def uses_compact_command_schema(
        self,
        *,
        request_tools: dict[str, ToolExecutable] | None,
        base_tool_map: dict[str, ToolExecutable],
        tool_calls: list[Any],
    ) -> bool:
        if not request_tools or not tool_calls:
            return False
        for tool_call in tool_calls:
            name = normalize_message_text(str(tool_call.function.name or ""))
            tool = request_tools.get(name)
            if is_compact_request_tool(tool):
                return True
            if str(getattr(tool, "chatinter_schema_mode", "") or "") == "full":
                continue
            if tool is not None and tool is not base_tool_map.get(name):
                return True
        return False

    def selected_command_tools(
        self,
        tools: dict[str, ToolExecutable],
        tool_calls: list[Any],
    ) -> dict[str, ToolExecutable]:
        selected: dict[str, ToolExecutable] = {}
        for tool_call in tool_calls:
            name = normalize_message_text(str(tool_call.function.name or ""))
            tool = tools.get(name)
            if tool is not None and _is_command_tool(tool):
                selected[name] = tool
        return self.sort_tool_map(selected)

    def compact_schema_upgrade_prompt(self) -> str:
        protocol = self.profile.protocol
        provider_hint = normalize_message_text(
            str(
                getattr(
                    getattr(protocol, "schema_exposure", None),
                    "compact_upgrade_prompt",
                    "",
                )
                or "Call the selected full-schema tool if it fits."
            )
        )
        return (
            "You selected compact plugin capability card(s). "
            "Now use the selected real command tool(s) with the full schema "
            "and fill arguments from the user's current task. "
            "If the selected tool is not actually appropriate, answer briefly "
            "instead of calling it. "
            f"{provider_hint}"
        )

    def tool_result_message(
        self,
        *,
        tool_call: Any,
        function_name: str,
        result: Any,
    ) -> LLMMessage:
        """Create provider-compatible tool result message.

        The lower LLM adapters still serialize LLMMessage into provider JSON,
        but runtime no longer needs to know whether that becomes OpenAI
        `tool`, Gemini `functionResponse`, or another protocol shape.
        """

        payload = self.tool_result_payload(result)
        return LLMMessage.tool_response(
            tool_call_id=str(getattr(tool_call, "id", "") or ""),
            function_name=function_name,
            result=payload,
        )

    def tool_result_payload(self, result: Any) -> Any:
        protocol = self.profile.protocol
        if protocol is None:
            return result
        return adapt_tool_result_payload_for_protocol(
            result,
            profile=protocol,
        )

    def sanitize_external_tool_name(
        self,
        name: str,
        *,
        namespace: str = "",
    ) -> str:
        """Normalize MCP/external tool names before registry insertion."""

        profile = self.profile.mcp
        return sanitize_external_tool_name_for_protocol(
            name,
            namespace=namespace,
            mcp=profile,
        )

    def mcp_metadata(self) -> dict[str, Any]:
        return self.profile.mcp.to_metadata()

    def _tool_sort_key(
        self,
        name: str,
        tool: ToolExecutable,
        *,
        required_tool_names: set[str],
    ) -> tuple[int, int, int, int, str]:
        binding = getattr(tool, "binding", None)
        normalized_name = normalize_message_text(name)
        if binding is None:
            return (0, 0, 0, 0, normalized_name)
        candidate = getattr(binding, "candidate", None)
        command_id = normalize_message_text(str(getattr(binding, "command_id", "")))
        selected = 1 if normalized_name in required_tool_names else 0
        exact = 1 if bool(getattr(candidate, "exact_protected", False)) else 0
        score = int(float(getattr(candidate, "score", 0.0) or 0.0) * 100)
        return (
            1,
            -selected,
            -exact,
            -score,
            command_id or normalized_name,
        )

    def _should_use_compact_command_schema(
        self,
        tools: dict[str, ToolExecutable],
        *,
        full_schema_names: set[str],
        force: bool = False,
    ) -> bool:
        command_tool_count = sum(1 for tool in tools.values() if _is_command_tool(tool))
        if command_tool_count <= len(full_schema_names):
            return False
        if force:
            return True
        if self.should_use_compact_schema(tool_count=command_tool_count):
            return command_tool_count > self.profile.full_schema_tool_cap
        return any(
            self._is_compact_schema_candidate(tool)
            for tool in tools.values()
            if _is_command_tool(tool)
        )

    def _full_schema_tool_names(
        self,
        tools: dict[str, ToolExecutable],
        *,
        required_tool_names: set[str],
    ) -> set[str]:
        selected: list[tuple[str, ToolExecutable]] = []
        for name, tool in tools.items():
            if not _is_command_tool(tool):
                continue
            if name in required_tool_names or self._is_full_schema_candidate(tool):
                selected.append((name, tool))
        cap = max(
            1,
            min(
                _AUTO_FULL_SCHEMA_TOOL_CAP,
                int(self.profile.full_schema_tool_cap or 1),
            ),
        )
        return {name for name, _tool in selected[:cap]}

    def _is_full_schema_candidate(self, tool: ToolExecutable) -> bool:
        binding = getattr(tool, "binding", None)
        candidate = getattr(binding, "candidate", None)
        if candidate is None:
            return False
        if bool(getattr(candidate, "exact_protected", False)):
            return True
        features = getattr(candidate, "features", None)
        exact_score = float(getattr(features, "exact_score", 0.0) or 0.0)
        schema_score = float(getattr(features, "schema_score", 0.0) or 0.0)
        context_score = float(getattr(features, "context_score", 0.0) or 0.0)
        reliability_score = float(getattr(features, "reliability_score", 0.0) or 0.0)
        param_failure_score = float(
            getattr(features, "param_failure_score", 0.0) or 0.0
        )
        score = float(getattr(candidate, "score", 0.0) or 0.0)
        if reliability_score >= 8.0 and param_failure_score >= -3.0 and score >= 80.0:
            return True
        high_reliability = _is_high_reliability_candidate(candidate)
        if high_reliability and (
            score >= 90.0 or exact_score > 0 or schema_score + context_score >= 12.0
        ):
            return True
        return (
            exact_score > 0
            or score >= 180.0
            or (score >= 120.0 and schema_score + context_score >= 8.0)
        )

    def _is_compact_schema_candidate(self, tool: ToolExecutable) -> bool:
        binding = getattr(tool, "binding", None)
        candidate = getattr(binding, "candidate", None)
        if candidate is None:
            return False
        if _is_low_reliability_candidate(candidate):
            return True
        return not self._is_full_schema_candidate(tool)


def is_compact_request_tool(tool: ToolExecutable | None) -> bool:
    return str(getattr(tool, "chatinter_schema_mode", "") or "") == "compact"


def is_light_request_tool(tool: ToolExecutable | None) -> bool:
    return str(getattr(tool, "chatinter_schema_mode", "") or "") == "light"


def _light_tool_definition(definition: ToolDefinition) -> ToolDefinition:
    parameters = {"type": "object", "properties": {}, "required": []}
    if hasattr(definition, "model_copy"):
        return definition.model_copy(update={"parameters": parameters})
    return ToolDefinition(
        name=str(getattr(definition, "name", "") or ""),
        description=str(getattr(definition, "description", "") or ""),
        parameters=parameters,
    )


def sanitize_json_schema(
    schema: dict[str, Any],
    *,
    dialect: SchemaDialect,
    protocol: ProviderProtocolProfile | None = None,
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}
    cleaned = _sanitize_schema_node(
        copy.deepcopy(schema),
        dialect=dialect,
        protocol=protocol,
    )
    if not isinstance(cleaned, dict):
        return {"type": "object", "properties": {}, "required": []}
    if cleaned.get("type") is None and "properties" in cleaned:
        cleaned["type"] = "object"
    if cleaned.get("type") == "object":
        cleaned.setdefault("properties", {})
        cleaned.setdefault("required", [])
    return cleaned


def _sanitize_schema_node(
    value: Any,
    *,
    dialect: SchemaDialect,
    protocol: ProviderProtocolProfile | None = None,
) -> Any:
    if isinstance(value, list):
        return [
            _sanitize_schema_node(item, dialect=dialect, protocol=protocol)
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    schema_policy = protocol.schema if protocol is not None else None
    unsupported_keys = set(
        getattr(schema_policy, "unsupported_keys", _DEFAULT_UNSUPPORTED_SCHEMA_KEYS)
        or _DEFAULT_UNSUPPORTED_SCHEMA_KEYS
    )
    strip_additional = bool(
        getattr(schema_policy, "strip_additional_properties", False)
    )
    strip_strict = bool(getattr(schema_policy, "strip_strict", False))
    compose_oneof_as_anyof = bool(
        getattr(schema_policy, "compose_oneof_as_anyof", False)
    )
    compose_allof_as_anyof = bool(
        getattr(schema_policy, "compose_allof_as_anyof", False)
    )
    param_description_limit = int(
        getattr(
            schema_policy,
            "max_param_description_chars",
            _MAX_PARAM_DESCRIPTION_CHARS,
        )
        or _MAX_PARAM_DESCRIPTION_CHARS
    )
    nullable_style = normalize_message_text(
        str(getattr(schema_policy, "nullable_style", "nullable") or "nullable")
    )

    result: dict[str, Any] = {}
    for key, raw in value.items():
        if key in unsupported_keys:
            continue
        if key == "additionalProperties" and strip_additional:
            continue
        if key == "strict" and strip_strict:
            continue
        if key == "description" and isinstance(raw, str):
            result[key] = _clip_text(raw, param_description_limit)
            continue
        if key in {"anyOf", "oneOf", "allOf"} and isinstance(raw, list):
            variants = [
                _sanitize_schema_node(item, dialect=dialect, protocol=protocol)
                for item in raw
            ]
            if (key == "oneOf" and compose_oneof_as_anyof) or (
                key == "allOf" and compose_allof_as_anyof
            ):
                if variants:
                    result["anyOf"] = variants
                continue
            result[key] = variants
            continue
        result[key] = _sanitize_schema_node(raw, dialect=dialect, protocol=protocol)

    schema_type = result.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        if len(non_null) == 1:
            result["type"] = non_null[0]
            if "null" in schema_type:
                if nullable_style == "type_union":
                    result["type"] = [non_null[0], "null"]
                elif nullable_style != "drop":
                    result.setdefault("nullable", True)
    if result.get("type") == "object":
        properties = result.get("properties")
        if not isinstance(properties, dict):
            result["properties"] = {}
        required = result.get("required")
        if not isinstance(required, list):
            result["required"] = []
    if result.get("type") == "array" and "items" not in result:
        result["items"] = {"type": "string"}
    return result


def _sanitize_tool_name(value: str, *, mcp: MCPToolProtocolProfile) -> str:
    name = normalize_message_text(value)
    name = re.sub(mcp.name_pattern, "_", name).strip("_")
    if not name:
        return "chatinter_tool"
    return name[: max(int(mcp.max_name_length or 64), 1)]


def _clip_text(value: str, limit: int) -> str:
    text = normalize_message_text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _is_command_tool(tool: ToolExecutable) -> bool:
    return getattr(tool, "binding", None) is not None


def _is_high_reliability_candidate(candidate: Any) -> bool:
    features = getattr(candidate, "features", None)
    snapshot = getattr(candidate, "tool", None)
    reliability_score = float(getattr(features, "reliability_score", 0.0) or 0.0)
    false_trigger_score = float(getattr(features, "false_trigger_score", 0.0) or 0.0)
    reliability = float(getattr(snapshot, "reliability", 0.5) or 0.5)
    return reliability >= 0.72 or (
        reliability_score >= 8.0 and false_trigger_score >= -4.0
    )


def _is_low_reliability_candidate(candidate: Any) -> bool:
    features = getattr(candidate, "features", None)
    snapshot = getattr(candidate, "tool", None)
    reliability_score = float(getattr(features, "reliability_score", 0.0) or 0.0)
    false_trigger_score = float(getattr(features, "false_trigger_score", 0.0) or 0.0)
    param_failure_score = float(getattr(features, "param_failure_score", 0.0) or 0.0)
    reliability = float(getattr(snapshot, "reliability", 0.5) or 0.5)
    return reliability < 0.35 or (
        reliability_score < -8.0
        or false_trigger_score < -8.0
        or param_failure_score < -8.0
    )


__all__ = [
    "MCPToolProtocolProfile",
    "ProviderAdjustedTool",
    "ProviderCapabilityAdapter",
    "ProviderCapabilityProfile",
    "ProviderPreparedRequest",
    "ProviderToolSchemaPlan",
    "is_compact_request_tool",
    "is_light_request_tool",
    "sanitize_json_schema",
]
