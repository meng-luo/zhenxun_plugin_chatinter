"""Provider and MCP protocol profiles for ChatInter.

This module is the policy/source-of-truth layer for provider differences.  The
runtime asks for one profile and applies it; it should not branch on provider
quirks such as schema dialect, tool_choice shape or MCP naming rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Literal

from .route_text import normalize_message_text

ProviderFamily = Literal["openai", "gemini", "anthropic", "custom"]
SchemaDialect = Literal["openai_strict", "gemini", "generic"]
ToolChoiceMode = Literal["none", "auto", "required", "named"]
ToolResultMessageFormat = Literal[
    "openai_tool",
    "gemini_function_response",
    "anthropic_tool_result",
    "generic",
]
SchemaExposureMode = Literal["full", "compact", "auto"]

@dataclass(frozen=True)
class ProviderSchemaPolicy:
    dialect: SchemaDialect = "generic"
    unsupported_keys: tuple[str, ...] = (
        "$schema",
        "$id",
        "examples",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
    )
    strip_additional_properties: bool = False
    strip_strict: bool = False
    nullable_style: str = "nullable"
    compose_oneof_as_anyof: bool = False
    compose_allof_as_anyof: bool = False
    max_tool_description_chars: int = 1800
    max_param_description_chars: int = 700

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderToolChoicePolicy:
    supports_tools: bool = True
    supports_required: bool = False
    supports_named: bool = False
    supports_parallel: bool = False
    unsupported_required_fallback: ToolChoiceMode = "auto"
    unsupported_named_fallback: ToolChoiceMode = "required"

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderSchemaExposurePolicy:
    prefers_compact: bool = False
    full_schema_tool_cap: int = 8
    auto_command_tool_cap: int = 32
    required_command_tool_cap: int = 24
    compact_upgrade_prompt: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MCPToolProtocolProfile:
    enabled: bool = True
    namespace_separator: str = "__"
    max_name_length: int = 64
    name_pattern: str = r"[^0-9A-Za-z_-]+"
    arguments_json_only: bool = True
    result_message_format: ToolResultMessageFormat = "generic"
    max_result_chars: int = 12000
    supports_streaming_observations: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderProtocolProfile:
    model_name: str
    family: ProviderFamily
    max_tools: int
    supports_image_input: bool
    schema: ProviderSchemaPolicy
    tool_choice: ProviderToolChoicePolicy
    schema_exposure: ProviderSchemaExposurePolicy
    tool_result_message_format: ToolResultMessageFormat
    mcp: MCPToolProtocolProfile = field(default_factory=MCPToolProtocolProfile)
    source: str = "defaults"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "family": self.family,
            "max_tools": self.max_tools,
            "supports_tools": self.tool_choice.supports_tools,
            "supports_image_input": self.supports_image_input,
            "supports_parallel_tool_calls": self.tool_choice.supports_parallel,
            "supports_required_tool_choice": self.tool_choice.supports_required,
            "supports_named_tool_choice": self.tool_choice.supports_named,
            "prefers_compact_command_schema": self.schema_exposure.prefers_compact,
            "full_schema_tool_cap": self.schema_exposure.full_schema_tool_cap,
            "auto_command_tool_cap": self.schema_exposure.auto_command_tool_cap,
            "required_command_tool_cap": self.schema_exposure.required_command_tool_cap,
            "schema_dialect": self.schema.dialect,
            "schema": self.schema.to_metadata(),
            "tool_choice": self.tool_choice.to_metadata(),
            "schema_exposure": self.schema_exposure.to_metadata(),
            "tool_result_message_format": self.tool_result_message_format,
            "mcp": self.mcp.to_metadata(),
            "source": self.source,
        }


def load_provider_protocol_profile(
    model_name: str | None,
    *,
    supports_tools: bool,
    supports_image_input: bool,
) -> ProviderProtocolProfile:
    name = normalize_message_text(str(model_name or "")) or "unknown"
    family = infer_provider_family(name)
    return _default_profile(
        model_name=name,
        family=family,
        supports_tools=supports_tools,
        supports_image_input=supports_image_input,
    )


def infer_provider_family(model_name: str) -> ProviderFamily:
    lowered = normalize_message_text(model_name).casefold()
    if "gemini" in lowered:
        return "gemini"
    if "claude" in lowered or "anthropic" in lowered:
        return "anthropic"
    if any(
        token in lowered
        for token in (
            "gpt",
            "o1",
            "o3",
            "o4",
            "openai",
            "chatgpt",
            "deepseek",
            "qwen",
            "glm",
            "doubao",
        )
    ):
        return "openai"
    return "custom"


def adapt_tool_choice_for_policy(
    tool_choice: str | dict[str, Any] | None,
    *,
    has_tools: bool,
    policy: ProviderToolChoicePolicy,
) -> str | dict[str, Any] | None:
    if not has_tools or not policy.supports_tools:
        return None
    if tool_choice is None or tool_choice == "none":
        return None
    if isinstance(tool_choice, dict):
        if policy.supports_named:
            return tool_choice
        return _fallback_tool_choice(policy.unsupported_named_fallback, policy)
    if tool_choice == "required" and not policy.supports_required:
        return _fallback_tool_choice(policy.unsupported_required_fallback, policy)
    if tool_choice in {"auto", "required"}:
        return tool_choice
    return "auto"


def adapt_tool_result_payload_for_protocol(
    result: Any,
    *,
    profile: ProviderProtocolProfile,
) -> Any:
    """Return provider-safe tool result payload.

    ChatInter keeps observations provider-neutral internally.  The final shape
    exposed to the model is selected from the external protocol profile.
    """

    payload = _compact_tool_result_value(result, max_chars=profile.mcp.max_result_chars)
    result_format = profile.tool_result_message_format
    if result_format == "gemini_function_response":
        return payload if isinstance(payload, dict) else {"result": payload}
    if result_format == "anthropic_tool_result":
        return payload if isinstance(payload, dict) else {"content": payload}
    return payload


def sanitize_external_tool_name_for_protocol(
    name: str,
    *,
    namespace: str,
    mcp: MCPToolProtocolProfile,
) -> str:
    raw = normalize_message_text(name)
    if namespace:
        raw = normalize_message_text(namespace) + mcp.namespace_separator + raw
    sanitized = re.sub(mcp.name_pattern, "_", raw).strip("_")
    if not sanitized:
        sanitized = "external_tool"
    return sanitized[: max(int(mcp.max_name_length or 64), 1)]


def _fallback_tool_choice(
    mode: ToolChoiceMode,
    policy: ProviderToolChoicePolicy,
) -> str | None:
    if mode == "none":
        return None
    if mode == "required" and policy.supports_required:
        return "required"
    if mode == "named" and policy.supports_named:
        return "auto"
    return "auto"


def _default_profile(
    *,
    model_name: str,
    family: ProviderFamily,
    supports_tools: bool,
    supports_image_input: bool,
) -> ProviderProtocolProfile:
    if family == "openai":
        schema = ProviderSchemaPolicy(dialect="openai_strict")
        tool_choice = ProviderToolChoicePolicy(
            supports_tools=supports_tools,
            supports_required=True,
            supports_named=True,
            supports_parallel=True,
        )
        exposure = ProviderSchemaExposurePolicy(
            prefers_compact=_is_small_model(model_name),
            full_schema_tool_cap=10,
            auto_command_tool_cap=32,
            required_command_tool_cap=24,
        )
        result_format: ToolResultMessageFormat = "openai_tool"
        max_tools = 120
    elif family == "gemini":
        schema = ProviderSchemaPolicy(
            dialect="gemini",
            strip_additional_properties=True,
            strip_strict=True,
            compose_oneof_as_anyof=True,
            compose_allof_as_anyof=True,
            max_tool_description_chars=1600,
            max_param_description_chars=520,
        )
        tool_choice = ProviderToolChoicePolicy(
            supports_tools=supports_tools,
            supports_required=True,
            supports_named=True,
            supports_parallel=False,
        )
        exposure = ProviderSchemaExposurePolicy(
            prefers_compact=True,
            full_schema_tool_cap=6,
            auto_command_tool_cap=24,
            required_command_tool_cap=18,
            compact_upgrade_prompt="Return a functionCall if a real command fits.",
        )
        result_format = "gemini_function_response"
        max_tools = 64
    elif family == "anthropic":
        schema = ProviderSchemaPolicy(dialect="generic")
        tool_choice = ProviderToolChoicePolicy(
            supports_tools=supports_tools,
            supports_required=True,
            supports_named=False,
            supports_parallel=True,
        )
        exposure = ProviderSchemaExposurePolicy(
            prefers_compact=False,
            full_schema_tool_cap=8,
            auto_command_tool_cap=32,
            required_command_tool_cap=24,
        )
        result_format = "anthropic_tool_result"
        max_tools = 96
    else:
        schema = ProviderSchemaPolicy(
            dialect="generic",
            max_tool_description_chars=1600,
            max_param_description_chars=600,
        )
        tool_choice = ProviderToolChoicePolicy(
            supports_tools=supports_tools,
            supports_required=False,
            supports_named=False,
            supports_parallel=False,
        )
        exposure = ProviderSchemaExposurePolicy(
            prefers_compact=True,
            full_schema_tool_cap=8,
            auto_command_tool_cap=24,
            required_command_tool_cap=20,
        )
        result_format = "generic"
        max_tools = 96
    mcp = MCPToolProtocolProfile(
        enabled=True,
        max_name_length=128 if family == "anthropic" else 64,
        result_message_format=result_format,
        max_result_chars=(
            8000 if family == "gemini" else 10000 if family == "custom" else 12000
        ),
        supports_streaming_observations=family == "anthropic",
    )
    return ProviderProtocolProfile(
        model_name=model_name,
        family=family,
        max_tools=max_tools,
        supports_image_input=supports_image_input,
        schema=schema,
        tool_choice=tool_choice,
        schema_exposure=exposure,
        tool_result_message_format=result_format,
        mcp=mcp,
        source="defaults",
    )


def _is_small_model(model_name: str) -> bool:
    lowered = model_name.casefold()
    return "mini" in lowered or "flash" in lowered or "lite" in lowered


def _compact_tool_result_value(value: Any, *, max_chars: int) -> Any:
    limit = max(int(max_chars or 0), 0)
    if limit <= 0:
        return value
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return value
    preview = text[: max(limit - 3, 0)] + "..."
    return {
        "truncated": True,
        "original_type": type(value).__name__,
        "preview": preview,
    }



__all__ = [
    "MCPToolProtocolProfile",
    "ProviderProtocolProfile",
    "ProviderSchemaExposurePolicy",
    "ProviderSchemaPolicy",
    "ProviderToolChoicePolicy",
    "SchemaDialect",
    "SchemaExposureMode",
    "ToolChoiceMode",
    "ToolResultMessageFormat",
    "adapt_tool_choice_for_policy",
    "adapt_tool_result_payload_for_protocol",
    "infer_provider_family",
    "load_provider_protocol_profile",
    "sanitize_external_tool_name_for_protocol",
]
