"""Provider/model capability adapter for ChatInter Agent requests.

This layer keeps ChatInter from assuming every model/provider accepts the same
tool count, tool_choice shape, multimodal input, or JSON schema dialect.
It is intentionally conservative: unknown OpenAI-compatible gateways get safe
limits instead of optimistic "send everything" behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal
import copy
import re

from zhenxun.services.llm import LLMContentPart, LLMMessage
from zhenxun.services.llm.types.capabilities import (
    ModelModality,
    get_model_capabilities,
)
from zhenxun.services.llm.types.models import ToolDefinition, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .route_text import normalize_message_text

ProviderFamily = Literal["openai", "gemini", "anthropic", "custom"]
SchemaDialect = Literal["openai_strict", "gemini", "generic"]
ToolResultMessageFormat = Literal["openai_tool", "gemini_function_response", "generic"]

_OPENAI_MAX_TOOLS = 128
_SAFE_OPENAI_TOOL_CAP = 120
_SAFE_GEMINI_TOOL_CAP = 64
_SAFE_GENERIC_TOOL_CAP = 96
_MAX_TOOL_DESCRIPTION_CHARS = 1800
_MAX_PARAM_DESCRIPTION_CHARS = 700
_UNSUPPORTED_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "examples",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
}


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
    tool_result_message_format: ToolResultMessageFormat

    def to_metadata(self) -> dict[str, Any]:
        return {
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
            "tool_result_message_format": self.tool_result_message_format,
        }


class ProviderAdjustedTool:
    """Tool view that sanitizes the definition for the target provider."""

    def __init__(
        self,
        *,
        executable: ToolExecutable,
        adapter: "ProviderCapabilityAdapter",
        schema_mode: Literal["full", "compact"] = "full",
    ) -> None:
        self.executable = executable
        self.adapter = adapter
        self.chatinter_schema_mode = schema_mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self.executable, name)

    async def get_definition(self) -> ToolDefinition:
        definition = await self.executable.get_definition()
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
        lowered = name.casefold()
        family = _infer_family(lowered)
        capabilities = get_model_capabilities(name)
        supports_tools = bool(capabilities.supports_tool_calling)
        supports_image = ModelModality.IMAGE in capabilities.input_modalities
        max_tools = _infer_max_tools(family=family, model_name=lowered)
        schema_dialect = _schema_dialect(family)
        profile = ProviderCapabilityProfile(
            model_name=name,
            family=family,
            schema_dialect=schema_dialect,
            max_tools=max_tools,
            supports_tools=supports_tools,
            supports_image_input=supports_image,
            supports_parallel_tool_calls=_supports_parallel_tool_calls(lowered, family),
            supports_required_tool_choice=_supports_required_tool_choice(lowered, family),
            supports_named_tool_choice=_supports_named_tool_choice(lowered, family),
            prefers_compact_command_schema=_prefers_compact_schema(lowered, family),
            full_schema_tool_cap=_infer_full_schema_tool_cap(family=family),
            tool_result_message_format=_tool_result_format(family),
        )
        return cls(profile)

    @property
    def max_tools(self) -> int:
        return max(0, int(self.profile.max_tools or 0))

    def command_tool_capacity(self, *, reserved_tools: int = 0) -> int:
        if not self.profile.supports_tools:
            return 0
        return max(0, self.max_tools - max(int(reserved_tools or 0), 0))

    def adapt_tool_choice(
        self,
        tool_choice: str | dict[str, Any] | None,
        *,
        has_tools: bool,
    ) -> str | dict[str, Any] | None:
        if not has_tools or not self.profile.supports_tools:
            return None
        if tool_choice is None or tool_choice == "none":
            return None
        if isinstance(tool_choice, dict):
            if self.profile.supports_named_tool_choice:
                return tool_choice
            return "required" if self.profile.supports_required_tool_choice else "auto"
        if tool_choice == "required" and not self.profile.supports_required_tool_choice:
            return "auto"
        if tool_choice in {"auto", "required"}:
            return tool_choice
        return "auto"

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
        schema_modes: dict[str, Literal["full", "compact"]] | None = None,
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

    def adapt_messages(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        if self.profile.supports_image_input:
            return messages
        changed = False
        adapted: list[LLMMessage] = []
        for message in messages:
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
        return adapted if changed else messages

    def sanitize_tool_definition(self, definition: ToolDefinition) -> ToolDefinition:
        name = _sanitize_tool_name(definition.name)
        description = _clip_text(definition.description, _MAX_TOOL_DESCRIPTION_CHARS)
        parameters = sanitize_json_schema(
            definition.parameters or {},
            dialect=self.profile.schema_dialect,
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

    def schema_mode_for_tool(
        self,
        tool_name: str,
        *,
        full_schema_names: set[str],
    ) -> Literal["full", "compact"]:
        return "full" if tool_name in full_schema_names else "compact"


def is_compact_request_tool(tool: ToolExecutable | None) -> bool:
    return str(getattr(tool, "chatinter_schema_mode", "") or "") == "compact"


def sanitize_json_schema(
    schema: dict[str, Any],
    *,
    dialect: SchemaDialect,
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}
    cleaned = _sanitize_schema_node(copy.deepcopy(schema), dialect=dialect)
    if not isinstance(cleaned, dict):
        return {"type": "object", "properties": {}, "required": []}
    if cleaned.get("type") is None and "properties" in cleaned:
        cleaned["type"] = "object"
    if cleaned.get("type") == "object":
        cleaned.setdefault("properties", {})
        cleaned.setdefault("required", [])
    return cleaned


def _sanitize_schema_node(value: Any, *, dialect: SchemaDialect) -> Any:
    if isinstance(value, list):
        return [_sanitize_schema_node(item, dialect=dialect) for item in value]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    for key, raw in value.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if dialect == "gemini" and key in {"additionalProperties", "strict"}:
            continue
        if key == "description" and isinstance(raw, str):
            result[key] = _clip_text(raw, _MAX_PARAM_DESCRIPTION_CHARS)
            continue
        if key in {"anyOf", "oneOf", "allOf"} and isinstance(raw, list):
            variants = [_sanitize_schema_node(item, dialect=dialect) for item in raw]
            if dialect == "gemini" and key in {"oneOf", "allOf"}:
                if variants:
                    result["anyOf"] = variants
                continue
            result[key] = variants
            continue
        result[key] = _sanitize_schema_node(raw, dialect=dialect)

    schema_type = result.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        if len(non_null) == 1:
            result["type"] = non_null[0]
            if "null" in schema_type:
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


def _infer_family(model_name: str) -> ProviderFamily:
    if "gemini" in model_name:
        return "gemini"
    if "claude" in model_name or "anthropic" in model_name:
        return "anthropic"
    if any(
        token in model_name
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


def _schema_dialect(family: ProviderFamily) -> SchemaDialect:
    if family == "gemini":
        return "gemini"
    if family == "openai":
        return "openai_strict"
    return "generic"


def _infer_max_tools(*, family: ProviderFamily, model_name: str) -> int:
    if family == "openai":
        return _SAFE_OPENAI_TOOL_CAP
    if family == "gemini":
        return _SAFE_GEMINI_TOOL_CAP
    if family == "anthropic":
        return _SAFE_GENERIC_TOOL_CAP
    if "gpt" in model_name or "chatgpt" in model_name:
        return min(_SAFE_OPENAI_TOOL_CAP, _OPENAI_MAX_TOOLS - 8)
    return _SAFE_GENERIC_TOOL_CAP


def _supports_parallel_tool_calls(model_name: str, family: ProviderFamily) -> bool:
    if family == "gemini":
        return False
    if family == "anthropic":
        return True
    if family == "openai":
        return True
    return False


def _supports_required_tool_choice(model_name: str, family: ProviderFamily) -> bool:
    if family in {"openai", "gemini", "anthropic"}:
        return True
    return "gpt" in model_name or "gemini" in model_name


def _supports_named_tool_choice(model_name: str, family: ProviderFamily) -> bool:
    if family == "openai":
        return True
    if family == "gemini":
        return True
    return False


def _prefers_compact_schema(model_name: str, family: ProviderFamily) -> bool:
    if family == "gemini":
        return True
    if family == "custom":
        return True
    return "mini" in model_name or "flash" in model_name or "lite" in model_name


def _infer_full_schema_tool_cap(*, family: ProviderFamily) -> int:
    if family == "gemini":
        return 6
    if family == "openai":
        return 10
    return 8


def _tool_result_format(family: ProviderFamily) -> ToolResultMessageFormat:
    if family == "gemini":
        return "gemini_function_response"
    if family == "openai":
        return "openai_tool"
    return "generic"


def _sanitize_tool_name(value: str) -> str:
    name = normalize_message_text(value)
    name = re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")
    if not name:
        return "chatinter_tool"
    return name[:64]


def _clip_text(value: str, limit: int) -> str:
    text = normalize_message_text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _is_command_tool(tool: ToolExecutable) -> bool:
    return getattr(tool, "binding", None) is not None


__all__ = [
    "ProviderAdjustedTool",
    "ProviderCapabilityAdapter",
    "ProviderCapabilityProfile",
    "is_compact_request_tool",
    "sanitize_json_schema",
]
