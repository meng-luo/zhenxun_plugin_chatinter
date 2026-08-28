"""ChatInter local compatibility layer for the refactored AI service.

ChatInter keeps a compact local LLM surface internally and converts it to the
host bot's ``zhenxun.services.ai.*`` API at the boundary.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import inspect
import json
from typing import Any, Protocol

import json_repair
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class LLMContentPart:
    type: str = "text"
    text: str | None = None
    thought_text: str | None = None
    image_source: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def text_part(cls, text: str) -> "LLMContentPart":
        return cls(type="text", text=text)

    @classmethod
    def thought_part(
        cls,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "LLMContentPart":
        return cls(type="thought", thought_text=text, metadata=metadata)

    @classmethod
    def image_base64_part(
        cls, data: str, mime_type: str = "image/png"
    ) -> "LLMContentPart":
        return cls(type="image", image_source=data, mime_type=mime_type)


@dataclass
class LLMMessage:
    role: str
    content: str | list[LLMContentPart]
    name: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None
    thought_signature: str | None = None
    metadata: dict[str, Any] | None = None
    content_parts: list[Any] | None = None

    @classmethod
    def user(cls, content: str | list[LLMContentPart]) -> "LLMMessage":
        return cls(role="user", content=content)

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        return cls(role="system", content=content)

    @classmethod
    def assistant_text_response(
        cls,
        content: str | list[LLMContentPart],
    ) -> "LLMMessage":
        return cls(role="assistant", content=content)

    @classmethod
    def assistant_tool_calls(
        cls,
        tool_calls: list[Any],
        content: str | list[LLMContentPart] = "",
        scope: str | None = None,
        thought_text: str | None = None,
        content_parts: list[Any] | None = None,
        source_model: str | None = None,
        source_api_type: str | None = None,
        provider_replay_kind: str | None = None,
        provider_replay_payload: list[dict[str, Any]] | None = None,
    ) -> "LLMMessage":
        if content_parts is not None:
            content_parts = _content_parts_with_tool_calls(content_parts, tool_calls)
            content = _legacy_content_view(content_parts)
        elif thought_text is not None:
            parts = (
                [LLMContentPart.text_part(content)]
                if isinstance(content, str) and content
                else list(content)
                if isinstance(content, list)
                else []
            )
            parts.append(LLMContentPart.thought_part(thought_text))
            content = parts
        metadata: dict[str, Any] = {}
        if scope:
            metadata["scope"] = scope
        if source_model:
            metadata["source_model"] = source_model
            if source_api_type:
                metadata["source_api_type"] = str(source_api_type)
        if provider_replay_payload:
            metadata["provider_replay_kind"] = (
                provider_replay_kind or "responses_output"
            )
            metadata["provider_replay_payload"] = copy.deepcopy(provider_replay_payload)
        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            metadata=metadata or None,
            content_parts=content_parts,
        )

    @classmethod
    def tool_response(
        cls,
        tool_call_id: str,
        function_name: str,
        result: Any,
    ) -> "LLMMessage":
        return cls(
            role="tool",
            content=json.dumps(result, ensure_ascii=False, default=str),
            name=function_name,
            tool_call_id=tool_call_id,
        )

    def model_copy(self, *, update: dict[str, Any] | None = None) -> "LLMMessage":
        data = {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "thought_signature": self.thought_signature,
            "metadata": self.metadata,
            "content_parts": self.content_parts,
        }
        data.update(update or {})
        return LLMMessage(**data)


class ToolDefinition(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any
    display_content: str | None = None
    is_error: bool = False
    is_retryable: bool = True

    def as_error(self, is_retryable: bool = True) -> "ToolResult":
        self.is_error = True
        self.is_retryable = is_retryable
        return self

    def as_fatal(self) -> "ToolResult":
        return self.as_error(is_retryable=False)


@dataclass
class LLMToolFunction:
    name: str
    arguments: str = ""


@dataclass
class LLMToolCall:
    id: str
    function: LLMToolFunction
    thought_signature: str | None = None
    metadata: dict[str, Any] | None = None
    content_parts: list[Any] | None = None
    type: str = "function"


class ToolExecutable(Protocol):
    async def get_definition(self) -> ToolDefinition: ...

    async def execute(
        self, context: Any | None = None, **kwargs: Any
    ) -> ToolResult: ...


@dataclass
class RunContext:
    session_id: str | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope and not self.extra:
            self.extra = self.scope
        if self.extra and not self.scope:
            self.scope = self.extra
        if self.extra and not self.state:
            self.state = self.extra


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # type: ignore

    effort: Any | None = None
    show_thoughts: bool | None = None


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # type: ignore

    mode: str = "AUTO"


class LLMGenerationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")  # type: ignore

    reasoning: ReasoningConfig | None = None
    tool_config: ToolConfig | None = None
    max_tokens: int | None = None
    response_format: Any | None = None
    response_schema: dict[str, Any] | None = None
    response_mime_type: str | None = None
    structured_output_strategy: str | None = None
    validation_policy: dict[str, Any] | None = None

    def merge_with(self, other: "LLMGenerationConfig | Any | None") -> Any:
        if other is None:
            return self
        if not isinstance(other, LLMGenerationConfig):
            return other
        data = self.model_dump(exclude_none=True)
        other_data = other.model_dump(exclude_none=True)
        for key, value in other_data.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return LLMGenerationConfig(**data)


def _new_generation_config(config: Any | None) -> Any | None:
    if config is None:
        return None
    if not isinstance(config, LLMGenerationConfig):
        return config
    from zhenxun.services.ai.core.options import GenerationConfig

    result = GenerationConfig()
    if config.reasoning and config.reasoning.effort:
        result.common.reasoning_effort = config.reasoning.effort
        if config.reasoning.show_thoughts is not None:
            result.gemini_options.include_thoughts = config.reasoning.show_thoughts
    if config.max_tokens is not None:
        result.common.max_tokens = config.max_tokens
    if config.response_format is not None:
        result.output.response_format = config.response_format
    if config.response_schema is not None:
        result.output.response_schema = config.response_schema
    if config.response_mime_type is not None:
        result.output.response_mime_type = config.response_mime_type
    if config.structured_output_strategy is not None:
        result.output.structured_output_strategy = config.structured_output_strategy
    if config.validation_policy is not None:
        result.validation_policy = copy.deepcopy(config.validation_policy)
    if config.tool_config and config.tool_config.mode:
        mode = str(config.tool_config.mode or "AUTO").upper()
        if mode in {"ANY", "REQUIRED"}:
            result.tools.mode = "ANY"
        elif mode == "NONE":
            result.tools.mode = "NONE"
        else:
            result.tools.mode = "AUTO"
    return result


class ReasoningEffort(str):
    NONE = "NONE"
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    XHIGH = "XHIGH"
    MAX = "MAX"


_MISSING = object()


class _CompatResponse:
    def __init__(self, response: Any):
        self._response = response
        self.text = str(getattr(response, "text", "") or "")
        self.raw_response = getattr(response, "raw_response", None)
        self.usage_info = getattr(response, "usage_info", None)
        self.parsed_obj = getattr(response, "parsed_obj", None)
        self.content_parts = copy.deepcopy(
            list(getattr(response, "content_parts", []) or [])
        )
        replay_items = _raw_responses_output_items(self.raw_response)
        self.reasoning_replay_items = copy.deepcopy(replay_items)
        self.tool_calls = [
            _legacy_tool_call(call) for call in getattr(response, "tool_calls", [])
        ]
        self.thought_text = _response_thought_text(response)
        self.thought_signature = getattr(response, "thought_signature", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


LLMResponse = _CompatResponse


def _field_value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value[name] if name in value else _MISSING
    fields_set = getattr(value, "model_fields_set", None)
    if fields_set is not None and name not in fields_set:
        return _MISSING
    return getattr(value, name, _MISSING)


def _raw_responses_output_items(raw_response: Any) -> list[dict[str, Any]]:
    output = _field_value(raw_response, "output")
    if not isinstance(output, list | tuple) or not output:
        return []
    if not all(isinstance(item, dict) for item in output):
        return []
    return copy.deepcopy(list(output))


def response_reasoning_replay_items(response: Any) -> list[dict[str, Any]]:
    items = getattr(response, "reasoning_replay_items", _MISSING)
    if isinstance(items, list) and all(isinstance(item, dict) for item in items):
        return copy.deepcopy(items)
    return _raw_responses_output_items(getattr(response, "raw_response", None))


def _response_thought_text(response: Any) -> str | None:
    from zhenxun.services.ai.core.messages import ThoughtPart

    thoughts = [
        part.thought_text
        for part in getattr(response, "content_parts", [])
        if isinstance(part, ThoughtPart)
    ]
    return "\n".join(thoughts) if thoughts else None


def _part_value(part: Any, name: str, default: Any = None) -> Any:
    return (
        part.get(name, default)
        if isinstance(part, dict)
        else getattr(part, name, default)
    )


def _content_parts_with_tool_calls(
    content_parts: list[Any],
    tool_calls: list[Any],
) -> list[Any]:
    replacements = {
        str(getattr(call, "id", "") or ""): _new_tool_call(call)
        for call in tool_calls
        if str(getattr(call, "id", "") or "")
    }
    result: list[Any] = []
    emitted: set[str] = set()
    for part in content_parts:
        is_tool_call = str(
            _part_value(part, "type", "") or ""
        ).casefold() == "tool_call" or isinstance(part, LLMToolCall)
        if not is_tool_call:
            result.append(copy.deepcopy(part))
            continue
        call_id = str(_part_value(part, "id", "") or "")
        replacement = replacements.get(call_id)
        if replacement is None:
            continue
        result.append(replacement)
        emitted.add(call_id)
    result.extend(
        replacement
        for call_id, replacement in replacements.items()
        if call_id not in emitted
    )
    return result


def _legacy_tool_call(call: Any) -> LLMToolCall:
    args = getattr(call, "args", "")
    if not isinstance(args, str):
        args = json.dumps(args or {}, ensure_ascii=False, default=str)
    metadata = getattr(call, "metadata", None) or {}
    return LLMToolCall(
        id=str(getattr(call, "id", "") or ""),
        function=LLMToolFunction(
            name=str(getattr(call, "tool_name", "") or ""),
            arguments=args,
        ),
        thought_signature=metadata.get("thought_signature"),
        metadata=copy.deepcopy(metadata) or None,
    )


def _new_tool_call(call: Any) -> Any:
    from zhenxun.services.ai.core.messages import BaseContentPart as NewPart

    if hasattr(call, "function"):
        name = str(getattr(call.function, "name", "") or "")
        raw_args = getattr(call.function, "arguments", "") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = raw_args
        converted = NewPart.tool_call_part(
            str(getattr(call, "id", "") or ""), name, args
        )
        metadata = getattr(call, "metadata", None)
        metadata = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
        signature = getattr(call, "thought_signature", None)
        if signature:
            metadata.setdefault("thought_signature", signature)
        if metadata:
            converted.metadata = metadata
        return converted
    return call


def _legacy_content_view(content_parts: list[Any]) -> list[LLMContentPart]:
    result: list[LLMContentPart] = []
    for part in content_parts:
        part_type = str(_part_value(part, "type", "") or "")
        metadata = copy.deepcopy(_part_value(part, "metadata"))
        if part_type == "text":
            result.append(
                LLMContentPart(
                    type="text",
                    text=str(_part_value(part, "text", "") or ""),
                    metadata=metadata,
                )
            )
        elif part_type == "thought":
            result.append(
                LLMContentPart.thought_part(
                    str(_part_value(part, "thought_text", "") or ""),
                    metadata=metadata,
                )
            )
    return result


def _new_standard_content_parts(content_parts: list[Any]) -> list[Any]:
    from zhenxun.services.ai.core.messages import BaseContentPart, LLMContentPart
    from zhenxun.utils.pydantic_compat import parse_as

    result: list[Any] = []
    for part in content_parts:
        if isinstance(part, BaseContentPart):
            result.append(copy.deepcopy(part))
        elif isinstance(part, dict):
            result.append(parse_as(LLMContentPart, copy.deepcopy(part)))
        else:
            result.append(_new_content_part(part))
    return result


def _new_content_part(part: Any) -> Any:
    from zhenxun.services.ai.core.messages import BaseContentPart as NewPart

    def value(name: str, default: Any = None) -> Any:
        return (
            part.get(name, default)
            if isinstance(part, dict)
            else getattr(part, name, default)
        )

    part_type = str(value("type", "") or "")
    metadata = value("metadata")
    if part_type == "text":
        converted = NewPart.text_part(str(value("text", "") or ""))
        if isinstance(metadata, dict):
            converted.metadata = copy.deepcopy(metadata)
        return converted
    if part_type == "thought":
        converted = NewPart.thought_part(str(value("thought_text", "") or ""))
        if isinstance(metadata, dict):
            converted.metadata = copy.deepcopy(metadata)
        return converted
    if part_type == "image":
        image_source = str(value("image_source", "") or "")
        if image_source:
            converted = NewPart.image_base64_part(
                image_source,
                str(value("mime_type", "") or "image/png"),
            )
            if isinstance(metadata, dict):
                converted.metadata = copy.deepcopy(metadata)
            return converted
    return part


def _new_content(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        if not content.strip():
            return []
        from zhenxun.services.ai.core.messages import BaseContentPart as NewPart

        return [NewPart.text_part(content)]
    if isinstance(content, list):
        result: list[Any] = []
        for item in content:
            if isinstance(item, str):
                from zhenxun.services.ai.core.messages import BaseContentPart as NewPart

                result.append(NewPart.text_part(item))
            else:
                result.append(_new_content_part(item))
        return result
    return _new_content(str(content))


def _new_message(message: Any) -> Any:
    from zhenxun.services.ai.core.messages import LLMMessage as NewMessage

    if not isinstance(message, LLMMessage):
        return message
    role = str(message.role or "user")
    if role == "system":
        converted = NewMessage.system(str(message.content or ""))
    elif role == "assistant":
        if message.content_parts is not None:
            from zhenxun.services.ai.core.messages import AssistantMessage

            converted = AssistantMessage(
                content=_new_standard_content_parts(message.content_parts)
            )
        elif message.tool_calls:
            converted = NewMessage.assistant_tool_calls(
                [_new_tool_call(call) for call in message.tool_calls],
                _new_content(message.content),
                scope=(message.metadata or {}).get("scope"),
            )
        else:
            converted = NewMessage.assistant_text_response(
                _new_content(message.content)
            )
    elif role == "tool":
        converted = NewMessage.tool_response(
            message.tool_call_id or "",
            message.name or "tool",
            message.content,
        )
    else:
        converted = NewMessage.user(_new_content(message.content))
    converted.metadata = (
        dict(message.metadata) if isinstance(message.metadata, dict) else None
    )
    return converted


def _new_messages(messages: list[Any]) -> list[Any]:
    return [_new_message(message) for message in messages]


def _new_tools(tools: Any | None) -> list[Any] | None:
    if tools is None:
        return None
    return list(tools.values()) if isinstance(tools, dict) else list(tools)


def _accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class AI:
    def __init__(self, session_id: str | None = None, **_: Any):
        self.session_id = session_id

    async def generate_internal(
        self,
        messages: list[Any],
        model: str | None = None,
        config: Any | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        timeout: float | None = None,
        prompt_cache_key: str | None = None,
        cancellation_token: Any | None = None,
    ) -> _CompatResponse:
        from .host_llm import HostLLMClient, resolve_host_model_candidates

        candidate = model
        if not candidate or "/" not in candidate:
            candidate = (await resolve_host_model_candidates(candidate))[0].name

        response = await HostLLMClient().invoke(
            candidate=candidate,
            messages=_new_messages(messages),
            config=_new_generation_config(config),
            tools=_new_tools(tools),
            tool_choice=tool_choice,
            timeout=timeout,
            prompt_cache_key=prompt_cache_key,
            cancellation_token=cancellation_token,
        )
        return _CompatResponse(response)

    async def generate_structured(
        self,
        message: Any,
        response_model: type[BaseModel],
        model: str | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        instruction: str | None = None,
        timeout: float | None = None,
        template_vars: dict[str, Any] | None = None,
        config: Any | None = None,
        max_validation_retries: int | None = None,
        validation_callback: Any | None = None,
        error_prompt_template: str | None = None,
        auto_thinking: bool = False,
        usage_callback: Any | None = None,
    ) -> Any:
        from zhenxun.services.ai.llm.api import generate_structured

        del tools, tool_choice, template_vars, validation_callback, auto_thinking
        payload = _new_messages(message) if isinstance(message, list) else message
        kwargs = {
            "message": payload,
            "response_model": response_model,
            "model": model,
            "instruction": instruction,
            "timeout": timeout,
            "config": _new_generation_config(config),
            "max_retries": max_validation_retries,
            "error_prompt_template": error_prompt_template,
        }
        if usage_callback is not None and _accepts_keyword(
            generate_structured,
            "usage_callback",
        ):
            kwargs["usage_callback"] = usage_callback
        return await generate_structured(**kwargs)


class ToolInvoker:
    def __init__(self, callbacks: list[Any] | None = None):
        self.callbacks = callbacks or []

    async def execute_tool_call(
        self,
        tool_call: Any,
        available_tools: dict[str, ToolExecutable],
        context: Any | None = None,
    ) -> tuple[Any, ToolResult]:
        try:
            executable, args, validation_error = await validate_tool_call_arguments(
                tool_call,
                available_tools,
            )
            if validation_error is not None:
                return tool_call, validation_error
            if executable is None or args is None:
                raise RuntimeError("tool validation returned no executable")
            result = await executable.execute(context=context, **args)
            if isinstance(result, ToolResult):
                return tool_call, result
            return tool_call, ToolResult(
                output=getattr(result, "output", result),
                display_content=getattr(result, "display_content", None),
                is_error=bool(getattr(result, "is_error", False)),
                is_retryable=bool(getattr(result, "is_retryable", True)),
            )
        except Exception as exc:
            return tool_call, ToolResult(
                output=f"System Execution Error: {exc!s}",
                display_content=str(exc),
                is_error=True,
            )


async def validate_tool_call_arguments(
    tool_call: Any,
    available_tools: dict[str, ToolExecutable],
) -> tuple[ToolExecutable | None, dict[str, Any] | None, ToolResult | None]:
    tool_name, args_raw = _tool_call_name_and_arguments(tool_call)
    executable = available_tools.get(tool_name)
    if executable is None:
        return (
            None,
            None,
            _tool_argument_error(
                tool_name,
                status="tool_not_found",
                validation_error="unknown_tool",
                error=f"未知工具：{tool_name or '<empty>'}",
            ),
        )

    if isinstance(args_raw, str):
        try:
            parsed = json.loads(args_raw)
        except (TypeError, ValueError) as exc:
            if bool(
                getattr(
                    executable,
                    "chatinter_ignore_unknown_top_level_arguments",
                    False,
                )
            ):
                try:
                    parsed = json_repair.loads(args_raw, skip_json_loads=True)
                except Exception:
                    parsed = None
            else:
                parsed = None
            if parsed is None:
                return (
                    executable,
                    None,
                    _tool_argument_error(
                        tool_name,
                        validation_error="invalid_json",
                        error=f"工具参数不是有效 JSON：{exc}",
                    ),
                )
    else:
        parsed = args_raw
    if not isinstance(parsed, dict):
        return (
            executable,
            None,
            _tool_argument_error(
                tool_name,
                validation_error="arguments_not_object",
                error="工具参数必须是 JSON object。",
            ),
        )

    definition = await executable.get_definition()
    schema = definition.parameters if isinstance(definition.parameters, dict) else {}
    validation = _validate_json_schema(parsed, schema, path="$")
    if (
        validation is not None
        and validation[0] == "unexpected_arguments"
        and bool(
            getattr(
                executable,
                "chatinter_ignore_unknown_top_level_arguments",
                False,
            )
        )
    ):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            # Plugin dispatch tools share one model-visible catalog. A model can
            # occasionally copy a top-level field from a neighbouring dispatch
            # schema. Unknown fields are never executable input, so discard them
            # once and validate the complete declared schema again.
            parsed = {key: value for key, value in parsed.items() if key in properties}
            validation = _validate_json_schema(parsed, schema, path="$")
    if validation is not None:
        validation_error, path, error = validation
        return (
            executable,
            None,
            _tool_argument_error(
                tool_name,
                validation_error=validation_error,
                error=error,
                field=path,
            ),
        )
    return executable, parsed, None


async def normalize_responses_tool_argument_envelope(
    tool_call: Any,
    available_tools: dict[str, ToolExecutable],
) -> tuple[Any, bool]:
    """Unwrap one provider-added ``arguments`` envelope after strict validation."""

    tool_name, args_raw = _tool_call_name_and_arguments(tool_call)
    executable = available_tools.get(tool_name)
    if executable is None or not isinstance(args_raw, str):
        return tool_call, False
    try:
        outer = json.loads(args_raw)
    except (TypeError, ValueError):
        return tool_call, False
    if not isinstance(outer, dict) or set(outer) != {"arguments"}:
        return tool_call, False

    definition = await executable.get_definition()
    schema = definition.parameters if isinstance(definition.parameters, dict) else {}
    properties = schema.get("properties")
    if isinstance(properties, dict) and "arguments" in properties:
        return tool_call, False

    inner = outer["arguments"]
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except (TypeError, ValueError):
            return tool_call, False
    if not isinstance(inner, dict):
        return tool_call, False
    if _validate_json_schema(inner, schema, path="$") is not None:
        return tool_call, False

    normalized = copy.deepcopy(tool_call)
    function = getattr(normalized, "function", None)
    if function is None:
        return tool_call, False
    function.arguments = json.dumps(
        inner,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return normalized, True


def _tool_call_name_and_arguments(tool_call: Any) -> tuple[str, Any]:
    if hasattr(tool_call, "function"):
        function = tool_call.function
        return str(function.name or ""), function.arguments
    return (
        str(getattr(tool_call, "tool_name", "") or ""),
        getattr(tool_call, "args", {}),
    )


def _validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
) -> tuple[str, str, str] | None:
    if "const" in schema and not _json_values_equal(value, schema["const"]):
        return "const_mismatch", path, f"参数 {path} 必须等于 schema 指定值。"
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(
        _json_values_equal(value, candidate) for candidate in enum
    ):
        return "enum_mismatch", path, f"参数 {path} 不在允许值范围内。"

    expected = schema.get("type")
    expected_types = [expected] if isinstance(expected, str) else expected
    if isinstance(expected_types, list | tuple) and expected_types:
        if not any(_matches_json_type(value, item) for item in expected_types):
            names = ", ".join(str(item) for item in expected_types)
            return "type_mismatch", path, f"参数 {path} 类型无效，期望：{names}"

    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        required = required if isinstance(required, list | tuple) else ()
        missing = [str(name) for name in required if str(name) not in value]
        if missing:
            return (
                "missing_required",
                path,
                "缺少必填参数：" + ", ".join(missing),
            )
        if schema.get("additionalProperties") is False:
            unexpected = sorted(str(name) for name in value if name not in properties)
            if unexpected:
                return (
                    "unexpected_arguments",
                    path,
                    "包含未定义参数：" + ", ".join(unexpected),
                )
        additional_schema = schema.get("additionalProperties")
        for name, item in value.items():
            property_schema = properties.get(name)
            if not isinstance(property_schema, dict):
                property_schema = (
                    additional_schema if isinstance(additional_schema, dict) else None
                )
            if property_schema is None:
                continue
            error = _validate_json_schema(
                item,
                property_schema,
                path=f"{path}.{name}",
            )
            if error is not None:
                return error

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        item_schema = schema["items"]
        for index, item in enumerate(value):
            error = _validate_json_schema(item, item_schema, path=f"{path}[{index}]")
            if error is not None:
                return error
    return None


def _matches_json_type(value: Any, expected: Any) -> bool:
    validators = {
        "array": lambda: isinstance(value, list),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "null": lambda: value is None,
        "number": lambda: isinstance(value, int | float)
        and not isinstance(value, bool),
        "object": lambda: isinstance(value, dict),
        "string": lambda: isinstance(value, str),
    }
    validator = validators.get(str(expected))
    return bool(validator and validator())


def _json_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _tool_argument_error(
    tool_name: str,
    *,
    status: str = "invalid_tool_arguments",
    validation_error: str,
    error: str,
    **details: Any,
) -> ToolResult:
    return ToolResult(
        output={
            "ok": False,
            "status": status,
            "tool_name": tool_name,
            "validation_error": validation_error,
            "error": error,
            "retryable": True,
            "need_continue": True,
            **details,
        },
        display_content=error,
        is_error=True,
        is_retryable=True,
    )


__all__ = [
    "AI",
    "LLMContentPart",
    "LLMGenerationConfig",
    "LLMMessage",
    "LLMResponse",
    "LLMToolCall",
    "LLMToolFunction",
    "ReasoningConfig",
    "ReasoningEffort",
    "RunContext",
    "ToolConfig",
    "ToolDefinition",
    "ToolExecutable",
    "ToolInvoker",
    "ToolResult",
    "normalize_responses_tool_argument_envelope",
    "response_reasoning_replay_items",
    "validate_tool_call_arguments",
]
