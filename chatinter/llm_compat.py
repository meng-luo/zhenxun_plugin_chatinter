"""ChatInter local compatibility layer for the refactored AI service.

ChatInter keeps a compact local LLM surface internally and converts it to the
host bot's ``zhenxun.services.ai.*`` API at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class LLMContentPart:
    type: str = "text"
    text: str | None = None
    thought_text: str | None = None
    image_source: str | None = None
    mime_type: str | None = None

    @classmethod
    def text_part(cls, text: str) -> "LLMContentPart":
        return cls(type="text", text=text)

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
    ) -> "LLMMessage":
        metadata = {"scope": scope} if scope else None
        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            metadata=metadata,
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


class _CompatResponse:
    def __init__(self, response: Any):
        self._response = response
        self.text = str(getattr(response, "text", "") or "")
        self.raw_response = getattr(response, "raw_response", None)
        self.usage_info = getattr(response, "usage_info", None)
        self.parsed_obj = getattr(response, "parsed_obj", None)
        self.tool_calls = [
            _legacy_tool_call(call) for call in getattr(response, "tool_calls", [])
        ]
        self.thought_text = getattr(response, "thought_text", None)
        self.thought_signature = getattr(response, "thought_signature", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


LLMResponse = _CompatResponse


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
        return NewPart.tool_call_part(str(getattr(call, "id", "") or ""), name, args)
    return call


def _new_content_part(part: Any) -> Any:
    from zhenxun.services.ai.core.messages import BaseContentPart as NewPart

    part_type = str(getattr(part, "type", "") or "")
    if part_type == "text":
        return NewPart.text_part(str(getattr(part, "text", "") or ""))
    if part_type == "image":
        image_source = str(getattr(part, "image_source", "") or "")
        if image_source:
            return NewPart.image_base64_part(
                image_source,
                str(getattr(part, "mime_type", "") or "image/png"),
            )
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
        return NewMessage.system(str(message.content or ""))
    if role == "assistant":
        if message.tool_calls:
            return NewMessage.assistant_tool_calls(
                [_new_tool_call(call) for call in message.tool_calls],
                _new_content(message.content),
                scope=(message.metadata or {}).get("scope"),
            )
        return NewMessage.assistant_text_response(_new_content(message.content))
    if role == "tool":
        return NewMessage.tool_response(
            message.tool_call_id or "",
            message.name or "tool",
            message.content,
        )
    return NewMessage.user(_new_content(message.content))


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
    ) -> _CompatResponse:
        from zhenxun.services.ai.llm.api import generate

        if tools:
            response = await _generate_with_tools(
                messages=_new_messages(messages),
                model=model,
                config=_new_generation_config(config),
                tools=_new_tools(tools),
                tool_choice=tool_choice,
                timeout=timeout,
                session_id=self.session_id,
            )
        else:
            response = await generate(
                messages=_new_messages(messages),
                model=model,
                config=_new_generation_config(config),
                timeout=timeout,
                extra={"session_id": self.session_id} if self.session_id else None,
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


async def _generate_with_tools(
    *,
    messages: list[Any],
    model: str | None,
    config: Any | None,
    tools: list[Any] | None,
    tool_choice: str | dict[str, Any] | None,
    timeout: float | None,
    session_id: str | None,
) -> Any:
    from zhenxun.services.ai.core.messages import ChatRequest
    from zhenxun.services.ai.llm.engine.router import LLMOrchestrator

    request = ChatRequest(
        messages=messages,
        config=config,
        tools=tools,
        tool_choice=tool_choice,
        timeout=timeout,
        extra={"session_id": session_id} if session_id else {},
    )
    return await LLMOrchestrator.invoke(
        request,
        model_name=model,
        task="chat",
        override_config=config,
    )


class ToolInvoker:
    def __init__(self, callbacks: list[Any] | None = None):
        self.callbacks = callbacks or []

    async def execute_tool_call(
        self,
        tool_call: Any,
        available_tools: dict[str, ToolExecutable],
        context: Any | None = None,
    ) -> tuple[Any, ToolResult]:
        if hasattr(tool_call, "function"):
            tool_name = str(tool_call.function.name or "")
            args_raw = tool_call.function.arguments or "{}"
        else:
            tool_name = str(getattr(tool_call, "tool_name", "") or "")
            args_raw = getattr(tool_call, "args", {}) or {}
        args: dict[str, Any] = {}
        if isinstance(args_raw, str):
            try:
                parsed = json.loads(args_raw or "{}")
                if isinstance(parsed, dict):
                    args = parsed
            except Exception:
                args = {}
        elif isinstance(args_raw, dict):
            args = args_raw
        executable = available_tools.get(tool_name)
        if executable is None:
            return tool_call, ToolResult(
                output=f"Error: Tool '{tool_name}' not found.",
                display_content=f"tool not found: {tool_name}",
                is_error=True,
            )
        try:
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


async def embed_documents(texts: list[str]) -> list[list[float]]:
    from zhenxun.services.ai.llm.api import embed

    response = await embed(texts, task="document")
    return response.embeddings


async def embed_query(text: str) -> list[float]:
    from zhenxun.services.ai.llm.api import embed

    response = await embed(text, task="query")
    return response.vector


def list_embedding_models() -> list[Any]:
    try:
        from zhenxun.services.ai.llm.manager import list_embedding_models as _list

        return list(_list() or [])
    except Exception:
        return []


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
    "embed_documents",
    "embed_query",
    "list_embedding_models",
]
