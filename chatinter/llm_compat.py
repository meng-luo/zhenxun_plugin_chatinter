"""Lightweight LLM protocol types for ChatInter core modules.

These classes intentionally avoid importing ``zhenxun.services`` so pure
routing/runtime modules can be imported before NoneBot is initialized.  They
mirror the small attribute surface ChatInter uses at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from zhenxun.services.llm import LLMMessage as LLMMessage
    from zhenxun.services.llm.tools import RunContext as RunContext
    from zhenxun.services.llm.types.models import (
        LLMContentPart as LLMContentPart,
    )
    from zhenxun.services.llm.types.models import (
        LLMToolCall as LLMToolCall,
    )
    from zhenxun.services.llm.types.models import (
        LLMToolFunction as LLMToolFunction,
    )
    from zhenxun.services.llm.types.models import (
        ToolDefinition as ToolDefinition,
    )
    from zhenxun.services.llm.types.models import ToolResult as ToolResult
    from zhenxun.services.llm.types.protocols import (
        ToolExecutable as ToolExecutable,
    )

else:

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

    @dataclass
    class LLMMessage:
        role: str
        content: str | list[LLMContentPart]
        name: str | None = None
        tool_calls: list[Any] | None = None
        tool_call_id: str | None = None
        thought_signature: str | None = None

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
        ) -> "LLMMessage":
            return cls(role="assistant", content=content, tool_calls=tool_calls)

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
            }
            data.update(update or {})
            return LLMMessage(**data)

    class ToolDefinition(BaseModel):
        name: str = Field(..., description="工具的唯一名称标识")
        description: str = Field(..., description="工具功能的清晰描述")
        parameters: dict[str, Any] = Field(default_factory=dict)

    class ToolResult(BaseModel):
        output: Any
        display_content: str | None = None

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


__all__ = [
    "LLMContentPart",
    "LLMMessage",
    "LLMToolCall",
    "LLMToolFunction",
    "RunContext",
    "ToolDefinition",
    "ToolExecutable",
    "ToolResult",
]
