"""Shared boundary types for ChatInter scenario agents.

Only cross-scenario plumbing belongs here.  Plugin selection, superuser
permissions, approval, MCP and worktree logic should stay in their own agents.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..llm_compat import LLMMessage
from ..native_executor import NativeCommandExecutionContext
from ..native_route import NativeRouteReport
from ..provider_capability import ProviderCapabilityAdapter
from ..response_defaults import EMPTY_REPLY_TEXT
from ..route_text import normalize_reply_text
from ..turn_runtime import TurnBudgetController, estimate_text_tokens

if TYPE_CHECKING:
    from ..context_budget import ChatContextBundle
    from ..llm_compat import ToolResult
    from ..main_request_models import MainRequestResult
    from ..mixed_tool_catalog import MixedToolCatalog

ProgressHook = Callable[[str], Awaitable[None] | None]
AgentKind = Literal["unified_chat", "superuser"]
AgentObservationStatus = Literal["ok", "error", "fallback"]


@dataclass(frozen=True, slots=True)
class ToolScope:
    """Explicit tool boundary for one ChatInter scenario."""

    kind: AgentKind
    allow_plugin: bool = False


UNIFIED_CHAT_TOOL_SCOPE = ToolScope(kind="unified_chat", allow_plugin=True)


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """Lightweight cross-agent observation for boundary-level diagnostics."""

    kind: str
    status: AgentObservationStatus = "ok"
    message: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Unified return wrapper for scenario agents.

    The payload remains ``MainRequestResult`` for compatibility with the
    existing pipeline.  The wrapper gives callers a stable boundary while each
    agent keeps its own business strategy.
    """

    agent_kind: AgentKind
    main_result: "MainRequestResult"
    observations: tuple[AgentObservation, ...] = ()
    tool_scope: ToolScope | None = None
    elapsed_ms: int = 0

    def to_main_result(self) -> "MainRequestResult":
        return self.main_result


@dataclass(slots=True)
class UnifiedChatRequest:
    """Request envelope for the unified chat + plugin-invocation turn."""

    message_text: str
    session_key: str | None
    budget_controller: TurnBudgetController | None
    messages: list[LLMMessage]
    report: NativeRouteReport
    scenario: str = "private_chat"
    user_id: str = ""
    group_id: str | None = None
    bot_id: str | None = None
    platform: str | None = None
    channel_id: str | None = None
    command_candidate_text: str = ""
    tools: dict[str, Any] | None = None
    tool_catalog: MixedToolCatalog | None = None
    command_context: NativeCommandExecutionContext | None = None
    context_bundle: ChatContextBundle | None = None
    context_xml: str = ""


AgentRequest = UnifiedChatRequest


class ChatInterAgent(Protocol):
    """Minimal protocol implemented by all ChatInter scenario agents."""

    async def run(self, request: AgentRequest) -> AgentResult:
        """Run one scenario-specific agent turn."""
        ...


def provider_adapter_for(
    model_name: str | None,
    *,
    api_type: str | None = None,
    capabilities: Any | None = None,
) -> ProviderCapabilityAdapter:
    """Return the provider adapter used by an agent without owning policy."""

    return ProviderCapabilityAdapter.for_model(
        model_name,
        api_type=api_type,
        capabilities=capabilities,
    )


def normalize_tool_result_output(value: Any) -> dict[str, Any]:
    """Return a small dict payload for tool/fallback observations."""

    if isinstance(value, dict):
        return dict(value)
    if value in (None, "", [], ()):
        return {}
    return {"value": value}


def fallback_text(text: str | None, *, default: str = EMPTY_REPLY_TEXT) -> str:
    """Normalize final text and provide the shared empty-response fallback."""

    return normalize_reply_text(str(text or "")) or default


def record_prompt_tokens(
    *,
    budget_controller: TurnBudgetController | None,
    messages: Sequence[object],
) -> None:
    if budget_controller is None:
        return
    budget_controller.record_prompt_use(
        estimated_tokens=estimate_prompt_tokens(messages)
    )


def estimate_prompt_tokens(messages: Sequence[object]) -> int:
    total = 0
    for message in messages:
        total += 4
        if str(getattr(message, "role", "") or "") == "tool":
            total += 40
        content = getattr(message, "content", "")
        if isinstance(content, list):
            for part in content:
                part_type = str(getattr(part, "type", "") or "").casefold()
                if part_type == "image" or getattr(part, "image_source", None):
                    total += 1_032
                    continue
                total += estimate_text_tokens(
                    str(
                        getattr(part, "text", "")
                        or getattr(part, "thought_text", "")
                        or ""
                    )
                )
        else:
            total += estimate_text_tokens(str(content or ""))
        for value in (
            getattr(message, "name", None),
            getattr(message, "tool_call_id", None),
            getattr(message, "thought_signature", None),
        ):
            if value:
                total += estimate_text_tokens(str(value))
        for call in getattr(message, "tool_calls", None) or ():
            function = getattr(call, "function", None)
            total += estimate_text_tokens(str(getattr(function, "name", "") or ""))
            arguments = getattr(function, "arguments", "") or ""
            if not isinstance(arguments, str):
                arguments = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            total += estimate_text_tokens(arguments)
    return total


def observation_from_tool_result(
    *,
    kind: str,
    tool_result: "ToolResult | None" = None,
    status: AgentObservationStatus = "ok",
    message: str = "",
) -> AgentObservation:
    output = normalize_tool_result_output(
        getattr(tool_result, "output", None) if tool_result is not None else None
    )
    return AgentObservation(
        kind=kind,
        status=status,
        message=fallback_text(message, default=""),
        metadata=output or None,
    )


def error_observation(*, kind: str, error: BaseException) -> AgentObservation:
    """Convert an exception into a boundary observation without handling policy."""

    return AgentObservation(
        kind=kind,
        status="error",
        message=f"{type(error).__name__}: {error}"[:500],
    )


__all__ = [
    "UNIFIED_CHAT_TOOL_SCOPE",
    "AgentObservation",
    "AgentRequest",
    "AgentResult",
    "ChatInterAgent",
    "ProgressHook",
    "ToolScope",
    "UnifiedChatRequest",
    "error_observation",
    "estimate_prompt_tokens",
    "fallback_text",
    "normalize_tool_result_output",
    "observation_from_tool_result",
    "provider_adapter_for",
    "record_prompt_tokens",
]
