"""Shared boundary types for ChatInter scenario agents.

Only cross-scenario plumbing belongs here.  Plugin selection, superuser
permissions, approval, MCP and worktree logic should stay in their own agents.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from zhenxun.services.llm import LLMMessage

from ..models.pydantic_models import PluginKnowledgeBase
from ..native_executor import ExecuteNativeRoute
from ..native_route import NativeRouteReport
from ..provider_capability import ProviderCapabilityAdapter
from ..route_text import normalize_message_text
from ..turn_runtime import TurnBudgetController, estimate_text_tokens

if TYPE_CHECKING:
    from zhenxun.services.llm.types.models import ToolResult

    from ..main_request_models import MainRequestResult

ProgressHook = Callable[[str], Awaitable[None] | None]
AgentKind = Literal["plugin_command", "superuser", "private_chat"]
AgentObservationStatus = Literal["ok", "error", "fallback"]


@dataclass(frozen=True, slots=True)
class ToolScope:
    """Explicit tool boundary for one ChatInter scenario."""

    kind: AgentKind
    allow_plugin: bool = False
    allow_superuser: bool = False
    allow_mcp: bool = False


PLUGIN_COMMAND_TOOL_SCOPE = ToolScope(kind="plugin_command", allow_plugin=True)
SUPERUSER_TOOL_SCOPE = ToolScope(
    kind="superuser",
    allow_superuser=True,
    allow_mcp=True,
)
PRIVATE_CHAT_TOOL_SCOPE = ToolScope(kind="private_chat")


class LegacyMainRequestKwargs(TypedDict, total=False):
    """Optional keyword arguments accepted by the legacy main request entry."""

    route_completed_hook: Any
    reply_hook: Any
    enable_plugin_tools: bool
    initial_command_exposure: bool
    enable_agent_tools: bool
    progress_hook: Any
    _skip_agent_wrapper: bool


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
class AgentRequest:
    """Common request envelope used by scenario-specific ChatInter agents."""

    message_text: str
    knowledge_base: PluginKnowledgeBase
    session_key: str | None
    budget_controller: TurnBudgetController | None
    has_reply: bool
    command_tools: list[Any] | None
    messages: list[LLMMessage]
    route_executor: ExecuteNativeRoute
    kwargs: LegacyMainRequestKwargs
    report: NativeRouteReport | None = None


class ChatInterAgent(Protocol):
    """Minimal protocol implemented by all ChatInter scenario agents."""

    async def run(self, request: AgentRequest) -> AgentResult:
        """Run one scenario-specific agent turn."""
        ...


def provider_adapter_for(model_name: str | None) -> ProviderCapabilityAdapter:
    """Return the provider adapter used by an agent without owning policy."""

    return ProviderCapabilityAdapter.for_model(model_name)


def normalize_tool_result_output(value: Any) -> dict[str, Any]:
    """Return a small dict payload for tool/fallback observations."""

    if isinstance(value, dict):
        return dict(value)
    if value in (None, "", [], ()):
        return {}
    return {"value": value}


def fallback_text(
    text: str | None, *, default: str = "我暂时没想好怎么回答你。"
) -> str:
    """Normalize final text and provide the shared empty-response fallback."""

    return normalize_message_text(str(text or "")) or default


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
        content = getattr(message, "content", "")
        if isinstance(content, list):
            total += sum(
                estimate_text_tokens(str(getattr(part, "text", "") or ""))
                for part in content
            )
        else:
            total += estimate_text_tokens(str(content or ""))
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
    "PLUGIN_COMMAND_TOOL_SCOPE",
    "PRIVATE_CHAT_TOOL_SCOPE",
    "SUPERUSER_TOOL_SCOPE",
    "AgentObservation",
    "AgentRequest",
    "AgentResult",
    "ChatInterAgent",
    "LegacyMainRequestKwargs",
    "ProgressHook",
    "ToolScope",
    "error_observation",
    "estimate_prompt_tokens",
    "fallback_text",
    "normalize_tool_result_output",
    "observation_from_tool_result",
    "provider_adapter_for",
    "record_prompt_tokens",
]
