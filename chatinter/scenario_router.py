"""Scenario routing for ChatInter.

ChatInter is intentionally a fallback layer.  The router keeps the three
runtime shapes separate so group plugin selection, private chat, and future
superuser agent tools do not leak into one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nonebot.adapters import Bot, Event

from .event_runtime import event_is_private, resolve_superuser
from .route_text import normalize_message_text


def agent_session_is_active(session_key: str) -> bool:
    from .superuser_agent.store import agent_session_is_active as is_active

    return is_active(session_key)


class ChatInterScenario(str, Enum):
    SKIP = "skip_chatinter"
    GROUP_PLUGIN_SELECTOR = "group_plugin_selector"
    PRIVATE_CHAT = "private_chat"
    SUPERUSER_AGENT = "superuser_agent"


@dataclass(frozen=True)
class ScenarioRoute:
    scenario: ChatInterScenario
    reason: str
    allow_plugin_tools: bool

    @property
    def should_handle(self) -> bool:
        return self.scenario is not ChatInterScenario.SKIP


def resolve_chatinter_scenario(
    *,
    bot: Bot,
    event: Event,
    raw_message: str,
    user_id: str,
    group_id: str | None,
    route_modules: set[str] | None,
) -> ScenarioRoute:
    """Resolve the one ChatInter runtime shape for this event."""

    if route_modules:
        return ScenarioRoute(
            scenario=ChatInterScenario.SKIP,
            reason="already_routed_by_plugin",
            allow_plugin_tools=False,
        )

    message = normalize_message_text(raw_message)
    if not message:
        return ScenarioRoute(
            scenario=ChatInterScenario.SKIP,
            reason="empty_message",
            allow_plugin_tools=False,
        )
    private_superuser = resolve_superuser(bot, str(user_id)) and event_is_private(event)
    if private_superuser and agent_session_is_active(str(user_id)):
        return ScenarioRoute(
            scenario=ChatInterScenario.SUPERUSER_AGENT,
            reason="private_superuser_agent_active",
            allow_plugin_tools=False,
        )

    from .config import mixed_chat_message_should_skip, private_plugin_tools_enabled

    if mixed_chat_message_should_skip(message):
        return ScenarioRoute(
            scenario=ChatInterScenario.SKIP,
            reason="configured_prefix_skip",
            allow_plugin_tools=False,
        )

    if group_id:
        return ScenarioRoute(
            scenario=ChatInterScenario.GROUP_PLUGIN_SELECTOR,
            reason="group_fallback",
            allow_plugin_tools=True,
        )

    return ScenarioRoute(
        scenario=ChatInterScenario.PRIVATE_CHAT,
        reason="private_user_chat",
        allow_plugin_tools=private_plugin_tools_enabled(),
    )


__all__ = [
    "ChatInterScenario",
    "ScenarioRoute",
    "resolve_chatinter_scenario",
]
