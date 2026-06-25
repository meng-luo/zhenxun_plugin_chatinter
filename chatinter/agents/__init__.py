"""Scenario-specific ChatInter agent wrappers.

The package defines stable boundaries for the future split between the light
plugin command router, the full superuser agent, and private chat.  P0 keeps
runtime behavior unchanged by delegating to the existing main_request module.
"""

from typing import TYPE_CHECKING

from .core import (
    PLUGIN_COMMAND_TOOL_SCOPE,
    PRIVATE_CHAT_TOOL_SCOPE,
    SUPERUSER_TOOL_SCOPE,
    AgentObservation,
    AgentRequest,
    AgentResult,
    ChatInterAgent,
    LegacyMainRequestKwargs,
    ProgressHook,
    ToolScope,
)

if TYPE_CHECKING:
    from .plugin_command_agent import PluginCommandAgent
    from .private_chat_agent import PrivateChatAgent
    from .superuser_agent import SuperuserAgent

__all__ = [
    "PLUGIN_COMMAND_TOOL_SCOPE",
    "PRIVATE_CHAT_TOOL_SCOPE",
    "SUPERUSER_TOOL_SCOPE",
    "AgentObservation",
    "AgentRequest",
    "AgentResult",
    "ChatInterAgent",
    "LegacyMainRequestKwargs",
    "PluginCommandAgent",
    "PrivateChatAgent",
    "ProgressHook",
    "SuperuserAgent",
    "ToolScope",
]
