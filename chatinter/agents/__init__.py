"""Scenario agent exports without eager cross-scenario imports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_reply_agent import ChatReplyAgent
    from .core import (
        PLUGIN_COMMAND_TOOL_SCOPE,
        PRIVATE_CHAT_TOOL_SCOPE,
        AgentObservation,
        AgentRequest,
        AgentResult,
        ChatInterAgent,
        PluginCommandRequest,
        PrivateChatRequest,
        ProgressHook,
        ToolScope,
    )
    from .plugin_command_agent import PluginCommandAgent
    from .private_chat_agent import PrivateChatAgent
    from .superuser_agent import SuperuserAgent, SuperuserRequest

__all__ = [
    "PLUGIN_COMMAND_TOOL_SCOPE",
    "PRIVATE_CHAT_TOOL_SCOPE",
    "AgentObservation",
    "AgentRequest",
    "AgentResult",
    "ChatInterAgent",
    "ChatReplyAgent",
    "PluginCommandAgent",
    "PluginCommandRequest",
    "PrivateChatAgent",
    "PrivateChatRequest",
    "ProgressHook",
    "SuperuserAgent",
    "SuperuserRequest",
    "ToolScope",
]
