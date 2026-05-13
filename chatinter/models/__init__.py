"""
ChatInter - 数据模型

包含:
- Tortoise ORM 模型 (数据库表)
- Pydantic 模型 (结构化输出)
"""

from zhenxun.models.chat_history import ChatHistory as _ChatHistory  # noqa: F401

# Pydantic 模型（结构化输出）
from .chat_history import (
    ChatInterChatHistory,
    ChatInterMemory,
    ChatInterPersonProfile,
    ChatInterThread,
    ChatInterThreadMessage,
)
from .pydantic_models import (
    CapabilityGraphSnapshot,
    CommandCandidateFeatures,
    CommandCandidateSnapshot,
    CommandCapability,
    CommandRequirement,
    CommandSlotSpec,
    CommandToolSnapshot,
    PluginCapability,
    PluginCommandSchema,
    PluginInfo,
    PluginKnowledgeBase,
    PluginReference,
)

__all__ = [
    "CapabilityGraphSnapshot",
    "ChatInterChatHistory",
    "ChatInterMemory",
    "ChatInterPersonProfile",
    "ChatInterThread",
    "ChatInterThreadMessage",
    "CommandCandidateFeatures",
    "CommandCandidateSnapshot",
    "CommandCapability",
    "CommandRequirement",
    "CommandSlotSpec",
    "CommandToolSnapshot",
    "PluginCapability",
    "PluginCommandSchema",
    "PluginInfo",
    "PluginKnowledgeBase",
    "PluginReference",
]
