"""
ChatInter - 数据模型

包含:
- Tortoise ORM 模型 (数据库表)
- Pydantic 模型 (结构化输出)
"""

# Tortoise ORM 模型（数据库表）
from .chat_history import ChatInterChatHistory

# Pydantic 模型（结构化输出）
from .pydantic_models import (
    IntentAnalysisResult,
    PluginInfo,
    PluginKnowledgeBase,
)

__all__ = [
    "ChatInterChatHistory",
    "IntentAnalysisResult",
    "PluginInfo",
    "PluginKnowledgeBase",
]
