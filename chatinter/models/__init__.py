"""
ChatInter - 数据模型

包含:
- Tortoise ORM 模型 (数据库表)
- Pydantic 模型 (结构化输出)
"""

# Pydantic 模型（结构化输出）
from .pydantic_models import (
    PluginInfo,
    PluginKnowledgeBase,
)

__all__ = [
    "PluginInfo",
    "PluginKnowledgeBase",
]
