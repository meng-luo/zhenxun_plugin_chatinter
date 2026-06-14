"""
ChatInter - 数据模型

包含 Pydantic 模型。ORM 模型由插件入口显式导入，避免纯逻辑模块
裸导入时触发数据库/NoneBot 服务初始化。
"""

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
