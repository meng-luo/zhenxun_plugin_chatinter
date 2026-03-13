"""
ChatInter - 插件信息注册表

收集和缓存插件信息，供意图分析使用。
只提供给 LLM 普通用户可访问的插件。
"""

import asyncio
from datetime import datetime
from typing import ClassVar

import nonebot

from zhenxun.configs.utils import PluginExtraData
from zhenxun.models.plugin_info import PluginInfo as DBPluginInfo
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase


class PluginRegistry:
    """插件信息注册表"""

    # 缓存相关
    _cache: ClassVar[dict[str, tuple[PluginKnowledgeBase, datetime]]] = {}
    _cache_ttl: ClassVar[int] = 300  # 缓存有效期（秒）
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get_plugin_knowledge_base(cls) -> PluginKnowledgeBase:
        """
        获取普通用户可访问的插件知识库

        返回:
            PluginKnowledgeBase: 插件知识库
        """
        cache_key = "normal_user"

        # 检查缓存
        async with cls._lock:
            if cache_key in cls._cache:
                cached_data, cached_time = cls._cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < cls._cache_ttl:
                    logger.debug("使用缓存的插件知识库")
                    return cached_data

        # 从数据库获取插件信息（只获取普通用户可访问的）
        knowledge_base = await cls._build_knowledge_base()

        # 更新缓存
        async with cls._lock:
            cls._cache[cache_key] = (knowledge_base, datetime.now())
            # 清理过期缓存
            cls._cleanup_cache()

        return knowledge_base

    @classmethod
    async def _build_knowledge_base(cls) -> PluginKnowledgeBase:
        """
        构建插件知识库（只包含普通用户可访问的插件）

        返回:
            PluginKnowledgeBase: 插件知识库
        """
        # 从数据库获取插件信息，只获取 NORMAL 和 DEPENDANT 类型
        db_plugins = await DBPluginInfo.filter(
            is_show=True,
            plugin_type__in=[PluginType.NORMAL, PluginType.DEPENDANT]
        ).all()

        plugins: list[PluginInfo] = []

        for db_plugin in db_plugins:
            # 获取 nonebot 插件元数据
            nb_plugin = nonebot.get_plugin_by_module_name(db_plugin.module_path)
            if not nb_plugin or not nb_plugin.metadata:
                continue

            # 解析插件额外数据
            extra_data = PluginExtraData(**nb_plugin.metadata.extra)

            # 提取命令列表
            commands = []
            if extra_data.commands:
                for cmd in extra_data.commands:
                    commands.append(cmd.command)

            # 构建插件信息
            plugin_info = PluginInfo(
                name=db_plugin.name,
                description=nb_plugin.metadata.description.strip() or "暂无描述",
                commands=commands,
                usage=nb_plugin.metadata.usage.strip() if nb_plugin.metadata.usage else None,
            )
            plugins.append(plugin_info)

        return PluginKnowledgeBase(plugins=plugins, user_role="普通用户")

    @classmethod
    def _cleanup_cache(cls):
        """清理过期的缓存"""
        now = datetime.now()
        expired_keys = [
            key
            for key, (_, cached_time) in cls._cache.items()
            if (now - cached_time).total_seconds() >= cls._cache_ttl
        ]
        for key in expired_keys:
            del cls._cache[key]

    @classmethod
    def clear_cache(cls):
        """清空所有缓存"""
        cls._cache.clear()
        logger.info("插件知识库缓存已清空")

    @classmethod
    async def preload_cache(cls):
        """
        预加载缓存 - 在插件启动时调用，提前缓存普通用户的知识库
        """
        logger.info("开始预加载 ChatInter 插件知识库缓存...")

        try:
            # 预缓存普通用户知识库
            normal_cache = await cls._build_knowledge_base()
            cls._cache["normal_user"] = (normal_cache, datetime.now())
            logger.info(
                f"ChatInter 知识库缓存预加载完成，"
                f"共缓存 {len(normal_cache.plugins)} 个插件"
            )

        except Exception as e:
            logger.error(f"预加载知识库缓存失败：{e}")


async def get_user_plugin_knowledge() -> PluginKnowledgeBase:
    """
    获取普通用户的插件知识库（便捷函数）

    返回:
        PluginKnowledgeBase: 插件知识库
    """
    return await PluginRegistry.get_plugin_knowledge_base()
