"""
ChatInter - 插件信息注册表

收集和缓存插件信息，供意图分析使用。
只提供给 LLM 普通用户可访问的插件。
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import importlib
import inspect
import re
from typing import Any, ClassVar

import nonebot

from zhenxun.configs.utils import PluginExtraData
from zhenxun.services.cache.runtime_cache import PluginInfoMemoryCache
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType

from .metadata_builder import AutoMetadataBuilder
from .models.pydantic_models import PluginInfo, PluginKnowledgeBase


@dataclass(frozen=True)
class PluginSelectionContext:
    query: str = ""
    session_id: str | None = None
    user_id: str | None = None
    group_id: str | None = None
    is_superuser: bool = False


class PluginRegistry:
    """插件信息注册表"""

    # 缓存相关
    _cache: ClassVar[dict[str, tuple[PluginKnowledgeBase, datetime]]] = {}
    _cache_ttl: ClassVar[int] = 300  # 缓存有效期（秒）
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _max_matcher_commands: ClassVar[int] = 800
    _max_discovered_commands: ClassVar[int] = 2000
    _command_discovery_entrypoints: ClassVar[tuple[str, ...]] = (
        "chatinter_command_discovery",
        "__chatinter_command_discovery__",
        "get_chatinter_commands",
        "__chatinter_skill_commands__",
    )
    _command_placeholder_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*(?:\[[^\]]+\]|<[^>]+>|\{[^}]+\})\s*"
    )
    _self_only_command_keywords: ClassVar[tuple[str, ...]] = ("签到", "打卡", "补签")
    _session_plugin_overrides: ClassVar[dict[str, dict[str, bool]]] = {}
    _group_plugin_overrides: ClassVar[dict[str, dict[str, bool]]] = {}

    @classmethod
    async def get_plugin_knowledge_base(
        cls, force_refresh: bool = False
    ) -> PluginKnowledgeBase:
        """
        获取普通用户可访问的插件知识库

        返回:
            PluginKnowledgeBase: 插件知识库
        """
        cache_key = "normal_user"

        # 检查缓存
        async with cls._lock:
            if not force_refresh and cache_key in cls._cache:
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
    async def get_runtime_plugin_knowledge_base(cls) -> PluginKnowledgeBase:
        plugins = cls._deduplicate_plugins(await cls._collect_runtime_plugins())
        return PluginKnowledgeBase(plugins=plugins, user_role="普通用户")

    @classmethod
    async def _build_knowledge_base(cls) -> PluginKnowledgeBase:
        """
        构建插件知识库（只包含普通用户可访问的插件）

        返回:
            PluginKnowledgeBase: 插件知识库
        """
        plugins_by_module = await cls._collect_runtime_plugins()
        await cls._merge_database_plugins(plugins_by_module)
        plugins = cls._deduplicate_plugins(plugins_by_module)
        return PluginKnowledgeBase(plugins=plugins, user_role="普通用户")

    @classmethod
    def _parse_extra_data(cls, raw_extra: object) -> PluginExtraData:
        try:
            data = raw_extra if isinstance(raw_extra, dict) else {}
            return PluginExtraData(**data)
        except Exception:
            return PluginExtraData()

    @classmethod
    def _extract_command_meta(
        cls,
        extra_data: PluginExtraData,
    ) -> list[PluginInfo.PluginCommandMeta]:
        command_metas: list[PluginInfo.PluginCommandMeta] = []
        for raw in extra_data.commands or []:
            command_text = str(getattr(raw, "command", "") or "").strip()
            if not command_text:
                continue
            params = [
                str(param).strip()
                for param in (getattr(raw, "params", []) or [])
            ]
            params = [param for param in params if param]
            params = cls._merge_unique_strings(
                params,
                cls._extract_command_params_from_text(command_text),
            )
            examples: list[str] = []
            for item in getattr(raw, "examples", []) or []:
                exec_text = str(getattr(item, "exec", "") or "").strip()
                if exec_text:
                    examples.append(exec_text)
            command_metas.append(
                cls._with_command_meta_defaults(
                    command=command_text,
                    params=params,
                    examples=examples,
                    text_min=cls._safe_int(getattr(raw, "text_min", None)),
                    text_max=cls._safe_int(getattr(raw, "text_max", None)),
                    image_min=cls._safe_int(getattr(raw, "image_min", None)),
                    image_max=cls._safe_int(getattr(raw, "image_max", None)),
                    allow_at=cls._safe_bool(getattr(raw, "allow_at", None)),
                    actor_scope=getattr(raw, "actor_scope", None),
                    target_requirement=getattr(raw, "target_requirement", None),
                    target_sources=getattr(raw, "target_sources", None),
                    allow_sticky_arg=getattr(raw, "allow_sticky_arg", None),
                )
            )
        return command_metas

    @classmethod
    def _extract_command_params_from_text(cls, command_text: str) -> list[str]:
        normalized = str(command_text or "").strip()
        if not normalized:
            return []
        params: list[str] = []
        for raw_token in re.findall(r"[\[\(<｟]([^]\)>｠]+)[\]\)>｠]", normalized):
            token = str(raw_token or "").strip()
            if not token:
                continue
            token = token.lstrip("?*+")
            token = token.split("=", 1)[0]
            token = token.split(":", 1)[0]
            token = token.split(" ", 1)[0]
            token = cls._normalize_command(token)
            if token:
                params.append(token)
        return cls._merge_unique_strings(params, [])

    @classmethod
    def _build_command_meta_from_commands(
        cls,
        commands: list[str],
    ) -> list[PluginInfo.PluginCommandMeta]:
        result: list[PluginInfo.PluginCommandMeta] = []
        for command in commands:
            command_text = str(command).strip()
            if not command_text:
                continue
            result.append(cls._with_command_meta_defaults(command=command_text))
        return result

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return None

    @staticmethod
    def _normalize_access_level(value: object) -> str:
        level = str(value or "").strip().lower()
        if level in {"public", "admin", "superuser", "restricted"}:
            return level
        return "public"

    @classmethod
    def _merge_access_level(
        cls,
        left: object = None,
        right: object = None,
    ) -> str:
        left_level = cls._normalize_access_level(left)
        right_level = cls._normalize_access_level(right)
        if left_level == right_level:
            return left_level
        if "restricted" in {left_level, right_level}:
            return "restricted"
        levels = {left_level, right_level} - {"public"}
        if not levels:
            return "public"
        if levels == {"admin"}:
            return "admin"
        if levels == {"superuser"}:
            return "superuser"
        return "restricted"

    @classmethod
    def _is_public_command_meta(cls, meta: PluginInfo.PluginCommandMeta) -> bool:
        return cls._normalize_access_level(getattr(meta, "access_level", None)) == "public"

    @classmethod
    def _filter_public_command_meta(
        cls,
        metas: list[PluginInfo.PluginCommandMeta],
    ) -> list[PluginInfo.PluginCommandMeta]:
        return [meta for meta in metas if cls._is_public_command_meta(meta)]

    @classmethod
    def _infer_actor_scope(cls, command: str, actor_scope: object) -> str:
        normalize = str(command or "").strip()
        parsed = str(actor_scope or "").strip().lower()
        if parsed in {"self_only", "allow_other"}:
            return parsed
        if normalize.startswith("我的") or any(
            keyword in normalize for keyword in cls._self_only_command_keywords
        ):
            return "self_only"
        return "allow_other"

    @staticmethod
    def _infer_target_requirement(
        *,
        actor_scope: str,
        target_requirement: object,
        allow_at: bool | None,
        image_min: int | None,
    ) -> str:
        parsed = str(target_requirement or "").strip().lower()
        if parsed in {"none", "optional", "required"}:
            return parsed
        if actor_scope == "self_only":
            return "none"
        if allow_at or (image_min or 0) > 0:
            return "optional"
        return "none"

    @staticmethod
    def _normalize_target_sources(
        *,
        actor_scope: str,
        allow_at: bool | None,
        target_sources: object,
    ) -> list[str]:
        if isinstance(target_sources, (list, tuple)):
            parsed = [
                str(item).strip().lower()
                for item in target_sources
                if str(item).strip().lower() in {"at", "reply", "nickname", "self"}
            ]
        else:
            parsed = []
        if not parsed:
            if actor_scope == "self_only":
                return ["self"]
            if allow_at:
                return ["at", "reply", "nickname"]
            return []
        deduped: list[str] = []
        for item in parsed:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _infer_allow_sticky_arg(
        *,
        allow_sticky_arg: object,
        allow_at: bool | None,
        text_max: int | None,
    ) -> bool:
        parsed = str(allow_sticky_arg or "").strip().lower()
        if parsed in {"1", "true", "yes", "on"}:
            return True
        if parsed in {"0", "false", "no", "off"}:
            return False
        return bool(allow_at and (text_max is None or text_max <= 0))

    @classmethod
    def _with_command_meta_defaults(
        cls,
        *,
        command: str,
        aliases: list[str] | tuple[str, ...] | None = None,
        params: list[str] | tuple[str, ...] | None = None,
        examples: list[str] | tuple[str, ...] | None = None,
        text_min: int | None = None,
        text_max: int | None = None,
        image_min: int | None = None,
        image_max: int | None = None,
        allow_at: bool | None = None,
        actor_scope: object = None,
        target_requirement: object = None,
        target_sources: object = None,
        allow_sticky_arg: object = None,
        access_level: object = None,
    ) -> PluginInfo.PluginCommandMeta:
        normalized_command = str(command or "").strip()
        resolved_actor_scope = cls._infer_actor_scope(normalized_command, actor_scope)
        resolved_target_requirement = cls._infer_target_requirement(
            actor_scope=resolved_actor_scope,
            target_requirement=target_requirement,
            allow_at=allow_at,
            image_min=image_min,
        )
        resolved_target_sources = cls._normalize_target_sources(
            actor_scope=resolved_actor_scope,
            allow_at=allow_at,
            target_sources=target_sources,
        )
        resolved_allow_sticky_arg = cls._infer_allow_sticky_arg(
            allow_sticky_arg=allow_sticky_arg,
            allow_at=allow_at,
            text_max=text_max,
        )
        resolved_access_level = cls._normalize_access_level(access_level)
        return PluginInfo.PluginCommandMeta(
            command=normalized_command,
            aliases=cls._merge_unique_strings(aliases, []),
            params=cls._merge_unique_strings(params, []),
            examples=cls._merge_unique_strings(examples, []),
            text_min=text_min,
            text_max=text_max,
            image_min=image_min,
            image_max=image_max,
            allow_at=allow_at,
            actor_scope=resolved_actor_scope,
            target_requirement=resolved_target_requirement,
            target_sources=resolved_target_sources,
            allow_sticky_arg=resolved_allow_sticky_arg,
            access_level=resolved_access_level,
        )

    @classmethod
    def _meta_to_dict(cls, meta: PluginInfo.PluginCommandMeta) -> dict:
        if hasattr(meta, "model_dump"):
            return meta.model_dump()
        if hasattr(meta, "dict"):
            return meta.dict()
        return {
            "command": str(getattr(meta, "command", "") or "").strip(),
            "aliases": list(getattr(meta, "aliases", []) or []),
            "params": list(getattr(meta, "params", []) or []),
            "examples": list(getattr(meta, "examples", []) or []),
            "text_min": cls._safe_int(getattr(meta, "text_min", None)),
            "text_max": cls._safe_int(getattr(meta, "text_max", None)),
            "image_min": cls._safe_int(getattr(meta, "image_min", None)),
            "image_max": cls._safe_int(getattr(meta, "image_max", None)),
            "allow_at": cls._safe_bool(getattr(meta, "allow_at", None)),
            "actor_scope": str(getattr(meta, "actor_scope", "") or "").strip().lower()
            or None,
            "target_requirement": str(
                getattr(meta, "target_requirement", "") or ""
            ).strip().lower()
            or None,
            "target_sources": list(getattr(meta, "target_sources", []) or []),
            "allow_sticky_arg": cls._safe_bool(getattr(meta, "allow_sticky_arg", None)),
            "access_level": cls._normalize_access_level(
                getattr(meta, "access_level", None)
            ),
        }

    @staticmethod
    def _merge_unique_strings(
        left: list[str] | tuple[str, ...] | None,
        right: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        result: list[str] = []
        for collection in (left or [], right or []):
            if isinstance(collection, (list, tuple)):
                iterable = collection
            else:
                iterable = [collection]
            for value in iterable:
                text = str(value).strip()
                if text and text not in result:
                    result.append(text)
        return result

    @classmethod
    def _merge_command_meta_groups(
        cls,
        *groups: list[PluginInfo.PluginCommandMeta],
    ) -> list[PluginInfo.PluginCommandMeta]:
        merged: dict[str, PluginInfo.PluginCommandMeta] = {}
        for metas in groups:
            for meta in metas:
                command_text = str(getattr(meta, "command", "") or "").strip()
                if not command_text:
                    continue
                key = command_text.lower()
                current = merged.get(key)
                if current is None:
                    merged[key] = cls._with_command_meta_defaults(
                        **cls._meta_to_dict(meta)
                    )
                    continue
                left = cls._meta_to_dict(current)
                right = cls._meta_to_dict(meta)
                merged[key] = cls._with_command_meta_defaults(
                    command=left.get("command") or right.get("command") or command_text,
                    aliases=cls._merge_unique_strings(
                        left.get("aliases"), right.get("aliases")
                    ),
                    params=cls._merge_unique_strings(
                        left.get("params"), right.get("params")
                    ),
                    examples=cls._merge_unique_strings(
                        left.get("examples"), right.get("examples")
                    ),
                    text_min=left.get("text_min")
                    if left.get("text_min") is not None
                    else right.get("text_min"),
                    text_max=left.get("text_max")
                    if left.get("text_max") is not None
                    else right.get("text_max"),
                    image_min=left.get("image_min")
                    if left.get("image_min") is not None
                    else right.get("image_min"),
                    image_max=left.get("image_max")
                    if left.get("image_max") is not None
                    else right.get("image_max"),
                    allow_at=left.get("allow_at")
                    if left.get("allow_at") is not None
                    else right.get("allow_at"),
                    actor_scope=left.get("actor_scope") or right.get("actor_scope"),
                    target_requirement=left.get("target_requirement")
                    or right.get("target_requirement"),
                    target_sources=cls._merge_unique_strings(
                        left.get("target_sources"), right.get("target_sources")
                    ),
                    allow_sticky_arg=left.get("allow_sticky_arg")
                    if left.get("allow_sticky_arg") is not None
                    else right.get("allow_sticky_arg"),
                    access_level=cls._merge_access_level(
                        left.get("access_level"), right.get("access_level")
                    ),
                )
        return sorted(merged.values(), key=lambda item: (len(item.command), item.command))

    @classmethod
    def _command_meta_richness(
        cls,
        meta: PluginInfo.PluginCommandMeta,
    ) -> tuple[int, int, int, int, int, int]:
        aliases = len(getattr(meta, "aliases", []) or [])
        params = len(getattr(meta, "params", []) or [])
        examples = len(getattr(meta, "examples", []) or [])
        text_score = sum(
            1
            for value in (
                getattr(meta, "text_min", None),
                getattr(meta, "text_max", None),
                getattr(meta, "image_min", None),
                getattr(meta, "image_max", None),
            )
            if value is not None
        )
        sticky = int(bool(getattr(meta, "allow_sticky_arg", False)))
        allow_at = int(bool(getattr(meta, "allow_at", False)))
        return (params, text_score, aliases, examples, sticky, allow_at)

    @classmethod
    def _canonicalize_command_meta_groups(
        cls,
        metas: list[PluginInfo.PluginCommandMeta],
    ) -> list[PluginInfo.PluginCommandMeta]:
        if len(metas) <= 1:
            return metas

        command_to_index: dict[str, int] = {}
        for index, meta in enumerate(metas):
            command = str(getattr(meta, "command", "") or "").strip()
            if command:
                command_to_index[command.casefold()] = index

        parent: dict[int, int] = {index: index for index in range(len(metas))}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for index, meta in enumerate(metas):
            aliases = {
                str(alias).strip().casefold()
                for alias in (getattr(meta, "aliases", []) or [])
                if str(alias).strip()
            }
            for alias in aliases:
                other_index = command_to_index.get(alias)
                if other_index is None or other_index == index:
                    continue
                union(index, other_index)

        groups: dict[int, list[PluginInfo.PluginCommandMeta]] = {}
        for index, meta in enumerate(metas):
            groups.setdefault(find(index), []).append(meta)

        canonicalized: list[PluginInfo.PluginCommandMeta] = []
        for items in groups.values():
            if len(items) == 1:
                canonicalized.append(items[0])
                continue
            canonical = max(items, key=cls._command_meta_richness)
            payload = cls._meta_to_dict(canonical)
            for item in items:
                if item is canonical:
                    continue
                item_payload = cls._meta_to_dict(item)
                payload["aliases"] = cls._merge_unique_strings(
                    payload.get("aliases"), [item_payload.get("command") or ""]
                )
                payload["aliases"] = cls._merge_unique_strings(
                    payload.get("aliases"), item_payload.get("aliases")
                )
                payload["params"] = cls._merge_unique_strings(
                    payload.get("params"), item_payload.get("params")
                )
                payload["examples"] = cls._merge_unique_strings(
                    payload.get("examples"), item_payload.get("examples")
                )
                payload["target_sources"] = cls._merge_unique_strings(
                    payload.get("target_sources"), item_payload.get("target_sources")
                )
                payload["access_level"] = cls._merge_access_level(
                    payload.get("access_level"), item_payload.get("access_level")
                )
                for field in (
                    "text_min",
                    "text_max",
                    "image_min",
                    "image_max",
                    "allow_at",
                    "actor_scope",
                    "target_requirement",
                    "allow_sticky_arg",
                    "access_level",
                ):
                    if payload.get(field) is None and item_payload.get(field) is not None:
                        payload[field] = item_payload.get(field)
            canonicalized.append(cls._with_command_meta_defaults(**payload))

        return cls._merge_command_meta_groups(canonicalized)

    @classmethod
    def _fold_plugin_alias_command_meta(
        cls,
        metas: list[PluginInfo.PluginCommandMeta],
        *,
        plugin_aliases: list[str] | tuple[str, ...] | None = None,
    ) -> list[PluginInfo.PluginCommandMeta]:
        alias_heads = {
            *(
                cls._normalize_command(alias).casefold()
                for alias in (plugin_aliases or [])
                if cls._normalize_command(alias)
            ),
        }
        alias_heads = {head for head in alias_heads if head}
        if not alias_heads or len(metas) <= 1:
            return metas

        target_candidates = [
            meta
            for meta in metas
            if cls._normalize_command(getattr(meta, "command", "")).casefold()
            not in alias_heads
        ]
        if not target_candidates:
            return metas

        alias_items = [
            meta
            for meta in metas
            if cls._normalize_command(getattr(meta, "command", "")).casefold()
            in alias_heads
        ]
        if not alias_items:
            return metas

        target = max(target_candidates, key=cls._command_meta_richness)
        target_payload = cls._meta_to_dict(target)
        changed = False

        for item in alias_items:
            if item is target:
                continue
            item_payload = cls._meta_to_dict(item)
            if not item_payload.get("command"):
                continue
            changed = True
            target_payload["aliases"] = cls._merge_unique_strings(
                target_payload.get("aliases"),
                [item_payload.get("command") or ""],
            )
            target_payload["aliases"] = cls._merge_unique_strings(
                target_payload.get("aliases"), item_payload.get("aliases")
            )
            target_payload["params"] = cls._merge_unique_strings(
                target_payload.get("params"), item_payload.get("params")
            )
            target_payload["examples"] = cls._merge_unique_strings(
                target_payload.get("examples"), item_payload.get("examples")
            )
            target_payload["target_sources"] = cls._merge_unique_strings(
                target_payload.get("target_sources"), item_payload.get("target_sources")
            )
            target_payload["access_level"] = cls._merge_access_level(
                target_payload.get("access_level"), item_payload.get("access_level")
            )
            for field in (
                "text_min",
                "text_max",
                "image_min",
                "image_max",
                "allow_at",
                "actor_scope",
                "target_requirement",
                "allow_sticky_arg",
                "access_level",
            ):
                if (
                    target_payload.get(field) is None
                    and item_payload.get(field) is not None
                ):
                    target_payload[field] = item_payload.get(field)

        if not changed:
            return metas

        folded: list[PluginInfo.PluginCommandMeta] = [target]
        for item in metas:
            if item is target:
                continue
            command_fold = cls._normalize_command(getattr(item, "command", "")).casefold()
            if command_fold in alias_heads:
                continue
            folded.append(item)
        folded[0] = cls._with_command_meta_defaults(**target_payload)
        return cls._merge_command_meta_groups(folded)

    @classmethod
    def _load_plugin_module(cls, module_name: str, loaded_plugin=None):
        module_obj = getattr(loaded_plugin, "module", None)
        if module_obj is not None:
            return module_obj
        if not module_name:
            return None
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    @classmethod
    def _parse_discovery_item(
        cls,
        item: object,
    ) -> PluginInfo.PluginCommandMeta | None:
        if isinstance(item, str):
            command_text = str(item).strip()
            if command_text:
                return cls._with_command_meta_defaults(command=command_text)
            return None
        if not isinstance(item, dict):
            return None
        command_text = str(item.get("command") or item.get("head") or "").strip()
        if not command_text:
            return None
        schema = item.get("schema")
        if not isinstance(schema, dict):
            schema = {}
        text_schema = schema.get("text")
        image_schema = schema.get("image")
        if not isinstance(text_schema, dict):
            text_schema = {}
        if not isinstance(image_schema, dict):
            image_schema = {}
        aliases = item.get("aliases")
        params = item.get("params")
        examples = item.get("examples")
        if not isinstance(aliases, list | tuple):
            aliases = []
        if not isinstance(params, list | tuple):
            params = []
        if not isinstance(examples, list | tuple):
            examples = []
        normalized_examples: list[str] = []
        for example in examples:
            if isinstance(example, dict):
                text = str(example.get("exec") or example.get("example") or "").strip()
            else:
                text = str(example).strip()
            if text:
                normalized_examples.append(text)
        return cls._with_command_meta_defaults(
            command=command_text,
            aliases=[
                str(alias).strip() for alias in aliases if str(alias or "").strip()
            ],
            params=[str(param).strip() for param in params if str(param or "").strip()],
            examples=normalized_examples,
            text_min=cls._safe_int(item.get("text_min"))
            if item.get("text_min") is not None
            else cls._safe_int(text_schema.get("min")),
            text_max=cls._safe_int(item.get("text_max"))
            if item.get("text_max") is not None
            else cls._safe_int(text_schema.get("max")),
            image_min=cls._safe_int(item.get("image_min"))
            if item.get("image_min") is not None
            else cls._safe_int(image_schema.get("min")),
            image_max=cls._safe_int(item.get("image_max"))
            if item.get("image_max") is not None
            else cls._safe_int(image_schema.get("max")),
            allow_at=cls._safe_bool(item.get("allow_at"))
            if item.get("allow_at") is not None
            else cls._safe_bool(schema.get("allow_at")),
            actor_scope=item.get("actor_scope", schema.get("actor_scope")),
            target_requirement=item.get(
                "target_requirement", schema.get("target_requirement")
            ),
            target_sources=item.get("target_sources", schema.get("target_sources")),
            allow_sticky_arg=item.get(
                "allow_sticky_arg", schema.get("allow_sticky_arg")
            ),
            access_level=item.get("access_level", schema.get("access_level")),
        )

    @classmethod
    def _parse_discovery_payload(
        cls,
        payload: object,
    ) -> list[PluginInfo.PluginCommandMeta]:
        items: list[object] = []
        if isinstance(payload, dict):
            candidates = payload.get("commands")
            if isinstance(candidates, list | tuple):
                items.extend(list(candidates))
            elif payload.get("command"):
                items.append(payload)
        elif isinstance(payload, list | tuple):
            items.extend(list(payload))
        elif hasattr(payload, "commands"):
            candidates = getattr(payload, "commands", None)
            if isinstance(candidates, list | tuple):
                items.extend(list(candidates))
        metas: list[PluginInfo.PluginCommandMeta] = []
        for item in items:
            meta = cls._parse_discovery_item(item)
            if meta is not None:
                metas.append(meta)
        if len(metas) > cls._max_discovered_commands:
            metas = metas[: cls._max_discovered_commands]
        return metas

    @classmethod
    async def _discover_command_meta_from_plugin(
        cls,
        module_name: str,
        loaded_plugin=None,
    ) -> list[PluginInfo.PluginCommandMeta]:
        module_obj = cls._load_plugin_module(module_name, loaded_plugin)
        if module_obj is None:
            return []
        discovered: list[PluginInfo.PluginCommandMeta] = []
        for entrypoint in cls._command_discovery_entrypoints:
            candidate = getattr(module_obj, entrypoint, None)
            if candidate is None:
                continue
            payload = None
            if callable(candidate):
                try:
                    payload = candidate()
                    if inspect.isawaitable(payload):
                        payload = await payload
                except Exception as exc:
                    logger.debug(
                        "ChatInter 动态命令发现调用失败: "
                        f"module={module_name}, entrypoint={entrypoint}, error={exc}"
                    )
                    continue
            else:
                payload = candidate
            if inspect.isawaitable(payload):
                try:
                    payload = await payload
                except Exception as exc:
                    logger.debug(
                        "ChatInter 动态命令发现 await 失败: "
                        f"module={module_name}, entrypoint={entrypoint}, error={exc}"
                    )
                    continue
            discovered.extend(cls._parse_discovery_payload(payload))
        discovered.extend(
            cls._parse_discovery_payload(
                await AutoMetadataBuilder.build(
                    module_name=module_name,
                    module_obj=module_obj,
                    loaded_plugin=loaded_plugin,
                )
            )
        )
        return cls._merge_command_meta_groups(discovered)

    @classmethod
    def _extract_commands(
        cls,
        extra_data: PluginExtraData,
        command_meta: list[PluginInfo.PluginCommandMeta] | None = None,
    ) -> list[str]:
        commands: list[str] = []
        for meta in command_meta or []:
            cls._append_command(commands, meta.command)
            for alias in meta.aliases:
                cls._append_command(commands, alias)

        raw_aliases = extra_data.aliases or []
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        for alias in raw_aliases:
            if alias:
                cls._append_command(commands, str(alias))
        return commands

    @classmethod
    def _is_runtime_plugin_allowed(cls, module_name: str, loaded_plugin=None) -> bool:
        return bool(module_name and loaded_plugin is not None)

    @classmethod
    async def _build_plugin_info(
        cls,
        *,
        module_name: str,
        metadata=None,
        extra_data: PluginExtraData,
        loaded_plugin=None,
        fallback_name: str | None = None,
        admin_level: int | None = None,
        limit_superuser: bool | None = None,
    ) -> PluginInfo | None:
        command_meta = cls._extract_command_meta(extra_data)
        discovered_meta = await cls._discover_command_meta_from_plugin(
            module_name, loaded_plugin
        )
        command_meta = cls._merge_command_meta_groups(command_meta, discovered_meta)
        resolved_name = (
            str(fallback_name or getattr(metadata, "name", "") or "").strip()
            or str(getattr(loaded_plugin, "name", "") or "").strip()
            or module_name.rsplit(".", 1)[-1]
        )
        commands = cls._extract_commands(extra_data, command_meta)
        if loaded_plugin is not None:
            matcher_commands = cls._extract_commands_from_matchers(loaded_plugin)
            if matcher_commands:
                commands = cls._merge_unique_strings(commands, matcher_commands)
            else:
                matcher_commands = []
            if not commands:
                commands = matcher_commands
            matcher_meta = cls._build_command_meta_from_commands(commands)
            command_meta = cls._merge_command_meta_groups(command_meta, matcher_meta)
            if matcher_commands:
                commands, command_meta = cls._filter_to_matcher_executable(
                    commands=commands,
                    command_meta=command_meta,
                    matcher_commands=matcher_commands,
                )
        command_meta = cls._fold_plugin_alias_command_meta(
            command_meta,
            plugin_aliases=extra_data.aliases,
        )
        command_meta = cls._canonicalize_command_meta_groups(command_meta)
        command_meta = cls._filter_public_command_meta(command_meta)
        commands = cls._extract_commands(extra_data, command_meta)
        if not commands:
            return None

        setting = extra_data.setting
        resolved_limit_superuser = (
            bool(limit_superuser)
            if limit_superuser is not None
            else bool(getattr(setting, "limit_superuser", False))
        )
        resolved_admin_level = (
            admin_level if admin_level is not None else extra_data.admin_level
        )
        resolved_description = (
            str(getattr(metadata, "description", "") or "").strip()
            or "暂无描述"
        )
        resolved_usage = (
            str(getattr(metadata, "usage", "") or "").strip()
            if getattr(metadata, "usage", None)
            else None
        )
        return PluginInfo(
            module=module_name,
            name=resolved_name,
            description=resolved_description,
            commands=commands,
            aliases=sorted(
                {
                    str(alias).strip()
                    for alias in (extra_data.aliases or [])
                    if str(alias).strip()
                }
            ),
            command_meta=command_meta,
            usage=resolved_usage,
            admin_level=resolved_admin_level,
            limit_superuser=resolved_limit_superuser,
        )

    @classmethod
    def _is_public_plugin(cls, extra_data: PluginExtraData) -> bool:
        if int(extra_data.admin_level or 0) > 0:
            return False
        if bool(getattr(extra_data, "limit_superuser", False)):
            return False
        plugin_type = getattr(extra_data, "plugin_type", PluginType.NORMAL)
        if plugin_type in {
            PluginType.SUPERUSER,
            PluginType.ADMIN,
            PluginType.SUPER_AND_ADMIN,
            PluginType.HIDDEN,
        }:
            return False
        setting = extra_data.setting
        if isinstance(setting, dict):
            return not bool(setting.get("limit_superuser", False))
        if bool(getattr(setting, "limit_superuser", False)):
            return False
        return True

    @classmethod
    async def _collect_runtime_plugins(cls) -> dict[str, PluginInfo]:
        plugins_by_module: dict[str, PluginInfo] = {}
        for loaded_plugin in nonebot.get_loaded_plugins():
            module_name = str(getattr(loaded_plugin, "module_name", "") or "").strip()
            if not module_name or module_name in plugins_by_module:
                continue
            if not cls._is_runtime_plugin_allowed(module_name, loaded_plugin):
                continue
            metadata = getattr(loaded_plugin, "metadata", None)
            extra_data = cls._parse_extra_data(getattr(metadata, "extra", None))
            if not cls._is_public_plugin(extra_data):
                continue
            plugin_info = await cls._build_plugin_info(
                module_name=module_name,
                metadata=metadata,
                extra_data=extra_data,
                loaded_plugin=loaded_plugin,
            )
            if plugin_info is not None:
                plugins_by_module[module_name] = plugin_info
        return plugins_by_module

    @classmethod
    async def _merge_database_plugins(
        cls,
        plugins_by_module: dict[str, PluginInfo],
    ) -> None:
        try:
            db_plugins = await cls._load_db_plugins()
        except Exception as exc:
            logger.debug(
                "ChatInter 插件知识库数据库增强失败，已回退到运行时插件: "
                f"{exc}"
            )
            return

        for db_plugin in db_plugins.values():
            if db_plugin.plugin_type in {
                PluginType.SUPERUSER,
                PluginType.ADMIN,
                PluginType.SUPER_AND_ADMIN,
                PluginType.HIDDEN,
            }:
                continue
            if int(db_plugin.admin_level or 0) > 0:
                continue
            if bool(db_plugin.limit_superuser):
                continue

            module_candidates = [
                str(db_plugin.module or "").strip(),
                str(db_plugin.module_path or "").strip(),
            ]
            module_name = next((item for item in module_candidates if item), "")
            if not module_name:
                continue

            runtime_plugin = plugins_by_module.get(module_name)
            if runtime_plugin is not None:
                plugins_by_module[module_name] = runtime_plugin.model_copy(
                    update={
                        "name": str(db_plugin.name or runtime_plugin.name).strip()
                        or runtime_plugin.name,
                        "admin_level": db_plugin.admin_level,
                        "limit_superuser": bool(db_plugin.limit_superuser),
                    }
                )
                continue

            if not db_plugin.load_status:
                continue

            nb_plugin = nonebot.get_plugin_by_module_name(str(db_plugin.module_path))
            if not nb_plugin or not nb_plugin.metadata:
                continue
            extra_data = cls._parse_extra_data(nb_plugin.metadata.extra)
            if not cls._is_public_plugin(extra_data):
                continue
            plugin_info = await cls._build_plugin_info(
                module_name=module_name,
                metadata=nb_plugin.metadata,
                extra_data=extra_data,
                loaded_plugin=nb_plugin,
                fallback_name=str(db_plugin.name or "").strip() or None,
                admin_level=db_plugin.admin_level,
                limit_superuser=bool(db_plugin.limit_superuser),
            )
            if plugin_info is not None:
                plugins_by_module[module_name] = plugin_info

    @classmethod
    async def _load_db_plugins(cls):
        await PluginInfoMemoryCache.ensure_loaded()
        return await PluginInfoMemoryCache.get_all() or {}

    @classmethod
    def _deduplicate_plugins(
        cls,
        plugins_by_module: dict[str, PluginInfo],
    ) -> list[PluginInfo]:
        def module_priority(module_name: str) -> tuple[int, int]:
            if module_name.startswith("zhenxun.plugins."):
                return (0, -len(module_name))
            if module_name.startswith("zhenxun.builtin_plugins."):
                return (1, -len(module_name))
            return (2, -len(module_name))

        ordered = sorted(
            plugins_by_module.values(),
            key=lambda item: (
                module_priority(item.module),
                -(len(item.commands) + len(item.aliases)),
                item.module,
            ),
        )

        deduplicated: list[PluginInfo] = []
        seen_fingerprints: set[tuple[str, tuple[str, ...]]] = set()
        for plugin in ordered:
            command_fingerprint = tuple(
                sorted(
                    {
                        cmd.strip().lower()
                        for cmd in plugin.commands
                        if cmd.strip()
                    }
                )
            )
            fingerprint = (
                plugin.name.strip().lower(),
                command_fingerprint,
            )
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            deduplicated.append(plugin)
        return sorted(deduplicated, key=lambda item: item.module)

    @classmethod
    def _extract_commands_from_matchers(cls, nb_plugin) -> list[str]:
        commands: list[str] = []
        seen: set[str] = set()
        matcher_meta = AutoMetadataBuilder._extract_matcher_command_data(
            loaded_plugin=nb_plugin,
        )
        for payload in matcher_meta:
            candidates = [str(payload.get("command") or "").strip()]
            raw_aliases = payload.get("aliases") or []
            if isinstance(raw_aliases, (set, list, tuple, frozenset)):
                candidates.extend(
                    str(alias).strip() for alias in raw_aliases if str(alias).strip()
                )
            for candidate in candidates:
                normalized = cls._normalize_command(candidate)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                commands.append(normalized)
        commands.sort(key=lambda cmd: (len(cmd), cmd))
        if len(commands) > cls._max_matcher_commands:
            commands = commands[: cls._max_matcher_commands]
        return commands

    @classmethod
    def _build_matcher_command_lookup(
        cls,
        matcher_commands: list[str],
    ) -> set[str]:
        lookup: set[str] = set()
        for raw in matcher_commands:
            normalized = cls._normalize_command(raw)
            if not normalized:
                continue
            lookup.add(normalized.casefold())
            lookup.add(normalized.split(" ", 1)[0].casefold())
        return lookup

    @classmethod
    def _command_matches_matcher_lookup(
        cls,
        command_text: str,
        matcher_lookup: set[str],
    ) -> bool:
        normalized = cls._normalize_command(command_text)
        if not normalized:
            return False
        folded = normalized.casefold()
        if folded in matcher_lookup:
            return True
        return normalized.split(" ", 1)[0].casefold() in matcher_lookup

    @classmethod
    def _filter_to_matcher_executable(
        cls,
        *,
        commands: list[str],
        command_meta: list[PluginInfo.PluginCommandMeta],
        matcher_commands: list[str],
    ) -> tuple[list[str], list[PluginInfo.PluginCommandMeta]]:
        if not matcher_commands:
            return commands, command_meta

        matcher_lookup = cls._build_matcher_command_lookup(matcher_commands)
        filtered_commands = [
            command
            for command in commands
            if cls._command_matches_matcher_lookup(command, matcher_lookup)
        ]

        filtered_meta: list[PluginInfo.PluginCommandMeta] = []
        for meta in command_meta:
            payload = cls._meta_to_dict(meta)
            command_text = str(payload.get("command") or "").strip()
            matched_command = cls._command_matches_matcher_lookup(
                command_text, matcher_lookup
            )

            original_aliases = payload.get("aliases", [])
            matched_aliases = [
                alias
                for alias in original_aliases
                if cls._command_matches_matcher_lookup(alias, matcher_lookup)
            ]

            if not matched_command and not matched_aliases:
                continue

            if not matched_command and matched_aliases:
                payload["command"] = matched_aliases[0]
                matched_aliases = matched_aliases[1:]
            normalized_command = cls._normalize_command(str(payload.get("command", "")))
            alias_source = original_aliases if matched_command else matched_aliases
            payload["aliases"] = [
                alias
                for alias in alias_source
                if cls._normalize_command(alias).casefold()
                != normalized_command.casefold()
            ]
            filtered_meta.append(cls._with_command_meta_defaults(**payload))

        filtered_commands = cls._merge_unique_strings(filtered_commands, matcher_commands)

        if not filtered_meta:
            filtered_meta = cls._build_command_meta_from_commands(filtered_commands)
        else:
            filtered_meta = cls._merge_command_meta_groups(
                filtered_meta,
                cls._build_command_meta_from_commands(filtered_commands),
            )

        filtered_commands = cls._extract_commands(
            PluginExtraData(),
            filtered_meta,
        )
        if not filtered_commands:
            filtered_commands = matcher_commands[:]
        return filtered_commands, filtered_meta

    @classmethod
    def _normalize_command(cls, command: str) -> str:
        text = str(command or "").strip()
        if not text:
            return ""
        text = cls._command_placeholder_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        return text

    @classmethod
    def _append_command(cls, commands: list[str], command: str) -> None:
        normalized = cls._normalize_command(command)
        if not normalized:
            return
        if normalized not in commands:
            commands.append(normalized)

    @staticmethod
    def _extract_regex_head(pattern: str) -> str | None:
        normalized = pattern.strip()
        if not normalized:
            return None
        normalized = normalized.lstrip("^")
        if normalized.startswith("(?:"):
            return None
        if normalized.startswith("(?"):
            return None
        parts = re.split(r"[\[\(\.\*\+\?\|\$\\]", normalized, maxsplit=1)
        head = parts[0].strip()
        if not head:
            return None
        if any(ch in head for ch in "{}:"):
            return None
        return head

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
    def _is_plugin_enabled(
        cls,
        plugin: PluginInfo,
        selection_context: PluginSelectionContext | None,
    ) -> bool:
        if selection_context is None:
            return True
        session_id = str(selection_context.session_id or "").strip()
        group_id = str(selection_context.group_id or "").strip()
        keys = {plugin.module, plugin.name}
        if session_id:
            overrides = cls._session_plugin_overrides.get(session_id, {})
            for key in keys:
                if key in overrides:
                    return overrides[key]
        if group_id:
            overrides = cls._group_plugin_overrides.get(group_id, {})
            for key in keys:
                if key in overrides:
                    return overrides[key]
        return True

    @classmethod
    def _is_plugin_authorized(
        cls,
        plugin: PluginInfo,
        selection_context: PluginSelectionContext | None,
    ) -> bool:
        if selection_context is None:
            return True
        if plugin.limit_superuser and selection_context.is_superuser:
            return False
        admin_level = int(plugin.admin_level or 0)
        if admin_level > 0 and not selection_context.is_superuser:
            return False
        return True

    @classmethod
    def filter_knowledge_base(
        cls,
        knowledge_base: PluginKnowledgeBase,
        selection_context: PluginSelectionContext | None = None,
    ) -> PluginKnowledgeBase:
        if not knowledge_base.plugins:
            return knowledge_base
        selected: list[PluginInfo] = []
        for plugin in knowledge_base.plugins:
            if not cls._is_plugin_enabled(plugin, selection_context):
                continue
            if not cls._is_plugin_authorized(plugin, selection_context):
                continue
            selected.append(plugin)
        return PluginKnowledgeBase(
            plugins=selected,
            user_role=knowledge_base.user_role,
        )

    @classmethod
    async def set_plugin_enabled(
        cls,
        *,
        plugin_key: str,
        enabled: bool,
        session_id: str | None = None,
        group_id: str | None = None,
    ) -> None:
        key = str(plugin_key or "").strip()
        if not key or (not session_id and not group_id):
            return
        async with cls._lock:
            if session_id:
                sid = str(session_id).strip()
                if sid:
                    cls._session_plugin_overrides.setdefault(sid, {})[key] = enabled
            if group_id:
                gid = str(group_id).strip()
                if gid:
                    cls._group_plugin_overrides.setdefault(gid, {})[key] = enabled

    @classmethod
    async def reset_dynamic_overrides(
        cls,
        *,
        session_id: str | None = None,
        group_id: str | None = None,
    ) -> None:
        async with cls._lock:
            if session_id:
                cls._session_plugin_overrides.pop(str(session_id).strip(), None)
            if group_id:
                cls._group_plugin_overrides.pop(str(group_id).strip(), None)

    @classmethod
    async def preload_cache(cls, *, force_refresh: bool = False):
        """
        预加载缓存 - 在插件启动时调用，提前缓存普通用户的知识库
        """
        logger.info("开始预加载 ChatInter 插件知识库缓存...")

        try:
            normal_cache = await cls.get_plugin_knowledge_base(
                force_refresh=force_refresh
            )
            cls._cache["normal_user"] = (normal_cache, datetime.now())
            logger.info(
                f"ChatInter 知识库缓存预加载完成，"
                f"共缓存 {len(normal_cache.plugins)} 个插件"
            )

        except Exception as e:
            logger.error(f"预加载知识库缓存失败：{e}")


async def get_user_plugin_knowledge(
    force_refresh: bool = False,
) -> PluginKnowledgeBase:
    """
    获取普通用户的插件知识库（便捷函数）

    返回:
        PluginKnowledgeBase: 插件知识库
    """
    return await PluginRegistry.get_plugin_knowledge_base(force_refresh=force_refresh)


async def get_runtime_plugin_knowledge() -> PluginKnowledgeBase:
    return await PluginRegistry.get_runtime_plugin_knowledge_base()
