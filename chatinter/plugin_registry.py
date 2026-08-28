"""
ChatInter - 插件信息注册表

收集和缓存插件信息，供意图分析使用。
只提供给 LLM 普通用户可访问的插件。
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib
import inspect
import re
import traceback
from types import SimpleNamespace
from typing import Any, ClassVar, Literal, cast

import nonebot

from zhenxun.configs.utils import PluginExtraData
from zhenxun.services.cache.runtime_cache import (
    GroupMemoryCache,
    PluginInfoMemoryCache,
    _parse_block_modules,
)
from zhenxun.services.log import logger
from zhenxun.utils.enum import BlockType, PluginType

from .capability_graph import build_capability_graph_snapshot
from .command_meta_enrichment import enrich_command_meta_payload
from .metadata_builder import AutoMetadataBuilder
from .models.pydantic_models import (
    CapabilityGraphSnapshot,
    CommandToolSnapshot,
    PluginInfo,
    PluginKnowledgeBase,
    PluginReference,
    SemanticToolContract,
)
from .plugin_reference import (
    build_command_tool_snapshots,
    build_plugin_references,
)
from .route_text import normalize_message_text


@dataclass(frozen=True)
class PluginSelectionContext:
    query: str = ""
    session_id: str | None = None
    user_id: str | None = None
    group_id: str | None = None
    is_superuser: bool = False
    event_type: str = "message"
    adapter: str = ""
    is_private: bool = False
    has_image: bool = False
    has_at: bool = False
    has_reply: bool = False
    has_verified_target: bool = False
    verified_target_source: str = ""
    supports_image: bool = True
    supports_at: bool = True
    supports_reply: bool = True
    addressee_user_id: str | None = None
    addressee_source: str = ""
    thread_id: str = ""
    intervention_action: str = ""


class PluginRegistry:
    """插件信息注册表"""

    _cache: ClassVar[dict[str, tuple[PluginKnowledgeBase, datetime]]] = {}
    _cache_active_plugin_modules: ClassVar[dict[str, frozenset[str]]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _command_tool_cache: ClassVar[
        dict[
            tuple[object, ...],
            tuple[list[CommandToolSnapshot], int, PluginKnowledgeBase],
        ]
    ] = {}
    _command_tool_cache_order: ClassVar[list[tuple[object, ...]]] = []
    _command_tool_cache_max: ClassVar[int] = 16
    _filter_kb_cache: ClassVar[dict[tuple[object, ...], "PluginKnowledgeBase"]] = {}
    _filter_kb_cache_order: ClassVar[list[tuple[object, ...]]] = []
    _filter_kb_cache_max: ClassVar[int] = 32
    _knowledge_revision: ClassVar[int] = 0
    _knowledge_build_task: ClassVar[asyncio.Task[PluginKnowledgeBase] | None] = None
    _knowledge_build_forced: ClassVar[bool] = False
    _argument_source_rank: ClassVar[dict[str, int]] = {
        "unknown": 0,
        "identity_fallback": 1,
        "usage": 2,
        "declared": 3,
        "discovery": 4,
        "runtime_parser": 5,
        "runtime_handler": 6,
    }

    @classmethod
    def invalidate_knowledge_cache(cls) -> None:
        """插件安装、卸载或更新后使知识库与命令 schema 缓存失效。"""
        cls._cache.clear()
        cls._cache_active_plugin_modules.clear()
        cls._clear_command_tool_cache(bump_revision=True)

    _command_discovery_entrypoints: ClassVar[tuple[str, ...]] = (
        "chatinter_command_discovery",
        "__chatinter_command_discovery__",
        "get_chatinter_commands",
        "__chatinter_skill_commands__",
    )
    _command_placeholder_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*(?:\[[^\]]+\]|<[^>]+>|\{[^}]+\})\s*"
    )
    _ascii_target_terms: ClassVar[set[str]] = {
        "at",
        "user",
        "member",
        "target",
        "nickname",
    }
    _cjk_target_terms: ClassVar[tuple[str, ...]] = (
        "用户",
        "成员",
        "群友",
        "目标",
        "对象",
        "昵称",
    )
    _session_plugin_overrides: ClassVar[dict[str, dict[str, bool]]] = {}
    _group_plugin_overrides: ClassVar[dict[str, dict[str, bool]]] = {}
    _restricted_plugin_types: ClassVar[set[PluginType]] = {
        PluginType.SUPERUSER,
        PluginType.ADMIN,
        PluginType.SUPER_AND_ADMIN,
        PluginType.HIDDEN,
        PluginType.DEPENDANT,
        PluginType.PARENT,
    }
    _infra_module_tails: ClassVar[set[str]] = {
        "admin_help",
        "auto_backup",
        "auto_update",
        "bot_manage",
        "broadcast",
        "check",
        "chkdsk_hook",
        "clear_data",
        "exec_sql",
        "fg_manage",
        "group_manage",
        "group_member_update",
        "group_update",
        "hooks",
        "init",
        "init_config",
        "init_plugin",
        "init_task",
        "limiter_hook",
        "llm_manager",
        "plugin_config_manager",
        "plugin_store",
        "plugin_switch",
        "restart",
        "scheduler",
        "scheduler_admin",
        "scheduler_adm",
        "set_admin",
        "super_help",
        "update_fg_info",
        "web_ui",
        "withdraw_hook",
    }
    _infra_module_roots: ClassVar[set[str]] = {
        "zhenxun.builtin_plugins.admin",
        "zhenxun.builtin_plugins.superuser",
    }
    _infra_module_markers: ClassVar[tuple[str, ...]] = (
        ".builtin_plugins.hooks",
        ".builtin_plugins.init",
        ".builtin_plugins.scheduler",
        ".services.",
        ".webui",
        ".web_ui",
    )

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
        retry_forced_refresh = False
        async with cls._lock:
            if not force_refresh and cache_key in cls._cache:
                cached_data, _cached_time = cls._cache[cache_key]
                if not cls._active_plugin_modules_changed(cache_key):
                    logger.debug("使用缓存的插件知识库")
                    return cached_data
                logger.info("检测到活动插件集合变化，刷新 ChatInter 插件知识快照")
            task = cls._knowledge_build_task
            if force_refresh and task is not None and not cls._knowledge_build_forced:
                cls._cache.clear()
                cls._cache_active_plugin_modules.clear()
                cls._clear_command_tool_cache(bump_revision=True)
                retry_forced_refresh = True
            elif task is None:
                if force_refresh:
                    cls._cache.clear()
                    cls._cache_active_plugin_modules.clear()
                    cls._clear_command_tool_cache(bump_revision=True)
                task = asyncio.create_task(
                    cls._build_and_cache_knowledge(
                        cache_key=cache_key,
                        revision=cls._knowledge_revision,
                    )
                )
                cls._knowledge_build_task = task
                cls._knowledge_build_forced = force_refresh
        result = await asyncio.shield(task)
        if retry_forced_refresh:
            return await cls.get_plugin_knowledge_base(force_refresh=True)
        return result

    @classmethod
    async def _build_and_cache_knowledge(
        cls,
        *,
        cache_key: str,
        revision: int,
    ) -> PluginKnowledgeBase:
        current_task = asyncio.current_task()
        try:
            await PluginInfoMemoryCache.ensure_loaded()
            active_modules_before = cls._active_plugin_modules()
            knowledge_base = await cls._build_knowledge_base()
            active_modules_after = cls._active_plugin_modules()
            async with cls._lock:
                if (
                    revision == cls._knowledge_revision
                    and active_modules_before == active_modules_after
                ):
                    cls._cache[cache_key] = (knowledge_base, datetime.now())
                    cls._cache_active_plugin_modules[cache_key] = active_modules_after
                    cls._clear_command_tool_cache(bump_revision=True)
            return knowledge_base
        finally:
            async with cls._lock:
                if cls._knowledge_build_task is current_task:
                    cls._knowledge_build_task = None
                    cls._knowledge_build_forced = False

    @classmethod
    async def shutdown(cls) -> None:
        async with cls._lock:
            task = cls._knowledge_build_task
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    def _active_plugin_modules() -> frozenset[str]:
        snapshots = getattr(PluginInfoMemoryCache, "_by_module", {})
        active = {
            f"cache:{module}"
            f":al{getattr(snapshot, 'admin_level', '')}"
            f":ls{int(bool(getattr(snapshot, 'limit_superuser', False)))}"
            for module, snapshot in snapshots.items()
            if bool(getattr(snapshot, "load_status", True))
            and bool(getattr(snapshot, "status", True))
        }
        active.update(
            f"runtime:{module_name}"
            for plugin in nonebot.get_loaded_plugins()
            if (module_name := str(getattr(plugin, "module_name", "") or "").strip())
        )
        return frozenset(active)

    @classmethod
    def _active_plugin_modules_changed(cls, cache_key: str) -> bool:
        previous = cls._cache_active_plugin_modules.get(cache_key)
        if previous is None:
            return False
        return previous != cls._active_plugin_modules()

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
                str(param).strip() for param in (getattr(raw, "params", []) or [])
            ]
            params = [param for param in params if param]
            params = cls._merge_unique_strings(
                params,
                cls._extract_command_params_from_text(command_text),
            )
            examples: list[str] = []
            example_descriptions: list[str] = []
            for item in getattr(raw, "examples", []) or []:
                exec_text = str(getattr(item, "exec", "") or "").strip()
                if exec_text:
                    examples.append(exec_text)
                description_text = str(getattr(item, "description", "") or "").strip()
                if description_text:
                    example_descriptions.append(description_text)
            description = str(getattr(raw, "description", "") or "").strip()
            if example_descriptions:
                description = cls._merge_text_fields(
                    description,
                    "；".join(example_descriptions),
                )
            command_metas.append(
                cls._with_command_meta_defaults(
                    command=command_text,
                    description=description,
                    prefixes=getattr(raw, "prefixes", None),
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
                    argument_source="declared",
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
    def _infer_aliases_from_examples(
        cls,
        command: str,
        examples: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        """Infer alternate command heads from executable examples only.

        This stays generic: it never checks plugin/module names, only whether an
        example starts with a compact command head that differs from the primary
        command.
        """

        primary = cls._normalize_command(command).casefold()
        aliases: list[str] = []
        for example in examples or []:
            text = str(example or "").strip()
            if not text:
                continue
            head_part = cls._command_placeholder_pattern.split(text, maxsplit=1)[0]
            head_tokens = head_part.split(maxsplit=1)
            candidate = cls._normalize_command(head_tokens[0]) if head_tokens else ""
            if not candidate:
                text_tokens = text.split(maxsplit=1)
                candidate = (
                    cls._normalize_command(text_tokens[0]) if text_tokens else ""
                )
            if not candidate:
                continue
            if candidate.casefold() == primary:
                continue
            if len(candidate) > 32 or any(mark in candidate for mark in "，。！？；"):
                continue
            cls._append_command(aliases, candidate)
        return aliases

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
            result.append(
                cls._with_command_meta_defaults(
                    command=command_text,
                    argument_source="identity_fallback",
                )
            )
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
    def _normalize_argument_source(cls, value: object) -> str:
        source = str(value or "").strip().lower()
        return source if source in cls._argument_source_rank else "unknown"

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
        return (
            cls._normalize_access_level(getattr(meta, "access_level", None)) == "public"
        )

    @classmethod
    def _filter_public_command_meta(
        cls,
        metas: list[PluginInfo.PluginCommandMeta],
    ) -> list[PluginInfo.PluginCommandMeta]:
        return [meta for meta in metas if cls._is_public_command_meta(meta)]

    @classmethod
    def _infer_actor_scope(cls, command: str, actor_scope: object) -> str:
        del command
        parsed = str(actor_scope or "").strip().lower()
        if parsed in {"self_only", "allow_other"}:
            return parsed
        return "allow_other"

    @classmethod
    def _infer_target_requirement(
        cls,
        *,
        actor_scope: str,
        target_requirement: object,
        allow_at: bool | None,
        image_min: int | None,
        params: list[str] | tuple[str, ...] | None = None,
        examples: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        parsed = str(target_requirement or "").strip().lower()
        if parsed in {"none", "optional", "required"}:
            return parsed
        if actor_scope == "self_only":
            return "none"
        text = normalize_message_text(
            " ".join([*(params or ()), *(examples or ())])
        ).lower()
        target_like = cls._contains_target_term(text)
        if target_like:
            return "required"
        if allow_at or (image_min or 0) > 0:
            return "optional"
        return "none"

    @classmethod
    def _contains_target_term(cls, text: str) -> bool:
        normalized = normalize_message_text(text).lower()
        if not normalized:
            return False
        if "@" in normalized or any(
            term in normalized for term in cls._cjk_target_terms
        ):
            return True
        return any(
            token in cls._ascii_target_terms
            for token in re.findall(r"[a-z]+", normalized)
        )

    @staticmethod
    def _normalize_target_sources(
        *,
        actor_scope: str,
        allow_at: bool | None,
        target_sources: object,
    ) -> list[str]:
        if isinstance(target_sources, list | tuple):
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
        prefixes: list[str] | tuple[str, ...] | None = None,
        params: list[str] | tuple[str, ...] | None = None,
        description: str = "",
        examples: list[str] | tuple[str, ...] | None = None,
        text_min: int | None = None,
        text_max: int | None = None,
        image_min: int | None = None,
        image_max: int | None = None,
        allow_at: bool | None = None,
        choices: dict[str, list[str]] | None = None,
        slot_choices: dict[str, list[str]] | None = None,
        slot_types: dict[str, str] | None = None,
        slot_renderers: dict[str, str] | None = None,
        shortcut_renders: list[dict[str, object]] | None = None,
        actor_scope: object = None,
        target_requirement: object = None,
        target_sources: object = None,
        requires_reply: object = None,
        requires_private: object = None,
        requires_to_me: object = None,
        allow_sticky_arg: object = None,
        access_level: object = None,
        argument_source: object = None,
    ) -> PluginInfo.PluginCommandMeta:
        normalized_command = str(command or "").strip()
        if slot_choices is None:
            slot_choices = choices
        normalized_params = cls._merge_unique_strings(params, [])
        normalized_examples = cls._merge_unique_strings(examples, [])
        normalized_aliases = cls._merge_unique_strings(
            aliases,
            cls._infer_aliases_from_examples(
                normalized_command,
                normalized_examples,
            ),
        )
        resolved_actor_scope = cls._infer_actor_scope(normalized_command, actor_scope)
        resolved_target_requirement = cls._infer_target_requirement(
            actor_scope=resolved_actor_scope,
            target_requirement=target_requirement,
            allow_at=allow_at,
            image_min=image_min,
            params=normalized_params,
            examples=normalized_examples,
        )
        resolved_target_sources = cls._normalize_target_sources(
            actor_scope=resolved_actor_scope,
            allow_at=allow_at,
            target_sources=target_sources,
        )
        resolved_requires_reply = bool(cls._safe_bool(requires_reply))
        resolved_requires_private = bool(cls._safe_bool(requires_private))
        resolved_requires_to_me = bool(cls._safe_bool(requires_to_me))
        resolved_allow_sticky_arg = cls._infer_allow_sticky_arg(
            allow_sticky_arg=allow_sticky_arg,
            allow_at=allow_at,
            text_max=text_max,
        )
        resolved_access_level = cls._normalize_access_level(access_level)
        return PluginInfo.PluginCommandMeta(
            command=normalized_command,
            aliases=normalized_aliases,
            prefixes=cls._merge_unique_strings(prefixes, []),
            params=normalized_params,
            choices=dict(slot_choices or {}),
            slot_types=cls._merge_slot_mapping(slot_types),
            slot_renderers=cls._merge_slot_mapping(slot_renderers),
            shortcut_renders=list(shortcut_renders or []),
            description=str(description or "").strip(),
            examples=normalized_examples,
            text_min=text_min,
            text_max=text_max,
            image_min=image_min,
            image_max=image_max,
            allow_at=allow_at,
            actor_scope=cast(Literal["self_only", "allow_other"], resolved_actor_scope),
            target_requirement=cast(
                Literal["none", "optional", "required"],
                resolved_target_requirement,
            ),
            target_sources=cast(
                list[Literal["at", "reply", "nickname", "self"]],
                resolved_target_sources,
            ),
            requires_reply=resolved_requires_reply,
            requires_private=resolved_requires_private,
            requires_to_me=resolved_requires_to_me,
            allow_sticky_arg=resolved_allow_sticky_arg,
            argument_source=cls._normalize_argument_source(argument_source),
            access_level=cast(
                Literal["public", "admin", "superuser", "restricted"],
                resolved_access_level,
            ),
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
            "prefixes": list(getattr(meta, "prefixes", []) or []),
            "params": list(getattr(meta, "params", []) or []),
            "slot_choices": dict(
                getattr(meta, "choices", None)
                or getattr(meta, "slot_choices", None)
                or {}
            ),
            "slot_types": dict(getattr(meta, "slot_types", None) or {}),
            "slot_renderers": dict(getattr(meta, "slot_renderers", None) or {}),
            "shortcut_renders": list(getattr(meta, "shortcut_renders", []) or []),
            "description": str(getattr(meta, "description", "") or "").strip(),
            "examples": list(getattr(meta, "examples", []) or []),
            "text_min": cls._safe_int(getattr(meta, "text_min", None)),
            "text_max": cls._safe_int(getattr(meta, "text_max", None)),
            "image_min": cls._safe_int(getattr(meta, "image_min", None)),
            "image_max": cls._safe_int(getattr(meta, "image_max", None)),
            "allow_at": cls._safe_bool(getattr(meta, "allow_at", None)),
            "actor_scope": str(getattr(meta, "actor_scope", "") or "").strip().lower()
            or None,
            "target_requirement": str(getattr(meta, "target_requirement", "") or "")
            .strip()
            .lower()
            or None,
            "target_sources": list(getattr(meta, "target_sources", []) or []),
            "requires_reply": bool(getattr(meta, "requires_reply", False)),
            "requires_private": bool(getattr(meta, "requires_private", False)),
            "requires_to_me": bool(getattr(meta, "requires_to_me", False)),
            "allow_sticky_arg": cls._safe_bool(getattr(meta, "allow_sticky_arg", None)),
            "argument_source": cls._normalize_argument_source(
                getattr(meta, "argument_source", None)
            ),
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
            if isinstance(collection, list | tuple):
                iterable = collection
            else:
                iterable = [collection]
            for value in iterable:
                text = str(value).strip()
                if text and text not in result:
                    result.append(text)
        return result

    @classmethod
    def _merge_slot_choices(cls, *values: object) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for value in values:
            if not isinstance(value, dict):
                continue
            for raw_key, raw_choices in value.items():
                key = str(raw_key or "").strip()
                if not key:
                    continue
                if isinstance(raw_choices, str):
                    choices = [raw_choices]
                elif isinstance(raw_choices, list | tuple | set | frozenset):
                    choices = [str(choice) for choice in raw_choices]
                else:
                    continue
                merged[key] = cls._merge_unique_strings(merged.get(key), choices)
        return merged

    @classmethod
    def _merge_slot_mapping(cls, *values: object) -> dict[str, str]:
        merged: dict[str, str] = {}
        for value in values:
            if not isinstance(value, dict):
                continue
            for raw_key, raw_item in value.items():
                key = str(raw_key or "").strip()
                item = str(raw_item or "").strip()
                if key and item and key not in merged:
                    merged[key] = item
        return merged

    @classmethod
    def _merge_shortcut_renders(cls, *values: object) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for value in values:
            if not isinstance(value, list | tuple):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                alias = str(item.get("alias") or "").strip()
                render = str(item.get("render") or "").strip()
                if not alias or not render:
                    continue
                marker = (alias.casefold(), render.casefold())
                if marker in seen:
                    continue
                seen.add(marker)
                args = item.get("args")
                raw_optional_params = item.get("optional_params")
                payload: dict[str, object] = {
                    "alias": alias,
                    "render": render,
                    "args": [str(arg).strip() for arg in args if str(arg or "").strip()]
                    if isinstance(args, list | tuple)
                    else [],
                }
                optional_params = (
                    [
                        str(param).strip()
                        for param in raw_optional_params
                        if str(param or "").strip()
                    ]
                    if isinstance(raw_optional_params, list | tuple)
                    else []
                )
                if optional_params:
                    payload["optional_params"] = optional_params
                merged.append(payload)
        return merged

    @staticmethod
    def _merge_text_fields(*values: object) -> str:
        parts: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in parts:
                parts.append(text)
        return "；".join(parts)

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
                argument_contract = cls._merge_argument_meta(left, right)
                merged[key] = cls._with_command_meta_defaults(
                    command=left.get("command") or right.get("command") or command_text,
                    aliases=cls._merge_unique_strings(
                        left.get("aliases"), right.get("aliases")
                    ),
                    prefixes=cls._merge_unique_strings(
                        left.get("prefixes"), right.get("prefixes")
                    ),
                    params=argument_contract["params"],
                    slot_choices=cls._merge_slot_choices(
                        left.get("slot_choices"), right.get("slot_choices")
                    ),
                    slot_types=cls._merge_slot_mapping(
                        left.get("slot_types"), right.get("slot_types")
                    ),
                    slot_renderers=cls._merge_slot_mapping(
                        left.get("slot_renderers"), right.get("slot_renderers")
                    ),
                    shortcut_renders=cls._merge_shortcut_renders(
                        left.get("shortcut_renders"), right.get("shortcut_renders")
                    ),
                    description=cls._merge_text_fields(
                        left.get("description"), right.get("description")
                    ),
                    examples=cls._merge_unique_strings(
                        left.get("examples"), right.get("examples")
                    ),
                    text_min=argument_contract["text_min"],
                    text_max=argument_contract["text_max"],
                    image_min=argument_contract["image_min"],
                    image_max=argument_contract["image_max"],
                    allow_at=left.get("allow_at")
                    if left.get("allow_at") is not None
                    else right.get("allow_at"),
                    actor_scope=left.get("actor_scope") or right.get("actor_scope"),
                    target_requirement=left.get("target_requirement")
                    or right.get("target_requirement"),
                    target_sources=cls._merge_unique_strings(
                        left.get("target_sources"), right.get("target_sources")
                    ),
                    requires_reply=bool(left.get("requires_reply"))
                    or bool(right.get("requires_reply")),
                    requires_private=bool(left.get("requires_private"))
                    or bool(right.get("requires_private")),
                    requires_to_me=bool(left.get("requires_to_me"))
                    or bool(right.get("requires_to_me")),
                    allow_sticky_arg=left.get("allow_sticky_arg")
                    if left.get("allow_sticky_arg") is not None
                    else right.get("allow_sticky_arg"),
                    argument_source=argument_contract["argument_source"],
                    access_level=cls._merge_access_level(
                        left.get("access_level"), right.get("access_level")
                    ),
                )
        return sorted(
            merged.values(), key=lambda item: (len(item.command), item.command)
        )

    @classmethod
    def _merge_argument_meta(
        cls,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> dict[str, Any]:
        fields = ("text_min", "text_max", "image_min", "image_max")

        def has_facts(payload: dict[str, Any]) -> bool:
            return bool(payload.get("params")) or any(
                payload.get(field) is not None for field in fields
            )

        left_source = cls._normalize_argument_source(left.get("argument_source"))
        right_source = cls._normalize_argument_source(right.get("argument_source"))
        left_facts = has_facts(left)
        right_facts = has_facts(right)
        left_rank = cls._argument_source_rank[left_source] if left_facts else -1
        right_rank = cls._argument_source_rank[right_source] if right_facts else -1

        if left_rank != right_rank:
            primary, secondary = (
                (left, right) if left_rank > right_rank else (right, left)
            )
            source = left_source if left_rank > right_rank else right_source
            result = {
                "params": cls._merge_unique_strings(primary.get("params"), []),
                **{field: primary.get(field) for field in fields},
                "argument_source": source,
            }
            for field in fields:
                if result[field] is None:
                    result[field] = secondary.get(field)
            if not result["params"] and result["text_max"] != 0:
                result["params"] = cls._merge_unique_strings(
                    secondary.get("params"), []
                )
            return result

        source = left_source if left_rank >= 0 else right_source
        result = {
            "params": cls._merge_unique_strings(
                left.get("params"), right.get("params")
            ),
            "argument_source": source,
        }
        for field in fields:
            left_value = left.get(field)
            right_value = right.get(field)
            if left_value is None:
                result[field] = right_value
            elif right_value is None:
                result[field] = left_value
            elif field.endswith("_min"):
                result[field] = max(int(left_value), int(right_value))
            elif int(left_value) == 0 or int(right_value) == 0:
                result[field] = 0
            else:
                result[field] = min(int(left_value), int(right_value))
        return result

    @classmethod
    def _command_meta_richness(
        cls,
        meta: PluginInfo.PluginCommandMeta,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
        aliases = len(getattr(meta, "aliases", []) or [])
        prefixes = len(getattr(meta, "prefixes", []) or [])
        params = len(getattr(meta, "params", []) or [])
        description = int(bool(getattr(meta, "description", "") or ""))
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
        requires_reply = int(bool(getattr(meta, "requires_reply", False)))
        requires_private = int(bool(getattr(meta, "requires_private", False)))
        requires_to_me = int(bool(getattr(meta, "requires_to_me", False)))
        return (
            params,
            description,
            text_score,
            aliases,
            prefixes,
            examples,
            sticky,
            allow_at,
            requires_reply,
            requires_private,
            requires_to_me,
        )

    @classmethod
    def _canonicalize_command_meta_groups(
        cls,
        metas: list[PluginInfo.PluginCommandMeta],
    ) -> list[PluginInfo.PluginCommandMeta]:
        return cls._prune_sibling_head_aliases(cls._merge_command_meta_groups(metas))

    @classmethod
    def _prune_sibling_head_aliases(
        cls,
        metas: list[PluginInfo.PluginCommandMeta],
    ) -> list[PluginInfo.PluginCommandMeta]:
        heads = {
            cls._normalize_command(str(getattr(meta, "command", "") or "")).casefold()
            for meta in metas
            if cls._normalize_command(str(getattr(meta, "command", "") or ""))
        }
        if not heads:
            return metas

        pruned: list[PluginInfo.PluginCommandMeta] = []
        for meta in metas:
            payload = cls._meta_to_dict(meta)
            command_fold = cls._normalize_command(
                str(payload.get("command") or "")
            ).casefold()
            payload["aliases"] = [
                alias
                for alias in payload.get("aliases", [])
                if (alias_fold := cls._normalize_command(str(alias or "")).casefold())
                and alias_fold != command_fold
                and alias_fold not in heads
            ]
            pruned.append(cls._with_command_meta_defaults(**payload))
        return pruned

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
        command_heads = {
            cls._normalize_command(getattr(meta, "command", "")).casefold()
            for meta in metas
            if cls._normalize_command(getattr(meta, "command", ""))
        }
        alias_heads = {head for head in alias_heads if head not in command_heads}
        if not alias_heads:
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
            target_payload["description"] = cls._merge_text_fields(
                target_payload.get("description"), item_payload.get("description")
            )
            target_payload["examples"] = cls._merge_unique_strings(
                target_payload.get("examples"), item_payload.get("examples")
            )
            target_payload["prefixes"] = cls._merge_unique_strings(
                target_payload.get("prefixes"), item_payload.get("prefixes")
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
            command_fold = cls._normalize_command(
                getattr(item, "command", "")
            ).casefold()
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
                return cls._with_command_meta_defaults(
                    command=command_text,
                    argument_source="identity_fallback",
                )
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
        prefixes = item.get("prefixes")
        params = item.get("params")
        examples = item.get("examples")
        description = str(item.get("description") or item.get("desc") or "").strip()
        if not isinstance(aliases, list | tuple):
            aliases = []
        if not isinstance(prefixes, list | tuple):
            prefixes = []
        if not isinstance(params, list | tuple):
            params = []
        if not isinstance(examples, list | tuple):
            examples = []
        normalized_examples: list[str] = []
        for example in examples:
            if isinstance(example, dict):
                text = str(example.get("exec") or example.get("example") or "").strip()
                example_description = str(example.get("description") or "").strip()
                if example_description:
                    description = cls._merge_text_fields(
                        description, example_description
                    )
            else:
                text = str(example).strip()
            if text:
                normalized_examples.append(text)
        return cls._with_command_meta_defaults(
            command=command_text,
            description=description,
            aliases=[
                str(alias).strip() for alias in aliases if str(alias or "").strip()
            ],
            prefixes=[
                str(prefix).strip() for prefix in prefixes if str(prefix or "").strip()
            ],
            params=[str(param).strip() for param in params if str(param or "").strip()],
            slot_choices=item.get("slot_choices", schema.get("slot_choices")),
            slot_types=item.get("slot_types", schema.get("slot_types")),
            slot_renderers=item.get(
                "slot_renderers", schema.get("slot_renderers")
            ),
            shortcut_renders=item.get(
                "shortcut_renders", schema.get("shortcut_renders")
            ),
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
            requires_reply=item.get("requires_reply", schema.get("requires_reply")),
            requires_private=item.get(
                "requires_private", schema.get("requires_private")
            ),
            requires_to_me=item.get("requires_to_me", schema.get("requires_to_me")),
            allow_sticky_arg=item.get(
                "allow_sticky_arg", schema.get("allow_sticky_arg")
            ),
            argument_source=item.get("argument_source", "discovery"),
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
                enrich_command_meta_payload(
                    module_name,
                    await AutoMetadataBuilder.build(
                        module_name=module_name,
                        module_obj=module_obj,
                        loaded_plugin=loaded_plugin,
                    ),
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
    def _clean_usage_line(cls, line: str) -> str:
        text = str(line or "").strip()
        text = text.strip("` \t")
        text = re.sub(r"^[\-\*\d\.\)、)\s]+", "", text).strip()
        text = re.sub(
            r"^(?:命令|用法|示例|格式|usage|example)\s*[:：]\s*", "", text, flags=re.I
        )
        return text.strip()

    @staticmethod
    def _split_usage_description(line: str) -> tuple[str, str]:
        for separator in (" -- ", " - ", "：", ":"):
            if separator in line:
                left, right = line.split(separator, 1)
                return left.strip(), right.strip()
        return line.strip(), ""

    @classmethod
    def _line_starts_with_command(cls, line: str, command: str) -> bool:
        normalized_line = cls._normalize_command(line).casefold()
        normalized_command = cls._normalize_command(command).casefold()
        if not normalized_line or not normalized_command:
            return False
        if normalized_line == normalized_command:
            return True
        return normalized_line.startswith(normalized_command + " ")

    @classmethod
    def _extract_usage_command_meta(
        cls,
        usage: str | None,
        command_meta: list[PluginInfo.PluginCommandMeta],
    ) -> list[PluginInfo.PluginCommandMeta]:
        if not usage:
            return []

        known_heads: list[str] = []
        meta_by_head: dict[str, PluginInfo.PluginCommandMeta] = {}
        for meta in command_meta:
            cls._append_command(known_heads, meta.command)
            normalized_command = cls._normalize_command(meta.command).casefold()
            if normalized_command:
                meta_by_head[normalized_command] = meta
            for alias in meta.aliases:
                cls._append_command(known_heads, alias)
                normalized_alias = cls._normalize_command(alias).casefold()
                if normalized_alias:
                    meta_by_head[normalized_alias] = meta
        known_heads = sorted(known_heads, key=len, reverse=True)

        result: list[PluginInfo.PluginCommandMeta] = []
        for raw_line in str(usage or "").splitlines():
            line = cls._clean_usage_line(raw_line)
            if not line or line in {"```", "~~~"}:
                continue
            command_part, description = cls._split_usage_description(line)
            matched_head = next(
                (
                    head
                    for head in known_heads
                    if cls._line_starts_with_command(command_part, head)
                ),
                "",
            )
            if not matched_head and cls._command_placeholder_pattern.search(
                command_part
            ):
                before_placeholder = cls._command_placeholder_pattern.split(
                    command_part, maxsplit=1
                )[0]
                head_parts = before_placeholder.split(maxsplit=1)
                if head_parts:
                    matched_head = cls._normalize_command(head_parts[0])
            if not matched_head:
                continue
            matched_meta = meta_by_head.get(
                cls._normalize_command(matched_head).casefold()
            )
            if (
                not description
                and matched_meta is not None
                and matched_meta.text_max == 0
                and cls._argument_source_rank[
                    cls._normalize_argument_source(matched_meta.argument_source)
                ]
                >= cls._argument_source_rank["runtime_parser"]
            ):
                prose_tail = re.search(r"\s+--(?=\S)", command_part)
                if prose_tail is not None:
                    description = command_part[prose_tail.end() :].strip()
                    command_part = command_part[: prose_tail.start()].strip()
            if len(matched_head) > 32 or any(
                mark in matched_head for mark in "，。！？；"
            ):
                continue
            params = cls._extract_command_params_from_text(command_part)
            result.append(
                cls._with_command_meta_defaults(
                    command=matched_head,
                    params=params,
                    description=description,
                    examples=[command_part],
                    argument_source="usage",
                )
            )
        return result

    @classmethod
    def _is_runtime_plugin_allowed(cls, module_name: str, loaded_plugin=None) -> bool:
        return bool(
            module_name
            and loaded_plugin is not None
            and not cls._is_infrastructure_module(module_name)
        )

    @classmethod
    def _is_infrastructure_module(
        cls,
        module_name: str,
        plugin_name: str | None = None,
    ) -> bool:
        normalized = str(module_name or "").strip().lower()
        if not normalized:
            return True
        tail = normalized.rsplit(".", 1)[-1]
        if tail in cls._infra_module_tails:
            return True
        if any(
            normalized == root or normalized.startswith(f"{root}.")
            for root in cls._infra_module_roots
        ):
            return True
        if any(marker in normalized for marker in cls._infra_module_markers):
            return True

        name_text = normalize_message_text(plugin_name or "").lower()
        return bool(
            name_text
            and name_text
            in {
                "webui",
                "webui管理",
                "ui管理",
                "重启",
                "广播",
                "数据库操作",
                "插件商店",
                "插件配置管理",
                "功能开关",
                "llm模型管理",
                "bot管理",
                "好友群组列表",
                "管理群操作",
                "超级用户帮助",
                "群组管理员帮助",
            }
        )

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
        status: bool = True,
        block_type: BlockType | str | None = None,
        load_status: bool = True,
        block_keys: list[str] | None = None,
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
        resolved_usage = (
            str(getattr(metadata, "usage", "") or "").strip()
            if getattr(metadata, "usage", None)
            else None
        )
        commands = cls._extract_commands(extra_data, command_meta)
        matcher_commands: list[str] = []
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
        command_meta = cls._merge_command_meta_groups(
            command_meta,
            cls._extract_usage_command_meta(resolved_usage, command_meta),
        )
        if loaded_plugin is not None:
            command_meta = cls._fold_matcher_alias_command_meta(
                command_meta,
                loaded_plugin=loaded_plugin,
            )
        command_meta = cls._fold_plugin_alias_command_meta(
            command_meta,
            plugin_aliases=list(extra_data.aliases or []),
        )
        if matcher_commands:
            commands, command_meta = cls._filter_to_matcher_executable(
                commands=commands,
                command_meta=command_meta,
                matcher_commands=matcher_commands,
            )
        command_meta = cls._canonicalize_command_meta_groups(command_meta)
        command_meta = cls._filter_public_command_meta(command_meta)
        commands = cls._merge_unique_strings(
            [meta.command for meta in command_meta if meta.command],
            [],
        )
        if not commands:
            return None
        semantic_tools = cls._extract_semantic_tool_contracts(
            extra_data,
            loaded_plugin=loaded_plugin,
            command_meta=command_meta,
        )

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
            str(getattr(metadata, "description", "") or "").strip() or "暂无描述"
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
            introduction=str(extra_data.introduction or "").strip() or None,
            precautions=cls._merge_unique_strings(
                list(extra_data.precautions or []),
                [],
            ),
            semantic_tools=semantic_tools,
            admin_level=resolved_admin_level,
            limit_superuser=resolved_limit_superuser,
            status=bool(status),
            block_type=cls._normalize_block_type(block_type),
            load_status=bool(load_status),
            block_keys=cls._merge_unique_strings([module_name], block_keys or []),
        )

    @classmethod
    def _extract_semantic_tool_contracts(
        cls,
        extra_data: PluginExtraData,
        *,
        loaded_plugin: object | None,
        command_meta: list[PluginInfo.PluginCommandMeta],
    ) -> list[SemanticToolContract]:
        contracts: list[SemanticToolContract] = []
        command_heads = cls._merge_unique_strings(
            [meta.command for meta in command_meta if meta.command],
            [],
        )
        seen_names: set[str] = set()
        for raw_tool in extra_data.smart_tools or []:
            name = normalize_message_text(str(getattr(raw_tool, "name", "") or ""))
            key = name.casefold()
            if not name or key in seen_names:
                continue
            seen_names.add(key)
            parameters = cls._semantic_tool_parameters(
                getattr(raw_tool, "parameters", None)
            )
            bound_commands = cls._commands_bound_to_smart_handler(
                getattr(raw_tool, "func", None),
                loaded_plugin=loaded_plugin,
            )
            if not bound_commands and len(command_heads) == 1:
                bound_commands = list(command_heads)
            contracts.append(
                SemanticToolContract(
                    name=name,
                    description=normalize_message_text(
                        str(getattr(raw_tool, "description", "") or "")
                    ),
                    parameters=parameters,
                    bound_commands=bound_commands,
                )
            )
        return sorted(contracts, key=lambda item: (item.name.casefold(), item.name))

    @staticmethod
    def _semantic_tool_parameters(raw_parameters: object | None) -> dict[str, Any]:
        if raw_parameters is None:
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        if hasattr(raw_parameters, "model_dump"):
            payload = raw_parameters.model_dump(mode="json")
        elif isinstance(raw_parameters, dict):
            payload = dict(raw_parameters)
        else:
            payload = {}
        properties = payload.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        normalized_properties: dict[str, dict[str, Any]] = {}
        for raw_name, raw_schema in properties.items():
            name = normalize_message_text(str(raw_name or ""))
            if not name:
                continue
            if hasattr(raw_schema, "model_dump"):
                schema = raw_schema.model_dump(mode="json")
            elif isinstance(raw_schema, dict):
                schema = dict(raw_schema)
            else:
                schema = {}
            normalized_properties[name] = {
                key: value
                for key, value in schema.items()
                if key in {"type", "description", "enum", "default"}
            }
        required = [
            name
            for item in payload.get("required", []) or []
            if (name := normalize_message_text(str(item or "")))
            and name in normalized_properties
        ]
        return {
            "type": "object",
            "properties": normalized_properties,
            "required": list(dict.fromkeys(required)),
            "additionalProperties": False,
        }

    @classmethod
    def _commands_bound_to_smart_handler(
        cls,
        func: object | None,
        *,
        loaded_plugin: object | None,
    ) -> list[str]:
        if func is None or loaded_plugin is None:
            return []
        matched: list[object] = []
        for matcher in AutoMetadataBuilder._iter_plugin_matchers(loaded_plugin):
            if any(
                cls._same_callable(getattr(handler, "call", None), func)
                for handler in getattr(matcher, "handlers", []) or []
            ):
                matched.append(matcher)
        if len(matched) != 1:
            return []
        proxy = SimpleNamespace(
            matcher=[matched[0]],
            sub_plugins=set(),
            module_name="",
            module=None,
        )
        payloads = AutoMetadataBuilder._extract_matcher_command_data(
            loaded_plugin=proxy
        )
        return cls._merge_unique_strings(
            [
                str(payload.get("command") or "")
                for payload in payloads
                if isinstance(payload, dict)
            ],
            [],
        )

    @staticmethod
    def _same_callable(left: object | None, right: object | None) -> bool:
        if left is None or right is None:
            return False
        if left is right:
            return True
        left_func = getattr(left, "__func__", left)
        right_func = getattr(right, "__func__", right)
        if left_func is right_func:
            return True
        try:
            return inspect.unwrap(cast(Any, left_func)) is inspect.unwrap(
                cast(Any, right_func)
            )
        except (TypeError, ValueError):
            return False

    @classmethod
    def _is_public_plugin(cls, extra_data: PluginExtraData) -> bool:
        if int(extra_data.admin_level or 0) > 0:
            return False
        if bool(getattr(extra_data, "limit_superuser", False)):
            return False
        plugin_type = getattr(extra_data, "plugin_type", PluginType.NORMAL)
        if plugin_type in cls._restricted_plugin_types:
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
                "ChatInter 插件知识库数据库增强失败，已回退到运行时插件: " f"{exc}"
            )
            return

        for db_plugin in db_plugins.values():
            if db_plugin.plugin_type in cls._restricted_plugin_types:
                continue
            if int(db_plugin.admin_level or 0) > 0:
                continue
            if bool(db_plugin.limit_superuser):
                continue

            module_path = str(db_plugin.module_path or "").strip()
            module_short_name = str(db_plugin.module or "").strip()
            module_name = module_path or module_short_name
            if not module_name:
                continue
            if cls._is_infrastructure_module(module_name, str(db_plugin.name or "")):
                continue

            runtime_plugin = plugins_by_module.get(
                module_path
            ) or plugins_by_module.get(module_short_name)
            if not cls._is_db_plugin_loadable(db_plugin):
                plugins_by_module.pop(module_path, None)
                plugins_by_module.pop(module_short_name, None)
                continue

            if runtime_plugin is not None:
                plugins_by_module.pop(module_path, None)
                plugins_by_module.pop(module_short_name, None)
                plugins_by_module[runtime_plugin.module] = runtime_plugin.model_copy(
                    update={
                        "name": str(db_plugin.name or runtime_plugin.name).strip()
                        or runtime_plugin.name,
                        "admin_level": db_plugin.admin_level,
                        "limit_superuser": bool(db_plugin.limit_superuser),
                        "status": bool(db_plugin.status),
                        "block_type": cls._normalize_block_type(db_plugin.block_type),
                        "load_status": bool(db_plugin.load_status),
                        "block_keys": cls._merge_unique_strings(
                            runtime_plugin.block_keys,
                            [runtime_plugin.module, module_path, module_short_name],
                        ),
                    }
                )
                continue

            nb_plugin = nonebot.get_plugin_by_module_name(module_path)
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
                status=bool(db_plugin.status),
                block_type=db_plugin.block_type,
                load_status=bool(db_plugin.load_status),
                block_keys=[module_path, module_short_name],
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
                sorted({cmd.strip().lower() for cmd in plugin.commands if cmd.strip()})
            )
            fingerprint = (
                plugin.name.strip().lower(),
                command_fingerprint,
            )
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            deduplicated.append(plugin)

        parent_modules_to_remove: set[str] = set()

        for plugin in deduplicated:
            parent_module = plugin.module

            children = [
                p
                for p in deduplicated
                if p.module != parent_module
                and p.module.startswith(parent_module + ".")
            ]
            if not children:
                continue

            children_commands: set[str] = set()
            for child in children:
                for cmd in child.commands:
                    normalized_cmd = cmd.strip().lower()
                    if normalized_cmd:
                        children_commands.add(normalized_cmd)

            parent_commands: set[str] = set()
            for cmd in plugin.commands:
                normalized_cmd = cmd.strip().lower()
                if normalized_cmd:
                    parent_commands.add(normalized_cmd)

            if parent_commands and parent_commands <= children_commands:
                parent_modules_to_remove.add(parent_module)
                logger.debug(
                    f"ChatInter 去重: 移除父模块 {parent_module}，"
                    f"其 {len(parent_commands)} 个命令已被 "
                    f"{len(children)} 个子模块完全覆盖"
                )

        if parent_modules_to_remove:
            deduplicated = [
                p for p in deduplicated if p.module not in parent_modules_to_remove
            ]

        return sorted(deduplicated, key=lambda item: item.module)

    @classmethod
    def _extract_commands_from_matchers(cls, nb_plugin) -> list[str]:
        commands: list[str] = []
        seen: set[str] = set()
        matcher_meta = AutoMetadataBuilder._extract_matcher_command_data(
            loaded_plugin=nb_plugin,
        )
        alias_identities = {
            normalized.casefold()
            for payload in matcher_meta
            for alias in [
                *(payload.get("aliases") or ()),
                *cls._shortcut_aliases_from_payload(payload),
            ]
            if (normalized := cls._normalize_command(str(alias or "")))
        }
        for payload in matcher_meta:
            normalized = cls._normalize_command(
                str(payload.get("command") or "").strip()
            )
            if (
                not normalized
                or normalized.casefold() in alias_identities
                or normalized in seen
            ):
                continue
            seen.add(normalized)
            commands.append(normalized)
        commands.sort(key=lambda cmd: (len(cmd), cmd))
        return commands

    @classmethod
    def _shortcut_aliases_from_payload(cls, payload: dict[str, object]) -> list[str]:
        raw_shortcuts = payload.get("shortcut_renders")
        if not isinstance(raw_shortcuts, list | tuple):
            return []
        aliases: list[str] = []
        for item in raw_shortcuts:
            if not isinstance(item, dict):
                continue
            alias = cls._normalize_command(str(item.get("alias") or ""))
            if alias:
                aliases.append(alias)
        return aliases

    @classmethod
    def _fold_matcher_alias_command_meta(
        cls,
        metas: list[PluginInfo.PluginCommandMeta],
        *,
        loaded_plugin: object,
    ) -> list[PluginInfo.PluginCommandMeta]:
        alias_targets: dict[str, set[str]] = {}
        for payload in AutoMetadataBuilder._extract_matcher_command_data(
            loaded_plugin=loaded_plugin,
        ):
            command = cls._normalize_command(str(payload.get("command") or ""))
            if not command:
                continue
            # Parser aliases are alternate spellings of one command. Shortcuts
            # with their own render/arguments are independent executable views.
            for alias in payload.get("aliases") or ():
                normalized_alias = cls._normalize_command(str(alias or ""))
                if (
                    normalized_alias
                    and normalized_alias.casefold() != command.casefold()
                ):
                    alias_targets.setdefault(normalized_alias.casefold(), set()).add(
                        command.casefold()
                    )

        unique_targets = {
            alias: next(iter(targets))
            for alias, targets in alias_targets.items()
            if len(targets) == 1
        }
        if not unique_targets:
            return metas

        by_head = {
            cls._normalize_command(meta.command).casefold(): meta
            for meta in metas
            if cls._normalize_command(meta.command)
        }
        folded_aliases: set[str] = set()
        replacements: dict[str, PluginInfo.PluginCommandMeta] = {}
        for alias_head, target_head in unique_targets.items():
            alias_meta = by_head.get(alias_head)
            target_meta = replacements.get(target_head) or by_head.get(target_head)
            if alias_meta is None or target_meta is None or alias_meta is target_meta:
                continue
            target_payload = cls._meta_to_dict(target_meta)
            alias_payload = cls._meta_to_dict(alias_meta)
            target_payload["aliases"] = cls._merge_unique_strings(
                target_payload.get("aliases"),
                [str(alias_payload.get("command") or "")],
            )
            target_payload["aliases"] = cls._merge_unique_strings(
                target_payload.get("aliases"), alias_payload.get("aliases")
            )
            target_payload["description"] = cls._merge_text_fields(
                target_payload.get("description"), alias_payload.get("description")
            )
            target_payload["examples"] = cls._merge_unique_strings(
                target_payload.get("examples"), alias_payload.get("examples")
            )
            replacements[target_head] = cls._with_command_meta_defaults(
                **target_payload
            )
            folded_aliases.add(alias_head)

        return cls._merge_command_meta_groups(
            [
                replacements.get(head, meta)
                for meta in metas
                if (head := cls._normalize_command(meta.command).casefold())
                not in folded_aliases
            ]
        )

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
            stripped = cls._strip_leading_command_prefix(normalized)
            if stripped:
                lookup.add(stripped.casefold())
                lookup.add(stripped.split(" ", 1)[0].casefold())
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
        if normalized.split(" ", 1)[0].casefold() in matcher_lookup:
            return True
        stripped = cls._strip_leading_command_prefix(normalized)
        if not stripped:
            return False
        stripped_folded = stripped.casefold()
        if stripped_folded in matcher_lookup:
            return True
        return stripped.split(" ", 1)[0].casefold() in matcher_lookup

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
        executable_lookup = set(matcher_lookup)
        command_identity_lookup = set(matcher_lookup)
        for meta in command_meta:
            payload = cls._meta_to_dict(meta)
            if not cls._command_matches_matcher_lookup(
                str(payload.get("command") or ""), matcher_lookup
            ):
                continue
            for identity in payload.get("aliases") or ():
                normalized = cls._normalize_command(str(identity or ""))
                if normalized:
                    command_identity_lookup.add(normalized.casefold())
            for identity in cls._shortcut_aliases_from_payload(payload):
                normalized = cls._normalize_command(str(identity or ""))
                if normalized:
                    folded = normalized.casefold()
                    executable_lookup.add(folded)
                    command_identity_lookup.add(folded)
        filtered_commands = [
            command
            for command in commands
            if cls._command_matches_matcher_lookup(command, command_identity_lookup)
        ]

        filtered_meta: list[PluginInfo.PluginCommandMeta] = []
        for meta in command_meta:
            payload = cls._meta_to_dict(meta)
            command_text = str(payload.get("command") or "").strip()
            matched_command = cls._command_matches_matcher_lookup(
                command_text, executable_lookup
            )

            original_aliases = payload.get("aliases", [])
            matched_aliases = [
                alias
                for alias in original_aliases
                if cls._command_matches_matcher_lookup(alias, command_identity_lookup)
            ]
            shortcut_aliases = cls._shortcut_aliases_from_payload(payload)
            matched_shortcuts = [
                alias
                for alias in shortcut_aliases
                if cls._command_matches_matcher_lookup(alias, executable_lookup)
            ]

            if not matched_command and not matched_aliases and not matched_shortcuts:
                continue

            if not matched_command:
                if matched_aliases:
                    payload["command"] = matched_aliases[0]
                    matched_aliases = matched_aliases[1:]
                elif matched_shortcuts:
                    payload["command"] = matched_shortcuts[0]
                    matched_shortcuts = matched_shortcuts[1:]
            normalized_command = cls._normalize_command(str(payload.get("command", "")))
            alias_source = original_aliases if matched_command else matched_aliases
            payload["aliases"] = [
                alias
                for alias in alias_source
                if cls._normalize_command(alias).casefold()
                != normalized_command.casefold()
            ]
            filtered_meta.append(cls._with_command_meta_defaults(**payload))

        filtered_commands = cls._merge_unique_strings(
            filtered_commands, matcher_commands
        )

        if not filtered_meta:
            filtered_meta = cls._build_command_meta_from_commands(matcher_commands)
        else:
            filtered_meta = cls._merge_command_meta_groups(
                filtered_meta,
                cls._build_command_meta_from_commands(matcher_commands),
            )

        filtered_commands = cls._extract_commands(
            PluginExtraData(),
            filtered_meta,
        )
        if not filtered_commands:
            filtered_commands = matcher_commands[:]
        return filtered_commands, filtered_meta

    @staticmethod
    def _strip_leading_command_prefix(command: str) -> str:
        normalized = normalize_message_text(command)
        if not normalized:
            return ""
        if normalized.startswith("/"):
            return normalize_message_text(normalized[1:])
        if normalized.startswith("／"):
            return normalize_message_text(normalized[1:])
        return normalized

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
    def clear_cache(cls):
        """清空所有缓存"""
        cls._cache.clear()
        cls._cache_active_plugin_modules.clear()
        cls._clear_command_tool_cache(bump_revision=True)
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
    def _is_command_tool_available(
        cls,
        tool: CommandToolSnapshot,
        selection_context: PluginSelectionContext,
    ) -> bool:
        requires = tool.requires or {}
        if requires.get("image") and not selection_context.supports_image:
            return False
        if requires.get("at") and not selection_context.supports_at:
            return False
        if requires.get("reply") and not selection_context.supports_reply:
            return False
        if requires.get("private") and not selection_context.is_private:
            return False
        if tool.payload_policy in {"image_only", "text_or_image"}:
            if not selection_context.supports_image:
                return False
        return True

    @classmethod
    def _normalize_block_type(cls, block_type: BlockType | str | None) -> str | None:
        if block_type is None:
            return None
        value = getattr(block_type, "value", block_type)
        text = str(value or "").strip().upper()
        return text or None

    @classmethod
    def _is_db_plugin_loadable(cls, db_plugin: object) -> bool:
        if not bool(getattr(db_plugin, "load_status", True)):
            return False
        block_type = cls._normalize_block_type(getattr(db_plugin, "block_type", None))
        if (
            not bool(getattr(db_plugin, "status", True))
            and block_type == BlockType.ALL.value
        ):
            return False
        return True

    @classmethod
    def _is_plugin_status_allowed(
        cls,
        plugin: PluginInfo,
        selection_context: PluginSelectionContext | None,
    ) -> bool:
        if not plugin.load_status:
            return False
        block_type = cls._normalize_block_type(plugin.block_type)
        if plugin.status:
            return True
        if block_type == BlockType.ALL.value:
            return bool(selection_context and selection_context.is_superuser)
        if selection_context is None:
            return True
        if selection_context.is_superuser:
            return True
        if block_type == BlockType.GROUP.value and selection_context.group_id:
            return False
        if block_type == BlockType.PRIVATE.value and selection_context.is_private:
            return False
        return True

    @classmethod
    def _is_group_plugin_allowed(
        cls,
        plugin: PluginInfo,
        selection_context: PluginSelectionContext | None,
    ) -> bool:
        if selection_context is None:
            return True
        if selection_context.is_superuser:
            return True
        group_id = str(selection_context.group_id or "").strip()
        if not group_id:
            return True
        group = GroupMemoryCache.get_if_ready(group_id)
        if group is None:
            return True
        block_raw = getattr(group, "block_plugin", "") or ""
        block_set = getattr(group, "block_plugin_set", None)
        if block_set is None or (not block_set and block_raw):
            block_set = _parse_block_modules(block_raw)
        super_block_raw = getattr(group, "superuser_block_plugin", "") or ""
        super_block_set = getattr(group, "superuser_block_plugin_set", None)
        if super_block_set is None or (not super_block_set and super_block_raw):
            super_block_set = _parse_block_modules(super_block_raw)
        keys = set(plugin.block_keys or [])
        keys.add(plugin.module)
        return not (keys & set(block_set)) and not (keys & set(super_block_set))

    @classmethod
    def _is_plugin_authorized(
        cls,
        plugin: PluginInfo,
        selection_context: PluginSelectionContext | None,
    ) -> bool:
        if selection_context is None:
            return True
        if plugin.limit_superuser and not selection_context.is_superuser:
            return False
        admin_level = int(plugin.admin_level or 0)
        if admin_level > 0 and not selection_context.is_superuser:
            return False
        return True

    @classmethod
    def _is_allowed_plugin_info(cls, plugin: PluginInfo) -> bool:
        if not plugin.load_status:
            return False
        if not plugin.commands and not plugin.command_meta:
            return False
        if cls._is_infrastructure_module(plugin.module, plugin.name):
            return False
        if bool(plugin.limit_superuser):
            return False
        if int(plugin.admin_level or 0) > 0:
            return False
        public_meta = cls._filter_public_command_meta(plugin.command_meta)
        return bool(public_meta or not plugin.command_meta)

    @classmethod
    def filter_knowledge_base(
        cls,
        knowledge_base: PluginKnowledgeBase,
        selection_context: PluginSelectionContext | None = None,
    ) -> PluginKnowledgeBase:
        """权限、开关、群内屏蔽的统一过滤边界。"""

        if not knowledge_base.plugins:
            return knowledge_base

        _sel_key = cls._command_tool_selection_cache_key(selection_context)
        _kb_identity = cls._command_tool_knowledge_identity(knowledge_base)
        if _kb_identity is not None:
            _fk: tuple[object, ...] = (
                "identity",
                cls._knowledge_revision,
                _kb_identity,
                _sel_key,
            )
        else:
            _digest = hashlib.blake2b(digest_size=16)
            _digest.update(knowledge_base.model_dump_json().encode("utf-8", "ignore"))
            _fk = ("content", cls._knowledge_revision, _digest.hexdigest(), _sel_key)
        _cached_filtered = cls._filter_kb_cache.get(_fk)
        if _cached_filtered is not None:
            try:
                cls._filter_kb_cache_order.remove(_fk)
            except ValueError:
                pass
            cls._filter_kb_cache_order.append(_fk)
            return _cached_filtered
        selected: list[PluginInfo] = []
        for plugin in knowledge_base.plugins:
            if not cls._is_allowed_plugin_info(plugin):
                logger.debug(
                    f"插件 {plugin.name}（{plugin.module}）被过滤，原因："
                    "_is_allowed_plugin_info 返回 False"
                )
                continue
            if not cls._is_plugin_status_allowed(plugin, selection_context):
                logger.debug(
                    f"插件 {plugin.name}（{plugin.module}）被过滤，原因：插件状态不允许"
                )
                continue
            if not cls._is_group_plugin_allowed(plugin, selection_context):
                logger.debug(
                    f"插件 {plugin.name}（{plugin.module}）被过滤，原因：群内屏蔽"
                )
                continue
            if not cls._is_plugin_enabled(plugin, selection_context):
                logger.debug(
                    f"插件 {plugin.name}（{plugin.module}）被过滤，原因：插件未启用"
                )
                continue
            if not cls._is_plugin_authorized(plugin, selection_context):
                logger.debug(
                    f"插件 {plugin.name}（{plugin.module}）被过滤，原因：权限不足"
                )
                continue
            selected.append(plugin)
        _result = PluginKnowledgeBase(
            plugins=selected,
            user_role=knowledge_base.user_role,
        )
        cls._filter_kb_cache[_fk] = _result
        cls._filter_kb_cache_order.append(_fk)
        while len(cls._filter_kb_cache_order) > cls._filter_kb_cache_max:
            cls._filter_kb_cache.pop(cls._filter_kb_cache_order.pop(0), None)
        return _result

    @classmethod
    def build_capability_graph(
        cls,
        knowledge_base: PluginKnowledgeBase,
        *,
        selection_context: PluginSelectionContext | None = None,
        limit: int | None = None,
    ) -> CapabilityGraphSnapshot:
        """构建安全过滤后的插件能力图。"""
        if selection_context is not None:
            source = cls.filter_knowledge_base(
                knowledge_base,
                selection_context=selection_context,
            )
        else:
            selected: list[PluginInfo] = []
            for plugin in knowledge_base.plugins:
                if cls._is_allowed_plugin_info(plugin):
                    if not cls._is_plugin_status_allowed(plugin, None):
                        continue
                    selected.append(plugin)
            source = PluginKnowledgeBase(
                plugins=selected,
                user_role=knowledge_base.user_role,
            )
        return build_capability_graph_snapshot(source, limit=limit)

    @classmethod
    def build_plugin_references(
        cls,
        knowledge_base: PluginKnowledgeBase,
        *,
        selection_context: PluginSelectionContext | None = None,
        limit: int | None = None,
    ) -> list[PluginReference]:
        graph = cls.build_capability_graph(
            knowledge_base,
            selection_context=selection_context,
            limit=limit,
        )
        return build_plugin_references(graph, limit=limit)

    @classmethod
    def build_command_tool_snapshots(
        cls,
        knowledge_base: PluginKnowledgeBase,
        *,
        selection_context: PluginSelectionContext | None = None,
        limit: int | None = None,
    ) -> list[CommandToolSnapshot]:
        """构建 router 可见命令；必须复用同一权限过滤边界。"""

        snapshots = cls._get_cached_command_tool_snapshots(
            knowledge_base,
            selection_context=selection_context,
        )
        if selection_context is not None:
            snapshots = [
                snapshot
                for snapshot in snapshots
                if cls._is_command_tool_available(snapshot, selection_context)
            ]
        if limit is not None:
            return snapshots[: max(int(limit), 0)]
        return snapshots

    @classmethod
    def _get_cached_command_tool_snapshots(
        cls,
        knowledge_base: PluginKnowledgeBase,
        *,
        selection_context: PluginSelectionContext | None,
    ) -> list[CommandToolSnapshot]:
        from .command_metadata_overrides import load_command_overrides

        overrides = load_command_overrides()
        cache_key = (
            cls._command_tool_cache_key(
                knowledge_base,
                selection_context=selection_context,
            ),
            overrides.version,
        )
        cached = cls._command_tool_cache.get(cache_key)
        if cached is not None:
            try:
                cls._command_tool_cache_order.remove(cache_key)
            except ValueError:
                pass
            cls._command_tool_cache_order.append(cache_key)
            return list(cached[0])

        graph = cls.build_capability_graph(
            knowledge_base,
            selection_context=selection_context,
            limit=None,
        )
        snapshots = overrides.apply(
            cls._dedupe_execution_identity_snapshots(
                build_command_tool_snapshots(graph, limit=None)
            )
        )
        snapshots = cls._attach_semantic_tool_contracts(
            snapshots,
            knowledge_base=knowledge_base,
        )
        cls._command_tool_cache[cache_key] = (
            list(snapshots),
            len(snapshots),
            knowledge_base,
        )
        cls._command_tool_cache_order.append(cache_key)
        while len(cls._command_tool_cache_order) > cls._command_tool_cache_max:
            old_key = cls._command_tool_cache_order.pop(0)
            cls._command_tool_cache.pop(old_key, None)
        return list(snapshots)

    @staticmethod
    def _dedupe_execution_identity_snapshots(
        snapshots: list[CommandToolSnapshot],
    ) -> list[CommandToolSnapshot]:
        deduped: list[CommandToolSnapshot] = []
        seen: set[tuple[object, ...]] = set()
        for snapshot in snapshots:
            matcher_key = str(snapshot.matcher_key or "").strip().casefold()
            source_signature = str(snapshot.source_signature or "").strip()
            if not matcher_key and not source_signature:
                deduped.append(snapshot)
                continue
            execution_id = (
                ("matcher", matcher_key)
                if matcher_key
                else ("source", source_signature)
            )
            identity = (
                execution_id,
                normalize_message_text(snapshot.render).casefold(),
                tuple(slot.model_dump_json() for slot in snapshot.slots),
            )
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(snapshot)
        return deduped

    @classmethod
    def _attach_semantic_tool_contracts(
        cls,
        snapshots: list[CommandToolSnapshot],
        *,
        knowledge_base: PluginKnowledgeBase,
    ) -> list[CommandToolSnapshot]:
        updated = list(snapshots)
        for plugin in knowledge_base.plugins:
            module_key = normalize_message_text(plugin.module).casefold()
            if not module_key:
                continue
            for contract in plugin.semantic_tools:
                bound = {
                    normalize_message_text(value).casefold()
                    for value in contract.bound_commands
                    if normalize_message_text(value)
                }
                if not bound:
                    continue
                matches = [
                    index
                    for index, snapshot in enumerate(updated)
                    if normalize_message_text(snapshot.plugin_module).casefold()
                    == module_key
                    and normalize_message_text(snapshot.head).casefold() in bound
                ]
                if len(matches) != 1:
                    continue
                index = matches[0]
                snapshot = updated[index]
                if not cls._semantic_contract_matches_slots(contract, snapshot):
                    continue
                meta = dict(snapshot.meta or {})
                meta["semantic_tool_name"] = contract.name
                meta["semantic_contract"] = contract.model_dump(mode="json")
                use_cases = cls._merge_unique_strings(
                    snapshot.use_cases,
                    [contract.description, *contract.use_cases],
                )
                anti_use_cases = cls._merge_unique_strings(
                    snapshot.anti_use_cases,
                    contract.anti_use_cases,
                )
                update: dict[str, object] = {
                    "description": snapshot.description or contract.description,
                    "capability_text": snapshot.capability_text or contract.description,
                    "use_cases": use_cases,
                    "anti_use_cases": anti_use_cases,
                    "meta": meta,
                }
                for field_name in (
                    "output_mode",
                    "side_effect",
                    "source_of_truth",
                    "requires_real_tool",
                    "entity_scope",
                    "requires_real_result",
                    "execution_policy",
                ):
                    value = getattr(contract, field_name)
                    if value is not None:
                        update[field_name] = value
                if contract.risk is not None:
                    update["risk"] = contract.risk
                    update["risk_level"] = contract.risk
                if contract.intent_types:
                    update["intent_types"] = cls._merge_unique_strings(
                        snapshot.intent_types,
                        contract.intent_types,
                    )
                updated[index] = snapshot.model_copy(
                    update=update,
                )
        return updated

    @staticmethod
    def _semantic_contract_matches_slots(
        contract: SemanticToolContract,
        snapshot: CommandToolSnapshot,
    ) -> bool:
        parameters = dict(contract.parameters or {})
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return False
        property_names = {
            normalize_message_text(str(name or "")) for name in properties
        }
        slot_by_name = {
            normalize_message_text(slot.name): slot
            for slot in snapshot.slots
            if normalize_message_text(slot.name)
        }
        if not property_names or property_names != set(slot_by_name):
            return False
        required = {
            normalize_message_text(str(name or ""))
            for name in parameters.get("required", []) or []
        }
        slot_required = {name for name, slot in slot_by_name.items() if slot.required}
        if required != slot_required:
            return False
        expected_types = {
            "text": "string",
            "str": "string",
            "image": "string",
            "at": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }
        for name, slot in slot_by_name.items():
            raw_schema = properties.get(name)
            if not isinstance(raw_schema, dict):
                return False
            declared = raw_schema.get("type")
            if isinstance(declared, list):
                declared_types = {str(item) for item in declared}
            else:
                declared_types = {str(declared or "")}
            expected = expected_types.get(slot.type, "string")
            if expected not in declared_types:
                return False
        return True

    @classmethod
    def _command_tool_cache_key(
        cls,
        knowledge_base: PluginKnowledgeBase,
        *,
        selection_context: PluginSelectionContext | None,
    ) -> tuple[object, ...]:
        selection_key = cls._command_tool_selection_cache_key(selection_context)
        knowledge_identity = cls._command_tool_knowledge_identity(knowledge_base)
        if knowledge_identity is not None:
            return (
                "identity",
                cls._knowledge_revision,
                knowledge_identity,
                selection_key,
            )

        digest = hashlib.blake2b(digest_size=16)
        digest.update(knowledge_base.model_dump_json().encode("utf-8", "ignore"))

        return (
            "content",
            cls._knowledge_revision,
            digest.hexdigest(),
            selection_key,
        )

    @classmethod
    def _command_tool_knowledge_identity(
        cls,
        knowledge_base: PluginKnowledgeBase,
    ) -> tuple[object, ...] | None:
        for cached_knowledge, _cached_time in cls._cache.values():
            if cached_knowledge is knowledge_base:
                return (id(cached_knowledge), "all")
            positions = {
                id(plugin): index
                for index, plugin in enumerate(cached_knowledge.plugins)
            }
            projection: list[int] = []
            for plugin in knowledge_base.plugins:
                index = positions.get(id(plugin))
                if (
                    index is None
                    or cached_knowledge.plugins[index] is not plugin
                ):
                    break
                projection.append(index)
            else:
                return (
                    id(cached_knowledge),
                    tuple(projection),
                    knowledge_base.user_role,
                )
        return None

    @classmethod
    def _command_tool_selection_cache_key(
        cls,
        selection_context: PluginSelectionContext | None,
    ) -> tuple[object, ...]:
        if selection_context is None:
            return ("none",)
        group_id = str(selection_context.group_id or "").strip()
        return (
            group_id,
            bool(selection_context.is_superuser),
            bool(selection_context.is_private),
            cls._group_block_cache_signature(group_id),
            cls._session_override_cache_signature(selection_context.session_id),
        )

    @classmethod
    def _session_override_cache_signature(
        cls,
        session_id: str | None,
    ) -> tuple[tuple[str, bool], ...]:
        """会话级插件开关签名；无覆盖的会话共享同一缓存条目。"""
        normalized = str(session_id or "").strip()
        if not normalized:
            return ()
        overrides = cls._session_plugin_overrides.get(normalized)
        if not overrides:
            return ()
        return tuple(
            sorted((str(key), bool(value)) for key, value in overrides.items())
        )

    @staticmethod
    def _group_block_cache_signature(
        group_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not group_id:
            return (), ()
        group = GroupMemoryCache.get_if_ready(group_id)
        if group is None:
            return (), ()

        def _block_set(raw_name: str, set_name: str) -> tuple[str, ...]:
            raw = getattr(group, raw_name, "") or ""
            values = getattr(group, set_name, None)
            if values is None or (not values and raw):
                values = _parse_block_modules(raw)
            return tuple(sorted(str(value) for value in (values or set())))

        return (
            _block_set("block_plugin", "block_plugin_set"),
            _block_set("superuser_block_plugin", "superuser_block_plugin_set"),
        )

    @classmethod
    def _clear_command_tool_cache(cls, *, bump_revision: bool = False) -> None:
        cls._command_tool_cache.clear()
        cls._command_tool_cache_order.clear()
        cls._filter_kb_cache.clear()
        cls._filter_kb_cache_order.clear()
        if bump_revision:
            cls._knowledge_revision += 1

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
            cls._clear_command_tool_cache(bump_revision=True)

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
            cls._clear_command_tool_cache(bump_revision=True)

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
            logger.info(
                f"ChatInter 知识库缓存预加载完成，"
                f"共缓存 {len(normal_cache.plugins)} 个插件"
            )

        except Exception as e:
            logger.error(
                "预加载知识库缓存失败：" f"{e}\n{traceback.format_exc(limit=8)}"
            )


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
