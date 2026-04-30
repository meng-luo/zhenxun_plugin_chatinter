"""
ChatInter 插件能力图。

这一层只负责把 PluginInfo 转成结构化能力，不负责真正执行插件。
"""

from __future__ import annotations

import time

from .models.pydantic_models import (
    CapabilityGraphSnapshot,
    CommandCapability,
    CommandRequirement,
    PluginCapability,
    PluginInfo,
    PluginKnowledgeBase,
)
from .route_text import normalize_message_text

_INFRA_MODULE_TAILS = {
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
_INFRA_MODULE_MARKERS = (
    ".builtin_plugins.hooks",
    ".builtin_plugins.init",
    ".builtin_plugins.scheduler",
    ".services.",
    ".webui",
    ".web_ui",
)
_PRIVATE_ACCESS_LEVELS = {"admin", "superuser", "restricted"}


def _append_unique(target: list[str], value: object) -> None:
    text = normalize_message_text(str(value or ""))
    if text and text not in target:
        target.append(text)


def _normalize_bool(value: bool | None) -> bool:
    return bool(value) if value is not None else False


def _module_tail(module: str) -> str:
    normalized = normalize_message_text(module)
    return normalized.rsplit(".", 1)[-1] if normalized else ""


def is_public_capability_source(plugin: PluginInfo) -> bool:
    """保守判断插件是否允许进入能力图。"""
    module = normalize_message_text(plugin.module)
    name = normalize_message_text(plugin.name)
    if not module or not name:
        return False
    if not plugin.commands and not plugin.command_meta:
        return False
    if plugin.limit_superuser:
        return False
    if int(plugin.admin_level or 0) > 0:
        return False
    module_lower = module.lower()
    if _module_tail(module_lower) in _INFRA_MODULE_TAILS:
        return False
    if any(marker in module_lower for marker in _INFRA_MODULE_MARKERS):
        return False
    public_meta = [
        meta
        for meta in plugin.command_meta
        if normalize_message_text(meta.access_level).lower()
        not in _PRIVATE_ACCESS_LEVELS
    ]
    return bool(public_meta or not plugin.command_meta)


def requirement_from_meta(meta: PluginInfo.PluginCommandMeta) -> CommandRequirement:
    return CommandRequirement(
        params=[
            normalize_message_text(param)
            for param in meta.params
            if normalize_message_text(param)
        ],
        text_min=max(int(meta.text_min or 0), 0),
        text_max=meta.text_max,
        image_min=max(int(meta.image_min or 0), 0),
        image_max=meta.image_max,
        allow_at=_normalize_bool(meta.allow_at),
        actor_scope=meta.actor_scope,
        target_requirement=meta.target_requirement,
        target_sources=list(meta.target_sources),
        requires_reply=bool(meta.requires_reply),
        requires_private=bool(meta.requires_private),
        requires_to_me=bool(meta.requires_to_me),
    )


def capability_from_meta(
    meta: PluginInfo.PluginCommandMeta,
) -> CommandCapability | None:
    command = normalize_message_text(meta.command)
    if not command:
        return None
    aliases: list[str] = []
    prefixes: list[str] = []
    examples: list[str] = []
    for alias in meta.aliases:
        _append_unique(aliases, alias)
    for prefix in meta.prefixes:
        _append_unique(prefixes, prefix)
    for example in meta.examples:
        _append_unique(examples, example)
    return CommandCapability(
        command=command,
        aliases=aliases,
        prefixes=prefixes,
        examples=examples,
        requirement=requirement_from_meta(meta),
        allow_sticky_arg=bool(meta.allow_sticky_arg),
    )


def fallback_capabilities_from_plugin(plugin: PluginInfo) -> list[CommandCapability]:
    commands: list[str] = []
    aliases: list[str] = []
    for command in plugin.commands:
        _append_unique(commands, command)
    for alias in plugin.aliases:
        _append_unique(aliases, alias)

    capabilities: list[CommandCapability] = []
    for index, command in enumerate(commands):
        command_aliases = aliases if index == 0 else []
        capabilities.append(CommandCapability(command=command, aliases=command_aliases))
    return capabilities


def capability_from_plugin(plugin: PluginInfo) -> PluginCapability | None:
    if not is_public_capability_source(plugin):
        return None

    commands: list[CommandCapability] = []
    seen_commands: set[str] = set()
    for meta in plugin.command_meta:
        if normalize_message_text(meta.access_level).lower() in _PRIVATE_ACCESS_LEVELS:
            continue
        capability = capability_from_meta(meta)
        if capability is None or capability.command in seen_commands:
            continue
        seen_commands.add(capability.command)
        commands.append(capability)

    if not commands:
        commands = fallback_capabilities_from_plugin(plugin)

    if not commands:
        return None

    aliases: list[str] = []
    for alias in plugin.aliases:
        _append_unique(aliases, alias)

    return PluginCapability(
        module=normalize_message_text(plugin.module),
        name=normalize_message_text(plugin.name),
        description=normalize_message_text(plugin.description),
        usage=normalize_message_text(plugin.usage or "") or None,
        commands=commands,
        aliases=aliases,
        public=True,
    )


def build_capability_graph_snapshot(
    knowledge_base: PluginKnowledgeBase,
    *,
    limit: int | None = None,
) -> CapabilityGraphSnapshot:
    plugins: list[PluginCapability] = []
    max_items = max(int(limit), 0) if limit is not None else None
    if max_items == 0:
        return CapabilityGraphSnapshot(
            plugins=[],
            user_role=knowledge_base.user_role,
            created_at=time.time(),
        )

    for plugin in knowledge_base.plugins:
        capability = capability_from_plugin(plugin)
        if capability is None:
            continue
        plugins.append(capability)
        if max_items is not None and len(plugins) >= max_items:
            break

    return CapabilityGraphSnapshot(
        plugins=plugins,
        user_role=knowledge_base.user_role,
        created_at=time.time(),
    )
