"""Event/runtime helpers for ChatInter handler."""

from __future__ import annotations

import time

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger

from .plugin_registry import PluginRegistry
from .route_text import normalize_message_text, should_force_knowledge_refresh

_HANDLED_MESSAGE_IDS: set[str] = set()
_MAX_HANDLED_CACHE = 1000
_KNOWLEDGE_REFRESH_COOLDOWN = 30.0
_last_knowledge_refresh_ts = 0.0
_ENABLE_PLUGINS_ATTR = "_chatinter_enable_plugins"
_DISABLE_PLUGINS_ATTR = "_chatinter_disable_plugins"


def is_already_handled(event: Event) -> bool:
    message_id = getattr(event, "message_id", None)
    if not message_id:
        return False
    return str(message_id) in _HANDLED_MESSAGE_IDS


def mark_as_handled(event: Event) -> None:
    message_id = getattr(event, "message_id", None)
    if not message_id:
        return
    _HANDLED_MESSAGE_IDS.add(str(message_id))
    if len(_HANDLED_MESSAGE_IDS) > _MAX_HANDLED_CACHE:
        for old_id in list(_HANDLED_MESSAGE_IDS)[: len(_HANDLED_MESSAGE_IDS) // 2]:
            _HANDLED_MESSAGE_IDS.discard(old_id)


def get_nickname(session: Uninfo) -> str:
    return (
        getattr(session.user, "nick", None)
        or getattr(session.user, "name", None)
        or str(session.user.id)
    )


def resolve_superuser(bot: Bot, user_id: str) -> bool:
    platform_user = f"{getattr(bot, 'type', '')}:{user_id}"
    return user_id in BotConfig.superusers or platform_user in BotConfig.superusers


def event_type_name(event: Event) -> str:
    return str(getattr(event, "post_type", "") or event.__class__.__name__)


def event_adapter_name(bot: Bot) -> str:
    return str(getattr(bot, "type", "") or bot.__class__.__module__.split(".")[0])


def event_is_private(event: Event) -> bool:
    message_type = str(getattr(event, "message_type", "") or "").lower()
    if message_type:
        return message_type == "private"
    return "private" in event.__class__.__name__.lower()


def iter_runtime_plugin_overrides(event: Event, attr_name: str) -> set[str]:
    values = getattr(event, attr_name, None)
    if not values:
        return set()
    if isinstance(values, str):
        values = {values}
    return {
        normalize_message_text(str(item or ""))
        for item in values
        if normalize_message_text(str(item or ""))
    }


async def apply_runtime_plugin_overrides(
    *,
    event: Event,
    session_key: str | None,
    group_id: str | None,
) -> None:
    global _last_knowledge_refresh_ts

    enable_modules = iter_runtime_plugin_overrides(event, _ENABLE_PLUGINS_ATTR)
    disable_modules = iter_runtime_plugin_overrides(event, _DISABLE_PLUGINS_ATTR)
    force_refresh = should_force_knowledge_refresh(
        list(enable_modules | disable_modules)
    )
    if not force_refresh:
        return

    now = time.monotonic()
    if (now - _last_knowledge_refresh_ts) < _KNOWLEDGE_REFRESH_COOLDOWN:
        return
    _last_knowledge_refresh_ts = now
    await PluginRegistry.preload_cache(force_refresh=True)
    logger.debug(
        "ChatInter 已按运行态插件覆盖刷新知识库: "
        f"session={session_key or group_id or 'global'} "
        f"enable={sorted(enable_modules)} disable={sorted(disable_modules)}"
    )


__all__ = [
    "apply_runtime_plugin_overrides",
    "event_adapter_name",
    "event_is_private",
    "event_type_name",
    "get_nickname",
    "is_already_handled",
    "mark_as_handled",
    "resolve_superuser",
]
