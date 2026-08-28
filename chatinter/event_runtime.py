"""Event/runtime helpers for ChatInter handler."""

from __future__ import annotations

from collections import OrderedDict

from nonebot import get_driver
from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger

from .plugin_registry import PluginRegistry
from .route_text import normalize_message_text

_HANDLED_MESSAGE_IDS: OrderedDict[str, None] = OrderedDict()
_MAX_HANDLED_CACHE = 1000
_DISABLE_PLUGINS_ATTR = "_chatinter_disable_plugins"


def _handled_message_key(event: Event, *, session_key: str) -> str:
    message_id = getattr(event, "message_id", None)
    normalized_session = str(session_key or "").strip()
    if not message_id or not normalized_session:
        return ""
    return f"{normalized_session}\0{message_id}"


def is_already_handled(event: Event, *, session_key: str) -> bool:
    key = _handled_message_key(event, session_key=session_key)
    return bool(key and key in _HANDLED_MESSAGE_IDS)


def mark_as_handled(event: Event, *, session_key: str) -> None:
    normalized_id = _handled_message_key(event, session_key=session_key)
    if not normalized_id:
        return
    if normalized_id in _HANDLED_MESSAGE_IDS:
        return
    _HANDLED_MESSAGE_IDS[normalized_id] = None
    while len(_HANDLED_MESSAGE_IDS) > _MAX_HANDLED_CACHE:
        _HANDLED_MESSAGE_IDS.popitem(last=False)


def get_nickname(session: Uninfo) -> str:
    return (
        getattr(session.user, "nick", None)
        or getattr(session.user, "name", None)
        or str(session.user.id)
    )


def resolve_superuser(bot: Bot, user_id: str) -> bool:
    platform = str(getattr(bot, "type", "") or "")
    platform_user = f"{platform}:{user_id}" if platform else user_id
    driver_superusers = {
        str(item) for item in getattr(get_driver().config, "superusers", set())
    }
    platform_superusers = {str(item) for item in BotConfig.get_superuser(platform)}
    return (
        user_id in driver_superusers
        or platform_user in driver_superusers
        or user_id in platform_superusers
        or platform_user in platform_superusers
    )


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
    disable_modules = iter_runtime_plugin_overrides(event, _DISABLE_PLUGINS_ATTR)
    if not disable_modules:
        return
    for module in sorted(disable_modules):
        await PluginRegistry.set_plugin_enabled(
            plugin_key=module,
            enabled=False,
            session_id=session_key,
            group_id=group_id,
        )
    logger.debug(
        "ChatInter 已应用运行态插件关闭: "
        f"session={session_key or group_id or 'global'} "
        f"disable={sorted(disable_modules)}"
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
