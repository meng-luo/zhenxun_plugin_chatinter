"""
ChatInter - 主处理器

实现消息处理流程，支持多模态输入（图片识别）。
使用 UniMessage 统一处理消息。
"""

import asyncio
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
import hashlib
import re
import time

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger
from zhenxun.utils.message import MessageUtils

from .chat_handler import (
    handle_chat_message,
    normalize_ai_reply_text,
    replace_mention_ids_with_names,
    reroute_to_plugin,
)
from .config import get_config_value
from .lifecycle import LifecyclePayload, get_lifecycle_manager
from .memory import _chat_memory
from .plugin_registry import (
    PluginRegistry,
    PluginSelectionContext,
    get_user_plugin_knowledge,
)
from .route_engine import (
    RouteResolveResult,
    resolve_llm_route,
    resolve_pre_route,
    resolve_semantic_route,
)
from .route_text import (
    ROUTE_ACTION_WORDS,
    contains_any,
    collect_placeholders,
    has_negative_route_intent,
    is_usage_question,
    match_command_head,
    normalize_action_phrases,
    normalize_message_text,
    should_force_knowledge_refresh,
    strip_invoke_prefix,
)
from .trace import StageTrace
from .utils.multimodal import extract_images_from_message
from .utils.unimsg_utils import remove_reply_segment, uni_to_text_with_tags

_HANDLED_MESSAGE_IDS: set[str] = set()
_MAX_HANDLED_CACHE = 1000
_KNOWLEDGE_REFRESH_COOLDOWN = 30.0
_last_knowledge_refresh_ts = 0.0
_ENABLE_PLUGINS_ATTR = "_chatinter_enable_plugins"
_DISABLE_PLUGINS_ATTR = "_chatinter_disable_plugins"
_AT_ID_TOKEN_PATTERN = re.compile(
    r"\[@(\d{5,20})\]|(?<![0-9A-Za-z_])@(\d{5,20})(?=(?:\s|$|[的，,。.!！？?]))"
)
_FOLLOWUP_IMAGE_HINTS = (
    "[image]",
    "发你的头像",
    "发头像",
    "发送头像",
    "发送图片",
    "发图",
    "发一张图",
    "补一张图",
    "先发图",
)
_FOLLOWUP_MEME_HINTS = ("表情", "表情包", "梗图", "头像")
_PENDING_IMAGE_TTL = 60.0
_PENDING_IMAGE_FOLLOWUPS: dict[str, "PendingImageFollowup"] = {}
_PENDING_TARGET_TTL = 120.0
_PENDING_TARGET_FOLLOWUPS: dict[str, "PendingTargetFollowup"] = {}
_PLACEHOLDER_SEGMENT_PATTERN = re.compile(r"\[@[^\]]+\]|\[image(?:#\d+)?\]")
_REPLY_TAG_PATTERN = re.compile(r"\[reply:[^\]]+\]", re.IGNORECASE)
_REPLY_REF_HINTS = (
    "回复",
    "引用",
    "上面",
    "这条",
    "这张",
    "这图",
    "这个图",
    "这张图",
    "用这张",
)
_SELF_REF_HINTS = ("我", "自己", "本人", "我的", "我自己", "自己的")
_THIRD_PERSON_HINTS = ("他", "她", "ta", "对方", "那位", "这个人", "上面那位")
_SOFT_INVOKE_PREFIXES = (
    "请你",
    "麻烦你",
    "请帮我",
    "麻烦帮我",
    "请给我",
    "麻烦给我",
    "能不能帮我",
    "能否帮我",
    "可以帮我",
    "你帮我",
    "帮我",
    "给我",
    "替我",
)
_EXECUTION_INTENT_HINTS = (
    "帮我",
    "帮忙",
    "请",
    "麻烦",
    "执行",
    "调用",
    "使用",
    "打开",
    "关闭",
    "开启",
    "禁用",
    "设置",
    "查看",
    "看看",
    "看下",
    "查询",
    "生成",
    "制作",
    "发送",
    "来个",
    "来一个",
    "来一张",
    "做个",
    "做一个",
    "做一张",
    "再来个",
    "再来一个",
    "再来一张",
)
_ROUTE_META_CHAT_HINTS = (
    "刚有人说",
    "有人说了",
    "我觉得挺有意思",
    "只是提到",
    "不是在让你执行",
    "不是让你执行",
)
_CUTE_NOTIFY_TEMPLATES = (
    "好、好啦，真寻这就帮你{target}。",
    "收到啦，我马上就去{target}。",
    "唔，知道啦，这就给你{target}。",
    "诶嘿，安排安排，这就{target}。",
    "等一下下，我这就帮你{target}。",
    "哼哼，这点小事马上给你{target}。",
)
_CUTE_MEME_NOTIFY_TEMPLATES = (
    "好、好啦，真寻这就做{target}。",
    "收到啦，马上给你做{target}。",
    "诶嘿，开工咯，这就做{target}。",
    "等我一下下，这就把{target}做出来。",
    "哼，这个我超会，立刻做{target}。",
    "软乎乎开工中，马上给你{target}。",
)
_CUTE_MEME_HELPER_TEMPLATES = (
    "好、好啦，这就给你{target}。",
    "收到啦，我马上帮你{target}。",
    "唔，知道啦，这就去{target}。",
    "安排上啦，立刻给你{target}。",
    "等一下下，这就帮你{target}。",
)
_MEME_HELPER_COMMANDS = {"表情搜索", "表情详情", "启用表情", "更新表情"}
_FUZZY_TARGET_HINT_PATTERN = re.compile(
    r"(?:给|帮|替|让|叫|喊|请)(?!我|自己|本人)(?P<name>[A-Za-z0-9\u4e00-\u9fff]{1,16}?)(?=(?:做|整|弄|来|发|签|点|查|看|问|生成|制作|的|表情|头像|图片|图|一下|一张|一个|个|张|首|[\s，,。.!！？?]|$))"
)
_FUZZY_TARGET_SUFFIX_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9\u4e00-\u9fff]{2,16})(?:的)?(?=(?:表情|头像|图片|图|看书|签到|打卡|一直|敲|吃|摸|抱|捶|顶|打|贴|摸摸|[\s，,。.!！？?]|$))"
)
_SELF_ONLY_ACTION_KEYWORDS = ("签到", "打卡", "补签")
_TARGET_REQUIRED_ACTION_HINTS = ("给", "帮", "替", "让", "叫")
_TECHNICAL_REQUEST_HINT_WORDS = (
    "nonebot",
    "插件",
    "bot",
    "代码",
    "脚本",
    "函数",
    "类",
    "接口",
    "报错",
    "bug",
    "错误",
    "调试",
    "配置",
    "部署",
    "安装",
    "开发",
    "仓库",
    "git",
    "pull",
    "push",
)
_NON_SELF_TARGET_PATTERN = re.compile(r"(?:给|帮|替|让|叫|喊|请)(?!我|自己|本人)")
_GROUP_MEMBER_PROFILE_CACHE_TTL = 90.0
_GROUP_MEMBER_PROFILE_CACHE: dict[
    str, tuple[float, list[dict[str, str | tuple[str, ...]]]]
] = {}
_GROUP_ACTIVE_RANK_CACHE_TTL = 30.0
_GROUP_ACTIVE_RANK_CACHE: dict[str, tuple[float, dict[str, float]]] = {}
_NICKNAME_RESOLUTION_MEMORY_TTL = 12 * 3600.0
_NICKNAME_RESOLUTION_MEMORY: dict[str, tuple[float, str]] = {}


@dataclass(frozen=True)
class PendingImageFollowup:
    user_id: str
    group_id: str | None
    original_message: str
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class PendingTargetFollowup:
    user_id: str
    group_id: str | None
    original_message: str
    target_hint: str
    candidate_user_ids: tuple[str, ...]
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class RouteExecutionPlan:
    command: str
    need_followup: bool = False
    followup_message: str | None = None
    wait_for_image: bool = False


class ChannelName(str, Enum):
    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    FINAL = "final"


@dataclass
class TurnChannelEnvelope:
    analysis: list[str] = field(default_factory=list)
    commentary: list[str] = field(default_factory=list)
    final: str = ""

    def add(self, channel: ChannelName, content: str) -> None:
        raw_text = str(content or "")
        if channel is ChannelName.FINAL:
            # 最终回复保留换行和代码块格式，仅裁剪首尾空白。
            text = raw_text.strip()
            if text:
                self.final = text
            return

        text = normalize_message_text(raw_text)
        if not text:
            return
        if channel is ChannelName.ANALYSIS:
            self.analysis.append(text)
        else:
            self.commentary.append(text)


def _log_turn_channels(envelope: TurnChannelEnvelope) -> None:
    if envelope.analysis:
        logger.debug("[ChatInter][analysis] " + " | ".join(envelope.analysis))
    if envelope.commentary:
        logger.debug("[ChatInter][commentary] " + " | ".join(envelope.commentary))


def _message_matches_known_command_prefix(
    message_text: str,
    knowledge_base,
) -> bool:
    plugins = getattr(knowledge_base, "plugins", None) or []
    if not plugins:
        return False

    normalized = normalize_message_text(message_text or "")
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if not stripped:
        return False

    candidates: list[str] = [stripped]
    soft_stripped = stripped
    while soft_stripped:
        matched = next(
            (
                prefix
                for prefix in _SOFT_INVOKE_PREFIXES
                if soft_stripped.lower().startswith(prefix.lower())
            ),
            None,
        )
        if matched is None:
            break
        soft_stripped = normalize_message_text(soft_stripped[len(matched) :])
        if soft_stripped and soft_stripped not in candidates:
            candidates.append(soft_stripped)
    if stripped.startswith("你") and len(stripped) > 1:
        trimmed = normalize_message_text(stripped[1:])
        if trimmed and trimmed not in candidates:
            candidates.append(trimmed)

    for plugin in plugins:
        for command in getattr(plugin, "commands", []) or []:
            command_text = normalize_message_text(str(command or ""))
            if not command_text:
                continue
            command_head = normalize_message_text(command_text.split(" ", 1)[0])
            if not command_head:
                continue
            for candidate in candidates:
                if _match_command_head_or_sticky(candidate, command_head):
                    return True
    return False


def _should_allow_rule_fallback(message_text: str, knowledge_base) -> bool:
    normalized = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    if not normalized:
        return False
    if has_negative_route_intent(normalized):
        return False
    if contains_any(normalized, _ROUTE_META_CHAT_HINTS):
        return False
    if is_usage_question(normalized):
        return False
    if _message_matches_known_command_prefix(normalized, knowledge_base):
        return True
    if contains_any(normalized, _EXECUTION_INTENT_HINTS):
        return True
    if "[image" in normalized:
        return True
    if "[@" in normalized and contains_any(normalized, _FOLLOWUP_MEME_HINTS):
        return True
    return False


async def _persist_final_only_dialog(
    *,
    envelope: TurnChannelEnvelope,
    user_id: str,
    group_id: str | None,
    nickname: str,
    user_message,
    bot_id: str | None,
) -> None:
    final_text = str(envelope.final or "").strip()
    if not final_text:
        return
    await _chat_memory.add_dialog(
        user_id=user_id,
        group_id=group_id,
        nickname=nickname,
        user_message=user_message,
        ai_response=final_text,
        bot_id=bot_id,
    )


def _build_pending_session_key(user_id: str, group_id: str | None) -> str:
    return f"{group_id or 'private'}:{user_id}"


def _prune_pending_image_followups(now: float | None = None) -> None:
    current = now if now is not None else time.monotonic()
    expired_keys = [
        key
        for key, pending in _PENDING_IMAGE_FOLLOWUPS.items()
        if pending.expires_at <= current
    ]
    for key in expired_keys:
        _PENDING_IMAGE_FOLLOWUPS.pop(key, None)


def _prune_pending_target_followups(now: float | None = None) -> None:
    current = now if now is not None else time.monotonic()
    expired_keys = [
        key
        for key, pending in _PENDING_TARGET_FOLLOWUPS.items()
        if pending.expires_at <= current
    ]
    for key in expired_keys:
        _PENDING_TARGET_FOLLOWUPS.pop(key, None)


def set_pending_image_followup(
    user_id: str,
    group_id: str | None,
    original_message: str,
) -> None:
    now = time.monotonic()
    _prune_pending_image_followups(now)
    key = _build_pending_session_key(user_id, group_id)
    _PENDING_IMAGE_FOLLOWUPS[key] = PendingImageFollowup(
        user_id=user_id,
        group_id=group_id,
        original_message=original_message.strip(),
        created_at=now,
        expires_at=now + _PENDING_IMAGE_TTL,
    )


def clear_pending_image_followup(user_id: str, group_id: str | None) -> None:
    _prune_pending_image_followups()
    key = _build_pending_session_key(user_id, group_id)
    _PENDING_IMAGE_FOLLOWUPS.pop(key, None)


def has_pending_image_followup(user_id: str, group_id: str | None) -> bool:
    _prune_pending_image_followups()
    key = _build_pending_session_key(user_id, group_id)
    return key in _PENDING_IMAGE_FOLLOWUPS


def set_pending_target_followup(
    user_id: str,
    group_id: str | None,
    original_message: str,
    *,
    target_hint: str = "",
    candidate_user_ids: tuple[str, ...] = (),
) -> None:
    now = time.monotonic()
    _prune_pending_target_followups(now)
    key = _build_pending_session_key(user_id, group_id)
    _PENDING_TARGET_FOLLOWUPS[key] = PendingTargetFollowup(
        user_id=user_id,
        group_id=group_id,
        original_message=normalize_message_text(original_message),
        target_hint=normalize_message_text(target_hint),
        candidate_user_ids=tuple(str(uid) for uid in candidate_user_ids if str(uid).isdigit()),
        created_at=now,
        expires_at=now + _PENDING_TARGET_TTL,
    )


def get_pending_target_followup(
    user_id: str,
    group_id: str | None,
) -> PendingTargetFollowup | None:
    _prune_pending_target_followups()
    key = _build_pending_session_key(user_id, group_id)
    return _PENDING_TARGET_FOLLOWUPS.get(key)


def clear_pending_target_followup(user_id: str, group_id: str | None) -> None:
    _prune_pending_target_followups()
    key = _build_pending_session_key(user_id, group_id)
    _PENDING_TARGET_FOLLOWUPS.pop(key, None)


def claim_pending_image_followup(
    user_id: str, group_id: str | None
) -> PendingImageFollowup | None:
    _prune_pending_image_followups()
    key = _build_pending_session_key(user_id, group_id)
    return _PENDING_IMAGE_FOLLOWUPS.pop(key, None)


def build_pending_followup_message(
    original_message: str,
    followup_message_text: str,
) -> str:
    synthetic = original_message.strip()
    followup_text = str(followup_message_text or "").strip()
    if not followup_text:
        return synthetic
    extra_text = re.sub(r"\[image[^\]]*\]", " ", followup_text)
    extra_text = re.sub(r"\s+", " ", extra_text).strip(
        " ：:，,。.!！?？-[](){}<>【】（）《》「」『』"
    )
    parts = [synthetic]
    if extra_text:
        parts.append(extra_text)
    if "[image" in followup_text and "[image" not in synthetic:
        parts.append("[image]")
    return " ".join(part for part in parts if part).strip()


def build_pending_target_followup_message(
    original_message: str,
    followup_message_text: str,
    target_user_id: str,
) -> str:
    base = normalize_message_text(original_message)
    followup_text = normalize_message_text(followup_message_text)
    target_token = f"[@{target_user_id}]"
    parts = [base]
    if target_token not in base:
        parts.append(target_token)
    if "[image" in followup_text and "[image" not in base:
        parts.append("[image]")
    return normalize_message_text(" ".join(part for part in parts if part))


async def resolve_pending_target_followup_user_id(
    pending: PendingTargetFollowup,
    followup_message_text: str,
    group_id: str | None,
) -> str | None:
    tagged = normalize_message_text(followup_message_text)
    for token in _extract_at_tokens(tagged):
        user_id = _extract_user_id_from_at_token(token)
        if user_id and (
            not pending.candidate_user_ids or user_id in set(pending.candidate_user_ids)
        ):
            return user_id

    if not group_id:
        return None
    hint = _extract_fuzzy_target_hint(tagged) or tagged
    if len(_normalize_alias_key(hint)) < 2:
        return None
    profiles = await _get_group_member_profiles_for_fuzzy(group_id)
    if pending.candidate_user_ids:
        allowed = set(pending.candidate_user_ids)
        profiles = [
            profile
            for profile in profiles
            if str(profile.get("user_id") or "").strip() in allowed
        ]
    active_scores = await _get_group_recent_active_scores(group_id)
    matched, ambiguous, _ = _pick_fuzzy_target_profile(
        hint,
        profiles,
        active_scores,
        trigger_strength="strong",
    )
    if matched is None or ambiguous:
        return None
    user_id = str(matched.get("user_id") or "").strip()
    return user_id if user_id.isdigit() else None


def _should_wait_for_image_followup(
    message_text: str,
    reply_text: str,
) -> bool:
    normalized_message = str(message_text or "").strip()
    normalized_reply = str(reply_text or "").strip()
    if not normalized_message or not normalized_reply:
        return False
    if not any(hint in normalized_message for hint in _FOLLOWUP_MEME_HINTS):
        return False
    return any(hint in normalized_reply for hint in _FOLLOWUP_IMAGE_HINTS)


def _build_route_notify_text(
    plugin_name: str,
    plugin_module: str,
    route_command: str,
) -> str:
    def _render(template_pool: tuple[str, ...], *, target: str, seed: str) -> str:
        if not template_pool:
            return f"好哒，这就帮你{target}。"
        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(template_pool)
        return template_pool[index].format(target=target)

    normalized_command = normalize_message_text(route_command)
    command_head = normalized_command.split(" ", 1)[0] if normalized_command else ""
    target = command_head or normalize_message_text(plugin_name)
    seed_base = f"{plugin_module}|{plugin_name}|{normalized_command}"
    if not target:
        return _render(_CUTE_NOTIFY_TEMPLATES, target="处理一下", seed=seed_base)

    is_meme_like = "meme" in (plugin_module or "").lower() or "表情" in (plugin_name or "")
    if is_meme_like and not target.endswith("表情"):
        if target in _MEME_HELPER_COMMANDS:
            return _render(_CUTE_MEME_HELPER_TEMPLATES, target=target, seed=f"{seed_base}|helper")
        return _render(_CUTE_MEME_NOTIFY_TEMPLATES, target=f"{target}表情", seed=f"{seed_base}|meme")

    return _render(_CUTE_NOTIFY_TEMPLATES, target=target, seed=f"{seed_base}|default")


def _is_already_handled(event: Event) -> bool:
    """检查消息是否已被本插件处理过"""
    message_id = getattr(event, "message_id", None)
    if not message_id:
        return False
    return str(message_id) in _HANDLED_MESSAGE_IDS


def _mark_as_handled(event: Event):
    """标记消息已被处理"""
    message_id = getattr(event, "message_id", None)
    if not message_id:
        return
    if len(_HANDLED_MESSAGE_IDS) >= _MAX_HANDLED_CACHE:
        _HANDLED_MESSAGE_IDS.clear()
    _HANDLED_MESSAGE_IDS.add(str(message_id))


def _get_nickname(session: Uninfo) -> str:
    """获取用户昵称"""
    if (
        session.user
        and hasattr(session.user, "display_name")
        and session.user.display_name
    ):
        return session.user.display_name
    if session.user and hasattr(session.user, "name") and session.user.name:
        return session.user.name
    return "用户"


def _resolve_superuser(bot: Bot, user_id: str) -> bool:
    superusers = getattr(getattr(bot, "config", None), "superusers", set())
    return str(user_id) in {str(item) for item in superusers}


def _iter_runtime_plugin_overrides(event: Event, attr_name: str) -> set[str]:
    raw = getattr(event, attr_name, None)
    if raw is None:
        return set()
    if isinstance(raw, str):
        value = raw.strip()
        return {value} if value else set()
    if isinstance(raw, set | frozenset | tuple | list):
        values: set[str] = set()
        for item in raw:
            value = str(item).strip()
            if value:
                values.add(value)
        return values
    return set()


async def _apply_runtime_plugin_overrides(
    *,
    event: Event,
    session_key: str,
    group_id: str | None,
) -> None:
    await PluginRegistry.reset_dynamic_overrides(session_id=session_key)
    enable_keys = _iter_runtime_plugin_overrides(event, _ENABLE_PLUGINS_ATTR)
    disable_keys = _iter_runtime_plugin_overrides(event, _DISABLE_PLUGINS_ATTR)
    for key in enable_keys:
        await PluginRegistry.set_plugin_enabled(
            plugin_key=key,
            enabled=True,
            session_id=session_key,
            group_id=group_id,
        )
    for key in disable_keys:
        await PluginRegistry.set_plugin_enabled(
            plugin_key=key,
            enabled=False,
            session_id=session_key,
            group_id=group_id,
        )


def _extract_mentioned_user_ids(message_text: str) -> set[str]:
    mentioned_user_ids: set[str] = set()
    for match in _AT_ID_TOKEN_PATTERN.finditer(message_text or ""):
        user_id = (match.group(1) or match.group(2) or "").strip()
        if user_id:
            mentioned_user_ids.add(user_id)
    return mentioned_user_ids


def _build_mention_name_map(
    mention_profiles: dict[str, dict[str, str]],
) -> dict[str, str]:
    mention_name_map: dict[str, str] = {}
    for user_id, profile in mention_profiles.items():
        nickname = (
            str(profile.get("display_name") or profile.get("nickname") or "").strip()
        )
        if nickname:
            mention_name_map[user_id] = nickname
    return mention_name_map


def _normalize_alias_key(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(text or ""))
    return cleaned.lower().strip()


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


def _extract_user_id_from_at_token(token: str) -> str | None:
    text = normalize_message_text(token)
    if not text.startswith("[@") or not text.endswith("]"):
        return None
    user_id = text[2:-1].strip()
    return user_id if user_id.isdigit() else None


def _build_alias_keys(*names: str) -> tuple[str, ...]:
    keys: set[str] = set()
    for raw_name in names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        alias = _normalize_alias_key(name)
        if len(alias) >= 2:
            keys.add(alias)
            for size in (2, 3):
                if len(alias) >= size:
                    keys.add(alias[-size:])
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", name):
            normalized_chunk = _normalize_alias_key(chunk)
            if len(normalized_chunk) >= 2:
                keys.add(normalized_chunk)
                for size in (2, 3):
                    if len(normalized_chunk) >= size:
                        keys.add(normalized_chunk[-size:])
    return tuple(sorted(keys, key=len, reverse=True))


def _extract_fuzzy_target_hint(
    message_text: str,
    command_heads: set[str] | None = None,
) -> str:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return ""
    match = _FUZZY_TARGET_HINT_PATTERN.search(normalized)
    if match:
        return normalize_message_text(match.group("name") or "")

    if command_heads:
        for head in sorted(command_heads, key=len, reverse=True):
            normalized_head = normalize_message_text(head)
            if not normalized_head or not normalized.startswith(normalized_head):
                continue
            tail = normalize_message_text(normalized[len(normalized_head) :])
            tail = re.sub(r"^(?:给|帮|替|让|叫|喊|请)+", "", tail).strip()
            tail = re.sub(r"(?:做|整|弄|来|发|签|点|查|看|问|生成|制作).*$", "", tail)
            tail = tail.strip(" 的：:,，。.!！？?")
            if not tail:
                continue
            candidate = normalize_message_text(tail.split(" ", 1)[0])
            if _normalize_alias_key(candidate) in {"", "wo", "ziji"}:
                continue
            if candidate in _SELF_REF_HINTS:
                continue
            normalized_candidate = _normalize_alias_key(candidate)
            if len(normalized_candidate) > 16:
                continue
            if _is_technical_request_like(candidate):
                continue
            if len(normalized_candidate) >= 2:
                return candidate

    if contains_any(normalized, _TARGET_REQUIRED_ACTION_HINTS):
        suffix_match = _FUZZY_TARGET_SUFFIX_PATTERN.search(normalized)
        if suffix_match:
            candidate = normalize_message_text(suffix_match.group("name") or "")
            normalized_candidate = _normalize_alias_key(candidate)
            if len(normalized_candidate) > 16:
                return ""
            if _is_technical_request_like(candidate):
                return ""
            if len(normalized_candidate) >= 2:
                return candidate
    return ""


async def _get_group_member_profiles_for_fuzzy(
    group_id: str | None,
) -> list[dict[str, str | tuple[str, ...]]]:
    if not group_id:
        return []
    cache_key = str(group_id)
    now = time.monotonic()
    cached = _GROUP_MEMBER_PROFILE_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _GROUP_MEMBER_PROFILE_CACHE_TTL:
        return cached[1]

    try:
        from zhenxun.models.group_member_info import GroupInfoUser

        members = await GroupInfoUser.filter(group_id=group_id).all()
    except Exception as exc:
        logger.debug(f"加载群成员映射失败: {exc}")
        return []

    profiles: list[dict[str, str | tuple[str, ...]]] = []
    for member in members:
        user_id = str(member.user_id).strip()
        if not user_id.isdigit():
            continue
        nickname = (member.nickname or "").strip()
        user_name = (member.user_name or "").strip()
        display_name = (nickname or user_name).strip()
        if not display_name:
            continue
        uid = str(member.uid).strip() if member.uid is not None else ""
        platform = str(member.platform or "").strip() or "qq"
        alias_key = _normalize_alias_key(display_name)
        alias_keys = _build_alias_keys(display_name, nickname, user_name)
        profiles.append(
            {
                "user_id": user_id,
                "display_name": display_name,
                "nickname": nickname,
                "user_name": user_name,
                "uid": uid,
                "platform": platform,
                "alias_key": alias_key,
                "alias_keys": alias_keys,
            }
        )

    _GROUP_MEMBER_PROFILE_CACHE[cache_key] = (now, profiles)
    return profiles


async def _get_group_recent_active_scores(group_id: str | None) -> dict[str, float]:
    if not group_id:
        return {}
    cache_key = str(group_id)
    now = time.monotonic()
    cached = _GROUP_ACTIVE_RANK_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _GROUP_ACTIVE_RANK_CACHE_TTL:
        return cached[1]

    try:
        from zhenxun.models.chat_history import ChatHistory

        recent_rows = (
            await ChatHistory.filter(group_id=group_id)
            .order_by("-create_time", "-id")
            .limit(200)
            .values_list("user_id", flat=True)
        )
    except Exception as exc:
        logger.debug(f"加载群活跃度失败: {exc}")
        return {}

    score_map: dict[str, float] = {}
    rank = 0
    for raw_user_id in recent_rows:
        user_id = str(raw_user_id).strip()
        if not user_id.isdigit() or user_id in score_map:
            continue
        rank += 1
        score_map[user_id] = max(0.0, 0.08 - min(rank - 1, 10) * 0.006)
        if rank >= 20:
            break

    _GROUP_ACTIVE_RANK_CACHE[cache_key] = (now, score_map)
    return score_map


def _resolution_memory_key(group_id: str | None, target_hint: str) -> str:
    return f"{group_id or 'private'}:{_normalize_alias_key(target_hint)}"


def _remember_target_resolution(
    group_id: str | None,
    target_hint: str,
    user_id: str,
) -> None:
    normalized_hint = _normalize_alias_key(target_hint)
    if not normalized_hint or not str(user_id).isdigit():
        return
    _NICKNAME_RESOLUTION_MEMORY[
        _resolution_memory_key(group_id, target_hint)
    ] = (time.monotonic(), str(user_id))


def _lookup_remembered_target(
    group_id: str | None,
    target_hint: str,
) -> str | None:
    normalized_hint = _normalize_alias_key(target_hint)
    if not normalized_hint:
        return None
    cached = _NICKNAME_RESOLUTION_MEMORY.get(
        _resolution_memory_key(group_id, target_hint)
    )
    if not cached:
        return None
    ts, user_id = cached
    if (time.monotonic() - ts) > _NICKNAME_RESOLUTION_MEMORY_TTL:
        _NICKNAME_RESOLUTION_MEMORY.pop(
            _resolution_memory_key(group_id, target_hint), None
        )
        return None
    return user_id if str(user_id).isdigit() else None


def remember_target_resolution(
    group_id: str | None,
    target_hint: str,
    user_id: str,
) -> None:
    _remember_target_resolution(group_id, target_hint, user_id)


def _pick_fuzzy_target_profile(
    target_hint: str,
    profiles: list[dict[str, str | tuple[str, ...]]],
    active_scores: dict[str, float] | None = None,
    *,
    trigger_strength: str = "weak",
) -> tuple[
    dict[str, str | tuple[str, ...]] | None,
    list[dict[str, str | tuple[str, ...]]],
    float,
]:
    hint = _normalize_alias_key(target_hint)
    if len(hint) < 2:
        return None, [], 0.0

    strength = (trigger_strength or "weak").lower()
    if strength == "strong":
        ratio_threshold = 0.72
        match_threshold = 0.72
        ambiguous_top_threshold = 0.86
        ambiguous_gap_threshold = 0.08
    else:
        ratio_threshold = 0.80
        match_threshold = 0.86
        ambiguous_top_threshold = 0.92
        ambiguous_gap_threshold = 0.12

    ranked: list[tuple[float, dict[str, str | tuple[str, ...]]]] = []
    active_scores = active_scores or {}
    for profile in profiles:
        user_id = str(profile.get("user_id") or "").strip()
        alias_keys = profile.get("alias_keys") or ()
        if not isinstance(alias_keys, tuple):
            continue
        best_score = 0.0
        for alias in alias_keys:
            alias_text = str(alias or "").strip()
            if len(alias_text) < 2:
                continue
            if hint == alias_text:
                best_score = max(best_score, 1.0)
                continue
            if (hint in alias_text or alias_text in hint) and min(
                len(hint), len(alias_text)
            ) >= 4:
                overlap = min(len(hint), len(alias_text)) / max(len(hint), len(alias_text))
                best_score = max(best_score, 0.85 + overlap * 0.12)
                continue
            ratio = SequenceMatcher(None, hint, alias_text).ratio()
            if ratio >= ratio_threshold:
                best_score = max(best_score, ratio)
        if user_id and user_id in active_scores:
            best_score += active_scores[user_id]
        if best_score >= match_threshold:
            ranked.append((best_score, profile))

    if not ranked:
        return None, [], 0.0

    ranked.sort(
        key=lambda item: (
            item[0],
            len(str(item[1].get("display_name") or "")),
        ),
        reverse=True,
    )
    top_score, top_profile = ranked[0]
    if len(ranked) == 1:
        return top_profile, [], top_score

    second_score = ranked[1][0]
    if top_score < ambiguous_top_threshold or (
        top_score - second_score
    ) < ambiguous_gap_threshold:
        candidates: list[dict[str, str | tuple[str, ...]]] = []
        for _, profile in ranked[:5]:
            display_name = str(profile.get("display_name") or "").strip()
            user_id = str(profile.get("user_id") or "").strip()
            if display_name and user_id:
                candidates.append(profile)
        return None, candidates, top_score

    return top_profile, [], top_score


def _build_member_ambiguity_message(
    candidates: list[dict[str, str | tuple[str, ...]]],
) -> str:
    if not candidates:
        return "我不太确定你说的是谁。请直接@目标成员，我再继续。"
    display_options: list[str] = []
    for profile in candidates[:4]:
        display_name = str(profile.get("display_name") or "").strip()
        user_id = str(profile.get("user_id") or "").strip()
        if display_name and user_id:
            display_options.append(f"{display_name}(@{user_id})")
    if not display_options:
        return "我不太确定你说的是谁。请直接@目标成员，我再继续。"
    options = "、".join(display_options)
    return f"我匹配到好几个可能对象：{options}。请回复并@目标成员。"


def _is_self_only_action_message(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _SELF_ONLY_ACTION_KEYWORDS)


def _is_technical_request_like(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "").lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _TECHNICAL_REQUEST_HINT_WORDS)


def _contains_non_self_target_phrase(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    return _NON_SELF_TARGET_PATTERN.search(normalized) is not None


def _resolve_fuzzy_trigger_strength(
    *,
    original_message: str,
    route_message: str,
    command_heads: set[str] | None = None,
) -> str:
    normalized_original = normalize_message_text(original_message or "")
    normalized_route = normalize_message_text(route_message or "")
    if not normalized_original or not normalized_route:
        return ""
    if _extract_at_tokens(normalized_route):
        return ""
    if _is_technical_request_like(normalized_original) and not contains_any(
        normalized_original, _FOLLOWUP_MEME_HINTS
    ):
        return ""
    if _contains_non_self_target_phrase(normalized_original):
        return "strong"
    if _contains_third_person_reference(normalized_original):
        return "strong"
    if _needs_target_for_meme_request(normalized_original, normalized_route):
        return "strong"
    if command_heads:
        for head in sorted(command_heads, key=len, reverse=True):
            if head and _match_command_head_or_sticky(normalized_route, head):
                return "weak"
    return ""


def _match_command_head_or_sticky(message_text: str, command_head: str) -> bool:
    normalized_message = normalize_message_text(message_text or "")
    normalized_head = normalize_message_text(command_head or "")
    if not normalized_message or not normalized_head:
        return False
    if match_command_head(normalized_message, normalized_head):
        return True
    if not normalized_message.startswith(normalized_head):
        return False
    if len(normalized_message) <= len(normalized_head):
        return False
    sticky_tail = normalize_message_text(normalized_message[len(normalized_head) :])
    if not sticky_tail:
        return False
    # 粘连参数仅接受短昵称形态，避免把自然对话误判为目标名。
    if len(_normalize_alias_key(sticky_tail)) > 16:
        return False
    if _is_technical_request_like(sticky_tail):
        return False
    return True


def _needs_target_for_meme_request(message_text: str, route_message: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    if not contains_any(normalized, _FOLLOWUP_MEME_HINTS):
        return False
    if _contains_self_reference(normalized):
        return False
    if not (
        _contains_third_person_reference(normalized)
        or contains_any(normalized, _TARGET_REQUIRED_ACTION_HINTS)
    ):
        return False
    has_target = bool(_extract_at_tokens(route_message))
    has_image = bool(_extract_image_tokens(route_message))
    return not has_target and not has_image


async def _build_mention_profiles(
    group_id: str | None,
    message_text: str,
    bot_id: str | None = None,
) -> dict[str, dict[str, str]]:
    mention_profiles: dict[str, dict[str, str]] = {}
    mentioned_user_ids = _extract_mentioned_user_ids(message_text)
    if not mentioned_user_ids:
        return mention_profiles

    if bot_id and bot_id in mentioned_user_ids:
        bot_name = (BotConfig.self_nickname or "").strip()
        mention_profiles[bot_id] = {
            "display_name": bot_name,
            "nickname": bot_name,
            "user_name": bot_name,
            "uid": "",
            "platform": "qq",
            "alias_key": _normalize_alias_key(bot_name),
        }

    if not group_id:
        return mention_profiles

    try:
        from zhenxun.models.group_member_info import GroupInfoUser

        members = await GroupInfoUser.filter(
            group_id=group_id,
            user_id__in=list(mentioned_user_ids),
        ).all()
    except Exception as exc:
        logger.debug(f"解析@昵称失败: {exc}")
        return mention_profiles

    for member in members:
        user_id = str(member.user_id)
        nickname = (member.nickname or "").strip()
        user_name = (member.user_name or "").strip()
        display_name = (nickname or user_name).strip()
        uid = str(member.uid).strip() if member.uid is not None else ""
        platform = str(member.platform or "").strip() or "qq"
        alias_key = _normalize_alias_key(display_name or user_name)

        if not display_name and not uid:
            continue
        mention_profiles[user_id] = {
            "display_name": display_name,
            "nickname": nickname,
            "user_name": user_name,
            "uid": uid,
            "platform": platform,
            "alias_key": alias_key,
        }

    return mention_profiles


def _append_mention_context_xml(
    context_xml: str,
    mention_name_map: dict[str, str],
    mention_profiles: dict[str, dict[str, str]] | None = None,
) -> str:
    profiles = mention_profiles or {}
    if not mention_name_map and not profiles:
        return context_xml
    mention_lines: list[str] = []
    if mention_name_map:
        mention_lines.append("<mentioned_users>")
        for user_id, nickname in mention_name_map.items():
            mention_lines.append(f"[@{user_id}]={_xml_escape(nickname)}")
        mention_lines.append("</mentioned_users>")

    if profiles:
        mention_lines.append("<mentioned_user_profiles>")
        for user_id, profile in profiles.items():
            display_name = _xml_escape(profile.get("display_name", ""))
            nickname = _xml_escape(profile.get("nickname", ""))
            user_name = _xml_escape(profile.get("user_name", ""))
            uid = _xml_escape(profile.get("uid", ""))
            platform = _xml_escape(profile.get("platform", ""))
            alias_key = _xml_escape(profile.get("alias_key", ""))
            mention_lines.append(
                f"[@{user_id}] "
                f"display_name={display_name}; "
                f"nickname={nickname}; "
                f"user_name={user_name}; "
                f"uid={uid}; "
                f"platform={platform}; "
                f"alias_key={alias_key}"
            )
        mention_lines.append("</mentioned_user_profiles>")

    return f"{context_xml}\n" + "\n".join(mention_lines)


def _collect_target_capable_command_heads(knowledge_base) -> set[str]:
    heads: set[str] = set()
    plugins = getattr(knowledge_base, "plugins", None) or []
    for plugin in plugins:
        for meta in getattr(plugin, "command_meta", None) or []:
            allow_at = bool(getattr(meta, "allow_at", False))
            image_min = int(getattr(meta, "image_min", 0) or 0)
            target_requirement = normalize_message_text(
                str(getattr(meta, "target_requirement", "") or "")
            ).lower() or "none"
            allow_sticky_arg = bool(getattr(meta, "allow_sticky_arg", False))
            if (
                not allow_at
                and image_min <= 0
                and target_requirement == "none"
                and not allow_sticky_arg
            ):
                continue
            command_text = normalize_message_text(str(getattr(meta, "command", "") or ""))
            if command_text:
                heads.add(normalize_message_text(command_text.split(" ", 1)[0]))
            for alias in getattr(meta, "aliases", None) or []:
                alias_text = normalize_message_text(str(alias or ""))
                if alias_text:
                    heads.add(normalize_message_text(alias_text.split(" ", 1)[0]))
    return {head for head in heads if head}


async def _resolve_route_with_capability_recovery(
    message_text: str,
    knowledge_base,
    *,
    include_semantic: bool = True,
) -> RouteResolveResult | None:
    llm_route, allow_rule_fallback = await resolve_llm_route(
        message_text, knowledge_base
    )
    if llm_route is not None:
        return llm_route
    if allow_rule_fallback and not _should_allow_rule_fallback(
        message_text, knowledge_base
    ):
        logger.debug(
            "ChatInter 路由隔离：消息缺少执行信号，跳过 pre/semantic 规则回退"
        )
        allow_rule_fallback = False
    if not allow_rule_fallback:
        return None

    pre_route = resolve_pre_route(message_text, knowledge_base)
    if pre_route is not None:
        return pre_route
    if not include_semantic:
        return None
    return resolve_semantic_route(message_text, knowledge_base)


def _build_target_modules(
    decision: RouteResolveResult,
    selection_plugins,
) -> set[str]:
    target_modules = {decision.decision.plugin_module}
    for plugin in selection_plugins:
        if plugin.name == decision.decision.plugin_name:
            target_modules.add(plugin.module)
    return target_modules


def _normalize_head(command_text: str) -> str:
    normalized = normalize_message_text(command_text or "")
    if not normalized:
        return ""
    return normalize_message_text(normalized.split(" ", 1)[0])


def _iter_meta_aliases(meta) -> set[str]:
    aliases = getattr(meta, "aliases", None) or []
    values: set[str] = set()
    for alias in aliases:
        normalized = normalize_message_text(str(alias or ""))
        if normalized:
            values.add(normalized)
    return values


def _find_route_command_schema(route_result: RouteResolveResult, knowledge_plugins):
    decision = route_result.decision
    head = _normalize_head(decision.command)
    if not head:
        return None
    for plugin in knowledge_plugins:
        if (
            plugin.module != decision.plugin_module
            and plugin.name != decision.plugin_name
        ):
            continue
        for meta in plugin.command_meta:
            command_head = normalize_message_text(getattr(meta, "command", ""))
            if not command_head:
                continue
            if head == command_head or head in _iter_meta_aliases(meta):
                return meta
    return None


def _is_schema_self_only(schema) -> bool:
    actor_scope = normalize_message_text(str(getattr(schema, "actor_scope", "") or "")).lower()
    return actor_scope == "self_only"


def _should_block_self_only_action(
    *,
    schema,
    route_command: str,
    original_message: str,
    requester_user_id: str,
) -> bool:
    if not _is_schema_self_only(schema):
        return False

    command_targets = {
        extracted_id
        for token in _extract_at_tokens(route_command)
        if (extracted_id := _extract_user_id_from_at_token(token))
    }
    if any(target != requester_user_id for target in command_targets):
        return True

    normalized_message = normalize_message_text(original_message or "")
    if not normalized_message:
        return False

    if _contains_self_reference(normalized_message):
        return False
    if _contains_non_self_target_phrase(normalized_message):
        return True
    if _contains_third_person_reference(normalized_message):
        return True
    return False


def _extract_at_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _AT_ID_TOKEN_PATTERN.finditer(text or ""):
        user_id = (match.group(1) or match.group(2) or "").strip()
        if not user_id:
            continue
        token = f"[@{user_id}]"
        if token not in tokens:
            tokens.append(token)
    return tokens


def _extract_image_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in collect_placeholders(text or ""):
        if token.lower().startswith("[image"):
            if token not in tokens:
                tokens.append(token)
    return tokens


def _contains_reply_reference_hint(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    if _REPLY_TAG_PATTERN.search(normalized):
        return True
    return any(hint in normalized for hint in _REPLY_REF_HINTS)


def _contains_third_person_reference(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    return any(hint in normalized for hint in _THIRD_PERSON_HINTS)


def _extract_reply_sender_id(event: Event) -> str | None:
    reply = getattr(event, "reply", None)
    if reply is None:
        return None
    sender = getattr(reply, "sender", None)
    if sender is None and isinstance(reply, dict):
        sender = reply.get("sender")
    if sender is None:
        return None
    user_id = None
    if isinstance(sender, dict):
        user_id = sender.get("user_id")
    else:
        user_id = getattr(sender, "user_id", None)
    if user_id is None:
        return None
    text = str(user_id).strip()
    return text if text.isdigit() else None


def _build_route_message_with_explicit_context(
    *,
    message_text: str,
    user_id: str,
    reply_image_count: int,
    reply_sender_id: str | None,
) -> str:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return normalized

    should_enrich = (
        not is_usage_question(normalized)
        and (
            contains_any(normalized, ROUTE_ACTION_WORDS)
            or contains_any(normalized, _FOLLOWUP_MEME_HINTS)
            or "[image" in normalized
            or "[@" in normalized
            or _contains_reply_reference_hint(normalized)
        )
    )
    if not should_enrich:
        return normalized

    at_tokens = _extract_at_tokens(normalized)
    image_tokens = _extract_image_tokens(normalized)
    enriched = normalized

    if not at_tokens and _contains_self_reference(normalized):
        enriched = normalize_message_text(f"{enriched} [@{user_id}]")
        at_tokens.append(f"[@{user_id}]")

    if (
        not at_tokens
        and reply_sender_id
        and _contains_third_person_reference(normalized)
    ):
        enriched = normalize_message_text(f"{enriched} [@{reply_sender_id}]")
        at_tokens.append(f"[@{reply_sender_id}]")

    if (
        reply_image_count > 0
        and not image_tokens
        and _contains_reply_reference_hint(normalized)
    ):
        suffix = " ".join("[image]" for _ in range(reply_image_count))
        enriched = normalize_message_text(f"{enriched} {suffix}")

    return enriched


async def _enrich_route_message_with_fuzzy_target(
    *,
    group_id: str | None,
    request_user_id: str,
    original_message: str,
    route_message: str,
    mention_profiles: dict[str, dict[str, str]],
    command_heads: set[str] | None = None,
) -> tuple[str, dict[str, dict[str, str]], str | None]:
    if not group_id:
        return route_message, mention_profiles, None
    if _extract_at_tokens(route_message):
        return route_message, mention_profiles, None

    trigger_strength = _resolve_fuzzy_trigger_strength(
        original_message=original_message,
        route_message=route_message,
        command_heads=command_heads,
    )
    if not trigger_strength:
        return route_message, mention_profiles, None

    target_hint = _extract_fuzzy_target_hint(route_message, command_heads)
    if not target_hint:
        return route_message, mention_profiles, None

    profiles = await _get_group_member_profiles_for_fuzzy(group_id)
    if not profiles:
        return route_message, mention_profiles, None

    remembered_user_id = _lookup_remembered_target(group_id, target_hint)
    if remembered_user_id:
        remembered_profile = next(
            (
                profile
                for profile in profiles
                if str(profile.get("user_id") or "").strip() == remembered_user_id
            ),
            None,
        )
        if remembered_profile is not None:
            user_id = remembered_user_id
            enriched_message = normalize_message_text(f"{route_message} [@{user_id}]")
            mention_profiles = dict(mention_profiles)
            mention_profiles[user_id] = {
                "display_name": str(remembered_profile.get("display_name") or "").strip(),
                "nickname": str(remembered_profile.get("nickname") or "").strip(),
                "user_name": str(remembered_profile.get("user_name") or "").strip(),
                "uid": str(remembered_profile.get("uid") or "").strip(),
                "platform": str(remembered_profile.get("platform") or "qq").strip() or "qq",
                "alias_key": str(remembered_profile.get("alias_key") or "").strip(),
            }
            logger.debug(
                "ChatInter 昵称记忆命中: "
                f"hint='{target_hint}' -> {mention_profiles[user_id].get('display_name')}(@{user_id})"
            )
            return enriched_message, mention_profiles, None

    active_scores = await _get_group_recent_active_scores(group_id)
    matched, ambiguous_candidates, top_score = _pick_fuzzy_target_profile(
        target_hint,
        profiles,
        active_scores,
        trigger_strength=trigger_strength,
    )
    if ambiguous_candidates:
        if request_user_id:
            candidate_user_ids = tuple(
                str(profile.get("user_id") or "").strip()
                for profile in ambiguous_candidates
                if str(profile.get("user_id") or "").strip().isdigit()
            )
            set_pending_target_followup(
                request_user_id,
                group_id,
                original_message,
                target_hint=target_hint,
                candidate_user_ids=candidate_user_ids,
            )
        return (
            route_message,
            mention_profiles,
            _build_member_ambiguity_message(ambiguous_candidates),
        )
    if matched is None:
        if _needs_target_for_meme_request(original_message, route_message):
            if request_user_id:
                set_pending_target_followup(
                    request_user_id,
                    group_id,
                    original_message,
                    target_hint=target_hint,
                    candidate_user_ids=(),
                )
            return (
                route_message,
                mention_profiles,
                "要帮别人做的话，请直接@目标成员，或者发对方头像。你回复一下@我就继续处理。",
            )
        return route_message, mention_profiles, None

    user_id = str(matched.get("user_id") or "").strip()
    if not user_id.isdigit():
        return route_message, mention_profiles, None

    enriched_message = normalize_message_text(f"{route_message} [@{user_id}]")
    mention_profiles = dict(mention_profiles)
    mention_profiles[user_id] = {
        "display_name": str(matched.get("display_name") or "").strip(),
        "nickname": str(matched.get("nickname") or "").strip(),
        "user_name": str(matched.get("user_name") or "").strip(),
        "uid": str(matched.get("uid") or "").strip(),
        "platform": str(matched.get("platform") or "qq").strip() or "qq",
        "alias_key": str(matched.get("alias_key") or "").strip(),
    }
    logger.debug(
        "ChatInter 昵称模糊映射命中: "
        f"hint='{target_hint}' -> {mention_profiles[user_id].get('display_name')}(@{user_id})"
    )
    if top_score >= 0.90:
        _remember_target_resolution(group_id, target_hint, user_id)
    return enriched_message, mention_profiles, None


def _build_reply_image_segments_for_reroute(
    reply_images_data,
):
    if not reply_images_data:
        return []
    try:
        from nonebot.adapters.onebot.v11 import MessageSegment
    except Exception:
        return []

    segments = []
    seen_files: set[str] = set()
    for image in reply_images_data:
        file_id = str(getattr(image, "id", "") or "").strip()
        url = str(getattr(image, "url", "") or "").strip()
        path = getattr(image, "path", None)
        if not file_id and not url and not path:
            seg_type = getattr(image, "type", "")
            if seg_type == "image":
                seg_data = getattr(image, "data", {}) or {}
                file_id = str(seg_data.get("file", "") or "").strip()
                url = str(seg_data.get("url", "")).strip()
                path = seg_data.get("file")
        preferred_file_id = (
            file_id
            if file_id
            and not file_id.startswith(("http://", "https://", "base64://"))
            else ""
        )
        if preferred_file_id:
            key = f"id:{preferred_file_id}"
            if key in seen_files:
                continue
            try:
                if url:
                    segments.append(
                        MessageSegment(
                            "image",
                            {
                                "file": preferred_file_id,
                                "url": url,
                                "cache": "true",
                                "proxy": "true",
                            },
                        )
                    )
                else:
                    segments.append(MessageSegment.image(file=preferred_file_id))
                seen_files.add(key)
            except Exception:
                pass
            else:
                continue
        if url:
            key = f"url:{url}"
            if key in seen_files:
                continue
            try:
                segments.append(
                    MessageSegment(
                        "image",
                        {
                            "file": url,
                            "url": url,
                            "cache": "true",
                            "proxy": "true",
                        },
                    )
                )
                seen_files.add(key)
            except Exception:
                continue
            continue
        if path:
            path_text = str(path)
            key = f"path:{path_text}"
            if key in seen_files:
                continue
            try:
                segments.append(MessageSegment.image(file=path_text))
                seen_files.add(key)
            except Exception:
                continue
    return segments


def _extract_text_token_count(command_text: str) -> int:
    normalized = normalize_message_text(command_text)
    if not normalized:
        return 0
    parts = normalized.split(" ", 1)
    payload = parts[1] if len(parts) > 1 else ""
    payload = _PLACEHOLDER_SEGMENT_PATTERN.sub(" ", payload)
    payload = normalize_message_text(payload)
    if not payload:
        return 0
    return len([token for token in payload.split(" ") if token])


def _contains_self_reference(message_text: str) -> bool:
    normalized = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in ("我", "自己", "本人", "我的", "我自己", "自己的")
    )


def _build_followup_message(
    *,
    image_missing: int,
    text_missing: int,
    allow_at: bool,
) -> str:
    hints: list[str] = []
    if image_missing > 0:
        if allow_at:
            hints.append(f"还需要 {image_missing} 张图片（可发图或@目标）")
        else:
            hints.append(f"还需要 {image_missing} 张图片")
    if text_missing > 0:
        hints.append(f"还需要 {text_missing} 段文字")
    joined = "，".join(hints) if hints else "参数不足"
    return f"这个命令{joined}，请补充后我继续执行。"


def _build_target_required_message(schema) -> str:
    sources = {
        normalize_message_text(str(item or "")).lower()
        for item in (getattr(schema, "target_sources", None) or [])
    }
    hints: list[str] = []
    if "at" in sources:
        hints.append("直接@目标成员")
    if "reply" in sources:
        hints.append("回复对方消息并@")
    if "nickname" in sources:
        hints.append("补充完整昵称")
    if not hints:
        hints = ["补充目标成员（@或昵称）"]
    return "这个命令需要目标对象，请" + "、".join(hints) + "后我继续执行。"


def _is_image_related_route(route_result: RouteResolveResult) -> bool:
    plugin_name = str(route_result.decision.plugin_name or "").lower()
    module_name = str(route_result.decision.plugin_module or "").lower()
    return (
        "meme" in module_name
        or "表情" in plugin_name
        or "image" in module_name
        or "图片" in plugin_name
    )


def _append_unique_tokens(command: str, tokens: list[str]) -> str:
    normalized_command = normalize_message_text(command or "")
    if not normalized_command:
        return normalized_command
    merged: list[str] = []
    existing_placeholders = set(collect_placeholders(normalized_command))
    for token in tokens:
        text = normalize_message_text(token)
        if not text:
            continue
        if text in existing_placeholders:
            continue
        existing_placeholders.add(text)
        merged.append(text)
    if not merged:
        return normalized_command
    return normalize_message_text(f"{normalized_command} {' '.join(merged)}")


def _clamp_command_text_tokens(command: str, text_max_raw) -> str:
    normalized_command = normalize_message_text(command or "")
    if not normalized_command:
        return normalized_command
    if text_max_raw is None:
        return normalized_command
    try:
        text_max = int(text_max_raw)
    except Exception:
        return normalized_command
    text_max = max(text_max, 0)

    parts = normalized_command.split(" ", 1)
    command_head = parts[0]
    if len(parts) < 2:
        return command_head

    kept_tokens: list[str] = []
    text_count = 0
    for raw_token in parts[1].split(" "):
        token = normalize_message_text(raw_token)
        if not token:
            continue
        if _PLACEHOLDER_SEGMENT_PATTERN.fullmatch(token):
            kept_tokens.append(token)
            continue
        if text_count < text_max:
            kept_tokens.append(token)
            text_count += 1

    if kept_tokens:
        return normalize_message_text(f"{command_head} {' '.join(kept_tokens)}")
    return command_head


def _prepare_route_execution_plan(
    *,
    route_result: RouteResolveResult,
    knowledge_plugins,
    current_message: str,
    user_id: str,
) -> RouteExecutionPlan:
    command = normalize_message_text(route_result.decision.command or "")
    if not command:
        return RouteExecutionPlan(command="")

    schema = _find_route_command_schema(route_result, knowledge_plugins)
    if schema is None:
        if not _is_image_related_route(route_result):
            return RouteExecutionPlan(command=command)
        merged_at = _extract_at_tokens(current_message)
        if not merged_at and _contains_self_reference(current_message):
            merged_at.append(f"[@{user_id}]")
        merged_images = _extract_image_tokens(current_message)
        merged_tokens = [*merged_at, *merged_images]
        if merged_tokens:
            command = _append_unique_tokens(command, merged_tokens)
        return RouteExecutionPlan(command=command)

    command = _clamp_command_text_tokens(command, getattr(schema, "text_max", None))

    image_min = max(int(getattr(schema, "image_min", 0) or 0), 0)
    text_min = max(int(getattr(schema, "text_min", 0) or 0), 0)
    target_requirement = (
        normalize_message_text(str(getattr(schema, "target_requirement", "") or "")).lower()
        or "none"
    )
    allow_at_raw = getattr(schema, "allow_at", None)
    allow_at = allow_at_raw is True or allow_at_raw is None

    command_at = _extract_at_tokens(command)
    command_images = _extract_image_tokens(command)
    message_at = _extract_at_tokens(current_message)
    message_images = _extract_image_tokens(current_message)

    merged_at = command_at[:]
    for token in message_at:
        if token not in merged_at:
            merged_at.append(token)
    merged_images = command_images[:]
    for token in message_images:
        if token not in merged_images:
            merged_images.append(token)

    if image_min > 0 and allow_at and not merged_at and _contains_self_reference(
        current_message
    ):
        self_at = f"[@{user_id}]"
        merged_at.append(self_at)

    if target_requirement == "required" and not (merged_at or merged_images):
        if allow_at and _contains_self_reference(current_message):
            merged_at.append(f"[@{user_id}]")
        else:
            return RouteExecutionPlan(
                command=command,
                need_followup=True,
                followup_message=_build_target_required_message(schema),
                wait_for_image=False,
            )

    if allow_at:
        image_count = len(merged_images) + len(merged_at)
    else:
        image_count = len(merged_images)
    text_count = _extract_text_token_count(command)

    image_missing = max(image_min - image_count, 0)
    text_missing = max(text_min - text_count, 0)
    if image_missing > 0 or text_missing > 0:
        return RouteExecutionPlan(
            command=command,
            need_followup=True,
            followup_message=_build_followup_message(
                image_missing=image_missing,
                text_missing=text_missing,
                allow_at=allow_at,
            ),
            wait_for_image=image_missing > 0,
        )

    if allow_at and merged_at:
        command = _append_unique_tokens(command, merged_at)

    return RouteExecutionPlan(command=command)


async def _execute_route_decision(
    *,
    bot: Bot,
    event: Event,
    trace: StageTrace,
    route_result: RouteResolveResult,
    knowledge_plugins,
    user_id: str,
    group_id: str | None,
    nickname: str,
    user_message,
    bot_id: str | None,
    current_message: str,
    extra_image_segments=None,
) -> bool:
    decision = route_result.decision
    envelope = TurnChannelEnvelope()
    envelope.add(
        ChannelName.ANALYSIS,
        (
            f"route stage={route_result.stage} plugin={decision.plugin_name} "
            f"module={decision.plugin_module} source={decision.source}"
        ),
    )
    target_modules = _build_target_modules(route_result, knowledge_plugins)
    execution_plan = _prepare_route_execution_plan(
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        current_message=current_message,
        user_id=str(user_id),
    )
    if execution_plan.need_followup:
        followup_text = execution_plan.followup_message or "这个命令参数不足，请补充后再试。"
        envelope.add(ChannelName.COMMENTARY, "route requires additional parameters")
        envelope.add(ChannelName.FINAL, followup_text)
        _log_turn_channels(envelope)
        logger.info(
            "技能路由需要补充参数，进入待补全："
            f"stage={route_result.stage}, "
            f"plugin={decision.plugin_name}, "
            f"command={decision.command}, "
            f"followup={followup_text}"
        )
        await _persist_final_only_dialog(
            envelope=envelope,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            user_message=user_message,
            bot_id=bot_id,
        )
        trace.stage("persist")
        if execution_plan.wait_for_image:
            set_pending_image_followup(user_id, group_id, current_message)
        await MessageUtils.build_message(envelope.final).send()
        trace.stage("notify")
        trace.finish()
        return True

    route_command = execution_plan.command or decision.command
    logger.info(
        "触发技能路由："
        f"stage={route_result.stage}, "
        f"plugin={decision.plugin_name}, "
        f"module={decision.plugin_module}, "
        f"command={route_command}, "
        f"source={decision.source}"
    )

    response_text = _build_route_notify_text(
        plugin_name=decision.plugin_name,
        plugin_module=decision.plugin_module,
        route_command=route_command,
    )
    envelope.add(ChannelName.COMMENTARY, f"reroute command: {route_command}")
    envelope.add(ChannelName.FINAL, response_text)
    _log_turn_channels(envelope)
    await _persist_final_only_dialog(
        envelope=envelope,
        user_id=user_id,
        group_id=group_id,
        nickname=nickname,
        user_message=user_message,
        bot_id=bot_id,
    )
    trace.stage("persist")
    await MessageUtils.build_message(envelope.final).send()
    trace.stage("notify")

    success = await reroute_to_plugin(
        bot,
        event,
        route_command,
        target_modules=target_modules,
        extra_image_segments=extra_image_segments,
    )
    if success:
        trace.stage("route")
        trace.finish()
    return success


async def handle_fallback(
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message=None,
    route_modules: set[str] | None = None,
    cached_plain_text: str | None = None,
    current_message_override: str | None = None,
    from_pending_followup: bool = False,
) -> bool:
    """消息处理器

    当消息未被其他插件处理时，使用 AI 分析用户意图并响应。

    参数:
        bot: Bot 实例
        event: 事件对象
        session: Uninfo 会话信息
        raw_message: 原始消息文本
        message: 原始消息对象（可选）

    返回:
        bool: 是否处理成功
    """
    if not get_config_value("ENABLE_FALLBACK", True):
        logger.debug("ChatInter 功能已禁用")
        return

    if _is_already_handled(event):
        logger.debug("消息已被处理，跳过")
        return

    if route_modules:
        logger.debug("命中已有命令路由，跳过 ChatInter fallback")
        return

    _mark_as_handled(event)
    trace = StageTrace(
        "chatinter",
        tags={
            "user": str(getattr(session.user, "id", "")),
            "group": str(getattr(session.group, "id", ""))
            if session.group
            else "private",
            "message_id": str(getattr(event, "message_id", "")),
        },
    )

    user_id = session.user.id
    group_id = session.group.id if session.group else None
    nickname = _get_nickname(session)
    bot_id = str(bot.self_id) if hasattr(bot, "self_id") else None
    model_name = get_config_value("INTENT_MODEL", None)
    session_key = str(group_id or user_id)
    is_superuser = _resolve_superuser(bot, str(user_id))
    lifecycle = get_lifecycle_manager()
    current_message = raw_message
    chat_system_prompt = ""
    enriched_context_xml = ""
    mention_name_map: dict[str, str] = {}
    mention_profiles: dict[str, dict[str, str]] = {}
    try:
        if not from_pending_followup:
            clear_pending_image_followup(user_id, group_id)
        event_message = event.get_message()
    except Exception:
        event_message = None

    try:
        await _apply_runtime_plugin_overrides(
            event=event,
            session_key=session_key,
            group_id=str(group_id) if group_id else None,
        )
        knowledge_base = await get_user_plugin_knowledge()
        trace.stage("knowledge")

        # 尝试使用 UniMessage 传递
        uni_msg = None
        if message:
            try:
                uni_msg = UniMessage.of(message)
            except Exception:
                pass

        (
            chat_system_prompt,
            context_xml,
            reply_images_data,
        ) = await _chat_memory.build_full_context(
            user_id,
            group_id,
            nickname,
            uni_msg or raw_message,
            bot,
            bot_id,
            event,
        )
        trace.stage("context")

        # 优先使用 UniMessage 处理
        if current_message_override is not None:
            current_message = current_message_override.strip()
        elif event_message is not None:
            current_message = uni_to_text_with_tags(event_message)
        elif uni_msg:
            current_msg = remove_reply_segment(uni_msg)
            current_message = uni_to_text_with_tags(current_msg)
        elif cached_plain_text:
            current_message = cached_plain_text.strip()
        else:
            # 无法解析为 UniMessage 时，使用原始消息文本
            current_message = raw_message.strip()

        mention_profiles = await _build_mention_profiles(
            str(group_id) if group_id else None,
            current_message,
            bot_id=bot_id,
        )
        command_heads = _collect_target_capable_command_heads(knowledge_base)
        reply_sender_id = _extract_reply_sender_id(event)
        reply_image_count = len(reply_images_data or [])
        if reply_image_count > 0:
            logger.debug(f"Reply 中解析到图片 {reply_image_count} 张，将用于路由重放")
        reply_image_segments_for_reroute = _build_reply_image_segments_for_reroute(
            reply_images_data
        )
        route_message_base = _build_route_message_with_explicit_context(
            message_text=current_message,
            user_id=str(user_id),
            reply_image_count=reply_image_count,
            reply_sender_id=reply_sender_id,
        )
        route_message, mention_profiles, fuzzy_prompt = await _enrich_route_message_with_fuzzy_target(
            group_id=str(group_id) if group_id else None,
            request_user_id=str(user_id),
            original_message=current_message,
            route_message=route_message_base,
            mention_profiles=mention_profiles,
            command_heads=command_heads,
        )
        mention_name_map = _build_mention_name_map(mention_profiles)
        if mention_name_map or mention_profiles:
            context_xml = _append_mention_context_xml(
                context_xml,
                mention_name_map,
                mention_profiles,
            )
            logger.debug(
                "解析到@信息映射: "
                + ", ".join(
                    (
                        f"{mapped_user_id}->{profile.get('display_name')}"
                        + (
                            f"(uid:{profile.get('uid')})"
                            if profile.get("uid")
                            else ""
                        )
                    )
                    for mapped_user_id, profile in mention_profiles.items()
                )
            )

        if fuzzy_prompt:
            envelope = TurnChannelEnvelope()
            envelope.add(ChannelName.ANALYSIS, "fuzzy target requires clarification")
            envelope.add(ChannelName.FINAL, fuzzy_prompt)
            _log_turn_channels(envelope)
            await _persist_final_only_dialog(
                envelope=envelope,
                user_id=user_id,
                group_id=group_id,
                nickname=nickname,
                user_message=uni_msg or current_message,
                bot_id=bot_id,
            )
            trace.stage("persist")
            await MessageUtils.build_message(envelope.final).send()
            trace.stage("send")
            trace.finish()
            return

        if _needs_target_for_meme_request(current_message, route_message):
            envelope = TurnChannelEnvelope()
            envelope.add(ChannelName.ANALYSIS, "target required for meme request")
            envelope.add(
                ChannelName.FINAL,
                "要帮别人制作的话，请补充完整昵称、直接@对方，或者发对方头像。",
            )
            _log_turn_channels(envelope)
            await _persist_final_only_dialog(
                envelope=envelope,
                user_id=user_id,
                group_id=group_id,
                nickname=nickname,
                user_message=uni_msg or current_message,
                bot_id=bot_id,
            )
            trace.stage("persist")
            await MessageUtils.build_message(envelope.final).send()
            trace.stage("send")
            trace.finish()
            return

        if route_message != current_message:
            logger.debug(
                "ChatInter 路由上下文增强："
                f"before='{current_message}' -> after='{route_message}'"
            )
        selection_context = PluginSelectionContext(
            query=route_message,
            session_id=session_key,
            user_id=str(user_id),
            group_id=str(group_id) if group_id else None,
            is_superuser=is_superuser,
        )
        knowledge_base = PluginRegistry.filter_knowledge_base(
            knowledge_base,
            selection_context=selection_context,
        )

        global _last_knowledge_refresh_ts
        if should_force_knowledge_refresh(route_message, knowledge_base):
            now = time.monotonic()
            if now - _last_knowledge_refresh_ts >= _KNOWLEDGE_REFRESH_COOLDOWN:
                _last_knowledge_refresh_ts = now
                refreshed_knowledge = await get_user_plugin_knowledge(
                    force_refresh=True
                )
                if len(refreshed_knowledge.plugins) > len(knowledge_base.plugins):
                    knowledge_base = PluginRegistry.filter_knowledge_base(
                        refreshed_knowledge,
                        selection_context=selection_context,
                    )
                    logger.info(
                        "检测到插件知识可能不完整，已执行一次自愈刷新："
                        f"{len(knowledge_base.plugins)} 个插件"
                    )

        route_result = await _resolve_route_with_capability_recovery(
            route_message,
            knowledge_base,
            include_semantic=True,
        )
        if route_result is not None:
            route_schema = _find_route_command_schema(route_result, knowledge_base.plugins)
            block_self_only = False
            if route_schema is not None:
                block_self_only = _should_block_self_only_action(
                    schema=route_schema,
                    route_command=route_result.decision.command,
                    original_message=current_message,
                    requester_user_id=str(user_id),
                )
            elif _is_self_only_action_message(route_result.decision.command):
                route_targets = {
                    extracted_id
                    for token in _extract_at_tokens(route_result.decision.command)
                    if (extracted_id := _extract_user_id_from_at_token(token))
                }
                block_self_only = any(target != str(user_id) for target in route_targets)
                if not block_self_only and not _contains_self_reference(current_message):
                    block_self_only = (
                        _contains_non_self_target_phrase(current_message)
                        or _contains_third_person_reference(current_message)
                    )
            if block_self_only:
                envelope = TurnChannelEnvelope()
                envelope.add(ChannelName.ANALYSIS, "blocked self-only action for others")
                envelope.add(
                    ChannelName.FINAL,
                    "这类功能只能本人触发，不能代他人执行。请让对方自己发送命令。",
                )
                _log_turn_channels(envelope)
                await _persist_final_only_dialog(
                    envelope=envelope,
                    user_id=user_id,
                    group_id=group_id,
                    nickname=nickname,
                    user_message=uni_msg or current_message,
                    bot_id=bot_id,
                )
                trace.stage("persist")
                await MessageUtils.build_message(envelope.final).send()
                trace.stage("send")
                trace.finish()
                return
        if route_result is not None and await _execute_route_decision(
            bot=bot,
            event=event,
            trace=trace,
            route_result=route_result,
            knowledge_plugins=knowledge_base.plugins,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            user_message=uni_msg or raw_message,
            bot_id=bot_id,
            current_message=route_message,
            extra_image_segments=reply_image_segments_for_reroute,
        ):
            return
        if route_result is not None:
            logger.warning(
                "技能路由重路由失败，降级为聊天处理："
                f"stage={route_result.stage}, "
                f"plugin={route_result.decision.plugin_name}, "
                f"command={route_result.decision.command}"
            )

        # 提取图片（多模态处理）
        source_for_media = event_message or uni_msg or message or raw_message
        image_parts = await extract_images_from_message(
            bot, event, source_for_media
        )
        if image_parts:
            logger.debug(f"当前消息中包含 {len(image_parts)} 张图片")

        # 提取回复链中的图片（直接使用 Image Segment 处理）
        if reply_images_data:
            from .utils.multimodal import _process_image_segment

            for img_seg in reply_images_data:
                image_part = await _process_image_segment(img_seg)
                if image_part:
                    image_parts.append(image_part)
            if reply_images_data:
                logger.debug(f"回复链中包含 {len(reply_images_data)} 张图片")
        trace.stage("media")
        enriched_context_xml = context_xml
        before_chat_payload = LifecyclePayload(
            user_id=user_id,
            group_id=group_id,
            message_text=current_message,
            system_prompt=chat_system_prompt,
            context_xml=enriched_context_xml,
            model_name=model_name,
            metadata={"phase": "chat_fallback"},
        )
        await lifecycle.dispatch("before_agent", before_chat_payload)
        reply = await handle_chat_message(
            message=before_chat_payload.message_text,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            mention_name_map=mention_name_map,
        )
        trace.stage("chat_fallback")
        reply_text = (
            str(reply)
            if reply is not None and str(reply).strip()
            else "我暂时没想好怎么回答你。"
        )
        before_chat_payload.response_text = reply_text
        await lifecycle.dispatch("after_agent", before_chat_payload)
        reply_text = (
            before_chat_payload.response_text
            if before_chat_payload.response_text is not None
            else reply_text
        )
        reply_text = normalize_ai_reply_text(reply_text or "")
        reply_text = replace_mention_ids_with_names(
            reply_text, mention_name_map
        )
        envelope = TurnChannelEnvelope()
        envelope.add(ChannelName.ANALYSIS, "chat fallback")
        envelope.add(ChannelName.FINAL, reply_text)
        _log_turn_channels(envelope)
        await _persist_final_only_dialog(
            envelope=envelope,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            user_message=uni_msg or current_message,
            bot_id=bot_id,
        )
        trace.stage("persist")
        if _should_wait_for_image_followup(current_message, envelope.final):
            set_pending_image_followup(user_id, group_id, current_message)
            logger.debug(
                "ChatInter 已登记待补图会话: "
                f"user={user_id}, group={group_id or 'private'}"
            )
        await MessageUtils.build_message(envelope.final).send()
        trace.stage("send")
        trace.finish()
        return

    except asyncio.CancelledError:
        group_name = group_id or "private"
        logger.debug(
            f"ChatInter 当前会话任务被中断: user={user_id}, group={group_name}"
        )
        return
    except Exception as e:
        await lifecycle.dispatch(
            "on_error",
            LifecyclePayload(
                user_id=user_id,
                group_id=group_id,
                message_text=current_message,
                system_prompt=chat_system_prompt,
                context_xml=enriched_context_xml,
                model_name=model_name,
                metadata={"error": str(e)},
            ),
        )
        logger.error(f"ChatInter 处理失败：{e}")
        await MessageUtils.build_failure_message().send()
        trace.stage("error")
        trace.finish()
        return


__all__ = [
    "build_pending_followup_message",
    "build_pending_target_followup_message",
    "claim_pending_image_followup",
    "clear_pending_target_followup",
    "get_pending_target_followup",
    "handle_fallback",
    "has_pending_image_followup",
    "remember_target_resolution",
    "resolve_pending_target_followup_user_id",
    "_resolve_route_with_capability_recovery",
]
