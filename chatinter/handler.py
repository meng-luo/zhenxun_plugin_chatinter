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

from .addressee_resolver import AddresseeResult, resolve_addressee
from .agent_gate import decide_agent_gate
from .agent_runner import run_chatinter_agent
from .chat_dialogue_planner import ChatDialoguePlan, plan_chat_dialogue
from .chat_feedback import ChatFeedbackStore
from .chat_handler import (
    handle_chat_message,
    normalize_ai_reply_text,
    replace_mention_ids_with_names,
    reroute_to_plugin,
)
from .chat_quality_guard import refine_chat_reply
from .command_planner import CommandPlanDecision, plan_command
from .config import get_config_value, get_mcp_endpoints, get_model_name
from .context_packer import DialogueContextPack
from .event_context import ChatInterEventContext, build_event_context
from .execution_observer import (
    EXECUTION_REASON_CANCELLED,
    EXECUTION_REASON_CHAT_COMPLETED,
    EXECUTION_REASON_CHAT_REWRITTEN,
    EXECUTION_REASON_ERROR,
    EXECUTION_REASON_REROUTE_FAILED,
    EXECUTION_REASON_ROUTE_SUCCESS,
    EXECUTION_REASON_USAGE_REPLIED,
    ExecutionObservation,
    record_execution_observation,
    start_execution_observation,
)
from .feedback_keys import (
    FEEDBACK_REASON_DIRECT_TARGET_REQUIRED as _FEEDBACK_REASON_DIRECT_TARGET_REQUIRED,
)
from .feedback_keys import (
    FEEDBACK_REASON_FUZZY_CLARIFY as _FEEDBACK_REASON_FUZZY_CLARIFY,
)
from .feedback_keys import (
    FEEDBACK_REASON_MISSING_PARAMS as _FEEDBACK_REASON_MISSING_PARAMS,
)
from .feedback_keys import (
    FEEDBACK_REASON_REROUTE_FAILED as _FEEDBACK_REASON_REROUTE_FAILED,
)
from .feedback_keys import (
    FEEDBACK_REASON_ROUTE_SUCCESS as _FEEDBACK_REASON_ROUTE_SUCCESS,
)
from .feedback_keys import (
    FEEDBACK_REASON_SELF_ONLY_BLOCKED as _FEEDBACK_REASON_SELF_ONLY_BLOCKED,
)
from .feedback_keys import (
    FEEDBACK_REASON_TARGET_REQUIRED as _FEEDBACK_REASON_TARGET_REQUIRED,
)
from .intent_classifier import IntentClassification, classify_message_intent
from .intervention_router import InterventionDecision, decide_intervention
from .knowledge_rag import PluginRAGService
from .memory import _chat_memory
from .middleware import TurnMiddlewareState, get_middleware_manager
from .models.pydantic_models import PluginKnowledgeBase
from .person_registry import PersonProfile, get_person_profile, upsert_seen_person
from .plugin_adapters import (
    AdapterTargetPolicy,
    get_adapter_notification_policy,
    get_adapter_target_policy,
)
from .plugin_registry import (
    PluginRegistry,
    PluginSelectionContext,
    get_user_plugin_knowledge,
)
from .route_engine import (
    RouteAttemptReport,
    RouteResolveResult,
    resolve_llm_router,
)
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_placeholders,
    contains_any,
    has_negative_route_intent,
    is_usage_question,
    match_command_head_canonical,
    normalize_action_phrases,
    normalize_message_text,
    parse_command_with_head,
    should_force_knowledge_refresh,
)
from .schema_policy import (
    resolve_command_target_policy,
    schema_is_self_only,
)
from .skill_registry import (
    SkillRouteDecision,
    _extract_explicit_value,
    _extract_schema_argument_tokens,
)
from .thread_resolver import ThreadContext, resolve_thread_context
from .trace import StageTrace
from .turn_metrics import (
    build_turn_metrics_snapshot,
    emit_turn_metrics,
    record_route_observation,
)
from .turn_runtime import TurnBudgetController
from .utils.multimodal import extract_images_from_message
from .utils.unimsg_utils import remove_reply_segment, uni_to_text_with_tags

_HANDLED_MESSAGE_IDS: set[str] = set()
_MAX_HANDLED_CACHE = 1000
_KNOWLEDGE_REFRESH_COOLDOWN = 30.0
_last_knowledge_refresh_ts = 0.0
_ENABLE_PLUGINS_ATTR = "_chatinter_enable_plugins"
_DISABLE_PLUGINS_ATTR = "_chatinter_disable_plugins"
_AT_ID_TOKEN_PATTERN = re.compile(
    r"\[@([^\]\s]+)\]|(?<![0-9A-Za-z_])@(\d{5,20})(?=(?:\s|$|[的，,。.!！？?]))"
)
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
_INTENT_REFRESH_PUNCTUATION = ("。", "！", "？", "；", ";")
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
_ROUTE_FEEDBACK_REWARD = {
    _FEEDBACK_REASON_ROUTE_SUCCESS: 1.0,
    _FEEDBACK_REASON_MISSING_PARAMS: -0.35,
    _FEEDBACK_REASON_TARGET_REQUIRED: -0.40,
    _FEEDBACK_REASON_SELF_ONLY_BLOCKED: -0.55,
    _FEEDBACK_REASON_REROUTE_FAILED: -0.45,
    _FEEDBACK_REASON_FUZZY_CLARIFY: -0.20,
    _FEEDBACK_REASON_DIRECT_TARGET_REQUIRED: -0.30,
}
_GROUP_MEMBER_PROFILE_CACHE_TTL = 90.0
_GROUP_MEMBER_PROFILE_CACHE_MAX = 256
_GROUP_MEMBER_PROFILE_CACHE: dict[
    str, tuple[float, list[dict[str, str | tuple[str, ...]]]]
] = {}
_GROUP_ACTIVE_RANK_CACHE_TTL = 30.0
_GROUP_ACTIVE_RANK_CACHE_MAX = 256
_GROUP_ACTIVE_RANK_CACHE: dict[str, tuple[float, dict[str, float]]] = {}
_NICKNAME_RESOLUTION_MEMORY_TTL = 12 * 3600.0
_NICKNAME_RESOLUTION_MEMORY_MAX = 2048
_NICKNAME_RESOLUTION_MEMORY: dict[str, tuple[float, str]] = {}
_ROUTE_CONTINUATION_FRAME_TTL = 10 * 60.0
_ROUTE_CONTINUATION_FRAME_CACHE_MAX = 512


@dataclass(frozen=True)
class RouteExecutionPlan:
    command: str
    need_followup: bool = False
    followup_message: str | None = None
    feedback_reason: str | None = None
    image_missing: int = 0
    text_missing: int = 0
    allow_at: bool | None = None


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


async def _handle_chat_dialogue_special_case(
    *,
    plan: ChatDialoguePlan,
    trace: StageTrace,
    user_id: str,
    group_id: str | None,
    nickname: str,
    user_message,
    bot_id: str | None,
    session_key: str,
    current_message: str,
    route_report: RouteAttemptReport | None,
    budget_controller: TurnBudgetController | None,
    finalize_callback,
) -> bool:
    if plan.kind != "recap":
        return False
    recap = await _chat_memory.build_recent_conversation_recap(user_id, group_id)
    envelope = TurnChannelEnvelope()
    trace.update_tags(path="chat", outcome="chat_recap")
    envelope.add(ChannelName.ANALYSIS, "chat dialogue special: recap")
    envelope.add(ChannelName.FINAL, recap)
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
    trace.stage("send")
    ChatFeedbackStore.record(
        session_id=session_key,
        kind="chat_completed",
        message_text=current_message,
        reply_text=envelope.final,
        weight=0.2,
    )
    if finalize_callback is not None:
        await finalize_callback(
            response_text=envelope.final,
            phase="post_gate:chat_recap",
        )
    _tag_execution_observation(
        trace,
        record_execution_observation(
            action="chat",
            success=True,
            reason=EXECUTION_REASON_CHAT_COMPLETED,
            session_id=session_key,
            route_stage="chat_recap",
            message_preview=current_message,
        ),
    )
    _finish_trace(
        trace=trace,
        user_id=str(user_id),
        group_id=group_id,
        message_preview=current_message,
        route_report=route_report,
        budget_controller=budget_controller,
    )
    return True


def _build_route_notify_text(
    plugin_name: str,
    plugin_module: str,
    route_command: str,
    command_id: str = "",
) -> str:
    def _render(template_pool: tuple[str, ...], *, target: str, seed: str) -> str:
        if not template_pool:
            return f"好哒，这就帮你{target}。"
        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(template_pool)
        return template_pool[index].format(target=target)

    normalized_command = normalize_message_text(route_command)
    command_head = normalized_command.split(" ", 1)[0] if normalized_command else ""
    target = normalize_message_text(plugin_name) or command_head
    seed_base = f"{plugin_module}|{plugin_name}|{normalized_command}"
    if not target:
        return _render(_CUTE_NOTIFY_TEMPLATES, target="处理一下", seed=seed_base)

    policy = get_adapter_notification_policy(
        plugin_module=plugin_module,
        command_id=command_id,
    )
    if policy is not None and policy.default_templates:
        if target in policy.helper_heads and policy.helper_templates:
            return _render(
                policy.helper_templates,
                target=target,
                seed=f"{seed_base}|helper",
            )
        notify_target = target
        if policy.target_suffix and policy.target_suffix not in notify_target:
            notify_target = f"{notify_target}{policy.target_suffix}"
        return _render(
            policy.default_templates,
            target=notify_target,
            seed=f"{seed_base}|adapter",
        )

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
    display_name = str(getattr(session.user, "display_name", "") or "").strip()
    if display_name:
        return display_name
    name = str(getattr(session.user, "name", "") or "").strip()
    if name:
        return name
    return "用户"


def _resolve_superuser(bot: Bot, user_id: str) -> bool:
    superusers = getattr(getattr(bot, "config", None), "superusers", set())
    return str(user_id) in {str(item) for item in superusers}


def _event_type_name(event: Event) -> str:
    try:
        return str(event.get_type() or "")
    except Exception:
        return str(getattr(event, "post_type", "") or "")


def _event_adapter_name(bot: Bot) -> str:
    return str(getattr(bot, "type", "") or getattr(bot, "adapter", "") or "")


def _event_is_private(event: Event) -> bool:
    message_type = str(getattr(event, "message_type", "") or "").lower()
    if message_type == "private":
        return True
    try:
        return str(event.get_type() or "").lower() == "private"
    except Exception:
        return False


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
        nickname = str(
            profile.get("display_name") or profile.get("nickname") or ""
        ).strip()
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
    if not user_id or user_id in {"所有人", "all"}:
        return None
    return user_id


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
            if not normalized_head:
                continue
            parsed = parse_command_with_head(
                normalized,
                normalized_head,
                allow_sticky=True,
            )
            if parsed is None:
                continue
            tail = normalize_message_text(parsed.payload_text or parsed.prefix_text)
            tail = re.sub(r"^(?:给|帮|替|让|叫|喊|请|把|将)+", "", tail).strip()
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
        if not user_id:
            continue
        nickname = str(getattr(member, "nickname", "") or "").strip()
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
    if len(_GROUP_MEMBER_PROFILE_CACHE) > _GROUP_MEMBER_PROFILE_CACHE_MAX:
        for _evict_key in sorted(
            _GROUP_MEMBER_PROFILE_CACHE,
            key=lambda k: _GROUP_MEMBER_PROFILE_CACHE[k][0],
        )[: len(_GROUP_MEMBER_PROFILE_CACHE) - _GROUP_MEMBER_PROFILE_CACHE_MAX]:
            _GROUP_MEMBER_PROFILE_CACHE.pop(_evict_key, None)
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
        if not user_id or user_id in score_map:
            continue
        rank += 1
        score_map[user_id] = max(0.0, 0.08 - min(rank - 1, 10) * 0.006)
        if rank >= 20:
            break

    _GROUP_ACTIVE_RANK_CACHE[cache_key] = (now, score_map)
    if len(_GROUP_ACTIVE_RANK_CACHE) > _GROUP_ACTIVE_RANK_CACHE_MAX:
        for _evict_key in sorted(
            _GROUP_ACTIVE_RANK_CACHE,
            key=lambda k: _GROUP_ACTIVE_RANK_CACHE[k][0],
        )[: len(_GROUP_ACTIVE_RANK_CACHE) - _GROUP_ACTIVE_RANK_CACHE_MAX]:
            _GROUP_ACTIVE_RANK_CACHE.pop(_evict_key, None)
    return score_map


def _resolution_memory_key(group_id: str | None, target_hint: str) -> str:
    return f"{group_id or 'private'}:{_normalize_alias_key(target_hint)}"


def _remember_target_resolution(
    group_id: str | None,
    target_hint: str,
    user_id: str,
) -> None:
    normalized_hint = _normalize_alias_key(target_hint)
    user_id = str(user_id).strip()
    if not normalized_hint or not user_id:
        return
    _NICKNAME_RESOLUTION_MEMORY[_resolution_memory_key(group_id, target_hint)] = (
        time.monotonic(),
        user_id,
    )
    if len(_NICKNAME_RESOLUTION_MEMORY) > _NICKNAME_RESOLUTION_MEMORY_MAX:
        for _evict_key in sorted(
            _NICKNAME_RESOLUTION_MEMORY,
            key=lambda k: _NICKNAME_RESOLUTION_MEMORY[k][0],
        )[: len(_NICKNAME_RESOLUTION_MEMORY) - _NICKNAME_RESOLUTION_MEMORY_MAX]:
            _NICKNAME_RESOLUTION_MEMORY.pop(_evict_key, None)


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
    user_id = str(user_id).strip()
    return user_id or None


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
                overlap = min(len(hint), len(alias_text)) / max(
                    len(hint), len(alias_text)
                )
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
    if (
        top_score < ambiguous_top_threshold
        or (top_score - second_score) < ambiguous_gap_threshold
    ):
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
        return "我不太确定你说的是谁。请重新发送完整命令，并直接@目标成员。"
    display_options: list[str] = []
    for profile in candidates[:4]:
        display_name = str(profile.get("display_name") or "").strip()
        user_id = str(profile.get("user_id") or "").strip()
        if display_name and user_id:
            display_options.append(f"{display_name}(@{user_id})")
    if not display_options:
        return "我不太确定你说的是谁。请重新发送完整命令，并直接@目标成员。"
    options = "、".join(display_options)
    return f"我匹配到好几个可能对象：{options}。请重新发送完整命令并@目标成员。"


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
    target_policy: AdapterTargetPolicy | None = None,
    command_heads: set[str] | None = None,
) -> str:
    policy = target_policy or AdapterTargetPolicy()
    normalized_original = normalize_message_text(original_message or "")
    normalized_route = normalize_message_text(route_message or "")
    if not normalized_original or not normalized_route:
        return ""
    if _extract_at_tokens(normalized_route):
        return ""
    if _is_technical_request_like(
        normalized_original
    ) and not _has_adapter_context_hint(normalized_original, policy):
        return ""
    if _contains_non_self_target_phrase(normalized_original):
        return "strong"
    if _contains_third_person_reference(normalized_original):
        return "strong"
    if _needs_target_for_route(
        normalized_original,
        normalized_route,
        target_policy=policy,
    ):
        return "strong"
    if command_heads:
        for head in sorted(command_heads, key=len, reverse=True):
            if head and parse_command_with_head(
                normalized_route,
                head,
                allow_sticky=True,
            ):
                return "weak"
    return ""


def _needs_target_for_route(
    message_text: str,
    route_message: str,
    *,
    target_policy: AdapterTargetPolicy | None = None,
) -> bool:
    policy = target_policy or AdapterTargetPolicy()
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    if not policy.require_target_for_third_person:
        return False
    if not _has_adapter_context_hint(normalized, policy):
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
    if has_target and policy.allow_at_as_target:
        return False
    if has_image and policy.allow_image_as_target:
        return False
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
        nickname = str(getattr(member, "nickname", "") or "").strip()
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


async def _build_dialogue_context_pack(
    *,
    event_context: ChatInterEventContext,
    mention_profiles: dict[str, dict[str, str]] | None = None,
) -> tuple[
    DialogueContextPack,
    PersonProfile | None,
    AddresseeResult,
    ThreadContext,
    InterventionDecision,
]:
    speaker_profile = await get_person_profile(
        user_id=event_context.user_id,
        group_id=event_context.group_id,
        fallback_name=event_context.nickname,
    )
    await upsert_seen_person(
        user_id=event_context.user_id,
        group_id=event_context.group_id,
        nickname=event_context.nickname,
    )
    addressee = resolve_addressee(
        event_context=event_context,
        bot_names=(BotConfig.self_nickname or "",),
        mention_profiles=mention_profiles,
        speaker_profile=speaker_profile,
    )
    thread = resolve_thread_context(
        event_context=event_context,
        addressee=addressee,
    )
    route_signal = (
        contains_any(event_context.normalized_text, ROUTE_ACTION_WORDS)
        or bool(event_context.mentions)
        or bool(event_context.images)
    )
    intervention = decide_intervention(
        event_context=event_context,
        addressee=addressee,
        route_signal=route_signal,
    )
    pack = DialogueContextPack(
        event_context=event_context,
        speaker_profile=speaker_profile,
        addressee=addressee,
        thread=thread,
    )
    return pack, speaker_profile, addressee, thread, intervention


def _collect_target_capable_command_heads(knowledge_base) -> set[str]:
    heads: set[str] = set()
    plugins = getattr(knowledge_base, "plugins", None) or []
    for plugin in plugins:
        plugin_policy = _get_route_target_policy(
            plugin_module=getattr(plugin, "module", ""),
            plugin_name=getattr(plugin, "name", ""),
        )
        for meta in getattr(plugin, "command_meta", None) or []:
            policy = resolve_command_target_policy(
                meta,
                adapter_policy=plugin_policy,
            )
            image_min = int(getattr(meta, "image_min", 0) or 0)
            allow_sticky_arg = bool(getattr(meta, "allow_sticky_arg", False))
            if (
                not policy.allow_at
                and not policy.allow_image_as_target
                and not policy.allow_reply_image_as_target
                and image_min <= 0
                and policy.target_requirement == "none"
                and not allow_sticky_arg
            ):
                continue
            command_text = normalize_message_text(
                str(getattr(meta, "command", "") or "")
            )
            if command_text:
                heads.add(normalize_message_text(command_text.split(" ", 1)[0]))
            for alias in getattr(meta, "aliases", None) or []:
                alias_text = normalize_message_text(str(alias or ""))
                if alias_text:
                    heads.add(normalize_message_text(alias_text.split(" ", 1)[0]))
    return {head for head in heads if head}


def _get_route_target_policy(
    *,
    plugin_module: str = "",
    plugin_name: str = "",
    command_id: str = "",
) -> AdapterTargetPolicy:
    return get_adapter_target_policy(
        plugin_module=plugin_module,
        plugin_name=plugin_name,
        command_id=command_id,
    )


def _route_target_policy_from_result(
    route_result: RouteResolveResult,
) -> AdapterTargetPolicy:
    return _get_route_target_policy(
        plugin_module=route_result.decision.plugin_module,
        plugin_name=route_result.decision.plugin_name,
        command_id=route_result.command_id or "",
    )


def _has_adapter_context_hint(
    message_text: str,
    policy: AdapterTargetPolicy,
) -> bool:
    hints = tuple(policy.context_hints or ())
    if not hints:
        return False
    return contains_any(normalize_message_text(message_text or ""), hints)


def _finish_trace(
    *,
    trace: StageTrace,
    user_id: str,
    group_id: str | None,
    message_preview: str,
    route_report: RouteAttemptReport | None,
    budget_controller: TurnBudgetController | None = None,
) -> None:
    total_seconds = trace.finish()
    emit_turn_metrics(
        build_turn_metrics_snapshot(
            trace=trace,
            total_seconds=total_seconds,
            route_report=route_report,
            budget_controller=budget_controller,
        )
    )
    record_route_observation(
        user_id=user_id,
        group_id=group_id,
        message_preview=message_preview,
        trace_tags=dict(trace.tags),
        route_report=route_report,
    )


def _tag_execution_observation(
    trace: StageTrace,
    observation: ExecutionObservation,
) -> None:
    trace.update_tags(
        exec_action=observation.action,
        exec_success=int(observation.success),
        exec_reason=observation.reason,
        exec_latency_ms=observation.latency_ms,
    )


def _route_report_value(
    route_report: RouteAttemptReport | None,
    name: str,
    default: object = 0,
):
    if route_report is None:
        return default
    return getattr(route_report, name, default)


def _route_report_observer_kwargs(
    route_report: RouteAttemptReport | None,
) -> dict[str, object]:
    return {
        "candidate_total": _route_report_value(route_report, "candidate_total", 0),
        "tool_candidates": _route_report_value(route_report, "tool_candidates", 0),
        "no_hit_recovery_attempts": _route_report_value(
            route_report,
            "no_hit_recovery_attempts",
            0,
        ),
        "no_hit_recovery_success": _route_report_value(
            route_report,
            "no_hit_recovery_success",
            0,
        ),
        "no_hit_recovery_query": _route_report_value(
            route_report,
            "no_hit_recovery_query",
            "",
        ),
        "no_hit_recovery_reason": _route_report_value(
            route_report,
            "no_hit_recovery_reason",
            "",
        ),
        "rerank_attempts": _route_report_value(route_report, "rerank_attempts", 0),
        "rerank_success": _route_report_value(route_report, "rerank_success", 0),
        "rerank_no_available": _route_report_value(
            route_report,
            "rerank_no_available",
            0,
        ),
        "rerank_stage": _route_report_value(route_report, "rerank_stage", ""),
        "rerank_reason": _route_report_value(route_report, "rerank_reason", ""),
    }


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


def _is_public_command_meta(meta) -> bool:
    return (
        normalize_message_text(
            str(getattr(meta, "access_level", "public") or "public")
        ).lower()
        == "public"
    )


def _find_route_command_schema(route_result: RouteResolveResult, knowledge_plugins):
    decision = route_result.decision
    head = _normalize_head(decision.command)
    command_id = normalize_message_text(route_result.command_id or "").casefold()
    if not head and not command_id:
        return None
    exact_module_plugins = [
        plugin
        for plugin in knowledge_plugins
        if plugin.module == decision.plugin_module
    ]
    candidate_plugins = exact_module_plugins or [
        plugin for plugin in knowledge_plugins if plugin.name == decision.plugin_name
    ]
    for plugin in candidate_plugins:
        if command_id:
            for meta in plugin.command_meta:
                meta_id = normalize_message_text(
                    str(getattr(meta, "command_id", "") or "")
                ).casefold()
                if meta_id and meta_id == command_id:
                    return meta
        plugin_aliases = {
            _normalize_head(alias).casefold()
            for alias in (getattr(plugin, "aliases", None) or [])
            if _normalize_head(alias)
        }
        for meta in plugin.command_meta:
            if not _is_public_command_meta(meta):
                continue
            command_head = normalize_message_text(getattr(meta, "command", ""))
            if not command_head:
                continue
            if match_command_head_canonical(head, command_head) or any(
                match_command_head_canonical(head, alias)
                for alias in _iter_meta_aliases(meta)
            ):
                return meta
        if head in plugin_aliases and len(plugin.command_meta) == 1:
            return plugin.command_meta[0]
    return None


def _is_route_command_executable(
    route_result: RouteResolveResult,
    knowledge_plugins,
) -> bool:
    decision = route_result.decision
    head = _normalize_head(decision.command)
    if not head:
        return False

    exact_module_plugins = [
        plugin
        for plugin in knowledge_plugins
        if plugin.module == decision.plugin_module
    ]
    candidate_plugins = exact_module_plugins or [
        plugin for plugin in knowledge_plugins if plugin.name == decision.plugin_name
    ]
    if not candidate_plugins:
        return False

    for plugin in candidate_plugins:
        for meta in getattr(plugin, "command_meta", None) or []:
            if not _is_public_command_meta(meta):
                continue
            command_head = normalize_message_text(getattr(meta, "command", ""))
            if command_head and match_command_head_canonical(head, command_head):
                return True
            for alias in getattr(meta, "aliases", None) or []:
                alias_head = normalize_message_text(str(alias or ""))
                if alias_head and match_command_head_canonical(head, alias_head):
                    return True
        for command in getattr(plugin, "commands", None) or []:
            command_head = _normalize_head(str(command or ""))
            if command_head and match_command_head_canonical(head, command_head):
                return True
    return False


def _is_schema_self_only(schema) -> bool:
    return schema_is_self_only(schema)


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
    target_policy: AdapterTargetPolicy | None = None,
) -> str:
    policy = target_policy or AdapterTargetPolicy()
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return normalized

    should_enrich = not is_usage_question(normalized) and (
        contains_any(normalized, ROUTE_ACTION_WORDS)
        or _has_adapter_context_hint(normalized, policy)
        or "[image" in normalized
        or "[@" in normalized
        or _contains_reply_reference_hint(normalized)
    )
    if not should_enrich:
        return normalized

    at_tokens = _extract_at_tokens(normalized)
    image_tokens = _extract_image_tokens(normalized)
    enriched = normalized

    if not at_tokens and _contains_strong_self_reference(normalized):
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
        and (policy.allow_reply_image_as_target or policy.allow_image_as_target)
        and _contains_reply_reference_hint(normalized)
    ):
        suffix = " ".join("[image]" for _ in range(reply_image_count))
        enriched = normalize_message_text(f"{enriched} {suffix}")

    return enriched


def _should_retry_intent_with_refreshed_knowledge(
    message_text: str,
    intent_profile: IntentClassification,
) -> bool:
    if intent_profile.explicit_command or intent_profile.kind != "chat":
        return False
    if intent_profile.reason not in {"no_route_signal", "weak_route_signal"}:
        return False

    normalized = normalize_message_text(message_text or "")
    if (
        not normalized
        or is_usage_question(normalized)
        or has_negative_route_intent(normalized)
    ):
        return False
    if len(normalized) > 24:
        return False
    if any(token in normalized for token in _INTENT_REFRESH_PUNCTUATION):
        return False

    tokens = [token for token in normalized.split(" ") if token]
    if not tokens:
        return False
    if len(tokens) == 1:
        return 2 <= len(tokens[0]) <= 12
    if len(tokens) == 2:
        return len(tokens[0]) <= 4 and len(tokens[1]) <= 12
    return False


async def _enrich_route_message_with_fuzzy_target(
    *,
    group_id: str | None,
    original_message: str,
    route_message: str,
    mention_profiles: dict[str, dict[str, str]],
    target_policy: AdapterTargetPolicy | None = None,
    command_heads: set[str] | None = None,
) -> tuple[str, dict[str, dict[str, str]], str | None]:
    if not group_id:
        return route_message, mention_profiles, None
    if _extract_at_tokens(route_message):
        return route_message, mention_profiles, None

    trigger_strength = _resolve_fuzzy_trigger_strength(
        original_message=original_message,
        route_message=route_message,
        target_policy=target_policy,
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
                "display_name": str(
                    remembered_profile.get("display_name") or ""
                ).strip(),
                "nickname": str(remembered_profile.get("nickname") or "").strip(),
                "user_name": str(remembered_profile.get("user_name") or "").strip(),
                "uid": str(remembered_profile.get("uid") or "").strip(),
                "platform": str(remembered_profile.get("platform") or "qq").strip()
                or "qq",
                "alias_key": str(remembered_profile.get("alias_key") or "").strip(),
            }
            logger.debug(
                "ChatInter 昵称记忆命中: "
                f"hint='{target_hint}' -> "
                f"{mention_profiles[user_id].get('display_name')}(@{user_id})"
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
        return (
            route_message,
            mention_profiles,
            _build_member_ambiguity_message(ambiguous_candidates),
        )
    if matched is None:
        policy = target_policy or AdapterTargetPolicy()
        if _needs_target_for_route(
            original_message,
            route_message,
            target_policy=policy,
        ):
            return (
                route_message,
                mention_profiles,
                policy.target_missing_message
                or "要帮别人做的话，请重新发送完整命令，并补充目标成员。",
            )
        return route_message, mention_profiles, None

    user_id = str(matched.get("user_id") or "").strip()
    if not user_id:
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
        f"hint='{target_hint}' -> "
        f"{mention_profiles[user_id].get('display_name')}(@{user_id})"
    )
    if top_score >= 0.90:
        _remember_target_resolution(group_id, target_hint, user_id)
    return enriched_message, mention_profiles, None


def _select_adapter_policy_for_message(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> AdapterTargetPolicy:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return AdapterTargetPolicy()
    best_policy = AdapterTargetPolicy()
    best_score = 0
    for plugin in knowledge_base.plugins:
        policy = _get_route_target_policy(
            plugin_module=plugin.module,
            plugin_name=plugin.name,
        )
        if not policy.context_hints:
            continue
        score = 0
        if policy.media_related:
            score += 1
        if _has_adapter_context_hint(normalized, policy):
            score += 4
        command_texts: list[str] = []
        command_texts.extend(str(command or "") for command in plugin.commands)
        command_texts.extend(str(alias or "") for alias in plugin.aliases)
        for meta in getattr(plugin, "command_meta", None) or []:
            command_texts.append(str(getattr(meta, "command", "") or ""))
            command_texts.extend(
                str(alias or "") for alias in getattr(meta, "aliases", None) or []
            )
        if any(
            text and text in normalized
            for text in (normalize_message_text(item) for item in command_texts)
        ):
            score += 3
        if score > best_score:
            best_policy = policy
            best_score = score
    return best_policy if best_score > 0 else AdapterTargetPolicy()


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
            if file_id and not file_id.startswith(("http://", "https://", "base64://"))
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


def _resolve_feedback_reward(reason: str) -> float:
    return float(_ROUTE_FEEDBACK_REWARD.get(reason, 0.0))


def _build_route_slot_feedback(
    *,
    reason: str,
    route_message: str,
    route_command: str,
    image_missing: int = 0,
    text_missing: int = 0,
    allow_at: bool | None = None,
) -> dict[str, float]:
    slot_scores: dict[str, float] = {}
    has_command_head = bool(_normalize_head(route_command))
    has_target_signal = bool(_extract_at_tokens(route_message))
    has_image_signal = bool(_extract_image_tokens(route_message))
    has_text_signal = _extract_text_token_count(route_command) > 0

    if has_command_head:
        slot_scores["command_head"] = (
            1.0 if reason == _FEEDBACK_REASON_ROUTE_SUCCESS else -0.6
        )

    if reason == _FEEDBACK_REASON_ROUTE_SUCCESS:
        if has_target_signal:
            slot_scores["target"] = 0.35
        if has_image_signal:
            slot_scores["image"] = 0.35
        if has_text_signal:
            slot_scores["text"] = 0.25
        return slot_scores

    if reason == _FEEDBACK_REASON_SELF_ONLY_BLOCKED:
        slot_scores["target"] = -0.95
        return slot_scores

    if reason in {
        _FEEDBACK_REASON_TARGET_REQUIRED,
        _FEEDBACK_REASON_DIRECT_TARGET_REQUIRED,
        _FEEDBACK_REASON_FUZZY_CLARIFY,
    }:
        slot_scores["target"] = -0.65
        return slot_scores

    if reason == _FEEDBACK_REASON_MISSING_PARAMS:
        if image_missing > 0:
            slot_scores["image"] = -0.90
        if text_missing > 0:
            slot_scores["text"] = -0.75
        if allow_at and not has_target_signal:
            slot_scores["target"] = -0.55
        return slot_scores

    if reason == _FEEDBACK_REASON_REROUTE_FAILED:
        slot_scores["command_head"] = -0.85
        return slot_scores

    return slot_scores


async def _record_route_feedback(
    *,
    session_id: str | None,
    modules: set[str] | list[str],
    reason: str,
    route_message: str,
    route_command: str,
    image_missing: int = 0,
    text_missing: int = 0,
    allow_at: bool | None = None,
) -> None:
    normalized_modules = {
        normalize_message_text(str(module or ""))
        for module in modules
        if normalize_message_text(str(module or ""))
    }
    if not session_id or not normalized_modules:
        return
    slot_feedback = _build_route_slot_feedback(
        reason=reason,
        route_message=route_message,
        route_command=route_command,
        image_missing=image_missing,
        text_missing=text_missing,
        allow_at=allow_at,
    )
    try:
        await PluginRAGService.update_session_feedback(
            session_id=session_id,
            modules=normalized_modules,
            reward=_resolve_feedback_reward(reason),
            reason=reason,
            slot_feedback=slot_feedback or None,
        )
    except Exception as exc:
        logger.debug(f"更新 ChatInter 路由反馈失败: {exc}")


def _contains_self_reference(message_text: str) -> bool:
    normalized = normalize_message_text(normalize_action_phrases(message_text or ""))
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in ("我", "自己", "本人", "我的", "我自己", "自己的")
    )


def _contains_strong_self_reference(message_text: str) -> bool:
    normalized = normalize_message_text(normalize_action_phrases(message_text or ""))
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in ("我的", "我自己", "自己的", "本人", "本人的", "自己")
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
    return f"这个命令{joined}，请重新发送完整命令。"


def _build_planner_followup_message(missing: list[str]) -> str:
    labels: list[str] = []
    for item in missing:
        normalized = normalize_message_text(item).lower()
        if normalized in {"text", "文本", "文字", "参数", "内容"}:
            label = "要处理的文字"
        elif normalized in {"image", "图片", "图", "照片"}:
            label = "图片"
        elif normalized in {"reply", "回复", "引用"}:
            label = "回复上下文"
        else:
            label = item
        if label and label not in labels:
            labels.append(label)
    joined = "、".join(labels) if labels else "必要参数"
    return f"这个命令还需要{joined}，请补充后我再帮你执行。"


def _planner_missing_contains(missing: list[str], names: set[str]) -> bool:
    return any(normalize_message_text(item).lower() in names for item in missing)


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
    return "这个命令需要目标对象，请" + "、".join(hints) + "后重新发送完整命令。"


def _build_route_result_from_intent(
    intent: IntentClassification,
) -> RouteResolveResult | None:
    command_head = normalize_message_text(intent.command_head or "")
    plugin_name = str(intent.plugin_name or "").strip()
    plugin_module = str(intent.plugin_module or "").strip()
    if not command_head or not plugin_name or not plugin_module:
        return None
    payload_text = normalize_message_text(intent.payload_text or "")
    schema = intent.schema
    rewrite_command = normalize_message_text(intent.rewrite_command or "")
    command = rewrite_command if rewrite_command else command_head
    if payload_text:
        command = normalize_message_text(f"{command_head} {payload_text}")
    if schema is not None:
        command = _clamp_command_text_tokens(
            command,
            getattr(schema, "text_max", None),
        )
    command_id = normalize_message_text(str(getattr(schema, "command_id", "") or ""))
    return RouteResolveResult(
        decision=SkillRouteDecision(
            plugin_name=plugin_name,
            plugin_module=plugin_module,
            command=command,
            source="intent",
            skill_kind="intent",
        ),
        stage="intent",
        command_id=command_id or None,
    )


def _build_intent_clarification_message(intent: IntentClassification) -> str:
    if intent.kind == "help":
        if intent.command_head:
            return f"如果你是想问用法，可以直接说“真寻帮助{intent.command_head}”。"
        return "如果你是想问插件用法，可以直接说“真寻帮助 插件名”。"
    if intent.kind == "ambiguous":
        return (
            "这句更像是在发起功能调用，但我还不能确定具体命令。"
            "请补齐命令、图片或@目标后重新发送完整命令。"
        )
    return "这个请求还缺少关键信息，请补齐参数后重新发送完整命令。"


def _find_route_plugin_info(route_result: RouteResolveResult, knowledge_plugins):
    exact_module_plugins = [
        plugin
        for plugin in knowledge_plugins
        if plugin.module == route_result.decision.plugin_module
    ]
    if exact_module_plugins:
        return exact_module_plugins[0]
    for plugin in knowledge_plugins:
        if plugin.name == route_result.decision.plugin_name:
            return plugin
    return None


def _build_plugin_usage_fallback_message(
    *,
    route_result: RouteResolveResult,
    knowledge_plugins,
    current_message: str,
) -> str:
    plugin = _find_route_plugin_info(route_result, knowledge_plugins)
    command_head = _normalize_head(route_result.decision.command)
    schema = _find_route_command_schema(route_result, knowledge_plugins)
    plugin_name = route_result.decision.plugin_name
    if plugin is not None:
        plugin_name = plugin.name
    usage_line = normalize_message_text(str(getattr(plugin, "usage", "") or ""))
    example_lines: list[str] = []
    if schema is not None and plugin is not None:
        for meta in getattr(plugin, "command_meta", None) or []:
            if normalize_message_text(getattr(meta, "command", "")) != command_head:
                continue
            for example in getattr(meta, "examples", None) or []:
                normalized = normalize_message_text(str(example or ""))
                if normalized and normalized not in example_lines:
                    example_lines.append(normalized)
            break
    if not example_lines and plugin is not None:
        for meta in getattr(plugin, "command_meta", None) or []:
            for example in getattr(meta, "examples", None) or []:
                normalized = normalize_message_text(str(example or ""))
                if normalized and normalized not in example_lines:
                    example_lines.append(normalized)
                if len(example_lines) >= 2:
                    break
            if len(example_lines) >= 2:
                break

    hints: list[str] = []
    if schema is not None:
        if getattr(schema, "params", None):
            hints.append("参数: " + " / ".join(str(item) for item in schema.params))
        text_min = max(int(getattr(schema, "text_min", 0) or 0), 0)
        image_min = max(int(getattr(schema, "image_min", 0) or 0), 0)
        policy = resolve_command_target_policy(
            schema,
            adapter_policy=_route_target_policy_from_result(route_result),
        )
        if text_min > 0:
            hints.append(f"至少需要 {text_min} 段文本")
        if image_min > 0:
            if policy.allow_at:
                hints.append(f"至少需要 {image_min} 个图片/目标（可发图或@）")
            else:
                hints.append(f"至少需要 {image_min} 张图片")
        if policy.target_requirement == "required":
            hints.append("需要明确目标")
    intent_hint = "如果你是想调用这个插件，可以这样用："
    if is_usage_question(current_message):
        intent_hint = "这个插件的用法大致是："
    command_label = (
        command_head
        or normalize_message_text(route_result.decision.command)
        or "全部命令"
    )
    lines = [
        intent_hint,
        f"{plugin_name}：{command_label}",
    ]
    if usage_line:
        lines.append(f"用法：{usage_line}")
    if hints:
        lines.append("要求：" + "；".join(hints))
    if example_lines:
        lines.append("示例：" + " | ".join(example_lines[:2]))
    return "\n".join(line for line in lines if line)


def _is_image_related_route(route_result: RouteResolveResult) -> bool:
    return _route_target_policy_from_result(route_result).media_related


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


def _extract_command_payload_tokens(command: str) -> list[str]:
    normalized_command = normalize_message_text(command or "")
    if not normalized_command:
        return []
    parts = normalized_command.split(" ", 1)
    if len(parts) < 2:
        return []
    tokens: list[str] = []
    for raw_token in parts[1].split(" "):
        token = normalize_message_text(raw_token)
        if not token:
            continue
        tokens.append(token)
    return tokens


def _remove_tokens_from_command(command: str, tokens: list[str]) -> str:
    normalized_command = normalize_message_text(command or "")
    if not normalized_command or not tokens:
        return normalized_command
    parts = normalized_command.split(" ")
    head = normalize_message_text(parts[0] if parts else "")
    if not head:
        return ""
    token_set = {normalize_message_text(token) for token in tokens if token}
    payload = [
        token_text
        for token in parts[1:]
        if (token_text := normalize_message_text(token)) and token_text not in token_set
    ]
    if payload:
        return normalize_message_text(f"{head} {' '.join(payload)}")
    return head


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


def _schema_accepts_text_payload(schema) -> bool:
    text_min = max(int(getattr(schema, "text_min", 0) or 0), 0)
    if text_min > 0:
        return True
    text_max = getattr(schema, "text_max", None)
    if text_max is not None:
        try:
            return int(text_max) > 0
        except Exception:
            return False
    return bool(getattr(schema, "params", None))


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
        if _is_self_only_action_message(command):
            at_tokens = _extract_at_tokens(command)
            if at_tokens:
                command = _remove_tokens_from_command(command, at_tokens)
            return RouteExecutionPlan(command=command)
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

    schema_head = _normalize_head(getattr(schema, "command", ""))
    command_head = _normalize_head(command)
    if schema_head and command_head and schema_head != command_head:
        tail = normalize_message_text(command[len(command_head) :].strip())
        command = (
            normalize_message_text(f"{schema_head} {tail}".strip())
            if tail
            else schema_head
        )

    existing_payload_tokens = set(_extract_command_payload_tokens(command))
    payload_tokens: list[str] = []
    explicit_value = normalize_message_text(_extract_explicit_value(current_message))
    accepts_text_payload = _schema_accepts_text_payload(schema)
    if explicit_value and accepts_text_payload:
        payload_tokens.extend(
            token
            for token in explicit_value.split(" ")
            if token
            and token not in payload_tokens
            and token not in existing_payload_tokens
        )
    schema_tokens = _extract_schema_argument_tokens(current_message, schema)
    for token in schema_tokens:
        if (
            token
            and token not in payload_tokens
            and token not in existing_payload_tokens
        ):
            payload_tokens.append(token)
    if not payload_tokens and accepts_text_payload:
        parsed_payload = ""
        try:
            parsed = parse_command_with_head(
                current_message,
                schema_head or command_head,
                allow_sticky=bool(getattr(schema, "allow_sticky_arg", False)),
                max_prefix_len=16,
            )
            parsed_payload = normalize_message_text(
                (parsed.payload_text if parsed else "") or ""
            )
        except Exception:
            parsed_payload = ""
        if parsed_payload:
            for token in parsed_payload.split(" "):
                if (
                    token
                    and token not in payload_tokens
                    and token not in existing_payload_tokens
                ):
                    payload_tokens.append(token)
    if payload_tokens:
        command = _append_unique_tokens(command, payload_tokens)

    if not getattr(schema, "params", None):
        command = _clamp_command_text_tokens(command, getattr(schema, "text_max", None))

    image_min = max(int(getattr(schema, "image_min", 0) or 0), 0)
    text_min = max(int(getattr(schema, "text_min", 0) or 0), 0)
    policy = resolve_command_target_policy(
        schema,
        adapter_policy=_route_target_policy_from_result(route_result),
    )
    target_requirement = policy.target_requirement
    allow_at = policy.allow_at
    if allow_at:
        command_at = _extract_at_tokens(command)
    else:
        command_at = []
        disallowed_at = _extract_at_tokens(command)
        if disallowed_at:
            command = _remove_tokens_from_command(command, disallowed_at)
    command_images = _extract_image_tokens(command)
    message_images = _extract_image_tokens(current_message)

    merged_at: list[str] = []
    if allow_at:
        merged_at = command_at[:]
        for token in _extract_at_tokens(current_message):
            if token not in merged_at:
                merged_at.append(token)
        if target_requirement == "none" and merged_at:
            command = _remove_tokens_from_command(command, merged_at)
            merged_at = []
    merged_images = command_images[:]
    for token in message_images:
        if token not in merged_images:
            merged_images.append(token)

    if (
        image_min > 0
        and allow_at
        and not merged_at
        and _contains_self_reference(current_message)
    ):
        self_at = f"[@{user_id}]"
        merged_at.append(self_at)

    if target_requirement == "required" and not (merged_at or merged_images):
        if allow_at and _contains_self_reference(current_message):
            merged_at.append(f"[@{user_id}]")
        else:
            return RouteExecutionPlan(
                command=_apply_route_command_prefixes(command, schema),
                need_followup=True,
                followup_message=(
                    policy.target_missing_message
                    or _build_target_required_message(schema)
                ),
                feedback_reason=_FEEDBACK_REASON_TARGET_REQUIRED,
                allow_at=allow_at,
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
            command=_apply_route_command_prefixes(command, schema),
            need_followup=True,
            followup_message=_build_followup_message(
                image_missing=image_missing,
                text_missing=text_missing,
                allow_at=allow_at,
            ),
            feedback_reason=_FEEDBACK_REASON_MISSING_PARAMS,
            image_missing=image_missing,
            text_missing=text_missing,
            allow_at=allow_at,
        )

    if allow_at and merged_at:
        command = _append_unique_tokens(command, merged_at)

    return RouteExecutionPlan(command=_apply_route_command_prefixes(command, schema))


def _plan_route_command(
    *,
    route_result: RouteResolveResult,
    knowledge_plugins,
    current_message: str,
    has_reply: bool,
    image_count: int,
) -> CommandPlanDecision:
    knowledge_base = PluginKnowledgeBase(
        plugins=list(knowledge_plugins),
        user_role="普通用户",
    )
    references = PluginRegistry.build_plugin_references(knowledge_base)
    decision = route_result.decision
    return plan_command(
        action="execute",
        plugin_module=decision.plugin_module,
        plugin_name=decision.plugin_name,
        command=decision.command,
        command_id=route_result.command_id,
        slots=route_result.slots,
        references=references,
        current_message=current_message,
        has_reply=has_reply,
        image_count=image_count,
        reason=f"route_stage:{route_result.stage}",
    )


def _apply_command_plan_to_route_result(
    route_result: RouteResolveResult,
    command_plan: CommandPlanDecision,
) -> RouteResolveResult:
    final_command = normalize_message_text(command_plan.final_command or "")
    final_command = final_command or normalize_message_text(
        route_result.decision.command
    )
    merged_slots = {**route_result.slots, **dict(command_plan.slots or {})}
    command_id = command_plan.command_id or route_result.command_id
    missing = tuple(command_plan.missing or route_result.missing)
    if not final_command and (
        command_id == route_result.command_id
        and merged_slots == route_result.slots
        and missing == route_result.missing
    ):
        return route_result
    decision = route_result.decision
    if (
        normalize_message_text(decision.command) == final_command
        and command_id == route_result.command_id
        and merged_slots == route_result.slots
        and missing == route_result.missing
    ):
        return route_result
    return RouteResolveResult(
        decision=SkillRouteDecision(
            plugin_name=decision.plugin_name,
            plugin_module=decision.plugin_module,
            command=final_command,
            source=decision.source,
            skill_kind=decision.skill_kind,
        ),
        stage=route_result.stage,
        report=route_result.report,
        command_id=command_id,
        slots=merged_slots,
        missing=missing,
        selected_rank=route_result.selected_rank,
        selected_score=route_result.selected_score,
        selected_reason=route_result.selected_reason,
    )


def _apply_route_command_prefixes(command: str, schema) -> str:
    normalized = normalize_message_text(command)
    if not normalized or schema is None:
        return normalized
    raw_prefixes = getattr(schema, "prefixes", None) or []
    prefixes: list[str] = []
    for prefix in raw_prefixes:
        prefix_text = normalize_message_text(str(prefix or ""))
        if prefix_text and prefix_text not in prefixes:
            prefixes.append(prefix_text)
    if not prefixes:
        return normalized
    if any(normalized.startswith(prefix) for prefix in prefixes):
        return normalized
    return normalize_message_text(f"{prefixes[0]}{normalized}")


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
    session_id: str | None = None,
    has_reply: bool = False,
    extra_image_segments=None,
    route_report: RouteAttemptReport | None = None,
    budget_controller: TurnBudgetController | None = None,
    finalize_callback=None,
) -> bool:
    planned_image_count = len(_extract_image_tokens(current_message))
    if extra_image_segments:
        try:
            planned_image_count += len(extra_image_segments)
        except TypeError:
            planned_image_count += 1
    command_plan = _plan_route_command(
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        current_message=current_message,
        has_reply=has_reply,
        image_count=planned_image_count,
    )
    trace.update_tags(
        planner_action=command_plan.action,
        planner_command=command_plan.final_command or "",
        planner_missing=",".join(command_plan.missing),
    )
    route_result = _apply_command_plan_to_route_result(route_result, command_plan)
    decision = route_result.decision
    route_head = normalize_message_text(str(decision.command or "").split(" ", 1)[0])
    trace.update_tags(
        path="plugin",
        route_stage=route_result.stage,
        route_plugin=decision.plugin_name,
        route_module=decision.plugin_module,
        route_head=route_head or "unknown",
    )
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
    if not execution_plan.need_followup and command_plan.action == "clarify":
        execution_plan = RouteExecutionPlan(
            command=command_plan.final_command or decision.command,
            need_followup=True,
            followup_message=_build_planner_followup_message(command_plan.missing),
            feedback_reason=_FEEDBACK_REASON_MISSING_PARAMS,
            image_missing=1
            if _planner_missing_contains(command_plan.missing, {"image", "图片"})
            else 0,
            text_missing=1
            if _planner_missing_contains(
                command_plan.missing,
                {"text", "文本", "文字", "参数", "内容"},
            )
            else 0,
        )
    if execution_plan.need_followup:
        trace.set_tag("outcome", "plugin_missing_fallback_chat")
        logger.debug(
            "技能路由参数不足，取消待补全并回退纯对话："
            f"stage={route_result.stage}, "
            f"plugin={decision.plugin_name}, "
            f"command={decision.command}, "
            f"missing={command_plan.missing}"
        )
        return False

    route_command = execution_plan.command or decision.command
    trace.set_tag(
        "route_head", normalize_message_text(str(route_command).split(" ", 1)[0])
    )
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
        command_id=route_result.command_id or "",
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

    execution_frame = start_execution_observation(
        action="execute",
        plugin_module=decision.plugin_module,
        plugin_name=decision.plugin_name,
        command_id=route_result.command_id,
        command=route_command,
        route_stage=route_result.stage,
        session_id=session_id,
        message_preview=current_message,
        selected_rank=route_result.selected_rank,
        selected_score=route_result.selected_score,
        selected_reason=route_result.selected_reason,
        **_route_report_observer_kwargs(route_report),
    )
    success = await reroute_to_plugin(
        bot,
        event,
        route_command,
        target_modules=target_modules,
        extra_image_segments=extra_image_segments,
    )
    if success:
        _tag_execution_observation(
            trace,
            execution_frame.finish(
                success=True,
                reason=EXECUTION_REASON_ROUTE_SUCCESS,
            ),
        )
        trace.set_tag("outcome", "plugin_reroute")
        await _record_route_feedback(
            session_id=session_id,
            modules=target_modules,
            reason=_FEEDBACK_REASON_ROUTE_SUCCESS,
            route_message=current_message,
            route_command=route_command,
        )
        trace.stage("route")
        if finalize_callback is not None:
            await finalize_callback()
        _finish_trace(
            trace=trace,
            user_id=str(user_id),
            group_id=group_id,
            message_preview=current_message,
            route_report=route_report,
            budget_controller=budget_controller,
        )
    else:
        _tag_execution_observation(
            trace,
            execution_frame.finish(
                success=False,
                reason=EXECUTION_REASON_REROUTE_FAILED,
            ),
        )
        trace.set_tag("outcome", "plugin_reroute_failed")
        await _record_route_feedback(
            session_id=session_id,
            modules=target_modules,
            reason=_FEEDBACK_REASON_REROUTE_FAILED,
            route_message=current_message,
            route_command=route_command,
        )
    return success


async def _handle_router_usage_response(
    *,
    trace: StageTrace,
    route_result: RouteResolveResult,
    knowledge_plugins,
    route_message: str,
    user_id: str,
    group_id: str | None,
    nickname: str,
    user_message,
    bot_id: str | None,
    session_id: str | None,
    route_report: RouteAttemptReport | None,
    budget_controller: TurnBudgetController | None,
    finalize_callback=None,
) -> None:
    execution_frame = start_execution_observation(
        action="usage",
        plugin_module=route_result.decision.plugin_module,
        plugin_name=route_result.decision.plugin_name,
        command_id=route_result.command_id,
        command=route_result.decision.command,
        route_stage=route_result.stage,
        session_id=session_id,
        message_preview=route_message,
        selected_rank=route_result.selected_rank,
        selected_score=route_result.selected_score,
        selected_reason=route_result.selected_reason,
        **_route_report_observer_kwargs(route_report),
    )
    trace.update_tags(path="clarify", outcome="plugin_usage_redirect")
    envelope = TurnChannelEnvelope()
    envelope.add(ChannelName.ANALYSIS, "router plugin usage")
    envelope.add(
        ChannelName.FINAL,
        _build_plugin_usage_fallback_message(
            route_result=route_result,
            knowledge_plugins=knowledge_plugins,
            current_message=route_message,
        ),
    )
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
    trace.stage("send")
    if finalize_callback is not None:
        await finalize_callback(
            response_text=envelope.final,
            phase="post_gate:router_usage",
        )
    _tag_execution_observation(
        trace,
        execution_frame.finish(
            success=True,
            reason=EXECUTION_REASON_USAGE_REPLIED,
        ),
    )
    _finish_trace(
        trace=trace,
        user_id=str(user_id),
        group_id=group_id,
        message_preview=route_message,
        route_report=route_report,
        budget_controller=budget_controller,
    )


async def _build_chat_fallback_reply(
    *,
    bot: Bot,
    event: Event,
    user_id: str,
    group_id: str | None,
    nickname: str,
    model_name: str | None,
    mention_name_map: dict[str, str],
    session_key: str,
    current_message: str,
    middleware_state: TurnMiddlewareState,
    dialogue_plan: ChatDialoguePlan,
    image_parts,
    budget_controller: TurnBudgetController,
    router_force_pure_chat: bool,
) -> tuple[str, bool, bool]:
    intent_profile = middleware_state.intent
    if intent_profile is None:
        raise RuntimeError("missing intent profile for chat fallback")
    intent_timeout = int(get_config_value("INTENT_TIMEOUT", 20) or 20)
    agent_gate = decide_agent_gate(
        config_enabled=bool(get_config_value("ENABLE_AGENT_MODE", True)),
        intent=intent_profile,
        message_text=middleware_state.message_text,
        has_images=bool(image_parts),
        has_mcp_endpoints=bool(get_mcp_endpoints()),
    )
    agent_enabled = False if router_force_pure_chat else agent_gate.enabled
    logger.debug(
        f"ChatInter agent gate: enabled={agent_enabled} reason={agent_gate.reason}"
    )
    reply: str | UniMessage | None = None
    if agent_enabled:
        middleware_state.metadata = {"phase": "agent_fallback"}
        await get_middleware_manager().dispatch("before_agent", middleware_state)
        try:
            agent_response = await run_chatinter_agent(
                bot=bot,
                event=event,
                user_id=str(user_id),
                group_id=str(group_id) if group_id else None,
                model=model_name,
                timeout=max(intent_timeout, 5),
                system_prompt=middleware_state.system_prompt,
                context_xml=middleware_state.context_xml,
                message_text=middleware_state.message_text,
                image_parts=image_parts or None,
                budget_controller=budget_controller,
            )
            if agent_response and str(agent_response.text or "").strip():
                reply = str(agent_response.text)
            usage = (
                agent_response.usage_info
                if agent_response and isinstance(agent_response.usage_info, dict)
                else {}
            )
            logger.debug(
                "chatinter agent reply ready: "
                f"prompt_tokens={usage.get('prompt_tokens', 0)} "
                f"completion_tokens={usage.get('completion_tokens', 0)} "
                f"total_tokens={usage.get('total_tokens', 0)}"
            )
        except Exception as exc:
            logger.warning(f"ChatInter agent 执行失败，降级普通对话: {exc}")
    if reply is None:
        reply = await handle_chat_message(
            message=middleware_state.message_text,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            mention_name_map=mention_name_map,
            session_key=session_key,
            budget_controller=budget_controller,
            dialogue_plan=dialogue_plan,
            context_xml=middleware_state.context_xml,
        )

    reply_text = (
        str(reply)
        if reply is not None and str(reply).strip()
        else "我暂时没想好怎么回答你。"
    )
    middleware_state.response_text = reply_text
    if agent_enabled:
        await get_middleware_manager().dispatch("after_agent", middleware_state)
    await get_middleware_manager().dispatch("after_chat", middleware_state)
    reply_text = (
        middleware_state.response_text
        if middleware_state.response_text is not None
        else reply_text
    )
    reply_text = normalize_ai_reply_text(reply_text or "")
    refined_reply_text = await refine_chat_reply(
        plan=dialogue_plan,
        user_message=current_message,
        reply_text=reply_text,
        context_xml=middleware_state.context_xml,
        budget_controller=budget_controller,
    )
    rewritten = refined_reply_text != reply_text
    reply_text = replace_mention_ids_with_names(refined_reply_text, mention_name_map)
    return reply_text, rewritten, agent_enabled


async def handle_fallback(
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message=None,
    route_modules: set[str] | None = None,
    cached_plain_text: str | None = None,
) -> None:
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
    global _last_knowledge_refresh_ts

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
    model_name = get_model_name()
    session_key = str(group_id or user_id)
    is_superuser = _resolve_superuser(bot, str(user_id))
    middleware = get_middleware_manager()
    budget_controller = TurnBudgetController.for_session(session_key)
    current_message = raw_message
    chat_system_prompt = ""
    enriched_context_xml = ""
    route_report: RouteAttemptReport | None = None
    intent_profile: IntentClassification | None = None
    event_context: ChatInterEventContext | None = None
    dialogue_context_pack: DialogueContextPack | None = None
    addressee_result: AddresseeResult | None = None
    thread_context: ThreadContext | None = None
    intervention_decision: InterventionDecision | None = None
    router_force_pure_chat = False
    completion_disabled_force_chat = False
    mention_name_map: dict[str, str] = {}
    mention_profiles: dict[str, dict[str, str]] = {}
    middleware_state = TurnMiddlewareState(
        session_key=session_key,
        user_id=str(user_id),
        group_id=str(group_id) if group_id else None,
        message_text=raw_message,
        system_prompt="",
        context_xml="",
        model_name=model_name,
        budget_controller=budget_controller,
        metadata={"phase": "pre_gate"},
    )
    post_gate_dispatched = False

    async def _dispatch_post_gate(
        *,
        response_text: str | None = None,
        phase: str = "post_gate",
    ) -> None:
        nonlocal post_gate_dispatched
        if post_gate_dispatched:
            return
        if response_text is not None:
            middleware_state.response_text = response_text
        middleware_state.metadata = {
            **middleware_state.metadata,
            "phase": phase,
        }
        await middleware.dispatch("post_gate", middleware_state)
        post_gate_dispatched = True

    try:
        event_message = event.get_message()
    except Exception:
        event_message = None

    try:
        await middleware.dispatch("pre_gate", middleware_state)
        ChatFeedbackStore.inspect_user_followup(
            session_id=session_key,
            message_text=raw_message,
        )
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

        event_context = build_event_context(
            bot=bot,
            event=event,
            session=session,
            raw_message=raw_message,
            nickname=nickname,
            event_message=event_message,
            uni_msg=uni_msg,
            cached_plain_text=cached_plain_text,
        )
        mention_profiles = await _build_mention_profiles(
            str(group_id) if group_id else None,
            event_context.message_text_with_tags,
            bot_id=bot_id,
        )
        (
            dialogue_context_pack,
            _speaker_profile,
            addressee_result,
            thread_context,
            intervention_decision,
        ) = await _build_dialogue_context_pack(
            event_context=event_context,
            mention_profiles=mention_profiles,
        )
        trace.update_tags(
            addressee_source=addressee_result.source,
            addressee_confidence=f"{addressee_result.confidence:.2f}",
            thread_id=thread_context.thread_id,
            intervention=intervention_decision.action,
            intervention_reason=intervention_decision.reason,
        )

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
            dialogue_context_pack,
        )
        trace.stage("context")

        # 优先使用 UniMessage 处理
        if event_message is not None:
            current_message = uni_to_text_with_tags(event_message)
        elif uni_msg:
            current_msg = remove_reply_segment(uni_msg)
            current_message = uni_to_text_with_tags(current_msg)
        elif cached_plain_text:
            current_message = cached_plain_text.strip()
        else:
            # 无法解析为 UniMessage 时，使用原始消息文本
            current_message = raw_message.strip()

        middleware_state.message_text = current_message
        middleware_state.system_prompt = chat_system_prompt
        middleware_state.context_xml = context_xml
        middleware_state.metadata = {"phase": "intent_routing"}
        await middleware.dispatch("before_intent", middleware_state)
        chat_system_prompt = middleware_state.system_prompt
        context_xml = middleware_state.context_xml
        budget_report = middleware_state.metadata.get("budget_report")
        if isinstance(budget_report, dict):
            logger.debug(
                "ChatInter intent budget: "
                f"before={budget_report.get('before_tokens')} "
                f"after={budget_report.get('after_tokens')} "
                f"budget={budget_report.get('budget')} "
                f"ratio={budget_report.get('ratio')}"
            )
        trace.stage("intent_budget")

        command_heads = _collect_target_capable_command_heads(knowledge_base)
        reply_sender_id = (
            event_context.reply.sender_id
            if event_context is not None and event_context.reply is not None
            else _extract_reply_sender_id(event)
        )
        reply_image_count = len(reply_images_data or [])
        has_reply = bool(reply_sender_id) or reply_image_count > 0
        if reply_image_count > 0:
            logger.debug(f"Reply 中解析到图片 {reply_image_count} 张，将用于路由重放")
        reply_image_segments_for_reroute = _build_reply_image_segments_for_reroute(
            reply_images_data
        )
        pre_route_target_policy = _select_adapter_policy_for_message(
            current_message,
            knowledge_base,
        )
        route_message_base = _build_route_message_with_explicit_context(
            message_text=current_message,
            user_id=str(user_id),
            reply_image_count=reply_image_count,
            reply_sender_id=reply_sender_id,
            target_policy=pre_route_target_policy,
        )
        (
            route_message,
            mention_profiles,
            fuzzy_prompt,
        ) = await _enrich_route_message_with_fuzzy_target(
            group_id=str(group_id) if group_id else None,
            original_message=current_message,
            route_message=route_message_base,
            mention_profiles=mention_profiles,
            target_policy=pre_route_target_policy,
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
                        + (f"(uid:{profile.get('uid')})" if profile.get("uid") else "")
                    )
                    for mapped_user_id, profile in mention_profiles.items()
                )
            )
        if fuzzy_prompt:
            completion_disabled_force_chat = True
            trace.set_tag("outcome", "target_clarify_fallback_chat")
            logger.debug(
                "目标信息需要补全，已禁用追问，继续走对话回退：" f"{fuzzy_prompt}"
            )

        if _needs_target_for_route(
            current_message,
            route_message,
            target_policy=pre_route_target_policy,
        ):
            completion_disabled_force_chat = True
            trace.set_tag("outcome", "target_required_fallback_chat")
            logger.debug(
                "路由目标缺失，已禁用追问，继续走对话回退："
                f"{pre_route_target_policy.target_missing_message or '-'}"
            )

        if route_message != current_message:
            logger.debug(
                "ChatInter 路由上下文增强："
                f"before='{current_message}' -> after='{route_message}'"
            )
        if completion_disabled_force_chat:
            knowledge_base = PluginKnowledgeBase(
                plugins=[],
                user_role=knowledge_base.user_role,
            )
        middleware_state.route_message = route_message
        middleware_state.metadata = {"phase": "route_selection"}
        await middleware.dispatch("before_route", middleware_state)
        route_message = middleware_state.route_message or route_message
        selection_context = PluginSelectionContext(
            query=route_message,
            session_id=session_key,
            user_id=str(user_id),
            group_id=str(group_id) if group_id else None,
            is_superuser=is_superuser,
            event_type=_event_type_name(event),
            adapter=_event_adapter_name(bot),
            is_private=_event_is_private(event),
            has_image=bool(_extract_image_tokens(route_message)),
            has_at=bool(_extract_at_tokens(route_message)),
            has_reply=has_reply,
            addressee_user_id=addressee_result.target_user_id
            if addressee_result
            else None,
            addressee_source=addressee_result.source if addressee_result else "",
            thread_id=thread_context.thread_id if thread_context else "",
            intervention_action=intervention_decision.action
            if intervention_decision
            else "",
        )
        knowledge_base = PluginRegistry.filter_knowledge_base(
            knowledge_base,
            selection_context=selection_context,
        )

        if should_force_knowledge_refresh(route_message, knowledge_base):
            now = time.monotonic()
            if now - _last_knowledge_refresh_ts >= _KNOWLEDGE_REFRESH_COOLDOWN:
                _last_knowledge_refresh_ts = now
                refreshed_knowledge = await get_user_plugin_knowledge(
                    force_refresh=True
                )
                filtered_knowledge = PluginRegistry.filter_knowledge_base(
                    refreshed_knowledge,
                    selection_context=selection_context,
                )
                if len(filtered_knowledge.plugins) > len(knowledge_base.plugins):
                    knowledge_base = filtered_knowledge
                    logger.info(
                        "检测到插件知识可能不完整，已执行一次自愈刷新："
                        f"{len(knowledge_base.plugins)} 个插件"
                    )

        command_tools = PluginRegistry.build_command_tool_snapshots(
            knowledge_base,
            selection_context=selection_context,
        )

        intent_profile = classify_message_intent(route_message, knowledge_base)
        trace.update_tags(
            intent_kind=intent_profile.kind,
            intent_reason=intent_profile.reason,
        )
        logger.debug(
            "ChatInter intent classify: "
            f"kind={intent_profile.kind} "
            f"reason={intent_profile.reason} "
            f"explicit={intent_profile.explicit_command} "
            f"command={intent_profile.command_head or '-'} "
            f"chat_subkind={getattr(intent_profile, 'chat_subkind', 'general_chat')} "
            f"confidence={intent_profile.confidence:.2f}"
        )
        middleware_state.intent = intent_profile
        middleware_state.route_message = route_message
        middleware_state.metadata = {
            "phase": "after_intent",
            "intent_kind": intent_profile.kind,
            "intent_reason": intent_profile.reason,
        }
        await middleware.dispatch("after_intent", middleware_state)
        chat_system_prompt = middleware_state.system_prompt
        context_xml = middleware_state.context_xml
        route_message = middleware_state.route_message or route_message

        router_decision, route_result, route_report = await resolve_llm_router(
            route_message,
            knowledge_base,
            session_key=session_key,
            budget_controller=budget_controller,
            has_reply=has_reply,
            command_tools=command_tools,
        )
        trace.update_tags(
            router_action=router_decision.action,
            router_confidence=f"{router_decision.confidence:.2f}",
            router_reason=router_decision.reason or "",
            router_plugin=route_result.decision.plugin_module if route_result else "",
            router_command=route_result.decision.command if route_result else "",
        )
        if route_report is not None:
            trace.update_tags(
                route_reason=route_report.final_reason,
                route_candidates=route_report.candidate_total,
                route_attempts=route_report.attempts,
                route_tool_candidates=route_report.tool_candidates,
            )
        logger.debug(
            "ChatInter router result: "
            f"action={router_decision.action} "
            f"confidence={router_decision.confidence:.2f} "
            f"reason={router_decision.reason or '-'} "
            f"module={route_result.decision.plugin_module if route_result else '-'} "
            f"command={route_result.decision.command if route_result else '-'}"
        )
        middleware_state.metadata = {
            "phase": "route_completed",
            "router_action": router_decision.action,
            "route_reason": route_report.final_reason if route_report else "",
        }
        await middleware.dispatch("after_route", middleware_state)

        if router_decision.action == "usage" and route_result is not None:
            await _handle_router_usage_response(
                trace=trace,
                route_result=route_result,
                knowledge_plugins=knowledge_base.plugins,
                route_message=route_message,
                user_id=user_id,
                group_id=group_id,
                nickname=nickname,
                user_message=uni_msg or current_message,
                bot_id=bot_id,
                session_id=session_key,
                route_report=route_report,
                budget_controller=budget_controller,
                finalize_callback=_dispatch_post_gate,
            )
            return

        if router_decision.action == "clarify":
            trace.update_tags(path="chat", outcome="router_clarify_fallback_chat")
            logger.debug(
                "Router 需要补参，已禁用待补全追问，回退纯对话："
                f"missing={','.join(router_decision.missing)} "
                f"reason={router_decision.reason or '-'}"
            )

        if router_decision.action == "execute" and route_result is not None:
            if await _execute_route_decision(
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
                session_id=session_key,
                has_reply=has_reply,
                extra_image_segments=reply_image_segments_for_reroute,
                route_report=route_report,
                budget_controller=budget_controller,
                finalize_callback=_dispatch_post_gate,
            ):
                return
            logger.warning(
                "Router 技能路由重路由失败，降级为纯会话处理："
                f"plugin={route_result.decision.plugin_name}, "
                f"command={route_result.decision.command}"
            )

        router_force_pure_chat = True

        # 提取图片（多模态处理）
        source_for_media = event_message or uni_msg or message or raw_message
        image_parts = await extract_images_from_message(source_for_media)
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
        dialogue_plan = plan_chat_dialogue(
            message_text=current_message,
            intent=intent_profile,
            has_images=bool(image_parts),
            has_reply=has_reply,
        )
        trace.update_tags(
            chat_kind=dialogue_plan.kind,
            chat_style=dialogue_plan.style,
            chat_reason=dialogue_plan.reason,
        )
        if await _handle_chat_dialogue_special_case(
            plan=dialogue_plan,
            trace=trace,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            user_message=uni_msg or current_message,
            bot_id=bot_id,
            session_key=session_key,
            current_message=current_message,
            route_report=route_report,
            budget_controller=budget_controller,
            finalize_callback=_dispatch_post_gate,
        ):
            return
        middleware_state.message_text = current_message
        middleware_state.system_prompt = chat_system_prompt
        middleware_state.context_xml = enriched_context_xml
        middleware_state.metadata = {"phase": "chat_fallback"}
        await middleware.dispatch("before_chat", middleware_state)
        chat_system_prompt = middleware_state.system_prompt
        enriched_context_xml = middleware_state.context_xml
        budget_report = middleware_state.metadata.get("budget_report")
        if isinstance(budget_report, dict):
            logger.debug(
                "ChatInter agent budget: "
                f"before={budget_report.get('before_tokens')} "
                f"after={budget_report.get('after_tokens')} "
                f"budget={budget_report.get('budget')} "
                f"ratio={budget_report.get('ratio')}"
            )
        trace.stage("agent_budget")
        chat_execution_frame = start_execution_observation(
            action="chat",
            route_stage="chat",
            session_id=session_key,
            message_preview=current_message,
            **_route_report_observer_kwargs(route_report),
        )
        reply_text, rewritten, agent_enabled = await _build_chat_fallback_reply(
            bot=bot,
            event=event,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            model_name=model_name,
            mention_name_map=mention_name_map,
            session_key=session_key,
            current_message=current_message,
            middleware_state=middleware_state,
            dialogue_plan=dialogue_plan,
            image_parts=image_parts,
            budget_controller=budget_controller,
            router_force_pure_chat=router_force_pure_chat,
        )
        trace.update_tags(
            agent_enabled=int(agent_enabled),
            chat_rewritten=int(rewritten),
        )
        trace.stage("chat_fallback")
        envelope = TurnChannelEnvelope()
        trace.update_tags(path="chat", outcome="chat_fallback")
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
        await MessageUtils.build_message(envelope.final).send()
        trace.stage("send")
        await _dispatch_post_gate(
            response_text=envelope.final,
            phase="post_gate:chat_fallback",
        )
        ChatFeedbackStore.record(
            session_id=session_key,
            kind="chat_rewritten" if rewritten else "chat_completed",
            message_text=current_message,
            reply_text=envelope.final,
            weight=0.35 if rewritten else 0.2,
        )
        _tag_execution_observation(
            trace,
            chat_execution_frame.finish(
                success=True,
                reason=EXECUTION_REASON_CHAT_REWRITTEN
                if rewritten
                else EXECUTION_REASON_CHAT_COMPLETED,
            ),
        )
        _finish_trace(
            trace=trace,
            user_id=str(user_id),
            group_id=group_id,
            message_preview=current_message,
            route_report=route_report,
            budget_controller=budget_controller,
        )
        return

    except asyncio.CancelledError:
        trace.update_tags(path="cancelled", outcome="cancelled")
        _tag_execution_observation(
            trace,
            record_execution_observation(
                action="chat",
                success=False,
                reason=EXECUTION_REASON_CANCELLED,
                session_id=session_key,
                message_preview=current_message,
            ),
        )
        group_name = group_id or "private"
        logger.debug(
            f"ChatInter 当前会话任务被中断: user={user_id}, group={group_name}"
        )
        await _dispatch_post_gate(phase="post_gate:cancelled")
        _finish_trace(
            trace=trace,
            user_id=str(user_id),
            group_id=group_id,
            message_preview=current_message,
            route_report=route_report,
            budget_controller=budget_controller,
        )
        return
    except Exception as e:
        trace.update_tags(path="error", outcome="error")
        _tag_execution_observation(
            trace,
            record_execution_observation(
                action="chat",
                success=False,
                reason=EXECUTION_REASON_ERROR,
                session_id=session_key,
                message_preview=current_message,
            ),
        )
        middleware_state.message_text = current_message
        middleware_state.system_prompt = chat_system_prompt
        middleware_state.context_xml = enriched_context_xml
        middleware_state.metadata = {"phase": "error", "error": str(e)}
        await middleware.dispatch("on_error", middleware_state)
        logger.error(f"ChatInter 处理失败：{e}")
        await MessageUtils.build_failure_message().send()
        trace.stage("error")
        await _dispatch_post_gate(phase="post_gate:error")
        _finish_trace(
            trace=trace,
            user_id=str(user_id),
            group_id=group_id,
            message_preview=current_message,
            route_report=route_report,
            budget_controller=budget_controller,
        )
        return


__all__ = [
    "handle_fallback",
    "remember_target_resolution",
]
