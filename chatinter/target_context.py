"""Mention, nickname, and fuzzy target context helpers for ChatInter."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import time

from nonebot.adapters import Bot

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger

from .route_execution import (
    contains_self_reference,
    contains_third_person_reference,
    extract_at_tokens,
    extract_image_tokens,
    has_adapter_context_hint,
)
from .route_text import (
    contains_any,
    normalize_message_text,
    parse_command_with_head,
    strip_invoke_prefix,
)
from .target_policy import TargetPolicy

_AT_ID_TOKEN_PATTERN = re.compile(
    r"\[@([^\]\s]+)\]|(?<![0-9A-Za-z_])@(\d{5,20})(?=(?:\s|$|[\u7684\uff0c,\u3002.!！？?]))"
)
_SELF_REF_HINTS = (
    "\u6211",
    "\u81ea\u5df1",
    "\u672c\u4eba",
    "\u6211\u7684",
    "\u6211\u81ea\u5df1",
    "\u81ea\u5df1\u7684",
)
_FUZZY_TARGET_HINT_PATTERN = re.compile(
    r"(?:\u7ed9|\u5e2e|\u66ff|\u8ba9|\u53eb|\u558a|\u8bf7)"
    r"(?!\u6211|\u81ea\u5df1|\u672c\u4eba)"
    r"(?P<name>[A-Za-z0-9\u4e00-\u9fff]{1,16}?)"
    r"(?=(?:\u505a|\u6574|\u5f04|\u6765|\u53d1|\u7b7e|\u70b9|\u67e5|\u770b|\u95ee|\u751f\u6210|\u5236\u4f5c|\u7684|\u8868\u60c5|\u5934\u50cf|\u56fe\u7247|\u56fe|\u4e00\u4e0b|\u4e00\u5f20|\u4e00\u4e2a|\u4e2a|\u5f20|\u9996|[\s\uff0c,\u3002.!\uff01\uff1f?]|$))"
)
_FUZZY_TARGET_SUFFIX_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9\u4e00-\u9fff]{2,16})(?:\u7684)?"
    r"(?=(?:\u8868\u60c5|\u5934\u50cf|\u56fe\u7247|\u56fe|\u770b\u4e66|\u7b7e\u5230|\u6253\u5361|\u4e00\u76f4|\u6572|\u5403|\u6478|\u62b1|\u6376|\u9876|\u6253|\u8d34|\u6478\u6478|[\s\uff0c,\u3002.!\uff01\uff1f?]|$))"
)
_SELF_ONLY_ACTION_KEYWORDS = ("\u7b7e\u5230", "\u6253\u5361", "\u8865\u7b7e")
_TARGET_REQUIRED_ACTION_HINTS = (
    "\u7ed9",
    "\u5e2e",
    "\u66ff",
    "\u8ba9",
    "\u53eb",
)
_TECHNICAL_REQUEST_HINT_WORDS = (
    "nonebot",
    "\u63d2\u4ef6",
    "bot",
    "\u4ee3\u7801",
    "\u811a\u672c",
    "\u51fd\u6570",
    "\u7c7b",
    "\u63a5\u53e3",
    "\u62a5\u9519",
    "bug",
    "\u9519\u8bef",
    "\u8c03\u8bd5",
    "\u914d\u7f6e",
    "\u90e8\u7f72",
    "\u5b89\u88c5",
    "\u5f00\u53d1",
    "\u4ed3\u5e93",
    "git",
    "pull",
    "push",
)
_IDENTITY_PENDING_PATTERNS = (
    re.compile(
        r"(?:\u77e5\u9053|\u8ba4\u8bc6|\u4e86\u89e3|\u4f60\u77e5\u9053|\u4f60\u8ba4\u8bc6|\u4f60\u4e86\u89e3)"
        r"(?P<entity>[A-Za-z0-9\u4e00-\u9fff]{1,16})"
        r"(?:\u5417|\u561b|\u4e48|\u4e0d|\u6ca1\u6709|\u6ca1)?"
    ),
    re.compile(
        r"(?P<entity>[A-Za-z0-9\u4e00-\u9fff]{1,16})"
        r"(?:\u662f\u8c01|\u662f\u5565|\u4ec0\u4e48\u4eba|\u54ea\u4f4d|\u662f\u54ea\u4e2a|\u662f\u54ea\u4f4d)"
    ),
    re.compile(
        r"(?:\u8c01|\u54ea\u4e2a|\u54ea\u4f4d)(?:\u53eb|\u662f|\u6635\u79f0\u662f|\u7fa4\u6635\u79f0\u662f)"
        r"(?P<entity>[A-Za-z0-9\u4e00-\u9fff]{1,16})"
    ),
)
_NON_SELF_TARGET_PATTERN = re.compile(
    r"(?:\u7ed9|\u5e2e|\u66ff|\u8ba9|\u53eb|\u558a|\u8bf7)"
    r"(?!\u6211|\u81ea\u5df1|\u672c\u4eba)"
)
_TARGET_HINT_LEADING_NOISE_PATTERN = re.compile(
    r"^(?:"
    r"给|帮|替|让|叫|喊|请|把|将|用|拿|"
    r"做个|做一个|做一张|做|制作一个|制作一张|制作|"
    r"整一个|整一张|整|弄一个|弄一张|弄|来个|来一个|来一张|来"
    r")+"
)
_TARGET_HINT_TRAILING_NOISE_PATTERN = re.compile(
    r"(?:的)?(?:表情包?|梗图|头像|图片|照片|图|" r"一下|一把|一个|一张|个|张)+$"
)
_TARGET_HINT_ACTION_BOUNDARY_PATTERN = re.compile(
    r"(?:\u505a\u6210|\u53d8\u6210|\u5236\u6210|\u505a\u4e3a|\u4f5c\u4e3a|\u751f\u6210|\u5236\u4f5c|\u505a|\u6574|\u5f04|\u6765|\u53d1|\u53d8|\u8f6c).*$"
)
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


def _extract_pending_entities(message_text: str) -> tuple[str, ...]:
    normalized = normalize_message_text(message_text)
    if not normalized:
        return ()
    compact = normalized.replace(" ", "")
    values: list[str] = []
    for pattern in _IDENTITY_PENDING_PATTERNS:
        for match in pattern.finditer(compact):
            entity = normalize_message_text(match.group("entity") or "")
            if not entity:
                continue
            if entity in _SELF_REF_HINTS or entity in {"你", "真寻", "小真寻", "bot"}:
                continue
            if _is_technical_request_like(entity):
                continue
            values.append(entity[:24])
    return tuple(dict.fromkeys(values))[:4]


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


def _clean_fuzzy_target_hint(candidate: str) -> str:
    text = normalize_message_text(candidate)
    if not text:
        return ""
    previous = ""
    while text and text != previous:
        previous = text
        text = normalize_message_text(strip_invoke_prefix(text))
        text = normalize_message_text(_TARGET_HINT_LEADING_NOISE_PATTERN.sub("", text))
        text = normalize_message_text(
            _TARGET_HINT_ACTION_BOUNDARY_PATTERN.sub("", text)
        )
        text = normalize_message_text(_TARGET_HINT_TRAILING_NOISE_PATTERN.sub("", text))
        text = text.strip(" 的：:,，。.!！？?")
    if not text:
        return ""
    if text in _SELF_REF_HINTS:
        return ""
    normalized = _normalize_alias_key(text)
    if normalized in {"", "wo", "ziji"}:
        return ""
    if len(normalized) < 2 or len(normalized) > 16:
        return ""
    if _is_technical_request_like(text):
        return ""
    return text


def _extract_fuzzy_target_hint(
    message_text: str,
    command_heads: set[str] | None = None,
) -> str:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return ""
    match = _FUZZY_TARGET_HINT_PATTERN.search(normalized)
    if match:
        candidate = _clean_fuzzy_target_hint(match.group("name") or "")
        if candidate:
            return candidate

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
            for raw_part in (parsed.prefix_text, parsed.payload_text):
                candidate = _clean_fuzzy_target_hint(raw_part)
                if candidate:
                    return candidate
            tail = normalize_message_text(parsed.payload_text or parsed.prefix_text)
            tail = re.sub(r"^(?:给|帮|替|让|叫|喊|请|把|将)+", "", tail).strip()
            tail = re.sub(r"(?:做|整|弄|来|发|签|点|查|看|问|生成|制作).*$", "", tail)
            tail = tail.strip(" 的：:,，。.!！？?")
            if not tail:
                continue
            candidate = _clean_fuzzy_target_hint(tail.split(" ", 1)[0])
            if candidate:
                return candidate

    if contains_any(normalized, _TARGET_REQUIRED_ACTION_HINTS):
        suffix_match = _FUZZY_TARGET_SUFFIX_PATTERN.search(normalized)
        if suffix_match:
            candidate = _clean_fuzzy_target_hint(suffix_match.group("name") or "")
            if candidate:
                return candidate
    return ""


async def _get_group_member_profiles_for_fuzzy(
    group_id: str | None,
    bot: Bot | None = None,
) -> list[dict[str, str | tuple[str, ...]]]:
    if not group_id:
        return []
    cache_key = str(group_id)
    now = time.monotonic()
    cached = _GROUP_MEMBER_PROFILE_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _GROUP_MEMBER_PROFILE_CACHE_TTL:
        return cached[1]

    profiles: list[dict[str, str | tuple[str, ...]]] = []
    try:
        from zhenxun.models.group_member_info import GroupInfoUser

        members = await GroupInfoUser.filter(group_id=group_id).all()
    except Exception as exc:
        logger.debug(f"加载群成员映射失败: {exc}")
        members = []

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

    for profile in await _get_adapter_group_member_profiles(group_id, bot):
        user_id = str(profile.get("user_id") or "").strip()
        if not user_id or any(
            str(item.get("user_id") or "").strip() == user_id for item in profiles
        ):
            continue
        profiles.append(profile)

    # Unit tests and freshly joined groups may not have GroupInfoUser persisted
    # yet.  Fall back to recent chat history so nickname-target commands can
    # still resolve known speakers without plugin-specific shortcuts.
    for profile in await _get_recent_chat_member_profiles(group_id):
        user_id = str(profile.get("user_id") or "").strip()
        if not user_id or any(
            str(item.get("user_id") or "").strip() == user_id for item in profiles
        ):
            continue
        profiles.append(profile)

    _GROUP_MEMBER_PROFILE_CACHE[cache_key] = (now, profiles)
    if len(_GROUP_MEMBER_PROFILE_CACHE) > _GROUP_MEMBER_PROFILE_CACHE_MAX:
        for _evict_key in sorted(
            _GROUP_MEMBER_PROFILE_CACHE,
            key=lambda k: _GROUP_MEMBER_PROFILE_CACHE[k][0],
        )[: len(_GROUP_MEMBER_PROFILE_CACHE) - _GROUP_MEMBER_PROFILE_CACHE_MAX]:
            _GROUP_MEMBER_PROFILE_CACHE.pop(_evict_key, None)
    return profiles


async def _get_recent_chat_member_profiles(
    group_id: str | None,
) -> list[dict[str, str | tuple[str, ...]]]:
    if not group_id:
        return []
    try:
        from .models.chat_history import ChatInterPersonProfile

        rows = (
            await ChatInterPersonProfile.filter(group_id=group_id)
            .order_by("-last_seen", "-id")
            .limit(300)
            .values("user_id", "nickname", "group_card", "aliases")
        )
    except Exception as exc:
        logger.debug(f"加载近期群聊昵称失败: {exc}")
        return []

    profiles: list[dict[str, str | tuple[str, ...]]] = []
    seen: set[str] = set()
    for row in rows:
        user_id = str(row.get("user_id") or "").strip()
        nickname = str(row.get("nickname") or "").strip()
        group_card = str(row.get("group_card") or "").strip()
        aliases = _split_alias_text(str(row.get("aliases") or ""))
        display_name = group_card or nickname or (aliases[0] if aliases else "")
        if not user_id or user_id in seen or not display_name:
            continue
        seen.add(user_id)
        profiles.append(
            {
                "user_id": user_id,
                "display_name": display_name,
                "nickname": nickname,
                "user_name": group_card or nickname,
                "uid": "",
                "platform": "qq",
                "alias_key": _normalize_alias_key(display_name),
                "alias_keys": _build_alias_keys(
                    display_name,
                    nickname,
                    group_card,
                    *aliases,
                ),
            }
        )
    return profiles


async def _get_adapter_group_member_profiles(
    group_id: str | None,
    bot: Bot | None,
) -> list[dict[str, str | tuple[str, ...]]]:
    if not group_id or bot is None:
        return []
    try:
        rows = await bot.call_api("get_group_member_list", group_id=int(group_id))
    except Exception as exc:
        logger.debug(f"通过适配器加载群成员映射失败: {exc}")
        return []
    if not isinstance(rows, list | tuple):
        return []

    profiles: list[dict[str, str | tuple[str, ...]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        user_id = str(row.get("user_id") or row.get("uid") or "").strip()
        if not user_id:
            continue
        card = str(row.get("card") or "").strip()
        nickname = str(row.get("nickname") or "").strip()
        remark = str(row.get("remark") or "").strip()
        user_name = card or nickname or remark
        display_name = user_name or user_id
        alias_keys = _build_alias_keys(display_name, card, nickname, remark)
        profiles.append(
            {
                "user_id": user_id,
                "display_name": display_name,
                "nickname": nickname,
                "user_name": user_name,
                "uid": str(row.get("uid") or "").strip(),
                "platform": "qq",
                "alias_key": _normalize_alias_key(display_name),
                "alias_keys": alias_keys,
            }
        )
    return profiles


def _split_alias_text(raw: str) -> tuple[str, ...]:
    values: list[str] = []
    for item in re.split(r"[、,，\n\r\t ]+", raw or ""):
        text = normalize_message_text(item)
        if text and text not in values:
            values.append(text)
    return tuple(values[:12])


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
    target_policy: TargetPolicy | None = None,
    command_heads: set[str] | None = None,
) -> str:
    policy = target_policy or TargetPolicy()
    normalized_original = normalize_message_text(original_message or "")
    normalized_route = normalize_message_text(route_message or "")
    if not normalized_original or not normalized_route:
        return ""
    if extract_at_tokens(normalized_route):
        return ""
    if _is_technical_request_like(normalized_original) and not has_adapter_context_hint(
        normalized_original, policy
    ):
        return ""
    if _contains_non_self_target_phrase(normalized_original):
        return "strong"
    if contains_third_person_reference(normalized_original):
        return "strong"
    if _needs_target_for_route(
        normalized_original,
        normalized_route,
        target_policy=policy,
    ):
        return "strong"
    if command_heads and has_adapter_context_hint(normalized_original, policy):
        # Command-aware media/action requests often omit explicit "帮/给" words,
        # e.g. "做个番茄的敲表情".  Treat a clean nickname hint plus a known
        # target-capable command head as enough signal to try nickname lookup.
        if _extract_fuzzy_target_hint(normalized_route, command_heads):
            return "weak"
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
    target_policy: TargetPolicy | None = None,
) -> bool:
    policy = target_policy or TargetPolicy()
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    if not policy.require_target_for_third_person:
        return False
    if not has_adapter_context_hint(normalized, policy):
        return False
    if contains_self_reference(normalized):
        return False
    if not (
        contains_third_person_reference(normalized)
        or contains_any(normalized, _TARGET_REQUIRED_ACTION_HINTS)
    ):
        return False
    has_target = bool(extract_at_tokens(route_message))
    has_image = bool(extract_image_tokens(route_message))
    if has_target and policy.allow_at_as_target:
        return False
    if has_image and policy.allow_image_as_target:
        return False
    return not has_target and not has_image


async def _enrich_route_message_with_fuzzy_target(
    *,
    group_id: str | None,
    bot: Bot | None = None,
    original_message: str,
    route_message: str,
    mention_profiles: dict[str, dict[str, str]],
    target_policy: TargetPolicy | None = None,
    command_heads: set[str] | None = None,
) -> tuple[str, dict[str, dict[str, str]], str | None]:
    if not group_id:
        return route_message, mention_profiles, None
    if extract_at_tokens(route_message):
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

    profiles = await _get_group_member_profiles_for_fuzzy(group_id, bot=bot)
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
        policy = target_policy or TargetPolicy()
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


async def _build_mention_profiles(
    group_id: str | None,
    message_text: str,
    bot_id: str | None = None,
    bot: Bot | None = None,
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

    remaining_user_ids = {
        user_id for user_id in mentioned_user_ids if user_id not in mention_profiles
    }
    try:
        from zhenxun.models.group_member_info import GroupInfoUser

        members = await GroupInfoUser.filter(
            group_id=group_id,
            user_id__in=list(remaining_user_ids),
        ).all()
    except Exception as exc:
        logger.debug(f"解析@昵称失败: {exc}")
        members = []

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

    remaining_user_ids = {
        user_id for user_id in mentioned_user_ids if user_id not in mention_profiles
    }
    if remaining_user_ids and bot is not None:
        for user_id in remaining_user_ids:
            profile = await _get_adapter_group_member_profile(
                group_id=group_id,
                user_id=user_id,
                bot=bot,
            )
            if profile:
                mention_profiles[user_id] = profile

    return mention_profiles


async def _get_adapter_group_member_profile(
    *,
    group_id: str,
    user_id: str,
    bot: Bot,
) -> dict[str, str] | None:
    try:
        row = await bot.call_api(
            "get_group_member_info",
            group_id=int(group_id),
            user_id=int(user_id),
        )
    except Exception as exc:
        logger.debug(f"通过适配器解析@昵称失败: {exc}")
        return None
    if not isinstance(row, dict):
        return None
    card = str(row.get("card") or "").strip()
    nickname = str(row.get("nickname") or "").strip()
    display_name = card or nickname or user_id
    return {
        "display_name": display_name,
        "nickname": nickname,
        "user_name": card or nickname,
        "uid": str(row.get("uid") or "").strip(),
        "platform": "qq",
        "alias_key": _normalize_alias_key(display_name),
    }


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


extract_pending_entities = _extract_pending_entities
extract_mentioned_user_ids = _extract_mentioned_user_ids
build_mention_name_map = _build_mention_name_map
build_mention_profiles = _build_mention_profiles
append_mention_context_xml = _append_mention_context_xml
enrich_route_message_with_fuzzy_target = _enrich_route_message_with_fuzzy_target
extract_fuzzy_target_hint = _extract_fuzzy_target_hint
resolve_fuzzy_trigger_strength = _resolve_fuzzy_trigger_strength
needs_target_for_route = _needs_target_for_route
is_technical_request_like = _is_technical_request_like

__all__ = [
    "append_mention_context_xml",
    "build_mention_name_map",
    "build_mention_profiles",
    "enrich_route_message_with_fuzzy_target",
    "extract_fuzzy_target_hint",
    "extract_mentioned_user_ids",
    "extract_pending_entities",
    "is_technical_request_like",
    "needs_target_for_route",
    "remember_target_resolution",
    "resolve_fuzzy_trigger_strength",
]
