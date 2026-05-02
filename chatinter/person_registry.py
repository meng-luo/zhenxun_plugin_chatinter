from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import time
from typing import Any

from .route_text import normalize_message_text

_PROFILE_CACHE_TTL = 300.0
_PROFILE_CACHE_MAX = 2048
_profile_cache: dict[str, tuple[float, "PersonProfile"]] = {}
_ALIAS_SPLIT_PATTERN = re.compile(r"[\s,，/、|;；]+")


@dataclass(frozen=True)
class PersonProfile:
    user_id: str
    group_id: str | None
    nickname: str = ""
    group_card: str = ""
    aliases: tuple[str, ...] = ()
    known_facts: tuple[str, ...] = ()
    relationship: str = ""
    confidence: float = 0.0
    last_seen: datetime | None = None

    @property
    def display_name(self) -> str:
        return self.group_card or self.nickname or self.user_id


def _cache_key(group_id: str | None, user_id: str) -> str:
    return f"{group_id or 'private'}:{user_id}"


def _normalize_alias(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).lower()


def _split_aliases(raw: str) -> tuple[str, ...]:
    aliases: list[str] = []
    seen: set[str] = set()
    for part in _ALIAS_SPLIT_PATTERN.split(str(raw or "")):
        item = normalize_message_text(part)
        key = _normalize_alias(item)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        aliases.append(item)
    return tuple(aliases[:8])


def _trim_cache() -> None:
    now = time.monotonic()
    expired = [
        key
        for key, (ts, _) in _profile_cache.items()
        if now - ts > _PROFILE_CACHE_TTL
    ]
    for key in expired:
        _profile_cache.pop(key, None)
    if len(_profile_cache) <= _PROFILE_CACHE_MAX:
        return
    evict_count = len(_profile_cache) - _PROFILE_CACHE_MAX
    for key in sorted(
        _profile_cache,
        key=lambda item: _profile_cache[item][0],
    )[:evict_count]:
        _profile_cache.pop(key, None)


async def get_person_profile(
    *,
    user_id: str,
    group_id: str | None,
    fallback_name: str = "",
) -> PersonProfile:
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        return PersonProfile(user_id="", group_id=group_id, nickname=fallback_name)
    key = _cache_key(group_id, normalized_user_id)
    now = time.monotonic()
    cached = _profile_cache.get(key)
    if cached and now - cached[0] <= _PROFILE_CACHE_TTL:
        return cached[1]

    profile = await _load_person_profile(
        user_id=normalized_user_id,
        group_id=group_id,
        fallback_name=fallback_name,
    )
    _profile_cache[key] = (now, profile)
    _trim_cache()
    return profile


async def upsert_seen_person(
    *,
    user_id: str,
    group_id: str | None,
    nickname: str,
) -> None:
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        return
    try:
        model = _get_person_model()
        if model is None:
            return
        existing = await model.filter(
            user_id=normalized_user_id,
            group_id=group_id,
        ).first()
        if existing is None:
            await model.create(
                user_id=normalized_user_id,
                group_id=group_id,
                nickname=normalize_message_text(nickname),
                confidence=0.65,
            )
        else:
            new_name = normalize_message_text(nickname)
            if new_name and new_name != existing.nickname:
                aliases = set(_split_aliases(getattr(existing, "aliases", "") or ""))
                if existing.nickname:
                    aliases.add(existing.nickname)
                existing.aliases = "、".join(sorted(aliases))[:512]
                existing.nickname = new_name
            existing.confidence = max(float(existing.confidence or 0.0), 0.65)
            await existing.save()
    except Exception:
        return
    finally:
        _profile_cache.pop(_cache_key(group_id, normalized_user_id), None)


def format_profile_lines(profile: PersonProfile, *, prefix: str = "") -> list[str]:
    if not profile.user_id:
        return []
    label = f"{prefix}user" if prefix else "user"
    lines = [f"{label}_id={_xml_escape(profile.user_id)}"]
    if profile.display_name:
        lines.append(f"{label}_name={_xml_escape(profile.display_name)}")
    if profile.aliases:
        lines.append(f"{label}_aliases={_xml_escape('、'.join(profile.aliases[:6]))}")
    if profile.known_facts:
        lines.append(f"{label}_facts={_xml_escape('；'.join(profile.known_facts[:4]))}")
    if profile.relationship:
        lines.append(f"{label}_relationship={_xml_escape(profile.relationship)}")
    if profile.confidence:
        lines.append(f"{label}_confidence={profile.confidence:.2f}")
    return lines


async def _load_person_profile(
    *,
    user_id: str,
    group_id: str | None,
    fallback_name: str,
) -> PersonProfile:
    nickname = normalize_message_text(fallback_name)
    group_card = ""
    aliases: tuple[str, ...] = ()
    known_facts: tuple[str, ...] = ()
    relationship = ""
    confidence = 0.4 if nickname else 0.0
    last_seen: datetime | None = None

    try:
        if group_id:
            from zhenxun.models.group_member_info import GroupInfoUser

            member = await GroupInfoUser.filter(
                group_id=group_id,
                user_id=user_id,
            ).first()
            if member is not None:
                member_nickname = str(getattr(member, "nickname", "") or "").strip()
                user_name = str(getattr(member, "user_name", "") or "").strip()
                group_card = member_nickname
                nickname = member_nickname or user_name or nickname
                aliases = _split_aliases(
                    "、".join(
                        item
                        for item in (member_nickname, user_name, fallback_name)
                        if item
                    )
                )
                confidence = max(confidence, 0.75)
    except Exception:
        pass

    try:
        model = _get_person_model()
        if model is not None:
            row = await model.filter(user_id=user_id, group_id=group_id).first()
            if row is None and group_id is not None:
                row = await model.filter(user_id=user_id, group_id=None).first()
            if row is not None:
                nickname = normalize_message_text(
                    getattr(row, "nickname", "") or nickname
                )
                group_card = normalize_message_text(
                    getattr(row, "group_card", "") or group_card
                )
                stored_aliases = _split_aliases(getattr(row, "aliases", "") or "")
                aliases = tuple(dict.fromkeys((*aliases, *stored_aliases)))[:8]
                known_facts = tuple(
                    item
                    for item in _split_aliases(
                        str(getattr(row, "known_facts", "") or "").replace(
                            "；", "、"
                        )
                    )
                    if item
                )[:6]
                relationship = normalize_message_text(
                    getattr(row, "relationship", "") or ""
                )
                confidence = max(
                    confidence,
                    float(getattr(row, "confidence", 0.0) or 0.0),
                )
                last_seen = getattr(row, "last_seen", None)
    except Exception:
        pass

    return PersonProfile(
        user_id=user_id,
        group_id=group_id,
        nickname=nickname,
        group_card=group_card,
        aliases=aliases,
        known_facts=known_facts,
        relationship=relationship,
        confidence=confidence,
        last_seen=last_seen,
    )


def _get_person_model() -> Any | None:
    try:
        from .models.chat_history import ChatInterPersonProfile

        return ChatInterPersonProfile
    except Exception:
        return None


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


__all__ = [
    "PersonProfile",
    "format_profile_lines",
    "get_person_profile",
    "upsert_seen_person",
]
