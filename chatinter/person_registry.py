from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
import time
from typing import Any

from .route_text import normalize_message_text

_PROFILE_CACHE_TTL = 300.0
_PROFILE_CACHE_MAX = 2048
_GROUP_ALIAS_CACHE_TTL = 300.0
_GROUP_ALIAS_CACHE_MAX = 128
_profile_cache: dict[str, tuple[float, "PersonProfile"]] = {}
_group_alias_cache: dict[str, tuple[float, list["PersonProfile"]]] = {}
_ALIAS_SPLIT_PATTERN = re.compile(r"[\s,，/、|;；]+")


@dataclass(frozen=True)
class PersonProfile:
    user_id: str
    group_id: str | None
    nickname: str = ""
    group_card: str = ""
    aliases: tuple[str, ...] = ()
    alias_weights: tuple[tuple[str, float], ...] = ()
    alias_sources: tuple[tuple[str, str], ...] = ()
    known_facts: tuple[str, ...] = ()
    relationship: str = ""
    conflict_state: str = ""
    confidence: float = 0.0
    last_seen: datetime | None = None

    @property
    def display_name(self) -> str:
        return self.group_card or self.nickname or self.user_id


@dataclass(frozen=True)
class AliasCandidate:
    profile: PersonProfile
    score: float
    matched_alias: str = ""


def _cache_key(group_id: str | None, user_id: str) -> str:
    return f"{group_id or 'private'}:{user_id}"


def _normalize_alias(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).lower()


def normalize_alias_key(value: str) -> str:
    return _normalize_alias(value)


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


def _load_json_dict(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _dump_json_dict(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)[:2048]


def _merge_alias_weight(
    weights: dict[str, float],
    alias: str,
    *,
    weight: float,
) -> None:
    key = _normalize_alias(alias)
    if len(key) < 2:
        return
    weights[key] = max(float(weights.get(key, 0.0) or 0.0), weight)


def _trim_cache() -> None:
    now = time.monotonic()
    expired = [
        key for key, (ts, _) in _profile_cache.items() if now - ts > _PROFILE_CACHE_TTL
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
        new_name = normalize_message_text(nickname)
        if existing is None:
            alias_weights: dict[str, float] = {}
            alias_sources: dict[str, str] = {}
            for alias in _split_aliases(new_name):
                key = _normalize_alias(alias)
                _merge_alias_weight(alias_weights, alias, weight=0.65)
                alias_sources[key] = "seen"
            await model.create(
                user_id=normalized_user_id,
                group_id=group_id,
                nickname=new_name,
                aliases="、".join(_split_aliases(new_name)),
                alias_weights=_dump_json_dict(alias_weights),
                alias_sources=_dump_json_dict(alias_sources),
                confidence=0.65,
            )
        else:
            alias_weights = _load_json_dict(
                str(getattr(existing, "alias_weights", "") or "")
            )
            alias_sources = _load_json_dict(
                str(getattr(existing, "alias_sources", "") or "")
            )
            if new_name and new_name != existing.nickname:
                aliases = set(_split_aliases(getattr(existing, "aliases", "") or ""))
                if existing.nickname:
                    aliases.add(existing.nickname)
                    _merge_alias_weight(alias_weights, existing.nickname, weight=0.55)
                    alias_sources[_normalize_alias(existing.nickname)] = "old_nickname"
                existing.aliases = "、".join(sorted(aliases))[:512]
                existing.nickname = new_name
            for alias in _split_aliases(new_name):
                _merge_alias_weight(alias_weights, alias, weight=0.65)
                alias_sources.setdefault(_normalize_alias(alias), "seen")
            existing.alias_weights = _dump_json_dict(alias_weights)
            existing.alias_sources = _dump_json_dict(alias_sources)
            existing.confidence = max(float(existing.confidence or 0.0), 0.65)
            await existing.save()
    except Exception:
        return
    finally:
        _profile_cache.pop(_cache_key(group_id, normalized_user_id), None)
        if group_id:
            _group_alias_cache.pop(str(group_id), None)


def format_profile_lines(profile: PersonProfile, *, prefix: str = "") -> list[str]:
    if not profile.user_id:
        return []
    label = f"{prefix}user" if prefix else "user"
    lines = [f"{label}_id={_xml_escape(profile.user_id)}"]
    if profile.display_name:
        lines.append(f"{label}_name={_xml_escape(profile.display_name)}")
    if profile.aliases:
        lines.append(f"{label}_aliases={_xml_escape('、'.join(profile.aliases[:6]))}")
    if profile.conflict_state:
        lines.append(f"{label}_conflict_state={_xml_escape(profile.conflict_state)}")
    if profile.known_facts:
        lines.append(f"{label}_facts={_xml_escape('；'.join(profile.known_facts[:4]))}")
    if profile.relationship:
        lines.append(f"{label}_relationship={_xml_escape(profile.relationship)}")
    if profile.confidence:
        lines.append(f"{label}_confidence={profile.confidence:.2f}")
    return lines


async def resolve_alias_candidates(
    *,
    group_id: str | None,
    text: str,
    exclude_user_id: str | None = None,
    limit: int = 5,
) -> list[AliasCandidate]:
    alias_key = _normalize_alias(text)
    if not group_id or len(alias_key) < 2:
        return []
    profiles = await _load_group_profiles(group_id)
    scored: list[AliasCandidate] = []
    exclude = str(exclude_user_id or "").strip()
    for profile in profiles:
        if exclude and profile.user_id == exclude:
            continue
        score, matched_alias = _score_alias_match(alias_key, profile)
        if score <= 0:
            continue
        scored.append(AliasCandidate(profile, score, matched_alias))
    scored.sort(key=lambda item: (item.score, item.profile.confidence), reverse=True)
    return scored[: max(int(limit or 0), 0)]


async def _load_person_profile(
    *,
    user_id: str,
    group_id: str | None,
    fallback_name: str,
) -> PersonProfile:
    nickname = normalize_message_text(fallback_name)
    group_card = ""
    aliases: tuple[str, ...] = ()
    alias_weights: tuple[tuple[str, float], ...] = ()
    alias_sources: tuple[tuple[str, str], ...] = ()
    known_facts: tuple[str, ...] = ()
    relationship = ""
    conflict_state = ""
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
                alias_weights_raw = _load_json_dict(
                    str(getattr(row, "alias_weights", "") or "")
                )
                alias_weights = tuple(
                    (str(key), float(value or 0.0))
                    for key, value in alias_weights_raw.items()
                )
                alias_sources_raw = _load_json_dict(
                    str(getattr(row, "alias_sources", "") or "")
                )
                alias_sources = tuple(
                    (str(key), str(value or ""))
                    for key, value in alias_sources_raw.items()
                )
                known_facts = tuple(
                    item
                    for item in _split_aliases(
                        str(getattr(row, "known_facts", "") or "").replace("；", "、")
                    )
                    if item
                )[:6]
                relationship = normalize_message_text(
                    getattr(row, "relationship", "") or ""
                )
                conflict_state = normalize_message_text(
                    getattr(row, "conflict_state", "") or ""
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
        alias_weights=alias_weights,
        alias_sources=alias_sources,
        known_facts=known_facts,
        relationship=relationship,
        conflict_state=conflict_state,
        confidence=confidence,
        last_seen=last_seen,
    )


async def _load_group_profiles(group_id: str) -> list[PersonProfile]:
    key = str(group_id or "").strip()
    if not key:
        return []
    now = time.monotonic()
    cached = _group_alias_cache.get(key)
    if cached and now - cached[0] <= _GROUP_ALIAS_CACHE_TTL:
        return cached[1]

    profiles: dict[str, PersonProfile] = {}
    try:
        from zhenxun.models.group_member_info import GroupInfoUser

        members = await GroupInfoUser.filter(group_id=group_id).all()
        for member in members:
            user_id = str(member.user_id or "").strip()
            if not user_id:
                continue
            nickname = str(getattr(member, "nickname", "") or "").strip()
            user_name = str(member.user_name or "").strip()
            aliases = _split_aliases(
                "、".join(item for item in (nickname, user_name) if item)
            )
            profiles[user_id] = PersonProfile(
                user_id=user_id,
                group_id=group_id,
                nickname=user_name or nickname,
                group_card=nickname,
                aliases=aliases,
                confidence=0.72,
            )
    except Exception:
        pass

    try:
        model = _get_person_model()
        if model is not None:
            rows = await model.filter(group_id=group_id).all()
            for row in rows:
                user_id = str(getattr(row, "user_id", "") or "").strip()
                if not user_id:
                    continue
                base = profiles.get(user_id)
                stored_aliases = _split_aliases(getattr(row, "aliases", "") or "")
                alias_weights_raw = _load_json_dict(
                    str(getattr(row, "alias_weights", "") or "")
                )
                alias_sources_raw = _load_json_dict(
                    str(getattr(row, "alias_sources", "") or "")
                )
                profiles[user_id] = PersonProfile(
                    user_id=user_id,
                    group_id=group_id,
                    nickname=normalize_message_text(
                        getattr(row, "nickname", "") or (base.nickname if base else "")
                    ),
                    group_card=normalize_message_text(
                        getattr(row, "group_card", "")
                        or (base.group_card if base else "")
                    ),
                    aliases=tuple(
                        dict.fromkeys(
                            (*(base.aliases if base else ()), *stored_aliases)
                        )
                    )[:8],
                    alias_weights=tuple(
                        (str(k), float(v or 0.0)) for k, v in alias_weights_raw.items()
                    ),
                    alias_sources=tuple(
                        (str(k), str(v or "")) for k, v in alias_sources_raw.items()
                    ),
                    conflict_state=normalize_message_text(
                        getattr(row, "conflict_state", "") or ""
                    ),
                    confidence=max(
                        float(getattr(row, "confidence", 0.0) or 0.0),
                        base.confidence if base else 0.0,
                    ),
                )
    except Exception:
        pass

    result = list(profiles.values())
    _group_alias_cache[key] = (now, result)
    if len(_group_alias_cache) > _GROUP_ALIAS_CACHE_MAX:
        evict_count = len(_group_alias_cache) - _GROUP_ALIAS_CACHE_MAX
        for cache_key in sorted(
            _group_alias_cache,
            key=lambda item: _group_alias_cache[item][0],
        )[:evict_count]:
            _group_alias_cache.pop(cache_key, None)
    return result


def _score_alias_match(alias_key: str, profile: PersonProfile) -> tuple[float, str]:
    candidates: dict[str, tuple[float, str]] = {}
    for alias, weight in (
        (profile.nickname, 0.72),
        (profile.group_card, 0.78),
    ):
        key = _normalize_alias(alias)
        if len(key) >= 2:
            candidates[key] = (max(candidates.get(key, (0.0, ""))[0], weight), alias)
    for alias in profile.aliases:
        key = _normalize_alias(alias)
        if len(key) >= 2:
            candidates[key] = (max(candidates.get(key, (0.0, ""))[0], 0.68), alias)
    for alias, weight in profile.alias_weights:
        key = _normalize_alias(alias)
        if len(key) >= 2:
            candidates[key] = (
                max(candidates.get(key, (0.0, ""))[0], float(weight or 0.0)),
                alias,
            )

    best_score = 0.0
    best_alias = ""
    for candidate, (weight, alias) in candidates.items():
        score = 0.0
        if alias_key == candidate:
            score = weight + 0.25
        elif len(alias_key) >= 3 and (alias_key in candidate or candidate in alias_key):
            score = weight + 0.08
        if score > best_score:
            best_score = score
            best_alias = alias or candidate
    if profile.conflict_state:
        best_score -= 0.18
    return max(best_score, 0.0), best_alias


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
    "AliasCandidate",
    "PersonProfile",
    "format_profile_lines",
    "get_person_profile",
    "normalize_alias_key",
    "resolve_alias_candidates",
    "upsert_seen_person",
]
