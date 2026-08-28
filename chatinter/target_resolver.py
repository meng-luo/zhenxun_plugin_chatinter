"""Command-aware target resolution and missing-context gate.

The pre-route pass can only guess target policy from the raw message.  Once a
native tool has selected a concrete command, this module re-checks the command
schema and resolves nickname targets again before rerouting to NoneBot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from nonebot.adapters import Bot

from .member_similarity import (
    ALIAS_AMBIGUOUS_GAP,
    ALIAS_AMBIGUOUS_TOP,
    ALIAS_MATCH_THRESHOLD,
    score_alias_in_message,
    score_member_alias,
)
from .native_route import NativeRouteResult
from .person_registry import AliasCandidate, PersonProfile, normalize_alias_key
from .route_execution import (
    extract_at_tokens,
    extract_image_tokens,
    find_route_command_schema,
)
from .route_text import normalize_message_text
from .schema_policy import CommandTargetPolicy, resolve_command_target_policy
from .target_context import build_mention_profiles

TargetResolveStatus = Literal[
    "not_needed",
    "present",
    "resolved",
    "ambiguous",
    "missing",
    "invalid",
]

VerifiedActionTargetSource = Literal[
    "at",
    "reply",
    "alias",
    "self_nickname",
    "unknown",
]

_EXPLICIT_AT_PATTERN = re.compile(
    r"\[@(?P<bracket>[^\]\s]+)\]" r"|(?<![0-9A-Za-z_])@(?P<plain>\d{5,20})(?!\d)"
)

# Characters that, immediately before an alias, mark it as a directive object.
_ALIAS_DIRECTIVE_PREFIX_CHARS = frozenset("给帮替让叫喊请@")
_ALIAS_SCRIPT_RUN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class TargetResolveResult:
    status: TargetResolveStatus
    message_text: str
    mention_profiles: dict[str, dict[str, str]] = field(default_factory=dict)
    prompt: str = ""
    target_hint: str = ""
    resolved_target_ids: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.status in {"ambiguous", "invalid"} or (
            self.status == "missing" and bool(self.prompt)
        )

    @property
    def resolved(self) -> bool:
        return self.status in {"present", "resolved"}


@dataclass(frozen=True)
class VerifiedActionTarget:
    """A single group member that may be supplied to a target-aware command."""

    user_id: str | None = None
    source: VerifiedActionTargetSource = "unknown"
    confidence: float = 0.0
    ambiguous: bool = False

    @property
    def is_resolved(self) -> bool:
        return bool(self.user_id) and not self.ambiguous


def resolve_verified_action_target(
    *,
    event_context,
    addressee,
    speaker_profile=None,
    reply_has_image: bool = False,
) -> VerifiedActionTarget:
    """Bind structural event evidence; nickname choices use the turn ledger."""

    if bool(getattr(event_context, "is_private", False)):
        return VerifiedActionTarget()

    current_user_id = str(getattr(event_context, "user_id", "") or "").strip()
    bot_id = str(getattr(event_context, "bot_id", "") or "").strip()
    member_mentions = tuple(
        dict.fromkeys(
            str(getattr(item, "user_id", "") or "").strip()
            for item in tuple(getattr(event_context, "mentions", ()) or ())
            if str(getattr(item, "user_id", "") or "").strip()
            not in {current_user_id, bot_id}
        )
    )
    if len(member_mentions) == 1:
        mention_id = member_mentions[0]
        return VerifiedActionTarget(
            user_id=mention_id,
            source="at",
            confidence=0.95,
        )
    if len(member_mentions) > 1:
        return VerifiedActionTarget(source="at", ambiguous=True)

    speaker_target = _resolve_verified_speaker_alias_target(
        event_context=event_context,
        speaker_profile=speaker_profile,
        current_user_id=current_user_id,
        bot_id=bot_id,
    )
    if speaker_target.is_resolved:
        return speaker_target

    source = str(getattr(addressee, "source", "") or "")
    target_user_id = str(getattr(addressee, "target_user_id", "") or "").strip()
    if (
        source == "reply"
        and not reply_has_image
        and not bool(getattr(addressee, "ambiguous", False))
        and target_user_id
        and target_user_id not in {current_user_id, bot_id}
    ):
        return VerifiedActionTarget(
            user_id=target_user_id,
            source="reply",
            confidence=float(getattr(addressee, "confidence", 0.84) or 0.84),
        )

    return VerifiedActionTarget()


def resolve_verified_action_target_from_group_profiles(
    *,
    event_context,
    addressee,
    group_profiles: tuple[dict[str, str | tuple[str, ...]], ...],
    speaker_profile=None,
    reply_has_image: bool = False,
) -> VerifiedActionTarget:
    """Verify a target against a live, current-group member snapshot."""

    alias_candidates: list[AliasCandidate] = []
    for raw_profile in group_profiles:
        user_id = str(raw_profile.get("user_id") or "").strip()
        if not user_id:
            continue
        nickname = str(raw_profile.get("nickname") or "").strip()
        group_card = str(
            raw_profile.get("user_name")
            or raw_profile.get("display_name")
            or ""
        ).strip()
        aliases: list[str] = []
        for value in (
            raw_profile.get("display_name"),
            raw_profile.get("nickname"),
            raw_profile.get("user_name"),
        ):
            text = str(value or "").strip()
            if text and text not in aliases:
                aliases.append(text)
        alias_entries = raw_profile.get("alias_entries") or ()
        if isinstance(alias_entries, tuple):
            for entry in alias_entries:
                text = str(getattr(entry, "source", "") or "").strip()
                if text and text not in aliases:
                    aliases.append(text)
        profile = PersonProfile(
            user_id=user_id,
            group_id=str(getattr(event_context, "group_id", "") or "") or None,
            nickname=nickname,
            group_card=group_card,
            aliases=tuple(aliases),
            confidence=0.72,
        )
        alias_candidates.extend(
            AliasCandidate(profile=profile, score=1.0, matched_alias=alias)
            for alias in aliases
            if len(normalize_alias_key(alias)) >= 2
        )

    return resolve_verified_action_target(
        event_context=event_context,
        addressee=addressee,
        speaker_profile=speaker_profile,
        reply_has_image=reply_has_image,
    )


def _has_competing_alias_evidence(
    *,
    event_context,
    alias_candidates,
    excluded_user_ids: set[str],
) -> bool:
    """True when the message literally names a member other than the @ target."""

    message_text = normalize_message_text(
        str(getattr(event_context, "message_text_with_tags", "") or "")
    )
    if not message_text:
        return False
    hits, _ = _literal_alias_hits(
        message_text=message_text,
        alias_candidates=alias_candidates,
        excluded_user_ids={item for item in excluded_user_ids if item},
    )
    return bool(hits)


def _resolve_verified_speaker_alias_target(
    *,
    event_context,
    speaker_profile,
    current_user_id: str,
    bot_id: str,
) -> VerifiedActionTarget:
    if speaker_profile is None or not current_user_id or current_user_id == bot_id:
        return VerifiedActionTarget()
    profile_user_id = str(getattr(speaker_profile, "user_id", "") or "").strip()
    if profile_user_id != current_user_id:
        return VerifiedActionTarget()
    if str(getattr(speaker_profile, "conflict_state", "") or "").strip():
        return VerifiedActionTarget()

    message_text = normalize_message_text(
        str(getattr(event_context, "message_text_with_tags", "") or "")
    ).rstrip(" 　\t\r\n，,。.!！？?：:；;")
    nickname = normalize_message_text(
        str(getattr(event_context, "nickname", "") or "")
    )
    if not message_text or not nickname:
        return VerifiedActionTarget()
    nickname_key = normalize_alias_key(nickname)
    generic_self_keys = {
        normalize_alias_key(value)
        for value in ("我", "自己", "本人", "我的", "我自己", "自己的")
    }
    if not 2 <= len(nickname_key) <= 16 or nickname_key in generic_self_keys:
        return VerifiedActionTarget()
    if not message_text.casefold().endswith(nickname.casefold()):
        return VerifiedActionTarget()
    prefix = message_text[: -len(nickname)]
    if (
        prefix
        and nickname[0].isascii()
        and nickname[0].isalnum()
        and prefix[-1].isascii()
        and prefix[-1].isalnum()
    ):
        return VerifiedActionTarget()
    return VerifiedActionTarget(
        user_id=current_user_id,
        source="self_nickname",
        confidence=0.96,
    )


def _alias_occurrence_neighbors(
    message_text: str,
    alias_key: str,
) -> tuple[tuple[str, str], ...]:
    """(previous raw char, next normalized char) for each alias occurrence."""

    compact: list[str] = []
    origins: list[int] = []
    for index, char in enumerate(message_text):
        normalized = normalize_alias_key(char)
        if not normalized:
            continue
        compact.append(normalized)
        origins.append(index)
    compact_key = "".join(compact)
    if not alias_key or alias_key not in compact_key:
        return ()
    neighbors: list[tuple[str, str]] = []
    start = compact_key.find(alias_key)
    while start >= 0:
        end = start + len(alias_key)
        origin = origins[start]
        previous = message_text[origin - 1] if origin > 0 else ""
        following = compact_key[end] if end < len(compact_key) else ""
        neighbors.append((previous, following))
        start = compact_key.find(alias_key, start + 1)
    return tuple(neighbors)


def _is_cjk_char(char: str) -> bool:
    return bool(char) and "一" <= char <= "鿿"


def alias_has_target_evidence(message_text: str, alias_key: str) -> bool:
    """Reject alias hits that are ambient chatter rather than a real callout."""

    alias_key = normalize_alias_key(alias_key)
    if not alias_key:
        return False
    # A bare alias is a callout, not an execution target instruction.
    if normalize_alias_key(message_text) == alias_key:
        return False
    for previous, following in _alias_occurrence_neighbors(message_text, alias_key):
        # Possessive syntax is explicit evidence that the alias names a target.
        if following == "的":
            return True
        # CJK aliases cannot authorize a prefix inside a longer CJK compound;
        # transitions between writing systems are natural boundaries.
        if _is_cjk_char(following) and any(_is_cjk_char(char) for char in alias_key):
            continue
        # Directive prefixes place the alias in an explicit target slot.
        if previous in _ALIAS_DIRECTIVE_PREFIX_CHARS:
            return True
        # A bounded alias or one at the end of the message is standalone.
        return True
    return False


def _literal_alias_hits(
    *,
    message_text: str,
    alias_candidates,
    excluded_user_ids: set[str],
) -> tuple[list[tuple[int, float, str, str]], bool]:
    """Re-score alias candidates on literal containment in the raw message.

    Returns ``(hits, has_conflicted)`` where each hit is
    ``(alias_key_length, score, user_id, alias_key)``.
    """

    hits: list[tuple[int, float, str, str]] = []
    has_conflicted = False
    for candidate in alias_candidates or ():
        profile = getattr(candidate, "profile", None)
        if profile is None:
            continue
        user_id = str(getattr(profile, "user_id", "") or "").strip()
        if not user_id or user_id in excluded_user_ids:
            continue
        alias_key = normalize_alias_key(
            str(getattr(candidate, "matched_alias", "") or "")
        )
        score = score_alias_in_message(message_text, alias_key)
        if score <= 0.0 or not alias_has_target_evidence(message_text, alias_key):
            continue
        if str(getattr(profile, "conflict_state", "") or "").strip():
            has_conflicted = True
            continue
        hits.append((len(alias_key), score, user_id, alias_key))
    # Prefer the longest literal identity so its shorter fragments do not
    # create artificial ambiguity.
    hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return hits, has_conflicted


def _fuzzy_alias_spans(message_text: str, alias_key: str) -> tuple[str, ...]:
    """Return bounded message spans that may be a misspelled alias."""

    alias_key = normalize_alias_key(alias_key)
    message_key = normalize_alias_key(message_text)
    if len(alias_key) < 3 or not message_key or message_key == alias_key:
        return ()
    alias_has_cjk = any(_is_cjk_char(char) for char in alias_key)
    spans: list[str] = []
    for raw_run in _ALIAS_SCRIPT_RUN_PATTERN.findall(message_text):
        run = normalize_alias_key(raw_run)
        if len(run) < 2:
            continue
        run_has_cjk = any(_is_cjk_char(char) for char in run)
        if run_has_cjk != alias_has_cjk:
            continue
        if not alias_has_cjk:
            spans.append(run)
            continue
        minimum = max(len(alias_key) - 1, 2)
        maximum = min(len(alias_key) + 1, len(run))
        for size in range(minimum, maximum + 1):
            for start in range(0, len(run) - size + 1):
                spans.append(run[start : start + size])
    return tuple(dict.fromkeys(spans))


def _fuzzy_alias_score(message_text: str, alias_key: str) -> float:
    """Score a bounded typo without reviving rejected literal compounds."""

    alias_key = normalize_alias_key(alias_key)
    if not alias_key:
        return 0.0
    if score_alias_in_message(message_text, alias_key) > 0.0:
        return 0.0
    return max(
        (
            score_member_alias(span, alias_key)
            for span in _fuzzy_alias_spans(message_text, alias_key)
        ),
        default=0.0,
    )


def _fuzzy_alias_hits(
    *,
    message_text: str,
    alias_candidates,
    excluded_user_ids: set[str],
) -> tuple[list[tuple[float, str]], bool]:
    """Resolve only a unique, high-confidence approximate member identity."""

    scores_by_user: dict[str, float] = {}
    conflicted = False
    for candidate in alias_candidates or ():
        profile = getattr(candidate, "profile", None)
        if profile is None:
            continue
        user_id = str(getattr(profile, "user_id", "") or "").strip()
        if not user_id or user_id in excluded_user_ids:
            continue
        alias_key = normalize_alias_key(
            str(getattr(candidate, "matched_alias", "") or "")
        )
        try:
            candidate_score = float(getattr(candidate, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            candidate_score = 0.0
        score = min(_fuzzy_alias_score(message_text, alias_key), candidate_score)
        if score < ALIAS_MATCH_THRESHOLD:
            continue
        if str(getattr(profile, "conflict_state", "") or "").strip():
            conflicted = True
            continue
        scores_by_user[user_id] = max(scores_by_user.get(user_id, 0.0), score)
    hits = sorted(
        ((score, user_id) for user_id, score in scores_by_user.items()),
        reverse=True,
    )
    return hits, conflicted


def _resolve_verified_alias_target(
    *,
    event_context,
    alias_candidates,
    current_user_id: str,
    bot_id: str,
) -> VerifiedActionTarget:
    message_text = normalize_message_text(
        str(getattr(event_context, "message_text_with_tags", "") or "")
    )
    if not message_text:
        return VerifiedActionTarget()

    hits, has_conflicted = _literal_alias_hits(
        message_text=message_text,
        alias_candidates=alias_candidates,
        excluded_user_ids={current_user_id, bot_id},
    )
    if not hits:
        fuzzy_hits, fuzzy_conflicted = _fuzzy_alias_hits(
            message_text=message_text,
            alias_candidates=alias_candidates,
            excluded_user_ids={current_user_id, bot_id},
        )
        if fuzzy_conflicted:
            return VerifiedActionTarget(source="alias", ambiguous=True)
        if not fuzzy_hits:
            if has_conflicted:
                return VerifiedActionTarget(source="alias", ambiguous=True)
            return VerifiedActionTarget()
        top_score, top_user_id = fuzzy_hits[0]
        if len(fuzzy_hits) > 1:
            second_score, second_user_id = fuzzy_hits[1]
            if (
                second_user_id != top_user_id
                and (
                    top_score < ALIAS_AMBIGUOUS_TOP
                    or top_score - second_score < ALIAS_AMBIGUOUS_GAP
                )
            ):
                return VerifiedActionTarget(source="alias", ambiguous=True)
        return VerifiedActionTarget(
            user_id=top_user_id,
            source="alias",
            confidence=min(top_score, 0.96),
        )
    if has_conflicted:
        return VerifiedActionTarget(source="alias", ambiguous=True)

    top_length, top_score, top_user_id, _top_alias_key = hits[0]
    if len(hits) > 1:
        second_length, _, second_user_id, _ = hits[1]
        # Only equally strong literal evidence for two different people is a
        # genuine ambiguity.
        if second_length == top_length and second_user_id != top_user_id:
            return VerifiedActionTarget(source="alias", ambiguous=True)
    if top_score < ALIAS_AMBIGUOUS_TOP:
        return VerifiedActionTarget()
    return VerifiedActionTarget(
        user_id=top_user_id,
        source="alias",
        confidence=min(top_score, 0.96),
    )


def _route_command_policy(
    route_result: NativeRouteResult,
    knowledge_plugins,
) -> tuple[object | None, CommandTargetPolicy | None]:
    schema = find_route_command_schema(route_result, knowledge_plugins)
    command_policy = (
        resolve_command_target_policy(schema) if schema is not None else None
    )
    return schema, command_policy


def _schema_image_min(schema: object | None) -> int:
    if schema is None:
        return 0
    try:
        return max(int(getattr(schema, "image_min", 0) or 0), 0)
    except Exception:
        return 0


def _schema_accepts_image_context(schema: object | None) -> bool:
    if schema is None:
        return False
    try:
        policy = resolve_command_target_policy(schema)
        if policy.allow_image_as_target or policy.allow_reply_image_as_target:
            return True
    except Exception:
        pass
    if _schema_image_min(schema) > 0:
        return True
    if not hasattr(schema, "image_max"):
        return False
    image_max = getattr(schema, "image_max", 0)
    if image_max is None:
        return True
    try:
        return int(image_max) > 0
    except Exception:
        return False


def _has_target_or_image_context(*messages: str) -> bool:
    for message in messages:
        if extract_at_tokens(message) or extract_image_tokens(message):
            return True
    return False


def _append_unique_context_tokens(message_text: str, *sources: str) -> str:
    message = normalize_message_text(message_text)
    existing = set(extract_at_tokens(message)) | set(extract_image_tokens(message))
    tokens: list[str] = []
    for source in sources:
        for token in [*extract_at_tokens(source), *extract_image_tokens(source)]:
            if token in existing:
                continue
            existing.add(token)
            tokens.append(token)
    if not tokens:
        return message
    return normalize_message_text(f"{message} {' '.join(tokens)}")


def _explicit_at_ids(message_text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _EXPLICIT_AT_PATTERN.finditer(message_text or ""):
        user_id = (match.group("bracket") or match.group("plain") or "").strip()
        if user_id and user_id not in values:
            values.append(user_id)
    return tuple(values)


def _canonicalize_explicit_at_tokens(message_text: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        user_id = (match.group("bracket") or match.group("plain") or "").strip()
        return f"[@{user_id}]" if user_id else match.group(0)

    return normalize_message_text(_EXPLICIT_AT_PATTERN.sub(replace_match, message_text))


def _remove_explicit_at_ids(
    message_text: str,
    excluded_ids: set[str],
) -> str:
    if not excluded_ids:
        return normalize_message_text(message_text)

    def replace_match(match: re.Match[str]) -> str:
        user_id = (match.group("bracket") or match.group("plain") or "").strip()
        return "" if user_id in excluded_ids else match.group(0)

    return normalize_message_text(_EXPLICIT_AT_PATTERN.sub(replace_match, message_text))


def _remove_context_placeholders(message_text: str) -> str:
    text = _remove_explicit_at_ids(
        message_text,
        set(_explicit_at_ids(message_text)),
    )
    for token in extract_image_tokens(text):
        text = text.replace(token, "")
    return normalize_message_text(text)


def _identity_key(value: object) -> str:
    return "".join(
        character.casefold() for character in str(value or "") if character.isalnum()
    )


def _target_label_key(target_hint: str) -> str:
    without_target = _EXPLICIT_AT_PATTERN.sub("", target_hint or "")
    for token in extract_image_tokens(without_target):
        without_target = without_target.replace(token, "")
    return _identity_key(without_target)


def _profile_identity_keys(
    user_id: str,
    profile: dict[str, str],
) -> set[str]:
    return {
        key
        for key in (
            _identity_key(user_id),
            _identity_key(profile.get("display_name")),
            _identity_key(profile.get("nickname")),
            _identity_key(profile.get("user_name")),
            _identity_key(profile.get("alias_key")),
            _identity_key(profile.get("uid")),
        )
        if key
    }


def _conflicting_target_sources(*sources: tuple[str, ...]) -> bool:
    distinct = [frozenset(source) for source in sources if source]
    return bool(distinct) and any(source != distinct[0] for source in distinct[1:])


async def _resolve_verified_explicit_targets(
    *,
    group_id: str | None,
    bot_id: str | None,
    task_message: str,
    ambient_message: str,
    target_hint: str,
    mention_profiles: dict[str, dict[str, str]],
    trusted_target_ids: tuple[str, ...] = (),
) -> TargetResolveResult | None:
    hint_ids = _explicit_at_ids(target_hint)
    task_ids = _explicit_at_ids(task_message)
    ambient_ids = _explicit_at_ids(ambient_message)
    normalized_bot_id = str(bot_id or "").strip()
    excluded_ids: set[str] = set()
    if normalized_bot_id and normalized_bot_id not in hint_ids:
        excluded_ids.add(normalized_bot_id)
        task_ids = tuple(
            user_id for user_id in task_ids if user_id != normalized_bot_id
        )
        ambient_ids = tuple(
            user_id for user_id in ambient_ids if user_id != normalized_bot_id
        )
    if not hint_ids and not task_ids and not ambient_ids:
        return None

    if _conflicting_target_sources(hint_ids, task_ids, ambient_ids):
        return TargetResolveResult(
            status="ambiguous",
            message_text=task_message,
            mention_profiles=mention_profiles,
            target_hint=target_hint,
        )

    target_ids = tuple(dict.fromkeys((*hint_ids, *task_ids, *ambient_ids)))
    trusted_ids = {
        str(user_id).strip()
        for user_id in trusted_target_ids
        if str(user_id).strip() in target_ids
    }
    verified_profiles = {
        str(user_id).strip(): dict(profile)
        for user_id, profile in mention_profiles.items()
        if str(user_id).strip() and isinstance(profile, dict)
    }
    for user_id in trusted_ids:
        verified_profiles.setdefault(user_id, {"user_id": user_id, "uid": user_id})
    unresolved_ids = [
        user_id for user_id in target_ids if user_id not in verified_profiles
    ]
    if unresolved_ids and group_id:
        local_profiles = await build_mention_profiles(
            group_id,
            " ".join(f"[@{user_id}]" for user_id in unresolved_ids),
            bot_id=bot_id,
            bot=None,
        )
        verified_profiles.update(local_profiles)
    if any(user_id not in verified_profiles for user_id in target_ids):
        return TargetResolveResult(
            status="invalid",
            message_text=task_message,
            mention_profiles=verified_profiles,
            target_hint=target_hint,
        )

    label_key = _target_label_key(target_hint)
    if label_key:
        if len(hint_ids) != 1:
            return TargetResolveResult(
                status="ambiguous",
                message_text=task_message,
                mention_profiles=verified_profiles,
                target_hint=target_hint,
            )
        hinted_id = hint_ids[0]
        if label_key not in _profile_identity_keys(
            hinted_id,
            verified_profiles[hinted_id],
        ):
            return TargetResolveResult(
                status="invalid",
                message_text=task_message,
                mention_profiles=verified_profiles,
                target_hint=target_hint,
            )

    canonical_hint = _canonicalize_explicit_at_tokens(target_hint)
    canonical_task = _canonicalize_explicit_at_tokens(
        _remove_explicit_at_ids(task_message, excluded_ids)
    )
    canonical_ambient = _canonicalize_explicit_at_tokens(
        _remove_explicit_at_ids(ambient_message, excluded_ids)
    )
    enriched_message = _append_unique_context_tokens(
        canonical_task,
        canonical_hint,
        canonical_ambient,
    )
    return TargetResolveResult(
        status=(
            "resolved"
            if enriched_message != normalize_message_text(task_message)
            else "present"
        ),
        message_text=enriched_message,
        mention_profiles=verified_profiles,
        target_hint=canonical_hint,
        resolved_target_ids=target_ids,
    )


def _command_can_use_target(
    *,
    schema: object | None,
    command_policy: CommandTargetPolicy | None,
) -> bool:
    if command_policy is not None:
        if (
            command_policy.allow_at_as_target
            or command_policy.allow_image_as_target
            or command_policy.allow_reply_image_as_target
            or command_policy.media_related
            or command_policy.target_requirement != "none"
        ):
            return True
    return _schema_accepts_image_context(schema)


async def resolve_execution_target(
    *,
    group_id: str | None,
    bot: Bot | None = None,
    bot_id: str | None = None,
    route_result: NativeRouteResult,
    knowledge_plugins,
    task_message: str,
    ambient_message: str,
    target_hint: str = "",
    trusted_target_ids: tuple[str, ...] = (),
    mention_profiles: dict[str, dict[str, str]] | None = None,
    use_ambient_target_context: bool = False,
) -> TargetResolveResult:
    schema, command_policy = _route_command_policy(
        route_result,
        knowledge_plugins,
    )
    mention_profiles = dict(mention_profiles or {})
    if not _command_can_use_target(
        schema=schema,
        command_policy=command_policy,
    ):
        return TargetResolveResult(
            status="not_needed",
            message_text=task_message,
            mention_profiles=mention_profiles,
        )

    explicit_target_hint = normalize_message_text(target_hint)
    allow_at_target = bool(
        command_policy is not None and command_policy.allow_at_as_target
    )
    image_target_allowed = bool(
        _schema_accepts_image_context(schema)
        or (
            command_policy is not None
            and (
                command_policy.allow_image_as_target
                or command_policy.allow_reply_image_as_target
            )
        )
    )
    excluded_bot_ids = (
        {str(bot_id).strip()}
        if bot_id
        and str(bot_id).strip()
        and str(bot_id).strip() not in _explicit_at_ids(explicit_target_hint)
        else set()
    )
    target_task_message = _remove_explicit_at_ids(
        task_message,
        excluded_bot_ids,
    )
    if use_ambient_target_context:
        target_task_message = _remove_context_placeholders(target_task_message)
    target_ambient_message = _remove_explicit_at_ids(
        ambient_message,
        excluded_bot_ids,
    )
    if allow_at_target:
        explicit_resolution = await _resolve_verified_explicit_targets(
            group_id=group_id,
            bot_id=bot_id,
            task_message=target_task_message,
            ambient_message=target_ambient_message,
            target_hint=explicit_target_hint,
            mention_profiles=mention_profiles,
            trusted_target_ids=trusted_target_ids,
        )
        if explicit_resolution is not None:
            return explicit_resolution
    if image_target_allowed and extract_image_tokens(explicit_target_hint):
        image_context = " ".join(extract_image_tokens(explicit_target_hint))
        enriched_message = _append_unique_context_tokens(
            target_task_message,
            image_context,
        )
        return TargetResolveResult(
            status=(
                "resolved" if enriched_message != target_task_message else "present"
            ),
            message_text=enriched_message,
            mention_profiles=mention_profiles,
            target_hint=explicit_target_hint,
        )

    if image_target_allowed and any(
        extract_image_tokens(message)
        for message in (target_task_message, target_ambient_message)
    ):
        image_context = " ".join(
            dict.fromkeys(
                [
                    *extract_image_tokens(target_task_message),
                    *extract_image_tokens(target_ambient_message),
                ]
            )
        )
        return TargetResolveResult(
            status="present",
            message_text=_append_unique_context_tokens(
                target_task_message,
                image_context,
            ),
            mention_profiles=mention_profiles,
            target_hint=explicit_target_hint,
        )

    target_required = (
        command_policy is not None and command_policy.target_requirement == "required"
    )
    if target_required:
        return TargetResolveResult(
            status="missing",
            message_text=target_task_message,
            mention_profiles=mention_profiles,
            target_hint=explicit_target_hint,
        )

    return TargetResolveResult(
        status="not_needed",
        message_text=target_task_message,
        mention_profiles=mention_profiles,
        target_hint=explicit_target_hint,
    )


__all__ = [
    "TargetResolveResult",
    "VerifiedActionTarget",
    "alias_has_target_evidence",
    "resolve_execution_target",
    "resolve_verified_action_target",
    "resolve_verified_action_target_from_group_profiles",
]
