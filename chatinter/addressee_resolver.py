from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .event_context import ChatInterEventContext
from .person_registry import (
    AliasCandidate,
    PersonProfile,
    normalize_alias_key,
    resolve_alias_candidates,
)
from .route_text import normalize_message_text

AddresseeSource = Literal[
    "self",
    "at",
    "reply",
    "alias",
    "private",
    "broadcast",
    "unknown",
]


@dataclass(frozen=True)
class AddresseeResult:
    target_user_id: str | None
    target_name: str | None
    source: AddresseeSource
    confidence: float
    ambiguous: bool = False
    reason: str = ""

    @property
    def is_bot_target(self) -> bool:
        return self.source in {"self", "private"}


_SELF_NAME_HINTS = ("小真寻", "真寻", "机器人", "bot")
_BROADCAST_HINTS = ("大家", "全体", "有人", "有没有人", "谁来")
_ALIAS_INTENT_HINTS = (
    "在吗",
    "看看",
    "看下",
    "问问",
    "告诉",
    "给",
    "帮",
    "叫",
    "找",
    "说",
)
_THIRD_PERSON_REPLY_HINTS = ("他", "她", "ta", "对方", "那位", "上面那位", "刚才那位")


async def resolve_addressee(
    *,
    event_context: ChatInterEventContext,
    bot_names: tuple[str, ...] = (),
    mention_profiles: dict[str, dict[str, str]] | None = None,
    speaker_profile: PersonProfile | None = None,
    alias_candidates: list[AliasCandidate] | None = None,
) -> AddresseeResult:
    text = normalize_message_text(event_context.message_text_with_tags)
    bot_id = str(event_context.bot_id or "").strip()
    mention_profiles = mention_profiles or {}

    if event_context.is_private:
        return AddresseeResult(bot_id, None, "private", 1.0, reason="private")

    if bot_id and any(item.user_id == bot_id for item in event_context.mentions):
        return AddresseeResult(bot_id, None, "self", 1.0, reason="mentioned_bot")

    for mention in event_context.mentions:
        if mention.user_id == event_context.user_id:
            continue
        profile = mention_profiles.get(mention.user_id, {})
        name = str(profile.get("display_name") or profile.get("nickname") or "").strip()
        return AddresseeResult(
            mention.user_id,
            name or mention.user_id,
            "at",
            0.95,
            reason="mentioned_user",
        )

    if event_context.reply and event_context.reply.sender_id:
        target_id = str(event_context.reply.sender_id).strip()
        if bot_id and target_id == bot_id:
            return AddresseeResult(bot_id, None, "self", 0.92, reason="reply_to_bot")
        if any(hint in text for hint in _THIRD_PERSON_REPLY_HINTS) or not text:
            return AddresseeResult(
                target_id,
                target_id,
                "reply",
                0.84,
                reason="reply_target",
            )

    normalized_names = tuple(
        normalize_message_text(name).lower()
        for name in bot_names
        if normalize_message_text(name)
    )
    lowered = text.lower()
    if any(name and name in lowered for name in (*normalized_names, *_SELF_NAME_HINTS)):
        return AddresseeResult(bot_id or None, None, "self", 0.78, reason="bot_name")

    alias_result = await _resolve_alias_addressee(
        event_context=event_context,
        message_text=text,
        alias_candidates=alias_candidates,
    )
    if alias_result is not None:
        return alias_result

    if any(hint in text for hint in _BROADCAST_HINTS):
        return AddresseeResult(None, None, "broadcast", 0.45, reason="broadcast_hint")

    if speaker_profile and speaker_profile.display_name:
        # 当前发送者姓名只作为 speaker，不反推 addressee，避免把自述误判为目标。
        pass

    return AddresseeResult(None, None, "unknown", 0.0, reason="no_signal")


async def _resolve_alias_addressee(
    *,
    event_context: ChatInterEventContext,
    message_text: str,
    alias_candidates: list[AliasCandidate] | None,
) -> AddresseeResult | None:
    normalized = normalize_message_text(message_text)
    if not normalized or not event_context.group_id:
        return None
    if not _has_alias_context(normalized):
        return None

    candidates = alias_candidates
    if candidates is None:
        candidates = await resolve_alias_candidates(
            group_id=event_context.group_id,
            text=normalized,
            exclude_user_id=event_context.user_id,
            limit=5,
        )
    if not candidates:
        return None

    top = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        if top.score < 0.92 or (top.score - second.score) < 0.12:
            names = "、".join(item.profile.display_name for item in candidates[:3])
            return AddresseeResult(
                None,
                names or None,
                "alias",
                top.score,
                ambiguous=True,
                reason="alias_ambiguous",
            )
    if top.score < 0.86:
        return None
    return AddresseeResult(
        top.profile.user_id,
        top.profile.display_name,
        "alias",
        min(top.score, 0.96),
        ambiguous=bool(top.profile.conflict_state),
        reason=f"alias:{normalize_alias_key(top.matched_alias)}",
    )


def _has_alias_context(text: str) -> bool:
    if any(hint in text for hint in _ALIAS_INTENT_HINTS):
        return True
    return len(text) <= 16


def format_addressee_xml(result: AddresseeResult) -> list[str]:
    lines = ["<addressee>"]
    lines.append(f"source={result.source}")
    lines.append(f"confidence={result.confidence:.2f}")
    lines.append(f"ambiguous={int(result.ambiguous)}")
    if result.target_user_id:
        lines.append(f"target_user_id={_xml_escape(result.target_user_id)}")
    if result.target_name:
        lines.append(f"target_name={_xml_escape(result.target_name)}")
    if result.reason:
        lines.append(f"reason={_xml_escape(result.reason)}")
    lines.append("</addressee>")
    return lines


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


__all__ = ["AddresseeResult", "format_addressee_xml", "resolve_addressee"]
