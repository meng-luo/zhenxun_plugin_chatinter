from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, cast

from nonebot.adapters import Bot, Event, Message
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_uninfo import Uninfo

from .route_text import normalize_message_text
from .utils.unimsg_utils import (
    extract_reply_from_message,
    remove_reply_segment,
    uni_to_text_with_tags,
)

_AT_ID_TOKEN_PATTERN = re.compile(
    r"\[@([^\]\s]+)\]|(?<![0-9A-Za-z_])@(\d{5,20})(?=(?:\s|$|[的，,。.!！？?]))"
)
_IMAGE_TOKEN_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)


@dataclass(frozen=True)
class MentionTarget:
    user_id: str
    raw_token: str


@dataclass(frozen=True)
class ReplyContext:
    message_id: str | None = None
    sender_id: str | None = None
    text: str = ""


@dataclass(frozen=True)
class ImageContext:
    token: str
    source: str = "message"


@dataclass(frozen=True)
class ChatInterEventContext:
    adapter: str
    bot_id: str | None
    event_id: str | None
    user_id: str
    group_id: str | None
    nickname: str
    plain_text: str
    normalized_text: str
    message_text_with_tags: str
    mentions: list[MentionTarget] = field(default_factory=list)
    reply: ReplyContext | None = None
    images: list[ImageContext] = field(default_factory=list)
    is_private: bool = False
    is_to_me: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def has_reply(self) -> bool:
        return self.reply is not None and bool(
            self.reply.message_id or self.reply.sender_id or self.reply.text
        )

    @property
    def mentioned_user_ids(self) -> list[str]:
        return [item.user_id for item in self.mentions]


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


def _event_is_to_me(event: Event, bot_id: str | None, message_text: str) -> bool:
    if bool(getattr(event, "to_me", False)):
        return True
    if bot_id and f"[@{bot_id}]" in message_text:
        return True
    return False


def _extract_mentions(message_text: str) -> list[MentionTarget]:
    mentions: list[MentionTarget] = []
    seen: set[str] = set()
    for match in _AT_ID_TOKEN_PATTERN.finditer(message_text or ""):
        user_id = str(match.group(1) or match.group(2) or "").strip()
        if not user_id or user_id in {"所有人", "all"} or user_id in seen:
            continue
        seen.add(user_id)
        mentions.append(MentionTarget(user_id=user_id, raw_token=f"[@{user_id}]"))
    return mentions


def _extract_images(message_text: str) -> list[ImageContext]:
    images: list[ImageContext] = []
    seen: set[str] = set()
    for match in _IMAGE_TOKEN_PATTERN.finditer(message_text or ""):
        token = match.group(0)
        if token in seen:
            continue
        seen.add(token)
        images.append(ImageContext(token=token))
    return images


def _extract_reply_context(
    event: Event,
    raw_message: str | UniMessage | None,
) -> ReplyContext | None:
    reply_id: str | None = None
    if isinstance(raw_message, UniMessage):
        reply_id = extract_reply_from_message(raw_message)

    reply = getattr(event, "reply", None)
    sender_id: str | None = None
    reply_text = ""
    if reply is not None:
        if not reply_id:
            reply_id = (
                str(
                    getattr(reply, "message_id", "") or getattr(reply, "id", "") or ""
                ).strip()
                or None
            )
        sender = getattr(reply, "sender", None)
        if sender is None and isinstance(reply, dict):
            sender = reply.get("sender")
        if isinstance(sender, dict):
            raw_sender_id = sender.get("user_id")
        else:
            raw_sender_id = getattr(sender, "user_id", None)
        if raw_sender_id is not None:
            sender_id = str(raw_sender_id).strip() or None
        raw_reply_message: Any = None
        if isinstance(reply, dict):
            raw_reply_message = reply.get("message") or reply.get("raw_message")
        else:
            raw_reply_message = getattr(reply, "message", None) or getattr(
                reply, "raw_message", None
            )
        if raw_reply_message is not None:
            try:
                reply_text = uni_to_text_with_tags(raw_reply_message)
            except Exception:
                reply_text = str(raw_reply_message or "")

    if not reply_id and not sender_id and not reply_text:
        return None
    return ReplyContext(message_id=reply_id, sender_id=sender_id, text=reply_text)


def build_event_context(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    nickname: str,
    event_message: object | None = None,
    uni_msg: UniMessage | None = None,
    cached_plain_text: str | None = None,
) -> ChatInterEventContext:
    bot_id = str(bot.self_id) if hasattr(bot, "self_id") else None
    user_id = str(session.user.id)
    group_id = str(session.group.id) if session.group else None

    if event_message is not None:
        message_text = uni_to_text_with_tags(
            cast(UniMessage | Message | str, event_message)
        )
    elif uni_msg is not None:
        message_text = uni_to_text_with_tags(remove_reply_segment(uni_msg))
    elif cached_plain_text:
        message_text = cached_plain_text.strip()
    else:
        message_text = str(raw_message or "").strip()

    plain_text = normalize_message_text(message_text)
    normalized_text = normalize_message_text(message_text)
    mentions = _extract_mentions(message_text)
    reply = _extract_reply_context(event, uni_msg or raw_message)
    images = _extract_images(message_text)

    event_id = (
        str(
            getattr(event, "message_id", "")
            or getattr(event, "event_id", "")
            or getattr(event, "id", "")
            or ""
        ).strip()
        or None
    )

    return ChatInterEventContext(
        adapter=_event_adapter_name(bot),
        bot_id=bot_id,
        event_id=event_id,
        user_id=user_id,
        group_id=group_id,
        nickname=nickname,
        plain_text=plain_text,
        normalized_text=normalized_text,
        message_text_with_tags=message_text,
        mentions=mentions,
        reply=reply,
        images=images,
        is_private=_event_is_private(event),
        is_to_me=_event_is_to_me(event, bot_id, message_text),
    )


__all__ = [
    "ChatInterEventContext",
    "ImageContext",
    "MentionTarget",
    "ReplyContext",
    "build_event_context",
]
