from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from zhenxun.configs.config import BotConfig
from zhenxun.models.chat_history import ChatHistory
from zhenxun.services.llm import LLMMessage

from .models.chat_history import ChatInterChatHistory
from .person_registry import format_person_history_label, get_person_profile
from .route_text import normalize_message_text
from .utils.unimsg_utils import uni_to_text_with_tags

_HISTORY_MESSAGE_CLIP = 220
_CHATROOM_LINE_CLIP = 180


@dataclass(frozen=True)
class AstrHistoryPayload:
    """Astr-like history package for one LLM request.

    The policy is intentionally simple: recent conversation turns are supplied as
    normal role messages, while noisy platform chatroom history is a compact
    chronological context block. This replaces the old selector/XML recall path.
    """

    messages: list[LLMMessage]
    chatroom_lines: list[str]


async def build_astr_history_payload(
    *,
    session_id: str,
    user_id: str,
    group_id: str | None,
    bot_id: str | None,
    current_message_text: str,
    dialog_limit: int,
    chatroom_limit: int,
) -> AstrHistoryPayload:
    dialog_messages = await _build_dialog_messages(
        session_id=session_id,
        group_id=group_id,
        dialog_limit=dialog_limit,
    )
    chatroom_lines = await _build_chatroom_lines(
        user_id=user_id,
        group_id=group_id,
        bot_id=bot_id,
        current_message_text=current_message_text,
        chatroom_limit=chatroom_limit,
    )
    return AstrHistoryPayload(messages=dialog_messages, chatroom_lines=chatroom_lines)


async def _build_dialog_messages(
    *,
    session_id: str,
    group_id: str | None,
    dialog_limit: int,
) -> list[LLMMessage]:
    limit = max(int(dialog_limit or 0), 0)
    if limit <= 0:
        return []

    dialogs = await ChatInterChatHistory.get_recent_dialogs(session_id, limit)
    messages: list[LLMMessage] = []
    for dialog in dialogs:
        user_text = _clean_history_text(dialog.user_message, _HISTORY_MESSAGE_CLIP)
        if user_text:
            sender = await _format_sender(
                user_id=str(dialog.user_id or ""),
                group_id=group_id,
                fallback_name=str(dialog.nickname or ""),
                bot_id=None,
            )
            messages.append(LLMMessage.user(f"{sender}: {user_text}"))

        assistant_text = _clean_history_text(
            dialog.ai_response or "",
            _HISTORY_MESSAGE_CLIP,
        )
        if assistant_text:
            messages.append(LLMMessage.assistant_text_response(assistant_text))
    return messages


async def _build_chatroom_lines(
    *,
    user_id: str,
    group_id: str | None,
    bot_id: str | None,
    current_message_text: str,
    chatroom_limit: int,
) -> list[str]:
    limit = max(int(chatroom_limit or 0), 0)
    if limit <= 0 or not group_id:
        return []

    rows = (
        await ChatHistory.filter(group_id=group_id)
        .order_by("-create_time", "-id")
        .limit(limit + 3)
    )
    current_normalized = _normalize_for_compare(current_message_text)
    selected = []
    for row in reversed(rows):
        content = _clean_history_text(
            row.plain_text or row.text or "",
            _CHATROOM_LINE_CLIP,
        )
        if not content:
            continue
        if (
            str(row.user_id or "") == str(user_id or "")
            and _normalize_for_compare(content) == current_normalized
        ):
            continue
        selected.append((row, content))
    if len(selected) > limit:
        selected = selected[-limit:]

    lines: list[str] = []
    for row, content in selected:
        timestamp = (
            row.create_time.strftime("%m-%d %H:%M:%S")
            if row.create_time
            else "??:??:??"
        )
        row_user_id = str(row.user_id or "")
        is_bot_message = bool(bot_id and row_user_id == str(bot_id))
        sender = await _format_sender(
            user_id=row_user_id,
            group_id=group_id,
            fallback_name="",
            bot_id=bot_id if is_bot_message else None,
        )
        lines.append(f"[{timestamp}] {sender}: {content}")
    return lines


async def _format_sender(
    *,
    user_id: str,
    group_id: str | None,
    fallback_name: str,
    bot_id: str | None,
) -> str:
    if bot_id and user_id == str(bot_id):
        return f"[name={BotConfig.self_nickname}; user_id={user_id}]"
    if not group_id:
        name = normalize_message_text(fallback_name) or user_id
        return f"[name={name}; user_id={user_id}]"
    profile = await get_person_profile(
        user_id=user_id,
        group_id=group_id,
        fallback_name=fallback_name,
    )
    return format_person_history_label(profile, fallback_name=fallback_name)


def append_chatroom_history_context(
    lines: list[str],
    chatroom_lines: Iterable[str],
) -> None:
    materialized = [line for line in chatroom_lines if str(line or "").strip()]
    if not materialized:
        return
    lines.append("<chatroom_history>")
    lines.append("policy=recent_chronological_platform_messages")
    lines.extend(materialized)
    lines.append("</chatroom_history>")


def _clean_history_text(value: object, limit: int) -> str:
    text = uni_to_text_with_tags(str(value or ""))
    text = _strip_channel_markers(text)
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[: max(24, limit - 1)].rstrip()}…"


def _strip_channel_markers(text: str) -> str:
    normalized = str(text or "")
    if not normalized:
        return ""
    for marker in ("[analysis]", "[commentary]", "analysis:", "commentary:"):
        normalized = normalized.replace(marker, "")
    return normalized.strip()


def _normalize_for_compare(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


__all__ = [
    "AstrHistoryPayload",
    "append_chatroom_history_context",
    "build_astr_history_payload",
]
