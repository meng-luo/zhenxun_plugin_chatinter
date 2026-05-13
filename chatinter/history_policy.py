from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

from zhenxun.configs.config import BotConfig
from zhenxun.models.chat_history import ChatHistory
from zhenxun.services.llm import LLMMessage

from .models.chat_history import ChatInterChatHistory
from .person_registry import format_person_history_label, get_person_profile
from .route_text import normalize_message_text
from .turn_runtime import estimate_text_tokens
from .utils.unimsg_utils import uni_to_text_with_tags

_HISTORY_MESSAGE_CLIP = 220
_CHATROOM_LINE_CLIP = 180
_TURN_HISTORY_TOKEN_BUDGET = 3200
_MIN_RECENT_TURNS = 3
_SUMMARY_FETCH_LIMIT = 24
_SUMMARY_MAX_LINES = 10
_SUMMARY_USER_CLIP = 96
_SUMMARY_RESULT_CLIP = 128


@dataclass(frozen=True)
class AstrHistoryPayload:
    """Astr-like history package for one LLM request.

    The policy is intentionally simple: recent conversation turns are supplied as
    normal role messages, while noisy platform chatroom history is a compact
    chronological context block. This replaces the old selector/XML recall path.
    """

    messages: list[LLMMessage]
    chatroom_lines: list[str]


@dataclass(frozen=True)
class _HistoryTurn:
    messages: list[LLMMessage]
    summary: str
    token_cost: int


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
    dialog_messages = await _build_turn_managed_dialog_messages(
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


async def _build_turn_managed_dialog_messages(
    *,
    session_id: str,
    group_id: str | None,
    dialog_limit: int,
) -> list[LLMMessage]:
    limit = max(int(dialog_limit or 0), 0)
    if limit <= 0:
        return []

    fetch_limit = max(limit, min(_SUMMARY_FETCH_LIMIT, limit + _SUMMARY_MAX_LINES))
    dialogs = await ChatInterChatHistory.get_recent_dialogs(session_id, fetch_limit)
    turns: list[_HistoryTurn] = []
    for dialog in dialogs:
        timeline_messages = await _timeline_to_history_messages(
            dialog,
            group_id=group_id,
        )
        if timeline_messages:
            turns.append(
                _HistoryTurn(
                    messages=timeline_messages,
                    summary=await _timeline_to_summary_line(
                        dialog,
                        group_id=group_id,
                    ),
                    token_cost=sum(
                        _message_token_cost(message) for message in timeline_messages
                    ),
                )
            )
    if not turns:
        return []

    kept_reversed: list[_HistoryTurn] = []
    omitted: list[_HistoryTurn] = []
    used_tokens = 0
    min_recent_turns = min(_MIN_RECENT_TURNS, limit)
    for turn in reversed(turns):
        should_keep = (
            len(kept_reversed) < min_recent_turns
            or (
                len(kept_reversed) < limit
                and used_tokens + turn.token_cost <= _TURN_HISTORY_TOKEN_BUDGET
            )
        )
        if should_keep:
            kept_reversed.append(turn)
            used_tokens += turn.token_cost
        else:
            omitted.append(turn)

    kept = list(reversed(kept_reversed))
    messages: list[LLMMessage] = []
    summary_lines = [
        turn.summary
        for turn in reversed(omitted)
        if turn.summary
    ][-_SUMMARY_MAX_LINES:]
    if summary_lines:
        messages.append(_compressed_summary_message(summary_lines))
    for turn in kept:
        messages.extend(turn.messages)
    return messages


async def _timeline_to_history_messages(
    dialog: ChatInterChatHistory,
    *,
    group_id: str | None,
) -> list[LLMMessage]:
    timeline = dialog.get_timeline()
    if not timeline:
        return []
    messages: list[LLMMessage] = []
    sender = await _format_sender(
        user_id=str(dialog.user_id or ""),
        group_id=group_id,
        fallback_name=str(dialog.nickname or ""),
        bot_id=None,
    )
    for item in timeline:
        role = str(item.get("role", "") or "")
        kind = str(item.get("kind", "") or "")
        content = _clean_history_text(
            _timeline_content(item),
            _HISTORY_MESSAGE_CLIP,
        )
        if role == "user" and kind == "current_user":
            if content:
                messages.append(LLMMessage.user(f"{sender}: {content}"))
            continue
        if kind == "tool_call":
            tool_name = _clean_history_text(item.get("tool_name", ""), 80)
            arguments = _clean_history_text(
                _timeline_metadata_text(item, "arguments"),
                120,
            )
            text = f"[tool_call] {tool_name}"
            if arguments:
                text += f" {arguments}"
            messages.append(LLMMessage.assistant_text_response(text))
            continue
        if kind == "tool_result":
            tool_name = _clean_history_text(item.get("tool_name", ""), 80)
            result_text = content or _clean_history_text(
                _timeline_metadata_text(item, "output"),
                _HISTORY_MESSAGE_CLIP,
            )
            if result_text:
                messages.append(
                    LLMMessage.assistant_text_response(
                        f"[tool_result] {tool_name}: {result_text}"
                    )
                )
            continue
        if role == "assistant" and kind == "final_output" and content:
            messages.append(LLMMessage.assistant_text_response(content))
    return messages


async def _timeline_to_summary_line(
    dialog: ChatInterChatHistory,
    *,
    group_id: str | None,
) -> str:
    timeline = dialog.get_timeline()
    if not timeline:
        return ""
    sender = await _format_sender(
        user_id=str(dialog.user_id or ""),
        group_id=group_id,
        fallback_name=str(dialog.nickname or ""),
        bot_id=None,
    )
    user_text = ""
    final_text = ""
    tool_result_text = ""
    tool_names: list[str] = []
    for item in timeline:
        role = str(item.get("role", "") or "")
        kind = str(item.get("kind", "") or "")
        if role == "user" and kind == "current_user" and not user_text:
            user_text = _clean_history_text(
                _timeline_content(item),
                _SUMMARY_USER_CLIP,
            )
            continue
        if kind == "tool_call":
            tool_name = _clean_history_text(item.get("tool_name", ""), 64)
            if tool_name and tool_name not in tool_names:
                tool_names.append(tool_name)
            continue
        if role == "assistant" and kind == "final_output" and not final_text:
            final_text = _clean_history_text(
                _timeline_content(item),
                _SUMMARY_RESULT_CLIP,
            )
            continue
        if kind == "tool_result" and not tool_result_text:
            tool_result_text = _clean_history_text(
                _timeline_content(item)
                or _timeline_metadata_text(item, "output"),
                _SUMMARY_RESULT_CLIP,
            )

    result_text = final_text or tool_result_text
    if not user_text and not result_text and not tool_names:
        return ""
    parts = [f"speaker={sender}"]
    if user_text:
        parts.append(f"user={user_text}")
    if tool_names:
        parts.append(f"tools={','.join(tool_names[:4])}")
    if result_text:
        parts.append(f"result={result_text}")
    return " | ".join(parts)


def _compressed_summary_message(summary_lines: list[str]) -> LLMMessage:
    lines = [
        "<compressed_history_summary>",
        "policy=older_turns_summarized_recent_turns_kept_verbatim",
        *summary_lines,
        "</compressed_history_summary>",
    ]
    return LLMMessage.user("\n".join(lines))


def _message_token_cost(message: LLMMessage) -> int:
    content = message.content
    if isinstance(content, str):
        return estimate_text_tokens(content)
    total = 0
    for part in content:
        total += estimate_text_tokens(part.text or part.thought_text or "")
        if part.image_source:
            total += 48
    return max(total, 1)


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


def _timeline_content(item: dict) -> str:
    content = item.get("content", "")
    if content:
        return str(content)
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        output = metadata.get("output")
        if isinstance(output, dict):
            outputs = output.get("outputs")
            if isinstance(outputs, list):
                return "\n".join(str(value or "") for value in outputs if value)
            return str(output.get("message", "") or "")
    return ""


def _timeline_metadata_text(item: dict, key: str) -> str:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get(key)
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


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
