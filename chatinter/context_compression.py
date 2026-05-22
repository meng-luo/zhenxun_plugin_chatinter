"""Turn-local context compression for ChatInter AgentRuntime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from zhenxun.services.llm import LLMContentPart, LLMMessage

from .artifact_store import get_artifact_store, summarize_artifact_text
from .route_text import normalize_message_text
from .turn_runtime import estimate_text_tokens

_MAX_PROMPT_TOKENS = 4500
_TARGET_PROMPT_TOKENS = 3000
_MAX_SUMMARY_LINES = 24
_MAX_SUMMARY_CHARS = 2600
_TOOL_ARGUMENT_LIMIT = 360
_TOOL_RESULT_LIMIT = 520


@dataclass(frozen=True)
class ContextCompressionResult:
    messages: list[LLMMessage]
    changed: bool
    before_tokens: int
    after_tokens: int
    compressed_tool_pairs: int = 0
    summarized_messages: int = 0
    summary: str = ""


def compress_agent_messages(
    messages: list[LLMMessage],
    *,
    trace_id: str,
    max_prompt_tokens: int = _MAX_PROMPT_TOKENS,
    target_prompt_tokens: int = _TARGET_PROMPT_TOKENS,
) -> ContextCompressionResult:
    before = estimate_messages_tokens(messages)
    working, pair_count = _compress_completed_tool_pairs(messages, trace_id=trace_id)
    after_pairs = estimate_messages_tokens(working)
    summarized_count = 0
    summary = ""
    if after_pairs > max_prompt_tokens:
        working, summarized_count, summary = _summarize_middle_messages(
            working,
            trace_id=trace_id,
            target_prompt_tokens=target_prompt_tokens,
        )
    after = estimate_messages_tokens(working)
    changed = pair_count > 0 or summarized_count > 0 or after < before
    return ContextCompressionResult(
        messages=working,
        changed=changed,
        before_tokens=before,
        after_tokens=after,
        compressed_tool_pairs=pair_count,
        summarized_messages=summarized_count,
        summary=summary
        or f"compressed_tool_pairs={pair_count}; tokens {before}->{after}",
    )


def estimate_messages_tokens(messages: list[LLMMessage]) -> int:
    total = 0
    for message in messages:
        total += _message_token_cost(message)
    return total


def _compress_completed_tool_pairs(
    messages: list[LLMMessage],
    *,
    trace_id: str,
) -> tuple[list[LLMMessage], int]:
    compressed: list[LLMMessage] = []
    count = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = list(message.tool_calls or []) if message.role == "assistant" else []
        if not tool_calls:
            compressed.append(message)
            index += 1
            continue
        tool_ids = {str(getattr(item, "id", "") or "") for item in tool_calls}
        cursor = index + 1
        tool_messages: list[LLMMessage] = []
        while cursor < len(messages):
            next_message = messages[cursor]
            if next_message.role != "tool":
                break
            if str(next_message.tool_call_id or "") not in tool_ids:
                break
            tool_messages.append(next_message)
            cursor += 1
        answered_ids = {str(item.tool_call_id or "") for item in tool_messages}
        if tool_ids and tool_ids <= answered_ids:
            compressed.append(
                LLMMessage.assistant_text_response(
                    _tool_pair_summary(
                        tool_calls=tool_calls,
                        tool_messages=tool_messages,
                        trace_id=trace_id,
                    )
                )
            )
            count += 1
            index = cursor
            continue
        compressed.append(message)
        index += 1
    return compressed, count


def _tool_pair_summary(
    *,
    tool_calls: list[Any],
    tool_messages: list[LLMMessage],
    trace_id: str,
) -> str:
    lines = ["<tool_observation_summary>"]
    response_by_id = {str(message.tool_call_id or ""): message for message in tool_messages}
    for tool_call in tool_calls:
        tool_id = str(getattr(tool_call, "id", "") or "")
        function = getattr(tool_call, "function", None)
        name = normalize_message_text(str(getattr(function, "name", "") or ""))
        arguments = str(getattr(function, "arguments", "") or "")
        argument_summary = _artifact_or_summary(
            arguments,
            trace_id=trace_id,
            source=f"tool_call:{name}:arguments",
            limit=_TOOL_ARGUMENT_LIMIT,
        )
        result_summary = ""
        response = response_by_id.get(tool_id)
        if response is not None:
            result_summary = _artifact_or_summary(
                _message_text(response),
                trace_id=trace_id,
                source=f"tool_call:{name}:result",
                limit=_TOOL_RESULT_LIMIT,
            )
        line = f"tool={name or 'unknown'}; args={argument_summary}"
        if result_summary:
            line += f"; result={result_summary}"
        lines.append(line[:900])
    lines.append("</tool_observation_summary>")
    return "\n".join(lines)


def _summarize_middle_messages(
    messages: list[LLMMessage],
    *,
    trace_id: str,
    target_prompt_tokens: int,
) -> tuple[list[LLMMessage], int, str]:
    if len(messages) <= 10:
        return messages, 0, ""
    prefix_end = _prefix_end(messages)
    suffix_count = min(8, max(4, len(messages) // 3))
    suffix_start = max(prefix_end, len(messages) - suffix_count)
    while suffix_start > prefix_end and messages[suffix_start].role == "tool":
        suffix_start -= 1
    middle = messages[prefix_end:suffix_start]
    if not middle:
        return messages, 0, ""
    summary_lines = [
        line
        for line in (_message_summary(item, trace_id=trace_id) for item in middle)
        if line
    ][-_MAX_SUMMARY_LINES:]
    if not summary_lines:
        return messages, 0, ""
    omitted_ref = get_artifact_store().store_text(
        "\n".join(_message_full_text(item) for item in middle),
        artifact_type="text",
        trace_id=trace_id,
        source="context_compression:omitted_messages",
        force_file=True,
    )
    artifact_line = ""
    if omitted_ref is not None:
        artifact_line = f"artifact_id={omitted_ref.artifact_id}; summary={omitted_ref.summary}"
    summary_text = "\n".join(
        [
            "<compressed_turn_context>",
            "policy=tool_pairs_and_old_messages_summarized_artifacts_hold_full_text",
            artifact_line,
            *summary_lines,
            "</compressed_turn_context>",
        ]
    )[:_MAX_SUMMARY_CHARS]
    rebuilt = [
        *messages[:prefix_end],
        LLMMessage.user(summary_text),
        *messages[suffix_start:],
    ]
    if estimate_messages_tokens(rebuilt) <= target_prompt_tokens:
        return rebuilt, len(middle), summary_text
    return _hard_clip_messages(
        rebuilt,
        trace_id=trace_id,
        target_prompt_tokens=target_prompt_tokens,
        summarized_count=len(middle),
        summary_text=summary_text,
    )


def _hard_clip_messages(
    messages: list[LLMMessage],
    *,
    trace_id: str,
    target_prompt_tokens: int,
    summarized_count: int,
    summary_text: str,
) -> tuple[list[LLMMessage], int, str]:
    prefix_end = _prefix_end(messages)
    kept_suffix: list[LLMMessage] = []
    used = sum(_message_token_cost(message) for message in messages[:prefix_end])
    summary_message = LLMMessage.user(summary_text)
    used += _message_token_cost(summary_message)
    for message in reversed(messages[prefix_end + 1 :]):
        cost = _message_token_cost(message)
        if kept_suffix and used + cost > target_prompt_tokens:
            break
        kept_suffix.append(message)
        used += cost
    kept_suffix.reverse()
    clipped_count = max(len(messages) - prefix_end - 1 - len(kept_suffix), 0)
    if clipped_count <= 0:
        return messages, summarized_count, summary_text
    clip_ref = get_artifact_store().store_text(
        "\n".join(_message_full_text(item) for item in messages[prefix_end + 1 : -len(kept_suffix) or None]),
        artifact_type="text",
        trace_id=trace_id,
        source="context_compression:hard_clip",
        force_file=True,
    )
    clip_line = ""
    if clip_ref is not None:
        clip_line = f"\nclipped_artifact_id={clip_ref.artifact_id}; clipped_summary={clip_ref.summary}"
    clipped_summary = summary_text.replace(
        "</compressed_turn_context>",
        f"hard_clipped_messages={clipped_count}{clip_line}\n</compressed_turn_context>",
    )
    return [*messages[:prefix_end], LLMMessage.user(clipped_summary), *kept_suffix], summarized_count + clipped_count, clipped_summary


def _prefix_end(messages: list[LLMMessage]) -> int:
    if not messages:
        return 0
    index = 1 if messages[0].role == "system" else 0
    while index < len(messages):
        message = messages[index]
        if message.role != "user":
            break
        text = normalize_message_text(_message_text(message))
        if not _is_static_context_message(text):
            break
        index += 1
    return index


def _is_static_context_message(text: str) -> bool:
    if not text:
        return False
    prefixes = (
        "<qq_context>",
        "<event_context>",
        "<turn_identity>",
        "<chatroom_history>",
        "<dialogue_state>",
        "<memory",
        "<capability",
        "<compressed_history_summary>",
    )
    return any(text.startswith(prefix) for prefix in prefixes)


def _message_summary(message: LLMMessage, *, trace_id: str) -> str:
    role = normalize_message_text(message.role)
    if message.tool_calls:
        names = []
        for call in message.tool_calls:
            function = getattr(call, "function", None)
            names.append(normalize_message_text(str(getattr(function, "name", "") or "")))
        return f"{role}: tool_calls={','.join(name for name in names if name)}"
    text = _message_text(message)
    if len(text) > 500:
        text = _artifact_or_summary(
            text,
            trace_id=trace_id,
            source=f"context_message:{role}",
            limit=220,
        )
    else:
        text = summarize_artifact_text(text, limit=220)
    if not text:
        return ""
    return f"{role}: {text}"


def _message_full_text(message: LLMMessage) -> str:
    payload = {
        "role": message.role,
        "content": _message_text(message),
        "name": message.name,
        "tool_call_id": message.tool_call_id,
    }
    if message.tool_calls:
        payload["tool_calls"] = [str(call) for call in message.tool_calls]
    return json.dumps(payload, ensure_ascii=False, default=str)


def _message_text(message: LLMMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if not isinstance(part, LLMContentPart):
            parts.append(str(part))
            continue
        if part.text:
            parts.append(part.text)
        if part.thought_text:
            parts.append(part.thought_text)
        if part.image_source:
            ref = get_artifact_store().store_reference(
                artifact_type="image",
                summary="image omitted from compressed context",
                source="context_compression:image",
                path="" if part.image_source.startswith("data:") else part.image_source[:500],
                mime_type=part.mime_type or "",
                size=len(part.image_source),
            )
            parts.append(f"[image_artifact:{ref.artifact_id}] {ref.summary}")
    return "\n".join(parts)


def _artifact_or_summary(
    text: str,
    *,
    trace_id: str,
    source: str,
    limit: int,
) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return normalize_message_text(raw)
    ref = get_artifact_store().store_text(
        raw,
        artifact_type="text",
        trace_id=trace_id,
        source=source,
        force_file=True,
    )
    if ref is None:
        return summarize_artifact_text(raw, limit=limit)
    return f"[artifact:{ref.artifact_id}] {ref.summary}"


def _message_token_cost(message: LLMMessage) -> int:
    content = message.content
    tool_cost = 0
    if message.tool_calls:
        for call in message.tool_calls:
            function = getattr(call, "function", None)
            tool_cost += estimate_text_tokens(str(getattr(function, "name", "") or ""))
            tool_cost += estimate_text_tokens(str(getattr(function, "arguments", "") or ""))
    if isinstance(content, str):
        return max(estimate_text_tokens(content) + tool_cost, 1)
    total = 0
    for part in content:
        if not isinstance(part, LLMContentPart):
            total += estimate_text_tokens(str(part))
            continue
        total += estimate_text_tokens(part.text or part.thought_text or "")
        if part.image_source:
            total += 48
    return max(total + tool_cost, 1)


__all__ = [
    "ContextCompressionResult",
    "compress_agent_messages",
    "estimate_messages_tokens",
]
