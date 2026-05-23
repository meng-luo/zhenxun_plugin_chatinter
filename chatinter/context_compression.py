"""Turn-local context compression for ChatInter AgentRuntime."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2s
import json
import re
from typing import Any, Iterable

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
_PROTECTED_MESSAGE_LIMIT = 900
_MAX_PROTECTED_MESSAGES = 10
_COMPRESSION_MARKER = "<compressed_turn_context"
_TOOL_SUMMARY_MARKER = "<tool_observation_summary"
_SUMMARY_CLOSE = "</compressed_turn_context>"
_IMPORTANT_RESULT_KEYS = (
    "ok",
    "status",
    "command_id",
    "rendered_command",
    "matched_plugin",
    "plugin_module",
    "task_text",
    "messages_sent_summary",
    "error",
    "retryable",
    "need_continue",
    "remaining_task_hint",
    "approval_id",
    "task_id",
    "event_id",
    "artifact_id",
    "run_id",
    "operation_id",
    "eval_id",
    "loop_id",
    "checkpoint_id",
)


@dataclass(frozen=True)
class ContextCompressionResult:
    messages: list[LLMMessage]
    changed: bool
    before_tokens: int
    after_tokens: int
    compressed_tool_pairs: int = 0
    summarized_messages: int = 0
    deduped_messages: int = 0
    pruned_tool_results: int = 0
    protected_messages: int = 0
    compression_fingerprint: str = ""
    skipped_reason: str = ""
    summary: str = ""
    policy_trace: tuple[str, ...] = ()


def compress_agent_messages(
    messages: list[LLMMessage],
    *,
    trace_id: str,
    max_prompt_tokens: int = _MAX_PROMPT_TOKENS,
    target_prompt_tokens: int = _TARGET_PROMPT_TOKENS,
    protected_terms: Iterable[str] | None = None,
) -> ContextCompressionResult:
    from .context_engine import get_context_engine

    return get_context_engine().compress(
        messages,
        trace_id=trace_id,
        max_prompt_tokens=max_prompt_tokens,
        target_prompt_tokens=target_prompt_tokens,
        protected_terms=protected_terms,
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
    protected_terms: tuple[str, ...],
) -> tuple[list[LLMMessage], int, int, int]:
    compressed: list[LLMMessage] = []
    count = 0
    pruned_tool_results = 0
    protected_messages = 0
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
            protected_messages += sum(
                1
                for item in [message, *tool_messages]
                if _message_contains_protected_term(item, protected_terms)
            )
            compressed.append(
                LLMMessage.assistant_text_response(
                    _tool_pair_summary(
                        tool_calls=tool_calls,
                        tool_messages=tool_messages,
                        trace_id=trace_id,
                        protected_terms=protected_terms,
                    )
                )
            )
            count += 1
            pruned_tool_results += len(tool_messages)
            index = cursor
            continue
        compressed.append(message)
        index += 1
    return compressed, count, pruned_tool_results, protected_messages


def _tool_pair_summary(
    *,
    tool_calls: list[Any],
    tool_messages: list[LLMMessage],
    trace_id: str,
    protected_terms: tuple[str, ...],
) -> str:
    pair_fingerprint = _short_hash(
        "\n".join(
            [
                *(str(getattr(call, "id", "")) for call in tool_calls),
                *(_message_text(message) for message in tool_messages),
            ]
        )
    )
    lines = [f"<tool_observation_summary fingerprint={pair_fingerprint}>"]
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
            result_text = _message_text(response)
            result_summary = _tool_result_summary(
                result_text,
                trace_id=trace_id,
                source=f"tool_call:{name}:result",
                limit=_TOOL_RESULT_LIMIT,
                protected_terms=protected_terms,
            )
        line = f"tool={name or 'unknown'}; args={argument_summary}"
        if result_summary:
            line += f"; result={result_summary}"
        protected_refs = _protected_excerpts(
            f"{arguments}\n{result_summary}",
            protected_terms,
        )
        if protected_refs:
            line += f"; protected_refs={protected_refs}"
        lines.append(line[:900])
    lines.append("</tool_observation_summary>")
    return "\n".join(lines)


def _summarize_middle_messages(
    messages: list[LLMMessage],
    *,
    trace_id: str,
    target_prompt_tokens: int,
    protected_terms: tuple[str, ...],
    compression_fingerprint: str,
) -> tuple[list[LLMMessage], int, str, int]:
    if len(messages) <= 10:
        return messages, 0, "", 0
    prefix_end = _prefix_end(messages)
    suffix_count = min(8, max(4, len(messages) // 3))
    suffix_start = max(prefix_end, len(messages) - suffix_count)
    while suffix_start > prefix_end and messages[suffix_start].role == "tool":
        suffix_start -= 1
    middle = messages[prefix_end:suffix_start]
    if not middle:
        return messages, 0, "", 0
    protected_middle: list[LLMMessage] = []
    summarized_middle: list[LLMMessage] = []
    for item in middle:
        if _message_contains_protected_term(item, protected_terms):
            protected_middle.append(
                _compact_protected_message(
                    item,
                    trace_id=trace_id,
                    protected_terms=protected_terms,
                )
            )
            continue
        summarized_middle.append(item)
    if len(protected_middle) > _MAX_PROTECTED_MESSAGES:
        summarized_middle.extend(protected_middle[: -_MAX_PROTECTED_MESSAGES])
        protected_middle = protected_middle[-_MAX_PROTECTED_MESSAGES:]
    summary_lines = [
        line
        for line in (
            _message_summary(
                item,
                trace_id=trace_id,
                protected_terms=protected_terms,
            )
            for item in summarized_middle
        )
        if line
    ][-_MAX_SUMMARY_LINES:]
    if not summary_lines:
        return messages, 0, "", len(protected_middle)
    omitted_ref = get_artifact_store().store_text(
        "\n".join(_message_full_text(item) for item in summarized_middle),
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
            f"<compressed_turn_context fingerprint={compression_fingerprint}>",
            "policy=tool_result_pruning_dedup_active_task_protection",
            artifact_line,
            f"summarized_messages={len(summarized_middle)}; protected_messages_kept={len(protected_middle)}",
            *summary_lines,
            "</compressed_turn_context>",
        ]
    )[:_MAX_SUMMARY_CHARS]
    rebuilt = [
        *messages[:prefix_end],
        LLMMessage.user(summary_text),
        *protected_middle,
        *messages[suffix_start:],
    ]
    if estimate_messages_tokens(rebuilt) <= target_prompt_tokens:
        return rebuilt, len(summarized_middle), summary_text, len(protected_middle)
    return _hard_clip_messages(
        rebuilt,
        trace_id=trace_id,
        target_prompt_tokens=target_prompt_tokens,
        summarized_count=len(middle),
        summary_text=summary_text,
        protected_terms=protected_terms,
        protected_count=len(protected_middle),
    )


def _hard_clip_messages(
    messages: list[LLMMessage],
    *,
    trace_id: str,
    target_prompt_tokens: int,
    summarized_count: int,
    summary_text: str,
    protected_terms: tuple[str, ...],
    protected_count: int,
) -> tuple[list[LLMMessage], int, str, int]:
    prefix_end = _prefix_end(messages)
    used = sum(_message_token_cost(message) for message in messages[:prefix_end])
    summary_message = LLMMessage.user(summary_text)
    used += _message_token_cost(summary_message)
    candidates = list(messages[prefix_end + 1 :])
    keep_indices: set[int] = set()
    for idx, message in enumerate(candidates):
        if _message_contains_protected_term(message, protected_terms):
            keep_indices.add(idx)
    for idx in reversed(range(len(candidates))):
        message = candidates[idx]
        cost = _message_token_cost(message)
        if idx in keep_indices:
            used += cost
            continue
        if keep_indices and used + cost > target_prompt_tokens:
            continue
        if not keep_indices and used + cost > target_prompt_tokens:
            break
        keep_indices.add(idx)
        used += cost
    kept_suffix = [candidates[idx] for idx in sorted(keep_indices)]
    clipped_count = max(len(candidates) - len(kept_suffix), 0)
    if clipped_count <= 0:
        return messages, summarized_count, summary_text, protected_count
    clip_ref = get_artifact_store().store_text(
        "\n".join(
            _message_full_text(item)
            for idx, item in enumerate(candidates)
            if idx not in keep_indices
        ),
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
    return (
        [*messages[:prefix_end], LLMMessage.user(clipped_summary), *kept_suffix],
        summarized_count + clipped_count,
        clipped_summary,
        protected_count,
    )


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
    if index < len(messages) and messages[index].role == "user":
        text = normalize_message_text(_message_text(messages[index]))
        if text and not text.startswith(_COMPRESSION_MARKER):
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


def _message_summary(
    message: LLMMessage,
    *,
    trace_id: str,
    protected_terms: tuple[str, ...],
) -> str:
    role = normalize_message_text(message.role)
    if message.tool_calls:
        names = []
        for call in message.tool_calls:
            function = getattr(call, "function", None)
            names.append(normalize_message_text(str(getattr(function, "name", "") or "")))
        protected_refs = _protected_excerpts(
            _message_full_text(message),
            protected_terms,
        )
        suffix = f"; protected_refs={protected_refs}" if protected_refs else ""
        return f"{role}: tool_calls={','.join(name for name in names if name)}{suffix}"
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
    protected_refs = _protected_excerpts(text, protected_terms)
    if protected_refs and protected_refs not in text:
        text = f"{text}; protected_refs={protected_refs}"
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


def _tool_result_summary(
    text: str,
    *,
    trace_id: str,
    source: str,
    limit: int,
    protected_terms: tuple[str, ...],
) -> str:
    raw = str(text or "")
    payload = _loads_json_object(raw)
    if not isinstance(payload, dict):
        return _artifact_or_summary(
            raw,
            trace_id=trace_id,
            source=source,
            limit=limit,
        )
    compact: dict[str, Any] = {}
    for key in _IMPORTANT_RESULT_KEYS:
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        compact[key] = _compact_json_field(
            value,
            trace_id=trace_id,
            source=f"{source}:{key}",
        )
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list | tuple):
        compact["artifacts"] = [
            _compact_artifact_payload(item)
            for item in artifacts
            if isinstance(item, dict) and item.get("artifact_id")
        ][:8]
    protected_refs = _protected_excerpts(raw, protected_terms)
    if protected_refs:
        compact["protected_refs"] = protected_refs
    raw_for_store = raw if len(raw) > limit else ""
    if raw_for_store:
        ref = get_artifact_store().store_text(
            raw_for_store,
            artifact_type="plugin_output",
            trace_id=trace_id,
            source=source,
            force_file=True,
        )
        if ref is not None:
            compact["full_result_artifact_id"] = ref.artifact_id
            compact["full_result_summary"] = ref.summary
    if not compact:
        return _artifact_or_summary(
            raw,
            trace_id=trace_id,
            source=source,
            limit=limit,
        )
    return summarize_artifact_text(
        json.dumps(compact, ensure_ascii=False, default=str),
        limit=max(limit, 900),
    )


def _compact_json_field(
    value: Any,
    *,
    trace_id: str,
    source: str,
) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if len(value) <= _TOOL_RESULT_LIMIT:
            return normalize_message_text(value)
        ref = get_artifact_store().store_text(
            value,
            artifact_type="plugin_output",
            trace_id=trace_id,
            source=source,
            force_file=True,
        )
        if ref is None:
            return summarize_artifact_text(value, limit=_TOOL_RESULT_LIMIT)
        return {"artifact_id": ref.artifact_id, "summary": ref.summary}
    if isinstance(value, list | tuple):
        return [
            _compact_json_field(item, trace_id=trace_id, source=source)
            for item in list(value)[:8]
        ]
    if isinstance(value, dict):
        return {
            str(key): _compact_json_field(item, trace_id=trace_id, source=source)
            for key, item in list(value.items())[:12]
        }
    return summarize_artifact_text(str(value), limit=_TOOL_RESULT_LIMIT)


def _compact_artifact_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_id": normalize_message_text(str(item.get("artifact_id", "") or "")),
    }
    for key in ("type", "summary", "path", "source"):
        value = normalize_message_text(str(item.get(key, "") or ""))
        if value:
            payload[key] = summarize_artifact_text(value, limit=180)
    if item.get("size") is not None:
        payload["size"] = _safe_int(item.get("size"))
    return payload


def _loads_json_object(text: str) -> Any:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value


def _sanitize_tool_protocol(
    messages: list[LLMMessage],
    *,
    trace_id: str,
    protected_terms: tuple[str, ...],
) -> tuple[list[LLMMessage], int]:
    sanitized: list[LLMMessage] = []
    pruned = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = list(message.tool_calls or []) if message.role == "assistant" else []
        if tool_calls:
            tool_ids = {
                str(getattr(call, "id", "") or "")
                for call in tool_calls
                if str(getattr(call, "id", "") or "")
            }
            cursor = index + 1
            tool_messages: list[LLMMessage] = []
            while cursor < len(messages) and messages[cursor].role == "tool":
                if str(messages[cursor].tool_call_id or "") not in tool_ids:
                    break
                tool_messages.append(messages[cursor])
                cursor += 1
            answered_ids = {str(item.tool_call_id or "") for item in tool_messages}
            if tool_ids and tool_ids <= answered_ids:
                sanitized.append(message)
                sanitized.extend(tool_messages)
                index = cursor
                continue
            pruned += 1 + len(tool_messages)
            sanitized.append(
                LLMMessage.assistant_text_response(
                    _incomplete_tool_call_summary(
                        tool_calls=tool_calls,
                        tool_messages=tool_messages,
                        missing_ids=sorted(tool_ids - answered_ids),
                        trace_id=trace_id,
                        protected_terms=protected_terms,
                    )
                )
            )
            index = cursor
            continue
        if message.role != "tool":
            sanitized.append(message)
            index += 1
            continue
        pruned += 1
        summary = _tool_result_summary(
            _message_text(message),
            trace_id=trace_id,
            source=f"orphan_tool_result:{message.name or 'unknown'}",
            limit=_TOOL_RESULT_LIMIT,
            protected_terms=protected_terms,
        )
        sanitized.append(
            LLMMessage.assistant_text_response(
                "\n".join(
                    [
                        "<orphan_tool_result_pruned>",
                        f"tool={normalize_message_text(str(message.name or 'unknown'))}",
                        f"tool_call_id={normalize_message_text(str(message.tool_call_id or ''))}",
                        f"result={summary}",
                        "</orphan_tool_result_pruned>",
                    ]
                )
            )
        )
        index += 1
    return sanitized, pruned


def _incomplete_tool_call_summary(
    *,
    tool_calls: list[Any],
    tool_messages: list[LLMMessage],
    missing_ids: list[str],
    trace_id: str,
    protected_terms: tuple[str, ...],
) -> str:
    lines = ["<incomplete_tool_call_pruned>"]
    if missing_ids:
        lines.append(f"missing_tool_result_ids={','.join(missing_ids)}")
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        name = normalize_message_text(str(getattr(function, "name", "") or ""))
        arguments = str(getattr(function, "arguments", "") or "")
        protected_refs = _protected_excerpts(arguments, protected_terms)
        argument_summary = _artifact_or_summary(
            arguments,
            trace_id=trace_id,
            source=f"incomplete_tool_call:{name}:arguments",
            limit=_TOOL_ARGUMENT_LIMIT,
        )
        line = f"tool={name or 'unknown'}; args={argument_summary}"
        if protected_refs:
            line += f"; protected_refs={protected_refs}"
        lines.append(line[:900])
    for message in tool_messages:
        result_summary = _tool_result_summary(
            _message_text(message),
            trace_id=trace_id,
            source=f"incomplete_tool_call:{message.name or 'unknown'}:result",
            limit=_TOOL_RESULT_LIMIT,
            protected_terms=protected_terms,
        )
        lines.append(f"partial_result={result_summary}"[:900])
    lines.append("</incomplete_tool_call_pruned>")
    return "\n".join(lines)


def _dedupe_repeated_messages(
    messages: list[LLMMessage],
    *,
    protected_terms: tuple[str, ...],
) -> tuple[list[LLMMessage], int]:
    if len(messages) <= 4:
        return messages, 0
    prefix_end = _prefix_end(messages)
    protected_indices = {
        index
        for index, message in enumerate(messages)
        if _message_contains_protected_term(message, protected_terms)
    }
    keep_last_start = max(prefix_end, len(messages) - 4)
    seen: dict[str, int] = {}
    deduped: list[LLMMessage] = []
    skipped = 0
    for index, message in enumerate(messages):
        if index < prefix_end or index >= keep_last_start or index in protected_indices:
            deduped.append(message)
            continue
        fingerprint = _dedupe_fingerprint(message)
        if not fingerprint:
            deduped.append(message)
            continue
        if fingerprint in seen:
            seen[fingerprint] += 1
            skipped += 1
            continue
        seen[fingerprint] = 1
        deduped.append(message)
    if skipped <= 0:
        return messages, 0
    dedupe_summary = LLMMessage.user(
        "\n".join(
            [
                "<context_dedup_summary>",
                f"deduped_repeated_messages={skipped}",
                "policy=identical_old_tool_summaries_or_results_collapsed",
                "</context_dedup_summary>",
            ]
        )
    )
    insert_at = min(max(prefix_end, 0), len(deduped))
    return [*deduped[:insert_at], dedupe_summary, *deduped[insert_at:]], skipped


def _dedupe_fingerprint(message: LLMMessage) -> str:
    if message.role == "system":
        return ""
    text = normalize_message_text(_message_text(message))
    if not text:
        return ""
    if not (
        text.startswith(_TOOL_SUMMARY_MARKER)
        or text.startswith("<orphan_tool_result_pruned>")
        or message.role == "tool"
        or "full_result_artifact_id=" in text
        or "full_result_artifact_id" in text
    ):
        return ""
    return _short_hash(
        json.dumps(
            {
                "role": message.role,
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "content": _normalize_for_hash(text),
            },
            ensure_ascii=False,
            default=str,
        )
    )


def _compact_protected_message(
    message: LLMMessage,
    *,
    trace_id: str,
    protected_terms: tuple[str, ...],
) -> LLMMessage:
    text = _message_text(message)
    if _message_token_cost(message) <= _PROTECTED_MESSAGE_LIMIT:
        return message
    protected_refs = _protected_excerpts(text, protected_terms)
    ref = get_artifact_store().store_text(
        _message_full_text(message),
        artifact_type="text",
        trace_id=trace_id,
        source="context_compression:protected_message",
        force_file=True,
    )
    lines = [
        "<protected_context_message>",
        f"role={message.role}",
        f"name={message.name or ''}",
        f"tool_call_id={message.tool_call_id or ''}",
        f"protected_refs={protected_refs}",
    ]
    if ref is not None:
        lines.append(f"artifact_id={ref.artifact_id}; summary={ref.summary}")
    lines.append("</protected_context_message>")
    return LLMMessage.user("\n".join(lines))


def _message_contains_protected_term(
    message: LLMMessage,
    protected_terms: tuple[str, ...],
) -> bool:
    if not protected_terms:
        return False
    text = _message_full_text(message).lower()
    return any(term.lower() in text for term in protected_terms if term)


def _protected_excerpts(text: str, protected_terms: tuple[str, ...]) -> str:
    if not protected_terms:
        return ""
    raw = str(text or "")
    lowered = raw.lower()
    hits: list[str] = []
    for term in protected_terms:
        if not term:
            continue
        index = lowered.find(term.lower())
        if index < 0:
            continue
        start = max(index - 48, 0)
        end = min(index + len(term) + 48, len(raw))
        excerpt = normalize_message_text(raw[start:end])
        if excerpt and excerpt not in hits:
            hits.append(excerpt)
    return " | ".join(hits[:8])


def _normalize_protected_terms(values: Iterable[str]) -> tuple[str, ...]:
    terms: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or ""))
        if not text or len(text) < 3:
            continue
        if text not in terms:
            terms.append(text)
    return tuple(terms[:80])


def _count_completed_tool_pairs(messages: list[LLMMessage]) -> int:
    count = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = list(message.tool_calls or []) if message.role == "assistant" else []
        if not tool_calls:
            index += 1
            continue
        tool_ids = {
            str(getattr(item, "id", "") or "")
            for item in tool_calls
            if str(getattr(item, "id", "") or "")
        }
        cursor = index + 1
        answered: set[str] = set()
        while cursor < len(messages) and messages[cursor].role == "tool":
            tool_call_id = str(messages[cursor].tool_call_id or "")
            if tool_call_id not in tool_ids:
                break
            answered.add(tool_call_id)
            cursor += 1
        if tool_ids and tool_ids <= answered:
            count += 1
            index = cursor
            continue
        index += 1
    return count


def _has_orphan_tool_messages(messages: list[LLMMessage]) -> bool:
    pending_tool_ids: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            pending_tool_ids = {
                str(getattr(call, "id", "") or "")
                for call in list(message.tool_calls or [])
                if str(getattr(call, "id", "") or "")
            }
            continue
        if message.role == "tool":
            tool_call_id = str(message.tool_call_id or "")
            if tool_call_id not in pending_tool_ids:
                return True
            pending_tool_ids.discard(tool_call_id)
            continue
        pending_tool_ids = set()
    return False


def _has_recent_compression_marker(messages: list[LLMMessage]) -> bool:
    for message in messages[-10:]:
        text = normalize_message_text(_message_text(message))
        if text.startswith(_COMPRESSION_MARKER) or text.startswith("<context_dedup_summary>"):
            return True
    return False


def _messages_fingerprint(messages: list[LLMMessage]) -> str:
    return _short_hash(
        "\n".join(
            f"{message.role}:{message.name or ''}:{message.tool_call_id or ''}:"
            f"{_short_hash(_message_full_text(message))}"
            for message in messages
        )
    )


def _normalize_for_hash(text: str) -> str:
    normalized = normalize_message_text(text)
    normalized = re.sub(r"fingerprint=[a-fA-F0-9]+", "fingerprint=*", normalized)
    normalized = re.sub(r"ci_[a-z_]+_[a-zA-Z0-9]+_[a-fA-F0-9]+", "ci_artifact_*", normalized)
    return normalized


def _short_hash(text: str) -> str:
    return blake2s(str(text or "").encode("utf-8", errors="ignore"), digest_size=8).hexdigest()


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


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
