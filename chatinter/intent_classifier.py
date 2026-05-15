"""Lightweight ChatInter intent prefilter.

The AgentRuntime is now responsible for deciding whether to retrieve and call
tools.  This module only annotates the turn with cheap hints for metrics,
dialogue planning, and middleware.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from .models.pydantic_models import PluginKnowledgeBase
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_weak_route_signals,
    contains_any,
    has_negative_route_intent,
    is_usage_question,
    normalize_action_phrases,
    normalize_message_text,
    strip_invoke_prefix,
)

IntentKind = Literal["chat", "help", "execute", "execute_need_arg", "ambiguous"]
ChatDialogueKind = Literal[
    "general_chat",
    "recap",
    "identity_query",
    "memory_confirm",
    "explain_context",
]
IntentSchemaState = Literal[
    "unknown",
    "ready",
    "missing_target",
    "missing_image",
    "missing_text",
]

_AT_TOKEN_PATTERN = re.compile(r"\[@[^\]\s]+\]")
_IMAGE_TOKEN_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_SELF_REF_HINTS = ("我", "自己", "本人", "我自己", "自己的")
_WEAK_ROUTE_HINTS = ("帮我", "给我", "请", "麻烦", "查看", "看看", "看下", "查询")
_CHAT_RECAP_HINTS = (
    "我们说了些什么",
    "我们说了什么",
    "我们聊了些什么",
    "我们聊了什么",
    "回顾一下",
    "总结一下",
    "前面说了什么",
    "刚才说了什么",
    "刚刚说了什么",
)
_CHAT_IDENTITY_TARGET_PATTERNS = (
    re.compile(
        r"(?:知道|认识|了解|想问|问一下|请问|你知道)?"
        r"(?P<hint>[A-Za-z0-9\u4e00-\u9fff]{1,16})"
        r"(?:是谁|是啥|什么人|哪位|是谁呀|是谁吗|是谁嘛|是谁啊)"
    ),
    re.compile(
        r"(?:知道|认识|了解|你知道|你认识|你了解)"
        r"(?P<hint>[A-Za-z0-9\u4e00-\u9fff]{1,16})"
        r"(?:吗|嘛|么|不|没有|没)"
    ),
    re.compile(r"(?P<hint>[A-Za-z0-9\u4e00-\u9fff]{1,16})(?:是谁|是啥|什么人|哪位)"),
)
_CHAT_SELF_IDENTITY_HINTS = ("我是谁", "我叫啥", "我叫什么", "我是哪位")
_CHAT_MEMORY_TARGET_PATTERNS = (
    re.compile(
        r"(?P<hint>[A-Za-z0-9\u4e00-\u9fff]{1,16})"
        r"(?:是(?:他|她|TA|ta|本人|这个人|那个人)"
        r"|就是(?:他|她|TA|ta|这个人|那个人))"
    ),
    re.compile(
        r"(?:以后叫|就叫|叫他|叫她|叫它|记住(?:这个)?(?:名字|称呼)?叫)"
        r"(?P<hint>[A-Za-z0-9\u4e00-\u9fff]{1,16})"
    ),
)
_CHAT_EXPLAIN_HINTS = ("什么意思", "是什么意思", "指的什么", "解释一下", "说明一下")
_CHAT_EXPLAIN_CONTEXT_HINTS = (
    "前面",
    "上面",
    "刚才",
    "刚刚",
    "之前",
    "这个",
    "那个",
    "上下文",
    "前文",
)


@dataclass(frozen=True)
class IntentClassification:
    kind: IntentKind
    reason: str
    explicit_command: bool = False
    plugin_name: str | None = None
    plugin_module: str | None = None
    command_head: str | None = None
    payload_text: str = ""
    schema: object | None = None
    confidence: float = 0.0
    schema_state: IntentSchemaState = "unknown"
    rewrite_command: str = ""
    chat_subkind: ChatDialogueKind = "general_chat"
    chat_target_hint: str = ""


def classify_message_intent(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> IntentClassification:
    _ = knowledge_base
    normalized = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    if not normalized:
        return IntentClassification(kind="chat", reason="empty_message", confidence=1.0)
    if has_negative_route_intent(normalized):
        return IntentClassification(
            kind="chat",
            reason="negative_route_intent",
            confidence=0.96,
        )

    chat_subkind, target_hint, chat_reason = _classify_chat_dialogue(normalized)
    if chat_subkind != "general_chat":
        return IntentClassification(
            kind="chat",
            reason=chat_reason,
            confidence=0.92,
            chat_subkind=chat_subkind,
            chat_target_hint=target_hint,
        )
    if is_usage_question(normalized):
        return IntentClassification(
            kind="help",
            reason="usage_question_prefilter",
            confidence=0.72,
        )
    if _has_route_prefilter_signal(normalized):
        return IntentClassification(
            kind="ambiguous",
            reason="route_signal_prefilter",
            confidence=0.58,
        )
    return IntentClassification(
        kind="chat",
        reason="no_route_signal",
        confidence=0.86,
    )


def _classify_chat_dialogue(
    normalized_message: str,
) -> tuple[ChatDialogueKind, str, str]:
    compact = normalize_message_text(normalized_message).replace(" ", "")
    if not compact:
        return "general_chat", "", "general_chat"

    if contains_any(compact, _CHAT_RECAP_HINTS):
        return "recap", "", "recap_request"

    memory_hint = _extract_chat_target_hint(compact, _CHAT_MEMORY_TARGET_PATTERNS)
    if memory_hint or contains_any(
        compact,
        ("记住了吗", "记住了么", "记一下", "记住这个", "以后叫", "就叫"),
    ):
        return "memory_confirm", memory_hint, "memory_confirm_request"

    identity_hint = _extract_chat_target_hint(
        compact,
        _CHAT_IDENTITY_TARGET_PATTERNS,
    )
    if identity_hint or contains_any(compact, _CHAT_SELF_IDENTITY_HINTS):
        return "identity_query", identity_hint, "identity_query_request"

    if contains_any(compact, _CHAT_EXPLAIN_CONTEXT_HINTS) and contains_any(
        compact,
        _CHAT_EXPLAIN_HINTS,
    ):
        return "explain_context", "", "context_explain_request"
    return "general_chat", "", "general_chat"


def _extract_chat_target_hint(
    normalized_message: str,
    patterns: tuple[re.Pattern[str], ...],
) -> str:
    compact = normalize_message_text(normalized_message).replace(" ", "")
    for pattern in patterns:
        match = pattern.search(compact)
        if not match:
            continue
        hint = normalize_message_text(match.groupdict().get("hint", ""))
        if hint and hint not in _SELF_REF_HINTS and len(hint) <= 16:
            return hint
    return ""


def _has_route_prefilter_signal(normalized_message: str) -> bool:
    if bool(_AT_TOKEN_PATTERN.search(normalized_message)) or bool(
        _IMAGE_TOKEN_PATTERN.search(normalized_message)
    ):
        return True
    if collect_weak_route_signals(normalized_message):
        return True
    if contains_any(normalized_message, _WEAK_ROUTE_HINTS):
        return True
    return any(
        word
        for word in ROUTE_ACTION_WORDS
        if word and word not in _WEAK_ROUTE_HINTS and word in normalized_message
    )


__all__ = [
    "IntentClassification",
    "IntentKind",
    "IntentSchemaState",
    "classify_message_intent",
]
