"""Lightweight speech-act guard for command/chat boundary cases."""

from __future__ import annotations

import re
from typing import Literal

from .route_text import (
    ROUTE_ACTION_WORDS,
    contains_any,
    has_chat_context_hint,
    has_negative_route_intent,
    is_usage_question,
    normalize_action_phrases,
    normalize_message_text,
    strip_invoke_prefix,
)

SpeechAct = Literal[
    "perform_command",
    "ask_usage",
    "discuss_command",
    "casual_chat",
    "clarify",
]

_DISCUSSION_MARKERS = (
    "这个词",
    "这个概念",
    "这件事",
    "这种",
    "作为",
    "本身",
    "为什么",
    "是不是",
    "是什么",
    "什么意思",
    "怎么理解",
    "优点",
    "缺点",
    "挺",
    "文化",
    "设计",
    "活动",
    "口头禅",
    "不是插件",
    "公平吗",
    "经济学",
    "会怎样",
    "系统从",
    "模型",
    "训练",
    "原理",
    "期望怎么算",
    "概率",
    "只想聊",
    "不想听歌",
    "这件事",
    "件事",
)

_STRONG_PERFORM_MARKERS = (
    "帮我",
    "请",
    "麻烦",
    "给我",
    "替我",
    "把",
    "将",
    "来个",
    "来一个",
    "来一张",
    "来点",
    "来首",
    "做个",
    "做一个",
    "做一张",
    "打开",
    "执行",
    "调用",
    "发",
    "发送",
    "查一下",
    "看一下",
    "找一下",
    "搜一下",
    "翻译一下",
    "想看",
    "有哪些",
    "整",
    "选一个",
    "挑一个",
    "帮我选",
    "放一首",
    "放首",
    "播一首",
    "播首",
    "掷骰子",
    "掷个骰子",
    "投骰子",
)
_META_DISCUSSION_MARKERS = (
    "架构",
    "架构上",
    "怎么设计",
    "如何设计",
    "怎样设计",
    "系统设计",
    "管理系统",
    "设计方案",
    "原理",
    "技术背后",
)
_DISCUSSION_QUESTION_MARKERS = (
    "为什么",
    "是什么",
    "什么意思",
    "怎么理解",
    "如何理解",
    "这件事",
    "件事",
    "文化",
    "原理",
    "公平吗",
    "经济学",
    "期望怎么算",
    "模型",
    "训练",
    "容易",
)

_ABBR_QUERY_PATTERN = re.compile(
    r"(?:[0-9A-Za-z_]{2,}\s*(?:是)?(?:什么|啥|哪个)?缩写)"
    r"|(?:缩写\s*[0-9A-Za-z_]{2,})",
    re.IGNORECASE,
)


def classify_speech_act(
    text: str,
    *,
    has_image: bool = False,
    has_at: bool = False,
    has_reply: bool = False,
) -> SpeechAct:
    normalized = normalize_message_text(normalize_action_phrases(text or ""))
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if not stripped:
        return "casual_chat"
    if has_negative_route_intent(stripped):
        return "discuss_command"
    if not (has_image or has_at or has_reply) and contains_any(
        stripped, _META_DISCUSSION_MARKERS
    ):
        return "discuss_command"
    if has_chat_context_hint(stripped) and not contains_any(
        stripped, _STRONG_PERFORM_MARKERS
    ):
        return "discuss_command"
    if (
        not (has_image or has_at or has_reply)
        and has_chat_context_hint(stripped)
        and contains_any(stripped, _DISCUSSION_QUESTION_MARKERS)
    ):
        return "discuss_command"
    if is_usage_question(stripped):
        return "ask_usage"
    if _ABBR_QUERY_PATTERN.search(stripped):
        return "perform_command"
    if has_image or has_at or has_reply:
        if contains_any(stripped, ROUTE_ACTION_WORDS) or contains_any(
            stripped, _STRONG_PERFORM_MARKERS
        ):
            return "perform_command"
        return "clarify"
    if contains_any(stripped, _STRONG_PERFORM_MARKERS):
        if has_chat_context_hint(stripped) and any(
            marker in stripped for marker in ("不是", "不要", "不用", "别", "陪我聊")
        ):
            return "discuss_command"
        return "perform_command"
    if contains_any(stripped, _DISCUSSION_MARKERS):
        return "discuss_command"
    if contains_any(stripped, ROUTE_ACTION_WORDS):
        return "perform_command"
    return "casual_chat"


def should_force_chat(
    text: str,
    *,
    has_image: bool = False,
    has_at: bool = False,
    has_reply: bool = False,
) -> bool:
    return classify_speech_act(
        text,
        has_image=has_image,
        has_at=has_at,
        has_reply=has_reply,
    ) in {"discuss_command", "casual_chat"}


__all__ = ["SpeechAct", "classify_speech_act", "should_force_chat"]
