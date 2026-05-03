"""Lightweight tool-intent probe for query expansion."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .route_text import (
    collect_placeholders,
    contains_any,
    has_negative_route_intent,
    is_usage_question,
    normalize_message_text,
    strip_invoke_prefix,
)
from .speech_act import classify_speech_act

_ACTION_HINTS = (
    "帮我",
    "请",
    "麻烦",
    "给我",
    "替我",
    "把",
    "查",
    "查一下",
    "查询",
    "搜",
    "搜一下",
    "搜索",
    "找",
    "找一下",
    "看",
    "看一下",
    "识别",
    "解析",
    "生成",
    "制作",
    "做",
    "来",
    "来一",
    "发",
    "发送",
    "添加",
    "新增",
    "删除",
    "翻译",
    "播放",
    "打开",
    "关闭",
    "统计",
)
_DISCUSSION_HINTS = (
    "聊聊",
    "讨论",
    "觉得",
    "这个词",
    "这个概念",
    "作为",
    "优点",
    "缺点",
    "为什么",
    "什么意思",
)
_TOOL_INTENT_MIN_LEN = 4


class ToolIntentProbe(BaseModel):
    has_tool_intent: bool = Field(description="用户是否像是在请求调用某个工具")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")


def probe_tool_intent(message_text: str, *, has_reply: bool = False) -> ToolIntentProbe:
    normalized = normalize_message_text(strip_invoke_prefix(message_text or ""))
    if not normalized or len(normalized) < _TOOL_INTENT_MIN_LEN:
        return ToolIntentProbe(has_tool_intent=False, reason="too_short")
    if has_negative_route_intent(normalized):
        return ToolIntentProbe(has_tool_intent=False, reason="negative_route_intent")
    speech_act = classify_speech_act(normalized, has_reply=has_reply)
    if speech_act == "ask_usage":
        return ToolIntentProbe(has_tool_intent=True, confidence=0.72, reason="usage")
    if speech_act in {"discuss_command", "casual_chat"} and contains_any(
        normalized, _DISCUSSION_HINTS
    ):
        return ToolIntentProbe(has_tool_intent=False, reason=f"speech_act:{speech_act}")
    placeholders = collect_placeholders(normalized)
    has_image = any(token.startswith("[image") for token in placeholders)
    has_at = any(
        token.startswith("[@") or token.startswith("@") for token in placeholders
    )
    has_context = has_image or has_at or has_reply
    if has_context and contains_any(normalized, _ACTION_HINTS):
        return ToolIntentProbe(
            has_tool_intent=True,
            confidence=0.78,
            reason="context_action",
        )
    if contains_any(normalized, _ACTION_HINTS):
        return ToolIntentProbe(has_tool_intent=True, confidence=0.68, reason="action")
    if is_usage_question(normalized):
        return ToolIntentProbe(has_tool_intent=True, confidence=0.62, reason="usage")
    return ToolIntentProbe(has_tool_intent=False, confidence=0.2, reason="no_signal")


__all__ = ["ToolIntentProbe", "probe_tool_intent"]
