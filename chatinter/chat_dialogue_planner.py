from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .intent_classifier import IntentClassification
from .route_text import contains_any, normalize_message_text

ChatDialogueKind = Literal[
    "casual_chat",
    "factual_qa",
    "emotional_support",
    "recap",
    "identity_query",
    "memory_update",
    "explain_context",
    "complex_reasoning",
]

ChatDialogueStyle = Literal[
    "concise",
    "detailed",
    "empathetic",
    "step_by_step",
]

_EMOTION_HINTS = (
    "难受",
    "累",
    "焦虑",
    "烦",
    "崩溃",
    "委屈",
    "失眠",
    "不开心",
    "想哭",
    "压力",
    "害怕",
)
_FACTUAL_HINTS = (
    "是什么",
    "是啥",
    "为什么",
    "原因",
    "区别",
    "原理",
    "资料",
    "规则",
    "怎么回事",
)
_COMPLEX_HINTS = (
    "分析",
    "方案",
    "设计",
    "实现",
    "步骤",
    "排查",
    "优化",
    "对比",
    "架构",
    "总结",
)
_MEMORY_UPDATE_HINTS = (
    "记住",
    "记一下",
    "以后叫",
    "叫我",
    "我喜欢",
    "我不喜欢",
    "别叫",
)


@dataclass(frozen=True)
class ChatDialoguePlan:
    kind: ChatDialogueKind
    confidence: float = 0.72
    target_hint: str = ""
    need_grounding: bool = False
    need_memory_lookup: bool = False
    need_rewrite_check: bool = False
    style: ChatDialogueStyle = "concise"
    reason: str = "default"

    @property
    def is_complex(self) -> bool:
        return self.kind in {"complex_reasoning", "factual_qa"}


def plan_chat_dialogue(
    *,
    message_text: str,
    intent: IntentClassification | None = None,
    has_images: bool = False,
    has_reply: bool = False,
) -> ChatDialoguePlan:
    normalized = normalize_message_text(message_text or "")
    chat_subkind = getattr(intent, "chat_subkind", "general_chat") if intent else ""
    target_hint = normalize_message_text(
        getattr(intent, "chat_target_hint", "") if intent else ""
    )

    if chat_subkind == "recap":
        return ChatDialoguePlan(
            kind="recap",
            confidence=0.9,
            need_memory_lookup=True,
            style="concise",
            reason="intent_recap",
        )
    if chat_subkind == "identity_query":
        return ChatDialoguePlan(
            kind="identity_query",
            confidence=0.86,
            target_hint=target_hint,
            need_memory_lookup=True,
            style="concise",
            reason="intent_identity_query",
        )
    if chat_subkind == "memory_confirm":
        return ChatDialoguePlan(
            kind="memory_update",
            confidence=0.84,
            target_hint=target_hint,
            need_memory_lookup=True,
            style="concise",
            reason="intent_memory_confirm",
        )
    if chat_subkind == "explain_context":
        return ChatDialoguePlan(
            kind="explain_context",
            confidence=0.84,
            need_grounding=True,
            need_memory_lookup=True,
            need_rewrite_check=True,
            style="detailed",
            reason="intent_explain_context",
        )

    if not normalized:
        return ChatDialoguePlan(kind="casual_chat", confidence=0.5, reason="empty")
    if contains_any(normalized, _EMOTION_HINTS):
        return ChatDialoguePlan(
            kind="emotional_support",
            confidence=0.78,
            need_memory_lookup=True,
            style="empathetic",
            reason="emotion_hint",
        )
    if contains_any(normalized, _MEMORY_UPDATE_HINTS):
        return ChatDialoguePlan(
            kind="memory_update",
            confidence=0.75,
            need_memory_lookup=True,
            style="concise",
            reason="memory_hint",
        )
    if (
        has_images
        or has_reply
        or len(normalized) >= 56
        or contains_any(normalized, _COMPLEX_HINTS)
    ):
        return ChatDialoguePlan(
            kind="complex_reasoning",
            confidence=0.72,
            need_grounding=True,
            need_memory_lookup=True,
            need_rewrite_check=True,
            style="step_by_step",
            reason="complex_hint",
        )
    if contains_any(normalized, _FACTUAL_HINTS):
        return ChatDialoguePlan(
            kind="factual_qa",
            confidence=0.7,
            need_grounding=True,
            need_rewrite_check=True,
            style="detailed",
            reason="factual_hint",
        )

    return ChatDialoguePlan(kind="casual_chat", reason="general_chat")


__all__ = [
    "ChatDialogueKind",
    "ChatDialoguePlan",
    "ChatDialogueStyle",
    "plan_chat_dialogue",
]
