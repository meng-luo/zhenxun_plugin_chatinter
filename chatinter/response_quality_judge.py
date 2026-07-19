"""Lightweight final reply quality checks for chat turns.

The judge is deliberately small and runtime-oriented.  It does not replace the
model or add plugin-specific routing; it only catches generic failure modes
before a message is sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .chat_dialogue_planner import DialogueState
from .route_text import normalize_message_text

QualityAction = Literal["ok", "revise", "block"]

_STIFF_PHRASES = (
    "尊敬的用户",
    "您好，关于您的问题",
    "作为一个人工智能",
    "我无法提供情感",
)
_QUESTION_MARKERS = ("吗", "么", "嘛", "？", "?", "怎么", "为什么", "是什么", "是啥")
_GENERIC_FILLER_REPLIES = frozenset(
    {
        "好的",
        "嗯嗯",
        "可以",
        "收到",
        "明白了",
        "我知道了",
        "了解",
    }
)
_REFUSAL_MARKERS = (
    "我不能",
    "无法帮助",
    "无法提供",
    "不能协助",
    "不能回答",
    "不方便回答",
)
_UNFOUNDED_CERTAINTY = (
    "一定",
    "绝对",
    "肯定是",
    "毫无疑问",
    "百分百",
)
_SOURCE_REQUEST_MARKERS = (
    "来源",
    "依据",
    "资料",
    "引用",
    "链接",
    "文档",
    "出处",
)
_HALLUCINATION_MARKERS = (
    "根据上面的图片",
    "如图所示",
    "我已经执行",
    "已经为你完成",
    "我查到了",
)
@dataclass(frozen=True)
class ResponseQualityResult:
    action: QualityAction
    reason: str = ""
    revised_text: str = ""
    instruction: str = ""
    severity: str = "info"
    rule_hits: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.action == "ok"


class ResponseQualityJudge:
    @staticmethod
    def judge_chat_only(
        *,
        final_text: str,
        original_message: str,
        dialogue_state: DialogueState | None,
    ) -> ResponseQualityResult:
        """Conversational quality check without tool/execution dependencies."""

        reply = normalize_message_text(final_text)
        message = normalize_message_text(original_message)
        if not reply:
            return ResponseQualityResult(
                action="block",
                reason="empty_final_reply",
                revised_text="我暂时没想好怎么回答你。",
                severity="error",
                rule_hits=("empty_final_reply",),
            )
        rule_hits = _quality_rule_hits(
            message=message,
            reply=reply,
            state=dialogue_state,
        )
        if "unsafe_refusal_needed" in rule_hits:
            return ResponseQualityResult(
                action="block",
                reason="unsafe_refusal_needed",
                revised_text="这个我不能帮你完成，但可以换个安全的方向一起看。",
                severity="error",
                rule_hits=tuple(rule_hits),
            )
        if _looks_stiff(reply, dialogue_state):
            return ResponseQualityResult(
                action="revise",
                reason="too_stiff_for_dialogue_state",
                revised_text=_soften_reply(reply),
                severity="warn",
                rule_hits=(*rule_hits, "too_stiff_for_dialogue_state"),
            )
        if _looks_off_topic(message, reply, dialogue_state):
            return ResponseQualityResult(
                action="revise",
                reason="possible_off_topic",
                instruction=(
                    "Reply should answer the user's current message directly; "
                    "avoid generic filler."
                ),
                severity="warn",
                rule_hits=(*rule_hits, "possible_off_topic"),
            )
        if "too_long_for_short_state" in rule_hits:
            return ResponseQualityResult(
                action="revise",
                reason="too_long_for_short_state",
                instruction="Compress the reply; keep conclusion first.",
                severity="warn",
                rule_hits=tuple(rule_hits),
            )
        if rule_hits:
            return ResponseQualityResult(
                action="ok",
                reason=rule_hits[0],
                severity="info",
                rule_hits=tuple(rule_hits),
            )
        return ResponseQualityResult(action="ok")


def _looks_stiff(reply: str, state: DialogueState | None) -> bool:
    if not any(phrase in reply for phrase in _STIFF_PHRASES):
        return False
    if state is None:
        return True
    return state.tone in {"casual", "warm", "playful", "empathetic"}


def _soften_reply(reply: str) -> str:
    softened = reply
    for phrase in _STIFF_PHRASES:
        softened = softened.replace(phrase, "")
    return normalize_message_text(softened).lstrip("，,。.!！？?") or reply


def _looks_off_topic(
    message: str,
    reply: str,
    state: DialogueState | None,
) -> bool:
    if not message or not reply:
        return False
    if state is not None and state.dialogue_purpose in {"chat", "support"}:
        return False
    if len(reply) <= 8 and any(marker in message for marker in _QUESTION_MARKERS):
        return True
    if reply in _GENERIC_FILLER_REPLIES and len(message) >= 12:
        return True
    return False


def _quality_rule_hits(
    *,
    message: str,
    reply: str,
    state: DialogueState | None,
) -> list[str]:
    hits: list[str] = []
    if len(reply) <= 2 and len(message) >= 8:
        hits.append("too_short_for_context")
    if _generic_filler(message, reply):
        hits.append("generic_filler")
    if _unhelpful_refusal(message, reply):
        hits.append("unhelpful_refusal")
    if _unsafe_request(message) and not _has_refusal(reply):
        hits.append("unsafe_refusal_needed")
    if _unsupported_grounding_claim(message, reply):
        hits.append("unsupported_grounding_claim")
    if _overconfident_without_basis(message, reply):
        hits.append("overconfident_without_basis")
    if _too_long_for_state(reply, state):
        hits.append("too_long_for_short_state")
    return hits


def _generic_filler(message: str, reply: str) -> bool:
    return reply in _GENERIC_FILLER_REPLIES and len(message) >= 10


def _unhelpful_refusal(message: str, reply: str) -> bool:
    if not any(marker in reply for marker in _REFUSAL_MARKERS):
        return False
    return "为什么" in message or "怎么" in message or len(message) >= 10


def _unsafe_request(message: str) -> bool:
    unsafe_markers = (
        "盗号",
        "破解",
        "绕过验证",
        "窃取",
        "木马",
        "炸群",
        "刷屏攻击",
    )
    return any(marker in message for marker in unsafe_markers)


def _has_refusal(reply: str) -> bool:
    return any(marker in reply for marker in _REFUSAL_MARKERS) or "不能" in reply


def _unsupported_grounding_claim(message: str, reply: str) -> bool:
    if any(marker in message for marker in _SOURCE_REQUEST_MARKERS):
        return False
    return any(marker in reply for marker in _HALLUCINATION_MARKERS)


def _overconfident_without_basis(message: str, reply: str) -> bool:
    if any(marker in message for marker in _SOURCE_REQUEST_MARKERS):
        return False
    if "不确定" in reply or "可能" in reply or "看起来" in reply:
        return False
    return any(marker in reply for marker in _UNFOUNDED_CERTAINTY)


def _too_long_for_state(reply: str, state: DialogueState | None) -> bool:
    if state is None or state.response_length != "short":
        return False
    return len(reply) > 360 and state.dialogue_purpose in {"chat", "support"}


__all__ = [
    "ResponseQualityJudge",
    "ResponseQualityResult",
]
