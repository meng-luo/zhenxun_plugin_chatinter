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


@dataclass(frozen=True)
class ResponseQualityResult:
    action: QualityAction
    reason: str = ""
    revised_text: str = ""
    instruction: str = ""

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
            )
        if _looks_stiff(reply, dialogue_state):
            return ResponseQualityResult(
                action="revise",
                reason="too_stiff_for_dialogue_state",
                revised_text=_soften_reply(reply),
            )
        if _looks_off_topic(message, reply, dialogue_state):
            return ResponseQualityResult(
                action="revise",
                reason="possible_off_topic",
                instruction=(
                    "Reply should answer the user's current message directly; "
                    "avoid generic filler."
                ),
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
    if reply in {"好的", "嗯嗯", "可以", "收到"} and len(message) >= 12:
        return True
    return False


__all__ = [
    "ResponseQualityJudge",
    "ResponseQualityResult",
]
