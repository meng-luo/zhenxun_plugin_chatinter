"""Group chat reply policy for ordinary conversation.

This policy is intentionally chat-only.  It can tune reply length/posture and
whether the bot should mostly "接话", but it must not decide plugin/tool use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .chat_dialogue_planner import DialogueState
from .route_text import normalize_message_text

GroupChatPosture = Literal[
    "listen",
    "brief_react",
    "answer",
    "clarify",
    "support",
    "stay_quiet",
]


@dataclass(frozen=True)
class GroupChatPolicyDecision:
    posture: GroupChatPosture = "brief_react"
    max_sentences: int = 2
    should_ask_followup: bool = False
    reason: str = "default"

    def to_xml(self) -> str:
        return "\n".join(
            (
                "<group_chat_reply_policy>",
                f"posture={self.posture}",
                f"max_sentences={self.max_sentences}",
                f"should_ask_followup={int(self.should_ask_followup)}",
                f"reason={self.reason}",
                "rule=Chat-only wording policy; never select tools or plugin commands.",
                "</group_chat_reply_policy>",
            )
        )


def decide_group_chat_policy(
    *,
    message_text: str,
    is_group: bool,
    is_to_me: bool,
    dialogue_state: DialogueState | None,
    thread_context: Any | None = None,
) -> GroupChatPolicyDecision:
    text = normalize_message_text(message_text)
    if not is_group:
        return GroupChatPolicyDecision(
            posture="answer",
            max_sentences=3,
            should_ask_followup=bool(getattr(dialogue_state, "need_followup", False)),
            reason="private_chat",
        )
    if not is_to_me and len(text) <= 4:
        return GroupChatPolicyDecision(
            posture="brief_react",
            max_sentences=1,
            should_ask_followup=False,
            reason="short_group_chatter",
        )
    if dialogue_state is not None and dialogue_state.user_emotion in {
        "stressed",
        "sad",
        "angry",
    }:
        return GroupChatPolicyDecision(
            posture="support",
            max_sentences=2,
            should_ask_followup=dialogue_state.need_followup,
            reason=f"emotion_{dialogue_state.user_emotion}",
        )
    if dialogue_state is not None and dialogue_state.dialogue_purpose in {
        "answer",
        "plan",
    }:
        return GroupChatPolicyDecision(
            posture="answer",
            max_sentences=4 if dialogue_state.response_length == "long" else 2,
            should_ask_followup=dialogue_state.need_followup,
            reason=f"purpose_{dialogue_state.dialogue_purpose}",
        )
    if getattr(thread_context, "confidence", 0.0) and not is_to_me:
        return GroupChatPolicyDecision(
            posture="brief_react",
            max_sentences=1,
            should_ask_followup=False,
            reason="ambient_thread_reply",
        )
    return GroupChatPolicyDecision(
        posture="brief_react" if is_group else "answer",
        max_sentences=2,
        should_ask_followup=bool(getattr(dialogue_state, "need_followup", False)),
        reason="default_group_chat",
    )


__all__ = ["GroupChatPolicyDecision", "decide_group_chat_policy"]
