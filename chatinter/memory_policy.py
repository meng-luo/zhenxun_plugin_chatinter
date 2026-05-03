from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .route_text import normalize_message_text

MemoryAction = Literal["write", "skip", "digest"]
MemoryScope = Literal["user", "group", "thread"]

_QUESTION_SUFFIXES = ("吗", "么", "嘛", "？", "?")
_LOW_VALUE_MEMORY_HINTS = (
    "记住了吗",
    "你记得吗",
    "还记得吗",
    "忘了吗",
)
_PERSONAL_MEMORY_HINTS = (
    "叫我",
    "喊我",
    "称呼我",
    "我喜欢",
    "我不喜欢",
    "别叫我",
)
_GROUP_DIGEST_HINTS = (
    "我们",
    "大家",
    "群里",
    "这个群",
    "刚才",
    "前面",
    "总结",
    "决定",
    "约好",
)
_GROUP_DIGEST_STRONG_HINTS = ("决定", "约好", "总结")


@dataclass(frozen=True)
class MemoryPolicyDecision:
    action: MemoryAction
    scope: MemoryScope = "user"
    confidence: float = 0.0
    reason: str = ""

    @property
    def should_write(self) -> bool:
        return self.action in {"write", "digest"}


def decide_memory_policy(
    *,
    message_text: str,
    response_text: str = "",
    group_id: str | None = None,
    thread_id: str | None = None,
    memory_candidate_count: int = 0,
) -> MemoryPolicyDecision:
    text = normalize_message_text(message_text or "")
    response = normalize_message_text(response_text or "")
    if not text:
        return MemoryPolicyDecision("skip", reason="empty_message")
    if text.endswith(_QUESTION_SUFFIXES) or any(
        hint in text for hint in _LOW_VALUE_MEMORY_HINTS
    ):
        return MemoryPolicyDecision("skip", reason="question_or_memory_probe")
    if memory_candidate_count > 0:
        return MemoryPolicyDecision(
            "write",
            scope="user",
            confidence=0.86,
            reason="explicit_personal_memory",
        )
    if any(hint in text for hint in _PERSONAL_MEMORY_HINTS):
        return MemoryPolicyDecision(
            "write",
            scope="user",
            confidence=0.62,
            reason="personal_memory_hint",
        )
    if group_id and thread_id and _should_digest_group_turn(text, response):
        return MemoryPolicyDecision(
            "digest",
            scope="thread",
            confidence=0.58,
            reason="group_thread_digest_candidate",
        )
    return MemoryPolicyDecision("skip", reason="no_memory_signal")


def _should_digest_group_turn(message_text: str, response_text: str) -> bool:
    if not message_text:
        return False
    if len(message_text) >= 16 and any(
        hint in message_text for hint in _GROUP_DIGEST_STRONG_HINTS
    ):
        return True
    if len(message_text) >= 36 and any(
        hint in message_text for hint in _GROUP_DIGEST_HINTS
    ):
        return True
    merged = f"{message_text} {response_text}"
    return len(merged) >= 80 and any(hint in merged for hint in _GROUP_DIGEST_HINTS)


__all__ = [
    "MemoryPolicyDecision",
    "decide_memory_policy",
]
