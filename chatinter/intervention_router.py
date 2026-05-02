from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .addressee_resolver import AddresseeResult
from .event_context import ChatInterEventContext
from .route_text import ROUTE_ACTION_WORDS, contains_any, normalize_message_text

InterventionAction = Literal["reply", "route", "ignore"]


@dataclass(frozen=True)
class InterventionDecision:
    action: InterventionAction
    confidence: float
    reason: str


_CONTINUATION_HINTS = ("继续", "然后呢", "那", "所以", "刚才", "上面", "这个")


def decide_intervention(
    *,
    event_context: ChatInterEventContext,
    addressee: AddresseeResult,
    route_signal: bool,
) -> InterventionDecision:
    if event_context.is_private:
        return InterventionDecision("reply", 1.0, "private")
    if event_context.is_to_me or addressee.is_bot_target:
        return InterventionDecision(
            "route" if route_signal else "reply",
            0.96,
            "addressed_to_bot",
        )
    if route_signal:
        return InterventionDecision("route", 0.84, "route_signal")
    if event_context.reply and addressee.source == "self":
        return InterventionDecision("reply", 0.9, "reply_to_bot")
    text = normalize_message_text(event_context.message_text_with_tags)
    if any(hint in text for hint in _CONTINUATION_HINTS) and event_context.reply:
        return InterventionDecision("reply", 0.62, "reply_continuation")
    return InterventionDecision("reply", 0.5, "fallback_matcher_invoked")


def has_route_signal(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    return (
        contains_any(normalized, ROUTE_ACTION_WORDS)
        or "[image" in normalized
        or "[@" in normalized
    )


__all__ = ["InterventionDecision", "decide_intervention", "has_route_signal"]
