"""Route-level feedback for recently executed ChatInter routes.

This module stays deliberately small: it observes the result of an already
chosen route and records user follow-up corrections/confirmations.  It never
performs routing or plugin execution by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, ClassVar, Literal

from .execution_observer import (
    EXECUTION_REASON_ROUTE_CONFIRMED,
    EXECUTION_REASON_ROUTE_USER_CORRECTED,
    record_execution_observation,
)
from .route_text import normalize_message_text

RouteFollowupFeedback = Literal["route_user_corrected", "route_confirmed"]

_CORRECTION_HINTS = (
    "不是",
    "不对",
    "错了",
    "你理解错",
    "不是这个",
    "不是这个意思",
    "我不是说",
)
_CONFIRM_HINTS = ("谢谢", "感谢", "对的", "就是这个", "可以的", "懂了")


@dataclass(frozen=True)
class LastRouteOutcome:
    timestamp: float
    session_id: str
    message_key: str
    command_id: str
    slots: dict[str, str] = field(default_factory=dict)
    plugin_module: str = ""
    plugin_name: str = ""
    command: str = ""
    route_stage: str = ""
    success: bool = False
    reason: str = ""


class RouteFeedbackStore:
    _last_routes: ClassVar[dict[str, LastRouteOutcome]] = {}
    _last_route_ttl: ClassVar[float] = 4 * 60.0
    _max_last_routes: ClassVar[int] = 256

    @classmethod
    def record_route_outcome(
        cls,
        *,
        session_id: str | None,
        message_text: str,
        route_result: Any | None,
        route_command: str = "",
        success: bool,
        reason: str,
    ) -> None:
        normalized_session = normalize_message_text(session_id or "")
        message_key = _message_key(message_text)
        if not normalized_session or not message_key or route_result is None:
            return

        command_id = normalize_message_text(getattr(route_result, "command_id", ""))
        decision = getattr(route_result, "decision", None)
        plugin_module = normalize_message_text(getattr(decision, "plugin_module", ""))
        plugin_name = normalize_message_text(getattr(decision, "plugin_name", ""))
        command = normalize_message_text(
            route_command or getattr(decision, "command", "")
        )
        route_stage = normalize_message_text(getattr(route_result, "stage", ""))
        if not command_id and not plugin_module:
            return

        now = time.monotonic()
        slots = _normalize_slots(getattr(route_result, "slots", {}) or {})
        last = LastRouteOutcome(
            timestamp=now,
            session_id=normalized_session,
            message_key=message_key,
            command_id=command_id,
            slots=slots,
            plugin_module=plugin_module,
            plugin_name=plugin_name,
            command=command,
            route_stage=route_stage,
            success=bool(success),
            reason=normalize_message_text(reason),
        )
        cls._last_routes[normalized_session] = last
        cls._prune(now)

    @classmethod
    def inspect_user_followup(
        cls,
        *,
        session_id: str | None,
        message_text: str,
    ) -> RouteFollowupFeedback | None:
        normalized_session = normalize_message_text(session_id or "")
        normalized_message = normalize_message_text(message_text)
        if not normalized_session or not normalized_message:
            return None
        now = time.monotonic()
        cls._prune(now)
        last = cls._last_routes.get(normalized_session)
        if last is None or now - last.timestamp > cls._last_route_ttl:
            return None
        if any(hint in normalized_message for hint in _CORRECTION_HINTS):
            cls._record_followup_feedback(
                last,
                success=False,
                reason=EXECUTION_REASON_ROUTE_USER_CORRECTED,
                message_text=normalized_message,
            )
            cls._last_routes.pop(normalized_session, None)
            return "route_user_corrected"
        if any(hint in normalized_message for hint in _CONFIRM_HINTS):
            cls._record_followup_feedback(
                last,
                success=True,
                reason=EXECUTION_REASON_ROUTE_CONFIRMED,
                message_text=normalized_message,
            )
            cls._last_routes.pop(normalized_session, None)
            return "route_confirmed"
        return None

    @classmethod
    def clear(cls) -> None:
        cls._last_routes.clear()

    @classmethod
    def _record_followup_feedback(
        cls,
        last: LastRouteOutcome,
        *,
        success: bool,
        reason: str,
        message_text: str,
    ) -> None:
        record_execution_observation(
            action="execute",
            success=success,
            reason=reason,
            plugin_module=last.plugin_module,
            plugin_name=last.plugin_name,
            command_id=last.command_id,
            command=last.command,
            route_stage=last.route_stage or "route_feedback",
            session_id=last.session_id,
            message_preview=message_text,
        )

    @classmethod
    def _prune(cls, now: float) -> None:
        expired_last = [
            key
            for key, value in cls._last_routes.items()
            if now - value.timestamp > cls._last_route_ttl
        ]
        for key in expired_last:
            cls._last_routes.pop(key, None)

        if len(cls._last_routes) > cls._max_last_routes:
            stale_sessions = sorted(
                cls._last_routes.items(),
                key=lambda item: item[1].timestamp,
            )[:32]
            for key, _ in stale_sessions:
                cls._last_routes.pop(key, None)


def _message_key(message_text: str) -> str:
    return normalize_message_text(message_text)[:240]


def _normalize_slots(raw_slots: dict[str, Any]) -> dict[str, str]:
    slots: dict[str, str] = {}
    for key, value in raw_slots.items():
        name = normalize_message_text(str(key or ""))
        slot_value = normalize_message_text(str(value or ""))
        if name and slot_value:
            slots[name] = slot_value
    return slots


__all__ = [
    "LastRouteOutcome",
    "RouteFeedbackStore",
    "RouteFollowupFeedback",
]
