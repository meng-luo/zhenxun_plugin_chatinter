from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import time
from typing import Any, ClassVar, Literal

from .route_text import normalize_message_text

ChatFeedbackKind = Literal[
    "chat_completed",
    "chat_rewritten",
    "chat_empty",
    "user_corrected",
    "user_thanks",
    "followup_same_topic",
]

_CORRECTION_HINTS = ("不是", "不对", "你理解错", "不是这个意思", "我不是说")
_THANKS_HINTS = ("谢谢", "感谢", "对的", "就是这个", "可以的", "懂了")


@dataclass(frozen=True)
class ChatFeedbackRecord:
    timestamp: float
    session_id: str
    kind: ChatFeedbackKind
    message_preview: str = ""
    reply_preview: str = ""
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = round(self.timestamp, 3)
        return payload


class ChatFeedbackStore:
    _records: ClassVar[deque[ChatFeedbackRecord]] = deque(maxlen=400)
    _last_chat: ClassVar[dict[str, tuple[float, str, str]]] = {}
    _ttl: ClassVar[float] = 1800.0

    @classmethod
    def record(
        cls,
        *,
        session_id: str | None,
        kind: ChatFeedbackKind,
        message_text: str = "",
        reply_text: str = "",
        weight: float = 0.0,
    ) -> None:
        normalized_session = normalize_message_text(session_id or "")
        if not normalized_session:
            return
        now = time.monotonic()
        cls._records.append(
            ChatFeedbackRecord(
                timestamp=now,
                session_id=normalized_session,
                kind=kind,
                message_preview=normalize_message_text(message_text)[:120],
                reply_preview=normalize_message_text(reply_text)[:120],
                weight=float(weight or 0.0),
            )
        )
        if kind in {"chat_completed", "chat_rewritten"}:
            cls._last_chat[normalized_session] = (
                now,
                normalize_message_text(message_text),
                normalize_message_text(reply_text),
            )
        cls._record_memory_feedback(
            session_id=normalized_session,
            kind=kind,
            weight=float(weight or 0.0),
        )
        cls._prune(now)

    @classmethod
    def inspect_user_followup(
        cls,
        *,
        session_id: str | None,
        message_text: str,
    ) -> ChatFeedbackKind | None:
        normalized_session = normalize_message_text(session_id or "")
        normalized_message = normalize_message_text(message_text)
        if not normalized_session or not normalized_message:
            return None
        now = time.monotonic()
        cls._prune(now)
        if any(hint in normalized_message for hint in _CORRECTION_HINTS):
            cls.record(
                session_id=normalized_session,
                kind="user_corrected",
                message_text=normalized_message,
                weight=-1.0,
            )
            return "user_corrected"
        if any(hint in normalized_message for hint in _THANKS_HINTS):
            cls.record(
                session_id=normalized_session,
                kind="user_thanks",
                message_text=normalized_message,
                weight=0.45,
            )
            return "user_thanks"
        last = cls._last_chat.get(normalized_session)
        if last is None:
            return None
        last_ts, last_message, _last_reply = last
        if now - last_ts > 240:
            return None
        if _shared_token_count(last_message, normalized_message) >= 2:
            cls.record(
                session_id=normalized_session,
                kind="followup_same_topic",
                message_text=normalized_message,
                weight=-0.25,
            )
            return "followup_same_topic"
        return None

    @classmethod
    def recent(cls, limit: int = 20) -> list[dict[str, Any]]:
        cls._prune(time.monotonic())
        return [item.to_dict() for item in list(cls._records)[-max(limit, 1) :]]

    @classmethod
    def clear(cls) -> None:
        cls._records.clear()
        cls._last_chat.clear()
        try:
            from .memory_feedback_reranker import MemoryFeedbackReranker

            MemoryFeedbackReranker.clear()
        except Exception:
            pass

    @classmethod
    def _prune(cls, now: float) -> None:
        expired_sessions = [
            session
            for session, (ts, _message, _reply) in cls._last_chat.items()
            if now - ts > cls._ttl
        ]
        for session in expired_sessions:
            cls._last_chat.pop(session, None)

    @staticmethod
    def _record_memory_feedback(
        *,
        session_id: str,
        kind: ChatFeedbackKind,
        weight: float,
    ) -> None:
        if kind not in {"user_corrected", "user_thanks", "followup_same_topic"}:
            return
        try:
            from .memory_feedback_reranker import MemoryFeedbackReranker

            MemoryFeedbackReranker.record_feedback(
                session_id=session_id,
                kind=kind,
                weight=weight,
            )
        except Exception:
            pass


def _shared_token_count(left: str, right: str) -> int:
    left_tokens = {
        token for token in normalize_message_text(left).split(" ") if len(token) >= 2
    }
    right_tokens = {
        token for token in normalize_message_text(right).split(" ") if len(token) >= 2
    }
    if not left_tokens or not right_tokens:
        return 0
    return len(left_tokens & right_tokens)


__all__ = [
    "ChatFeedbackKind",
    "ChatFeedbackRecord",
    "ChatFeedbackStore",
]
