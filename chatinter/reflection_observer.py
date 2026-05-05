from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal

from .route_text import normalize_message_text

ReflectionAction = Literal["memory_write", "memory_skip", "memory_digest"]


@dataclass(frozen=True)
class ReflectionObservation:
    timestamp: str
    action: ReflectionAction
    session_id: str = ""
    user_id: str = ""
    group_id: str = ""
    thread_id: str = ""
    reason: str = ""
    written: int = 0
    candidate_count: int = 0
    message_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReflectionObserver:
    _records: ClassVar[deque[ReflectionObservation]] = deque(maxlen=400)
    _capacity: ClassVar[int] = 400

    @classmethod
    def configure(cls, *, max_records: int | None = None) -> None:
        if max_records is not None:
            cls._capacity = max(int(max_records), 50)
        if cls._records.maxlen != cls._capacity:
            cls._records = deque(cls._records, maxlen=cls._capacity)

    @classmethod
    def record(
        cls,
        *,
        action: ReflectionAction,
        session_id: str | None = None,
        user_id: str | None = None,
        group_id: str | None = None,
        thread_id: str | None = None,
        reason: str | None = None,
        written: int = 0,
        candidate_count: int = 0,
        message_text: str = "",
    ) -> ReflectionObservation:
        cls.configure()
        observation = ReflectionObservation(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            action=action,
            session_id=normalize_message_text(session_id or ""),
            user_id=normalize_message_text(user_id or ""),
            group_id=normalize_message_text(group_id or ""),
            thread_id=normalize_message_text(thread_id or ""),
            reason=normalize_message_text(reason or ""),
            written=max(int(written or 0), 0),
            candidate_count=max(int(candidate_count or 0), 0),
            message_preview=normalize_message_text(message_text or "")[:120],
        )
        cls._records.append(observation)
        return observation

    @classmethod
    def snapshot(cls, limit: int = 200) -> dict[str, Any]:
        cls.configure()
        rows = list(cls._records)[-max(int(limit or 0), 1) :]
        if not rows:
            return {
                "total": 0,
                "action_counts": {},
                "reason_counts": {},
                "written": 0,
                "candidate_count": 0,
                "recent": [],
            }
        return {
            "total": len(rows),
            "action_counts": dict(Counter(row.action for row in rows)),
            "reason_counts": dict(Counter(row.reason for row in rows if row.reason)),
            "written": sum(row.written for row in rows),
            "candidate_count": sum(row.candidate_count for row in rows),
            "recent": [row.to_dict() for row in rows[-8:]],
        }

    @classmethod
    def clear(cls) -> None:
        cls._records.clear()


def record_reflection_observation(**kwargs: Any) -> ReflectionObservation:
    return ReflectionObserver.record(**kwargs)


def get_reflection_observer_snapshot(limit: int = 200) -> dict[str, Any]:
    return ReflectionObserver.snapshot(limit=limit)


__all__ = [
    "ReflectionObservation",
    "ReflectionObserver",
    "get_reflection_observer_snapshot",
    "record_reflection_observation",
]
