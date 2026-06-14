from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal

from .route_text import normalize_message_text

ReflectionAction = Literal["memory_write", "memory_skip", "memory_digest"]
_OVERWRITE_RISK_THRESHOLD = 0.45
_DIGEST_RISK_THRESHOLD = 0.35


@dataclass(frozen=True)
class ReflectionObservation:
    timestamp: str
    action: ReflectionAction
    session_id: str = ""
    user_id: str = ""
    group_id: str = ""
    thread_id: str = ""
    reason: str = ""
    policy_action: str = ""
    policy_scope: str = ""
    policy_confidence: float = 0.0
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
        policy_action: str | None = None,
        policy_scope: str | None = None,
        policy_confidence: float = 0.0,
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
            policy_action=normalize_message_text(policy_action or ""),
            policy_scope=normalize_message_text(policy_scope or ""),
            policy_confidence=max(float(policy_confidence or 0.0), 0.0),
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
                "policy_action_counts": {},
                "policy_scope_counts": {},
                "write_rate": None,
                "digest_rate": None,
                "skip_rate": None,
                "risk_flags": [],
                "written": 0,
                "candidate_count": 0,
                "recent": [],
            }
        action_counts = Counter(row.action for row in rows)
        policy_action_counts = Counter(
            row.policy_action for row in rows if row.policy_action
        )
        policy_scope_counts = Counter(
            row.policy_scope for row in rows if row.policy_scope
        )
        write_like = action_counts.get("memory_write", 0) + action_counts.get(
            "memory_digest",
            0,
        )
        digest_count = action_counts.get("memory_digest", 0)
        total = len(rows)
        write_rate = _rate(write_like, total)
        digest_rate = _rate(digest_count, total)
        skip_rate = _rate(action_counts.get("memory_skip", 0), total)
        return {
            "total": total,
            "action_counts": dict(action_counts),
            "reason_counts": dict(Counter(row.reason for row in rows if row.reason)),
            "policy_action_counts": dict(policy_action_counts),
            "policy_scope_counts": dict(policy_scope_counts),
            "write_rate": write_rate,
            "digest_rate": digest_rate,
            "skip_rate": skip_rate,
            "risk_flags": _risk_flags(
                write_rate=write_rate,
                digest_rate=digest_rate,
                action_counts=action_counts,
                total=total,
            ),
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


def render_reflection_observer_summary(limit: int = 200) -> str:
    payload = get_reflection_observer_snapshot(limit=limit)
    if not payload.get("total"):
        return "ChatInter 记忆反思最近 0 条"
    action_counts = _format_counts(payload.get("action_counts"))
    reason_counts = _format_counts(payload.get("reason_counts"), limit=5)
    policy_actions = _format_counts(payload.get("policy_action_counts"))
    policy_scopes = _format_counts(payload.get("policy_scope_counts"))
    risk_flags = (
        payload.get("risk_flags") if isinstance(payload.get("risk_flags"), list) else []
    )
    lines = [
        f"ChatInter 记忆反思最近 {payload.get('total')} 条",
        f"action: {action_counts or 'none'}",
        f"policy_action: {policy_actions or 'none'}",
        f"policy_scope: {policy_scopes or 'none'}",
        (
            "rates: "
            f"write={payload.get('write_rate')}, "
            f"digest={payload.get('digest_rate')}, "
            f"skip={payload.get('skip_rate')}"
        ),
        f"written={payload.get('written')} candidates={payload.get('candidate_count')}",
        f"top_reasons: {reason_counts or 'none'}",
    ]
    if risk_flags:
        lines.append("risk: " + ", ".join(str(item) for item in risk_flags))
    return "\n".join(lines)


def reflection_metrics_snapshot(limit: int = 200) -> dict[str, Any]:
    """Host-metrics friendly read-only snapshot."""

    payload = get_reflection_observer_snapshot(limit=limit)
    return {
        "chatinter.reflection.total": int(payload.get("total") or 0),
        "chatinter.reflection.written": int(payload.get("written") or 0),
        "chatinter.reflection.candidate_count": int(
            payload.get("candidate_count") or 0
        ),
        "chatinter.reflection.write_rate": payload.get("write_rate"),
        "chatinter.reflection.digest_rate": payload.get("digest_rate"),
        "chatinter.reflection.skip_rate": payload.get("skip_rate"),
        "chatinter.reflection.risk_flags": payload.get("risk_flags", []),
    }


def _risk_flags(
    *,
    write_rate: float | None,
    digest_rate: float | None,
    action_counts: Counter[str],
    total: int,
) -> list[str]:
    flags: list[str] = []
    if total < 20:
        return flags
    if write_rate is not None and write_rate >= _OVERWRITE_RISK_THRESHOLD:
        flags.append("high_memory_write_rate")
    if digest_rate is not None and digest_rate >= _DIGEST_RISK_THRESHOLD:
        flags.append("high_digest_rate")
    if action_counts.get("memory_skip", 0) == 0:
        flags.append("no_skip_samples")
    return flags


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _format_counts(value: Any, *, limit: int = 8) -> str:
    if not isinstance(value, dict):
        return ""
    rows = sorted(value.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))
    return ", ".join(f"{key}={count}" for key, count in rows[:limit])


__all__ = [
    "ReflectionObservation",
    "ReflectionObserver",
    "get_reflection_observer_snapshot",
    "record_reflection_observation",
    "reflection_metrics_snapshot",
    "render_reflection_observer_summary",
]
