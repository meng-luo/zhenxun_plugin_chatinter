"""
ChatInter 执行观察器。

该模块只记录 ChatInter 决策进入执行后的结果，不负责路由、不负责权限、
不负责真正调用插件，避免把观察层变成新的执行链。
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
import time
from typing import Any, ClassVar, Literal

from .route_text import normalize_message_text

ExecutionAction = Literal["chat", "usage", "clarify", "execute"]

EXECUTION_REASON_SUCCESS = "success"
EXECUTION_REASON_ROUTE_SUCCESS = "route_success"
EXECUTION_REASON_CHAT_COMPLETED = "chat_completed"
EXECUTION_REASON_USAGE_REPLIED = "usage_replied"
EXECUTION_REASON_CLARIFY_REQUESTED = "clarify_requested"
EXECUTION_REASON_MISSING_PARAMS = "missing_params"
EXECUTION_REASON_MISSING_TEXT = "missing_text"
EXECUTION_REASON_MISSING_IMAGE = "missing_image"
EXECUTION_REASON_MISSING_REPLY = "missing_reply"
EXECUTION_REASON_PERMISSION_DENIED = "permission_denied"
EXECUTION_REASON_PLUGIN_NOT_LOADED = "plugin_not_loaded"
EXECUTION_REASON_INVALID_COMMAND = "invalid_command"
EXECUTION_REASON_REROUTE_FAILED = "reroute_failed"
EXECUTION_REASON_TIMEOUT = "timeout"
EXECUTION_REASON_LLM_ERROR = "llm_error"
EXECUTION_REASON_CANCELLED = "cancelled"
EXECUTION_REASON_ERROR = "error"


@dataclass(frozen=True)
class ExecutionObservation:
    timestamp: str
    action: ExecutionAction
    success: bool
    reason: str
    latency_ms: int
    plugin_module: str = ""
    plugin_name: str = ""
    command_id: str = ""
    command: str = ""
    route_stage: str = ""
    session_id: str = ""
    message_preview: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionFrame:
    action: ExecutionAction
    plugin_module: str = ""
    plugin_name: str = ""
    command_id: str = ""
    command: str = ""
    route_stage: str = ""
    session_id: str = ""
    message_preview: str = ""
    started_at: float = field(default_factory=time.perf_counter)

    def finish(
        self,
        *,
        success: bool,
        reason: str,
        token_usage: dict[str, int] | None = None,
    ) -> ExecutionObservation:
        return ExecutionObserver.record(
            action=self.action,
            success=success,
            reason=reason,
            started_at=self.started_at,
            plugin_module=self.plugin_module,
            plugin_name=self.plugin_name,
            command_id=self.command_id,
            command=self.command,
            route_stage=self.route_stage,
            session_id=self.session_id,
            message_preview=self.message_preview,
            token_usage=token_usage,
        )


class ExecutionObserver:
    _records: ClassVar[deque[ExecutionObservation]] = deque(maxlen=400)
    _capacity: ClassVar[int] = 400
    _command_feedback: ClassVar[dict[str, float]] = {}
    _session_command_feedback: ClassVar[dict[str, dict[str, float]]] = {}
    _module_feedback: ClassVar[dict[str, float]] = {}

    @classmethod
    def configure(cls, *, max_records: int | None = None) -> None:
        if max_records is not None:
            cls._capacity = max(int(max_records), 50)
        if cls._records.maxlen != cls._capacity:
            cls._records = deque(cls._records, maxlen=cls._capacity)

    @classmethod
    def start(
        cls,
        *,
        action: ExecutionAction,
        plugin_module: str | None = None,
        plugin_name: str | None = None,
        command_id: str | None = None,
        command: str | None = None,
        route_stage: str | None = None,
        session_id: str | None = None,
        message_preview: str | None = None,
    ) -> ExecutionFrame:
        return ExecutionFrame(
            action=action,
            plugin_module=normalize_message_text(plugin_module or ""),
            plugin_name=normalize_message_text(plugin_name or ""),
            command_id=normalize_message_text(command_id or ""),
            command=normalize_message_text(command or ""),
            route_stage=normalize_message_text(route_stage or ""),
            session_id=normalize_message_text(session_id or ""),
            message_preview=normalize_message_text(message_preview or "")[:120],
        )

    @classmethod
    def record(
        cls,
        *,
        action: ExecutionAction,
        success: bool,
        reason: str,
        started_at: float | None = None,
        plugin_module: str | None = None,
        plugin_name: str | None = None,
        command_id: str | None = None,
        command: str | None = None,
        route_stage: str | None = None,
        session_id: str | None = None,
        message_preview: str | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> ExecutionObservation:
        start = started_at if started_at is not None else time.perf_counter()
        latency_ms = max(int((time.perf_counter() - start) * 1000), 0)
        observation = ExecutionObservation(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            action=action,
            success=bool(success),
            reason=normalize_message_text(reason) or EXECUTION_REASON_SUCCESS,
            latency_ms=latency_ms,
            plugin_module=normalize_message_text(plugin_module or ""),
            plugin_name=normalize_message_text(plugin_name or ""),
            command_id=normalize_message_text(command_id or ""),
            command=normalize_message_text(command or ""),
            route_stage=normalize_message_text(route_stage or ""),
            session_id=normalize_message_text(session_id or ""),
            message_preview=normalize_message_text(message_preview or "")[:120],
            token_usage=dict(token_usage or {}),
        )
        cls.configure()
        cls._records.append(observation)
        cls._record_command_feedback(observation)
        return observation

    @classmethod
    def _record_command_feedback(cls, observation: ExecutionObservation) -> None:
        command_id = normalize_message_text(observation.command_id)
        plugin_module = normalize_message_text(observation.plugin_module)
        if not command_id and not plugin_module:
            return

        delta = 0.0
        if observation.action == "execute":
            delta = 1.0 if observation.success else -1.2
        elif observation.action == "clarify":
            delta = -0.35
        elif observation.action == "usage" and observation.success:
            delta = 0.25
        elif not observation.success:
            delta = -0.5
        if not delta:
            return

        def clamp(value: float) -> float:
            return max(min(value, 36.0), -72.0)

        if command_id:
            cls._command_feedback[command_id] = clamp(
                cls._command_feedback.get(command_id, 0.0) + delta
            )
            session_id = normalize_message_text(observation.session_id)
            if session_id:
                session_bucket = cls._session_command_feedback.setdefault(
                    session_id, {}
                )
                session_bucket[command_id] = clamp(
                    session_bucket.get(command_id, 0.0) + delta
                )
                if len(session_bucket) > 256:
                    weakest = sorted(
                        session_bucket.items(),
                        key=lambda item: abs(item[1]),
                    )[:32]
                    for key, _ in weakest:
                        session_bucket.pop(key, None)
        if plugin_module:
            cls._module_feedback[plugin_module] = clamp(
                cls._module_feedback.get(plugin_module, 0.0) + delta * 0.35
            )

    @classmethod
    def snapshot(cls, limit: int = 200) -> dict[str, Any]:
        cls.configure()
        rows = list(cls._records)[-max(int(limit or 0), 1) :]
        if not rows:
            return {
                "total": 0,
                "action_counts": {},
                "reason_counts": {},
                "success_counts": {},
                "top_plugins": {},
                "avg_latency_ms": 0.0,
                "recent_failures": [],
            }
        action_counts = Counter(row.action for row in rows)
        reason_counts = Counter(row.reason for row in rows)
        success_counts = Counter("success" if row.success else "failed" for row in rows)
        top_plugins = Counter(
            row.plugin_name or row.plugin_module
            for row in rows
            if row.action == "execute" and (row.plugin_name or row.plugin_module)
        )
        avg_latency_ms = sum(row.latency_ms for row in rows) / len(rows)
        recent_failures = [row.to_dict() for row in rows if not row.success][-8:]
        return {
            "total": len(rows),
            "action_counts": dict(action_counts),
            "reason_counts": dict(reason_counts),
            "success_counts": dict(success_counts),
            "top_plugins": dict(top_plugins.most_common(8)),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "recent_failures": recent_failures,
        }

    @classmethod
    def clear(cls) -> None:
        cls._records.clear()
        cls._command_feedback.clear()
        cls._session_command_feedback.clear()
        cls._module_feedback.clear()

    @classmethod
    def command_feedback_score(
        cls,
        *,
        command_id: str | None = None,
        session_id: str | None = None,
        plugin_module: str | None = None,
    ) -> float:
        score = 0.0
        normalized_command_id = normalize_message_text(command_id or "")
        normalized_session_id = normalize_message_text(session_id or "")
        normalized_module = normalize_message_text(plugin_module or "")
        if normalized_command_id:
            score += cls._command_feedback.get(normalized_command_id, 0.0)
            if normalized_session_id:
                score += cls._session_command_feedback.get(
                    normalized_session_id, {}
                ).get(normalized_command_id, 0.0)
        if normalized_module:
            score += cls._module_feedback.get(normalized_module, 0.0)
        return max(min(score, 48.0), -96.0)


def start_execution_observation(**kwargs: Any) -> ExecutionFrame:
    return ExecutionObserver.start(**kwargs)


def record_execution_observation(**kwargs: Any) -> ExecutionObservation:
    return ExecutionObserver.record(**kwargs)


def get_execution_observer_snapshot(limit: int = 200) -> dict[str, Any]:
    return ExecutionObserver.snapshot(limit=limit)


def get_command_feedback_score(
    *,
    command_id: str | None = None,
    session_id: str | None = None,
    plugin_module: str | None = None,
) -> float:
    return ExecutionObserver.command_feedback_score(
        command_id=command_id,
        session_id=session_id,
        plugin_module=plugin_module,
    )


def render_execution_observer_summary(limit: int = 200) -> str:
    payload = get_execution_observer_snapshot(limit=limit)
    if payload["total"] <= 0:
        return "暂无 ChatInter 执行观测数据。"
    lines = [
        f"ChatInter 执行最近 {payload['total']} 条",
        "action: "
        + ", ".join(f"{k}={v}" for k, v in sorted(payload["action_counts"].items())),
        "reason: "
        + ", ".join(f"{k}={v}" for k, v in sorted(payload["reason_counts"].items())),
        f"avg_latency_ms={payload['avg_latency_ms']}",
    ]
    top_plugins = payload.get("top_plugins") or {}
    if top_plugins:
        lines.append(
            "top_plugins: " + ", ".join(f"{k}={v}" for k, v in top_plugins.items())
        )
    recent_failures = payload.get("recent_failures") or []
    if recent_failures:
        lines.append("recent_failures:")
        for item in recent_failures[-5:]:
            lines.append(
                f"- {item['timestamp']} {item['action']} {item['reason']} "
                f"{item['plugin_name'] or item['plugin_module'] or '-'} "
                f"| {item['message_preview']}"
            )
    return "\n".join(lines)


__all__ = [
    "EXECUTION_REASON_CANCELLED",
    "EXECUTION_REASON_CHAT_COMPLETED",
    "EXECUTION_REASON_CLARIFY_REQUESTED",
    "EXECUTION_REASON_ERROR",
    "EXECUTION_REASON_INVALID_COMMAND",
    "EXECUTION_REASON_LLM_ERROR",
    "EXECUTION_REASON_MISSING_IMAGE",
    "EXECUTION_REASON_MISSING_PARAMS",
    "EXECUTION_REASON_MISSING_REPLY",
    "EXECUTION_REASON_MISSING_TEXT",
    "EXECUTION_REASON_PERMISSION_DENIED",
    "EXECUTION_REASON_PLUGIN_NOT_LOADED",
    "EXECUTION_REASON_REROUTE_FAILED",
    "EXECUTION_REASON_ROUTE_SUCCESS",
    "EXECUTION_REASON_SUCCESS",
    "EXECUTION_REASON_TIMEOUT",
    "EXECUTION_REASON_USAGE_REPLIED",
    "ExecutionAction",
    "ExecutionFrame",
    "ExecutionObservation",
    "ExecutionObserver",
    "get_command_feedback_score",
    "get_execution_observer_snapshot",
    "record_execution_observation",
    "render_execution_observer_summary",
    "start_execution_observation",
]
