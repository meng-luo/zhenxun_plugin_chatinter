"""Unified feedback store for ChatInter.

This module is the single place that turns recent user follow-ups and execution
outcomes into bounded in-memory feedback.  It keeps feedback observational:
selection/execution still happens in the native tool chain.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import re
import time
from types import SimpleNamespace
from typing import Any, ClassVar, Literal, cast

from .feedback_keys import (
    FEEDBACK_REASON_DIRECT_TARGET_REQUIRED,
    FEEDBACK_REASON_FUZZY_CLARIFY,
    FEEDBACK_REASON_MISSING_PARAMS,
    FEEDBACK_REASON_REROUTE_FAILED,
    FEEDBACK_REASON_ROUTE_SUCCESS,
    FEEDBACK_REASON_SELF_ONLY_BLOCKED,
    FEEDBACK_REASON_TARGET_REQUIRED,
)
from .log_compat import logger
from .persistence import read_json, state_path, utc_now_iso, write_json
from .route_text import normalize_message_text

FeedbackDomain = Literal["chat", "plugin", "followup", "runtime"]
FeedbackKind = Literal[
    "chat_completed",
    "chat_rewritten",
    "chat_empty",
    "user_corrected",
    "user_thanks",
    "followup_same_topic",
    "route_success",
    "reroute_failed",
    "missing_params",
    "target_required",
    "self_only_blocked",
    "fuzzy_target_clarify",
    "direct_target_required",
    "route_user_corrected",
    "route_confirmed",
    "runtime_guardrail",
]

_CORRECTION_HINTS = (
    "不是",
    "不对",
    "错了",
    "你理解错",
    "不是这个",
    "不是这个意思",
    "我不是说",
)
_THANKS_HINTS = ("谢谢", "感谢", "对的", "就是这个", "可以的", "懂了")
_EXECUTION_REASON_ROUTE_CONFIRMED = "route_confirmed"
_EXECUTION_REASON_ROUTE_USER_CORRECTED = "route_user_corrected"
_EXECUTION_REASON_MISSING_IMAGE = "missing_image"
_EXECUTION_REASON_MISSING_REPLY = "missing_reply"
_EXECUTION_REASON_MISSING_TEXT = "missing_text"
_EXECUTION_REASON_CLARIFY_REQUESTED = "clarify_requested"
_EXECUTION_REASON_PERMISSION_DENIED = "permission_denied"
_EXECUTION_REASON_PLUGIN_NOT_LOADED = "plugin_not_loaded"
_EXECUTION_REASON_INVALID_COMMAND = "invalid_command"
_EXECUTION_REASON_TIMEOUT = "timeout"
_EXECUTION_REASON_LLM_ERROR = "llm_error"
_EXECUTION_REASON_CANCELLED = "cancelled"
_EXECUTION_REASON_ERROR = "error"
_EXECUTION_REASON_PLUGIN_NO_VISIBLE_OUTPUT = "plugin_completed_without_visible_output"
_PLUGIN_FEEDBACK_REWARD = {
    FEEDBACK_REASON_ROUTE_SUCCESS: 1.0,
    FEEDBACK_REASON_MISSING_PARAMS: -0.35,
    FEEDBACK_REASON_TARGET_REQUIRED: -0.40,
    FEEDBACK_REASON_SELF_ONLY_BLOCKED: -0.55,
    FEEDBACK_REASON_REROUTE_FAILED: -0.45,
    FEEDBACK_REASON_FUZZY_CLARIFY: -0.20,
    FEEDBACK_REASON_DIRECT_TARGET_REQUIRED: -0.30,
}
_CAPABILITY_FEEDBACK_PATH = state_path("capability_feedback.json")
_CAPABILITY_FEEDBACK_VERSION = "chatinter.capability_feedback.v1"
_TRAJECTORY_FEEDBACK_PATH = state_path("trajectory_feedback.json")
_TRAJECTORY_FEEDBACK_VERSION = "chatinter.trajectory_feedback.v1"
_LONGTERM_FEEDBACK_TTL_SECONDS = 90 * 24 * 3600
_CONTEXT_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,6}", re.I)
_PLACEHOLDER_CONTEXT_PATTERN = re.compile(
    r"\[@[^\]]+\]|\[image(?:#\d+)?\]|\[reply:[^\]]+\]",
    re.I,
)
_PARAM_FAILURE_REASONS = {
    _EXECUTION_REASON_MISSING_IMAGE,
    _EXECUTION_REASON_MISSING_REPLY,
    _EXECUTION_REASON_MISSING_TEXT,
    FEEDBACK_REASON_MISSING_PARAMS,
    _EXECUTION_REASON_CLARIFY_REQUESTED,
    FEEDBACK_REASON_TARGET_REQUIRED,
    FEEDBACK_REASON_SELF_ONLY_BLOCKED,
    FEEDBACK_REASON_FUZZY_CLARIFY,
    FEEDBACK_REASON_DIRECT_TARGET_REQUIRED,
}
_FALSE_TRIGGER_REASONS = {
    _EXECUTION_REASON_ROUTE_USER_CORRECTED,
    FEEDBACK_REASON_REROUTE_FAILED,
}


@dataclass(frozen=True)
class FeedbackRecord:
    timestamp: float
    domain: FeedbackDomain
    session_id: str
    kind: FeedbackKind
    message_preview: str = ""
    reply_preview: str = ""
    plugin_module: str = ""
    plugin_name: str = ""
    command_id: str = ""
    command: str = ""
    route_stage: str = ""
    success: bool | None = None
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = round(self.timestamp, 3)
        return payload


@dataclass(frozen=True)
class _LastChatOutcome:
    timestamp: float
    message: str
    reply: str


@dataclass(frozen=True)
class _LastPluginOutcome:
    timestamp: float
    session_id: str
    command_id: str
    slots: dict[str, str] = field(default_factory=dict)
    plugin_module: str = ""
    plugin_name: str = ""
    command: str = ""
    route_stage: str = ""
    success: bool = False
    reason: str = ""


@dataclass(frozen=True)
class _CommandSuccessExample:
    timestamp: float
    message: str
    command_id: str
    slots: dict[str, str]
    rendered_command: str
    plugin_module: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "message": self.message[:180],
            "command_id": self.command_id,
            "slots": dict(self.slots),
            "rendered_command": self.rendered_command,
            "plugin_module": self.plugin_module,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "_CommandSuccessExample | None":
        if not isinstance(payload, dict):
            return None
        command_id = normalize_message_text(str(payload.get("command_id", "") or ""))
        if not command_id:
            return None
        return cls(
            timestamp=_safe_float(payload.get("timestamp")),
            message=normalize_message_text(str(payload.get("message", "") or "")),
            command_id=command_id,
            slots=_normalize_slots(payload.get("slots") or {}),
            rendered_command=normalize_message_text(
                str(payload.get("rendered_command", "") or "")
            ),
            plugin_module=normalize_message_text(
                str(payload.get("plugin_module", "") or "")
            ),
        )


@dataclass
class _ReliabilityStats:
    success_count: float = 0.0
    failure_count: float = 0.0
    false_trigger_count: float = 0.0
    correction_count: float = 0.0
    param_failure_count: float = 0.0
    latency_total_ms: float = 0.0
    latency_count: float = 0.0
    last_updated: float = 0.0

    @property
    def total_count(self) -> float:
        return self.success_count + self.failure_count

    @property
    def avg_latency_ms(self) -> float:
        if self.latency_count <= 0:
            return 0.0
        return self.latency_total_ms / self.latency_count

    def add(
        self,
        *,
        success: float = 0.0,
        failure: float = 0.0,
        false_trigger: float = 0.0,
        correction: float = 0.0,
        param_failure: float = 0.0,
        latency_ms: float = 0.0,
        now: float,
        weight: float = 1.0,
    ) -> None:
        self.success_count += max(float(success or 0.0), 0.0) * weight
        self.failure_count += max(float(failure or 0.0), 0.0) * weight
        self.false_trigger_count += max(float(false_trigger or 0.0), 0.0) * weight
        self.correction_count += max(float(correction or 0.0), 0.0) * weight
        self.param_failure_count += max(float(param_failure or 0.0), 0.0) * weight
        latency = max(float(latency_ms or 0.0), 0.0)
        if latency:
            self.latency_total_ms += latency * weight
            self.latency_count += weight
        self.last_updated = now

    def scaled(self, factor: float) -> "_ReliabilityStats":
        return _ReliabilityStats(
            success_count=self.success_count * factor,
            failure_count=self.failure_count * factor,
            false_trigger_count=self.false_trigger_count * factor,
            correction_count=self.correction_count * factor,
            param_failure_count=self.param_failure_count * factor,
            latency_total_ms=self.latency_total_ms * factor,
            latency_count=self.latency_count * factor,
            last_updated=self.last_updated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_count": round(self.success_count, 6),
            "failure_count": round(self.failure_count, 6),
            "false_trigger_count": round(self.false_trigger_count, 6),
            "correction_count": round(self.correction_count, 6),
            "param_failure_count": round(self.param_failure_count, 6),
            "latency_total_ms": round(self.latency_total_ms, 3),
            "latency_count": round(self.latency_count, 6),
            "last_updated": round(self.last_updated, 3),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "_ReliabilityStats":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            success_count=_safe_float(payload.get("success_count")),
            failure_count=_safe_float(payload.get("failure_count")),
            false_trigger_count=_safe_float(payload.get("false_trigger_count")),
            correction_count=_safe_float(payload.get("correction_count")),
            param_failure_count=_safe_float(payload.get("param_failure_count")),
            latency_total_ms=_safe_float(payload.get("latency_total_ms")),
            latency_count=_safe_float(payload.get("latency_count")),
            last_updated=_safe_float(payload.get("last_updated")),
        )


@dataclass(frozen=True)
class CommandFeedbackProfile:
    command_id: str = ""
    plugin_module: str = ""
    feedback_score: float = 0.0
    reliability_score: float = 0.0
    false_trigger_score: float = 0.0
    success_count: float = 0.0
    failure_count: float = 0.0
    false_trigger_count: float = 0.0
    correction_count: float = 0.0
    param_failure_count: float = 0.0
    avg_latency_ms: float = 0.0
    total_count: float = 0.0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    false_trigger_rate: float = 0.0
    param_failure_rate: float = 0.0
    latency_score: float = 0.0
    param_failure_score: float = 0.0
    low_reliability: bool = False
    high_reliability: bool = False

    @property
    def exposure_score(self) -> float:
        return (
            self.feedback_score * 0.35
            + self.reliability_score
            + self.false_trigger_score
            + self.param_failure_score
            + self.latency_score
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "plugin_module": self.plugin_module,
            "feedback_score": round(self.feedback_score, 3),
            "reliability_score": round(self.reliability_score, 3),
            "false_trigger_score": round(self.false_trigger_score, 3),
            "success_count": round(self.success_count, 3),
            "failure_count": round(self.failure_count, 3),
            "false_trigger_count": round(self.false_trigger_count, 3),
            "correction_count": round(self.correction_count, 3),
            "param_failure_count": round(self.param_failure_count, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_count": round(self.total_count, 3),
            "success_rate": round(self.success_rate, 3),
            "failure_rate": round(self.failure_rate, 3),
            "false_trigger_rate": round(self.false_trigger_rate, 3),
            "param_failure_rate": round(self.param_failure_rate, 3),
            "latency_score": round(self.latency_score, 3),
            "param_failure_score": round(self.param_failure_score, 3),
            "low_reliability": self.low_reliability,
            "high_reliability": self.high_reliability,
        }


class FeedbackStore:
    _records: ClassVar[deque[FeedbackRecord]] = deque(maxlen=600)
    _last_chat: ClassVar[dict[str, _LastChatOutcome]] = {}
    _last_plugin: ClassVar[dict[str, _LastPluginOutcome]] = {}
    _command_feedback: ClassVar[dict[str, float]] = {}
    _command_feedback_ts: ClassVar[dict[str, float]] = {}
    _session_command_feedback: ClassVar[dict[str, dict[str, float]]] = {}
    _session_command_feedback_ts: ClassVar[dict[str, dict[str, float]]] = {}
    _module_feedback: ClassVar[dict[str, float]] = {}
    _module_feedback_ts: ClassVar[dict[str, float]] = {}
    _reason_feedback: ClassVar[dict[str, dict[str, float]]] = {}
    _command_reliability: ClassVar[dict[str, _ReliabilityStats]] = {}
    _session_command_reliability: ClassVar[dict[str, dict[str, _ReliabilityStats]]] = {}
    _module_reliability: ClassVar[dict[str, _ReliabilityStats]] = {}
    _context_command_reliability: ClassVar[dict[str, dict[str, _ReliabilityStats]]] = {}
    _command_success_examples: ClassVar[dict[str, list[_CommandSuccessExample]]] = {}
    _longterm_loaded: ClassVar[bool] = False
    _longterm_dirty: ClassVar[bool] = False
    _last_longterm_save: ClassVar[float] = 0.0
    _longterm_save_interval: ClassVar[float] = 8.0
    _chat_ttl: ClassVar[float] = 1800.0
    _plugin_ttl: ClassVar[float] = 4 * 60.0
    _execution_feedback_ttl: ClassVar[float] = 6 * 3600.0
    _max_last_plugin: ClassVar[int] = 256
    _max_command_feedback: ClassVar[int] = 2048
    _max_session_feedback: ClassVar[int] = 512
    _max_module_feedback: ClassVar[int] = 1024
    _max_context_feedback: ClassVar[int] = 2048
    _max_success_examples_per_command: ClassVar[int] = 12

    @classmethod
    def record_execution_observation(cls, observation: Any) -> None:
        cls._ensure_longterm_loaded()
        observation = _normalize_observation_for_feedback(observation)
        command_id = normalize_message_text(getattr(observation, "command_id", ""))
        plugin_module = normalize_message_text(
            getattr(observation, "plugin_module", "")
        )
        if not command_id and not plugin_module:
            return

        now = time.monotonic()
        cls._prune(now)
        cls._record_reliability_observation(observation, now=now)
        cls._record_context_feedback_observation(observation, now=now)
        cls._maybe_save_longterm(now)
        delta = cls._execution_feedback_delta(observation)
        if not delta:
            return
        selected_rank = int(getattr(observation, "selected_rank", 0) or 0)
        success = bool(getattr(observation, "success", False))
        if selected_rank > 1 and success:
            delta += min(selected_rank, 8) * 0.08
        if selected_rank == 1 and not success:
            delta -= 0.15

        if command_id:
            cls._command_feedback[command_id] = _clamp_command_feedback(
                cls._command_feedback.get(command_id, 0.0) + delta
            )
            cls._command_feedback_ts[command_id] = now
            session_id = normalize_message_text(getattr(observation, "session_id", ""))
            if session_id:
                session_bucket = cls._session_command_feedback.setdefault(
                    session_id, {}
                )
                session_ts_bucket = cls._session_command_feedback_ts.setdefault(
                    session_id, {}
                )
                session_bucket[command_id] = _clamp_command_feedback(
                    session_bucket.get(command_id, 0.0) + delta
                )
                session_ts_bucket[command_id] = now
                if len(session_bucket) > 256:
                    weakest = sorted(
                        session_bucket.items(),
                        key=lambda item: abs(item[1]),
                    )[:32]
                    for key, _ in weakest:
                        session_bucket.pop(key, None)
                        session_ts_bucket.pop(key, None)

        if plugin_module:
            reason = normalize_message_text(getattr(observation, "reason", ""))
            module_weight = 0.35
            if not command_id and reason in {
                FEEDBACK_REASON_REROUTE_FAILED,
            }:
                module_weight = 1.0
            cls._module_feedback[plugin_module] = _clamp_command_feedback(
                cls._module_feedback.get(plugin_module, 0.0) + delta * module_weight
            )
            cls._module_feedback_ts[plugin_module] = now

        reason = normalize_message_text(getattr(observation, "reason", ""))
        if command_id and reason:
            reason_bucket = cls._reason_feedback.setdefault(reason, {})
            reason_bucket[command_id] = _clamp_command_feedback(
                reason_bucket.get(command_id, 0.0) + delta
            )
            if len(reason_bucket) > 256:
                weakest = sorted(reason_bucket.items(), key=lambda item: abs(item[1]))[
                    :32
                ]
                for key, _ in weakest:
                    reason_bucket.pop(key, None)

    @classmethod
    def command_feedback_score(
        cls,
        *,
        command_id: str | None = None,
        session_id: str | None = None,
        plugin_module: str | None = None,
    ) -> float:
        cls._ensure_longterm_loaded()
        score = 0.0
        now = time.monotonic()
        cls._prune(now)
        normalized_command_id = normalize_message_text(command_id or "")
        normalized_session_id = normalize_message_text(session_id or "")
        normalized_module = normalize_message_text(plugin_module or "")
        if normalized_command_id:
            score += cls._fresh_feedback_value(
                cls._command_feedback,
                cls._command_feedback_ts,
                normalized_command_id,
                now,
            )
            if normalized_session_id:
                score += cls._fresh_feedback_value(
                    cls._session_command_feedback.get(normalized_session_id, {}),
                    cls._session_command_feedback_ts.get(normalized_session_id, {}),
                    normalized_command_id,
                    now,
                )
        if normalized_module:
            score += cls._fresh_feedback_value(
                cls._module_feedback,
                cls._module_feedback_ts,
                normalized_module,
                now,
            )
        return max(min(score, 48.0), -96.0)

    @classmethod
    def command_feedback_profile(
        cls,
        *,
        command_id: str | None = None,
        session_id: str | None = None,
        plugin_module: str | None = None,
    ) -> CommandFeedbackProfile:
        cls._ensure_longterm_loaded()
        now = time.monotonic()
        cls._prune(now)
        normalized_command_id = normalize_message_text(command_id or "")
        normalized_session_id = normalize_message_text(session_id or "")
        normalized_module = normalize_message_text(plugin_module or "")
        combined = _ReliabilityStats()

        if normalized_command_id:
            _merge_reliability_stats(
                combined,
                cls._fresh_stats(
                    cls._command_reliability.get(normalized_command_id),
                    now,
                ),
                weight=1.0,
            )
            if normalized_session_id:
                _merge_reliability_stats(
                    combined,
                    cls._fresh_stats(
                        cls._session_command_reliability.get(
                            normalized_session_id,
                            {},
                        ).get(normalized_command_id),
                        now,
                    ),
                    weight=1.35,
                )

        if normalized_module:
            _merge_reliability_stats(
                combined,
                cls._fresh_stats(
                    cls._module_reliability.get(normalized_module),
                    now,
                ),
                weight=0.35,
            )

        feedback_score = cls.command_feedback_score(
            command_id=normalized_command_id,
            session_id=normalized_session_id,
            plugin_module=normalized_module,
        )
        return _profile_from_stats(
            command_id=normalized_command_id,
            plugin_module=normalized_module,
            feedback_score=feedback_score,
            stats=combined,
        )

    @classmethod
    def record_chat(
        cls,
        *,
        session_id: str | None,
        kind: FeedbackKind,
        message_text: str = "",
        reply_text: str = "",
        weight: float = 0.0,
    ) -> None:
        normalized_session = normalize_message_text(session_id or "")
        if not normalized_session:
            return
        now = time.monotonic()
        normalized_message = normalize_message_text(message_text)
        normalized_reply = normalize_message_text(reply_text)
        cls._records.append(
            FeedbackRecord(
                timestamp=now,
                domain="chat",
                session_id=normalized_session,
                kind=kind,
                message_preview=normalized_message[:120],
                reply_preview=normalized_reply[:120],
                weight=float(weight or 0.0),
            )
        )
        if kind in {"chat_completed", "chat_rewritten"}:
            cls._last_chat[normalized_session] = _LastChatOutcome(
                timestamp=now,
                message=normalized_message,
                reply=normalized_reply,
            )
        cls._record_memory_feedback(
            session_id=normalized_session,
            kind=kind,
            weight=float(weight or 0.0),
        )
        cls._prune(now)

    @classmethod
    def record_runtime_guardrail(
        cls,
        *,
        session_id: str | None,
        reason: str,
        severity: str,
        message_text: str = "",
        command_id: str = "",
        tool_name: str = "",
        weight: float = -0.12,
    ) -> None:
        """Record runtime loop protection in the same bounded feedback store."""

        normalized_session = normalize_message_text(session_id or "")
        if not normalized_session:
            return
        now = time.monotonic()
        normalized_command_id = normalize_message_text(command_id)
        normalized_tool = normalize_message_text(tool_name)
        normalized_reason = normalize_message_text(reason) or "runtime_guardrail"
        normalized_severity = normalize_message_text(severity) or "light"
        cls._records.append(
            FeedbackRecord(
                timestamp=now,
                domain="runtime",
                session_id=normalized_session,
                kind="runtime_guardrail",
                message_preview=normalize_message_text(message_text)[:120],
                command_id=normalized_command_id,
                command=normalized_tool,
                route_stage=f"runtime_guardrail:{normalized_severity}",
                success=False,
                weight=float(weight or 0.0),
            )
        )
        if normalized_command_id:
            delta = _clamp_command_feedback(float(weight or 0.0))
            cls._command_feedback[normalized_command_id] = _clamp_command_feedback(
                cls._command_feedback.get(normalized_command_id, 0.0) + delta
            )
            cls._command_feedback_ts[normalized_command_id] = now
            reason_bucket = cls._reason_feedback.setdefault(normalized_reason, {})
            reason_bucket[normalized_command_id] = _clamp_command_feedback(
                reason_bucket.get(normalized_command_id, 0.0) + delta
            )
        if normalized_tool and not normalized_command_id:
            reason_bucket = cls._reason_feedback.setdefault(normalized_reason, {})
            reason_bucket[normalized_tool] = _clamp_command_feedback(
                reason_bucket.get(normalized_tool, 0.0) + float(weight or 0.0)
            )
        cls._prune(now)

    @classmethod
    async def record_plugin_outcome(
        cls,
        *,
        session_id: str | None,
        message_text: str,
        route_result: Any | None,
        modules: set[str] | list[str] | tuple[str, ...],
        route_command: str = "",
        success: bool,
        reason: str,
        image_missing: int = 0,
        text_missing: int = 0,
        allow_at: bool | None = None,
    ) -> None:
        normalized_session = normalize_message_text(session_id or "")
        if not normalized_session or route_result is None:
            return

        normalized_reason = normalize_message_text(reason) or (
            FEEDBACK_REASON_ROUTE_SUCCESS if success else FEEDBACK_REASON_REROUTE_FAILED
        )
        decision = getattr(route_result, "decision", None)
        plugin_module = normalize_message_text(getattr(decision, "plugin_module", ""))
        plugin_name = normalize_message_text(getattr(decision, "plugin_name", ""))
        command_id = normalize_message_text(getattr(route_result, "command_id", ""))
        command = normalize_message_text(
            route_command or getattr(decision, "command", "")
        )
        route_stage = normalize_message_text(getattr(route_result, "stage", ""))
        if not command_id and not plugin_module:
            return

        now = time.monotonic()
        cls._last_plugin[normalized_session] = _LastPluginOutcome(
            timestamp=now,
            session_id=normalized_session,
            command_id=command_id,
            slots=_normalize_slots(getattr(route_result, "slots", {}) or {}),
            plugin_module=plugin_module,
            plugin_name=plugin_name,
            command=command,
            route_stage=route_stage,
            success=bool(success),
            reason=normalized_reason,
        )
        cls._records.append(
            FeedbackRecord(
                timestamp=now,
                domain="plugin",
                session_id=normalized_session,
                kind=_coerce_feedback_kind(normalized_reason),
                message_preview=normalize_message_text(message_text)[:120],
                plugin_module=plugin_module,
                plugin_name=plugin_name,
                command_id=command_id,
                command=command,
                route_stage=route_stage,
                success=bool(success),
                weight=_plugin_feedback_reward(normalized_reason),
            )
        )
        cls._record_context_feedback(
            command_id=command_id,
            plugin_module=plugin_module,
            message_text=message_text,
            rendered_command=command,
            slots=_normalize_slots(getattr(route_result, "slots", {}) or {}),
            success=bool(success),
            reason=normalized_reason,
            now=now,
        )
        cls._maybe_save_longterm(now)
        cls._prune(now)
        await cls._record_plugin_rag_feedback(
            session_id=normalized_session,
            modules=modules,
            reason=normalized_reason,
            route_message=message_text,
            route_command=command,
            image_missing=image_missing,
            text_missing=text_missing,
            allow_at=allow_at,
        )

    @classmethod
    def inspect_user_followup(
        cls,
        *,
        session_id: str | None,
        message_text: str,
    ) -> FeedbackKind | None:
        normalized_session = normalize_message_text(session_id or "")
        normalized_message = normalize_message_text(message_text)
        if not normalized_session or not normalized_message:
            return None
        now = time.monotonic()
        cls._prune(now)

        plugin_feedback = cls._inspect_plugin_followup(
            normalized_session,
            normalized_message,
            now,
        )
        if plugin_feedback is not None:
            return plugin_feedback

        if any(hint in normalized_message for hint in _CORRECTION_HINTS):
            cls.record_chat(
                session_id=normalized_session,
                kind="user_corrected",
                message_text=normalized_message,
                weight=-1.0,
            )
            return "user_corrected"
        if any(hint in normalized_message for hint in _THANKS_HINTS):
            cls.record_chat(
                session_id=normalized_session,
                kind="user_thanks",
                message_text=normalized_message,
                weight=0.45,
            )
            return "user_thanks"

        last_chat = cls._last_chat.get(normalized_session)
        if last_chat is None or now - last_chat.timestamp > 240:
            return None
        if _shared_token_count(last_chat.message, normalized_message) >= 2:
            cls.record_chat(
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
        cls._last_plugin.clear()
        cls.clear_execution_feedback()
        try:
            from .memory_feedback_reranker import MemoryFeedbackReranker

            MemoryFeedbackReranker.clear()
        except Exception:
            pass

    @classmethod
    def clear_execution_feedback(cls) -> None:
        cls._command_feedback.clear()
        cls._command_feedback_ts.clear()
        cls._session_command_feedback.clear()
        cls._session_command_feedback_ts.clear()
        cls._module_feedback.clear()
        cls._module_feedback_ts.clear()
        cls._reason_feedback.clear()
        cls._command_reliability.clear()
        cls._session_command_reliability.clear()
        cls._module_reliability.clear()
        cls._context_command_reliability.clear()
        cls._command_success_examples.clear()
        cls._longterm_loaded = True
        cls._longterm_dirty = True
        cls._save_longterm(time.monotonic())

    @classmethod
    def _inspect_plugin_followup(
        cls,
        session_id: str,
        message_text: str,
        now: float,
    ) -> FeedbackKind | None:
        last = cls._last_plugin.get(session_id)
        if last is None or now - last.timestamp > cls._plugin_ttl:
            return None
        if any(hint in message_text for hint in _CORRECTION_HINTS):
            cls._record_plugin_followup(
                last,
                success=False,
                reason=_EXECUTION_REASON_ROUTE_USER_CORRECTED,
                message_text=message_text,
            )
            cls._last_plugin.pop(session_id, None)
            return "route_user_corrected"
        if any(hint in message_text for hint in _THANKS_HINTS):
            cls._record_plugin_followup(
                last,
                success=True,
                reason=_EXECUTION_REASON_ROUTE_CONFIRMED,
                message_text=message_text,
            )
            cls._last_plugin.pop(session_id, None)
            return "route_confirmed"
        return None

    @classmethod
    def _record_plugin_followup(
        cls,
        last: _LastPluginOutcome,
        *,
        success: bool,
        reason: str,
        message_text: str,
    ) -> None:
        now = time.monotonic()
        cls._records.append(
            FeedbackRecord(
                timestamp=now,
                domain="followup",
                session_id=last.session_id,
                kind=_coerce_feedback_kind(reason),
                message_preview=normalize_message_text(message_text)[:120],
                plugin_module=last.plugin_module,
                plugin_name=last.plugin_name,
                command_id=last.command_id,
                command=last.command,
                route_stage=last.route_stage,
                success=success,
                weight=0.35 if success else -1.25,
            )
        )
        from .execution_observer import record_execution_observation

        record_execution_observation(
            action="execute",
            success=success,
            reason=reason,
            plugin_module=last.plugin_module,
            plugin_name=last.plugin_name,
            command_id=last.command_id,
            command=last.command,
            route_stage=last.route_stage or "feedback",
            session_id=last.session_id,
            message_preview=message_text,
        )

    @classmethod
    async def _record_plugin_rag_feedback(
        cls,
        *,
        session_id: str,
        modules: set[str] | list[str] | tuple[str, ...],
        reason: str,
        route_message: str,
        route_command: str,
        image_missing: int = 0,
        text_missing: int = 0,
        allow_at: bool | None = None,
    ) -> None:
        normalized_modules = {
            normalize_message_text(str(module or ""))
            for module in modules
            if normalize_message_text(str(module or ""))
        }
        if not normalized_modules:
            return
        slot_feedback = _build_plugin_slot_feedback(
            reason=reason,
            route_message=route_message,
            route_command=route_command,
            image_missing=image_missing,
            text_missing=text_missing,
            allow_at=allow_at,
        )
        try:
            from .knowledge_rag import PluginRAGService

            await PluginRAGService.update_session_feedback(
                session_id=session_id,
                modules=normalized_modules,
                reward=_plugin_feedback_reward(reason),
                reason=reason,
                slot_feedback=slot_feedback or None,
            )
        except Exception as exc:
            logger.debug(f"更新 ChatInter 统一反馈失败: {exc}")

    @classmethod
    def _record_memory_feedback(
        cls,
        *,
        session_id: str,
        kind: FeedbackKind,
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

    @classmethod
    def _execution_feedback_delta(cls, observation: Any) -> float:
        reason = normalize_message_text(getattr(observation, "reason", ""))
        action = normalize_message_text(getattr(observation, "action", ""))
        success = bool(getattr(observation, "success", False))
        if reason == _EXECUTION_REASON_ROUTE_CONFIRMED:
            return 0.35
        if reason == _EXECUTION_REASON_ROUTE_USER_CORRECTED:
            return -1.25
        if action == "execute" and success:
            return 1.0
        if action == "usage" and success:
            return 0.25
        if action == "clarify":
            return -0.18
        if reason in _PARAM_FAILURE_REASONS:
            return -0.35
        if reason == _EXECUTION_REASON_PERMISSION_DENIED:
            return -0.04
        if reason == _EXECUTION_REASON_PLUGIN_NOT_LOADED:
            return -0.2
        if reason in {
            FEEDBACK_REASON_REROUTE_FAILED,
            _EXECUTION_REASON_ROUTE_USER_CORRECTED,
        }:
            return -1.4
        if reason in {
            _EXECUTION_REASON_INVALID_COMMAND,
            _EXECUTION_REASON_PLUGIN_NO_VISIBLE_OUTPUT,
        }:
            return -0.18
        if reason in {
            _EXECUTION_REASON_TIMEOUT,
            _EXECUTION_REASON_LLM_ERROR,
            _EXECUTION_REASON_CANCELLED,
            _EXECUTION_REASON_ERROR,
        }:
            return -0.45
        return -0.5 if not success else 0.0

    @classmethod
    def _record_reliability_observation(cls, observation: Any, *, now: float) -> None:
        command_id = normalize_message_text(getattr(observation, "command_id", ""))
        plugin_module = normalize_message_text(
            getattr(observation, "plugin_module", "")
        )
        if not command_id and not plugin_module:
            return
        success, failure, false_trigger, correction = _reliability_delta(observation)
        param_failure = _param_failure_delta(observation)
        latency_ms = _observation_latency_ms(observation)
        if not any(
            (success, failure, false_trigger, correction, param_failure, latency_ms)
        ):
            return

        if command_id:
            cls._command_reliability.setdefault(command_id, _ReliabilityStats()).add(
                success=success,
                failure=failure,
                false_trigger=false_trigger,
                correction=correction,
                param_failure=param_failure,
                latency_ms=latency_ms,
                now=now,
            )
            session_id = normalize_message_text(getattr(observation, "session_id", ""))
            if session_id:
                bucket = cls._session_command_reliability.setdefault(session_id, {})
                bucket.setdefault(command_id, _ReliabilityStats()).add(
                    success=success,
                    failure=failure,
                    false_trigger=false_trigger,
                    correction=correction,
                    param_failure=param_failure,
                    latency_ms=latency_ms,
                    now=now,
                )

        if plugin_module:
            module_weight = 0.55 if command_id else 1.0
            cls._module_reliability.setdefault(
                plugin_module,
                _ReliabilityStats(),
            ).add(
                success=success,
                failure=failure,
                false_trigger=false_trigger,
                correction=correction,
                param_failure=param_failure,
                latency_ms=latency_ms,
                now=now,
                weight=module_weight,
            )
        cls._longterm_dirty = True

    @classmethod
    def _record_context_feedback_observation(
        cls,
        observation: Any,
        *,
        now: float,
    ) -> None:
        command_id = normalize_message_text(getattr(observation, "command_id", ""))
        plugin_module = normalize_message_text(
            getattr(observation, "plugin_module", "")
        )
        if not command_id:
            return
        output = getattr(observation, "output", None)
        if not isinstance(output, dict):
            return
        message_text = normalize_message_text(str(output.get("task_text", "") or ""))
        slots = _normalize_slots(output.get("slots") or {})
        rendered_command = normalize_message_text(
            str(output.get("rendered_command", "") or "")
        )
        cls._record_context_feedback(
            command_id=command_id,
            plugin_module=plugin_module,
            message_text=message_text,
            rendered_command=rendered_command,
            slots=slots,
            success=bool(getattr(observation, "success", False)),
            reason=normalize_message_text(getattr(observation, "reason", "")),
            now=now,
        )

    @classmethod
    def _record_context_feedback(
        cls,
        *,
        command_id: str,
        plugin_module: str,
        message_text: str,
        rendered_command: str,
        slots: dict[str, str],
        success: bool,
        reason: str,
        now: float,
    ) -> None:
        command_id = normalize_message_text(command_id)
        if not command_id:
            return
        context_key = _context_key(message_text)
        if context_key:
            bucket = cls._context_command_reliability.setdefault(context_key, {})
            stats = bucket.setdefault(command_id, _ReliabilityStats())
            stats.add(
                success=1.0 if success else 0.0,
                failure=0.0 if success else 1.0,
                false_trigger=1.0 if reason in _FALSE_TRIGGER_REASONS else 0.0,
                param_failure=1.0 if reason in _PARAM_FAILURE_REASONS else 0.0,
                now=now,
            )
            if len(bucket) > 128:
                _trim_stats_map(bucket, 128)
        if success:
            example = _CommandSuccessExample(
                timestamp=now,
                message=normalize_message_text(message_text)[:180],
                command_id=command_id,
                slots=dict(slots or {}),
                rendered_command=normalize_message_text(rendered_command),
                plugin_module=normalize_message_text(plugin_module),
            )
            examples = cls._command_success_examples.setdefault(command_id, [])
            examples.append(example)
            deduped: dict[tuple[str, str], _CommandSuccessExample] = {}
            for item in examples:
                key = (item.message, item.rendered_command)
                previous = deduped.get(key)
                if previous is None or item.timestamp >= previous.timestamp:
                    deduped[key] = item
            cls._command_success_examples[command_id] = sorted(
                deduped.values(),
                key=lambda item: item.timestamp,
                reverse=True,
            )[: cls._max_success_examples_per_command]
        cls._longterm_dirty = True

    @classmethod
    def contextual_command_feedback_profile(
        cls,
        *,
        message_text: str,
        command_id: str | None = None,
    ) -> CommandFeedbackProfile:
        cls._ensure_longterm_loaded()
        normalized_command_id = normalize_message_text(command_id or "")
        context_key = _context_key(message_text)
        if not normalized_command_id or not context_key:
            return CommandFeedbackProfile(command_id=normalized_command_id)
        now = time.monotonic()
        cls._prune(now)
        stats = cls._fresh_stats(
            cls._context_command_reliability.get(context_key, {}).get(
                normalized_command_id
            ),
            now,
        )
        return _profile_from_stats(
            command_id=normalized_command_id,
            plugin_module="",
            feedback_score=0.0,
            stats=stats,
        )

    @classmethod
    def command_success_examples(
        cls,
        *,
        command_id: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        cls._ensure_longterm_loaded()
        normalized_command_id = normalize_message_text(command_id or "")
        if not normalized_command_id:
            return []
        now = time.monotonic()
        cls._prune(now)
        examples = cls._command_success_examples.get(normalized_command_id, [])
        return [item.to_dict() for item in examples[: max(1, min(int(limit or 8), 24))]]

    @classmethod
    def _ensure_longterm_loaded(cls) -> None:
        if cls._longterm_loaded:
            return
        payload = read_json(_CAPABILITY_FEEDBACK_PATH, {})
        if not isinstance(payload, dict):
            cls._longterm_loaded = True
            return
        try:
            now = time.monotonic()
            saved_monotonic = _safe_float(payload.get("saved_monotonic"))
            saved_wall_time = _safe_float(payload.get("saved_wall_time"))
            wall_now = time.time()
            cls._command_reliability.update(
                _load_stats_bucket(
                    payload.get("commands"),
                    saved_monotonic=saved_monotonic,
                    saved_wall_time=saved_wall_time,
                    now=now,
                    wall_now=wall_now,
                )
            )
            cls._module_reliability.update(
                _load_stats_bucket(
                    payload.get("modules"),
                    saved_monotonic=saved_monotonic,
                    saved_wall_time=saved_wall_time,
                    now=now,
                    wall_now=wall_now,
                )
            )
            cls._context_command_reliability.update(
                _load_nested_stats_bucket(
                    payload.get("contexts"),
                    saved_monotonic=saved_monotonic,
                    saved_wall_time=saved_wall_time,
                    now=now,
                    wall_now=wall_now,
                )
            )
            cls._command_success_examples.update(
                _load_success_examples(
                    payload.get("success_examples"),
                    saved_monotonic=saved_monotonic,
                    saved_wall_time=saved_wall_time,
                    now=now,
                    wall_now=wall_now,
                )
            )
            command_scores = _load_score_bucket(
                payload.get("scores"),
                payload.get("score_ts"),
                saved_monotonic=saved_monotonic,
                saved_wall_time=saved_wall_time,
                now=now,
                wall_now=wall_now,
            )
            cls._command_feedback.update(command_scores)
            cls._command_feedback_ts.update({key: now for key in command_scores})
            module_scores = _load_score_bucket(
                payload.get("module_scores"),
                payload.get("module_score_ts"),
                saved_monotonic=saved_monotonic,
                saved_wall_time=saved_wall_time,
                now=now,
                wall_now=wall_now,
            )
            cls._module_feedback.update(module_scores)
            cls._module_feedback_ts.update({key: now for key in module_scores})
        except Exception as exc:
            logger.debug(f"加载 ChatInter capability 反馈失败: {exc}")
        cls._longterm_loaded = True
        cls._prune(time.monotonic())

    @classmethod
    def _maybe_save_longterm(cls, now: float) -> None:
        if not cls._longterm_dirty:
            return
        if now - cls._last_longterm_save < cls._longterm_save_interval:
            return
        cls._save_longterm(now)

    @classmethod
    def _save_longterm(cls, now: float) -> None:
        cls._prune(now)
        payload = {
            "version": _CAPABILITY_FEEDBACK_VERSION,
            "saved_at": utc_now_iso(),
            "saved_monotonic": round(now, 3),
            "saved_wall_time": round(time.time(), 3),
            "ttl_seconds": _LONGTERM_FEEDBACK_TTL_SECONDS,
            "commands": _dump_stats_bucket(cls._command_reliability),
            "modules": _dump_stats_bucket(cls._module_reliability),
            "contexts": _dump_nested_stats_bucket(cls._context_command_reliability),
            "success_examples": _dump_success_examples(cls._command_success_examples),
            "scores": _dump_float_bucket(cls._command_feedback),
            "score_ts": _dump_float_bucket(cls._command_feedback_ts),
            "module_scores": _dump_float_bucket(cls._module_feedback),
            "module_score_ts": _dump_float_bucket(cls._module_feedback_ts),
        }
        try:
            write_json(_CAPABILITY_FEEDBACK_PATH, payload)
            cls._last_longterm_save = now
            cls._longterm_dirty = False
        except Exception as exc:
            logger.debug(f"保存 ChatInter capability 反馈失败: {exc}")

    @classmethod
    def _fresh_feedback_value(
        cls,
        values: dict[str, float],
        timestamps: dict[str, float],
        key: str,
        now: float,
    ) -> float:
        value = values.get(key, 0.0)
        if not value:
            return 0.0
        updated_at = timestamps.get(key, now)
        age = max(now - updated_at, 0.0)
        if age >= cls._execution_feedback_ttl:
            return 0.0
        freshness = max(0.25, 1.0 - age / cls._execution_feedback_ttl)
        return value * freshness

    @classmethod
    def _fresh_stats(
        cls,
        stats: _ReliabilityStats | None,
        now: float,
    ) -> _ReliabilityStats:
        if stats is None or stats.last_updated <= 0:
            return _ReliabilityStats()
        age = max(now - stats.last_updated, 0.0)
        if age >= _LONGTERM_FEEDBACK_TTL_SECONDS:
            return _ReliabilityStats()
        freshness = max(0.35, 1.0 - age / _LONGTERM_FEEDBACK_TTL_SECONDS)
        return stats.scaled(freshness)

    @classmethod
    def _prune(cls, now: float) -> None:
        expired_chat = [
            session
            for session, item in cls._last_chat.items()
            if now - item.timestamp > cls._chat_ttl
        ]
        for session in expired_chat:
            cls._last_chat.pop(session, None)

        expired_plugin = [
            session
            for session, item in cls._last_plugin.items()
            if now - item.timestamp > cls._plugin_ttl
        ]
        for session in expired_plugin:
            cls._last_plugin.pop(session, None)

        if len(cls._last_plugin) > cls._max_last_plugin:
            stale_sessions = sorted(
                cls._last_plugin.items(),
                key=lambda item: item[1].timestamp,
            )[:32]
            for session, _ in stale_sessions:
                cls._last_plugin.pop(session, None)

        expired_commands = [
            key
            for key, updated_at in cls._command_feedback_ts.items()
            if now - updated_at > cls._execution_feedback_ttl
        ]
        for key in expired_commands:
            cls._command_feedback.pop(key, None)
            cls._command_feedback_ts.pop(key, None)

        expired_modules = [
            key
            for key, updated_at in cls._module_feedback_ts.items()
            if now - updated_at > cls._execution_feedback_ttl
        ]
        for key in expired_modules:
            cls._module_feedback.pop(key, None)
            cls._module_feedback_ts.pop(key, None)

        expired_sessions: list[str] = []
        for session_id, bucket in list(cls._session_command_feedback.items()):
            ts_bucket = cls._session_command_feedback_ts.get(session_id, {})
            expired_keys = [
                key
                for key, updated_at in ts_bucket.items()
                if now - updated_at > cls._execution_feedback_ttl
            ]
            for key in expired_keys:
                bucket.pop(key, None)
                ts_bucket.pop(key, None)
            if not bucket:
                expired_sessions.append(session_id)
        for session_id in expired_sessions:
            cls._session_command_feedback.pop(session_id, None)
            cls._session_command_feedback_ts.pop(session_id, None)

        _trim_feedback_map(
            cls._command_feedback,
            cls._command_feedback_ts,
            cls._max_command_feedback,
        )
        _trim_feedback_map(
            cls._module_feedback,
            cls._module_feedback_ts,
            cls._max_module_feedback,
        )
        _trim_stats_map(cls._command_reliability, cls._max_command_feedback)
        _trim_stats_map(cls._module_reliability, cls._max_module_feedback)
        if len(cls._session_command_feedback) > cls._max_session_feedback:
            stale_sessions = sorted(
                cls._session_command_feedback_ts.items(),
                key=lambda item: max(item[1].values(), default=0.0),
            )[:64]
            for session_id, _ in stale_sessions:
                cls._session_command_feedback.pop(session_id, None)
                cls._session_command_feedback_ts.pop(session_id, None)

        expired_stats_commands = [
            key
            for key, stats in cls._command_reliability.items()
            if now - stats.last_updated > _LONGTERM_FEEDBACK_TTL_SECONDS
        ]
        for key in expired_stats_commands:
            cls._command_reliability.pop(key, None)

        expired_stats_modules = [
            key
            for key, stats in cls._module_reliability.items()
            if now - stats.last_updated > _LONGTERM_FEEDBACK_TTL_SECONDS
        ]
        for key in expired_stats_modules:
            cls._module_reliability.pop(key, None)

        expired_contexts: list[str] = []
        for context_key, bucket in list(cls._context_command_reliability.items()):
            expired_keys = [
                key
                for key, stats in bucket.items()
                if now - stats.last_updated > _LONGTERM_FEEDBACK_TTL_SECONDS
            ]
            for key in expired_keys:
                bucket.pop(key, None)
            _trim_stats_map(bucket, 128)
            if not bucket:
                expired_contexts.append(context_key)
        for context_key in expired_contexts:
            cls._context_command_reliability.pop(context_key, None)

        if len(cls._context_command_reliability) > cls._max_context_feedback:
            stale_contexts = sorted(
                cls._context_command_reliability,
                key=lambda key: max(
                    (
                        stats.last_updated
                        for stats in cls._context_command_reliability[key].values()
                    ),
                    default=0.0,
                ),
            )[: len(cls._context_command_reliability) - cls._max_context_feedback]
            for context_key in stale_contexts:
                cls._context_command_reliability.pop(context_key, None)

        for command_id, examples in list(cls._command_success_examples.items()):
            fresh = [
                item
                for item in examples
                if now - item.timestamp <= _LONGTERM_FEEDBACK_TTL_SECONDS
            ][: cls._max_success_examples_per_command]
            if fresh:
                cls._command_success_examples[command_id] = fresh
            else:
                cls._command_success_examples.pop(command_id, None)

        expired_stats_sessions: list[str] = []
        for session_id, bucket in list(cls._session_command_reliability.items()):
            expired_keys = [
                key
                for key, stats in bucket.items()
                if now - stats.last_updated > cls._execution_feedback_ttl
            ]
            for key in expired_keys:
                bucket.pop(key, None)
            _trim_stats_map(bucket, 256)
            if not bucket:
                expired_stats_sessions.append(session_id)
        for session_id in expired_stats_sessions:
            cls._session_command_reliability.pop(session_id, None)


def _plugin_feedback_reward(reason: str) -> float:
    return float(_PLUGIN_FEEDBACK_REWARD.get(reason, 0.0))


def record_command_observation_feedback(
    *,
    output: dict[str, Any],
    action: str = "execute",
    session_id: str | None = None,
    latency_ms: float = 0,
    selected_rank: int = 0,
    selected_score: float = 0.0,
    selected_reason: str = "",
) -> None:
    """Record structured command-tool observations without requiring reroute logs.

    The Agent loop sees every tool result, including local validation failures
    and guardrail outputs.  This adapter turns the standard observation dict
    into the same feedback path as ``ExecutionObserver``.
    """

    if not isinstance(output, dict):
        return
    command_id = normalize_message_text(str(output.get("command_id", "") or ""))
    plugin_module = normalize_message_text(str(output.get("plugin_module", "") or ""))
    if not command_id and not plugin_module:
        return
    FeedbackStore.record_execution_observation(
        SimpleNamespace(
            action=normalize_message_text(action) or "execute",
            success=bool(output.get("ok")),
            reason=_reason_from_observation_output(output),
            latency_ms=max(int(float(latency_ms or 0)), 0),
            plugin_module=plugin_module,
            plugin_name=normalize_message_text(
                str(output.get("matched_plugin", "") or "")
            ),
            command_id=command_id,
            command=normalize_message_text(
                str(output.get("rendered_command", "") or "")
            ),
            route_stage=normalize_message_text(str(output.get("status", "") or "")),
            session_id=normalize_message_text(session_id or ""),
            message_preview=normalize_message_text(
                str(output.get("task_text", "") or "")
            )[:120],
            token_usage={},
            candidate_total=0,
            tool_candidates=0,
            selected_rank=max(int(selected_rank or 0), 0),
            selected_score=float(selected_score or 0.0),
            selected_reason=normalize_message_text(selected_reason)[:120],
            output=dict(output),
        )
    )


def record_trajectory_eval_feedback(record: dict[str, Any]) -> None:
    """Persist coarse trajectory metrics for feedback-driven tuning.

    Command-level feedback remains the source for individual capability
    reliability.  This projection keeps turn-level health: hit rate, false
    triggers, multi-task coverage, latency and estimated prompt cost.
    """

    if not isinstance(record, dict):
        return
    scenario = normalize_message_text(str(record.get("scenario", "") or "unknown"))
    eval_class = _trajectory_eval_class(record)
    payload = read_json(
        _TRAJECTORY_FEEDBACK_PATH,
        {
            "version": _TRAJECTORY_FEEDBACK_VERSION,
            "updated_at": utc_now_iso(),
            "overall": _empty_trajectory_bucket(),
            "by_scenario": {},
            "by_eval_class": {},
        },
    )
    if not isinstance(payload, dict):
        payload = {
            "version": _TRAJECTORY_FEEDBACK_VERSION,
            "updated_at": utc_now_iso(),
            "overall": _empty_trajectory_bucket(),
            "by_scenario": {},
            "by_eval_class": {},
        }
    payload["version"] = _TRAJECTORY_FEEDBACK_VERSION
    payload["updated_at"] = utc_now_iso()
    payload["overall"] = _update_trajectory_bucket(
        payload.get("overall"),
        record=record,
    )
    by_scenario = payload.setdefault("by_scenario", {})
    if isinstance(by_scenario, dict):
        by_scenario[scenario] = _update_trajectory_bucket(
            by_scenario.get(scenario),
            record=record,
        )
        _trim_trajectory_buckets(by_scenario, 64)
    by_eval_class = payload.setdefault("by_eval_class", {})
    if isinstance(by_eval_class, dict):
        by_eval_class[eval_class] = _update_trajectory_bucket(
            by_eval_class.get(eval_class),
            record=record,
        )
        _trim_trajectory_buckets(by_eval_class, 64)
    write_json(_TRAJECTORY_FEEDBACK_PATH, payload)


def _reason_from_observation_output(output: dict[str, Any]) -> str:
    if output.get("ok"):
        return FEEDBACK_REASON_ROUTE_SUCCESS
    missing = output.get("missing")
    if isinstance(missing, list | tuple | set) and missing:
        normalized_missing = {
            normalize_message_text(str(item or "")) for item in missing
        }
        if normalized_missing & {"target", "at", "user", "用户", "目标"}:
            return FEEDBACK_REASON_TARGET_REQUIRED
        return FEEDBACK_REASON_MISSING_PARAMS
    reason = normalize_message_text(
        str(
            output.get("reason")
            or output.get("guardrail_reason")
            or output.get("status")
            or ""
        )
    )
    if reason:
        return reason
    return _EXECUTION_REASON_ERROR


def _normalize_observation_for_feedback(observation: Any) -> Any:
    output = getattr(observation, "output", None)
    if not isinstance(output, dict):
        return observation
    if not normalize_message_text(getattr(observation, "command_id", "")):
        setattr_if_possible(observation, "command_id", output.get("command_id", ""))
    if not normalize_message_text(getattr(observation, "plugin_module", "")):
        setattr_if_possible(
            observation, "plugin_module", output.get("plugin_module", "")
        )
    if not normalize_message_text(getattr(observation, "plugin_name", "")):
        setattr_if_possible(
            observation, "plugin_name", output.get("matched_plugin", "")
        )
    if not normalize_message_text(getattr(observation, "command", "")):
        setattr_if_possible(observation, "command", output.get("rendered_command", ""))
    return observation


def setattr_if_possible(target: Any, name: str, value: Any) -> None:
    try:
        setattr(target, name, normalize_message_text(str(value or "")))
    except Exception:
        pass


def get_command_feedback_score(
    *,
    command_id: str | None = None,
    session_id: str | None = None,
    plugin_module: str | None = None,
) -> float:
    return FeedbackStore.command_feedback_score(
        command_id=command_id,
        session_id=session_id,
        plugin_module=plugin_module,
    )


def get_command_feedback_profile(
    *,
    command_id: str | None = None,
    session_id: str | None = None,
    plugin_module: str | None = None,
) -> CommandFeedbackProfile:
    return FeedbackStore.command_feedback_profile(
        command_id=command_id,
        session_id=session_id,
        plugin_module=plugin_module,
    )


def get_contextual_command_feedback_profile(
    *,
    message_text: str,
    command_id: str | None = None,
) -> CommandFeedbackProfile:
    return FeedbackStore.contextual_command_feedback_profile(
        message_text=message_text,
        command_id=command_id,
    )


def get_command_success_examples(
    *,
    command_id: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    return FeedbackStore.command_success_examples(
        command_id=command_id,
        limit=limit,
    )


def _clamp_command_feedback(value: float) -> float:
    return max(min(value, 36.0), -72.0)


def _trim_feedback_map(
    values: dict[str, float],
    timestamps: dict[str, float],
    capacity: int,
) -> None:
    if len(values) <= capacity:
        return
    stale = sorted(
        values,
        key=lambda key: (timestamps.get(key, 0.0), abs(values.get(key, 0.0))),
    )[: max(len(values) - capacity, 64)]
    for key in stale:
        values.pop(key, None)
        timestamps.pop(key, None)


def _trim_stats_map(values: dict[str, _ReliabilityStats], capacity: int) -> None:
    if len(values) <= capacity:
        return
    stale = sorted(
        values,
        key=lambda key: (
            values[key].last_updated,
            values[key].success_count + values[key].failure_count,
        ),
    )[: max(len(values) - capacity, 64)]
    for key in stale:
        values.pop(key, None)


def _dump_stats_bucket(
    values: dict[str, _ReliabilityStats],
) -> dict[str, dict[str, Any]]:
    return {
        key: stats.to_dict()
        for key, stats in values.items()
        if normalize_message_text(key) and stats.last_updated > 0
    }


def _dump_nested_stats_bucket(
    values: dict[str, dict[str, _ReliabilityStats]],
) -> dict[str, dict[str, dict[str, Any]]]:
    payload: dict[str, dict[str, dict[str, Any]]] = {}
    for context_key, bucket in values.items():
        key = normalize_message_text(context_key)
        if not key:
            continue
        dumped = _dump_stats_bucket(bucket)
        if dumped:
            payload[key] = dumped
    return payload


def _dump_success_examples(
    values: dict[str, list[_CommandSuccessExample]],
) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for command_id, examples in values.items():
        key = normalize_message_text(command_id)
        if not key:
            continue
        rows = [item.to_dict() for item in examples[:12]]
        if rows:
            payload[key] = rows
    return payload


def _load_stats_bucket(
    payload: Any,
    *,
    saved_monotonic: float,
    saved_wall_time: float,
    now: float,
    wall_now: float,
) -> dict[str, _ReliabilityStats]:
    if not isinstance(payload, dict):
        return {}
    wall_age = max(wall_now - saved_wall_time, 0.0) if saved_wall_time > 0 else 0.0
    result: dict[str, _ReliabilityStats] = {}
    for raw_key, raw_stats in payload.items():
        key = normalize_message_text(str(raw_key or ""))
        if not key:
            continue
        stats = _ReliabilityStats.from_dict(raw_stats)
        if stats.last_updated <= 0:
            continue
        age_at_save = max(saved_monotonic - stats.last_updated, 0.0)
        age = age_at_save + wall_age
        if age >= _LONGTERM_FEEDBACK_TTL_SECONDS:
            continue
        freshness = max(0.35, 1.0 - age / _LONGTERM_FEEDBACK_TTL_SECONDS)
        stats = stats.scaled(freshness)
        stats.last_updated = now
        result[key] = stats
    return result


def _load_nested_stats_bucket(
    payload: Any,
    *,
    saved_monotonic: float,
    saved_wall_time: float,
    now: float,
    wall_now: float,
) -> dict[str, dict[str, _ReliabilityStats]]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, dict[str, _ReliabilityStats]] = {}
    for raw_context, raw_bucket in payload.items():
        context_key = normalize_message_text(str(raw_context or ""))
        if not context_key:
            continue
        bucket = _load_stats_bucket(
            raw_bucket,
            saved_monotonic=saved_monotonic,
            saved_wall_time=saved_wall_time,
            now=now,
            wall_now=wall_now,
        )
        if bucket:
            result[context_key] = bucket
    return result


def _load_success_examples(
    payload: Any,
    *,
    saved_monotonic: float,
    saved_wall_time: float,
    now: float,
    wall_now: float,
) -> dict[str, list[_CommandSuccessExample]]:
    if not isinstance(payload, dict):
        return {}
    wall_age = max(wall_now - saved_wall_time, 0.0) if saved_wall_time > 0 else 0.0
    result: dict[str, list[_CommandSuccessExample]] = {}
    for raw_command_id, raw_items in payload.items():
        command_id = normalize_message_text(str(raw_command_id or ""))
        if not command_id or not isinstance(raw_items, list | tuple):
            continue
        examples: list[_CommandSuccessExample] = []
        for raw_item in raw_items:
            item = _CommandSuccessExample.from_dict(raw_item)
            if item is None:
                continue
            age_at_save = max(saved_monotonic - item.timestamp, 0.0)
            age = age_at_save + wall_age
            if age >= _LONGTERM_FEEDBACK_TTL_SECONDS:
                continue
            examples.append(
                _CommandSuccessExample(
                    timestamp=now - age,
                    message=item.message,
                    command_id=item.command_id,
                    slots=item.slots,
                    rendered_command=item.rendered_command,
                    plugin_module=item.plugin_module,
                )
            )
        if examples:
            result[command_id] = sorted(
                examples,
                key=lambda item: item.timestamp,
                reverse=True,
            )[:12]
    return result


def _dump_float_bucket(values: dict[str, float]) -> dict[str, float]:
    return {
        key: round(float(value or 0.0), 6)
        for key, value in values.items()
        if normalize_message_text(key) and float(value or 0.0)
    }


def _load_float_bucket(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for raw_key, raw_value in payload.items():
        key = normalize_message_text(str(raw_key or ""))
        value = _safe_float(raw_value)
        if key and value:
            result[key] = value
    return result


def _load_score_bucket(
    score_payload: Any,
    timestamp_payload: Any,
    *,
    saved_monotonic: float,
    saved_wall_time: float,
    now: float,
    wall_now: float,
) -> dict[str, float]:
    scores = _load_float_bucket(score_payload)
    timestamps = _load_float_bucket(timestamp_payload)
    wall_age = max(wall_now - saved_wall_time, 0.0) if saved_wall_time > 0 else 0.0
    result: dict[str, float] = {}
    for key, value in scores.items():
        age_at_save = max(saved_monotonic - timestamps.get(key, saved_monotonic), 0.0)
        age = age_at_save + wall_age
        if age >= _LONGTERM_FEEDBACK_TTL_SECONDS:
            continue
        freshness = max(0.35, 1.0 - age / _LONGTERM_FEEDBACK_TTL_SECONDS)
        result[key] = _clamp_command_feedback(value * freshness)
    return result


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _context_key(message_text: str) -> str:
    normalized = normalize_message_text(message_text)
    if not normalized:
        return ""
    normalized = _PLACEHOLDER_CONTEXT_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"\d+", "#", normalized)
    tokens: list[str] = []
    for token in _CONTEXT_TOKEN_PATTERN.findall(normalized.casefold()):
        token = normalize_message_text(token)
        if not token or token in {"真寻", "小真寻", "bot", "请", "帮我", "一下"}:
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= 12:
            break
    if not tokens:
        return ""
    return " ".join(tokens)


def _merge_reliability_stats(
    target: _ReliabilityStats,
    source: _ReliabilityStats,
    *,
    weight: float,
) -> None:
    if source.last_updated <= 0:
        return
    target.success_count += source.success_count * weight
    target.failure_count += source.failure_count * weight
    target.false_trigger_count += source.false_trigger_count * weight
    target.correction_count += source.correction_count * weight
    target.param_failure_count += source.param_failure_count * weight
    target.latency_total_ms += source.latency_total_ms * weight
    target.latency_count += source.latency_count * weight
    target.last_updated = max(target.last_updated, source.last_updated)


def _profile_from_stats(
    *,
    command_id: str,
    plugin_module: str,
    feedback_score: float,
    stats: _ReliabilityStats,
) -> CommandFeedbackProfile:
    success = max(float(stats.success_count or 0.0), 0.0)
    failure = max(float(stats.failure_count or 0.0), 0.0)
    false_trigger = max(float(stats.false_trigger_count or 0.0), 0.0)
    correction = max(float(stats.correction_count or 0.0), 0.0)
    param_failure = max(float(stats.param_failure_count or 0.0), 0.0)
    avg_latency_ms = max(float(stats.avg_latency_ms or 0.0), 0.0)
    total = success + failure
    smoothed_total = total + 3.0
    success_rate = (success + 1.5) / smoothed_total if smoothed_total else 0.5
    failure_rate = (failure + 1.0) / smoothed_total if smoothed_total else 0.0
    false_trigger_rate = false_trigger / max(total, 1.0)
    param_failure_rate = param_failure / max(total, 1.0)

    sample_weight = min(total / 8.0, 1.0)
    reliability_score = (success_rate - failure_rate) * 18.0 * sample_weight
    false_trigger_score = -min(
        42.0,
        (false_trigger_rate * 36.0 + correction * 4.5) * sample_weight,
    )
    param_failure_score = -min(
        28.0,
        (param_failure_rate * 26.0 + param_failure * 1.8) * sample_weight,
    )
    latency_score = _latency_reliability_score(avg_latency_ms, sample_weight)
    low_reliability = total >= 3.0 and (
        success_rate < 0.38
        or false_trigger_rate >= 0.28
        or false_trigger >= 2.0
        or param_failure_rate >= 0.42
    )
    high_reliability = (
        total >= 3.0
        and success_rate >= 0.72
        and false_trigger_rate <= 0.08
        and param_failure_rate <= 0.12
        and failure_rate <= 0.35
        and avg_latency_ms <= 9000.0
    )
    return CommandFeedbackProfile(
        command_id=command_id,
        plugin_module=plugin_module,
        feedback_score=feedback_score,
        reliability_score=max(min(reliability_score, 18.0), -24.0),
        false_trigger_score=false_trigger_score,
        latency_score=latency_score,
        param_failure_score=param_failure_score,
        success_count=success,
        failure_count=failure,
        false_trigger_count=false_trigger,
        correction_count=correction,
        param_failure_count=param_failure,
        avg_latency_ms=avg_latency_ms,
        total_count=total,
        success_rate=success_rate,
        failure_rate=failure_rate,
        false_trigger_rate=false_trigger_rate,
        param_failure_rate=param_failure_rate,
        low_reliability=low_reliability,
        high_reliability=high_reliability,
    )


def _latency_reliability_score(avg_latency_ms: float, sample_weight: float) -> float:
    if avg_latency_ms <= 0:
        return 0.0
    if avg_latency_ms <= 1200:
        return 2.0 * sample_weight
    if avg_latency_ms <= 3500:
        return 0.8 * sample_weight
    if avg_latency_ms <= 9000:
        return 0.0
    if avg_latency_ms <= 18000:
        return -2.0 * sample_weight
    return -4.5 * sample_weight


def _reliability_delta(observation: Any) -> tuple[float, float, float, float]:
    reason = normalize_message_text(getattr(observation, "reason", ""))
    action = normalize_message_text(getattr(observation, "action", ""))
    success = bool(getattr(observation, "success", False))
    if reason == _EXECUTION_REASON_ROUTE_CONFIRMED:
        return 0.55, 0.0, 0.0, 0.0
    if reason == _EXECUTION_REASON_ROUTE_USER_CORRECTED:
        return 0.0, 1.0, 1.0, 1.0
    if action == "execute" and success:
        return 1.0, 0.0, 0.0, 0.0
    if action == "usage" and success:
        return 0.35, 0.0, 0.0, 0.0
    if reason == FEEDBACK_REASON_REROUTE_FAILED:
        return 0.0, 1.0, 0.35, 0.0
    if reason in {
        _EXECUTION_REASON_INVALID_COMMAND,
        _EXECUTION_REASON_PLUGIN_NO_VISIBLE_OUTPUT,
    }:
        return 0.0, 0.35, 0.0, 0.0
    if reason in {
        _EXECUTION_REASON_MISSING_IMAGE,
        _EXECUTION_REASON_MISSING_REPLY,
        _EXECUTION_REASON_MISSING_TEXT,
        FEEDBACK_REASON_MISSING_PARAMS,
        _EXECUTION_REASON_CLARIFY_REQUESTED,
    }:
        return 0.0, 0.35, 0.0, 0.0
    if reason in {
        FEEDBACK_REASON_TARGET_REQUIRED,
        FEEDBACK_REASON_SELF_ONLY_BLOCKED,
        FEEDBACK_REASON_FUZZY_CLARIFY,
        FEEDBACK_REASON_DIRECT_TARGET_REQUIRED,
    }:
        return 0.0, 0.45, 0.15, 0.0
    if reason == _EXECUTION_REASON_PERMISSION_DENIED:
        return 0.0, 0.12, 0.0, 0.0
    if reason == _EXECUTION_REASON_PLUGIN_NOT_LOADED:
        return 0.0, 0.35, 0.0, 0.0
    if reason in {
        _EXECUTION_REASON_TIMEOUT,
        _EXECUTION_REASON_LLM_ERROR,
        _EXECUTION_REASON_CANCELLED,
        _EXECUTION_REASON_ERROR,
    }:
        return 0.0, 0.45, 0.0, 0.0
    if action == "execute" and not success:
        return 0.0, 0.65, 0.0, 0.0
    return 0.0, 0.0, 0.0, 0.0


def _param_failure_delta(observation: Any) -> float:
    reason = normalize_message_text(getattr(observation, "reason", ""))
    if reason in {
        _EXECUTION_REASON_MISSING_IMAGE,
        _EXECUTION_REASON_MISSING_REPLY,
        _EXECUTION_REASON_MISSING_TEXT,
        FEEDBACK_REASON_MISSING_PARAMS,
        _EXECUTION_REASON_CLARIFY_REQUESTED,
        FEEDBACK_REASON_TARGET_REQUIRED,
        FEEDBACK_REASON_SELF_ONLY_BLOCKED,
        FEEDBACK_REASON_FUZZY_CLARIFY,
        FEEDBACK_REASON_DIRECT_TARGET_REQUIRED,
    }:
        return 1.0
    output = getattr(observation, "output", None)
    if isinstance(output, dict):
        status = normalize_message_text(str(output.get("status", "")))
        missing = output.get("missing")
        if status in {"failed", "tool_execution_exception"} and missing:
            return 1.0
    return 0.0


def _observation_latency_ms(observation: Any) -> float:
    latency = _safe_float(getattr(observation, "latency_ms", 0.0))
    if latency > 0:
        return latency
    output = getattr(observation, "output", None)
    if isinstance(output, dict):
        for key in ("latency_ms", "duration_ms", "elapsed_ms"):
            latency = _safe_float(output.get(key))
            if latency > 0:
                return latency
    return 0.0


def _coerce_feedback_kind(reason: str) -> FeedbackKind:
    normalized = normalize_message_text(reason)
    if normalized == _EXECUTION_REASON_ROUTE_CONFIRMED:
        return "route_confirmed"
    if normalized == _EXECUTION_REASON_ROUTE_USER_CORRECTED:
        return "route_user_corrected"
    if normalized in _PLUGIN_FEEDBACK_REWARD:
        return cast(FeedbackKind, normalized)
    if normalized == "chat_rewritten":
        return "chat_rewritten"
    if normalized == "chat_empty":
        return "chat_empty"
    return "route_success" if normalized == "success" else "reroute_failed"


def _build_plugin_slot_feedback(
    *,
    reason: str,
    route_message: str,
    route_command: str,
    image_missing: int = 0,
    text_missing: int = 0,
    allow_at: bool | None = None,
) -> dict[str, float]:
    slot_scores: dict[str, float] = {}
    has_command_head = bool(_normalize_head(route_command))
    has_target_signal = bool(_extract_at_tokens(route_message))
    has_image_signal = bool(_extract_image_tokens(route_message))
    has_text_signal = _extract_text_token_count(route_command) > 0

    if has_command_head:
        slot_scores["command_head"] = (
            1.0 if reason == FEEDBACK_REASON_ROUTE_SUCCESS else -0.6
        )

    if reason == FEEDBACK_REASON_ROUTE_SUCCESS:
        if has_target_signal:
            slot_scores["target"] = 0.35
        if has_image_signal:
            slot_scores["image"] = 0.35
        if has_text_signal:
            slot_scores["text"] = 0.25
        return slot_scores

    if reason == FEEDBACK_REASON_SELF_ONLY_BLOCKED:
        slot_scores["target"] = -0.95
        return slot_scores

    if reason in {
        FEEDBACK_REASON_TARGET_REQUIRED,
        FEEDBACK_REASON_DIRECT_TARGET_REQUIRED,
        FEEDBACK_REASON_FUZZY_CLARIFY,
    }:
        slot_scores["target"] = -0.65
        return slot_scores

    if reason == FEEDBACK_REASON_MISSING_PARAMS:
        if image_missing > 0:
            slot_scores["image"] = -0.90
        if text_missing > 0:
            slot_scores["text"] = -0.75
        if allow_at and not has_target_signal:
            slot_scores["target"] = -0.55
        return slot_scores

    if reason == FEEDBACK_REASON_REROUTE_FAILED:
        slot_scores["command_head"] = -0.85
        return slot_scores

    return slot_scores


def _extract_text_token_count(command_text: str) -> int:
    normalized = normalize_message_text(command_text)
    if not normalized:
        return 0
    parts = normalized.split(" ", 1)
    payload = parts[1] if len(parts) > 1 else ""
    payload = normalize_message_text(_PLACEHOLDER_SEGMENT_PATTERN.sub(" ", payload))
    if not payload:
        return 0
    return len([token for token in payload.split(" ") if token])


def _extract_at_tokens(message_text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _AT_TOKEN_PATTERN.finditer(message_text))


def _extract_image_tokens(message_text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0) for match in _IMAGE_TOKEN_PATTERN.finditer(message_text)
    )


def _normalize_head(command: str) -> str:
    normalized = normalize_message_text(command)
    return normalize_message_text(normalized.split(" ", 1)[0]) if normalized else ""


def _normalize_slots(raw_slots: dict[str, Any]) -> dict[str, str]:
    slots: dict[str, str] = {}
    if not isinstance(raw_slots, dict):
        return slots
    for key, value in raw_slots.items():
        name = normalize_message_text(str(key or ""))
        slot_value = normalize_message_text(str(value or ""))
        if name and slot_value:
            slots[name] = slot_value
    return slots


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


def _trajectory_eval_class(record: dict[str, Any]) -> str:
    scenario = normalize_message_text(str(record.get("scenario", "") or ""))
    if scenario == "superuser_agent" or record.get("agent_mode") == "superuser_agent":
        return "superuser_agent"
    task_ledger = record.get("task_ledger")
    tasks = task_ledger.get("tasks") if isinstance(task_ledger, dict) else []
    if isinstance(tasks, list) and len(tasks) > 1:
        return "multi_tool"
    obligation = normalize_message_text(str(record.get("tool_obligation", "") or ""))
    if obligation == "required":
        return "real_tool_required"
    selected_tools = record.get("selected_tools")
    if isinstance(selected_tools, list) and selected_tools:
        return "tool_selected"
    if obligation == "none":
        return "direct_chat"
    return "ambiguous_or_auto"


def _empty_trajectory_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "hit_total": 0,
        "hit_count": 0,
        "false_trigger_count": 0,
        "multi_task_total": 0,
        "multi_task_covered_count": 0,
        "tool_call_total": 0,
        "latency_ms_total": 0,
        "prompt_tokens_total": 0,
        "failed_total": 0,
        "hit_rate": None,
        "false_trigger_rate": None,
        "multi_task_coverage_rate": None,
        "avg_tool_calls": 0.0,
        "avg_latency_ms": None,
        "avg_prompt_tokens": None,
        "last_trace_id": "",
        "updated_at": utc_now_iso(),
    }


def _update_trajectory_bucket(
    bucket: Any,
    *,
    record: dict[str, Any],
) -> dict[str, Any]:
    current = _empty_trajectory_bucket()
    if isinstance(bucket, dict):
        current.update(bucket)
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    current["total"] = int(current.get("total", 0) or 0) + 1
    hit = _trajectory_bool_or_none(record.get("hit", evaluation.get("hit")))
    if hit is not None:
        current["hit_total"] = int(current.get("hit_total", 0) or 0) + 1
        if hit:
            current["hit_count"] = int(current.get("hit_count", 0) or 0) + 1
    if bool(record.get("false_trigger", evaluation.get("false_trigger"))):
        current["false_trigger_count"] = (
            int(current.get("false_trigger_count", 0) or 0) + 1
        )
    multi = _trajectory_bool_or_none(
        record.get("multi_task_covered", evaluation.get("multi_task_covered"))
    )
    if multi is not None:
        current["multi_task_total"] = int(current.get("multi_task_total", 0) or 0) + 1
        if multi:
            current["multi_task_covered_count"] = (
                int(current.get("multi_task_covered_count", 0) or 0) + 1
            )
    current["tool_call_total"] = int(current.get("tool_call_total", 0) or 0) + int(
        evaluation.get("tool_call_count", 0) or 0
    )
    current["latency_ms_total"] = int(current.get("latency_ms_total", 0) or 0) + max(
        _trajectory_nested_int(record, "latency", "total_ms"), 0
    )
    current["prompt_tokens_total"] = int(
        current.get("prompt_tokens_total", 0) or 0
    ) + max(_trajectory_nested_int(record, "token", "prompt_estimated"), 0)
    if normalize_message_text(str(record.get("status", "") or "")) == "failed":
        current["failed_total"] = int(current.get("failed_total", 0) or 0) + 1
    current["last_trace_id"] = normalize_message_text(str(record.get("trace_id", "")))
    current["updated_at"] = utc_now_iso()
    total = int(current.get("total", 0) or 0)
    hit_total = int(current.get("hit_total", 0) or 0)
    multi_total = int(current.get("multi_task_total", 0) or 0)
    current["hit_rate"] = _trajectory_rate(
        int(current.get("hit_count", 0) or 0), hit_total
    )
    current["false_trigger_rate"] = _trajectory_rate(
        int(current.get("false_trigger_count", 0) or 0),
        total,
    )
    current["multi_task_coverage_rate"] = _trajectory_rate(
        int(current.get("multi_task_covered_count", 0) or 0),
        multi_total,
    )
    current["avg_tool_calls"] = (
        round(
            float(current.get("tool_call_total", 0) or 0) / float(total),
            3,
        )
        if total
        else 0.0
    )
    current["avg_latency_ms"] = (
        round(
            float(current.get("latency_ms_total", 0) or 0) / float(total),
            2,
        )
        if total
        else None
    )
    current["avg_prompt_tokens"] = (
        round(
            float(current.get("prompt_tokens_total", 0) or 0) / float(total),
            2,
        )
        if total
        else None
    )
    return current


def _trajectory_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _trajectory_nested_int(payload: dict[str, Any], section: str, key: str) -> int:
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        return -1
    try:
        return int(section_payload.get(key, -1) or 0)
    except Exception:
        return -1


def _trajectory_rate(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(float(count) / float(total), 4)


def _trim_trajectory_buckets(values: dict[str, Any], capacity: int) -> None:
    if len(values) <= capacity:
        return
    stale = sorted(
        values,
        key=lambda key: normalize_message_text(
            str((values.get(key) or {}).get("updated_at", ""))
        ),
    )[: max(len(values) - capacity, 1)]
    for key in stale:
        values.pop(key, None)


_AT_TOKEN_PATTERN = re.compile(r"\[@[^\]\s]+\]|@\S+")
_IMAGE_TOKEN_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.I)
_PLACEHOLDER_SEGMENT_PATTERN = re.compile(
    r"\[@[^\]\s]+\]|\[image(?:#\d+)?\]",
    re.I,
)

__all__ = [
    "CommandFeedbackProfile",
    "FeedbackDomain",
    "FeedbackKind",
    "FeedbackRecord",
    "FeedbackStore",
    "get_command_feedback_profile",
    "get_command_feedback_score",
    "get_command_success_examples",
    "get_contextual_command_feedback_profile",
    "record_command_observation_feedback",
    "record_trajectory_eval_feedback",
]
