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
from typing import Any, ClassVar, Literal, cast

from zhenxun.services import logger

from .feedback_keys import (
    FEEDBACK_REASON_DIRECT_TARGET_REQUIRED,
    FEEDBACK_REASON_FUZZY_CLARIFY,
    FEEDBACK_REASON_MISSING_PARAMS,
    FEEDBACK_REASON_REROUTE_FAILED,
    FEEDBACK_REASON_ROUTE_SUCCESS,
    FEEDBACK_REASON_SELF_ONLY_BLOCKED,
    FEEDBACK_REASON_TARGET_REQUIRED,
)
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
_PLUGIN_FEEDBACK_REWARD = {
    FEEDBACK_REASON_ROUTE_SUCCESS: 1.0,
    FEEDBACK_REASON_MISSING_PARAMS: -0.35,
    FEEDBACK_REASON_TARGET_REQUIRED: -0.40,
    FEEDBACK_REASON_SELF_ONLY_BLOCKED: -0.55,
    FEEDBACK_REASON_REROUTE_FAILED: -0.45,
    FEEDBACK_REASON_FUZZY_CLARIFY: -0.20,
    FEEDBACK_REASON_DIRECT_TARGET_REQUIRED: -0.30,
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
    _chat_ttl: ClassVar[float] = 1800.0
    _plugin_ttl: ClassVar[float] = 4 * 60.0
    _execution_feedback_ttl: ClassVar[float] = 6 * 3600.0
    _max_last_plugin: ClassVar[int] = 256
    _max_command_feedback: ClassVar[int] = 2048
    _max_session_feedback: ClassVar[int] = 512
    _max_module_feedback: ClassVar[int] = 1024

    @classmethod
    def record_execution_observation(cls, observation: Any) -> None:
        command_id = normalize_message_text(getattr(observation, "command_id", ""))
        plugin_module = normalize_message_text(
            getattr(observation, "plugin_module", "")
        )
        if not command_id and not plugin_module:
            return

        delta = cls._execution_feedback_delta(observation)
        if not delta:
            return
        now = time.monotonic()
        cls._prune(now)
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
                _EXECUTION_REASON_INVALID_COMMAND,
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
        if reason in {
            _EXECUTION_REASON_MISSING_IMAGE,
            _EXECUTION_REASON_MISSING_REPLY,
            _EXECUTION_REASON_MISSING_TEXT,
            FEEDBACK_REASON_MISSING_PARAMS,
            _EXECUTION_REASON_CLARIFY_REQUESTED,
        }:
            return -0.12
        if reason == _EXECUTION_REASON_PERMISSION_DENIED:
            return -0.04
        if reason == _EXECUTION_REASON_PLUGIN_NOT_LOADED:
            return -0.2
        if reason in {
            _EXECUTION_REASON_INVALID_COMMAND,
            FEEDBACK_REASON_REROUTE_FAILED,
            _EXECUTION_REASON_ROUTE_USER_CORRECTED,
        }:
            return -1.4
        if reason in {
            _EXECUTION_REASON_TIMEOUT,
            _EXECUTION_REASON_LLM_ERROR,
            _EXECUTION_REASON_CANCELLED,
            _EXECUTION_REASON_ERROR,
        }:
            return -0.45
        return -0.5 if not success else 0.0

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
        if len(cls._session_command_feedback) > cls._max_session_feedback:
            stale_sessions = sorted(
                cls._session_command_feedback_ts.items(),
                key=lambda item: max(item[1].values(), default=0.0),
            )[:64]
            for session_id, _ in stale_sessions:
                cls._session_command_feedback.pop(session_id, None)
                cls._session_command_feedback_ts.pop(session_id, None)


def _plugin_feedback_reward(reason: str) -> float:
    return float(_PLUGIN_FEEDBACK_REWARD.get(reason, 0.0))


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


_AT_TOKEN_PATTERN = re.compile(r"\[@[^\]\s]+\]|@\S+")
_IMAGE_TOKEN_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.I)
_PLACEHOLDER_SEGMENT_PATTERN = re.compile(
    r"\[@[^\]\s]+\]|\[image(?:#\d+)?\]",
    re.I,
)

__all__ = [
    "FeedbackDomain",
    "FeedbackKind",
    "FeedbackRecord",
    "FeedbackStore",
    "get_command_feedback_score",
]
