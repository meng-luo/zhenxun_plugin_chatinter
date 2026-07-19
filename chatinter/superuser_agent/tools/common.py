"""Shared helpers for superuser Agent toolsets."""

from __future__ import annotations

from typing import Any

from ...llm_compat import ToolResult
from ..approval_store import create_pending_approval
from ..audit_log import record_audit_event
from ..permission_policy import PermissionResult, permission_reason_text

MAX_OUTPUT_CHARS = 60000
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_TIMEOUT_SECONDS = 1800.0


def tool_result(ok: bool, status: str, **payload: Any) -> ToolResult:
    output = {"ok": ok, "status": status, **payload}
    return ToolResult(output=output, display_content=_display_content(status, payload))


def _display_content(status: str, payload: dict[str, Any]) -> str:
    for key in ("summary", "stdout", "stderr", "error", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return compact_text(value.strip(), max_chars=1200)
    return status


def actor_from_context(context: Any | None) -> dict[str, str]:
    session_id = str(getattr(context, "session_id", "") or "")
    extra = getattr(context, "extra", None)
    user_id = ""
    run_id = ""
    trace_id = ""
    if isinstance(extra, dict):
        user_id = str(extra.get("actor_user_id", "") or "")
        run_id = str(extra.get("run_id", "") or "")
        trace_id = str(extra.get("trace_id", "") or "")
    user_id = user_id or session_id or "unknown"
    return {
        "user_id": user_id,
        "session_key": session_id or user_id,
        "run_id": run_id,
        "trace_id": trace_id,
    }


def coerce_max_chars(value: Any) -> int:
    try:
        return max(1, min(int(value or MAX_OUTPUT_CHARS), MAX_OUTPUT_CHARS))
    except (TypeError, ValueError):
        return MAX_OUTPUT_CHARS


def coerce_timeout(value: Any, *, default: float = DEFAULT_TIMEOUT_SECONDS) -> float:
    try:
        return max(1.0, min(float(value or default), MAX_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return default


def decode(data: bytes | None, *, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")[:max_chars]


def compact_text(value: str, *, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    return str(value or "")[: max(1, min(max_chars, MAX_OUTPUT_CHARS))]


def approval_required_result(
    *,
    actor: dict[str, str],
    action: str,
    payload: dict[str, Any],
    permission: PermissionResult,
) -> ToolResult:
    runtime_payload = dict(payload)
    if actor.get("run_id"):
        runtime_payload.setdefault("run_id", actor["run_id"])
    if actor.get("trace_id"):
        runtime_payload.setdefault("trace_id", actor["trace_id"])
    approval = create_pending_approval(
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=action,
        payload=runtime_payload,
        reason=permission_reason_text(permission),
        matched_pattern=permission.matched_pattern,
        permission_section=permission.section,
        permission_grant_key=permission.grant_key,
    )
    record_audit_event(
        event="approval_created",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=action,
        payload={"approval_id": approval.approval_id, **runtime_payload},
        result={"permission": permission.__dict__},
    )
    allow_conversation = bool(permission.section and permission.grant_key)
    choices = "/允许、/拒绝" + (" 或 /本对话允许" if allow_conversation else "")
    return tool_result(
        False,
        "approval_required",
        approval_required=True,
        approval=approval.to_public_payload(),
        instruction=f"运行时会直接处理 {choices}，并继续当前对话。",
    )


def permission_denied_result(
    *,
    actor: dict[str, str],
    action: str,
    payload: dict[str, Any],
    permission: PermissionResult,
) -> ToolResult:
    record_audit_event(
        event="permission_denied",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=action,
        payload=payload,
        result={"permission": permission.__dict__},
    )
    return tool_result(
        False,
        "permission_denied",
        reason=permission_reason_text(permission),
        **payload,
    )


def audited_error_result(
    *,
    actor: dict[str, str],
    action: str,
    payload: dict[str, Any],
    status: str,
    error: str = "",
) -> ToolResult:
    record_audit_event(
        event="operation_failed",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=action,
        payload=payload,
        result={"status": status, "error": error},
    )
    return tool_result(False, status, error=error, **payload)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_OUTPUT_CHARS",
    "MAX_TIMEOUT_SECONDS",
    "actor_from_context",
    "approval_required_result",
    "audited_error_result",
    "coerce_max_chars",
    "coerce_timeout",
    "compact_text",
    "decode",
    "permission_denied_result",
    "tool_result",
]
