"""Shared helpers for superuser Agent toolsets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zhenxun.services.llm.types.models import ToolResult

from ..approval_store import create_pending_approval
from ..audit_log import record_audit_event
from ..permission_policy import PermissionResult

MAX_OUTPUT_CHARS = 60000
DEFAULT_TIMEOUT_SECONDS = 20.0


def project_root() -> Path:
    return Path.cwd()


def tool_result(ok: bool, status: str, **payload: Any) -> ToolResult:
    output = {"ok": ok, "status": status, **payload}
    return ToolResult(output=output, display_content=status)


def actor_from_context(context: Any | None) -> dict[str, str]:
    session_id = str(getattr(context, "session_id", "") or "")
    extra = getattr(context, "extra", None)
    user_id = ""
    if isinstance(extra, dict):
        user_id = str(extra.get("actor_user_id", "") or "")
    user_id = user_id or session_id or "unknown"
    return {
        "user_id": user_id,
        "session_key": session_id or user_id,
    }


def coerce_max_chars(value: Any) -> int:
    try:
        return max(1, min(int(value or MAX_OUTPUT_CHARS), MAX_OUTPUT_CHARS))
    except (TypeError, ValueError):
        return MAX_OUTPUT_CHARS


def coerce_timeout(value: Any, *, default: float = DEFAULT_TIMEOUT_SECONDS) -> float:
    try:
        return max(1.0, min(float(value or default), 120.0))
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
    approval = create_pending_approval(
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=action,
        payload=payload,
        reason=permission.reason,
        matched_pattern=permission.matched_pattern,
    )
    record_audit_event(
        event="approval_created",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=action,
        payload={"approval_id": approval.approval_id, **payload},
        result={"permission": permission.__dict__},
    )
    return tool_result(
        False,
        "approval_required",
        approval_required=True,
        approval=approval.to_public_payload(),
        permission=permission.__dict__,
        instruction=(
            "请向超级用户说明此操作需要确认。用户确认后，调用 "
            "approve_pending_action 并传入 approval_id 执行。"
        ),
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
        permission=permission.__dict__,
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
    "actor_from_context",
    "approval_required_result",
    "audited_error_result",
    "coerce_max_chars",
    "coerce_timeout",
    "compact_text",
    "decode",
    "permission_denied_result",
    "project_root",
    "tool_result",
]
