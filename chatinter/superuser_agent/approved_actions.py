"""Execute user-approved actions from the fixed superuser tool set."""

from __future__ import annotations

from typing import Any

from ..llm_compat import ToolResult
from .approval_store import PendingApproval, approval_payload_matches_fingerprint
from .audit_log import record_audit_event
from .permission_policy import file_path_deny, shell_command_deny
from .tools.common import permission_denied_result, tool_result
from .tools.file_tools import (
    replace_in_file,
    write_file,
)
from .tools.shell_tools import run_shell_command

_APPROVED_ACTIONS = {"shell_command", "write_file", "replace_in_file"}


def validate_approved_action(
    *,
    approval: PendingApproval,
    actor: dict[str, str],
) -> ToolResult | None:
    if not approval_payload_matches_fingerprint(approval):
        return tool_result(
            False,
            "approval_payload_mismatch",
            approval_id=approval.approval_id,
            error="审批内容校验失败，操作未执行。",
        )
    if approval.action not in _APPROVED_ACTIONS:
        return tool_result(
            False,
            "approval_action_unknown",
            approval_id=approval.approval_id,
            action=approval.action,
            error="该操作不支持审批执行。",
        )
    if approval.action == "shell_command":
        denied = shell_command_deny(str(approval.payload.get("command", "") or ""))
    else:
        denied = file_path_deny(str(approval.payload.get("path", "") or ""))
    if denied is None:
        return None
    return permission_denied_result(
        actor=actor,
        action=approval.action,
        payload={"approval_id": approval.approval_id},
        permission=denied,
    )


async def execute_approved_action(
    *,
    approval: PendingApproval,
    actor: dict[str, str],
) -> ToolResult:
    validation_error = validate_approved_action(approval=approval, actor=actor)
    if validation_error is not None:
        return validation_error
    record_audit_event(
        event="approval_accepted",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=approval.action,
        payload={"approval_id": approval.approval_id, **approval.payload},
    )
    payload = approval.payload
    approval_id = approval.approval_id
    if approval.action == "shell_command":
        return await run_shell_command(
            command=str(payload.get("command", "") or ""),
            cwd=str(payload.get("cwd", "") or "") or None,
            actor=actor,
            approval_id=approval_id,
            timeout_seconds=_optional_float(payload.get("timeout_seconds")),
        )
    if approval.action == "write_file":
        return await write_file(
            path=str(payload.get("path", "") or ""),
            content=str(payload.get("content", "") or ""),
            create_dirs=bool(payload.get("create_dirs") or False),
            actor=actor,
            approval_id=approval_id,
            reason=str(payload.get("reason", "") or ""),
        )
    return await replace_in_file(
        path=str(payload.get("path", "") or ""),
        old_text=str(payload.get("old_text", "") or ""),
        new_text=str(payload.get("new_text", "") or ""),
        expected_replacements=_optional_int(payload.get("expected_replacements")),
        actor=actor,
        approval_id=approval_id,
        reason=str(payload.get("reason", "") or ""),
    )


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


__all__ = ["execute_approved_action", "validate_approved_action"]
