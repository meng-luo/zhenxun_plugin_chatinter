"""Execute user-approved actions from the fixed superuser tool set."""

from __future__ import annotations

from typing import Any

from ..llm_compat import ToolResult
from .approval_store import PendingApproval, approval_payload_matches_fingerprint
from .audit_log import record_audit_event
from .permission_policy import file_path_deny, shell_command_deny
from .tools.active_task_tools import (
    active_task_audit_payload,
    execute_active_task_control_payload,
    execute_active_task_create_payload,
    execute_active_task_update_payload,
    validate_active_task_approval_payload,
)
from .tools.common import permission_denied_result, tool_result
from .tools.file_tools import (
    apply_patch_text,
    build_replace_changes,
    patch_paths,
    replace_files,
    replace_in_file,
    write_file,
)
from .tools.shell_tools import run_shell_command, start_background_shell_command

_APPROVED_ACTIONS = {
    "shell_command",
    "write_file",
    "replace_in_file",
    "apply_patch",
    "active_task_create",
    "active_task_control",
    "active_task_update",
}


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
    if approval.action in {
        "active_task_create",
        "active_task_control",
        "active_task_update",
    }:
        return validate_active_task_approval_payload(
            action=approval.action,
            payload=approval.payload,
            actor=actor,
        )
    if approval.action == "shell_command":
        denied = shell_command_deny(str(approval.payload.get("command", "") or ""))
        denied_paths: list[str] = []
    elif approval.action == "apply_patch":
        denied = None
        denied_paths, error = patch_paths(
            str(approval.payload.get("patch", "") or ""),
            cwd=str(approval.payload.get("cwd", "") or "") or None,
        )
        if error:
            return tool_result(
                False,
                "approval_payload_invalid",
                approval_id=approval.approval_id,
                error=error,
            )
    elif approval.action == "replace_in_file":
        changes, error = build_replace_changes(approval.payload)
        if error:
            return tool_result(
                False,
                "approval_payload_invalid",
                approval_id=approval.approval_id,
                error=error,
            )
        denied = None
        denied_paths = [change.path for change in changes]
    else:
        denied = None
        denied_paths = [str(approval.payload.get("path", "") or "")]
    for path in denied_paths:
        denied = file_path_deny(path)
        if denied is not None:
            break
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
    audit_payload = (
        active_task_audit_payload(approval.payload)
        if approval.action.startswith("active_task_")
        else approval.payload
    )
    record_audit_event(
        event="approval_accepted",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=approval.action,
        payload={"approval_id": approval.approval_id, **audit_payload},
    )
    payload = approval.payload
    approval_id = approval.approval_id
    if approval.action == "active_task_create":
        return await execute_active_task_create_payload(payload, actor=actor)
    if approval.action == "active_task_control":
        return await execute_active_task_control_payload(payload, actor=actor)
    if approval.action == "active_task_update":
        return await execute_active_task_update_payload(payload, actor=actor)
    if approval.action == "shell_command":
        if str(payload.get("action", "") or "run") == "start":
            return start_background_shell_command(
                command=str(payload.get("command", "") or ""),
                cwd=str(payload.get("cwd", "") or "") or None,
                actor=actor,
                approval_id=approval_id,
                timeout_seconds=_optional_float(payload.get("timeout_seconds")),
            )
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
    if approval.action == "apply_patch":
        return await apply_patch_text(
            patch=str(payload.get("patch", "") or ""),
            cwd=str(payload.get("cwd", "") or "") or None,
            actor=actor,
            approval_id=approval_id,
        )
    changes, error = build_replace_changes(payload)
    if error:
        return tool_result(False, "approval_payload_invalid", error=error)
    if payload.get("changes") is None:
        change = changes[0]
        return await replace_in_file(
            path=change.path,
            old_text=change.old_text,
            new_text=change.new_text,
            expected_replacements=change.expected_replacements,
            replace_all=change.replace_all,
            actor=actor,
            approval_id=approval_id,
            reason=str(payload.get("reason", "") or ""),
        )
    return await replace_files(
        changes=changes,
        actor=actor,
        approval_id=approval_id,
        reason=str(payload.get("reason", "") or ""),
    )


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


__all__ = ["execute_approved_action", "validate_approved_action"]
