"""Approval tools for deferred superuser Agent actions."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..approval_store import (
    PendingApproval,
    consume_pending_approval,
    list_pending_approvals,
    reject_pending_approval,
)
from ..audit_log import record_audit_event
from ..registry import register_superuser_tool
from .common import actor_from_context, coerce_max_chars, coerce_timeout, tool_result


class ApprovePendingActionTool:
    name = "approve_pending_action"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：确认并执行一个之前因 ask 权限策略暂停的操作。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "approval_id": {
                        "type": "string",
                        "description": "待确认操作 ID。",
                    }
                },
                "required": ["approval_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        approval_id = str(kwargs.get("approval_id", "") or "").strip()
        approval = consume_pending_approval(
            approval_id=approval_id,
            user_id=actor["user_id"],
            session_key=actor["session_key"],
        )
        if approval is None:
            return tool_result(
                False,
                "approval_not_found_or_expired",
                approval_id=approval_id,
            )
        return await execute_approved_action(approval=approval, actor=actor)


class RejectPendingActionTool:
    name = "reject_pending_action"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：拒绝并移除一个待确认操作。",
            parameters={
                "type": "object",
                "properties": {
                    "approval_id": {
                        "type": "string",
                        "description": "待确认操作 ID。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "拒绝原因，可为空。",
                    },
                },
                "required": ["approval_id", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        approval_id = str(kwargs.get("approval_id", "") or "").strip()
        reason = str(kwargs.get("reason", "") or "")
        approval = reject_pending_approval(
            approval_id=approval_id,
            user_id=actor["user_id"],
            session_key=actor["session_key"],
        )
        if approval is None:
            return tool_result(
                False,
                "approval_not_found_or_expired",
                approval_id=approval_id,
            )
        record_audit_event(
            event="approval_rejected",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=approval.action,
            payload={"approval_id": approval.approval_id, "reason": reason},
            result={"rejected": True},
        )
        return tool_result(
            True,
            "approval_rejected",
            approval=approval.to_public_payload(),
            reason=reason,
        )


class ListPendingApprovalsTool:
    name = "list_pending_approvals"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：列出当前会话待确认操作。",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        approvals = list_pending_approvals(
            user_id=actor["user_id"],
            session_key=actor["session_key"],
        )
        return tool_result(
            True,
            "pending_approvals",
            approvals=[approval.to_public_payload() for approval in approvals],
        )


async def execute_approved_action(
    *,
    approval: PendingApproval,
    actor: dict[str, str],
) -> ToolResult:
    record_audit_event(
        event="approval_accepted",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=approval.action,
        payload={"approval_id": approval.approval_id, **approval.payload},
    )
    if approval.action == "shell_command":
        from .shell_tools import run_shell_command

        return await run_shell_command(
            command=str(approval.payload.get("command", "") or ""),
            cwd=str(approval.payload.get("cwd", "") or "") or None,
            actor=actor,
            approval_id=approval.approval_id,
            timeout_seconds=coerce_timeout(approval.payload.get("timeout_seconds")),
        )
    if approval.action == "git_command":
        from .git_tools import run_git_command

        return await run_git_command(
            args=str(approval.payload.get("args", "") or ""),
            cwd=str(approval.payload.get("cwd", "") or "") or None,
            actor=actor,
            approval_id=approval.approval_id,
            timeout_seconds=coerce_timeout(approval.payload.get("timeout_seconds")),
        )
    if approval.action == "server_status":
        from .server_tools import server_status

        return await server_status(
            path=str(approval.payload.get("path", "") or ""),
            actor=actor,
            approval_id=approval.approval_id,
        )
    if approval.action == "process_list":
        from .server_tools import process_list

        return await process_list(
            query=str(approval.payload.get("query", "") or ""),
            max_results=_coerce_int(approval.payload.get("max_results"), default=40, lower=1, upper=120),
            actor=actor,
            approval_id=approval.approval_id,
        )
    if approval.action == "server_command":
        from .server_tools import run_server_command

        return await run_server_command(
            command=str(approval.payload.get("command", "") or ""),
            cwd=str(approval.payload.get("cwd", "") or "") or None,
            actor=actor,
            approval_id=approval.approval_id,
            timeout_seconds=coerce_timeout(approval.payload.get("timeout_seconds")),
        )
    if approval.action == "plugin_dev_inspect":
        from .plugin_dev_tools import inspect_plugin

        return await inspect_plugin(
            plugin_name=str(approval.payload.get("plugin_name", "") or ""),
            plugin_root=str(approval.payload.get("plugin_root", "") or "") or None,
            max_files=_coerce_int(approval.payload.get("max_files"), default=80, lower=1, upper=300),
            actor=actor,
            approval_id=approval.approval_id,
        )
    if approval.action == "plugin_dev_scaffold":
        from .plugin_dev_tools import scaffold_plugin

        return await scaffold_plugin(
            plugin_name=str(approval.payload.get("plugin_name", "") or ""),
            display_name=str(approval.payload.get("display_name", "") or ""),
            command=str(approval.payload.get("command", "") or ""),
            description=str(approval.payload.get("description", "") or ""),
            author=str(approval.payload.get("author", "") or "ChatInter Agent"),
            menu_type=str(approval.payload.get("menu_type", "") or "功能"),
            plugin_root=str(approval.payload.get("plugin_root", "") or "") or None,
            overwrite=bool(approval.payload.get("overwrite") or False),
            actor=actor,
            approval_id=approval.approval_id,
        )
    if approval.action == "plugin_dev_write_file":
        from .plugin_dev_tools import write_plugin_file

        return await write_plugin_file(
            plugin_name=str(approval.payload.get("plugin_name", "") or ""),
            relative_path=str(approval.payload.get("relative_path", "") or ""),
            content=str(approval.payload.get("content", "") or ""),
            plugin_root=str(approval.payload.get("plugin_root", "") or "") or None,
            create_dirs=bool(approval.payload.get("create_dirs") if approval.payload.get("create_dirs") is not None else True),
            overwrite=bool(approval.payload.get("overwrite") if approval.payload.get("overwrite") is not None else True),
            reason=str(approval.payload.get("reason", "") or ""),
            actor=actor,
            approval_id=approval.approval_id,
        )
    if approval.action == "patch_apply":
        from .patch_tools import apply_prepared_patch

        return await apply_prepared_patch(
            actor=actor,
            operation_id=str(approval.payload.get("operation_id", "") or ""),
            approval_id=approval.approval_id,
            reason=str(approval.payload.get("reason", "") or ""),
        )
    if approval.action == "patch_prepare":
        from ..patch_operations import normalize_change
        from .patch_tools import prepare_patch_operation

        try:
            raw_changes = approval.payload.get("changes", [])
            changes = [
                normalize_change(item)
                for item in raw_changes
                if isinstance(item, dict)
            ]
        except Exception as exc:
            return tool_result(
                False,
                "patch_prepare_invalid_input",
                approval_id=approval.approval_id,
                error=str(exc),
            )
        return await prepare_patch_operation(
            actor=actor,
            changes=changes,
            reason=str(approval.payload.get("reason", "") or ""),
            approval_id=approval.approval_id,
        )
    if approval.action == "patch_rollback":
        from .patch_tools import rollback_prepared_patch

        return await rollback_prepared_patch(
            actor=actor,
            operation_id=str(approval.payload.get("operation_id", "") or ""),
            approval_id=approval.approval_id,
            reason=str(approval.payload.get("reason", "") or ""),
        )
    if approval.action == "background_task_start":
        from .background_tools import start_background_task

        payload = dict(approval.payload)
        return await start_background_task(
            actor=actor,
            command_type=str(payload.get("command_type", "") or ""),
            command=str(payload.get("command", "") or ""),
            args=[str(item) for item in payload.get("args", []) or []],
            cwd=str(payload.get("cwd", "") or "") or None,
            reason=str(payload.get("reason", "") or ""),
            rendered_command=str(payload.get("rendered_command", "") or ""),
            approval_id=approval.approval_id,
        )
    if approval.action == "background_task_cancel":
        from .background_tools import cancel_task

        return await cancel_task(
            actor=actor,
            task_id=str(approval.payload.get("task_id", "") or ""),
        )
    if approval.action == "uv_command":
        from .uv_tools import run_uv_command

        return await run_uv_command(
            args=str(approval.payload.get("args", "") or ""),
            cwd=str(approval.payload.get("cwd", "") or "") or None,
            actor=actor,
            approval_id=approval.approval_id,
            timeout_seconds=coerce_timeout(approval.payload.get("timeout_seconds")),
        )
    if approval.action == "python_exec":
        from .python_tools import run_python_code

        return await run_python_code(
            code=str(approval.payload.get("code", "") or ""),
            cwd=str(approval.payload.get("cwd", "") or "") or None,
            actor=actor,
            approval_id=approval.approval_id,
            timeout_seconds=coerce_timeout(approval.payload.get("timeout_seconds")),
        )
    if approval.action == "python_module":
        from .python_tools import run_python_module

        return await run_python_module(
            module=str(approval.payload.get("module", "") or ""),
            args=[str(item) for item in approval.payload.get("args", []) or []],
            cwd=str(approval.payload.get("cwd", "") or "") or None,
            actor=actor,
            approval_id=approval.approval_id,
            timeout_seconds=coerce_timeout(approval.payload.get("timeout_seconds")),
        )
    if approval.action == "read_file":
        from .file_tools import read_file

        return await read_file(
            path=str(approval.payload.get("path", "") or ""),
            max_chars=coerce_max_chars(approval.payload.get("max_chars")),
            actor=actor,
            approval_id=approval.approval_id,
        )
    if approval.action == "list_dir":
        from .file_tools import list_dir

        return await list_dir(
            path=str(approval.payload.get("path", "") or "."),
            actor=actor,
            approval_id=approval.approval_id,
        )
    if approval.action == "search_files":
        from .file_tools import search_files

        return await search_files(
            root=str(approval.payload.get("root", "") or "."),
            pattern=str(approval.payload.get("pattern", "") or "**/*"),
            contains=str(approval.payload.get("contains", "") or ""),
            max_results=_coerce_int(approval.payload.get("max_results"), default=50, lower=1, upper=200),
            actor=actor,
            approval_id=approval.approval_id,
        )
    if approval.action == "write_file":
        from .file_tools import write_file

        return await write_file(
            path=str(approval.payload.get("path", "") or ""),
            content=str(approval.payload.get("content", "") or ""),
            create_dirs=bool(approval.payload.get("create_dirs") or False),
            actor=actor,
            approval_id=approval.approval_id,
            reason=str(approval.payload.get("reason", "") or ""),
        )
    if approval.action == "append_file":
        from .file_tools import append_file

        return await append_file(
            path=str(approval.payload.get("path", "") or ""),
            content=str(approval.payload.get("content", "") or ""),
            create_dirs=bool(approval.payload.get("create_dirs") or False),
            actor=actor,
            approval_id=approval.approval_id,
            reason=str(approval.payload.get("reason", "") or ""),
        )
    if approval.action == "replace_in_file":
        from .file_tools import replace_in_file

        expected = approval.payload.get("expected_replacements")
        try:
            expected_replacements = int(expected) if expected not in (None, "") else None
        except (TypeError, ValueError):
            expected_replacements = None
        return await replace_in_file(
            path=str(approval.payload.get("path", "") or ""),
            old_text=str(approval.payload.get("old_text", "") or ""),
            new_text=str(approval.payload.get("new_text", "") or ""),
            expected_replacements=expected_replacements,
            actor=actor,
            approval_id=approval.approval_id,
            reason=str(approval.payload.get("reason", "") or ""),
        )
    return tool_result(
        False,
        "approval_action_unknown",
        approval_id=approval.approval_id,
        action=approval.action,
    )


def _coerce_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        return max(lower, min(int(value or default), upper))
    except (TypeError, ValueError):
        return default


register_superuser_tool(ApprovePendingActionTool)
register_superuser_tool(RejectPendingActionTool)
register_superuser_tool(ListPendingApprovalsTool)

__all__ = [
    "ApprovePendingActionTool",
    "ListPendingApprovalsTool",
    "RejectPendingActionTool",
    "execute_approved_action",
]
