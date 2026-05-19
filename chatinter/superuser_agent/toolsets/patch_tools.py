"""Transactional diff/patch/rollback tools for superuser Agent code edits."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..patch_operations import (
    FileChange,
    apply_patch_operation,
    create_patch_operation,
    get_patch_operation,
    list_patch_operations,
    normalize_change,
    rollback_patch_operation,
)
from ..permission_policy import PermissionResult, decide_file_write, decide_patch
from ..registry import register_superuser_tool
from .common import (
    actor_from_context,
    approval_required_result,
    permission_denied_result,
    tool_result,
)


class PatchPrepareTool:
    name = "patch_prepare"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：准备一次事务式代码/文本修改，只生成 diff 和 "
                "operation_id，不直接写入。之后用 patch_apply 应用。"
            ),
            parameters=_patch_changes_schema(
                include_reason=True,
                description="本次修改原因，便于审计和回滚。",
            ),
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        try:
            changes = _normalize_changes(kwargs.get("changes"))
            reason = str(kwargs.get("reason", "") or "")
        except Exception as exc:
            return tool_result(False, "patch_prepare_invalid_input", error=str(exc))
        return await prepare_patch_operation(
            actor=actor,
            changes=changes,
            reason=reason,
        )


async def prepare_patch_operation(
    *,
    actor: dict[str, str],
    changes: list[FileChange],
    reason: str,
    approval_id: str | None = None,
) -> ToolResult:
    patch_permission = decide_patch("patch_prepare")
    if patch_permission.decision == "deny":
        return permission_denied_result(
            actor=actor,
            action="patch_prepare",
            payload={"reason": reason, "files": [change.path for change in changes]},
            permission=patch_permission,
        )
    if patch_permission.decision == "ask" and not approval_id:
        return approval_required_result(
            actor=actor,
            action="patch_prepare",
            payload={
                "reason": reason,
                "changes": [_change_to_payload(change) for change in changes],
            },
            permission=patch_permission,
        )
    deny = _first_denied_change(changes)
    if deny is not None:
        change, permission = deny
        return permission_denied_result(
            actor=actor,
            action="patch_prepare",
            payload=_change_payload(change, reason=reason),
            permission=permission,
        )
    try:
        operation = create_patch_operation(
            actor=actor,
            changes=changes,
            action="patch_prepare",
            reason=reason,
            approval_id=approval_id,
        )
    except Exception as exc:
        return tool_result(False, "patch_prepare_failed", error=str(exc))
    return tool_result(
        True,
        "patch_prepared",
        operation=operation.public_payload(),
        instruction="检查 diff 后，如需落地请调用 patch_apply。",
    )


class PatchApplyTool:
    name = "patch_apply"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：应用 patch_prepare 生成的事务式修改。",
            parameters={
                "type": "object",
                "properties": {
                    "operation_id": {
                        "type": "string",
                        "description": "patch_prepare 返回的 operation_id。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么现在应用该修改。",
                    },
                },
                "required": ["operation_id", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        operation_id = str(kwargs.get("operation_id", "") or "").strip()
        reason = str(kwargs.get("reason", "") or "")
        return await apply_prepared_patch(
            actor=actor,
            operation_id=operation_id,
            reason=reason,
        )


class PatchRollbackTool:
    name = "patch_rollback"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：按 operation_id 回滚已应用的事务式修改。"
                "会恢复旧内容；新增文件会删除。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation_id": {
                        "type": "string",
                        "description": "要回滚的 operation_id。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "回滚原因。",
                    },
                },
                "required": ["operation_id", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        operation_id = str(kwargs.get("operation_id", "") or "").strip()
        reason = str(kwargs.get("reason", "") or "")
        return await rollback_prepared_patch(
            actor=actor,
            operation_id=operation_id,
            reason=reason,
        )


class PatchShowTool:
    name = "patch_show"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：查看 patch operation 详情或列出最近操作。",
            parameters={
                "type": "object",
                "properties": {
                    "operation_id": {
                        "type": ["string", "null"],
                        "description": "为空则列出最近操作；否则查看指定操作。",
                    },
                    "include_content": {
                        "type": ["boolean", "null"],
                        "description": "是否包含 before/after 完整内容，默认 false。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "列出最近操作时的数量，默认 20。",
                    },
                },
                "required": ["operation_id", "include_content", "limit"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        operation_id = str(kwargs.get("operation_id", "") or "").strip()
        include_content = bool(kwargs.get("include_content") or False)
        limit = _coerce_limit(kwargs.get("limit"))
        if operation_id:
            operation = get_patch_operation(operation_id)
            if operation is None:
                return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
            if operation.user_id != actor["user_id"] or operation.session_key != actor["session_key"]:
                return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
            return tool_result(
                True,
                "patch_operation",
                operation=operation.public_payload(include_content=include_content),
            )
        operations = list_patch_operations(
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            limit=limit,
        )
        return tool_result(
            True,
            "patch_operations_listed",
            operations=[
                operation.public_payload(include_content=False)
                for operation in operations
            ],
            count=len(operations),
        )


def _patch_changes_schema(*, include_reason: bool, description: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "changes": {
            "type": "array",
            "description": "文件变更列表，支持 write/append/replace。",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目标文件路径。"},
                    "mode": {
                        "type": "string",
                        "enum": ["write", "append", "replace"],
                        "description": "write 覆盖全文；append 追加；replace 精确替换。",
                    },
                    "content": {
                        "type": ["string", "null"],
                        "description": "write/append 使用的内容。",
                    },
                    "old_text": {
                        "type": ["string", "null"],
                        "description": "replace 使用的原文。",
                    },
                    "new_text": {
                        "type": ["string", "null"],
                        "description": "replace 使用的新文。",
                    },
                    "create_dirs": {
                        "type": ["boolean", "null"],
                        "description": "父目录不存在时是否创建。",
                    },
                    "expected_replacements": {
                        "type": ["integer", "null"],
                        "description": "replace 期望替换次数。",
                    },
                },
                "required": [
                    "path",
                    "mode",
                    "content",
                    "old_text",
                    "new_text",
                    "create_dirs",
                    "expected_replacements",
                ],
                "additionalProperties": False,
            },
        },
    }
    required = ["changes"]
    if include_reason:
        properties["reason"] = {"type": ["string", "null"], "description": description}
        required.append("reason")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _normalize_changes(raw_changes: Any) -> list[FileChange]:
    if not isinstance(raw_changes, list):
        raise ValueError("changes must be a list")
    changes = [normalize_change(item) for item in raw_changes if isinstance(item, dict)]
    if not changes:
        raise ValueError("changes cannot be empty")
    return changes


async def apply_prepared_patch(
    *,
    actor: dict[str, str],
    operation_id: str,
    approval_id: str | None = None,
    reason: str = "",
) -> ToolResult:
    operation = get_patch_operation(operation_id)
    if operation is None:
        return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
    if operation.user_id != actor["user_id"] or operation.session_key != actor["session_key"]:
        return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
    permission_result = _authorize_patch_mutation(
        actor=actor,
        action="patch_apply",
        operation_id=operation_id,
        reason=reason,
        changes=operation.changes,
        approval_id=approval_id,
    )
    if permission_result is not None:
        return permission_result
    return apply_patch_operation(
        operation_id=operation_id,
        actor=actor,
        approval_id=approval_id,
    )


async def rollback_prepared_patch(
    *,
    actor: dict[str, str],
    operation_id: str,
    approval_id: str | None = None,
    reason: str = "",
) -> ToolResult:
    operation = get_patch_operation(operation_id)
    if operation is None:
        return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
    if operation.user_id != actor["user_id"] or operation.session_key != actor["session_key"]:
        return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
    permission_result = _authorize_patch_mutation(
        actor=actor,
        action="patch_rollback",
        operation_id=operation_id,
        reason=reason,
        changes=operation.changes,
        approval_id=approval_id,
    )
    if permission_result is not None:
        return permission_result
    return rollback_patch_operation(
        operation_id=operation_id,
        actor=actor,
        approval_id=approval_id,
    )


def _authorize_patch_mutation(
    *,
    actor: dict[str, str],
    action: str,
    operation_id: str,
    reason: str,
    changes: list[FileChange],
    approval_id: str | None,
) -> ToolResult | None:
    deny = _first_denied_change(changes)
    if deny is not None:
        change, permission = deny
        return permission_denied_result(
            actor=actor,
            action=action,
            payload={"operation_id": operation_id, "path": change.path},
            permission=permission,
        )
    if approval_id:
        return None
    patch_permission = decide_patch(action)
    payload = {"operation_id": operation_id, "reason": reason}
    if patch_permission.decision == "deny":
        return permission_denied_result(
            actor=actor,
            action=action,
            payload=payload,
            permission=patch_permission,
        )
    if patch_permission.decision == "ask":
        return approval_required_result(
            actor=actor,
            action=action,
            payload=payload,
            permission=patch_permission,
        )
    ask = _first_ask_change(changes)
    if ask is not None:
        change, permission = ask
        return approval_required_result(
            actor=actor,
            action=action,
            payload={**payload, "path": change.path},
            permission=permission,
        )
    return None


def _first_ask_change(
    changes: list[FileChange],
) -> tuple[FileChange, PermissionResult] | None:
    for change in changes:
        permission = decide_file_write(change.path)
        if permission.decision == "ask":
            return change, permission
    return None


def _first_denied_change(
    changes: list[FileChange],
) -> tuple[FileChange, PermissionResult] | None:
    for change in changes:
        permission = decide_file_write(change.path)
        if permission.decision == "deny":
            return change, permission
    return None


def _change_payload(change: FileChange, *, reason: str) -> dict[str, Any]:
    return {
        "path": change.path,
        "mode": change.mode,
        "reason": reason,
    }


def _change_to_payload(change: FileChange) -> dict[str, Any]:
    return {
        "path": change.path,
        "mode": change.mode,
        "content": change.content,
        "old_text": change.old_text,
        "new_text": change.new_text,
        "create_dirs": change.create_dirs,
        "expected_replacements": change.expected_replacements,
    }


def _coerce_limit(value: Any) -> int:
    try:
        return max(1, min(int(value or 20), 100))
    except (TypeError, ValueError):
        return 20


register_superuser_tool(PatchPrepareTool)
register_superuser_tool(PatchApplyTool)
register_superuser_tool(PatchRollbackTool)
register_superuser_tool(PatchShowTool)

__all__ = [
    "PatchApplyTool",
    "PatchPrepareTool",
    "PatchRollbackTool",
    "PatchShowTool",
    "apply_prepared_patch",
    "prepare_patch_operation",
    "rollback_prepared_patch",
]
