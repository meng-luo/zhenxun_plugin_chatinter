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
from ..workspace_isolation import resolve_working_path
from .common import (
    actor_from_context,
    approval_required_result,
    permission_denied_result,
    tool_result,
    worktree_id_from_context,
)

_ENGINEERING_LOOP_ID_FIELD = "engineering_loop_id"


class PatchPrepareTool:
    name = "patch_prepare"

    async def get_definition(self) -> ToolDefinition:
        schema = _patch_changes_schema(
            include_reason=True,
            description="本次修改原因，便于审计和回滚。",
        )
        schema["properties"][_ENGINEERING_LOOP_ID_FIELD] = {
            "type": ["string", "null"],
            "description": "可选：engineering_loop_start 返回的 loop_id，用于固定工程闭环。",
        }
        schema["required"].append(_ENGINEERING_LOOP_ID_FIELD)
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：准备一次事务式代码/文本修改，只生成 diff 和 "
                "operation_id，不直接写入。之后用 patch_apply 应用。"
            ),
            parameters=schema,
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
            worktree_id=worktree_id_from_context(context),
            changes=changes,
            reason=reason,
            engineering_loop_id=str(kwargs.get(_ENGINEERING_LOOP_ID_FIELD, "") or ""),
        )


async def prepare_patch_operation(
    *,
    actor: dict[str, str],
    changes: list[FileChange],
    reason: str,
    worktree_id: str = "",
    pre_resolved: bool = False,
    approval_id: str | None = None,
    engineering_loop_id: str = "",
) -> ToolResult:
    try:
        changes, isolation = _resolve_changes(
            actor=actor,
            changes=changes,
            worktree_id=worktree_id,
            pre_resolved=pre_resolved,
        )
    except Exception as exc:
        return tool_result(
            False,
            "patch_prepare_path_resolution_failed",
            error=str(exc),
            reason=reason,
        )
    patch_permission = decide_patch("patch_prepare")
    if patch_permission.decision == "deny":
        return permission_denied_result(
            actor=actor,
            action="patch_prepare",
            payload={
                "reason": reason,
                "files": [change.path for change in changes],
                "isolation": isolation,
            },
            permission=patch_permission,
        )
    if patch_permission.decision == "ask" and not approval_id:
        return approval_required_result(
            actor=actor,
            action="patch_prepare",
            payload={
                "reason": reason,
                "changes": [_change_to_payload(change) for change in changes],
                "pre_resolved": True,
                "isolation": isolation,
                _ENGINEERING_LOOP_ID_FIELD: engineering_loop_id,
            },
            permission=patch_permission,
        )
    deny = _first_denied_change(changes)
    if deny is not None:
        change, permission = deny
        return permission_denied_result(
            actor=actor,
            action="patch_prepare",
            payload={**_change_payload(change, reason=reason), "isolation": isolation},
            permission=permission,
        )
    try:
        _enforce_engineering_loop_patch_protocol(
            engineering_loop_id=engineering_loop_id,
            changes=changes,
        )
        operation = create_patch_operation(
            actor=actor,
            changes=changes,
            action="patch_prepare",
            reason=_reason_with_isolation(reason, isolation=isolation),
            approval_id=approval_id,
        )
    except Exception as exc:
        return tool_result(
            False,
            "patch_prepare_failed",
            error=str(exc),
            engineering_loop=_loop_payload(engineering_loop_id),
            retryable=True,
            need_continue=True,
            instruction=(
                "patch_prepare 被工程闭环协议拦截或生成失败。按返回的 "
                "engineering_loop/diagnosis 先重读代码或重新规划，不要跳过协议。"
            ),
        )
    _bind_engineering_loop_patch(
        engineering_loop_id=engineering_loop_id,
        operation_id=operation.operation_id,
    )
    return tool_result(
        True,
        "patch_prepared",
        operation=operation.public_payload(),
        engineering_loop=_loop_payload(engineering_loop_id),
        isolation=isolation,
        instruction=(
            "检查 diff 后，如需落地请调用 patch_apply。若属于工程闭环，"
            "patch_apply 后继续 engineering_eval_run 和 engineering_eval_gate。"
        ),
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
                    _ENGINEERING_LOOP_ID_FIELD: {
                        "type": ["string", "null"],
                        "description": "可选工程闭环 loop_id；传入后自动绑定 operation/eval。",
                    },
                },
                "required": ["operation_id", "reason", _ENGINEERING_LOOP_ID_FIELD],
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
            engineering_loop_id=str(kwargs.get(_ENGINEERING_LOOP_ID_FIELD, "") or ""),
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
                    _ENGINEERING_LOOP_ID_FIELD: {
                        "type": ["string", "null"],
                        "description": "可选工程闭环 loop_id；传入后记录 rollback 阶段。",
                    },
                },
                "required": ["operation_id", "reason", _ENGINEERING_LOOP_ID_FIELD],
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
            engineering_loop_id=str(kwargs.get(_ENGINEERING_LOOP_ID_FIELD, "") or ""),
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


def _resolve_changes(
    *,
    actor: dict[str, str],
    changes: list[FileChange],
    worktree_id: str,
    pre_resolved: bool,
) -> tuple[list[FileChange], dict[str, Any]]:
    if pre_resolved:
        return changes, {
            "isolated": False,
            "pre_resolved": True,
            "files": [
                {"requested": change.path, "resolved": change.path}
                for change in changes
            ],
        }
    resolved_changes: list[FileChange] = []
    resolutions: list[dict[str, Any]] = []
    for change in changes:
        path, isolation = resolve_working_path(
            change.path,
            actor=actor,
            worktree_id=worktree_id,
        )
        if isolation.get("invalid_worktree") or isolation.get("escaped_worktree"):
            raise ValueError("worktree path resolution failed for patch path")
        resolutions.append(isolation)
        resolved_changes.append(
            FileChange(
                path=path,
                mode=change.mode,
                content=change.content,
                old_text=change.old_text,
                new_text=change.new_text,
                create_dirs=change.create_dirs,
                expected_replacements=change.expected_replacements,
            )
        )
    return resolved_changes, _merge_isolation(resolutions)


def _merge_isolation(resolutions: list[dict[str, Any]]) -> dict[str, Any]:
    if not resolutions:
        return {"isolated": False}
    first = dict(resolutions[0])
    first["files"] = [
        {
            "requested": item.get("requested", ""),
            "resolved": item.get("resolved", ""),
            "mapped_from_main_workspace": bool(item.get("mapped_from_main_workspace")),
        }
        for item in resolutions
    ]
    first["all_isolated"] = all(bool(item.get("isolated")) for item in resolutions)
    return first


def _reason_with_isolation(reason: str, *, isolation: dict[str, Any]) -> str:
    text = str(reason or "")
    if not isolation.get("isolated"):
        return text
    if text.startswith("[worktree:"):
        return text
    prefix = (
        f"[worktree:{isolation.get('worktree_id', '')} "
        f"{isolation.get('branch_name', '')}] "
    )
    return prefix + text if text else prefix.rstrip()


async def apply_prepared_patch(
    *,
    actor: dict[str, str],
    operation_id: str,
    approval_id: str | None = None,
    reason: str = "",
    engineering_loop_id: str = "",
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
    result = apply_patch_operation(
        operation_id=operation_id,
        actor=actor,
        approval_id=approval_id,
    )
    _bind_engineering_loop_patch(
        engineering_loop_id=engineering_loop_id,
        operation_id=operation_id,
    )
    if isinstance(result.output, dict):
        result.output["engineering_loop"] = _loop_payload(engineering_loop_id)
    return result


async def rollback_prepared_patch(
    *,
    actor: dict[str, str],
    operation_id: str,
    approval_id: str | None = None,
    reason: str = "",
    engineering_loop_id: str = "",
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
    result = rollback_patch_operation(
        operation_id=operation_id,
        actor=actor,
        approval_id=approval_id,
    )
    _mark_engineering_loop_rollback(
        engineering_loop_id=engineering_loop_id,
        operation_id=operation_id,
    )
    if isinstance(result.output, dict):
        result.output["engineering_loop"] = _loop_payload(engineering_loop_id)
    return result


def _bind_engineering_loop_patch(
    *,
    engineering_loop_id: str,
    operation_id: str,
) -> None:
    if not engineering_loop_id or not operation_id:
        return
    try:
        from ..engineering_loop import bind_patch_operation

        bind_patch_operation(
            loop_id=engineering_loop_id,
            operation_id=operation_id,
        )
    except Exception:
        return


def _enforce_engineering_loop_patch_protocol(
    *,
    engineering_loop_id: str,
    changes: list[FileChange],
) -> None:
    if not engineering_loop_id:
        return
    try:
        from ..engineering_loop import get_engineering_loop

        loop = get_engineering_loop(engineering_loop_id)
    except Exception:
        loop = None
    if loop is None:
        return
    if loop.stage in {"created", "eval_failed"}:
        raise RuntimeError(
            "engineering loop protocol violation: call engineering_lsp_read and "
            "semantic_patch_plan before patch_prepare"
            if loop.stage == "created"
            else "engineering loop protocol violation: call engineering_failure_diagnose "
            "before preparing a second patch after eval failure"
        )
    if loop.diagnosis and not _loop_has_event_after(
        loop,
        event_kind="lsp_code_read",
        after_kind="failure_diagnosed",
    ):
        raise RuntimeError(
            "engineering loop protocol violation: reread code with engineering_lsp_read "
            "after engineering_failure_diagnose before second patch"
        )
    if loop.diagnosis and not _loop_has_event_after(
        loop,
        event_kind="semantic_patch_planned",
        after_kind="failure_diagnosed",
    ):
        raise RuntimeError(
            "engineering loop protocol violation: call semantic_patch_plan after "
            "engineering_failure_diagnose before second patch"
        )
    if not loop.semantic_patch_plan:
        raise RuntimeError(
            "engineering loop protocol violation: semantic_patch_plan is required before patch_prepare"
        )
    if loop.diagnosis:
        allowed = {str(path) for path in loop.diagnosis.get("files_to_reread", []) or []}
        changed = {str(change.path) for change in changes if str(change.path)}
        if allowed and not changed <= allowed:
            raise RuntimeError(
                "engineering loop protocol violation: second patch touches files outside diagnosis "
                f"scope: {sorted(changed - allowed)}"
            )


def _loop_has_event_after(
    loop: Any,
    *,
    event_kind: str,
    after_kind: str,
) -> bool:
    marker_ts = 0.0
    for event in getattr(loop, "events", []) or []:
        if getattr(event, "kind", "") == after_kind:
            marker_ts = max(marker_ts, float(getattr(event, "timestamp", 0.0) or 0.0))
    if not marker_ts:
        return False
    for event in getattr(loop, "events", []) or []:
        if (
            getattr(event, "kind", "") == event_kind
            and float(getattr(event, "timestamp", 0.0) or 0.0) > marker_ts
        ):
            return True
    return False


def _mark_engineering_loop_rollback(
    *,
    engineering_loop_id: str,
    operation_id: str,
) -> None:
    if not engineering_loop_id or not operation_id:
        return
    try:
        from ..engineering_loop import mark_loop_rolled_back

        mark_loop_rolled_back(
            loop_id=engineering_loop_id,
            operation_id=operation_id,
        )
    except Exception:
        return


def _loop_payload(engineering_loop_id: str) -> dict[str, Any] | None:
    if not engineering_loop_id:
        return None
    try:
        from ..engineering_loop import get_engineering_loop

        loop = get_engineering_loop(engineering_loop_id)
        return loop.public_payload(include_events=False) if loop else None
    except Exception:
        return None


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
