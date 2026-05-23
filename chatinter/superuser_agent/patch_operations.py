"""Durable diff/patch/rollback operation layer for superuser Agent writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import difflib
import hashlib
from pathlib import Path
import time
import uuid
from typing import Any, Literal

from zhenxun.services.llm.types.models import ToolResult

from ..persistence import read_json, state_path, write_json
from .audit_log import record_audit_event

ChangeMode = Literal["write", "append", "replace"]
OperationStatus = Literal["prepared", "applied", "rolled_back", "failed"]

_OPERATIONS_PATH = state_path("patch_operations.json")
_MAX_DIFF_CHARS = 12000
_MAX_CONTENT_CHARS = 1_000_000
_OPERATIONS: dict[str, "PatchOperation"] = {}
_LOADED = False


@dataclass(frozen=True)
class FileChange:
    path: str
    mode: ChangeMode
    content: str = ""
    old_text: str = ""
    new_text: str = ""
    create_dirs: bool = False
    expected_replacements: int | None = None


@dataclass
class FileSnapshot:
    path: str
    existed_before: bool
    before_content: str = ""
    after_content: str = ""
    diff: str = ""
    replacements: int = 0
    before_sha256: str = ""
    after_sha256: str = ""
    before_size: int = 0
    after_size: int = 0
    before_mtime_ns: int = 0
    after_mtime_ns: int = 0


@dataclass
class PatchOperation:
    operation_id: str
    user_id: str
    session_key: str
    action: str
    reason: str
    changes: list[FileChange]
    snapshots: list[FileSnapshot]
    status: OperationStatus = "prepared"
    approval_id: str | None = None
    bound_eval_id: str = ""
    workspace_lock: dict[str, str] = field(default_factory=dict)
    workspace_lock_details: dict[str, Any] = field(default_factory=dict)
    pre_checkpoint_id: str = ""
    post_checkpoint_id: str = ""
    rollback_checkpoint_id: str = ""
    failure_checkpoint_id: str = ""
    last_recovery_plan: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""

    def public_payload(self, *, include_content: bool = False) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "user_id": self.user_id,
            "session_key": self.session_key,
            "action": self.action,
            "reason": self.reason,
            "status": self.status,
            "approval_id": self.approval_id,
            "bound_eval_id": self.bound_eval_id,
            "workspace_lock": dict(self.workspace_lock),
            "workspace_lock_details": dict(self.workspace_lock_details),
            "pre_checkpoint_id": self.pre_checkpoint_id,
            "post_checkpoint_id": self.post_checkpoint_id,
            "rollback_checkpoint_id": self.rollback_checkpoint_id,
            "failure_checkpoint_id": self.failure_checkpoint_id,
            "last_recovery_plan": dict(self.last_recovery_plan),
            "created_at": int(self.created_at),
            "updated_at": int(self.updated_at),
            "error": self.error,
            "files": [
                {
                    "path": snapshot.path,
                    "existed_before": snapshot.existed_before,
                    "before_chars": len(snapshot.before_content),
                    "after_chars": len(snapshot.after_content),
                    "replacements": snapshot.replacements,
                    "before_sha256": snapshot.before_sha256,
                    "after_sha256": snapshot.after_sha256,
                    "before_size": snapshot.before_size,
                    "after_size": snapshot.after_size,
                    "before_mtime_ns": snapshot.before_mtime_ns,
                    "after_mtime_ns": snapshot.after_mtime_ns,
                    "diff": snapshot.diff,
                    **(
                        {
                            "before_content": snapshot.before_content,
                            "after_content": snapshot.after_content,
                        }
                        if include_content
                        else {}
                    ),
                }
                for snapshot in self.snapshots
            ],
            "changes": [asdict(change) for change in self.changes],
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "user_id": self.user_id,
            "session_key": self.session_key,
            "action": self.action,
            "reason": self.reason,
            "changes": [asdict(change) for change in self.changes],
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
            "status": self.status,
            "approval_id": self.approval_id,
            "bound_eval_id": self.bound_eval_id,
            "workspace_lock": dict(self.workspace_lock),
            "workspace_lock_details": dict(self.workspace_lock_details),
            "pre_checkpoint_id": self.pre_checkpoint_id,
            "post_checkpoint_id": self.post_checkpoint_id,
            "rollback_checkpoint_id": self.rollback_checkpoint_id,
            "failure_checkpoint_id": self.failure_checkpoint_id,
            "last_recovery_plan": dict(self.last_recovery_plan),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }


def create_patch_operation(
    *,
    actor: dict[str, str],
    changes: list[FileChange],
    action: str,
    reason: str = "",
    approval_id: str | None = None,
) -> PatchOperation:
    _ensure_loaded()
    if not changes:
        raise ValueError("patch operation requires at least one change")
    conflict = _prepared_operation_conflict(changes)
    if conflict:
        raise RuntimeError(conflict)
    snapshots = [_build_snapshot(change) for change in changes]
    operation = PatchOperation(
        operation_id=uuid.uuid4().hex[:12],
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=action,
        reason=reason,
        changes=changes,
        snapshots=snapshots,
        approval_id=approval_id,
        workspace_lock=_workspace_lock_for_snapshots(snapshots),
        workspace_lock_details=_workspace_lock_details_for_snapshots(snapshots),
    )
    operation.pre_checkpoint_id = _write_checkpoint(operation, phase="pre_apply")
    _OPERATIONS[operation.operation_id] = operation
    _save_operations()
    record_audit_event(
        event="patch_operation_prepared",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action=action,
        payload={
            "operation_id": operation.operation_id,
            "reason": reason,
            "files": [snapshot.path for snapshot in snapshots],
            "approval_id": approval_id,
        },
    )
    return operation


def apply_patch_operation(
    *,
    operation_id: str,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    operation = get_patch_operation(operation_id)
    if operation is None:
        return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
    if operation.user_id != actor["user_id"] or operation.session_key != actor["session_key"]:
        return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
    if operation.status == "applied":
        return tool_result(
            True,
            "patch_operation_already_applied",
            operation=operation.public_payload(),
        )
    if operation.status == "rolled_back":
        return tool_result(False, "patch_operation_already_rolled_back", operation_id=operation_id)
    if operation.status == "failed":
        return tool_result(
            False,
            "patch_operation_failed_needs_reprepare",
            operation=operation.public_payload(),
            recovery_plan=operation.last_recovery_plan
            or _patch_recovery_plan(
                operation,
                phase="apply",
                error=operation.error or "operation already failed",
            ),
            instruction="该 patch operation 已失败。为避免旧快照误写，请重读文件后重新 patch_prepare。",
        )
    try:
        lock_error = _workspace_lock_error(operation)
        if lock_error:
            raise RuntimeError(lock_error)
        conflict = _prepared_operation_conflict(
            operation.changes,
            ignore_operation_id=operation.operation_id,
        )
        if conflict:
            raise RuntimeError(conflict)
        for change, snapshot in zip(operation.changes, operation.snapshots, strict=False):
            target = Path(change.path)
            if change.create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            elif not target.parent.exists():
                raise FileNotFoundError(str(target.parent))
        for snapshot in operation.snapshots:
            if not _snapshot_matches_current(snapshot):
                raise RuntimeError(
                    f"file changed since patch_prepare: {snapshot.path}"
                )
        touched: list[FileSnapshot] = []
        for change, snapshot in zip(operation.changes, operation.snapshots, strict=False):
            target = Path(change.path)
            target.write_text(snapshot.after_content, encoding="utf-8")
            snapshot.after_mtime_ns = _mtime_ns(target)
            snapshot.after_size = _file_size(target)
            snapshot.after_sha256 = _sha256(
                target.read_text(encoding="utf-8", errors="replace")
            )
            touched.append(snapshot)
        operation.status = "applied"
        operation.approval_id = approval_id or operation.approval_id
        operation.post_checkpoint_id = _write_checkpoint(operation, phase="post_apply")
        operation.updated_at = time.time()
        operation.error = ""
        _save_operations()
        eval_run = _bind_eval_after_apply(operation, actor=actor)
        record_audit_event(
            event="patch_operation_applied",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=operation.action,
            payload={
                "operation_id": operation.operation_id,
                "approval_id": operation.approval_id,
                "files": [snapshot.path for snapshot in operation.snapshots],
            },
            result={"ok": True},
        )
        checkpoints = _checkpoint_payload(operation)
        return tool_result(
            True,
            "patch_operation_applied",
            operation=operation.public_payload(),
            eval=eval_run.public_payload() if eval_run is not None else None,
            checkpoints=checkpoints,
            next_tool="engineering_eval_run" if eval_run is not None else "engineering_eval_plan",
            instruction=(
                "patch 已应用并绑定工程 eval。下一步调用 engineering_eval_run "
                "执行建议测试；如果失败，根据 observation 决定重读、二次 patch 或 rollback。"
            ),
        )
    except Exception as exc:
        _restore_snapshots(locals().get("touched", []))
        operation.status = "failed"
        operation.error = str(exc)
        operation.failure_checkpoint_id = _write_checkpoint(
            operation,
            phase="apply_failed",
            error=str(exc),
        )
        operation.last_recovery_plan = _patch_recovery_plan(
            operation,
            phase="apply",
            error=str(exc),
        )
        operation.updated_at = time.time()
        _save_operations()
        record_audit_event(
            event="patch_operation_failed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=operation.action,
            payload={"operation_id": operation.operation_id},
            result={"error": str(exc)},
        )
        return tool_result(
            False,
            "patch_operation_apply_failed",
            operation_id=operation.operation_id,
            operation=operation.public_payload(),
            error=str(exc),
            recovery_plan=operation.last_recovery_plan,
            checkpoints=_checkpoint_payload(operation),
            retryable=True,
            need_continue=True,
            instruction=(
                "patch_apply 失败。不要重复应用同一个 stale operation；按 recovery_plan "
                "重读冲突文件，重新 patch_prepare，必要时先检查 checkpoint。"
            ),
        )


def rollback_patch_operation(
    *,
    operation_id: str,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    operation = get_patch_operation(operation_id)
    if operation is None:
        return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
    if operation.user_id != actor["user_id"] or operation.session_key != actor["session_key"]:
        return tool_result(False, "patch_operation_not_found", operation_id=operation_id)
    if operation.status == "rolled_back":
        return tool_result(
            True,
            "patch_operation_already_rolled_back",
            operation=operation.public_payload(),
        )
    if operation.status != "applied":
        return tool_result(
            False,
            "patch_operation_not_applied",
            operation=operation.public_payload(),
        )
    try:
        lock_error = _rollback_lock_error(operation)
        if lock_error:
            raise RuntimeError(lock_error)
        operation.rollback_checkpoint_id = _write_checkpoint(
            operation,
            phase="pre_rollback",
        )
        for snapshot in reversed(operation.snapshots):
            target = Path(snapshot.path)
            if snapshot.existed_before:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(snapshot.before_content, encoding="utf-8")
            elif target.exists():
                target.unlink()
        operation.status = "rolled_back"
        operation.approval_id = approval_id or operation.approval_id
        operation.updated_at = time.time()
        operation.error = ""
        _save_operations()
        record_audit_event(
            event="patch_operation_rolled_back",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=operation.action,
            payload={
                "operation_id": operation.operation_id,
                "approval_id": approval_id,
                "files": [snapshot.path for snapshot in operation.snapshots],
            },
            result={"ok": True},
        )
        return tool_result(
            True,
            "patch_operation_rolled_back",
            operation=operation.public_payload(),
            checkpoints=_checkpoint_payload(operation),
            instruction="回滚已完成。如需继续工程任务，请重读相关文件并准备新的 patch/eval。",
        )
    except Exception as exc:
        operation.status = "failed"
        operation.error = str(exc)
        operation.failure_checkpoint_id = _write_checkpoint(
            operation,
            phase="rollback_failed",
            error=str(exc),
        )
        operation.last_recovery_plan = _patch_recovery_plan(
            operation,
            phase="rollback",
            error=str(exc),
        )
        operation.updated_at = time.time()
        _save_operations()
        return tool_result(
            False,
            "patch_operation_rollback_failed",
            operation_id=operation.operation_id,
            operation=operation.public_payload(),
            error=str(exc),
            recovery_plan=operation.last_recovery_plan,
            checkpoints=_checkpoint_payload(operation),
            retryable=True,
            need_continue=True,
        )


def apply_changes_transaction(
    *,
    actor: dict[str, str],
    changes: list[FileChange],
    action: str,
    reason: str = "",
    approval_id: str | None = None,
) -> ToolResult:
    try:
        operation = create_patch_operation(
            actor=actor,
            changes=changes,
            action=action,
            reason=reason,
            approval_id=approval_id,
        )
    except Exception as exc:
        return tool_result(False, "patch_operation_prepare_failed", error=str(exc))
    result = apply_patch_operation(
        operation_id=operation.operation_id,
        actor=actor,
        approval_id=approval_id,
    )
    if isinstance(result.output, dict):
        result.output.setdefault("operation_id", operation.operation_id)
    return result


def get_patch_operation(operation_id: str) -> PatchOperation | None:
    _ensure_loaded()
    return _OPERATIONS.get(str(operation_id or "").strip())


def list_patch_operations(
    *,
    user_id: str,
    session_key: str,
    limit: int = 20,
) -> list[PatchOperation]:
    _ensure_loaded()
    limit = max(1, min(int(limit or 20), 100))
    operations = [
        operation
        for operation in _OPERATIONS.values()
        if operation.user_id == str(user_id or "")
        and operation.session_key == str(session_key or "")
    ]
    return sorted(operations, key=lambda item: item.updated_at, reverse=True)[:limit]


def normalize_change(raw: dict[str, Any]) -> FileChange:
    mode = str(raw.get("mode", "") or "").strip()
    if mode not in {"write", "append", "replace"}:
        raise ValueError("change.mode must be write, append, or replace")
    expected = raw.get("expected_replacements")
    expected_replacements = None
    if expected not in (None, ""):
        expected_replacements = max(1, int(expected))
    return FileChange(
        path=str(raw.get("path", "") or "").strip(),
        mode=mode,  # type: ignore[arg-type]
        content=str(raw.get("content", "") or ""),
        old_text=str(raw.get("old_text", "") or ""),
        new_text=str(raw.get("new_text", "") or ""),
        create_dirs=bool(raw.get("create_dirs") or False),
        expected_replacements=expected_replacements,
    )


def _build_snapshot(change: FileChange) -> FileSnapshot:
    if not change.path:
        raise ValueError("change.path is required")
    target = Path(change.path)
    existed = target.exists()
    before = target.read_text(encoding="utf-8", errors="replace") if existed else ""
    if len(before) > _MAX_CONTENT_CHARS or len(change.content) > _MAX_CONTENT_CHARS:
        raise ValueError("patch content is too large")
    after, replacements = _render_after_content(before, change)
    diff = _unified_diff(path=change.path, before=before, after=after)
    return FileSnapshot(
        path=str(target),
        existed_before=existed,
        before_content=before,
        after_content=after,
        diff=diff,
        replacements=replacements,
        before_sha256=_sha256(before),
        after_sha256=_sha256(after),
        before_size=_file_size(target) if existed else 0,
        after_size=len(after.encode("utf-8")),
        before_mtime_ns=_mtime_ns(target) if existed else 0,
        after_mtime_ns=0,
    )


def _render_after_content(before: str, change: FileChange) -> tuple[str, int]:
    if change.mode == "write":
        return change.content, 0
    if change.mode == "append":
        return before + change.content, 0
    if not change.old_text:
        raise ValueError("replace change requires old_text")
    replacements = before.count(change.old_text)
    if replacements <= 0:
        raise ValueError(f"replace text not found in {change.path}")
    if (
        change.expected_replacements is not None
        and replacements != change.expected_replacements
    ):
        raise ValueError(
            f"replace count mismatch in {change.path}: "
            f"found={replacements}, expected={change.expected_replacements}"
        )
    return before.replace(change.old_text, change.new_text), replacements


def _unified_diff(*, path: str, before: str, after: str) -> str:
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    return _compact_text(diff, max_chars=_MAX_DIFF_CHARS)


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    raw = read_json(_OPERATIONS_PATH, {})
    if not isinstance(raw, dict):
        return
    for operation_id, payload in raw.items():
        operation = _operation_from_payload(operation_id, payload)
        if operation is not None:
            _OPERATIONS[operation.operation_id] = operation


def _operation_from_payload(
    operation_id: object,
    payload: object,
) -> PatchOperation | None:
    if not isinstance(payload, dict):
        return None
    try:
        changes = [
            FileChange(**change)
            for change in payload.get("changes", [])
            if isinstance(change, dict)
        ]
        snapshots = [
            _snapshot_from_payload(snapshot)
            for snapshot in payload.get("snapshots", [])
            if isinstance(snapshot, dict)
        ]
        return PatchOperation(
            operation_id=str(payload.get("operation_id") or operation_id or ""),
            user_id=str(payload.get("user_id", "") or ""),
            session_key=str(payload.get("session_key", "") or ""),
            action=str(payload.get("action", "") or ""),
            reason=str(payload.get("reason", "") or ""),
            changes=changes,
            snapshots=snapshots,
            status=str(payload.get("status", "") or "prepared"),  # type: ignore[arg-type]
            approval_id=str(payload.get("approval_id", "") or "") or None,
            bound_eval_id=str(payload.get("bound_eval_id", "") or ""),
            workspace_lock=dict(payload.get("workspace_lock") or {}),
            workspace_lock_details=dict(payload.get("workspace_lock_details") or {}),
            pre_checkpoint_id=str(payload.get("pre_checkpoint_id", "") or ""),
            post_checkpoint_id=str(payload.get("post_checkpoint_id", "") or ""),
            rollback_checkpoint_id=str(payload.get("rollback_checkpoint_id", "") or ""),
            failure_checkpoint_id=str(payload.get("failure_checkpoint_id", "") or ""),
            last_recovery_plan=dict(payload.get("last_recovery_plan") or {}),
            created_at=float(payload.get("created_at") or time.time()),
            updated_at=float(payload.get("updated_at") or time.time()),
            error=str(payload.get("error", "") or ""),
        )
    except Exception:
        return None


def _save_operations() -> None:
    write_json(
        _OPERATIONS_PATH,
        {
            operation_id: operation.to_record()
            for operation_id, operation in sorted(_OPERATIONS.items())
        },
    )


def _restore_snapshots(snapshots: list[FileSnapshot]) -> None:
    for snapshot in reversed(snapshots):
        target = Path(snapshot.path)
        try:
            if snapshot.existed_before:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(snapshot.before_content, encoding="utf-8")
            elif target.exists():
                target.unlink()
        except Exception:
            continue


def _snapshot_matches_current(snapshot: FileSnapshot) -> bool:
    target = Path(snapshot.path)
    if not snapshot.existed_before:
        return not target.exists()
    if not target.exists():
        return False
    current = target.read_text(encoding="utf-8", errors="replace")
    return _sha256(current) == snapshot.before_sha256


def _snapshot_matches_after(snapshot: FileSnapshot) -> bool:
    target = Path(snapshot.path)
    if not target.exists():
        return False
    current = target.read_text(encoding="utf-8", errors="replace")
    return _sha256(current) == snapshot.after_sha256


def _workspace_lock_for_snapshots(snapshots: list[FileSnapshot]) -> dict[str, str]:
    return {
        snapshot.path: snapshot.before_sha256
        for snapshot in snapshots
        if snapshot.existed_before
    }


def _workspace_lock_details_for_snapshots(
    snapshots: list[FileSnapshot],
) -> dict[str, Any]:
    return {
        snapshot.path: {
            "existed_before": snapshot.existed_before,
            "before_sha256": snapshot.before_sha256,
            "before_size": snapshot.before_size,
            "before_mtime_ns": snapshot.before_mtime_ns,
            "parent_exists": Path(snapshot.path).parent.exists(),
            "parent_mtime_ns": _mtime_ns(Path(snapshot.path).parent),
            "path_kind": _path_kind(Path(snapshot.path)),
        }
        for snapshot in snapshots
    }


def _workspace_lock_error(operation: PatchOperation) -> str:
    for snapshot in operation.snapshots:
        target = Path(snapshot.path)
        if not snapshot.existed_before:
            if target.exists():
                return _lock_error_message(
                    operation,
                    snapshot,
                    phase="before_apply",
                    reason="new_file_already_exists",
                )
            continue
        if not _snapshot_matches_current(snapshot):
            return _lock_error_message(
                operation,
                snapshot,
                phase="before_apply",
                reason="content_changed_since_prepare",
            )
        if snapshot.before_size and _file_size(target) != snapshot.before_size:
            return _lock_error_message(
                operation,
                snapshot,
                phase="before_apply",
                reason="size_changed_since_prepare",
            )
    return ""


def _rollback_lock_error(operation: PatchOperation) -> str:
    for snapshot in operation.snapshots:
        if not snapshot.existed_before and not Path(snapshot.path).exists():
            continue
        if not _snapshot_matches_after(snapshot):
            return _lock_error_message(
                operation,
                snapshot,
                phase="before_rollback",
                reason="content_changed_after_apply",
            )
    return ""


def _bind_eval_after_apply(
    operation: PatchOperation,
    *,
    actor: dict[str, str],
):
    if operation.bound_eval_id:
        try:
            from .engineering_eval import get_engineering_eval

            eval_run = get_engineering_eval(operation.bound_eval_id)
            if eval_run is not None:
                return eval_run
        except Exception:
            pass
    try:
        from .engineering_eval import create_engineering_eval, suggest_test_commands

        files = [snapshot.path for snapshot in operation.snapshots]
        eval_run = create_engineering_eval(
            actor=actor,
            task=operation.reason or operation.action or "patch apply validation",
            files=files,
            tests=suggest_test_commands(files=files, task=operation.reason),
            rollback_operation_id=operation.operation_id,
            patch_operation_id=operation.operation_id,
        )
        operation.bound_eval_id = eval_run.eval_id
        operation.updated_at = time.time()
        _save_operations()
        return eval_run
    except Exception as exc:
        operation.last_recovery_plan = {
            "phase": "bind_eval_after_apply",
            "operation_id": operation.operation_id,
            "error": str(exc),
            "recommended_next_action": "call_engineering_eval_plan_manually",
            "next_tools": ["engineering_eval_plan", "engineering_eval_status"],
            "files_to_validate": [snapshot.path for snapshot in operation.snapshots],
            "instruction": (
                "Patch applied but automatic eval binding failed. Create an "
                "engineering_eval_plan for the changed files before finalizing."
            ),
        }
        operation.updated_at = time.time()
        _save_operations()
        return None


def _snapshot_from_payload(payload: dict[str, Any]) -> FileSnapshot:
    before = str(payload.get("before_content", "") or "")
    after = str(payload.get("after_content", "") or "")
    return FileSnapshot(
        path=str(payload.get("path", "") or ""),
        existed_before=bool(payload.get("existed_before")),
        before_content=before,
        after_content=after,
        diff=str(payload.get("diff", "") or ""),
        replacements=int(payload.get("replacements", 0) or 0),
        before_sha256=str(payload.get("before_sha256", "") or "") or _sha256(before),
        after_sha256=str(payload.get("after_sha256", "") or "") or _sha256(after),
        before_size=int(payload.get("before_size", 0) or 0),
        after_size=int(payload.get("after_size", 0) or 0),
        before_mtime_ns=int(payload.get("before_mtime_ns", 0) or 0),
        after_mtime_ns=int(payload.get("after_mtime_ns", 0) or 0),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return 0


def _prepared_operation_conflict(
    changes: list[FileChange],
    *,
    ignore_operation_id: str = "",
) -> str:
    _ensure_loaded()
    paths = {str(Path(change.path)) for change in changes if change.path}
    if not paths:
        return ""
    for operation in _OPERATIONS.values():
        if operation.operation_id == ignore_operation_id:
            continue
        if operation.status != "prepared":
            continue
        other_paths = {snapshot.path for snapshot in operation.snapshots}
        overlap = sorted(paths & other_paths)
        if overlap:
            return (
                "workspace dirty lock failed: another prepared patch operation "
                f"touches {overlap[0]} (operation_id={operation.operation_id})"
            )
    return ""


def _write_checkpoint(
    operation: PatchOperation,
    *,
    phase: str,
    error: str = "",
) -> str:
    checkpoint_id = f"{operation.operation_id}_{phase}_{uuid.uuid4().hex[:8]}"
    write_json(
        state_path("patch_checkpoints", f"{checkpoint_id}.json"),
        {
            "checkpoint_id": checkpoint_id,
            "operation_id": operation.operation_id,
            "phase": phase,
            "created_at": time.time(),
            "status": operation.status,
            "action": operation.action,
            "reason": operation.reason,
            "error": error,
            "approval_id": operation.approval_id,
            "workspace_lock_details": operation.workspace_lock_details,
            "current_workspace": {
                snapshot.path: _current_file_state(snapshot.path)
                for snapshot in operation.snapshots
            },
            "files": [
                {
                    "path": snapshot.path,
                    "existed_before": snapshot.existed_before,
                    "before_sha256": snapshot.before_sha256,
                    "after_sha256": snapshot.after_sha256,
                    "before_size": snapshot.before_size,
                    "after_size": snapshot.after_size,
                    "before_mtime_ns": snapshot.before_mtime_ns,
                    "after_mtime_ns": snapshot.after_mtime_ns,
                    "diff": snapshot.diff,
                    "before_content": snapshot.before_content,
                    "after_content": snapshot.after_content,
                }
                for snapshot in operation.snapshots
            ],
        },
    )
    return checkpoint_id


def _checkpoint_payload(operation: PatchOperation) -> dict[str, str]:
    return {
        "pre_checkpoint_id": operation.pre_checkpoint_id,
        "post_checkpoint_id": operation.post_checkpoint_id,
        "rollback_checkpoint_id": operation.rollback_checkpoint_id,
        "failure_checkpoint_id": operation.failure_checkpoint_id,
    }


def _patch_recovery_plan(
    operation: PatchOperation,
    *,
    phase: str,
    error: str,
) -> dict[str, Any]:
    files = [snapshot.path for snapshot in operation.snapshots]
    lower = str(error or "").lower()
    dirty_lock = "workspace dirty lock failed" in lower or "file changed" in lower
    rollback = phase == "rollback"
    next_tools = ["patch_show"]
    if dirty_lock or rollback:
        next_tools.extend(["read_file", "search_files"])
    if not rollback:
        next_tools.extend(["patch_prepare", "patch_apply"])
    else:
        next_tools.extend(["read_file", "patch_prepare"])
    return {
        "phase": phase,
        "error": str(error or ""),
        "operation_id": operation.operation_id,
        "status": operation.status,
        "dirty_lock_failed": dirty_lock,
        "files_to_reread": files[:20],
        "checkpoint_ids": _checkpoint_payload(operation),
        "next_tools": _dedupe(next_tools),
        "recommended_next_action": (
            "reread_changed_files_then_reprepare_patch"
            if dirty_lock
            else "inspect_checkpoint_then_reprepare_patch"
        ),
        "instruction": (
            "Treat this patch operation as stale. Do not retry the same operation "
            "unchanged. Use patch_show/read_file to inspect current files, then "
            "prepare a new focused patch. If rollback failed, avoid overwriting "
            "unknown user changes until the current content has been reviewed."
        ),
    }


def _lock_error_message(
    operation: PatchOperation,
    snapshot: FileSnapshot,
    *,
    phase: str,
    reason: str,
) -> str:
    current = _current_file_state(snapshot.path)
    return (
        "workspace dirty lock failed: "
        f"{reason}: {snapshot.path} "
        f"(operation_id={operation.operation_id}, phase={phase}, "
        f"expected_before_sha256={snapshot.before_sha256}, "
        f"expected_after_sha256={snapshot.after_sha256}, "
        f"current_sha256={current.get('sha256', '')}, "
        f"current_size={current.get('size', 0)}, "
        f"current_mtime_ns={current.get('mtime_ns', 0)})"
    )


def _current_file_state(path: str) -> dict[str, Any]:
    target = Path(path)
    existed = target.exists()
    content = ""
    if existed and target.is_file():
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = ""
    return {
        "exists": existed,
        "kind": _path_kind(target),
        "sha256": _sha256(content) if existed and target.is_file() else "",
        "size": _file_size(target) if existed else 0,
        "mtime_ns": _mtime_ns(target) if existed else 0,
        "parent_exists": target.parent.exists(),
        "parent_mtime_ns": _mtime_ns(target.parent),
    }


def _path_kind(path: Path) -> str:
    try:
        if path.is_file():
            return "file"
        if path.is_dir():
            return "dir"
        if path.exists():
            return "other"
    except Exception:
        return "unknown"
    return "missing"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _compact_text(value: str, *, max_chars: int) -> str:
    return str(value or "")[: max(1, max_chars)]


def tool_result(ok: bool, status: str, **payload: Any) -> ToolResult:
    return ToolResult(
        output={"ok": ok, "status": status, **payload},
        display_content=status,
    )


__all__ = [
    "FileChange",
    "PatchOperation",
    "apply_changes_transaction",
    "apply_patch_operation",
    "create_patch_operation",
    "get_patch_operation",
    "list_patch_operations",
    "normalize_change",
    "rollback_patch_operation",
]
