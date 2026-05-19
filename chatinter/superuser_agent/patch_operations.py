"""Durable diff/patch/rollback operation layer for superuser Agent writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import difflib
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
    )
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
    touched: list[FileSnapshot] = []
    try:
        for change, snapshot in zip(operation.changes, operation.snapshots, strict=False):
            target = Path(change.path)
            if change.create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            elif not target.parent.exists():
                raise FileNotFoundError(str(target.parent))
        for change, snapshot in zip(operation.changes, operation.snapshots, strict=False):
            target = Path(change.path)
            target.write_text(snapshot.after_content, encoding="utf-8")
            touched.append(snapshot)
        operation.status = "applied"
        operation.approval_id = approval_id or operation.approval_id
        operation.updated_at = time.time()
        operation.error = ""
        _save_operations()
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
        return tool_result(
            True,
            "patch_operation_applied",
            operation=operation.public_payload(),
        )
    except Exception as exc:
        _restore_snapshots(touched)
        operation.status = "failed"
        operation.error = str(exc)
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
            error=str(exc),
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
        )
    except Exception as exc:
        operation.status = "failed"
        operation.error = str(exc)
        operation.updated_at = time.time()
        _save_operations()
        return tool_result(
            False,
            "patch_operation_rollback_failed",
            operation_id=operation.operation_id,
            error=str(exc),
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
            FileSnapshot(**snapshot)
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
