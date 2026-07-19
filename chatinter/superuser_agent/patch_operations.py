"""Atomic text-file writes for the fixed Superuser tool set."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Literal

from ..llm_compat import ToolResult

ChangeMode = Literal["write", "replace"]


@dataclass(frozen=True)
class FileChange:
    path: str
    mode: ChangeMode
    content: str = ""
    old_text: str = ""
    new_text: str = ""
    create_dirs: bool = False
    expected_replacements: int | None = None


def apply_changes_transaction(
    *,
    actor: dict[str, str],
    changes: list[FileChange],
    action: str,
    reason: str = "",
    approval_id: str | None = None,
) -> ToolResult:
    del actor, reason
    if len(changes) != 1:
        return _result(False, "patch_operation_apply_failed", error="one file required")
    change = changes[0]
    try:
        file_result = _apply_atomic_change(change)
    except Exception as exc:
        return _result(
            False,
            "patch_operation_apply_failed",
            error=str(exc),
            retryable=True,
            need_continue=True,
        )
    return _result(
        True,
        "patch_operation_applied",
        operation={
            "action": action,
            "approval_id": approval_id,
            "files": [file_result],
        },
    )


def _apply_atomic_change(change: FileChange) -> dict[str, Any]:
    if not change.path:
        raise ValueError("path is required")
    target = Path(change.path)
    existed = target.exists()
    if existed and not target.is_file():
        raise ValueError(f"not a file: {target}")
    before = target.read_text(encoding="utf-8") if existed else ""
    after, replacements = _render_content(before, change)

    if change.create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    elif not target.parent.exists():
        raise FileNotFoundError(str(target.parent))

    fd, temp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(after)
            handle.flush()
            os.fsync(handle.fileno())
        if existed:
            if not target.is_file() or target.read_text(encoding="utf-8") != before:
                raise RuntimeError(f"file changed during update: {target}")
            os.chmod(temp_path, stat.S_IMODE(target.stat().st_mode))
        elif target.exists():
            raise RuntimeError(f"file appeared during update: {target}")
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "path": str(target),
        "before_chars": len(before),
        "after_chars": len(after),
        "replacements": replacements,
        "before_sha256": _sha256(before),
        "after_sha256": _sha256(after),
    }


def _render_content(before: str, change: FileChange) -> tuple[str, int]:
    if change.mode == "write":
        return change.content, 0
    if change.mode != "replace":
        raise ValueError(f"unsupported change mode: {change.mode}")
    if not change.old_text:
        raise ValueError("old_text is required")
    replacements = before.count(change.old_text)
    if replacements == 0:
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


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _result(ok: bool, status: str, **payload: Any) -> ToolResult:
    return ToolResult(
        output={"ok": ok, "status": status, **payload},
        display_content=status,
    )


__all__ = ["FileChange", "apply_changes_transaction"]
