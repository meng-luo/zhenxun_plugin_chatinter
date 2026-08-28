"""Atomic text-file writes for the fixed Superuser tool set."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
from itertools import islice
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Literal

from ..llm_compat import ToolResult

ChangeMode = Literal["write", "replace", "delete"]


@dataclass(frozen=True)
class FileChange:
    path: str
    mode: ChangeMode
    content: str = ""
    old_text: str = ""
    new_text: str = ""
    create_dirs: bool = False
    expected_replacements: int | None = None
    replace_all: bool = False
    expected_sha256: str = ""
    require_absent: bool = False


@dataclass
class _PreparedChange:
    change: FileChange
    target: Path
    existed: bool
    before: str
    after: str
    replacements: int
    staged: Path | None = None
    backup: Path | None = None


def apply_changes_transaction(
    *,
    actor: dict[str, str],
    changes: list[FileChange],
    action: str,
    reason: str = "",
    approval_id: str | None = None,
) -> ToolResult:
    del actor, reason
    if not changes or len(changes) > 20:
        return _result(
            False,
            "patch_operation_apply_failed",
            error="between 1 and 20 file changes are required",
        )
    try:
        if len(changes) == 1 and changes[0].mode != "delete":
            file_results = [_apply_atomic_change(changes[0])]
        else:
            file_results = _apply_multiple_changes(changes)
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
            "files": file_results,
        },
    )


def _apply_multiple_changes(changes: list[FileChange]) -> list[dict[str, Any]]:
    targets = [Path(change.path).resolve() for change in changes]
    normalized = [os.path.normcase(str(target)) for target in targets]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate file paths are not allowed")

    prepared = [
        _prepare_change(change, target)
        for change, target in zip(changes, targets)
    ]
    committed: list[_PreparedChange] = []
    try:
        for item in prepared:
            if item.change.create_dirs:
                item.target.parent.mkdir(parents=True, exist_ok=True)
            elif not item.target.parent.exists():
                raise FileNotFoundError(str(item.target.parent))
            if item.change.mode != "delete":
                item.staged = _write_temp(item.target, item.after, suffix=".tmp")
            if item.existed:
                item.backup = _write_temp(item.target, item.before, suffix=".bak")

        for item in prepared:
            if not _target_unchanged(item):
                raise RuntimeError(f"file changed during update: {item.target}")

        for item in prepared:
            if item.change.mode == "delete":
                item.target.unlink()
            else:
                assert item.staged is not None
                os.replace(item.staged, item.target)
                item.staged = None
            committed.append(item)
    except Exception as exc:
        rollback_errors = _rollback_changes(committed)
        if rollback_errors:
            raise RuntimeError(
                f"{exc}; rollback failed: {'; '.join(rollback_errors)}"
            ) from exc
        raise
    finally:
        for item in prepared:
            if item.staged is not None:
                item.staged.unlink(missing_ok=True)
            if item.backup is not None:
                item.backup.unlink(missing_ok=True)

    return [_file_result(item) for item in prepared]


def _prepare_change(change: FileChange, target: Path) -> _PreparedChange:
    if not change.path:
        raise ValueError("path is required")
    existed = target.exists()
    if change.require_absent and existed:
        raise FileExistsError(str(target))
    if existed and not target.is_file():
        raise ValueError(f"not a file: {target}")
    before = target.read_text(encoding="utf-8") if existed else ""
    if change.expected_sha256 and _sha256(before) != change.expected_sha256:
        raise RuntimeError(f"file changed before update: {target}")
    after, replacements = _render_content(before, change)
    return _PreparedChange(
        change=change,
        target=target,
        existed=existed,
        before=before,
        after=after,
        replacements=replacements,
    )


def _write_temp(target: Path, content: str, *, suffix: str) -> Path:
    fd, temp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=suffix,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and target.is_file():
            os.chmod(temp_path, stat.S_IMODE(target.stat().st_mode))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _target_unchanged(item: _PreparedChange) -> bool:
    if item.existed:
        return (
            item.target.is_file()
            and item.target.read_text(encoding="utf-8") == item.before
        )
    return not item.target.exists()


def _rollback_changes(committed: list[_PreparedChange]) -> list[str]:
    errors: list[str] = []
    for item in reversed(committed):
        try:
            if item.existed and item.backup is not None:
                os.replace(item.backup, item.target)
                item.backup = None
            else:
                item.target.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"{item.target}: {exc}")
    return errors


def _file_result(item: _PreparedChange) -> dict[str, Any]:
    return {
        "path": str(item.target),
        "before_chars": len(item.before),
        "after_chars": len(item.after),
        "replacements": item.replacements,
        "diff": _diff_preview(item.before, item.after, path=str(item.target)),
        "before_sha256": _sha256(item.before),
        "after_sha256": _sha256(item.after),
    }


def _apply_atomic_change(change: FileChange) -> dict[str, Any]:
    if not change.path:
        raise ValueError("path is required")
    target = Path(change.path)
    existed = target.exists()
    if change.require_absent and existed:
        raise FileExistsError(str(target))
    if existed and not target.is_file():
        raise ValueError(f"not a file: {target}")
    before = target.read_text(encoding="utf-8") if existed else ""
    if change.expected_sha256 and _sha256(before) != change.expected_sha256:
        raise RuntimeError(f"file changed before update: {target}")
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
        "diff": _diff_preview(before, after, path=str(target)),
        "before_sha256": _sha256(before),
        "after_sha256": _sha256(after),
    }


def _render_content(before: str, change: FileChange) -> tuple[str, int]:
    if change.mode == "write":
        return change.content, 0
    if change.mode == "delete":
        if not before and not Path(change.path).is_file():
            raise FileNotFoundError(change.path)
        return "", 0
    if change.mode != "replace":
        raise ValueError(f"unsupported change mode: {change.mode}")
    if not change.old_text:
        raise ValueError("old_text is required")
    replacements = before.count(change.old_text)
    if replacements == 0:
        raise ValueError(
            f"replace text not found in {change.path}; "
            f"{_closest_text_hint(before, change.old_text)}"
        )
    if change.replace_all:
        if change.expected_replacements is not None and (
            replacements != change.expected_replacements
        ):
            raise ValueError(
                f"replace count mismatch in {change.path}: "
                f"found={replacements}, expected={change.expected_replacements}"
            )
        return before.replace(change.old_text, change.new_text), replacements
    if replacements != 1:
        raise ValueError(
            f"replace text is ambiguous in {change.path}: found={replacements}; "
            "provide more surrounding context or set replace_all=true explicitly"
        )
    if change.expected_replacements not in (None, 1):
        raise ValueError(
            f"replace count mismatch in {change.path}: found=1, "
            f"expected={change.expected_replacements}"
        )
    return before.replace(change.old_text, change.new_text, 1), 1


def _closest_text_hint(content: str, expected: str) -> str:
    expected_lines = expected.splitlines() or [expected]
    candidates = content.splitlines()
    match = difflib.get_close_matches(expected_lines[0], candidates, n=1, cutoff=0.45)
    if match:
        return f"closest line: {match[0]!r}"
    if expected.strip() != expected:
        return "hint: check leading/trailing whitespace and line endings"
    return "hint: re-read the file and include more surrounding context"


def _diff_preview(before: str, after: str, *, path: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{path} (before)",
        tofile=f"{path} (after)",
        lineterm="",
        n=2,
    )
    return "\n".join(islice(lines, 80))[:8000]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _result(ok: bool, status: str, **payload: Any) -> ToolResult:
    return ToolResult(
        output={"ok": ok, "status": status, **payload},
        display_content=status,
    )


__all__ = ["FileChange", "apply_changes_transaction"]
