"""Parse V4A patches into validated Superuser file changes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from .patch_operations import FileChange

_MAX_PATCH_CHARS = 200_000
_MAX_PATCH_FILES = 20


@dataclass
class PatchHunk:
    hint: str = ""
    lines: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PatchOperation:
    kind: str
    path: str
    destination: str = ""
    hunks: list[PatchHunk] = field(default_factory=list)


def patch_file_changes(
    patch: str,
    *,
    cwd: str | None = None,
) -> tuple[list[FileChange], str]:
    operations, error = parse_v4a_patch(patch)
    if error:
        return [], error
    root = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    changes: list[FileChange] = []
    try:
        for operation in operations:
            source = _resolve_path(root, operation.path)
            if operation.kind == "add":
                content = _added_content(operation)
                changes.append(
                    FileChange(
                        path=str(source),
                        mode="write",
                        content=content,
                        create_dirs=True,
                        require_absent=True,
                    )
                )
                continue

            if not source.is_file():
                raise FileNotFoundError(str(source))
            before = source.read_text(encoding="utf-8")
            expected_sha256 = _sha256(before)
            if operation.kind == "delete":
                changes.append(
                    FileChange(
                        path=str(source),
                        mode="delete",
                        expected_sha256=expected_sha256,
                    )
                )
                continue

            after = _apply_hunks(before, operation)
            if operation.kind == "move" or operation.destination:
                destination = _resolve_path(root, operation.destination)
                if destination.exists():
                    raise FileExistsError(str(destination))
                changes.extend(
                    [
                        FileChange(
                            path=str(destination),
                            mode="write",
                            content=after,
                            create_dirs=True,
                            require_absent=True,
                        ),
                        FileChange(
                            path=str(source),
                            mode="delete",
                            expected_sha256=expected_sha256,
                        ),
                    ]
                )
            else:
                changes.append(
                    FileChange(
                        path=str(source),
                        mode="write",
                        content=after,
                        expected_sha256=expected_sha256,
                    )
                )
        if len(changes) > _MAX_PATCH_FILES:
            raise ValueError(f"patch exceeds {_MAX_PATCH_FILES} file changes")
    except (OSError, ValueError) as exc:
        return [], str(exc)
    return changes, ""


def parse_v4a_patch(patch: str) -> tuple[list[PatchOperation], str]:
    text = str(patch or "")
    if not text.strip():
        return [], "patch is required"
    if len(text) > _MAX_PATCH_CHARS:
        return [], f"patch exceeds {_MAX_PATCH_CHARS} characters"
    lines = text.splitlines()
    try:
        start = lines.index("*** Begin Patch")
        end = lines.index("*** End Patch", start + 1)
    except ValueError:
        return [], "patch must contain *** Begin Patch and *** End Patch"

    operations: list[PatchOperation] = []
    current: PatchOperation | None = None
    current_hunk: PatchHunk | None = None

    def finish() -> str:
        nonlocal current, current_hunk
        if current is None:
            return ""
        if current_hunk is not None and current_hunk.lines:
            current.hunks.append(current_hunk)
        if current.kind == "update" and not current.hunks:
            return f"update has no hunks: {current.path}"
        operations.append(current)
        current = None
        current_hunk = None
        return ""

    for line in lines[start + 1 : end]:
        header = _operation_header(line)
        if header is not None:
            error = finish()
            if error:
                return [], error
            current = header
            continue
        if line.startswith("*** Move to: ") and current is not None:
            current.destination = line.removeprefix("*** Move to: ").strip()
            continue
        if line.startswith("@@") and current is not None:
            if current_hunk is not None and current_hunk.lines:
                current.hunks.append(current_hunk)
            hint = line[2:].removesuffix("@@").strip()
            current_hunk = PatchHunk(hint=hint)
            continue
        if current is None or not line:
            continue
        if current.kind == "add":
            if not line.startswith("+"):
                return [], f"add lines must start with +: {current.path}"
            current_hunk = current_hunk or PatchHunk()
            current_hunk.lines.append(("+", line[1:]))
            continue
        if current.kind in {"update", "move"}:
            if line[:1] not in {" ", "+", "-"}:
                return [], f"invalid hunk line in {current.path}: {line}"
            current_hunk = current_hunk or PatchHunk()
            current_hunk.lines.append((line[0], line[1:]))
            continue
        return [], f"unexpected patch content for {current.path}"

    error = finish()
    if error:
        return [], error
    if not operations:
        return [], "patch contains no file operations"
    if len(operations) > _MAX_PATCH_FILES:
        return [], f"patch exceeds {_MAX_PATCH_FILES} file operations"
    return operations, ""


def _operation_header(line: str) -> PatchOperation | None:
    headers = {
        "*** Add File: ": "add",
        "*** Update File: ": "update",
        "*** Delete File: ": "delete",
    }
    for prefix, kind in headers.items():
        if line.startswith(prefix):
            return PatchOperation(kind=kind, path=line.removeprefix(prefix).strip())
    if line.startswith("*** Move File: "):
        value = line.removeprefix("*** Move File: ")
        source, separator, destination = value.partition(" -> ")
        if separator:
            return PatchOperation(
                kind="move",
                path=source.strip(),
                destination=destination.strip(),
            )
    return None


def _added_content(operation: PatchOperation) -> str:
    lines = [content for hunk in operation.hunks for prefix, content in hunk.lines]
    return "\n".join(lines) + ("\n" if lines else "")


def _apply_hunks(before: str, operation: PatchOperation) -> str:
    content = before
    for index, hunk in enumerate(operation.hunks, start=1):
        old = "\n".join(
            line for prefix, line in hunk.lines if prefix in {" ", "-"}
        )
        new = "\n".join(
            line for prefix, line in hunk.lines if prefix in {" ", "+"}
        )
        if old:
            matches = content.count(old)
            if matches != 1:
                raise ValueError(
                    f"hunk {index} in {operation.path} matched {matches} times"
                )
            content = content.replace(old, new, 1)
            continue
        if not hunk.hint:
            raise ValueError(
                f"addition-only hunk {index} in {operation.path} requires a hint"
            )
        matches = content.count(hunk.hint)
        if matches != 1:
            raise ValueError(
                f"hunk hint in {operation.path} matched {matches} times"
            )
        content = content.replace(hunk.hint, f"{hunk.hint}\n{new}", 1)
    return content


def _resolve_path(root: Path, value: str) -> Path:
    if not str(value or "").strip():
        raise ValueError("patch file path is required")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["PatchOperation", "parse_v4a_patch", "patch_file_changes"]
