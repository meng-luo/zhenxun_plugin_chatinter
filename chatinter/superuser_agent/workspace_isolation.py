"""Git worktree isolation for superuser Agent engineering tasks.

The main repository may contain user edits or a live bot checkout.  Superuser
Agent code changes should be able to run in an isolated worktree first, then
be reviewed or applied intentionally instead of mutating the primary checkout
by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
import uuid

from ..persistence import read_json, state_path, write_json
from .audit_log import record_audit_event

_WORKTREES_PATH = state_path("worktree_sessions.json")
_WORKTREE_ROOT = state_path("worktrees")
_SESSIONS: dict[str, "WorktreeSession"] = {}
_LOADED = False
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class WorktreeSession:
    worktree_id: str
    user_id: str
    session_key: str
    repo_root: str
    worktree_path: str
    branch_name: str
    base_ref: str
    base_commit: str
    reason: str = ""
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    removed_at: float = 0.0
    error: str = ""

    def public_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = int(self.created_at)
        payload["updated_at"] = int(self.updated_at)
        payload["removed_at"] = int(self.removed_at) if self.removed_at else 0
        payload["current"] = self.status == "active"
        payload["exists"] = Path(self.worktree_path).exists()
        return payload

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def create_worktree_session(
    *,
    actor: dict[str, str],
    repo_root: str | None = None,
    base_ref: str = "HEAD",
    branch_name: str = "",
    reason: str = "",
) -> WorktreeSession:
    """Create a detached engineering worktree for the current session."""

    _ensure_loaded()
    root = _resolve_repo_root(repo_root)
    existing = active_worktree_session(
        user_id=actor["user_id"],
        session_key=actor["session_key"],
    )
    if (
        existing is not None
        and Path(existing.worktree_path).exists()
        and Path(existing.repo_root).resolve() == root
    ):
        return existing
    worktree_id = uuid.uuid4().hex[:10]
    safe_branch = _safe_branch_name(
        branch_name or f"chatinter/{actor['user_id']}/{worktree_id}"
    )
    worktree_path = (_WORKTREE_ROOT / worktree_id).resolve()
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    base_ref = str(base_ref or "HEAD").strip() or "HEAD"
    base_commit = _run_git(["rev-parse", base_ref], cwd=root).strip()
    _run_git(
        ["worktree", "add", "-b", safe_branch, str(worktree_path), base_ref],
        cwd=root,
    )
    session = WorktreeSession(
        worktree_id=worktree_id,
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        repo_root=str(root),
        worktree_path=str(worktree_path),
        branch_name=safe_branch,
        base_ref=base_ref,
        base_commit=base_commit,
        reason=str(reason or ""),
    )
    _SESSIONS[worktree_id] = session
    _save_sessions()
    record_audit_event(
        event="worktree_created",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action="worktree_create",
        payload=session.public_payload(),
        result={"ok": True},
    )
    return session


def get_worktree_session(worktree_id: str) -> WorktreeSession | None:
    _ensure_loaded()
    return _SESSIONS.get(str(worktree_id or "").strip())


def list_worktree_sessions(
    *,
    user_id: str,
    session_key: str,
    include_removed: bool = False,
    limit: int = 20,
) -> list[WorktreeSession]:
    _ensure_loaded()
    rows = [
        session
        for session in _SESSIONS.values()
        if session.user_id == str(user_id or "")
        and session.session_key == str(session_key or "")
        and (include_removed or session.status == "active")
    ]
    rows.sort(key=lambda item: item.updated_at, reverse=True)
    return rows[: max(1, min(int(limit or 20), 100))]


def active_worktree_session(
    *,
    user_id: str,
    session_key: str,
) -> WorktreeSession | None:
    rows = list_worktree_sessions(
        user_id=user_id,
        session_key=session_key,
        include_removed=False,
        limit=1,
    )
    return rows[0] if rows else None


def resolve_working_path(
    path: str | None,
    *,
    actor: dict[str, str] | None = None,
    worktree_id: str = "",
    prefer_worktree: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Resolve a user path against the active worktree when appropriate."""

    raw_path = str(path or "").strip()
    session = _resolve_session(
        actor=actor,
        worktree_id=worktree_id,
        prefer_worktree=prefer_worktree,
    )
    invalid_worktree = bool(worktree_id and session is None and prefer_worktree)
    base = Path(session.worktree_path) if session is not None else Path.cwd()
    if not raw_path:
        resolved = base
    else:
        raw = Path(raw_path)
        resolved = _map_path_for_session(raw, session=session, base=base)
    try:
        resolved_path = resolved.resolve()
    except Exception:
        resolved_path = resolved
    return str(resolved_path), _resolution_payload(
        session,
        requested=raw_path,
        resolved=str(resolved_path),
        invalid_worktree=invalid_worktree,
        escaped_worktree=(
            _escaped_worktree(raw_path, resolved_path, session)
            if session is not None
            else False
        ),
    )


def resolve_cwd(
    cwd: str | None,
    *,
    actor: dict[str, str] | None = None,
    worktree_id: str = "",
    prefer_worktree: bool = True,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve cwd; empty cwd becomes the active worktree path when present."""

    raw_cwd = str(cwd or "").strip()
    session = _resolve_session(
        actor=actor,
        worktree_id=worktree_id,
        prefer_worktree=prefer_worktree,
    )
    invalid_worktree = bool(worktree_id and session is None and prefer_worktree)
    if not raw_cwd and session is None:
        return None, _resolution_payload(
            None,
            requested=raw_cwd,
            resolved="",
            invalid_worktree=invalid_worktree,
        )
    base = Path(session.worktree_path) if session is not None else Path.cwd()
    target = (
        _map_path_for_session(Path(raw_cwd), session=session, base=base)
        if raw_cwd
        else base
    )
    try:
        resolved = target.resolve()
    except Exception:
        resolved = target
    return str(resolved), _resolution_payload(
        session,
        requested=raw_cwd,
        resolved=str(resolved),
        invalid_worktree=invalid_worktree,
        escaped_worktree=_escaped_worktree(raw_cwd, resolved, session)
        if session is not None
        else False,
    )


def worktree_status(
    *,
    actor: dict[str, str],
    worktree_id: str = "",
) -> dict[str, Any]:
    session = _resolve_session(
        actor=actor,
        worktree_id=worktree_id,
        prefer_worktree=True,
    )
    if session is None:
        return {"active": False}
    if not Path(session.worktree_path).exists():
        session.status = "missing"
        session.updated_at = time.time()
        _save_sessions()
        return {"active": False, "missing": True, "session": session.public_payload()}
    status = _run_git(["status", "--short"], cwd=Path(session.worktree_path))
    diff_stat = _run_git(["diff", "--stat"], cwd=Path(session.worktree_path))
    return {
        "active": True,
        "session": session.public_payload(),
        "git_status": status,
        "diff_stat": diff_stat,
    }


def remove_worktree_session(
    *,
    actor: dict[str, str],
    worktree_id: str,
    force: bool = False,
) -> WorktreeSession | None:
    _ensure_loaded()
    session = get_worktree_session(worktree_id)
    if session is None:
        return None
    if (
        session.user_id != actor["user_id"]
        or session.session_key != actor["session_key"]
    ):
        return None
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(session.worktree_path)
    try:
        _run_git(args, cwd=Path(session.repo_root))
    except Exception:
        if force and Path(session.worktree_path).exists():
            shutil.rmtree(session.worktree_path, ignore_errors=True)
    session.status = "removed"
    session.removed_at = time.time()
    session.updated_at = time.time()
    _save_sessions()
    record_audit_event(
        event="worktree_removed",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action="worktree_remove",
        payload={"worktree_id": worktree_id, "force": force},
        result={"ok": True},
    )
    return session


def _resolve_session(
    *,
    actor: dict[str, str] | None,
    worktree_id: str,
    prefer_worktree: bool,
) -> WorktreeSession | None:
    if not prefer_worktree:
        return None
    if worktree_id:
        session = get_worktree_session(worktree_id)
        if session is None or session.status != "active":
            return None
        if not Path(session.worktree_path).exists():
            session.status = "missing"
            session.updated_at = time.time()
            _save_sessions()
            return None
        if actor is not None and (
            session.user_id != actor["user_id"]
            or session.session_key != actor["session_key"]
        ):
            return None
        return session
    if actor is None:
        return None
    session = active_worktree_session(
        user_id=actor["user_id"],
        session_key=actor["session_key"],
    )
    if session is not None and not Path(session.worktree_path).exists():
        session.status = "missing"
        session.updated_at = time.time()
        _save_sessions()
        return None
    return session


def _resolution_payload(
    session: WorktreeSession | None,
    *,
    requested: str,
    resolved: str,
    invalid_worktree: bool = False,
    escaped_worktree: bool = False,
) -> dict[str, Any]:
    if session is None:
        return {
            "isolated": False,
            "requested": requested,
            "resolved": resolved,
            "invalid_worktree": invalid_worktree,
        }
    mapped = _was_main_workspace_path(
        requested=requested,
        session=session,
        resolved=resolved,
    )
    return {
        "isolated": True,
        "requested": requested,
        "resolved": resolved,
        "worktree_id": session.worktree_id,
        "worktree_path": session.worktree_path,
        "branch_name": session.branch_name,
        "base_commit": session.base_commit,
        "mapped_from_main_workspace": mapped,
        "escaped_worktree": escaped_worktree,
    }


def _map_path_for_session(
    raw: Path,
    *,
    session: WorktreeSession | None,
    base: Path,
) -> Path:
    if session is None:
        return raw if raw.is_absolute() else base / raw
    if not raw.is_absolute():
        return base / raw
    try:
        absolute = raw.resolve()
    except Exception:
        absolute = raw
    repo_root = Path(session.repo_root).resolve()
    worktree_root = Path(session.worktree_path).resolve()
    if _path_is_relative_to(absolute, worktree_root):
        return absolute
    if _path_is_relative_to(absolute, repo_root):
        return worktree_root / absolute.relative_to(repo_root)
    return absolute


def _was_main_workspace_path(
    *,
    requested: str,
    session: WorktreeSession,
    resolved: str,
) -> bool:
    if not requested:
        return False
    raw = Path(requested)
    if not raw.is_absolute():
        return False
    try:
        raw_abs = raw.resolve()
        resolved_abs = Path(resolved).resolve()
        repo_root = Path(session.repo_root).resolve()
        worktree_root = Path(session.worktree_path).resolve()
    except Exception:
        return False
    return (
        _path_is_relative_to(raw_abs, repo_root)
        and not _path_is_relative_to(raw_abs, worktree_root)
        and _path_is_relative_to(resolved_abs, worktree_root)
    )


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _escaped_worktree(
    requested: str,
    resolved_path: Path,
    session: WorktreeSession | None,
) -> bool:
    if session is None:
        return False
    raw = Path(str(requested or ""))
    if not requested or raw.is_absolute():
        return False
    try:
        resolved = resolved_path.resolve()
        worktree_root = Path(session.worktree_path).resolve()
    except Exception:
        return False
    return not _path_is_relative_to(resolved, worktree_root)


def _resolve_repo_root(repo_root: str | None) -> Path:
    root = Path(str(repo_root or "").strip() or Path.cwd())
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    top = _run_git(["rev-parse", "--show-toplevel"], cwd=root).strip()
    return Path(top).resolve()


def _run_git(args: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "git failed")
    return proc.stdout.strip()


def _safe_branch_name(value: str) -> str:
    name = _SLUG_RE.sub("-", str(value or "").strip()).strip("-/.")
    if not name:
        name = "chatinter-worktree"
    if not name.startswith("chatinter/"):
        name = "chatinter/" + name
    return name[:96].rstrip("-/.") or f"chatinter/{uuid.uuid4().hex[:8]}"


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    raw = read_json(_WORKTREES_PATH, {})
    if not isinstance(raw, dict):
        return
    for worktree_id, payload in raw.items():
        session = _session_from_payload(worktree_id, payload)
        if session is not None:
            _SESSIONS[session.worktree_id] = session


def _session_from_payload(
    worktree_id: object,
    payload: object,
) -> WorktreeSession | None:
    if not isinstance(payload, dict):
        return None
    try:
        return WorktreeSession(
            worktree_id=str(payload.get("worktree_id") or worktree_id or ""),
            user_id=str(payload.get("user_id", "") or ""),
            session_key=str(payload.get("session_key", "") or ""),
            repo_root=str(payload.get("repo_root", "") or ""),
            worktree_path=str(payload.get("worktree_path", "") or ""),
            branch_name=str(payload.get("branch_name", "") or ""),
            base_ref=str(payload.get("base_ref", "") or "HEAD"),
            base_commit=str(payload.get("base_commit", "") or ""),
            reason=str(payload.get("reason", "") or ""),
            status=str(payload.get("status", "") or "active"),
            created_at=float(payload.get("created_at") or time.time()),
            updated_at=float(payload.get("updated_at") or time.time()),
            removed_at=float(payload.get("removed_at") or 0.0),
            error=str(payload.get("error", "") or ""),
        )
    except Exception:
        return None


def _save_sessions() -> None:
    write_json(
        _WORKTREES_PATH,
        {
            worktree_id: session.to_record()
            for worktree_id, session in sorted(_SESSIONS.items())
        },
    )


__all__ = [
    "WorktreeSession",
    "active_worktree_session",
    "create_worktree_session",
    "get_worktree_session",
    "list_worktree_sessions",
    "remove_worktree_session",
    "resolve_cwd",
    "resolve_working_path",
    "worktree_status",
]
