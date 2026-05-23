"""Audit log for superuser agent operations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..runtime_events import emit_runtime_event

_AUDIT_LOG_PATH = Path("data/log/chatinter_agent_audit.log")
_MAX_QUERY_LINES = 2000


def record_audit_event(
    *,
    event: str,
    user_id: str,
    session_key: str,
    action: str,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "user_id": str(user_id or ""),
        "session_key": str(session_key or ""),
        "action": str(action or ""),
        "payload": payload or {},
        "result": result or {},
    }
    try:
        _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Audit must never break the bot turn. Tool results still contain the
        # user-visible status if logging fails.
        pass
    try:
        emit_runtime_event(
            kind="audit",
            status="info",
            source=f"audit:{event}",
            session_key=entry["session_key"],
            user_id=entry["user_id"],
            summary=f"{event}:{action}",
            payload={
                "event": event,
                "action": action,
                "payload": payload or {},
                "result": result or {},
            },
        )
    except Exception:
        pass


def query_audit_events(
    *,
    limit: int = 50,
    user_id: str = "",
    session_key: str = "",
    action: str = "",
    event: str = "",
    contains: str = "",
) -> list[dict[str, Any]]:
    if not _AUDIT_LOG_PATH.exists():
        return []
    limit = max(1, min(int(limit or 50), 200))
    filters = {
        "user_id": str(user_id or ""),
        "session_key": str(session_key or ""),
        "action": str(action or ""),
        "event": str(event or ""),
    }
    contains_text = str(contains or "")
    entries: list[dict[str, Any]] = []
    try:
        lines = _AUDIT_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    for line in reversed(lines[-_MAX_QUERY_LINES:]):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if not isinstance(entry, dict):
            continue
        if any(filters[key] and str(entry.get(key, "")) != filters[key] for key in filters):
            continue
        if contains_text and contains_text not in json.dumps(entry, ensure_ascii=False):
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def audit_log_path() -> Path:
    return _AUDIT_LOG_PATH


__all__ = ["audit_log_path", "query_audit_events", "record_audit_event"]
