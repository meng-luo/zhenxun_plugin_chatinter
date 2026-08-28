"""Audit log for superuser agent operations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any

from zhenxun.utils.log_sanitizer import sanitize_for_logging

_AUDIT_LOG_PATH = Path("data/log/chatinter_agent_audit.log")
_MAX_QUERY_LINES = 2000
_MAX_LOG_BYTES = 2_000_000
_MAX_TEXT_CHARS = 1000
_AUDIT_LOCK = threading.Lock()
_REDACTED_KEYS = {
    "api_key",
    "authorization",
    "content",
    "env",
    "environment",
    "new_text",
    "old_text",
    "password",
    "secret",
    "stderr",
    "stdout",
    "token",
}


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
        "payload": _compact_value(payload or {}),
        "result": _compact_value(result or {}),
    }
    sanitized = sanitize_for_logging(entry)
    if isinstance(sanitized, dict):
        entry = sanitized
    try:
        with _AUDIT_LOCK:
            _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_full()
            with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
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
        lines = _AUDIT_LOG_PATH.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except Exception:
        return []
    for line in reversed(lines[-_MAX_QUERY_LINES:]):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if not isinstance(entry, dict):
            continue
        if any(
            filters[key] and str(entry.get(key, "")) != filters[key] for key in filters
        ):
            continue
        if contains_text and contains_text not in json.dumps(entry, ensure_ascii=False):
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def audit_log_path() -> Path:
    return _AUDIT_LOG_PATH


def _rotate_if_full() -> None:
    try:
        if (
            not _AUDIT_LOG_PATH.exists()
            or _AUDIT_LOG_PATH.stat().st_size < _MAX_LOG_BYTES
        ):
            return
        backup = _AUDIT_LOG_PATH.with_name(_AUDIT_LOG_PATH.name + ".1")
        os.replace(_AUDIT_LOG_PATH, backup)
    except OSError:
        return


def _compact_value(value: Any, *, key: str = "") -> Any:
    if key.lower() in _REDACTED_KEYS:
        return f"[redacted:{len(str(value or ''))}]"
    if isinstance(value, dict):
        return {
            str(item_key): _compact_value(item, key=str(item_key))
            for item_key, item in list(value.items())[:40]
        }
    if isinstance(value, list | tuple):
        return [_compact_value(item) for item in value[:40]]
    if isinstance(value, str) and len(value) > _MAX_TEXT_CHARS:
        return value[:_MAX_TEXT_CHARS] + "...[truncated]"
    return value


__all__ = ["audit_log_path", "query_audit_events", "record_audit_event"]
