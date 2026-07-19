"""Durable snapshots for the Superuser Agent conversation."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
import time
from typing import Any
import uuid

from ..artifact_store import get_artifact_store
from ..llm_compat import (
    LLMMessage,
    LLMToolCall,
    LLMToolFunction,
    ToolExecutable,
)
from ..persistence import (
    read_json,
    state_path,
    to_jsonable,
    utc_now_iso,
    write_json,
)
from .permission_policy import get_default_permission_mode, resolve_permission_mode
from .state import (
    AgentBudgetState,
    AgentRunState,
    ToolExecutionRecord,
    append_artifact_refs,
)

_CANCEL_SIGNALS: set[str] = set()
_CANCEL_SIGNALS_MAX = 512
_ACTIVITY_RING_MAX_RUNS = 128
_ACTIVITY_RING_MAX_ITEMS = 40
_ACTIVITY_RING: dict[str, deque[dict[str, Any]]] = {}
_IMPORTANT_PERSIST_STAGES = frozenset(
    {
        "started",
        "tool_observation",
        "tool_execution_started",
        "tool_execution_completed",
        "tool_execution_reconciled",
        "tool_protocol_repaired",
        "semantic_compression_failed",
        "semantic_context_compressed",
        "paused",
        "cancelled",
        "completed",
        "failed",
    }
)
_PERSIST_THROTTLE_SECONDS = 15.0
_LAST_PERSIST_AT: dict[str, float] = {}
_AGENT_SESSIONS_PATH = state_path("agent_sessions.json")
_AGENT_SESSIONS_LOCK = threading.RLock()
_RETENTION_INTERVAL_SECONDS = 6 * 60 * 60
_SNAPSHOT_RETENTION_SECONDS = 30 * 24 * 60 * 60
_TRACE_ALIAS_RETENTION_SECONDS = 24 * 60 * 60
_LAST_RETENTION_AT = 0.0
_TRACE_ALIAS_KIND = "agent_run_trace_alias"


def get_agent_session(session_key: str) -> dict[str, Any]:
    key = str(session_key or "").strip()
    if not key:
        return {}
    with _AGENT_SESSIONS_LOCK:
        sessions = _read_agent_sessions()
        value = sessions.get(key)
        if not isinstance(value, dict):
            return {}
        session, changed = _normalize_agent_session(value)
        if changed:
            sessions[key] = session
            write_json(_AGENT_SESSIONS_PATH, sessions)
        return _agent_session_view(session)


def agent_session_is_active(session_key: str) -> bool:
    return bool(get_agent_session(session_key).get("agent_mode_active"))


def get_active_agent_run_id(session_key: str) -> str:
    conversation = get_active_conversation(session_key)
    return str(conversation.get("run_id", "") or "") if conversation else ""


def create_conversation(
    session_key: str,
    *,
    name: str = "",
    run_id: str = "",
    permission_mode: str = "",
) -> dict[str, Any] | None:
    key = str(session_key or "").strip()
    if not key:
        return None
    with _AGENT_SESSIONS_LOCK:
        sessions, session = _agent_session_state(key)
        conversation_id = str(session["next_conversation_id"])
        session["next_conversation_id"] = int(conversation_id) + 1
        now = utc_now_iso()
        conversation = {
            "name": str(name or "").strip() or f"会话 {conversation_id}",
            "run_id": str(run_id or _new_run_id()),
            "permission_mode": resolve_permission_mode(permission_mode)
            if permission_mode
            else get_default_permission_mode(),
            "created_at": now,
            "last_used_at": now,
            "archived": False,
        }
        session["conversations"][conversation_id] = conversation
        session["active_conversation_id"] = conversation_id
        session["agent_mode_active"] = True
        _save_agent_session(key, sessions, session)
        return _conversation_view(conversation_id, conversation)


def list_conversations(
    session_key: str,
    *,
    archived: bool | None = None,
) -> list[dict[str, Any]]:
    session = get_agent_session(session_key)
    conversations = session.get("conversations")
    if not isinstance(conversations, dict):
        return []
    rows = [
        _conversation_view(conversation_id, conversation)
        for conversation_id, conversation in conversations.items()
        if isinstance(conversation, dict)
        and (archived is None or bool(conversation.get("archived")) is archived)
    ]
    rows.sort(key=lambda item: _conversation_sort_key(str(item["id"])))
    return rows


def get_active_conversation(session_key: str) -> dict[str, Any] | None:
    session = get_agent_session(session_key)
    conversation_id = str(session.get("active_conversation_id", "") or "")
    conversations = session.get("conversations")
    if not conversation_id or not isinstance(conversations, dict):
        return None
    conversation = conversations.get(conversation_id)
    if not isinstance(conversation, dict) or conversation.get("archived"):
        return None
    return _conversation_view(conversation_id, conversation)


def switch_conversation(
    session_key: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    return _update_conversation(
        session_key,
        conversation_id,
        operation="switch",
    )


def rename_conversation(
    session_key: str,
    conversation_id: str,
    name: str,
) -> dict[str, Any] | None:
    normalized_name = str(name or "").strip()
    if not normalized_name:
        return None
    return _update_conversation(
        session_key,
        conversation_id,
        operation="rename",
        name=normalized_name,
    )


def archive_conversation(
    session_key: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    return _update_conversation(
        session_key,
        conversation_id,
        operation="archive",
    )


def restore_conversation(
    session_key: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    return _update_conversation(
        session_key,
        conversation_id,
        operation="restore",
    )


def set_conversation_permission_mode(
    session_key: str,
    conversation_id: str,
    mode: str,
) -> dict[str, Any] | None:
    return _update_conversation(
        session_key,
        conversation_id,
        operation="permission_mode",
        permission_mode=resolve_permission_mode(mode),
    )


def delete_conversation(
    session_key: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    key = str(session_key or "").strip()
    target = str(conversation_id or "").strip()
    if not key or not target:
        return None
    with _AGENT_SESSIONS_LOCK:
        sessions, session = _agent_session_state(key)
        conversation = session["conversations"].pop(target, None)
        if not isinstance(conversation, dict):
            return None
        if session["active_conversation_id"] == target:
            session["active_conversation_id"] = ""
            session["agent_mode_active"] = False
        _save_agent_session(key, sessions, session)
        _delete_agent_run_snapshot(str(conversation.get("run_id", "") or ""))
        return _conversation_view(target, conversation)


def activate_agent_session(session_key: str, *, run_id: str = "") -> None:
    key = str(session_key or "").strip()
    if not key:
        return
    with _AGENT_SESSIONS_LOCK:
        sessions, session = _agent_session_state(key)
        conversation_id = str(session["active_conversation_id"] or "")
        conversation = session["conversations"].get(conversation_id)
        if not isinstance(conversation, dict) or conversation.get("archived"):
            create_conversation(key, run_id=run_id)
            return
        if run_id:
            conversation["run_id"] = str(run_id)
        conversation["last_used_at"] = utc_now_iso()
        session["agent_mode_active"] = True
        _save_agent_session(key, sessions, session)


def deactivate_agent_session(session_key: str) -> str:
    """Close Agent mode while retaining the selected conversation."""

    key = str(session_key or "").strip()
    if not key:
        return ""
    with _AGENT_SESSIONS_LOCK:
        sessions, session = _agent_session_state(key)
        conversation_id = str(session["active_conversation_id"] or "")
        conversation = session["conversations"].get(conversation_id)
        run_id = (
            str(conversation.get("run_id", "") or "")
            if isinstance(conversation, dict)
            else ""
        )
        session["agent_mode_active"] = False
        _save_agent_session(key, sessions, session)
        return run_id


def clear_agent_session_context(session_key: str) -> str:
    key = str(session_key or "").strip()
    if not key:
        return ""
    with _AGENT_SESSIONS_LOCK:
        sessions, session = _agent_session_state(key)
        conversation_id = str(session["active_conversation_id"] or "")
        conversation = session["conversations"].get(conversation_id)
        if not isinstance(conversation, dict) or conversation.get("archived"):
            create_conversation(key)
            return ""
        previous = str(conversation.get("run_id", "") or "")
        conversation["run_id"] = _new_run_id()
        conversation["last_used_at"] = utc_now_iso()
        session["agent_mode_active"] = True
        _save_agent_session(key, sessions, session)
        _delete_agent_run_snapshot(previous)
        return previous


def archive_agent_session(session_key: str) -> str:
    conversation = get_active_conversation(session_key)
    if conversation is None:
        deactivate_agent_session(session_key)
        return ""
    archived = archive_conversation(session_key, str(conversation["id"]))
    return str(archived.get("run_id", "") or "") if archived else ""


def _agent_session_state(
    session_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sessions = _read_agent_sessions()
    current = sessions.get(session_key)
    if not isinstance(current, dict):
        return sessions, _empty_agent_session()
    session, changed = _normalize_agent_session(current)
    if changed:
        sessions[session_key] = session
    return sessions, session


def _delete_agent_run_snapshot(run_id: str) -> None:
    safe_run_id = _safe_trace_id(run_id)
    if not safe_run_id:
        return
    run_dir = _run_snapshot_path(safe_run_id).parent
    if not run_dir.exists():
        return
    for path in run_dir.glob("*.json"):
        try:
            payload = read_json(path, None)
            payload_run_id = (
                _safe_trace_id(str(payload.get("run_id", "") or ""))
                if isinstance(payload, dict)
                else ""
            )
            if (
                _safe_trace_id(path.stem) == safe_run_id
                or payload_run_id == safe_run_id
            ):
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _save_agent_session(
    key: str,
    sessions: dict[str, Any],
    session: dict[str, Any],
) -> None:
    session["updated_at"] = utc_now_iso()
    sessions[key] = session
    write_json(_AGENT_SESSIONS_PATH, sessions)


def _update_conversation(
    session_key: str,
    conversation_id: str,
    *,
    operation: str,
    name: str = "",
    permission_mode: str = "",
) -> dict[str, Any] | None:
    key = str(session_key or "").strip()
    target = str(conversation_id or "").strip()
    if not key or not target:
        return None
    with _AGENT_SESSIONS_LOCK:
        sessions, session = _agent_session_state(key)
        conversation = session["conversations"].get(target)
        if not isinstance(conversation, dict):
            return None
        if operation == "switch":
            if conversation.get("archived"):
                return None
            session["active_conversation_id"] = target
            session["agent_mode_active"] = True
            conversation["last_used_at"] = utc_now_iso()
        elif operation == "rename":
            conversation["name"] = name
        elif operation == "archive":
            conversation["archived"] = True
            if session["active_conversation_id"] == target:
                session["active_conversation_id"] = ""
                session["agent_mode_active"] = False
        elif operation == "restore":
            conversation["archived"] = False
        elif operation == "permission_mode":
            conversation["permission_mode"] = permission_mode
        else:
            return None
        _save_agent_session(key, sessions, session)
        return _conversation_view(target, conversation)


def _read_agent_sessions() -> dict[str, Any]:
    sessions = read_json(_AGENT_SESSIONS_PATH, {})
    return sessions if isinstance(sessions, dict) else {}


def _empty_agent_session() -> dict[str, Any]:
    return {
        "agent_mode_active": False,
        "active_conversation_id": "",
        "next_conversation_id": 1,
        "conversations": {},
        "updated_at": utc_now_iso(),
    }


def _normalize_agent_session(value: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    now = utc_now_iso()
    default_permission_mode = get_default_permission_mode()
    raw_conversations = value.get("conversations")
    conversations: dict[str, dict[str, Any]] = {}
    if isinstance(raw_conversations, dict):
        for raw_id, raw_conversation in raw_conversations.items():
            conversation_id = str(raw_id or "").strip()
            if not conversation_id or not isinstance(raw_conversation, dict):
                continue
            created_at = str(raw_conversation.get("created_at", "") or now)
            conversations[conversation_id] = {
                "name": str(raw_conversation.get("name", "") or "").strip()
                or f"会话 {conversation_id}",
                "run_id": str(raw_conversation.get("run_id", "") or ""),
                "permission_mode": resolve_permission_mode(
                    str(
                        raw_conversation.get("permission_mode", "")
                        or default_permission_mode
                    )
                ),
                "created_at": created_at,
                "last_used_at": str(
                    raw_conversation.get("last_used_at", "") or created_at
                ),
                "archived": bool(raw_conversation.get("archived", False)),
            }
    elif "active" in value or "run_id" in value:
        updated_at = str(value.get("updated_at", "") or now)
        conversations["1"] = {
            "name": "会话 1",
            "run_id": str(value.get("run_id", "") or _new_run_id()),
            "permission_mode": default_permission_mode,
            "created_at": updated_at,
            "last_used_at": updated_at,
            "archived": False,
        }

    active_id = str(
        value.get(
            "active_conversation_id",
            "1" if conversations and not isinstance(raw_conversations, dict) else "",
        )
        or ""
    )
    active = conversations.get(active_id)
    if not isinstance(active, dict) or active.get("archived"):
        active_id = ""
    numeric_ids = [int(item) for item in conversations if item.isdigit()]
    next_id = max(
        _positive_int(value.get("next_conversation_id"), default=1),
        max(numeric_ids, default=0) + 1,
        1,
    )
    session = {
        "agent_mode_active": bool(
            value.get("agent_mode_active", value.get("active", False))
        )
        and bool(active_id),
        "active_conversation_id": active_id,
        "next_conversation_id": next_id,
        "conversations": conversations,
        "updated_at": str(value.get("updated_at", "") or now),
    }
    return session, session != value


def _agent_session_view(session: dict[str, Any]) -> dict[str, Any]:
    view = dict(session)
    view["conversations"] = {
        str(key): dict(value)
        for key, value in session.get("conversations", {}).items()
        if isinstance(value, dict)
    }
    conversation_id = str(session.get("active_conversation_id", "") or "")
    conversation = view["conversations"].get(conversation_id, {})

    view["active"] = bool(session.get("agent_mode_active"))
    view["run_id"] = str(conversation.get("run_id", "") or "")
    return view


def _conversation_view(
    conversation_id: str,
    conversation: dict[str, Any],
) -> dict[str, Any]:
    return {"id": str(conversation_id), **dict(conversation)}


def _conversation_sort_key(conversation_id: str) -> tuple[int, int | str]:
    if conversation_id.isdigit():
        return 0, int(conversation_id)
    return 1, conversation_id


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _positive_int(value: Any, *, default: int) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError, OverflowError):
        return max(int(default), 1)


def cleanup_agent_run_storage(*, now: float | None = None) -> dict[str, int]:
    """Remove expired snapshots and trace aliases."""

    now_ts = float(now if now is not None else time.time())
    run_dir = _run_snapshot_path("retention").parent
    protected_run_ids = _retained_session_run_ids()
    records: list[tuple[Any, dict[str, Any], float, str, str]] = []
    for path in run_dir.glob("*.json") if run_dir.exists() else ():
        payload = read_json(path, None)
        if not isinstance(payload, dict):
            continue
        stem = _safe_trace_id(path.stem)
        run_id = _safe_trace_id(
            str(payload.get("run_id", "") or payload.get("trace_id", ""))
        )
        trace_id = _safe_trace_id(str(payload.get("trace_id", "") or ""))
        updated_at = _payload_timestamp(payload, fallback=path.stat().st_mtime)
        records.append((path, payload, updated_at, run_id, trace_id))
        canonical_exists = bool(run_id) and _run_snapshot_path(run_id).exists()
        if str(payload.get("status", "") or "") in {"running", "paused"} and (
            stem == run_id or not canonical_exists
        ):
            protected_run_ids.add(run_id or stem)

    protected_snapshot_ids = set(protected_run_ids)
    for run_id in tuple(protected_run_ids):
        payload = get_agent_run_snapshot(run_id)
        if isinstance(payload, dict):
            trace_id = _safe_trace_id(str(payload.get("trace_id", "") or ""))
            if trace_id:
                protected_snapshot_ids.add(trace_id)

    snapshots_deleted = 0
    for path, payload, updated_at, run_id, _trace_id in records:
        stem = _safe_trace_id(path.stem)
        if run_id in protected_run_ids or stem in protected_snapshot_ids:
            continue
        canonical_exists = bool(run_id) and _run_snapshot_path(run_id).exists()
        is_trace_alias = bool(run_id) and stem != run_id and canonical_exists
        retention = (
            _TRACE_ALIAS_RETENTION_SECONDS
            if is_trace_alias
            else _SNAPSHOT_RETENTION_SECONDS
        )
        if now_ts - updated_at <= retention:
            continue
        if not is_trace_alias and str(payload.get("status", "") or "") in {
            "running",
            "paused",
        }:
            continue
        try:
            path.unlink(missing_ok=True)
            snapshots_deleted += 1
        except OSError:
            continue

    artifact_stats = get_artifact_store().cleanup_expired(now=now_ts)
    return {"snapshots_deleted": snapshots_deleted, **artifact_stats}


def _maybe_cleanup_agent_run_storage() -> None:
    global _LAST_RETENTION_AT
    now_monotonic = time.monotonic()
    if (
        _LAST_RETENTION_AT
        and now_monotonic - _LAST_RETENTION_AT < _RETENTION_INTERVAL_SECONDS
    ):
        return
    _LAST_RETENTION_AT = now_monotonic
    try:
        cleanup_agent_run_storage()
    except Exception:
        return


def _retained_session_run_ids() -> set[str]:
    sessions = read_json(_AGENT_SESSIONS_PATH, {})
    if not isinstance(sessions, dict):
        return set()
    retained: set[str] = set()
    for value in sessions.values():
        if not isinstance(value, dict):
            continue
        session, _ = _normalize_agent_session(value)
        conversations = session.get("conversations", {})
        if not isinstance(conversations, dict):
            continue
        for conversation in conversations.values():
            if not isinstance(conversation, dict):
                continue
            run_id = _safe_trace_id(str(conversation.get("run_id", "") or ""))
            if run_id:
                retained.add(run_id)
    retained.discard("")
    return retained


def _retained_session_artifact_ids() -> set[str]:
    retained: set[str] = set()
    for run_id in _retained_session_run_ids():
        snapshot = get_agent_run_snapshot(run_id)
        if not isinstance(snapshot, dict):
            continue
        retained.update(_text_list(snapshot.get("artifact_refs")))
    retained.discard("")
    return retained


def _payload_timestamp(payload: dict[str, Any], *, fallback: float) -> float:
    value = str(payload.get("updated_at") or payload.get("ts") or "").strip()
    if not value:
        return float(fallback)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return float(fallback)


def signal_agent_run_cancel(run_id: str) -> None:
    safe = _safe_trace_id(str(run_id or ""))
    if not safe:
        return
    if len(_CANCEL_SIGNALS) >= _CANCEL_SIGNALS_MAX:
        _CANCEL_SIGNALS.clear()
    _CANCEL_SIGNALS.add(safe)


def is_agent_run_cancel_signaled(run_id: str) -> bool:
    safe = _safe_trace_id(str(run_id or ""))
    return bool(safe) and safe in _CANCEL_SIGNALS


def clear_agent_run_cancel_signal(run_id: str) -> None:
    safe = _safe_trace_id(str(run_id or ""))
    _CANCEL_SIGNALS.discard(safe)


def list_agent_run_activities(run_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
    safe = _safe_trace_id(str(run_id or ""))
    if not safe:
        return []
    rows = list(_ACTIVITY_RING.get(safe, ()))
    return rows[-max(1, min(int(limit or 10), _ACTIVITY_RING_MAX_ITEMS)) :]


def persist_agent_run_state(
    state: Any,
    *,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    try:
        activity = _activity_state_payload(
            state,
            stage=stage,
            metadata=metadata or {},
        )
        _remember_activity(activity)
        if not _should_persist_snapshot(activity):
            return True
        payload = _state_payload(state, stage=stage, metadata=metadata or {})
        _write_canonical_snapshot(payload)
        _mark_snapshot_persisted(payload)
    except Exception:
        return False
    _maybe_cleanup_agent_run_storage()
    return True


def get_agent_run_snapshot(trace_id: str) -> dict[str, Any] | None:
    safe_trace = _safe_trace_id(trace_id)
    if not safe_trace:
        return None
    payload = read_json(_run_snapshot_path(safe_trace), None)
    if _is_trace_alias(payload):
        run_id = _safe_trace_id(str(payload.get("run_id", "") or ""))
        if not run_id or run_id == safe_trace:
            return None
        payload = read_json(_run_snapshot_path(run_id), None)
        if _is_trace_alias(payload):
            return None
    return payload if isinstance(payload, dict) else None


def get_agent_run_messages(run_id: str) -> list[LLMMessage]:
    snapshot = get_agent_run_snapshot(run_id)
    return _messages_from_payload(snapshot.get("messages", [])) if snapshot else []


def persist_agent_run_messages(
    run_id: str,
    *,
    messages: list[LLMMessage],
    current_context_tokens: int,
    stage: str,
    artifact_ids: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Replace conversation messages without rewriting the rest of the run."""

    snapshot = get_agent_run_snapshot(run_id)
    if not isinstance(snapshot, dict):
        return False
    snapshot["updated_at"] = utc_now_iso()
    snapshot["stage"] = str(stage or "context_updated")
    snapshot["messages"] = to_jsonable(messages)
    snapshot["compression_failure_fingerprint"] = ""
    snapshot["compression_failure_count"] = 0
    artifact_refs = _text_list(snapshot.get("artifact_refs"))
    append_artifact_refs(artifact_refs, artifact_ids)
    snapshot["artifact_refs"] = artifact_refs
    budget = dict(snapshot.get("budget") or {})
    budget["current_context_tokens"] = max(int(current_context_tokens or 0), 0)
    budget["last_usage_message_count"] = len(messages)
    budget["last_usage_schema_tokens"] = 0
    snapshot["budget"] = budget
    del metadata
    _write_canonical_snapshot(snapshot, fallback_run_id=run_id)
    _maybe_cleanup_agent_run_storage()
    return True


def load_agent_run_state(
    run_id: str,
    *,
    tool_map: dict[str, ToolExecutable],
) -> AgentRunState | None:
    snapshot = get_agent_run_snapshot(run_id)
    if not isinstance(snapshot, dict):
        return None
    state = _state_from_snapshot(snapshot, tool_map=tool_map)
    if state is None:
        return None
    return state


def update_agent_run_status(
    run_id: str,
    *,
    status: str,
    reason: str = "",
    metadata: dict[str, Any] | None = None,
    clear_pending_approval: bool = False,
) -> dict[str, Any] | None:
    if str(status or "") == "cancelled":
        signal_agent_run_cancel(run_id)
    snapshot = get_agent_run_snapshot(run_id)
    if not isinstance(snapshot, dict):
        return None
    snapshot["updated_at"] = utc_now_iso()
    snapshot["status"] = str(status or "")
    snapshot["paused_reason"] = "" if status != "paused" else str(reason or "")
    snapshot["stop_reason"] = str(reason or status or "")
    if clear_pending_approval:
        snapshot["pending_approval"] = ""
        snapshot.pop("waiting_approval_ids", None)
    del metadata
    _write_canonical_snapshot(snapshot, fallback_run_id=run_id)
    _maybe_cleanup_agent_run_storage()
    return snapshot


def list_agent_run_snapshots(
    *,
    session_key: str = "",
    limit: int = 20,
) -> list[dict[str, Any]]:
    run_dir = state_path("agent_runs")
    if not run_dir.exists():
        return []
    normalized_session = str(session_key or "")
    rows: list[dict[str, Any]] = []
    for path in run_dir.glob("*.json"):
        payload = read_json(path, None)
        if not isinstance(payload, dict):
            continue
        if _is_trace_alias(payload):
            continue
        if (
            normalized_session
            and str(payload.get("session_key", "")) != normalized_session
        ):
            continue
        rows.append(_compact_run_snapshot(payload))
    rows.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return rows[: max(1, min(int(limit or 20), 100))]


def _state_payload(
    state: Any,
    *,
    stage: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    del metadata
    return {
        "version": 2,
        "updated_at": utc_now_iso(),
        "stage": str(stage or ""),
        "run_id": str(
            getattr(state, "run_id", "") or getattr(state, "trace_id", "") or ""
        ),
        "trace_id": str(getattr(state, "trace_id", "") or ""),
        "session_key": str(getattr(state, "session_key", "") or ""),
        "status": str(getattr(state, "status", "") or ""),
        "paused_reason": str(getattr(state, "paused_reason", "") or ""),
        "pending_approval": str(getattr(state, "pending_approval", "") or ""),
        "artifact_refs": to_jsonable(getattr(state, "artifact_refs", [])),
        "step": int(getattr(state, "step", 0) or 0),
        "max_steps": int(getattr(state, "max_steps", 0) or 0),
        "cost_checkpoint_tokens": int(getattr(state, "cost_checkpoint_tokens", 0) or 0),
        "compression_failure_fingerprint": str(
            getattr(state, "compression_failure_fingerprint", "") or ""
        ),
        "compression_failure_count": max(
            int(getattr(state, "compression_failure_count", 0) or 0),
            0,
        ),
        "stop_reason": str(getattr(state, "stop_reason", "") or ""),
        "final_text": str(getattr(state, "final_text", "") or ""),
        "delivery_complete": bool(getattr(state, "delivery_complete", False)),
        "final_source": str(getattr(state, "final_source", "") or ""),
        "tool_executions": to_jsonable(getattr(state, "tool_executions", [])),
        "budget": to_jsonable(getattr(state, "budget", None)),
        "messages": to_jsonable(getattr(state, "messages", [])),
    }


def _write_canonical_snapshot(
    payload: dict[str, Any],
    *,
    fallback_run_id: str = "",
) -> None:
    run_id = _safe_trace_id(
        str(payload.get("run_id", "") or fallback_run_id or payload.get("trace_id", ""))
    )
    if not run_id:
        raise ValueError("snapshot run_id is required")
    write_json(_run_snapshot_path(run_id), payload, compact=True)

    trace_id = _safe_trace_id(str(payload.get("trace_id", "") or ""))
    if not trace_id or trace_id == run_id:
        return
    alias_path = _run_snapshot_path(trace_id)
    if alias_path.exists():
        return
    try:
        write_json(
            alias_path,
            {
                "version": 1,
                "kind": _TRACE_ALIAS_KIND,
                "run_id": run_id,
                "trace_id": trace_id,
                "updated_at": str(payload.get("updated_at", "") or utc_now_iso()),
            },
            compact=True,
        )
    except OSError:
        pass


def _is_trace_alias(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("kind") == _TRACE_ALIAS_KIND


def _activity_state_payload(
    state: Any,
    *,
    stage: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = list(getattr(state, "metrics", []) or [])
    metric = metrics[-1] if metrics else None
    observation = getattr(metric, "observation", None)
    return {
        "version": 2,
        "updated_at": utc_now_iso(),
        "stage": str(stage or ""),
        "run_id": str(
            getattr(state, "run_id", "") or getattr(state, "trace_id", "") or ""
        ),
        "trace_id": str(getattr(state, "trace_id", "") or ""),
        "session_key": str(getattr(state, "session_key", "") or ""),
        "status": str(getattr(state, "status", "") or ""),
        "paused_reason": str(getattr(state, "paused_reason", "") or ""),
        "pending_approval": str(getattr(state, "pending_approval", "") or ""),
        "step": int(getattr(state, "step", 0) or 0),
        "stop_reason": str(getattr(state, "stop_reason", "") or ""),
        "activity": {
            "kind": str(getattr(metric, "kind", "") or ""),
            "tool_name": str(getattr(metric, "tool_name", "") or ""),
            "ok": getattr(observation, "ok", None),
        },
        "metadata": to_jsonable(metadata),
    }


def _should_persist_snapshot(snapshot: dict[str, Any]) -> bool:
    stage = str(snapshot.get("stage", "") or "")
    status = str(snapshot.get("status", "") or "")
    if stage in _IMPORTANT_PERSIST_STAGES:
        return True
    if status in {"paused", "cancelled", "completed", "failed"}:
        return True
    if stage == "paused" and snapshot.get("pending_approval"):
        return True

    key = str(snapshot.get("run_id", "") or snapshot.get("trace_id", "") or "")
    if not key:
        return False
    last = _LAST_PERSIST_AT.get(key, 0.0)
    return time.monotonic() - last >= _PERSIST_THROTTLE_SECONDS


def _mark_snapshot_persisted(snapshot: dict[str, Any]) -> None:
    key = str(snapshot.get("run_id", "") or snapshot.get("trace_id", "") or "")
    if not key:
        return
    _LAST_PERSIST_AT[key] = time.monotonic()
    while len(_LAST_PERSIST_AT) > _ACTIVITY_RING_MAX_RUNS:
        _LAST_PERSIST_AT.pop(next(iter(_LAST_PERSIST_AT)))


def _remember_activity(snapshot: dict[str, Any]) -> None:
    stage = str(snapshot.get("stage", "") or "")
    activity = _activity_from_snapshot(snapshot, stage=stage)
    if not activity:
        return
    keys = {
        _safe_trace_id(str(snapshot.get("run_id", "") or "")),
        _safe_trace_id(str(snapshot.get("trace_id", "") or "")),
    }
    for key in {item for item in keys if item}:
        ring = _ACTIVITY_RING.setdefault(
            key,
            deque(maxlen=_ACTIVITY_RING_MAX_ITEMS),
        )
        ring.append(activity)
    while len(_ACTIVITY_RING) > _ACTIVITY_RING_MAX_RUNS:
        _ACTIVITY_RING.pop(next(iter(_ACTIVITY_RING)))


def _activity_from_snapshot(
    snapshot: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    activity = snapshot.get("activity")
    activity = activity if isinstance(activity, dict) else {}
    tool_name = str(activity.get("tool_name", "") or "")
    if stage in {"tool_calls", "tool_observation"} and tool_name:
        return _activity_payload(
            snapshot,
            stage=stage,
            tool_name=tool_name,
            ok=activity.get("ok") if isinstance(activity.get("ok"), bool) else None,
        )
    if stage == "paused" and snapshot.get("pending_approval"):
        return _activity_payload(snapshot, stage=stage, tool_name="approval_required")
    return {}


def _activity_payload(
    snapshot: dict[str, Any],
    *,
    stage: str,
    tool_name: str,
    ok: bool | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "tool_name": str(tool_name or ""),
        "ok": ok,
        "step": int(snapshot.get("step", 0) or 0),
        "updated_at": str(snapshot.get("updated_at", "") or ""),
    }


def _run_snapshot_path(trace_id: str):
    safe_trace = _safe_trace_id(trace_id)
    return state_path("agent_runs", f"{safe_trace or 'unknown'}.json")


def _safe_trace_id(trace_id: str) -> str:
    return "".join(ch for ch in str(trace_id or "") if ch.isalnum() or ch in {"-", "_"})


def _compact_run_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": payload.get("run_id", payload.get("trace_id", "")),
        "trace_id": payload.get("trace_id", ""),
        "session_key": payload.get("session_key", ""),
        "updated_at": payload.get("updated_at", ""),
        "stage": payload.get("stage", ""),
        "status": payload.get("status", ""),
        "paused_reason": payload.get("paused_reason", ""),
        "pending_approval": payload.get("pending_approval", ""),
        "artifact_refs": payload.get("artifact_refs", [])[:10]
        if isinstance(payload.get("artifact_refs"), list)
        else [],
        "step": payload.get("step", 0),
        "max_steps": payload.get("max_steps", 0),
        "stop_reason": payload.get("stop_reason", ""),
        "final_text": str(payload.get("final_text", "") or "")[:500],
    }


def _state_from_snapshot(
    snapshot: dict[str, Any],
    *,
    tool_map: dict[str, ToolExecutable],
) -> AgentRunState | None:
    try:
        state = AgentRunState(
            trace_id=str(snapshot.get("trace_id", "") or snapshot.get("run_id", "")),
            run_id=str(snapshot.get("run_id", "") or snapshot.get("trace_id", "")),
            session_key=str(snapshot.get("session_key", "") or "") or None,
            messages=_messages_from_payload(snapshot.get("messages", [])),
            tool_map=dict(tool_map),
            status=str(snapshot.get("status", "") or "running"),
            paused_reason=str(snapshot.get("paused_reason", "") or ""),
            pending_approval=_pending_approval_from_snapshot(snapshot),
            artifact_refs=_text_list(snapshot.get("artifact_refs")),
            tool_executions=_tool_executions_from_payload(
                snapshot.get("tool_executions", [])
            ),
            stop_reason=str(snapshot.get("stop_reason", "") or "running"),
            step=int(snapshot.get("step", 0) or 0),
            max_steps=int(snapshot.get("max_steps", 5) or 5),
            cost_checkpoint_tokens=int(
                snapshot.get(
                    "cost_checkpoint_tokens",
                    snapshot.get("max_total_tokens", 0),
                )
                or 0
            ),
            compression_failure_fingerprint=str(
                snapshot.get("compression_failure_fingerprint", "") or ""
            ),
            compression_failure_count=max(
                int(snapshot.get("compression_failure_count", 0) or 0),
                0,
            ),
            budget=_budget_from_payload(snapshot.get("budget", {})),
            final_text=str(snapshot.get("final_text", "") or ""),
            delivery_complete=bool(snapshot.get("delivery_complete", False)),
            final_source=str(snapshot.get("final_source", "") or ""),
        )
    except Exception:
        return None
    return state


def _messages_from_payload(value: Any) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    if not isinstance(value, list | tuple):
        return messages
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            payload = dict(item)
            payload["tool_calls"] = _tool_calls_from_payload(
                payload.get("tool_calls", [])
            )
            messages.append(LLMMessage(**payload))
        except Exception:
            continue
    return messages


def _tool_executions_from_payload(value: Any) -> list[ToolExecutionRecord]:
    records: list[ToolExecutionRecord] = []
    if not isinstance(value, list | tuple):
        return records
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            records.append(
                ToolExecutionRecord(
                    tool_call_id=str(item.get("tool_call_id", "") or ""),
                    tool_name=str(item.get("tool_name", "") or ""),
                    fingerprint=str(item.get("fingerprint", "") or ""),
                    status=str(item.get("status", "") or "started"),
                    step=int(item.get("step", 0) or 0),
                    started_at=float(item.get("started_at", 0.0) or 0.0),
                    completed_at=float(item.get("completed_at", 0.0) or 0.0),
                    result_status=str(item.get("result_status", "") or ""),
                )
            )
        except Exception:
            continue
    return records


def _tool_calls_from_payload(value: Any) -> list[LLMToolCall]:
    calls: list[LLMToolCall] = []
    if not isinstance(value, list | tuple):
        return calls
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            function = item.get("function")
            calls.append(
                LLMToolCall(
                    id=str(item.get("id", "") or ""),
                    function=LLMToolFunction(
                        name=str((function or {}).get("name", "") or ""),
                        arguments=str((function or {}).get("arguments", "") or ""),
                    ),
                    thought_signature=item.get("thought_signature"),
                )
            )
        except Exception:
            continue
    return calls


def _budget_from_payload(value: Any) -> AgentBudgetState:
    if not isinstance(value, dict):
        return AgentBudgetState()
    legacy_prompt_tokens = int(value.get("prompt_tokens", 0) or 0)
    return AgentBudgetState(
        classifier_calls=int(value.get("classifier_calls", 0) or 0),
        hook_calls=int(value.get("hook_calls", 0) or 0),
        tool_calls=int(value.get("tool_calls", 0) or 0),
        tool_batches=int(value.get("tool_batches", 0) or 0),
        run_input_tokens=int(value.get("run_input_tokens", legacy_prompt_tokens) or 0),
        run_output_tokens=int(value.get("run_output_tokens", 0) or 0),
        current_context_tokens=int(value.get("current_context_tokens", 0) or 0),
        last_usage_message_count=int(value.get("last_usage_message_count", 0) or 0),
        last_usage_schema_tokens=int(value.get("last_usage_schema_tokens", 0) or 0),
        model_calls=int(value.get("model_calls", 0) or 0),
        durations_ms=dict(value.get("durations_ms") or {}),
    )


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if str(item or "")]


def _pending_approval_from_snapshot(snapshot: dict[str, Any]) -> str:
    current = str(snapshot.get("pending_approval", "") or "")
    if current:
        return current
    legacy = _text_list(snapshot.get("waiting_approval_ids"))
    return legacy[-1] if legacy else ""


get_artifact_store().set_protected_ids_provider(_retained_session_artifact_ids)


__all__ = [
    "activate_agent_session",
    "agent_session_is_active",
    "archive_agent_session",
    "archive_conversation",
    "cleanup_agent_run_storage",
    "clear_agent_run_cancel_signal",
    "clear_agent_session_context",
    "create_conversation",
    "deactivate_agent_session",
    "delete_conversation",
    "get_active_agent_run_id",
    "get_active_conversation",
    "get_agent_run_messages",
    "get_agent_run_snapshot",
    "get_agent_session",
    "list_agent_run_activities",
    "list_agent_run_snapshots",
    "list_conversations",
    "load_agent_run_state",
    "persist_agent_run_messages",
    "persist_agent_run_state",
    "rename_conversation",
    "restore_conversation",
    "set_conversation_permission_mode",
    "switch_conversation",
    "update_agent_run_status",
]
