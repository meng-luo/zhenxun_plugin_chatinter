"""Unified runtime event protocol for ChatInter Agent operations.

RuntimeEvent is the append-only fact stream for AgentRun, tool progress,
observations, artifacts, approvals, task graph updates and background jobs.
Snapshots are still useful for fast resume, but replay/projection from this
stream is the durable source for audit and recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Iterable, Literal
import uuid

from .persistence import append_jsonl, read_json, state_path, to_jsonable, write_json
from .route_text import normalize_message_text

RuntimeEventKind = Literal[
    "agent_run",
    "model_request",
    "tool_call",
    "tool_observation",
    "observation",
    "artifact",
    "approval",
    "background_job",
    "task_graph",
    "task_ledger",
    "guardrail",
    "todo",
    "audit",
    "system",
]

RuntimeEventStatus = Literal[
    "created",
    "started",
    "progress",
    "waiting",
    "completed",
    "failed",
    "cancelled",
    "blocked",
    "expired",
    "info",
]

_EVENTS_JSONL_PATH = state_path("runtime_events", "events.jsonl")
_EVENTS_INDEX_PATH = state_path("runtime_events", "events_index.json")
_MAX_INDEX_EVENTS = 1200
_MAX_REPLAY_SCAN_EVENTS = 20000
_MAX_PAYLOAD_CHARS = 3600
_INDEX: dict[str, dict[str, Any]] = {}
_LOADED = False


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    kind: RuntimeEventKind
    status: RuntimeEventStatus
    source: str = ""
    run_id: str = ""
    trace_id: str = ""
    session_key: str = ""
    user_id: str = ""
    step: int = 0
    parent_event_id: str = ""
    related_ids: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[dict[str, Any], ...] = ()
    created_at: float = field(default_factory=time.time)

    def public_payload(self, *, include_payload: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = int(self.created_at)
        payload["artifacts"] = list(self.artifacts)
        if not include_payload:
            payload.pop("payload", None)
        return payload

    def compact_payload(self) -> dict[str, Any]:
        payload = self.public_payload(include_payload=False)
        payload["payload_keys"] = sorted(self.payload)[:24]
        return payload


@dataclass
class RuntimeEventProjection:
    """Replay-friendly state derived from RuntimeEvent facts."""

    run_id: str = ""
    trace_id: str = ""
    session_key: str = ""
    user_id: str = ""
    status: str = ""
    paused_reason: str = ""
    stop_reason: str = ""
    recovery_action: str = ""
    step: int = 0
    last_event_id: str = ""
    last_event_at: float = 0.0
    event_count: int = 0
    kinds: dict[str, int] = field(default_factory=dict)
    status_by_kind: dict[str, str] = field(default_factory=dict)
    waiting_approval_ids: list[str] = field(default_factory=list)
    background_task_ids: list[str] = field(default_factory=list)
    observation_event_ids: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    background_jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_graph: dict[str, Any] = field(default_factory=dict)
    task_ledger: dict[str, Any] = field(default_factory=dict)
    pending_tasks: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    guardrails: list[dict[str, Any]] = field(default_factory=list)

    def public_payload(self, *, include_details: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "session_key": self.session_key,
            "user_id": self.user_id,
            "status": self.status,
            "paused_reason": self.paused_reason,
            "stop_reason": self.stop_reason,
            "recovery_action": self.recovery_action,
            "step": self.step,
            "last_event_id": self.last_event_id,
            "last_event_at": int(self.last_event_at or 0),
            "event_count": self.event_count,
            "kinds": dict(sorted(self.kinds.items())),
            "status_by_kind": dict(sorted(self.status_by_kind.items())),
            "waiting_approval_ids": list(self.waiting_approval_ids),
            "background_task_ids": list(self.background_task_ids),
            "observation_event_ids": list(self.observation_event_ids[-50:]),
            "artifact_refs": list(self.artifact_refs[-50:]),
            "pending_tasks": list(self.pending_tasks[-20:]),
            "completed_tasks": list(self.completed_tasks[-20:]),
        }
        if include_details:
            payload.update(
                {
                    "tool_calls": list(self.tool_calls[-40:]),
                    "observations": list(self.observations[-60:]),
                    "approvals": dict(self.approvals),
                    "background_jobs": dict(self.background_jobs),
                    "artifacts": dict(self.artifacts),
                    "task_graph": dict(self.task_graph),
                    "task_ledger": dict(self.task_ledger),
                    "guardrails": list(self.guardrails[-20:]),
                }
            )
        return payload


def emit_runtime_event(
    *,
    kind: RuntimeEventKind,
    status: RuntimeEventStatus = "info",
    source: str = "",
    run_id: str = "",
    trace_id: str = "",
    session_key: str = "",
    user_id: str = "",
    step: int = 0,
    parent_event_id: str = "",
    related_ids: dict[str, Any] | None = None,
    summary: str = "",
    payload: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> RuntimeEvent:
    """Append one compact runtime event.

    This function must be best-effort and never raise to callers.  The returned
    event is still useful in memory even if persistence fails.
    """

    event = RuntimeEvent(
        event_id=uuid.uuid4().hex[:12],
        kind=kind,
        status=status,
        source=normalize_message_text(source),
        run_id=normalize_message_text(run_id),
        trace_id=normalize_message_text(trace_id),
        session_key=normalize_message_text(session_key),
        user_id=normalize_message_text(user_id),
        step=max(int(step or 0), 0),
        parent_event_id=normalize_message_text(parent_event_id),
        related_ids={
            normalize_message_text(str(key)): normalize_message_text(str(value))
            for key, value in dict(related_ids or {}).items()
            if normalize_message_text(str(key)) and normalize_message_text(str(value))
        },
        summary=normalize_message_text(summary)[:700],
        payload=_compact_payload(payload or {}),
        artifacts=tuple(_compact_artifacts(artifacts or ())),
    )
    try:
        _remember_event(event)
    except Exception:
        pass
    return event


def emit_runtime_event_from_state(
    state: Any,
    *,
    kind: RuntimeEventKind,
    status: RuntimeEventStatus = "info",
    source: str = "",
    summary: str = "",
    payload: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    related_ids: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return emit_runtime_event(
        kind=kind,
        status=status,
        source=source,
        run_id=str(getattr(state, "run_id", "") or ""),
        trace_id=str(getattr(state, "trace_id", "") or ""),
        session_key=str(getattr(state, "session_key", "") or ""),
        step=int(getattr(state, "step", 0) or 0),
        summary=summary,
        payload=payload,
        artifacts=artifacts,
        related_ids=related_ids,
    )


def list_runtime_events(
    *,
    run_id: str = "",
    trace_id: str = "",
    session_key: str = "",
    kind: str = "",
    status: str = "",
    source_contains: str = "",
    after_event_id: str = "",
    limit: int = 50,
    include_payload: bool = False,
) -> list[dict[str, Any]]:
    _ensure_loaded()
    normalized_run = normalize_message_text(run_id)
    normalized_trace = normalize_message_text(trace_id)
    normalized_session = normalize_message_text(session_key)
    normalized_kind = normalize_message_text(kind)
    normalized_status = normalize_message_text(status)
    source_filter = normalize_message_text(source_contains).lower()
    after_seen = not bool(after_event_id)
    rows = sorted(_INDEX.values(), key=lambda item: float(item.get("created_at") or 0.0))
    result: list[dict[str, Any]] = []
    for payload in rows:
        if payload.get("event_id") == after_event_id:
            after_seen = True
            continue
        if not after_seen:
            continue
        if normalized_run and payload.get("run_id") != normalized_run:
            continue
        if normalized_trace and payload.get("trace_id") != normalized_trace:
            continue
        if normalized_session and payload.get("session_key") != normalized_session:
            continue
        if normalized_kind and payload.get("kind") != normalized_kind:
            continue
        if normalized_status and payload.get("status") != normalized_status:
            continue
        if source_filter and source_filter not in str(payload.get("source", "")).lower():
            continue
        result.append(
            dict(payload)
            if include_payload
            else _without_payload(dict(payload))
        )
    return result[-max(1, min(int(limit or 50), 300)) :]


def get_runtime_event(event_id: str) -> dict[str, Any] | None:
    _ensure_loaded()
    payload = _INDEX.get(normalize_message_text(event_id))
    return dict(payload) if isinstance(payload, dict) else None


def replay_runtime_events(
    *,
    run_id: str = "",
    trace_id: str = "",
    session_key: str = "",
    user_id: str = "",
    kind: str = "",
    after_event_id: str = "",
    limit: int = 1000,
    include_payload: bool = True,
) -> list[dict[str, Any]]:
    """Replay persisted events from the append-only JSONL stream.

    Unlike ``list_runtime_events`` this scans the durable event log, so it can
    reconstruct older runs even after the in-memory/index window has rotated.
    """

    normalized_run = normalize_message_text(run_id)
    normalized_trace = normalize_message_text(trace_id)
    normalized_session = normalize_message_text(session_key)
    normalized_user = normalize_message_text(user_id)
    normalized_kind = normalize_message_text(kind)
    after_seen = not bool(after_event_id)
    rows = _read_persisted_event_rows()
    result: list[dict[str, Any]] = []
    for payload in rows:
        event_id = normalize_message_text(str(payload.get("event_id", "") or ""))
        if event_id == after_event_id:
            after_seen = True
            continue
        if not after_seen:
            continue
        if normalized_run and payload.get("run_id") != normalized_run:
            continue
        if normalized_trace and payload.get("trace_id") != normalized_trace:
            continue
        if normalized_session and payload.get("session_key") != normalized_session:
            continue
        if normalized_user and payload.get("user_id") != normalized_user:
            continue
        if normalized_kind and payload.get("kind") != normalized_kind:
            continue
        result.append(dict(payload) if include_payload else _without_payload(dict(payload)))
    return result[-max(1, min(int(limit or 1000), _MAX_REPLAY_SCAN_EVENTS)) :]


def project_runtime_state(
    *,
    run_id: str = "",
    trace_id: str = "",
    session_key: str = "",
    user_id: str = "",
    include_details: bool = True,
    limit: int = 5000,
) -> dict[str, Any]:
    """Replay a RuntimeEvent stream into a compact resumable projection."""

    events = replay_runtime_events(
        run_id=run_id,
        trace_id=trace_id,
        session_key=session_key,
        user_id=user_id,
        limit=limit,
        include_payload=True,
    )
    events = _with_related_session_events(
        events,
        run_id=run_id,
        trace_id=trace_id,
        session_key=session_key,
        user_id=user_id,
        limit=limit,
    )
    projection = build_runtime_projection(events)
    if run_id and not projection.run_id:
        projection.run_id = normalize_message_text(run_id)
    if trace_id and not projection.trace_id:
        projection.trace_id = normalize_message_text(trace_id)
    if session_key and not projection.session_key:
        projection.session_key = normalize_message_text(session_key)
    if user_id and not projection.user_id:
        projection.user_id = normalize_message_text(user_id)
    return projection.public_payload(include_details=include_details)


def _with_related_session_events(
    events: list[dict[str, Any]],
    *,
    run_id: str,
    trace_id: str,
    session_key: str,
    user_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Add session-scoped lifecycle events related to a run projection.

    Approval/background/artifact events are sometimes emitted by lower-level
    stores before an AgentRun id is known.  Replay therefore starts with the
    run-scoped stream, discovers related ids, then folds in matching session
    events so resume sees consumed approvals and finished background tasks.
    """

    if not events or not session_key or not (run_id or trace_id):
        return events
    projection = build_runtime_projection(events)
    related = {
        *projection.waiting_approval_ids,
        *projection.background_task_ids,
        *projection.observation_event_ids,
        *projection.artifact_refs,
    }
    related = {normalize_message_text(item) for item in related if item}
    if not related:
        return events
    existing_ids = {
        normalize_message_text(str(item.get("event_id", "") or ""))
        for item in events
        if isinstance(item, dict)
    }
    supplemental = replay_runtime_events(
        session_key=session_key,
        user_id=user_id,
        limit=limit,
        include_payload=True,
    )
    merged = list(events)
    for event in supplemental:
        event_id = normalize_message_text(str(event.get("event_id", "") or ""))
        if not event_id or event_id in existing_ids:
            continue
        if _event_matches_related_ids(event, related):
            merged.append(event)
            existing_ids.add(event_id)
    return sorted(merged, key=lambda item: float(item.get("created_at") or 0.0))


def build_runtime_projection(events: Iterable[dict[str, Any]]) -> RuntimeEventProjection:
    projection = RuntimeEventProjection()
    for payload in sorted(
        (dict(item) for item in events if isinstance(item, dict)),
        key=lambda item: float(item.get("created_at") or 0.0),
    ):
        _apply_event_to_projection(projection, payload)
    return projection


def rebuild_runtime_event_index(*, max_events: int = _MAX_INDEX_EVENTS) -> int:
    """Rebuild the bounded query index from the append-only event log."""

    global _INDEX
    rows = _read_persisted_event_rows(max_scan=max(max_events, _MAX_INDEX_EVENTS))
    selected = rows[-max(1, min(int(max_events or _MAX_INDEX_EVENTS), _MAX_REPLAY_SCAN_EVENTS)) :]
    _INDEX = {
        normalize_message_text(str(item.get("event_id", "") or "")): dict(item)
        for item in selected
        if normalize_message_text(str(item.get("event_id", "") or ""))
    }
    _trim_index()
    write_json(_EVENTS_INDEX_PATH, _INDEX)
    return len(_INDEX)


def _remember_event(event: RuntimeEvent) -> None:
    _ensure_loaded()
    payload = event.public_payload(include_payload=True)
    _INDEX[event.event_id] = payload
    _trim_index()
    write_json(_EVENTS_INDEX_PATH, _INDEX)
    append_jsonl(_EVENTS_JSONL_PATH, payload)


def _read_persisted_event_rows(
    *,
    max_scan: int = _MAX_REPLAY_SCAN_EVENTS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if _EVENTS_JSONL_PATH.exists():
        try:
            lines = _EVENTS_JSONL_PATH.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except Exception:
            lines = []
        for line in lines[-max(1, min(int(max_scan or _MAX_REPLAY_SCAN_EVENTS), 100000)) :]:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    if rows:
        return rows
    _ensure_loaded()
    return sorted(_INDEX.values(), key=lambda item: float(item.get("created_at") or 0.0))


def _apply_event_to_projection(
    projection: RuntimeEventProjection,
    event: dict[str, Any],
) -> None:
    kind = normalize_message_text(str(event.get("kind", "") or "system"))
    status = normalize_message_text(str(event.get("status", "") or "info"))
    event_id = normalize_message_text(str(event.get("event_id", "") or ""))
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    related_ids = (
        event.get("related_ids") if isinstance(event.get("related_ids"), dict) else {}
    )
    projection.event_count += 1
    projection.kinds[kind] = projection.kinds.get(kind, 0) + 1
    projection.status_by_kind[kind] = status
    projection.last_event_id = event_id or projection.last_event_id
    projection.last_event_at = float(event.get("created_at") or projection.last_event_at or 0.0)
    projection.run_id = projection.run_id or normalize_message_text(
        str(event.get("run_id", "") or "")
    )
    projection.trace_id = projection.trace_id or normalize_message_text(
        str(event.get("trace_id", "") or "")
    )
    projection.session_key = projection.session_key or normalize_message_text(
        str(event.get("session_key", "") or "")
    )
    projection.user_id = projection.user_id or normalize_message_text(
        str(event.get("user_id", "") or "")
    )
    projection.step = max(projection.step, _safe_int(event.get("step")))

    _collect_related_refs(projection, related_ids)
    _collect_artifacts(projection, event.get("artifacts"))

    if kind == "agent_run":
        _apply_agent_run_event(projection, status=status, payload=payload)
    elif kind == "tool_call":
        projection.tool_calls.append(_compact_projection_event(event))
        projection.tool_calls = projection.tool_calls[-80:]
    elif kind in {"tool_observation", "observation"}:
        _apply_observation_event(projection, event, payload)
    elif kind == "approval":
        _apply_approval_event(projection, status=status, payload=payload, related_ids=related_ids)
    elif kind == "background_job":
        _apply_background_event(projection, status=status, payload=payload, related_ids=related_ids)
    elif kind == "artifact":
        _apply_artifact_event(projection, payload=payload, artifacts=event.get("artifacts"))
    elif kind == "task_graph":
        graph = _extract_task_payload(payload, key="graph")
        if graph:
            projection.task_graph = graph
    elif kind == "task_ledger":
        ledger = _extract_task_payload(payload, key="ledger")
        if ledger:
            projection.task_ledger = ledger
        _collect_task_lists(projection, payload)
    elif kind == "guardrail":
        projection.guardrails.append(_compact_projection_event(event))
        projection.guardrails = projection.guardrails[-40:]


def _apply_agent_run_event(
    projection: RuntimeEventProjection,
    *,
    status: str,
    payload: dict[str, Any],
) -> None:
    run_status = normalize_message_text(str(payload.get("status", "") or ""))
    if run_status:
        projection.status = run_status
    elif status in {"waiting", "completed", "failed", "cancelled"}:
        projection.status = "paused" if status == "waiting" else status
    projection.paused_reason = normalize_message_text(
        str(payload.get("paused_reason", "") or projection.paused_reason)
    )
    projection.stop_reason = normalize_message_text(
        str(payload.get("stop_reason", "") or projection.stop_reason)
    )
    projection.recovery_action = normalize_message_text(
        str(payload.get("recovery_action", "") or projection.recovery_action)
    )
    for key, target in (
        ("waiting_approval_ids", projection.waiting_approval_ids),
        ("background_task_ids", projection.background_task_ids),
        ("observation_event_ids", projection.observation_event_ids),
        ("artifact_refs", projection.artifact_refs),
    ):
        value = payload.get(key)
        if isinstance(value, list | tuple):
            for item in value:
                _append_unique(target, str(item or ""))
    _collect_task_lists(projection, payload)


def _apply_observation_event(
    projection: RuntimeEventProjection,
    event: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    projection.observations.append(_compact_projection_event(event))
    projection.observations = projection.observations[-120:]
    task_text = normalize_message_text(str(payload.get("task_text", "") or ""))
    if task_text:
        if bool(payload.get("ok")):
            _mark_completed(projection, task_text)
        elif bool(payload.get("need_continue")):
            _append_unique(projection.pending_tasks, task_text)
    hint = normalize_message_text(str(payload.get("remaining_task_hint", "") or ""))
    if hint:
        _append_unique(projection.pending_tasks, hint)
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    _collect_related_refs(projection, output)
    _collect_artifacts(projection, output.get("artifacts"))


def _apply_approval_event(
    projection: RuntimeEventProjection,
    *,
    status: str,
    payload: dict[str, Any],
    related_ids: dict[str, Any],
) -> None:
    approval_id = _first_text(
        related_ids.get("approval_id"),
        payload.get("approval_id"),
    )
    if not approval_id:
        return
    projection.approvals[approval_id] = {
        "approval_id": approval_id,
        "status": status,
        "action": normalize_message_text(str(payload.get("action", "") or "")),
        "reason": normalize_message_text(str(payload.get("reason", "") or "")),
        "ttl_seconds": _safe_int(payload.get("ttl_seconds")),
    }
    if status == "waiting":
        _append_unique(projection.waiting_approval_ids, approval_id)
    elif status in {"completed", "cancelled", "expired", "failed"}:
        projection.waiting_approval_ids = [
            item for item in projection.waiting_approval_ids if item != approval_id
        ]


def _apply_background_event(
    projection: RuntimeEventProjection,
    *,
    status: str,
    payload: dict[str, Any],
    related_ids: dict[str, Any],
) -> None:
    task_id = _first_text(
        related_ids.get("background_task_id"),
        payload.get("task_id"),
    )
    event_id = _first_text(
        related_ids.get("observation_event_id"),
        payload.get("event_id"),
    )
    if event_id:
        _append_unique(projection.observation_event_ids, event_id)
    if not task_id:
        return
    projection.background_jobs[task_id] = {
        "task_id": task_id,
        "status": status,
        "action": normalize_message_text(str(payload.get("action", "") or "")),
        "kind": normalize_message_text(str(payload.get("kind", "") or "")),
        "returncode": payload.get("returncode"),
        "error": normalize_message_text(str(payload.get("error", "") or "")),
        "last_event_id": event_id,
    }
    if status in {"progress", "started", "waiting", "info"}:
        _append_unique(projection.background_task_ids, task_id)
    elif status in {"completed", "failed", "cancelled"}:
        _append_unique(projection.background_task_ids, task_id)


def _apply_artifact_event(
    projection: RuntimeEventProjection,
    *,
    payload: dict[str, Any],
    artifacts: Any,
) -> None:
    _collect_artifacts(projection, artifacts)
    artifact_id = normalize_message_text(str(payload.get("artifact_id", "") or ""))
    if artifact_id:
        projection.artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "type": normalize_message_text(str(payload.get("type", "") or "")),
            "summary": normalize_message_text(str(payload.get("summary", "") or "")),
            "size": _safe_int(payload.get("size")),
            "source": normalize_message_text(str(payload.get("source", "") or "")),
        }
        _append_unique(projection.artifact_refs, artifact_id)


def _collect_related_refs(
    projection: RuntimeEventProjection,
    payload: dict[str, Any],
) -> None:
    _append_unique(
        projection.waiting_approval_ids,
        _first_text(
            payload.get("approval_id"),
            _nested_payload_get(payload, "approval", "approval_id"),
        ),
    )
    _append_unique(
        projection.background_task_ids,
        _first_text(
            payload.get("background_task_id"),
            payload.get("task_id"),
            _nested_payload_get(payload, "task", "task_id"),
        ),
    )
    _append_unique(
        projection.observation_event_ids,
        _first_text(
            payload.get("observation_event_id"),
            payload.get("event_id"),
            _nested_payload_get(payload, "event", "event_id"),
            _nested_payload_get(payload, "observation_event", "event_id"),
        ),
    )
    _append_unique(
        projection.artifact_refs,
        _first_text(
            payload.get("artifact_id"),
            _nested_payload_get(payload, "artifact", "artifact_id"),
        ),
    )


def _collect_artifacts(projection: RuntimeEventProjection, artifacts: Any) -> None:
    if not isinstance(artifacts, list | tuple):
        return
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        artifact_id = normalize_message_text(str(item.get("artifact_id", "") or ""))
        if not artifact_id:
            continue
        projection.artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "type": normalize_message_text(str(item.get("type", "") or "")),
            "summary": normalize_message_text(str(item.get("summary", "") or "")),
            "size": _safe_int(item.get("size")),
            "source": normalize_message_text(str(item.get("source", "") or "")),
        }
        _append_unique(projection.artifact_refs, artifact_id)


def _collect_task_lists(
    projection: RuntimeEventProjection,
    payload: dict[str, Any],
) -> None:
    for key, target in (
        ("pending_tasks", projection.pending_tasks),
        ("completed_tasks", projection.completed_tasks),
    ):
        value = payload.get(key)
        if isinstance(value, list | tuple):
            for item in value:
                if isinstance(item, dict):
                    text = str(item.get("text") or item.get("goal") or "")
                else:
                    text = str(item or "")
                _append_unique(target, text)
    ledger = _extract_task_payload(payload, key="ledger")
    if ledger:
        tasks = ledger.get("tasks") if isinstance(ledger, dict) else []
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                goal = normalize_message_text(str(task.get("goal", "") or ""))
                status = normalize_message_text(str(task.get("status", "") or ""))
                if status == "completed":
                    _mark_completed(projection, goal)
                elif goal:
                    _append_unique(projection.pending_tasks, goal)


def _extract_task_payload(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    candidates = [
        metadata.get(key),
        metadata.get("task_" + key),
        metadata.get("task_graph" if key == "graph" else "task_ledger"),
        payload.get(key),
        payload.get("task_" + key),
        payload.get("task_graph" if key == "graph" else "task_ledger"),
    ]
    for item in candidates:
        if isinstance(item, dict):
            return dict(item)
    return {}


def _compact_projection_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "event_id": event.get("event_id", ""),
        "kind": event.get("kind", ""),
        "status": event.get("status", ""),
        "source": event.get("source", ""),
        "step": event.get("step", 0),
        "summary": event.get("summary", ""),
        "tool_name": payload.get("tool_name", metadata.get("tool_name", "")),
        "command_id": payload.get("command_id", ""),
        "task_text": payload.get("task_text", ""),
        "ok": payload.get("ok"),
        "created_at": event.get("created_at", 0),
        "related_ids": event.get("related_ids", {}),
    }


def _append_unique(target: list[str], value: str) -> None:
    normalized = normalize_message_text(str(value or ""))
    if normalized and normalized not in target:
        target.append(normalized)
        del target[:-120]


def _mark_completed(projection: RuntimeEventProjection, value: str) -> None:
    normalized = normalize_message_text(str(value or ""))
    if not normalized:
        return
    _append_unique(projection.completed_tasks, normalized)
    projection.pending_tasks = [
        item for item in projection.pending_tasks if item != normalized
    ]


def _event_matches_related_ids(event: dict[str, Any], related: set[str]) -> bool:
    if not related:
        return False
    values: list[str] = []
    related_ids = event.get("related_ids") if isinstance(event.get("related_ids"), dict) else {}
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    artifacts = event.get("artifacts")
    for source in (related_ids, payload):
        values.extend(_flat_related_values(source))
    if isinstance(artifacts, list | tuple):
        for artifact in artifacts:
            if isinstance(artifact, dict):
                values.extend(_flat_related_values(artifact))
    return any(normalize_message_text(value) in related for value in values)


def _flat_related_values(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    interesting = {
        "approval_id",
        "background_task_id",
        "task_id",
        "observation_event_id",
        "event_id",
        "artifact_id",
        "command_id",
        "tool_call_id",
    }
    for key, value in payload.items():
        if key in interesting and value is not None:
            result.append(str(value))
        elif isinstance(value, dict):
            result.extend(_flat_related_values(value))
        elif isinstance(value, list | tuple):
            for item in value:
                if isinstance(item, dict):
                    result.extend(_flat_related_values(item))
    return result


def _first_text(*values: Any) -> str:
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text:
            return text
    return ""


def _nested_payload_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    raw = read_json(_EVENTS_INDEX_PATH, {})
    if not isinstance(raw, dict):
        return
    for event_id, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        normalized_id = normalize_message_text(
            str(payload.get("event_id") or event_id or "")
        )
        if normalized_id:
            _INDEX[normalized_id] = dict(payload)
    _trim_index()


def _trim_index() -> None:
    if len(_INDEX) <= _MAX_INDEX_EVENTS:
        return
    rows = sorted(_INDEX.items(), key=lambda item: float(item[1].get("created_at") or 0.0))
    for event_id, _payload in rows[: max(len(rows) - _MAX_INDEX_EVENTS, 0)]:
        _INDEX.pop(event_id, None)


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compacted = to_jsonable(payload)
    try:
        text = json.dumps(compacted, ensure_ascii=False, default=str)
    except Exception:
        text = str(compacted)
    if len(text) <= _MAX_PAYLOAD_CHARS:
        return compacted if isinstance(compacted, dict) else {"value": compacted}
    return {
        "truncated": True,
        "summary": normalize_message_text(text)[:_MAX_PAYLOAD_CHARS],
        "original_chars": len(text),
    }


def _compact_artifacts(
    artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        artifact_id = normalize_message_text(str(item.get("artifact_id", "") or ""))
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        result.append(
            {
                "artifact_id": artifact_id,
                "type": normalize_message_text(str(item.get("type", "") or "")),
                "summary": normalize_message_text(str(item.get("summary", "") or ""))[:300],
                "size": _safe_int(item.get("size")),
                **(
                    {"source": normalize_message_text(str(item.get("source", "") or ""))}
                    if item.get("source")
                    else {}
                ),
            }
        )
    return result[:24]


def _without_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("payload", None)
    return payload


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "RuntimeEvent",
    "RuntimeEventKind",
    "RuntimeEventProjection",
    "RuntimeEventStatus",
    "build_runtime_projection",
    "emit_runtime_event",
    "emit_runtime_event_from_state",
    "get_runtime_event",
    "list_runtime_events",
    "project_runtime_state",
    "rebuild_runtime_event_index",
    "replay_runtime_events",
]
