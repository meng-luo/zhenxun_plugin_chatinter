"""Durable AgentRun snapshots for ChatInter.

Agent runs are short-lived in normal group chat, but superuser private turns can
span long tool loops.  Persisting every transition gives us a restart-safe audit
trail and a recovery surface for future resumable runs.
"""

from __future__ import annotations

import json
from typing import Any, cast

from zhenxun.services.llm import LLMMessage
from zhenxun.services.llm.types.models import (
    LLMToolCall,
    LLMToolFunction,
    ToolResult,
)
from zhenxun.services.llm.types.protocols import ToolExecutable

from .agent_state import (
    AgentBudgetState,
    AgentObservation,
    AgentRunState,
    AgentRuntimeTimelineItem,
    CompletedTask,
    PendingTask,
)
from .persistence import (
    append_jsonl,
    read_json,
    state_path,
    to_jsonable,
    utc_now_iso,
    write_json,
)
from .runtime_events import (
    RuntimeEventStatus,
    emit_runtime_event,
    project_runtime_state,
)
from .task_graph import task_graph_from_payload, task_graph_to_payload
from .task_ledger import (
    capability_ledger_from_payload,
    task_ledger_from_payload,
)

# In-process cancel signals (P0-1): the cancel tool flips a flag here so the
# runtime loop can react without re-reading the persisted snapshot every step.
_CANCEL_SIGNALS: set[str] = set()
_CANCEL_SIGNALS_MAX = 512


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


def persist_agent_run_state(
    state: Any,
    *,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        payload = _state_payload(state, stage=stage, metadata=metadata or {})
        write_json(_run_snapshot_path(str(state.trace_id)), payload)
        run_id = str(getattr(state, "run_id", "") or "")
        if run_id and run_id != str(getattr(state, "trace_id", "") or ""):
            write_json(_run_snapshot_path(run_id), payload)
        append_jsonl(_run_events_path(), _event_payload(payload))
        _emit_agent_run_runtime_event(payload)
    except Exception:
        # Persistence must never break a user turn or tool execution.
        return


def get_agent_run_snapshot(trace_id: str) -> dict[str, Any] | None:
    safe_trace = _safe_trace_id(trace_id)
    if not safe_trace:
        return None
    payload = read_json(_run_snapshot_path(safe_trace), None)
    return payload if isinstance(payload, dict) else None


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
    projection = project_agent_run_state(
        run_id=run_id,
        session_key=str(snapshot.get("session_key", "") or ""),
        include_details=False,
    )
    _merge_projection_refs(state, projection)
    return state


def project_agent_run_state(
    *,
    run_id: str = "",
    trace_id: str = "",
    session_key: str = "",
    include_details: bool = True,
) -> dict[str, Any]:
    return project_runtime_state(
        run_id=run_id,
        trace_id=trace_id,
        session_key=session_key,
        include_details=include_details,
    )


def update_agent_run_status(
    run_id: str,
    *,
    status: str,
    reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if str(status or "") == "cancelled":
        signal_agent_run_cancel(run_id)
    snapshot = get_agent_run_snapshot(run_id)
    if not isinstance(snapshot, dict):
        return None
    snapshot["updated_at"] = utc_now_iso()
    snapshot["status"] = str(status or "")
    snapshot["paused_reason"] = "" if status != "paused" else str(reason or "")
    snapshot["stop_reason"] = str(status or "")
    snapshot["recovery_action"] = str(reason or "")
    snapshot["metadata"] = to_jsonable(
        {**dict(snapshot.get("metadata") or {}), **dict(metadata or {})}
    )
    safe_run = _safe_trace_id(str(snapshot.get("run_id", "") or run_id))
    safe_trace = _safe_trace_id(str(snapshot.get("trace_id", "") or ""))
    write_json(_run_snapshot_path(safe_run), snapshot)
    if safe_trace and safe_trace != safe_run:
        write_json(_run_snapshot_path(safe_trace), snapshot)
    append_jsonl(_run_events_path(), _event_payload({**snapshot, "stage": status}))
    _emit_agent_run_runtime_event({**snapshot, "stage": status})
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
        if (
            normalized_session
            and str(payload.get("session_key", "")) != normalized_session
        ):
            continue
        rows.append(_compact_run_snapshot(payload))
    rows.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return rows[: max(1, min(int(limit or 20), 100))]


def query_agent_run_events(
    *,
    trace_id: str = "",
    session_key: str = "",
    limit: int = 50,
) -> list[dict[str, Any]]:
    path = _run_events_path()
    if not path.exists():
        return []
    normalized_trace = _safe_trace_id(trace_id)
    normalized_session = str(session_key or "")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    for line in reversed(lines[-2000:]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if normalized_trace and str(payload.get("trace_id", "")) != normalized_trace:
            continue
        if (
            normalized_session
            and str(payload.get("session_key", "")) != normalized_session
        ):
            continue
        rows.append(payload)
        if len(rows) >= max(1, min(int(limit or 50), 200)):
            break
    return rows


def _state_payload(
    state: Any,
    *,
    stage: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": utc_now_iso(),
        "stage": str(stage or ""),
        "run_id": str(
            getattr(state, "run_id", "") or getattr(state, "trace_id", "") or ""
        ),
        "trace_id": str(getattr(state, "trace_id", "") or ""),
        "session_key": str(getattr(state, "session_key", "") or ""),
        "status": str(getattr(state, "status", "") or ""),
        "paused_reason": str(getattr(state, "paused_reason", "") or ""),
        "resume_cursor": to_jsonable(getattr(state, "resume_cursor", {})),
        "waiting_approval_ids": to_jsonable(getattr(state, "waiting_approval_ids", [])),
        "background_task_ids": to_jsonable(getattr(state, "background_task_ids", [])),
        "observation_event_ids": to_jsonable(
            getattr(state, "observation_event_ids", [])
        ),
        "artifact_refs": to_jsonable(getattr(state, "artifact_refs", [])),
        "step": int(getattr(state, "step", 0) or 0),
        "max_steps": int(getattr(state, "max_steps", 0) or 0),
        "stop_reason": str(getattr(state, "stop_reason", "") or ""),
        "recovery_action": str(getattr(state, "recovery_action", "") or ""),
        "final_text": str(getattr(state, "final_text", "") or ""),
        "tool_names": sorted(str(name) for name in getattr(state, "tool_map", {})),
        "tool_obligation": str(getattr(state, "tool_obligation", "") or ""),
        "tool_obligation_reason": str(
            getattr(state, "tool_obligation_reason", "") or ""
        ),
        "agent_complexity_mode": str(getattr(state, "agent_complexity_mode", "") or ""),
        "agent_complexity_reason": str(
            getattr(state, "agent_complexity_reason", "") or ""
        ),
        "required_tool_names": to_jsonable(getattr(state, "required_tool_names", ())),
        "tool_calls": to_jsonable(getattr(state, "tool_calls", [])),
        "observations": to_jsonable(getattr(state, "observations", [])),
        "pending_tasks": to_jsonable(getattr(state, "pending_tasks", [])),
        "completed_tasks": to_jsonable(getattr(state, "completed_tasks", [])),
        "task_graph": to_jsonable(
            task_graph_to_payload(getattr(state, "task_graph", None))
        ),
        "task_graph_interceptions": int(
            getattr(state, "task_graph_interceptions", 0) or 0
        ),
        "capability_ledger": to_jsonable(_capability_ledger_payload(state)),
        "task_ledger": to_jsonable(_task_ledger_payload(state)),
        "resume_integrity": _resume_integrity_payload(state),
        "budget": to_jsonable(getattr(state, "budget", None)),
        "timeline": to_jsonable(getattr(state, "timeline", [])),
        "messages": to_jsonable(getattr(state, "messages", [])),
        "metadata": to_jsonable(metadata),
    }


def _event_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": snapshot["updated_at"],
        "stage": snapshot["stage"],
        "run_id": snapshot.get("run_id", snapshot.get("trace_id", "")),
        "trace_id": snapshot["trace_id"],
        "session_key": snapshot["session_key"],
        "status": snapshot.get("status", ""),
        "paused_reason": snapshot.get("paused_reason", ""),
        "waiting_approval_ids": snapshot.get("waiting_approval_ids", []),
        "background_task_ids": snapshot.get("background_task_ids", []),
        "observation_event_ids": snapshot.get("observation_event_ids", []),
        "step": snapshot["step"],
        "stop_reason": snapshot["stop_reason"],
        "recovery_action": snapshot["recovery_action"],
        "tool_call_count": len(snapshot.get("tool_calls", [])),
        "observation_count": len(snapshot.get("observations", [])),
        "pending_task_count": len(snapshot.get("pending_tasks", [])),
        "completed_task_count": len(snapshot.get("completed_tasks", [])),
        "task_graph_status": (snapshot.get("task_graph", {}) or {}).get("status", "")
        if isinstance(snapshot.get("task_graph"), dict)
        else "",
        "task_ledger_status": _compact_task_ledger_status(snapshot.get("task_ledger")),
        "agent_complexity_mode": snapshot.get("agent_complexity_mode", ""),
        "metadata": snapshot.get("metadata", {}),
    }


def _emit_agent_run_runtime_event(snapshot: dict[str, Any]) -> None:
    emit_runtime_event(
        kind="agent_run",
        status=_runtime_status_from_snapshot(snapshot),
        source=f"agent_run_store:{snapshot.get('stage', '')}",
        run_id=str(snapshot.get("run_id", "") or snapshot.get("trace_id", "")),
        trace_id=str(snapshot.get("trace_id", "") or ""),
        session_key=str(snapshot.get("session_key", "") or ""),
        step=int(snapshot.get("step", 0) or 0),
        summary=str(snapshot.get("stage", "") or snapshot.get("status", "")),
        payload={
            "stage": snapshot.get("stage", ""),
            "status": snapshot.get("status", ""),
            "paused_reason": snapshot.get("paused_reason", ""),
            "stop_reason": snapshot.get("stop_reason", ""),
            "recovery_action": snapshot.get("recovery_action", ""),
            "waiting_approval_ids": snapshot.get("waiting_approval_ids", []),
            "background_task_ids": snapshot.get("background_task_ids", []),
            "observation_event_ids": snapshot.get("observation_event_ids", []),
            "artifact_refs": snapshot.get("artifact_refs", []),
            "tool_call_count": len(snapshot.get("tool_calls", []) or []),
            "observation_count": len(snapshot.get("observations", []) or []),
            "pending_task_count": len(snapshot.get("pending_tasks", []) or []),
            "completed_task_count": len(snapshot.get("completed_tasks", []) or []),
            "metadata": snapshot.get("metadata", {}),
        },
        related_ids={
            "approval_id": _last_text(snapshot.get("waiting_approval_ids")),
            "background_task_id": _last_text(snapshot.get("background_task_ids")),
            "observation_event_id": _last_text(snapshot.get("observation_event_ids")),
            "artifact_id": _last_text(snapshot.get("artifact_refs")),
        },
    )


def _runtime_status_from_snapshot(snapshot: dict[str, Any]) -> RuntimeEventStatus:
    status = str(snapshot.get("status", "") or "")
    stage = str(snapshot.get("stage", "") or "")
    if status == "paused":
        return cast(RuntimeEventStatus, "waiting")
    if status in {"completed", "failed", "cancelled"}:
        return cast(RuntimeEventStatus, status)
    if stage in {"started", "step_started", "model_request"}:
        return cast(RuntimeEventStatus, "started")
    if stage in {"guardrail"}:
        return cast(RuntimeEventStatus, "blocked")
    return cast(RuntimeEventStatus, "progress")


def _last_text(value: Any) -> str:
    if isinstance(value, list | tuple) and value:
        return str(value[-1] or "")
    return ""


def _run_snapshot_path(trace_id: str):
    safe_trace = _safe_trace_id(trace_id)
    return state_path("agent_runs", f"{safe_trace or 'unknown'}.json")


def _run_events_path():
    return state_path("agent_runs", "events.jsonl")


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
        "waiting_approval_ids": payload.get("waiting_approval_ids", [])[:10]
        if isinstance(payload.get("waiting_approval_ids"), list)
        else [],
        "background_task_ids": payload.get("background_task_ids", [])[:10]
        if isinstance(payload.get("background_task_ids"), list)
        else [],
        "artifact_refs": payload.get("artifact_refs", [])[:10]
        if isinstance(payload.get("artifact_refs"), list)
        else [],
        "observation_event_ids": payload.get("observation_event_ids", [])[:10]
        if isinstance(payload.get("observation_event_ids"), list)
        else [],
        "step": payload.get("step", 0),
        "max_steps": payload.get("max_steps", 0),
        "stop_reason": payload.get("stop_reason", ""),
        "recovery_action": payload.get("recovery_action", ""),
        "tool_names": payload.get("tool_names", [])[:40]
        if isinstance(payload.get("tool_names"), list)
        else [],
        "tool_call_count": len(payload.get("tool_calls", []) or []),
        "observation_count": len(payload.get("observations", []) or []),
        "agent_complexity_mode": payload.get("agent_complexity_mode", ""),
        "agent_complexity_reason": payload.get("agent_complexity_reason", ""),
        "pending_task_count": len(payload.get("pending_tasks", []) or []),
        "completed_task_count": len(payload.get("completed_tasks", []) or []),
        "task_graph": _compact_task_graph(payload.get("task_graph")),
        "task_ledger": _compact_task_ledger(payload.get("task_ledger")),
        "resume_integrity": payload.get("resume_integrity", {}),
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
            resume_cursor=dict(snapshot.get("resume_cursor") or {}),
            waiting_approval_ids=_text_list(snapshot.get("waiting_approval_ids")),
            background_task_ids=_text_list(snapshot.get("background_task_ids")),
            observation_event_ids=_text_list(snapshot.get("observation_event_ids")),
            artifact_refs=_text_list(snapshot.get("artifact_refs")),
            tool_calls=_tool_calls_from_payload(snapshot.get("tool_calls", [])),
            observations=_observations_from_payload(snapshot.get("observations", [])),
            pending_tasks=_pending_tasks_from_payload(
                snapshot.get("pending_tasks", [])
            ),
            completed_tasks=_completed_tasks_from_payload(
                snapshot.get("completed_tasks", [])
            ),
            task_graph=task_graph_from_payload(snapshot.get("task_graph")),
            task_graph_interceptions=int(
                snapshot.get("task_graph_interceptions", 0) or 0
            ),
            capability_ledger=capability_ledger_from_payload(
                snapshot.get("capability_ledger")
            ),
            task_ledger=task_ledger_from_payload(snapshot.get("task_ledger")),
            stop_reason=str(snapshot.get("stop_reason", "") or "running"),
            recovery_action=str(snapshot.get("recovery_action", "") or "") or None,
            step=int(snapshot.get("step", 0) or 0),
            max_steps=int(snapshot.get("max_steps", 5) or 5),
            budget=_budget_from_payload(snapshot.get("budget", {})),
            timeline=_timeline_from_payload(snapshot.get("timeline", [])),
            final_text=str(snapshot.get("final_text", "") or ""),
            tool_obligation=str(snapshot.get("tool_obligation", "") or "auto"),
            tool_obligation_reason=str(
                snapshot.get("tool_obligation_reason", "") or "resumed_run"
            ),
            agent_complexity_mode=str(
                snapshot.get("agent_complexity_mode", "") or "standard"
            ),
            agent_complexity_reason=str(
                snapshot.get("agent_complexity_reason", "") or ""
            ),
            required_tool_names=tuple(_text_list(snapshot.get("required_tool_names"))),
        )
    except Exception:
        return None
    return state


def _merge_projection_refs(
    state: AgentRunState,
    projection: dict[str, Any],
) -> None:
    if not isinstance(projection, dict):
        return
    for attr, key in (
        ("waiting_approval_ids", "waiting_approval_ids"),
        ("background_task_ids", "background_task_ids"),
        ("observation_event_ids", "observation_event_ids"),
        ("artifact_refs", "artifact_refs"),
    ):
        target = getattr(state, attr, None)
        if not isinstance(target, list):
            continue
        for item in projection.get(key, []) or []:
            text = str(item or "")
            if text and text not in target:
                target.append(text)
    projected_status = str(projection.get("status", "") or "")
    if projected_status in {"paused", "cancelled", "completed", "failed"}:
        state.status = projected_status
    paused_reason = str(projection.get("paused_reason", "") or "")
    if paused_reason:
        state.paused_reason = paused_reason
    stop_reason = str(projection.get("stop_reason", "") or "")
    if stop_reason:
        state.stop_reason = stop_reason
    recovery_action = str(projection.get("recovery_action", "") or "")
    if recovery_action:
        state.recovery_action = recovery_action


def _messages_from_payload(value: Any) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    if not isinstance(value, list | tuple):
        return messages
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            messages.append(LLMMessage(**item))
        except Exception:
            continue
    return messages


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


def _observations_from_payload(value: Any) -> list[AgentObservation]:
    rows: list[AgentObservation] = []
    if not isinstance(value, list | tuple):
        return rows
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            result_payload = item.get("result")
            result = (
                ToolResult(**result_payload)
                if isinstance(result_payload, dict)
                else None
            )
            rows.append(
                AgentObservation(
                    tool_call_id=str(item.get("tool_call_id", "") or ""),
                    tool_name=str(item.get("tool_name", "") or ""),
                    command_id=str(item.get("command_id", "") or ""),
                    rendered_command=str(item.get("rendered_command", "") or ""),
                    matched_plugin=str(item.get("matched_plugin", "") or ""),
                    task_text=str(item.get("task_text", "") or ""),
                    ok=bool(item.get("ok")),
                    need_continue=bool(item.get("need_continue")),
                    remaining_task_hint=str(item.get("remaining_task_hint", "") or ""),
                    error=str(item.get("error", "") or ""),
                    artifacts=tuple(
                        dict(artifact)
                        for artifact in item.get("artifacts", []) or []
                        if isinstance(artifact, dict)
                    ),
                    step=int(item.get("step", 0) or 0),
                    result=result,
                    output=dict(item.get("output") or {}),
                )
            )
        except Exception:
            continue
    return rows


def _pending_tasks_from_payload(value: Any) -> list[PendingTask]:
    rows: list[PendingTask] = []
    if not isinstance(value, list | tuple):
        return rows
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(
                PendingTask(
                    text=str(item.get("text", "") or ""),
                    source=str(item.get("source", "") or "snapshot"),
                    command_id=str(item.get("command_id", "") or ""),
                    step=int(item.get("step", 0) or 0),
                )
            )
        except Exception:
            continue
    return rows


def _completed_tasks_from_payload(value: Any) -> list[CompletedTask]:
    rows: list[CompletedTask] = []
    if not isinstance(value, list | tuple):
        return rows
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(
                CompletedTask(
                    text=str(item.get("text", "") or ""),
                    command_id=str(item.get("command_id", "") or ""),
                    rendered_command=str(item.get("rendered_command", "") or ""),
                    matched_plugin=str(item.get("matched_plugin", "") or ""),
                    ok=bool(item.get("ok", True)),
                    step=int(item.get("step", 0) or 0),
                )
            )
        except Exception:
            continue
    return rows


def _timeline_from_payload(value: Any) -> list[AgentRuntimeTimelineItem]:
    rows: list[AgentRuntimeTimelineItem] = []
    if not isinstance(value, list | tuple):
        return rows
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(
                AgentRuntimeTimelineItem(
                    role=str(item.get("role", "") or ""),
                    kind=str(item.get("kind", "") or ""),
                    content=str(item.get("content", "") or ""),
                    tool_name=str(item.get("tool_name", "") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        except Exception:
            continue
    return rows


def _budget_from_payload(value: Any) -> AgentBudgetState:
    if not isinstance(value, dict):
        return AgentBudgetState()
    return AgentBudgetState(
        classifier_calls=int(value.get("classifier_calls", 0) or 0),
        hook_calls=int(value.get("hook_calls", 0) or 0),
        tool_calls=int(value.get("tool_calls", 0) or 0),
        tool_batches=int(value.get("tool_batches", 0) or 0),
        prompt_tokens=int(value.get("prompt_tokens", 0) or 0),
        durations_ms=dict(value.get("durations_ms") or {}),
    )


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if str(item or "")]


def _compact_task_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    tasks = value.get("tasks", []) if isinstance(value.get("tasks"), list) else []
    return {
        "graph_id": value.get("graph_id", ""),
        "status": value.get("status", ""),
        "reason": str(value.get("reason", "") or "")[:300],
        "tasks": [
            {
                "task_id": item.get("task_id", ""),
                "goal": str(item.get("goal", "") or "")[:300],
                "status": item.get("status", ""),
                "reason": str(item.get("reason", "") or "")[:300],
                "artifact_count": len(item.get("artifacts", []) or []),
                "observation_count": len(item.get("observations", []) or []),
            }
            for item in tasks[:12]
            if isinstance(item, dict)
        ],
    }


def _compact_task_ledger_status(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    tasks = value.get("tasks", []) if isinstance(value.get("tasks"), list) else []
    if not tasks:
        return "empty"
    statuses = [
        str(item.get("status", "") or "") for item in tasks if isinstance(item, dict)
    ]
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    if any(status == "pending" for status in statuses):
        return "in_progress"
    return "partial"


def _compact_task_ledger(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    tasks = value.get("tasks", []) if isinstance(value.get("tasks"), list) else []
    return {
        "ledger_id": value.get("ledger_id", ""),
        "reason": str(value.get("reason", "") or "")[:300],
        "status": _compact_task_ledger_status(value),
        "tasks": [
            {
                "task_id": item.get("task_id", ""),
                "goal": str(item.get("goal", "") or "")[:300],
                "status": item.get("status", ""),
                "requires_real_tool": bool(item.get("requires_real_tool")),
                "covered_by": item.get("covered_by", [])[:6]
                if isinstance(item.get("covered_by"), list)
                else [],
            }
            for item in tasks[:12]
            if isinstance(item, dict)
        ],
    }


def _resume_integrity_payload(state: Any) -> dict[str, Any]:
    task_graph = getattr(state, "task_graph", None)
    task_ledger = getattr(state, "task_ledger", None)
    waiting_approval_ids = list(getattr(state, "waiting_approval_ids", []) or [])
    background_task_ids = list(getattr(state, "background_task_ids", []) or [])
    observation_event_ids = list(getattr(state, "observation_event_ids", []) or [])
    return {
        "can_resume_original_context": True,
        "has_task_graph": task_graph is not None,
        "task_graph_status": getattr(task_graph, "status", "")
        if task_graph is not None
        else "",
        "task_graph_incomplete_count": len(
            getattr(task_graph, "incomplete_tasks", []) or []
        )
        if task_graph is not None
        else 0,
        "has_task_ledger": task_ledger is not None,
        "task_ledger_incomplete_count": len(
            getattr(task_ledger, "incomplete_goals", []) or []
        )
        if task_ledger is not None
        else 0,
        "waiting_approval_count": len(waiting_approval_ids),
        "background_task_count": len(background_task_ids),
        "last_observation_event_id": observation_event_ids[-1]
        if observation_event_ids
        else "",
    }


def _capability_ledger_payload(state: Any) -> dict[str, Any] | None:
    ledger = getattr(state, "capability_ledger", None)
    if ledger is None:
        return None
    return ledger.to_payload()


def _task_ledger_payload(state: Any) -> dict[str, Any] | None:
    ledger = getattr(state, "task_ledger", None)
    if ledger is None:
        return None
    return ledger.to_payload()


__all__ = [
    "get_agent_run_snapshot",
    "list_agent_run_snapshots",
    "load_agent_run_state",
    "persist_agent_run_state",
    "project_agent_run_state",
    "query_agent_run_events",
    "update_agent_run_status",
]
