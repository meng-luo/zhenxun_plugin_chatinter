"""Durable AgentRun snapshots for ChatInter.

Agent runs are short-lived in normal group chat, but superuser private turns can
span long tool loops.  Persisting every transition gives us a restart-safe audit
trail and a recovery surface for future resumable runs.
"""

from __future__ import annotations

from typing import Any

from .persistence import append_jsonl, state_path, to_jsonable, utc_now_iso, write_json


def persist_agent_run_state(
    state: Any,
    *,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        payload = _state_payload(state, stage=stage, metadata=metadata or {})
        write_json(_run_snapshot_path(str(state.trace_id)), payload)
        append_jsonl(_run_events_path(), _event_payload(payload))
    except Exception:
        # Persistence must never break a user turn or tool execution.
        return


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
        "trace_id": str(getattr(state, "trace_id", "") or ""),
        "session_key": str(getattr(state, "session_key", "") or ""),
        "step": int(getattr(state, "step", 0) or 0),
        "max_steps": int(getattr(state, "max_steps", 0) or 0),
        "stop_reason": str(getattr(state, "stop_reason", "") or ""),
        "recovery_action": str(getattr(state, "recovery_action", "") or ""),
        "final_text": str(getattr(state, "final_text", "") or ""),
        "tool_names": sorted(str(name) for name in getattr(state, "tool_map", {})),
        "tool_calls": to_jsonable(getattr(state, "tool_calls", [])),
        "observations": to_jsonable(getattr(state, "observations", [])),
        "pending_tasks": to_jsonable(getattr(state, "pending_tasks", [])),
        "completed_tasks": to_jsonable(getattr(state, "completed_tasks", [])),
        "budget": to_jsonable(getattr(state, "budget", None)),
        "timeline": to_jsonable(getattr(state, "timeline", [])),
        "messages": to_jsonable(getattr(state, "messages", [])),
        "metadata": to_jsonable(metadata),
    }


def _event_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": snapshot["updated_at"],
        "stage": snapshot["stage"],
        "trace_id": snapshot["trace_id"],
        "session_key": snapshot["session_key"],
        "step": snapshot["step"],
        "stop_reason": snapshot["stop_reason"],
        "recovery_action": snapshot["recovery_action"],
        "tool_call_count": len(snapshot.get("tool_calls", [])),
        "observation_count": len(snapshot.get("observations", [])),
        "pending_task_count": len(snapshot.get("pending_tasks", [])),
        "completed_task_count": len(snapshot.get("completed_tasks", [])),
        "metadata": snapshot.get("metadata", {}),
    }


def _run_snapshot_path(trace_id: str):
    safe_trace = "".join(ch for ch in trace_id if ch.isalnum() or ch in {"-", "_"})
    return state_path("agent_runs", f"{safe_trace or 'unknown'}.json")


def _run_events_path():
    return state_path("agent_runs", "events.jsonl")


__all__ = ["persist_agent_run_state"]
