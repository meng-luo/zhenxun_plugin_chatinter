"""Durable trajectory records for ChatInter agent runs.

Trajectory records are intentionally observational.  They do not decide routing
or tool selection; they preserve enough evidence for later evals to explain
whether a turn was accurate, expensive, slow, or over-eager.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from ..persistence import (
    append_jsonl,
    read_json,
    state_path,
    to_jsonable,
    utc_now_iso,
    write_json,
)
from ..route_text import normalize_message_text
from .state import AgentObservation, AgentRunState, AgentRuntimeTimelineItem

SCHEMA_VERSION = "chatinter.trajectory.v1"
_MAX_TEXT = 600
_MAX_TOOLS = 180
_MAX_TIMELINE_ITEMS = 160
_MAX_OBSERVATIONS = 80
_MAX_ARGUMENT_TEXT = 1200


def record_agent_trajectory(
    *,
    state: AgentRunState,
    input_message: str,
    started_at: float,
    latency_ms: float,
    run_context_extra: dict[str, Any] | None = None,
    project: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Append one completed/paused/failed AgentRun trajectory to durable JSONL."""

    record = build_agent_trajectory_record(
        state=state,
        input_message=input_message,
        started_at=started_at,
        latency_ms=latency_ms,
        run_context_extra=run_context_extra,
    )
    path = trajectory_jsonl_path()
    append_jsonl(path, record)
    write_json(state_path("trajectories", "latest.json"), record)
    if project:
        _record_feedback_projection(record)
        _record_eval_projection(record)
    return path, record


def build_agent_trajectory_record(
    *,
    state: AgentRunState,
    input_message: str,
    started_at: float,
    latency_ms: float,
    run_context_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = dict(run_context_extra or {})
    selected_tools = _selected_tools(state.metrics)
    observations = [
        _observation_payload(item) for item in state.runtime_observations()
    ]
    evaluation = _evaluate_record_fields(
        selected_tools=selected_tools,
        observations=observations,
        status=state.status,
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": state.trace_id,
        "run_id": state.run_id,
        "eval_case_id": normalize_message_text(
            str(extra.get("eval_case_id", "") or "")
        ),
        "eval_layer": normalize_message_text(str(extra.get("eval_layer", "") or "")),
        "eval_expectation": normalize_message_text(
            str(extra.get("eval_expectation", "") or "")
        ),
        "session_key": state.session_key or "",
        "created_at": utc_now_iso(),
        "started_at": _iso_from_timestamp(started_at),
        "scenario": "superuser_agent",
        "agent_mode": normalize_message_text(str(extra.get("agent_mode", ""))),
        "input_message": _clip(input_message),
        "exposed_tools": _exposed_tool_payloads(state),
        "selected_tools": selected_tools,
        "tool_calls": _tool_call_payloads(state.metrics),
        "observations": observations,
        "final_reply": _clip(state.final_text),
        "delivery_complete": state.delivery_complete,
        "final_source": state.final_source,
        "token": {
            "prompt_estimated": int(state.budget.run_input_tokens or 0),
            "completion_tokens": int(state.budget.run_output_tokens or 0),
            "run_input_tokens": int(state.budget.run_input_tokens or 0),
            "run_output_tokens": int(state.budget.run_output_tokens or 0),
            "current_context_tokens": int(state.budget.current_context_tokens or 0),
            **_token_estimate_stats(state.metrics),
        },
        "latency": {
            "total_ms": max(int(float(latency_ms or 0)), 0),
            "stage_ms": dict(state.budget.durations_ms or {}),
        },
        "model": _model_request_stats(state.metrics, state=state),
        "compression": _compression_stats(state.metrics),
        "status": state.status,
        "paused_reason": state.paused_reason,
        "stop_reason": state.stop_reason,
        "steps": state.step,
        "budget": to_jsonable(state.budget),
        "cost_checkpoint_reached": state.cost_checkpoint_reached(),
        "hit": evaluation["hit"],
        "false_trigger": evaluation["false_trigger"],
        "evaluation": evaluation,
        "provider_capability": _json_dict(extra.get("provider_capability")),
        "timeline_digest": _timeline_digest(state.metrics),
        **_side_effect_stats(state),
    }
    return _drop_empty(record)


def trajectory_jsonl_path(day: str | None = None) -> Path:
    normalized_day = normalize_message_text(day or "")
    if not normalized_day:
        normalized_day = datetime.now(timezone.utc).date().isoformat()
    return state_path("trajectories", f"{normalized_day}.jsonl")


def load_trajectory_records(
    *,
    path: Path | str | None = None,
    limit: int = 500,
    scenario: str | None = None,
) -> list[dict[str, Any]]:
    """Load recent trajectory records from JSONL, newest last."""

    source = Path(path) if path is not None else trajectory_jsonl_path()
    if not source.exists():
        return []
    max_items = max(int(limit or 0), 1)
    scenario_filter = normalize_message_text(scenario or "")
    records: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if (
            scenario_filter
            and normalize_message_text(str(payload.get("scenario", "")))
            != scenario_filter
        ):
            continue
        records.append(payload)
    return records[-max_items:]


def latest_trajectory_record() -> dict[str, Any]:
    payload = read_json(state_path("trajectories", "latest.json"), {})
    return payload if isinstance(payload, dict) else {}


def _evaluate_record_fields(
    *,
    selected_tools: list[str],
    observations: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    actionable_observations = [
        item
        for item in observations
        if item.get("tool_name") and not item.get("synthetic")
    ]
    ok_observations = [item for item in actionable_observations if item.get("ok")]
    tool_selected = bool(selected_tools)
    hit: bool | None
    if status not in {"completed", "paused"}:
        hit = False
    elif tool_selected:
        hit = bool(ok_observations)
    else:
        hit = None

    return {
        "hit": hit,
        "false_trigger": False,
        "ok_observation_count": len(ok_observations),
        "tool_call_count": len(selected_tools),
        "observation_count": len(observations),
    }


def _selected_tools(timeline: list[AgentRuntimeTimelineItem]) -> list[str]:
    result: list[str] = []
    for item in timeline:
        if item.kind != "tool_call":
            continue
        name = normalize_message_text(item.tool_name)
        if name:
            result.append(name)
    return result


def _model_request_stats(
    timeline: list[AgentRuntimeTimelineItem],
    *,
    state: AgentRunState,
) -> dict[str, int]:
    requests = [item for item in timeline if item.kind == "model_request"]
    main_model_calls = len(requests)
    total_model_calls = max(int(state.budget.model_calls or 0), 0)
    return {
        "selected_tool_count": max(
            (
                int(item.metadata.get("selected_tool_count", 0) or 0)
                for item in requests
            ),
            default=0,
        ),
        "schema_chars": sum(
            int(item.metadata.get("schema_chars", 0) or 0) for item in requests
        ),
        "model_calls": main_model_calls,
        "main_model_calls": main_model_calls,
        "summary_model_calls": max(total_model_calls - main_model_calls, 0),
        "total_model_calls": total_model_calls,
    }


def _token_estimate_stats(
    timeline: list[AgentRuntimeTimelineItem],
) -> dict[str, int | float | str]:
    usage = [item for item in timeline if item.kind == "model_usage"]
    if not usage:
        return {}
    estimated = sum(
        max(int(item.metadata.get("estimated_prompt_tokens", 0) or 0), 0)
        for item in usage
    )
    provider = sum(
        max(int(item.metadata.get("provider_prompt_tokens", 0) or 0), 0)
        for item in usage
    )
    sources = {
        normalize_message_text(str(item.metadata.get("estimate_source", "") or ""))
        for item in usage
    }
    sources.discard("")
    source = next(iter(sources)) if len(sources) == 1 else "mixed"
    return {
        "estimated_prompt_tokens": estimated,
        "provider_prompt_tokens": provider,
        "estimate_ratio": round(estimated / provider, 4)
        if provider and sources == {"provider"}
        else None,
        "estimate_source": source or "local",
    }


def _side_effect_stats(state: AgentRunState) -> dict[str, int]:
    records = list(state.tool_executions)
    executed = [
        item
        for item in records
        if item.status in {"completed", "failed", "cancelled"}
    ]
    fingerprints = Counter(
        str(item.fingerprint) for item in executed if str(item.fingerprint or "")
    )
    return {
        "side_effect_requested": len(records),
        "side_effect_executed": len(executed),
        "side_effect_not_executed": sum(
            item.status == "not_executed" for item in records
        ),
        "side_effect_uncertain": sum(
            item.status in {"started", "uncertain"} for item in records
        ),
        "side_effect_duplicate_executions": sum(
            count - 1 for count in fingerprints.values() if count > 1
        ),
    }


def _compression_stats(
    timeline: list[AgentRuntimeTimelineItem],
) -> dict[str, int]:
    successful = [
        item
        for item in timeline
        if item.kind
        in {"semantic_context_compression", "context_tool_results_pruned"}
    ]
    semantic = [
        item for item in successful if item.kind == "semantic_context_compression"
    ]
    failed_attempts = sum(
        item.kind == "semantic_compression_failed" for item in timeline
    )
    return {
        "count": len(successful),
        "semantic_count": len(semantic),
        "token_savings": sum(
            max(
                int(item.metadata.get("before_tokens", 0) or 0)
                - int(item.metadata.get("after_tokens", 0) or 0),
                0,
            )
            for item in successful
        ),
        "pruned_tool_results": sum(
            max(int(item.metadata.get("pruned_tool_results", 0) or 0), 0)
            for item in successful
        ),
        "failure_count": failed_attempts,
        "compression_failed_attempts": failed_attempts,
        "low_savings_count": sum(
            bool(item.metadata.get("low_savings")) for item in semantic
        ),
        "summary_input_dropped_rounds": sum(
            max(
                int(item.metadata.get("summary_input_dropped_rounds", 0) or 0),
                0,
            )
            for item in semantic
        ),
    }


def _tool_call_payloads(
    timeline: list[AgentRuntimeTimelineItem],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in timeline:
        if item.kind != "tool_call":
            continue
        payloads.append(
            _drop_empty(
                {
                    "step": item.metadata.get("step"),
                    "tool_name": item.tool_name,
                    "arguments": _compact_value(item.metadata.get("arguments")),
                }
            )
        )
    return payloads[:_MAX_OBSERVATIONS]


def _observation_payload(observation: AgentObservation) -> dict[str, Any]:
    output = observation.output if isinstance(observation.output, dict) else {}
    payload = {
        "step": observation.step,
        "tool_name": observation.tool_name,
        "command_id": observation.command_id,
        "rendered_command": observation.rendered_command,
        "task_text": _clip(observation.task_text),
        "ok": observation.ok,
        "need_continue": observation.need_continue,
        "remaining_task_hint": _clip(observation.remaining_task_hint),
        "error": _clip(observation.error),
        "retryable": bool(output.get("retryable")),
        "status": normalize_message_text(str(output.get("status", ""))),
        "messages_sent_summary": _messages_sent_summary(output),
        "artifacts": _artifact_summary(output, observation.artifacts),
        "synthetic": _is_synthetic_observation(observation),
    }
    return _drop_empty(payload)


def _exposed_tool_payloads(state: AgentRunState) -> list[dict[str, str]]:
    return [{"tool_name": name} for name in sorted(state.tool_map)][:_MAX_TOOLS]


def _timeline_digest(
    timeline: list[AgentRuntimeTimelineItem],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in timeline[-_MAX_TIMELINE_ITEMS:]:
        result.append(
            _drop_empty(
                {
                    "role": item.role,
                    "kind": item.kind,
                    "tool_name": item.tool_name,
                    "content": _clip(item.content, limit=240),
                    "metadata": _compact_value(item.metadata, limit=600),
                }
            )
        )
    return result


def _messages_sent_summary(output: dict[str, Any]) -> list[str]:
    for key in (
        "messages_sent_summary",
        "messages_sent",
        "sent_messages",
        "visible_outputs",
    ):
        value = output.get(key)
        if isinstance(value, str):
            text = normalize_message_text(value)
            return [text[:240]] if text else []
        if isinstance(value, list | tuple):
            result: list[str] = []
            for item in value[:8]:
                if isinstance(item, dict):
                    text = normalize_message_text(
                        str(
                            item.get("summary")
                            or item.get("text")
                            or item.get("message")
                            or item.get("content")
                            or item
                        )
                    )
                else:
                    text = normalize_message_text(str(item or ""))
                if text:
                    result.append(text[:240])
            if result:
                return result
    return []


def _artifact_summary(
    output: dict[str, Any],
    artifacts: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    raw = list(artifacts)
    output_artifacts = output.get("artifacts")
    if isinstance(output_artifacts, list | tuple):
        raw.extend(item for item in output_artifacts if isinstance(item, dict))
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw[:16]:
        artifact_id = normalize_message_text(str(item.get("artifact_id", "") or ""))
        if artifact_id and artifact_id in seen:
            continue
        if artifact_id:
            seen.add(artifact_id)
        result.append(
            _drop_empty(
                {
                    "artifact_id": artifact_id,
                    "type": normalize_message_text(str(item.get("type", "") or "")),
                    "summary": _clip(str(item.get("summary", "") or ""), limit=200),
                    "path": normalize_message_text(str(item.get("path", "") or "")),
                }
            )
        )
    return result


def _is_synthetic_observation(observation: AgentObservation) -> bool:
    if observation.command_id or observation.rendered_command:
        return False
    status = normalize_message_text(str(observation.output.get("status", "")))
    return observation.tool_name.startswith("runtime_") or status.startswith(
        ("guardrail", "coverage", "validator")
    )


def _record_feedback_projection(record: dict[str, Any]) -> None:
    try:
        from ..feedback import record_trajectory_eval_feedback

        record_trajectory_eval_feedback(record)
    except Exception:
        return


def _record_eval_projection(record: dict[str, Any]) -> None:
    try:
        from ..trajectory_eval import record_trajectory_eval

        record_trajectory_eval(record)
    except Exception:
        return


def _iso_from_timestamp(value: float) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat()


def _json_dict(value: Any) -> dict[str, Any]:
    payload = to_jsonable(value)
    return payload if isinstance(payload, dict) else {}


def _compact_value(value: Any, *, limit: int = _MAX_ARGUMENT_TEXT) -> Any:
    payload = to_jsonable(value)
    if isinstance(payload, str):
        return _clip(payload, limit=limit)
    if isinstance(payload, dict):
        return {
            str(key): _compact_value(item, limit=max(160, limit // 2))
            for key, item in list(payload.items())[:24]
        }
    if isinstance(payload, list):
        return [
            _compact_value(item, limit=max(160, limit // 2)) for item in payload[:24]
        ]
    return payload


def _clip(value: Any, *, limit: int = _MAX_TEXT) -> str:
    text = normalize_message_text(str(value or ""))
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items() if value not in ("", [], {}, None)
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_agent_trajectory_record",
    "latest_trajectory_record",
    "load_trajectory_records",
    "record_agent_trajectory",
    "trajectory_jsonl_path",
]
