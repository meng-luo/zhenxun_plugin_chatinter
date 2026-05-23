"""Layered evaluation summaries for real ChatInter trajectories.

TrajectoryStore records raw runs.  This module turns those runs into durable,
queryable scoreboards so architectural changes can be judged by hit rate,
false triggers, multi-task coverage, cost and latency instead of anecdotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .persistence import read_json, state_path, to_jsonable, utc_now_iso, write_json
from .route_text import normalize_message_text

EVAL_SCHEMA_VERSION = "chatinter.trajectory_eval.v1"
_MAX_RECENT_FAILURES = 50
_MAX_RECENT_RECORDS = 200
_MAX_PLUGIN_ROWS = 200


@dataclass
class MetricBucket:
    """Incremental aggregate for one trajectory layer."""

    total: int = 0
    hit_known: int = 0
    hit: int = 0
    failed: int = 0
    false_trigger: int = 0
    required_total: int = 0
    required_hit: int = 0
    optional_total: int = 0
    optional_hit: int = 0
    none_total: int = 0
    none_clean: int = 0
    multi_known: int = 0
    multi_covered: int = 0
    superuser_total: int = 0
    superuser_completed: int = 0
    superuser_paused: int = 0
    token_prompt_total: int = 0
    latency_total_ms: int = 0
    latency_max_ms: int = 0
    steps_total: int = 0
    tool_calls_total: int = 0
    observation_total: int = 0

    def add(self, metrics: dict[str, Any]) -> None:
        self.total += 1
        if metrics["hit"] is not None:
            self.hit_known += 1
            if metrics["hit"]:
                self.hit += 1
            else:
                self.failed += 1
        if metrics["false_trigger"]:
            self.false_trigger += 1
        obligation = metrics["tool_obligation"]
        if obligation == "required":
            self.required_total += 1
            if metrics["hit"]:
                self.required_hit += 1
        elif obligation == "auto":
            self.optional_total += 1
            if metrics["hit"]:
                self.optional_hit += 1
        elif obligation == "none":
            self.none_total += 1
            if not metrics["false_trigger"]:
                self.none_clean += 1
        if metrics["multi_task_covered"] is not None:
            self.multi_known += 1
            if metrics["multi_task_covered"]:
                self.multi_covered += 1
        if metrics["scenario"] == "superuser_agent":
            self.superuser_total += 1
            if metrics["status"] == "completed":
                self.superuser_completed += 1
            elif metrics["status"] == "paused":
                self.superuser_paused += 1
        self.token_prompt_total += metrics["prompt_tokens"]
        self.latency_total_ms += metrics["latency_ms"]
        self.latency_max_ms = max(self.latency_max_ms, metrics["latency_ms"])
        self.steps_total += metrics["steps"]
        self.tool_calls_total += metrics["tool_call_count"]
        self.observation_total += metrics["observation_count"]

    def to_payload(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "hit_known": self.hit_known,
            "hit": self.hit,
            "failed": self.failed,
            "hit_rate": _rate(self.hit, self.hit_known),
            "false_trigger": self.false_trigger,
            "false_trigger_rate": _rate(self.false_trigger, self.total),
            "required_total": self.required_total,
            "required_hit_rate": _rate(self.required_hit, self.required_total),
            "optional_total": self.optional_total,
            "optional_hit_rate": _rate(self.optional_hit, self.optional_total),
            "none_total": self.none_total,
            "none_clean_rate": _rate(self.none_clean, self.none_total),
            "multi_known": self.multi_known,
            "multi_coverage_rate": _rate(self.multi_covered, self.multi_known),
            "superuser_total": self.superuser_total,
            "superuser_completion_rate": _rate(
                self.superuser_completed,
                self.superuser_total,
            ),
            "superuser_pause_rate": _rate(self.superuser_paused, self.superuser_total),
            "avg_prompt_tokens": _avg(self.token_prompt_total, self.total),
            "avg_latency_ms": _avg(self.latency_total_ms, self.total),
            "max_latency_ms": self.latency_max_ms,
            "avg_steps": _avg(self.steps_total, self.total),
            "avg_tool_calls": _avg(self.tool_calls_total, self.total),
            "avg_observations": _avg(self.observation_total, self.total),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "MetricBucket":
        if not isinstance(payload, dict):
            return cls()
        bucket = cls()
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            try:
                setattr(bucket, field_name, int(payload.get(field_name, 0) or 0))
            except Exception:
                setattr(bucket, field_name, 0)
        return bucket


@dataclass
class TrajectoryEvalState:
    schema_version: str = EVAL_SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    total: MetricBucket = field(default_factory=MetricBucket)
    by_scenario: dict[str, MetricBucket] = field(default_factory=dict)
    by_layer: dict[str, MetricBucket] = field(default_factory=dict)
    by_obligation: dict[str, MetricBucket] = field(default_factory=dict)
    by_agent_mode: dict[str, MetricBucket] = field(default_factory=dict)
    by_plugin: dict[str, MetricBucket] = field(default_factory=dict)
    recent_failures: list[dict[str, Any]] = field(default_factory=list)
    recent_records: list[dict[str, Any]] = field(default_factory=list)

    def add_record(self, record: dict[str, Any]) -> None:
        metrics = trajectory_metrics(record)
        self.updated_at = utc_now_iso()
        self.total.add(metrics)
        _bucket(self.by_scenario, metrics["scenario"]).add(metrics)
        _bucket(self.by_layer, metrics["layer"]).add(metrics)
        _bucket(self.by_obligation, metrics["tool_obligation"]).add(metrics)
        _bucket(self.by_agent_mode, metrics["agent_mode"] or "default").add(metrics)
        for plugin in metrics["plugins"]:
            _bucket(self.by_plugin, plugin).add(metrics)
        self.by_plugin = _trim_bucket_map(self.by_plugin, limit=_MAX_PLUGIN_ROWS)
        compact = _compact_record(record, metrics)
        self.recent_records.append(compact)
        self.recent_records = self.recent_records[-_MAX_RECENT_RECORDS:]
        if _is_failure(metrics):
            self.recent_failures.append(compact)
            self.recent_failures = self.recent_failures[-_MAX_RECENT_FAILURES:]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total": _bucket_record(self.total),
            "by_scenario": _bucket_map_record(self.by_scenario),
            "by_layer": _bucket_map_record(self.by_layer),
            "by_obligation": _bucket_map_record(self.by_obligation),
            "by_agent_mode": _bucket_map_record(self.by_agent_mode),
            "by_plugin": _bucket_map_record(self.by_plugin),
            "recent_failures": list(self.recent_failures),
            "recent_records": list(self.recent_records),
        }

    @classmethod
    def from_record(cls, payload: Any) -> "TrajectoryEvalState":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            schema_version=normalize_message_text(
                str(payload.get("schema_version") or EVAL_SCHEMA_VERSION)
            )
            or EVAL_SCHEMA_VERSION,
            created_at=normalize_message_text(str(payload.get("created_at") or ""))
            or utc_now_iso(),
            updated_at=normalize_message_text(str(payload.get("updated_at") or ""))
            or utc_now_iso(),
            total=MetricBucket.from_payload(
                _raw_bucket_payload(payload.get("total"))
            ),
            by_scenario=_bucket_map_from_payload(payload.get("by_scenario")),
            by_layer=_bucket_map_from_payload(payload.get("by_layer")),
            by_obligation=_bucket_map_from_payload(payload.get("by_obligation")),
            by_agent_mode=_bucket_map_from_payload(payload.get("by_agent_mode")),
            by_plugin=_bucket_map_from_payload(payload.get("by_plugin")),
            recent_failures=_dict_list(payload.get("recent_failures"))[
                -_MAX_RECENT_FAILURES:
            ],
            recent_records=_dict_list(payload.get("recent_records"))[
                -_MAX_RECENT_RECORDS:
            ],
        )


def record_trajectory_eval(record: dict[str, Any]) -> Path:
    """Update daily and all-time eval summaries for one real trajectory."""

    day = _record_day(record)
    daily_path = trajectory_eval_path(day=day)
    all_time_path = trajectory_eval_path(day="all")
    latest_path = state_path("trajectory_eval", "latest.json")

    daily = load_trajectory_eval(day=day)
    daily.add_record(record)
    write_json(daily_path, daily.to_record())

    all_time = load_trajectory_eval(day="all")
    all_time.add_record(record)
    write_json(all_time_path, all_time.to_record())

    latest_payload = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "updated_at": utc_now_iso(),
        "daily_path": str(daily_path),
        "all_time_path": str(all_time_path),
        "daily": daily.to_record(),
        "all_time": all_time.to_record(),
    }
    write_json(latest_path, latest_payload)
    return daily_path


def load_trajectory_eval(*, day: str = "all") -> TrajectoryEvalState:
    return TrajectoryEvalState.from_record(read_json(trajectory_eval_path(day=day), {}))


def trajectory_eval_path(*, day: str = "all") -> Path:
    normalized = normalize_message_text(day or "all") or "all"
    return state_path("trajectory_eval", f"{normalized}.json")


def latest_trajectory_eval() -> dict[str, Any]:
    payload = read_json(state_path("trajectory_eval", "latest.json"), {})
    return payload if isinstance(payload, dict) else {}


def trajectory_metrics(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
    latency = record.get("latency") if isinstance(record.get("latency"), dict) else {}
    token = record.get("token") if isinstance(record.get("token"), dict) else {}
    scenario = normalize_message_text(str(record.get("scenario") or "unknown"))
    obligation = normalize_message_text(str(record.get("tool_obligation") or "none"))
    selected_tools = _text_list(record.get("selected_tools"))
    observations = _dict_list(record.get("observations"))
    task_ledger = record.get("task_ledger") if isinstance(record.get("task_ledger"), dict) else {}
    metrics = {
        "trace_id": normalize_message_text(str(record.get("trace_id") or "")),
        "run_id": normalize_message_text(str(record.get("run_id") or "")),
        "scenario": scenario or "unknown",
        "layer": _layer_for_record(record, scenario=scenario),
        "agent_mode": normalize_message_text(str(record.get("agent_mode") or "")),
        "status": normalize_message_text(str(record.get("status") or "")),
        "tool_obligation": obligation if obligation in {"none", "auto", "required"} else "none",
        "hit": _optional_bool(record.get("hit", evaluation.get("hit"))),
        "false_trigger": bool(record.get("false_trigger", evaluation.get("false_trigger", False))),
        "multi_task_covered": _optional_bool(
            record.get("multi_task_covered", evaluation.get("multi_task_covered"))
        ),
        "latency_ms": _int(latency.get("total_ms")),
        "prompt_tokens": _int(token.get("prompt_estimated")),
        "steps": _int(record.get("steps")),
        "tool_call_count": _int(evaluation.get("tool_call_count"), fallback=len(selected_tools)),
        "observation_count": _int(evaluation.get("observation_count"), fallback=len(observations)),
        "selected_tools": selected_tools,
        "plugins": _plugins_for_record(record, observations),
        "task_count": _task_count(task_ledger),
    }
    return metrics


def _layer_for_record(record: dict[str, Any], *, scenario: str) -> str:
    if scenario == "superuser_agent":
        return "superuser_long_task"
    obligation = normalize_message_text(str(record.get("tool_obligation") or "none"))
    selected_tools = _text_list(record.get("selected_tools"))
    task_count = _task_count(
        record.get("task_ledger") if isinstance(record.get("task_ledger"), dict) else {}
    )
    if task_count > 1:
        return "multi_tool"
    if scenario in {"group_plugin_selector", "tool_run"}:
        return "real_tool_required" if obligation == "required" else "plugin_optional"
    if selected_tools and obligation == "none":
        return "chat_false_trigger"
    if not selected_tools and obligation == "none":
        return "direct_chat"
    return scenario or "unknown"


def _plugins_for_record(
    record: dict[str, Any],
    observations: list[dict[str, Any]],
) -> list[str]:
    plugins: list[str] = []
    for observation in observations:
        plugin = normalize_message_text(str(observation.get("matched_plugin") or ""))
        if plugin and plugin not in plugins:
            plugins.append(plugin)
    for item in _dict_list(record.get("exposed_tools")):
        plugin = normalize_message_text(
            str(item.get("plugin_module") or item.get("plugin_name") or "")
        )
        if plugin and plugin not in plugins:
            plugins.append(plugin)
    return plugins[:16] or ["none"]


def _compact_record(record: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": normalize_message_text(str(record.get("created_at") or "")),
        "trace_id": metrics["trace_id"],
        "run_id": metrics["run_id"],
        "scenario": metrics["scenario"],
        "layer": metrics["layer"],
        "status": metrics["status"],
        "tool_obligation": metrics["tool_obligation"],
        "hit": metrics["hit"],
        "false_trigger": metrics["false_trigger"],
        "multi_task_covered": metrics["multi_task_covered"],
        "latency_ms": metrics["latency_ms"],
        "prompt_tokens": metrics["prompt_tokens"],
        "steps": metrics["steps"],
        "tool_call_count": metrics["tool_call_count"],
        "observation_count": metrics["observation_count"],
        "selected_tools": metrics["selected_tools"][:8],
        "plugins": metrics["plugins"][:8],
        "input_message": _clip(str(record.get("input_message") or ""), limit=180),
        "stop_reason": normalize_message_text(str(record.get("stop_reason") or "")),
        "paused_reason": normalize_message_text(str(record.get("paused_reason") or "")),
    }


def _is_failure(metrics: dict[str, Any]) -> bool:
    return (
        metrics["hit"] is False
        or bool(metrics["false_trigger"])
        or metrics["multi_task_covered"] is False
        or metrics["status"] == "failed"
    )


def _record_day(record: dict[str, Any]) -> str:
    value = normalize_message_text(str(record.get("created_at") or ""))
    if len(value) >= 10:
        return value[:10]
    return datetime.now(timezone.utc).date().isoformat()


def _bucket(target: dict[str, MetricBucket], key: str) -> MetricBucket:
    normalized = normalize_message_text(key) or "unknown"
    if normalized not in target:
        target[normalized] = MetricBucket()
    return target[normalized]


def _bucket_record(bucket: MetricBucket) -> dict[str, Any]:
    payload = to_jsonable(bucket)
    derived = bucket.to_payload()
    if isinstance(payload, dict):
        payload["derived"] = derived
        return payload
    return {"derived": derived}


def _bucket_map_record(mapping: dict[str, MetricBucket]) -> dict[str, Any]:
    return {
        key: _bucket_record(bucket)
        for key, bucket in sorted(
            mapping.items(),
            key=lambda item: (-item[1].total, item[0]),
        )
    }


def _bucket_map_from_payload(payload: Any) -> dict[str, MetricBucket]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, MetricBucket] = {}
    for key, value in payload.items():
        normalized = normalize_message_text(str(key or ""))
        if not normalized:
            continue
        result[normalized] = MetricBucket.from_payload(_raw_bucket_payload(value))
    return result


def _raw_bucket_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if key != "derived"}


def _trim_bucket_map(
    mapping: dict[str, MetricBucket],
    *,
    limit: int,
) -> dict[str, MetricBucket]:
    rows = sorted(mapping.items(), key=lambda item: (-item[1].total, item[0]))
    return dict(rows[: max(1, int(limit or 1))])


def _task_count(task_ledger: dict[str, Any]) -> int:
    tasks = task_ledger.get("tasks") if isinstance(task_ledger, dict) else None
    return len(tasks) if isinstance(tasks, list) else 0


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[str] = []
    for item in value:
        text = normalize_message_text(str(item or ""))
        if text:
            result.append(text)
    return result


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = normalize_message_text(str(value)).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _int(value: Any, *, fallback: int = 0) -> int:
    try:
        return max(int(float(value or 0)), 0)
    except (TypeError, ValueError):
        return max(int(fallback or 0), 0)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _avg(total: int, count: int) -> float | None:
    if count <= 0:
        return None
    return round(float(total) / float(count), 2)


def _clip(value: str, *, limit: int) -> str:
    text = normalize_message_text(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


__all__ = [
    "EVAL_SCHEMA_VERSION",
    "MetricBucket",
    "TrajectoryEvalState",
    "latest_trajectory_eval",
    "load_trajectory_eval",
    "record_trajectory_eval",
    "trajectory_eval_path",
    "trajectory_metrics",
]
