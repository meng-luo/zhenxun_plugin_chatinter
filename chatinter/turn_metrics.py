from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from zhenxun.services import logger

from .config import ROUTE_OBSERVER_MAX_RECORDS
from .native_route import NativeRouteReport
from .route_text import normalize_message_text
from .trace import StageTrace
from .turn_runtime import TurnBudgetController


@dataclass(frozen=True)
class TurnMetricsSnapshot:
    trace_name: str
    total_ms: float
    queue_wait_ms: float
    stage_ms: dict[str, float]
    tags: dict[str, str]
    route_reason: str | None
    route_attempts: int
    route_tool_attempts: int
    route_candidates: int
    route_tool_candidates: int
    route_tool_choices: int
    route_prompt_full_candidates: int
    runtime_budget: dict[str, object] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_turn_metrics_snapshot(
    *,
    trace: StageTrace,
    total_seconds: float,
    route_report: NativeRouteReport | None = None,
    budget_controller: TurnBudgetController | None = None,
) -> TurnMetricsSnapshot:
    runtime_budget = (
        asdict(budget_controller.snapshot()) if budget_controller is not None else None
    )
    return TurnMetricsSnapshot(
        trace_name=trace.name,
        total_ms=round(total_seconds * 1000, 2),
        queue_wait_ms=_queue_wait_ms(trace),
        stage_ms={
            name: round(cost * 1000, 2) for name, cost in getattr(trace, "_stages", ())
        },
        tags=dict(trace.tags),
        route_reason=(route_report.final_reason if route_report else None),
        route_attempts=(route_report.attempts if route_report else 0),
        route_tool_attempts=(route_report.tool_attempts if route_report else 0),
        route_candidates=(route_report.candidate_total if route_report else 0),
        route_tool_candidates=(route_report.tool_candidates if route_report else 0),
        route_tool_choices=(route_report.tool_choice_count if route_report else 0),
        route_prompt_full_candidates=(
            route_report.prompt_full_candidates if route_report else 0
        ),
        runtime_budget=runtime_budget,
    )


def _queue_wait_ms(trace: StageTrace) -> float:
    try:
        return round(max(float(trace.tags.get("queue_wait_ms", 0.0)), 0.0), 2)
    except (TypeError, ValueError):
        return 0.0


def emit_turn_metrics(snapshot: TurnMetricsSnapshot) -> None:
    logger.debug(f"ChatInter turn metrics: {snapshot.to_dict()}")


@dataclass(frozen=True)
class RouteObservation:
    timestamp: str
    trace_id: str
    user_id: str
    group_id: str
    message_preview: str
    path: str
    outcome: str
    route_stage: str
    route_plugin: str
    route_module: str
    route_head: str
    candidate_total: int
    lexical_candidates: int
    direct_candidates: int
    vector_candidates: int
    attempts: int
    tool_candidates: int
    tool_attempts: int
    tool_choice_count: int
    prompt_full_candidates: int
    final_reason: str
    failure_layer: str = ""
    plugin_outcome: str = ""
    exact_identity_ids: str = ""
    strict_identity_match_modes: str = ""
    exposed_command_ids: str = ""
    available_command_count: int = 0
    selected_command_ids: str = ""
    selected_skill: str = ""
    discovery_source: str = ""
    retrieval_query_count: int = 0
    discovery_candidate_count: int = 0
    candidate_displayed: int = 0
    candidate_omitted: int = 0
    candidate_exposure_count: int = 0
    selected_command_id: str = ""
    selected_capability_id: str = ""
    execution_validation_reason: str = ""
    identity_spans: str = ""
    person_candidate_count: int = 0
    candidate_sources: str = ""
    selected_target_ref: str = ""
    target_resolution_mode: str = ""
    target_validation_reason: str = ""
    self_identity_candidate: bool = False
    protocol_argument_retries: int = 0
    protocol_format_retries: int = 0
    protocol_text_only_retries: int = 0
    protocol_text_suppressed: int = 0
    tool_argument_envelope_repairs: int = 0
    protocol_tool_name_count: int = 0
    model_requests: int = 0
    tool_executions: int = 0
    response_quality: str = ""
    response_quality_action: str = ""
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_observed_prompt_tokens: int = 0
    cache_unknown_prompt_tokens: int = 0
    cache_observed_model_calls: int = 0
    cache_unknown_model_calls: int = 0


class _RouteObserver:
    def __init__(self) -> None:
        self._records: deque[RouteObservation] = deque(maxlen=self._capacity())

    def _capacity(self) -> int:
        return max(int(ROUTE_OBSERVER_MAX_RECORDS), 50)

    def record(self, record: RouteObservation) -> None:
        if self._records.maxlen != self._capacity():
            self._records = deque(self._records, maxlen=self._capacity())
        self._records.append(record)

    def snapshot(self, limit: int = 200) -> dict[str, Any]:
        rows = list(self._records)[-max(int(limit or 0), 1) :]
        if not rows:
            return {
                "total": 0,
                "path_counts": {},
                "outcome_counts": {},
                "stage_counts": {},
                "quality_counts": {},
                "quality_action_counts": {},
                "failure_layer_counts": {},
                "top_plugins": {},
                "avg_candidate_total": 0.0,
                "avg_tool_candidates": 0.0,
                "avg_prompt_full_candidates": 0.0,
                "prompt_tokens_total": 0,
                "cached_prompt_tokens_total": 0,
                "cache_observed_prompt_tokens_total": 0,
                "cache_unknown_prompt_tokens_total": 0,
                "cache_observed_model_calls": 0,
                "cache_unknown_model_calls": 0,
                "prompt_cache_hit_rate": 0.0,
                "recent_failures": [],
            }

        path_counts = Counter(row.path for row in rows if row.path)
        outcome_counts = Counter(row.outcome for row in rows if row.outcome)
        stage_counts = Counter(row.route_stage for row in rows if row.route_stage)
        top_plugins = Counter(
            row.route_plugin
            for row in rows
            if row.path == "plugin" and row.route_plugin
        )
        avg_candidate_total = sum(row.candidate_total for row in rows) / len(rows)
        avg_tool_candidates = sum(row.tool_candidates for row in rows) / len(rows)
        avg_prompt_full_candidates = sum(
            row.prompt_full_candidates for row in rows
        ) / len(rows)
        prompt_tokens_total = sum(row.prompt_tokens for row in rows)
        cached_tokens_total = sum(row.cached_prompt_tokens for row in rows)
        observed_prompt_tokens_total = sum(
            row.cache_observed_prompt_tokens for row in rows
        )
        unknown_prompt_tokens_total = sum(
            row.cache_unknown_prompt_tokens for row in rows
        )
        observed_model_calls = sum(row.cache_observed_model_calls for row in rows)
        unknown_model_calls = sum(row.cache_unknown_model_calls for row in rows)
        cache_hit_rate = (
            round(cached_tokens_total / observed_prompt_tokens_total, 4)
            if observed_prompt_tokens_total > 0
            else 0.0
        )
        recent_failures = [
            asdict(row)
            for row in rows
            if row.outcome
            not in {
                "plugin_reroute",
                "chat_fallback",
                "tool_completed",
                "chat_completed",
            }
        ][-8:]
        quality_counts = Counter(
            row.response_quality for row in rows if row.response_quality
        )
        quality_action_counts = Counter(
            row.response_quality_action for row in rows if row.response_quality_action
        )
        failure_layer_counts = Counter(
            row.failure_layer for row in rows if row.failure_layer
        )
        return {
            "total": len(rows),
            "path_counts": dict(path_counts),
            "outcome_counts": dict(outcome_counts),
            "stage_counts": dict(stage_counts),
            "quality_counts": dict(quality_counts),
            "quality_action_counts": dict(quality_action_counts),
            "failure_layer_counts": dict(failure_layer_counts),
            "top_plugins": dict(top_plugins.most_common(8)),
            "avg_candidate_total": round(avg_candidate_total, 2),
            "avg_tool_candidates": round(avg_tool_candidates, 2),
            "avg_prompt_full_candidates": round(avg_prompt_full_candidates, 2),
            "prompt_tokens_total": prompt_tokens_total,
            "cached_prompt_tokens_total": cached_tokens_total,
            "cache_observed_prompt_tokens_total": observed_prompt_tokens_total,
            "cache_unknown_prompt_tokens_total": unknown_prompt_tokens_total,
            "cache_observed_model_calls": observed_model_calls,
            "cache_unknown_model_calls": unknown_model_calls,
            "prompt_cache_hit_rate": cache_hit_rate,
            "recent_failures": recent_failures,
        }


_OBSERVER = _RouteObserver()


def record_route_observation(
    *,
    user_id: str,
    group_id: str | None,
    message_preview: str,
    trace_tags: dict[str, str],
    route_report: Any | None = None,
    prompt_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    cache_observed_prompt_tokens: int = 0,
    cache_unknown_prompt_tokens: int = 0,
    cache_observed_model_calls: int = 0,
    cache_unknown_model_calls: int = 0,
) -> None:
    route_stage = str(trace_tags.get("route_stage", "") or "")
    route_plugin = str(trace_tags.get("route_plugin", "") or "")
    route_module = str(trace_tags.get("route_module", "") or "")
    route_head = str(trace_tags.get("route_head", "") or "")
    response_quality = str(trace_tags.get("response_quality", "") or "")
    response_quality_action = str(trace_tags.get("response_quality_action", "") or "")
    final_reason = ""
    candidate_total = 0
    lexical_candidates = 0
    direct_candidates = 0
    vector_candidates = 0
    attempts = 0
    tool_candidates = 0
    tool_attempts = 0
    tool_choice_count = 0
    prompt_full_candidates = 0
    if route_report is not None:
        route_stage = route_stage or str(
            getattr(route_report, "selected_stage", "") or ""
        )
        route_plugin = route_plugin or str(
            getattr(route_report, "selected_plugin", "") or ""
        )
        route_module = route_module or str(
            getattr(route_report, "selected_module", "") or ""
        )
        if not route_head:
            route_head = normalize_message_text(
                str(getattr(route_report, "selected_command", "") or "").split(
                    " ",
                    1,
                )[0]
            )
        final_reason = str(getattr(route_report, "final_reason", "") or "")
        candidate_total = int(getattr(route_report, "candidate_total", 0) or 0)
        lexical_candidates = int(getattr(route_report, "lexical_candidates", 0) or 0)
        direct_candidates = int(getattr(route_report, "direct_candidates", 0) or 0)
        vector_candidates = int(getattr(route_report, "vector_candidates", 0) or 0)
        attempts = int(getattr(route_report, "attempts", 0) or 0)
        tool_candidates = int(getattr(route_report, "tool_candidates", 0) or 0)
        tool_attempts = int(getattr(route_report, "tool_attempts", 0) or 0)
        tool_choice_count = int(getattr(route_report, "tool_choice_count", 0) or 0)
        prompt_full_candidates = int(
            getattr(route_report, "prompt_full_candidates", 0) or 0
        )

    _OBSERVER.record(
        RouteObservation(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            trace_id=str(trace_tags.get("message_id", "") or ""),
            user_id=str(user_id or ""),
            group_id=str(group_id or "private"),
            message_preview=normalize_message_text(message_preview)[:120],
            path=str(trace_tags.get("path", "") or ""),
            outcome=str(trace_tags.get("outcome", "") or ""),
            route_stage=route_stage,
            route_plugin=route_plugin,
            route_module=route_module,
            route_head=route_head,
            candidate_total=candidate_total,
            lexical_candidates=lexical_candidates,
            direct_candidates=direct_candidates,
            vector_candidates=vector_candidates,
            attempts=attempts,
            tool_candidates=tool_candidates,
            tool_attempts=tool_attempts,
            tool_choice_count=tool_choice_count,
            prompt_full_candidates=prompt_full_candidates,
            final_reason=final_reason,
            failure_layer=str(trace_tags.get("failure_layer", "") or ""),
            plugin_outcome=str(trace_tags.get("plugin_outcome", "") or ""),
            exact_identity_ids=str(trace_tags.get("exact_identity_ids", "") or ""),
            strict_identity_match_modes=str(
                trace_tags.get("strict_identity_match_modes", "") or ""
            ),
            exposed_command_ids=str(
                trace_tags.get("exposed_command_ids", "") or ""
            ),
            available_command_count=max(
                int(float(trace_tags.get("available_command_count", 0) or 0)), 0
            ),
            selected_command_ids=str(
                trace_tags.get("selected_command_ids", "") or ""
            ),
            selected_skill=str(trace_tags.get("selected_skill", "") or ""),
            discovery_source=str(trace_tags.get("discovery_source", "") or ""),
            retrieval_query_count=max(
                int(float(trace_tags.get("retrieval_query_count", 0) or 0)),
                0,
            ),
            discovery_candidate_count=max(
                int(float(trace_tags.get("candidate_count", 0) or 0)),
                0,
            ),
            candidate_displayed=max(
                int(float(trace_tags.get("candidate_displayed", 0) or 0)),
                0,
            ),
            candidate_omitted=max(
                int(float(trace_tags.get("candidate_omitted", 0) or 0)),
                0,
            ),
            candidate_exposure_count=max(
                int(float(trace_tags.get("candidate_exposure_count", 0) or 0)),
                0,
            ),
            selected_command_id=str(
                trace_tags.get("selected_command_id", "") or ""
            ),
            selected_capability_id=str(
                trace_tags.get("selected_capability_id", "") or ""
            ),
            execution_validation_reason=str(
                trace_tags.get("execution_validation_reason", "") or ""
            ),
            identity_spans=str(trace_tags.get("identity_spans", "") or ""),
            person_candidate_count=max(
                int(float(trace_tags.get("person_candidate_count", 0) or 0)),
                0,
            ),
            candidate_sources=str(trace_tags.get("candidate_sources", "") or ""),
            selected_target_ref=str(
                trace_tags.get("selected_target_ref", "") or ""
            ),
            target_resolution_mode=str(
                trace_tags.get("target_resolution_mode", "") or ""
            ),
            target_validation_reason=str(
                trace_tags.get("target_validation_reason", "") or ""
            ),
            self_identity_candidate=bool(
                float(trace_tags.get("self_identity_candidate", 0) or 0)
            ),
            protocol_argument_retries=max(
                int(float(trace_tags.get("protocol_argument_retries", 0) or 0)), 0
            ),
            protocol_format_retries=max(
                int(float(trace_tags.get("protocol_format_retries", 0) or 0)), 0
            ),
            protocol_text_only_retries=max(
                int(float(trace_tags.get("protocol_text_only_retries", 0) or 0)), 0
            ),
            protocol_text_suppressed=max(
                int(float(trace_tags.get("protocol_text_suppressed", 0) or 0)), 0
            ),
            tool_argument_envelope_repairs=max(
                int(
                    float(
                        trace_tags.get("tool_argument_envelope_repairs", 0) or 0
                    )
                ),
                0,
            ),
            protocol_tool_name_count=max(
                int(float(trace_tags.get("protocol_tool_name_count", 0) or 0)), 0
            ),
            model_requests=max(
                int(float(trace_tags.get("agent_model_requests", 0) or 0)),
                0,
            ),
            tool_executions=max(
                int(float(trace_tags.get("agent_tool_executions", 0) or 0)),
                0,
            ),
            response_quality=response_quality,
            response_quality_action=response_quality_action,
            prompt_tokens=max(int(prompt_tokens or 0), 0),
            cached_prompt_tokens=max(int(cached_prompt_tokens or 0), 0),
            cache_observed_prompt_tokens=max(int(cache_observed_prompt_tokens or 0), 0),
            cache_unknown_prompt_tokens=max(int(cache_unknown_prompt_tokens or 0), 0),
            cache_observed_model_calls=max(int(cache_observed_model_calls or 0), 0),
            cache_unknown_model_calls=max(int(cache_unknown_model_calls or 0), 0),
        )
    )


def get_route_observer_snapshot(limit: int = 200) -> dict[str, Any]:
    return _OBSERVER.snapshot(limit=limit)


def render_route_observer_summary(limit: int = 200) -> str:
    payload = get_route_observer_snapshot(limit=limit)
    if payload["total"] <= 0:
        return "暂无 ChatInter 路由观测数据。"
    lines = [
        f"ChatInter 最近 {payload['total']} 条",
        "path: "
        + ", ".join(f"{k}={v}" for k, v in sorted(payload["path_counts"].items())),
        "outcome: "
        + ", ".join(f"{k}={v}" for k, v in sorted(payload["outcome_counts"].items())),
        "stage: "
        + ", ".join(f"{k}={v}" for k, v in sorted(payload["stage_counts"].items())),
        "quality: "
        + ", ".join(f"{k}={v}" for k, v in sorted(payload["quality_counts"].items())),
        "failure_layer: "
        + ", ".join(
            f"{k}={v}" for k, v in sorted(payload["failure_layer_counts"].items())
        ),
        (
            f"avg_candidates={payload['avg_candidate_total']}, "
            f"avg_tool_candidates={payload['avg_tool_candidates']}"
        ),
        (f"avg_prompt_full_schema={payload['avg_prompt_full_candidates']}"),
        (
            f"prompt_cache: hit_rate={payload['prompt_cache_hit_rate'] * 100:.1f}%, "
            f"cached={payload['cached_prompt_tokens_total']}, "
            f"observed_prompt={payload['cache_observed_prompt_tokens_total']}, "
            f"unknown_prompt={payload['cache_unknown_prompt_tokens_total']}"
        ),
    ]
    top_plugins = payload.get("top_plugins") or {}
    if top_plugins:
        lines.append(
            "top_plugins: " + ", ".join(f"{k}={v}" for k, v in top_plugins.items())
        )
    recent_failures = payload.get("recent_failures") or []
    if recent_failures:
        lines.append("recent_failures:")
        for item in recent_failures[-5:]:
            lines.append(
                f"- {item['timestamp']} {item['outcome']} "
                f"{item['route_plugin'] or item['route_module'] or '-'} "
                f"| {item['message_preview']}"
            )
    return "\n".join(lines)


__all__ = [
    "RouteObservation",
    "TurnMetricsSnapshot",
    "build_turn_metrics_snapshot",
    "emit_turn_metrics",
    "get_route_observer_snapshot",
    "record_route_observation",
    "render_route_observer_summary",
]
