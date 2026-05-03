from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from zhenxun.services import logger

from .config import ROUTE_OBSERVER_MAX_RECORDS
from .route_engine import RouteAttemptReport
from .route_text import normalize_message_text
from .trace import StageTrace
from .turn_runtime import TurnBudgetController


@dataclass(frozen=True)
class TurnMetricsSnapshot:
    trace_name: str
    total_ms: float
    stage_ms: dict[str, float]
    tags: dict[str, str]
    route_reason: str | None
    route_attempts: int
    route_tool_attempts: int
    route_candidates: int
    route_tool_candidates: int
    route_tool_choices: int
    route_prompt_full_candidates: int
    route_prompt_compact_candidates: int
    route_prompt_name_only_candidates: int
    query_expansion_attempts: int = 0
    query_expansion_success: int = 0
    query_expansion_query: str = ""
    query_expansion_reason: str = ""
    rerank_attempts: int = 0
    rerank_success: int = 0
    rerank_no_available: int = 0
    rerank_stage: str = ""
    rerank_reason: str = ""
    runtime_budget: dict[str, object] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_turn_metrics_snapshot(
    *,
    trace: StageTrace,
    total_seconds: float,
    route_report: RouteAttemptReport | None = None,
    budget_controller: TurnBudgetController | None = None,
) -> TurnMetricsSnapshot:
    runtime_budget = (
        asdict(budget_controller.snapshot()) if budget_controller is not None else None
    )
    return TurnMetricsSnapshot(
        trace_name=trace.name,
        total_ms=round(total_seconds * 1000, 2),
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
        route_prompt_compact_candidates=(
            route_report.prompt_compact_candidates if route_report else 0
        ),
        route_prompt_name_only_candidates=(
            route_report.prompt_name_only_candidates if route_report else 0
        ),
        query_expansion_attempts=(
            route_report.query_expansion_attempts if route_report else 0
        ),
        query_expansion_success=(
            route_report.query_expansion_success if route_report else 0
        ),
        query_expansion_query=(
            route_report.query_expansion_query if route_report else ""
        ),
        query_expansion_reason=(
            route_report.query_expansion_reason if route_report else ""
        ),
        rerank_attempts=(route_report.rerank_attempts if route_report else 0),
        rerank_success=(route_report.rerank_success if route_report else 0),
        rerank_no_available=(route_report.rerank_no_available if route_report else 0),
        rerank_stage=(route_report.rerank_stage if route_report else ""),
        rerank_reason=(route_report.rerank_reason if route_report else ""),
        runtime_budget=runtime_budget,
    )


def emit_turn_metrics(snapshot: TurnMetricsSnapshot) -> None:
    logger.debug(f"ChatInter turn metrics: {snapshot.to_dict()}")


@dataclass(frozen=True)
class RouteObservation:
    timestamp: str
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
    prompt_compact_candidates: int
    prompt_name_only_candidates: int
    query_expansion_attempts: int
    query_expansion_success: int
    query_expansion_query: str
    query_expansion_reason: str
    rerank_attempts: int
    rerank_success: int
    rerank_no_available: int
    rerank_stage: str
    rerank_reason: str
    final_reason: str


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
                "top_plugins": {},
                "avg_candidate_total": 0.0,
                "avg_tool_candidates": 0.0,
                "avg_prompt_full_candidates": 0.0,
                "avg_prompt_compact_candidates": 0.0,
                "avg_prompt_name_only_candidates": 0.0,
                "recent_failures": [],
                "query_expansion_attempts": 0,
                "query_expansion_success": 0,
                "rerank_attempts": 0,
                "rerank_success": 0,
                "rerank_no_available": 0,
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
        avg_prompt_compact_candidates = sum(
            row.prompt_compact_candidates for row in rows
        ) / len(rows)
        avg_prompt_name_only_candidates = sum(
            row.prompt_name_only_candidates for row in rows
        ) / len(rows)
        recent_failures = [
            asdict(row)
            for row in rows
            if row.outcome not in {"plugin_reroute", "chat_fallback"}
        ][-8:]
        return {
            "total": len(rows),
            "path_counts": dict(path_counts),
            "outcome_counts": dict(outcome_counts),
            "stage_counts": dict(stage_counts),
            "top_plugins": dict(top_plugins.most_common(8)),
            "avg_candidate_total": round(avg_candidate_total, 2),
            "avg_tool_candidates": round(avg_tool_candidates, 2),
            "avg_prompt_full_candidates": round(avg_prompt_full_candidates, 2),
            "avg_prompt_compact_candidates": round(avg_prompt_compact_candidates, 2),
            "avg_prompt_name_only_candidates": round(
                avg_prompt_name_only_candidates,
                2,
            ),
            "recent_failures": recent_failures,
            "query_expansion_attempts": sum(
                row.query_expansion_attempts for row in rows
            ),
            "query_expansion_success": sum(row.query_expansion_success for row in rows),
            "rerank_attempts": sum(row.rerank_attempts for row in rows),
            "rerank_success": sum(row.rerank_success for row in rows),
            "rerank_no_available": sum(row.rerank_no_available for row in rows),
        }


_OBSERVER = _RouteObserver()


def record_route_observation(
    *,
    user_id: str,
    group_id: str | None,
    message_preview: str,
    trace_tags: dict[str, str],
    route_report: Any | None = None,
) -> None:
    route_stage = str(trace_tags.get("route_stage", "") or "")
    route_plugin = str(trace_tags.get("route_plugin", "") or "")
    route_module = str(trace_tags.get("route_module", "") or "")
    route_head = str(trace_tags.get("route_head", "") or "")
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
    prompt_compact_candidates = 0
    prompt_name_only_candidates = 0
    query_expansion_attempts = 0
    query_expansion_success = 0
    query_expansion_query = ""
    query_expansion_reason = ""
    rerank_attempts = 0
    rerank_success = 0
    rerank_no_available = 0
    rerank_stage = ""
    rerank_reason = ""
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
        prompt_compact_candidates = int(
            getattr(route_report, "prompt_compact_candidates", 0) or 0
        )
        prompt_name_only_candidates = int(
            getattr(route_report, "prompt_name_only_candidates", 0) or 0
        )
        query_expansion_attempts = int(
            getattr(route_report, "query_expansion_attempts", 0) or 0
        )
        query_expansion_success = int(
            getattr(route_report, "query_expansion_success", 0) or 0
        )
        query_expansion_query = str(
            getattr(route_report, "query_expansion_query", "") or ""
        )
        query_expansion_reason = str(
            getattr(route_report, "query_expansion_reason", "") or ""
        )
        rerank_attempts = int(getattr(route_report, "rerank_attempts", 0) or 0)
        rerank_success = int(getattr(route_report, "rerank_success", 0) or 0)
        rerank_no_available = int(getattr(route_report, "rerank_no_available", 0) or 0)
        rerank_stage = str(getattr(route_report, "rerank_stage", "") or "")
        rerank_reason = str(getattr(route_report, "rerank_reason", "") or "")

    _OBSERVER.record(
        RouteObservation(
            timestamp=datetime.now().isoformat(timespec="seconds"),
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
            prompt_compact_candidates=prompt_compact_candidates,
            prompt_name_only_candidates=prompt_name_only_candidates,
            query_expansion_attempts=query_expansion_attempts,
            query_expansion_success=query_expansion_success,
            query_expansion_query=query_expansion_query,
            query_expansion_reason=query_expansion_reason,
            rerank_attempts=rerank_attempts,
            rerank_success=rerank_success,
            rerank_no_available=rerank_no_available,
            rerank_stage=rerank_stage,
            rerank_reason=rerank_reason,
            final_reason=final_reason,
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
        (
            f"avg_candidates={payload['avg_candidate_total']}, "
            f"avg_tool_candidates={payload['avg_tool_candidates']}"
        ),
        (
            f"avg_prompt_levels=full:{payload['avg_prompt_full_candidates']}, "
            f"compact:{payload['avg_prompt_compact_candidates']}, "
            f"name:{payload['avg_prompt_name_only_candidates']}"
        ),
        (
            f"rerank={payload.get('rerank_success', 0)}/"
            f"{payload.get('rerank_attempts', 0)}, "
            f"no_tool={payload.get('rerank_no_available', 0)}"
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
