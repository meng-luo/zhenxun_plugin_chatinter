from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .config import ROUTE_OBSERVER_MAX_RECORDS
from .route_text import normalize_message_text


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
                "recent_failures": [],
            }

        path_counts = Counter(row.path for row in rows if row.path)
        outcome_counts = Counter(row.outcome for row in rows if row.outcome)
        stage_counts = Counter(row.route_stage for row in rows if row.route_stage)
        top_plugins = Counter(
            row.route_plugin for row in rows if row.path == "plugin" and row.route_plugin
        )
        avg_candidate_total = sum(row.candidate_total for row in rows) / len(rows)
        avg_tool_candidates = sum(row.tool_candidates for row in rows) / len(rows)
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
    if route_report is not None:
        route_stage = route_stage or str(getattr(route_report, "selected_stage", "") or "")
        route_plugin = route_plugin or str(getattr(route_report, "selected_plugin", "") or "")
        route_module = route_module or str(getattr(route_report, "selected_module", "") or "")
        if not route_head:
            route_head = normalize_message_text(
                str(getattr(route_report, "selected_command", "") or "").split(" ", 1)[0]
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
        "path: " + ", ".join(f"{k}={v}" for k, v in sorted(payload["path_counts"].items())),
        "outcome: " + ", ".join(f"{k}={v}" for k, v in sorted(payload["outcome_counts"].items())),
        "stage: " + ", ".join(f"{k}={v}" for k, v in sorted(payload["stage_counts"].items())),
        (
            f"avg_candidates={payload['avg_candidate_total']}, "
            f"avg_tool_candidates={payload['avg_tool_candidates']}"
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
    "get_route_observer_snapshot",
    "record_route_observation",
    "render_route_observer_summary",
]
