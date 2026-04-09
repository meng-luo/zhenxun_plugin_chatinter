from __future__ import annotations

from dataclasses import asdict, dataclass

from zhenxun.services import logger

from .route_engine import RouteAttemptReport
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
            name: round(cost * 1000, 2)
            for name, cost in getattr(trace, "_stages", ())
        },
        tags=dict(trace.tags),
        route_reason=(route_report.final_reason if route_report else None),
        route_attempts=(route_report.attempts if route_report else 0),
        route_tool_attempts=(route_report.tool_attempts if route_report else 0),
        route_candidates=(route_report.candidate_total if route_report else 0),
        route_tool_candidates=(route_report.tool_candidates if route_report else 0),
        route_tool_choices=(route_report.tool_choice_count if route_report else 0),
        runtime_budget=runtime_budget,
    )


def emit_turn_metrics(snapshot: TurnMetricsSnapshot) -> None:
    logger.debug(f"ChatInter turn metrics: {snapshot.to_dict()}")


__all__ = [
    "TurnMetricsSnapshot",
    "build_turn_metrics_snapshot",
    "emit_turn_metrics",
]
