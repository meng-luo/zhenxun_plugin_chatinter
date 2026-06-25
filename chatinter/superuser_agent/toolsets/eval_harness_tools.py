"""Eval Harness inspection tools for superuser Agent."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ...eval_runner import (
    build_trend_report,
    ensure_eval_harness_files,
    eval_harness_paths,
    render_eval_report_markdown,
    run_eval_harness,
)
from ...persistence import read_json
from ..registry import register_superuser_tool
from .common import actor_from_context, tool_result


class EvalHarnessRunTool:
    name = "eval_harness_run"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：基于真实 trajectory JSONL 运行 ChatInter "
                "固定分层回归评估，输出阈值检查、失败样本和对比报告。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "trajectory_path": {
                        "type": ["string", "null"],
                        "description": (
                            "可选 trajectory JSONL 路径；" "为空使用当天默认轨迹。"
                        ),
                    },
                    "dataset_path": {
                        "type": ["string", "null"],
                        "description": "可选固定测试集 JSON 路径。",
                    },
                    "thresholds_path": {
                        "type": ["string", "null"],
                        "description": "可选阈值 JSON 路径。",
                    },
                    "baseline_path": {
                        "type": ["string", "null"],
                        "description": "可选 baseline report 路径。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多读取最近多少条轨迹，默认 1000。",
                    },
                    "run_name": {
                        "type": ["string", "null"],
                        "description": "本次评估名称。",
                    },
                    "write_baseline": {
                        "type": ["boolean", "null"],
                        "description": "是否把本次报告写为 baseline。",
                    },
                    "archive_failures": {
                        "type": ["boolean", "null"],
                        "description": "是否归档失败样本，默认 true。",
                    },
                },
                "required": [
                    "trajectory_path",
                    "dataset_path",
                    "thresholds_path",
                    "baseline_path",
                    "limit",
                    "run_name",
                    "write_baseline",
                    "archive_failures",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        report = run_eval_harness(
            trajectory_path=str(kwargs.get("trajectory_path") or "") or None,
            dataset_path=str(kwargs.get("dataset_path") or "") or None,
            thresholds_path=str(kwargs.get("thresholds_path") or "") or None,
            baseline_path=str(kwargs.get("baseline_path") or "") or None,
            limit=_coerce_limit(kwargs.get("limit")),
            run_name=str(kwargs.get("run_name") or ""),
            write_baseline=bool(kwargs.get("write_baseline") or False),
            archive_failures=kwargs.get("archive_failures") is not False,
        )
        latest = read_json(eval_harness_paths().latest_report, {})
        return tool_result(
            bool(report.get("passed")),
            "eval_harness_passed" if report.get("passed") else "eval_harness_failed",
            run_id=report.get("run_id"),
            passed=report.get("passed"),
            summary=report.get("summary"),
            checks=report.get("checks"),
            failure_count=len(report.get("failures", []) or []),
            missing_count=len(report.get("missing_cases", []) or []),
            comparison=report.get("comparison"),
            trend=report.get("trend"),
            quality_verdict=report.get("quality_verdict"),
            report_json=latest.get("report_json") if isinstance(latest, dict) else "",
            report_md=latest.get("report_md") if isinstance(latest, dict) else "",
            trend_json=latest.get("trend_json") if isinstance(latest, dict) else "",
            trend_md=latest.get("trend_md") if isinstance(latest, dict) else "",
            report_preview=render_eval_report_markdown(report)[:1800],
        )


class EvalHarnessStatusTool:
    name = "eval_harness_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查看 Eval Harness 固定测试集、"
                "阈值、最新报告和 baseline 状态。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "include_cases": {
                        "type": ["boolean", "null"],
                        "description": "是否返回测试用例明细，默认 false。",
                    }
                },
                "required": ["include_cases"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        paths = eval_harness_paths()
        ensure_eval_harness_files()
        dataset = read_json(paths.dataset, {})
        thresholds = read_json(paths.thresholds, {})
        latest = read_json(paths.latest_report, {})
        baseline = read_json(paths.baseline, {})
        include_cases = bool(kwargs.get("include_cases") or False)
        cases = dataset.get("cases") if isinstance(dataset, dict) else []
        payload: dict[str, Any] = {
            "paths": {
                "dataset": str(paths.dataset),
                "thresholds": str(paths.thresholds),
                "reports": str(paths.reports),
                "failures": str(paths.failures),
                "latest_report": str(paths.latest_report),
                "baseline": str(paths.baseline),
                "history": str(paths.history),
                "trends": str(paths.trends),
            },
            "case_count": len(cases) if isinstance(cases, list) else 0,
            "layers": _layer_counts(cases if isinstance(cases, list) else []),
            "thresholds": thresholds,
            "latest_summary": latest.get("summary") if isinstance(latest, dict) else {},
            "latest_run_id": latest.get("run_id") if isinstance(latest, dict) else "",
            "latest_quality_verdict": latest.get("quality_verdict")
            if isinstance(latest, dict)
            else "",
            "latest_trend": latest.get("trend") if isinstance(latest, dict) else {},
            "baseline_run_id": baseline.get("run_id")
            if isinstance(baseline, dict)
            else "",
            "report_json": latest.get("report_json")
            if isinstance(latest, dict)
            else "",
            "report_md": latest.get("report_md") if isinstance(latest, dict) else "",
            "trend_json": latest.get("trend_json") if isinstance(latest, dict) else "",
            "trend_md": latest.get("trend_md") if isinstance(latest, dict) else "",
        }
        if include_cases:
            payload["cases"] = cases
        return tool_result(True, "eval_harness_status", **payload)


class EvalHarnessTrendTool:
    name = "eval_harness_trend"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查看 Eval Harness token/latency/"
                "准确率趋势，用于判断能力提升还是 prompt 更激进。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "history_limit": {
                        "type": ["integer", "null"],
                        "description": "趋势窗口大小，默认 12。",
                    }
                },
                "required": ["history_limit"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        latest = read_json(eval_harness_paths().latest_report, {})
        if not isinstance(latest, dict) or not latest:
            return tool_result(False, "eval_harness_no_latest_report")
        trend = build_trend_report(
            current=latest,
            history_limit=_coerce_limit(kwargs.get("history_limit"), default=12),
        )
        return tool_result(
            bool(trend.get("available")),
            "eval_harness_trend",
            trend=trend,
            quality_verdict=latest.get("quality_verdict", ""),
            latest_run_id=latest.get("run_id", ""),
        )


def _coerce_limit(value: Any, *, default: int = 1000) -> int:
    try:
        return max(1, min(int(value or default), 10000))
    except (TypeError, ValueError):
        return default


def _layer_counts(cases: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in cases:
        if not isinstance(item, dict):
            continue
        layer = str(item.get("layer") or "unknown")
        counts[layer] = counts.get(layer, 0) + 1
    return dict(sorted(counts.items()))


register_superuser_tool(
    EvalHarnessRunTool,
    category="eval",
    risk="low",
    read_only=False,
    destructive=False,
    side_effect="execute",
    produces_artifacts=True,
    tags=("eval", "trajectory", "regression"),
)
register_superuser_tool(
    EvalHarnessStatusTool,
    category="eval",
    risk="low",
    read_only=True,
    tags=("eval", "trajectory", "regression"),
)
register_superuser_tool(
    EvalHarnessTrendTool,
    category="eval",
    risk="low",
    read_only=True,
    tags=("eval", "trajectory", "trend", "cost", "latency"),
)

__all__ = ["EvalHarnessRunTool", "EvalHarnessStatusTool", "EvalHarnessTrendTool"]
