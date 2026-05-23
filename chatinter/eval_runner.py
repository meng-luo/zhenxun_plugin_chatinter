"""Productized trajectory eval harness for ChatInter.

This module turns recorded trajectories into repeatable regression reports:
fixed layered test cases, thresholds, failure archives and baseline comparison.
It deliberately stays outside runtime policy; it observes behavior after runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

try:
    from .eval_dataset import (
        DATASET_SCHEMA_VERSION,
        DEFAULT_EVAL_CASES,
        DEFAULT_THRESHOLDS,
        THRESHOLD_SCHEMA_VERSION,
    )
except ImportError:  # Allows `py zhenxun/plugins/chatinter/eval_runner.py`.
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from eval_dataset import (  # type: ignore[no-redef]
        DATASET_SCHEMA_VERSION,
        DEFAULT_EVAL_CASES,
        DEFAULT_THRESHOLDS,
        THRESHOLD_SCHEMA_VERSION,
    )

HARNESS_SCHEMA_VERSION = "chatinter.eval_harness.v1"
_DEFAULT_LIMIT = 1000
_STATE_ROOT = Path("data/chatinter_agent")
_DEFAULT_COMPARE_KEYS = (
    "case_coverage",
    "pass_rate",
    "hit_rate",
    "false_trigger_rate",
    "multi_coverage_rate",
    "tool_call_pressure",
    "avg_latency_ms",
    "avg_prompt_tokens",
)


@dataclass(frozen=True)
class EvalHarnessPaths:
    root: Path
    dataset: Path
    thresholds: Path
    reports: Path
    failures: Path
    trends: Path
    latest_report: Path
    baseline: Path
    history: Path


def eval_harness_paths() -> EvalHarnessPaths:
    root = _state_path("eval_harness")
    return EvalHarnessPaths(
        root=root,
        dataset=root / "dataset.json",
        thresholds=root / "thresholds.json",
        reports=root / "reports",
        failures=root / "failures",
        trends=root / "trends",
        latest_report=root / "latest_report.json",
        baseline=root / "baseline_report.json",
        history=root / "history.jsonl",
    )


def ensure_eval_harness_files(
    *,
    dataset_path: Path | str | None = None,
    thresholds_path: Path | str | None = None,
) -> dict[str, Path]:
    """Create default dataset/threshold files if they do not exist."""

    paths = eval_harness_paths()
    dataset = Path(dataset_path) if dataset_path is not None else paths.dataset
    thresholds = Path(thresholds_path) if thresholds_path is not None else paths.thresholds
    if not dataset.exists():
        _write_json(
            dataset,
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "updated_at": _utc_now_iso(),
                "cases": list(DEFAULT_EVAL_CASES),
            },
        )
    else:
        _merge_default_dataset_cases(dataset)
    if not thresholds.exists():
        payload = dict(DEFAULT_THRESHOLDS)
        payload.setdefault("updated_at", _utc_now_iso())
        _write_json(thresholds, payload)
    else:
        _merge_default_thresholds(thresholds)
    return {"dataset": dataset, "thresholds": thresholds}


def load_eval_dataset(path: Path | str | None = None) -> list[dict[str, Any]]:
    paths = ensure_eval_harness_files(dataset_path=path)
    payload = _read_json(paths["dataset"], {})
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        return [dict(item) for item in DEFAULT_EVAL_CASES]
    return [_normalize_case(item) for item in cases if isinstance(item, dict)]


def load_eval_thresholds(path: Path | str | None = None) -> dict[str, Any]:
    paths = ensure_eval_harness_files(thresholds_path=path)
    payload = _read_json(paths["thresholds"], {})
    if not isinstance(payload, dict):
        return dict(DEFAULT_THRESHOLDS)
    return _merge_thresholds(DEFAULT_THRESHOLDS, payload)


def run_eval_harness(
    *,
    trajectory_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
    thresholds_path: Path | str | None = None,
    baseline_path: Path | str | None = None,
    limit: int = _DEFAULT_LIMIT,
    run_name: str = "",
    write_baseline: bool = False,
    archive_failures: bool = True,
) -> dict[str, Any]:
    """Evaluate recorded trajectories against the fixed layered dataset."""

    ensure_eval_harness_files(dataset_path=dataset_path, thresholds_path=thresholds_path)
    cases = load_eval_dataset(dataset_path)
    thresholds = load_eval_thresholds(thresholds_path)
    default_trajectory_path = _trajectory_jsonl_path()
    records = _load_trajectory_records(
        path=Path(trajectory_path) if trajectory_path is not None else default_trajectory_path,
        limit=max(1, int(limit or _DEFAULT_LIMIT)),
    )
    report = build_eval_report(
        cases=cases,
        thresholds=thresholds,
        records=records,
        run_name=run_name,
        trajectory_path=str(trajectory_path or default_trajectory_path),
    )
    baseline = _load_baseline(baseline_path)
    if baseline:
        report["comparison"] = compare_eval_reports(report, baseline)
    else:
        report["comparison"] = {"available": False, "reason": "baseline_not_found"}
    report["trend"] = build_trend_report(current=report)
    report["quality_verdict"] = classify_quality_change(report)
    _persist_eval_report(
        report,
        write_baseline=write_baseline,
        archive_failures=archive_failures,
        baseline_path=Path(baseline_path) if baseline_path is not None else None,
    )
    return report


def build_eval_report(
    *,
    cases: list[dict[str, Any]],
    thresholds: dict[str, Any],
    records: list[dict[str, Any]],
    run_name: str = "",
    trajectory_path: str = "",
) -> dict[str, Any]:
    run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    normalized_cases = [_normalize_case(case) for case in cases]
    records_by_input = _records_by_input(records)
    case_results = [
        evaluate_case(case, _records_for_case(case, records_by_input))
        for case in normalized_cases
    ]
    layers = _summarize_layers(case_results)
    summary = _summarize_global(case_results)
    checks = evaluate_thresholds(
        summary=summary,
        layers=layers,
        thresholds=thresholds,
    )
    failures = [
        result
        for result in case_results
        if not result.get("passed") and not result.get("missing")
    ]
    missing_cases = [result for result in case_results if result.get("missing")]
    report = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "run_id": run_id,
        "run_name": _normalize_text(run_name),
        "created_at": _utc_now_iso(),
        "trajectory_path": trajectory_path,
        "dataset": {
            "schema_version": DATASET_SCHEMA_VERSION,
            "case_count": len(normalized_cases),
            "layers": _case_layer_counts(normalized_cases),
        },
        "threshold_schema_version": thresholds.get(
            "schema_version",
            THRESHOLD_SCHEMA_VERSION,
        ),
        "summary": summary,
        "layers": layers,
        "cost_latency": _cost_latency_summary(case_results),
        "checks": checks,
        "passed": bool(checks.get("passed")) and not failures,
        "case_results": case_results,
        "failures": failures,
        "missing_cases": missing_cases,
        "failure_archive_policy": {
            "groups": [
                "runtime_failed",
                "expected_tool_call_missing",
                "tool_call_without_success_observation",
                "chat_case_triggered_tool",
                "multi_task_not_covered",
                "cost_or_latency_regression",
                "missing_trajectory",
            ],
            "purpose": "preserve failed examples for regression triage without changing runtime policy",
        },
    }
    report["passed"] = bool(checks.get("passed")) and all(
        result.get("passed") or result.get("missing") for result in case_results
    )
    return report


def evaluate_case(
    case: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    record = _latest_record(records)
    if record is None:
        return {
            "case_id": case["id"],
            "layer": case["layer"],
            "expectation": case["expectation"],
            "input_message": case["input_message"],
            "matched": False,
            "missing": True,
            "passed": False,
            "reason": "no_matching_trajectory",
            "tags": list(case.get("tags", [])),
        }
    metrics = _trajectory_metrics(record)
    observations = _dict_list(record.get("observations"))
    selected_tools = _text_list(record.get("selected_tools"))
    passed, reason = _case_passed(
        case=case,
        record=record,
        metrics=metrics,
        observations=observations,
        selected_tools=selected_tools,
    )
    return {
        "case_id": case["id"],
        "layer": case["layer"],
        "expectation": case["expectation"],
        "input_message": case["input_message"],
        "matched": True,
        "missing": False,
        "passed": passed,
        "reason": reason,
        "trace_id": metrics.get("trace_id", ""),
        "run_id": metrics.get("run_id", ""),
        "status": metrics.get("status", ""),
        "tool_obligation": metrics.get("tool_obligation", ""),
        "hit": metrics.get("hit"),
        "false_trigger": bool(metrics.get("false_trigger")),
        "multi_task_covered": metrics.get("multi_task_covered"),
        "latency_ms": metrics.get("latency_ms", 0),
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "steps": metrics.get("steps", 0),
        "tool_call_count": metrics.get("tool_call_count", 0),
        "observation_count": metrics.get("observation_count", 0),
        "over_tooling": _case_over_tooling(case, metrics, selected_tools),
        "selected_tools": selected_tools[:8],
        "ok_action_observation_count": _ok_action_observation_count(observations),
        "plugins": metrics.get("plugins", [])[:8],
        "final_reply": _clip(str(record.get("final_reply", "") or ""), limit=220),
        "stop_reason": _normalize_text(str(record.get("stop_reason", "") or "")),
        "paused_reason": _normalize_text(str(record.get("paused_reason", "") or "")),
        "failure_evidence": _failure_evidence(record, observations),
        "tags": list(case.get("tags", [])),
    }


def evaluate_thresholds(
    *,
    summary: dict[str, Any],
    layers: dict[str, dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    global_thresholds = thresholds.get("global") if isinstance(thresholds, dict) else {}
    if isinstance(global_thresholds, dict):
        checks.extend(_threshold_checks("global", summary, global_thresholds))
    layer_thresholds = thresholds.get("layers") if isinstance(thresholds, dict) else {}
    if isinstance(layer_thresholds, dict):
        for layer, spec in layer_thresholds.items():
            if not isinstance(spec, dict):
                continue
            checks.extend(_threshold_checks(f"layer:{layer}", layers.get(layer, {}), spec))
    failed = [check for check in checks if not check.get("passed")]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "checks": checks,
        "failed": failed,
    }


def compare_eval_reports(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    current_summary = current.get("summary") if isinstance(current.get("summary"), dict) else {}
    baseline_summary = baseline.get("summary") if isinstance(baseline.get("summary"), dict) else {}
    summary_delta = _metric_delta(current_summary, baseline_summary)
    layer_delta: dict[str, Any] = {}
    current_layers = current.get("layers") if isinstance(current.get("layers"), dict) else {}
    baseline_layers = baseline.get("layers") if isinstance(baseline.get("layers"), dict) else {}
    for layer in sorted(set(current_layers) | set(baseline_layers)):
        layer_delta[layer] = _metric_delta(
            current_layers.get(layer, {}) if isinstance(current_layers.get(layer), dict) else {},
            baseline_layers.get(layer, {}) if isinstance(baseline_layers.get(layer), dict) else {},
        )
    regressions = _comparison_regressions(summary_delta, layer_delta)
    interpretation = _comparison_interpretation(
        summary_delta=summary_delta,
        regressions=regressions,
    )
    return {
        "available": True,
        "baseline_run_id": _normalize_text(str(baseline.get("run_id", "") or "")),
        "current_run_id": _normalize_text(str(current.get("run_id", "") or "")),
        "summary_delta": summary_delta,
        "layer_delta": layer_delta,
        "regressions": regressions,
        "regression_count": len(regressions),
        "interpretation": interpretation,
    }


def build_trend_report(
    *,
    current: dict[str, Any],
    history_limit: int = 12,
) -> dict[str, Any]:
    history = _load_report_history(limit=max(int(history_limit or 12), 2))
    previous = [
        item for item in history if item.get("run_id") != current.get("run_id")
    ]
    if not previous:
        return {"available": False, "reason": "not_enough_history"}
    window = [*previous[-(history_limit - 1) :], _history_entry(current)]
    signals = _trend_signals(window)
    return {
        "available": True,
        "window_size": len(window),
        "run_ids": [item.get("run_id") for item in window],
        "signals": signals,
        "interpretation": _trend_interpretation(signals),
    }


def classify_quality_change(report: dict[str, Any]) -> str:
    comparison = report.get("comparison") if isinstance(report.get("comparison"), dict) else {}
    trend = report.get("trend") if isinstance(report.get("trend"), dict) else {}
    if comparison.get("available"):
        interpretation = str(comparison.get("interpretation") or "")
        if interpretation in {
            "likely_capability_gain",
            "likely_tool_accuracy_gain",
        }:
            return interpretation
        if "prompt_pressure" in interpretation:
            return "prompt_more_aggressive_not_clear_gain"
        if "cost_regression" in interpretation or "latency_regression" in interpretation:
            return "cost_or_latency_regression_without_clear_gain"
        if comparison.get("regression_count"):
            return "metric_regression"
    if trend.get("available"):
        interpretation = str(trend.get("interpretation") or "")
        if interpretation:
            return interpretation
    return "stable_or_insufficient_baseline"


def render_eval_report_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        f"# ChatInter Eval Report {report.get('run_id', '')}",
        "",
        f"- created_at: {report.get('created_at', '')}",
        f"- passed: {report.get('passed', False)}",
        f"- case_coverage: {summary.get('case_coverage')}",
        f"- pass_rate: {summary.get('pass_rate')}",
        f"- hit_rate: {summary.get('hit_rate')}",
        f"- false_trigger_rate: {summary.get('false_trigger_rate')}",
        f"- tool_call_pressure: {summary.get('tool_call_pressure')}",
        f"- avg_latency_ms: {summary.get('avg_latency_ms')}",
        f"- avg_prompt_tokens: {summary.get('avg_prompt_tokens')}",
        f"- quality_verdict: {report.get('quality_verdict', '')}",
        "",
        "## Layers",
        "",
        "| layer | cases | matched | pass_rate | hit_rate | false_trigger_rate | tool_pressure | avg_latency_ms | avg_prompt_tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    layers = report.get("layers") if isinstance(report.get("layers"), dict) else {}
    for layer, payload in sorted(layers.items()):
        if not isinstance(payload, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(layer),
                    str(payload.get("case_count", 0)),
                    str(payload.get("matched_count", 0)),
                    str(payload.get("pass_rate")),
                    str(payload.get("hit_rate")),
                    str(payload.get("false_trigger_rate")),
                    str(payload.get("tool_call_pressure")),
                    str(payload.get("avg_latency_ms")),
                    str(payload.get("avg_prompt_tokens")),
                ]
            )
            + " |"
        )
    failures = report.get("failures") if isinstance(report.get("failures"), list) else []
    missing_cases = (
        report.get("missing_cases")
        if isinstance(report.get("missing_cases"), list)
        else []
    )
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("No failed cases.")
    else:
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            lines.append(
                f"- `{failure.get('case_id', '')}` [{failure.get('layer', '')}] "
                f"{failure.get('reason', '')} / input: {failure.get('input_message', '')}"
            )
            evidence = failure.get("failure_evidence")
            if isinstance(evidence, dict):
                compact = ", ".join(
                    f"{key}={value}"
                    for key, value in evidence.items()
                    if value not in ("", [], {}, None)
                )
                if compact:
                    lines.append(f"  evidence: {compact}")
    lines.extend(["", "## Missing Cases", ""])
    if not missing_cases:
        lines.append("No missing cases.")
    else:
        for item in missing_cases:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- `{item.get('case_id', '')}` [{item.get('layer', '')}] "
                f"input: {item.get('input_message', '')}"
            )
    comparison = report.get("comparison") if isinstance(report.get("comparison"), dict) else {}
    regressions = comparison.get("regressions") if isinstance(comparison.get("regressions"), list) else []
    lines.extend(["", "## Comparison", ""])
    if not comparison.get("available"):
        lines.append(str(comparison.get("reason", "baseline_not_found")))
    elif not regressions:
        interpretation = comparison.get("interpretation")
        lines.append("No metric regression against baseline.")
        if interpretation:
            lines.append(f"interpretation: {interpretation}")
    else:
        interpretation = comparison.get("interpretation")
        if interpretation:
            lines.append(f"interpretation: {interpretation}")
        for item in regressions:
            lines.append(
                f"- `{item.get('scope')}.{item.get('metric')}`: "
                f"{item.get('baseline')} -> {item.get('current')} "
                f"(delta {item.get('delta')})"
            )
    trend = report.get("trend") if isinstance(report.get("trend"), dict) else {}
    lines.extend(["", "## Trend", ""])
    if not trend.get("available"):
        lines.append(str(trend.get("reason", "trend_history_not_available")))
    else:
        lines.append(f"window_size: {trend.get('window_size')}")
        lines.append(f"interpretation: {trend.get('interpretation')}")
        for item in trend.get("signals", []) if isinstance(trend.get("signals"), list) else []:
            if isinstance(item, dict):
                lines.append(
                    f"- `{item.get('metric')}`: {item.get('first')} -> "
                    f"{item.get('latest')} (delta {item.get('delta')})"
                )
    return "\n".join(lines) + "\n"


def _case_passed(
    *,
    case: dict[str, Any],
    record: dict[str, Any],
    metrics: dict[str, Any],
    observations: list[dict[str, Any]],
    selected_tools: list[str],
) -> tuple[bool, str]:
    expectation = _normalize_text(str(case.get("expectation", "") or ""))
    min_tool_calls = _int(case.get("min_tool_calls"), fallback=0)
    tool_call_count = _int(metrics.get("tool_call_count"), fallback=len(selected_tools))
    status = _normalize_text(str(metrics.get("status", "") or ""))
    hit = metrics.get("hit")
    false_trigger = bool(metrics.get("false_trigger"))
    action_ok = _ok_action_observation_count(observations)
    if status == "failed":
        return False, "runtime_failed"
    if expectation == "must_call_tool":
        if tool_call_count < max(1, min_tool_calls):
            return False, "expected_tool_call_missing"
        if action_ok <= 0:
            return False, "tool_call_without_success_observation"
        return True, "real_tool_called"
    if expectation == "direct_chat":
        if selected_tools or tool_call_count > 0 or false_trigger:
            return False, "chat_case_triggered_tool"
        return True, "direct_chat_without_tool"
    if expectation == "no_tool_available":
        if action_ok > 0:
            return False, "unexpected_action_tool_succeeded"
        if false_trigger:
            return False, "false_trigger_on_unsupported_case"
        return True, "unsupported_handled_without_action_success"
    if expectation == "multi_tool":
        if tool_call_count < max(2, min_tool_calls):
            return False, "multi_tool_call_count_too_low"
        if metrics.get("multi_task_covered") is False:
            return False, "multi_task_not_covered"
        if hit is False:
            return False, "multi_tool_hit_failed"
        return True, "multi_tool_covered"
    if expectation == "native_continuation":
        if tool_call_count < max(1, min_tool_calls):
            return False, "continuation_not_handled_by_chatinter"
        if false_trigger:
            return False, "continuation_false_trigger"
        return True, "continuation_handled"
    if expectation == "superuser_task":
        allow_paused = bool(case.get("allow_paused"))
        if status == "paused" and allow_paused and tool_call_count >= max(1, min_tool_calls):
            return True, "superuser_task_paused_safely"
        if status != "completed":
            return False, "superuser_task_not_completed"
        if tool_call_count < max(1, min_tool_calls):
            return False, "superuser_tool_call_count_too_low"
        return True, "superuser_task_completed"
    return False, f"unknown_expectation:{expectation}"


def _summarize_global(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_bucket(case_results)


def _summarize_layers(case_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in case_results:
        grouped.setdefault(str(result.get("layer") or "unknown"), []).append(result)
    return {layer: _summarize_bucket(items) for layer, items in sorted(grouped.items())}


def _summarize_bucket(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    matched = [item for item in case_results if item.get("matched")]
    passed = [item for item in matched if item.get("passed")]
    hit_known = [item for item in matched if item.get("hit") is not None]
    hit_ok = [item for item in hit_known if item.get("hit") is True]
    false_triggers = [item for item in matched if item.get("false_trigger")]
    multi_known = [item for item in matched if item.get("multi_task_covered") is not None]
    multi_ok = [item for item in multi_known if item.get("multi_task_covered")]
    superuser = [item for item in matched if item.get("layer") == "superuser_long_task"]
    superuser_ok_or_paused = [
        item
        for item in superuser
        if item.get("status") in {"completed", "paused"}
    ]
    return {
        "case_count": total,
        "matched_count": len(matched),
        "missing_count": total - len(matched),
        "passed_count": len(passed),
        "failed_count": len(matched) - len(passed),
        "case_coverage": _rate(len(matched), total),
        "pass_rate": _rate(len(passed), len(matched)),
        "hit_rate": _rate(len(hit_ok), len(hit_known)),
        "false_trigger_rate": _rate(len(false_triggers), len(matched)),
        "multi_coverage_rate": _rate(len(multi_ok), len(multi_known)),
        "superuser_completion_or_pause_rate": _rate(
            len(superuser_ok_or_paused),
            len(superuser),
        ),
        "avg_latency_ms": _avg(
            sum(_int(item.get("latency_ms")) for item in matched),
            len(matched),
        ),
        "avg_prompt_tokens": _avg(
            sum(_int(item.get("prompt_tokens")) for item in matched),
            len(matched),
        ),
        "avg_tool_calls": _avg(
            sum(_int(item.get("tool_call_count")) for item in matched),
            len(matched),
        ),
        "tool_call_pressure": _avg(
            sum(_int(item.get("tool_call_count")) for item in matched),
            len(matched),
        ),
        "over_tooling_rate": _rate(
            len([item for item in matched if item.get("over_tooling")]),
            len(matched),
        ),
    }


def _cost_latency_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [item for item in case_results if item.get("matched")]
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for item in matched:
        by_layer.setdefault(str(item.get("layer") or "unknown"), []).append(item)
    return {
        "global": _cost_latency_bucket(matched),
        "layers": {
            layer: _cost_latency_bucket(items)
            for layer, items in sorted(by_layer.items())
        },
    }


def _cost_latency_bucket(items: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(_int(item.get("latency_ms")) for item in items)
    tokens = sorted(_int(item.get("prompt_tokens")) for item in items)
    tool_calls = sorted(_int(item.get("tool_call_count")) for item in items)
    return {
        "count": len(items),
        "latency_ms": {
            "avg": _avg(sum(latencies), len(latencies)),
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "max": latencies[-1] if latencies else None,
        },
        "prompt_tokens": {
            "avg": _avg(sum(tokens), len(tokens)),
            "p50": _percentile(tokens, 0.50),
            "p90": _percentile(tokens, 0.90),
            "max": tokens[-1] if tokens else None,
        },
        "tool_calls": {
            "avg": _avg(sum(tool_calls), len(tool_calls)),
            "p50": _percentile(tool_calls, 0.50),
            "p90": _percentile(tool_calls, 0.90),
            "max": tool_calls[-1] if tool_calls else None,
        },
    }


def _threshold_checks(
    scope: str,
    metrics: dict[str, Any],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key, raw_threshold in spec.items():
        if not key.startswith(("min_", "max_")):
            continue
        metric_name = key[4:]
        current = metrics.get(metric_name)
        threshold = _float(raw_threshold)
        if current is None:
            passed = False
            reason = "metric_missing"
        else:
            current_float = _float(current)
            if key.startswith("min_"):
                passed = current_float >= threshold
                reason = "below_min" if not passed else "ok"
            else:
                passed = current_float <= threshold
                reason = "above_max" if not passed else "ok"
        checks.append(
            {
                "scope": scope,
                "metric": metric_name,
                "operator": "min" if key.startswith("min_") else "max",
                "threshold": threshold,
                "current": current,
                "passed": passed,
                "reason": reason,
            }
        )
    return checks


def _records_by_input(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = _normalize_text(str(record.get("eval_case_id") or ""))
        if key:
            grouped.setdefault(f"case:{key}", []).append(record)
        input_key = _normalize_text(str(record.get("input_message") or ""))
        if input_key:
            grouped.setdefault(f"input:{input_key}", []).append(record)
    return grouped


def _records_for_case(
    case: dict[str, Any],
    records_by_input: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in _case_match_keys(case):
        for record in records_by_input.get(key, []):
            record_key = _normalize_text(
                str(record.get("trace_id") or record.get("run_id") or id(record))
            )
            if record_key in seen:
                continue
            seen.add(record_key)
            result.append(record)
    return result


def _case_match_keys(case: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    case_id = _normalize_text(str(case.get("id") or ""))
    if case_id:
        keys.append(f"case:{case_id}")
    input_message = _normalize_text(str(case.get("input_message") or ""))
    if input_message:
        keys.append(f"input:{input_message}")
    for alias in case.get("alternate_inputs", []) or []:
        alias_text = _normalize_text(str(alias or ""))
        if alias_text:
            keys.append(f"input:{alias_text}")
    return keys


def _latest_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return sorted(records, key=lambda item: str(item.get("created_at") or ""))[-1]


def _ok_action_observation_count(observations: list[dict[str, Any]]) -> int:
    count = 0
    for item in observations:
        if not item.get("ok"):
            continue
        if item.get("synthetic"):
            continue
        command_id = _normalize_text(str(item.get("command_id") or ""))
        tool_name = _normalize_text(str(item.get("tool_name") or ""))
        status = _normalize_text(str(item.get("status") or ""))
        if tool_name in {"retrieve_plugin_commands", "runtime_event_list", "runtime_event_read"}:
            continue
        if command_id or status not in {"retrieved", "capability_candidates_retrieved"}:
            count += 1
    return count


def _persist_eval_report(
    report: dict[str, Any],
    *,
    write_baseline: bool,
    archive_failures: bool,
    baseline_path: Path | None,
) -> None:
    paths = eval_harness_paths()
    run_id = _normalize_text(str(report.get("run_id") or "eval")) or "eval"
    report_json = paths.reports / f"{run_id}.json"
    report_md = paths.reports / f"{run_id}.md"
    _write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(render_eval_report_markdown(report), encoding="utf-8")
    _append_history(paths.history, report, report_json=report_json, report_md=report_md)
    trend_report = build_trend_report(current=report)
    trend_json = paths.trends / f"{run_id}.json"
    trend_md = paths.trends / f"{run_id}.md"
    _write_json(trend_json, trend_report)
    trend_md.parent.mkdir(parents=True, exist_ok=True)
    trend_md.write_text(_render_trend_markdown(trend_report), encoding="utf-8")
    _write_json(
        paths.latest_report,
        {
            **report,
            "trend": trend_report,
            "report_json": str(report_json),
            "report_md": str(report_md),
            "trend_json": str(trend_json),
            "trend_md": str(trend_md),
        },
    )
    if archive_failures:
        failure_path = paths.failures / f"{run_id}.jsonl"
        failure_summary_path = paths.failures / f"{run_id}_summary.json"
        latest_failure_index = paths.failures / "latest_failure_summary.json"
        grouped: dict[str, list[dict[str, Any]]] = {}
        for failure in report.get("failures", []) if isinstance(report.get("failures"), list) else []:
            payload = {"kind": "failure", **failure}
            _append_jsonl(failure_path, payload)
            grouped.setdefault(str(failure.get("reason") or "unknown"), []).append(payload)
        for missing in report.get("missing_cases", []) if isinstance(report.get("missing_cases"), list) else []:
            payload = {"kind": "missing", **missing}
            _append_jsonl(failure_path, payload)
            grouped.setdefault("missing_trajectory", []).append(payload)
        failure_summary = {
            "run_id": run_id,
            "created_at": report.get("created_at"),
            "failure_count": sum(len(items) for items in grouped.values()),
            "failure_path": str(failure_path),
            "groups": {
                reason: {
                    "count": len(items),
                    "cases": [
                        {
                            "case_id": item.get("case_id"),
                            "layer": item.get("layer"),
                            "input_message": item.get("input_message"),
                            "trace_id": item.get("trace_id"),
                            "reason": item.get("reason"),
                        }
                        for item in items[:20]
                    ],
                }
                for reason, items in sorted(grouped.items())
            },
        }
        _write_json(failure_summary_path, failure_summary)
        _write_json(latest_failure_index, failure_summary)
    if write_baseline:
        _write_json(baseline_path or paths.baseline, report)


def _load_baseline(path: Path | str | None) -> dict[str, Any]:
    source = Path(path) if path is not None else eval_harness_paths().baseline
    payload = _read_json(source, {})
    return payload if isinstance(payload, dict) else {}


def _append_history(
    path: Path,
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> None:
    entry = _history_entry(report)
    entry["report_json"] = str(report_json)
    entry["report_md"] = str(report_md)
    _append_jsonl(path, entry)


def _load_report_history(*, limit: int = 20) -> list[dict[str, Any]]:
    path = eval_harness_paths().history
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip().lstrip("\ufeff")
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows[-max(1, int(limit or 20)) :]


def _history_entry(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    comparison = (
        report.get("comparison") if isinstance(report.get("comparison"), dict) else {}
    )
    return {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "run_id": report.get("run_id"),
        "run_name": report.get("run_name"),
        "created_at": report.get("created_at"),
        "passed": bool(report.get("passed")),
        "quality_verdict": report.get("quality_verdict", ""),
        "summary": {
            key: summary.get(key)
            for key in [
                "case_coverage",
                "pass_rate",
                "hit_rate",
                "false_trigger_rate",
                "multi_coverage_rate",
                "tool_call_pressure",
                "avg_latency_ms",
                "avg_prompt_tokens",
            ]
        },
        "failure_count": len(report.get("failures", []) or []),
        "missing_count": len(report.get("missing_cases", []) or []),
        "comparison_interpretation": comparison.get("interpretation", ""),
        "regression_count": comparison.get("regression_count", 0),
    }


def _trend_signals(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(history) < 2:
        return []
    first = history[0].get("summary") if isinstance(history[0].get("summary"), dict) else {}
    latest = history[-1].get("summary") if isinstance(history[-1].get("summary"), dict) else {}
    signals: list[dict[str, Any]] = []
    for metric in _DEFAULT_COMPARE_KEYS:
        old = first.get(metric)
        new = latest.get(metric)
        if old is None or new is None:
            continue
        delta = round(_float(new) - _float(old), 4)
        signals.append(
            {
                "metric": metric,
                "first": old,
                "latest": new,
                "delta": delta,
                "direction": _trend_direction(metric, delta),
            }
        )
    return signals


def _trend_direction(metric: str, delta: float) -> str:
    if abs(delta) < 0.0001:
        return "flat"
    lower_is_better = {
        "false_trigger_rate",
        "tool_call_pressure",
        "avg_latency_ms",
        "avg_prompt_tokens",
    }
    if metric in lower_is_better:
        return "better" if delta < 0 else "worse"
    return "better" if delta > 0 else "worse"


def _trend_interpretation(signals: list[dict[str, Any]]) -> str:
    if not signals:
        return "not_enough_signal"
    lookup = {str(item.get("metric")): item for item in signals}
    pass_delta = _float((lookup.get("pass_rate") or {}).get("delta"))
    hit_delta = _float((lookup.get("hit_rate") or {}).get("delta"))
    pressure_delta = _float((lookup.get("tool_call_pressure") or {}).get("delta"))
    latency_delta = _float((lookup.get("avg_latency_ms") or {}).get("delta"))
    token_delta = _float((lookup.get("avg_prompt_tokens") or {}).get("delta"))
    false_delta = _float((lookup.get("false_trigger_rate") or {}).get("delta"))
    if pass_delta > 0.03 and false_delta <= 0.02 and token_delta <= 1500:
        return "trend_capability_improving"
    if hit_delta > 0.03 and pressure_delta <= 0.5:
        return "trend_tool_accuracy_improving"
    if pressure_delta > 0.5 and pass_delta <= 0.02:
        return "trend_prompt_pressure_increasing"
    if token_delta > 1500 and pass_delta <= 0.02:
        return "trend_token_cost_increasing_without_gain"
    if latency_delta > 1500 and pass_delta <= 0.02:
        return "trend_latency_increasing_without_gain"
    return "trend_stable"


def _render_trend_markdown(trend: dict[str, Any]) -> str:
    lines = ["# ChatInter Eval Trend", ""]
    if not trend.get("available"):
        lines.append(str(trend.get("reason", "trend_history_not_available")))
        return "\n".join(lines) + "\n"
    lines.append(f"- window_size: {trend.get('window_size')}")
    lines.append(f"- interpretation: {trend.get('interpretation')}")
    lines.extend(["", "| metric | first | latest | delta | direction |", "|---|---:|---:|---:|---|"])
    for item in trend.get("signals", []) if isinstance(trend.get("signals"), list) else []:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| {item.get('metric')} | {item.get('first')} | {item.get('latest')} | "
            f"{item.get('delta')} | {item.get('direction')} |"
        )
    return "\n".join(lines) + "\n"


def _metric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _DEFAULT_COMPARE_KEYS:
        cur = current.get(key)
        old = baseline.get(key)
        if cur is None or old is None:
            result[key] = {"current": cur, "baseline": old, "delta": None}
            continue
        delta = round(_float(cur) - _float(old), 4)
        result[key] = {"current": cur, "baseline": old, "delta": delta}
    return result


def _comparison_regressions(
    summary_delta: dict[str, Any],
    layer_delta: dict[str, Any],
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for scope, delta_map in [("global", summary_delta), *[(f"layer:{k}", v) for k, v in layer_delta.items()]]:
        if not isinstance(delta_map, dict):
            continue
        for metric, payload in delta_map.items():
            if not isinstance(payload, dict) or payload.get("delta") is None:
                continue
            delta = _float(payload.get("delta"))
            bad = False
            if metric in {"case_coverage", "pass_rate"} and delta < -0.03:
                bad = True
            elif metric == "false_trigger_rate" and delta > 0.03:
                bad = True
            elif metric == "tool_call_pressure" and delta > 0.5:
                bad = True
            elif metric in {"avg_latency_ms", "avg_prompt_tokens"} and delta > 1500:
                bad = True
            if bad:
                regressions.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "current": payload.get("current"),
                        "baseline": payload.get("baseline"),
                        "delta": delta,
                    }
                )
    return regressions


def _comparison_interpretation(
    *,
    summary_delta: dict[str, Any],
    regressions: list[dict[str, Any]],
) -> str:
    pass_delta = _delta_value(summary_delta, "pass_rate")
    hit_delta = _delta_value(summary_delta, "hit_rate")
    false_delta = _delta_value(summary_delta, "false_trigger_rate")
    pressure_delta = _delta_value(summary_delta, "tool_call_pressure")
    latency_delta = _delta_value(summary_delta, "avg_latency_ms")
    token_delta = _delta_value(summary_delta, "avg_prompt_tokens")
    if regressions:
        if pressure_delta > 0.5 and pass_delta <= 0.02:
            return "possible_prompt_pressure: more tool calls without meaningful pass-rate gain"
        if token_delta > 1500 and pass_delta <= 0.02:
            return "possible_cost_regression: higher token spend without clear ability gain"
        if latency_delta > 1500 and pass_delta <= 0.02:
            return "possible_latency_regression: slower without clear ability gain"
        return "metric_regression_detected"
    if pass_delta > 0.03 and false_delta <= 0.02 and token_delta <= 1500:
        return "likely_capability_gain"
    if hit_delta > 0.03 and pressure_delta <= 0.5:
        return "likely_tool_accuracy_gain"
    if pressure_delta > 0.5 and pass_delta <= 0.02:
        return "possible_prompt_pressure_without_regression_threshold"
    return "stable_or_neutral"


def _delta_value(delta_map: dict[str, Any], key: str) -> float:
    payload = delta_map.get(key)
    if not isinstance(payload, dict):
        return 0.0
    return _float(payload.get("delta"))


def _case_over_tooling(
    case: dict[str, Any],
    metrics: dict[str, Any],
    selected_tools: list[str],
) -> bool:
    expectation = _normalize_text(str(case.get("expectation", "") or ""))
    tool_call_count = _int(metrics.get("tool_call_count"), fallback=len(selected_tools))
    if expectation == "direct_chat":
        return tool_call_count > 0
    min_calls = _int(case.get("min_tool_calls"), fallback=0)
    if min_calls and tool_call_count > max(min_calls + 2, min_calls * 2):
        return True
    return False


def _failure_evidence(
    record: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = [
        _clip(str(item.get("error", "") or ""), limit=180)
        for item in observations
        if _normalize_text(str(item.get("error", "") or ""))
    ]
    return {
        "trace_id": _normalize_text(str(record.get("trace_id", "") or "")),
        "selected_tools": _text_list(record.get("selected_tools"))[:6],
        "tool_obligation": _normalize_text(str(record.get("tool_obligation", "") or "")),
        "stop_reason": _normalize_text(str(record.get("stop_reason", "") or "")),
        "errors": errors[:3],
        "final_reply": _clip(str(record.get("final_reply", "") or ""), limit=180),
    }


def _normalize_case(payload: dict[str, Any]) -> dict[str, Any]:
    case_id = _normalize_text(str(payload.get("id") or ""))
    input_message = _normalize_text(str(payload.get("input_message") or ""))
    layer = _normalize_text(str(payload.get("layer") or "unknown")) or "unknown"
    expectation = _normalize_text(str(payload.get("expectation") or "")) or layer
    tags = [_normalize_text(str(tag or "")) for tag in payload.get("tags", []) if _normalize_text(str(tag or ""))]
    return {
        **payload,
        "id": case_id or uuid.uuid5(uuid.NAMESPACE_URL, input_message).hex[:12],
        "input_message": input_message,
        "layer": layer,
        "expectation": expectation,
        "scenario": _normalize_text(str(payload.get("scenario") or "")),
        "tags": tags,
    }


def _merge_default_dataset_cases(path: Path) -> None:
    payload = _read_json(path, {})
    if not isinstance(payload, dict):
        return
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        return
    existing_ids = {
        _normalize_text(str(item.get("id") or ""))
        for item in raw_cases
        if isinstance(item, dict)
    }
    merged = list(raw_cases)
    changed = False
    for item in DEFAULT_EVAL_CASES:
        case_id = _normalize_text(str(item.get("id") or ""))
        if not case_id or case_id in existing_ids:
            continue
        merged.append(dict(item))
        existing_ids.add(case_id)
        changed = True
    if not changed:
        return
    payload["schema_version"] = DATASET_SCHEMA_VERSION
    payload["updated_at"] = _utc_now_iso()
    payload["cases"] = merged
    _write_json(path, payload)


def _merge_default_thresholds(path: Path) -> None:
    payload = _read_json(path, {})
    if not isinstance(payload, dict):
        return
    merged = _merge_thresholds(DEFAULT_THRESHOLDS, payload)
    if merged == payload:
        return
    merged["schema_version"] = THRESHOLD_SCHEMA_VERSION
    merged["updated_at"] = _utc_now_iso()
    _write_json(path, merged)


def _merge_thresholds(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(default, ensure_ascii=False))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_thresholds(result[key], value)
        else:
            result[key] = value
    return result


def _case_layer_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        layer = _normalize_text(str(case.get("layer") or "unknown")) or "unknown"
        counts[layer] = counts.get(layer, 0) + 1
    return dict(sorted(counts.items()))


def _trajectory_jsonl_path(day: str | None = None) -> Path:
    normalized_day = _normalize_text(day or "")
    if not normalized_day:
        normalized_day = datetime.now(timezone.utc).date().isoformat()
    return _state_path("trajectories", f"{normalized_day}.jsonl")


def _load_trajectory_records(*, path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip().lstrip("\ufeff")
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records[-max(1, int(limit or _DEFAULT_LIMIT)) :]


def _trajectory_metrics(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
    latency = record.get("latency") if isinstance(record.get("latency"), dict) else {}
    token = record.get("token") if isinstance(record.get("token"), dict) else {}
    scenario = _normalize_text(str(record.get("scenario") or "unknown"))
    obligation = _normalize_text(str(record.get("tool_obligation") or "none"))
    selected_tools = _text_list(record.get("selected_tools"))
    observations = _dict_list(record.get("observations"))
    task_ledger = record.get("task_ledger") if isinstance(record.get("task_ledger"), dict) else {}
    return {
        "trace_id": _normalize_text(str(record.get("trace_id") or "")),
        "run_id": _normalize_text(str(record.get("run_id") or "")),
        "scenario": scenario or "unknown",
        "layer": _layer_for_record(record, scenario=scenario),
        "agent_mode": _normalize_text(str(record.get("agent_mode") or "")),
        "status": _normalize_text(str(record.get("status") or "")),
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


def _layer_for_record(record: dict[str, Any], *, scenario: str) -> str:
    if scenario == "superuser_agent":
        return "superuser_long_task"
    obligation = _normalize_text(str(record.get("tool_obligation") or "none"))
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
        plugin = _normalize_text(str(observation.get("matched_plugin") or ""))
        if plugin and plugin not in plugins:
            plugins.append(plugin)
    for item in _dict_list(record.get("exposed_tools")):
        plugin = _normalize_text(
            str(item.get("plugin_module") or item.get("plugin_name") or "")
        )
        if plugin and plugin not in plugins:
            plugins.append(plugin)
    return plugins[:16] or ["none"]


def _task_count(task_ledger: dict[str, Any]) -> int:
    tasks = task_ledger.get("tasks") if isinstance(task_ledger, dict) else None
    return len(tasks) if isinstance(tasks, list) else 0


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = _normalize_text(str(value)).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _state_path(*parts: str) -> Path:
    path = _STATE_ROOT
    for part in parts:
        path = path / str(part).strip().strip("/\\")
    return path


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(_to_jsonable(payload), ensure_ascii=False, default=str))
        fp.write("\n")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump(mode="json"))
        except Exception:
            return str(value)
    return str(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [_normalize_text(str(item or "")) for item in value if _normalize_text(str(item or ""))]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _int(value: Any, *, fallback: int = 0) -> int:
    try:
        return max(int(float(value or 0)), 0)
    except (TypeError, ValueError):
        return max(int(fallback or 0), 0)


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _avg(total: int, count: int) -> float | None:
    if count <= 0:
        return None
    return round(float(total) / float(count), 2)


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    bounded = max(0.0, min(float(percentile or 0.0), 1.0))
    index = int(round((len(values) - 1) * bounded))
    return values[max(0, min(index, len(values) - 1))]


def _clip(value: str, *, limit: int) -> str:
    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ChatInter trajectory eval harness")
    parser.add_argument("--trajectory-path", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--no-archive-failures", action="store_true")
    args = parser.parse_args(argv)
    report = run_eval_harness(
        trajectory_path=args.trajectory_path or None,
        dataset_path=args.dataset or None,
        thresholds_path=args.thresholds or None,
        baseline_path=args.baseline or None,
        limit=args.limit,
        run_name=args.run_name,
        write_baseline=args.write_baseline,
        archive_failures=not args.no_archive_failures,
    )
    print(json.dumps(_to_jsonable(report["summary"]), ensure_ascii=False, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HARNESS_SCHEMA_VERSION",
    "build_eval_report",
    "compare_eval_reports",
    "build_trend_report",
    "classify_quality_change",
    "ensure_eval_harness_files",
    "eval_harness_paths",
    "evaluate_case",
    "evaluate_thresholds",
    "load_eval_dataset",
    "load_eval_thresholds",
    "render_eval_report_markdown",
    "run_eval_harness",
]
