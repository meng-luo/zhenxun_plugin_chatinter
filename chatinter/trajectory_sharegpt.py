"""Export ChatInter trajectories to ShareGPT-style training records.

The exporter is offline-only.  It does not participate in routing or runtime
policy; it converts observed trajectories into portable JSON/JSONL artifacts
for future route-model or reply-model fine-tuning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
import uuid

SHAREGPT_EXPORT_SCHEMA_VERSION = "chatinter.sharegpt_export.v1"
_STATE_ROOT = Path("data/chatinter_agent")
_DEFAULT_LIMIT = 1000


def export_trajectories_to_sharegpt(
    *,
    trajectory_path: Path | str | None = None,
    output_path: Path | str | None = None,
    limit: int = _DEFAULT_LIMIT,
    scenario: str = "",
    target: str = "route",
    include_failed: bool = False,
    jsonl: bool = True,
) -> dict[str, Any]:
    """Export recent trajectories as ShareGPT-style conversations."""

    source = (
        Path(trajectory_path) if trajectory_path is not None else _trajectory_path()
    )
    records = _load_records(source, limit=max(1, int(limit or _DEFAULT_LIMIT)))
    selected = [
        record
        for record in records
        if _record_matches(record, scenario=scenario, include_failed=include_failed)
    ]
    conversations = [
        item
        for record in selected
        if (item := _sharegpt_record(record, target=target)) is not None
    ]
    output = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(target=target, jsonl=jsonl)
    )
    _write_export(output, conversations, jsonl=jsonl)
    return {
        "schema_version": SHAREGPT_EXPORT_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "trajectory_path": str(source),
        "output_path": str(output),
        "target": target,
        "format": "jsonl" if jsonl else "json",
        "source_count": len(records),
        "selected_count": len(selected),
        "exported_count": len(conversations),
    }


def _sharegpt_record(record: dict[str, Any], *, target: str) -> dict[str, Any] | None:
    input_message = _normalize_text(str(record.get("input_message") or ""))
    if not input_message:
        return None
    assistant_text = _assistant_text(record, target=target)
    if not assistant_text:
        return None
    trace_id = _normalize_text(str(record.get("trace_id") or ""))
    run_id = _normalize_text(str(record.get("run_id") or ""))
    return {
        "id": trace_id or run_id or uuid.uuid4().hex,
        "conversations": [
            {"from": "human", "value": input_message},
            {"from": "gpt", "value": assistant_text},
        ],
        "metadata": _metadata(record, target=target),
    }


def _assistant_text(record: dict[str, Any], *, target: str) -> str:
    normalized_target = _normalize_text(target).lower() or "route"
    if normalized_target == "reply":
        return _normalize_text(str(record.get("final_reply") or ""))
    route_payload = {
        "action": _route_action(record),
        "scenario": _normalize_text(str(record.get("scenario") or "")),
        "tool_obligation": _normalize_text(str(record.get("tool_obligation") or "")),
        "selected_tools": _text_list(record.get("selected_tools"))[:12],
        "required_tool_names": _text_list(record.get("required_tool_names"))[:12],
        "status": _normalize_text(str(record.get("status") or "")),
        "hit": _optional_bool(record.get("hit")),
        "false_trigger": bool(record.get("false_trigger", False)),
        "multi_task_covered": _optional_bool(record.get("multi_task_covered")),
    }
    return json.dumps(_drop_empty(route_payload), ensure_ascii=False, sort_keys=True)


def _route_action(record: dict[str, Any]) -> str:
    selected_tools = _text_list(record.get("selected_tools"))
    obligation = _normalize_text(str(record.get("tool_obligation") or ""))
    if selected_tools:
        return "call_tool"
    if obligation == "required":
        return "missing_required_tool"
    return "chat"


def _metadata(record: dict[str, Any], *, target: str) -> dict[str, Any]:
    latency = _dict_or_empty(record.get("latency"))
    token = _dict_or_empty(record.get("token"))
    evaluation = _dict_or_empty(record.get("evaluation"))
    return _drop_empty(
        {
            "schema_version": SHAREGPT_EXPORT_SCHEMA_VERSION,
            "target": _normalize_text(target) or "route",
            "trace_id": _normalize_text(str(record.get("trace_id") or "")),
            "run_id": _normalize_text(str(record.get("run_id") or "")),
            "eval_case_id": _normalize_text(str(record.get("eval_case_id") or "")),
            "created_at": _normalize_text(str(record.get("created_at") or "")),
            "scenario": _normalize_text(str(record.get("scenario") or "")),
            "agent_mode": _normalize_text(str(record.get("agent_mode") or "")),
            "tool_obligation": _normalize_text(
                str(record.get("tool_obligation") or "")
            ),
            "selected_tools": _text_list(record.get("selected_tools"))[:12],
            "plugins": _plugins(record)[:12],
            "status": _normalize_text(str(record.get("status") or "")),
            "hit": _optional_bool(record.get("hit")),
            "false_trigger": bool(record.get("false_trigger", False)),
            "multi_task_covered": _optional_bool(record.get("multi_task_covered")),
            "latency_ms": _int(latency.get("total_ms")),
            "prompt_tokens": _int(token.get("prompt_estimated")),
            "steps": _int(record.get("steps")),
            "tool_call_count": _int(evaluation.get("tool_call_count")),
            "observation_count": _int(evaluation.get("observation_count")),
        }
    )


def _plugins(record: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for observation in _dict_list(record.get("observations")):
        plugin = _normalize_text(str(observation.get("matched_plugin") or ""))
        if plugin and plugin not in result:
            result.append(plugin)
    for item in _dict_list(record.get("exposed_tools")):
        plugin = _normalize_text(
            str(item.get("plugin_module") or item.get("plugin_name") or "")
        )
        if plugin and plugin not in result:
            result.append(plugin)
    return result


def _record_matches(
    record: dict[str, Any],
    *,
    scenario: str,
    include_failed: bool,
) -> bool:
    scenario_filter = _normalize_text(scenario)
    if (
        scenario_filter
        and _normalize_text(str(record.get("scenario") or "")) != scenario_filter
    ):
        return False
    if include_failed:
        return True
    return _normalize_text(str(record.get("status") or "")) in {"completed", "paused"}


def _load_records(path: Path, *, limit: int) -> list[dict[str, Any]]:
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
    return rows[-max(1, int(limit or _DEFAULT_LIMIT)) :]


def _write_export(path: Path, rows: list[dict[str, Any]], *, jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False, default=str))
                fp.write("\n")
        return
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _default_output_path(*, target: str, jsonl: bool) -> Path:
    suffix = "jsonl" if jsonl else "json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_target = _normalize_text(target).lower() or "route"
    return _state_path("exports", f"sharegpt_{safe_target}_{stamp}.{suffix}")


def _trajectory_path(day: str | None = None) -> Path:
    normalized_day = _normalize_text(day or "")
    if not normalized_day:
        normalized_day = datetime.now(timezone.utc).date().isoformat()
    return _state_path("trajectories", f"{normalized_day}.jsonl")


def _state_path(*parts: str) -> Path:
    path = _STATE_ROOT
    for part in parts:
        path = path / str(part).strip().strip("/\\")
    return path


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items() if value not in ("", [], {}, None)
    }


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [text for item in value if (text := _normalize_text(str(item or "")))]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


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


def _int(value: Any, *, fallback: int = 0) -> int:
    try:
        return max(int(float(value or 0)), 0)
    except (TypeError, ValueError):
        return max(int(fallback or 0), 0)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export ChatInter trajectories to ShareGPT format",
    )
    parser.add_argument("--trajectory-path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--target", choices=("route", "reply"), default="route")
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = export_trajectories_to_sharegpt(
        trajectory_path=args.trajectory_path or None,
        output_path=args.output or None,
        limit=args.limit,
        scenario=args.scenario,
        target=args.target,
        include_failed=args.include_failed,
        jsonl=not args.json,
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SHAREGPT_EXPORT_SCHEMA_VERSION",
    "export_trajectories_to_sharegpt",
]
