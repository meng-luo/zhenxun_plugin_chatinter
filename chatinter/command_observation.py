"""Model-facing observations for native command execution."""

from __future__ import annotations

from typing import Any

from .artifact_store import get_artifact_store, summarize_artifact_text
from .route_text import normalize_message_text

_MAX_OBSERVED_MESSAGES = 8
_MAX_OBSERVED_ARTIFACTS = 12


def build_command_observation(
    *,
    ok: bool,
    command_id: str | None,
    rendered_command: str | None,
    matched_plugin: str | None,
    messages_sent: list[str] | tuple[str, ...] | None = None,
    task_text: str = "",
    ambient_message: str = "",
    trace_id: str = "",
    error: str = "",
    missing: list[str] | tuple[str, ...] | None = None,
    retryable: bool = False,
    plugin_module: str = "",
    artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    remaining_task_hint: str | None = None,
) -> dict[str, Any]:
    """Build the compact, model-facing payload command tools return.

    Observation is deliberately descriptive instead of prescriptive: it records
    what happened and leaves the next action to the agent loop.
    """

    sent, generated_artifacts = _compact_messages(
        messages_sent or (),
        trace_id=trace_id,
    )
    remaining = normalize_message_text(remaining_task_hint or "")
    artifact_payloads = _compact_artifacts(
        [
            *(artifacts or ()),
            *generated_artifacts,
        ]
    )
    payload: dict[str, Any] = {
        "status": "success" if ok else "failed",
        "ok": bool(ok),
        "command_id": normalize_message_text(command_id or ""),
        "rendered_command": normalize_message_text(rendered_command or ""),
        "matched_plugin": normalize_message_text(matched_plugin or plugin_module or ""),
        "task_text": normalize_message_text(task_text),
        "messages_sent": sent[:_MAX_OBSERVED_MESSAGES],
        "artifacts": artifact_payloads[:_MAX_OBSERVED_ARTIFACTS],
        "need_continue": bool(remaining),
        "remaining_task_hint": remaining,
        "error": summarize_artifact_text(normalize_message_text(error)),
        "retryable": bool(retryable),
    }
    if plugin_module:
        payload["plugin_module"] = normalize_message_text(plugin_module)
    if trace_id:
        payload["trace_id"] = normalize_message_text(trace_id)
    if missing:
        payload["missing"] = [
            normalize_message_text(str(item or ""))
            for item in missing
            if normalize_message_text(str(item or ""))
        ]
    return payload


def _compact_messages(
    messages: list[str] | tuple[str, ...],
    *,
    trace_id: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    compacted: list[str] = []
    artifacts: list[dict[str, Any]] = []
    store = get_artifact_store()
    for item in messages:
        text = normalize_message_text(str(item or ""))
        if not text:
            continue
        summary = summarize_artifact_text(text)
        compacted.append(summary)
        if len(text) > len(summary):
            ref = store.store_text(
                text,
                artifact_type="plugin_output",
                trace_id=trace_id,
                source="plugin_send",
                force_file=True,
            )
            if ref is not None:
                artifacts.append(ref.to_dict())
    return compacted, artifacts


def _compact_artifacts(
    artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, dict) or not item.get("artifact_id"):
            continue
        payload: dict[str, Any] = {
            "artifact_id": normalize_message_text(str(item.get("artifact_id", ""))),
            "type": normalize_message_text(str(item.get("type", ""))),
            "summary": summarize_artifact_text(str(item.get("summary", "") or "")),
            "size": _safe_int(item.get("size")),
        }
        for key in ("mime_type", "path", "source"):
            value = normalize_message_text(str(item.get(key, "") or ""))
            if value:
                payload[key] = value
        compacted.append(payload)
    return compacted


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "build_command_observation",
]
