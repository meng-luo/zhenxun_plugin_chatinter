"""Final reply contract helpers for ChatInter agent runtime."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.tools import RunContext

from .route_text import normalize_message_text
from .superuser_agent.tool_guardrail import SUPERUSER_AGENT_MODES


def final_contract_text(
    state: Any,
    run_context: RunContext,
    final_text: str,
) -> str:
    extra = getattr(run_context, "extra", None)
    agent_mode = (
        normalize_message_text(str(extra.get("agent_mode", "") or ""))
        if isinstance(extra, dict)
        else ""
    )
    if agent_mode not in SUPERUSER_AGENT_MODES:
        return final_text
    return superuser_final_contract_reply(state, final_text)


def observation_contract_reply(observation: Any) -> str:
    output = observation.output if isinstance(observation.output, dict) else {}
    artifacts = _artifact_ids(output, getattr(observation, "artifacts", ()))
    prefix = "已完成" if bool(getattr(observation, "ok", False)) else "失败"
    detail = _observation_detail(observation, output)
    parts = [prefix]
    if detail:
        parts.append(detail)
    if artifacts:
        parts.append("artifact_id: " + "、".join(artifacts[:5]))
    return "；".join(parts) + "。"


def superuser_final_contract_reply(state: Any, final_text: str) -> str:
    observations = list(getattr(state, "observations", []) or [])
    if not observations:
        return final_text

    completed: list[str] = []
    failed: list[str] = []
    approvals: list[str] = []
    artifacts: list[str] = []
    changed_files: list[str] = []
    worktrees: list[str] = []

    for observation in observations[-8:]:
        output = observation.output if isinstance(observation.output, dict) else {}
        status = normalize_message_text(str(output.get("status", "") or ""))
        detail = _observation_detail(observation, output)
        label = _observation_label(observation, status, detail)
        if bool(output.get("approval_required")) or status == "approval_required":
            approval_id = _first_nested_text(
                output,
                ("approval_id",),
                ("approval", "approval_id"),
            )
            approvals.append(approval_id or label)
        elif bool(getattr(observation, "ok", False)):
            completed.append(label)
        else:
            failed.append(label)
        for artifact_id in _artifact_ids(output, getattr(observation, "artifacts", ())):
            _append_unique(artifacts, artifact_id)
        _collect_changed_files(observation, output, changed_files)
        _collect_worktrees(output, worktrees)

    for artifact_id in getattr(state, "artifact_refs", []) or []:
        _append_unique(artifacts, normalize_message_text(str(artifact_id or "")))

    parts: list[str] = []
    if completed:
        parts.append("已完成：" + "；".join(completed[:5]))
    if failed:
        parts.append("失败：" + "；".join(failed[:5]))
    if approvals:
        parts.append("需确认：" + "、".join(approvals[:5]))
    if changed_files:
        parts.append("修改文件：" + "、".join(changed_files[:8]))
    if worktrees:
        parts.append("worktree：" + "、".join(worktrees[:3]))
    if artifacts:
        parts.append("artifact_id：" + "、".join(artifacts[:8]))
    if not parts:
        return final_text
    return "\n".join(parts)


def _observation_label(observation: Any, status: str, detail: str) -> str:
    task_text = normalize_message_text(str(getattr(observation, "task_text", "") or ""))
    tool_name = normalize_message_text(str(getattr(observation, "tool_name", "") or ""))
    return (task_text or detail or status or tool_name or "工具调用")[:180]


def _append_unique(items: list[str], value: str) -> None:
    value = normalize_message_text(str(value or ""))
    if value and value not in items:
        items.append(value)


def _first_nested_text(output: dict[str, Any], *paths: tuple[str, ...]) -> str:
    for path in paths:
        value: Any = output
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        text = normalize_message_text(str(value or ""))
        if text:
            return text
    return ""


def _collect_changed_files(
    observation: Any,
    output: dict[str, Any],
    result: list[str],
) -> None:
    for file_item in _iter_patch_files(output.get("operation")):
        _append_unique(result, file_item)
    for file_item in _iter_patch_files(output.get("patch_operation")):
        _append_unique(result, file_item)
    tool_name = normalize_message_text(str(getattr(observation, "tool_name", "") or ""))
    if tool_name in {"write_file", "append_file", "replace_in_file"}:
        _append_unique(result, normalize_message_text(str(output.get("path") or "")))


def _iter_patch_files(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    files = value.get("files")
    if not isinstance(files, list | tuple):
        return []
    result: list[str] = []
    for item in files:
        if isinstance(item, dict):
            path = normalize_message_text(str(item.get("path") or ""))
            if path:
                result.append(path)
        elif isinstance(item, str) and item.strip():
            result.append(normalize_message_text(item))
    return result


def _collect_worktrees(output: dict[str, Any], result: list[str]) -> None:
    worktree = output.get("worktree")
    if isinstance(worktree, dict):
        _append_unique(
            result,
            _first_nested_text(worktree, ("worktree_id",), ("path",)),
        )
    isolation = output.get("isolation")
    if isinstance(isolation, dict):
        _append_unique(
            result,
            _first_nested_text(isolation, ("worktree_id",), ("worktree_path",)),
        )


def _observation_detail(observation: Any, output: dict[str, Any]) -> str:
    for key in (
        "summary",
        "messages_sent_summary",
        "stdout",
        "stderr",
        "error",
        "message",
        "status",
    ):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_message_text(value)[:500]
    error = normalize_message_text(str(getattr(observation, "error", "") or ""))
    if error:
        return error[:500]
    result = getattr(observation, "result", None)
    display = normalize_message_text(str(getattr(result, "display_content", "") or ""))
    return display[:500]


def _artifact_ids(
    output: dict[str, Any],
    observation_artifacts: tuple[dict[str, Any], ...],
) -> list[str]:
    result: list[str] = []

    def add(value: Any) -> None:
        artifact_id = normalize_message_text(str(value or ""))
        if artifact_id and artifact_id not in result:
            result.append(artifact_id)

    add(output.get("artifact_id"))
    artifact = output.get("artifact")
    if isinstance(artifact, dict):
        add(artifact.get("artifact_id"))
    groups = [output.get("artifacts"), observation_artifacts]
    for group in groups:
        if not isinstance(group, list | tuple):
            continue
        for item in group:
            if isinstance(item, dict):
                add(item.get("artifact_id"))
    return result
