from __future__ import annotations

import re
from typing import Any

from .route_text import normalize_message_text
from .task_ledger import (
    TASK_STATUS_COMPLETED,
    CapabilityLedger,
    TaskLedger,
    TaskLedgerEntry,
)

_CQ_PATTERN = re.compile(r"\[CQ:[^\]]+\]")


def synthesize_local_task_ledger_reply(
    *,
    task_ledger: TaskLedger,
    observations: list[Any],
    capability_ledger: CapabilityLedger | None = None,
    looks_like_multi_task_turn: bool,
) -> str:
    if not _can_synthesize(
        task_ledger=task_ledger,
        observations=observations,
        capability_ledger=capability_ledger,
        looks_like_multi_task_turn=looks_like_multi_task_turn,
    ):
        return ""

    lines: list[str] = []
    for task in task_ledger.tasks:
        observation = _observation_for_task(task, observations)
        if observation is None:
            continue
        output = _observation_output(observation)
        if output:
            lines.append(f"{_goal_label(task.goal)}：{output}")
        else:
            lines.append(_goal_label(task.goal))
    if not lines:
        return ""
    return "已完成：" + "；".join(lines)


def can_silently_finish_task_ledger(
    *,
    task_ledger: TaskLedger,
    observations: list[Any],
    capability_ledger: CapabilityLedger | None = None,
    looks_like_multi_task_turn: bool,
) -> bool:
    return _can_synthesize(
        task_ledger=task_ledger,
        observations=observations,
        capability_ledger=capability_ledger,
        looks_like_multi_task_turn=looks_like_multi_task_turn,
    )


def _can_synthesize(
    *,
    task_ledger: TaskLedger,
    observations: list[Any],
    capability_ledger: CapabilityLedger | None,
    looks_like_multi_task_turn: bool,
) -> bool:
    tasks = list(task_ledger.tasks or [])
    if not tasks or any(task.status != TASK_STATUS_COMPLETED for task in tasks):
        return False
    if len(tasks) < 2 and not _has_bot_state_capability(tasks, capability_ledger):
        return False
    if not looks_like_multi_task_turn and not _has_bot_state_capability(
        tasks, capability_ledger
    ):
        return False
    for task in tasks:
        if _observation_for_task(task, observations) is None:
            return False
    if len(observations) == 1 and len(tasks) > 1:
        return _has_bot_state_capability(tasks, capability_ledger)
    return True


def _has_bot_state_capability(
    tasks: list[TaskLedgerEntry],
    capability_ledger: CapabilityLedger | None,
) -> bool:
    if capability_ledger is None:
        return False
    for task in tasks:
        for command_id in task.expected_capabilities:
            entry = capability_ledger.by_command_id(command_id)
            if entry is None or entry.source_of_truth != "bot_state":
                return False
    return bool(tasks)


def _observation_for_task(task: TaskLedgerEntry, observations: list[Any]) -> Any | None:
    capabilities = {normalize_message_text(item) for item in task.expected_capabilities}
    goal = normalize_message_text(task.goal)
    for observation in observations:
        command_id = normalize_message_text(str(getattr(observation, "command_id", "")))
        task_text = normalize_message_text(str(getattr(observation, "task_text", "")))
        if command_id and command_id in capabilities:
            return observation
        if goal and task_text and (goal in task_text or task_text in goal):
            return observation
    if len(observations) == 1 and ("observation" in task.covered_by or capabilities):
        return observations[0]
    return None


def _observation_output(observation: Any) -> str:
    output = getattr(observation, "output", None)
    if isinstance(output, dict):
        artifacts = output.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                if artifact.get("type") == "plugin_output":
                    text = _clean_output(str(artifact.get("summary", "") or ""))
                    if text:
                        return text
        for key in ("messages_sent_summary", "visible_output"):
            value = output.get(key)
            if isinstance(value, str) and value:
                return _clean_output(value)
    return _clean_output(str(getattr(observation, "messages_sent_summary", "") or ""))


def _clean_output(value: str) -> str:
    text = _CQ_PATTERN.sub("", str(value or ""))
    if "plugin_output:" in text:
        text = text.split("plugin_output:", 1)[1]
    text = " ".join(text.replace("|", " ").split()).strip()
    return text[:220]


def _goal_label(goal: str) -> str:
    text = normalize_message_text(goal).rstrip("。.!！")
    return text


__all__ = [
    "can_silently_finish_task_ledger",
    "synthesize_local_task_ledger_reply",
]
