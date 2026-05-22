"""TaskGraph verifier for superuser Agent observations and final checks."""

from __future__ import annotations

from typing import Any
import json

from pydantic import BaseModel, Field

from zhenxun.services import logger
from zhenxun.services.llm import AI

from .agent_state import AgentObservation
from .route_text import normalize_message_text
from .task_graph import (
    TASK_STATUS_BLOCKED,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_IN_PROGRESS,
    TASK_STATUS_PENDING,
    TaskGraph,
)


class TaskVerificationItem(BaseModel):
    task_id: str = ""
    status: str = TASK_STATUS_PENDING
    reason: str = ""


class TaskVerificationResult(BaseModel):
    updates: list[TaskVerificationItem] = Field(default_factory=list)
    missing_tasks: list[str] = Field(default_factory=list)
    reason: str = ""


class AgentVerifier:
    """Update TaskGraph from observations and block premature final replies."""

    def __init__(
        self,
        *,
        trace_id: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
    ) -> None:
        self.ai = AI(session_id=f"chatinter-verifier:{trace_id}")
        self.model_name = model_name
        self.generation_config = generation_config
        self.timeout = max(4.0, min(float(timeout or 10.0), 16.0))

    async def verify_observation(
        self,
        *,
        graph: TaskGraph | None,
        observation: AgentObservation,
        available_tools: list[dict[str, Any]],
    ) -> TaskVerificationResult:
        if graph is None or not graph.tasks:
            return TaskVerificationResult(reason="no_task_graph")
        payload = {
            "mode": "after_observation",
            "task_graph": graph.to_public_payload(),
            "observation": _observation_payload(observation),
            "available_tools": available_tools[:40],
        }
        result = await self._judge(payload)
        _apply_updates(graph, result, observation=observation)
        return result

    async def verify_final(
        self,
        *,
        graph: TaskGraph | None,
        final_reply: str,
        available_tools: list[dict[str, Any]],
    ) -> TaskVerificationResult:
        if graph is None or not graph.tasks:
            return TaskVerificationResult(reason="no_task_graph")
        graph.refresh_status()
        if not graph.incomplete_tasks:
            return TaskVerificationResult(reason="all_tasks_completed")
        payload = {
            "mode": "final_check",
            "task_graph": graph.to_public_payload(),
            "final_reply": normalize_message_text(final_reply),
            "available_tools": available_tools[:40],
        }
        result = await self._judge(payload)
        _apply_updates(graph, result)
        if not result.missing_tasks:
            result.missing_tasks.extend(
                task.goal for task in graph.incomplete_tasks[:8] if task.goal
            )
        return result

    async def _judge(self, payload: dict[str, Any]) -> TaskVerificationResult:
        try:
            return await self.ai.generate_structured(
                json.dumps(payload, ensure_ascii=False),
                TaskVerificationResult,
                model=self.model_name,
                config=self.generation_config,
                instruction=_VERIFIER_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                auto_thinking=False,
            )
        except Exception as exc:
            logger.warning(f"[ChatInter] agent verifier failed: {exc}")
            return TaskVerificationResult(
                missing_tasks=_fallback_missing_tasks(payload),
                reason="verifier_failed",
            )


def _apply_updates(
    graph: TaskGraph,
    result: TaskVerificationResult,
    *,
    observation: AgentObservation | None = None,
) -> None:
    for update in result.updates:
        task = graph.task_by_id(update.task_id)
        if task is None:
            continue
        status = _normalize_status(update.status)
        if status:
            task.status = status
        reason = normalize_message_text(update.reason)
        if reason:
            task.reason = reason
        if observation is not None:
            task.add_observation(_observation_payload(observation))
    graph.refresh_status()


def _observation_payload(observation: AgentObservation) -> dict[str, Any]:
    output = observation.output or {}
    return {
        "ok": observation.ok,
        "tool_name": observation.tool_name,
        "command_id": observation.command_id,
        "rendered_command": observation.rendered_command,
        "matched_plugin": observation.matched_plugin,
        "task_text": observation.task_text,
        "status": output.get("status", "success" if observation.ok else "failed"),
        "need_continue": observation.need_continue,
        "remaining_task_hint": observation.remaining_task_hint,
        "error": observation.error,
        "messages_sent_summary": normalize_message_text(
            str(output.get("messages_sent_summary", "") or "")
        ),
        "visible_output": bool(output.get("visible_output")),
        "artifacts": output.get("artifacts", []),
        "messages_sent": _compact_list(output.get("messages_sent"), limit=4),
    }


def _compact_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[str] = []
    for item in value[: max(1, limit)]:
        text = normalize_message_text(str(item or ""))
        if text:
            result.append(text[:240])
    return result


def _normalize_status(status: str) -> str:
    normalized = normalize_message_text(status)
    if normalized in {
        TASK_STATUS_PENDING,
        TASK_STATUS_IN_PROGRESS,
        TASK_STATUS_COMPLETED,
        TASK_STATUS_BLOCKED,
        TASK_STATUS_FAILED,
    }:
        return normalized
    return ""


def _fallback_missing_tasks(payload: dict[str, Any]) -> list[str]:
    graph = payload.get("task_graph") if isinstance(payload, dict) else {}
    tasks = graph.get("tasks", []) if isinstance(graph, dict) else []
    missing: list[str] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        if str(item.get("status", "")) == TASK_STATUS_COMPLETED:
            continue
        goal = normalize_message_text(str(item.get("goal", "") or ""))
        if goal:
            missing.append(goal)
    return missing[:8]


_VERIFIER_INSTRUCTION = """
你是超级用户 Agent 的任务图验收器，只做结构化判断。

判断规则：
- task 完成必须有 observation 支持，不能因为 final_reply 声称完成就算完成。
- after_observation：根据 observation 判断它证明了哪些 task 的 acceptance_criteria。
- final_check：返回仍未完成、未被 observation 证明的任务。
- 如果工具返回 approval_required，相关任务通常是 blocked/in_progress，而不是 completed。
- 如果后台任务刚 started，相关任务是 in_progress，直到 status completed 且 returncode=0。
- 如果工具失败但可重试，任务保持 in_progress；不可重试或权限拒绝可 blocked/failed。
- 不要创造新任务；missing_tasks 只来自 task_graph 中未完成的 goal。

只返回 JSON：
{
  "updates": [
    {"task_id": "task_1", "status": "completed", "reason": ""}
  ],
  "missing_tasks": [],
  "reason": ""
}
""".strip()


__all__ = [
    "AgentVerifier",
    "TaskVerificationItem",
    "TaskVerificationResult",
]
