"""LLM-backed lightweight TaskGraph planner for superuser Agent mode."""

from __future__ import annotations

from typing import Any
import json

from pydantic import BaseModel, Field

from zhenxun.services import logger
from zhenxun.services.llm import AI

from .route_text import normalize_message_text
from .task_graph import TaskGraph, TaskGraphTask, fallback_task_graph


class PlannedTask(BaseModel):
    goal: str = Field(default="")
    required_tools: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class PlannerResult(BaseModel):
    should_plan: bool = Field(default=False)
    tasks: list[PlannedTask] = Field(default_factory=list)
    reason: str = ""


class AgentPlanner:
    """Create a compact, explicit task graph before the Agent loop starts."""

    def __init__(
        self,
        *,
        trace_id: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
    ) -> None:
        self.ai = AI(session_id=f"chatinter-planner:{trace_id}")
        self.model_name = model_name
        self.generation_config = generation_config
        self.timeout = max(4.0, min(float(timeout or 10.0), 16.0))

    async def plan(
        self,
        *,
        original_goal: str,
        available_tools: list[dict[str, Any]],
        resumed_graph: TaskGraph | None = None,
    ) -> TaskGraph | None:
        if resumed_graph is not None and resumed_graph.tasks:
            return resumed_graph
        goal = normalize_message_text(original_goal)
        if not goal:
            return None
        payload = {
            "original_goal": goal,
            "available_tools": available_tools[:80],
        }
        try:
            result = await self.ai.generate_structured(
                json.dumps(payload, ensure_ascii=False),
                PlannerResult,
                model=self.model_name,
                config=self.generation_config,
                instruction=_PLANNER_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                auto_thinking=False,
            )
        except Exception as exc:
            logger.warning(f"[ChatInter] agent planner failed: {exc}")
            return fallback_task_graph(
                original_goal=goal,
                available_tools=available_tools,
            )
        if not result.should_plan:
            return None
        tasks = _tasks_from_result(result)
        if not tasks:
            return fallback_task_graph(
                original_goal=goal,
                available_tools=available_tools,
            )
        graph = TaskGraph.create(original_goal=goal, tasks=tasks)
        graph.reason = normalize_message_text(result.reason)
        return graph


def _tasks_from_result(result: PlannerResult) -> list[TaskGraphTask]:
    tasks: list[TaskGraphTask] = []
    for index, task in enumerate(result.tasks[:8], start=1):
        goal = normalize_message_text(task.goal)
        if not goal:
            continue
        tasks.append(
            TaskGraphTask(
                task_id=f"task_{index}",
                goal=goal,
                required_tools=_text_list(task.required_tools),
                acceptance_criteria=_text_list(task.acceptance_criteria),
            )
        )
    return tasks


def _text_list(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = normalize_message_text(value)
        if text and text not in result:
            result.append(text)
    return result[:10]


_PLANNER_INSTRUCTION = """
你是超级用户 Agent 的轻量任务规划器。只输出任务图，不执行任务。

规划规则：
- 只有当用户请求需要真实工具、代码/文件/服务器操作、检查、修改、测试、后台任务、
  多步骤排查时 should_plan=true。
- 纯聊天、解释概念、一般建议、无需检查真实环境的问题 should_plan=false。
- 任务数量尽量少，通常 1-5 个；每个 task 必须有清晰 goal。
- required_tools 只填 available_tools 里出现过的 tool 名；不确定可留空。
- acceptance_criteria 写可验收事实，例如“读取目标文件并确认入口函数”、
  “生成 patch 并通过 py_compile”、“后台任务状态 completed 且 returncode=0”。
- 不要把“回复用户”单独列为任务；最终总结由 AgentRuntime 处理。

只返回 JSON：
{
  "should_plan": true,
  "tasks": [
    {
      "goal": "",
      "required_tools": [],
      "acceptance_criteria": []
    }
  ],
  "reason": ""
}
""".strip()


__all__ = ["AgentPlanner", "PlannedTask", "PlannerResult"]
