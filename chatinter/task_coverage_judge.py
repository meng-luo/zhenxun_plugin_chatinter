"""Semantic task coverage judge for the ChatInter agent loop."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from zhenxun.services import logger
from zhenxun.services.llm import AI

from .route_text import normalize_message_text


class TaskCoverageResult(BaseModel):
    covered: bool = Field(default=True)
    missing_tasks: list[str] = Field(default_factory=list)
    unsupported_tasks: list[str] = Field(default_factory=list)
    reason: str = ""


class TaskLedgerItem(BaseModel):
    task_id: str = ""
    goal: str = ""
    intent_type: str = "unknown"
    requires_real_tool: bool = False
    expected_capabilities: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    reason: str = ""


class TaskLedgerPlanResult(BaseModel):
    tasks: list[TaskLedgerItem] = Field(default_factory=list)
    reason: str = ""


class TaskLedgerCoverageResult(BaseModel):
    covered: bool = Field(default=True)
    covered_task_ids: list[str] = Field(default_factory=list)
    missing_task_ids: list[str] = Field(default_factory=list)
    needs_capability_retrieval_task_ids: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    unsupported_tasks: list[dict[str, str]] = Field(default_factory=list)
    failed_tasks: list[dict[str, str]] = Field(default_factory=list)
    reason: str = ""


class TaskCoverageJudge:
    """LLM-backed validator that checks task coverage without local splitting."""

    def __init__(
        self,
        *,
        trace_id: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
    ) -> None:
        self.ai = AI(session_id=f"chatinter-coverage:{trace_id}")
        self.model_name = model_name
        self.generation_config = generation_config
        self.timeout = max(4.0, min(float(timeout or 12.0), 18.0))

    async def judge(
        self,
        *,
        original_message: str,
        observations: list[dict[str, Any]],
        final_reply: str,
        available_tools: list[dict[str, Any]],
        pending_tasks: list[str],
        mode: str,
        completed_tasks: list[str] | None = None,
    ) -> TaskCoverageResult:
        payload = {
            "mode": normalize_message_text(mode),
            "original_message": normalize_message_text(original_message),
            "observations": observations,
            "completed_tasks": [
                normalize_message_text(task)
                for task in (completed_tasks or [])
                if normalize_message_text(task)
            ],
            "final_reply": normalize_message_text(final_reply),
            "available_tools": available_tools,
            "pending_tasks": [
                normalize_message_text(task)
                for task in pending_tasks
                if normalize_message_text(task)
            ],
        }
        try:
            return await self.ai.generate_structured(
                json.dumps(payload, ensure_ascii=False),
                TaskCoverageResult,
                model=self.model_name,
                config=self.generation_config,
                instruction=_COVERAGE_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                auto_thinking=False,
            )
        except Exception as exc:
            logger.warning(f"[ChatInter] task coverage judge failed: {exc}")
            return TaskCoverageResult(covered=True, reason="judge_failed")

    async def plan_task_ledger(
        self,
        *,
        original_message: str,
        available_capabilities: list[dict[str, Any]],
    ) -> TaskLedgerPlanResult:
        payload = {
            "original_message": normalize_message_text(original_message),
            "available_capabilities": available_capabilities,
        }
        try:
            return await self.ai.generate_structured(
                json.dumps(payload, ensure_ascii=False),
                TaskLedgerPlanResult,
                model=self.model_name,
                config=self.generation_config,
                instruction=_TASK_LEDGER_PLAN_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                auto_thinking=False,
            )
        except Exception as exc:
            logger.warning(f"[ChatInter] task ledger planning failed: {exc}")
            return TaskLedgerPlanResult(reason="planner_failed")

    async def judge_task_ledger(
        self,
        *,
        task_ledger: dict[str, Any],
        capability_ledger: dict[str, Any],
        observations: list[dict[str, Any]],
        final_reply: str,
    ) -> TaskLedgerCoverageResult:
        payload = {
            "task_ledger": task_ledger,
            "capability_ledger": capability_ledger,
            "observations": observations,
            "final_reply": normalize_message_text(final_reply),
        }
        try:
            return await self.ai.generate_structured(
                json.dumps(payload, ensure_ascii=False),
                TaskLedgerCoverageResult,
                model=self.model_name,
                config=self.generation_config,
                instruction=_TASK_LEDGER_COVERAGE_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                auto_thinking=False,
            )
        except Exception as exc:
            logger.warning(f"[ChatInter] task ledger coverage failed: {exc}")
            return TaskLedgerCoverageResult(covered=True, reason="judge_failed")


_COVERAGE_INSTRUCTION = """
你是 ChatInter 的任务覆盖率验收器，只做结构化判断，不执行任务。

判断规则：
- 只检查用户原始消息里明确要求机器人完成的插件/动作/查询类任务。
- 普通聊天、寒暄、讨论概念、解释背景不算 missing task。
- 已完成任务必须能在 observations 里找到对应工具结果。
- 不要因为 final_reply 说完成就认为完成。
- completed_tasks 是从 ok=true observations 推导出的事实摘要。
- 判断覆盖时优先相信 completed_tasks 和 observations。
- observations 中 ok=true 的命令执行记录就是工具结果。
- 如果 messages_sent_summary 或 visible_output 存在，说明已有可见输出。
- 插件已发图/文本时必须相信 messages_sent_summary/visible_output。
- 不要因为原始 messages_sent 为空而判“没有吐出可见结果”。
- mode=initial_scan 时，把明确可执行任务放入 missing_tasks。
- mode=initial_scan 且纯聊天时返回 covered=true。
- mode=final_check 时，比较原始消息、pending_tasks、observations。
- mode=final_check 还要结合 final_reply 找出未被工具结果覆盖的任务。
- 如果任务明确存在但 available_tools 没有合适工具，放入 unsupported_tasks。
- 不要把缺少合适工具的任务放入 missing_tasks。
- 不要按“然后/最后/顺便”等词机械切分，只按语义任务判断。

只返回 JSON：
{
  "covered": true,
  "missing_tasks": [],
  "unsupported_tasks": [],
  "reason": ""
}
""".strip()


_TASK_LEDGER_PLAN_INSTRUCTION = """
你是 ChatInter 的任务账本规划器。
你的工作是把用户原始消息中的明确任务列成结构化 TaskLedger。

规则：
- 只列真实任务：插件动作、查询状态、生成/制作图片或文件。
- 也包括抽取随机结果、翻译/转换、需要机器人能力的操作。
- 不要把寒暄、语气、背景解释、普通讨论、概念咨询列成任务。
- 不要按“然后/最后/顺便”等连接词机械切分。
- 只按语义上独立的目标列任务。
- 不决定调用哪个工具。
- expected_capabilities 只填明显相关的 command_id/tool/head/plugin。
- 如果任务可以直接聊天回答且不需要真实工具，requires_real_tool=false。
- 如果任务必须依赖真实工具结果，requires_real_tool=true。
- 这包括“完成/查到/生成/制作/发送/签到”等承诺。
- task_id 使用 task_1、task_2 这种稳定短 ID。

只返回 JSON：
{
  "tasks": [
    {
      "task_id": "task_1",
      "goal": "",
      "intent_type": "query|generate|action|translate|status|random|chat|unknown",
      "requires_real_tool": false,
      "expected_capabilities": [],
      "acceptance_criteria": [],
      "reason": ""
    }
  ],
  "reason": ""
}
""".strip()


_TASK_LEDGER_COVERAGE_INSTRUCTION = """
你是 ChatInter 的任务覆盖验收器。
你只根据 TaskLedger、CapabilityLedger 和真实 observations 判断覆盖。
你不执行任务。

规则：
- 任务是否完成必须以 observations 中 ok=true 的真实工具结果为依据。
- final_reply 说“完成了”不能当证据。
- requires_real_tool=false 的聊天/解释任务，final_reply 已回答即可 covered。
- requires_real_tool=true 的任务，必须找到相关 observation。
- observation 的 command_id/tool/task_text/rendered_command 要能对应任务目标。
- 如果 CapabilityLedger 没有明显可用能力，但可通过 retrieve/catalog 补查。
- 此时优先把 task_id 放入 needs_capability_retrieval_task_ids。
- 同时给 retrieval_queries，不要直接判 unsupported。
- 只有多次 retrieve/catalog 仍无结果，或明显超出 bot 能力范围。
- 这种情况下才放入 unsupported_tasks。
- 如果有能力但还没执行或证据不足，放入 missing_task_ids。
- 不要新增任务；所有 task_id 必须来自 TaskLedger。

只返回 JSON：
{
  "covered": true,
  "covered_task_ids": [],
  "missing_task_ids": [],
  "needs_capability_retrieval_task_ids": [],
  "retrieval_queries": [],
  "unsupported_tasks": [{"task_id": "", "reason": ""}],
  "failed_tasks": [{"task_id": "", "reason": ""}],
  "reason": ""
}
""".strip()


__all__ = [
    "TaskCoverageJudge",
    "TaskCoverageResult",
    "TaskLedgerCoverageResult",
    "TaskLedgerItem",
    "TaskLedgerPlanResult",
]
