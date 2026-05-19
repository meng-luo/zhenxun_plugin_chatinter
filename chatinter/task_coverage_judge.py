"""Semantic task coverage judge for the ChatInter agent loop."""

from __future__ import annotations

from typing import Any
import json

from pydantic import BaseModel, Field

from zhenxun.services import logger
from zhenxun.services.llm import AI

from .route_text import normalize_message_text


class TaskCoverageResult(BaseModel):
    covered: bool = Field(default=True)
    missing_tasks: list[str] = Field(default_factory=list)
    unsupported_tasks: list[str] = Field(default_factory=list)
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
    ) -> TaskCoverageResult:
        payload = {
            "mode": normalize_message_text(mode),
            "original_message": normalize_message_text(original_message),
            "observations": observations,
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


_COVERAGE_INSTRUCTION = """
你是 ChatInter 的任务覆盖率验收器，只做结构化判断，不执行任务。

判断规则：
- 只检查用户原始消息里明确要求机器人完成的插件/动作/查询类任务。
- 普通聊天、寒暄、讨论概念、解释背景不算 missing task。
- 已完成任务必须能在 observations 里找到对应工具结果；不要因为 final_reply 说完成就认为完成。
- observations 中 ok=true 的命令执行记录就是工具结果；messages_sent/artifacts 为空只表示没有捕获到可见输出，不代表任务未执行。
- 如果 mode=initial_scan，把原始消息中的明确可执行任务放入 missing_tasks；纯聊天返回 covered=true。
- 如果 mode=final_check，比较 original_message、pending_tasks、observations 和 final_reply，找出仍未被工具结果覆盖的任务。
- 如果任务明确存在但 available_tools 看起来没有合适工具，放入 unsupported_tasks，而不是 missing_tasks。
- 不要按“然后/最后/顺便”等词机械切分，只按语义任务判断。

只返回 JSON：
{
  "covered": true,
  "missing_tasks": [],
  "unsupported_tasks": [],
  "reason": ""
}
""".strip()


__all__ = [
    "TaskCoverageJudge",
    "TaskCoverageResult",
]
