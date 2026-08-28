"""Read-only parallel investigation tool for the Superuser Agent."""

from __future__ import annotations

from typing import Any

from ...llm_compat import ToolDefinition, ToolResult
from .common import tool_result


class DelegateTasksTool:
    name = "delegate_tasks"
    read_only = True

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "并行委派两个相互独立、需要多步只读调查的子任务；"
                "父 Agent 仍负责最终判断、修改和验证。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "调查范围、问题和预期结论。",
                                }
                            },
                            "required": ["task"],
                            "additionalProperties": False,
                        },
                    },
                    "artifact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                        "description": "子任务可读取的当前会话 artifact ID。",
                    },
                },
                "required": ["tasks"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        parent_runtime = (
            getattr(context, "extra", {}).get("_subagent_parent_runtime")
            if context is not None and isinstance(getattr(context, "extra", None), dict)
            else None
        )
        if parent_runtime is None:
            return tool_result(
                False,
                "delegation_unavailable",
                error="当前运行上下文不支持子任务委派。",
            ).as_fatal()
        if int(getattr(parent_runtime, "_delegation_batches_succeeded", 0)) >= 1:
            return tool_result(
                False,
                "delegation_limit_reached",
                error="本轮已完成一次并行调查，请使用已有证据继续。",
            ).as_fatal()

        from ..subagent import run_delegated_tasks

        return await run_delegated_tasks(
            parent_runtime=parent_runtime,
            tasks=kwargs.get("tasks"),
            artifact_ids=kwargs.get("artifact_ids"),
        )


__all__ = ["DelegateTasksTool"]
