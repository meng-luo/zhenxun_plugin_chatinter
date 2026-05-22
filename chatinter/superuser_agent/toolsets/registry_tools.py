"""Superuser Agent registry inspection tools."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..registry import (
    invalidate_superuser_tool_check_cache,
    register_superuser_tool,
    superuser_tool_cards,
)
from .common import actor_from_context, tool_result


class ToolRegistryStatusTool:
    name = "tool_registry_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查看 Agent 工具注册表、分类、风险、approval "
                "策略和可用性。适合在复杂任务前确认可用工具。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": ["string", "null"],
                        "description": "可选分类过滤，例如 file/patch/eval/background/artifact。",
                    },
                    "available_only": {
                        "type": ["boolean", "null"],
                        "description": "是否只列可用工具，默认 true。",
                    },
                    "invalidate_cache": {
                        "type": ["boolean", "null"],
                        "description": "是否先清理 check_fn 可用性缓存，默认 false。",
                    },
                },
                "required": ["category", "available_only", "invalidate_cache"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        if bool(kwargs.get("invalidate_cache") or False):
            invalidate_superuser_tool_check_cache()
        category = str(kwargs.get("category", "") or "").strip()
        available_only = (
            True
            if kwargs.get("available_only") is None
            else bool(kwargs.get("available_only"))
        )
        cards = [
            card.public_payload()
            for card in superuser_tool_cards(available_only=available_only)
            if not category or card.category == category
        ]
        return tool_result(
            True,
            "tool_registry_status",
            tools=cards,
            count=len(cards),
            category=category,
            available_only=available_only,
        )


register_superuser_tool(
    ToolRegistryStatusTool,
    category="registry",
    risk="low",
    read_only=True,
)

__all__ = ["ToolRegistryStatusTool"]
