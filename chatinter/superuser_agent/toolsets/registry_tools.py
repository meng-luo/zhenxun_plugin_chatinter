"""Superuser Agent registry inspection tools."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..registry import (
    build_superuser_agent_tool_bundle,
    invalidate_superuser_tool_check_cache,
    register_superuser_tool,
    superuser_tool_cards,
)
from ..tool_preset import get_session_tool_preset, preset_allows_card
from .common import actor_from_context, tool_result


class ToolRegistryStatusTool:
    name = "tool_registry_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查看 Agent 工具注册表、分类、风险、approval "
                "策略和可用性；也可按 query/category 搜索并注入长尾工具。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": ["string", "null"],
                        "description": "可选工具搜索词，例如 文件/patch/git/压测。",
                    },
                    "category": {
                        "type": ["string", "null"],
                        "description": (
                            "可选分类过滤，例如 "
                            "file/patch/eval/background/artifact。"
                        ),
                    },
                    "inject": {
                        "type": ["boolean", "null"],
                        "description": (
                            "是否把匹配到的长尾工具注入当前 Agent，默认 false。"
                        ),
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
                "required": [
                    "query",
                    "category",
                    "inject",
                    "available_only",
                    "invalidate_cache",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        if bool(kwargs.get("invalidate_cache") or False):
            invalidate_superuser_tool_check_cache()
        query = str(kwargs.get("query", "") or "").strip()
        category = str(kwargs.get("category", "") or "").strip()
        inject = bool(kwargs.get("inject") or False)
        available_only = (
            True
            if kwargs.get("available_only") is None
            else bool(kwargs.get("available_only"))
        )
        preset = get_session_tool_preset(str(getattr(context, "session_id", "") or ""))
        selected_names = _selected_tool_names(
            query=query,
            category=category,
            preset=preset,
        )
        cards = [
            card.public_payload()
            for card in superuser_tool_cards(available_only=available_only)
            if (not category or card.category == category)
            and (not selected_names or card.name in selected_names)
            and preset_allows_card(preset, card)
        ]
        injected: list[str] = []
        if inject:
            injected = _inject_tools(context, query=query, category=category)
        return tool_result(
            True,
            "tool_registry_status",
            tools=cards,
            injected_tools=injected,
            count=len(cards),
            category=category,
            query=query,
            available_only=available_only,
        )


def _selected_tool_names(*, query: str, category: str, preset: str) -> set[str]:
    search_text = query or category
    if not search_text:
        return set()
    bundle = build_superuser_agent_tool_bundle(message_text=search_text, limit=16)
    return {card.name for card in bundle.cards if preset_allows_card(preset, card)}


def _inject_tools(
    context: Any | None,
    *,
    query: str,
    category: str,
) -> list[str]:
    extra = getattr(context, "extra", None)
    capability_registry = (
        extra.get("capability_registry") if isinstance(extra, dict) else None
    )
    if capability_registry is None or not hasattr(
        capability_registry, "register_superuser_tools"
    ):
        return []
    search_text = query or category
    if not search_text:
        return []
    preset = get_session_tool_preset(str(getattr(context, "session_id", "") or ""))
    bundle = build_superuser_agent_tool_bundle(message_text=search_text, limit=16)
    if category:
        allowed = {card.name for card in bundle.cards if card.category == category}
        tools = {name: tool for name, tool in bundle.tools.items() if name in allowed}
        cards = tuple(card for card in bundle.cards if card.name in tools)
    else:
        tools = bundle.tools
        cards = bundle.cards
    cards = tuple(card for card in cards if preset_allows_card(preset, card))
    allowed_by_preset = {card.name for card in cards}
    tools = {name: tool for name, tool in tools.items() if name in allowed_by_preset}
    existing = (
        set(capability_registry.executable_tool_map())
        if hasattr(capability_registry, "executable_tool_map")
        else set()
    )
    tools = {name: tool for name, tool in tools.items() if name not in existing}
    cards = tuple(card for card in cards if card.name in tools)
    if not tools:
        return []
    capability_registry.register_superuser_tools(tools, cards=cards)
    return sorted(tools)


register_superuser_tool(
    ToolRegistryStatusTool,
    category="registry",
    risk="low",
    read_only=True,
)

__all__ = ["ToolRegistryStatusTool"]
