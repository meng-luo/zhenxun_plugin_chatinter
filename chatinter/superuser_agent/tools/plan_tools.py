"""Persistent task plan tool for the Superuser runtime."""

from __future__ import annotations

from typing import Any

from ...llm_compat import ToolDefinition, ToolResult
from .common import tool_result

_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


class PlanTool:
    name = "plan"
    read_only = True

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="读取或替换当前任务的结构化执行计划。",
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": ["array", "null"],
                        "description": "完整计划；省略时只读取，空数组清除计划。",
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": sorted(_STATUSES),
                                },
                            },
                            "required": ["id", "content", "status"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        extra = getattr(context, "extra", None)
        plan = extra.get("plan_items") if isinstance(extra, dict) else None
        if not isinstance(plan, list):
            return tool_result(False, "plan_state_unavailable")
        if "items" not in kwargs or kwargs.get("items") is None:
            return tool_result(True, "plan_read", items=list(plan))
        items, error = _normalize_items(kwargs.get("items"))
        if error:
            return tool_result(False, "invalid_plan", error=error)
        plan[:] = items
        return tool_result(True, "plan_updated", items=list(plan))


def _normalize_items(value: Any) -> tuple[list[dict[str, str]], str]:
    if not isinstance(value, list) or len(value) > 20:
        return [], "items must be an array with at most 20 entries"
    normalized: list[dict[str, str]] = []
    ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            return [], f"items[{index}] must be an object"
        item_id = str(item.get("id", "") or "").strip()
        content = str(item.get("content", "") or "").strip()
        status = str(item.get("status", "") or "").strip()
        if not item_id or len(item_id) > 64 or item_id in ids:
            return [], f"items[{index}].id is invalid or duplicated"
        if not content or len(content) > 500:
            return [], f"items[{index}].content must contain 1-500 characters"
        if status not in _STATUSES:
            return [], f"items[{index}].status is invalid"
        ids.add(item_id)
        normalized.append({"id": item_id, "content": content, "status": status})
    return normalized, ""


__all__ = ["PlanTool"]
