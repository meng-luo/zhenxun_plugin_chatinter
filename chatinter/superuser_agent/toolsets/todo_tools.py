"""Claude-like Todo tools for superuser Agent planning and verification."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..registry import register_superuser_tool
from ..todo_store import read_todos, write_todos
from .common import actor_from_context, tool_result


class TodoReadTool:
    name = "todo_read"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：读取当前会话 Todo 列表。复杂工程任务应先维护 "
                "Todo，再按 Todo 推进工具调用和验收。"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        todo_list = read_todos(
            user_id=actor["user_id"],
            session_key=actor["session_key"],
        )
        return tool_result(True, "todo_read", todo_list=todo_list.public_payload())


class TodoWriteTool:
    name = "todo_write"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：写入/更新当前会话 Todo 列表。用于 Claude 风格的 "
                "读代码 -> 改代码 -> 跑测试 -> 回滚/总结闭环；每次只能有少量 "
                "in_progress Todo。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "新的 Todo 列表或要合并更新的 Todo 项。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "todo_id": {
                                    "type": ["string", "null"],
                                    "description": "已有 todo_id；新建可为空。",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "明确、可验收的任务内容。",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                        "cancelled",
                                    ],
                                },
                                "priority": {
                                    "type": ["string", "null"],
                                    "enum": ["low", "medium", "high"],
                                    "description": "优先级；不确定用 medium。",
                                },
                                "active_form": {
                                    "type": ["string", "null"],
                                    "description": (
                                        "进行中时的动词短语，例如"
                                        "“正在读取入口文件”。"
                                    ),
                                },
                                "related_tools": {
                                    "type": ["array", "null"],
                                    "items": {"type": "string"},
                                },
                                "related_artifacts": {
                                    "type": ["array", "null"],
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "todo_id",
                                "content",
                                "status",
                                "priority",
                                "active_form",
                                "related_tools",
                                "related_artifacts",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "replace": {
                        "type": ["boolean", "null"],
                        "description": (
                            "true 覆盖整个列表；false 按 todo_id "
                            "合并更新。默认 true。"
                        ),
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么更新 Todo。",
                    },
                },
                "required": ["todos", "replace", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        raw_todos = kwargs.get("todos")
        if not isinstance(raw_todos, list):
            return tool_result(
                False, "todo_invalid_input", error="todos must be a list"
            )
        try:
            todo_list = write_todos(
                user_id=actor["user_id"],
                session_key=actor["session_key"],
                todos=[item for item in raw_todos if isinstance(item, dict)],
                replace=True
                if kwargs.get("replace") is None
                else bool(kwargs.get("replace")),
            )
        except Exception as exc:
            return tool_result(False, "todo_write_failed", error=str(exc))
        return tool_result(
            True,
            "todo_written",
            todo_list=todo_list.public_payload(),
            reason=str(kwargs.get("reason", "") or ""),
            instruction=(
                "继续执行 in_progress 或 pending Todo。代码修改后用 patch/eval "
                "工具提供 observation，再把 Todo 标记为 completed。"
            ),
        )


register_superuser_tool(TodoReadTool, category="todo", risk="low", read_only=True)
register_superuser_tool(
    TodoWriteTool,
    category="todo",
    risk="low",
    read_only=False,
    destructive=False,
    side_effect="mutate",
    todo_relevant=True,
)

__all__ = ["TodoReadTool", "TodoWriteTool"]
