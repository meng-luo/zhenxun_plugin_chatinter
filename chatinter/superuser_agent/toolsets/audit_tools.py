"""Audit query tools for the superuser private Agent scenario."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..audit_log import audit_log_path, query_audit_events, record_audit_event
from ..registry import register_superuser_tool
from .common import actor_from_context, tool_result


class AuditLogQueryTool:
    name = "audit_log_query"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查询 ChatInter Agent 最近的审计日志，"
                "用于追踪审批、命令执行、文件写入、后台任务等操作。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多返回条数，默认 50，最大 200。",
                    },
                    "action": {
                        "type": ["string", "null"],
                        "description": "按 action 精确过滤，例如 shell_command。",
                    },
                    "event": {
                        "type": ["string", "null"],
                        "description": "按 event 精确过滤，例如 operation_executed。",
                    },
                    "current_session_only": {
                        "type": ["boolean", "null"],
                        "description": "是否只查询当前私聊会话，默认 true。",
                    },
                    "contains": {
                        "type": ["string", "null"],
                        "description": (
                            "可选全文过滤，例如 approval_id、operation_id、" "task_id。"
                        ),
                    },
                },
                "required": [
                    "limit",
                    "action",
                    "event",
                    "current_session_only",
                    "contains",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        limit = _coerce_int(kwargs.get("limit"), default=50, lower=1, upper=200)
        action = str(kwargs.get("action", "") or "").strip()
        event = str(kwargs.get("event", "") or "").strip()
        contains = str(kwargs.get("contains", "") or "").strip()
        current_session_only = kwargs.get("current_session_only")
        if current_session_only is None:
            current_session_only = True
        entries = query_audit_events(
            limit=limit,
            user_id=actor["user_id"] if current_session_only else "",
            session_key=actor["session_key"] if current_session_only else "",
            action=action,
            event=event,
            contains=contains,
        )
        record_audit_event(
            event="audit_queried",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="audit_log_query",
            payload={
                "limit": limit,
                "action": action,
                "event": event,
                "contains": contains,
                "current_session_only": bool(current_session_only),
            },
            result={"count": len(entries)},
        )
        return tool_result(
            True,
            "audit_log_queried",
            path=str(audit_log_path()),
            entries=entries,
            count=len(entries),
        )


def _coerce_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        return max(lower, min(int(value or default), upper))
    except (TypeError, ValueError):
        return default


register_superuser_tool(AuditLogQueryTool)

__all__ = ["AuditLogQueryTool"]
