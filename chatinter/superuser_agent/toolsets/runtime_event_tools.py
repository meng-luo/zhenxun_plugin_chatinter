"""Runtime event stream inspection tools for superuser Agent."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ...runtime_events import (
    get_runtime_event,
    list_runtime_events,
    project_runtime_state,
    rebuild_runtime_event_index,
    replay_runtime_events,
)
from ..registry import register_superuser_tool
from .common import actor_from_context, tool_result


class RuntimeEventListTool:
    name = "runtime_event_list"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查看统一 RuntimeEvent 事件流，覆盖 AgentRun、"
                "tool progress、observation、artifact、approval、background job、"
                "TaskGraph/TaskLedger。用于长任务恢复、审计和定位卡住位置。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "run_id": {"type": ["string", "null"], "description": "可选 run_id。"},
                    "trace_id": {"type": ["string", "null"], "description": "可选 trace_id。"},
                    "kind": {
                        "type": ["string", "null"],
                        "description": "可选事件类型，例如 approval/background_job/tool_observation。",
                    },
                    "status": {
                        "type": ["string", "null"],
                        "description": "可选状态，例如 waiting/progress/completed/failed。",
                    },
                    "source_contains": {
                        "type": ["string", "null"],
                        "description": "按 source 文本过滤。",
                    },
                    "after_event_id": {
                        "type": ["string", "null"],
                        "description": "只列出该事件之后的事件。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多返回数量，默认 50，最大 300。",
                    },
                    "include_payload": {
                        "type": ["boolean", "null"],
                        "description": "是否包含压缩 payload，默认 false。",
                    },
                },
                "required": [
                    "run_id",
                    "trace_id",
                    "kind",
                    "status",
                    "source_contains",
                    "after_event_id",
                    "limit",
                    "include_payload",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        events = list_runtime_events(
            run_id=str(kwargs.get("run_id", "") or ""),
            trace_id=str(kwargs.get("trace_id", "") or ""),
            session_key=actor["session_key"],
            kind=str(kwargs.get("kind", "") or ""),
            status=str(kwargs.get("status", "") or ""),
            source_contains=str(kwargs.get("source_contains", "") or ""),
            after_event_id=str(kwargs.get("after_event_id", "") or ""),
            limit=_coerce_limit(kwargs.get("limit")),
            include_payload=bool(kwargs.get("include_payload") or False),
        )
        return tool_result(
            True,
            "runtime_events_listed",
            events=events,
            count=len(events),
        )


class RuntimeEventReadTool:
    name = "runtime_event_read"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：读取单个 RuntimeEvent 的完整压缩 payload。",
            parameters={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "runtime event_id。"}
                },
                "required": ["event_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        event_id = str(kwargs.get("event_id", "") or "").strip()
        if not event_id:
            return tool_result(False, "runtime_event_id_required")
        event = get_runtime_event(event_id)
        if event is None:
            return tool_result(False, "runtime_event_not_found", event_id=event_id)
        return tool_result(True, "runtime_event", event=event)


class RuntimeEventReplayTool:
    name = "runtime_event_replay"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：从 append-only RuntimeEvent JSONL 重放事件。"
                "用于 index 轮转后的长任务审计、恢复和定位。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "run_id": {"type": ["string", "null"], "description": "可选 run_id。"},
                    "trace_id": {"type": ["string", "null"], "description": "可选 trace_id。"},
                    "kind": {"type": ["string", "null"], "description": "可选事件类型。"},
                    "after_event_id": {
                        "type": ["string", "null"],
                        "description": "只返回此事件之后的事件。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多返回数量，默认 200，最大 5000。",
                    },
                    "include_payload": {
                        "type": ["boolean", "null"],
                        "description": "是否包含压缩 payload，默认 false。",
                    },
                },
                "required": [
                    "run_id",
                    "trace_id",
                    "kind",
                    "after_event_id",
                    "limit",
                    "include_payload",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        events = replay_runtime_events(
            run_id=str(kwargs.get("run_id", "") or ""),
            trace_id=str(kwargs.get("trace_id", "") or ""),
            session_key=actor["session_key"],
            kind=str(kwargs.get("kind", "") or ""),
            after_event_id=str(kwargs.get("after_event_id", "") or ""),
            limit=_coerce_replay_limit(kwargs.get("limit")),
            include_payload=bool(kwargs.get("include_payload") or False),
        )
        return tool_result(
            True,
            "runtime_events_replayed",
            events=events,
            count=len(events),
        )


class RuntimeStateProjectTool:
    name = "runtime_state_project"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：把 RuntimeEvent 流投影成可恢复状态，汇总 "
                "approval/background/artifact/task/observation refs。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "run_id": {"type": ["string", "null"], "description": "可选 run_id。"},
                    "trace_id": {"type": ["string", "null"], "description": "可选 trace_id。"},
                    "include_details": {
                        "type": ["boolean", "null"],
                        "description": "是否返回详细 observations/tool_calls，默认 true。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多重放事件数，默认 5000。",
                    },
                },
                "required": ["run_id", "trace_id", "include_details", "limit"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        projection = project_runtime_state(
            run_id=str(kwargs.get("run_id", "") or ""),
            trace_id=str(kwargs.get("trace_id", "") or ""),
            session_key=actor["session_key"],
            include_details=True
            if kwargs.get("include_details") is None
            else bool(kwargs.get("include_details")),
            limit=_coerce_replay_limit(kwargs.get("limit")),
        )
        return tool_result(
            True,
            "runtime_state_projected",
            projection=projection,
        )


class RuntimeEventIndexRebuildTool:
    name = "runtime_event_index_rebuild"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：从 RuntimeEvent JSONL 重建快速查询 index。",
            parameters={
                "type": "object",
                "properties": {
                    "max_events": {
                        "type": ["integer", "null"],
                        "description": "保留到 index 的最近事件数，默认 1200。",
                    }
                },
                "required": ["max_events"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        count = rebuild_runtime_event_index(
            max_events=_coerce_replay_limit(kwargs.get("max_events")),
        )
        return tool_result(True, "runtime_event_index_rebuilt", count=count)


def _coerce_limit(value: Any) -> int:
    try:
        return max(1, min(int(value or 50), 300))
    except (TypeError, ValueError):
        return 50


def _coerce_replay_limit(value: Any) -> int:
    try:
        return max(1, min(int(value or 5000), 5000))
    except (TypeError, ValueError):
        return 5000


register_superuser_tool(
    RuntimeEventListTool,
    category="runtime",
    risk="low",
    read_only=True,
    tags=("runtime", "events", "observation"),
)
register_superuser_tool(
    RuntimeEventReadTool,
    category="runtime",
    risk="low",
    read_only=True,
    tags=("runtime", "events", "observation"),
)
register_superuser_tool(
    RuntimeEventReplayTool,
    category="runtime",
    risk="low",
    read_only=True,
    tags=("runtime", "events", "replay"),
)
register_superuser_tool(
    RuntimeStateProjectTool,
    category="runtime",
    risk="low",
    read_only=True,
    tags=("runtime", "events", "projection", "resume"),
)
register_superuser_tool(
    RuntimeEventIndexRebuildTool,
    category="runtime",
    risk="low",
    read_only=True,
    tags=("runtime", "events", "index"),
)

__all__ = [
    "RuntimeEventIndexRebuildTool",
    "RuntimeEventListTool",
    "RuntimeEventReadTool",
    "RuntimeEventReplayTool",
    "RuntimeStateProjectTool",
]
