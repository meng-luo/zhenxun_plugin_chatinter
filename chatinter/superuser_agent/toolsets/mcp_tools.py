"""Superuser Agent tools for MCP runtime management."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ...mcp_runtime import ensure_mcp_config, get_mcp_runtime_manager
from ..permission_policy import decide_server
from ..registry import register_superuser_tool
from .common import (
    actor_from_context,
    approval_required_result,
    permission_denied_result,
    tool_result,
)


class MCPRuntimeStatusTool:
    name = "mcp_runtime_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查看 ChatInter MCP runtime 状态，包括 SDK "
                "是否可用、配置路径、已连接 server、工具数量和失败隔离状态。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ensure_config": {
                        "type": ["boolean", "null"],
                        "description": "是否确保默认 MCP 配置文件存在，默认 true。",
                    }
                },
                "required": ["ensure_config"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        if kwargs.get("ensure_config") is not False:
            ensure_mcp_config()
        status = await get_mcp_runtime_manager().status()
        return tool_result(True, "mcp_runtime_status", **status.to_payload())


class MCPRuntimeReloadTool:
    name = "mcp_runtime_reload"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：重载 MCP server 配置并重新发现工具。会断开并重连 "
                "MCP server，可能影响正在进行的 MCP 工具调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "server_names": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "可选 server 名列表；为空则重载全部 MCP server。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "重载原因，写入审计/确认上下文。",
                    },
                },
                "required": ["server_names", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        server_names = [
            str(item or "").strip()
            for item in (kwargs.get("server_names") or [])
            if str(item or "").strip()
        ]
        command = "mcp_runtime_reload " + (" ".join(server_names) if server_names else "*")
        permission = decide_server(command)
        payload = {"server_names": server_names, "reason": kwargs.get("reason") or ""}
        if permission.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="mcp_runtime_reload",
                payload=payload,
                permission=permission,
            )
        if permission.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="mcp_runtime_reload",
                payload=payload,
                permission=permission,
            )
        result = await get_mcp_runtime_manager().reload(server_names=server_names or None)
        return tool_result(True, "mcp_runtime_reloaded", **result.to_payload())


class MCPRuntimeRefreshTool:
    name = "mcp_runtime_refresh"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：刷新单个 MCP server 的工具列表。适合 server "
                "支持动态工具变更或收到工具列表异常时使用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "要刷新的 MCP server 名。",
                    }
                },
                "required": ["server_name"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        server_name = str(kwargs.get("server_name") or "").strip()
        if not server_name:
            return tool_result(False, "mcp_runtime_refresh_missing_server")
        status = await get_mcp_runtime_manager().refresh_server(server_name)
        if status is None:
            return tool_result(
                False,
                "mcp_runtime_server_not_loaded",
                server_name=server_name,
            )
        return tool_result(
            status.status == "connected",
            "mcp_runtime_refreshed",
            server=status.to_payload(),
        )


register_superuser_tool(
    MCPRuntimeStatusTool,
    category="mcp",
    risk="low",
    read_only=True,
    description="查看 MCP runtime 状态和配置。",
    tags=("mcp", "provider", "runtime"),
)
register_superuser_tool(
    MCPRuntimeReloadTool,
    category="mcp",
    risk="medium",
    read_only=False,
    approval_mode="policy",
    description="重载 MCP server 连接和工具列表。",
    tags=("mcp", "reload", "provider", "runtime"),
)
register_superuser_tool(
    MCPRuntimeRefreshTool,
    category="mcp",
    risk="low",
    read_only=True,
    description="刷新单个 MCP server 的工具列表。",
    tags=("mcp", "refresh", "provider", "runtime"),
)

__all__ = [
    "MCPRuntimeRefreshTool",
    "MCPRuntimeReloadTool",
    "MCPRuntimeStatusTool",
]
