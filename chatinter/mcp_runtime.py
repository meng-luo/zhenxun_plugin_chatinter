"""Runtime MCP server integration for ChatInter.

This module intentionally sits below CapabilityRegistry.  It owns the concrete
MCP lifecycle (connect, discover, reload, reconnect and failure isolation) and
returns ordinary ToolExecutable objects that the unified capability layer can
expose safely.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from zhenxun.services import logger
from zhenxun.services.llm.types.models import ToolDefinition, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .route_text import normalize_message_text

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is present in normal deployments.
    yaml = None  # type: ignore[assignment]

_CONFIG_PATH = Path("data") / "config.yaml"
_CONFIG_SECTION = "chatinter"
_CONFIG_KEY = "MCP_SERVERS"
_DEFAULT_CONNECT_TIMEOUT = 12.0
_DEFAULT_TOOL_TIMEOUT = 60.0
_DEFAULT_FAILURE_ISOLATION_SECONDS = 60.0
_MAX_RESULT_CHARS = 60000


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    enabled: bool = True
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    url: str = ""
    transport: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = _DEFAULT_CONNECT_TIMEOUT
    sse_read_timeout: float = 300.0
    session_read_timeout: float = 60.0
    tool_timeout: float = _DEFAULT_TOOL_TIMEOUT
    include_tools: frozenset[str] = field(default_factory=frozenset)
    exclude_tools: frozenset[str] = field(default_factory=frozenset)
    max_tools: int = 80
    failure_isolation_seconds: float = _DEFAULT_FAILURE_ISOLATION_SECONDS

    @property
    def has_endpoint(self) -> bool:
        return bool(self.url or self.command)

    def signature(self) -> str:
        payload = {
            "enabled": self.enabled,
            "command": self.command,
            "args": list(self.args),
            "env": self.env,
            "cwd": self.cwd,
            "url": self.url,
            "transport": self.transport,
            "headers": self.headers,
            "timeout": self.timeout,
            "sse_read_timeout": self.sse_read_timeout,
            "session_read_timeout": self.session_read_timeout,
            "tool_timeout": self.tool_timeout,
            "include_tools": sorted(self.include_tools),
            "exclude_tools": sorted(self.exclude_tools),
            "max_tools": self.max_tools,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MCPServerStatus:
    name: str
    enabled: bool
    status: str
    tool_count: int = 0
    error: str = ""
    failure_count: int = 0
    isolated_until: float = 0.0
    config_signature: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "status": self.status,
            "tool_count": self.tool_count,
            "error": self.error,
            "failure_count": self.failure_count,
            "isolated_until": round(self.isolated_until, 3) if self.isolated_until else 0,
            "config_signature": self.config_signature[:12],
        }


@dataclass
class MCPDiscoveryResult:
    tools_by_server: dict[str, dict[str, ToolExecutable]] = field(default_factory=dict)
    statuses: list[MCPServerStatus] = field(default_factory=list)
    sdk_available: bool = False
    config_path: str = ""
    error: str = ""

    @property
    def tool_count(self) -> int:
        return sum(len(tools) for tools in self.tools_by_server.values())

    def to_payload(self) -> dict[str, Any]:
        return {
            "sdk_available": self.sdk_available,
            "config_path": self.config_path,
            "tool_count": self.tool_count,
            "error": self.error,
            "servers": [status.to_payload() for status in self.statuses],
        }


class MCPToolExecutable:
    """ToolExecutable wrapper around one remote MCP tool."""

    def __init__(self, *, server: "MCPServerConnection", raw_tool: Any) -> None:
        self.server = server
        self.raw_tool = raw_tool
        self.raw_name = _tool_name(raw_tool)
        self.name = self.raw_name
        self.description = _tool_description(raw_tool)
        self.input_schema = normalize_mcp_input_schema(_tool_input_schema(raw_tool))

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.raw_name,
            description=self.description or f"MCP tool {self.raw_name}",
            parameters=self.input_schema,
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        _ = context
        return await self.server.call_tool(self.raw_name, kwargs)


class MCPServerConnection:
    """One isolated MCP server connection."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.status = MCPServerStatus(
            name=config.name,
            enabled=config.enabled,
            status="new",
            config_signature=config.signature(),
        )
        self._exit_stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self._tools: list[Any] = []
        self._rpc_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._sdk: dict[str, Any] | None = None

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def should_skip_for_failure_isolation(self) -> bool:
        return bool(self.status.isolated_until and time.time() < self.status.isolated_until)

    async def ensure_connected(self, *, force: bool = False) -> None:
        if not self.config.enabled:
            self.status.status = "disabled"
            return
        if not self.config.has_endpoint:
            self._mark_failed("missing_endpoint", isolate=False)
            return
        if self.should_skip_for_failure_isolation() and not force:
            self.status.status = "isolated"
            return
        if self._session is not None and not force:
            return
        async with self._connect_lock:
            if self._session is not None and not force:
                return
            await self.close()
            self._sdk = _load_mcp_sdk()
            if not self._sdk["available"]:
                self._mark_failed(str(self._sdk["error"]), isolate=False)
                return
            try:
                await asyncio.wait_for(
                    self._connect_with_sdk(self._sdk),
                    timeout=max(float(self.config.timeout or _DEFAULT_CONNECT_TIMEOUT), 1.0),
                )
                await self.refresh_tools(force=True)
                self.status.status = "connected"
                self.status.error = ""
                self.status.isolated_until = 0.0
            except Exception as exc:
                await self.close()
                self._mark_failed(f"{type(exc).__name__}: {exc}")

    async def refresh_tools(self, *, force: bool = False) -> None:
        if self._session is None:
            await self.ensure_connected(force=force)
            return
        async with self._refresh_lock:
            try:
                async with self._rpc_lock:
                    result = await self._session.list_tools()
                tools = list(getattr(result, "tools", []) or [])
                self._tools = _filter_tools(tools, self.config)
                self.status.tool_count = len(self._tools)
                self.status.status = "connected"
                self.status.error = ""
            except Exception as exc:
                self._mark_failed(f"{type(exc).__name__}: {exc}")
                await self.close()

    def build_executables(self) -> dict[str, ToolExecutable]:
        return {
            _tool_name(tool): MCPToolExecutable(server=self, raw_tool=tool)
            for tool in self._tools
            if _tool_name(tool)
        }

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if self._session is None:
            await self.ensure_connected(force=True)
        if self._session is None:
            return ToolResult(
                output={
                    "ok": False,
                    "status": "mcp_server_unavailable",
                    "server": self.config.name,
                    "tool": tool_name,
                    "error": self.status.error,
                    "retryable": True,
                },
                display_content="mcp_server_unavailable",
            )

        async def _call_once() -> Any:
            async with self._rpc_lock:
                read_timeout = timedelta(
                    seconds=max(float(self.config.tool_timeout or _DEFAULT_TOOL_TIMEOUT), 1.0)
                )
                try:
                    return await self._session.call_tool(
                        name=tool_name,
                        arguments=arguments or {},
                        read_timeout_seconds=read_timeout,
                    )
                except TypeError:
                    return await self._session.call_tool(
                        name=tool_name,
                        arguments=arguments or {},
                    )

        try:
            result = await _call_once()
        except Exception as exc:
            if _looks_like_closed_resource(exc):
                logger.warning(
                    f"[ChatInter MCP] {self.config.name}.{tool_name} 连接已关闭，尝试重连",
                    "mcp_runtime",
                )
                await self.ensure_connected(force=True)
                try:
                    result = await _call_once()
                except Exception as retry_exc:
                    self._mark_failed(f"{type(retry_exc).__name__}: {retry_exc}")
                    return _mcp_error_result(self.config.name, tool_name, retry_exc)
            else:
                self._mark_failed(f"{type(exc).__name__}: {exc}", isolate=False)
                return _mcp_error_result(self.config.name, tool_name, exc)

        payload = _serialize_mcp_result(result)
        return ToolResult(
            output={
                "ok": True,
                "status": "mcp_tool_completed",
                "server": self.config.name,
                "tool": tool_name,
                "result": payload,
            },
            display_content=_mcp_display_summary(payload),
        )

    async def close(self) -> None:
        self._session = None
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as exc:
                logger.debug(
                    f"[ChatInter MCP] 关闭 MCP server {self.config.name} 失败: {exc}",
                    "mcp_runtime",
                )
        self._exit_stack = None

    async def _connect_with_sdk(self, sdk: dict[str, Any]) -> None:
        stack = AsyncExitStack()
        self._exit_stack = stack
        session_kwargs = {
            "read_timeout_seconds": timedelta(
                seconds=max(float(self.config.session_read_timeout or 60.0), 1.0)
            ),
        }
        handler = self._message_handler()
        if _call_accepts_keyword(sdk["ClientSession"], "message_handler"):
            session_kwargs["message_handler"] = handler

        if self.config.url:
            transport = normalize_message_text(self.config.transport or "sse")
            if transport in {"streamable_http", "streamablehttp", "http"}:
                streamable = sdk.get("streamablehttp_client")
                if streamable is None:
                    raise RuntimeError("MCP streamable_http transport is unavailable")
                streams = await stack.enter_async_context(
                    streamable(
                        url=self.config.url,
                        headers=self.config.headers,
                        timeout=timedelta(seconds=max(self.config.timeout, 1.0)),
                        sse_read_timeout=timedelta(
                            seconds=max(self.config.sse_read_timeout, 1.0)
                        ),
                    )
                )
                read_stream, write_stream = streams[0], streams[1]
                self._session = await stack.enter_async_context(
                    sdk["ClientSession"](
                        read_stream=read_stream,
                        write_stream=write_stream,
                        **session_kwargs,
                    )
                )
            else:
                sse = sdk.get("sse_client")
                if sse is None:
                    raise RuntimeError("MCP SSE transport is unavailable")
                streams = await stack.enter_async_context(
                    sse(
                        url=self.config.url,
                        headers=self.config.headers,
                        timeout=max(self.config.timeout, 1.0),
                        sse_read_timeout=max(self.config.sse_read_timeout, 1.0),
                    )
                )
                self._session = await stack.enter_async_context(
                    sdk["ClientSession"](*streams, **session_kwargs)
                )
        else:
            stdio = sdk.get("stdio_client")
            params_cls = sdk.get("StdioServerParameters")
            if stdio is None or params_cls is None:
                raise RuntimeError("MCP stdio transport is unavailable")
            env = dict(os.environ)
            env.update(self.config.env or {})
            if os.name == "nt":
                env.setdefault("PATHEXT", os.environ.get("PATHEXT", ""))
            params_payload: dict[str, Any] = {
                "command": self.config.command,
                "args": list(self.config.args),
                "env": env,
            }
            if self.config.cwd:
                params_payload["cwd"] = self.config.cwd
            params = params_cls(**params_payload)
            streams = await stack.enter_async_context(stdio(params))
            self._session = await stack.enter_async_context(
                sdk["ClientSession"](*streams, **session_kwargs)
            )
        init_result = await self._session.initialize()
        self.status.status = "initialized"
        setattr(self, "initialize_result", init_result)

    def _message_handler(self):
        async def _handler(message: Any) -> None:
            if _is_tools_changed_notification(message):
                logger.info(
                    f"[ChatInter MCP] {self.config.name} 收到 tools/list_changed，后台刷新",
                    "mcp_runtime",
                )
                asyncio.create_task(self.refresh_tools(force=True))

        return _handler

    def _mark_failed(self, error: str, *, isolate: bool = True) -> None:
        self.status.status = "failed"
        self.status.error = str(error or "unknown_error")[:800]
        self.status.failure_count += 1
        if isolate:
            seconds = max(float(self.config.failure_isolation_seconds or 0.0), 0.0)
            if seconds:
                self.status.isolated_until = time.time() + min(
                    seconds * max(self.status.failure_count, 1),
                    300.0,
                )
        logger.warning(
            f"[ChatInter MCP] server={self.config.name} failed: {self.status.error}",
            "mcp_runtime",
        )


class MCPRuntimeManager:
    """Process-wide MCP manager with per-server failure isolation."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConnection] = {}
        self._lock = asyncio.Lock()
        self._last_config_signature = ""

    async def discover_tools(self, *, force: bool = False) -> MCPDiscoveryResult:
        async with self._lock:
            configs = load_mcp_server_configs()
            sdk = _load_mcp_sdk()
            result = MCPDiscoveryResult(
                sdk_available=bool(sdk["available"]),
                config_path=str(_CONFIG_PATH),
                error="" if sdk["available"] else str(sdk["error"]),
            )
            if not sdk["available"]:
                result.statuses = [
                    MCPServerStatus(
                        name=config.name,
                        enabled=config.enabled,
                        status="sdk_unavailable" if config.enabled else "disabled",
                        error=str(sdk["error"]) if config.enabled else "",
                        config_signature=config.signature(),
                    )
                    for config in configs
                ]
                return result

            config_signature = _configs_signature(configs)
            if force or config_signature != self._last_config_signature:
                await self._reconcile_servers(configs)
                self._last_config_signature = config_signature

            for config in configs:
                server = self._servers.get(config.name)
                if server is None:
                    status = MCPServerStatus(
                        name=config.name,
                        enabled=config.enabled,
                        status="disabled" if not config.enabled else "missing",
                        config_signature=config.signature(),
                    )
                    result.statuses.append(status)
                    continue
                await server.ensure_connected(force=force)
                if server.status.status == "connected":
                    result.tools_by_server[config.name] = server.build_executables()
                result.statuses.append(server.status)
            return result

    async def reload(self, *, server_names: list[str] | None = None) -> MCPDiscoveryResult:
        async with self._lock:
            selected = {
                normalize_message_text(name)
                for name in (server_names or [])
                if normalize_message_text(name)
            }
            for name, server in list(self._servers.items()):
                if not selected or name in selected:
                    await server.close()
                    self._servers.pop(name, None)
            self._last_config_signature = ""
        return await self.discover_tools(force=True)

    async def refresh_server(self, name: str) -> MCPServerStatus | None:
        normalized = normalize_message_text(name)
        async with self._lock:
            server = self._servers.get(normalized)
            if server is None:
                return None
            await server.refresh_tools(force=True)
            return server.status

    async def shutdown(self) -> None:
        async with self._lock:
            for server in list(self._servers.values()):
                await server.close()
            self._servers.clear()
            self._last_config_signature = ""

    async def status(self) -> MCPDiscoveryResult:
        configs = load_mcp_server_configs()
        sdk = _load_mcp_sdk()
        statuses: list[MCPServerStatus] = []
        for config in configs:
            server = self._servers.get(config.name)
            if server is not None:
                statuses.append(server.status)
            else:
                statuses.append(
                    MCPServerStatus(
                        name=config.name,
                        enabled=config.enabled,
                        status="not_loaded" if config.enabled else "disabled",
                        config_signature=config.signature(),
                    )
                )
        return MCPDiscoveryResult(
            statuses=statuses,
            sdk_available=bool(sdk["available"]),
            config_path=str(_CONFIG_PATH),
            error="" if sdk["available"] else str(sdk["error"]),
        )

    async def _reconcile_servers(self, configs: list[MCPServerConfig]) -> None:
        desired = {config.name: config for config in configs}
        for name in list(self._servers):
            if name not in desired:
                await self._servers[name].close()
                self._servers.pop(name, None)
        for name, config in desired.items():
            old = self._servers.get(name)
            if old is None or old.config.signature() != config.signature():
                if old is not None:
                    await old.close()
                self._servers[name] = MCPServerConnection(config)


_MANAGER = MCPRuntimeManager()


def get_mcp_runtime_manager() -> MCPRuntimeManager:
    return _MANAGER


def ensure_mcp_config() -> Path:
    if not _CONFIG_PATH.exists():
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(_default_mcp_config_text(), encoding="utf-8")
        return _CONFIG_PATH
    text = _CONFIG_PATH.read_text(encoding="utf-8")
    if _has_mcp_servers_key(text):
        return _CONFIG_PATH
    _CONFIG_PATH.write_text(_insert_mcp_servers_key(text), encoding="utf-8")
    return _CONFIG_PATH


def load_mcp_server_configs() -> list[MCPServerConfig]:
    ensure_mcp_config()
    if yaml is None:
        logger.warning("[ChatInter MCP] PyYAML 不可用，MCP 配置无法读取", "mcp_runtime")
        return []
    try:
        raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"[ChatInter MCP] MCP 配置读取失败: {exc}", "mcp_runtime")
        return []
    chatinter_raw = raw.get(_CONFIG_SECTION) if isinstance(raw, dict) else {}
    if not isinstance(chatinter_raw, dict):
        return []
    servers_raw = chatinter_raw.get(_CONFIG_KEY) or {}
    if not isinstance(servers_raw, dict):
        return []
    return [
        _server_config_from_raw(name, config)
        for name, config in servers_raw.items()
        if isinstance(config, dict)
    ]


def normalize_mcp_input_schema(schema: Any) -> dict[str, Any]:
    """Repair common MCP JSON Schema variants for provider compatibility."""

    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}

    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node
        repaired: dict[str, Any] = {}
        for key, value in node.items():
            out_key = "$defs" if key == "definitions" else key
            if key in {"examples", "default", "deprecated", "readOnly", "writeOnly"}:
                continue
            if key == "$ref" and isinstance(value, str) and value.startswith("#/definitions/"):
                repaired[out_key] = "#/$defs/" + value[len("#/definitions/") :]
            else:
                repaired[out_key] = visit(value)

        for union_key in ("anyOf", "oneOf"):
            variants = repaired.get(union_key)
            if isinstance(variants, list):
                non_null = [
                    variant
                    for variant in variants
                    if not (isinstance(variant, dict) and variant.get("type") == "null")
                ]
                if len(non_null) == 1 and len(non_null) != len(variants):
                    merged = dict(non_null[0])
                    merged["nullable"] = True
                    repaired.pop(union_key, None)
                    repaired.update(visit(merged))

        type_value = repaired.get("type")
        if isinstance(type_value, list):
            non_null_types = [item for item in type_value if item != "null"]
            if len(non_null_types) == 1:
                repaired["type"] = non_null_types[0]
                if len(non_null_types) != len(type_value):
                    repaired["nullable"] = True
        if not repaired.get("type") and (
            "properties" in repaired or "required" in repaired
        ):
            repaired["type"] = "object"
        if repaired.get("type") == "object":
            props = repaired.get("properties")
            if not isinstance(props, dict):
                props = {}
                repaired["properties"] = props
            required = repaired.get("required")
            if isinstance(required, list):
                valid = [item for item in required if isinstance(item, str) and item in props]
                if valid:
                    repaired["required"] = valid
                else:
                    repaired.pop("required", None)
        return repaired

    normalized = visit(schema)
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}
    if normalized.get("type") == "object" and "properties" not in normalized:
        normalized["properties"] = {}
    if "type" not in normalized:
        normalized["type"] = "object"
        normalized.setdefault("properties", {})
    return normalized


def _server_config_from_raw(name: Any, raw: dict[str, Any]) -> MCPServerConfig:
    tools = raw.get("tools") if isinstance(raw.get("tools"), dict) else {}
    return MCPServerConfig(
        name=normalize_message_text(str(name or "")) or "mcp_server",
        enabled=_bool(raw.get("enabled", raw.get("active", True)), True),
        command=str(raw.get("command", "") or "").strip(),
        args=tuple(str(item) for item in _list(raw.get("args"))),
        env={str(k): str(v) for k, v in (raw.get("env") or {}).items()}
        if isinstance(raw.get("env"), dict)
        else {},
        cwd=str(raw.get("cwd", "") or "").strip(),
        url=str(raw.get("url", "") or "").strip(),
        transport=normalize_message_text(str(raw.get("transport", raw.get("type", "")) or "")),
        headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()}
        if isinstance(raw.get("headers"), dict)
        else {},
        timeout=_float(raw.get("timeout"), _DEFAULT_CONNECT_TIMEOUT),
        sse_read_timeout=_float(raw.get("sse_read_timeout"), 300.0),
        session_read_timeout=_float(raw.get("session_read_timeout"), 60.0),
        tool_timeout=_float(raw.get("tool_timeout"), _DEFAULT_TOOL_TIMEOUT),
        include_tools=frozenset(str(item) for item in _list(tools.get("include"))),
        exclude_tools=frozenset(str(item) for item in _list(tools.get("exclude"))),
        max_tools=_int(raw.get("max_tools"), 80),
        failure_isolation_seconds=_float(
            raw.get("failure_isolation_seconds"),
            _DEFAULT_FAILURE_ISOLATION_SECONDS,
        ),
    )


def _load_mcp_sdk() -> dict[str, Any]:
    try:
        import mcp  # type: ignore

        try:
            from mcp.client.sse import sse_client  # type: ignore
        except Exception:
            sse_client = None
        try:
            from mcp.client.streamable_http import streamablehttp_client  # type: ignore
        except Exception:
            streamablehttp_client = None
        try:
            from mcp.client.stdio import stdio_client  # type: ignore
        except Exception:
            stdio_client = getattr(mcp, "stdio_client", None)
        return {
            "available": True,
            "error": "",
            "ClientSession": getattr(mcp, "ClientSession"),
            "StdioServerParameters": getattr(mcp, "StdioServerParameters", None),
            "stdio_client": stdio_client,
            "sse_client": sse_client,
            "streamablehttp_client": streamablehttp_client,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"MCP SDK unavailable: {type(exc).__name__}: {exc}",
        }


def _filter_tools(tools: list[Any], config: MCPServerConfig) -> list[Any]:
    selected: list[Any] = []
    for tool in tools:
        name = _tool_name(tool)
        if not name:
            continue
        if config.include_tools and name not in config.include_tools:
            continue
        if config.exclude_tools and name in config.exclude_tools:
            continue
        selected.append(tool)
        if len(selected) >= max(int(config.max_tools or 1), 1):
            break
    return selected


def _serialize_mcp_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        try:
            return _compact_payload(result.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(result, "dict"):
        try:
            return _compact_payload(result.dict())
        except Exception:
            pass
    if isinstance(result, dict | list | str | int | float | bool) or result is None:
        return _compact_payload(result)
    return _compact_payload({"repr": repr(result)})


def _compact_payload(value: Any) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) <= _MAX_RESULT_CHARS:
        return value
    return {
        "truncated": True,
        "text": text[:_MAX_RESULT_CHARS],
        "original_chars": len(text),
    }


def _mcp_display_summary(payload: Any) -> str:
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list):
            texts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
            if texts:
                return " ".join(texts)[:300]
        if payload.get("isError"):
            return "mcp_tool_error"
    return "mcp_tool_completed"


def _mcp_error_result(server: str, tool_name: str, exc: Exception) -> ToolResult:
    return ToolResult(
        output={
            "ok": False,
            "status": "mcp_tool_error",
            "server": server,
            "tool": tool_name,
            "error": f"{type(exc).__name__}: {exc}",
            "retryable": _looks_like_closed_resource(exc),
        },
        display_content="mcp_tool_error",
    )


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "").strip()


def _tool_description(tool: Any) -> str:
    return str(getattr(tool, "description", "") or "").strip()


def _tool_input_schema(tool: Any) -> Any:
    return (
        getattr(tool, "inputSchema", None)
        or getattr(tool, "input_schema", None)
        or getattr(tool, "parameters", None)
        or {}
    )


def _looks_like_closed_resource(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(token in text for token in ("closedresource", "brokenresource", "closed resource", "connection reset", "session is closed"))


def _is_tools_changed_notification(message: Any) -> bool:
    method = str(getattr(message, "method", "") or "")
    root = getattr(message, "root", None)
    root_method = str(getattr(root, "method", "") or "")
    text = str(message)
    return (
        method == "notifications/tools/list_changed"
        or root_method == "notifications/tools/list_changed"
        or "tools/list_changed" in text
    )


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        return False
    return keyword in signature.parameters


def _configs_signature(configs: list[MCPServerConfig]) -> str:
    text = json.dumps(
        {config.name: config.signature() for config in configs},
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


def _has_mcp_servers_key(text: str) -> bool:
    return re.search(r"(?m)^\s{2}MCP_SERVERS\s*:", text) is not None


def _insert_mcp_servers_key(text: str) -> str:
    block = _mcp_config_block()
    lines = text.splitlines()
    chatinter_index: int | None = None
    for index, line in enumerate(lines):
        if re.match(r"^chatinter\s*:\s*(?:#.*)?$", line):
            chatinter_index = index
            break
    if chatinter_index is None:
        suffix = "\n" if text and not text.endswith("\n") else ""
        return f"{text}{suffix}{_default_mcp_config_text()}"

    insert_at = len(lines)
    for index in range(chatinter_index + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t", "#")):
            insert_at = index
            break
    lines[insert_at:insert_at] = block.rstrip("\n").splitlines()
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _mcp_config_block() -> str:
    return """  # MCP_SERVERS: 超级用户私聊 Agent 可用的 MCP server 配置。仅 superuser_agent 场景暴露，不进入群聊/普通私聊。
  # 示例:
  # MCP_SERVERS:
  #   filesystem:
  #     enabled: false
  #     command: npx
  #     args: ["-y", "@modelcontextprotocol/server-filesystem", "C:/zhenxun"]
  #     tools:
  #       include: []
  #       exclude: []
  #     max_tools: 80
  #     timeout: 12
  #     tool_timeout: 60
  MCP_SERVERS: {}
"""


def _default_mcp_config_text() -> str:
    return """chatinter:
""" + _mcp_config_block()


__all__ = [
    "MCPDiscoveryResult",
    "MCPRuntimeManager",
    "MCPServerConfig",
    "MCPServerConnection",
    "MCPServerStatus",
    "MCPToolExecutable",
    "ensure_mcp_config",
    "get_mcp_runtime_manager",
    "load_mcp_server_configs",
    "normalize_mcp_input_schema",
]
