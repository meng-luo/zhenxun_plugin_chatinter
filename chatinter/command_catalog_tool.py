"""Catalog tool that injects executable command schemas on demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .command_index import CommandCandidate, dump_candidate_for_prompt
from .native_command_tools import build_native_command_tools
from .route_text import normalize_message_text
from .tool_retriever import CommandToolRetriever

COMMAND_CATALOG_TOOL_NAME = "retrieve_plugin_commands"
_DEFAULT_COMMAND_TOOL_CAP = 96


@dataclass
class CommandCatalogState:
    retriever: CommandToolRetriever
    max_command_tools: int = _DEFAULT_COMMAND_TOOL_CAP
    _candidates: dict[str, CommandCandidate] = field(default_factory=dict)
    _tool_map: dict[str, ToolExecutable] = field(default_factory=dict)
    _command_order: list[str] = field(default_factory=list)
    retrieve_count: int = 0

    @property
    def candidates(self) -> list[CommandCandidate]:
        return [
            self._candidates[command_id]
            for command_id in self._command_order
            if command_id in self._candidates
        ]

    @property
    def tool_map(self) -> dict[str, ToolExecutable]:
        return dict(self._tool_map)

    @property
    def injected_count(self) -> int:
        return len(self._tool_map)

    def inject(self, candidates: list[CommandCandidate]) -> list[ToolExecutable]:
        tools = build_native_command_tools(candidates)
        for tool in tools:
            command_id = normalize_message_text(tool.binding.command_id)
            if not command_id:
                continue
            self._candidates[command_id] = tool.binding.candidate
            self._tool_map[tool.binding.tool_name] = tool
            if command_id in self._command_order:
                self._command_order.remove(command_id)
            self._command_order.append(command_id)

        self._trim_to_cap()
        return [cast(ToolExecutable, tool) for tool in tools]

    def _trim_to_cap(self) -> None:
        cap = max(1, int(self.max_command_tools or _DEFAULT_COMMAND_TOOL_CAP))
        while len(self._tool_map) > cap and self._command_order:
            command_id = self._command_order.pop(0)
            candidate = self._candidates.pop(command_id, None)
            if candidate is None:
                continue
            for tool_name, tool in list(self._tool_map.items()):
                binding = getattr(tool, "binding", None)
                if (
                    normalize_message_text(getattr(binding, "command_id", ""))
                    == command_id
                ):
                    self._tool_map.pop(tool_name, None)


class CommandCatalogTool:
    """Retrieve relevant commands, then inject their executable schemas."""

    def __init__(self, state: CommandCatalogState):
        self.state = state

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=COMMAND_CATALOG_TOOL_NAME,
            description=(
                "检索真寻插件命令能力。用户可能需要插件能力，但当前还没有对应"
                "命令 schema 时，先调用此工具。工具返回后，相关命令会作为"
                "可执行 function tools 注入下一轮模型请求。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "用于检索插件能力的自然语言子任务。多任务时可以填原句，"
                            "也可以填某个未完成子任务。"
                        ),
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": (
                            "最多返回并注入多少个相关命令；通常 12-32 即可。"
                        ),
                    },
                },
                "required": ["query", "limit"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        query = normalize_message_text(str(kwargs.get("query", "") or ""))
        limit = kwargs.get("limit")
        if not query:
            query = _context_message_text(context)
        result = self.state.retriever.retrieve(query, limit=limit)
        self.state.retrieve_count += 1
        candidates = list(result.candidates)
        injected_tools = self.state.inject(candidates)
        _sync_native_context_candidates(context, self.state.candidates)

        payload = {
            "ok": True,
            "status": "retrieved",
            "query": result.query,
            "retrieved": len(candidates),
            "total_commands": result.total_commands,
            "injected_command_tools": len(injected_tools),
            "active_command_tools": self.state.injected_count,
            "commands": [
                _candidate_payload(candidate, index=index)
                for index, candidate in enumerate(candidates, 1)
            ],
            "next_step": (
                "如果这些命令中有合适的工具，请在下一步调用对应命令工具；"
                "如果没有合适命令，可以换查询词再次调用 retrieve_plugin_commands，"
                "或直接聊天说明没有合适插件。"
            ),
        }
        return ToolResult(
            output=payload,
            display_content=(
                f"检索到 {len(candidates)} 个相关命令，"
                f"已注入 {len(injected_tools)} 个可执行 schema。"
            ),
        )


def _candidate_payload(candidate: CommandCandidate, *, index: int) -> dict[str, Any]:
    payload = dump_candidate_for_prompt(candidate, index=index)
    payload["tool_injected"] = True
    return payload


def _context_message_text(context: Any | None) -> str:
    extra = getattr(context, "extra", None)
    if not isinstance(extra, dict):
        return ""
    native_context = extra.get("native_command_context")
    return normalize_message_text(getattr(native_context, "message_text", "") or "")


def _sync_native_context_candidates(
    context: Any | None,
    candidates: list[CommandCandidate],
) -> None:
    extra = getattr(context, "extra", None)
    if not isinstance(extra, dict):
        return
    native_context = extra.get("native_command_context")
    if native_context is not None and hasattr(native_context, "candidates"):
        native_context.candidates = list(candidates)


__all__ = [
    "COMMAND_CATALOG_TOOL_NAME",
    "CommandCatalogState",
    "CommandCatalogTool",
]
