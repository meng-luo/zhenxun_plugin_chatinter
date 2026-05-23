"""Catalog tool that injects executable command schemas on demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from .command_index import CommandCandidate
from .route_text import normalize_message_text
from .tool_retriever import CommandToolRetriever

COMMAND_CATALOG_TOOL_NAME = "retrieve_plugin_commands"
_DEFAULT_COMMAND_TOOL_CAP = 119


@dataclass
class CommandCatalogState:
    retriever: CommandToolRetriever
    max_command_tools: int = _DEFAULT_COMMAND_TOOL_CAP
    _candidates: dict[str, CommandCandidate] = field(default_factory=dict)
    _command_order: list[str] = field(default_factory=list)
    retrieve_count: int = 0
    last_query: str = ""
    last_retrieved: int = 0
    last_injected: int = 0

    @property
    def candidates(self) -> list[CommandCandidate]:
        return [
            self._candidates[command_id]
            for command_id in self._command_order
            if command_id in self._candidates
        ]

    @property
    def injected_count(self) -> int:
        return len(self.retriever.registry.executable_tool_map_by_kind("plugin_command"))

    def inject(self, candidates: list[CommandCandidate]) -> list[Any]:
        for candidate in candidates:
            command_id = normalize_message_text(candidate.schema.command_id)
            if not command_id:
                continue
            self._candidates[command_id] = candidate
            if command_id in self._command_order:
                self._command_order.remove(command_id)
            self._command_order.append(command_id)

        self._trim_to_cap()
        self._sort_stable()
        return self.retriever.registry.inject_plugin_command_candidates(
            self.candidates,
            max_command_tools=self.max_command_tools,
        )

    def replace(self, candidates: list[CommandCandidate]) -> list[Any]:
        self._candidates.clear()
        self._command_order.clear()
        self.retriever.registry.clear_plugin_command_tools()
        return self.inject(candidates)

    def _trim_to_cap(self) -> None:
        cap = max(1, int(self.max_command_tools or _DEFAULT_COMMAND_TOOL_CAP))
        while len(self._command_order) > cap and self._command_order:
            command_id = self._command_order.pop(0)
            self._candidates.pop(command_id, None)

    def _sort_stable(self) -> None:
        candidates = [
            self._candidates[command_id]
            for command_id in self._command_order
            if command_id in self._candidates
        ]
        candidates.sort(
            key=lambda candidate: (
                bool(candidate.exact_protected),
                float(candidate.score or 0.0),
                normalize_message_text(candidate.schema.command_id),
            ),
            reverse=True,
        )
        self._command_order = [
            normalize_message_text(candidate.schema.command_id)
            for candidate in candidates
            if normalize_message_text(candidate.schema.command_id)
        ]


class CommandCatalogTool:
    """Retrieve relevant command capabilities, then inject executable schemas."""

    def __init__(self, state: CommandCatalogState):
        self.state = state

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=COMMAND_CATALOG_TOOL_NAME,
            description=(
                "检索真寻插件命令能力。它只做候选召回，不替你决定要执行哪个"
                "命令。仅当当前已暴露的命令工具没有合适 schema、但用户可能"
                "需要插件能力时，才调用此工具补查长尾能力；返回后再根据注入"
                "的完整 schema 选择具体工具。"
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
        self.state.last_query = result.query
        self.state.last_retrieved = len(candidates)
        self.state.last_injected = len(injected_tools)
        _sync_native_context_candidates(context, self.state.candidates)

        guardrail_hint = _guardrail_hint(
            retrieved=len(candidates),
            active_tools=self.state.injected_count,
        )
        payload = {
            "ok": True,
            "status": "capability_candidates_retrieved",
            "query": result.query,
            "retrieved": len(candidates),
            "total_commands": result.total_commands,
            "injected_command_tools": len(injected_tools),
            "active_command_tools": self.state.injected_count,
            "selection_policy": (
                "local_recall_only; rank/score/reason 只用于缩小候选，"
                "最终是否执行、执行哪个命令必须由模型结合完整 schema 判断。"
            ),
            "commands": [
                _candidate_payload(result.capability_payloads, index=index)
                for index in range(1, len(candidates) + 1)
            ],
            "next_step": (
                "如果新注入的命令工具与用户子任务匹配，可以在下一步调用对应"
                "工具；如果没有合适命令，可以换查询词再次检索，或直接聊天"
                "说明没有合适插件。不要因为 rank 靠前就执行。"
            ),
        }
        if guardrail_hint:
            payload["guardrail_hint"] = guardrail_hint
        return ToolResult(
            output=payload,
            display_content=(
                f"检索到 {len(candidates)} 个相关命令，"
                f"已注入 {len(injected_tools)} 个可执行 schema。"
            ),
        )


def _candidate_payload(
    payloads: tuple[dict[str, Any], ...],
    *,
    index: int,
) -> dict[str, Any]:
    raw = dict(payloads[index - 1]) if index - 1 < len(payloads) else {}
    capability_raw = raw.get("capability")
    capability: dict[str, Any] = (
        capability_raw if isinstance(capability_raw, dict) else {}
    )
    target_policy_raw = capability.get("target_policy")
    target_policy = (
        target_policy_raw
        if isinstance(target_policy_raw, dict)
        else {}
    )
    compact: dict[str, Any] = {
        "rank": raw.get("rank", index),
        "score": raw.get("score"),
        "command_id": raw.get("command_id"),
        "plugin_name": raw.get("plugin_name"),
        "plugin_module": raw.get("plugin_module"),
        "head": raw.get("head"),
        "role": raw.get("role"),
        "payload_policy": raw.get("payload_policy"),
        "target_policy": {
            "requirement": target_policy.get("requirement")
            or raw.get("target_requirement"),
            "sources": target_policy.get("sources") or raw.get("target_sources") or [],
            "allow_at": target_policy.get("allow_at", raw.get("allow_at")),
        },
        "slots": _compact_slots(raw.get("slots")),
        "render": raw.get("render"),
        "output_mode": capability.get("output_mode"),
        "requires_real_tool": capability.get("requires_real_tool"),
        "source_of_truth": capability.get("source_of_truth"),
        "selection_note": "已注入同名 command tool；若匹配当前任务，请调用对应工具。",
        "tool_injected": True,
    }
    reason = normalize_message_text(str(raw.get("reason", "") or ""))
    if reason:
        compact["reason"] = reason[:180]
    aliases = raw.get("aliases")
    if isinstance(aliases, list) and aliases:
        compact["aliases"] = [
            normalize_message_text(str(item or ""))
            for item in aliases[:4]
            if normalize_message_text(str(item or ""))
        ]
    description = normalize_message_text(str(raw.get("description", "") or ""))
    if description:
        compact["description"] = description[:160]
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _compact_slots(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    slots: list[dict[str, Any]] = []
    for item in value[:6]:
        if not isinstance(item, dict):
            continue
        slot = {
            "name": item.get("name"),
            "type": item.get("type"),
            "required": item.get("required"),
        }
        aliases = item.get("aliases")
        if isinstance(aliases, list) and aliases:
            slot["aliases"] = [
                normalize_message_text(str(alias or ""))
                for alias in aliases[:3]
                if normalize_message_text(str(alias or ""))
            ]
        description = normalize_message_text(str(item.get("description", "") or ""))
        if description:
            slot["description"] = description[:80]
        slots.append(
            {key: item_value for key, item_value in slot.items() if item_value not in (None, "", [])}
        )
    return slots


def _guardrail_hint(*, retrieved: int, active_tools: int) -> str:
    if retrieved <= 0:
        return (
            "catalog_no_result: 如果换查询词后仍无结果，应直接回复没有合适插件。"
        )
    if active_tools >= 72:
        return (
            "catalog_over_injected: 当前已注入较多命令，应优先选择已有工具，"
            "不要继续扩大检索。"
        )
    return ""


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
