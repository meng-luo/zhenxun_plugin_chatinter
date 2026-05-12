"""Native function-call tool wrappers for ChatInter command schemas."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from .command_index import CommandCandidate
from .models.pydantic_models import CommandSlotSpec
from .route_text import normalize_message_text

_TOOL_NAME_PREFIX = "ci_cmd_"
_TOOL_NAME_DIGEST_SIZE = 5
_DESCRIPTION_MAX_LEN = 900


@dataclass(frozen=True)
class NativeCommandToolBinding:
    tool_name: str
    candidate: CommandCandidate

    @property
    def command_id(self) -> str:
        return self.candidate.schema.command_id


class NativeCommandTool:
    """A no-op executable used to expose one command as one native tool."""

    def __init__(self, binding: NativeCommandToolBinding):
        self.binding = binding

    async def get_definition(self) -> ToolDefinition:
        schema = self.binding.candidate.schema
        return ToolDefinition(
            name=self.binding.tool_name,
            description=_build_tool_description(self.binding.candidate),
            parameters=_build_parameters(schema.slots),
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        _ = context
        return ToolResult(
            output={
                "command_id": self.binding.command_id,
                "slots": kwargs,
            },
            display_content=f"selected {self.binding.command_id}",
        )


def build_native_command_tools(
    candidates: list[CommandCandidate],
) -> tuple[list[NativeCommandTool], dict[str, NativeCommandToolBinding]]:
    tools: list[NativeCommandTool] = []
    bindings: dict[str, NativeCommandToolBinding] = {}
    seen_command_ids: set[str] = set()

    for candidate in candidates:
        command_id = normalize_message_text(candidate.schema.command_id)
        if not command_id or command_id in seen_command_ids:
            continue
        seen_command_ids.add(command_id)
        tool_name = _safe_tool_name(command_id)
        binding = NativeCommandToolBinding(
            tool_name=tool_name,
            candidate=candidate,
        )
        tools.append(NativeCommandTool(binding))
        bindings[tool_name] = binding

    return tools, bindings


def parse_native_tool_arguments(tool_call: Any) -> dict[str, Any]:
    function = getattr(tool_call, "function", None)
    raw_arguments = getattr(function, "arguments", None)
    if isinstance(raw_arguments, dict):
        return {
            str(key): value
            for key, value in raw_arguments.items()
            if normalize_message_text(str(key or ""))
        }
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        return {}
    try:
        payload = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if normalize_message_text(str(key or ""))
    }


def _safe_tool_name(command_id: str) -> str:
    digest = hashlib.blake2s(
        command_id.encode("utf-8", errors="ignore"),
        digest_size=_TOOL_NAME_DIGEST_SIZE,
    ).hexdigest()
    return f"{_TOOL_NAME_PREFIX}{digest}"


def _build_tool_description(candidate: CommandCandidate) -> str:
    schema = candidate.schema
    parts = [
        f"插件: {candidate.plugin_name}",
        f"命令: {schema.head}",
        f"用途: {schema.description or candidate.reason or schema.head}",
    ]
    if schema.aliases:
        parts.append("别名: " + " / ".join(schema.aliases[:8]))
    if schema.retrieval_phrases:
        parts.append("可响应说法: " + " / ".join(schema.retrieval_phrases[:8]))
    if schema.requires:
        requires = [
            key
            for key, required in schema.requires.items()
            if required and normalize_message_text(key)
        ]
        if requires:
            parts.append("上下文需求: " + " / ".join(requires))
    target_requirement = normalize_message_text(
        str(getattr(schema, "target_requirement", "") or "")
    )
    target_sources = [
        normalize_message_text(str(item or ""))
        for item in getattr(schema, "target_sources", []) or []
        if normalize_message_text(str(item or ""))
    ]
    if target_requirement and target_requirement != "none":
        parts.append(f"目标要求: {target_requirement}")
    if target_sources:
        parts.append("目标来源: " + " / ".join(target_sources))
    if getattr(schema, "allow_at", None):
        parts.append("允许使用 @ 或 [@user_id] 作为目标。")
    if schema.command_role and schema.command_role != "execute":
        parts.append(f"命令类型: {schema.command_role}")
    parts.append(
        "只在用户明确要执行该功能或查询该功能用法时调用；"
        "普通闲聊、讨论命令概念、缺少必要上下文时不要调用。"
    )
    description = "\n".join(parts)
    return description[:_DESCRIPTION_MAX_LEN]


def _build_parameters(slots: list[CommandSlotSpec]) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    seen: set[str] = set()
    for slot in slots:
        name = normalize_message_text(slot.name)
        if not name or name in seen:
            continue
        seen.add(name)
        properties[name] = _slot_to_property(slot)
        # OpenAI strict function schemas require every object property to appear
        # in `required`. Optional command slots therefore accept null explicitly.
        required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _slot_to_property(slot: CommandSlotSpec) -> dict[str, Any]:
    json_type = {
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }.get(slot.type, "string")
    schema_type: str | list[str]
    if slot.required:
        schema_type = json_type
    else:
        schema_type = [json_type, "null"]
    description_parts = [
        slot.description or slot.name,
        "必填" if slot.required else "可选；未提供时传 null",
    ]
    if slot.aliases:
        description_parts.append("别名: " + " / ".join(slot.aliases[:6]))
    if slot.default is not None:
        description_parts.append(f"默认: {slot.default}")
    if slot.type in {"at", "image"}:
        description_parts.append("使用已有占位符，例如 [@user_id] 或 [image#1]")
    return {
        "type": schema_type,
        "description": "；".join(
            normalize_message_text(item) for item in description_parts if item
        ),
    }


__all__ = [
    "NativeCommandTool",
    "NativeCommandToolBinding",
    "build_native_command_tools",
    "parse_native_tool_arguments",
]
