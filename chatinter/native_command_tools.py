"""Native function-call tool wrappers for ChatInter command schemas."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from .command_index import CommandCandidate
from .command_observation import build_command_observation
from .models.pydantic_models import (
    CommandSlotSpec,
    CommandToolSnapshot,
    PluginCommandSchema,
)
from .route_text import normalize_message_text
from .task_frame import TASK_TEXT_FIELD

_TOOL_NAME_PREFIX = "ci_cmd_"
_TOOL_NAME_DIGEST_SIZE = 5


@dataclass(frozen=True)
class NativeCommandToolBinding:
    tool_name: str
    candidate: CommandCandidate

    @property
    def command_id(self) -> str:
        return self.candidate.schema.command_id


class NativeCommandTool:
    """Executable wrapper exposing one plugin command as one native tool."""

    def __init__(self, binding: NativeCommandToolBinding):
        self.binding = binding

    async def get_definition(self) -> ToolDefinition:
        schema = self.binding.candidate.schema
        return ToolDefinition(
            name=self.binding.tool_name,
            description=_build_tool_description(self.binding.candidate),
            parameters=_build_parameters(
                schema,
                snapshot=self.binding.candidate.tool,
            ),
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        executor = _resolve_native_executor(context)
        if executor is None:
            return ToolResult(
                output=build_command_observation(
                    ok=False,
                    command_id=self.binding.command_id,
                    rendered_command=self.binding.candidate.schema.head,
                    matched_plugin=self.binding.candidate.plugin_name,
                    error="Native command execution context is missing.",
                    plugin_module=self.binding.candidate.plugin_module,
                ),
                display_content=f"{self.binding.command_id} 缺少执行上下文",
            )
        try:
            return await executor.execute_tool(binding=self.binding, raw_slots=kwargs)
        except Exception as exc:
            return ToolResult(
                output=build_command_observation(
                    ok=False,
                    command_id=self.binding.command_id,
                    rendered_command=self.binding.candidate.schema.head,
                    matched_plugin=self.binding.candidate.plugin_name,
                    error=str(exc),
                    plugin_module=self.binding.candidate.plugin_module,
                ),
                display_content=f"{self.binding.command_id} 执行失败",
            )


def build_native_command_tools(
    candidates: list[CommandCandidate],
) -> list[NativeCommandTool]:
    tools: list[NativeCommandTool] = []
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

    return tools


def _safe_tool_name(command_id: str) -> str:
    digest = hashlib.blake2s(
        command_id.encode("utf-8", errors="ignore"),
        digest_size=_TOOL_NAME_DIGEST_SIZE,
    ).hexdigest()
    return f"{_TOOL_NAME_PREFIX}{digest}"


def _resolve_native_executor(context: Any | None) -> Any | None:
    extra = getattr(context, "extra", None)
    if not isinstance(extra, dict):
        return None
    executor = extra.get("native_command_context")
    if executor is None or not hasattr(executor, "execute_tool"):
        return None
    return executor


def _build_tool_description(candidate: CommandCandidate) -> str:
    schema = candidate.schema
    snapshot = candidate.tool
    parts = [
        "ChatInter command tool. 调用此工具会真实触发对应 NoneBot 插件命令。",
        f"plugin_name: {candidate.plugin_name}",
        f"plugin_module: {candidate.plugin_module}",
        f"command_id: {schema.command_id}",
        f"head: {schema.head}",
        f"render: {schema.render}",
        f"role: {schema.command_role}",
        f"payload_policy: {schema.payload_policy}",
        f"extra_text_policy: {schema.extra_text_policy}",
        f"source: {schema.source}",
        f"confidence: {schema.confidence:.2f}",
        f"local_recall_reason: {candidate.reason}",
    ]
    description = schema.description or getattr(snapshot, "capability_text", "")
    if description:
        parts.append(f"description: {description}")
    if snapshot is not None and snapshot.usage:
        parts.append(f"usage: {snapshot.usage}")
    if snapshot is not None and snapshot.capability_text:
        parts.append(f"capability: {snapshot.capability_text}")
    if snapshot is not None and snapshot.task_verbs:
        parts.append("task_verbs: " + _join_values(snapshot.task_verbs))
    if snapshot is not None and snapshot.input_requirements:
        parts.append("input_requirements: " + _join_values(snapshot.input_requirements))
    if snapshot is not None and snapshot.examples:
        parts.append("examples: " + _join_values(snapshot.examples))
    if schema.aliases:
        parts.append("aliases: " + _join_values(schema.aliases))
    if schema.retrieval_phrases:
        parts.append("retrieval_phrases: " + _join_values(schema.retrieval_phrases))
    if schema.requires:
        requires = [
            key
            for key, required in schema.requires.items()
            if required and normalize_message_text(key)
        ]
        if requires:
            parts.append("requires_context: " + _join_values(requires))
    target_requirement = normalize_message_text(
        str(getattr(schema, "target_requirement", "") or "")
    )
    target_sources = [
        normalize_message_text(str(item or ""))
        for item in getattr(schema, "target_sources", []) or []
        if normalize_message_text(str(item or ""))
    ]
    if target_requirement and target_requirement != "none":
        parts.append(f"target_requirement: {target_requirement}")
    if target_sources:
        parts.append("target_sources: " + _join_values(target_sources))
    if getattr(schema, "allow_at", None):
        parts.append("allow_at: true，允许使用 @ 或 [@user_id] 作为目标。")
    if schema.actor_scope:
        parts.append(f"actor_scope: {schema.actor_scope}")
    if schema.slots:
        parts.append(
            "slots: "
            + _join_values(
                _slot_signature(slot)
                for slot in schema.slots
                if normalize_message_text(slot.name)
            )
        )
    parts.append(
        "Call policy: 只有用户明确要执行该功能、查询该功能用法，或自然语言需求"
        "明显对应该命令时才调用。普通闲聊、讨论命令概念、候选不匹配时不要调用。"
    )
    parts.append(
        "Multi-task policy: 如果用户一句话里有多个任务，调用本工具时 task_text "
        "只能包含本命令负责的子任务，不要带上前后其他命令。"
    )
    return "\n".join(part for part in parts if normalize_message_text(part))


def _build_parameters(
    schema: PluginCommandSchema,
    *,
    snapshot: CommandToolSnapshot | None,
) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {
        TASK_TEXT_FIELD: {
            "type": ["string", "null"],
            "description": (
                "当前工具调用对应的用户子任务原文。多任务消息必须只填写本工具"
                "负责的片段，例如“看一下我的信息”，不要包含其他任务。"
            ),
        }
    }
    required: list[str] = [TASK_TEXT_FIELD]
    seen: set[str] = set()
    for slot in schema.slots:
        name = normalize_message_text(slot.name)
        if not name or name in seen:
            continue
        seen.add(name)
        properties[name] = _slot_to_property(slot, schema=schema, snapshot=snapshot)
        # OpenAI strict function schemas require every object property to appear
        # in `required`. Optional command slots therefore accept null explicitly.
        required.append(name)
    return {
        "type": "object",
        "description": _build_parameter_root_description(schema, snapshot=snapshot),
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _slot_to_property(
    slot: CommandSlotSpec,
    *,
    schema: PluginCommandSchema,
    snapshot: CommandToolSnapshot | None,
) -> dict[str, Any]:
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
        f"slot_type={slot.type}",
        f"command_id={schema.command_id}",
    ]
    if slot.aliases:
        description_parts.append("aliases: " + _join_values(slot.aliases))
    if slot.default is not None:
        description_parts.append(f"默认: {slot.default}")
    if slot.type == "at":
        description_parts.append(
            "目标用户请使用已有 [@user_id] 占位符；不要臆造陌生用户 ID。"
        )
    if slot.type == "image":
        description_parts.append(
            "图片请使用已有 [image#1] 占位符；没有图片上下文时传 null。"
        )
    if snapshot is not None and snapshot.input_requirements:
        description_parts.append(
            "input_requirements: " + _join_values(snapshot.input_requirements)
        )
    return {
        "type": schema_type,
        "description": "；".join(
            normalize_message_text(item) for item in description_parts if item
        ),
    }


def _build_parameter_root_description(
    schema: PluginCommandSchema,
    *,
    snapshot: CommandToolSnapshot | None,
) -> str:
    parts = [
        f"Full command argument schema for {schema.command_id}.",
        f"head={schema.head}",
        f"render={schema.render}",
        f"role={schema.command_role}",
        f"payload_policy={schema.payload_policy}",
    ]
    true_requires = [
        key for key, value in (schema.requires or {}).items() if bool(value)
    ]
    if true_requires:
        parts.append("requires=" + _join_values(true_requires))
    if schema.target_requirement != "none":
        parts.append(f"target_requirement={schema.target_requirement}")
    if schema.target_sources:
        parts.append("target_sources=" + _join_values(schema.target_sources))
    if snapshot is not None and snapshot.capability_text:
        parts.append(f"capability={snapshot.capability_text}")
    return "；".join(parts)


def _slot_signature(slot: CommandSlotSpec) -> str:
    parts = [
        slot.name,
        slot.type,
        "required" if slot.required else "optional",
    ]
    if slot.aliases:
        parts.append("aliases=" + ",".join(slot.aliases))
    if slot.description:
        parts.append(slot.description)
    return "(" + ";".join(normalize_message_text(part) for part in parts if part) + ")"


def _join_values(values: Iterable[object]) -> str:
    result: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text and text not in result:
            result.append(text)
    return " / ".join(result)


__all__ = [
    "NativeCommandTool",
    "NativeCommandToolBinding",
    "build_native_command_tools",
]
