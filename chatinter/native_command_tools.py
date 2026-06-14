"""Native function-call tool wrappers for ChatInter command schemas."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from typing import Any

from .command_index import CommandCandidate
from .command_observation import build_command_observation
from .llm_compat import ToolDefinition, ToolResult
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


class CompactNativeCommandTool:
    """Lightweight schema view for first-pass command selection.

    The tool name stays identical to the executable command tool.  The runtime
    re-queries with the selected full schema before execution, mirroring Astr's
    skills-like mode while keeping dispatch through the same executor.
    """

    def __init__(self, executable: NativeCommandTool):
        self.executable = executable
        self.binding = executable.binding

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.binding.tool_name,
            description=_build_compact_tool_description(self.binding.candidate),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        return await self.executable.execute(context=context, **kwargs)


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


def compact_command_tool_view(
    executable: Any,
) -> Any:
    if isinstance(executable, NativeCommandTool):
        return CompactNativeCommandTool(executable)
    return executable


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
        f"recall_signal: {candidate.reason}",
        "recall_policy: 本工具来自能力检索候选；rank/score 不是执行决策。",
    ]
    description = schema.description or getattr(snapshot, "capability_text", "")
    if description:
        parts.append(f"description: {description}")
    if snapshot is not None and snapshot.usage:
        parts.append(f"usage: {snapshot.usage}")
    if snapshot is not None and snapshot.capability_text:
        parts.append(f"capability: {snapshot.capability_text}")
    if snapshot is not None:
        card_lines = _capability_card_lines(snapshot)
        if card_lines:
            parts.extend(card_lines)
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
        "Multi-task policy: task_text 是模型对本次工具调用负责内容的标注；"
        "不要把无关任务写进同一个工具调用。无法明确标注时可传 null。"
    )
    return "\n".join(part for part in parts if normalize_message_text(part))


def _build_compact_tool_description(candidate: CommandCandidate) -> str:
    schema = candidate.schema
    snapshot = candidate.tool
    description = _clip_text(
        schema.description or getattr(snapshot, "capability_text", ""), 90
    )
    parts = [
        "Compact capability card; selecting it only asks runtime for full schema.",
        f"command_id: {schema.command_id}",
        f"head: {schema.head}",
        f"role: {schema.command_role}",
        f"payload_policy: {schema.payload_policy}",
    ]
    if description:
        parts.append(f"description: {description}")
    if snapshot is not None:
        card_lines = _compact_capability_card_lines(snapshot)
        if card_lines:
            parts.extend(card_lines)
    if schema.aliases:
        parts.append("aliases: " + _join_values(schema.aliases[:3], limit=80))
    if schema.retrieval_phrases:
        parts.append("phrases: " + _join_values(schema.retrieval_phrases[:3], limit=80))
    if schema.slots:
        parts.append(
            "slots_summary: "
            + _join_values(
                (
                    f"{slot.name}:{slot.type}:{'req' if slot.required else 'opt'}"
                    for slot in schema.slots[:4]
                    if normalize_message_text(slot.name)
                ),
                limit=96,
            )
        )
    return "\n".join(part for part in parts if normalize_message_text(part))


def _capability_card_lines(snapshot: CommandToolSnapshot) -> list[str]:
    lines: list[str] = []
    if snapshot.source_of_truth:
        lines.append(f"source_of_truth: {snapshot.source_of_truth}")
    lines.append(
        "requires_real_tool: " + str(bool(snapshot.requires_real_tool)).lower()
    )
    if snapshot.output_mode:
        lines.append(f"output_mode: {snapshot.output_mode}")
    if snapshot.entity_scope:
        lines.append(f"entity_scope: {snapshot.entity_scope}")
    if snapshot.side_effect:
        lines.append(f"side_effect: {snapshot.side_effect}")
    if snapshot.risk:
        lines.append(f"risk: {snapshot.risk}")
    elif snapshot.risk_level:
        lines.append(f"risk: {snapshot.risk_level}")
    lines.append(f"reliability: {float(snapshot.reliability or 0.0):.2f}")
    lines.append(f"schema_quality: {float(snapshot.schema_quality or 0.0):.2f}")
    lines.append(f"soft_tool: {str(bool(snapshot.soft_tool)).lower()}")
    if snapshot.intent_types:
        lines.append("intent_types: " + _join_values(snapshot.intent_types))
    lines.append(
        "requires_real_result: " + str(bool(snapshot.requires_real_result)).lower()
    )
    lines.append(f"generative: {str(bool(snapshot.generative)).lower()}")
    if snapshot.execution_policy:
        lines.append(f"execution_policy: {snapshot.execution_policy}")
    if snapshot.use_cases:
        lines.append("use_cases: " + _join_values(snapshot.use_cases))
    if snapshot.anti_use_cases:
        lines.append("anti_use_cases: " + _join_values(snapshot.anti_use_cases))
    return lines


def _compact_capability_card_lines(snapshot: CommandToolSnapshot) -> list[str]:
    lines: list[str] = []
    if snapshot.source_of_truth:
        lines.append(f"source_of_truth: {snapshot.source_of_truth}")
    lines.append(
        "requires_real_tool: " + str(bool(snapshot.requires_real_tool)).lower()
    )
    if snapshot.output_mode:
        lines.append(f"output_mode: {snapshot.output_mode}")
    if snapshot.side_effect:
        lines.append(f"side_effect: {snapshot.side_effect}")
    risk = snapshot.risk or snapshot.risk_level
    if risk:
        lines.append(f"risk: {risk}")
    if snapshot.intent_types:
        lines.append(
            "intent_types: " + _join_values(snapshot.intent_types[:4], limit=80)
        )
    if snapshot.use_cases:
        lines.append("use_cases: " + _join_values(snapshot.use_cases[:2], limit=96))
    if snapshot.anti_use_cases:
        lines.append(
            "anti_use_cases: " + _join_values(snapshot.anti_use_cases[:1], limit=96)
        )
    return lines


def _build_parameters(
    schema: PluginCommandSchema,
    *,
    snapshot: CommandToolSnapshot | None,
) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {
        TASK_TEXT_FIELD: {
            "type": ["string", "null"],
            "description": (
                "当前工具调用对应的用户任务标注。多工具场景建议只填写本工具"
                "负责的内容；无法明确标注时传 null。"
            ),
        },
        "target_hint": {
            "type": ["string", "null"],
            "description": _target_hint_description(schema),
        },
        "payload_hint": {
            "type": ["string", "null"],
            "description": _payload_hint_description(schema),
        },
    }
    required: list[str] = [TASK_TEXT_FIELD, "target_hint", "payload_hint"]
    seen: set[str] = {TASK_TEXT_FIELD, "target_hint", "payload_hint"}
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
    choices = list(getattr(slot, "choices", []) or [])
    if choices:
        description_parts.append(
            "可选值: " + _join_values([str(choice) for choice in choices], limit=160)
        )
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
    payload: dict[str, Any] = {
        "type": schema_type,
        "description": "；".join(
            normalize_message_text(item) for item in description_parts if item
        ),
    }
    if choices and json_type == "string":
        enum_values = [normalize_message_text(str(choice)) for choice in choices]
        if not slot.required:
            enum_values = [*enum_values, None]
        payload["enum"] = enum_values
    return payload


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
        f"extra_text_policy={schema.extra_text_policy}",
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


def _target_hint_description(schema: PluginCommandSchema) -> str:
    parts = [
        "目标策略提示，不直接作为插件参数渲染；用于说明本次调用的目标来源。",
        f"target_requirement={schema.target_requirement}",
        f"actor_scope={schema.actor_scope}",
    ]
    if schema.target_sources:
        parts.append("target_sources=" + _join_values(schema.target_sources))
    if schema.allow_at is not None:
        parts.append(f"allow_at={schema.allow_at}")
    return "；".join(parts)


def _payload_hint_description(schema: PluginCommandSchema) -> str:
    return "；".join(
        [
            "负载策略提示，不直接作为插件参数渲染；用于说明文本/图片/尾巴如何提供。",
            f"payload_policy={schema.payload_policy}",
            f"extra_text_policy={schema.extra_text_policy}",
            "图片上下文请用 [image#N]，目标用户请用 [@user_id]。",
        ]
    )


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


def _join_values(values: Iterable[object], *, limit: int = 240) -> str:
    result: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text and text not in result:
            result.append(_clip_text(text, max(24, limit // 3)))
    return _clip_text(" / ".join(result), limit)


def _clip_text(text: str, limit: int) -> str:
    normalized = normalize_message_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 1)].rstrip() + "…"


__all__ = [
    "CompactNativeCommandTool",
    "NativeCommandTool",
    "NativeCommandToolBinding",
    "build_native_command_tools",
    "compact_command_tool_view",
]
