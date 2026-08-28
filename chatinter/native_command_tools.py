"""Native function-call tool wrappers for ChatInter command schemas."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import re
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
from .task_frame import (
    TARGET_REF_FIELD,
    TARGET_REF_SCHEMA_DESCRIPTION,
    TARGET_REFS_FIELD,
    TARGET_REFS_SCHEMA_DESCRIPTION,
    TASK_TEXT_FIELD,
)

_TOOL_NAME_PREFIX = "ci_cmd_"
_TOOL_NAME_DIGEST_SIZE = 5
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class NativeCommandToolBinding:
    tool_name: str
    candidate: CommandCandidate

    @property
    def command_id(self) -> str:
        return self.candidate.schema.command_id


class NativeCommandTool:
    """Executable wrapper exposing one plugin command as one native tool."""

    def __init__(
        self,
        binding: NativeCommandToolBinding,
        *,
        execution_context: Any | None = None,
    ):
        self.binding = binding
        self._execution_context = execution_context

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
        executor = self._execution_context or _resolve_native_executor(context)
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
    *,
    execution_context: Any | None = None,
) -> list[NativeCommandTool]:
    tools: list[NativeCommandTool] = []
    seen_command_ids: set[str] = set()
    seen_tool_names: set[str] = set()

    for candidate in candidates:
        command_id = normalize_message_text(candidate.schema.command_id)
        if not command_id or command_id in seen_command_ids:
            continue
        seen_command_ids.add(command_id)
        preferred_name = _semantic_tool_name(candidate.tool)
        tool_name = _safe_tool_name(
            command_id,
            preferred_name=(
                preferred_name if preferred_name not in seen_tool_names else None
            ),
        )
        if tool_name in seen_tool_names:
            tool_name = _safe_tool_name(command_id)
        seen_tool_names.add(tool_name)
        binding = NativeCommandToolBinding(
            tool_name=tool_name,
            candidate=candidate,
        )
        tools.append(
            NativeCommandTool(
                binding,
                execution_context=execution_context,
            )
        )

    return tools


def _safe_tool_name(
    command_id: str,
    *,
    preferred_name: str | None = None,
) -> str:
    preferred = normalize_message_text(str(preferred_name or ""))
    if _TOOL_NAME_PATTERN.fullmatch(preferred):
        return preferred
    digest = hashlib.blake2s(
        command_id.encode("utf-8", errors="ignore"),
        digest_size=_TOOL_NAME_DIGEST_SIZE,
    ).hexdigest()
    return f"{_TOOL_NAME_PREFIX}{digest}"


def _semantic_tool_name(snapshot: CommandToolSnapshot | None) -> str:
    if snapshot is None or not isinstance(snapshot.meta, dict):
        return ""
    return normalize_message_text(str(snapshot.meta.get("semantic_tool_name") or ""))


def _semantic_contract(snapshot: CommandToolSnapshot | None) -> dict[str, Any]:
    if snapshot is None or not isinstance(snapshot.meta, dict):
        return {}
    contract = snapshot.meta.get("semantic_contract")
    return dict(contract) if isinstance(contract, dict) else {}


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
    contract = _semantic_contract(snapshot)
    description = normalize_message_text(str(contract.get("description") or ""))
    description = description or schema.description or getattr(
        snapshot, "capability_text", ""
    )
    parts = [description or f"执行{candidate.plugin_name}的{schema.head}功能。"]
    if description:
        parts.append("调用会真实执行对应插件功能并返回实际结果。")
    if snapshot is not None and snapshot.usage:
        parts.append(f"用法：{snapshot.usage}")
    if snapshot is not None and snapshot.input_requirements:
        parts.append("输入：" + _join_values(snapshot.input_requirements))
    if snapshot is not None and snapshot.examples:
        parts.append("示例：" + _join_values(snapshot.examples))
    if snapshot is not None and snapshot.use_cases:
        parts.append("适用：" + _join_values(snapshot.use_cases))
    if snapshot is not None and snapshot.anti_use_cases:
        parts.append("不适用：" + _join_values(snapshot.anti_use_cases))
    if schema.requires:
        requires = [
            key
            for key, required in schema.requires.items()
            if required and normalize_message_text(key)
        ]
        if requires:
            parts.append("所需上下文：" + _join_values(requires))
    target_requirement = normalize_message_text(
        str(getattr(schema, "target_requirement", "") or "")
    )
    target_sources = [
        normalize_message_text(str(item or ""))
        for item in getattr(schema, "target_sources", []) or []
        if normalize_message_text(str(item or ""))
    ]
    if target_requirement and target_requirement != "none":
        parts.append(f"目标要求：{target_requirement}")
    if target_sources:
        parts.append("目标来源：" + _join_values(target_sources))
    target_is_context_owned = any(slot.type == "at" for slot in schema.slots)
    if target_is_context_owned:
        parts.append("目标对象由代码从本轮消息、@、回复和群成员上下文中解析。")
    if schema.slots:
        parts.append(
            "参数："
            + _join_values(
                _slot_signature(slot)
                for slot in schema.slots
                if normalize_message_text(slot.name) and slot.type != "at"
            )
        )
    if snapshot is not None:
        parts.append(
            f"输出：{snapshot.output_mode}；副作用：{snapshot.side_effect}；"
            f"执行策略：{snapshot.execution_policy}。"
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
                "当前工具调用对应的用户任务标注。多工具场景建议只填写本工具"
                "负责的内容；无法明确标注时传 null。"
            ),
        },
        TARGET_REF_FIELD: {
            "type": ["string", "null"],
            "description": TARGET_REF_SCHEMA_DESCRIPTION,
        },
        TARGET_REFS_FIELD: {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
            "minItems": 2,
            "maxItems": 4,
            "uniqueItems": True,
            "description": TARGET_REFS_SCHEMA_DESCRIPTION,
        },
    }
    required: list[str] = [TASK_TEXT_FIELD]
    seen: set[str] = {TASK_TEXT_FIELD}
    for slot in schema.slots:
        name = normalize_message_text(slot.name)
        if not name or name in seen or slot.type == "at":
            continue
        seen.add(name)
        properties[name] = _slot_to_property(slot, schema=schema, snapshot=snapshot)
        if slot.required:
            required.append(name)
    contract = _semantic_contract(snapshot)
    contract_parameters = contract.get("parameters")
    if isinstance(contract_parameters, dict):
        contract_properties = contract_parameters.get("properties")
        if isinstance(contract_properties, dict):
            for name, raw_property in contract_properties.items():
                if name not in properties or not isinstance(raw_property, dict):
                    continue
                properties[name].update(
                    {
                        key: value
                        for key, value in raw_property.items()
                        if key in {"type", "description", "enum", "default"}
                    }
                )
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
    "NativeCommandTool",
    "NativeCommandToolBinding",
    "build_native_command_tools",
]
