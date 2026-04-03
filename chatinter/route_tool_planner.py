from dataclasses import dataclass
import re
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .models.pydantic_models import PluginKnowledgeBase
from .route_text import normalize_message_text
from .schema_policy import resolve_command_target_policy
from .skill_registry import (
    SkillCommandSchema,
    SkillRankedCandidate,
    SkillSearchResult,
    SkillSpec,
    skill_search,
)

_DEFAULT_TOOLS_PER_SKILL = 2
_DESCRIPTION_TRIM = 120
_GLOBAL_HELP_HEAD_PRIORITY = ("功能", "帮助", "help")
_AT_TOKEN_PATTERN = re.compile(r"\[@\d{5,20}\]|(?<![0-9A-Za-z_])@\d{5,20}(?=(?:\s|$|[的，,。.!！？?]))")
_IMAGE_TOKEN_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_SELF_REF_HINTS = ("我", "自己", "本人", "我自己", "自己的")


@dataclass(frozen=True)
class RoutePlannerToolSpec:
    tool_name: str
    plugin_name: str
    plugin_module: str
    command_head: str
    schema: SkillCommandSchema | None
    text_allowed: bool
    text_required: bool


@dataclass(frozen=True)
class _MessageCapabilitySnapshot:
    at_count: int
    image_count: int
    has_self_ref: bool


class RoutePlannerTool(ToolExecutable):
    def __init__(self, spec: RoutePlannerToolSpec, definition: ToolDefinition):
        self.spec = spec
        self._definition = definition

    async def get_definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        _ = context
        payload = {
            "plugin_name": self.spec.plugin_name,
            "plugin_module": self.spec.plugin_module,
            "command_head": self.spec.command_head,
            "text": normalize_message_text(str(kwargs.get("text", "") or "")),
        }
        return ToolResult(output=payload, display_content=str(payload))


def _trim_text(text: str, *, limit: int = _DESCRIPTION_TRIM) -> str:
    normalized = normalize_message_text(text)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def _find_schema(skill: SkillSpec, command_head: str) -> SkillCommandSchema | None:
    normalized_head = normalize_message_text(command_head).casefold()
    if not normalized_head:
        return None
    for schema in skill.command_schemas:
        if normalize_message_text(schema.command).casefold() == normalized_head:
            return schema
        for alias in schema.aliases:
            if normalize_message_text(alias).casefold() == normalized_head:
                return schema
    return None


def _is_text_allowed(schema: SkillCommandSchema | None) -> bool:
    if schema is None:
        return True
    text_min = schema.text_min or 0
    text_max = schema.text_max
    if text_min > 0:
        return True
    if text_max is None:
        return True
    return text_max > 0


def _is_text_required(schema: SkillCommandSchema | None) -> bool:
    if schema is None:
        return False
    return (schema.text_min or 0) > 0


def _build_text_property(schema: SkillCommandSchema | None) -> dict[str, Any]:
    if schema is None:
        return {
            "type": "string",
            "description": (
                "可选文本参数。只填写纯文本内容，不要重复填写命令头、[@...]、[image]。"
            ),
        }

    text_min = schema.text_min or 0
    text_max = schema.text_max
    if text_max is None:
        range_text = f"至少 {text_min} 段文本" if text_min > 0 else "可选文本"
    elif text_max == 0:
        range_text = "该命令不接受文本参数"
    elif text_min == text_max:
        range_text = f"需要 {text_min} 段文本"
    else:
        range_text = f"需要 {text_min}-{text_max} 段文本"

    description = (
        f"{range_text}。只填写纯文本内容，不要重复填写命令头、[@...]、[image]。"
    )
    property_payload: dict[str, Any] = {
        "type": "string",
        "description": description,
    }
    if text_max == 1:
        property_payload["maxLength"] = 80
    return property_payload


def _build_tool_definition(
    *,
    spec: RoutePlannerToolSpec,
    skill: SkillSpec,
    message_snapshot: _MessageCapabilitySnapshot,
) -> ToolDefinition:
    schema = spec.schema
    description_parts = [
        f"调用插件“{spec.plugin_name}”的命令“{spec.command_head}”。",
    ]
    if skill.description:
        description_parts.append(_trim_text(skill.description))
    if skill.usage:
        description_parts.append(f"用法: {_trim_text(skill.usage)}")
    if schema is not None:
        policy = resolve_command_target_policy(schema)
        parameter_bits: list[str] = []
        if spec.text_allowed:
            if spec.text_required:
                parameter_bits.append("文本必填")
            else:
                parameter_bits.append("文本可选")
        else:
            parameter_bits.append("无文本参数")
        if (schema.image_min or 0) > 0:
            parameter_bits.append(f"至少 {schema.image_min} 张图")
        if policy.allow_at:
            parameter_bits.append("@可作为目标")
        if policy.target_requirement == "required":
            parameter_bits.append("必须指定目标")
        if policy.actor_scope == "self_only":
            parameter_bits.append("仅本人可执行")
        if parameter_bits:
            description_parts.append("约束: " + "；".join(parameter_bits))
        context_bits = _build_message_fit_bits(schema, message_snapshot)
        if context_bits:
            description_parts.append("当前消息状态: " + "；".join(context_bits))
    if skill.examples:
        description_parts.append(
            "示例: "
            + " | ".join(_trim_text(item, limit=48) for item in skill.examples[:2])
        )

    description_parts.append(
        "仅当当前用户消息明显对应这个命令时调用；不要在 text 中重复填写 [@...] 或 [image]。"
    )

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    if spec.text_allowed:
        parameters["properties"]["text"] = _build_text_property(schema)
        if spec.text_required:
            parameters["required"] = ["text"]

    return ToolDefinition(
        name=spec.tool_name,
        description="\n".join(part for part in description_parts if part),
        parameters=parameters,
    )


def _build_message_snapshot(message_text: str) -> _MessageCapabilitySnapshot:
    normalized = normalize_message_text(message_text)
    lowered = normalized.lower()
    return _MessageCapabilitySnapshot(
        at_count=len(_AT_TOKEN_PATTERN.findall(normalized)),
        image_count=len(_IMAGE_TOKEN_PATTERN.findall(normalized)),
        has_self_ref=any(hint in lowered for hint in _SELF_REF_HINTS),
    )


def _build_message_fit_bits(
    schema: SkillCommandSchema,
    message_snapshot: _MessageCapabilitySnapshot,
) -> list[str]:
    policy = resolve_command_target_policy(schema)
    bits: list[str] = []
    image_min = max(int(schema.image_min or 0), 0)
    if message_snapshot.at_count > 0 and policy.allow_at:
        bits.append(f"已带 {message_snapshot.at_count} 个@目标")
    elif message_snapshot.has_self_ref and policy.allow_at:
        bits.append("含自指，可映射自己")
    elif policy.target_requirement == "required":
        bits.append("当前未给出明确目标，可能进入补参")

    if image_min > 0:
        if message_snapshot.image_count >= image_min:
            bits.append(f"已带 {message_snapshot.image_count} 张图")
        elif not policy.allow_at or (
            message_snapshot.at_count <= 0 and not message_snapshot.has_self_ref
        ):
            bits.append(f"缺少图片/目标（至少需要 {image_min} 个）")
    return bits


def _iter_candidate_heads(
    skill: SkillSpec,
    *,
    helper_mode: bool,
    matched_command: str | None,
    exact_head_hit: bool,
    limit: int,
) -> list[str]:
    values: list[str] = []

    def _append(items: tuple[str, ...] | list[str]) -> None:
        for item in items:
            command_head = normalize_message_text(item)
            if command_head and command_head not in values:
                values.append(command_head)
            if len(values) >= limit:
                return

    matched = normalize_message_text(matched_command or "")
    if matched:
        _append([matched])

    if helper_mode:
        _append(skill.helper_commands)
        if not values:
            _append(skill.commands)
        return values[:limit]

    if matched and matched in skill.helper_commands and exact_head_hit:
        _append([matched])

    _append(skill.action_commands)
    if len(values) < limit:
        _append(skill.commands)
    return values[:limit]


def _pick_global_help_head(skill: SkillSpec) -> str | None:
    ordered_pool = [
        *skill.helper_commands,
        *skill.action_commands,
        *skill.commands,
    ]
    normalized_heads = [
        normalize_message_text(head)
        for head in ordered_pool
        if normalize_message_text(head)
    ]
    for preferred in _GLOBAL_HELP_HEAD_PRIORITY:
        preferred_text = normalize_message_text(preferred)
        for head in normalized_heads:
            if head == preferred_text:
                return head
    return normalized_heads[0] if normalized_heads else None


def _is_global_help_skill(skill: SkillSpec) -> bool:
    pool = [
        *skill.helper_commands,
        *skill.action_commands,
        *skill.commands,
    ]
    normalized_pool = [
        normalize_message_text(item).lower()
        for item in pool
        if normalize_message_text(item)
    ]
    if any(item in _GLOBAL_HELP_HEAD_PRIORITY for item in normalized_pool):
        return True
    plugin_name = normalize_message_text(skill.plugin_name).lower()
    module_name = normalize_message_text(skill.plugin_module).lower()
    return "help" in module_name or "帮助" in plugin_name or "功能" in plugin_name


def _append_global_help_tool(
    *,
    search_result: SkillSearchResult,
    executables: list[ToolExecutable],
    spec_map: dict[str, RoutePlannerToolSpec],
    seen: set[tuple[str, str]],
    tool_index_start: int,
    max_tools: int,
    message_snapshot: _MessageCapabilitySnapshot,
) -> int:
    if len(executables) >= max_tools:
        return tool_index_start

    for skill in search_result.registry.skills:
        if not _is_global_help_skill(skill):
            continue

        command_head = _pick_global_help_head(skill)
        if not command_head:
            continue

        dedupe_key = (skill.plugin_module, command_head.casefold())
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        schema = _find_schema(skill, command_head)
        spec = RoutePlannerToolSpec(
            tool_name=f"route_choice_{tool_index_start:02d}",
            plugin_name=skill.plugin_name,
            plugin_module=skill.plugin_module,
            command_head=command_head,
            schema=schema,
            text_allowed=_is_text_allowed(schema),
            text_required=_is_text_required(schema),
        )
        definition = _build_tool_definition(
            spec=spec,
            skill=skill,
            message_snapshot=message_snapshot,
        )
        executables.append(RoutePlannerTool(spec, definition))
        spec_map[spec.tool_name] = spec
        return tool_index_start + 1

    return tool_index_start


def build_route_planner_tools(
    *,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    max_tools: int,
    tools_per_skill: int = _DEFAULT_TOOLS_PER_SKILL,
    candidate_modules: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[ToolExecutable], dict[str, RoutePlannerToolSpec]]:
    if max_tools <= 0 or not knowledge_base.plugins:
        return [], {}

    search_result = skill_search(
        message_text,
        knowledge_base,
        include_usage=True,
        include_similarity=True,
    )
    helper_mode = search_result.is_usage
    executables: list[ToolExecutable] = []
    spec_map: dict[str, RoutePlannerToolSpec] = {}
    seen: set[tuple[str, str]] = set()
    tool_index = 1
    message_snapshot = _build_message_snapshot(message_text)
    module_order = [
        normalize_message_text(module)
        for module in (candidate_modules or ())
        if normalize_message_text(module)
    ]
    ranked_by_module = {
        normalize_message_text(candidate.skill.plugin_module): candidate
        for candidate in search_result.ranked_candidates
    }
    skills_by_module = {
        normalize_message_text(skill.plugin_module): skill
        for skill in search_result.registry.skills
    }

    if helper_mode:
        tool_index = _append_global_help_tool(
            search_result=search_result,
            executables=executables,
            spec_map=spec_map,
            seen=seen,
            tool_index_start=tool_index,
            max_tools=max_tools,
            message_snapshot=message_snapshot,
        )

    ordered_candidates: list[SkillRankedCandidate | None] = []
    seen_modules: set[str] = set()
    for module in module_order:
        if module in seen_modules:
            continue
        seen_modules.add(module)
        ordered_candidates.append(ranked_by_module.get(module))
    for candidate in search_result.ranked_candidates:
        module = candidate.skill.plugin_module
        if module in seen_modules:
            continue
        seen_modules.add(module)
        ordered_candidates.append(candidate)

    for idx, candidate in enumerate(ordered_candidates):
        if len(executables) >= max_tools:
            break
        if candidate is None:
            continue
        normalized_module = normalize_message_text(candidate.skill.plugin_module)
        if candidate.score <= 0 and normalized_module not in module_order:
            continue
        if idx > 0 and candidate.score < 3.0 and not candidate.exact_head_hit:
            if normalized_module not in module_order:
                continue

        skill = candidate.skill
        candidate_heads = _iter_candidate_heads(
            skill,
            helper_mode=helper_mode,
            matched_command=candidate.matched_command,
            exact_head_hit=candidate.exact_head_hit,
            limit=max(tools_per_skill, 1),
        )
        for command_head in candidate_heads:
            if len(executables) >= max_tools:
                break
            dedupe_key = (skill.plugin_module, command_head.casefold())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            schema = _find_schema(skill, command_head)
            spec = RoutePlannerToolSpec(
                tool_name=f"route_choice_{tool_index:02d}",
                plugin_name=skill.plugin_name,
                plugin_module=skill.plugin_module,
                command_head=command_head,
                schema=schema,
                text_allowed=_is_text_allowed(schema),
                text_required=_is_text_required(schema),
            )
            definition = _build_tool_definition(
                spec=spec,
                skill=skill,
                message_snapshot=message_snapshot,
            )
            executables.append(RoutePlannerTool(spec, definition))
            spec_map[spec.tool_name] = spec
            tool_index += 1

    if len(executables) >= max_tools:
        return executables, spec_map

    for module in module_order:
        if len(executables) >= max_tools:
            break
        skill = skills_by_module.get(module)
        if skill is None:
            continue
        candidate_heads = _iter_candidate_heads(
            skill,
            helper_mode=helper_mode,
            matched_command=None,
            exact_head_hit=False,
            limit=max(tools_per_skill, 1),
        )
        for command_head in candidate_heads:
            if len(executables) >= max_tools:
                break
            dedupe_key = (skill.plugin_module, command_head.casefold())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            schema = _find_schema(skill, command_head)
            spec = RoutePlannerToolSpec(
                tool_name=f"route_choice_{tool_index:02d}",
                plugin_name=skill.plugin_name,
                plugin_module=skill.plugin_module,
                command_head=command_head,
                schema=schema,
                text_allowed=_is_text_allowed(schema),
                text_required=_is_text_required(schema),
            )
            definition = _build_tool_definition(
                spec=spec,
                skill=skill,
                message_snapshot=message_snapshot,
            )
            executables.append(RoutePlannerTool(spec, definition))
            spec_map[spec.tool_name] = spec
            tool_index += 1

    return executables, spec_map
