"""Command-level schemas for ChatInter routing.

插件命令最终仍走原 NoneBot matcher；这里仅把自然语言意图转换为稳定的
command_id + slots，再确定性渲染回原命令文本。
"""

from __future__ import annotations

import re
from typing import Any

from .models.pydantic_models import (
    CommandCapability,
    CommandSlotSpec,
    PluginCommandSchema,
)
from .plugin_adapters import (
    build_adapter_schemas,
    derive_adapter_semantic_aliases,
)
from .route_text import normalize_message_text

_COMMAND_PARAM_TOKEN_PATTERN = re.compile(r"\s*[?*+]?\[[^\]]+\]")
_COMMAND_PARAM_BRACKET_PATTERN = re.compile(r"\s*[?*+]?[<(｟][^>)｠]+[>)｠]")


def _slot(
    name: str,
    slot_type: str = "text",
    *,
    required: bool = False,
    default: Any = None,
    aliases: list[str] | None = None,
    description: str = "",
) -> CommandSlotSpec:
    return CommandSlotSpec(
        name=name,
        type=slot_type,  # type: ignore[arg-type]
        required=required,
        default=default,
        aliases=list(aliases or []),
        description=description,
    )


def _schema(
    command_id: str,
    head: str,
    *,
    aliases: list[str] | None = None,
    description: str = "",
    slots: list[CommandSlotSpec] | None = None,
    render: str | None = None,
    requires: dict[str, bool] | None = None,
    allow_at: bool | None = None,
    actor_scope: str = "allow_other",
    target_requirement: str = "none",
    target_sources: list[str] | None = None,
    command_role: str = "execute",
    payload_policy: str = "none",
    extra_text_policy: str = "keep",
    source: str = "override",
    confidence: float = 0.85,
    matcher_key: str | None = None,
    retrieval_phrases: list[str] | None = None,
) -> PluginCommandSchema:
    normalized_head = normalize_message_text(head)
    normalized_aliases = [
        text
        for text in (normalize_message_text(alias) for alias in list(aliases or []))
        if text
    ]
    normalized_description = normalize_message_text(description)
    phrase_values = [
        normalized_head,
        *normalized_aliases,
        normalized_description,
        command_id,
        *(retrieval_phrases or []),
    ]
    phrases: list[str] = []
    for value in phrase_values:
        text = normalize_message_text(value)
        if text and text not in phrases:
            phrases.append(text)
    return PluginCommandSchema(
        command_id=command_id,
        head=normalized_head or head,
        aliases=list(dict.fromkeys(normalized_aliases)),
        description=normalized_description,
        slots=list(slots or []),
        render=render or head,
        requires={
            "text": False,
            "image": False,
            "reply": False,
            "at": False,
            **dict(requires or {}),
        },
        allow_at=allow_at,  # type: ignore[arg-type]
        actor_scope=actor_scope,  # type: ignore[arg-type]
        target_requirement=target_requirement,  # type: ignore[arg-type]
        target_sources=list(target_sources or []),  # type: ignore[arg-type]
        command_role=command_role,  # type: ignore[arg-type]
        payload_policy=payload_policy,  # type: ignore[arg-type]
        extra_text_policy=extra_text_policy,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
        matcher_key=matcher_key,
        retrieval_phrases=phrases,
    )


def normalize_schema_command_head(command: str | None) -> str:
    """Return the executable command head without parameter placeholders."""

    normalized = normalize_message_text(command or "")
    if not normalized:
        return ""
    candidate = _COMMAND_PARAM_TOKEN_PATTERN.split(normalized, maxsplit=1)[0]
    candidate = _COMMAND_PARAM_BRACKET_PATTERN.split(candidate, maxsplit=1)[0]
    candidate = normalize_message_text(candidate)
    if not candidate:
        return ""
    return normalize_message_text(candidate.split(" ", 1)[0])


def _command_id(module: str, head: str) -> str:
    safe_module = re.sub(r"[^0-9A-Za-z_]+", "_", module.rsplit(".", 1)[-1])
    safe_head = re.sub(r"\s+", "_", normalize_message_text(head))
    return f"{safe_module}.{safe_head or 'command'}"


def _requires_from_capability(command: CommandCapability) -> dict[str, bool]:
    requirement = command.requirement
    return {
        # `requires` is a hard execution gate. Optional slots (for example
        # `我的信息 ?[at]` or `随机小猪 [数量]`) must not become required text.
        "text": bool(requirement.text_min > 0),
        "image": bool(requirement.image_min > 0),
        "reply": bool(requirement.requires_reply),
        "private": bool(requirement.requires_private),
        "to_me": bool(requirement.requires_to_me),
        "at": bool(
            requirement.allow_at
            or "at" in requirement.target_sources
            or requirement.target_requirement == "required"
        ),
    }


def _is_internal_media_param(name: str, requirement: Any) -> bool:
    normalized = normalize_message_text(name).lower()
    if normalized not in {"meme_params", "img", "image", "images", "图片"}:
        return False
    return (
        max(int(getattr(requirement, "text_min", 0) or 0), 0) <= 0
        and max(int(getattr(requirement, "image_min", 0) or 0), 0) > 0
    )


def _payload_policy_from_capability(command: CommandCapability) -> tuple[str, str]:
    requirement = command.requirement
    if requirement.image_min > 0 and requirement.text_min <= 0:
        return "image_only", "discard"
    if requirement.text_min > 0:
        return "text", "slot_only"
    if requirement.params:
        return "slots", "slot_only"
    return "none", "keep"


def _slot_type_from_name(name: str) -> str:
    normalized = normalize_message_text(name).lower()
    if any(
        token in normalized
        for token in (
            "num",
            "count",
            "amount",
            "金币",
            "数量",
            "金额",
            "次数",
            "份",
            "个数",
        )
    ):
        return "int"
    if any(token in normalized for token in ("image", "图片", "图", "照片")):
        return "image"
    if any(token in normalized for token in ("at", "user", "用户", "目标", "对象")):
        return "at"
    return "text"


def _slot_aliases_from_name(name: str) -> list[str]:
    normalized = normalize_message_text(name)
    alias_map = {
        "amount": ["金额", "金币", "总额"],
        "num": ["数量", "个数", "份数"],
        "count": ["数量", "次数", "个数"],
        "text": ["文本", "内容"],
        "content": ["文本", "内容"],
        "target": ["目标", "对象"],
        "image": ["图片", "图"],
    }
    aliases = [normalized] if normalized else []
    for key, values in alias_map.items():
        if key in normalized.lower() or normalized in values:
            aliases.extend(values)
    result: list[str] = []
    for alias in aliases:
        text = normalize_message_text(alias)
        if text and text not in result:
            result.append(text)
    return result


def _slot_description(name: str, slot_type: str) -> str:
    normalized = normalize_message_text(name)
    if not normalized:
        return ""
    if slot_type == "int":
        return f"{normalized}，通常填写数字"
    if slot_type == "image":
        return f"{normalized}，需要图片上下文"
    if slot_type == "at":
        return f"{normalized}，需要@、回复或昵称目标"
    return f"{normalized}文本"


def _command_description(command: CommandCapability, head: str) -> str:
    parts: list[str] = []
    examples = [
        normalize_message_text(example)
        for example in command.examples
        if normalize_message_text(example)
    ]
    if examples:
        parts.append("示例: " + " / ".join(examples))
    requirement = command.requirement
    requirement_parts: list[str] = []
    if requirement.params:
        requirement_parts.append("参数: " + " ".join(requirement.params))
    if requirement.text_min > 0:
        requirement_parts.append(f"至少{requirement.text_min}段文本")
    if requirement.image_min > 0:
        requirement_parts.append(f"至少{requirement.image_min}张图片")
    if requirement.requires_reply:
        requirement_parts.append("需要回复上下文")
    if requirement.target_requirement == "required":
        requirement_parts.append("需要明确目标")
    if requirement_parts:
        parts.append("；".join(requirement_parts))
    if not parts:
        parts.append(f"执行“{head}”命令")
    description = "；".join(parts)
    return description.rstrip()


def schema_from_capability(
    module: str,
    command: CommandCapability,
) -> PluginCommandSchema | None:
    raw_command = normalize_message_text(command.command)
    head = normalize_schema_command_head(raw_command) or raw_command
    if not head:
        return None
    slots: list[CommandSlotSpec] = []
    requirement = command.requirement
    raw_params = [
        normalize_message_text(str(param or ""))
        for param in requirement.params
        if normalize_message_text(str(param or ""))
    ]
    raw_params = [
        param
        for param in raw_params
        if not _is_internal_media_param(param, requirement)
    ]
    if not raw_params and requirement.text_min > 0:
        raw_params = ["text"]
    for index, slot_name in enumerate(raw_params):
        slot_type = _slot_type_from_name(slot_name)
        slots.append(
            _slot(
                slot_name,
                slot_type,
                required=requirement.text_min > index,
                aliases=_slot_aliases_from_name(slot_name),
                description=_slot_description(slot_name, slot_type),
            )
        )
    render = head
    if slots:
        render = " ".join([head, *[f"{{{slot.name}}}" for slot in slots]])
    payload_policy, extra_text_policy = _payload_policy_from_capability(command)
    aliases = [
        *command.aliases,
        *derive_adapter_semantic_aliases(
            head,
            module=module,
            image_required=requirement.image_min > 0,
        ),
    ]
    return _schema(
        _command_id(module, head),
        head,
        aliases=list(dict.fromkeys(alias for alias in aliases if alias)),
        description=_command_description(command, head),
        slots=slots,
        render=render,
        requires=_requires_from_capability(command),
        allow_at=requirement.allow_at,
        actor_scope=requirement.actor_scope,
        target_requirement=requirement.target_requirement,
        target_sources=list(requirement.target_sources),
        command_role="template" if requirement.image_min > 0 else "execute",
        payload_policy=payload_policy,
        extra_text_policy=extra_text_policy,
        source="matcher",
        confidence=0.68,
        matcher_key=f"{module}:{head}",
        retrieval_phrases=[raw_command] if raw_command and raw_command != head else [],
    )


def build_command_schemas(
    module: str,
    commands: list[CommandCapability],
) -> list[PluginCommandSchema]:
    module_key = normalize_message_text(module)
    adapter_schemas = build_adapter_schemas(module_key, commands)
    if adapter_schemas is not None:
        return adapter_schemas

    schemas: list[PluginCommandSchema] = []
    seen: set[str] = set()
    for command in commands:
        schema = schema_from_capability(module_key, command)
        if schema is None or schema.command_id in seen:
            continue
        seen.add(schema.command_id)
        schemas.append(schema)
    return schemas


def complete_slots(
    schema: PluginCommandSchema,
    *,
    slots: dict[str, Any] | None = None,
    message_text: str = "",
    arguments_text: str = "",
) -> tuple[dict[str, Any], list[str]]:
    _ = (message_text, arguments_text)
    merged: dict[str, Any] = {}
    slot_by_key: dict[str, CommandSlotSpec] = {}
    for slot in schema.slots:
        for key in (slot.name, *slot.aliases):
            normalized = normalize_message_text(str(key or ""))
            if normalized:
                slot_by_key[normalized] = slot

    for key, value in dict(slots or {}).items():
        slot = slot_by_key.get(normalize_message_text(str(key or "")))
        if slot is None:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not normalize_message_text(value):
            continue
        merged[slot.name] = value

    missing: list[str] = []
    for slot in schema.slots:
        if slot.name not in merged and slot.default is not None:
            merged[slot.name] = slot.default
        if slot.required and slot.name not in merged:
            missing.append(slot.name)
    return merged, missing


def render_command(
    schema: PluginCommandSchema,
    *,
    slots: dict[str, Any] | None = None,
    message_text: str = "",
    arguments_text: str = "",
) -> tuple[str, list[str]]:
    completed, missing = complete_slots(
        schema,
        slots=slots,
        message_text=message_text,
        arguments_text=arguments_text,
    )
    if missing:
        return schema.head, missing
    values = {
        slot.name: normalize_message_text(str(completed.get(slot.name, "")))
        for slot in schema.slots
    }
    try:
        rendered = schema.render.format_map(values)
    except Exception:
        rendered = schema.head
    return normalize_message_text(rendered), []
