"""Command-level schemas for ChatInter routing.

插件命令最终仍走原 NoneBot matcher；这里仅把自然语言意图转换为稳定的
command_id + slots，再确定性渲染回原命令文本。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models.pydantic_models import (
    CommandCapability,
    CommandRequirement,
    CommandSlotSpec,
    PluginCommandSchema,
)
from .route_text import normalize_message_text

_COMMAND_PARAM_TOKEN_PATTERN = re.compile(r"\s*[?*+]?\[[^\]]+\]")
_COMMAND_PARAM_BRACKET_PATTERN = re.compile(r"\s*[?*+]?[<(｟][^>)｠]+[>)｠]")
_PLACEHOLDER_NAME_PATTERN = re.compile(r"[\[\(<｟{]([^\]\)>｠}]+)[\]\)>｠}]")
_USAGE_LINE_PREFIX_PATTERN = re.compile(
    r"^[\-\*\d\.\)、)\s]*(?:命令|用法|示例|格式|usage|example)?\s*[:：]?\s*",
    re.IGNORECASE,
)
_HELP_ROLE_TERMS = ("帮助", "说明", "用法", "教程", "参数", "示例", "文档", "列表")
_RANDOM_ROLE_TERMS = ("随机", "抽", "roll", "掷", "选择", "塔罗")
_TEMPLATE_ROLE_TERMS = ("生成", "制作", "做", "表情", "模板", "绘制", "画图", "图片")
_QUERY_ROLE_TERMS = ("查询", "查看", "搜索", "识别", "解析", "翻译", "统计", "排行")
_SELF_SCOPE_TERMS = ("我的", "自己", "本人", "个人", "我")
_TARGET_TERMS = (
    "@",
    "at",
    "qq",
    "user",
    "member",
    "target",
    "nickname",
    "用户",
    "成员",
    "群友",
    "目标",
    "对象",
    "昵称",
)


def _slot(
    name: str,
    slot_type: str = "text",
    *,
    required: bool = False,
    default: Any = None,
    aliases: list[str] | None = None,
    description: str = "",
    choices: list[str] | None = None,
) -> CommandSlotSpec:
    return CommandSlotSpec(
        name=name,
        type=slot_type,  # type: ignore[arg-type]
        required=required,
        default=default,
        aliases=list(aliases or []),
        description=description,
        choices=_normalize_choices(choices or []),
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


def _clean_usage_line(line: str) -> str:
    text = str(line or "").strip().strip("`")
    text = _USAGE_LINE_PREFIX_PATTERN.sub("", text).strip()
    return normalize_message_text(text)


def _collect_relevant_usage_lines(
    usage: str | None,
    *,
    head: str,
    aliases: list[str],
) -> list[str]:
    if not usage:
        return []
    heads = [text for text in [head, *aliases] if normalize_message_text(text)]
    heads = sorted(dict.fromkeys(heads), key=len, reverse=True)
    lines: list[str] = []
    for raw_line in str(usage or "").splitlines():
        line = _clean_usage_line(raw_line)
        if not line:
            continue
        if any(_line_starts_with_head(line, candidate) for candidate in heads):
            if line not in lines:
                lines.append(line)
    return lines


def _line_starts_with_head(line: str, head: str) -> bool:
    normalized_line = normalize_message_text(line).casefold()
    normalized_head = normalize_message_text(head).casefold()
    if not normalized_line or not normalized_head:
        return False
    return normalized_line == normalized_head or normalized_line.startswith(
        normalized_head + " "
    )


def _extract_placeholder_names(text: str) -> list[str]:
    names: list[str] = []
    for raw in _PLACEHOLDER_NAME_PATTERN.findall(str(text or "")):
        name = normalize_message_text(str(raw or ""))
        name = name.lstrip("?*+")
        name = name.split("=", 1)[0].split(":", 1)[0].split(" ", 1)[0]
        if name and name not in names:
            names.append(name)
    return names


def _required_placeholder_names(text: str) -> set[str]:
    required: set[str] = set()
    for match in re.finditer(r"([<(｟{])([^>\)｠}]+)[>\)｠}]", str(text or "")):
        raw_name = normalize_message_text(str(match.group(2) or ""))
        raw_name = raw_name.lstrip("?*+")
        name = raw_name.split("=", 1)[0].split(":", 1)[0].split(" ", 1)[0]
        if name:
            required.add(name)
    return required


def _command_id(module: str, head: str) -> str:
    safe_module = re.sub(r"[^0-9A-Za-z_]+", "_", module.rsplit(".", 1)[-1])
    safe_head = re.sub(r"\s+", "_", normalize_message_text(head))
    return f"{safe_module}.{safe_head or 'command'}"


def _requires_from_capability(command: CommandCapability) -> dict[str, bool]:
    requirement = command.requirement
    allow_at = _capability_accepts_at_target(command)
    return {
        # `requires` is a hard execution gate. Optional slots (for example
        # `我的信息 ?[at]` or `随机小猪 [数量]`) must not become required text.
        "text": bool(requirement.text_min > 0),
        "image": bool(requirement.image_min > 0),
        "reply": bool(requirement.requires_reply),
        "private": bool(requirement.requires_private),
        "to_me": bool(requirement.requires_to_me),
        "at": allow_at,
    }


def _normalize_param_name(name: str) -> str:
    return normalize_message_text(name).lower()


def _capability_accepts_image_input(command: CommandCapability) -> bool:
    requirement = command.requirement
    image_min = max(int(requirement.image_min or 0), 0)
    image_max = requirement.image_max
    if image_min > 0:
        return True
    if image_max is None:
        return True
    try:
        return int(image_max) > 0
    except (TypeError, ValueError):
        return False


def _capability_has_explicit_at_slot(command: CommandCapability) -> bool:
    return any(
        _slot_type_from_name(str(param or "")) == "at"
        for param in command.requirement.params
    )


def _capability_accepts_at_target(command: CommandCapability) -> bool:
    requirement = command.requirement
    if _capability_has_explicit_at_slot(command):
        return True
    if not _capability_accepts_image_input(command):
        return False
    return bool(
        requirement.allow_at
        or "at" in requirement.target_sources
        or requirement.target_requirement == "required"
    )


def _target_requirement_from_capability(command: CommandCapability) -> str:
    requirement = command.requirement
    if _capability_accepts_at_target(command):
        return requirement.target_requirement
    if requirement.target_requirement == "required":
        return "required"
    return "none"


def _target_sources_from_capability(command: CommandCapability) -> list[str]:
    if _capability_accepts_at_target(command):
        return list(command.requirement.target_sources)
    return [
        source
        for source in command.requirement.target_sources
        if source not in {"at", "reply", "nickname", "self"}
    ]


def _is_internal_media_param(name: str, requirement: Any) -> bool:
    normalized = _normalize_param_name(name)
    media_named = normalized in {"img", "image", "images", "图片"} or (
        _slot_type_from_name(name) == "image"
    )
    aggregate_media = _is_generic_aggregate_param(name)
    if not (media_named or aggregate_media):
        return False
    return (
        max(int(getattr(requirement, "text_min", 0) or 0), 0) <= 0
        and max(int(getattr(requirement, "image_min", 0) or 0), 0) > 0
    )


def _is_generic_aggregate_param(name: str) -> bool:
    normalized = _normalize_param_name(name)
    return (
        normalized
        in {
            "params",
            "args",
            "arguments",
            "content",
            "contents",
        }
        or normalized.endswith("_params")
        or normalized.endswith("params")
    )


def _payload_policy_from_capability(command: CommandCapability) -> tuple[str, str]:
    requirement = command.requirement
    if requirement.image_min > 0 and (requirement.text_min > 0 or requirement.params):
        return "text_or_image", "slot_only"
    if requirement.image_min > 0:
        return "image_only", "discard"
    if requirement.text_min > 0:
        return "text", "slot_only"
    if requirement.params:
        return "slots", "slot_only"
    return "none", "keep"


def _slot_type_from_name(name: str) -> str:
    normalized = _normalize_param_name(name)
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
            "人数",
            "等级",
            "页",
        )
    ):
        return "int"
    if any(token in normalized for token in ("ratio", "rate", "概率", "比例", "倍率")):
        return "float"
    if any(token in normalized for token in ("image", "图片", "图", "照片")):
        return "image"
    if any(token in normalized for token in _TARGET_TERMS):
        return "at"
    return "text"


def _normalize_choices(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text and text not in result:
            result.append(text)
    return result


def _slot_choices_from_requirement(
    requirement: CommandRequirement,
) -> dict[str, list[str]]:
    raw = getattr(requirement, "choices", None) or getattr(
        requirement, "slot_choices", None
    )
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[str]] = {}
    for key, values in raw.items():
        name = normalize_message_text(str(key or ""))
        if not name:
            continue
        if isinstance(values, str):
            choices = [values]
        elif isinstance(values, list | tuple | set | frozenset):
            choices = [str(value) for value in values]
        else:
            continue
        normalized = _normalize_choices(choices)
        if normalized:
            result[name] = normalized
    return result


def _slot_type_from_examples(name: str, values: list[str]) -> str:
    by_name = _slot_type_from_name(name)
    if by_name != "text":
        return by_name
    normalized_values = [
        normalize_message_text(value)
        for value in values
        if normalize_message_text(value)
    ]
    if not normalized_values:
        return by_name
    if all(re.fullmatch(r"[-+]?\d+", value) for value in normalized_values):
        return "int"
    if all(
        re.fullmatch(r"[-+]?(?:\d+\.\d+|\d+)", value) for value in normalized_values
    ):
        return "float"
    if all(
        value.startswith("@") or (value.isdigit() and len(value) >= 5)
        for value in normalized_values
    ):
        return "at"
    return by_name


def _prepare_schema_param_names(
    raw_params: list[str],
    *,
    text_min: int,
    requirement: Any,
) -> list[str]:
    """Convert aggregate parser params into concrete schema slots.

    Some plugins expose a single variadic parser arg (for example `params`)
    while their runtime metadata knows the exact text/image bounds.  LLM tool
    schemas work better with explicit slots, so split generic aggregate params
    into `text1`, `text2`, ... when the command requires fixed text segments.
    """

    params = [
        param
        for param in raw_params
        if not _is_internal_media_param(param, requirement)
    ]
    if text_min <= 0:
        return params

    text_like_params = [
        param for param in params if _slot_type_from_name(param) == "text"
    ]
    if len(text_like_params) == 1 and _is_generic_aggregate_param(text_like_params[0]):
        params = [param for param in params if param != text_like_params[0]]
        text_like_params = []

    if not text_like_params:
        return [*params, *[f"text{index + 1}" for index in range(text_min)]]

    if len(text_like_params) < text_min:
        return [
            *params,
            *[f"text{index + 1}" for index in range(len(text_like_params), text_min)],
        ]

    return params


def _example_argument_rows(
    command: CommandCapability,
    *,
    head: str,
    usage_lines: list[str],
) -> list[list[str]]:
    rows: list[list[str]] = []
    examples = [
        *[
            normalize_message_text(example)
            for example in command.examples
            if normalize_message_text(example)
        ],
        *usage_lines,
    ]
    heads = [
        text
        for text in [head, command.command, *command.aliases]
        if normalize_message_text(text)
    ]
    heads = sorted(dict.fromkeys(heads), key=len, reverse=True)
    for example in examples:
        matched = next(
            (
                candidate
                for candidate in heads
                if _line_starts_with_head(example, candidate)
            ),
            "",
        )
        if not matched:
            continue
        tail = normalize_message_text(example[len(matched) :])
        if not tail:
            continue
        tail = re.split(r"\s+(?:--|-|：|:)\s+", tail, maxsplit=1)[0]
        values = [
            value
            for value in re.findall(r'"[^"]+"|“[^”]+”|\'[^\']+\'|\S+', tail)
            if value
        ]
        values = [value.strip("\"'“”") for value in values if value.strip("\"'“”")]
        if values:
            rows.append(values)
    return rows


def _infer_param_names_from_examples(
    command: CommandCapability,
    *,
    head: str,
    usage_lines: list[str],
    text_min: int,
) -> list[str]:
    if text_min > 0:
        return (
            ["text"]
            if text_min == 1
            else [f"text{index + 1}" for index in range(text_min)]
        )
    rows = _example_argument_rows(command, head=head, usage_lines=usage_lines)
    if not rows:
        return []
    max_count = min(max(len(row) for row in rows), 6)
    if max_count <= 0:
        return []
    if max_count == 1:
        return ["text"]
    metadata_text = normalize_message_text(
        " ".join(
            [
                head,
                command.description,
                " ".join(command.examples),
                " ".join(usage_lines),
            ]
        )
    ).lower()
    names: list[str] = []
    for index in range(max_count):
        column_values = [row[index] for row in rows if len(row) > index]
        slot_type = _slot_type_from_examples("", column_values)
        if slot_type in {"int", "float"}:
            if index == 0 and any(
                token in metadata_text
                for token in ("金额", "金币", "余额", "积分", "价格", "总额")
            ):
                names.append("amount")
            elif any(
                token in metadata_text
                for token in ("数量", "人数", "个数", "份数", "次数")
            ):
                names.append("count" if "count" not in names else f"count{index + 1}")
            else:
                names.append(f"number{index + 1}")
        elif slot_type == "at":
            names.append("target")
        else:
            names.append(f"arg{index + 1}")
    return names


def _example_values_for_slot(
    rows: list[list[str]],
    index: int,
) -> list[str]:
    return [row[index] for row in rows if len(row) > index]


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
    if slot_type == "float":
        return f"{normalized}，通常填写小数或数字"
    if slot_type == "image":
        return f"{normalized}，需要图片上下文"
    if slot_type == "at":
        return f"{normalized}，需要@、回复或昵称目标"
    return f"{normalized}文本"


def _description_with_choices(description: str, choices: list[str]) -> str:
    normalized_choices = _normalize_choices(choices)
    if not normalized_choices:
        return description
    choices_text = "可选值: " + "/".join(normalized_choices[:16])
    base = normalize_message_text(description)
    return normalize_message_text(
        "；".join(part for part in (base, choices_text) if part)
    )


def _command_description(
    command: CommandCapability,
    head: str,
    *,
    usage_lines: list[str] | None = None,
    plugin_description: str = "",
) -> str:
    parts: list[str] = []
    command_description = normalize_message_text(command.description)
    if command_description:
        parts.append(command_description)
    plugin_desc = normalize_message_text(plugin_description)
    if plugin_desc and plugin_desc != "暂无描述":
        parts.append(plugin_desc)
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
    if usage_lines:
        parts.append("用法: " + " / ".join(usage_lines[:3]))
    if not parts:
        parts.append(f"执行“{head}”命令")
    description = "；".join(parts)
    return description.rstrip()


def _command_role_from_text(
    *,
    text: str,
    requirement: Any,
) -> str:
    normalized = normalize_message_text(text).casefold()
    if any(term.casefold() in normalized for term in _HELP_ROLE_TERMS):
        return "usage"
    if any(term.casefold() in normalized for term in _RANDOM_ROLE_TERMS):
        return "random"
    if requirement.image_min > 0 and any(
        term.casefold() in normalized for term in _TEMPLATE_ROLE_TERMS
    ):
        return "template"
    if any(term.casefold() in normalized for term in _TEMPLATE_ROLE_TERMS):
        return "template"
    if any(term.casefold() in normalized for term in _QUERY_ROLE_TERMS):
        return "execute"
    return "execute"


def _actor_scope_from_text(default: str, text: str) -> str:
    normalized = normalize_message_text(text)
    if normalized.startswith(_SELF_SCOPE_TERMS):
        return "self_only"
    return default


def _target_requirement_from_slots(
    default: str,
    slots: list[CommandSlotSpec],
) -> str:
    at_slots = [slot for slot in slots if slot.type == "at"]
    if not at_slots:
        return default
    if any(slot.required for slot in at_slots):
        return "required"
    return "optional" if default == "none" else default


def _target_sources_from_slots(
    default: list[str],
    slots: list[CommandSlotSpec],
) -> list[str]:
    if not any(slot.type == "at" for slot in slots):
        return default
    result = list(default)
    for source in ("at", "reply", "nickname"):
        if source not in result:
            result.append(source)
    return result


def schema_from_capability(
    module: str,
    command: CommandCapability,
    *,
    usage: str | None = None,
    plugin_description: str = "",
) -> PluginCommandSchema | None:
    raw_command = normalize_message_text(command.command)
    head = normalize_schema_command_head(raw_command) or raw_command
    if not head:
        return None
    usage_lines = _collect_relevant_usage_lines(
        usage,
        head=head,
        aliases=list(command.aliases),
    )
    slots: list[CommandSlotSpec] = []
    requirement = command.requirement
    raw_params = [
        normalize_message_text(str(param or ""))
        for param in requirement.params
        if normalize_message_text(str(param or ""))
    ]
    required_param_names: set[str] = set()
    for value in [raw_command, *usage_lines]:
        required_param_names.update(_required_placeholder_names(value))
        raw_params = [
            *raw_params,
            *[
                item
                for item in _extract_placeholder_names(value)
                if item and item not in raw_params
            ],
        ]
    text_min = max(int(requirement.text_min or 0), 0)
    if not raw_params:
        raw_params = _infer_param_names_from_examples(
            command,
            head=head,
            usage_lines=usage_lines,
            text_min=text_min,
        )
    raw_params = _prepare_schema_param_names(
        raw_params,
        text_min=text_min,
        requirement=requirement,
    )
    slot_choices = _slot_choices_from_requirement(requirement)
    example_rows = _example_argument_rows(command, head=head, usage_lines=usage_lines)
    inferred_from_examples = bool(example_rows and not requirement.params)
    for index, slot_name in enumerate(raw_params):
        choices = slot_choices.get(normalize_message_text(slot_name), [])
        slot_type = _slot_type_from_examples(
            slot_name,
            _example_values_for_slot(example_rows, index),
        )
        if choices and slot_type not in {"int", "float", "bool"}:
            slot_type = "text"
        text_slot_index = sum(
            1
            for previous in raw_params[:index]
            if _slot_type_from_name(previous) in {"text", "int", "float"}
        )
        image_slot_index = sum(
            1
            for previous in raw_params[:index]
            if _slot_type_from_name(previous) == "image"
        )
        slots.append(
            _slot(
                slot_name,
                slot_type,
                required=(
                    (
                        slot_type in {"text", "int", "float"}
                        and (
                            text_slot_index < text_min
                            or slot_name in required_param_names
                            or inferred_from_examples
                        )
                    )
                    or (
                        slot_type == "image"
                        and (
                            image_slot_index < max(int(requirement.image_min or 0), 0)
                            or slot_name in required_param_names
                        )
                    )
                    or (
                        slot_type == "at"
                        and (
                            requirement.target_requirement == "required"
                            or slot_name in required_param_names
                        )
                    )
                ),
                aliases=_slot_aliases_from_name(slot_name),
                description=_description_with_choices(
                    _slot_description(slot_name, slot_type),
                    choices,
                ),
                choices=choices,
            )
        )
    render = head
    if slots:
        render = " ".join([head, *[f"{{{slot.name}}}" for slot in slots]])
    payload_policy, extra_text_policy = _payload_policy_from_capability(command)
    if slots and payload_policy == "none":
        payload_policy, extra_text_policy = "slots", "slot_only"
    description = _command_description(
        command,
        head,
        usage_lines=usage_lines,
        plugin_description=plugin_description,
    )
    metadata_text = normalize_message_text(
        " ".join(
            [
                head,
                raw_command,
                description,
                command.description,
                " ".join(command.examples),
                " ".join(usage_lines),
                plugin_description,
                " ".join(raw_params),
            ]
        )
    )
    command_role = _command_role_from_text(text=metadata_text, requirement=requirement)
    actor_scope = _actor_scope_from_text(requirement.actor_scope, metadata_text)
    target_requirement = _target_requirement_from_slots(
        _target_requirement_from_capability(command),
        slots,
    )
    target_sources = _target_sources_from_slots(
        _target_sources_from_capability(command),
        slots,
    )
    confidence = 0.5
    if raw_params:
        confidence += 0.12
    if command.description:
        confidence += 0.1
    if command.examples:
        confidence += 0.08
    if usage_lines:
        confidence += 0.08
    if requirement.image_min > 0 or requirement.text_min > 0:
        confidence += 0.07
    confidence = min(confidence, 0.9)
    aliases = list(command.aliases)
    requires = _requires_from_capability(command)
    if any(slot.required and slot.type in {"text", "int", "float"} for slot in slots):
        requires["text"] = True
    if any(slot.required and slot.type == "image" for slot in slots):
        requires["image"] = True
    if any(slot.type == "at" for slot in slots):
        requires["at"] = True
    schema = _schema(
        _command_id(module, head),
        head,
        aliases=list(dict.fromkeys(alias for alias in aliases if alias)),
        description=description,
        slots=slots,
        render=render,
        requires=requires,
        allow_at=_capability_accepts_at_target(command)
        or any(slot.type == "at" for slot in slots),
        actor_scope=actor_scope,
        target_requirement=target_requirement,
        target_sources=target_sources,
        command_role=command_role,
        payload_policy=payload_policy,
        extra_text_policy=extra_text_policy,
        source="metadata"
        if (raw_params or command.description or command.examples or usage_lines)
        else "matcher",
        confidence=confidence,
        matcher_key=f"{module}:{head}",
        retrieval_phrases=[
            phrase
            for phrase in [
                raw_command if raw_command and raw_command != head else "",
                command.description,
                *command.examples,
                *usage_lines,
                plugin_description,
            ]
            if normalize_message_text(phrase)
        ],
    )
    schema.shortcut_renders = list(command.shortcut_renders or [])
    return schema


def build_command_schemas(
    module: str,
    commands: list[CommandCapability],
    *,
    usage: str | None = None,
    plugin_description: str = "",
) -> list[PluginCommandSchema]:
    module_key = normalize_message_text(module)
    schemas: list[PluginCommandSchema] = []
    seen: set[str] = set()
    for command in commands:
        schema = schema_from_capability(
            module_key,
            command,
            usage=usage,
            plugin_description=plugin_description,
        )
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
        slot.name: _render_slot_value(schema, slot, completed.get(slot.name, ""))
        for slot in schema.slots
    }
    shortcut_render = _matched_shortcut_render(schema, message_text)
    if shortcut_render:
        shortcut_parts = shortcut_render.split()
        rendered_parts = shortcut_parts[:1] if shortcut_parts else [shortcut_render]
        rendered_parts.extend(
            values.get(slot.name, "") for slot in schema.slots if values.get(slot.name)
        )
        rendered_parts.extend(shortcut_parts[1:])
        rendered = " ".join(rendered_parts)
        return normalize_message_text(rendered), []
    try:
        rendered = schema.render.format_map(values)
    except Exception:
        rendered = schema.head
    return normalize_message_text(rendered), []


def _matched_shortcut_render(
    schema: PluginCommandSchema,
    message_text: str,
) -> str:
    text = normalize_message_text(message_text)
    if not text:
        return ""
    for shortcut in schema.shortcut_renders:
        if not isinstance(shortcut, dict):
            continue
        alias = normalize_message_text(str(shortcut.get("alias", "") or ""))
        render = normalize_message_text(str(shortcut.get("render", "") or ""))
        if alias and render and alias in text:
            return render
    return ""


def _render_slot_value(
    schema: PluginCommandSchema,
    slot: CommandSlotSpec,
    value: Any,
) -> str:
    """Render slot values so downstream command parsers keep slot boundaries."""

    text = normalize_message_text(str(value or ""))
    if not text:
        return ""
    if not _slot_needs_quoting(schema, slot, text):
        return text
    escaped = json.dumps(text, ensure_ascii=False)
    return escaped


def _slot_needs_quoting(
    schema: PluginCommandSchema,
    slot: CommandSlotSpec,
    text: str,
) -> bool:
    if slot.type not in {"text", "str"}:
        return False
    if " " not in text:
        return False
    if len(schema.slots) <= 1:
        return True
    return True
