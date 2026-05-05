"""Command-level schemas for ChatInter routing.

插件命令最终仍走原 NoneBot matcher；这里仅把自然语言意图转换为稳定的
command_id + slots，再确定性渲染回原命令文本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    extract_adapter_slots,
)
from .route_text import normalize_message_text, parse_command_with_head

_TEXT_PLACEHOLDER_PATTERN = re.compile(r"\{(?P<name>[A-Za-z_][0-9A-Za-z_]*)\}")
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+")
_URL_PAYLOAD_PATTERN = re.compile(
    r"https?://\S+|(?:BV|AV|av)[0-9A-Za-z]+",
    re.IGNORECASE,
)
_TEXT_TAIL_PREFIX_PATTERN = re.compile(
    r"^(?:这句话|这段话|这句|内容|文本|参数|链接|地址|是|为|叫|名称|名字|"
    r"：|:|-|，|,|。)+"
)
_EMPTY_TEXT_PAYLOAD_WORDS = {
    "一下",
    "一下子",
    "一下下",
    "看看",
    "看下",
    "帮我",
    "请",
    "麻烦",
    "吧",
    "一下吧",
    "下吧",
}


@dataclass(frozen=True)
class CommandSchemaSelection:
    """A scored schema choice.

    The selector keeps command choice deterministic when several schemas share the
    same head, while still accepting LLM-provided command_id as the strongest hint.
    """

    schema: PluginCommandSchema
    score: float
    reason: str
    slots: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()


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
        command_role=command_role,  # type: ignore[arg-type]
        payload_policy=payload_policy,  # type: ignore[arg-type]
        extra_text_policy=extra_text_policy,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
        matcher_key=matcher_key,
        retrieval_phrases=phrases,
    )


def _command_id(module: str, head: str) -> str:
    safe_module = re.sub(r"[^0-9A-Za-z_]+", "_", module.rsplit(".", 1)[-1])
    safe_head = re.sub(r"\s+", "_", normalize_message_text(head))
    return f"{safe_module}.{safe_head or 'command'}"


def _requires_from_capability(command: CommandCapability) -> dict[str, bool]:
    requirement = command.requirement
    params = [
        normalize_message_text(str(param or "")).lower()
        for param in requirement.params
        if normalize_message_text(str(param or ""))
    ]
    internal_media_params = {"meme_params", "img", "image", "images", "图片"}
    params_require_text = bool(params) and not (
        requirement.text_min <= 0
        and requirement.image_min > 0
        and all(param in internal_media_params for param in params)
    )
    return {
        "text": bool(requirement.text_min > 0 or params_require_text),
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
        for example in command.examples[:2]
        if normalize_message_text(example)
    ]
    if examples:
        parts.append("示例: " + " / ".join(examples))
    requirement = command.requirement
    requirement_parts: list[str] = []
    if requirement.params:
        requirement_parts.append("参数: " + " ".join(requirement.params[:4]))
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
    return description[:120].rstrip()


def schema_from_capability(
    module: str,
    command: CommandCapability,
) -> PluginCommandSchema | None:
    head = normalize_message_text(command.command)
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
    for index, slot_name in enumerate(raw_params[:4]):
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
        command_role="template" if requirement.image_min > 0 else "execute",
        payload_policy=payload_policy,
        extra_text_policy=extra_text_policy,
        source="matcher",
        confidence=0.68,
        matcher_key=f"{module}:{head}",
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


def _command_head(command: str | None) -> str:
    normalized = normalize_message_text(command or "")
    return normalize_message_text(normalized.split(" ", 1)[0]) if normalized else ""


def _command_tail(command: str | None) -> str:
    normalized = normalize_message_text(command or "")
    if not normalized or " " not in normalized:
        return ""
    return normalize_message_text(normalized.split(" ", 1)[1])


def _tokenize(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(normalize_message_text(text))
        if token
    }


def _schema_phrases(schema: PluginCommandSchema) -> list[str]:
    values = [
        schema.command_id,
        schema.head,
        *schema.aliases,
        schema.description,
        *schema.retrieval_phrases,
    ]
    for slot in schema.slots:
        values.extend([slot.name, slot.description, *slot.aliases])
    result: list[str] = []
    for value in values:
        text = normalize_message_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _schema_render_slots(schema: PluginCommandSchema) -> set[str]:
    return {
        normalize_message_text(match.group("name"))
        for match in _TEXT_PLACEHOLDER_PATTERN.finditer(schema.render or "")
        if normalize_message_text(match.group("name"))
    }


def _message_has_text_payload(text: str) -> bool:
    normalized = normalize_message_text(text)
    if not normalized:
        return False
    if re.search(r"\d", normalized):
        return True
    if len(normalized) >= 4:
        return True
    return bool(_tokenize(normalized))


def _score_schema(
    schema: PluginCommandSchema,
    *,
    command_id: str | None,
    command: str | None,
    message_text: str,
    arguments_text: str,
    slots: dict[str, Any] | None,
    action: str | None,
) -> CommandSchemaSelection:
    normalized_id = normalize_message_text(command_id or "").casefold()
    schema_id = normalize_message_text(schema.command_id).casefold()
    command_text = normalize_message_text(command or "")
    command_head = _command_head(command_text).casefold()
    command_tail = _command_tail(command_text)
    message = normalize_message_text(" ".join([message_text, arguments_text]))
    message_fold = message.casefold()
    action_text = normalize_message_text(action or "").casefold()
    score = 0.0
    reasons: list[str] = []

    if normalized_id:
        if normalized_id == schema_id:
            score += 1000.0
            reasons.append("command_id")
        else:
            score -= 24.0

    head = normalize_message_text(schema.head).casefold()
    aliases = [
        normalize_message_text(alias).casefold()
        for alias in schema.aliases
        if normalize_message_text(alias)
    ]
    if command_head:
        if command_head == head:
            score += 420.0
            reasons.append("head")
        elif command_head in aliases:
            score += 380.0
            reasons.append("alias_head")
        elif head and command_text.casefold().startswith(head):
            score += 260.0
            reasons.append("head_prefix")
        elif any(
            alias and command_text.casefold().startswith(alias) for alias in aliases
        ):
            score += 220.0
            reasons.append("alias_prefix")

    if message_fold:
        for alias in aliases:
            if alias and alias in message_fold:
                score += 130.0 + min(len(alias), 16)
                reasons.append("message_alias")
        if head and head in message_fold:
            score += 84.0 + min(len(head), 12)
            reasons.append("message_head")

        message_tokens = _tokenize(message)
        phrase_tokens: set[str] = set()
        for phrase in _schema_phrases(schema):
            phrase_tokens.update(_tokenize(phrase))
        overlap = len(message_tokens & phrase_tokens)
        if overlap:
            score += min(overlap * 10.0, 60.0)
            reasons.append("token_overlap")

    completed_slots, missing = complete_slots(
        schema,
        slots=slots,
        message_text=message_text,
        arguments_text=arguments_text or command_tail,
    )
    provided_slot_names = {
        normalize_message_text(str(name or ""))
        for name, value in dict(slots or {}).items()
        if normalize_message_text(str(name or "")) and value is not None
    }
    schema_slot_names = {normalize_message_text(slot.name) for slot in schema.slots}
    matched_provided = len(provided_slot_names & schema_slot_names)
    if matched_provided:
        score += matched_provided * 72.0
        reasons.append("slots")

    render_slots = _schema_render_slots(schema)
    completed_render_slots = {
        name
        for name in render_slots
        if name in completed_slots and completed_slots.get(name) is not None
    }
    if completed_render_slots:
        score += len(completed_render_slots) * 28.0
        reasons.append("completed_slots")

    if missing:
        penalty = 18.0 if action_text == "clarify" else 86.0
        score -= len(missing) * penalty
        reasons.append("missing")

    requires = schema.requires or {}
    has_payload = _message_has_text_payload(
        " ".join([command_tail, arguments_text, message_text])
    )
    if requires.get("text") and has_payload:
        score += 18.0
    elif not requires.get("text") and command_tail:
        score -= 16.0

    if not schema.slots and not requires.get("text") and action_text == "execute":
        score += 3.0

    return CommandSchemaSelection(
        schema=schema,
        score=score,
        reason=",".join(dict.fromkeys(reasons)) or "fallback",
        slots=completed_slots,
        missing=tuple(missing),
    )


def select_command_schema(
    schemas: list[PluginCommandSchema],
    *,
    command_id: str | None = None,
    command: str | None = None,
    message_text: str = "",
    arguments_text: str = "",
    slots: dict[str, Any] | None = None,
    action: str | None = None,
) -> CommandSchemaSelection | None:
    if not schemas:
        return None
    has_hint = any(
        normalize_message_text(value or "")
        for value in (command_id, command, message_text, arguments_text)
    ) or bool(slots)
    if not has_hint:
        return None

    selections = [
        _score_schema(
            schema,
            command_id=command_id,
            command=command,
            message_text=message_text,
            arguments_text=arguments_text,
            slots=slots,
            action=action,
        )
        for schema in schemas
    ]
    selections.sort(
        key=lambda item: (
            item.score,
            -len(item.missing),
            len(item.schema.slots),
            -len(item.schema.head),
        ),
        reverse=True,
    )
    best = selections[0]
    if best.score <= 0:
        return None
    return best


def find_command_schema(
    schemas: list[PluginCommandSchema],
    *,
    command_id: str | None = None,
    command: str | None = None,
) -> PluginCommandSchema | None:
    selection = select_command_schema(
        schemas,
        command_id=command_id,
        command=command,
    )
    return selection.schema if selection is not None else None


def _infer_adapter_slots(
    schema: PluginCommandSchema,
    message_text: str,
) -> dict[str, Any]:
    command_id = normalize_message_text(schema.command_id)
    if not command_id:
        return {}
    return extract_adapter_slots(command_id, message_text)


def _clean_text_payload(value: str) -> str:
    payload = normalize_message_text(value)
    while payload:
        cleaned = normalize_message_text(_TEXT_TAIL_PREFIX_PATTERN.sub("", payload))
        if cleaned == payload:
            break
        payload = cleaned
    return payload


def _clean_head_payload(raw_payload: str, head: str) -> str:
    payload = normalize_message_text(raw_payload)
    if not payload:
        return ""
    head_text = normalize_message_text(head)
    if head_text and payload.startswith(head_text):
        payload = normalize_message_text(payload[len(head_text) :])
    payload = re.sub(
        r"^(?:做一句|做一段|写一句|写一段|说一句|说一段|做|写|说|"
        r"内容(?:是|为)?|文本(?:是|为)?|文字(?:是|为)?|"
        r"：|:|，|,|。)+",
        "",
        payload,
    )
    payload = _clean_text_payload(payload)
    return "" if payload in _EMPTY_TEXT_PAYLOAD_WORDS else payload


def _extract_command_tail_payload(
    schema: PluginCommandSchema,
    message_text: str,
) -> str:
    heads = [schema.head, *schema.aliases]
    for head in heads:
        normalized_head = normalize_message_text(head)
        if not normalized_head:
            continue
        head_index = normalize_message_text(message_text).find(normalized_head)
        if head_index > 0:
            payload = _clean_head_payload(
                normalize_message_text(
                    message_text[head_index + len(normalized_head) :]
                ),
                normalized_head,
            )
            if payload:
                return payload
        parsed = parse_command_with_head(
            message_text,
            normalized_head,
            allow_sticky=True,
            max_prefix_len=12,
        )
        if parsed is None:
            continue
        payload = _clean_head_payload(parsed.payload_text, normalized_head)
        if payload:
            return payload
    return ""


def _slot_accepts_url(slot: CommandSlotSpec) -> bool:
    text = normalize_message_text(
        " ".join([slot.name, slot.description, *slot.aliases])
    ).casefold()
    return any(
        token in text for token in ("链接", "地址", "url", "bv", "av", "视频", "link")
    )


def _extract_url_payload(message_text: str) -> str:
    match = _URL_PAYLOAD_PATTERN.search(normalize_message_text(message_text))
    return match.group(0) if match else ""


def _fill_slots_from_payload(
    merged: dict[str, Any],
    schema: PluginCommandSchema,
    payload: str,
) -> None:
    argument_payload = _clean_text_payload(payload)
    if not argument_payload:
        return
    payload_tokens = [token for token in argument_payload.split(" ") if token]
    token_index = 0
    for slot in schema.slots:
        if slot.name in merged or slot.type == "text":
            continue
        if token_index >= len(payload_tokens):
            break
        value: Any = payload_tokens[token_index]
        token_index += 1
        if slot.type == "int":
            from .slot_extractors import parse_int_token

            parsed_value = parse_int_token(value)
            if parsed_value is None:
                continue
            value = parsed_value
        merged[slot.name] = value
    for slot in schema.slots:
        if slot.name in merged or slot.type != "text":
            continue
        merged[slot.name] = argument_payload
        break


def _fill_link_slots_from_message(
    merged: dict[str, Any],
    schema: PluginCommandSchema,
    message_text: str,
) -> None:
    url_payload = _extract_url_payload(message_text)
    if not url_payload:
        return
    for slot in schema.slots:
        if slot.name in merged or slot.type != "text":
            continue
        if not _slot_accepts_url(slot):
            continue
        merged[slot.name] = url_payload
        return


def complete_slots(
    schema: PluginCommandSchema,
    *,
    slots: dict[str, Any] | None = None,
    message_text: str = "",
    arguments_text: str = "",
) -> tuple[dict[str, Any], list[str]]:
    merged: dict[str, Any] = {}
    for key, value in dict(slots or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not normalize_message_text(value):
            continue
        merged[key] = value
    # Adapter extractors provide optional plugin-specific slot correction without
    # leaking those rules into the generic renderer.
    inferred = _infer_adapter_slots(schema, normalize_message_text(message_text))
    if not inferred and arguments_text:
        inferred = _infer_adapter_slots(schema, normalize_message_text(arguments_text))
    merged.update(
        inferred,
    )
    _fill_link_slots_from_message(merged, schema, message_text)
    argument_payload = normalize_message_text(arguments_text)
    if not argument_payload:
        argument_payload = _extract_command_tail_payload(schema, message_text)
    if argument_payload:
        _fill_slots_from_payload(merged, schema, argument_payload)

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
