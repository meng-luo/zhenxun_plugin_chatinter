"""
ChatInter 命令规划器。

Router 负责判断意图；CommandPlanner 负责把 Router 结果整理成可执行命令，
并用 PluginReference 做轻量约束校验。真正的插件权限和执行仍交给原链路。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .command_schema import complete_slots, render_command, select_command_schema
from .models.pydantic_models import PluginReference
from .route_text import (
    collect_placeholders,
    normalize_message_text,
    strip_invoke_prefix,
)


class CommandPlanDecision(BaseModel):
    action: Literal["execute", "clarify", "usage", "chat"] = Field(default="chat")
    plugin_module: str | None = None
    plugin_name: str | None = None
    command_id: str | None = None
    command_head: str | None = None
    slots: dict[str, object] = Field(default_factory=dict)
    arguments_text: str = ""
    final_command: str | None = None
    missing: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


def _command_head(command: str | None) -> str:
    normalized = normalize_message_text(command or "")
    return normalize_message_text(normalized.split(" ", 1)[0]) if normalized else ""


def _command_tail(command: str | None) -> str:
    normalized = normalize_message_text(command or "")
    if not normalized or " " not in normalized:
        return ""
    return normalize_message_text(normalized.split(" ", 1)[1])


def _find_reference(
    references: list[PluginReference],
    *,
    plugin_module: str | None,
    plugin_name: str | None,
) -> PluginReference | None:
    module = normalize_message_text(plugin_module or "").casefold()
    name = normalize_message_text(plugin_name or "").casefold()
    if module:
        for reference in references:
            if normalize_message_text(reference.module).casefold() == module:
                return reference
        tail_matches = [
            reference
            for reference in references
            if normalize_message_text(reference.module.rsplit(".", 1)[-1]).casefold()
            == module
        ]
        if len(tail_matches) == 1:
            return tail_matches[0]
    if name:
        for reference in references:
            if normalize_message_text(reference.name).casefold() == name:
                return reference
    return None


def _head_in_reference(head: str, reference: PluginReference) -> bool:
    normalized = normalize_message_text(head).casefold()
    if not normalized:
        return False
    for value in [*reference.commands, *reference.aliases]:
        candidate = normalize_message_text(value).casefold()
        if candidate and candidate == normalized:
            return True
    return False


def _default_head(reference: PluginReference | None) -> str:
    if reference is None:
        return ""
    for schema in reference.command_schemas:
        normalized = normalize_message_text(schema.head)
        if normalized:
            return normalized
    for command in reference.commands:
        normalized = normalize_message_text(command)
        if normalized:
            return normalized
    for alias in reference.aliases:
        normalized = normalize_message_text(alias)
        if normalized:
            return normalized
    return ""


def _has_image_context(message_text: str, image_count: int) -> bool:
    if image_count > 0:
        return True
    return any(
        token.lower().startswith("[image")
        for token in collect_placeholders(message_text or "")
    )


def _has_at_context(message_text: str) -> bool:
    return any(
        token.startswith("[@") or token.startswith("@")
        for token in collect_placeholders(message_text or "")
    )


def _text_payload_count(command: str) -> int:
    tail = _command_tail(command)
    if not tail:
        return 0
    for token in collect_placeholders(tail):
        tail = tail.replace(token, " ")
    tail = normalize_message_text(tail)
    if tail in {
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
    }:
        return 0
    return len([item for item in tail.split(" ") if item])


def _merge_command_and_arguments(command: str, arguments_text: str) -> str:
    normalized_command = normalize_message_text(command)
    arguments = normalize_message_text(arguments_text)
    if not normalized_command:
        return arguments
    if not arguments:
        return normalized_command
    if arguments in normalized_command:
        return normalized_command
    return normalize_message_text(f"{normalized_command} {arguments}")


def _extract_command_arguments_from_message(command: str, message_text: str) -> str:
    head = _command_head(command)
    if not head:
        return ""
    normalized_message = normalize_message_text(strip_invoke_prefix(message_text or ""))
    if not normalized_message:
        return ""
    command_text = normalize_message_text(command or "")
    if normalized_message == head:
        return ""
    if normalized_message.startswith(head + " "):
        tail = normalize_message_text(normalized_message[len(head) :])
        if tail in {"一下", "下", "一下吧", "下吧"}:
            return ""
        return tail
    if command_text and normalized_message.startswith(command_text + " "):
        tail = normalize_message_text(normalized_message[len(command_text) :])
        if tail in {"一下", "下", "一下吧", "下吧"}:
            return ""
        return tail
    return ""


def plan_command(
    *,
    action: Literal["chat", "execute", "usage", "clarify"],
    plugin_module: str | None,
    plugin_name: str | None,
    command: str | None,
    command_id: str | None = None,
    slots: dict[str, object] | None = None,
    arguments_text: str = "",
    references: list[PluginReference] | None = None,
    current_message: str = "",
    has_reply: bool = False,
    image_count: int = 0,
    confidence: float = 0.0,
    missing: list[str] | None = None,
    reason: str | None = None,
) -> CommandPlanDecision:
    if action in {"chat", "usage"}:
        return CommandPlanDecision(
            action=action,
            plugin_module=plugin_module,
            plugin_name=plugin_name,
            command_id=command_id,
            command_head=_command_head(command),
            slots=dict(slots or {}),
            arguments_text=normalize_message_text(arguments_text),
            final_command=normalize_message_text(command or "") or None,
            missing=list(missing or []),
            confidence=confidence,
            reason=reason or action,
        )

    reference = _find_reference(
        list(references or []),
        plugin_module=plugin_module,
        plugin_name=plugin_name,
    )
    final_command = _merge_command_and_arguments(command or "", arguments_text)
    effective_arguments_text = normalize_message_text(arguments_text)
    if not effective_arguments_text:
        effective_arguments_text = _extract_command_arguments_from_message(
            final_command or command or "",
            current_message,
        )
    schema = None
    if reference is not None:
        selection = select_command_schema(
            reference.command_schemas,
            command_id=command_id,
            command=final_command or command,
            message_text=current_message,
            arguments_text=effective_arguments_text,
            slots=slots,
            action=action,
        )
        if selection is not None:
            schema = selection.schema
    schema_missing: list[str] = []
    completed_slots = dict(slots or {})
    if schema is not None:
        completed_slots, schema_missing = complete_slots(
            schema,
            slots=completed_slots,
            message_text=current_message,
            arguments_text=effective_arguments_text,
        )
        rendered, schema_missing = render_command(
            schema,
            slots=completed_slots,
            message_text=current_message,
            arguments_text=effective_arguments_text,
        )
        if rendered:
            final_command = rendered
        command_id = schema.command_id

    head = _command_head(final_command)
    if not head:
        head = _default_head(reference)
        final_command = _merge_command_and_arguments(head, arguments_text)

    if (
        schema is None
        and reference is not None
        and head
        and not _head_in_reference(head, reference)
    ):
        default_head = _default_head(reference)
        if default_head:
            final_command = _merge_command_and_arguments(
                default_head,
                _command_tail(final_command),
            )
            head = default_head

    missing_items = [*list(missing or []), *schema_missing]
    if reference is not None:
        requires = (schema.requires if schema is not None else reference.requires) or {}
        context_text = f"{current_message} {final_command}"
        image_satisfied = _has_image_context(context_text, image_count) or (
            requires.get("at") and _has_at_context(context_text)
        )
        if requires.get("image") and not image_satisfied:
            if "image" not in missing_items and "图片" not in missing_items:
                missing_items.append("image")
        if requires.get("reply") and not has_reply:
            if "reply" not in missing_items and "回复" not in missing_items:
                missing_items.append("reply")
        if requires.get("text") and _text_payload_count(final_command) <= 0:
            if "text" not in missing_items and "文本" not in missing_items:
                missing_items.append("text")
        if requires.get("private"):
            if "private" not in missing_items and "私聊" not in missing_items:
                missing_items.append("private")
        if requires.get("to_me"):
            if "to_me" not in missing_items and "@机器人" not in missing_items:
                missing_items.append("to_me")

    planned_action: Literal["execute", "clarify"] = (
        "clarify" if action == "clarify" or missing_items else "execute"
    )
    return CommandPlanDecision(
        action=planned_action,
        plugin_module=plugin_module or (reference.module if reference else None),
        plugin_name=plugin_name or (reference.name if reference else None),
        command_id=command_id,
        command_head=head or None,
        slots=completed_slots,
        arguments_text=effective_arguments_text,
        final_command=normalize_message_text(final_command) or None,
        missing=missing_items,
        confidence=confidence,
        reason=reason or "planned",
    )
