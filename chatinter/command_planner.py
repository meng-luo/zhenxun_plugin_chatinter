"""Local command validator/renderer.

Command selection now belongs to the LLM tool call.  This module deliberately
does not pick another command; it only validates the selected schema and renders
the command text from provided slots.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .command_schema import complete_slots, render_command, select_command_schema
from .models.pydantic_models import PluginReference
from .route_text import normalize_message_text


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
    ambient_message: str = "",
    has_reply: bool = False,
    image_count: int = 0,
    confidence: float = 0.0,
    missing: list[str] | None = None,
    reason: str | None = None,
) -> CommandPlanDecision:
    _ = (ambient_message, has_reply, image_count)
    normalized_command = normalize_message_text(command or "")
    normalized_args = normalize_message_text(arguments_text)
    selected_slots = dict(slots or {})
    missing_items = list(missing or [])

    if action in {"chat", "usage"}:
        return CommandPlanDecision(
            action=action,
            plugin_module=plugin_module,
            plugin_name=plugin_name,
            command_id=command_id,
            command_head=_command_head(normalized_command),
            slots=selected_slots,
            arguments_text=normalized_args,
            final_command=normalized_command or None,
            missing=missing_items,
            confidence=confidence,
            reason=reason or action,
        )

    schema = _select_schema(
        references=list(references or []),
        plugin_module=plugin_module,
        plugin_name=plugin_name,
        command_id=command_id,
        command=normalized_command,
        message_text=current_message,
        arguments_text=normalized_args,
        slots=selected_slots,
        action=action,
    )
    if schema is not None:
        selected_slots, schema_missing = complete_slots(
            schema,
            slots=selected_slots,
            message_text=current_message,
            arguments_text=normalized_args,
        )
        rendered, render_missing = render_command(
            schema,
            slots=selected_slots,
            message_text=current_message,
            arguments_text=normalized_args,
        )
        normalized_command = rendered or normalized_command or schema.head
        command_id = schema.command_id
        missing_items.extend(schema_missing)
        missing_items.extend(render_missing)

    missing_items = list(dict.fromkeys(item for item in missing_items if item))
    planned_action: Literal["execute", "clarify"] = (
        "clarify" if action == "clarify" or missing_items else "execute"
    )
    return CommandPlanDecision(
        action=planned_action,
        plugin_module=plugin_module,
        plugin_name=plugin_name,
        command_id=command_id,
        command_head=_command_head(normalized_command),
        slots=selected_slots,
        arguments_text=normalized_args,
        final_command=normalized_command or None,
        missing=missing_items,
        confidence=confidence,
        reason=reason or "validated_render",
    )


def _select_schema(
    *,
    references: list[PluginReference],
    plugin_module: str | None,
    plugin_name: str | None,
    command_id: str | None,
    command: str,
    message_text: str,
    arguments_text: str,
    slots: dict[str, object],
    action: str,
):
    reference = _find_reference(
        references,
        plugin_module=plugin_module,
        plugin_name=plugin_name,
    )
    if reference is None:
        return None
    selection = select_command_schema(
        reference.command_schemas,
        command_id=command_id,
        command=command,
        message_text=message_text,
        arguments_text=arguments_text,
        slots=slots,
        action=action,
    )
    return selection.schema if selection is not None else None


def _find_reference(
    references: list[PluginReference],
    *,
    plugin_module: str | None,
    plugin_name: str | None,
) -> PluginReference | None:
    module = normalize_message_text(plugin_module or "").casefold()
    name = normalize_message_text(plugin_name or "").casefold()
    for reference in references:
        if module and normalize_message_text(reference.module).casefold() == module:
            return reference
        if name and normalize_message_text(reference.name).casefold() == name:
            return reference
    return None


def _command_head(command: str | None) -> str:
    normalized = normalize_message_text(command or "")
    return normalize_message_text(normalized.split(" ", 1)[0]) if normalized else ""


__all__ = ["CommandPlanDecision", "plan_command"]
