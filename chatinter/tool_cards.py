"""Stable model-visible cards for ChatInter plugin capabilities."""

from __future__ import annotations

from typing import Any

from .models.pydantic_models import CommandToolSnapshot
from .route_text import normalize_message_text
from .schema_policy import resolve_command_target_policy

_TEXT_LIMIT = 240
_USAGE_LIMIT = 600
_CANDIDATE_LIST_LIMIT = 8
_CANDIDATE_EXAMPLE_LIMIT = 3
def project_command_card(snapshot: CommandToolSnapshot) -> dict[str, Any]:
    card: dict[str, Any] = {
        "command_id": _clip(snapshot.command_id, 160),
        "plugin": _clip(snapshot.plugin_name, 120),
        "head": _clip(snapshot.head, 120),
        "aliases": [_clip(value, 120) for value in snapshot.aliases if _text(value)],
        "description": _clip(snapshot.description or snapshot.capability_text, 320),
        "command_role": snapshot.command_role,
        "slots": [_slot_card(slot) for slot in snapshot.slots],
        "render": _clip(snapshot.render or snapshot.head, 320),
        "accepted_inputs": _accepted_inputs(snapshot),
        "required_context": _required_context(snapshot),
    }
    usage = _text(snapshot.usage)
    if usage:
        card["usage"] = _clip(usage, _USAGE_LIMIT)
    examples = [_clip(value, 280) for value in snapshot.examples if _text(value)]
    if examples:
        card["examples"] = examples
    constraints = _constraints(snapshot)
    if constraints:
        card["constraints"] = constraints
    return _drop_empty(card)


def project_command_candidate_card(snapshot: CommandToolSnapshot) -> dict[str, Any]:
    card = project_command_card(snapshot)
    projected = {
        key: card[key]
        for key in (
            "command_id",
            "plugin",
            "head",
            "aliases",
            "description",
            "command_role",
            "slots",
            "render",
            "accepted_inputs",
            "required_context",
        )
        if key in card
    }
    for key in ("aliases",):
        values = projected.get(key)
        if isinstance(values, list):
            projected[key] = values[:_CANDIDATE_LIST_LIMIT]
    for key in ("examples",):
        values = card.get(key)
        if isinstance(values, list) and values:
            projected[key] = values[:_CANDIDATE_EXAMPLE_LIMIT]
    return _drop_empty(projected)


def _slot_card(slot: Any) -> dict[str, Any]:
    card = {
        "name": _clip(getattr(slot, "name", ""), 80),
        "type": _clip(getattr(slot, "type", ""), 40),
        "required": bool(getattr(slot, "required", False)),
        "aliases": [
            _clip(value, 80)
            for value in getattr(slot, "aliases", ()) or ()
            if _text(value)
        ],
        "description": _clip(getattr(slot, "description", ""), 180),
        "choices": [
            _clip(value, 80)
            for value in getattr(slot, "choices", ()) or ()
            if _text(value)
        ],
    }
    default = getattr(slot, "default", None)
    if default not in (None, ""):
        card["default"] = default
    return _drop_empty(card)


def _accepted_inputs(snapshot: CommandToolSnapshot) -> list[dict[str, Any]]:
    requires = dict(snapshot.requires or {})
    policy = resolve_command_target_policy(snapshot)
    image_required = (
        bool(requires.get("image"))
        or (snapshot.payload_policy == "image_only")
        or any(slot.type == "image" and slot.required for slot in snapshot.slots)
    )
    text_or_image = snapshot.payload_policy == "text_or_image"
    target_required = bool(requires.get("at")) or (
        snapshot.target_requirement == "required"
    )
    summaries: list[dict[str, Any]] = []
    if image_required or target_required or text_or_image:
        sources: list[str] = []
        if text_or_image:
            sources.append("文本参数")
        if policy.allow_image_as_target:
            sources.append("当前消息图片")
        if policy.allow_reply_image_as_target:
            sources.append("回复消息图片")
        if policy.allow_at_as_target:
            sources.extend(("@用户头像", "用户昵称"))
        labels = {
            "at": "@用户",
            "reply": "回复消息",
            "nickname": "用户昵称",
            "self": "当前用户",
        }
        for source in snapshot.target_sources:
            label = labels.get(source, source)
            if label and label not in sources:
                sources.append(label)
        if sources:
            summaries.append(
                {
                    "for": (
                        "图片或目标"
                        if image_required and target_required
                        else "文本或图片"
                        if text_or_image
                        else "图片"
                        if image_required
                        else "目标"
                    ),
                    "any_of": sources,
                }
            )
    if requires.get("text") and not any(
        slot.type == "text" and slot.required for slot in snapshot.slots
    ):
        summaries.append({"for": "文本", "any_of": ["task_text", "payload_hint"]})
    return summaries


def _required_context(snapshot: CommandToolSnapshot) -> list[str]:
    return ["回复消息"] if bool((snapshot.requires or {}).get("reply")) else []


def _constraints(snapshot: CommandToolSnapshot) -> list[str]:
    return ["仅对当前用户"] if snapshot.actor_scope == "self_only" else []


def _drop_empty(
    payload: dict[str, Any],
    *,
    keep_zero: set[str] | None = None,
) -> dict[str, Any]:
    keep_zero = keep_zero or set()
    return {
        key: value
        for key, value in payload.items()
        if key in keep_zero or value not in (None, "", [], (), {})
    }


def _text(value: object) -> str:
    return normalize_message_text(str(value or ""))


def _clip(value: object, limit: int = _TEXT_LIMIT) -> str:
    text = _text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)].rstrip() + "…"


__all__ = [
    "project_command_candidate_card",
    "project_command_card",
]
