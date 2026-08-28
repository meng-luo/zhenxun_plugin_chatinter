"""Native tool-call route validation.

The LLM chooses a concrete command tool.  This module only validates that call
against the selected command schema and renders the final NoneBot command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .command_index import CommandCandidate
from .command_schema import complete_slots, render_command
from .route_text import normalize_message_text

_AT_PLACEHOLDER_PATTERN = re.compile(r"\[@[^\]\s]+\]")
_AT_INLINE_PATTERN = re.compile(r"@\d{5,20}")
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_ROUTE_TRACE_SAMPLE_LIMIT = 12


@dataclass(frozen=True)
class SkillRouteDecision:
    plugin_name: str
    plugin_module: str
    command: str
    source: str
    skill_kind: str


@dataclass(frozen=True)
class NativeRouteResult:
    decision: SkillRouteDecision
    stage: str
    report: "NativeRouteReport | None" = None
    selected_candidate: CommandCandidate | None = None
    command_id: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    selected_rank: int = 0
    selected_score: float = 0.0
    selected_reason: str = ""


@dataclass
class NativeRouteReport:
    helper_mode: bool
    candidate_total: int = 0
    lexical_candidates: int = 0
    direct_candidates: int = 0
    vector_candidates: int = 0
    attempts: int = 0
    tool_attempts: int = 0
    tool_candidates: int = 0
    tool_choice_count: int = 0
    prompt_full_candidates: int = 0
    candidate_policy_reason: str = ""
    candidate_policy_limit: int = 0
    final_reason: str = "init"
    selected_stage: str | None = None
    selected_plugin: str | None = None
    selected_module: str | None = None
    selected_command: str | None = None
    attempt_modules: list[list[str]] = field(default_factory=list)

    def note_attempt(self, modules: list[str]) -> None:
        self.attempts += 1
        self.attempt_modules.append(modules[:_ROUTE_TRACE_SAMPLE_LIMIT])

    def note_tool_pool(self, tool_count: int, choice_count: int = 0) -> None:
        self.tool_attempts += 1
        self.tool_candidates = max(self.tool_candidates, max(tool_count, 0))
        self.tool_choice_count += max(choice_count, 0)

    def note_prompt_exposure(self, candidates: list[CommandCandidate]) -> None:
        self.prompt_full_candidates = max(
            self.prompt_full_candidates,
            len(candidates),
        )

    def note_tool_choice(self) -> None:
        self.tool_choice_count += 1

    def note_candidate_policy(self, *, reason: str, limit: int) -> None:
        self.candidate_policy_reason = normalize_message_text(reason)[:120]
        self.candidate_policy_limit = max(int(limit or 0), 0)

    def finalize(
        self,
        *,
        reason: str,
        stage: str | None = None,
        plugin_name: str | None = None,
        plugin_module: str | None = None,
        command: str | None = None,
    ) -> None:
        self.final_reason = reason
        if stage is not None:
            self.selected_stage = stage
        if plugin_name is not None:
            self.selected_plugin = plugin_name
        if plugin_module is not None:
            self.selected_module = plugin_module
        if command is not None:
            self.selected_command = command


class NativeSlotValue(BaseModel):
    name: str = Field(description="槽位名称，必须来自候选 schema")
    value: str = Field(default="", description="槽位值，统一以字符串填写")


class NativeRouteDecision(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(
        default="chat",
        description="chat=普通对话；execute=执行插件；usage=查询用法；clarify=需要补充信息",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    plugin_module: str | None = Field(default=None, description="插件模块")
    plugin_name: str | None = Field(default=None, description="插件名称")
    command_id: str | None = Field(default=None, description="命令 schema ID")
    command: str | None = Field(default=None, description="渲染后的插件命令")
    slots: list[NativeSlotValue] = Field(default_factory=list)
    arguments_text: str = ""
    missing: list[str] = Field(default_factory=list)
    reason: str | None = None

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots(cls, value: Any) -> list[NativeSlotValue]:
        return _slots_to_items(value)


class NativeCommandSelection(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(default="chat")
    command_id: str | None = None
    slots: list[NativeSlotValue] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots(cls, value: Any) -> list[NativeSlotValue]:
        return _slots_to_items(value)


def candidate_selection_to_native_route(
    *,
    selection: NativeCommandSelection,
    candidates: list[CommandCandidate],
    message_text: str,
    stage: str,
    candidate: CommandCandidate | None = None,
    has_reply: bool = False,
) -> tuple[NativeRouteDecision, NativeRouteResult | None] | None:
    if selection.action == "chat":
        return (
            NativeRouteDecision(
                action="chat",
                confidence=selection.confidence,
                reason=selection.reason or f"{stage}:chat",
            ),
            None,
        )

    command_id = normalize_message_text(selection.command_id or "")
    if not command_id:
        return None
    selected_candidate = candidate or _find_candidate(candidates, command_id=command_id)
    if selected_candidate is None:
        return None
    if normalize_message_text(selected_candidate.schema.command_id) != command_id:
        return None

    schema = selected_candidate.schema
    slots = _slots_to_dict(selection.slots)
    missing: list[str] = list(selection.missing or [])
    if selection.action == "usage":
        action: Literal["usage", "execute", "clarify"] = "usage"
        rendered = normalize_message_text(schema.head)
        slots = {}
    else:
        slots, schema_missing = complete_slots(
            schema,
            slots=slots,
        )
        rendered, render_missing = render_command(
            schema,
            slots=slots,
        )
        missing.extend(schema_missing)
        missing.extend(render_missing)
        missing.extend(_missing_context(schema, message_text, has_reply=has_reply))
        missing = list(dict.fromkeys(item for item in missing if item))
        action = "execute"

    command = rendered or normalize_message_text(schema.head)
    decision = NativeRouteDecision(
        action=action,
        confidence=selection.confidence,
        plugin_module=selected_candidate.plugin_module,
        plugin_name=selected_candidate.plugin_name,
        command_id=schema.command_id,
        command=command,
        slots=[] if action == "usage" else _slots_to_items(slots),
        missing=[] if action == "usage" else missing,
        reason=selection.reason or f"{stage}:{selected_candidate.reason}",
    )
    return decision, NativeRouteResult(
        decision=SkillRouteDecision(
            plugin_name=selected_candidate.plugin_name,
            plugin_module=selected_candidate.plugin_module,
            command=command,
            source=stage,
            skill_kind=stage,
        ),
        stage=stage,
        selected_candidate=selected_candidate,
        command_id=schema.command_id,
        slots=_slots_to_dict(decision.slots),
        missing=tuple(decision.missing),
        selected_rank=next(
            (
                index
                for index, item in enumerate(candidates, 1)
                if item.schema.command_id == schema.command_id
            ),
            0,
        ),
        selected_score=selected_candidate.score,
        selected_reason=selected_candidate.reason,
    )


def _find_candidate(
    candidates: list[CommandCandidate],
    *,
    command_id: str,
) -> CommandCandidate | None:
    for candidate in candidates:
        if normalize_message_text(candidate.schema.command_id) == command_id:
            return candidate
    return None


def _missing_context(schema: Any, message_text: str, *, has_reply: bool) -> list[str]:
    requires = getattr(schema, "requires", {}) or {}
    missing: list[str] = []
    has_image = bool(_IMAGE_PLACEHOLDER_PATTERN.search(message_text))
    has_at = bool(
        _AT_PLACEHOLDER_PATTERN.search(message_text)
        or _AT_INLINE_PATTERN.search(message_text)
    )
    image_satisfied = has_image or (requires.get("at") and has_at)
    if requires.get("image") and not image_satisfied:
        missing.append("image")
    if requires.get("reply") and not has_reply:
        missing.append("reply")
    if requires.get("at") and not has_at and not (requires.get("image") and has_image):
        missing.append("at")
    return missing


def _slots_to_items(value: Any) -> list[NativeSlotValue]:
    if not value:
        return []
    if isinstance(value, dict):
        return [
            NativeSlotValue(name=str(key), value=str(slot_value))
            for key, slot_value in value.items()
            if normalize_message_text(str(key or ""))
        ]
    items: list[NativeSlotValue] = []
    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, NativeSlotValue):
                name = item.name
                slot_value = item.value
            elif isinstance(item, dict):
                name = str(item.get("name", "") or "")
                slot_value = str(item.get("value", "") or "")
            else:
                continue
            if normalize_message_text(name):
                items.append(NativeSlotValue(name=name, value=slot_value))
    return items


def _slots_to_dict(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _slots_to_items(value):
        name = normalize_message_text(item.name)
        if name:
            result[name] = str(item.value or "")
    return result


__all__ = [
    "NativeCommandSelection",
    "NativeRouteDecision",
    "NativeRouteReport",
    "NativeRouteResult",
    "NativeSlotValue",
    "candidate_selection_to_native_route",
]
