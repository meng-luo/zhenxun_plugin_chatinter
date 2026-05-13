"""Native tool route data structures and local validation helpers."""

from dataclasses import dataclass, field
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .capability_graph import build_capability_graph_snapshot
from .command_index import (
    CommandCandidate,
    build_command_candidates,
)
from .command_schema import complete_slots, render_command
from .config import get_config_value
from .models.pydantic_models import PluginKnowledgeBase
from .plugin_reference import build_command_tool_snapshots
from .route_text import (
    normalize_message_text,
)
from .skill_registry import SkillRouteDecision
from .speech_act import classify_speech_act

_AT_PLACEHOLDER_PATTERN = re.compile(r"\[@[^\]\s]+\]")
_AT_INLINE_PATTERN = re.compile(r"@\d{5,20}")
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_ROUTE_TRACE_SAMPLE_LIMIT = 12


def _action_for_schema(
    *,
    schema: Any,
    speech_act: str,
    missing: list[str] | tuple[str, ...],
) -> Literal["chat", "execute", "usage", "clarify"]:
    role = normalize_message_text(getattr(schema, "command_role", "") or "").lower()
    if speech_act == "ask_usage" or role == "usage":
        return "usage"
    if role == "template" and missing:
        requires = getattr(schema, "requires", {}) or {}
        context_missing = {"image", "reply", "at", "context", "target"}
        if requires.get("image") or requires.get("reply") or requires.get("at"):
            if set(missing).issubset(context_missing):
                return "execute"
    if missing:
        return "clarify"
    return "execute"


def _has_payload_for_schema(schema: Any, message_text: str) -> bool:
    normalized = normalize_message_text(message_text)
    if not normalized:
        return False
    values = [
        normalize_message_text(getattr(schema, "head", "") or ""),
        *[
            normalize_message_text(alias)
            for alias in getattr(schema, "aliases", []) or []
            if normalize_message_text(alias)
        ],
    ]
    text = normalized
    for value in values:
        if value and value in text:
            text = text.replace(value, " ", 1)
    text = _AT_PLACEHOLDER_PATTERN.sub(" ", text)
    text = _IMAGE_PLACEHOLDER_PATTERN.sub(" ", text)
    for noise in (
        "帮我",
        "给我",
        "请",
        "麻烦",
        "做",
        "做个",
        "做一张",
        "来张",
        "来一张",
        "用",
        "让",
        "写",
        "内容是",
        "文字是",
        "牌子写",
        "一句",
        "一段",
        "这个",
        "这张",
        "表情",
        "表情包",
        "梗图",
        "图片",
        "：",
        ":",
        "，",
        ",",
    ):
        text = text.replace(noise, " ")
    return bool(normalize_message_text(text))


def _payload_slot_items(schema: Any, message_text: str) -> list["NativeSlotValue"]:
    if not _has_payload_for_schema(schema, message_text):
        return []
    slots = list(getattr(schema, "slots", []) or [])
    target = next(
        (
            slot
            for slot in slots
            if getattr(slot, "type", "") == "text"
            and bool(getattr(slot, "required", False))
        ),
        None,
    )
    if target is None:
        return []
    normalized = normalize_message_text(message_text)
    head = normalize_message_text(getattr(schema, "head", "") or "")
    payload = normalized
    aliases = [
        normalize_message_text(alias)
        for alias in getattr(schema, "aliases", []) or []
        if normalize_message_text(alias)
    ]
    for marker in [head, *aliases]:
        if marker and marker in payload:
            before, _sep, after = payload.partition(marker)
            payload = after or before
            break
    for prefix in (
        "做一句",
        "做一段",
        "说",
        "写",
        "内容是",
        "文字是",
        "牌子写",
        "：",
        ":",
        "，",
        ",",
        "一下",
    ):
        payload = payload.replace(prefix, " ")
    payload = normalize_message_text(payload)
    if not payload:
        return []
    return [NativeSlotValue(name=str(getattr(target, "name", "")), value=payload)]


@dataclass(frozen=True)
class NativeRouteResult:
    decision: SkillRouteDecision
    stage: str
    report: "NativeRouteReport | None" = None
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


class NativeRouteDecision(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(
        default="chat",
        description="chat=普通对话；execute=执行插件；usage=查询用法；clarify=需要补充信息",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    plugin_module: str | None = Field(default=None, description="必须来自 plugin_cards")
    plugin_name: str | None = Field(default=None, description="插件名称，可选")
    command_id: str | None = Field(default=None, description="优先使用的命令 schema ID")
    command: str | None = Field(default=None, description="插件命令头")
    slots: list[NativeSlotValue] = Field(
        default_factory=list,
        description="命令槽位列表，格式为 [{name,value}]，不要使用任意对象键",
    )
    arguments_text: str = Field(default="", description="命令后的自然语言参数")
    missing: list[str] = Field(default_factory=list, description="缺失信息")
    reason: str | None = Field(default=None, description="简短原因")

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots(cls, value: Any) -> list[NativeSlotValue]:
        return _slots_to_items(value)


class NativeCommandSelection(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(
        default="chat",
        description="chat=普通对话；execute=执行候选命令；usage=查询候选命令用法；clarify=需要补充信息",
    )
    command_id: str | None = Field(
        default=None,
        description="必须来自 candidates.command_id；chat 时为空",
    )
    slots: list[NativeSlotValue] = Field(
        default_factory=list,
        description=(
            "按候选 schema 填写的槽位列表，格式为 [{name,value}]，"
            "不要臆造 schema 之外的槽位"
        ),
    )
    missing: list[str] = Field(
        default_factory=list,
        description="缺失的必填槽位或上下文，例如 text/image/reply/target",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="简短选择理由")

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots(cls, value: Any) -> list[NativeSlotValue]:
        return _slots_to_items(value)


def _message_context_flags(
    message_text: str, *, has_reply: bool = False
) -> dict[str, bool]:
    return {
        "has_image": bool(_IMAGE_PLACEHOLDER_PATTERN.search(message_text)),
        "has_at": bool(
            _AT_PLACEHOLDER_PATTERN.search(message_text)
            or _AT_INLINE_PATTERN.search(message_text)
        ),
        "has_reply": has_reply,
    }


def _ensure_command_tools(
    knowledge_base: PluginKnowledgeBase,
    command_tools: list[Any] | None,
) -> list[Any]:
    if command_tools:
        return list(command_tools)
    graph = build_capability_graph_snapshot(knowledge_base)
    return list(build_command_tool_snapshots(graph))


def build_native_command_candidate_pool(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None = None,
    command_tools: list[Any] | None = None,
    limit: int | None = None,
    diversify: bool = True,
    include_unscored: bool = False,
) -> list[CommandCandidate]:
    tools = _ensure_command_tools(knowledge_base, command_tools)
    candidate_limit = limit
    if candidate_limit is None:
        if include_unscored:
            candidate_limit = len(tools)
        else:
            candidate_limit = max(
                int(get_config_value("ROUTE_COMMAND_CANDIDATE_LIMIT", 32) or 32),
                8,
            )
    return build_command_candidates(
        knowledge_base,
        message_text,
        limit=candidate_limit,
        session_id=session_key,
        diversify=diversify,
        tools=tools,
        include_unscored=include_unscored,
    )


def _selection_matches_command_context(
    *,
    selection: NativeCommandSelection,
    candidate: CommandCandidate,
    message_text: str,
    has_reply: bool,
) -> tuple[bool, str]:
    if selection.action == "usage":
        return True, ""
    requires = candidate.schema.requires or {}
    flags = _message_context_flags(message_text, has_reply=has_reply)
    image_satisfied = flags["has_image"] or (requires.get("at") and flags["has_at"])
    if requires.get("image") and not image_satisfied:
        return False, "missing image context"
    if requires.get("reply") and not flags["has_reply"]:
        return False, "missing reply context"
    if (
        requires.get("at")
        and not flags["has_at"]
        and not (requires.get("image") and flags["has_image"])
    ):
        return False, "missing at context"
    return True, ""


def candidate_selection_to_native_route(
    *,
    selection: NativeCommandSelection,
    candidates: list[CommandCandidate],
    message_text: str,
    stage: str,
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
    candidate = next(
        (
            item
            for item in candidates
            if normalize_message_text(item.schema.command_id) == command_id
        ),
        None,
    )
    if candidate is None:
        return None
    context_ok, context_reason = _selection_matches_command_context(
        selection=selection,
        candidate=candidate,
        message_text=message_text,
        has_reply=has_reply,
    )
    if not context_ok:
        missing_name = context_reason.rsplit(" ", 1)[-1] or "context"
        selection = NativeCommandSelection(
            action="clarify",
            command_id=candidate.schema.command_id,
            slots=_slots_to_items(selection.slots),
            missing=[*selection.missing, missing_name],
            confidence=min(selection.confidence, 0.82),
            reason=f"{selection.reason};validator:{context_reason}",
        )

    schema = candidate.schema
    route_slots = _slots_to_dict(selection.slots)
    if not route_slots:
        route_slots.update(_slots_to_dict(_payload_slot_items(schema, message_text)))
    if selection.action == "usage":
        rendered = schema.head
        schema_missing: list[str] | tuple[str, ...] = ()
        route_slots = {}
    else:
        route_slots, schema_missing = complete_slots(
            schema,
            slots=route_slots,
            message_text=message_text,
            arguments_text="",
        )
        if schema_missing and _has_payload_for_schema(schema, message_text):
            schema_missing = []
        rendered, schema_missing = render_command(
            schema,
            slots=route_slots,
            message_text=message_text,
            arguments_text="",
        )
        if schema_missing and _has_payload_for_schema(schema, message_text):
            schema_missing = []
    missing = [*selection.missing, *list(schema_missing)]
    action = _action_for_schema(
        schema=schema,
        speech_act=classify_speech_act(
            message_text,
            **_message_context_flags(message_text, has_reply=has_reply),
        ),
        missing=missing,
    )
    if selection.action == "clarify":
        action = "clarify"
    elif selection.action == "usage":
        action = "usage"
    if action == "execute" and missing:
        action = "clarify"
    if (
        action == "clarify"
        and selection.action == "execute"
        and getattr(schema, "command_role", "") == "template"
        and (getattr(schema, "requires", {}) or {}).get("image")
    ):
        context_missing = {"image", "reply", "at", "context", "target"}
        if (
            set(missing).issubset(context_missing)
            and _message_context_flags(
                message_text,
                has_reply=has_reply,
            )["has_image"]
        ):
            action = "execute"
            missing = []

    command = schema.head if action == "usage" else rendered or schema.head
    decision = NativeRouteDecision(
        action=action,
        confidence=selection.confidence,
        plugin_module=candidate.plugin_module,
        plugin_name=candidate.plugin_name,
        command_id=schema.command_id,
        command=command,
        slots=[] if action == "usage" else _slots_to_items(route_slots),
        missing=[] if action == "usage" else missing,
        reason=selection.reason or f"{stage}:{candidate.reason}",
    )
    native_route = NativeRouteResult(
        decision=SkillRouteDecision(
            plugin_name=candidate.plugin_name,
            plugin_module=candidate.plugin_module,
            command=command,
            source=stage,
            skill_kind=stage,
        ),
        stage=stage,
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
        selected_score=candidate.score,
        selected_reason=candidate.reason,
    )
    return decision, native_route


__all__ = [
    "NativeCommandSelection",
    "NativeRouteDecision",
    "NativeRouteReport",
    "NativeRouteResult",
    "NativeSlotValue",
    "build_native_command_candidate_pool",
    "candidate_selection_to_native_route",
]
