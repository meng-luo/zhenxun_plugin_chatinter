from dataclasses import dataclass, field
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from zhenxun.services.log import logger

from .capability_graph import build_capability_graph_snapshot
from .command_index import (
    CommandCandidate,
    build_candidate_snapshots,
    build_command_candidates,
)
from .command_schema import complete_slots, render_command
from .config import get_config_value
from .models.pydantic_models import PluginKnowledgeBase
from .plugin_adapters import (
    extract_adapter_slots,
    resolve_adapter_clarify_route,
)
from .plugin_reference import build_command_tool_snapshots
from .route_text import (
    STRONG_EXECUTE_WORDS,
    contains_any,
    has_chat_context_hint,
    has_negative_route_intent,
    is_usage_question,
    match_command_head,
    normalize_message_text,
    strip_invoke_prefix,
)
from .skill_registry import SkillRouteDecision
from .speech_act import classify_speech_act
from .tool_candidate_policy import select_rerank_candidates
from .tool_query_rewriter import (
    ToolQueryDescriptor,
    build_tool_query_descriptor,
    expand_tool_queries,
)
from .tool_reranker import ToolRerankDecision, request_tool_rerank
from .turn_runtime import TurnBudgetController

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


def _payload_slot_items(schema: Any, message_text: str) -> list["LLMSlotValue"]:
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
    return [LLMSlotValue(name=str(getattr(target, "name", "")), value=payload)]


@dataclass(frozen=True)
class RouteResolveResult:
    decision: SkillRouteDecision
    stage: str
    report: "RouteAttemptReport | None" = None
    command_id: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    selected_rank: int = 0
    selected_score: float = 0.0
    selected_reason: str = ""


@dataclass
class RouteAttemptReport:
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
    prompt_compact_candidates: int = 0
    prompt_name_only_candidates: int = 0
    query_expansion_attempts: int = 0
    query_expansion_success: int = 0
    query_expansion_query: str = ""
    query_expansion_reason: str = ""
    candidate_policy_reason: str = ""
    candidate_policy_limit: int = 0
    rerank_attempts: int = 0
    rerank_success: int = 0
    rerank_no_available: int = 0
    rerank_stage: str = ""
    rerank_reason: str = ""
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
        snapshots = build_candidate_snapshots(candidates)
        self.prompt_full_candidates = max(
            self.prompt_full_candidates,
            sum(1 for item in snapshots if item.prompt_level == "full"),
        )
        self.prompt_compact_candidates = max(
            self.prompt_compact_candidates,
            sum(1 for item in snapshots if item.prompt_level == "compact"),
        )
        self.prompt_name_only_candidates = max(
            self.prompt_name_only_candidates,
            sum(1 for item in snapshots if item.prompt_level == "name_only"),
        )

    def note_query_expansion(
        self,
        descriptor: ToolQueryDescriptor | None,
        *,
        queries: list[str],
    ) -> None:
        self.query_expansion_attempts += 1
        if descriptor is not None and len(queries) > 1:
            self.query_expansion_success += 1
            self.query_expansion_query = normalize_message_text(
                descriptor.capability_query
            )[:160]
            self.query_expansion_reason = normalize_message_text(
                descriptor.reason or descriptor.action
            )[:160]

    def note_candidate_policy(self, *, reason: str, limit: int) -> None:
        self.candidate_policy_reason = normalize_message_text(reason)[:120]
        self.candidate_policy_limit = max(int(limit or 0), 0)

    def note_rerank(
        self,
        *,
        stage: str,
        decision: ToolRerankDecision | None,
        success: bool,
    ) -> None:
        self.rerank_attempts += 1
        self.rerank_stage = normalize_message_text(stage)[:80]
        if success:
            self.rerank_success += 1
        if decision is None:
            return
        if decision.action == "no_available_tool":
            self.rerank_no_available += 1
        self.rerank_reason = normalize_message_text(
            decision.reason or decision.no_available_tool_reason or decision.action
        )[:160]

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


class LLMSlotValue(BaseModel):
    name: str = Field(description="槽位名称，必须来自候选 schema")
    value: str = Field(default="", description="槽位值，统一以字符串填写")


def _slots_to_items(value: Any) -> list[LLMSlotValue]:
    if not value:
        return []
    if isinstance(value, dict):
        return [
            LLMSlotValue(name=str(key), value=str(slot_value))
            for key, slot_value in value.items()
            if normalize_message_text(str(key or ""))
        ]
    items: list[LLMSlotValue] = []
    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, LLMSlotValue):
                name = item.name
                slot_value = item.value
            elif isinstance(item, dict):
                name = str(item.get("name", "") or "")
                slot_value = str(item.get("value", "") or "")
            else:
                continue
            if normalize_message_text(name):
                items.append(LLMSlotValue(name=name, value=slot_value))
    return items


def _slots_to_dict(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _slots_to_items(value):
        name = normalize_message_text(item.name)
        if name:
            result[name] = str(item.value or "")
    return result


class LLMRouterDecision(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(
        default="chat",
        description="chat=普通对话；execute=执行插件；usage=查询用法；clarify=需要补充信息",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    plugin_module: str | None = Field(default=None, description="必须来自 plugin_cards")
    plugin_name: str | None = Field(default=None, description="插件名称，可选")
    command_id: str | None = Field(default=None, description="优先使用的命令 schema ID")
    command: str | None = Field(default=None, description="插件命令头")
    slots: list[LLMSlotValue] = Field(
        default_factory=list,
        description="命令槽位列表，格式为 [{name,value}]，不要使用任意对象键",
    )
    arguments_text: str = Field(default="", description="命令后的自然语言参数")
    missing: list[str] = Field(default_factory=list, description="缺失信息")
    reason: str | None = Field(default=None, description="简短原因")

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots(cls, value: Any) -> list[LLMSlotValue]:
        return _slots_to_items(value)


class LLMCommandSelection(BaseModel):
    action: Literal["chat", "execute", "usage", "clarify"] = Field(
        default="chat",
        description="chat=普通对话；execute=执行候选命令；usage=查询候选命令用法；clarify=需要补充信息",
    )
    command_id: str | None = Field(
        default=None,
        description="必须来自 candidates.command_id；chat 时为空",
    )
    slots: list[LLMSlotValue] = Field(
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
    def _validate_slots(cls, value: Any) -> list[LLMSlotValue]:
        return _slots_to_items(value)


def _is_exact_command_candidate(candidate: CommandCandidate | None) -> bool:
    return bool(candidate and candidate.exact_protected)


def _should_force_chat_before_llm(message_text: str, speech_act: str) -> bool:
    normalized = normalize_message_text(message_text)
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if has_negative_route_intent(normalized) or has_negative_route_intent(stripped):
        return True
    if speech_act == "discuss_command":
        return not contains_any(stripped, STRONG_EXECUTE_WORDS)
    if speech_act != "casual_chat":
        return False
    if not has_chat_context_hint(stripped):
        return False
    return not contains_any(stripped, STRONG_EXECUTE_WORDS)


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


def _has_exact_tool_head(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    command_tools: list[Any] | None,
) -> bool:
    normalized = normalize_message_text(message_text)
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if not normalized:
        return False
    texts = [normalized]
    if stripped and stripped != normalized:
        texts.append(stripped)
    for tool in _ensure_command_tools(knowledge_base, command_tools):
        terms = [
            normalize_message_text(getattr(tool, "head", "") or ""),
            *[
                normalize_message_text(alias)
                for alias in getattr(tool, "aliases", []) or []
                if normalize_message_text(alias)
            ],
        ]
        if any(
            term and any(match_command_head(text, term) for text in texts)
            for term in terms
        ):
            return True
    return False


def _ensure_command_tools(
    knowledge_base: PluginKnowledgeBase,
    command_tools: list[Any] | None,
) -> list[Any]:
    if command_tools:
        return list(command_tools)
    graph = build_capability_graph_snapshot(knowledge_base)
    return list(build_command_tool_snapshots(graph))


def _build_command_candidate_pool(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None = None,
    command_tools: list[Any] | None = None,
    expanded_queries: list[str] | None = None,
    limit: int | None = None,
    diversify: bool = True,
) -> list[CommandCandidate]:
    tools = _ensure_command_tools(knowledge_base, command_tools)
    candidate_limit = limit
    if candidate_limit is None:
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
        expanded_queries=expanded_queries,
    )


def _has_adapter_slot_signal(candidate: CommandCandidate, message_text: str) -> bool:
    try:
        slots = extract_adapter_slots(candidate.schema.command_id, message_text)
    except Exception:
        return False
    if any(
        slot.name in slots
        for slot in candidate.schema.slots
        if normalize_message_text(slot.name)
    ):
        return True
    if not slots:
        return False
    requires = candidate.schema.requires or {}
    return bool(
        (requires.get("image") and "image" in slots)
        or (requires.get("reply") and "reply" in slots)
        or (requires.get("at") and ("at" in slots or "target" in slots))
    )


def _rerank_to_command_selection(
    decision: ToolRerankDecision,
) -> LLMCommandSelection | None:
    if decision.action == "no_available_tool":
        return LLMCommandSelection(
            action="chat",
            confidence=decision.confidence,
            reason=(
                "no_available_tool:"
                + normalize_message_text(
                    decision.no_available_tool_reason or decision.reason
                )
            ),
        )
    if decision.action == "chat":
        return LLMCommandSelection(
            action="chat",
            confidence=decision.confidence,
            reason=decision.reason or "tool_rerank_chat",
        )
    command_id = normalize_message_text(decision.command_id or "")
    if not command_id:
        return None
    return LLMCommandSelection(
        action=decision.action,
        command_id=command_id,
        slots=[
            LLMSlotValue(name=item.name, value=item.value)
            for item in decision.slots
            if normalize_message_text(item.name)
        ],
        missing=list(decision.missing),
        confidence=decision.confidence,
        reason=decision.reason or f"tool_rerank:{decision.action}",
    )


def _fallback_candidate_selection(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    has_reply: bool,
) -> LLMCommandSelection | None:
    if not candidates:
        return None
    flags = _message_context_flags(message_text, has_reply=has_reply)
    speech_act = classify_speech_act(message_text, **flags)
    if _should_force_chat_before_llm(message_text, speech_act):
        return LLMCommandSelection(
            action="chat",
            confidence=0.82,
            reason=f"speech_act:{speech_act}",
        )

    top = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = top.score - second_score
    if speech_act == "ask_usage" and (
        top.exact_protected or top.score >= 130.0 or is_usage_question(message_text)
    ):
        return LLMCommandSelection(
            action="usage",
            command_id=top.schema.command_id,
            confidence=min(max(top.score / 360.0, 0.0), 0.88),
            reason=f"local_unavailable_usage:{top.reason}",
        )

    # Local execution is only a safety net for classifier outages. Fuzzy or semantic
    # matches stay in chat unless the user hit an actual command head/alias with a
    # clear margin. Normal routing decisions belong to the LLM reranker.
    if not _is_exact_command_candidate(top):
        return None
    if speech_act in {"casual_chat", "discuss_command"}:
        return None
    if margin < 18.0 and top.schema.command_role not in {"catalog", "helper"}:
        return None

    slots, missing = complete_slots(
        top.schema,
        slots={},
        message_text=message_text,
        arguments_text="",
    )
    payload_items = _payload_slot_items(top.schema, message_text)
    if payload_items:
        slots.update(_slots_to_dict(payload_items))
        missing = []
    elif missing and _has_payload_for_schema(top.schema, message_text):
        missing = []
    action = _action_for_schema(
        schema=top.schema,
        speech_act=speech_act,
        missing=missing,
    )
    return LLMCommandSelection(
        action=action,
        command_id=top.schema.command_id,
        slots=_slots_to_items(slots),
        missing=[] if action == "usage" else list(missing),
        confidence=min(max(top.score / 420.0, 0.0), 0.86),
        reason=(
            "local_unavailable_exact:"
            f"{top.reason};score={top.score:.1f};margin={margin:.1f}"
        ),
    )


def _selection_matches_command_context(
    *,
    selection: LLMCommandSelection,
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


def _candidate_selection_to_route_result(
    *,
    selection: LLMCommandSelection,
    candidates: list[CommandCandidate],
    message_text: str,
    stage: str,
    has_reply: bool = False,
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    if selection.action == "chat":
        return (
            LLMRouterDecision(
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
        selection = LLMCommandSelection(
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
    decision = LLMRouterDecision(
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
    route_result = RouteResolveResult(
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
    return decision, route_result


async def _resolve_candidate_selection_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    report: RouteAttemptReport,
    command_tools: list[Any] | None = None,
    candidates: list[CommandCandidate] | None = None,
    stage: str = "command_selector",
    use_reranker: bool = False,
    recovery_query: str = "",
    recovery_reason: str = "",
    expanded_queries: list[str] | None = None,
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    if candidates is None:
        candidates = _build_command_candidate_pool(
            message_text,
            knowledge_base,
            session_key=session_key,
            command_tools=command_tools,
            expanded_queries=expanded_queries,
        )
    if not candidates:
        return None

    policy = select_rerank_candidates(
        candidates,
        stage=stage,
        base_limit=max(
            int(get_config_value("ROUTE_COMMAND_RERANK_LIMIT", 24) or 24),
            8,
        ),
        max_limit=max(
            int(get_config_value("ROUTE_COMMAND_RERANK_MAX_LIMIT", 48) or 48),
            16,
        ),
    )
    report.note_candidate_policy(reason=policy.reason, limit=policy.limit)
    candidates = policy.candidates
    report.candidate_total = max(report.candidate_total, len(candidates))
    report.note_tool_pool(len(candidates))
    report.note_prompt_exposure(candidates)

    # Very weak candidates are better left to query expansion or the chat fallback.
    if (
        stage != "query_expansion"
        and candidates[0].score < 80.0
        and not candidates[0].exact_protected
        and not any(
            _has_adapter_slot_signal(item, message_text) for item in candidates[:8]
        )
    ):
        return None

    selection: LLMCommandSelection | None = None
    rerank_decision: ToolRerankDecision | None = None
    try:
        rerank_decision = await request_tool_rerank(
            message_text=message_text,
            candidates=candidates,
            has_reply=has_reply,
            stage=stage if use_reranker else "command_selector",
            session_key=session_key,
            budget_controller=budget_controller,
            recovery_query=recovery_query,
            recovery_reason=recovery_reason,
        )
        report.note_rerank(
            stage=stage,
            decision=rerank_decision,
            success=bool(
                rerank_decision
                and rerank_decision.action in {"execute", "usage", "clarify"}
            ),
        )
        if rerank_decision is not None:
            selection = _rerank_to_command_selection(rerank_decision)
    except Exception as exc:
        logger.debug(f"ChatInter tool rerank failed, trying local fallback: {exc}")

    if selection is None and rerank_decision is None:
        clarify_route = resolve_adapter_clarify_route(message_text, candidates)
        if clarify_route is not None:
            selection = LLMCommandSelection(
                action="clarify",
                command_id=clarify_route.command_id,
                missing=list(clarify_route.missing),
                confidence=clarify_route.confidence,
                reason=f"adapter_fallback:{clarify_route.reason}",
            )
        else:
            selection = _fallback_candidate_selection(
                message_text=message_text,
                candidates=candidates,
                has_reply=has_reply,
            )

    if selection is None:
        return None
    result = _candidate_selection_to_route_result(
        selection=selection,
        candidates=candidates,
        message_text=message_text,
        stage=stage,
        has_reply=has_reply,
    )
    if result is not None:
        report.tool_choice_count += 1
    return result


async def _resolve_query_expansion_route(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    report: RouteAttemptReport,
    command_tools: list[Any] | None = None,
) -> tuple[LLMRouterDecision, RouteResolveResult | None] | None:
    tools = _ensure_command_tools(knowledge_base, command_tools)
    if not tools:
        return None
    descriptor = await build_tool_query_descriptor(
        message_text,
        tools=tools,
        has_reply=has_reply,
    )
    queries = expand_tool_queries(message_text, descriptor)
    report.note_query_expansion(descriptor, queries=queries)
    if descriptor is None or descriptor.action != "expand" or len(queries) <= 1:
        return None
    candidates = _build_command_candidate_pool(
        message_text,
        knowledge_base,
        session_key=session_key,
        command_tools=tools,
        expanded_queries=queries[1:],
        diversify=False,
    )
    if not candidates:
        return None
    top = candidates[0]
    if top.score < 88.0 and not top.exact_protected:
        return None
    result = await _resolve_candidate_selection_route(
        message_text,
        knowledge_base,
        session_key=session_key,
        budget_controller=budget_controller,
        has_reply=has_reply,
        report=report,
        command_tools=tools,
        candidates=candidates,
        stage="query_expansion",
        use_reranker=True,
        recovery_query=descriptor.capability_query,
        recovery_reason=descriptor.reason,
    )
    if result is None:
        return None
    decision, route_result = result
    decision.reason = normalize_message_text(
        f"{decision.reason};query_expansion:{descriptor.capability_query}"
    )
    if route_result is not None:
        route_result = RouteResolveResult(
            decision=route_result.decision,
            stage=route_result.stage,
            report=route_result.report,
            command_id=route_result.command_id,
            slots=route_result.slots,
            missing=route_result.missing,
            selected_rank=route_result.selected_rank,
            selected_score=route_result.selected_score,
            selected_reason=normalize_message_text(
                f"{route_result.selected_reason};"
                f"query_expansion:{descriptor.capability_query}"
            ),
        )
    return decision, route_result


async def resolve_llm_router(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None = None,
    budget_controller: TurnBudgetController | None = None,
    has_reply: bool = False,
    command_tools: list[Any] | None = None,
) -> tuple[LLMRouterDecision, RouteResolveResult | None, RouteAttemptReport]:
    normalized_message = normalize_message_text(message_text)
    report = RouteAttemptReport(helper_mode=is_usage_question(normalized_message))
    if not normalized_message:
        report.finalize(reason="router_empty")
        return LLMRouterDecision(action="chat", reason="empty"), None, report

    speech_act = classify_speech_act(
        normalized_message,
        **_message_context_flags(normalized_message, has_reply=has_reply),
    )
    if _should_force_chat_before_llm(
        normalized_message,
        speech_act,
    ) and not _has_exact_tool_head(normalized_message, knowledge_base, command_tools):
        decision = LLMRouterDecision(
            action="chat",
            confidence=0.82,
            reason=f"speech_act:{speech_act}",
        )
        report.finalize(reason=decision.reason or "speech_act_chat", stage="speech_act")
        return decision, None, report

    candidate_route = await _resolve_candidate_selection_route(
        normalized_message,
        knowledge_base,
        session_key=session_key,
        budget_controller=budget_controller,
        has_reply=has_reply,
        report=report,
        command_tools=command_tools,
        use_reranker=True,
    )
    if candidate_route is not None:
        decision, route_result = candidate_route
        report.finalize(
            reason=decision.reason or f"command_selector_{decision.action}",
            stage=(
                route_result.stage if route_result is not None else "command_selector"
            ),
            plugin_name=route_result.decision.plugin_name if route_result else None,
            plugin_module=route_result.decision.plugin_module if route_result else None,
            command=route_result.decision.command if route_result else None,
        )
        return decision, route_result, report

    expanded_route = await _resolve_query_expansion_route(
        normalized_message,
        knowledge_base,
        session_key=session_key,
        budget_controller=budget_controller,
        has_reply=has_reply,
        report=report,
        command_tools=command_tools,
    )
    if expanded_route is not None:
        decision, route_result = expanded_route
        report.finalize(
            reason=decision.reason or f"query_expansion_{decision.action}",
            stage=(
                route_result.stage if route_result is not None else "query_expansion"
            ),
            plugin_name=route_result.decision.plugin_name if route_result else None,
            plugin_module=route_result.decision.plugin_module if route_result else None,
            command=route_result.decision.command if route_result else None,
        )
        return decision, route_result, report

    report.finalize(reason="no_route_candidate", stage="none")
    return (
        LLMRouterDecision(action="chat", reason="no_route_candidate"),
        None,
        report,
    )


__all__ = [
    "LLMCommandSelection",
    "LLMRouterDecision",
    "RouteAttemptReport",
    "RouteResolveResult",
    "resolve_llm_router",
]
