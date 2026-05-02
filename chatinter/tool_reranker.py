"""LLM reranker/classifier for command candidates.

This layer is intentionally scoped to an already filtered candidate pool. It
may choose one installed command, ask for missing inputs, answer usage, or
decline the pool. It never invents commands outside the provided candidates.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .command_index import CommandCandidate, dump_candidate_for_prompt
from .route_text import normalize_message_text
from .turn_runtime import TurnBudgetController


def _debug_log(message: str) -> None:
    try:
        from zhenxun.services import logger

        logger.debug(message)
    except Exception:
        return


def _get_config_value(key: str, default: Any = None) -> Any:
    try:
        from .config import get_config_value

        return get_config_value(key, default)
    except Exception:
        return default


def _get_model_name() -> str | None:
    try:
        from .config import get_model_name

        return get_model_name()
    except Exception:
        return None


class ToolRerankSlotValue(BaseModel):
    name: str = Field(description="槽位名称，必须来自被选候选 schema")
    value: str = Field(default="", description="槽位值，统一以字符串填写")


def _slots_to_items(value: Any) -> list[ToolRerankSlotValue]:
    if not value:
        return []
    if isinstance(value, dict):
        return [
            ToolRerankSlotValue(name=str(key), value=str(slot_value))
            for key, slot_value in value.items()
            if normalize_message_text(str(key or ""))
        ]

    items: list[ToolRerankSlotValue] = []
    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, ToolRerankSlotValue):
                name = item.name
                slot_value = item.value
            elif isinstance(item, dict):
                name = str(item.get("name", "") or "")
                slot_value = str(item.get("value", "") or "")
            else:
                continue
            if normalize_message_text(name):
                items.append(ToolRerankSlotValue(name=name, value=str(slot_value)))
    return items


def slots_to_dict(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _slots_to_items(value):
        name = normalize_message_text(item.name)
        if name:
            result[name] = str(item.value or "")
    return result


class ToolRerankDecision(BaseModel):
    action: Literal[
        "execute",
        "usage",
        "clarify",
        "chat",
        "no_available_tool",
    ] = Field(
        default="chat",
        description=(
            "execute=执行候选命令；usage=解释候选用法；clarify=缺少参数；"
            "chat=普通对话；no_available_tool=有工具意图但候选都不合适"
        ),
    )
    command_id: str | None = Field(
        default=None,
        description="execute/usage/clarify 时必须来自 candidates.command_id",
    )
    slots: list[ToolRerankSlotValue] = Field(
        default_factory=list,
        description="按候选 schema 填写的槽位列表，不要臆造槽位",
    )
    missing: list[str] = Field(
        default_factory=list,
        description="缺失的必填槽位或上下文，例如 text/image/reply/target",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="简短选择理由")
    no_available_tool_reason: str = Field(
        default="",
        description="action=no_available_tool 时说明为什么候选无法满足诉求",
    )

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots(cls, value: Any) -> list[ToolRerankSlotValue]:
        return _slots_to_items(value)


def _build_context_flags(message_text: str, *, has_reply: bool) -> dict[str, bool]:
    normalized = normalize_message_text(message_text)
    return {
        "has_reply": has_reply,
        "has_image": "[image" in normalized.lower(),
        "has_at": "[@" in normalized or "@" in normalized,
    }


def _candidate_payload(candidates: list[CommandCandidate]) -> list[dict[str, object]]:
    return [
        dump_candidate_for_prompt(candidate, index=index)
        for index, candidate in enumerate(candidates, 1)
    ]


def build_tool_rerank_prompt(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    has_reply: bool,
    stage: str,
    recovery_query: str = "",
    recovery_reason: str = "",
) -> str:
    payload = {
        "message": normalize_message_text(message_text),
        "stage": stage,
        "context": _build_context_flags(message_text, has_reply=has_reply),
        "recovery": {
            "query": normalize_message_text(recovery_query),
            "reason": normalize_message_text(recovery_reason),
        }
        if recovery_query or recovery_reason
        else None,
        "task": (
            "在 candidates 中重排并分类。只允许选择一个 command_id；"
            "如果用户只是聊天/讨论，返回 chat；"
            "如果用户确实想调用工具但候选没有合适工具，返回 no_available_tool；"
            "如果询问怎么用/用法，返回 usage；"
            "如果工具明确但缺少必填输入，返回 clarify 并列出 missing；"
            "如果可执行，返回 execute 并填写 slots。"
        ),
        "candidates": _candidate_payload(candidates),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_tool_rerank_decision(
    decision: ToolRerankDecision,
    *,
    candidates: list[CommandCandidate],
) -> None:
    candidate_by_id = {
        normalize_message_text(candidate.schema.command_id): candidate
        for candidate in candidates
    }
    if decision.action not in {"execute", "usage", "clarify"}:
        return

    command_id = normalize_message_text(decision.command_id or "")
    if command_id not in candidate_by_id:
        raise ValueError(f"unknown command_id: {command_id}")

    selected = candidate_by_id[command_id]
    allowed_slots = {
        normalize_message_text(slot.name)
        for slot in selected.schema.slots
        if normalize_message_text(slot.name)
    }
    unknown_slots = [
        key
        for key in slots_to_dict(decision.slots)
        if normalize_message_text(str(key or "")) not in allowed_slots
    ]
    if unknown_slots:
        raise ValueError(f"unknown slots: {unknown_slots}")


async def request_tool_rerank(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    has_reply: bool,
    stage: str,
    session_key: str | None = None,
    budget_controller: TurnBudgetController | None = None,
    recovery_query: str = "",
    recovery_reason: str = "",
) -> ToolRerankDecision | None:
    if not candidates:
        return None
    timeout = _get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout_value = float(timeout) if timeout else None
    except (TypeError, ValueError):
        timeout_value = None

    label = f"{stage}_rerank"
    if budget_controller is not None and not budget_controller.allow_classifier(label):
        return None

    prompt = build_tool_rerank_prompt(
        message_text=message_text,
        candidates=candidates,
        has_reply=has_reply,
        stage=stage,
        recovery_query=recovery_query,
        recovery_reason=recovery_reason,
    )
    instruction = (
        "你是 ChatInter 的命令候选重排器。"
        "只能在 candidates 里选择 command_id，不能发明工具、命令、插件或槽位。"
        "命令头已由系统保存，输出 command_id 和 slots 即可。"
        "把普通聊天和工具调用严格分开；把候选不足与普通聊天区分开。"
        "exact_protected=true 表示用户输入了真实命令头，除非明显闲聊，否则优先保留。"
    )

    async def _validate(decision: ToolRerankDecision) -> None:
        validate_tool_rerank_decision(decision, candidates=candidates)

    try:
        started = time.perf_counter()
        from .prompt_guard import guard_prompt_sections

        guarded = guard_prompt_sections(
            session_key=session_key or "global",
            stage=label,
            system_prompt=instruction,
            user_text=prompt,
            controller=budget_controller,
        )
        from zhenxun.services import generate_structured

        decision = await generate_structured(
            guarded.user_text,
            ToolRerankDecision,
            model=_get_model_name(),
            instruction=guarded.system_prompt,
            timeout=timeout_value,
            validation_callback=_validate,
        )
        if budget_controller is not None:
            budget_controller.record_classifier(label, time.perf_counter() - started)
        return decision
    except Exception as exc:
        _debug_log(f"ChatInter tool rerank 失败: {exc}")
        return None


__all__ = [
    "ToolRerankDecision",
    "ToolRerankSlotValue",
    "build_tool_rerank_prompt",
    "request_tool_rerank",
    "slots_to_dict",
    "validate_tool_rerank_decision",
]
