"""Stable overflow broker for provider tool-count limits."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from .candidate_exposure import CandidateExposureLedger
from .llm_compat import ToolDefinition, ToolResult
from .route_text import normalize_message_text
from .sparse_retrieval import normalize_retrieval_queries
from .task_frame import (
    TARGET_REF_SCHEMA_DESCRIPTION,
    TARGET_REFS_SCHEMA_DESCRIPTION,
)
from .token_compat import estimate_text_tokens

OVERFLOW_SKILL_TOOL_NAME = "ci_skill_overflow"
_NULL_IDENTITIES = {"", "null", "none", "nil", "undefined"}
_FULL_LISTING_RECALLS = {
    "skill_full_listing",
    "gscore_full_listing",
    "gscore_no_recall",
}


class SkillOverflowTool:
    """Discover and delegate Skills omitted by a provider's tool limit."""

    name = OVERFLOW_SKILL_TOOL_NAME
    chatinter_plugin_tool_kind = "skill_dispatch"
    chatinter_ignore_unknown_top_level_arguments = True
    chatinter_required_tool = True

    def __init__(
        self,
        delegates: Mapping[str, Any],
        *,
        result_token_budget: int = 4_096,
        result_char_budget: int = 12_000,
        exposure_ledger: CandidateExposureLedger | None = None,
    ) -> None:
        self._delegates = dict(sorted(delegates.items()))
        self._result_token_budget = max(int(result_token_budget), 1)
        self._result_char_budget = max(int(result_char_budget), 1)
        self._exposure_ledger = exposure_ledger

    def with_result_budget(
        self,
        *,
        token_budget: int,
        char_budget: int,
    ) -> SkillOverflowTool:
        return SkillOverflowTool(
            self._delegates,
            result_token_budget=token_budget,
            result_char_budget=char_budget,
            exposure_ledger=self._exposure_ledger,
        )

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Provider 工具数量不足时的插件能力入口。只有当前可见的具体 Skill "
                "都不匹配时才使用。未持有真实 command_id/capability_id 时先检索；"
                "只能执行本轮候选中返回的真实 ID，不能猜测或生成 ID。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_text": {
                        "type": "string",
                        "minLength": 1,
                        "description": "当前工具调用对应的用户原话或任务片段",
                    },
                    "retrieval_queries": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 256},
                        "maxItems": 6,
                        "description": "保持原意的可选检索改写，只用于召回候选",
                    },
                    "skill_tool": {
                        "type": ["string", "null"],
                        "description": "候选卡返回的 overflow_delegate；未知时填 null",
                    },
                    "command_id": {
                        "type": ["string", "null"],
                        "description": "候选卡返回的真实本体 command_id",
                    },
                    "capability_id": {
                        "type": ["string", "null"],
                        "description": "候选卡返回的真实 GScore capability_id",
                    },
                    "command_text": {
                        "type": ["string", "null"],
                        "description": "GScore 候选要求的完整实际命令文本",
                    },
                    "target_ref": {
                        "type": ["string", "null"],
                        "description": TARGET_REF_SCHEMA_DESCRIPTION,
                    },
                    "target_refs": {
                        "type": ["array", "null"],
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 2,
                        "maxItems": 4,
                        "uniqueItems": True,
                        "description": TARGET_REFS_SCHEMA_DESCRIPTION,
                    },
                    "slots": {
                        "type": "object",
                        "description": "本体命令参数槽位",
                        "additionalProperties": True,
                    },
                },
                "required": ["task_text"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        task_text = normalize_message_text(str(kwargs.get("task_text") or ""))
        if not task_text:
            return self._invalid("missing_task_text")
        command_id = _identity(kwargs.get("command_id"))
        capability_id = _identity(kwargs.get("capability_id"))
        selected_id = command_id or capability_id
        requested_delegate = normalize_message_text(
            str(kwargs.get("skill_tool") or "")
        )
        if selected_id:
            delegate_name = self._resolve_exposed_delegate(
                selected_id,
                requested_delegate=requested_delegate,
            )
            if not delegate_name:
                return self._invalid("candidate_identity_not_exposed")
            delegate = self._delegates.get(delegate_name)
            if delegate is None:
                return self._invalid("overflow_delegate_unavailable")
            return await delegate.execute(
                context,
                **_delegate_arguments(
                    delegate,
                    task_text=task_text,
                    command_id=command_id,
                    capability_id=capability_id,
                    kwargs=kwargs,
                ),
            )
        return await self._discover(
            context=context,
            task_text=task_text,
            requested_delegate=requested_delegate,
            kwargs=kwargs,
        )

    async def _discover(
        self,
        *,
        context: Any | None,
        task_text: str,
        requested_delegate: str,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        if requested_delegate:
            delegate = self._delegates.get(requested_delegate)
            if delegate is None:
                return self._invalid("overflow_delegate_unavailable")
            delegates = ((requested_delegate, delegate),)
        else:
            delegates = tuple(self._delegates.items())

        candidate_buckets: list[list[dict[str, Any]]] = []
        delegate_candidates: dict[str, list[dict[str, Any]]] = {}
        total_candidates = 0
        for delegate_name, delegate in delegates:
            result = await delegate.execute(
                context,
                **_delegate_arguments(
                    delegate,
                    task_text=task_text,
                    command_id="",
                    capability_id="",
                    kwargs=kwargs,
                    defer_candidate_exposure=True,
                ),
            )
            output = result.output if isinstance(result.output, dict) else {}
            recall = normalize_message_text(str(output.get("recall") or ""))
            cards = [
                dict(card)
                for card in output.get("candidates", ())
                if isinstance(card, dict)
            ]
            total_candidates += int(output.get("candidate_count", len(cards)) or 0)
            if recall in _FULL_LISTING_RECALLS and not requested_delegate:
                continue
            projected: list[dict[str, Any]] = []
            for card in cards:
                identity = _card_identity(card)
                if not identity:
                    continue
                projected_card = dict(card)
                projected_card["skill_tool"] = self.name
                projected_card["overflow_delegate"] = delegate_name
                projected.append(projected_card)
            if projected:
                candidate_buckets.append(projected)
                delegate_candidates[delegate_name] = projected

        candidates = self._fit_round_robin(candidate_buckets)
        exposed_by_delegate: dict[str, list[str]] = {}
        for card in candidates:
            identity = _card_identity(card)
            delegate_name = normalize_message_text(
                str(card.get("overflow_delegate") or "")
            )
            if identity and delegate_name:
                exposed_by_delegate.setdefault(delegate_name, []).append(identity)
        for delegate_name, identities in exposed_by_delegate.items():
            delegate = self._delegates.get(delegate_name)
            expose = getattr(delegate, "expose_candidates", None)
            if callable(expose):
                expose(
                    identities,
                    source="overflow_sparse_multi_skill",
                    pending=True,
                )
            record = getattr(delegate, "record_discovery", None)
            if callable(record):
                local_total = len(delegate_candidates.get(delegate_name, ()))
                record(
                    source="overflow_sparse_multi_skill",
                    query_count=len(
                        normalize_retrieval_queries(
                            task_text,
                            kwargs.get("retrieval_queries"),
                        )
                    ),
                    candidate_count=local_total,
                    displayed_count=len(identities),
                    omitted_count=max(local_total - len(identities), 0),
                )
        if self._exposure_ledger is not None:
            queries = normalize_retrieval_queries(
                task_text,
                kwargs.get("retrieval_queries"),
            )
            self._exposure_ledger.record_discovery_summary(
                skill=self.name,
                source=(
                    "overflow_sparse_multi_skill"
                    if candidates
                    else "overflow_no_recall"
                ),
                query_count=len(queries),
                candidate_count=total_candidates,
                displayed_count=len(candidates),
                omitted_count=max(total_candidates - len(candidates), 0),
            )
        return ToolResult(
            output={
                "status": "selection_required",
                "plugin_execution": False,
                "executed": False,
                "recall": (
                    "overflow_sparse_multi_skill"
                    if candidates
                    else "overflow_no_recall"
                ),
                "candidates": candidates,
                "candidate_count": total_candidates,
                "displayed_candidate_count": len(candidates),
                "omitted_candidate_count": max(total_candidates - len(candidates), 0),
                "truncated": len(candidates) < total_candidates,
                "response_policy": (
                    "continue_with_real_candidate"
                    if candidates
                    else "chat_without_clarification"
                ),
            },
            display_content=f"{self.name}: selection_required",
            is_retryable=True,
        )

    def _resolve_exposed_delegate(
        self,
        identity: str,
        *,
        requested_delegate: str,
    ) -> str:
        names = (
            (requested_delegate,)
            if requested_delegate
            else tuple(self._delegates)
        )
        matches = []
        for name in names:
            delegate = self._delegates.get(name)
            owns = getattr(delegate, "owns_candidate_identity", None)
            exposed = getattr(delegate, "is_candidate_exposed", None)
            if (
                delegate is not None
                and callable(owns)
                and callable(exposed)
                and owns(identity)
                and exposed(identity)
            ):
                matches.append(name)
        return matches[0] if len(matches) == 1 else ""

    def _fit_round_robin(
        self,
        buckets: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        max_rank = max((len(bucket) for bucket in buckets), default=0)
        for rank in range(max_rank):
            for bucket in buckets:
                if rank >= len(bucket):
                    continue
                trial = [*selected, bucket[rank]]
                payload = {
                    "status": "selection_required",
                    "candidates": trial,
                }
                serialized = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                )
                if (
                    len(serialized) <= self._result_char_budget
                    and estimate_text_tokens(serialized) <= self._result_token_budget
                ):
                    selected = trial
        return selected

    @staticmethod
    def _invalid(reason: str) -> ToolResult:
        return ToolResult(
            output={
                "status": "invalid_tool_arguments",
                "plugin_execution": False,
                "executed": False,
                "reason": reason,
                "response_policy": "chat_without_clarification",
            },
            display_content=f"{OVERFLOW_SKILL_TOOL_NAME}: invalid_tool_arguments",
            is_error=True,
            is_retryable=False,
        )


def _delegate_arguments(
    delegate: Any,
    *,
    task_text: str,
    command_id: str,
    capability_id: str,
    kwargs: dict[str, Any],
    defer_candidate_exposure: bool = False,
) -> dict[str, Any]:
    kind = str(getattr(delegate, "chatinter_plugin_tool_kind", "") or "")
    arguments: dict[str, Any] = {
        "task_text": task_text,
        "retrieval_queries": kwargs.get("retrieval_queries"),
        "_defer_candidate_exposure": defer_candidate_exposure,
    }
    if kind == "gscore":
        if capability_id:
            arguments["capability_id"] = capability_id
            arguments["command_text"] = kwargs.get("command_text")
        return arguments
    if command_id:
        arguments["command_id"] = command_id
        arguments["target_ref"] = kwargs.get("target_ref")
        arguments["target_refs"] = kwargs.get("target_refs")
        arguments["slots"] = kwargs.get("slots") or {}
    return arguments


def _identity(value: object) -> str:
    identity = normalize_message_text(str(value or ""))
    return "" if identity.casefold() in _NULL_IDENTITIES else identity


def _card_identity(card: Mapping[str, Any]) -> str:
    return _identity(card.get("command_id") or card.get("capability_id"))


__all__ = ["OVERFLOW_SKILL_TOOL_NAME", "SkillOverflowTool"]
