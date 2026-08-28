"""Superuser Agent context compaction orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import copy
from dataclasses import dataclass, replace
from typing import Any

from ..host_llm import HostModelCandidate
from ..llm_compat import LLMMessage
from .context import (
    CompactionTrigger,
    ContextCompressionResult,
    SemanticSummaryResponse,
    compact_messages,
    estimate_messages_tokens,
    parse_semantic_summary,
    semantic_summary_repair_messages,
    summary_request_fits_model,
    summary_request_output_tokens,
)
from .state import AgentRunState, append_artifact_refs

SummaryInvoke = Callable[
    [HostModelCandidate, list[LLMMessage], int, int], Awaitable[Any]
]
PersistCompaction = Callable[[str, dict[str, Any]], bool]
VisibleMessages = Callable[[list[LLMMessage]], list[LLMMessage]]


@dataclass(frozen=True)
class SuperuserCompactionExecution:
    result: ContextCompressionResult
    installed: bool
    persistence_failed: bool = False
    semantic_failures: tuple[tuple[str, dict[str, Any]], ...] = ()


def ordered_summary_candidates(
    candidates: tuple[HostModelCandidate, ...],
    *,
    preferred_name: str | None,
) -> tuple[HostModelCandidate, ...]:
    preferred = str(preferred_name or "").strip().casefold()
    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (
            0 if preferred and item[1].name.strip().casefold() == preferred else 1,
            item[0],
        ),
    )
    result: list[HostModelCandidate] = []
    seen: set[tuple[str, str]] = set()
    for _, candidate in ordered:
        key = (candidate.name.strip().casefold(), candidate.api_type.strip().casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return tuple(result)


async def summarize_with_candidates(
    messages: list[LLMMessage],
    *,
    candidates: tuple[HostModelCandidate, ...],
    preferred_name: str | None,
    configured_context_tokens: int,
    invoke: SummaryInvoke,
    propagate_errors: tuple[type[Exception], ...] = (),
) -> SemanticSummaryResponse:
    attempts: list[str] = []
    fitted_candidates = 0
    for candidate in ordered_summary_candidates(
        candidates,
        preferred_name=preferred_name,
    ):
        max_input_tokens = candidate.context_window(configured_context_tokens)
        summary_token_target = summary_request_output_tokens(
            messages,
            max_input_tokens=max_input_tokens,
        )
        candidate_messages = _with_summary_output_target(
            messages,
            summary_token_target=summary_token_target,
        )
        if not summary_request_fits_model(
            candidate_messages,
            max_input_tokens=max_input_tokens,
            summary_token_target=summary_token_target,
        ):
            attempts.append(f"{candidate.name}:request_too_large")
            continue
        fitted_candidates += 1
        response, native_status = await _invoke_summary_candidate(
            candidate,
            candidate_messages,
            max_input_tokens=max_input_tokens,
            summary_token_target=summary_token_target,
            invoke=invoke,
            attempts=attempts,
            stage="native",
            propagate_errors=propagate_errors,
        )
        if response is not None and _valid_summary_response(response):
            attempts.append(f"{candidate.name}:native:ok")
            return SemanticSummaryResponse(
                text=str(response.text or ""),
                candidate_attempts=tuple(attempts),
            )
        if native_status == "error":
            continue

        repair_messages = semantic_summary_repair_messages(candidate_messages)
        if not summary_request_fits_model(
            repair_messages,
            max_input_tokens=max_input_tokens,
            summary_token_target=summary_token_target,
        ):
            attempts.append(f"{candidate.name}:repair_too_large")
            continue
        repaired, _ = await _invoke_summary_candidate(
            candidate,
            repair_messages,
            max_input_tokens=max_input_tokens,
            summary_token_target=summary_token_target,
            invoke=invoke,
            attempts=attempts,
            stage="repair",
            propagate_errors=propagate_errors,
        )
        if repaired is not None and _valid_summary_response(repaired):
            attempts.append(f"{candidate.name}:repair:ok")
            return SemanticSummaryResponse(
                text=str(repaired.text or ""),
                candidate_attempts=tuple(attempts),
            )
        attempts.append(f"{candidate.name}:invalid_summary")

    return SemanticSummaryResponse(
        text="",
        candidate_attempts=tuple(attempts),
        failure_reason=(
            "summary_candidates_exhausted"
            if fitted_candidates
            else "summary_request_too_large"
        ),
    )


async def _invoke_summary_candidate(
    candidate: HostModelCandidate,
    messages: list[LLMMessage],
    *,
    max_input_tokens: int,
    summary_token_target: int,
    invoke: SummaryInvoke,
    attempts: list[str],
    stage: str,
    propagate_errors: tuple[type[Exception], ...],
) -> tuple[Any | None, str]:
    try:
        response = await invoke(
            candidate,
            messages,
            max_input_tokens,
            summary_token_target,
        )
    except Exception as exc:
        if propagate_errors and isinstance(exc, propagate_errors):
            raise
        attempts.append(f"{candidate.name}:{stage}:error:{type(exc).__name__}")
        return None, "error"
    if getattr(response, "tool_calls", None):
        attempts.append(f"{candidate.name}:{stage}:tool_calls")
        return None, "invalid"
    return response, "ok"


def _valid_summary_response(response: Any) -> bool:
    return parse_semantic_summary(str(getattr(response, "text", "") or "")) is not None


def _with_summary_output_target(
    messages: list[LLMMessage],
    *,
    summary_token_target: int,
) -> list[LLMMessage]:
    opening = "<summary_output_token_target>"
    closing = "</summary_output_token_target>"
    replacement = f"{opening}{max(int(summary_token_target or 0), 1)}{closing}"
    result: list[LLMMessage] = []
    for message in messages:
        content = message.content
        if (
            isinstance(content, str)
            and content.strip().startswith(opening)
            and content.strip().endswith(closing)
        ):
            result.append(message.model_copy(update={"content": replacement}))
        else:
            result.append(message)
    return result


async def compact_superuser_context(
    state: AgentRunState,
    *,
    max_input_tokens: int,
    schema_tokens: int,
    output_reserve_tokens: int,
    prompt_tokens_before: int,
    summarize: Callable[
        [list[LLMMessage]], Awaitable[str | SemanticSummaryResponse]
    ],
    persist: PersistCompaction,
    visible_messages: VisibleMessages,
    trigger: CompactionTrigger,
    hard_required: bool,
    prune_tool_results: bool = True,
    summary_max_input_tokens: int | None = None,
    propagate_errors: tuple[type[Exception], ...] = (),
) -> SuperuserCompactionExecution:
    failures: list[tuple[str, dict[str, Any]]] = []
    result = await compact_messages(
        state.messages,
        trace_id=state.trace_id,
        max_input_tokens=max_input_tokens,
        summarize=summarize,
        schema_tokens=schema_tokens,
        output_reserve_tokens=output_reserve_tokens,
        force=hard_required or trigger in {"provider_overflow", "manual"},
        on_failure=lambda fingerprint, metadata: failures.append(
            (fingerprint, dict(metadata))
        ),
        propagate_errors=propagate_errors,
        prune_tool_results=prune_tool_results,
        prompt_tokens_before=prompt_tokens_before,
        summary_max_input_tokens=summary_max_input_tokens,
        tighter=trigger == "provider_overflow",
        trigger=trigger,
        checkpoint_state={
            "plan_items": [dict(item) for item in state.plan_items],
            "artifact_refs": list(state.artifact_refs),
        },
    )
    if failures:
        state.append_metric(
            role="system",
            kind="semantic_compression_failed",
            metadata={
                "step": state.step,
                "error": str(failures[-1][1].get("error", "") or ""),
                "candidate_attempts": result.candidate_attempts,
                "failure_count": len(failures),
            },
        )

    if result.artifact_persistence_failed and hard_required:
        persist(
            "semantic_compression_failed",
            {
                "failure_reason": "artifact_persistence_failed",
                "candidate_attempts": result.candidate_attempts,
            },
        )
        return SuperuserCompactionExecution(
            result=result,
            installed=False,
            semantic_failures=tuple(failures),
        )

    if not result.changed:
        if failures:
            persist(
                "semantic_compression_failed",
                {
                    "failure_reason": result.failure_reason,
                    "candidate_attempts": result.candidate_attempts,
                },
            )
        return SuperuserCompactionExecution(
            result=result,
            installed=False,
            semantic_failures=tuple(failures),
        )

    visible = visible_messages(result.messages)
    actual_after_tokens = estimate_messages_tokens(visible)
    before_tokens = max(int(result.before_tokens or 0), int(prompt_tokens_before or 0))
    savings_tokens = before_tokens - actual_after_tokens
    result = replace(
        result,
        before_tokens=before_tokens,
        after_tokens=actual_after_tokens,
        summary_candidate_tokens=actual_after_tokens,
        summary_savings_tokens=savings_tokens,
        summary_savings_ratio=(
            savings_tokens / before_tokens if before_tokens > 0 else 0.0
        ),
        low_savings=savings_tokens <= 0,
    )
    previous_messages = state.messages
    previous_budget = copy.deepcopy(state.budget)
    previous_artifact_refs = list(state.artifact_refs)
    metric_count = len(state.metrics)
    state.messages = list(result.messages)
    state.budget.current_context_tokens = actual_after_tokens
    state.budget.last_usage_message_count = len(visible)
    state.budget.last_usage_schema_tokens = 0
    state.artifact_refs = []
    append_artifact_refs(
        state.artifact_refs,
        (*previous_artifact_refs, *result.artifact_ids),
    )
    metadata = {
        "step": state.step,
        "before_tokens": result.before_tokens,
        "after_tokens": result.after_tokens,
        "strategy": result.strategy,
        "trigger": result.trigger,
        "summarized_messages": result.summarized_messages,
        "pruned_tool_results": result.pruned_tool_results,
        "protected_messages": result.protected_messages,
        "retained_rounds": result.retained_rounds,
        "dropped_rounds": result.dropped_rounds,
        "summary_savings_tokens": result.summary_savings_tokens,
        "summary_savings_ratio": result.summary_savings_ratio,
        "artifact_ids": result.artifact_ids,
        "candidate_attempts": result.candidate_attempts,
    }
    state.append_metric(
        role="system",
        kind="semantic_context_compression",
        content=result.summary,
        metadata=metadata,
    )
    if persist("semantic_context_compressed", metadata) is not False:
        return SuperuserCompactionExecution(
            result=result,
            installed=True,
            semantic_failures=tuple(failures),
        )

    state.messages = previous_messages
    state.budget = previous_budget
    state.artifact_refs = previous_artifact_refs
    del state.metrics[metric_count:]
    return SuperuserCompactionExecution(
        result=replace(result, failure_reason="compression_persistence_failed"),
        installed=False,
        persistence_failed=True,
        semantic_failures=tuple(failures),
    )


__all__ = [
    "SuperuserCompactionExecution",
    "compact_superuser_context",
    "ordered_summary_candidates",
    "summarize_with_candidates",
]
