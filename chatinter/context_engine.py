"""Pluggable context engine for ChatInter AgentRuntime.

The engine owns turn-local context reduction.  Individual policies stay small
and replaceable, while the old compression helpers remain as proven building
blocks behind the default policy chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from zhenxun.services.llm import LLMMessage

from .context_compression import (
    _MAX_PROMPT_TOKENS,
    _TARGET_PROMPT_TOKENS,
    ContextCompressionResult,
    _compress_completed_tool_pairs,
    _count_completed_tool_pairs,
    _dedupe_repeated_messages,
    _has_orphan_tool_messages,
    _has_recent_compression_marker,
    _messages_fingerprint,
    _normalize_protected_terms,
    _sanitize_tool_protocol,
    _summarize_middle_messages,
    estimate_messages_tokens,
)
from .route_text import normalize_message_text

_PROTECTED_REF_KEYS = {
    "run_id",
    "trace_id",
    "approval_id",
    "task_id",
    "event_id",
    "artifact_id",
    "operation_id",
    "eval_id",
    "loop_id",
    "checkpoint_id",
    "command_id",
    "tool_call_id",
    "task_text",
    "goal",
    "graph_id",
    "text",
}
_PROTECTED_REF_LIST_KEYS = {
    "waiting_approval_ids",
    "background_task_ids",
    "observation_event_ids",
    "artifact_refs",
    "incomplete_task_goals",
}
_PROTECTED_TERM_LIMIT = 180


@dataclass
class ContextEngineState:
    messages: list[LLMMessage]
    trace_id: str
    max_prompt_tokens: int = _MAX_PROMPT_TOKENS
    target_prompt_tokens: int = _TARGET_PROMPT_TOKENS
    protected_terms: tuple[str, ...] = ()
    context_refs: dict[str, Any] = field(default_factory=dict)
    before_tokens: int = 0
    fingerprint: str = ""
    compressed_tool_pairs: int = 0
    summarized_messages: int = 0
    deduped_messages: int = 0
    pruned_tool_results: int = 0
    protected_messages: int = 0
    summary: str = ""
    skipped_reason: str = ""
    stop_processing: bool = False
    policy_trace: list[str] = field(default_factory=list)

    def current_tokens(self) -> int:
        return estimate_messages_tokens(self.messages)

    def mark(self, policy_name: str, detail: str = "") -> None:
        entry = normalize_message_text(policy_name)
        if detail:
            entry = f"{entry}:{normalize_message_text(detail)}"
        if entry:
            self.policy_trace.append(entry[:180])


class ContextPolicy(Protocol):
    """Replaceable context reduction policy."""

    name: str

    def apply(self, state: ContextEngineState) -> ContextEngineState:
        """Return the updated context state."""


class ActiveTaskProtectionPolicy:
    """Normalize and enrich terms that should survive compression."""

    name = "active_task_protection"

    def apply(self, state: ContextEngineState) -> ContextEngineState:
        extra_terms = _protected_terms_from_refs(state.context_refs)
        if extra_terms:
            state.protected_terms = _normalize_protected_terms(
                [*state.protected_terms, *extra_terms]
            )
        state.mark(self.name, f"terms={len(state.protected_terms)}")
        return state


class AntiThrashingPolicy:
    """Avoid repeatedly compressing an already compact recent context."""

    name = "anti_thrashing"

    def apply(self, state: ContextEngineState) -> ContextEngineState:
        raw_pair_count = _count_completed_tool_pairs(state.messages)
        has_orphan_tool = _has_orphan_tool_messages(state.messages)
        if (
            state.before_tokens <= state.max_prompt_tokens
            and raw_pair_count == 0
            and not has_orphan_tool
            and _has_recent_compression_marker(state.messages)
        ):
            state.skipped_reason = "recent_compression_below_pressure"
            state.stop_processing = True
            state.mark(self.name, "skip_recent_below_pressure")
            return state
        state.mark(
            self.name,
            f"continue tokens={state.before_tokens} pairs={raw_pair_count}",
        )
        return state


class ToolResultPruningPolicy:
    """Compact completed assistant tool-call + tool-result pairs."""

    name = "tool_result_pruning"

    def apply(self, state: ContextEngineState) -> ContextEngineState:
        messages, pair_count, pruned_tool_results, protected_messages = (
            _compress_completed_tool_pairs(
                state.messages,
                trace_id=state.trace_id,
                protected_terms=state.protected_terms,
            )
        )
        state.messages = messages
        state.compressed_tool_pairs += pair_count
        state.pruned_tool_results += pruned_tool_results
        state.protected_messages += protected_messages
        state.mark(
            self.name,
            f"pairs={pair_count} pruned={pruned_tool_results}",
        )
        return state


class ToolProtocolSanitizerPolicy:
    """Turn orphan or incomplete tool protocol messages into summaries."""

    name = "tool_protocol_sanitizer"

    def apply(self, state: ContextEngineState) -> ContextEngineState:
        messages, orphan_count = _sanitize_tool_protocol(
            state.messages,
            trace_id=state.trace_id,
            protected_terms=state.protected_terms,
        )
        state.messages = messages
        state.pruned_tool_results += orphan_count
        state.mark(self.name, f"pruned={orphan_count}")
        return state


class DeduplicateMessagesPolicy:
    """Collapse repeated historical tool summaries/results."""

    name = "deduplicate_messages"

    def apply(self, state: ContextEngineState) -> ContextEngineState:
        messages, deduped_count = _dedupe_repeated_messages(
            state.messages,
            protected_terms=state.protected_terms,
        )
        state.messages = messages
        state.deduped_messages += deduped_count
        state.mark(self.name, f"deduped={deduped_count}")
        return state


class LongTermSummaryPolicy:
    """Summarize middle history under token pressure and keep task refs."""

    name = "long_term_summary"

    def apply(self, state: ContextEngineState) -> ContextEngineState:
        if state.current_tokens() <= state.max_prompt_tokens:
            state.mark(self.name, "skip_below_pressure")
            return state
        messages, summarized_count, summary, summarized_protected = (
            _summarize_middle_messages(
                state.messages,
                trace_id=state.trace_id,
                target_prompt_tokens=state.target_prompt_tokens,
                protected_terms=state.protected_terms,
                compression_fingerprint=state.fingerprint,
            )
        )
        state.messages = messages
        state.summarized_messages += summarized_count
        state.protected_messages += summarized_protected
        if summary:
            state.summary = summary
        state.mark(
            self.name,
            f"summarized={summarized_count} protected={summarized_protected}",
        )
        return state


class ArtifactReferencePolicy:
    """Final accounting policy for artifact-backed compression output.

    Artifact creation is performed by lower-level summarizers because they know
    exactly which raw payload is being omitted.  This policy keeps that behavior
    visible in the engine trace and is the replacement point for future stores.
    """

    name = "artifact_reference"

    def apply(self, state: ContextEngineState) -> ContextEngineState:
        state.mark(self.name, "store=default_artifact_store")
        return state


class ContextEngine:
    """Composable context compression engine."""

    def __init__(self, policies: Iterable[ContextPolicy] | None = None) -> None:
        self.policies: list[ContextPolicy] = list(policies or default_context_policies())

    def register_policy(
        self,
        policy: ContextPolicy,
        *,
        index: int | None = None,
    ) -> None:
        if index is None:
            self.policies.append(policy)
            return
        self.policies.insert(max(index, 0), policy)

    def compress(
        self,
        messages: list[LLMMessage],
        *,
        trace_id: str,
        max_prompt_tokens: int = _MAX_PROMPT_TOKENS,
        target_prompt_tokens: int = _TARGET_PROMPT_TOKENS,
        protected_terms: Iterable[str] | None = None,
        context_refs: dict[str, Any] | None = None,
    ) -> ContextCompressionResult:
        state = ContextEngineState(
            messages=messages,
            trace_id=trace_id,
            max_prompt_tokens=max_prompt_tokens,
            target_prompt_tokens=target_prompt_tokens,
            protected_terms=_normalize_protected_terms(protected_terms or ()),
            context_refs=dict(context_refs or {}),
            before_tokens=estimate_messages_tokens(messages),
            fingerprint=_messages_fingerprint(messages),
        )
        for policy in self.policies:
            state = policy.apply(state)
            if state.stop_processing:
                break
        after_tokens = estimate_messages_tokens(state.messages)
        changed = (
            state.compressed_tool_pairs > 0
            or state.summarized_messages > 0
            or state.deduped_messages > 0
            or state.pruned_tool_results > 0
            or after_tokens < state.before_tokens
        )
        return ContextCompressionResult(
            messages=state.messages,
            changed=changed,
            before_tokens=state.before_tokens,
            after_tokens=after_tokens,
            compressed_tool_pairs=state.compressed_tool_pairs,
            summarized_messages=state.summarized_messages,
            deduped_messages=state.deduped_messages,
            pruned_tool_results=state.pruned_tool_results,
            protected_messages=state.protected_messages,
            compression_fingerprint=state.fingerprint,
            skipped_reason=state.skipped_reason,
            summary=state.summary or _build_summary(state, after_tokens),
            policy_trace=tuple(state.policy_trace),
        )


def default_context_policies() -> list[ContextPolicy]:
    return [
        ActiveTaskProtectionPolicy(),
        AntiThrashingPolicy(),
        ToolResultPruningPolicy(),
        ToolProtocolSanitizerPolicy(),
        DeduplicateMessagesPolicy(),
        LongTermSummaryPolicy(),
        ArtifactReferencePolicy(),
    ]


_DEFAULT_CONTEXT_ENGINE: ContextEngine | None = None


def get_context_engine() -> ContextEngine:
    global _DEFAULT_CONTEXT_ENGINE
    if _DEFAULT_CONTEXT_ENGINE is None:
        _DEFAULT_CONTEXT_ENGINE = ContextEngine()
    return _DEFAULT_CONTEXT_ENGINE


def register_context_policy(
    policy: ContextPolicy,
    *,
    index: int | None = None,
) -> None:
    get_context_engine().register_policy(policy, index=index)


def _build_summary(state: ContextEngineState, after_tokens: int) -> str:
    return (
        f"compressed_tool_pairs={state.compressed_tool_pairs}; "
        f"pruned_tool_results={state.pruned_tool_results}; "
        f"deduped_messages={state.deduped_messages}; "
        f"protected_messages={state.protected_messages}; "
        f"tokens {state.before_tokens}->{after_tokens}; "
        f"policies={','.join(state.policy_trace)}"
    )


def _protected_terms_from_refs(context_refs: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add_scalar(value: Any) -> None:
        text = normalize_message_text(str(value or ""))
        if not text:
            return
        if len(text) > _PROTECTED_TERM_LIMIT:
            text = text[:_PROTECTED_TERM_LIMIT].rstrip()
        if text and text not in terms:
            terms.append(text)

    def add(value: Any, *, key: str = "") -> None:
        if value is None:
            return
        if isinstance(value, str | int | float):
            if key in _PROTECTED_REF_KEYS or key in _PROTECTED_REF_LIST_KEYS:
                add_scalar(value)
            return
        if isinstance(value, dict):
            for item_key, item in value.items():
                add(item, key=str(item_key))
            return
        if isinstance(value, list | tuple | set):
            for item in value:
                if isinstance(item, dict):
                    add(item, key=key)
                    continue
                if key in _PROTECTED_REF_LIST_KEYS:
                    add_scalar(item)

    for key, value in context_refs.items():
        add(value, key=str(key))
    return terms[:80]


__all__ = [
    "ActiveTaskProtectionPolicy",
    "AntiThrashingPolicy",
    "ArtifactReferencePolicy",
    "ContextEngine",
    "ContextEngineState",
    "ContextPolicy",
    "DeduplicateMessagesPolicy",
    "LongTermSummaryPolicy",
    "ToolProtocolSanitizerPolicy",
    "ToolResultPruningPolicy",
    "default_context_policies",
    "get_context_engine",
    "register_context_policy",
]
