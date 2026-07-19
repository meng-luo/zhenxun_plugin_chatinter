"""Capability retriever for ChatInter command tools.

Group plugin routing uses this as a recall-only static metadata searcher.  It
never exposes the full command list to the LLM and never reads execution
feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_registry import CapabilityRegistry
from .command_index import CommandCandidate, build_command_candidates
from .models.pydantic_models import PluginKnowledgeBase
from .route_text import normalize_message_text

_DEFAULT_RETRIEVAL_LIMIT = 24
_MAX_RETRIEVAL_LIMIT = 64


@dataclass(frozen=True)
class CommandRetrievalResult:
    query: str
    candidates: tuple[CommandCandidate, ...]
    total_commands: int


class CommandToolRetriever:
    """Local retriever that returns a narrow static-metadata candidate set."""

    def __init__(
        self,
        knowledge_base: PluginKnowledgeBase,
        *,
        session_id: str | None,
        tools: list[Any] | None = None,
    ) -> None:
        self.registry = CapabilityRegistry.from_knowledge_base(
            knowledge_base,
            session_id=session_id,
            tools=tools,
        )

    @property
    def total_commands(self) -> int:
        return self.registry.total_commands

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> CommandRetrievalResult:
        normalized_query = normalize_message_text(query)
        retrieval_limit = _coerce_limit(limit)
        knowledge_base = self.registry.knowledge_base or PluginKnowledgeBase(
            user_role=""
        )
        candidates = build_command_candidates(
            knowledge_base,
            normalized_query,
            limit=retrieval_limit,
            session_id=self.registry.session_id,
            diversify=True,
            tools=self.registry.tools,
            include_unscored=False,
            use_feedback=False,
            use_prefilter=False,
            static_metadata_only=True,
            router_context=context,
        )
        candidates = [
            candidate
            for candidate in candidates
            if self.registry.record_for(candidate.schema.command_id) is not None
        ]
        return CommandRetrievalResult(
            query=normalized_query,
            candidates=tuple(candidates),
            total_commands=self.total_commands,
        )


def _coerce_limit(limit: int | None) -> int:
    if limit is None:
        return _DEFAULT_RETRIEVAL_LIMIT
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = _DEFAULT_RETRIEVAL_LIMIT
    return max(1, min(value, _MAX_RETRIEVAL_LIMIT))


__all__ = [
    "CommandRetrievalResult",
    "CommandToolRetriever",
]
