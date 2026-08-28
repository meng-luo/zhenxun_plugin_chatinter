"""Capability retriever for ChatInter command tools.

Group plugin routing uses this as a recall-only static metadata searcher.  It
never exposes the full command list to the LLM and never reads execution
feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_graph import build_capability_graph_snapshot
from .command_index import (
    CommandCandidate,
    build_command_candidates,
)
from .models.pydantic_models import (
    CommandToolSnapshot,
    PluginKnowledgeBase,
)
from .plugin_reference import build_command_tool_snapshots
from .route_text import normalize_message_text

_DEFAULT_RETRIEVAL_LIMIT = 24
_DISPATCH_RETRIEVAL_LIMIT = 48


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
        self.knowledge_base = knowledge_base
        self.session_id = session_id
        self.tools = (
            [tool for tool in tools if isinstance(tool, CommandToolSnapshot)]
            if tools is not None
            else list(
                build_command_tool_snapshots(
                    build_capability_graph_snapshot(knowledge_base)
                )
            )
        )
        self._tools_by_id = {
            normalize_message_text(tool.command_id): tool
            for tool in self.tools
            if normalize_message_text(tool.command_id)
        }

    @property
    def total_commands(self) -> int:
        return len(self._tools_by_id)

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> CommandRetrievalResult:
        normalized_query = normalize_message_text(query)
        retrieval_limit = (
            _coerce_limit(limit) if limit is not None else _DISPATCH_RETRIEVAL_LIMIT
        )
        candidates = build_command_candidates(
            self.knowledge_base,
            normalized_query,
            limit=retrieval_limit,
            session_id=self.session_id,
            diversify=True,
            tools=self.tools,
            include_unscored=False,
            use_feedback=False,
            use_prefilter=False,
            static_metadata_only=True,
            router_context=context,
        )
        candidates = [
            candidate
            for candidate in candidates
            if normalize_message_text(candidate.schema.command_id) in self._tools_by_id
        ]
        return CommandRetrievalResult(
            query=normalized_query,
            candidates=tuple(candidates),
            total_commands=self.total_commands,
        )


def _coerce_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = _DEFAULT_RETRIEVAL_LIMIT
    return max(1, value)


__all__ = [
    "CommandRetrievalResult",
    "CommandToolRetriever",
]
