"""Capability retriever for ChatInter command tools.

The retriever keeps every known command discoverable without exposing every
command as a native function tool in the first model request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_graph import build_capability_graph_snapshot
from .command_index import CommandCandidate, retrieve_command_candidates
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .plugin_reference import build_command_tool_snapshots
from .route_text import normalize_message_text

_DEFAULT_RETRIEVAL_LIMIT = 24
_MAX_RETRIEVAL_LIMIT = 64


@dataclass(frozen=True)
class CommandRetrievalResult:
    query: str
    candidates: tuple[CommandCandidate, ...]
    total_commands: int


class CommandToolRetriever:
    """Local retriever that turns a natural-language query into command candidates."""

    def __init__(
        self,
        knowledge_base: PluginKnowledgeBase,
        *,
        session_id: str | None,
        tools: list[Any] | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.session_id = session_id
        self.tools = _ensure_command_tools(knowledge_base, tools)

    @property
    def total_commands(self) -> int:
        return len(self.tools)

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> CommandRetrievalResult:
        normalized_query = normalize_message_text(query)
        retrieval_limit = _coerce_limit(limit)
        candidates = retrieve_command_candidates(
            self.knowledge_base,
            normalized_query,
            limit=retrieval_limit,
            session_id=self.session_id,
            diversify=True,
            tools=self.tools,
        )
        return CommandRetrievalResult(
            query=normalized_query,
            candidates=tuple(candidates),
            total_commands=self.total_commands,
        )


def _ensure_command_tools(
    knowledge_base: PluginKnowledgeBase,
    tools: list[Any] | None,
) -> list[CommandToolSnapshot]:
    if tools:
        return [tool for tool in tools if isinstance(tool, CommandToolSnapshot)]
    graph = build_capability_graph_snapshot(knowledge_base)
    return list(build_command_tool_snapshots(graph))


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
