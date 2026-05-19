"""Capability retriever for ChatInter command tools.

The retriever keeps every known command discoverable without exposing every
command as a native function tool in the first model request.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .capability_registry import CapabilityRegistry
from .command_index import CommandCandidate, build_command_candidates
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .route_text import normalize_message_text
from .soft_tool_policy import filter_soft_candidates

_DEFAULT_RETRIEVAL_LIMIT = 24
_MAX_RETRIEVAL_LIMIT = 64
_DEFAULT_INITIAL_TOOL_CAP = 120
_DEFAULT_LARGE_PLUGIN_COMMAND_CAP = 8


@dataclass(frozen=True)
class CommandRetrievalResult:
    query: str
    candidates: tuple[CommandCandidate, ...]
    total_commands: int
    capability_payloads: tuple[dict[str, Any], ...] = ()


class CommandToolRetriever:
    """Local retriever that turns a natural-language query into command candidates."""

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
    ) -> CommandRetrievalResult:
        normalized_query = normalize_message_text(query)
        retrieval_limit = _coerce_limit(limit)
        candidates = self.registry.recall(
            normalized_query,
            limit=retrieval_limit,
            diversify=True,
        )
        return CommandRetrievalResult(
            query=normalized_query,
            candidates=tuple(candidates),
            total_commands=self.total_commands,
            capability_payloads=tuple(
                self.registry.candidate_payload(candidate, index=index)
                for index, candidate in enumerate(candidates, 1)
            ),
        )

    def initial_command_exposure(
        self,
        query: str,
        *,
        max_total: int = _DEFAULT_INITIAL_TOOL_CAP,
        large_plugin_command_cap: int = _DEFAULT_LARGE_PLUGIN_COMMAND_CAP,
    ) -> CommandRetrievalResult:
        """Build first-turn executable command tools.

        Policy:
        - plugins with <= ``large_plugin_command_cap`` commands expose all commands;
        - larger plugins expose only the top related commands for this query.

        This restores broad command-level choice without exceeding provider tool
        limits when a plugin contributes many command schemas.
        """

        normalized_query = normalize_message_text(query)
        grouped = _group_tools_by_plugin(self.registry.tools)
        selected: list[CommandCandidate] = []
        for plugin_tools in grouped.values():
            if len(plugin_tools) <= large_plugin_command_cap:
                plugin_candidates = build_command_candidates(
                    self.registry.knowledge_base,
                    normalized_query,
                    limit=None,
                    session_id=self.registry.session_id,
                    diversify=False,
                    tools=plugin_tools,
                    include_unscored=True,
                )
                selected.extend(
                    filter_soft_candidates(normalized_query, plugin_candidates)
                )
                continue
            plugin_candidates = build_command_candidates(
                self.registry.knowledge_base,
                normalized_query,
                limit=large_plugin_command_cap,
                session_id=self.registry.session_id,
                diversify=False,
                tools=plugin_tools,
                include_unscored=False,
            )
            selected.extend(
                filter_soft_candidates(normalized_query, plugin_candidates)
            )

        selected = _dedupe_and_trim_candidates(selected, max_total=max_total)
        return CommandRetrievalResult(
            query=normalized_query,
            candidates=tuple(selected),
            total_commands=self.total_commands,
            capability_payloads=tuple(
                self.registry.candidate_payload(candidate, index=index)
                for index, candidate in enumerate(selected, 1)
            ),
        )


def _coerce_limit(limit: int | None) -> int:
    if limit is None:
        return _DEFAULT_RETRIEVAL_LIMIT
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = _DEFAULT_RETRIEVAL_LIMIT
    return max(1, min(value, _MAX_RETRIEVAL_LIMIT))


def _group_tools_by_plugin(
    tools: list[CommandToolSnapshot],
) -> dict[str, list[CommandToolSnapshot]]:
    grouped: dict[str, list[CommandToolSnapshot]] = defaultdict(list)
    for tool in tools:
        key = normalize_message_text(tool.plugin_module or tool.plugin_name)
        if not key:
            key = "unknown"
        grouped[key].append(tool)
    return dict(grouped)


def _dedupe_and_trim_candidates(
    candidates: list[CommandCandidate],
    *,
    max_total: int,
) -> list[CommandCandidate]:
    by_id: dict[str, CommandCandidate] = {}
    for candidate in candidates:
        command_id = normalize_message_text(candidate.schema.command_id)
        if not command_id:
            continue
        previous = by_id.get(command_id)
        if previous is None or _candidate_rank_key(candidate) > _candidate_rank_key(
            previous
        ):
            by_id[command_id] = candidate

    ordered = sorted(
        by_id.values(),
        key=_candidate_rank_key,
        reverse=True,
    )
    cap = max(1, min(int(max_total or _DEFAULT_INITIAL_TOOL_CAP), 120))
    return ordered[:cap]


def _candidate_rank_key(
    candidate: CommandCandidate,
) -> tuple[int, float, float, int, str]:
    exact = 1 if candidate.exact_protected else 0
    score = float(candidate.score or 0.0)
    context = float(getattr(candidate.features, "context_score", 0.0) or 0.0)
    non_empty = 1 if score > 0 else 0
    return (
        exact,
        score,
        context,
        non_empty,
        normalize_message_text(candidate.schema.command_id),
    )


__all__ = [
    "CommandRetrievalResult",
    "CommandToolRetriever",
]
