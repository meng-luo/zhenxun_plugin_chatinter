"""Model-visible command candidate projection for mixed chat."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from .command_index import CommandCandidate, _schema_from_tool_snapshot
from .models.pydantic_models import CommandToolSnapshot
from .route_text import normalize_message_text
from .token_compat import estimate_text_tokens
from .tool_cards import project_command_candidate_card


def render_command_candidate_context(
    candidates: list[CommandCandidate],
    *,
    token_budget: int | None = None,
    skill_tools_by_command_id: Mapping[str, str] | None = None,
) -> str:
    cards: list[dict[str, Any]] = []
    budget = max(int(token_budget), 1) if token_budget is not None else None
    used_tokens = estimate_text_tokens('{"commands":[]}')
    for candidate in candidates:
        snapshot = candidate.tool
        if snapshot is None:
            continue
        card = project_command_candidate_card(snapshot)
        skill_tool = (skill_tools_by_command_id or {}).get(
            normalize_message_text(snapshot.command_id),
            "",
        )
        if skill_tool:
            card["skill_tool"] = skill_tool
        card_tokens = estimate_text_tokens(
            json.dumps(
                card,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
        if budget is not None and used_tokens + card_tokens > budget:
            break
        cards.append(card)
        used_tokens += card_tokens
    if not cards:
        return ""
    return json.dumps(
        {"commands": cards},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _candidate_from_snapshot(snapshot: CommandToolSnapshot) -> CommandCandidate:
    return CommandCandidate(
        plugin_module=snapshot.plugin_module,
        plugin_name=snapshot.plugin_name,
        schema=_schema_from_tool_snapshot(snapshot),
        score=0.0,
        reason="unified_skill_candidate",
        family=snapshot.family,
        tool=snapshot,
    )


__all__ = ["render_command_candidate_context"]
