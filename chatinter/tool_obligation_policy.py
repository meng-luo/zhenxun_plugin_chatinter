"""Compatibility facade for ChatInter tool-obligation decisions.

The production decision chain lives in ``main_request``.  This module keeps the
small test-facing API lightweight so importing it does not pull runtime services
until the full resolver is actually called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolGateSpendContext:
    has_high_reliability_candidate: bool = False


def _tool_gate_spend_context(
    *,
    candidates: list[Any],
    budget_controller: Any | None = None,
) -> ToolGateSpendContext:
    del budget_controller
    return ToolGateSpendContext(
        has_high_reliability_candidate=any(
            _candidate_is_high_reliability(candidate) for candidate in candidates
        )
    )


async def resolve_tool_obligation(**kwargs: Any) -> Any:
    from .plugin_command_support import _resolve_tool_obligation

    return await _resolve_tool_obligation(**kwargs)


def _candidate_is_high_reliability(candidate: Any) -> bool:
    snapshot = getattr(candidate, "tool", None)
    features = getattr(candidate, "features", None)
    reliability = float(getattr(snapshot, "reliability", 0.0) or 0.0)
    schema_quality = float(getattr(snapshot, "schema_quality", 0.0) or 0.0)
    score = float(getattr(candidate, "score", 0.0) or 0.0)
    feature_score = sum(
        float(getattr(features, name, 0.0) or 0.0)
        for name in (
            "exact_score",
            "lexical_score",
            "semantic_score",
            "context_score",
            "schema_score",
        )
    )
    return (
        bool(getattr(snapshot, "requires_real_tool", False))
        and reliability >= 0.72
        and schema_quality >= 0.65
        and max(score, feature_score) >= 80.0
    )


__all__ = [
    "ToolGateSpendContext",
    "_tool_gate_spend_context",
    "resolve_tool_obligation",
]
