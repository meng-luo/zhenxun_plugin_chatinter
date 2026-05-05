"""Adaptive candidate exposure policy for ChatInter tool reranking."""

from __future__ import annotations

from dataclasses import dataclass

from .command_index import CommandCandidate
from .route_text import normalize_message_text


@dataclass(frozen=True)
class CandidatePolicyDecision:
    candidates: list[CommandCandidate]
    limit: int
    reason: str
    top_score: float = 0.0
    margin: float = 0.0
    family_conflict: bool = False


def select_rerank_candidates(
    candidates: list[CommandCandidate],
    *,
    stage: str = "",
    base_limit: int = 24,
    max_limit: int = 48,
) -> CandidatePolicyDecision:
    if not candidates:
        return CandidatePolicyDecision(
            candidates=[],
            limit=0,
            reason="empty",
        )
    safe_base = max(int(base_limit or 0), 8)
    safe_max = max(int(max_limit or 0), safe_base)
    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    top_score = float(top.score)
    margin = top_score - float(second.score if second is not None else 0.0)
    top_family = normalize_message_text(top.family)
    family_conflict = bool(
        top_family
        and sum(
            1
            for item in candidates[: min(len(candidates), 12)]
            if item.family == top_family
        )
        >= 3
    )

    limit = safe_base
    reason_parts: list[str] = []
    if top.exact_protected and margin >= 80:
        limit = min(12, safe_base)
        reason_parts.append("exact_clear")
    if top_score < 180 or margin < 32:
        limit = max(limit, 36)
        reason_parts.append("weak_or_close")
    if family_conflict:
        limit = max(limit, 36)
        reason_parts.append("family_conflict")
    if stage == "query_expansion":
        limit = max(limit, 40)
        reason_parts.append(stage)
    limit = min(limit, safe_max)

    selected: list[CommandCandidate] = []
    seen_ids: set[str] = set()

    def keep(candidate: CommandCandidate) -> None:
        command_id = normalize_message_text(candidate.schema.command_id)
        if not command_id or command_id in seen_ids:
            return
        selected.append(candidate)
        seen_ids.add(command_id)

    for candidate in candidates:
        if candidate.exact_protected:
            keep(candidate)
    for candidate in candidates:
        if top_family and candidate.family == top_family:
            keep(candidate)
        if len(selected) >= limit:
            break
    for candidate in candidates:
        keep(candidate)
        if len(selected) >= limit:
            break

    return CandidatePolicyDecision(
        candidates=selected[:limit],
        limit=limit,
        reason=",".join(reason_parts) or "default",
        top_score=top_score,
        margin=margin,
        family_conflict=family_conflict,
    )


__all__ = ["CandidatePolicyDecision", "select_rerank_candidates"]
