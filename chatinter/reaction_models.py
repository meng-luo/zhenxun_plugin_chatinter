"""Local reaction-image data contracts for mixed chat."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ReactionDeliveryMode = Literal["append", "only"]


@dataclass(frozen=True, slots=True)
class ReactionSettings:
    enabled: bool
    root: Path
    import_root: Path
    semantic_search: bool
    auto_caption: bool
    auto_discovery: bool


@dataclass(frozen=True, slots=True)
class ReactionRecord:
    content_sha256: str
    relative_path: str
    category: str = ""
    category_description: str = ""
    caption: str = ""
    tags: tuple[str, ...] = ()
    visible_text: str = ""
    reply_intents: tuple[str, ...] = ()
    usage_scenarios: tuple[str, ...] = ()
    tones: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    target_relation: str = ""
    semantic_version: int = 0
    status: str = "pending"
    visual_fingerprint: str = ""
    provenance: str = ""
    source_version: str = ""
    size: int = 0
    mtime_ns: int = 0

    @property
    def reaction_id(self) -> str:
        return f"reaction:{self.content_sha256[:16]}"

    @property
    def semantic_text(self) -> str:
        values = (
            self.caption,
            self.category,
            self.category_description,
            *self.tags,
            self.visible_text,
            *self.reply_intents,
            *self.usage_scenarios,
            *self.tones,
            *self.actions,
            self.target_relation,
            Path(self.relative_path).stem.replace("_", " ").replace("-", " "),
        )
        return " / ".join(dict.fromkeys(value for value in values if value))[:2_000]

    @property
    def has_full_semantics(self) -> bool:
        return self.semantic_version >= 2 and bool(
            self.reply_intents
            or self.usage_scenarios
            or self.tones
            or self.actions
            or self.target_relation
        )

    def public_candidate(
        self,
        *,
        score: float,
        recently_used: bool = False,
        turns_ago: int | None = None,
    ) -> dict[str, Any]:
        return {
            "id": self.reaction_id,
            "caption": self.caption or self.category or Path(self.relative_path).stem,
            "tags": list(self.tags[:6]),
            "visible_text": self.visible_text[:160],
            "category": self.category,
            "reply_intents": list(self.reply_intents[:4]),
            "usage_scenarios": list(self.usage_scenarios[:2]),
            "tones": list(self.tones[:3]),
            "semantic_detail": "full" if self.has_full_semantics else "category",
            "score": round(max(min(float(score), 1.0), 0.0), 4),
            "recently_used": bool(recently_used),
            **(
                {"turns_ago": max(int(turns_ago), 1)}
                if recently_used and turns_ago is not None
                else {}
            ),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "relative_path": self.relative_path,
            "category": self.category,
            "category_description": self.category_description,
            "caption": self.caption,
            "tags": list(self.tags),
            "visible_text": self.visible_text,
            "reply_intents": list(self.reply_intents),
            "usage_scenarios": list(self.usage_scenarios),
            "tones": list(self.tones),
            "actions": list(self.actions),
            "target_relation": self.target_relation,
            "semantic_version": self.semantic_version,
            "status": self.status,
            "visual_fingerprint": self.visual_fingerprint,
            "provenance": self.provenance,
            "source_version": self.source_version,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True, slots=True)
class ReactionAction:
    reaction_id: str
    content_sha256: str
    path: Path
    root: Path
    mode: ReactionDeliveryMode
    reply_text: str
    fallback_text: str
    memory_text: str
    category: str = ""
    search_intent: str = ""


@dataclass(frozen=True, slots=True)
class RecentReactionFact:
    reaction_id: str
    category: str
    search_intent: str
    mode: ReactionDeliveryMode
    turns_ago: int


@dataclass(slots=True)
class ReactionTurnState:
    settings: ReactionSettings
    store: Any
    session_id: str
    category_catalog: tuple[tuple[str, str], ...] = ()
    records_snapshot: tuple[ReactionRecord, ...] = ()
    recent_reactions: tuple[RecentReactionFact, ...] = ()
    candidates: dict[str, ReactionRecord] = field(default_factory=dict)
    search_query: str = ""
    search_payload: dict[str, Any] | None = None
    action: ReactionAction | None = None
    terminal_reply_text: str | None = None
    terminal_memory_text: str = ""


def normalize_tags(value: Any, *, limit: int = 10) -> tuple[str, ...]:
    return normalize_semantic_list(value, limit=limit, item_limit=48)


def normalize_semantic_list(
    value: Any,
    *,
    limit: int,
    item_limit: int,
) -> tuple[str, ...]:
    values = value if isinstance(value, list | tuple | set | frozenset) else ()
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        tag = " ".join(item.split())[:item_limit]
        if tag and tag not in result:
            result.append(tag)
        if len(result) >= limit:
            break
    return tuple(result)


__all__ = [
    "ReactionAction",
    "ReactionDeliveryMode",
    "ReactionRecord",
    "ReactionSettings",
    "ReactionTurnState",
    "RecentReactionFact",
    "normalize_semantic_list",
    "normalize_tags",
]
