"""Generic target policy for command execution.

This is intentionally capability/schema driven.  It replaces the old
plugin-adapter compatibility layer, so target handling no longer depends on
handwritten plugin-specific adapters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetPolicy:
    family: str = "general"
    context_hints: tuple[str, ...] = ()
    media_related: bool = False
    allow_at_as_target: bool = False
    allow_image_as_target: bool = False
    allow_reply_image_as_target: bool = False
    require_target_for_third_person: bool = False


__all__ = ["TargetPolicy"]
