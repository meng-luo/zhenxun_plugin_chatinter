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
    target_missing_message: str = ""


def get_target_policy(
    *,
    plugin_module: str = "",
    plugin_name: str = "",
    command_id: str = "",
) -> TargetPolicy:
    _ = plugin_module, plugin_name, command_id
    return TargetPolicy()


__all__ = ["TargetPolicy", "get_target_policy"]
