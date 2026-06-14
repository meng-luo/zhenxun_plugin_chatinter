"""Helpers for mapping native command ids to tool names."""

from __future__ import annotations

from .route_text import normalize_message_text


def command_id_to_tool_name(command_id: str) -> str:
    normalized = normalize_message_text(command_id)
    return "ci_cmd_" + normalized.replace(".", "_").replace("-", "_")


__all__ = ["command_id_to_tool_name"]
