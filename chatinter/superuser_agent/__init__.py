"""Superuser-only ChatInter agent tools."""

from typing import Any

from .approval_store import list_pending_approvals

__all__ = [
    "build_superuser_agent_tool_bundle",
    "list_pending_approvals",
    "registered_superuser_tool_names",
    "superuser_tool_cards",
]


def __getattr__(name: str) -> Any:
    if name in {
        "build_superuser_agent_tool_bundle",
        "registered_superuser_tool_names",
        "superuser_tool_cards",
    }:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(name)
