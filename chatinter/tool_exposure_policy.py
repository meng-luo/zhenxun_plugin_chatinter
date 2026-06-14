"""Compatibility facade for command-tool exposure policy."""

from __future__ import annotations

from typing import Any


def apply_tool_exposure_policy(**kwargs: Any) -> None:
    from .main_request import _apply_tool_exposure_policy

    _apply_tool_exposure_policy(**kwargs)


__all__ = ["apply_tool_exposure_policy"]
