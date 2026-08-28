"""Preserve trigger AI metadata without enabling GScore's AI runtime."""

from __future__ import annotations

from typing import Literal, Callable
from functools import wraps

from gsuid_core.sv import SV

_METADATA_ATTRIBUTE = "__chatinter_bridge_to_ai__"
_CAPTURE_MODULE = __name__

TriggerType = Literal[
    "prefix",
    "suffix",
    "keyword",
    "fullmatch",
    "command",
    "file",
    "regex",
    "message",
    "meta",
]
TriggerHandler = Callable[..., object]
TriggerDecorator = Callable[[TriggerHandler], TriggerHandler]


def install_metadata_capture() -> None:
    current = SV._on
    original = current.__wrapped__ if current.__module__ == _CAPTURE_MODULE else current

    @wraps(original)
    def capture_on(
        self: SV,
        type: TriggerType,
        keyword: str | tuple[str, ...],
        block: bool = False,
        to_me: bool = False,
        prefix: bool = True,
        to_ai: str = "",
    ) -> TriggerDecorator:
        decorator = original(
            self,
            type,
            keyword,
            block,
            to_me,
            prefix,
            to_ai=to_ai,
        )
        description = to_ai.strip()

        def capture_handler(func: TriggerHandler) -> TriggerHandler:
            if description:
                setattr(func, _METADATA_ATTRIBUTE, description)
            return decorator(func)

        return capture_handler

    SV._on = capture_on


def trigger_ai_description(func: Callable[..., object]) -> str:
    current = func
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if _METADATA_ATTRIBUTE in current.__dict__:
            value = current.__dict__[_METADATA_ATTRIBUTE]
            return value.strip() if isinstance(value, str) else ""
        if "__wrapped__" not in current.__dict__:
            break
        wrapped = current.__dict__["__wrapped__"]
        if not callable(wrapped):
            break
        current = wrapped
    return ""


__all__ = ["install_metadata_capture", "trigger_ai_description"]
