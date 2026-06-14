"""Feedback boundary facade kept free of runtime-service imports."""

from __future__ import annotations

from typing import Any


def record_feedback_event(*args: Any, **kwargs: Any) -> None:
    from .feedback import FeedbackStore

    handler = getattr(FeedbackStore, "record", None)
    if callable(handler):
        handler(*args, **kwargs)


__all__ = ["record_feedback_event"]
