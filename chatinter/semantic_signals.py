"""Small semantic helpers shared by routing tests and policy code."""

from __future__ import annotations

from .route_text import normalize_message_text

_EXPLANATION_TERMS = (
    "解释",
    "展开",
    "意思",
    "含义",
    "说清楚",
    "好好说话",
    "说人话",
    "翻译一下",
)
_SHORT_TEXT_MARKERS = (
    "缩写",
    "简称",
    "黑话",
    "梗",
    "术语",
    "这个词",
    "这个",
)


def is_concrete_short_text_explanation_request(message_text: str) -> bool:
    text = normalize_message_text(message_text)
    if not text:
        return False
    return any(term in text for term in _EXPLANATION_TERMS) and any(
        marker in text for marker in _SHORT_TEXT_MARKERS
    )


__all__ = ["is_concrete_short_text_explanation_request"]
