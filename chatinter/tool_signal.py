"""Small text signal helpers for tool-routing code."""

from __future__ import annotations

from .route_text import normalize_message_text


def has_tool_signal(text: str) -> bool:
    normalized = normalize_message_text(text)
    return any(word in normalized for word in ("帮我", "查询", "发送", "生成", "处理"))


__all__ = ["has_tool_signal"]
