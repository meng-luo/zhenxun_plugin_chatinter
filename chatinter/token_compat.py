"""ChatInter token estimation compatibility boundary."""

from __future__ import annotations

import math
import re


def estimate_text_tokens(text: str) -> int:
    value = str(text or "")
    try:
        from zhenxun.services.ai.core.engine.token_counter import (
            TokenCounter as host_counter,
        )
    except Exception:
        host_counter = None
    if host_counter is not None:
        for name in ("count_text", "_count_text"):
            counter = getattr(host_counter, name, None)
            if not callable(counter):
                continue
            try:
                return max(int(counter(value)), 0)
            except (TypeError, ValueError):
                continue
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", value))
    return math.ceil(cjk_chars * 1.2 + (len(value) - cjk_chars) * 0.3)


__all__ = ["estimate_text_tokens"]
