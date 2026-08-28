"""ChatInter token estimation compatibility boundary."""

from __future__ import annotations

import math
import re
from typing import Any


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
    cjk_chars = len(
        re.findall(
            r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u3040-\u309f"
            r"\u30a0-\u30ff\uac00-\ud7af\u1100-\u11ff]",
            value,
        )
    )
    return math.ceil(cjk_chars * 1.2 + (len(value) - cjk_chars) * 0.3)


def parse_usage_info(usage_info: dict | None):
    from zhenxun.services.ai.core.engine.token_counter import (
        parse_usage_info as host_parse_usage_info,
    )

    usage = host_parse_usage_info(usage_info)
    if not isinstance(usage_info, dict):
        return usage
    if not any(
        key in usage_info
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens")
    ):
        return usage
    uncached = max(int(usage_info.get("input_tokens", 0) or 0), 0)
    cache_read = max(int(usage_info.get("cache_read_input_tokens", 0) or 0), 0)
    cache_creation = max(
        int(usage_info.get("cache_creation_input_tokens", 0) or 0),
        0,
    )
    completion = max(int(usage_info.get("output_tokens", 0) or 0), 0)
    usage.prompt_tokens = uncached + cache_read + cache_creation
    usage.completion_tokens = completion
    usage.total_tokens = usage.prompt_tokens + completion
    usage.prompt_cache_hit_tokens = cache_read
    usage.prompt_cache_miss_tokens = uncached + cache_creation
    return usage


def usage_reports_prompt_cache(usage_info: Any) -> bool:
    if not isinstance(usage_info, dict):
        return False
    if any(
        key in usage_info
        for key in (
            "cachedContentTokenCount",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    ):
        return True
    for key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage_info.get(key)
        if isinstance(details, dict) and "cached_tokens" in details:
            return True
    return False


__all__ = [
    "estimate_text_tokens",
    "parse_usage_info",
    "usage_reports_prompt_cache",
]
