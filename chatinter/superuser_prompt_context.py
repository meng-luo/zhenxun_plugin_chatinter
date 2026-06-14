from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .route_text import normalize_message_text


def build_superuser_tool_context(*, cards: Iterable[Any]) -> str:
    materialized = list(cards or [])
    names = [
        normalize_message_text(str(getattr(card, "name", "") or ""))
        for card in materialized
    ]
    read_only = [
        name
        for name, card in zip(names, materialized)
        if bool(getattr(card, "read_only", False))
    ]
    approval_sensitive = [
        name
        for name, card in zip(names, materialized)
        if str(getattr(card, "risk", "") or "") in {"high", "danger"}
        or not bool(getattr(card, "read_only", False))
    ]
    background = [
        name
        for name, card in zip(names, materialized)
        if bool(getattr(card, "background_capable", False))
    ]
    artifacts = [
        name
        for name, card in zip(names, materialized)
        if bool(getattr(card, "produces_artifacts", False))
    ]
    return "\n".join(
        [
            "<superuser_agent_tool_guidance>",
            "<available_superuser_tool_summary>",
            f"tool_count={len(materialized)}",
            f"tools={', '.join(name for name in names if name)}",
            f"read_only_first={', '.join(read_only)}",
            f"approval_sensitive={', '.join(approval_sensitive)}",
            f"background_capable={', '.join(background)}",
            f"produces_artifacts={', '.join(artifacts)}",
            "</available_superuser_tool_summary>",
            "推荐流程：读代码 -> 计划/准备补丁 -> 应用 -> 运行验收。",
            "优先使用只读工具确认状态；高风险写操作需要明确目标和回滚思路。",
            "</superuser_agent_tool_guidance>",
        ]
    )


__all__ = ["build_superuser_tool_context"]
