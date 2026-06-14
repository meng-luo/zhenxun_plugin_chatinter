from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .route_text import normalize_message_text


@dataclass(frozen=True)
class HistoryBudget:
    dialog_limit: int
    chatroom_limit: int
    turn_token_budget: int
    reason: str


def resolve_history_budget(
    state: Any,
    *,
    session_context_limit: int,
    is_group: bool,
) -> HistoryBudget:
    purpose = normalize_message_text(str(getattr(state, "dialogue_purpose", "")))
    response_length = normalize_message_text(
        str(getattr(state, "response_length", "short"))
    )
    group_policy = normalize_message_text(str(getattr(state, "group_reply_policy", "")))
    context_limit = max(int(session_context_limit or 0), 0)

    if purpose in {"agent_task", "tool", "workflow"} or response_length == "long":
        return HistoryBudget(
            dialog_limit=min(12, context_limit) if context_limit else 12,
            chatroom_limit=0 if not is_group else min(12, context_limit),
            turn_token_budget=1600,
            reason=f"expanded:{purpose or response_length}",
        )

    if purpose in {"identity", "answer", "status", "help"}:
        return HistoryBudget(
            dialog_limit=min(10, context_limit) if context_limit else 10,
            chatroom_limit=12 if is_group else 0,
            turn_token_budget=1300,
            reason=f"purpose:{purpose}",
        )

    if is_group and group_policy in {"brief_react", "brief", "react"}:
        return HistoryBudget(
            dialog_limit=min(4, context_limit) if context_limit else 4,
            chatroom_limit=6,
            turn_token_budget=650,
            reason="brief_group_chat",
        )

    return HistoryBudget(
        dialog_limit=min(8, context_limit) if context_limit else 8,
        chatroom_limit=8 if is_group else 0,
        turn_token_budget=1000,
        reason="default",
    )


__all__ = ["HistoryBudget", "resolve_history_budget"]
