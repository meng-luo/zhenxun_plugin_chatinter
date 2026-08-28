from __future__ import annotations

from .chat_dialogue_planner import DialogueState, normalize_message_text


def build_dialogue_state_prompt(
    state: DialogueState | None,
    *,
    current_message_text: str = "",
) -> str:
    if state is None:
        return ""
    _ = current_message_text
    parts: list[str] = []
    if state.continuity in {"same_topic", "followup"}:
        topic_hint = normalize_message_text(state.topic_hint)[:60]
        if topic_hint:
            parts.append(f"延续话题：{topic_hint}")
    return "；".join(parts) if parts else ""


__all__ = ["build_dialogue_state_prompt"]
