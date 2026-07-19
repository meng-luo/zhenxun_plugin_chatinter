from __future__ import annotations

from .chat_dialogue_planner import DialogueState, normalize_message_text

_TONE_HINTS = {
    "casual": "自然随意",
    "warm": "温和亲近",
    "focused": "专注直接",
    "playful": "轻松活泼",
    "serious": "认真克制",
    "empathetic": "温和共情",
}
_POSTURE_HINTS = {
    "listen": "先理解再回应，不机械复述",
    "answer": "直接回答，结论优先",
    "banter": "自然接话，可以适度打趣",
    "support": "先接住情绪，再给实际回应",
    "clarify": "缺少关键信息时只追问一个重点",
    "step": "按清晰步骤推进",
}
_LENGTH_HINTS = {
    "short": "简短自然",
    "medium": "适度展开",
    "long": "充分说明但避免重复",
}


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
            parts.append(f"承接话题：{topic_hint}")
    guidance = "，".join(
        (
            _TONE_HINTS.get(state.tone, "自然清晰"),
            _POSTURE_HINTS.get(state.reply_posture, "直接回应当前消息"),
            _LENGTH_HINTS.get(state.response_length, "按问题自然控制长度"),
        )
    )
    parts.append(f"回应方式：{guidance}")
    return "；".join(parts)


__all__ = ["build_dialogue_state_prompt"]
