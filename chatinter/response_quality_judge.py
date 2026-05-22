"""Lightweight final reply quality checks for chat turns.

The judge is deliberately small and runtime-oriented.  It does not replace the
model or add plugin-specific routing; it only catches generic failure modes
before a message is sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .chat_dialogue_planner import DialogueState
from .completion_validator import validate_final_reply
from .route_text import normalize_message_text

QualityAction = Literal["ok", "revise", "block"]

_STIFF_PHRASES = (
    "尊敬的用户",
    "您好，关于您的问题",
    "作为一个人工智能",
    "我无法提供情感",
)
_QUESTION_MARKERS = ("吗", "么", "嘛", "？", "?", "怎么", "为什么", "是什么", "是啥")
_ACTION_CLAIM_PATTERNS = (
    "我已",
    "我已经",
    "我这边已",
    "我这边已经",
    "帮你完成",
    "帮你处理",
    "帮你执行",
    "帮你生成",
    "帮你做成",
    "帮你发送",
    "帮你查询",
    "给你生成",
    "给你做",
    "处理好了",
    "做好了",
    "做成了",
    "生成好了",
    "发好了",
    "查到了",
    "执行完",
)


@dataclass(frozen=True)
class ResponseQualityResult:
    action: QualityAction
    reason: str = ""
    revised_text: str = ""
    instruction: str = ""

    @property
    def ok(self) -> bool:
        return self.action == "ok"


class ResponseQualityJudge:
    @staticmethod
    def should_judge(*, scenario: str, main_result: Any | None) -> bool:
        """Only judge ordinary chat, never plugin execution or superuser Agent.

        Tool/action correctness is handled by AgentRuntime observations and
        validators.  This judge is for conversational polish only.
        """

        if main_result is None:
            return False
        if normalize_message_text(scenario) == "superuser_agent":
            return False
        output = getattr(main_result, "output", None)
        if output is None:
            return False
        if normalize_message_text(str(getattr(output, "outcome", "") or "")) != "chat_completed":
            return False
        if not bool(getattr(output, "record_chat_feedback", False)):
            return False
        if bool(getattr(main_result, "handled_by_tools", False)):
            return False
        if bool(getattr(main_result, "tool_results", ()) or ()):
            return False
        return True

    @staticmethod
    def judge(
        *,
        final_text: str,
        original_message: str,
        dialogue_state: DialogueState | None,
        main_result: Any | None,
    ) -> ResponseQualityResult:
        reply = normalize_message_text(final_text)
        message = normalize_message_text(original_message)
        if not reply:
            return ResponseQualityResult(
                action="block",
                reason="empty_final_reply",
                revised_text="我暂时没想好怎么回答你。",
            )

        tool_validation = validate_final_reply(
            final_text=reply,
            tool_obligation=str(
                getattr(getattr(main_result, "output", None), "observation_reason", "")
                or ""
            ),
            observations=_extract_observations(main_result),
            pending_tasks=(),
            tool_map={},
        )
        if not tool_validation.ok:
            return ResponseQualityResult(
                action="block",
                reason=tool_validation.reason,
                revised_text="这个需要实际工具结果确认，我刚刚还没有拿到可靠结果。",
                instruction="Do not claim tool/action completion without observation.",
            )

        if _claims_action_without_tool_result(reply, main_result):
            return ResponseQualityResult(
                action="block",
                reason="action_claim_without_tool_result",
                revised_text="这个需要实际执行结果确认，我刚刚还没有拿到可靠结果。",
            )

        if _looks_stiff(reply, dialogue_state):
            return ResponseQualityResult(
                action="revise",
                reason="too_stiff_for_dialogue_state",
                revised_text=_soften_reply(reply),
            )

        if _looks_off_topic(message, reply, dialogue_state):
            return ResponseQualityResult(
                action="revise",
                reason="possible_off_topic",
                instruction=(
                    "Reply should answer the user's current message directly; "
                    "avoid generic filler."
                ),
            )

        return ResponseQualityResult(action="ok")


def _extract_observations(main_result: Any | None) -> list[Any]:
    if main_result is None:
        return []
    observations: list[Any] = []
    for execution in getattr(main_result, "executions", ()) or ():
        output = getattr(execution, "output", None)
        if isinstance(output, dict):
            observations.append(_ObservationProxy(output))
    for tool_result in getattr(main_result, "tool_results", ()) or ():
        output = getattr(tool_result, "output", None)
        if isinstance(output, dict):
            observations.append(_ObservationProxy(output))
    return observations


class _ObservationProxy:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.ok = bool(output.get("ok"))
        self.command_id = normalize_message_text(str(output.get("command_id", "")))


def _claims_action_without_tool_result(reply: str, main_result: Any | None) -> bool:
    if not any(pattern in reply for pattern in _ACTION_CLAIM_PATTERNS):
        return False
    if main_result is None:
        return True
    if getattr(main_result, "handled_by_tools", False):
        return False
    return not bool(
        getattr(main_result, "executions", ())
        or getattr(main_result, "tool_results", ())
    )


def _looks_stiff(reply: str, state: DialogueState | None) -> bool:
    if not any(phrase in reply for phrase in _STIFF_PHRASES):
        return False
    if state is None:
        return True
    return state.tone in {"casual", "warm", "playful", "empathetic"}


def _soften_reply(reply: str) -> str:
    softened = reply
    for phrase in _STIFF_PHRASES:
        softened = softened.replace(phrase, "")
    return normalize_message_text(softened).lstrip("，,。.!！？?") or reply


def _looks_off_topic(
    message: str,
    reply: str,
    state: DialogueState | None,
) -> bool:
    if not message or not reply:
        return False
    if state is not None and state.dialogue_purpose in {"chat", "support"}:
        return False
    if len(reply) <= 8 and any(marker in message for marker in _QUESTION_MARKERS):
        return True
    if reply in {"好的", "嗯嗯", "可以", "收到"} and len(message) >= 12:
        return True
    return False


__all__ = [
    "ResponseQualityJudge",
    "ResponseQualityResult",
]
