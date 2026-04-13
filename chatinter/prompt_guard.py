from __future__ import annotations

from dataclasses import dataclass

from zhenxun.services.llm.types.models import LLMMessage

from .turn_runtime import (
    TurnBudgetController,
    detect_prompt_cache_break,
    estimate_text_tokens,
    fingerprint_prompt_section,
    trim_text_to_tokens,
)

_SOFT_PROMPT_RATIO = 0.82
_MESSAGE_COMPACT_KEEP = 6


@dataclass(frozen=True)
class PromptGuardResult:
    system_prompt: str
    context_text: str
    user_text: str
    estimated_tokens: int
    cache_break: bool
    compacted: bool


def guard_prompt_sections(
    *,
    session_key: str,
    stage: str,
    system_prompt: str,
    context_text: str = "",
    user_text: str = "",
    controller: TurnBudgetController | None = None,
) -> PromptGuardResult:
    system_text = str(system_prompt or "")
    context_value = str(context_text or "")
    user_value = str(user_text or "")
    fingerprint = fingerprint_prompt_section(system_text, context_value[:512], user_value[:256])
    cache_break = detect_prompt_cache_break(
        session_key=session_key,
        stage=stage,
        fingerprint=fingerprint,
    )

    if controller is not None:
        remaining = controller.prompt_budget_remaining()
        token_budget = remaining if remaining > 0 else max(int(controller.prompt_budget_tokens * 0.1), 200)
    else:
        token_budget = 4000
    soft_budget = max(int(token_budget * _SOFT_PROMPT_RATIO), 48)
    estimated = estimate_text_tokens(f"{system_text}\n{context_value}\n{user_value}")
    compacted = False

    if estimated > soft_budget:
        compacted = True
        system_budget = max(int(soft_budget * 0.25), 180)
        context_budget = max(int(soft_budget * 0.45), 220)
        user_budget = max(soft_budget - system_budget - context_budget, 120)
        system_text = trim_text_to_tokens(system_text, system_budget)
        context_value = trim_text_to_tokens(context_value, context_budget)
        user_value = trim_text_to_tokens(user_value, user_budget)
        estimated = estimate_text_tokens(f"{system_text}\n{context_value}\n{user_value}")

    if controller is not None:
        controller.record_prompt_use(
            stage=stage,
            estimated_tokens=estimated,
            cache_break=cache_break,
            compacted=compacted,
        )

    return PromptGuardResult(
        system_prompt=system_text,
        context_text=context_value,
        user_text=user_value,
        estimated_tokens=estimated,
        cache_break=cache_break,
        compacted=compacted,
    )


def compact_agent_messages(
    *,
    session_key: str,
    stage: str,
    messages: list[LLMMessage],
    controller: TurnBudgetController | None = None,
) -> tuple[list[LLMMessage], bool]:
    if len(messages) <= _MESSAGE_COMPACT_KEEP:
        total_tokens = sum(_message_tokens(message) for message in messages)
        if controller is not None:
            controller.record_prompt_use(
                stage=stage,
                estimated_tokens=total_tokens,
                cache_break=False,
                compacted=False,
            )
        return messages, False

    if controller is not None:
        remaining = controller.prompt_budget_remaining()
        token_budget = remaining if remaining > 0 else max(int(controller.prompt_budget_tokens * 0.1), 200)
    else:
        token_budget = 4000
    total_tokens = sum(_message_tokens(message) for message in messages)
    if total_tokens <= token_budget:
        if controller is not None:
            controller.record_prompt_use(
                stage=stage,
                estimated_tokens=total_tokens,
                cache_break=False,
                compacted=False,
            )
        return messages, False

    system_messages = [message for message in messages if message.role == "system"]
    tail_messages = messages[-_MESSAGE_COMPACT_KEEP:]
    middle_messages = messages[len(system_messages) : max(len(messages) - _MESSAGE_COMPACT_KEEP, len(system_messages))]
    summary_lines: list[str] = []
    for message in middle_messages[-8:]:
        role = str(message.role or "unknown")
        snippet = _message_preview(message)
        if snippet:
            summary_lines.append(f"{role}: {snippet}")
    summary_text = "先前上下文摘要:\n" + "\n".join(summary_lines)
    compacted_messages = [*system_messages[:1]]
    compacted_messages.append(LLMMessage.system(trim_text_to_tokens(summary_text, 220)))
    compacted_messages.extend(tail_messages)

    compacted_tokens = sum(_message_tokens(message) for message in compacted_messages)
    cache_break = detect_prompt_cache_break(
        session_key=session_key,
        stage=stage,
        fingerprint=fingerprint_prompt_section(*(str(_message_preview(m)) for m in compacted_messages)),
    )
    if controller is not None:
        controller.record_prompt_use(
            stage=stage,
            estimated_tokens=compacted_tokens,
            cache_break=cache_break,
            compacted=True,
        )
    return compacted_messages, True


def _message_preview(message: LLMMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None) or getattr(part, "thought_text", None) or ""
            if text:
                parts.append(str(text))
        source = " ".join(parts)
    else:
        source = str(content or "")
    return " ".join(source.split())[:160]


def _message_tokens(message: LLMMessage) -> int:
    return estimate_text_tokens(_message_preview(message))


__all__ = [
    "PromptGuardResult",
    "compact_agent_messages",
    "guard_prompt_sections",
]
