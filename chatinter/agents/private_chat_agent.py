"""Lightweight private chat agent boundary.

Ordinary private chat is plain conversation: no plugin command index, no
runtime tools, no superuser/MCP tool registration.  Keeping it outside the
mixed AgentRuntime path avoids paying tool-routing costs for casual chat.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from zhenxun.services.llm import AI, LLMMessage

from ..config import (
    build_reasoning_generation_config,
    get_config_value,
    get_fallback_models,
    get_model_name,
)
from ..main_request_models import (
    MainRequestOutput,
    MainRequestResult,
    MainRequestTimelineItem,
)
from ..native_route import NativeRouteDecision, NativeRouteReport
from ..provider_failover import request_with_failover
from ..route_text import normalize_message_text
from .core import (
    PRIVATE_CHAT_TOOL_SCOPE,
    AgentObservation,
    AgentRequest,
    AgentResult,
    fallback_text,
    record_prompt_tokens,
)

if TYPE_CHECKING:
    from zhenxun.services.llm.types.models import LLMContentPart, LLMResponse

_PRIVATE_CHAT_STAGE = "private_chat_agent"
_PRIVATE_HISTORY_MESSAGES = 16
_PRIVATE_CONTEXT_CHAR_BUDGET = 9000
_PRIVATE_SYSTEM_TEXT_LIMIT = 3000
_PRIVATE_TEXT_LIMIT = 5000


class PrivateChatAgent:
    """Boundary for ordinary private conversation."""

    async def run(self, request: AgentRequest) -> AgentResult:
        del request.knowledge_base, request.has_reply
        del request.command_tools, request.route_executor, request.kwargs

        started = time.perf_counter()
        trace_id = f"private-{int(time.time() * 1000):x}"
        ai = AI(session_id=f"chatinter-private:{request.session_key or 'global'}")
        model_name = get_model_name()
        messages = _private_chat_messages(request.messages)

        if request.budget_controller is not None:
            record_prompt_tokens(
                budget_controller=request.budget_controller,
                messages=messages,
            )

        async def _do_request(model: str | None) -> "LLMResponse":
            return await ai.generate_internal(
                messages,
                model=model,
                config=build_reasoning_generation_config(),
                tools=None,
                tool_choice=None,
                timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
            )

        outcome = await request_with_failover(
            primary_model=model_name,
            fallback_models=get_fallback_models(),
            request_fn=_do_request,
            trace_id=trace_id,
        )
        reply = fallback_text(str(outcome.response.text or ""))
        timeline = (
            MainRequestTimelineItem(
                role="user",
                kind="current_user",
                content=request.message_text,
            ),
            MainRequestTimelineItem(
                role="assistant",
                kind="private_chat_response",
                content=reply,
                metadata={
                    "used_model": outcome.used_model or "<default>",
                    "fallback_attempts": [
                        {
                            "model": item.model,
                            "kind": item.kind,
                            "error": item.error,
                        }
                        for item in outcome.attempts
                    ],
                },
            ),
        )
        report = NativeRouteReport(helper_mode=False)
        report.note_candidate_policy(reason="private_chat_agent", limit=0)
        report.finalize(reason="private_chat_agent", stage=_PRIVATE_CHAT_STAGE)
        if request.budget_controller is not None:
            request.budget_controller.record_stage(
                _PRIVATE_CHAT_STAGE,
                time.perf_counter() - started,
            )
        result = MainRequestResult(
            decision=NativeRouteDecision(
                action="chat",
                confidence=0.88,
                reason="private_chat_agent",
            ),
            route_result=None,
            report=report,
            timeline=timeline,
            output=MainRequestOutput(
                final_text=reply,
                memory_text=_memory_text(request.message_text, reply),
                should_send=True,
                outcome="chat_completed",
                feedback_kind="chat_completed",
                record_chat_feedback=True,
                observation_reason="chat_completed",
            ),
        )
        return AgentResult(
            agent_kind="private_chat",
            main_result=result,
            observations=(AgentObservation(kind="single_llm_chat", status="ok"),),
            tool_scope=PRIVATE_CHAT_TOOL_SCOPE,
            elapsed_ms=max(int((time.perf_counter() - started) * 1000), 0),
        )


def _memory_text(user_text: str, reply_text: str) -> str:
    return "\n".join(
        item
        for item in (
            f"user/current_user: {normalize_message_text(user_text)}",
            f"assistant/final_output: {normalize_message_text(reply_text)}",
        )
        if normalize_message_text(item)
    )[:4000]


def _private_chat_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    if not messages:
        return []
    system = next((item for item in messages if item.role == "system"), None)
    chat_messages = [
        _compact_message(item)
        for item in messages
        if item.role in {"user", "assistant"} and not getattr(item, "tool_calls", None)
    ]
    selected = _select_recent_context(chat_messages)
    omitted = max(len(chat_messages) - len(selected), 0)
    result: list[LLMMessage] = []
    if system is not None:
        result.append(_compact_message(system, limit=_PRIVATE_SYSTEM_TEXT_LIMIT))
    if omitted:
        result.append(
            LLMMessage.system(
                f"私聊上下文已压缩：省略 {omitted} 条较早对话，仅保留最近相关上下文。"
            )
        )
    result.extend(selected)
    return result


def _select_recent_context(messages: list[LLMMessage]) -> list[LLMMessage]:
    selected: list[LLMMessage] = []
    used_chars = 0
    for message in reversed(messages):
        if len(selected) >= _PRIVATE_HISTORY_MESSAGES:
            break
        size = _content_size(message.content)
        if selected and used_chars + size > _PRIVATE_CONTEXT_CHAR_BUDGET:
            break
        selected.append(message)
        used_chars += size
    selected.reverse()
    return selected


def _compact_message(
    message: LLMMessage,
    *,
    limit: int = _PRIVATE_TEXT_LIMIT,
) -> LLMMessage:
    content = message.content
    if isinstance(content, str):
        content = normalize_message_text(content)[:limit]
    else:
        content = _compact_parts(content, limit=limit)
    return LLMMessage(role=message.role, content=content, name=message.name)


def _compact_parts(
    parts: list["LLMContentPart"],
    *,
    limit: int,
) -> list["LLMContentPart"]:
    result: list[LLMContentPart] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            try:
                part = part.model_copy(
                    update={"text": normalize_message_text(text)[:limit]}
                )
            except Exception:
                pass
        result.append(part)
    return result


def _content_size(content: str | list["LLMContentPart"]) -> int:
    if isinstance(content, str):
        return len(content)
    total = 0
    for part in content:
        text = getattr(part, "text", None)
        total += len(text) if isinstance(text, str) else 200
    return total


__all__ = ["PrivateChatAgent"]
