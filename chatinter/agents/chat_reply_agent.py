"""Lightweight private chat agent boundary.

Ordinary private chat is plain conversation: no plugin command index, no
runtime tools, no superuser/MCP tool registration.  Keeping it outside the
mixed AgentRuntime path avoids paying tool-routing costs for casual chat.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from zhenxun.services.ai.core.engine.token_counter import parse_usage_info

from ..config import (
    CHAT_RESPONSE_TIMEOUT_SECONDS,
    build_agent_generation_config,
    get_agent_max_output_tokens,
    get_agent_model,
    get_fallback_models,
    resolve_agent_context_window_tokens,
)
from ..llm_compat import AI
from ..main_request_models import (
    MainRequestOutput,
    MainRequestResult,
    MainRequestTimelineItem,
)
from ..native_route import NativeRouteDecision, NativeRouteReport
from ..provider_failover import request_with_failover
from ..route_text import normalize_message_text
from ..turn_runtime import estimate_text_tokens
from .core import (
    PRIVATE_CHAT_TOOL_SCOPE,
    AgentObservation,
    AgentResult,
    PrivateChatRequest,
    estimate_prompt_tokens,
    fallback_text,
)

if TYPE_CHECKING:
    from ..llm_compat import LLMResponse

_PRIVATE_CHAT_STAGE = "private_chat_agent"
_CHAT_PROTOCOL_MARGIN_TOKENS = 2_048


class ChatReplyAgent:
    """Boundary for private chat and group no-selection fallback replies."""

    async def run(self, request: PrivateChatRequest) -> AgentResult:
        started = time.perf_counter()
        trace_id = f"private-{int(time.time() * 1000):x}"
        ai = AI(session_id=f"chatinter-private:{request.session_key or 'global'}")
        model_name = get_agent_model("chat")
        generation_config = build_agent_generation_config("chat")
        messages = _fit_chat_messages(
            request.messages,
            max_input_tokens=resolve_agent_context_window_tokens(
                "chat",
                model_name,
            ),
            output_reserve_tokens=get_agent_max_output_tokens("chat"),
        )
        estimated_prompt_tokens = estimate_prompt_tokens(messages)

        async def _do_request(model: str | None) -> "LLMResponse":
            return await ai.generate_internal(
                messages,
                model=model,
                config=generation_config,
                tools=None,
                tool_choice=None,
                timeout=float(CHAT_RESPONSE_TIMEOUT_SECONDS),
            )

        outcome = await request_with_failover(
            primary_model=model_name,
            fallback_models=get_fallback_models(model_name),
            request_fn=_do_request,
            trace_id=trace_id,
        )
        response_text = str(outcome.response.text or "")
        reply = fallback_text(response_text)
        if request.budget_controller is not None:
            usage_info = getattr(outcome.response, "usage_info", None)
            if isinstance(usage_info, dict) and usage_info:
                usage = parse_usage_info(usage_info)
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
            else:
                prompt_tokens = estimated_prompt_tokens
                completion_tokens = estimate_text_tokens(response_text)
            request.budget_controller.record_model_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        timeline = (
            MainRequestTimelineItem(
                role="user",
                kind="current_user",
                content=request.message_text,
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


def _fit_chat_messages(
    messages: list[object],
    *,
    max_input_tokens: int,
    output_reserve_tokens: int,
) -> list[object]:
    """Drop only complete old dialogue groups when a chat prompt is oversized."""

    fitted = list(messages)
    limit = max(
        int(max_input_tokens)
        - max(int(output_reserve_tokens), 0)
        - _CHAT_PROTOCOL_MARGIN_TOKENS,
        1,
    )
    if estimate_prompt_tokens(fitted) <= limit or len(fitted) <= 2:
        return fitted

    stable = [fitted[0]]
    current = [fitted[-1]]
    groups: list[list[object]] = []
    for message in fitted[1:-1]:
        role = str(getattr(message, "role", "") or "")
        if role in {"system", "user"} or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)

    kept: list[list[object]] = []
    for group in reversed(groups):
        recent = [item for part in reversed(kept) for item in part]
        candidate = stable + group + recent + current
        if estimate_prompt_tokens(candidate) > limit:
            break
        kept.append(group)

    return stable + [item for group in reversed(kept) for item in group] + current


__all__ = ["ChatReplyAgent"]
