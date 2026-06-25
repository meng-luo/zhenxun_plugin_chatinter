"""Thin dispatcher for ChatInter scenario agents.

The heavy command-routing and AgentRuntime plumbing lives behind the scenario
agents.  This module keeps the historical public function and result types so
callers do not need to know which agent handled the turn.
"""

from __future__ import annotations

import time
from typing import Any

from zhenxun.services import logger
from zhenxun.services.llm import LLMMessage

from .agents.core import AgentRequest
from .main_request_models import (
    CandidateObligationEvaluation,
    MainRequestOutput,
    MainRequestReplyHook,
    MainRequestResult,
    MainRequestRouteHook,
    MainRequestTimelineItem,
    ToolObligationDecision,
)
from .main_request_support import (
    _apply_tool_exposure_policy,
    _candidates_for_task_routes,
    _fallback_result,
    _finalize_result,
    _resolve_tool_obligation,
    _result_from_task_execution_queue,
    _try_local_direct_command,
    _user_timeline_item,
)
from .models.pydantic_models import PluginKnowledgeBase
from .native_executor import ExecuteNativeRoute
from .native_route import NativeRouteReport
from .route_text import is_usage_question, normalize_message_text
from .turn_runtime import TurnBudgetController

_MAIN_STAGE = "main_request"


async def run_chatinter_main_request(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None,
    messages: list[LLMMessage],
    route_executor: ExecuteNativeRoute,
    route_completed_hook: MainRequestRouteHook | None = None,
    reply_hook: MainRequestReplyHook | None = None,
    enable_plugin_tools: bool = True,
    initial_command_exposure: bool = False,
    enable_agent_tools: bool = False,
    progress_hook: Any | None = None,
    _skip_agent_wrapper: bool = False,
) -> MainRequestResult:
    """Run one ChatInter turn and return the legacy result payload.

    ``enable_plugin_tools`` and ``enable_agent_tools`` remain part of the
    public API, but this module no longer mixes the two policies.  They only
    select one scenario agent.
    """

    normalized_message = normalize_message_text(message_text)
    report = NativeRouteReport(helper_mode=is_usage_question(normalized_message))
    started = time.perf_counter()

    try:
        result = await _run_main_request(
            normalized_message,
            knowledge_base,
            session_key=session_key,
            budget_controller=budget_controller,
            has_reply=has_reply,
            command_tools=command_tools,
            messages=messages,
            route_executor=route_executor,
            report=report,
            enable_plugin_tools=enable_plugin_tools,
            initial_command_exposure=initial_command_exposure,
            enable_agent_tools=enable_agent_tools,
            progress_hook=progress_hook,
            _skip_agent_wrapper=_skip_agent_wrapper,
        )
        return await _finalize_result(
            result,
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )
    except Exception as exc:
        logger.error(f"ChatInter main request failed: {exc}")
        return await _finalize_result(
            _fallback_result(
                report=report,
                reason=f"main_request_error:{type(exc).__name__}",
                reply="抱歉，我刚刚处理失败了。",
                timeline=[_user_timeline_item(normalized_message)],
            ),
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )
    finally:
        if budget_controller is not None:
            budget_controller.record_stage(_MAIN_STAGE, time.perf_counter() - started)


async def _dispatch_agent(
    request: AgentRequest,
    *,
    enable_plugin_tools: bool,
    enable_agent_tools: bool,
):
    if enable_agent_tools:
        from .agents.superuser_agent import SuperuserAgent

        return await SuperuserAgent().run(request)
    if enable_plugin_tools:
        from .agents.plugin_command_agent import PluginCommandAgent

        return await PluginCommandAgent(
            candidates_for_task_routes=_candidates_for_task_routes,
            result_from_task_execution_queue=_result_from_task_execution_queue,
            try_local_direct_command=_try_local_direct_command,
        ).run(request)
    from .agents.private_chat_agent import PrivateChatAgent

    return await PrivateChatAgent().run(request)


async def _run_main_request(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None,
    messages: list[LLMMessage],
    route_executor: ExecuteNativeRoute,
    report: NativeRouteReport,
    enable_plugin_tools: bool,
    initial_command_exposure: bool,
    enable_agent_tools: bool,
    progress_hook: Any | None = None,
    _skip_agent_wrapper: bool = False,
) -> MainRequestResult:
    """Compatibility dispatcher for older tests/imports."""

    request = AgentRequest(
        message_text=message_text,
        knowledge_base=knowledge_base,
        session_key=session_key,
        budget_controller=budget_controller,
        has_reply=has_reply,
        command_tools=command_tools,
        messages=messages,
        route_executor=route_executor,
        kwargs={
            "enable_plugin_tools": enable_plugin_tools,
            "initial_command_exposure": initial_command_exposure,
            "enable_agent_tools": enable_agent_tools,
            "progress_hook": progress_hook,
            "_skip_agent_wrapper": _skip_agent_wrapper,
        },
        report=report,
    )
    return (
        await _dispatch_agent(
            request,
            enable_plugin_tools=enable_plugin_tools,
            enable_agent_tools=enable_agent_tools,
        )
    ).to_main_result()


__all__ = [
    "CandidateObligationEvaluation",
    "MainRequestOutput",
    "MainRequestResult",
    "MainRequestTimelineItem",
    "ToolObligationDecision",
    "_apply_tool_exposure_policy",
    "_resolve_tool_obligation",
    "run_chatinter_main_request",
]
