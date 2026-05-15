"""Unified ChatInter main request runner.

This module prepares the turn prompt and command tools, then delegates the
model/tool loop to ``agent_runtime``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from inspect import isawaitable
import time
from typing import Any, cast

from zhenxun.services import logger
from zhenxun.services.llm import LLMContentPart, LLMMessage
from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types.models import ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .agent_runtime import AgentRuntime, AgentRuntimeTimelineItem
from .chat_dialogue_planner import ChatDialoguePlan
from .chat_strategy import build_chat_strategy_prompt
from .command_catalog_tool import (
    COMMAND_CATALOG_TOOL_NAME,
    CommandCatalogState,
    CommandCatalogTool,
)
from .config import build_reasoning_generation_config, get_config_value, get_model_name
from .models.pydantic_models import PluginKnowledgeBase
from .native_executor import (
    ExecuteNativeRoute,
    NativeCommandExecutionContext,
    NativeToolExecutionResult,
)
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
    NativeRouteResult,
)
from .route_text import is_usage_question, normalize_message_text
from .tool_retriever import CommandToolRetriever
from .turn_runtime import TurnBudgetController

_MAIN_STAGE = "main_request"
_MAIN_REQUEST_RULES = """
<chatinter_main_request>
你正在处理群聊消息：可以直接聊天，也可以调用候选插件工具。
候选工具是真实插件命令；调用工具会执行插件。
命令不会一次性全部暴露。你需要插件能力但当前没有对应命令 schema 时，
先调用 retrieve_plugin_commands 检索能力；检索结果会把相关命令 schema
注入下一轮工具列表，然后再调用真实命令工具执行。
每个注入后的命令工具已暴露完整 schema；只根据工具 description 和参数 schema
选择工具并填写参数，不要猜 schema 外参数。
每次工具调用必须填写 task_text：它是该工具负责的用户子任务原文。
如果一句话里有多个任务，请把每个工具调用的 task_text 拆成互不重叠的短片段；
不要把“然后/再/最后/顺便”等后续任务塞进前一个工具。
用户明确要执行插件能力、查询插件用法，或自然语言需求明显对应插件时才调用工具。
普通闲聊、玩梗、讨论命令概念、候选工具不匹配、目标不清时，不要调用工具，直接自然回复或简短说明需要的信息。
retrieve_plugin_commands 只负责发现能力，不代表插件任务已经完成。
真实命令工具返回的是 Observation：如果 need_continue=true，请优先根据
remaining_task_hint 继续检索或调用后续工具；不要把它当作最终回复。
你会在工具执行后继续看到工具结果；如果原始消息还有未完成的子任务，请继续调用后续工具。
所有子任务都已完成或不需要工具时，再给出最终回复。
</chatinter_main_request>
""".strip()


MainRequestRouteHook = Callable[["MainRequestResult"], Awaitable[None] | None]
MainRequestReplyHook = Callable[[str], Awaitable[str] | str]


@dataclass(frozen=True)
class MainRequestOutput:
    analysis: str = "main request"
    final_text: str = ""
    memory_text: str = ""
    should_send: bool = True
    outcome: str = "chat_completed"
    feedback_kind: str = "chat_completed"
    record_chat_feedback: bool = True
    observation_reason: str = "chat_completed"


@dataclass(frozen=True)
class MainRequestTimelineItem:
    role: str
    kind: str
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "kind": self.kind,
        }
        if self.content:
            payload["content"] = self.content
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class MainRequestResult:
    decision: NativeRouteDecision
    route_result: NativeRouteResult | None
    report: NativeRouteReport
    executions: tuple[NativeToolExecutionResult, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    timeline: tuple[MainRequestTimelineItem, ...] = ()
    output: MainRequestOutput = field(default_factory=MainRequestOutput)

    @property
    def handled_by_tools(self) -> bool:
        return any(item.route_result is not None for item in self.executions)


async def run_chatinter_main_request(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None,
    system_prompt: str,
    context_xml: str,
    history_messages: list[LLMMessage] | None,
    image_parts: list[LLMContentPart] | None,
    dialogue_plan: ChatDialoguePlan | None,
    route_executor: ExecuteNativeRoute,
    route_completed_hook: MainRequestRouteHook | None = None,
    reply_hook: MainRequestReplyHook | None = None,
) -> MainRequestResult:
    normalized_message = normalize_message_text(message_text)
    report = NativeRouteReport(helper_mode=is_usage_question(normalized_message))

    if budget_controller is not None and not budget_controller.allow_classifier(
        _MAIN_STAGE
    ):
        return await _finalize_result(
            _fallback_result(
                report=report,
                reason="main_request_budget_exhausted",
                reply="我现在有点忙，稍后再试试吧。",
                timeline=[_user_timeline_item(normalized_message)],
            ),
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
        )

    started = time.perf_counter()
    try:
        result = await _run_main_request(
            normalized_message,
            knowledge_base,
            session_key=session_key,
            budget_controller=budget_controller,
            has_reply=has_reply,
            command_tools=command_tools,
            system_prompt=system_prompt,
            context_xml=context_xml,
            history_messages=history_messages,
            image_parts=image_parts,
            dialogue_plan=dialogue_plan,
            route_executor=route_executor,
            report=report,
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
            budget_controller.record_classifier(
                _MAIN_STAGE,
                time.perf_counter() - started,
            )


async def _run_main_request(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None,
    system_prompt: str,
    context_xml: str,
    history_messages: list[LLMMessage] | None,
    image_parts: list[LLMContentPart] | None,
    dialogue_plan: ChatDialoguePlan | None,
    route_executor: ExecuteNativeRoute,
    report: NativeRouteReport,
) -> MainRequestResult:
    retriever = CommandToolRetriever(
        knowledge_base,
        session_id=session_key,
        tools=cast(Any, command_tools),
    )
    catalog_state = CommandCatalogState(retriever=retriever)

    report.note_candidate_policy(
        reason="main_request_catalog_retrieval",
        limit=retriever.total_commands,
    )
    report.candidate_total = max(report.candidate_total, retriever.total_commands)
    report.note_tool_pool(1)

    catalog_tool = CommandCatalogTool(catalog_state)
    system_prompt_text = _build_system_prompt(system_prompt, dialogue_plan)
    user_text = message_text
    if context_xml:
        user_text = (
            f"{context_xml}\n\n"
            f"<current_user_message>{message_text}</current_user_message>"
        )

    user_content: str | list[LLMContentPart]
    if image_parts:
        user_content = [LLMContentPart.text_part(user_text), *image_parts]
    else:
        user_content = user_text

    messages = [
        LLMMessage.system(system_prompt_text),
        *list(history_messages or []),
        LLMMessage.user(user_content),
    ]
    tool_map: dict[str, ToolExecutable] = {
        COMMAND_CATALOG_TOOL_NAME: cast(ToolExecutable, catalog_tool),
    }
    command_context = NativeCommandExecutionContext(
        candidates=[],
        has_reply=has_reply,
        report=report,
        route_executor=route_executor,
        message_text=message_text,
    )
    run_context = RunContext(
        session_id=session_key,
        extra={
            "native_command_context": command_context,
            "command_catalog_state": catalog_state,
        },
    )
    runtime = AgentRuntime(
        session_key=session_key,
        messages=messages,
        tool_map=tool_map,
        run_context=run_context,
        message_text=message_text,
        model_name=get_model_name(),
        generation_config=build_reasoning_generation_config(),
        timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
        budget_controller=budget_controller,
    )
    agent_result = await runtime.run()
    timeline = _convert_runtime_timeline(agent_result.timeline)
    if catalog_state.candidates:
        report.note_prompt_exposure(catalog_state.candidates)
        report.tool_candidates = max(
            report.tool_candidates,
            catalog_state.injected_count,
        )
    report.tool_choice_count += sum(
        1 for item in agent_result.timeline if item.kind == "tool_call"
    )
    return _finish_agent_response(
        report=report,
        executions=command_context.executions,
        tool_results=list(agent_result.tool_results),
        timeline=timeline,
        final_text=agent_result.final_text,
        stop_reason=agent_result.stop_reason,
    )


def _finish_text_response(
    *,
    report: NativeRouteReport,
    executions: list[NativeToolExecutionResult],
    text: str,
    reason: str,
    timeline: list[MainRequestTimelineItem],
) -> MainRequestResult:
    reply = normalize_message_text(text)
    if not reply:
        reply = _fallback_final_reply(executions) or "我暂时没想好怎么回答你。"
    decision = NativeRouteDecision(
        action="chat",
        confidence=0.9 if executions else 0.84,
        reason=reason,
    )
    report.finalize(reason=reason, stage=_MAIN_STAGE)
    return MainRequestResult(
        decision=decision,
        route_result=_first_route(executions),
        report=report,
        executions=tuple(executions),
        timeline=tuple(timeline),
        output=MainRequestOutput(final_text=reply, memory_text=reply),
    )


def _fallback_result(
    *,
    report: NativeRouteReport,
    reason: str,
    reply: str,
    timeline: list[MainRequestTimelineItem] | None = None,
) -> MainRequestResult:
    decision = NativeRouteDecision(action="chat", confidence=0.0, reason=reason)
    report.finalize(reason=reason, stage=_MAIN_STAGE)
    return MainRequestResult(
        decision=decision,
        route_result=None,
        report=report,
        timeline=(
            *(timeline or []),
            MainRequestTimelineItem(
                role="system",
                kind="fallback",
                content=reason,
            ),
        ),
        output=MainRequestOutput(final_text=reply, memory_text=reply),
    )


def _finish_agent_response(
    *,
    report: NativeRouteReport,
    executions: list[NativeToolExecutionResult],
    tool_results: list[ToolResult],
    timeline: list[MainRequestTimelineItem],
    final_text: str,
    stop_reason: str,
) -> MainRequestResult:
    if not executions:
        return _finish_text_response(
            report=report,
            executions=executions,
            text=final_text,
            reason=f"main_request:{stop_reason}",
            timeline=timeline,
        )

    if report.final_reason == "init":
        report.finalize(reason="main_request:tool_loop_completed", stage=_MAIN_STAGE)
    command_tool_results = [
        result for result in tool_results if not _is_catalog_tool_result(result)
    ]
    reply = normalize_message_text(final_text) or _tool_execution_reply(
        executions,
        command_tool_results,
    )
    memory_text = _tool_memory_text(executions, command_tool_results)
    if reply:
        memory_text = "\n".join(
            item for item in [memory_text, reply] if normalize_message_text(item)
        )
    memory_text = memory_text or reply
    return MainRequestResult(
        decision=NativeRouteDecision(
            action="chat",
            confidence=0.9 if executions else 0.35,
            reason=f"main_request:{stop_reason}",
        ),
        route_result=_first_route(executions),
        report=report,
        executions=tuple(executions),
        tool_results=tuple(command_tool_results),
        timeline=tuple(timeline),
        output=MainRequestOutput(
            final_text=reply,
            memory_text=memory_text,
            should_send=bool(reply),
            outcome="tool_completed",
            feedback_kind="tool_completed",
            record_chat_feedback=False,
            observation_reason="route_success"
            if any(item.success for item in executions)
            else "reroute_failed",
        ),
    )


def _is_catalog_tool_result(result: ToolResult) -> bool:
    output = result.output if isinstance(result.output, dict) else {}
    return output.get("status") == "retrieved"


async def _finalize_result(
    result: MainRequestResult,
    *,
    route_completed_hook: MainRequestRouteHook | None,
    reply_hook: MainRequestReplyHook | None,
) -> MainRequestResult:
    if route_completed_hook is not None:
        maybe_awaitable = route_completed_hook(result)
        if maybe_awaitable is not None:
            await maybe_awaitable

    output = result.output
    if not output.should_send:
        return result

    final_text = normalize_message_text(output.final_text)
    if not final_text:
        final_text = (
            _fallback_final_reply(list(result.executions)) or "我暂时没想好怎么回答你。"
        )
    if reply_hook is not None:
        maybe_reply = reply_hook(final_text)
        final_text = (
            await maybe_reply if isawaitable(maybe_reply) else str(maybe_reply or "")
        )
    final_text = normalize_message_text(final_text)
    if not final_text:
        final_text = "我暂时没想好怎么回答你。"
    memory_text = (
        normalize_message_text(output.memory_text)
        if result.handled_by_tools
        else final_text
    )
    memory_text = memory_text or final_text
    return replace(
        result,
        timeline=_with_final_timeline(
            result.timeline,
            final_text=final_text,
            should_send=True,
        ),
        output=replace(
            output,
            final_text=final_text,
            memory_text=memory_text,
            should_send=True,
        ),
    )


def _first_route(
    executions: list[NativeToolExecutionResult],
) -> NativeRouteResult | None:
    for execution in executions:
        if execution.route_result is not None:
            return execution.route_result
    return None


def _fallback_final_reply(executions: list[NativeToolExecutionResult]) -> str:
    if not executions:
        return ""
    success_count = sum(1 for item in executions if item.success)
    latest = executions[-1]
    if latest.display_text:
        return latest.display_text
    if success_count:
        return "处理好了。"
    message = str(latest.output.get("error", "") or latest.reason or "").strip()
    return message or "这个暂时没处理成功。"


def _user_timeline_item(message_text: str) -> MainRequestTimelineItem:
    return MainRequestTimelineItem(
        role="user",
        kind="current_user",
        content=message_text,
    )


def _convert_runtime_timeline(
    items: tuple[AgentRuntimeTimelineItem, ...],
) -> list[MainRequestTimelineItem]:
    return [
        MainRequestTimelineItem(
            role=item.role,
            kind=item.kind,
            content=item.content,
            tool_name=item.tool_name,
            metadata=dict(item.metadata),
        )
        for item in items
    ]


def _with_final_timeline(
    timeline: tuple[MainRequestTimelineItem, ...],
    *,
    final_text: str,
    should_send: bool,
) -> tuple[MainRequestTimelineItem, ...]:
    if not final_text and not should_send:
        return timeline
    return (
        *timeline,
        MainRequestTimelineItem(
            role="assistant",
            kind="final_output",
            content=final_text,
            metadata={"sent_by_chatinter": should_send},
        ),
    )


def _tool_memory_text(
    executions: list[NativeToolExecutionResult],
    tool_results: list[ToolResult],
) -> str:
    lines: list[str] = []
    for execution in executions:
        output = execution.output if isinstance(execution.output, dict) else {}
        messages_sent = output.get("messages_sent")
        if isinstance(messages_sent, list):
            for item in messages_sent[:4]:
                text = normalize_message_text(str(item or ""))
                if text:
                    lines.append(text)
        message = normalize_message_text(
            str(output.get("error", "") or execution.display_text or "")
        )
        if message:
            lines.append(message)
    for result in tool_results:
        output = result.output if isinstance(result.output, dict) else {}
        messages_sent = output.get("messages_sent")
        if isinstance(messages_sent, list):
            for item in messages_sent[:4]:
                text = normalize_message_text(str(item or ""))
                if text:
                    lines.append(text)
        error = normalize_message_text(str(output.get("error", "") or ""))
        if error:
            lines.append(error)
    return "\n".join(dict.fromkeys(lines))


def _tool_execution_reply(
    executions: list[NativeToolExecutionResult],
    tool_results: list[ToolResult],
) -> str:
    if executions:
        successful_observations = [
            item
            for item in executions
            if item.success and bool(item.output.get("messages_sent"))
        ]
        if successful_observations and len(successful_observations) == len(
            executions
        ) == len(tool_results):
            return ""
        messages = [
            normalize_message_text(item.display_text or item.reason)
            for item in executions
            if normalize_message_text(item.display_text or item.reason)
        ]
        if messages:
            return "\n".join(dict.fromkeys(messages))
        return _fallback_final_reply(executions)

    for result in tool_results:
        output = result.output if isinstance(result.output, dict) else {}
        messages_sent = output.get("messages_sent")
        if isinstance(messages_sent, list):
            message = normalize_message_text(
                "\n".join(str(item or "") for item in messages_sent if item)
            )
            if message:
                return message
        error = normalize_message_text(str(output.get("error", "") or ""))
        if error:
            return error
    return "工具调用没有成功执行，请换个说法再试。"


def _build_system_prompt(
    base_prompt: str,
    dialogue_plan: ChatDialoguePlan | None,
) -> str:
    parts = [normalize_message_text(base_prompt)]
    strategy_prompt = build_chat_strategy_prompt(dialogue_plan)
    if strategy_prompt:
        parts.append(strategy_prompt)
    parts.append(_MAIN_REQUEST_RULES)
    return "\n\n".join(part for part in parts if part)


__all__ = [
    "MainRequestOutput",
    "MainRequestResult",
    "MainRequestTimelineItem",
    "run_chatinter_main_request",
]
