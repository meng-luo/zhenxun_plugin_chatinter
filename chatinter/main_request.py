"""Unified ChatInter main request runner.

This module prepares the turn prompt and command tools, then delegates the
model/tool loop to ``agent_runtime``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from inspect import isawaitable
import re
import time
from typing import Any, cast
import uuid

from zhenxun.services import logger
from zhenxun.services.llm import LLMMessage
from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types.models import ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .agent_complexity import route_agent_complexity
from .agent_runtime import AgentRuntime
from .agent_state import AgentRunState, AgentRuntimeResult, AgentRuntimeTimelineItem
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
from .provider_capability import ProviderCapabilityAdapter
from .route_text import (
    is_usage_question,
    normalize_message_text,
)
from .soft_tool_policy import (
    filter_soft_candidates,
    is_high_reliability_candidate,
    request_strength_for_candidate,
    should_catalog_only_candidate,
    sort_exposure_candidates,
)
from .tool_intent_gate import ToolIntentGate, ToolIntentGateResult
from .tool_retriever import CommandToolRetriever
from .turn_runtime import TurnBudgetController

_MAIN_STAGE = "main_request"
_TOOL_INTENT_GATE_STAGE = "tool_intent_gate"
_NEGATIVE_TOOL_MARKERS = (
    "不是在让你",
    "不是让你",
    "不是叫你",
    "不是要你",
    "不要",
    "别",
    "不用",
    "无需",
    "不必",
    "不想",
    "不需要",
)
_NEGATIVE_TOOL_BRIDGE_TERMS = (
    "真的",
    "实际",
    "真正",
    "真",
    "直接",
    "去",
    "来",
    "再",
    "帮我",
    "给我",
    "执行",
    "调用",
    "使用",
    "运行",
    "触发",
    "进行",
)
_GENERIC_TOOL_ACTION_TERMS = (
    "执行",
    "调用",
    "使用",
    "运行",
    "触发",
    "操作",
    "命令",
    "工具",
    "插件",
)
_DISCUSSION_ONLY_CONSTRAINTS = (
    "只是讨论",
    "只是在讨论",
    "我在讨论",
    "只是聊",
    "只想聊",
    "只聊",
    "只是提到",
    "只是说说",
    "不是命令",
)
_DISCUSSION_INTENT_MARKERS = (
    "聊聊",
    "讨论",
    "为什么",
    "怎么看",
    "会不会",
    "是否应该",
    "应不应该",
    "边界",
    "机制",
    "原理",
    "原因",
    "隐喻",
    "表达",
    "体验",
    "兜底",
    "误判",
    "不应该触发",
)
_DISCUSSION_META_TERMS = (
    "插件",
    "命令",
    "工具",
    "触发",
    "误判",
    "边界",
    "api",
    "产品",
    "机制",
    "词",
    "梗",
    "表达",
    "召回率",
)
_STRONG_EXECUTION_REQUEST_TERMS = (
    "帮我查",
    "查询",
    "搜一下",
    "搜索",
    "生成",
    "制作",
    "做个",
    "画",
    "翻译成",
    "翻译为",
    "执行",
    "调用",
    "运行",
    "发一个",
    "发一下",
    "来一个",
    "来一张",
    "抽一",
    "签个到",
    "签到",
)
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


@dataclass(frozen=True)
class ToolObligationDecision:
    obligation: str
    reason: str
    required_tool_names: tuple[str, ...] = ()
    gate_result: ToolIntentGateResult | None = None


@dataclass(frozen=True)
class CandidateObligationEvaluation:
    candidate: Any
    score: float
    request_strength: Any
    capability_factor: float
    recall_factor: float
    reliability_factor: float
    schema_factor: float
    requires_real_tool: bool
    real_output_factor: float
    reason: str


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
            messages=messages,
            route_executor=route_executor,
            report=report,
            enable_plugin_tools=enable_plugin_tools,
            initial_command_exposure=initial_command_exposure,
            enable_agent_tools=enable_agent_tools,
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
    messages: list[LLMMessage],
    route_executor: ExecuteNativeRoute,
    report: NativeRouteReport,
    enable_plugin_tools: bool,
    initial_command_exposure: bool,
    enable_agent_tools: bool,
) -> MainRequestResult:
    model_name = get_model_name()
    provider_adapter = ProviderCapabilityAdapter.for_model(model_name)
    enable_plugin_tools = enable_plugin_tools and provider_adapter.profile.supports_tools
    enable_agent_tools = enable_agent_tools and provider_adapter.profile.supports_tools
    retriever = CommandToolRetriever(
        knowledge_base,
        session_id=session_key,
        tools=cast(Any, command_tools),
    )
    capability_registry = retriever.registry
    if enable_agent_tools:
        capability_registry.register_available_superuser_tools()
        mcp_status = await capability_registry.register_available_mcp_tools(
            provider_adapter=provider_adapter,
        )
    else:
        mcp_status = None
    base_tool_count = capability_registry.executable_tool_count(
        kind="superuser_tool",
    ) + capability_registry.executable_tool_count(
        kind="runtime_tool",
    ) + (1 if enable_plugin_tools else 0)
    command_tool_capacity = provider_adapter.command_tool_capacity(
        reserved_tools=base_tool_count,
    )
    if enable_plugin_tools and command_tool_capacity <= 0:
        enable_plugin_tools = False
        base_tool_count = capability_registry.executable_tool_count(
            kind="superuser_tool",
        )
        command_tool_capacity = provider_adapter.command_tool_capacity(
            reserved_tools=base_tool_count,
        )
    catalog_state = CommandCatalogState(
        retriever=retriever,
        max_command_tools=max(command_tool_capacity, 1),
    )

    report.note_candidate_policy(
        reason="main_request_catalog_retrieval"
        if enable_plugin_tools
        else "plugin_tools_disabled",
        limit=retriever.total_commands if enable_plugin_tools else 0,
    )
    if enable_plugin_tools:
        report.candidate_total = max(report.candidate_total, retriever.total_commands)
    report.note_tool_pool(1 if enable_plugin_tools else 0)

    tool_map: dict[str, ToolExecutable] = {}
    if enable_plugin_tools:
        catalog_tool = CommandCatalogTool(catalog_state)
        capability_registry.register_catalog_tool(
            tool_name=COMMAND_CATALOG_TOOL_NAME,
            executable=cast(ToolExecutable, catalog_tool),
        )
    tool_map = capability_registry.executable_tool_map()
    if tool_map:
        report.note_tool_pool(len(tool_map))
    command_context = NativeCommandExecutionContext(
        candidates=[],
        has_reply=has_reply,
        report=report,
        route_executor=route_executor,
        message_text=message_text,
    )
    if enable_plugin_tools and initial_command_exposure:
        initial_result = retriever.initial_command_exposure(
            message_text,
            max_total=command_tool_capacity,
        )
        catalog_state.inject(list(initial_result.candidates))
        command_context.candidates = catalog_state.candidates
        tool_map = capability_registry.executable_tool_map()
        report.note_prompt_exposure(catalog_state.candidates)
        report.tool_candidates = max(
            report.tool_candidates,
            catalog_state.injected_count,
        )
        report.note_tool_pool(len(tool_map))
        report.note_candidate_policy(
            reason="initial_grouped_command_exposure",
            limit=catalog_state.injected_count,
        )
    trace_id = uuid.uuid4().hex[:12]
    run_id = trace_id
    obligation_decision = await _resolve_tool_obligation(
        message_text=message_text,
        enable_plugin_tools=enable_plugin_tools,
        enable_agent_tools=enable_agent_tools,
        candidates=catalog_state.candidates,
        tool_map=tool_map,
        trace_id=trace_id,
        model_name=model_name,
        generation_config=build_reasoning_generation_config(),
        timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
        budget_controller=budget_controller,
    )
    if enable_plugin_tools:
        _apply_tool_exposure_policy(
            message_text=message_text,
            decision=obligation_decision,
            catalog_state=catalog_state,
            command_context=command_context,
            capability_registry=capability_registry,
            report=report,
            command_tool_capacity=command_tool_capacity,
            provider_adapter=provider_adapter,
        )
        tool_map = capability_registry.executable_tool_map()
    complexity_decision = route_agent_complexity(
        message_text=message_text,
        tool_map=tool_map,
        enable_agent_tools=enable_agent_tools,
    )
    run_context = RunContext(
        session_id=session_key,
        extra={
            "native_command_context": command_context,
            "command_catalog_state": catalog_state,
            "capability_registry": capability_registry,
            "provider_capability": provider_adapter.profile.to_metadata(),
            "mcp_status": mcp_status,
            "actor_user_id": session_key or "",
            "agent_mode": "superuser_agent" if enable_agent_tools else "chatinter",
            "enable_agent_tools": enable_agent_tools,
            "agent_complexity": complexity_decision.to_metadata(),
        },
    )
    state = AgentRunState.create(
        trace_id=trace_id,
        run_id=run_id,
        session_key=session_key,
        messages=messages,
        tool_map=tool_map,
        current_message=message_text,
        max_steps=8 if enable_agent_tools else 5,
        budget_controller=budget_controller,
        tool_obligation=obligation_decision.obligation,
        tool_obligation_reason=obligation_decision.reason,
        required_tool_names=obligation_decision.required_tool_names,
        agent_complexity_mode=complexity_decision.mode,
        agent_complexity_reason=complexity_decision.reason,
    )
    state.append_timeline(
        role="system",
        kind="provider_capability",
        metadata=provider_adapter.profile.to_metadata(),
    )
    if mcp_status is not None:
        state.append_timeline(
            role="system",
            kind="mcp_runtime",
            metadata=mcp_status,
        )
    state.append_timeline(
        role="system",
        kind="agent_complexity",
        metadata=complexity_decision.to_metadata(),
    )
    state.append_timeline(
        role="system",
        kind="tool_intent_gate",
        metadata=_tool_obligation_metadata(obligation_decision),
    )
    runtime = AgentRuntime(
        state=state,
        run_context=run_context,
        message_text=message_text,
        model_name=model_name,
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
    return _result_from_agent_runtime(
        report=report,
        executions=command_context.executions,
        agent_result=agent_result,
        timeline=timeline,
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


def _result_from_agent_runtime(
    *,
    report: NativeRouteReport,
    executions: list[NativeToolExecutionResult],
    agent_result: AgentRuntimeResult,
    timeline: list[MainRequestTimelineItem],
) -> MainRequestResult:
    stop_reason = agent_result.stop_reason
    reason = f"main_request:{stop_reason}"
    if report.final_reason == "init":
        first_route = _first_route(executions)
        report.finalize(
            reason=reason,
            stage=first_route.stage if first_route is not None else _MAIN_STAGE,
            plugin_name=first_route.decision.plugin_name
            if first_route is not None
            else None,
            plugin_module=first_route.decision.plugin_module
            if first_route is not None
            else None,
            command=first_route.decision.command if first_route is not None else None,
        )
    command_tool_results = [
        result
        for result in agent_result.tool_results
        if not _is_catalog_tool_result(result)
    ]
    final_text = normalize_message_text(agent_result.final_text)
    should_send = bool(final_text)
    memory_text = _timeline_memory_text(timeline, fallback=final_text)
    handled_by_tools = bool(executions or command_tool_results)
    return MainRequestResult(
        decision=NativeRouteDecision(
            action="chat",
            confidence=0.9 if handled_by_tools else 0.84,
            reason=reason,
        ),
        route_result=_first_route(executions),
        report=report,
        executions=tuple(executions),
        tool_results=tuple(command_tool_results),
        timeline=tuple(timeline),
        output=MainRequestOutput(
            final_text=final_text,
            memory_text=memory_text,
            should_send=should_send,
            outcome="tool_completed" if handled_by_tools else "chat_completed",
            feedback_kind="tool_completed" if handled_by_tools else "chat_completed",
            record_chat_feedback=not handled_by_tools,
            observation_reason="route_success"
            if any(item.success for item in executions)
            else "reroute_failed"
            if handled_by_tools
            else "chat_completed",
        ),
    )


def _is_catalog_tool_result(result: ToolResult) -> bool:
    output = result.output if isinstance(result.output, dict) else {}
    return output.get("status") in {
        "retrieved",
        "capability_candidates_retrieved",
    }


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
    final_timeline = _with_final_timeline(
        result.timeline,
        final_text=final_text,
        should_send=True,
    )
    memory_text = normalize_message_text(output.memory_text) or _timeline_memory_text(
        list(final_timeline),
        fallback=final_text,
    )
    return replace(
        result,
        timeline=final_timeline,
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


def _timeline_memory_text(
    timeline: list[MainRequestTimelineItem] | tuple[MainRequestTimelineItem, ...],
    *,
    fallback: str = "",
) -> str:
    lines: list[str] = []
    for item in timeline:
        text = _timeline_item_summary(item)
        if text:
            lines.append(text)
    if fallback:
        lines.append(normalize_message_text(f"assistant: {fallback}"))
    return "\n".join(dict.fromkeys(line for line in lines if line))[:4000]


def _timeline_item_summary(item: MainRequestTimelineItem) -> str:
    role = normalize_message_text(item.role)
    kind = normalize_message_text(item.kind)
    prefix = f"{role}/{kind}".strip("/")
    if item.tool_name:
        prefix = f"{prefix}:{normalize_message_text(item.tool_name)}"
    content = normalize_message_text(item.content)
    if not content:
        output = item.metadata.get("output") if isinstance(item.metadata, dict) else None
        content = _compact_output_summary(output)
    if not content:
        arguments = (
            item.metadata.get("arguments") if isinstance(item.metadata, dict) else None
        )
        content = _compact_output_summary(arguments)
    if not content:
        return ""
    return f"{prefix}: {content}"[:800]


def _compact_output_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return normalize_message_text(str(value or ""))[:500]
    parts: list[str] = []
    for key in (
        "status",
        "ok",
        "command_id",
        "rendered_command",
        "matched_plugin",
        "task_text",
        "error",
        "remaining_task_hint",
    ):
        item = value.get(key)
        if item not in ("", [], {}, None):
            parts.append(f"{key}={normalize_message_text(str(item))}")
    messages = value.get("messages_sent")
    if isinstance(messages, list) and messages:
        parts.append(
            "messages_sent="
            + " | ".join(
                normalize_message_text(str(message or ""))
                for message in messages[:3]
                if normalize_message_text(str(message or ""))
            )
        )
    artifacts = value.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        summaries = [
            normalize_message_text(str(item.get("summary", "") or ""))
            for item in artifacts[:3]
            if isinstance(item, dict)
            and normalize_message_text(str(item.get("summary", "") or ""))
        ]
        if summaries:
            parts.append("artifacts=" + " | ".join(summaries))
    return "；".join(parts)[:500]


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


async def _resolve_tool_obligation(
    *,
    message_text: str,
    enable_plugin_tools: bool,
    enable_agent_tools: bool,
    candidates: list[Any],
    tool_map: dict[str, ToolExecutable],
    trace_id: str,
    model_name: str | None,
    generation_config: Any,
    timeout: float,
    budget_controller: TurnBudgetController | None,
) -> ToolObligationDecision:
    if not tool_map:
        return ToolObligationDecision(obligation="none", reason="no_tools")
    if enable_agent_tools:
        return ToolObligationDecision(
            obligation="auto",
            reason="superuser_agent_tools_available",
        )
    if not enable_plugin_tools:
        return ToolObligationDecision(
            obligation="none",
            reason="plugin_tools_disabled",
        )
    command_tools = _command_tool_names(tool_map)
    if not command_tools:
        return ToolObligationDecision(obligation="auto", reason="catalog_only")
    if not candidates:
        return ToolObligationDecision(
            obligation="none",
            reason="no_command_candidates",
        )
    negative_reason = _negative_tool_request_reason(
        message_text,
        candidates=candidates,
    )
    if negative_reason:
        return ToolObligationDecision(
            obligation="none",
            reason=f"negative_tool_request:{negative_reason}",
        )

    cheap_decision = _cheap_tool_obligation_decision(
        message_text=message_text,
        candidates=candidates,
        tool_map=tool_map,
    )
    if cheap_decision is not None:
        return cheap_decision

    if budget_controller is not None and not budget_controller.allow_classifier(
        _TOOL_INTENT_GATE_STAGE
    ):
        return ToolObligationDecision(
            obligation="auto",
            reason="tool_intent_gate_budget_exhausted",
        )

    started = time.perf_counter()
    gate = ToolIntentGate(
        trace_id=trace_id,
        model_name=model_name,
        generation_config=generation_config,
        timeout=timeout,
    )
    result = await gate.judge(
        message_text=message_text,
        candidates=candidates,
        command_tool_count=len(command_tools),
    )
    if budget_controller is not None:
        budget_controller.record_classifier(
            _TOOL_INTENT_GATE_STAGE,
            time.perf_counter() - started,
        )

    required_tool_names: tuple[str, ...] = ()
    if result.obligation == "none":
        obligation = "none"
    elif result.obligation == "required":
        command_id_filter = {
            normalize_message_text(str(command_id or ""))
            for command_id in (
                result.required_command_ids or result.allowed_command_ids
            )
            if normalize_message_text(str(command_id or ""))
        }
        if not command_id_filter:
            allowed_soft_candidates = filter_soft_candidates(message_text, candidates)
            command_id_filter = {
                normalize_message_text(
                    str(getattr(getattr(candidate, "schema", None), "command_id", ""))
                )
                for candidate in allowed_soft_candidates
                if normalize_message_text(
                    str(getattr(getattr(candidate, "schema", None), "command_id", ""))
                )
            }
        obligation = "required"
        required_tool_names = _tool_names_for_command_ids(
            tool_map,
            tuple(command_id_filter),
        )
        if not required_tool_names:
            obligation = "auto"
    else:
        obligation = "auto"

    return ToolObligationDecision(
        obligation=obligation,
        reason=_gate_obligation_reason(result),
        required_tool_names=required_tool_names,
        gate_result=result,
    )


def _negative_tool_request_reason(
    message_text: str,
    *,
    candidates: list[Any],
) -> str:
    """Return a generic reason when the user explicitly says not to execute tools."""

    normalized = normalize_message_text(message_text)
    if not normalized:
        return ""
    if _has_discussion_only_constraint(normalized):
        return "discussion_only_constraint"
    if _has_negative_tool_action_phrase(
        normalized,
        capability_terms=_capability_action_terms(candidates),
    ):
        return "negative_action_phrase"
    return ""


def _has_discussion_only_constraint(text: str) -> bool:
    normalized = normalize_message_text(text)
    if any(term in normalized for term in _DISCUSSION_ONLY_CONSTRAINTS):
        return True
    if _has_strong_execution_request(normalized):
        return False
    return (
        any(marker in normalized for marker in _DISCUSSION_INTENT_MARKERS)
        and any(term in normalized for term in _DISCUSSION_META_TERMS)
    )


def _has_strong_execution_request(text: str) -> bool:
    normalized = normalize_message_text(text)
    return any(term in normalized for term in _STRONG_EXECUTION_REQUEST_TERMS)


def _has_negative_tool_action_phrase(
    text: str,
    *,
    capability_terms: tuple[str, ...],
) -> bool:
    normalized = normalize_message_text(text)
    if not normalized:
        return False
    for marker in _NEGATIVE_TOOL_MARKERS:
        start = 0
        while True:
            index = normalized.find(marker, start)
            if index < 0:
                break
            tail = normalized[index + len(marker) :]
            if _negative_tail_targets_tool_action(
                tail,
                capability_terms=capability_terms,
            ):
                return True
            start = index + len(marker)
    return False


def _negative_tail_targets_tool_action(
    tail: str,
    *,
    capability_terms: tuple[str, ...],
) -> bool:
    compact_tail = re.sub(r"[\s，,。.!！？?；;：:\"'“”‘’、]+", "", tail or "")
    if not compact_tail:
        return False
    compact_tail = _strip_negative_bridge_terms(compact_tail)
    if not compact_tail:
        return False
    search_area = compact_tail[:16]
    return any(term in search_area for term in _GENERIC_TOOL_ACTION_TERMS) or any(
        term in search_area for term in capability_terms
    )


def _strip_negative_bridge_terms(text: str) -> str:
    stripped = text
    changed = True
    while changed and stripped:
        changed = False
        for term in _NEGATIVE_TOOL_BRIDGE_TERMS:
            if term and stripped.startswith(term):
                stripped = stripped[len(term) :]
                changed = True
                break
    return stripped


def _capability_action_terms(candidates: list[Any]) -> tuple[str, ...]:
    terms: list[str] = []
    for candidate in candidates:
        schema = getattr(candidate, "schema", None)
        snapshot = getattr(candidate, "tool", None)
        raw_values: list[Any] = [
            getattr(schema, "head", ""),
            *list(getattr(schema, "aliases", []) or []),
            *list(getattr(schema, "retrieval_phrases", []) or []),
            *list(getattr(snapshot, "task_verbs", []) or []),
        ]
        for value in raw_values:
            term = _compact_capability_term(value)
            if term and term not in terms:
                terms.append(term)
    return tuple(terms[:64])


def _compact_capability_term(value: Any) -> str:
    text = normalize_message_text(str(value or ""))
    if not text:
        return ""
    compact = re.sub(r"[\s，,。.!！？?；;：:\"'“”‘’、]+", "", text)
    if len(compact) < 2 or len(compact) > 12:
        return ""
    return compact


def _cheap_tool_obligation_decision(
    *,
    message_text: str,
    candidates: list[Any],
    tool_map: dict[str, ToolExecutable],
) -> ToolObligationDecision | None:
    """Resolve obvious tool obligation without spending a classifier call."""

    if not candidates:
        return ToolObligationDecision(obligation="none", reason="cheap:no_candidates")

    actionable = [
        candidate for candidate in candidates if _candidate_is_actionable(candidate)
    ]
    if not actionable:
        return ToolObligationDecision(
            obligation="auto",
            reason="cheap:catalog_or_helper_only",
        )

    exact = [candidate for candidate in actionable if candidate.exact_protected]
    if exact:
        command_ids = _candidate_command_ids(exact)
        tool_names = _tool_names_for_command_ids(tool_map, tuple(command_ids))
        return ToolObligationDecision(
            obligation="required" if tool_names else "auto",
            reason="cheap:exact_command_candidate",
            required_tool_names=tool_names,
        )

    evaluations = [
        _candidate_obligation_evaluation(message_text, candidate)
        for candidate in actionable
    ]
    required_evaluations = [
        evaluation for evaluation in evaluations if _evaluation_requires_tool(evaluation)
    ]
    required = [evaluation.candidate for evaluation in required_evaluations]
    if required:
        ranked_required = _rank_required_candidates(required_evaluations)
        command_ids = _candidate_command_ids(
            [evaluation.candidate for evaluation in ranked_required[:8]]
        )
        tool_names = _tool_names_for_command_ids(tool_map, tuple(command_ids))
        best = ranked_required[0]
        return ToolObligationDecision(
            obligation="required" if tool_names else "auto",
            reason=f"cheap:structured_required:{best.reason}",
            required_tool_names=tool_names,
        )

    if any(_evaluation_allows_auto(evaluation) for evaluation in evaluations):
        best = max(evaluations, key=lambda item: item.score)
        return ToolObligationDecision(
            obligation="auto",
            reason=f"cheap:structured_auto:{best.reason}",
        )

    if _all_candidates_are_weak(actionable):
        return ToolObligationDecision(
            obligation="auto",
            reason="cheap:weak_or_explicit_only_candidates",
        )

    # Ambiguous but non-trivial: keep the LLM gate for the hard middle band.
    return None


def _rank_required_candidates(evaluations: list[Any]) -> list[Any]:
    return sorted(
        evaluations,
        key=lambda evaluation: (
            _candidate_has_constrained_schema(evaluation.candidate),
            float(getattr(evaluation, "score", 0.0) or 0.0),
            float(getattr(evaluation.candidate, "score", 0.0) or 0.0),
        ),
        reverse=True,
    )


def _candidate_has_constrained_schema(candidate: Any) -> bool:
    schema = getattr(candidate, "schema", None)
    if schema is None:
        return False
    if any(getattr(slot, "choices", None) for slot in getattr(schema, "slots", []) or []):
        return True
    shortcut_renders = getattr(schema, "shortcut_renders", None)
    if shortcut_renders:
        return True
    tool = getattr(candidate, "tool", None)
    meta = getattr(tool, "meta", None)
    if isinstance(meta, dict) and (
        meta.get("slot_choices") or meta.get("shortcut_renders")
    ):
        return True
    return False


def _candidate_command_ids(candidates: list[Any]) -> list[str]:
    result: list[str] = []
    for candidate in candidates:
        command_id = normalize_message_text(
            str(getattr(getattr(candidate, "schema", None), "command_id", "") or "")
        )
        if command_id and command_id not in result:
            result.append(command_id)
    return result


def _candidate_is_actionable(candidate: Any) -> bool:
    schema = getattr(candidate, "schema", None)
    role = normalize_message_text(str(getattr(schema, "command_role", "") or ""))
    return role not in {"catalog", "helper"}


def _candidate_is_concrete_or_external(candidate: Any) -> bool:
    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    output_mode = normalize_message_text(str(getattr(snapshot, "output_mode", "") or ""))
    side_effect = normalize_message_text(str(getattr(snapshot, "side_effect", "") or ""))
    if output_mode in {"image", "file", "action", "plugin_output"}:
        return True
    if side_effect in {"query", "send", "mutate"}:
        return True
    payload_policy = normalize_message_text(
        str(getattr(schema, "payload_policy", "") or "")
    )
    if payload_policy not in {"", "none"}:
        return True
    if getattr(schema, "slots", None):
        return True
    requires = dict(getattr(schema, "requires", {}) or {})
    return any(bool(requires.get(key)) for key in ("text", "image", "reply", "at"))


def _candidate_requires_real_result(candidate: Any) -> bool:
    snapshot = getattr(candidate, "tool", None)
    if snapshot is not None and hasattr(snapshot, "requires_real_result"):
        return bool(getattr(snapshot, "requires_real_result", True))
    return _candidate_is_concrete_or_external(candidate)


def _candidate_requires_real_tool(candidate: Any) -> bool:
    snapshot = getattr(candidate, "tool", None)
    if snapshot is not None and hasattr(snapshot, "requires_real_tool"):
        return bool(getattr(snapshot, "requires_real_tool", True))
    return _candidate_requires_real_result(candidate)


def _candidate_capability_factor(candidate: Any) -> float:
    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    factor = 1.0
    source = normalize_message_text(
        str(getattr(snapshot, "source_of_truth", "") or "unknown")
    )
    output_mode = normalize_message_text(str(getattr(snapshot, "output_mode", "") or ""))
    side_effect = normalize_message_text(str(getattr(snapshot, "side_effect", "") or ""))
    risk = normalize_message_text(
        str(
            getattr(snapshot, "risk", "")
            or getattr(snapshot, "risk_level", "")
            or "low"
        )
    )
    entity_scope = normalize_message_text(
        str(getattr(snapshot, "entity_scope", "") or "global")
    )
    policy = normalize_message_text(
        str(getattr(snapshot, "execution_policy", "") or "normal")
    )
    intent_types = {
        normalize_message_text(str(intent or "")).lower()
        for intent in list(getattr(snapshot, "intent_types", []) or [])
        if normalize_message_text(str(intent or ""))
    }

    if bool(getattr(snapshot, "requires_real_tool", False)):
        factor *= 1.35
    elif bool(getattr(snapshot, "requires_real_result", False)):
        factor *= 1.2
    if source in {"bot_state", "external_service", "local_state"}:
        factor *= 1.28
    elif source == "plugin_runtime":
        factor *= 1.15
    elif source == "model_knowledge":
        factor *= 0.72
    elif source in {"user_provided", "unknown"}:
        factor *= 0.9

    if output_mode in {"image", "file", "action"}:
        factor *= 1.28
    elif output_mode == "plugin_output":
        factor *= 1.15
    elif output_mode == "text":
        factor *= 0.96

    if side_effect in {"query", "send"}:
        factor *= 1.12
    elif side_effect == "mutate":
        factor *= 1.2

    if intent_types & {
        "query",
        "status",
        "generate",
        "media",
        "random",
        "transform",
        "play",
    }:
        factor *= 1.12
    if bool(getattr(snapshot, "generative", False)):
        factor *= 1.08
    if entity_scope in {"self_bot", "actor_user", "target_user", "group", "external"}:
        factor *= 1.08
    if risk == "medium":
        factor *= 0.96
    elif risk == "high":
        factor *= 0.9
    schema_quality = float(getattr(snapshot, "schema_quality", 0.5) or 0.5)
    reliability = float(getattr(snapshot, "reliability", 0.5) or 0.5)
    if schema_quality < 0.35:
        factor *= 0.82
    elif schema_quality >= 0.72:
        factor *= 1.08
    if reliability < 0.35:
        factor *= 0.84
    elif reliability >= 0.72:
        factor *= 1.08
    if policy == "explicit_only":
        factor *= 0.82
    elif policy == "confirmation_required":
        factor *= 0.86
    if bool(getattr(snapshot, "soft_tool", False)):
        factor *= 0.84
    payload_policy = normalize_message_text(
        str(getattr(schema, "payload_policy", "") or "")
    )
    if payload_policy not in {"", "none"}:
        factor *= 1.06
    if getattr(schema, "slots", None):
        factor *= 1.05
    return max(0.25, min(factor, 2.8))


def _candidate_obligation_evaluation(
    message_text: str,
    candidate: Any,
) -> CandidateObligationEvaluation:
    request_strength = request_strength_for_candidate(message_text, candidate)
    capability_factor = _candidate_capability_factor(candidate)
    recall_factor = _candidate_recall_factor(candidate)
    reliability_factor = _candidate_reliability_factor(candidate)
    schema_factor = _candidate_schema_factor(candidate)
    request_factor = _request_strength_factor(request_strength)
    requires_real_tool = _candidate_requires_real_tool(candidate)
    real_output_factor = _candidate_real_output_factor(candidate)
    score = request_factor * capability_factor * recall_factor
    score *= reliability_factor * schema_factor * real_output_factor
    snapshot = getattr(candidate, "tool", None)
    policy = normalize_message_text(
        str(getattr(snapshot, "execution_policy", "") or "normal")
    )
    if policy == "explicit_only" and not request_strength.explicit:
        score *= 0.48
    elif policy == "strong_intent" and request_strength.score < 2.7:
        score *= 0.68
    if bool(getattr(snapshot, "soft_tool", False)) and not request_strength.explicit:
        score *= 0.58
    if not requires_real_tool:
        score *= 0.62
    reason = (
        f"request={request_strength.score:.2f};"
        f"capability_factor={capability_factor:.2f};"
        f"recall_factor={recall_factor:.2f};"
        f"reliability_factor={reliability_factor:.2f};"
        f"schema_factor={schema_factor:.2f};"
        f"real_output_factor={real_output_factor:.2f};"
        f"score={score:.2f};"
        f"policy={policy or 'normal'};"
        f"{request_strength.reason}"
    )
    return CandidateObligationEvaluation(
        candidate=candidate,
        score=round(score, 3),
        request_strength=request_strength,
        capability_factor=capability_factor,
        recall_factor=recall_factor,
        reliability_factor=reliability_factor,
        schema_factor=schema_factor,
        requires_real_tool=requires_real_tool,
        real_output_factor=real_output_factor,
        reason=reason,
    )


def _evaluation_requires_tool(evaluation: CandidateObligationEvaluation) -> bool:
    if not evaluation.requires_real_tool:
        return False
    if not evaluation.request_strength.explicit:
        return False
    if evaluation.recall_factor < 0.66:
        return False
    if evaluation.reliability_factor < 0.52:
        return False
    if evaluation.schema_factor < 0.58:
        return False
    if evaluation.score >= 2.65:
        return True
    if (
        evaluation.request_strength.explicit
        and evaluation.real_output_factor >= 1.12
        and evaluation.score >= 2.05
        and evaluation.recall_factor >= 0.72
    ):
        return True
    if (
        evaluation.request_strength.explicit
        and "play" in set(getattr(evaluation.request_strength, "matched_intents", ()) or ())
        and evaluation.real_output_factor >= 1.12
        and evaluation.score >= 2.0
        and evaluation.recall_factor >= 0.45
    ):
        return True
    return (
        evaluation.request_strength.exact_command
        and evaluation.score >= 1.9
    )


def _evaluation_allows_auto(evaluation: CandidateObligationEvaluation) -> bool:
    if evaluation.recall_factor < 0.42:
        return False
    if evaluation.score >= 1.55:
        return True
    return (
        evaluation.request_strength.explicit
        and evaluation.requires_real_tool
        and evaluation.score >= 1.25
    )


def _request_strength_factor(request_strength: Any) -> float:
    score = float(getattr(request_strength, "score", 0.0) or 0.0)
    factor = 0.45 + min(max(score, 0.0), 6.0) / 3.0
    if bool(getattr(request_strength, "exact_command", False)):
        factor += 0.55
    elif bool(getattr(request_strength, "direct_mention", False)):
        factor += 0.25
    if bool(getattr(request_strength, "explicit", False)):
        factor += 0.25
    return max(0.25, min(factor, 3.2))


def _candidate_recall_factor(candidate: Any) -> float:
    confidence = _candidate_confidence_score(candidate)
    if bool(getattr(candidate, "exact_protected", False)):
        confidence = max(confidence, 145.0)
    return max(0.25, min(confidence / 110.0, 1.8))


def _candidate_reliability_factor(candidate: Any) -> float:
    features = getattr(candidate, "features", None)
    snapshot = getattr(candidate, "tool", None)
    feedback_reliability = float(
        getattr(features, "reliability_score", 0.0) or 0.0
    )
    false_trigger = float(getattr(features, "false_trigger_score", 0.0) or 0.0)
    param_failure = float(getattr(features, "param_failure_score", 0.0) or 0.0)
    latency = float(getattr(features, "latency_score", 0.0) or 0.0)
    prior = float(getattr(snapshot, "reliability", 0.5) or 0.5)
    factor = 0.72 + prior * 0.56
    factor += max(min(feedback_reliability, 18.0), -24.0) / 90.0
    factor += max(min(false_trigger, 0.0), -24.0) / 70.0
    factor += max(min(param_failure, 0.0), -18.0) / 80.0
    factor += max(min(latency, 6.0), -8.0) / 120.0
    if is_high_reliability_candidate(candidate):
        factor += 0.12
    return max(0.35, min(factor, 1.45))


def _candidate_schema_factor(candidate: Any) -> float:
    features = getattr(candidate, "features", None)
    snapshot = getattr(candidate, "tool", None)
    schema_score = float(getattr(features, "schema_score", 0.0) or 0.0)
    quality = float(getattr(snapshot, "schema_quality", 0.5) or 0.5)
    factor = 0.68 + quality * 0.52 + min(max(schema_score, 0.0), 22.5) / 120.0
    return max(0.45, min(factor, 1.4))


def _candidate_real_output_factor(candidate: Any) -> float:
    snapshot = getattr(candidate, "tool", None)
    if snapshot is None:
        return 1.0
    output_mode = normalize_message_text(str(getattr(snapshot, "output_mode", "") or ""))
    side_effect = normalize_message_text(str(getattr(snapshot, "side_effect", "") or ""))
    source = normalize_message_text(
        str(getattr(snapshot, "source_of_truth", "") or "")
    )
    intent_types = {
        normalize_message_text(str(intent or "")).lower()
        for intent in list(getattr(snapshot, "intent_types", []) or [])
        if normalize_message_text(str(intent or ""))
    }
    factor = 1.0
    if output_mode in {"image", "file", "action"}:
        factor *= 1.18
    elif output_mode == "plugin_output":
        factor *= 1.08
    if side_effect in {"query", "send", "mutate"}:
        factor *= 1.08
    if bool(intent_types & {"generate", "media", "transform", "random", "query", "play"}):
        factor *= 1.06
    if source in {"bot_state", "external_service", "local_state", "plugin_runtime"}:
        factor *= 1.06
    if bool(getattr(snapshot, "requires_real_tool", False)):
        factor *= 1.05
    if bool(getattr(snapshot, "generative", False)):
        factor *= 1.04
    return max(0.85, min(factor, 1.35))


def _candidate_confidence_score(candidate: Any) -> float:
    features = getattr(candidate, "features", None)
    exact = float(getattr(features, "exact_score", 0.0) or 0.0)
    lexical = float(getattr(features, "lexical_score", 0.0) or 0.0)
    semantic = float(getattr(features, "semantic_score", 0.0) or 0.0)
    context = float(getattr(features, "context_score", 0.0) or 0.0)
    schema = float(getattr(features, "schema_score", 0.0) or 0.0)
    score = float(getattr(candidate, "score", 0.0) or 0.0)
    return max(score, exact + lexical + semantic + context + schema)


def _all_candidates_are_weak(candidates: list[Any]) -> bool:
    if not candidates:
        return True
    best = max(_candidate_confidence_score(candidate) for candidate in candidates)
    if best >= 90.0:
        return False
    return all(not _candidate_is_concrete_or_external(candidate) for candidate in candidates)


def _command_tool_names(tool_map: dict[str, ToolExecutable]) -> tuple[str, ...]:
    names: list[str] = []
    for name, tool in tool_map.items():
        binding = getattr(tool, "binding", None)
        command_id = normalize_message_text(str(getattr(binding, "command_id", "")))
        if command_id:
            names.append(normalize_message_text(name))
    return tuple(name for name in names if name)


def _apply_tool_exposure_policy(
    *,
    message_text: str,
    decision: ToolObligationDecision,
    catalog_state: CommandCatalogState,
    command_context: NativeCommandExecutionContext,
    capability_registry: Any,
    report: NativeRouteReport,
    command_tool_capacity: int,
    provider_adapter: ProviderCapabilityAdapter,
) -> None:
    """Align first-turn executable tools with the structured gate."""

    current_candidates = list(catalog_state.candidates)
    if decision.obligation == "none":
        catalog_state.replace([])
        command_context.candidates = []
        report.note_tool_pool(capability_registry.executable_tool_count())
        report.note_candidate_policy(reason="tool_gate_no_command_exposure", limit=0)
        return
    if not current_candidates:
        return

    selected_ids = _command_ids_for_tool_names(
        capability_registry.executable_tool_map(),
        decision.required_tool_names,
    )
    if decision.gate_result is not None:
        selected_ids.update(
            normalize_message_text(str(command_id or ""))
            for command_id in (
                decision.gate_result.required_command_ids
                + decision.gate_result.allowed_command_ids
            )
            if normalize_message_text(str(command_id or ""))
        )

    current_candidates = _filter_catalog_only_candidates(
        current_candidates,
        message_text=message_text,
        selected_command_ids=selected_ids,
    )
    if not current_candidates:
        catalog_state.replace([])
        command_context.candidates = []
        report.note_tool_pool(capability_registry.executable_tool_count())
        report.note_candidate_policy(reason="feedback_catalog_only_exposure", limit=0)
        return

    if decision.obligation == "required":
        filtered = _required_exposure_candidates(
            message_text=message_text,
            candidates=current_candidates,
            selected_command_ids=selected_ids,
        )
    else:
        filtered = filter_soft_candidates(
            message_text,
            current_candidates,
            selected_command_ids=selected_ids,
        )

    exposure_cap = _command_exposure_cap(
        decision=decision,
        command_tool_capacity=command_tool_capacity,
        provider_adapter=provider_adapter,
    )
    filtered = sort_exposure_candidates(
        message_text,
        filtered,
        selected_command_ids=selected_ids,
    )[:exposure_cap]
    catalog_state.replace(filtered)
    command_context.candidates = catalog_state.candidates
    report.note_prompt_exposure(catalog_state.candidates)
    report.tool_candidates = max(report.tool_candidates, catalog_state.injected_count)
    report.note_tool_pool(capability_registry.executable_tool_count())
    report.note_candidate_policy(
        reason=f"tool_gate_{decision.obligation}_command_exposure",
        limit=catalog_state.injected_count,
    )


def _command_exposure_cap(
    *,
    decision: ToolObligationDecision,
    command_tool_capacity: int,
    provider_adapter: ProviderCapabilityAdapter,
) -> int:
    return provider_adapter.command_exposure_cap(
        obligation=decision.obligation,
        required_tool_count=len(decision.required_tool_names),
        command_tool_capacity=command_tool_capacity,
    )


def _filter_catalog_only_candidates(
    candidates: list[Any],
    *,
    message_text: str,
    selected_command_ids: set[str],
) -> list[Any]:
    result: list[Any] = []
    for candidate in candidates:
        if should_catalog_only_candidate(
            candidate,
            message_text=message_text,
            selected_command_ids=selected_command_ids,
        ):
            continue
        result.append(candidate)
    if result:
        return result
    return [
        candidate
        for candidate in candidates
        if normalize_message_text(
            str(getattr(getattr(candidate, "schema", None), "command_id", "") or "")
        )
        in selected_command_ids
        or bool(getattr(candidate, "exact_protected", False))
        or is_high_reliability_candidate(candidate)
    ]


def _required_exposure_candidates(
    *,
    message_text: str,
    candidates: list[Any],
    selected_command_ids: set[str],
) -> list[Any]:
    if selected_command_ids:
        selected = [
            candidate
            for candidate in candidates
            if normalize_message_text(
                str(getattr(getattr(candidate, "schema", None), "command_id", "") or "")
            )
            in selected_command_ids
        ]
        if selected:
            return selected
    filtered = filter_soft_candidates(
        message_text,
        candidates,
        selected_command_ids=selected_command_ids,
    )
    if filtered:
        return filtered
    return [
        candidate
        for candidate in candidates
        if not bool(getattr(getattr(candidate, "tool", None), "soft_tool", False))
    ] or candidates


def _command_ids_for_tool_names(
    tool_map: dict[str, ToolExecutable],
    tool_names: tuple[str, ...],
) -> set[str]:
    wanted_names = {
        normalize_message_text(str(name or ""))
        for name in tool_names
        if normalize_message_text(str(name or ""))
    }
    result: set[str] = set()
    if not wanted_names:
        return result
    for name, tool in tool_map.items():
        if normalize_message_text(name) not in wanted_names:
            continue
        binding = getattr(tool, "binding", None)
        command_id = normalize_message_text(str(getattr(binding, "command_id", "")))
        if command_id:
            result.add(command_id)
    return result


def _tool_names_for_command_ids(
    tool_map: dict[str, ToolExecutable],
    command_ids: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    wanted = {
        normalize_message_text(str(command_id or ""))
        for command_id in command_ids
        if normalize_message_text(str(command_id or ""))
    }
    if not wanted:
        return ()
    names: list[str] = []
    for name, tool in tool_map.items():
        binding = getattr(tool, "binding", None)
        command_id = normalize_message_text(str(getattr(binding, "command_id", "")))
        if command_id and command_id in wanted:
            names.append(normalize_message_text(name))
    return tuple(name for name in names if name)


def _gate_obligation_reason(result: ToolIntentGateResult) -> str:
    return normalize_message_text(
        "tool_intent_gate:"
        f"{result.intent_type}:{result.obligation}:"
        f"request_strength={result.request_strength}:"
        f"mention_only={int(result.mention_only)}:"
        f"needs_real_execution={int(result.needs_real_execution)}:"
        f"{result.reason or 'no_reason'}"
    )


def _tool_obligation_metadata(
    decision: ToolObligationDecision,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tool_obligation": decision.obligation,
        "tool_obligation_reason": decision.reason,
        "required_tool_names": list(decision.required_tool_names),
    }
    result = decision.gate_result
    if result is not None:
        metadata.update(
            {
                "gate_intent_type": result.intent_type,
                "gate_request_strength": result.request_strength,
                "gate_mention_only": result.mention_only,
                "gate_obligation": result.obligation,
                "gate_reason": result.reason,
                "required_command_ids": list(result.required_command_ids),
                "allowed_command_ids": list(result.allowed_command_ids),
                "candidate_intent_types": list(result.candidate_intent_types),
                "needs_real_execution": result.needs_real_execution,
            }
        )
    return metadata


__all__ = [
    "MainRequestOutput",
    "MainRequestResult",
    "MainRequestTimelineItem",
    "run_chatinter_main_request",
]
