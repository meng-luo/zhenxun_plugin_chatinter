"""AgentRuntime runner shared by scenario-specific agents."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm import LLMMessage
from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types.protocols import ToolExecutable

from ..agent_complexity import resolve_agent_run_budget
from ..agent_runtime import AgentRuntime
from ..agent_state import AgentRunState
from ..command_catalog_tool import CommandCatalogState
from ..config import build_reasoning_generation_config, get_config_value
from ..main_request_models import MainRequestResult, ToolObligationDecision
from ..native_executor import NativeCommandExecutionContext
from ..native_route import NativeRouteReport
from ..plugin_command_support import _tool_obligation_metadata
from ..provider_capability import ProviderCapabilityAdapter
from ..runtime_result import _convert_runtime_timeline, _result_from_agent_runtime
from ..task_planner_lite import task_items_to_payload
from ..turn_runtime import TurnBudgetController


async def _run_legacy_agent_runtime(
    *,
    message_text: str,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    messages: list[LLMMessage],
    report: NativeRouteReport,
    agent_mode: str,
    progress_hook: Any | None,
    provider_adapter: ProviderCapabilityAdapter,
    capability_registry: Any,
    command_context: NativeCommandExecutionContext,
    catalog_state: CommandCatalogState | None,
    tool_map: dict[str, ToolExecutable],
    mcp_status: Any | None,
    task_items: tuple[Any, ...],
    task_router_result: Any | None,
    trace_id: str,
    run_id: str,
    model_name: str | None,
    complexity_decision: Any,
    started: float,
    obligation_decision: ToolObligationDecision,
) -> MainRequestResult:
    del started
    enable_agent_tools = agent_mode in {"superuser_agent", "superuser_subagent"}
    enable_plugin_tools = agent_mode == "plugin_command"
    run_budget = resolve_agent_run_budget(
        mode=complexity_decision.mode,
        enable_agent_tools=enable_agent_tools,
        enable_plugin_tools=enable_plugin_tools,
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
            "agent_mode": agent_mode,
            "enable_agent_tools": enable_agent_tools,
            "trace_id": trace_id,
            "run_id": run_id,
            "agent_complexity": complexity_decision.to_metadata(),
            "task_planner_lite": task_items_to_payload(task_items)
            if task_items
            else None,
            "task_router": task_router_result.to_payload()
            if task_router_result is not None
            else None,
        },
    )
    state = AgentRunState.create(
        trace_id=trace_id,
        run_id=run_id,
        session_key=session_key,
        messages=messages,
        tool_map=tool_map,
        current_message=message_text,
        max_steps=run_budget.max_steps,
        max_total_tokens=run_budget.max_total_tokens,
        max_step_refunds=run_budget.max_step_refunds,
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
    if task_items:
        state.append_timeline(
            role="system",
            kind="task_planner_lite",
            metadata=task_items_to_payload(task_items),
        )
    if task_router_result is not None:
        state.append_timeline(
            role="system",
            kind="task_router",
            metadata=task_router_result.to_payload(),
        )
    state.append_timeline(
        role="system",
        kind="agent_run_budget",
        metadata=run_budget.to_metadata(),
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
        progress_hook=progress_hook if enable_agent_tools else None,
    )
    agent_result = await runtime.run()
    timeline = _convert_runtime_timeline(agent_result.timeline)
    if catalog_state is not None and catalog_state.candidates:
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


__all__ = ["_run_legacy_agent_runtime"]
