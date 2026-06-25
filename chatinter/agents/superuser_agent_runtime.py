"""Superuser AgentRuntime entry.

This module owns the private superuser path.  It intentionally does not build
plugin command retrievers or catalog state; group plugin routing lives in
PluginCommandAgent.
"""

from __future__ import annotations

import time
from typing import Any
import uuid

from zhenxun.services.llm import LLMMessage

from ..agent_complexity import route_agent_complexity
from ..capability_registry import CapabilityRegistry
from ..config import get_model_name
from ..main_request_models import MainRequestResult, ToolObligationDecision
from ..native_executor import (
    NativeCommandExecutionContext,
    NativeToolExecutionResult,
)
from ..native_route import NativeRouteReport
from ..provider_capability import ProviderCapabilityAdapter
from ..route_text import normalize_message_text
from ..superuser_agent.registry import (
    SuperuserToolBundle,
    build_superuser_agent_tool_bundle,
)
from ..superuser_agent.tool_preset import (
    get_session_tool_preset,
    preset_allows_card,
)
from ..task_planner_lite import plan_task_items
from .runtime_runner import _run_legacy_agent_runtime

_DELEGATE_TASK_TOOL_NAME = "delegate_task"


async def _superuser_route_executor_not_available(
    *_args: Any,
    **_kwargs: Any,
) -> NativeToolExecutionResult:
    raise RuntimeError("superuser agent does not execute plugin command routes")


async def run_superuser_agent_runtime(
    *,
    message_text: str,
    session_key: str | None,
    budget_controller: Any | None,
    messages: list[LLMMessage],
    report: NativeRouteReport,
    progress_hook: Any | None,
) -> MainRequestResult:
    """Run the superuser AgentRuntime without plugin command catalog plumbing."""

    model_name = get_model_name()
    provider_adapter = ProviderCapabilityAdapter.for_model(model_name)
    capability_registry = CapabilityRegistry.empty(session_id=session_key)
    tool_preset = get_session_tool_preset(session_key)
    bundle = _filter_bundle_by_preset(
        build_superuser_agent_tool_bundle(message_text=message_text),
        preset=tool_preset,
    )
    capability_registry.register_superuser_tools(
        bundle.tools,
        cards=bundle.cards,
    )
    mcp_status = None
    tool_map = capability_registry.executable_tool_map()
    task_items = plan_task_items(message_text)
    complexity_decision = route_agent_complexity(
        message_text=message_text,
        tool_map=tool_map,
        enable_agent_tools=True,
        local_task_count=len(task_items),
    )
    if _should_hide_delegate_task(complexity_mode=complexity_decision.mode):
        capability_registry.remove_executable_tool(
            _DELEGATE_TASK_TOOL_NAME,
            kind="superuser_tool",
        )
        tool_map = capability_registry.executable_tool_map()
    tool_obligation = "auto"
    tool_obligation_reason = (
        "superuser_agent_tools_available"
        if tool_preset == "default"
        else f"superuser_agent_tool_preset:{tool_preset}"
    )
    required_tool_names: tuple[str, ...] = ()
    if complexity_decision.mode == "readonly_fast":
        readonly_names = _readonly_fast_tool_names(
            message_text=message_text,
            cards=bundle.cards,
            available_tool_names=tuple(tool_map),
        )
        required_tool_names = readonly_names or tuple(tool_map)
        if required_tool_names:
            tool_obligation = "required"
            tool_obligation_reason = "readonly_fast_requires_real_observation"
    trace_id = uuid.uuid4().hex[:12]
    command_context = NativeCommandExecutionContext(
        candidates=[],
        has_reply=False,
        report=report,
        route_executor=_superuser_route_executor_not_available,
        message_text=message_text,
    )
    return await _run_legacy_agent_runtime(
        message_text=message_text,
        session_key=session_key,
        budget_controller=budget_controller,
        messages=messages,
        report=report,
        agent_mode="superuser_agent",
        progress_hook=progress_hook,
        provider_adapter=provider_adapter,
        capability_registry=capability_registry,
        command_context=command_context,
        catalog_state=None,
        tool_map=tool_map,
        mcp_status=mcp_status,
        task_items=task_items,
        task_router_result=None,
        trace_id=trace_id,
        run_id=trace_id,
        model_name=model_name,
        complexity_decision=complexity_decision,
        started=time.perf_counter(),
        obligation_decision=ToolObligationDecision(
            obligation=tool_obligation,
            reason=tool_obligation_reason,
            required_tool_names=required_tool_names,
        ),
    )


def _should_hide_delegate_task(*, complexity_mode: str) -> bool:
    """Keep delegate_task for complex engineering turns only."""

    return complexity_mode != "complex_pev"


def _filter_bundle_by_preset(
    bundle: SuperuserToolBundle,
    *,
    preset: str,
) -> SuperuserToolBundle:
    if preset == "default":
        return bundle
    cards = tuple(card for card in bundle.cards if preset_allows_card(preset, card))
    names = {card.name for card in cards}
    return SuperuserToolBundle(
        tools={name: tool for name, tool in bundle.tools.items() if name in names},
        cards=cards,
    )


def _readonly_fast_tool_names(
    *,
    message_text: str,
    cards: tuple[Any, ...],
    available_tool_names: tuple[str, ...],
) -> tuple[str, ...]:
    query = normalize_message_text(message_text).lower()
    scored: list[tuple[int, str]] = []
    for card in cards:
        if not bool(getattr(card, "read_only", False)):
            continue
        name = str(getattr(card, "name", "") or "")
        if name not in available_tool_names:
            continue
        haystack = " ".join(
            str(part or "").lower()
            for part in (
                name,
                getattr(card, "category", ""),
                getattr(card, "description", ""),
                " ".join(getattr(card, "tags", ()) or ()),
            )
        )
        score = _readonly_fast_score(query, name=name, haystack=haystack)
        if score > 0:
            scored.append((score, name))
    scored.sort(key=lambda item: item[0], reverse=True)
    return tuple(name for _score, name in scored[:3])


def _readonly_fast_score(query: str, *, name: str, haystack: str) -> int:
    score = 0
    lowered_name = name.lower()
    if lowered_name and lowered_name in query:
        score += 5
    for part in lowered_name.split("_"):
        if len(part) >= 3 and part in query:
            score += 2
    if "git" in query and lowered_name == "git_command":
        score += 5
    if any(word in query for word in ("日志", "log")) and lowered_name == "read_file":
        score += 3
    if any(word in query for word in ("文件", "读", "看")) and lowered_name in {
        "read_file",
        "list_dir",
        "search_files",
    }:
        score += 2
    if any(word in query for word in ("状态", "status")) and "status" in lowered_name:
        score += 2
    if any(token in haystack for token in query.split() if len(token) >= 2):
        score += 1
    return score


__all__ = ["run_superuser_agent_runtime"]
