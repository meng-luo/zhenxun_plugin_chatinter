"""Sub-agent delegation tools for superuser Agent mode."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any
import uuid

from zhenxun.services.llm import LLMMessage
from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ...agent_runtime import AgentRuntime
from ...agent_state import AgentRunState
from ...capability_registry import CapabilityRegistry
from ...config import (
    build_reasoning_generation_config,
    get_config_value,
    get_model_name,
)
from ...provider_capability import ProviderCapabilityAdapter
from ...route_text import normalize_message_text
from ..audit_log import record_audit_event
from ..registry import register_superuser_tool
from .common import actor_from_context, tool_result

_MAX_DELEGATE_TASKS = 3
_DEFAULT_DELEGATE_STEPS = 14
_MAX_DELEGATE_STEPS = 30
_DEFAULT_DELEGATE_TIMEOUT = 90.0
_MAX_DELEGATE_TIMEOUT = 180.0
_BLOCKED_TOOL_NAMES = {
    "delegate_task",
    "session_search",
}
_BLOCKED_PREFIXES = (
    "approve_",
    "reject_",
    "revoke_",
    "list_pending_",
    "memory_",
    "agent_run_resume",
)


@dataclass(frozen=True)
class _DelegateTask:
    title: str
    task: str


class DelegateTaskTool:
    name = "delegate_task"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：把一个或多个可独立处理的子任务委托给隔离"
                "子 AgentRuntime。子代理有独立历史和预算，只能使用受限工具集，"
                "不会再调用 delegate/approval/memory 类工具。适合并行排查、"
                "读文件、搜索、运行低耦合验证。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": ["string", "null"],
                        "description": "单个子任务。tasks 为空时使用。",
                    },
                    "tasks": {
                        "type": ["array", "null"],
                        "description": "批量子任务，最多 3 个。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": ["string", "null"],
                                    "description": "子任务标题。",
                                },
                                "task": {
                                    "type": "string",
                                    "description": "子任务内容。",
                                },
                            },
                            "required": ["title", "task"],
                            "additionalProperties": False,
                        },
                    },
                    "max_steps": {
                        "type": ["integer", "null"],
                        "description": "每个子代理最大步数，默认 14，最大 30。",
                    },
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "description": "每个子任务超时，默认 90 秒，最大 180 秒。",
                    },
                    "allowed_tools": {
                        "type": ["array", "null"],
                        "description": (
                            "可选工具名白名单；为空时使用默认受限超级用户工具集。"
                        ),
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "task",
                    "tasks",
                    "max_steps",
                    "timeout_seconds",
                    "allowed_tools",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        if _delegate_depth(context) >= 1:
            return tool_result(False, "delegate_recursion_blocked")
        allowed, reason = _delegate_context_allowed(context)
        if not allowed:
            return tool_result(False, "delegate_not_available", reason=reason)
        tasks = _parse_delegate_tasks(kwargs)
        if not tasks:
            return tool_result(False, "delegate_task_required")
        max_steps = _coerce_int(
            kwargs.get("max_steps"),
            default=_DEFAULT_DELEGATE_STEPS,
            minimum=2,
            maximum=_MAX_DELEGATE_STEPS,
        )
        timeout_seconds = _coerce_float(
            kwargs.get("timeout_seconds"),
            default=_DEFAULT_DELEGATE_TIMEOUT,
            minimum=5.0,
            maximum=_MAX_DELEGATE_TIMEOUT,
        )
        allowed_tools = _normalize_tool_names(kwargs.get("allowed_tools"))
        started = time.perf_counter()
        results = await asyncio.gather(
            *[
                _run_delegate_task(
                    task=item,
                    index=index,
                    parent_context=context,
                    actor=actor,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    allowed_tools=allowed_tools,
                )
                for index, item in enumerate(tasks, 1)
            ]
        )
        ok = all(bool(result.get("ok")) for result in results)
        payload = {
            "tasks": len(tasks),
            "max_steps": max_steps,
            "timeout_seconds": timeout_seconds,
            "elapsed_ms": max(int((time.perf_counter() - started) * 1000), 0),
            "results": results,
        }
        record_audit_event(
            event="delegate_task_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=self.name,
            payload={
                "tasks": [task.task for task in tasks],
                "max_steps": max_steps,
                "allowed_tools": sorted(allowed_tools) if allowed_tools else [],
            },
            result={"ok": ok, "result_count": len(results)},
        )
        return tool_result(ok, "delegate_task_completed", **payload)


async def _run_delegate_task(
    *,
    task: _DelegateTask,
    index: int,
    parent_context: Any | None,
    actor: dict[str, str],
    max_steps: int,
    timeout_seconds: float,
    allowed_tools: set[str],
) -> dict[str, Any]:
    trace_id = f"delegate-{uuid.uuid4().hex[:12]}"
    try:
        model_name = get_model_name()
        provider_adapter = ProviderCapabilityAdapter.for_model(model_name)
        capability_registry = CapabilityRegistry.empty(
            session_id=f"{actor['session_key']}:delegate:{index}",
        )
        capability_registry.register_available_superuser_tools(
            message_text=task.task,
            include_deferred=True,
        )
        _filter_delegate_registry(
            capability_registry,
            allowed_tools=allowed_tools,
        )
        tool_map = capability_registry.executable_tool_map()
        if not tool_map:
            return {
                "ok": False,
                "status": "delegate_no_tools_available",
                "title": task.title,
                "task": task.task,
            }
        run_context = _child_context(
            parent_context,
            actor=actor,
            trace_id=trace_id,
            capability_registry=capability_registry,
            provider_adapter=provider_adapter,
        )
        state = AgentRunState.create(
            trace_id=trace_id,
            run_id=trace_id,
            session_key=run_context.session_id,
            messages=_delegate_messages(task),
            tool_map=tool_map,
            current_message=f"子任务标题: {task.title}\n子任务内容: {task.task}",
            max_steps=max_steps,
            max_total_tokens=max_steps * 6000,
            max_step_refunds=max(1, max_steps // 3),
            agent_complexity_mode="delegate_task",
            agent_complexity_reason="subagent_delegation",
        )
        runtime = AgentRuntime(
            state=state,
            run_context=run_context,
            message_text=f"{task.title}\n{task.task}",
            model_name=model_name,
            generation_config=build_reasoning_generation_config(),
            timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
            budget_controller=None,
        )
        result = await asyncio.wait_for(runtime.run(), timeout=timeout_seconds)
        return {
            "ok": result.status != "failed",
            "status": result.status,
            "title": task.title,
            "task": task.task,
            "trace_id": result.trace_id,
            "run_id": result.run_id,
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "final_text": normalize_message_text(result.final_text)[:2000],
            "completed_tasks": [
                completed.__dict__ for completed in result.completed_tasks[:12]
            ],
            "pending_tasks": [
                pending.__dict__ for pending in result.pending_tasks[:12]
            ],
        }
    except TimeoutError:
        return {
            "ok": False,
            "status": "delegate_timeout",
            "title": task.title,
            "task": task.task,
            "trace_id": trace_id,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "delegate_failed",
            "title": task.title,
            "task": task.task,
            "trace_id": trace_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _filter_delegate_registry(
    registry: CapabilityRegistry,
    *,
    allowed_tools: set[str],
) -> None:
    for name in list(registry.tool_records):
        normalized = normalize_message_text(name)
        if _delegate_tool_blocked(normalized):
            registry.tool_records.pop(name, None)
            registry.generation += 1
            continue
        if allowed_tools and normalized not in allowed_tools:
            registry.tool_records.pop(name, None)
            registry.generation += 1


def _delegate_tool_blocked(tool_name: str) -> bool:
    if not tool_name or tool_name in _BLOCKED_TOOL_NAMES:
        return True
    return any(tool_name.startswith(prefix) for prefix in _BLOCKED_PREFIXES)


def _child_context(
    parent_context: Any | None,
    *,
    actor: dict[str, str],
    trace_id: str,
    capability_registry: CapabilityRegistry,
    provider_adapter: ProviderCapabilityAdapter,
) -> RunContext:
    parent_extra = getattr(parent_context, "extra", None)
    extra = dict(parent_extra) if isinstance(parent_extra, dict) else {}
    extra.update(
        {
            "actor_user_id": actor["user_id"],
            "agent_mode": "superuser_subagent",
            "enable_agent_tools": True,
            "delegate_depth": _delegate_depth(parent_context) + 1,
            "parent_session_key": actor["session_key"],
            "parent_trace_id": str(extra.get("trace_id", "") or ""),
            "delegate_trace_id": trace_id,
            "capability_registry": capability_registry,
            "provider_capability": provider_adapter.profile.to_metadata(),
            "mcp_status": {"disabled": "delegate_task"},
        }
    )
    return RunContext(
        session_id=f"{actor['session_key']}:delegate:{trace_id}",
        extra=extra,
    )


def _delegate_messages(task: _DelegateTask) -> list[LLMMessage]:
    return [
        LLMMessage.system(
            "\n".join(
                [
                    "你是 ChatInter 的隔离子代理，只处理父代理委托的子任务。",
                    "必须优先使用可用工具获取真实结果，"
                    "不要调用或要求 delegate/approval/memory 工具。",
                    "如果工具集不足，明确说明缺少什么能力；不要编造执行结果。",
                    "最终回答只总结本子任务的证据、结论、关键文件/命令/错误。",
                ]
            )
        )
    ]


def _parse_delegate_tasks(kwargs: dict[str, Any]) -> tuple[_DelegateTask, ...]:
    parsed: list[_DelegateTask] = []
    raw_tasks = kwargs.get("tasks")
    if isinstance(raw_tasks, list):
        for index, item in enumerate(raw_tasks[:_MAX_DELEGATE_TASKS], 1):
            if not isinstance(item, dict):
                continue
            task = normalize_message_text(str(item.get("task", "") or ""))
            if not task:
                continue
            title = normalize_message_text(str(item.get("title", "") or ""))
            parsed.append(_DelegateTask(title=title or f"task-{index}", task=task))
    if not parsed:
        task = normalize_message_text(str(kwargs.get("task", "") or ""))
        if task:
            parsed.append(_DelegateTask(title="task-1", task=task))
    return tuple(parsed[:_MAX_DELEGATE_TASKS])


def _normalize_tool_names(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        normalize_message_text(str(item or ""))
        for item in value
        if normalize_message_text(str(item or ""))
    }


def _delegate_depth(context: Any | None) -> int:
    extra = getattr(context, "extra", None)
    if not isinstance(extra, dict):
        return 0
    try:
        return max(int(extra.get("delegate_depth", 0) or 0), 0)
    except (TypeError, ValueError):
        return 0


def _delegate_context_allowed(context: Any | None) -> tuple[bool, str]:
    extra = getattr(context, "extra", None)
    if not isinstance(extra, dict):
        return False, "missing_run_context"
    if not bool(extra.get("enable_agent_tools")):
        return False, "agent_tools_disabled"
    if normalize_message_text(str(extra.get("agent_mode", "") or "")) != (
        "superuser_agent"
    ):
        return False, "not_superuser_agent"
    complexity = extra.get("agent_complexity")
    mode = ""
    if isinstance(complexity, dict):
        mode = normalize_message_text(str(complexity.get("mode", "") or ""))
    if mode != "complex_pev":
        return False, "not_complex_long_task"
    return True, "superuser_complex_task"


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _coerce_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


register_superuser_tool(
    DelegateTaskTool,
    category="delegate",
    risk="medium",
    approval_mode="policy",
    read_only=False,
    destructive=False,
    side_effect="execute",
    description="将独立子任务交给隔离子 AgentRuntime 并行处理。",
    tags=("delegate", "subagent", "parallel"),
    source_of_truth="local_state",
    requires_real_tool=True,
    output_mode="text",
    entity_scope="global",
    reliability=0.72,
    schema_quality=0.82,
    soft_tool=False,
)
