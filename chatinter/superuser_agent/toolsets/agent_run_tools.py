"""AgentRun persistence query tools for superuser private Agent mode."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm import LLMMessage
from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ...agent_complexity import route_agent_complexity
from ...agent_run_store import (
    get_agent_run_snapshot,
    list_agent_run_snapshots,
    load_agent_run_state,
    project_agent_run_state,
    query_agent_run_events,
    update_agent_run_status,
)
from ...agent_runtime import AgentRuntime
from ...agent_state import AgentObservation
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


class AgentRunStatusTool:
    name = "agent_run_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：查询持久化 AgentRun 状态。可查看当前/历史 "
                "trace 的 timeline、tool calls、observations、pending tasks。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "trace_id": {
                        "type": ["string", "null"],
                        "description": "AgentRun trace_id；为空则列出最近运行。",
                    },
                    "include_full_snapshot": {
                        "type": ["boolean", "null"],
                        "description": "是否返回完整快照，默认 false。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "列出最近运行或事件数量，默认 20。",
                    },
                    "current_session_only": {
                        "type": ["boolean", "null"],
                        "description": "是否只看当前会话，默认 true。",
                    },
                },
                "required": [
                    "trace_id",
                    "include_full_snapshot",
                    "limit",
                    "current_session_only",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        trace_id = str(kwargs.get("trace_id", "") or "").strip()
        include_full = bool(kwargs.get("include_full_snapshot") or False)
        current_session_only = kwargs.get("current_session_only")
        if current_session_only is None:
            current_session_only = True
        limit = _coerce_limit(kwargs.get("limit"), default=20, upper=100)
        session_key = actor["session_key"] if current_session_only else ""
        if trace_id:
            snapshot = get_agent_run_snapshot(trace_id)
            if snapshot is None:
                return tool_result(False, "agent_run_not_found", trace_id=trace_id)
            if session_key and str(snapshot.get("session_key", "")) != session_key:
                return tool_result(False, "agent_run_not_found", trace_id=trace_id)
            events = query_agent_run_events(
                trace_id=trace_id,
                session_key=session_key,
                limit=limit,
            )
            projection = project_agent_run_state(
                run_id=trace_id,
                session_key=session_key,
                include_details=False,
            )
            payload = snapshot if include_full else _compact_snapshot(snapshot)
            if isinstance(payload, dict):
                payload["runtime_projection"] = projection
            record_audit_event(
                event="agent_run_queried",
                user_id=actor["user_id"],
                session_key=actor["session_key"],
                action=self.name,
                payload={"trace_id": trace_id, "include_full_snapshot": include_full},
                result={"events": len(events)},
            )
            return tool_result(
                True,
                "agent_run_status",
                trace_id=trace_id,
                snapshot=payload,
                events=events,
            )
        runs = list_agent_run_snapshots(session_key=session_key, limit=limit)
        return tool_result(
            True,
            "agent_runs_listed",
            runs=runs,
            count=len(runs),
        )


class AgentRunResumeTool:
    name = "agent_run_resume"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：恢复一个 paused AgentRun。适用于确认已完成或后台任务"
                "结束后继续同一条任务链。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "要恢复的 AgentRun run_id/trace_id。",
                    },
                    "resume_message": {
                        "type": ["string", "null"],
                        "description": "补充给 Agent 的恢复说明；可为空。",
                    },
                    "max_steps": {
                        "type": ["integer", "null"],
                        "description": "恢复后最多继续几步，默认沿用快照或 6。",
                    },
                },
                "required": ["run_id", "resume_message", "max_steps"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        run_id = str(kwargs.get("run_id", "") or "").strip()
        if not run_id:
            return tool_result(False, "agent_run_id_required")
        snapshot = get_agent_run_snapshot(run_id)
        if not isinstance(snapshot, dict):
            return tool_result(False, "agent_run_not_found", run_id=run_id)
        if str(snapshot.get("session_key", "") or "") != actor["session_key"]:
            return tool_result(False, "agent_run_not_found", run_id=run_id)
        if str(snapshot.get("status", "") or "") == "cancelled":
            return tool_result(False, "agent_run_cancelled", run_id=run_id)
        if str(snapshot.get("status", "") or "") not in {"paused", "running"}:
            return tool_result(
                False,
                "agent_run_not_resumable",
                run_id=run_id,
                snapshot_status=snapshot.get("status", ""),
            )

        model_name = get_model_name()
        provider_adapter = ProviderCapabilityAdapter.for_model(model_name)
        capability_registry = CapabilityRegistry.empty(session_id=actor["session_key"])
        capability_registry.register_available_superuser_tools(
            message_text=str(kwargs.get("resume_message") or ""),
            include_deferred=True,
        )
        mcp_status = None
        tools = capability_registry.executable_tool_map()
        state = load_agent_run_state(run_id, tool_map=tools)
        if state is None:
            return tool_result(False, "agent_run_resume_load_failed", run_id=run_id)
        state.tool_map = tools
        state.status = "paused"
        state.max_steps = _resume_max_steps(
            kwargs.get("max_steps"),
            fallback=max(6, int(state.max_steps or 8) - int(state.step or 0)),
        )
        state.resume(reason="agent_run_resume")
        pre_resume_observation = _coerce_pre_resume_observation(
            kwargs.get("pre_resume_observation"),
            step=state.step,
        )
        if pre_resume_observation is not None:
            state.append_synthetic_observation(
                pre_resume_observation,
                timeline_kind="pre_resume_tool_observation",
                content=pre_resume_observation.error
                or str(pre_resume_observation.output.get("status", "") or ""),
                metadata={"source": "runtime_approval"},
            )
        resume_message = normalize_message_text(str(kwargs.get("resume_message") or ""))
        resume_context = _resume_context_message(snapshot, resume_message)
        complexity_decision = route_agent_complexity(
            message_text=resume_message or str(snapshot.get("final_text", "") or ""),
            tool_map=tools,
            enable_agent_tools=True,
            resumed_run=True,
        )
        state.agent_complexity_mode = complexity_decision.mode
        state.agent_complexity_reason = complexity_decision.reason
        state.messages.append(resume_context)
        state.append_timeline(
            role="user",
            kind="agent_resume_request",
            content=resume_message or "继续这个 AgentRun。",
            metadata={
                "run_id": state.run_id,
                "snapshot_status": snapshot.get("status", ""),
                "task_graph_preserved": state.task_graph is not None,
                "task_ledger_preserved": state.task_ledger is not None,
                "capability_ledger_preserved": bool(
                    state.capability_ledger.public_entries(limit=1)
                ),
                "resume_integrity": snapshot.get("resume_integrity", {}),
            },
        )
        state.append_timeline(
            role="system",
            kind="agent_complexity",
            metadata=complexity_decision.to_metadata(),
        )
        run_context = RunContext(
            session_id=actor["session_key"],
            extra={
                "actor_user_id": actor["user_id"],
                "agent_mode": "superuser_agent",
                "enable_agent_tools": True,
                "trace_id": state.trace_id,
                "run_id": state.run_id,
                "resumed_run_id": state.run_id,
                "agent_complexity": complexity_decision.to_metadata(),
                "capability_registry": capability_registry,
                "provider_capability": provider_adapter.profile.to_metadata(),
                "mcp_status": mcp_status,
            },
        )
        runtime = AgentRuntime(
            state=state,
            run_context=run_context,
            message_text=resume_message or "继续这个 AgentRun。",
            model_name=model_name,
            generation_config=build_reasoning_generation_config(),
            timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
            budget_controller=None,
        )
        result = await runtime.run()
        record_audit_event(
            event="agent_run_resumed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=self.name,
            payload={"run_id": run_id, "resume_message": resume_message},
            result={"status": result.status, "stop_reason": result.stop_reason},
        )
        return tool_result(
            result.status != "failed",
            "agent_run_resumed",
            run_id=result.run_id or run_id,
            run_status=result.status,
            paused_reason=result.paused_reason,
            stop_reason=result.stop_reason,
            final_text=result.final_text,
            steps=result.steps,
            waiting_approval_ids=list(state.waiting_approval_ids),
            background_task_ids=list(state.background_task_ids),
            observation_event_ids=list(state.observation_event_ids[-10:]),
            artifact_refs=list(state.artifact_refs[-10:]),
            task_graph=_compact_task_graph(
                state.task_graph.to_public_payload()
                if state.task_graph is not None
                else {}
            )
            or _compact_task_graph(snapshot.get("task_graph")),
            task_ledger=state.task_ledger.to_public_payload()
            if state.task_ledger is not None
            else snapshot.get("task_ledger", {}),
            resume_integrity={
                **dict(snapshot.get("resume_integrity") or {}),
                "runtime_projection": project_agent_run_state(
                    run_id=result.run_id or run_id,
                    session_key=actor["session_key"],
                    include_details=False,
                ),
            },
        )


class AgentRunCancelTool:
    name = "agent_run_cancel"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：取消一个 paused/running AgentRun，"
                "使其不能再恢复。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "要取消的 AgentRun run_id/trace_id。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "取消原因，可为空。",
                    },
                },
                "required": ["run_id", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        run_id = str(kwargs.get("run_id", "") or "").strip()
        reason = normalize_message_text(
            str(kwargs.get("reason") or "cancelled_by_user")
        )
        if not run_id:
            return tool_result(False, "agent_run_id_required")
        snapshot = get_agent_run_snapshot(run_id)
        if not isinstance(snapshot, dict):
            return tool_result(False, "agent_run_not_found", run_id=run_id)
        if str(snapshot.get("session_key", "") or "") != actor["session_key"]:
            return tool_result(False, "agent_run_not_found", run_id=run_id)
        updated = update_agent_run_status(
            run_id,
            status="cancelled",
            reason=reason,
            metadata={"cancelled_by": actor["user_id"]},
        )
        record_audit_event(
            event="agent_run_cancelled",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action=self.name,
            payload={"run_id": run_id, "reason": reason},
            result={"cancelled": updated is not None},
        )
        if updated is None:
            return tool_result(False, "agent_run_cancel_failed", run_id=run_id)
        return tool_result(
            True,
            "agent_run_cancelled",
            run_id=run_id,
            reason=reason,
            snapshot=_compact_snapshot(updated),
        )


def _compact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": snapshot.get("run_id", snapshot.get("trace_id", "")),
        "trace_id": snapshot.get("trace_id", ""),
        "session_key": snapshot.get("session_key", ""),
        "updated_at": snapshot.get("updated_at", ""),
        "stage": snapshot.get("stage", ""),
        "status": snapshot.get("status", ""),
        "paused_reason": snapshot.get("paused_reason", ""),
        "resume_cursor": snapshot.get("resume_cursor", {}),
        "waiting_approval_ids": snapshot.get("waiting_approval_ids", [])[-10:]
        if isinstance(snapshot.get("waiting_approval_ids"), list)
        else [],
        "background_task_ids": snapshot.get("background_task_ids", [])[-10:]
        if isinstance(snapshot.get("background_task_ids"), list)
        else [],
        "artifact_refs": snapshot.get("artifact_refs", [])[-10:]
        if isinstance(snapshot.get("artifact_refs"), list)
        else [],
        "observation_event_ids": snapshot.get("observation_event_ids", [])[-10:]
        if isinstance(snapshot.get("observation_event_ids"), list)
        else [],
        "step": snapshot.get("step", 0),
        "max_steps": snapshot.get("max_steps", 0),
        "stop_reason": snapshot.get("stop_reason", ""),
        "recovery_action": snapshot.get("recovery_action", ""),
        "final_text": str(snapshot.get("final_text", "") or "")[:800],
        "tool_names": snapshot.get("tool_names", [])[:60]
        if isinstance(snapshot.get("tool_names"), list)
        else [],
        "tool_calls": snapshot.get("tool_calls", [])[-12:]
        if isinstance(snapshot.get("tool_calls"), list)
        else [],
        "observations": snapshot.get("observations", [])[-12:]
        if isinstance(snapshot.get("observations"), list)
        else [],
        "pending_tasks": snapshot.get("pending_tasks", [])[-12:]
        if isinstance(snapshot.get("pending_tasks"), list)
        else [],
        "completed_tasks": snapshot.get("completed_tasks", [])[-12:]
        if isinstance(snapshot.get("completed_tasks"), list)
        else [],
        "task_graph": _compact_task_graph(snapshot.get("task_graph")),
        "budget": snapshot.get("budget", {}),
    }


def _compact_task_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    tasks = value.get("tasks", []) if isinstance(value.get("tasks"), list) else []
    return {
        "graph_id": value.get("graph_id", ""),
        "status": value.get("status", ""),
        "reason": str(value.get("reason", "") or "")[:300],
        "tasks": [
            {
                "task_id": item.get("task_id", ""),
                "goal": str(item.get("goal", "") or "")[:300],
                "required_tools": item.get("required_tools", [])[:8]
                if isinstance(item.get("required_tools"), list)
                else [],
                "acceptance_criteria": item.get("acceptance_criteria", [])[:5]
                if isinstance(item.get("acceptance_criteria"), list)
                else [],
                "status": item.get("status", ""),
                "reason": str(item.get("reason", "") or "")[:300],
                "observation_count": len(item.get("observations", []) or []),
                "artifacts": item.get("artifacts", [])[-5:]
                if isinstance(item.get("artifacts"), list)
                else [],
            }
            for item in tasks[:12]
            if isinstance(item, dict)
        ],
    }


def _coerce_limit(value: Any, *, default: int, upper: int) -> int:
    try:
        return max(1, min(int(value or default), upper))
    except (TypeError, ValueError):
        return default


def _resume_max_steps(value: Any, *, fallback: int) -> int:
    try:
        return max(1, min(int(value or fallback or 6), 12))
    except (TypeError, ValueError):
        return max(1, min(int(fallback or 6), 12))


def _resume_context_message(snapshot: dict[str, Any], resume_message: str):
    cursor = (
        snapshot.get("resume_cursor")
        if isinstance(snapshot.get("resume_cursor"), dict)
        else {}
    )
    background_ids = snapshot.get("background_task_ids", [])
    approval_ids = snapshot.get("waiting_approval_ids", [])
    task_graph = _compact_task_graph(snapshot.get("task_graph"))
    task_ledger = (
        snapshot.get("task_ledger")
        if isinstance(snapshot.get("task_ledger"), dict)
        else {}
    )
    resume_integrity = (
        snapshot.get("resume_integrity")
        if isinstance(snapshot.get("resume_integrity"), dict)
        else {}
    )
    return LLMMessage.user(
        "Resume AgentRun from persisted state.\n"
        f"run_id: {snapshot.get('run_id') or snapshot.get('trace_id')}\n"
        f"paused_reason: {snapshot.get('paused_reason', '')}\n"
        f"resume_cursor: {cursor}\n"
        f"waiting_approval_ids: {approval_ids}\n"
        f"background_task_ids: {background_ids}\n"
        f"resume_integrity: {resume_integrity}\n"
        f"task_graph: {task_graph}\n"
        f"task_ledger: {task_ledger}\n"
        f"user_resume_message: {resume_message or '继续之前的任务'}\n"
        "Continue the original TaskGraph/TaskLedger; do not restart planning unless "
        "the snapshot has no graph. If approval was required and has not been "
        "approved, explain what is waiting. If background_task_ids exist, call "
        "background_task_status before continuing."
    )


def _coerce_pre_resume_observation(
    value: Any,
    *,
    step: int,
) -> AgentObservation | None:
    if not isinstance(value, dict):
        return None
    output = value.get("output")
    output = dict(output) if isinstance(output, dict) else {}
    tool_name = normalize_message_text(str(value.get("tool_name", "") or ""))
    if not tool_name and not output:
        return None
    artifacts = output.get("artifacts")
    return AgentObservation(
        tool_call_id=normalize_message_text(str(value.get("tool_call_id", "") or ""))
        or "runtime_approval",
        tool_name=tool_name or "runtime_approval",
        task_text=normalize_message_text(str(value.get("task_text", "") or "")),
        ok=bool(output.get("ok")),
        need_continue=bool(output.get("need_continue")),
        remaining_task_hint=normalize_message_text(
            str(output.get("remaining_task_hint", "") or "")
        ),
        error=normalize_message_text(str(output.get("error", "") or "")),
        artifacts=tuple(
            dict(item) for item in artifacts or [] if isinstance(item, dict)
        )
        if isinstance(artifacts, list | tuple)
        else (),
        step=max(int(step or 0), 0),
        output=output,
    )


register_superuser_tool(AgentRunStatusTool)
register_superuser_tool(
    AgentRunResumeTool, risk="low", destructive=False, side_effect="mutate"
)
register_superuser_tool(
    AgentRunCancelTool, risk="low", destructive=True, side_effect="mutate"
)

__all__ = ["AgentRunCancelTool", "AgentRunResumeTool", "AgentRunStatusTool"]
