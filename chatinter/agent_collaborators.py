"""External collaborators for AgentRuntime.

AgentRuntime should own the model/tool loop, not every policy detail.  This
module keeps the main loop readable while preserving the same runtime chain:
ToolIntentGate -> TaskLedger -> Observation -> FinalValidator -> Guardrails.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from .agent_state import AgentObservation, AgentRunState
from .completion_validator import validate_final_reply
from .llm_compat import LLMMessage, RunContext, ToolResult
from .route_text import normalize_message_text
from .task_ledger import TaskLedger, TaskLedgerEntry

DIRECT_ANSWER_INTERCEPT_LIMIT = 1
FINAL_VALIDATION_INTERCEPT_LIMIT = 1
TASK_COVERAGE_INTERCEPT_LIMIT = 2
BACKGROUND_OBSERVATION_WAIT_SECONDS = 6.0
SAFE_NO_TOOL_RESULT_REPLY = (
    "我没有拿到真实工具执行结果，不能直接说已经完成。你可以换个说法，"
    "或稍后再让我试一次。"
)


class RuntimeView(Protocol):
    state: AgentRunState
    run_context: RunContext
    message_text: str
    coverage_judge: Any
    planner: Any
    verifier: Any

    def _available_tool_summaries(self, *, limit: int = 16) -> list[dict[str, Any]]: ...
    def _has_available_required_tools(self) -> bool: ...
    def _has_command_observation(self) -> bool: ...
    def _has_actionable_command_tools(self) -> bool: ...
    def _actionable_command_tool_count(self) -> int: ...
    def _persist_state(self, stage: str, **metadata: Any) -> None: ...
    def request_required_tool_choice_once(self) -> None: ...
    async def verify_after_observation(self) -> None: ...


class RuntimeTaskLedgerCoordinator:
    """Owns semantic task listing and TaskGraph verification."""

    def __init__(self, runtime: RuntimeView) -> None:
        self.runtime = runtime

    async def seed(self) -> None:
        await self.seed_task_graph()
        await self.seed_task_ledger()

    async def seed_task_graph(self) -> None:
        runtime = self.runtime
        if not self._complex_planner_enabled():
            return
        if not self._should_use_task_graph():
            return
        graph = await runtime.planner.plan(
            original_goal=runtime.message_text,
            available_tools=runtime._available_tool_summaries(limit=80),
            resumed_graph=runtime.state.task_graph,
        )
        if graph is None:
            return
        runtime.state.set_task_graph(graph, source="agent_planner")
        runtime.state.add_pending_tasks(
            [task.goal for task in graph.incomplete_tasks],
            source="task_graph_initial",
        )
        runtime.state.messages.append(
            LLMMessage.user(
                "TaskGraph initialized for this superuser Agent run:\n"
                + json.dumps(graph.to_public_payload(), ensure_ascii=False)
                + "\nUse tools to satisfy acceptance_criteria. Final answer must not "
                "claim unfinished tasks are complete."
            )
        )
        runtime._persist_state("task_graph_initialized", task_count=len(graph.tasks))

    async def verify_after_observation(self) -> None:
        runtime = self.runtime
        if not self._complex_verifier_enabled():
            return
        if runtime.state.task_graph is None or not runtime.state.observations:
            return
        observation = runtime.state.observations[-1]
        if not observation.tool_name:
            return
        result = await runtime.verifier.verify_observation(
            graph=runtime.state.task_graph,
            observation=observation,
            available_tools=runtime._available_tool_summaries(limit=40),
        )
        incomplete = runtime.state.incomplete_task_goals()
        runtime.state.replace_pending_tasks(
            incomplete,
            source="task_graph_verifier",
        )
        runtime.state.append_timeline(
            role="system",
            kind="task_graph_verification",
            metadata={
                "step": runtime.state.step,
                "mode": "after_observation",
                "updates": [item.model_dump() for item in result.updates],
                "missing_tasks": list(result.missing_tasks),
                "reason": normalize_message_text(result.reason),
                "graph": runtime.state.task_graph.to_public_payload(),
            },
        )

    async def seed_task_ledger(self) -> None:
        runtime = self.runtime
        if not self._should_use_task_ledger():
            return
        if runtime.state.task_ledger is not None:
            runtime.state.refresh_capability_ledger(
                runtime._available_tool_summaries(limit=0)
            )
            runtime.state.replace_pending_tasks(
                runtime.state.task_ledger.incomplete_goals,
                source="task_ledger_resumed",
            )
            runtime.state.append_timeline(
                role="system",
                kind="task_ledger_resumed",
                metadata={
                    "step": runtime.state.step,
                    "ledger": runtime.state.task_ledger.to_public_payload(),
                    "capability_ledger": runtime.state.capability_ledger.public_entries(
                        limit=0
                    ),
                },
            )
            runtime._persist_state(
                "task_ledger_resumed",
                task_count=len(runtime.state.task_ledger.tasks),
            )
            return
        capabilities = runtime._available_tool_summaries(limit=0)
        runtime.state.refresh_capability_ledger(capabilities)
        result = await runtime.coverage_judge.plan_task_ledger(
            original_message=runtime.message_text,
            available_capabilities=runtime.state.capability_ledger.public_entries(
                limit=0
            ),
        )
        tasks: list[TaskLedgerEntry] = []
        for index, item in enumerate(result.tasks[:12], 1):
            goal = normalize_message_text(item.goal)
            if not goal:
                continue
            tasks.append(
                TaskLedgerEntry(
                    task_id=normalize_message_text(item.task_id) or f"task_{index}",
                    goal=goal,
                    intent_type=normalize_message_text(item.intent_type) or "unknown",
                    requires_real_tool=bool(item.requires_real_tool),
                    expected_capabilities=_normalized_tasks(item.expected_capabilities),
                    acceptance_criteria=_normalized_tasks(item.acceptance_criteria),
                    reason=normalize_message_text(item.reason),
                )
            )
        if not tasks:
            return
        ledger = TaskLedger.create(
            original_message=runtime.message_text,
            tasks=tasks,
            reason=result.reason,
        )
        runtime.state.set_task_ledger(ledger, source="llm_task_ledger")
        runtime.state.replace_pending_tasks(
            ledger.incomplete_goals,
            source="task_ledger_initial",
        )
        runtime.state.messages.append(
            LLMMessage.user(
                "TaskLedger initialized by semantic task listing:\n"
                + json.dumps(ledger.to_public_payload(), ensure_ascii=False)
                + "\nComplete tasks by calling real tools when "
                "requires_real_tool=true. "
                "Do not rely on local connector words; each tool call should map to "
                "one listed task when possible. If no visible command schema can "
                "cover a ledger task, call retrieve_plugin_commands first instead "
                "of declaring the task unsupported."
            )
        )
        runtime._persist_state("task_ledger_initialized", task_count=len(ledger.tasks))

    def _should_use_task_graph(self) -> bool:
        runtime = self.runtime
        extra = getattr(runtime.run_context, "extra", None)
        if not isinstance(extra, dict) or not bool(extra.get("enable_agent_tools")):
            return False
        if runtime.state.task_graph is not None:
            return True
        return bool(runtime.message_text and runtime.state.tool_map)

    def _should_use_task_ledger(self) -> bool:
        runtime = self.runtime
        if runtime.state.task_ledger is not None:
            return True
        if not self.looks_like_multi_task_turn():
            return False
        if not runtime.message_text or not runtime.state.tool_map:
            return False
        if runtime.state.observations or runtime.state.pending_tasks:
            return False
        if runtime.state.tool_obligation == "required":
            return True
        return runtime._actionable_command_tool_count() > 1

    def looks_like_multi_task_turn(self) -> bool:
        runtime = self.runtime
        extra = getattr(runtime.run_context, "extra", None)
        if isinstance(extra, dict):
            task_plan = extra.get("task_planner_lite")
            if isinstance(task_plan, dict) and isinstance(task_plan.get("tasks"), list):
                return len(task_plan["tasks"]) >= 2
        text = runtime.message_text
        if not text:
            return False
        if any(
            marker in text
            for marker in ("然后", "最后", "顺便", "接着", "同时", "以及")
        ):
            return True
        return (
            text.count("，") + text.count(",") >= 2
            and runtime._actionable_command_tool_count() > 2
        )

    def _agent_complexity_metadata(self) -> dict[str, Any]:
        extra = getattr(self.runtime.run_context, "extra", None)
        metadata = extra.get("agent_complexity") if isinstance(extra, dict) else None
        return dict(metadata) if isinstance(metadata, dict) else {}

    def _complex_planner_enabled(self) -> bool:
        metadata = self._agent_complexity_metadata()
        if metadata:
            return bool(metadata.get("enable_planner"))
        return self.runtime.state.agent_complexity_mode in {"complex_pev", "standard"}

    def _complex_verifier_enabled(self) -> bool:
        metadata = self._agent_complexity_metadata()
        if metadata:
            return bool(metadata.get("enable_verifier"))
        return self.runtime.state.agent_complexity_mode in {"complex_pev", "standard"}


class RuntimeFinalGate:
    """Final acceptance chain: obligation -> ledger -> hallucination guard."""

    def __init__(self, runtime: RuntimeView) -> None:
        self.runtime = runtime

    async def final_acceptance_action(
        self,
        final_text: str,
        *,
        allow_retry: bool = True,
    ) -> str:
        obligation_action = self._direct_answer_intercept_action(
            final_text,
            allow_retry=allow_retry,
        )
        if obligation_action:
            return obligation_action
        ledger_action = await self._task_ledger_final_action(
            final_text,
            allow_retry=allow_retry,
        )
        if ledger_action:
            return ledger_action
        return self._unobserved_completion_action(
            final_text,
            allow_retry=allow_retry,
        )

    async def _task_ledger_final_action(
        self,
        final_text: str,
        *,
        allow_retry: bool = True,
    ) -> str:
        runtime = self.runtime
        if runtime.state.task_ledger is None:
            return ""
        runtime.state.refresh_capability_ledger(
            runtime._available_tool_summaries(limit=0)
        )
        missing_goals = _normalized_tasks(runtime.state.task_ledger.incomplete_goals)
        runtime.state.append_timeline(
            role="system",
            kind="task_ledger_coverage",
            metadata={
                "step": runtime.state.step,
                "mode": "observation_evidence_only",
                "covered": not missing_goals,
                "missing_tasks": missing_goals,
                "ledger": runtime.state.task_ledger.to_public_payload(),
                "observation_count": len(runtime.state.observations),
            },
        )
        if not missing_goals:
            return ""
        if (
            not allow_retry
            or runtime.state.coverage_interceptions >= TASK_COVERAGE_INTERCEPT_LIMIT
        ):
            runtime.state.replace_pending_tasks(
                missing_goals, source="task_ledger_final"
            )
            return "partial_final"
        runtime.state.coverage_interceptions += 1
        runtime.state.replace_pending_tasks(missing_goals, source="task_ledger_final")
        if runtime._has_actionable_command_tools():
            runtime.request_required_tool_choice_once()
        runtime.state.append_guardrail_observation(
            {
                "ok": False,
                "status": "runtime_task_ledger_coverage",
                "guardrail_reason": "task_ledger_missing",
                "reason": "task_ledger_missing",
                "covered": False,
                "task_ledger": runtime.state.task_ledger.to_public_payload(),
                "capability_ledger": runtime.state.capability_ledger.public_entries(
                    limit=0
                ),
                "missing_tasks": missing_goals,
                "catalog_tool": "retrieve_plugin_commands",
                "instruction": (
                    "Continue unfinished TaskLedger tasks using real tools when "
                    "available. If no matching command schema is visible, call "
                    "retrieve_plugin_commands before explaining that a task cannot "
                    "be completed."
                ),
                "need_continue": True,
                "retryable": True,
            },
            as_message=True,
        )
        return "retry"

    def _direct_answer_intercept_action(
        self,
        final_text: str,
        *,
        allow_retry: bool = True,
    ) -> str:
        runtime = self.runtime
        if (
            runtime.state.tool_obligation != "required"
            or not runtime._has_available_required_tools()
            or runtime._has_command_observation()
        ):
            return ""
        if (
            not allow_retry
            or runtime.state.direct_answer_interceptions
            >= DIRECT_ANSWER_INTERCEPT_LIMIT
        ):
            return "safe_final"
        runtime.state.direct_answer_interceptions += 1
        runtime.request_required_tool_choice_once()
        runtime.state.append_guardrail_observation(
            {
                "ok": False,
                "status": "runtime_tool_obligation",
                "guardrail_reason": "tool_required_but_model_answered_directly",
                "reason": "tool_required_but_model_answered_directly",
                "available_tools": runtime._available_tool_summaries(limit=16),
                "instruction": "select a real tool or explain why no tool applies",
                "model_answer_summary": final_text[:240],
                "need_continue": True,
                "retryable": True,
            },
            as_message=True,
        )
        return "retry"

    def _unobserved_completion_action(
        self,
        final_text: str,
        *,
        allow_retry: bool = True,
    ) -> str:
        runtime = self.runtime
        validation = validate_final_reply(
            final_text=final_text,
            tool_obligation=runtime.state.tool_obligation,
            observations=runtime.state.observations,
            pending_tasks=runtime.state.pending_tasks,
            tool_map=runtime.state.tool_map,
        )
        if validation.ok:
            return ""
        if (
            not allow_retry
            or runtime.state.final_validation_interceptions
            >= FINAL_VALIDATION_INTERCEPT_LIMIT
        ):
            return "safe_final"
        runtime.state.final_validation_interceptions += 1
        if runtime._has_available_required_tools():
            runtime.request_required_tool_choice_once()
        runtime.state.append_guardrail_observation(
            {
                "ok": False,
                "status": "runtime_final_validation",
                "guardrail_reason": validation.reason,
                "reason": validation.reason,
                "requires_observation": validation.requires_observation,
                "successful_observations": validation.successful_observations,
                "action_like_tools": validation.action_like_tools,
                "available_tools": runtime._available_tool_summaries(limit=16),
                "instruction": (
                    "Do not claim image/file/plugin/action completion without "
                    "a real tool observation. Call a real tool, or explain why "
                    "no tool applies."
                ),
                "model_answer_summary": final_text[:240],
                "need_continue": True,
                "retryable": True,
            },
            as_message=True,
        )
        return "retry"

    @staticmethod
    def partial_task_ledger_reply(state: AgentRunState) -> str:
        if state.task_ledger is None:
            return "我还没能确认任务账本是否全部完成。"
        missing = [task.goal for task in state.task_ledger.pending_tasks[:5]]
        if not missing:
            return "我还没能确认任务账本是否全部完成。"
        return "我只完成了部分任务，任务账本里这些还没验收完成：" + "；".join(missing)


class RuntimeObservationCoordinator:
    """Observation side effects: approval/background/todo sync."""

    def __init__(self, runtime: RuntimeView) -> None:
        self.runtime = runtime

    def pause_if_needed_after_observation(self, tool_result: ToolResult) -> bool:
        output = tool_result.output if isinstance(tool_result.output, dict) else {}
        if bool(output.get("approval_required")):
            approval_id = _first_text(
                output.get("approval_id"),
                _nested_get(output, "approval", "approval_id"),
            )
            approval_ids = (
                [approval_id]
                if approval_id
                else list(self.runtime.state.waiting_approval_ids[-3:])
            )
            self.runtime.state.pause(
                reason="approval_required",
                final_text=_approval_pause_reply(output, approval_ids=approval_ids),
                cursor={
                    "type": "approval",
                    "approval_ids": approval_ids,
                    "resume_instruction": (
                        "Runtime will consume the user's confirm/reject message "
                        "directly; do not call approve_pending_action unless the "
                        "user explicitly asks to manage approvals by tool."
                    ),
                },
            )
            return True
        return False

    async def background_pause_if_needed_after_observation(
        self,
        tool_result: ToolResult,
    ) -> bool:
        output = tool_result.output if isinstance(tool_result.output, dict) else {}
        status = normalize_message_text(str(output.get("status", "") or ""))
        task_id = _first_text(
            output.get("task_id"),
            _nested_get(output, "task", "task_id"),
        )
        if status != "background_task_started" or not task_id:
            return False
        actor = self.actor_from_run_context()
        from .superuser_agent.background_tasks import wait_for_observation_event

        event = await wait_for_observation_event(
            task_id=task_id,
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            after_event_id=_first_text(
                _nested_get(output, "observation_event", "event_id"),
                output.get("event_id"),
            ),
            timeout=BACKGROUND_OBSERVATION_WAIT_SECONDS,
            terminal_only=True,
        )
        if event is not None:
            self.append_background_observation_event(event)
            await self.runtime.verify_after_observation()
            self.runtime._persist_state(
                "background_observation_event",
                task_id=task_id,
                event_id=event.event_id,
                status=event.status,
            )
            return False
        self.runtime.state.pause(
            reason="waiting_background_task",
            final_text=(
                f"后台任务已启动，task_id={task_id}。"
                "它还在运行；稍后我可以自动/手动继续查看结果并完成后续步骤。"
            ),
            cursor={
                "type": "background_task",
                "task_ids": [task_id],
                "last_event_ids": list(self.runtime.state.observation_event_ids[-10:]),
                "resume_instruction": (
                    "Wait for ObservationEvent or call background_task_status, "
                    "then continue the task."
                ),
            },
        )
        return True

    def append_background_observation_event(self, event: Any) -> None:
        payload = event.public_payload()
        if event.event_id not in self.runtime.state.observation_event_ids:
            self.runtime.state.observation_event_ids.append(event.event_id)
        for artifact in payload.get("artifacts", []) or []:
            if not isinstance(artifact, dict):
                continue
            artifact_id = normalize_message_text(str(artifact.get("artifact_id") or ""))
            if artifact_id and artifact_id not in self.runtime.state.artifact_refs:
                self.runtime.state.artifact_refs.append(artifact_id)
        self.runtime.state.messages.append(
            LLMMessage.user(
                "Background task observation event:\n"
                + json.dumps(payload, ensure_ascii=False)
                + "\nUse this event as real tool evidence and continue the TaskGraph."
            )
        )
        synthetic = AgentObservation(
            tool_call_id=event.event_id,
            tool_name="background_task",
            task_text=f"background task {event.task_id}",
            ok=event.status == "completed",
            need_continue=event.status
            not in {"completed", "failed", "cancelled", "error"},
            error=event.error or event.stderr_tail
            if event.status != "completed"
            else "",
            artifacts=tuple(payload.get("artifacts", []) or []),
            step=self.runtime.state.step,
            output={
                "ok": event.status == "completed",
                "status": f"background_{event.status}",
                "event_id": event.event_id,
                "task_id": event.task_id,
                "task_text": f"background task {event.task_id}",
                "messages_sent": [],
                "artifacts": list(payload.get("artifacts", []) or []),
                "output_tail": event.output_tail,
                "stderr_tail": event.stderr_tail,
                "error": event.error or event.stderr_tail,
                "returncode": event.returncode,
            },
        )
        self.runtime.state.append_synthetic_observation(
            synthetic,
            timeline_kind="background_observation_event",
            content=payload.get("output_tail", "") or payload.get("stderr_tail", ""),
            metadata={"event": payload},
        )
        self.sync_todos_from_latest_observation()

    def sync_todos_from_latest_observation(self) -> None:
        extra = getattr(self.runtime.run_context, "extra", None)
        if not isinstance(extra, dict) or not bool(extra.get("enable_agent_tools")):
            return
        if not self.runtime.state.observations:
            return
        actor = self.actor_from_run_context()
        observation = self.runtime.state.observations[-1]
        payload = {
            "ok": observation.ok,
            "status": observation.output.get("status", ""),
            "tool_name": observation.tool_name,
            "task_text": observation.task_text,
            "artifacts": list(observation.artifacts),
        }
        try:
            from .superuser_agent.todo_store import update_todo_from_observation

            todo_list = update_todo_from_observation(
                user_id=actor["user_id"],
                session_key=actor["session_key"],
                observation=payload,
            )
        except Exception:
            return
        if todo_list is None:
            return
        self.runtime.state.append_timeline(
            role="system",
            kind="todo_sync",
            metadata={
                "step": self.runtime.state.step,
                "summary": todo_list.public_payload().get("summary", {}),
            },
        )

    def actor_from_run_context(self) -> dict[str, str]:
        extra = getattr(self.runtime.run_context, "extra", None)
        session_key = str(getattr(self.runtime.run_context, "session_id", "") or "")
        user_id = ""
        if isinstance(extra, dict):
            user_id = str(extra.get("actor_user_id", "") or "")
        user_id = user_id or session_key or "unknown"
        return {"user_id": user_id, "session_key": session_key or user_id}


def should_complete_after_plugin_observation(
    runtime: RuntimeView,
    *,
    looks_like_multi_task_turn: bool,
) -> bool:
    """Avoid an extra final LLM turn after visible group plugin output."""

    extra = getattr(runtime.run_context, "extra", None)
    if isinstance(extra, dict) and bool(extra.get("enable_agent_tools")):
        return False
    if runtime.state.task_graph is not None:
        return False
    if runtime.state.task_ledger is not None:
        return False
    if runtime.state.pending_tasks:
        return False
    if looks_like_multi_task_turn:
        return False
    if not runtime.state.observations:
        return False
    recent_commands = [
        observation
        for observation in runtime.state.observations
        if observation.command_id
    ]
    if not recent_commands:
        return False
    if any(observation.need_continue for observation in recent_commands):
        return False
    latest = recent_commands[-1]
    if not latest.ok:
        return False
    output = latest.output or {}
    messages_sent = output.get("messages_sent")
    messages_sent_summary = normalize_message_text(
        str(output.get("messages_sent_summary", "") or "")
    )
    artifacts = output.get("artifacts")
    has_visible_output = (
        bool(output.get("visible_output"))
        or bool(messages_sent_summary)
        or (
            isinstance(messages_sent, list | tuple)
            and any(normalize_message_text(str(item or "")) for item in messages_sent)
        )
    ) or (
        isinstance(artifacts, list | tuple)
        and any(isinstance(item, dict) for item in artifacts)
    )
    if not has_visible_output:
        return False
    return all(observation.ok for observation in recent_commands)


def _normalized_tasks(tasks: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for task in tasks:
        normalized = normalize_message_text(task)
        if normalized and normalized not in result:
            result.append(normalized)
    return result[:8]


def _compact_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[str] = []
    for item in value[: max(1, limit)]:
        text = normalize_message_text(str(item or ""))
        if text:
            result.append(text[:240])
    return result


def _compact_artifacts(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list | tuple):
        return []
    result: list[dict[str, str]] = []
    for item in value[:6]:
        if not isinstance(item, dict):
            continue
        artifact_type = normalize_message_text(str(item.get("type", "") or ""))
        artifact_id = normalize_message_text(str(item.get("artifact_id", "") or ""))
        if artifact_id or artifact_type:
            result.append(
                {
                    "artifact_id": artifact_id,
                    "type": artifact_type,
                    "summary": normalize_message_text(
                        str(item.get("summary", "") or "")
                    )[:200],
                }
            )
    return result


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_text(*values: Any) -> str:
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text:
            return text
    return ""


def _approval_pause_reply(
    output: dict[str, Any],
    *,
    approval_ids: list[str],
) -> str:
    raw_approval = output.get("approval")
    approval: dict[str, Any] = raw_approval if isinstance(raw_approval, dict) else {}
    action = normalize_message_text(str(approval.get("action", "") or ""))
    reason = normalize_message_text(
        str(
            approval.get("reason", "")
            or output.get("reason", "")
            or output.get("status", "")
        )
    )
    approval_text = "、".join(approval_ids) if approval_ids else "未知"
    parts = [f"这个操作需要确认，approval_id={approval_text}。"]
    if action:
        parts.append(f"操作：{action}。")
    if reason:
        parts.append(f"原因：{reason}。")
    parts.append("回复“确认”继续执行，或回复“取消”放弃。")
    return "".join(parts)


__all__ = [
    "BACKGROUND_OBSERVATION_WAIT_SECONDS",
    "SAFE_NO_TOOL_RESULT_REPLY",
    "RuntimeFinalGate",
    "RuntimeObservationCoordinator",
    "RuntimeTaskLedgerCoordinator",
    "should_complete_after_plugin_observation",
]
