"""Explicit run state for the ChatInter agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from zhenxun.services.llm import LLMMessage
from zhenxun.services.llm.types.models import LLMToolCall, LLMToolFunction, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .artifact_store import get_artifact_store, summarize_artifact_text
from .provider_capability import ProviderCapabilityAdapter
from .route_text import normalize_message_text
from .runtime_events import emit_runtime_event_from_state
from .task_frame import TASK_TEXT_FIELD
from .task_graph import TaskGraph
from .task_ledger import CapabilityLedger, TaskLedger


@dataclass(frozen=True)
class AgentRuntimeTimelineItem:
    role: str
    kind: str
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PendingTask:
    text: str
    source: str = "observation"
    command_id: str = ""
    step: int = 0


@dataclass(frozen=True)
class CompletedTask:
    text: str
    command_id: str = ""
    rendered_command: str = ""
    matched_plugin: str = ""
    ok: bool = True
    step: int = 0


@dataclass(frozen=True)
class AgentObservation:
    tool_call_id: str
    tool_name: str
    command_id: str = ""
    rendered_command: str = ""
    matched_plugin: str = ""
    task_text: str = ""
    ok: bool = False
    need_continue: bool = False
    remaining_task_hint: str = ""
    error: str = ""
    artifacts: tuple[dict[str, Any], ...] = ()
    step: int = 0
    result: ToolResult | None = None
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentBudgetState:
    classifier_calls: int = 0
    hook_calls: int = 0
    tool_calls: int = 0
    tool_batches: int = 0
    prompt_tokens: int = 0
    durations_ms: dict[str, float] = field(default_factory=dict)

    def add_prompt_tokens(self, estimated_tokens: int) -> None:
        self.prompt_tokens += max(int(estimated_tokens), 0)

    def update_from_controller(self, controller: Any | None) -> None:
        if controller is None:
            return
        snapshot = controller.snapshot()
        self.classifier_calls = snapshot.classifier_calls
        self.hook_calls = snapshot.hook_calls
        self.tool_calls = snapshot.tool_calls
        self.tool_batches = snapshot.tool_batches
        self.prompt_tokens = snapshot.prompt_tokens
        self.durations_ms = dict(snapshot.durations_ms)


@dataclass
class AgentRuntimeResult:
    final_text: str
    run_id: str = ""
    trace_id: str = ""
    status: str = "completed"
    paused_reason: str = ""
    tool_results: tuple[ToolResult, ...] = ()
    timeline: tuple[AgentRuntimeTimelineItem, ...] = ()
    messages: tuple[LLMMessage, ...] = ()
    stop_reason: str = "final_response"
    steps: int = 0
    budget: AgentBudgetState = field(default_factory=AgentBudgetState)
    pending_tasks: tuple[PendingTask, ...] = ()
    completed_tasks: tuple[CompletedTask, ...] = ()


@dataclass
class AgentRunState:
    trace_id: str
    session_key: str | None
    messages: list[LLMMessage]
    tool_map: dict[str, ToolExecutable]
    run_id: str = ""
    status: str = "running"
    paused_reason: str = ""
    resume_cursor: dict[str, Any] = field(default_factory=dict)
    waiting_approval_ids: list[str] = field(default_factory=list)
    background_task_ids: list[str] = field(default_factory=list)
    observation_event_ids: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    observations: list[AgentObservation] = field(default_factory=list)
    pending_tasks: list[PendingTask] = field(default_factory=list)
    completed_tasks: list[CompletedTask] = field(default_factory=list)
    stop_reason: str = "running"
    recovery_action: str | None = None
    step: int = 0
    max_steps: int = 5
    budget: AgentBudgetState = field(default_factory=AgentBudgetState)
    timeline: list[AgentRuntimeTimelineItem] = field(default_factory=list)
    final_text: str = ""
    tool_obligation: str = "none"
    tool_obligation_reason: str = ""
    required_tool_names: tuple[str, ...] = ()
    direct_answer_interceptions: int = 0
    final_validation_interceptions: int = 0
    coverage_interceptions: int = 0
    task_graph: TaskGraph | None = None
    task_graph_interceptions: int = 0
    capability_ledger: CapabilityLedger = field(default_factory=CapabilityLedger)
    task_ledger: TaskLedger | None = None
    agent_complexity_mode: str = "standard"
    agent_complexity_reason: str = ""

    @classmethod
    def create(
        cls,
        *,
        trace_id: str,
        run_id: str | None = None,
        session_key: str | None,
        messages: list[LLMMessage],
        tool_map: dict[str, ToolExecutable],
        current_message: str,
        max_steps: int,
        budget_controller: Any | None = None,
        tool_obligation: str = "none",
        tool_obligation_reason: str = "",
        required_tool_names: tuple[str, ...] = (),
        agent_complexity_mode: str = "standard",
        agent_complexity_reason: str = "",
    ) -> "AgentRunState":
        state = cls(
            trace_id=trace_id,
            run_id=normalize_message_text(run_id or trace_id),
            session_key=session_key,
            messages=list(messages),
            tool_map=dict(tool_map),
            max_steps=max(1, int(max_steps or 1)),
            tool_obligation=tool_obligation,
            tool_obligation_reason=normalize_message_text(tool_obligation_reason),
            required_tool_names=tuple(
                normalize_message_text(name) for name in required_tool_names if name
            ),
            agent_complexity_mode=normalize_message_text(agent_complexity_mode)
            or "standard",
            agent_complexity_reason=normalize_message_text(agent_complexity_reason),
        )
        state.capture_budget(budget_controller)
        state.append_timeline(
            role="user",
            kind="current_user",
            content=current_message,
        )
        return state

    def start_step(self) -> int:
        self.step += 1
        return self.step

    def append_timeline(
        self,
        *,
        role: str,
        kind: str,
        content: str = "",
        tool_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.timeline.append(
            AgentRuntimeTimelineItem(
                role=role,
                kind=kind,
                content=normalize_message_text(content),
                tool_name=tool_name,
                metadata=dict(metadata or {}),
            )
        )
        _emit_timeline_runtime_event(
            self,
            role=role,
            kind=kind,
            content=content,
            tool_name=tool_name,
            metadata=dict(metadata or {}),
        )

    def append_model_request(self, *, tool_count: int) -> None:
        task_graph_summary: dict[str, Any] = {}
        if self.task_graph is not None:
            task_graph_summary = {
                "graph_id": self.task_graph.graph_id,
                "status": self.task_graph.status,
                "incomplete_tasks": [
                    {"task_id": task.task_id, "goal": task.goal}
                    for task in self.task_graph.incomplete_tasks[:5]
                ],
            }
        self.append_timeline(
            role="system",
            kind="model_request",
            metadata={
                "run_id": self.run_id,
                "status": self.status,
                "step": self.step,
                "tool_count": tool_count,
                "tool_obligation": self.tool_obligation,
                "tool_obligation_reason": self.tool_obligation_reason,
                "required_tool_count": len(self.required_tool_names),
                "agent_complexity_mode": self.agent_complexity_mode,
                "agent_complexity_reason": self.agent_complexity_reason,
                "pending_tasks": [task.text for task in self.pending_tasks[-5:]],
                "task_graph": task_graph_summary,
                "task_ledger": self.task_ledger.to_public_payload()
                if self.task_ledger is not None
                else {},
                "capability_ledger": self.capability_ledger.public_entries(limit=12),
            },
        )

    def append_tool_calls(
        self,
        tool_calls: list[LLMToolCall],
        *,
        response_text: str,
    ) -> None:
        history_tool_calls = _compact_tool_calls_for_history(
            tool_calls,
            trace_id=self.trace_id,
        )
        self.tool_calls.extend(history_tool_calls)
        self.messages.append(
            LLMMessage.assistant_tool_calls(
                history_tool_calls,
                response_text,
            )
        )
        for tool_call in tool_calls:
            self.append_timeline(
                role="assistant",
                kind="tool_call",
                tool_name=str(tool_call.function.name or ""),
                metadata={
                    "step": self.step,
                    "arguments": _compact_tool_arguments(
                        tool_call.function.arguments,
                        trace_id=self.trace_id,
                        tool_name=str(tool_call.function.name or ""),
                    ),
                },
            )

    def append_tool_observation(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
        model_payload: dict[str, Any],
        provider_adapter: ProviderCapabilityAdapter | None = None,
    ) -> None:
        observation = self._build_observation(
            tool_call=tool_call,
            tool_result=tool_result,
        )
        self.observations.append(observation)
        _emit_observation_runtime_event(self, observation)
        self._update_resume_refs(observation)
        self._update_capability_ledger(observation)
        self._update_task_state(observation)
        self.messages.append(
            (
                provider_adapter.tool_result_message(
                    tool_call=tool_call,
                    function_name=tool_call.function.name,
                    result=model_payload,
                )
                if provider_adapter is not None
                else LLMMessage.tool_response(
                    tool_call_id=tool_call.id,
                    function_name=tool_call.function.name,
                    result=model_payload,
                )
            )
        )
        self.append_timeline(
            role="tool",
            kind="tool_result",
            content=_observation_content(observation),
            tool_name=observation.tool_name,
            metadata={
                "step": self.step,
                "output": observation.output or tool_result.output,
                "pending_tasks": [task.text for task in self.pending_tasks[-5:]],
            },
        )

    def append_synthetic_observation(
        self,
        observation: AgentObservation,
        *,
        timeline_kind: str,
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.observations.append(observation)
        _emit_observation_runtime_event(self, observation)
        self._update_resume_refs(observation)
        self._update_capability_ledger(observation)
        self._update_task_state(observation)
        self.append_timeline(
            role="tool",
            kind=timeline_kind,
            content=content,
            tool_name=observation.tool_name,
            metadata={
                "step": self.step,
                "output": observation.output,
                **dict(metadata or {}),
            },
        )

    def append_guardrail_observation(
        self,
        payload: dict[str, Any],
        *,
        as_message: bool = True,
        record_timeline: bool = True,
    ) -> None:
        reason = normalize_message_text(str(payload.get("guardrail_reason", "")))
        message = normalize_message_text(
            str(payload.get("message", "") or payload.get("error", ""))
        )
        self.recovery_action = reason or self.recovery_action
        if as_message:
            self.messages.append(
                LLMMessage.user(
                    "Runtime guardrail observation:\n"
                    + json.dumps(payload, ensure_ascii=False)
                )
            )
        if not record_timeline:
            return
        self.append_timeline(
            role="system",
            kind="runtime_guardrail",
            content=message or reason,
            metadata={
                "step": self.step,
                "payload": dict(payload),
                "as_message": as_message,
            },
        )

    def replace_pending_tasks(
        self,
        tasks: list[str] | tuple[str, ...],
        *,
        source: str,
        command_id: str = "",
    ) -> None:
        self.pending_tasks = []
        self.add_pending_tasks(tasks, source=source, command_id=command_id)

    def add_pending_tasks(
        self,
        tasks: list[str] | tuple[str, ...],
        *,
        source: str,
        command_id: str = "",
    ) -> None:
        for text in tasks:
            normalized = normalize_message_text(text)
            if not normalized or _has_task(self.pending_tasks, normalized):
                continue
            self.pending_tasks.append(
                PendingTask(
                    text=normalized,
                    source=source,
                    command_id=normalize_message_text(command_id),
                    step=self.step,
                )
            )

    def set_task_graph(self, graph: TaskGraph | None, *, source: str) -> None:
        self.task_graph = graph
        if graph is None:
            return
        self.append_timeline(
            role="system",
            kind="task_graph",
            metadata={
                "source": normalize_message_text(source),
                "graph": graph.to_public_payload(),
            },
        )

    def incomplete_task_goals(self) -> list[str]:
        if self.task_ledger is not None:
            return self.task_ledger.incomplete_goals
        if self.task_graph is None:
            return []
        return [task.goal for task in self.task_graph.incomplete_tasks if task.goal]

    def set_task_ledger(self, ledger: TaskLedger | None, *, source: str) -> None:
        self.task_ledger = ledger
        if ledger is None:
            return
        self.append_timeline(
            role="system",
            kind="task_ledger",
            metadata={
                "source": normalize_message_text(source),
                "ledger": ledger.to_public_payload(),
            },
        )

    def refresh_capability_ledger(self, tools: list[dict[str, Any]]) -> None:
        self.capability_ledger.refresh_tools(tools)

    def transition_force_final(self, reason: str) -> None:
        self.recovery_action = reason
        self.stop_reason = reason
        self.messages.append(
            LLMMessage.user(
                "工具调用已经结束或达到上限。请不要再调用工具，"
                "根据已经完成的工具结果直接给用户一个简短最终回复。"
            )
        )
        self.append_timeline(
            role="system",
            kind="recovery_action",
            content=reason,
            metadata={"step": self.step},
        )

    def pause(
        self,
        *,
        reason: str,
        cursor: dict[str, Any] | None = None,
        final_text: str = "",
    ) -> None:
        self.status = "paused"
        self.paused_reason = normalize_message_text(reason)
        self.stop_reason = f"paused:{self.paused_reason or 'unknown'}"
        self.resume_cursor = dict(cursor or {})
        if final_text:
            self.final_text = normalize_message_text(final_text)
        self.append_timeline(
            role="system",
            kind="agent_paused",
            content=self.paused_reason,
            metadata={
                "step": self.step,
                "resume_cursor": dict(self.resume_cursor),
                "waiting_approval_ids": list(self.waiting_approval_ids),
                "background_task_ids": list(self.background_task_ids),
                "observation_event_ids": list(self.observation_event_ids[-20:]),
                "artifact_refs": list(self.artifact_refs[-20:]),
            },
        )

    def resume(
        self,
        *,
        reason: str = "manual_resume",
    ) -> None:
        self.status = "running"
        self.paused_reason = ""
        self.stop_reason = "running"
        self.resume_cursor = {}
        self.append_timeline(
            role="system",
            kind="agent_resumed",
            content=normalize_message_text(reason),
            metadata={"step": self.step},
        )

    def cancel(self, *, reason: str = "") -> None:
        self.status = "cancelled"
        self.paused_reason = ""
        self.stop_reason = "cancelled"
        self.recovery_action = normalize_message_text(reason) or "cancelled_by_user"
        self.append_timeline(
            role="system",
            kind="agent_cancelled",
            content=self.recovery_action or "cancelled_by_user",
            metadata={"step": self.step},
        )

    def complete_final(self, final_text: str, *, reason: str) -> None:
        self.status = "completed"
        self.paused_reason = ""
        self.final_text = normalize_message_text(final_text)
        self.stop_reason = reason
        self.messages.append(LLMMessage.assistant_text_response(self.final_text))
        metadata: dict[str, Any] = {"step": self.step}
        if self.recovery_action:
            metadata["forced_final"] = self.recovery_action
        self.append_timeline(
            role="assistant",
            kind="assistant_text",
            content=self.final_text,
            metadata=metadata,
        )

    def record_prompt_use(
        self,
        *,
        estimated_tokens: int,
        budget_controller: Any | None,
    ) -> None:
        if budget_controller is None:
            self.budget.add_prompt_tokens(estimated_tokens)
            return
        budget_controller.record_prompt_use(estimated_tokens=estimated_tokens)
        self.capture_budget(budget_controller)

    def capture_budget(self, budget_controller: Any | None) -> None:
        self.budget.update_from_controller(budget_controller)

    def to_result(self) -> AgentRuntimeResult:
        return AgentRuntimeResult(
            final_text=self.final_text,
            run_id=self.run_id,
            trace_id=self.trace_id,
            status=self.status,
            paused_reason=self.paused_reason,
            tool_results=tuple(
                observation.result
                for observation in self.observations
                if observation.result is not None
            ),
            timeline=tuple(self.timeline),
            messages=tuple(self.messages),
            stop_reason=self.stop_reason,
            steps=self.step,
            budget=self.budget,
            pending_tasks=tuple(self.pending_tasks),
            completed_tasks=tuple(self.completed_tasks),
        )

    def _build_observation(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
    ) -> AgentObservation:
        output = tool_result.output if isinstance(tool_result.output, dict) else {}
        arguments = _parse_tool_arguments(tool_call.function.arguments)
        task_text = normalize_message_text(str(output.get("task_text", "")))
        if isinstance(arguments, dict):
            task_text = task_text or normalize_message_text(
                str(arguments.get(TASK_TEXT_FIELD) or "")
            )
        artifacts = output.get("artifacts")
        artifact_payloads = (
            tuple(
                dict(item)
                for item in artifacts
                if isinstance(item, dict) and item.get("artifact_id")
            )
            if isinstance(artifacts, list | tuple)
            else ()
        )
        return AgentObservation(
            tool_call_id=tool_call.id,
            tool_name=str(tool_call.function.name or ""),
            command_id=normalize_message_text(str(output.get("command_id", ""))),
            rendered_command=normalize_message_text(
                str(output.get("rendered_command", ""))
            ),
            matched_plugin=normalize_message_text(str(output.get("matched_plugin", ""))),
            task_text=task_text,
            ok=bool(output.get("ok")),
            need_continue=bool(output.get("need_continue")),
            remaining_task_hint=normalize_message_text(
                str(output.get("remaining_task_hint", ""))
            ),
            error=normalize_message_text(str(output.get("error", ""))),
            artifacts=artifact_payloads,
            step=self.step,
            result=tool_result,
            output=dict(output),
        )

    def _update_task_state(self, observation: AgentObservation) -> None:
        changed = False
        if self.task_ledger is not None and observation.ok:
            self.task_ledger.apply_coverage(
                covered_task_ids=_covered_ledger_task_ids(
                    self.task_ledger,
                    observation,
                ),
                unsupported_tasks=[],
                reason="observation_update",
            )
            changed = True
        if observation.task_text:
            self.pending_tasks = [
                task
                for task in self.pending_tasks
                if not _task_covered_by_observation(task, observation)
            ]
            self.completed_tasks.append(
                CompletedTask(
                    text=observation.task_text,
                    command_id=observation.command_id,
                    rendered_command=observation.rendered_command,
                    matched_plugin=observation.matched_plugin,
                    ok=observation.ok,
                    step=observation.step,
                )
            )
            changed = True
        if observation.need_continue and observation.remaining_task_hint:
            self.add_pending_tasks(
                [observation.remaining_task_hint],
                source="observation",
                command_id=observation.command_id,
            )
            changed = True
        if changed:
            _emit_task_state_runtime_event(self, observation)

    def _update_capability_ledger(self, observation: AgentObservation) -> None:
        self.capability_ledger.record_observation(
            tool_name=observation.tool_name,
            command_id=observation.command_id,
            plugin=observation.matched_plugin,
            ok=observation.ok,
            task_id=_best_matching_ledger_task_id(self.task_ledger, observation),
            error=observation.error,
        )

    def _update_resume_refs(self, observation: AgentObservation) -> None:
        output = observation.output or {}
        event_id = _first_text(output.get("event_id"))
        if event_id and event_id not in self.observation_event_ids:
            self.observation_event_ids.append(event_id)

        approval_id = _first_text(
            output.get("approval_id"),
            _nested_get(output, "approval", "approval_id"),
        )
        if approval_id and approval_id not in self.waiting_approval_ids:
            self.waiting_approval_ids.append(approval_id)

        task_id = _first_text(
            output.get("task_id"),
            _nested_get(output, "task", "task_id"),
        )
        if task_id and task_id not in self.background_task_ids:
            self.background_task_ids.append(task_id)

        event = output.get("observation_event")
        if isinstance(event, dict):
            nested_event_id = _first_text(event.get("event_id"))
            if nested_event_id and nested_event_id not in self.observation_event_ids:
                self.observation_event_ids.append(nested_event_id)
        nested_event = output.get("event")
        if isinstance(nested_event, dict):
            nested_event_id = _first_text(nested_event.get("event_id"))
            if nested_event_id and nested_event_id not in self.observation_event_ids:
                self.observation_event_ids.append(nested_event_id)

        artifact_groups = [output.get("artifacts")]
        if isinstance(event, dict):
            artifact_groups.append(event.get("artifacts"))
        if isinstance(nested_event, dict):
            artifact_groups.append(nested_event.get("artifacts"))
        for artifacts in artifact_groups:
            if not isinstance(artifacts, list | tuple):
                continue
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                artifact_id = normalize_message_text(str(item.get("artifact_id") or ""))
                if artifact_id and artifact_id not in self.artifact_refs:
                    self.artifact_refs.append(artifact_id)
        artifact_id = _first_text(
            output.get("artifact_id"),
            _nested_get(output, "artifact", "artifact_id"),
        )
        if artifact_id and artifact_id not in self.artifact_refs:
            self.artifact_refs.append(artifact_id)
        for artifact in observation.artifacts:
            artifact_id = normalize_message_text(str(artifact.get("artifact_id") or ""))
            if artifact_id and artifact_id not in self.artifact_refs:
                self.artifact_refs.append(artifact_id)


def _parse_tool_arguments(arguments: str) -> dict[str, Any] | str:
    text = str(arguments or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return text
    return value if isinstance(value, dict) else {"value": value}


def _compact_tool_calls_for_history(
    tool_calls: list[LLMToolCall],
    *,
    trace_id: str,
) -> list[LLMToolCall]:
    return [
        LLMToolCall(
            id=tool_call.id,
            function=LLMToolFunction(
                name=tool_call.function.name,
                arguments=_compact_tool_argument_string(
                    tool_call.function.arguments,
                    trace_id=trace_id,
                    tool_name=tool_call.function.name,
                ),
            ),
            thought_signature=tool_call.thought_signature,
        )
        for tool_call in tool_calls
    ]


def _compact_tool_arguments(
    arguments: str,
    *,
    trace_id: str,
    tool_name: str,
) -> dict[str, Any] | str:
    compacted = _compact_tool_argument_string(
        arguments,
        trace_id=trace_id,
        tool_name=tool_name,
    )
    return _parse_tool_arguments(compacted)


def _compact_tool_argument_string(
    arguments: str,
    *,
    trace_id: str,
    tool_name: str,
) -> str:
    text = str(arguments or "")
    if len(text) <= 900:
        return text
    ref = get_artifact_store().store_text(
        text,
        artifact_type="text",
        trace_id=trace_id,
        source=f"tool_call:{tool_name}:arguments",
        force_file=True,
    )
    if ref is None:
        return summarize_artifact_text(text, limit=360)
    payload = {
        "artifact_id": ref.artifact_id,
        "summary": ref.summary,
        "note": "tool arguments were compressed; original arguments are stored in ArtifactStore",
    }
    return json.dumps(payload, ensure_ascii=False)


def _observation_content(observation: AgentObservation) -> str:
    summary = normalize_message_text(
        str(observation.output.get("messages_sent_summary", "") or "")
    )
    if summary:
        return summary
    messages_sent = observation.output.get("messages_sent")
    if isinstance(messages_sent, list):
        content = "\n".join(
            normalize_message_text(str(item or ""))
            for item in messages_sent[:8]
            if normalize_message_text(str(item or ""))
        )
        if content:
            return content
    if observation.artifacts:
        content = "\n".join(
            normalize_message_text(str(item.get("summary", "") or ""))
            for item in observation.artifacts[:6]
            if normalize_message_text(str(item.get("summary", "") or ""))
        )
        if content:
            return content
    if observation.remaining_task_hint or observation.error:
        return observation.remaining_task_hint or observation.error
    if observation.result:
        return normalize_message_text(str(observation.result.display_content or ""))
    return ""


def _emit_timeline_runtime_event(
    state: AgentRunState,
    *,
    role: str,
    kind: str,
    content: str,
    tool_name: str,
    metadata: dict[str, Any],
) -> None:
    event_kind, event_status = _runtime_kind_status_from_timeline(kind)
    emit_runtime_event_from_state(
        state,
        kind=event_kind,
        status=event_status,
        source=f"timeline:{kind}",
        summary=content or tool_name or kind,
        payload={
            "role": role,
            "timeline_kind": kind,
            "tool_name": tool_name,
            "metadata": metadata,
        },
        artifacts=_artifacts_from_payload(metadata),
        related_ids=_related_ids_from_payload(metadata),
    )


def _emit_observation_runtime_event(
    state: AgentRunState,
    observation: AgentObservation,
) -> None:
    emit_runtime_event_from_state(
        state,
        kind="tool_observation" if observation.tool_name else "observation",
        status="completed" if observation.ok else "failed",
        source=f"tool:{observation.tool_name or 'synthetic'}",
        summary=_observation_content(observation),
        payload={
            "tool_call_id": observation.tool_call_id,
            "tool_name": observation.tool_name,
            "command_id": observation.command_id,
            "rendered_command": observation.rendered_command,
            "matched_plugin": observation.matched_plugin,
            "task_text": observation.task_text,
            "ok": observation.ok,
            "need_continue": observation.need_continue,
            "remaining_task_hint": observation.remaining_task_hint,
            "error": observation.error,
            "output": observation.output,
        },
        artifacts=list(observation.artifacts),
        related_ids={
            "tool_call_id": observation.tool_call_id,
            "command_id": observation.command_id,
            "background_task_id": _first_text(
                observation.output.get("task_id"),
                _nested_get(observation.output, "task", "task_id"),
            ),
            "approval_id": _first_text(
                observation.output.get("approval_id"),
                _nested_get(observation.output, "approval", "approval_id"),
            ),
            "observation_event_id": _first_text(
                observation.output.get("event_id"),
                _nested_get(observation.output, "event", "event_id"),
                _nested_get(observation.output, "observation_event", "event_id"),
            ),
        },
    )


def _emit_task_state_runtime_event(
    state: AgentRunState,
    observation: AgentObservation,
) -> None:
    payload: dict[str, Any] = {
        "source_observation": {
            "tool_name": observation.tool_name,
            "command_id": observation.command_id,
            "task_text": observation.task_text,
            "ok": observation.ok,
        },
        "pending_tasks": [
            {"text": task.text, "source": task.source, "command_id": task.command_id}
            for task in state.pending_tasks[-20:]
        ],
        "completed_tasks": [
            {
                "text": task.text,
                "command_id": task.command_id,
                "rendered_command": task.rendered_command,
                "matched_plugin": task.matched_plugin,
                "ok": task.ok,
            }
            for task in state.completed_tasks[-20:]
        ],
    }
    if state.task_ledger is not None:
        payload["task_ledger"] = state.task_ledger.to_public_payload()
    if state.task_graph is not None:
        payload["task_graph"] = state.task_graph.to_public_payload()
    emit_runtime_event_from_state(
        state,
        kind="task_ledger" if state.task_ledger is not None else "task_graph",
        status="progress",
        source="task_state:observation_update",
        summary=observation.task_text or observation.command_id or observation.tool_name,
        payload=payload,
        artifacts=list(observation.artifacts),
        related_ids={
            "tool_call_id": observation.tool_call_id,
            "command_id": observation.command_id,
        },
    )


def _runtime_kind_status_from_timeline(
    timeline_kind: str,
) -> tuple[str, str]:
    kind = normalize_message_text(timeline_kind)
    if kind == "model_request":
        return "model_request", "started"
    if kind == "tool_call":
        return "tool_call", "started"
    if kind in {"tool_result", "background_observation_event"}:
        return "tool_observation", "completed"
    if kind in {"task_graph", "task_graph_verification"}:
        return "task_graph", "progress"
    if kind.startswith("task_ledger"):
        return "task_ledger", "progress"
    if kind == "runtime_guardrail":
        return "guardrail", "blocked"
    if kind in {"agent_paused"}:
        return "agent_run", "waiting"
    if kind in {"agent_resumed"}:
        return "agent_run", "started"
    if kind in {"agent_cancelled"}:
        return "agent_run", "cancelled"
    if kind == "assistant_text":
        return "agent_run", "completed"
    if kind == "todo_sync":
        return "todo", "progress"
    return "system", "info"


def _artifacts_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    values = [payload.get("artifacts")]
    output = payload.get("output")
    if isinstance(output, dict):
        values.append(output.get("artifacts"))
    event = payload.get("event")
    if isinstance(event, dict):
        values.append(event.get("artifacts"))
    for value in values:
        if not isinstance(value, list | tuple):
            continue
        for item in value:
            if isinstance(item, dict) and item.get("artifact_id"):
                result.append(dict(item))
    return result


def _related_ids_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    output = payload.get("output")
    output = output if isinstance(output, dict) else {}
    event = payload.get("event")
    event = event if isinstance(event, dict) else {}
    return {
        "approval_id": _first_text(
            output.get("approval_id"),
            _nested_get(output, "approval", "approval_id"),
        ),
        "background_task_id": _first_text(
            output.get("task_id"),
            _nested_get(output, "task", "task_id"),
            event.get("task_id"),
        ),
        "observation_event_id": _first_text(
            output.get("event_id"),
            _nested_get(output, "event", "event_id"),
            _nested_get(output, "observation_event", "event_id"),
            event.get("event_id"),
        ),
    }


def _has_task(tasks: list[PendingTask], text: str) -> bool:
    normalized = normalize_message_text(text)
    return any(task.text == normalized for task in tasks)


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_text(*values: Any) -> str:
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text:
            return text
    return ""


def _same_or_nested_task(left: str, right: str) -> bool:
    first = normalize_message_text(left)
    second = normalize_message_text(right)
    return bool(first and second and (first in second or second in first))


def _task_covered_by_observation(
    task: PendingTask,
    observation: AgentObservation,
) -> bool:
    if not observation.ok:
        return False
    task_text = normalize_message_text(task.text)
    completed_text = normalize_message_text(observation.task_text)
    rendered_command = normalize_message_text(observation.rendered_command)
    command_id = normalize_message_text(observation.command_id)
    if completed_text and _same_or_nested_task(task_text, completed_text):
        return True
    if rendered_command and _same_or_nested_task(task_text, rendered_command):
        return True
    return bool(task.command_id and task.command_id == command_id)


def _covered_ledger_task_ids(
    ledger: TaskLedger,
    observation: AgentObservation,
) -> list[str]:
    task_id = _best_matching_ledger_task_id(ledger, observation)
    return [task_id] if task_id else []


def _best_matching_ledger_task_id(
    ledger: TaskLedger | None,
    observation: AgentObservation,
) -> str:
    if ledger is None:
        return ""
    candidates = [
        observation.task_text,
        observation.rendered_command,
        observation.command_id,
        observation.matched_plugin,
    ]
    for task in ledger.tasks:
        goal = normalize_message_text(task.goal)
        if any(_same_or_nested_task(goal, item) for item in candidates if item):
            return task.task_id
        command_id = normalize_message_text(observation.command_id)
        if command_id and command_id in {
            normalize_message_text(item) for item in task.expected_capabilities
        }:
            return task.task_id
    return ""


__all__ = [
    "AgentBudgetState",
    "AgentObservation",
    "AgentRunState",
    "AgentRuntimeResult",
    "AgentRuntimeTimelineItem",
    "CompletedTask",
    "PendingTask",
]
