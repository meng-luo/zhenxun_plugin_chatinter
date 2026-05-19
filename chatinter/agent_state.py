"""Explicit run state for the ChatInter agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from zhenxun.services.llm import LLMMessage
from zhenxun.services.llm.types.models import LLMToolCall, LLMToolFunction, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .artifact_store import get_artifact_store, summarize_artifact_text
from .route_text import normalize_message_text
from .task_frame import TASK_TEXT_FIELD


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

    @classmethod
    def create(
        cls,
        *,
        trace_id: str,
        session_key: str | None,
        messages: list[LLMMessage],
        tool_map: dict[str, ToolExecutable],
        current_message: str,
        max_steps: int,
        budget_controller: Any | None = None,
        tool_obligation: str = "none",
        tool_obligation_reason: str = "",
        required_tool_names: tuple[str, ...] = (),
    ) -> "AgentRunState":
        state = cls(
            trace_id=trace_id,
            session_key=session_key,
            messages=list(messages),
            tool_map=dict(tool_map),
            max_steps=max(1, int(max_steps or 1)),
            tool_obligation=tool_obligation,
            tool_obligation_reason=normalize_message_text(tool_obligation_reason),
            required_tool_names=tuple(
                normalize_message_text(name) for name in required_tool_names if name
            ),
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

    def append_model_request(self, *, tool_count: int) -> None:
        self.append_timeline(
            role="system",
            kind="model_request",
            metadata={
                "step": self.step,
                "tool_count": tool_count,
                "tool_obligation": self.tool_obligation,
                "tool_obligation_reason": self.tool_obligation_reason,
                "required_tool_count": len(self.required_tool_names),
                "pending_tasks": [task.text for task in self.pending_tasks[-5:]],
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
    ) -> None:
        observation = self._build_observation(
            tool_call=tool_call,
            tool_result=tool_result,
        )
        self.observations.append(observation)
        self._update_task_state(observation)
        self.messages.append(
            LLMMessage.tool_response(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                result=model_payload,
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

    def complete_final(self, final_text: str, *, reason: str) -> None:
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
        if observation.need_continue and observation.remaining_task_hint:
            self.add_pending_tasks(
                [observation.remaining_task_hint],
                source="observation",
                command_id=observation.command_id,
            )


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


def _has_task(tasks: list[PendingTask], text: str) -> bool:
    normalized = normalize_message_text(text)
    return any(task.text == normalized for task in tasks)


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


__all__ = [
    "AgentBudgetState",
    "AgentObservation",
    "AgentRunState",
    "AgentRuntimeResult",
    "AgentRuntimeTimelineItem",
    "CompletedTask",
    "PendingTask",
]
