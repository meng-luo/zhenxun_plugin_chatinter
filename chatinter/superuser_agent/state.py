"""Explicit run state for the ChatInter agent loop."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
import time
from typing import Any

from ..artifact_store import get_artifact_store, summarize_artifact_text
from ..llm_compat import (
    LLMMessage,
    LLMToolCall,
    LLMToolFunction,
    ToolExecutable,
    ToolResult,
)
from ..route_text import normalize_message_text, normalize_reply_text

_TASK_TEXT_FIELD = "task_text"
_MESSAGE_SOURCE_KEY = "chatinter_message_source"
_RUNTIME_CONTROL_SOURCE = "runtime_control"
_GROUP_WITH_NEXT_USER_KEY = "chatinter_group_with_next_user"


def runtime_control_message(
    content: str,
    *,
    group_with_next_user: bool = False,
) -> LLMMessage:
    metadata = {_MESSAGE_SOURCE_KEY: _RUNTIME_CONTROL_SOURCE}
    if group_with_next_user:
        metadata[_GROUP_WITH_NEXT_USER_KEY] = True
    return LLMMessage(
        role="user",
        content=content,
        metadata=metadata,
    )


def is_runtime_control_message(message: LLMMessage) -> bool:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    return (
        message.role == "user"
        and metadata.get(_MESSAGE_SOURCE_KEY) == _RUNTIME_CONTROL_SOURCE
    )


def groups_with_next_user_message(message: LLMMessage) -> bool:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    return is_runtime_control_message(message) and (
        metadata.get(_GROUP_WITH_NEXT_USER_KEY) is True
    )


@dataclass(frozen=True)
class AgentRunBudget:
    max_steps: int
    cost_checkpoint_tokens: int
    scenario: str = "superuser_agent"

    def to_metadata(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "max_steps": self.max_steps,
            "cost_checkpoint_tokens": self.cost_checkpoint_tokens,
        }


def resolve_superuser_agent_run_budget() -> AgentRunBudget:
    from ..config import AGENT_COST_CHECKPOINT_TOKENS, AGENT_STEP_BUDGETS

    steps = (AGENT_STEP_BUDGETS.get("superuser_agent") or {}).get("standard") or 90
    tokens = AGENT_COST_CHECKPOINT_TOKENS.get("superuser_agent") or 0
    return AgentRunBudget(
        max_steps=max(int(steps), 1),
        cost_checkpoint_tokens=max(int(tokens), 0),
    )


@dataclass(frozen=True)
class AgentRuntimeMetric:
    role: str
    kind: str
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    observation: AgentObservation | None = None


AgentRuntimeTimelineItem = AgentRuntimeMetric


@dataclass
class ToolExecutionRecord:
    tool_call_id: str
    tool_name: str
    fingerprint: str
    status: str = "started"
    step: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    result_status: str = ""


@dataclass(frozen=True)
class AgentObservation:
    tool_call_id: str
    tool_name: str
    command_id: str = ""
    rendered_command: str = ""
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
    tool_calls: int = 0
    tool_batches: int = 0
    run_input_tokens: int = 0
    run_output_tokens: int = 0
    current_context_tokens: int = 0
    last_usage_message_count: int = 0
    last_usage_schema_tokens: int = 0
    model_calls: int = 0
    durations_ms: dict[str, float] = field(default_factory=dict)

    def record_model_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        message_count: int = 0,
        schema_tokens: int = 0,
        update_context: bool = True,
    ) -> None:
        input_tokens = max(int(input_tokens), 0)
        output_tokens = max(int(output_tokens), 0)
        self.run_input_tokens += input_tokens
        self.run_output_tokens += output_tokens
        if update_context:
            self.current_context_tokens = input_tokens
            self.last_usage_message_count = max(int(message_count or 0), 0)
            self.last_usage_schema_tokens = max(int(schema_tokens or 0), 0)
        self.model_calls += 1


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
    delivery_complete: bool = False
    final_source: str = ""


@dataclass
class AgentRunState:
    """One execution turn within the durable conversation identified by run_id."""

    trace_id: str
    session_key: str | None
    messages: list[LLMMessage]
    tool_map: dict[str, ToolExecutable]
    run_id: str = ""
    status: str = "running"
    paused_reason: str = ""
    pending_approval: str = ""
    artifact_refs: list[str] = field(default_factory=list)
    plan_items: list[dict[str, str]] = field(default_factory=list)
    tool_executions: list[ToolExecutionRecord] = field(default_factory=list)
    stop_reason: str = "running"
    step: int = 0
    max_steps: int = 5
    cost_checkpoint_tokens: int = 0
    budget: AgentBudgetState = field(default_factory=AgentBudgetState)
    metrics: list[AgentRuntimeMetric] = field(default_factory=list)
    final_text: str = ""
    delivery_complete: bool = False
    final_source: str = ""

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
        cost_checkpoint_tokens: int = 0,
    ) -> "AgentRunState":
        state = cls(
            trace_id=trace_id,
            run_id=normalize_message_text(run_id or trace_id),
            session_key=session_key,
            messages=list(messages),
            tool_map=dict(tool_map),
            max_steps=max(1, int(max_steps or 1)),
            cost_checkpoint_tokens=max(int(cost_checkpoint_tokens or 0), 0),
        )
        state.append_metric(
            role="user",
            kind="current_user",
            content=current_message,
        )
        return state

    @classmethod
    def start_new_turn(
        cls,
        previous: "AgentRunState",
        *,
        trace_id: str,
        tool_map: dict[str, ToolExecutable],
        current_message: str,
        max_steps: int,
        cost_checkpoint_tokens: int = 0,
    ) -> "AgentRunState":
        """Start a fresh turn while retaining durable conversation context."""

        state = cls.create(
            trace_id=trace_id,
            run_id=previous.run_id,
            session_key=previous.session_key,
            messages=[*previous.messages, LLMMessage.user(current_message)],
            tool_map=tool_map,
            current_message=current_message,
            max_steps=max_steps,
            cost_checkpoint_tokens=cost_checkpoint_tokens,
        )
        state.artifact_refs = list(previous.artifact_refs)
        state.plan_items = [dict(item) for item in previous.plan_items]
        state.budget.current_context_tokens = previous.budget.current_context_tokens
        state.budget.last_usage_message_count = previous.budget.last_usage_message_count
        state.budget.last_usage_schema_tokens = previous.budget.last_usage_schema_tokens
        state.tool_executions = [
            replace(record)
            for record in previous.tool_executions
            if record.status in {"started", "uncertain"}
        ]
        return state

    def start_step(self) -> int:
        self.step += 1
        return self.step

    def cost_checkpoint_reached(self) -> bool:
        if self.cost_checkpoint_tokens <= 0:
            return False
        return (
            self.budget.run_input_tokens + self.budget.run_output_tokens
            >= self.cost_checkpoint_tokens
        )

    def append_metric(
        self,
        *,
        role: str,
        kind: str,
        content: str = "",
        tool_name: str = "",
        metadata: dict[str, Any] | None = None,
        observation: AgentObservation | None = None,
    ) -> None:
        self.metrics.append(
            AgentRuntimeMetric(
                role=role,
                kind=kind,
                content=normalize_message_text(content),
                tool_name=tool_name,
                metadata=dict(metadata or {}),
                observation=observation,
            )
        )

    def append_model_request(
        self,
        *,
        selected_tool_count: int,
        schema_chars: int,
    ) -> None:
        self.append_metric(
            role="system",
            kind="model_request",
            metadata={
                "run_id": self.run_id,
                "status": self.status,
                "step": self.step,
                "tool_count": selected_tool_count,
                "selected_tool_count": selected_tool_count,
                "schema_chars": schema_chars,
            },
        )

    def append_tool_calls(
        self,
        tool_calls: list[LLMToolCall],
        *,
        response_text: str,
        response_thought_text: str | None = None,
        response_content_parts: list[Any] | None = None,
        source_model: str | None = None,
        source_api_type: str | None = None,
        provider_replay_kind: str | None = None,
        provider_replay_payload: list[dict[str, Any]] | None = None,
    ) -> None:
        history_tool_calls: list[LLMToolCall] = []
        compacted_arguments: list[str] = []
        for tool_call in tool_calls:
            arguments, artifact_id = _compact_tool_argument_string(
                tool_call.function.arguments,
                trace_id=self.trace_id,
                tool_name=tool_call.function.name,
            )
            append_artifact_refs(self.artifact_refs, (artifact_id,))
            compacted_arguments.append(arguments)
            history_tool_calls.append(
                LLMToolCall(
                    id=tool_call.id,
                    function=LLMToolFunction(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                    thought_signature=tool_call.thought_signature,
                    metadata=(
                        copy.deepcopy(tool_call.metadata)
                        if tool_call.metadata
                        else None
                    ),
                )
            )
        self.messages.append(
            LLMMessage.assistant_tool_calls(
                history_tool_calls,
                response_text,
                thought_text=response_thought_text,
                content_parts=response_content_parts,
                source_model=source_model,
                source_api_type=source_api_type,
                provider_replay_kind=provider_replay_kind,
                provider_replay_payload=provider_replay_payload,
            )
        )
        for tool_call, arguments in zip(tool_calls, compacted_arguments, strict=True):
            self.append_metric(
                role="assistant",
                kind="tool_call",
                tool_name=str(tool_call.function.name or ""),
                metadata={
                    "step": self.step,
                    "arguments": _parse_tool_arguments(arguments),
                },
            )

    def append_tool_observation(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
        model_payload: dict[str, Any],
        provider_adapter: Any | None = None,
    ) -> None:
        observation = self._build_observation(
            tool_call=tool_call,
            tool_result=tool_result,
        )
        self._update_resume_refs(observation)
        self.messages.append(
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
        self.append_metric(
            role="tool",
            kind="tool_result",
            content=_observation_content(observation),
            tool_name=observation.tool_name,
            metadata={
                "step": self.step,
                "output": observation.output or tool_result.output,
            },
            observation=observation,
        )

    def start_tool_execution(
        self,
        tool_call: LLMToolCall,
        *,
        fingerprint: str,
    ) -> None:
        self.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=tool_call.id,
                tool_name=str(tool_call.function.name or ""),
                fingerprint=fingerprint,
                step=self.step,
                started_at=time.time(),
            )
        )
        self.append_metric(
            role="system",
            kind="tool_execution_started",
            tool_name=str(tool_call.function.name or ""),
            metadata={
                "step": self.step,
                "tool_call_id": tool_call.id,
                "call_fingerprint": fingerprint,
            },
        )

    def settle_tool_execution(
        self,
        *,
        fingerprint: str,
        status: str,
        result_status: str = "",
    ) -> None:
        record = next(
            (
                item
                for item in reversed(self.tool_executions)
                if item.fingerprint == fingerprint and item.status == "started"
            ),
            None,
        )
        if record is None:
            return
        record.status = status
        record.completed_at = time.time()
        record.result_status = normalize_message_text(result_status)
        self.append_metric(
            role="system",
            kind=f"tool_execution_{status}",
            tool_name=record.tool_name,
            metadata={
                "step": self.step,
                "tool_call_id": record.tool_call_id,
                "call_fingerprint": fingerprint,
                "result_status": record.result_status,
            },
        )

    def unsettled_tool_execution(
        self,
        fingerprint: str,
    ) -> ToolExecutionRecord | None:
        return next(
            (
                item
                for item in reversed(self.tool_executions)
                if item.fingerprint == fingerprint
                and item.status in {"started", "uncertain"}
            ),
            None,
        )

    def append_synthetic_observation(
        self,
        observation: AgentObservation,
        *,
        timeline_kind: str,
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._update_resume_refs(observation)
        self.append_metric(
            role="tool",
            kind=timeline_kind,
            content=content,
            tool_name=observation.tool_name,
            metadata={
                "step": self.step,
                "output": observation.output,
                **dict(metadata or {}),
            },
            observation=observation,
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
        if as_message:
            self.messages.append(runtime_control_message(message or reason))
        if not record_timeline:
            return
        self.append_metric(
            role="system",
            kind="runtime_guardrail",
            content=message or reason,
            metadata={
                "step": self.step,
                "payload": dict(payload),
                "as_message": as_message,
            },
        )

    def transition_force_final(self, reason: str) -> None:
        self.stop_reason = reason
        self.messages.append(
            runtime_control_message(
                "工具调用已经结束或达到上限。请不要再调用工具，"
                "根据已有工具结果直接给用户一个简短最终回复："
                "明确已完成/失败/需要确认；如有 artifact_id 必须列出；"
                "不要声称完成工具结果未证明的事项。"
            )
        )
        self.append_metric(
            role="system",
            kind="recovery_action",
            content=reason,
            metadata={"step": self.step},
        )

    def pause(
        self,
        *,
        reason: str,
        final_text: str = "",
    ) -> None:
        self.status = "paused"
        self.paused_reason = normalize_message_text(reason)
        self.stop_reason = f"paused:{self.paused_reason or 'unknown'}"
        self.delivery_complete = False
        if final_text:
            self.final_text = normalize_reply_text(final_text)
            self.final_source = "local_fallback"
        self.append_metric(
            role="system",
            kind="agent_paused",
            content=self.paused_reason,
            metadata={
                "step": self.step,
                "pending_approval": self.pending_approval,
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
        self.final_text = ""
        self.delivery_complete = False
        self.final_source = ""
        self.append_metric(
            role="system",
            kind="agent_resumed",
            content=normalize_message_text(reason),
            metadata={"step": self.step},
        )

    def cancel(self, *, reason: str = "") -> None:
        self.status = "cancelled"
        self.paused_reason = ""
        self.stop_reason = "cancelled"
        self.final_text = ""
        self.delivery_complete = False
        self.final_source = ""
        self.stop_reason = normalize_message_text(reason) or "cancelled_by_user"
        self.append_metric(
            role="system",
            kind="agent_cancelled",
            content=self.stop_reason,
            metadata={"step": self.step},
        )

    def complete_final(self, final_text: str, *, reason: str) -> None:
        self.status = "completed"
        self.paused_reason = ""
        self.final_text = normalize_reply_text(final_text)
        self.delivery_complete = True
        self.final_source = "model"
        self.stop_reason = reason
        self.messages.append(LLMMessage.assistant_text_response(self.final_text))
        self.append_metric(
            role="assistant",
            kind="assistant_text",
            content=self.final_text,
            metadata={"step": self.step},
        )

    def to_result(self) -> AgentRuntimeResult:
        return AgentRuntimeResult(
            final_text=self.final_text,
            run_id=self.run_id,
            trace_id=self.trace_id,
            status=self.status,
            paused_reason=self.paused_reason,
            tool_results=tuple(
                item.observation.result
                for item in self.metrics
                if item.observation is not None and item.observation.result is not None
            ),
            timeline=tuple(self.metrics),
            messages=tuple(self.messages),
            stop_reason=self.stop_reason,
            steps=self.step,
            budget=self.budget,
            delivery_complete=self.delivery_complete,
            final_source=self.final_source,
        )

    def runtime_observations(self) -> list[AgentObservation]:
        return [
            item.observation for item in self.metrics if item.observation is not None
        ]

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
                str(arguments.get(_TASK_TEXT_FIELD) or "")
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

    def _update_resume_refs(self, observation: AgentObservation) -> None:
        output = observation.output or {}
        approval_id = _first_text(
            output.get("approval_id"),
            _nested_get(output, "approval", "approval_id"),
        )
        if approval_id and bool(output.get("approval_required")):
            self.pending_approval = approval_id

        event = output.get("observation_event")
        nested_event = output.get("event")

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
                append_artifact_refs(
                    self.artifact_refs,
                    (str(item.get("artifact_id") or ""),),
                )
        artifact_id = _first_text(
            output.get("artifact_id"),
            _nested_get(output, "artifact", "artifact_id"),
        )
        append_artifact_refs(self.artifact_refs, (artifact_id,))
        for artifact in observation.artifacts:
            append_artifact_refs(
                self.artifact_refs,
                (str(artifact.get("artifact_id") or ""),),
            )


def repair_interrupted_tool_protocol(
    state: AgentRunState,
    *,
    provider_adapter: Any | None = None,
) -> dict[str, int]:
    """Make a restored transcript valid without replaying unfinished work."""

    repaired: list[LLMMessage] = []
    pending: dict[str, LLMToolCall] = {}
    orphan_results = 0
    interrupted_calls = 0
    uncertain_calls = 0

    def flush_pending() -> None:
        nonlocal interrupted_calls, uncertain_calls
        for tool_call in pending.values():
            record = next(
                (
                    item
                    for item in reversed(state.tool_executions)
                    if item.tool_call_id == tool_call.id
                    and item.status in {"started", "uncertain"}
                ),
                None,
            )
            if record is not None:
                tool_result = uncertain_tool_execution_result()
                if record.status == "started":
                    state.settle_tool_execution(
                        fingerprint=record.fingerprint,
                        status="uncertain",
                        result_status="tool_execution_uncertain",
                    )
                uncertain_calls += 1
            else:
                tool_result = _interrupted_tool_call_result()
                interrupted_calls += 1
            output = dict(tool_result.output)
            repaired.append(
                provider_adapter.tool_result_message(
                    tool_call=tool_call,
                    function_name=tool_call.function.name,
                    result=output,
                )
                if provider_adapter is not None
                else LLMMessage.tool_response(
                    tool_call_id=tool_call.id,
                    function_name=tool_call.function.name,
                    result=output,
                )
            )
            state.append_synthetic_observation(
                state._build_observation(
                    tool_call=tool_call,
                    tool_result=tool_result,
                ),
                timeline_kind=str(output.get("status", "tool_call_interrupted")),
                content=str(output.get("error", "")),
                metadata={"source": "transcript_recovery"},
            )
        pending.clear()

    for message in state.messages:
        if message.role == "tool":
            tool_call_id = str(message.tool_call_id or "")
            if tool_call_id and tool_call_id in pending:
                repaired.append(message)
                pending.pop(tool_call_id, None)
            else:
                orphan_results += 1
            continue
        if pending:
            flush_pending()
        tool_calls = (
            list(message.tool_calls or []) if message.role == "assistant" else []
        )
        tool_call_ids = [str(tool_call.id or "") for tool_call in tool_calls]
        if tool_calls and (
            any(not tool_call_id for tool_call_id in tool_call_ids)
            or len(set(tool_call_ids)) != len(tool_call_ids)
        ):
            interrupted_calls += len(tool_calls)
            repaired.append(
                runtime_control_message(
                    "此前模型返回的工具调用标识无效，相关调用未自动重放；"
                    "已持久化的副作用执行记录仍保留。"
                )
            )
            continue
        repaired.append(message)
        if tool_calls:
            pending.update(dict(zip(tool_call_ids, tool_calls, strict=True)))
    if pending:
        flush_pending()

    for record in state.tool_executions:
        if record.status == "started":
            tool_call = LLMToolCall(
                id=record.tool_call_id,
                function=LLMToolFunction(name=record.tool_name, arguments="{}"),
            )
            tool_result = uncertain_tool_execution_result()
            state.settle_tool_execution(
                fingerprint=record.fingerprint,
                status="uncertain",
                result_status="tool_execution_uncertain",
            )
            output = dict(tool_result.output)
            repaired.append(
                runtime_control_message(
                    f"工具 {record.tool_name} 的副作用调用状态不确定，"
                    "系统不会自动重放；请先检查外部状态再继续。"
                )
            )
            state.append_synthetic_observation(
                state._build_observation(
                    tool_call=tool_call,
                    tool_result=tool_result,
                ),
                timeline_kind="tool_execution_uncertain",
                content=str(output.get("error", "")),
                metadata={"source": "execution_record_recovery"},
            )
            uncertain_calls += 1

    state.messages = repaired
    result = {
        "orphan_results": orphan_results,
        "interrupted_calls": interrupted_calls,
        "uncertain_calls": uncertain_calls,
    }
    if any(result.values()):
        state.budget.last_usage_message_count = 0
        state.budget.last_usage_schema_tokens = 0
    return result


def uncertain_tool_execution_result() -> ToolResult:
    message = (
        "该有副作用工具的同一调用此前已开始，但完成状态未持久化。"
        "禁止自动重放；请先检查外部状态，必要时让用户确认后再处理。"
    )
    return ToolResult(
        output={
            "ok": False,
            "status": "tool_execution_uncertain",
            "error": message,
            "retryable": False,
        },
        display_content=message,
        is_error=True,
        is_retryable=False,
    )


def _interrupted_tool_call_result() -> ToolResult:
    message = "工具调用在执行完成前被中断，未执行的操作可由模型重新规划。"
    return ToolResult(
        output={
            "ok": False,
            "status": "tool_call_interrupted",
            "error": message,
            "retryable": True,
        },
        display_content=message,
        is_error=True,
        is_retryable=True,
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


def tool_call_fingerprint(tool_call: LLMToolCall) -> str:
    payload = {
        "tool_name": normalize_message_text(str(tool_call.function.name or "")),
        "arguments": _parse_tool_arguments(tool_call.function.arguments),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _compact_tool_argument_string(
    arguments: str,
    *,
    trace_id: str,
    tool_name: str,
) -> tuple[str, str]:
    text = str(arguments or "")
    if len(text) <= 900:
        return text, ""
    ref = get_artifact_store().store_text(
        text,
        artifact_type="text",
        trace_id=trace_id,
        source=f"tool_call:{tool_name}:arguments",
        force_file=True,
    )
    if ref is None:
        return summarize_artifact_text(text, limit=360), ""
    payload = {
        "artifact_id": ref.artifact_id,
        "summary": ref.summary,
        "note": (
            "tool arguments were compressed; original arguments are stored "
            "in ArtifactStore"
        ),
    }
    return json.dumps(payload, ensure_ascii=False), str(ref.artifact_id or "")


def append_artifact_refs(target: list[str], artifact_ids: tuple[str, ...]) -> None:
    existing = set(target)
    for value in artifact_ids:
        artifact_id = normalize_message_text(str(value or ""))
        if artifact_id and artifact_id not in existing:
            target.append(artifact_id)
            existing.add(artifact_id)


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


__all__ = [
    "AgentBudgetState",
    "AgentObservation",
    "AgentRunBudget",
    "AgentRunState",
    "AgentRuntimeMetric",
    "AgentRuntimeResult",
    "AgentRuntimeTimelineItem",
    "ToolExecutionRecord",
    "groups_with_next_user_message",
    "is_runtime_control_message",
    "repair_interrupted_tool_protocol",
    "resolve_superuser_agent_run_budget",
    "runtime_control_message",
    "tool_call_fingerprint",
    "uncertain_tool_execution_result",
]
