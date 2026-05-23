"""Policy chain for AgentRuntime.

AgentRuntime should be a small model/tool loop.  This chain owns the runtime
policies in one explicit order:
ToolIntentGate -> TaskLedger -> Observation -> FinalValidator -> Guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types.models import LLMToolCall, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .agent_collaborators import (
    SAFE_NO_TOOL_RESULT_REPLY,
    RuntimeFinalGate,
    RuntimeObservationCoordinator,
    RuntimeTaskLedgerCoordinator,
    should_complete_after_plugin_observation,
)
from .agent_planner import AgentPlanner
from .agent_state import AgentRunState
from .agent_verifier import AgentVerifier
from .provider_capability import ProviderCapabilityAdapter
from .route_text import normalize_message_text
from .runtime_guardrails import RuntimeGuardrailDecision, RuntimeGuardrails
from .task_coverage_judge import TaskCoverageJudge


@dataclass(frozen=True)
class PostObservationAction:
    """Result of running the Observation -> Guardrails side of the chain."""

    paused: bool = False
    stop_reason: str = ""


@dataclass(frozen=True)
class FinalAcceptance:
    """FinalValidator decision in runtime-friendly form."""

    action: Literal["accept", "retry", "complete"]
    final_text: str = ""
    reason: str = ""


class AgentRuntimeChain:
    """Single coordination point for all non-plumbing AgentRuntime policies."""

    def __init__(
        self,
        *,
        state: AgentRunState,
        run_context: RunContext,
        message_text: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
        provider_adapter: ProviderCapabilityAdapter,
        sync_tools: Callable[[], None],
        persist_state: Callable[..., None],
    ) -> None:
        self.state = state
        self.run_context = run_context
        self.message_text = normalize_message_text(message_text)
        self.provider_adapter = provider_adapter
        self._sync_tools = sync_tools
        self._persist_state = persist_state
        self._force_required_tool_choice_once = False
        self._post_batch_guardrails: list[RuntimeGuardrailDecision] = []

        self.guardrails = RuntimeGuardrails(
            session_id=state.session_key,
            max_elapsed_seconds=max(float(timeout or 0) * 3.0, 45.0),
        )
        self.coverage_judge = TaskCoverageJudge(
            trace_id=state.trace_id,
            model_name=model_name,
            generation_config=generation_config,
            timeout=timeout,
        )
        self.planner = AgentPlanner(
            trace_id=state.trace_id,
            model_name=model_name,
            generation_config=generation_config,
            timeout=timeout,
        )
        self.verifier = AgentVerifier(
            trace_id=state.trace_id,
            model_name=model_name,
            generation_config=generation_config,
            timeout=timeout,
        )
        self.task_ledger_coordinator = RuntimeTaskLedgerCoordinator(self)
        self.final_gate = RuntimeFinalGate(self)
        self.observation_coordinator = RuntimeObservationCoordinator(self)

    async def initialize(self) -> None:
        await self.task_ledger_coordinator.seed()

    def tool_choice_for_request(self) -> str | dict[str, Any] | None:
        """ToolIntentGate output adapted to provider capabilities."""

        if not self.state.tool_map:
            return None
        if self.state.tool_obligation == "none":
            return None
        if self.consume_required_tool_choice_once():
            return self.provider_adapter.adapt_tool_choice(
                "required",
                has_tools=bool(self.state.tool_map),
            )
        if (
            self.state.tool_obligation == "required"
            and self._has_available_required_tools()
            and not self._has_command_observation()
        ):
            return self.provider_adapter.adapt_tool_choice(
                "required",
                has_tools=bool(self.state.tool_map),
            )
        return self.provider_adapter.adapt_tool_choice(
            "auto",
            has_tools=bool(self.state.tool_map),
        )

    def tools_for_request(
        self,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, ToolExecutable] | None:
        if not self.state.tool_map:
            return None
        return self.provider_adapter.prepare_chatinter_tools_for_request(
            self.state.tool_map,
            tool_choice=tool_choice,
            required_tool_names=self.state.required_tool_names,
            tool_obligation=self.state.tool_obligation,
            has_command_observation=self._has_command_observation(),
        )

    def filter_tool_map(
        self,
        tool_map: dict[str, ToolExecutable],
    ) -> dict[str, ToolExecutable]:
        return self.guardrails.filter_tool_map(tool_map)

    def before_model_request(self) -> RuntimeGuardrailDecision | None:
        return self.guardrails.before_model_request(self.state)

    def on_budget_exhausted(self, *, call_count: int) -> RuntimeGuardrailDecision:
        return self.guardrails.on_budget_exhausted(call_count=call_count)

    def on_max_steps(self) -> RuntimeGuardrailDecision:
        return self.guardrails.on_max_steps()

    def before_tool_call(
        self,
        tool_call: LLMToolCall,
    ) -> RuntimeGuardrailDecision | None:
        return self.guardrails.before_tool_call(
            tool_call=tool_call,
            tool_map=self.state.tool_map,
        )

    def runtime_stopping_decision(
        self,
        tool_call: LLMToolCall,
    ) -> RuntimeGuardrailDecision:
        return RuntimeGuardrailDecision(
            reason="runtime_stopping",
            message="运行时已经决定停止工具调用，请等待最终回复。",
            severity="heavy",
            action="stop",
            tool_name=str(tool_call.function.name or ""),
        )

    def tool_result_for_decision(
        self,
        *,
        tool_call: LLMToolCall,
        decision: RuntimeGuardrailDecision,
    ) -> ToolResult:
        return self.guardrails.tool_result_for_decision(
            tool_call=tool_call,
            decision=decision,
        )

    def apply_guardrail(
        self,
        decision: RuntimeGuardrailDecision,
        *,
        as_message: bool,
        record_timeline: bool = True,
    ) -> None:
        if decision.should_block_tool:
            self.guardrails.block_tool(decision.tool_name)
        self.state.append_guardrail_observation(
            decision.to_payload(),
            as_message=as_message,
            record_timeline=record_timeline,
        )
        self._sync_tools()

    async def final_acceptance_action(
        self,
        final_text: str,
        *,
        allow_retry: bool = True,
    ) -> str:
        return await self.final_gate.final_acceptance_action(
            final_text,
            allow_retry=allow_retry,
        )

    async def accept_final(
        self,
        final_text: str,
        *,
        allow_retry: bool = True,
        fallback_reason: str = "final_response",
    ) -> FinalAcceptance:
        action = await self.final_acceptance_action(
            final_text,
            allow_retry=allow_retry,
        )
        if action == "retry":
            return FinalAcceptance(action="retry")
        if action == "safe_final":
            reason = (
                "final_validation_blocked"
                if fallback_reason == "final_response"
                else f"{fallback_reason}:final_validation_blocked"
            )
            return FinalAcceptance(
                action="complete",
                final_text=SAFE_NO_TOOL_RESULT_REPLY,
                reason=reason,
            )
        if action == "partial_final":
            reason = (
                "task_ledger_incomplete"
                if fallback_reason == "final_response"
                else f"{fallback_reason}:task_ledger_incomplete"
            )
            return FinalAcceptance(
                action="complete",
                final_text=RuntimeFinalGate.partial_task_ledger_reply(self.state),
                reason=reason,
            )
        return FinalAcceptance(
            action="accept",
            final_text=final_text,
            reason=fallback_reason,
        )

    async def complete_if_acceptable(
        self,
        final_text: str,
        *,
        allow_retry: bool = True,
        fallback_reason: str = "final_response",
    ) -> FinalAcceptance:
        decision = await self.accept_final(
            final_text,
            allow_retry=allow_retry,
            fallback_reason=fallback_reason,
        )
        if decision.action in {"accept", "complete"}:
            self.state.complete_final(decision.final_text, reason=decision.reason)
        return decision

    async def after_tool_observation(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
        pre_guardrail: RuntimeGuardrailDecision | None,
    ) -> PostObservationAction:
        self.observation_coordinator.sync_todos_from_latest_observation()
        await self.verify_after_observation()
        self._persist_state(
            "task_graph_verified",
            tool_name=str(tool_call.function.name or ""),
        )
        if self.observation_coordinator.pause_if_needed_after_observation(
            tool_result
        ):
            if self.state.task_graph is not None:
                self.state.task_graph.refresh_status()
            self._persist_state("paused", reason=self.state.paused_reason)
            return PostObservationAction(paused=True)
        if await self.observation_coordinator.background_pause_if_needed_after_observation(
            tool_result
        ):
            if self.state.task_graph is not None:
                self.state.task_graph.refresh_status()
            self._persist_state("paused", reason=self.state.paused_reason)
            return PostObservationAction(paused=True)
        if pre_guardrail is not None:
            return PostObservationAction()

        post_guardrail = self.guardrails.after_tool_result(
            tool_call=tool_call,
            tool_result=tool_result,
            tool_map=self.state.tool_map,
        )
        if post_guardrail is None:
            return PostObservationAction()
        self.apply_guardrail(post_guardrail, as_message=False)
        self._post_batch_guardrails.append(post_guardrail)
        if post_guardrail.should_stop:
            return PostObservationAction(stop_reason=post_guardrail.reason)
        return PostObservationAction()

    async def verify_after_observation(self) -> None:
        await self.task_ledger_coordinator.verify_after_observation()

    def flush_post_batch_guardrails(self) -> None:
        for guardrail in self._post_batch_guardrails:
            self.state.append_guardrail_observation(
                guardrail.to_payload(),
                as_message=True,
                record_timeline=False,
            )
            self._persist_state("guardrail", reason=guardrail.reason)
        self._post_batch_guardrails.clear()

    def should_finish_after_observation(self) -> bool:
        return should_complete_after_plugin_observation(
            self,
            looks_like_multi_task_turn=(
                self.task_ledger_coordinator.looks_like_multi_task_turn()
            ),
        )

    def request_required_tool_choice_once(self) -> None:
        self._force_required_tool_choice_once = True

    def consume_required_tool_choice_once(self) -> bool:
        if not self._force_required_tool_choice_once:
            return False
        self._force_required_tool_choice_once = False
        return True

    def _available_tool_summaries(self, *, limit: int = 16) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        max_items = int(limit or 0)
        for name, tool in self._stable_tool_map(self.state.tool_map).items():
            binding = getattr(tool, "binding", None)
            if binding is None:
                items.append(
                    {
                        "tool": name,
                        "description": normalize_message_text(
                            str(getattr(tool, "description", "") or "")
                        ),
                    }
                )
                if max_items > 0 and len(items) >= max_items:
                    break
                continue
            candidate = getattr(binding, "candidate", None)
            schema = getattr(candidate, "schema", None)
            command_id = normalize_message_text(
                str(getattr(binding, "command_id", "") or "")
            )
            if not command_id:
                continue
            snapshot = getattr(candidate, "tool", None)
            items.append(
                {
                    "tool": name,
                    "command_id": command_id,
                    "plugin": normalize_message_text(
                        str(getattr(candidate, "plugin_name", "") or "")
                    ),
                    "head": normalize_message_text(
                        str(getattr(schema, "head", "") or "")
                    ),
                    "role": normalize_message_text(
                        str(getattr(schema, "command_role", "") or "")
                    ),
                    "output_mode": normalize_message_text(
                        str(getattr(snapshot, "output_mode", "") or "")
                    ),
                    "requires_real_tool": bool(
                        getattr(snapshot, "requires_real_tool", False)
                    ),
                    "source_of_truth": normalize_message_text(
                        str(getattr(snapshot, "source_of_truth", "") or "")
                    ),
                }
            )
            if max_items > 0 and len(items) >= max_items:
                break
        return items

    def _stable_tool_map(
        self,
        tools: dict[str, ToolExecutable],
    ) -> dict[str, ToolExecutable]:
        return self.provider_adapter.sort_tool_map(
            tools,
            required_tool_names=self.state.required_tool_names,
        )

    def _has_available_required_tools(self) -> bool:
        required = {
            normalize_message_text(str(name or ""))
            for name in self.state.required_tool_names
            if normalize_message_text(str(name or ""))
        }
        if required:
            return any(
                normalize_message_text(name) in required
                for name in self.state.tool_map
            )
        return self._has_actionable_command_tools()

    def _has_command_observation(self) -> bool:
        return any(
            bool(observation.command_id)
            for observation in self.state.observations
        )

    def _has_actionable_command_tools(self) -> bool:
        return any(
            self._is_actionable_command_tool(tool)
            for tool in self.state.tool_map.values()
        )

    def _actionable_command_tool_count(self) -> int:
        return sum(
            1
            for tool in self.state.tool_map.values()
            if self._is_actionable_command_tool(tool)
        )

    @staticmethod
    def _is_actionable_command_tool(tool: ToolExecutable) -> bool:
        binding = getattr(tool, "binding", None)
        candidate = getattr(binding, "candidate", None)
        schema = getattr(candidate, "schema", None)
        if schema is None:
            return False
        role = normalize_message_text(str(getattr(schema, "command_role", "") or ""))
        payload_policy = normalize_message_text(
            str(getattr(schema, "payload_policy", "") or "")
        )
        if role in {"execute", "template", "random"}:
            return True
        if payload_policy in {"image_only", "text_or_image", "free_tail"}:
            return True
        return bool(getattr(schema, "slots", []) or [])


__all__ = ["AgentRuntimeChain", "FinalAcceptance", "PostObservationAction"]
