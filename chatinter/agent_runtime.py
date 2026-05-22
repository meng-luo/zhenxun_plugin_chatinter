"""Agent state machine runtime for ChatInter native command tools.

The runtime owns the model/tool loop:
LLM -> tool_calls -> execute -> observations -> LLM ... -> final text.
"""

from __future__ import annotations

import json
import time
from typing import Any

from zhenxun.services.llm import AI, LLMContentPart, LLMMessage
from zhenxun.services.llm.tools import RunContext, ToolInvoker
from zhenxun.services.llm.types.models import (
    LLMResponse,
    LLMToolCall,
    ToolResult,
)
from zhenxun.services.llm.types.protocols import ToolExecutable

from .agent_run_store import get_agent_run_snapshot, persist_agent_run_state
from .agent_state import (
    AgentObservation,
    AgentRunState,
    AgentRuntimeResult,
    AgentRuntimeTimelineItem,
)
from .agent_planner import AgentPlanner
from .agent_verifier import AgentVerifier
from .artifact_store import compact_tool_result_output, summarize_artifact_text
from .command_observation import build_command_observation
from .completion_validator import validate_final_reply
from .config import build_tool_generation_config
from .context_compression import compress_agent_messages
from .feedback import record_command_observation_feedback
from .native_command_tools import compact_command_tool_view
from .provider_capability import (
    ProviderCapabilityAdapter,
    is_compact_request_tool,
)
from .route_text import normalize_message_text
from .runtime_guardrails import RuntimeGuardrailDecision, RuntimeGuardrails
from .soft_tool_policy import (
    is_high_reliability_candidate,
    is_low_reliability_candidate,
)
from .task_frame import TASK_TEXT_FIELD
from .task_coverage_judge import TaskCoverageJudge
from .task_ledger import TaskLedger, TaskLedgerEntry
from .trajectory_store import record_agent_trajectory
from .turn_runtime import TurnBudgetController, estimate_text_tokens
from .superuser_agent.background_tasks import (
    ObservationEvent,
    wait_for_observation_event,
)
from .superuser_agent.todo_store import update_todo_from_observation

_MAIN_STAGE = "main_request"
_DIRECT_ANSWER_INTERCEPT_LIMIT = 1
_FINAL_VALIDATION_INTERCEPT_LIMIT = 1
_TASK_COVERAGE_INTERCEPT_LIMIT = 2
_BACKGROUND_OBSERVATION_WAIT_SECONDS = 6.0
_AUTO_FULL_SCHEMA_TOOL_CAP = 8
_SAFE_NO_TOOL_RESULT_REPLY = (
    "我没有拿到真实工具执行结果，不能直接说已经完成。你可以换个说法，"
    "或稍后再让我试一次。"
)

class AgentRuntime:
    """State-machine ReAct loop around ChatInter command tools."""

    def __init__(
        self,
        *,
        state: AgentRunState,
        run_context: RunContext,
        message_text: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
        budget_controller: TurnBudgetController | None = None,
    ) -> None:
        self.state = state
        self.run_context = run_context
        self.message_text = normalize_message_text(message_text)
        self.model_name = model_name
        self.generation_config = generation_config
        self.timeout = timeout
        self.budget_controller = budget_controller
        self.provider_adapter = ProviderCapabilityAdapter.for_model(model_name)
        self.base_tool_map = dict(state.tool_map)
        self.ai = AI(session_id=f"chatinter-main:{state.session_key or 'global'}")
        self.invoker = ToolInvoker()
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
        self._force_required_tool_choice_once = False
        self._trajectory_recorded = False

    async def run(self) -> AgentRuntimeResult:
        started_at = time.time()
        started_perf = time.perf_counter()
        if self.state.status != "running":
            self.state.status = "running"
            self.state.paused_reason = ""
        self._persist_state("started")
        try:
            await self._seed_task_graph()
            await self._seed_task_ledger()
            for _ in range(self.state.max_steps):
                if self._cancelled_externally():
                    self.state.cancel(reason="cancelled_by_agent_run_cancel")
                    self._persist_state("cancelled")
                    return self.state.to_result()
                self.state.start_step()
                self._sync_dynamic_tools()
                self._persist_state("step_started")
                self._compress_context_if_needed()
                request_guardrail = self.guardrails.before_model_request(self.state)
                if request_guardrail is not None:
                    self._apply_guardrail(request_guardrail, as_message=True)
                    self._persist_state(
                        "guardrail",
                        reason=request_guardrail.reason,
                    )
                    if request_guardrail.should_stop:
                        await self._force_final_response(reason=request_guardrail.reason)
                        return self.state.to_result()

                self._record_prompt_use()
                self.state.append_model_request(tool_count=len(self.state.tool_map))
                self._persist_state("model_request")
                tool_choice = self._tool_choice_for_request()
                tools_for_request = self._tools_for_request(tool_choice)
                response = await self._request_model(
                    tools=tools_for_request,
                    tool_choice=tool_choice,
                )
                tool_calls = list(response.tool_calls or [])
                original_tool_call_count = len(tool_calls)
                tool_calls = self.provider_adapter.tool_calls_for_execution(tool_calls)
                if original_tool_call_count > len(tool_calls):
                    self.state.append_guardrail_observation(
                        self.provider_adapter.parallel_tool_call_notice(
                            original_count=original_tool_call_count,
                            executed_count=len(tool_calls),
                        ),
                        as_message=True,
                        record_timeline=True,
                    )
                if self._uses_compact_command_schema(tools_for_request, tool_calls):
                    tool_calls = await self._resolve_compact_command_tool_calls(
                        tool_calls
                    )
                    if self.state.status == "completed":
                        self._persist_state("completed", reason=self.state.stop_reason)
                        return self.state.to_result()
                    if not tool_calls:
                        self._persist_state("compact_schema_resolution_empty")
                        continue
                if not tool_calls:
                    final_text = normalize_message_text(str(response.text or ""))
                    final_action = await self._final_acceptance_action(
                        final_text
                    )
                    if final_action == "retry":
                        self._persist_state("final_acceptance_retry")
                        continue
                    if final_action == "safe_final":
                        self.state.complete_final(
                            _SAFE_NO_TOOL_RESULT_REPLY,
                            reason="final_validation_blocked",
                        )
                        self._persist_state("completed", reason=self.state.stop_reason)
                        return self.state.to_result()
                    if final_action == "partial_final":
                        self.state.complete_final(
                            self._partial_task_ledger_reply(),
                            reason="task_ledger_incomplete",
                        )
                        self._persist_state("completed", reason=self.state.stop_reason)
                        return self.state.to_result()
                    self.state.complete_final(final_text, reason="final_response")
                    self._persist_state("completed")
                    return self.state.to_result()

                if not self._allow_tool_batch(len(tool_calls)):
                    budget_guardrail = self.guardrails.on_budget_exhausted(
                        call_count=len(tool_calls)
                    )
                    self._apply_guardrail(budget_guardrail, as_message=True)
                    self._persist_state(
                        "guardrail",
                        reason=budget_guardrail.reason,
                    )
                    await self._force_final_response(reason=budget_guardrail.reason)
                    return self.state.to_result()

                self.state.append_tool_calls(
                    tool_calls,
                    response_text=response.text or "",
                )
                self._persist_state("tool_calls", count=len(tool_calls))

                started = time.perf_counter()
                post_batch_guardrails: list[RuntimeGuardrailDecision] = []
                force_final_reason = ""
                for tool_call in tool_calls:
                    tool_call_started = time.perf_counter()
                    pre_guardrail = (
                        self.guardrails.before_tool_call(
                            tool_call=tool_call,
                            tool_map=self.state.tool_map,
                        )
                        if not force_final_reason
                        else RuntimeGuardrailDecision(
                            reason="runtime_stopping",
                            message="运行时已经决定停止工具调用，请等待最终回复。",
                            severity="heavy",
                            action="stop",
                            tool_name=str(tool_call.function.name or ""),
                        )
                    )
                    if pre_guardrail is not None:
                        self._apply_guardrail(pre_guardrail, as_message=False)
                        self._persist_state(
                            "guardrail",
                            reason=pre_guardrail.reason,
                            tool_name=str(tool_call.function.name or ""),
                        )
                        resolved_call = tool_call
                        tool_result = self.guardrails.tool_result_for_decision(
                            tool_call=tool_call,
                            decision=pre_guardrail,
                        )
                        if pre_guardrail.should_stop:
                            force_final_reason = pre_guardrail.reason
                    else:
                        try:
                            (
                                resolved_call,
                                tool_result,
                            ) = await self.invoker.execute_tool_call(
                                tool_call,
                                self.state.tool_map,
                                self.run_context,
                            )
                        except Exception as exc:
                            resolved_call = tool_call
                            tool_result = self._exception_tool_result(tool_call, exc)
                        tool_result = self._normalize_command_tool_result(
                            resolved_call,
                            tool_result,
                        )
                    tool_result = self._compact_tool_result_for_context(
                        resolved_call,
                        tool_result,
                    )
                    self._record_tool_feedback(
                        resolved_call,
                        tool_result,
                        latency_ms=max(
                            int((time.perf_counter() - tool_call_started) * 1000),
                            0,
                        ),
                    )
                    self._sync_dynamic_tools()
                    self.state.append_tool_observation(
                        tool_call=resolved_call,
                        tool_result=tool_result,
                        model_payload=self._tool_result_for_model(
                            resolved_call,
                            tool_result,
                        ),
                    )
                    self._persist_state(
                        "tool_observation",
                        tool_name=str(resolved_call.function.name or ""),
                    )
                    self._sync_todos_from_latest_observation()
                    await self._verify_task_graph_after_observation()
                    self._persist_state(
                        "task_graph_verified",
                        tool_name=str(resolved_call.function.name or ""),
                    )
                    if self._pause_if_needed_after_observation(tool_result):
                        if self.state.task_graph is not None:
                            self.state.task_graph.refresh_status()
                        self._persist_state("paused", reason=self.state.paused_reason)
                        return self.state.to_result()
                    if await self._background_pause_if_needed_after_observation(
                        tool_result
                    ):
                        if self.state.task_graph is not None:
                            self.state.task_graph.refresh_status()
                        self._persist_state("paused", reason=self.state.paused_reason)
                        return self.state.to_result()
                    if pre_guardrail is not None:
                        continue

                    post_guardrail = self.guardrails.after_tool_result(
                        tool_call=resolved_call,
                        tool_result=tool_result,
                        tool_map=self.state.tool_map,
                    )
                    if post_guardrail is not None:
                        self._apply_guardrail(post_guardrail, as_message=False)
                        post_batch_guardrails.append(post_guardrail)
                        if post_guardrail.should_stop:
                            force_final_reason = post_guardrail.reason
                self._record_tool_batch(time.perf_counter() - started)
                for guardrail in post_batch_guardrails:
                    self.state.append_guardrail_observation(
                        guardrail.to_payload(),
                        as_message=True,
                        record_timeline=False,
                    )
                    self._persist_state("guardrail", reason=guardrail.reason)
                if force_final_reason:
                    await self._force_final_response(reason=force_final_reason)
                    return self.state.to_result()
                if self._should_complete_after_plugin_observation():
                    self.state.complete_final(
                        "",
                        reason="tool_completed_no_final_llm",
                    )
                    self._persist_state("completed", reason=self.state.stop_reason)
                    return self.state.to_result()

            max_steps_guardrail = self.guardrails.on_max_steps()
            self._apply_guardrail(max_steps_guardrail, as_message=True)
            self._persist_state("guardrail", reason=max_steps_guardrail.reason)
            await self._force_final_response(reason=max_steps_guardrail.reason)
            return self.state.to_result()
        except Exception as exc:
            self.state.status = "failed"
            self.state.stop_reason = f"runtime_exception:{type(exc).__name__}"
            self.state.recovery_action = (
                self.state.recovery_action or self.state.stop_reason
            )
            self._persist_state("failed", error=str(exc))
            raise
        finally:
            self._record_trajectory_once(
                started_at=started_at,
                latency_ms=max(int((time.perf_counter() - started_perf) * 1000), 0),
            )

    async def _request_model(
        self,
        *,
        tools: dict[str, ToolExecutable] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> LLMResponse:
        request_tools = self._ensure_provider_request_tools(tools)
        adapted_tool_choice = self.provider_adapter.adapt_tool_choice(
            tool_choice,
            has_tools=bool(request_tools),
        )
        adapted_messages = self.provider_adapter.adapt_messages(self.state.messages)
        return await self.ai.generate_internal(
            adapted_messages,
            model=self.model_name,
            config=build_tool_generation_config(
                tool_choice=adapted_tool_choice,
                base=self.generation_config,
            ),
            tools=request_tools,
            tool_choice=adapted_tool_choice,
            timeout=self.timeout,
        )

    def _ensure_provider_request_tools(
        self,
        tools: dict[str, ToolExecutable] | None,
    ) -> dict[str, ToolExecutable] | None:
        if not tools:
            return None
        if all(
            str(getattr(tool, "chatinter_schema_mode", "") or "")
            in {"full", "compact"}
            for tool in tools.values()
        ):
            return tools
        schema_modes = {
            name: "full"
            for name in tools
        }
        return self.provider_adapter.prepare_tool_map_for_request(
            tools,
            required_tool_names=self.state.required_tool_names,
            schema_modes=schema_modes,  # type: ignore[arg-type]
        )

    async def _force_final_response(self, *, reason: str) -> None:
        self.state.transition_force_final(reason)
        self._compress_context_if_needed()
        self._persist_state("force_final_requested", reason=reason)
        response = await self._request_model(tools=None, tool_choice=None)
        final_text = normalize_message_text(str(response.text or ""))
        final_action = await self._final_acceptance_action(
            final_text,
            allow_retry=False,
        )
        if final_action == "safe_final":
            final_text = _SAFE_NO_TOOL_RESULT_REPLY
            reason = f"{reason}:final_validation_blocked"
        elif final_action == "partial_final":
            final_text = self._partial_task_ledger_reply()
            reason = f"{reason}:task_ledger_incomplete"
        self.state.complete_final(final_text, reason=reason)
        self._persist_state("completed", reason=reason)

    def _allow_tool_batch(self, call_count: int) -> bool:
        if self.budget_controller is None:
            self.state.budget.tool_calls += max(call_count, 0)
            self.state.budget.tool_batches += 1
            return True
        allowed = self.budget_controller.allow_tool_batch(
            call_count=call_count,
            batch_kind=_MAIN_STAGE,
        )
        self.state.capture_budget(self.budget_controller)
        return allowed

    def _record_tool_batch(self, duration: float) -> None:
        if self.budget_controller is None:
            self.state.budget.durations_ms[f"tool:{_MAIN_STAGE}"] = round(
                self.state.budget.durations_ms.get(f"tool:{_MAIN_STAGE}", 0.0)
                + max(duration, 0.0) * 1000,
                2,
            )
            return
        self.budget_controller.record_tool_batch(
            batch_kind=_MAIN_STAGE,
            duration=duration,
        )
        self.state.capture_budget(self.budget_controller)

    def _record_prompt_use(self) -> None:
        estimated_tokens = _estimate_prompt_tokens(self.state.messages)
        self.state.record_prompt_use(
            estimated_tokens=estimated_tokens,
            budget_controller=self.budget_controller,
        )

    def _sync_dynamic_tools(self) -> None:
        extra = getattr(self.run_context, "extra", None)
        if not isinstance(extra, dict):
            return
        catalog_state = extra.get("command_catalog_state")
        capability_registry = extra.get("capability_registry")
        dynamic_tools = getattr(catalog_state, "tool_map", None)
        if dynamic_tools is None:
            if capability_registry is None or not hasattr(
                capability_registry, "executable_tool_map"
            ):
                return
        if callable(dynamic_tools):
            dynamic_tools = dynamic_tools()
        if not isinstance(dynamic_tools, dict):
            dynamic_tools = {}
        if (
            capability_registry is not None
            and hasattr(capability_registry, "sync_command_tools")
            and catalog_state is not None
        ):
            capability_registry.sync_command_tools(
                list(getattr(catalog_state, "candidates", []) or []),
                dynamic_tools,
            )
        if capability_registry is not None and hasattr(
            capability_registry, "executable_tool_map"
        ):
            merged_tools = capability_registry.executable_tool_map()
        else:
            merged_tools = {**self.base_tool_map, **dynamic_tools}
        self.state.tool_map = self.guardrails.filter_tool_map(merged_tools)
        self.state.tool_map = self._stable_tool_map(self.state.tool_map)

    def _tool_choice_for_request(self) -> str | dict[str, Any] | None:
        if not self.state.tool_map:
            return None
        if self.state.tool_obligation == "none":
            return None
        if self._force_required_tool_choice_once:
            self._force_required_tool_choice_once = False
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

    def _tools_for_request(
        self,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, ToolExecutable] | None:
        if not self.state.tool_map:
            return None
        if self.state.tool_obligation == "none" and tool_choice is None:
            return None
        tools = self._stable_tool_map(self.state.tool_map)
        schema_modes: dict[str, str] = {name: "full" for name in tools}
        if self._should_use_compact_command_schema(tool_choice):
            full_schema_names = self._full_schema_tool_names_for_request(tool_choice)
            tools = {
                name: compact_command_tool_view(tool)
                if self._is_command_tool(tool) and name not in full_schema_names
                else tool
                for name, tool in tools.items()
            }
            schema_modes = {
                name: self.provider_adapter.schema_mode_for_tool(
                    name,
                    full_schema_names=full_schema_names,
                )
                if self._is_command_tool(self.state.tool_map.get(name, tool))
                else "full"
                for name, tool in tools.items()
            }
        return self.provider_adapter.prepare_tool_map_for_request(
            tools,
            required_tool_names=self.state.required_tool_names,
            schema_modes=schema_modes,  # type: ignore[arg-type]
        )

    def _stable_tool_map(
        self,
        tools: dict[str, ToolExecutable],
    ) -> dict[str, ToolExecutable]:
        return {
            name: tools[name]
            for name in sorted(
                tools,
                key=lambda name: self._tool_sort_key(name, tools[name]),
            )
        }

    def _tool_sort_key(
        self,
        name: str,
        tool: ToolExecutable,
    ) -> tuple[int, int, int, int, str]:
        binding = getattr(tool, "binding", None)
        if binding is None:
            # Keep catalog/agent tool order deterministic before command tools.
            return (0, 0, 0, 0, normalize_message_text(name))
        candidate = getattr(binding, "candidate", None)
        command_id = normalize_message_text(str(getattr(binding, "command_id", "")))
        selected = 1 if name in self.state.required_tool_names else 0
        exact = 1 if bool(getattr(candidate, "exact_protected", False)) else 0
        score = int(float(getattr(candidate, "score", 0.0) or 0.0) * 100)
        return (
            1,
            -selected,
            -exact,
            -score,
            command_id or normalize_message_text(name),
        )

    def _should_use_compact_command_schema(
        self,
        tool_choice: str | dict[str, Any] | None,
    ) -> bool:
        if tool_choice == "required":
            return False
        if self._has_command_observation():
            return False
        if self.state.tool_obligation == "required":
            return False
        command_tool_count = sum(
            1 for tool in self.state.tool_map.values() if self._is_command_tool(tool)
        )
        if self.provider_adapter.should_use_compact_schema(
            tool_count=command_tool_count,
        ):
            return command_tool_count > self.provider_adapter.profile.full_schema_tool_cap
        return any(
            self._is_compact_schema_candidate(tool)
            for tool in self.state.tool_map.values()
        )

    def _full_schema_tool_names_for_request(
        self,
        tool_choice: str | dict[str, Any] | None,
    ) -> set[str]:
        if tool_choice == "required" or self.state.tool_obligation == "required":
            return {
                name
                for name, tool in self.state.tool_map.items()
                if self._is_command_tool(tool)
            }
        selected: list[tuple[str, ToolExecutable]] = []
        for name, tool in self._stable_tool_map(self.state.tool_map).items():
            if not self._is_command_tool(tool):
                continue
            if self._is_full_schema_candidate(tool):
                selected.append((name, tool))
        cap = max(
            1,
            min(
                _AUTO_FULL_SCHEMA_TOOL_CAP,
                int(self.provider_adapter.profile.full_schema_tool_cap or 1),
            ),
        )
        return {name for name, _tool in selected[:cap]}

    def _is_full_schema_candidate(self, tool: ToolExecutable) -> bool:
        binding = getattr(tool, "binding", None)
        candidate = getattr(binding, "candidate", None)
        if candidate is None:
            return False
        if bool(getattr(candidate, "exact_protected", False)):
            return True
        features = getattr(candidate, "features", None)
        exact_score = float(getattr(features, "exact_score", 0.0) or 0.0)
        schema_score = float(getattr(features, "schema_score", 0.0) or 0.0)
        context_score = float(getattr(features, "context_score", 0.0) or 0.0)
        reliability_score = float(getattr(features, "reliability_score", 0.0) or 0.0)
        param_failure_score = float(
            getattr(features, "param_failure_score", 0.0) or 0.0
        )
        score = float(getattr(candidate, "score", 0.0) or 0.0)
        if reliability_score >= 8.0 and param_failure_score >= -3.0 and score >= 80.0:
            return True
        if is_high_reliability_candidate(candidate) and (
            score >= 90.0 or exact_score > 0 or schema_score + context_score >= 12.0
        ):
            return True
        return exact_score > 0 or score >= 180.0 or (
            score >= 120.0 and schema_score + context_score >= 8.0
        )

    def _is_compact_schema_candidate(self, tool: ToolExecutable) -> bool:
        binding = getattr(tool, "binding", None)
        candidate = getattr(binding, "candidate", None)
        if candidate is None:
            return False
        if is_low_reliability_candidate(candidate):
            return True
        return not self._is_full_schema_candidate(tool)

    def _uses_compact_command_schema(
        self,
        tools: dict[str, ToolExecutable] | None,
        tool_calls: list[LLMToolCall],
    ) -> bool:
        if not tools or not tool_calls:
            return False
        for tool_call in tool_calls:
            tool = tools.get(str(tool_call.function.name or ""))
            if is_compact_request_tool(tool):
                return True
            if str(getattr(tool, "chatinter_schema_mode", "") or "") == "full":
                continue
            if tool is not None and tool is not self.state.tool_map.get(
                str(tool_call.function.name or "")
            ):
                return True
        return False

    async def _resolve_compact_command_tool_calls(
        self,
        tool_calls: list[LLMToolCall],
    ) -> list[LLMToolCall]:
        selected: dict[str, ToolExecutable] = {}
        for tool_call in tool_calls:
            name = normalize_message_text(str(tool_call.function.name or ""))
            tool = self.state.tool_map.get(name)
            if tool is not None and self._is_command_tool(tool):
                selected[name] = tool
        if not selected:
            return tool_calls

        selected = self._stable_tool_map(selected)
        self.state.messages.append(
            LLMMessage.user(
                "You selected compact plugin capability card(s). "
                "Now call the selected real command tool(s) with the full schema "
                "and fill arguments from the user's current task. If the selected "
                "tool is not actually appropriate, answer briefly instead of "
                "calling it. If TaskLedger exists, map each call to one listed "
                "task goal in task_text; do not split by connector words."
            )
        )
        self.state.append_timeline(
            role="system",
            kind="compact_schema_upgrade",
            metadata={
                "step": self.state.step,
                "selected_tools": list(selected.keys()),
            },
        )
        response = await self._request_model(
            tools=selected,
            tool_choice="auto",
        )
        resolved = [
            call
            for call in list(response.tool_calls or [])
            if normalize_message_text(str(call.function.name or "")) in selected
        ]
        if resolved:
            return resolved
        final_text = normalize_message_text(str(response.text or ""))
        if final_text:
            completion_action = self._unobserved_completion_action(final_text)
            if completion_action == "retry":
                return []
            if completion_action == "safe_final":
                final_text = _SAFE_NO_TOOL_RESULT_REPLY
            self.state.complete_final(
                final_text,
                reason="compact_schema_direct_response",
            )
        return []

    async def _seed_task_graph(self) -> None:
        if not self._complex_planner_enabled():
            return
        if not self._should_use_task_graph():
            return
        graph = await self.planner.plan(
            original_goal=self.message_text,
            available_tools=self._available_tool_summaries(limit=80),
            resumed_graph=self.state.task_graph,
        )
        if graph is None:
            return
        self.state.set_task_graph(graph, source="agent_planner")
        self.state.add_pending_tasks(
            [task.goal for task in graph.incomplete_tasks],
            source="task_graph_initial",
        )
        self.state.messages.append(
            LLMMessage.user(
                "TaskGraph initialized for this superuser Agent run:\n"
                + json.dumps(graph.to_public_payload(), ensure_ascii=False)
                + "\nUse tools to satisfy acceptance_criteria. Final answer must not "
                "claim unfinished tasks are complete."
            )
        )
        self._persist_state("task_graph_initialized", task_count=len(graph.tasks))

    async def _verify_task_graph_after_observation(self) -> None:
        if not self._complex_verifier_enabled():
            return
        if self.state.task_graph is None or not self.state.observations:
            return
        observation = self.state.observations[-1]
        if not observation.tool_name:
            return
        result = await self.verifier.verify_observation(
            graph=self.state.task_graph,
            observation=observation,
            available_tools=self._available_tool_summaries(limit=40),
        )
        incomplete = self.state.incomplete_task_goals()
        self.state.replace_pending_tasks(
            incomplete,
            source="task_graph_verifier",
        )
        self.state.append_timeline(
            role="system",
            kind="task_graph_verification",
            metadata={
                "step": self.state.step,
                "mode": "after_observation",
                "updates": [item.model_dump() for item in result.updates],
                "missing_tasks": list(result.missing_tasks),
                "reason": normalize_message_text(result.reason),
                "graph": self.state.task_graph.to_public_payload(),
            },
        )

    async def _seed_task_ledger(self) -> None:
        if not self._should_use_task_ledger():
            return
        capabilities = self._available_tool_summaries(limit=0)
        self.state.refresh_capability_ledger(capabilities)
        result = await self.coverage_judge.plan_task_ledger(
            original_message=self.message_text,
            available_capabilities=self.state.capability_ledger.public_entries(
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
                    expected_capabilities=_normalized_tasks(
                        item.expected_capabilities
                    ),
                    acceptance_criteria=_normalized_tasks(
                        item.acceptance_criteria
                    ),
                    reason=normalize_message_text(item.reason),
                )
            )
        if not tasks:
            return
        ledger = TaskLedger.create(
            original_message=self.message_text,
            tasks=tasks,
            reason=result.reason,
        )
        self.state.set_task_ledger(ledger, source="llm_task_ledger")
        self.state.replace_pending_tasks(
            ledger.incomplete_goals,
            source="task_ledger_initial",
        )
        self.state.messages.append(
            LLMMessage.user(
                "TaskLedger initialized by semantic task listing:\n"
                + json.dumps(ledger.to_public_payload(), ensure_ascii=False)
                + "\nComplete tasks by calling real tools when requires_real_tool=true. "
                "Do not rely on local connector words; each tool call should map to "
                "one listed task when possible. If no visible command schema can "
                "cover a ledger task, call retrieve_plugin_commands first instead "
                "of declaring the task unsupported."
            )
        )
        self._persist_state("task_ledger_initialized", task_count=len(ledger.tasks))

    async def _final_acceptance_action(
        self,
        final_text: str,
        *,
        allow_retry: bool = True,
    ) -> str:
        """Single final gate: obligation -> ledger -> final reply validation.

        TaskGraph and CoverageJudge are allowed to prepare context earlier, but
        final acceptance is intentionally linear and evidence based:
        ToolIntentGate says whether a real tool is needed; TaskLedger says what
        tasks exist; Observations prove completion; FinalValidator blocks only
        over-claiming/hallucinated completion.
        """

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
        if self.state.task_ledger is None:
            return ""
        self.state.refresh_capability_ledger(self._available_tool_summaries(limit=0))
        missing_goals = _normalized_tasks(self.state.task_ledger.incomplete_goals)
        self.state.append_timeline(
            role="system",
            kind="task_ledger_coverage",
            metadata={
                "step": self.state.step,
                "mode": "observation_evidence_only",
                "covered": not missing_goals,
                "missing_tasks": missing_goals,
                "ledger": self.state.task_ledger.to_public_payload(),
                "observation_count": len(self.state.observations),
            },
        )
        if not missing_goals:
            return ""
        if (
            not allow_retry
            or self.state.coverage_interceptions >= _TASK_COVERAGE_INTERCEPT_LIMIT
        ):
            self.state.replace_pending_tasks(missing_goals, source="task_ledger_final")
            return "partial_final"
        self.state.coverage_interceptions += 1
        self.state.replace_pending_tasks(missing_goals, source="task_ledger_final")
        self._force_required_tool_choice_once = self._has_actionable_command_tools()
        self.state.append_guardrail_observation(
            {
                "ok": False,
                "status": "runtime_task_ledger_coverage",
                "guardrail_reason": "task_ledger_missing",
                "reason": "task_ledger_missing",
                "covered": False,
                "task_ledger": self.state.task_ledger.to_public_payload(),
                "capability_ledger": self.state.capability_ledger.public_entries(
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
        if (
            self.state.tool_obligation != "required"
            or not self._has_available_required_tools()
            or self._has_command_observation()
        ):
            return ""
        if (
            not allow_retry
            or self.state.direct_answer_interceptions >= _DIRECT_ANSWER_INTERCEPT_LIMIT
        ):
            return "safe_final"
        self.state.direct_answer_interceptions += 1
        self._force_required_tool_choice_once = True
        self.state.append_guardrail_observation(
            {
                "ok": False,
                "status": "runtime_tool_obligation",
                "guardrail_reason": "tool_required_but_model_answered_directly",
                "reason": "tool_required_but_model_answered_directly",
                "available_tools": self._available_tool_summaries(limit=16),
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
        validation = validate_final_reply(
            final_text=final_text,
            tool_obligation=self.state.tool_obligation,
            observations=self.state.observations,
            pending_tasks=self.state.pending_tasks,
            tool_map=self.state.tool_map,
        )
        if validation.ok:
            return ""
        if (
            not allow_retry
            or self.state.final_validation_interceptions
            >= _FINAL_VALIDATION_INTERCEPT_LIMIT
        ):
            return "safe_final"
        self.state.final_validation_interceptions += 1
        self._force_required_tool_choice_once = self._has_available_required_tools()
        self.state.append_guardrail_observation(
            {
                "ok": False,
                "status": "runtime_final_validation",
                "guardrail_reason": validation.reason,
                "reason": validation.reason,
                "requires_observation": validation.requires_observation,
                "successful_observations": validation.successful_observations,
                "action_like_tools": validation.action_like_tools,
                "available_tools": self._available_tool_summaries(limit=16),
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

    def _has_available_required_tools(self) -> bool:
        if self.state.required_tool_names:
            return any(
                name in self.state.tool_map for name in self.state.required_tool_names
            )
        return any(self._is_command_tool(tool) for tool in self.state.tool_map.values())

    def _has_command_observation(self) -> bool:
        return any(observation.command_id for observation in self.state.observations)

    def _has_successful_command_observation(self) -> bool:
        return any(
            observation.command_id and observation.ok
            for observation in self.state.observations
        )

    def _should_complete_after_plugin_observation(self) -> bool:
        """Avoid an extra final LLM turn after visible group plugin output.

        Plugin commands have already sent through NoneBot reroute.  When there
        is no pending continuation, returning an empty final reply saves one
        model call and avoids duplicating the plugin's own response.
        """

        extra = getattr(self.run_context, "extra", None)
        if isinstance(extra, dict) and bool(extra.get("enable_agent_tools")):
            return False
        if self.state.task_graph is not None:
            return False
        if self.state.task_ledger is not None:
            return False
        if self.state.pending_tasks:
            return False
        if self._looks_like_multi_task_turn():
            return False
        if not self.state.observations:
            return False
        recent_commands = [
            observation
            for observation in self.state.observations
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

    def _should_use_task_graph(self) -> bool:
        extra = getattr(self.run_context, "extra", None)
        if not isinstance(extra, dict) or not bool(extra.get("enable_agent_tools")):
            return False
        if self.state.task_graph is not None:
            return True
        return bool(self.message_text and self.state.tool_map)

    def _should_use_task_ledger(self) -> bool:
        if self.state.task_ledger is not None:
            return True
        if not self._looks_like_multi_task_turn():
            return False
        if not self.message_text or not self.state.tool_map:
            return False
        if self.state.observations or self.state.pending_tasks:
            return False
        if self.state.tool_obligation == "required":
            return True
        return self._actionable_command_tool_count() > 1

    def _looks_like_multi_task_turn(self) -> bool:
        text = self.message_text
        if not text:
            return False
        if any(marker in text for marker in ("然后", "最后", "顺便", "接着", "同时", "以及")):
            return True
        return text.count("，") + text.count(",") >= 2 and self._actionable_command_tool_count() > 2

    def _agent_complexity_metadata(self) -> dict[str, Any]:
        extra = getattr(self.run_context, "extra", None)
        metadata = extra.get("agent_complexity") if isinstance(extra, dict) else None
        return dict(metadata) if isinstance(metadata, dict) else {}

    def _complex_planner_enabled(self) -> bool:
        metadata = self._agent_complexity_metadata()
        if metadata:
            return bool(metadata.get("enable_planner"))
        return self.state.agent_complexity_mode in {"complex_pev", "standard"}

    def _complex_verifier_enabled(self) -> bool:
        metadata = self._agent_complexity_metadata()
        if metadata:
            return bool(metadata.get("enable_verifier"))
        return self.state.agent_complexity_mode in {"complex_pev", "standard"}

    def _observation_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for observation in self.state.observations[-12:]:
            output = observation.output or {}
            messages_sent = output.get("messages_sent")
            messages_sent_summary = normalize_message_text(
                str(output.get("messages_sent_summary", "") or "")
            )
            artifacts = output.get("artifacts")
            payloads.append(
                {
                    "ok": observation.ok,
                    "command_id": observation.command_id,
                    "rendered_command": observation.rendered_command,
                    "matched_plugin": observation.matched_plugin,
                    "task_text": observation.task_text,
                    "status": output.get("status", "success" if observation.ok else "failed"),
                    "need_continue": observation.need_continue,
                    "remaining_task_hint": observation.remaining_task_hint,
                    "error": observation.error,
                    "messages_sent_summary": messages_sent_summary,
                    "visible_output": bool(output.get("visible_output"))
                    or bool(messages_sent_summary),
                    "messages_sent": _compact_list(messages_sent, limit=4),
                    "artifacts": _compact_artifacts(artifacts),
                }
            )
        return payloads

    def _partial_task_ledger_reply(self) -> str:
        if self.state.task_ledger is None:
            return "我还没能确认任务账本是否全部完成。"
        missing = [task.goal for task in self.state.task_ledger.pending_tasks[:5]]
        if not missing:
            return "我还没能确认任务账本是否全部完成。"
        return "我只完成了部分任务，任务账本里这些还没验收完成：" + "；".join(missing)

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
    def _is_command_tool(tool: ToolExecutable) -> bool:
        binding = getattr(tool, "binding", None)
        return bool(normalize_message_text(str(getattr(binding, "command_id", ""))))

    def _is_actionable_command_tool(self, tool: ToolExecutable) -> bool:
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

    def _apply_guardrail(
        self,
        decision: RuntimeGuardrailDecision,
        *,
        as_message: bool,
    ) -> None:
        if decision.should_block_tool:
            self.guardrails.block_tool(decision.tool_name)
        self.state.append_guardrail_observation(
            decision.to_payload(),
            as_message=as_message,
        )
        self._sync_dynamic_tools()

    def _pause_if_needed_after_observation(self, tool_result: ToolResult) -> bool:
        output = tool_result.output if isinstance(tool_result.output, dict) else {}
        if bool(output.get("approval_required")):
            approval_id = _first_text(
                output.get("approval_id"),
                _nested_get(output, "approval", "approval_id"),
            )
            approval_ids = [approval_id] if approval_id else list(
                self.state.waiting_approval_ids[-3:]
            )
            self.state.pause(
                reason="approval_required",
                final_text=_approval_pause_reply(output, approval_ids=approval_ids),
                cursor={
                    "type": "approval",
                    "approval_ids": approval_ids,
                    "resume_instruction": (
                        "After the user approves, call approve_pending_action "
                        "then agent_run_resume."
                    ),
                },
            )
            return True
        return False

    async def _background_pause_if_needed_after_observation(
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
        actor = self._actor_from_run_context()
        event = await wait_for_observation_event(
            task_id=task_id,
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            after_event_id=_first_text(
                _nested_get(output, "observation_event", "event_id"),
                output.get("event_id"),
            ),
            timeout=_BACKGROUND_OBSERVATION_WAIT_SECONDS,
            terminal_only=True,
        )
        if event is not None:
            self._append_background_observation_event(event)
            await self._verify_task_graph_after_observation()
            self._persist_state(
                "background_observation_event",
                task_id=task_id,
                event_id=event.event_id,
                status=event.status,
            )
            return False
        self.state.pause(
            reason="waiting_background_task",
            final_text=(
                f"后台任务已启动，task_id={task_id}。"
                "它还在运行；稍后我可以自动/手动继续查看结果并完成后续步骤。"
            ),
            cursor={
                "type": "background_task",
                "task_ids": [task_id],
                "last_event_ids": list(self.state.observation_event_ids[-10:]),
                "resume_instruction": (
                    "Wait for ObservationEvent or call background_task_status, "
                    "then continue the task."
                ),
            },
        )
        return True

    def _append_background_observation_event(self, event: ObservationEvent) -> None:
        payload = event.public_payload()
        if event.event_id not in self.state.observation_event_ids:
            self.state.observation_event_ids.append(event.event_id)
        for artifact in payload.get("artifacts", []) or []:
            if not isinstance(artifact, dict):
                continue
            artifact_id = normalize_message_text(str(artifact.get("artifact_id") or ""))
            if artifact_id and artifact_id not in self.state.artifact_refs:
                self.state.artifact_refs.append(artifact_id)
        self.state.messages.append(
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
            need_continue=event.status not in {"completed", "failed", "cancelled", "error"},
            error=event.error or event.stderr_tail if event.status != "completed" else "",
            artifacts=tuple(payload.get("artifacts", []) or []),
            step=self.state.step,
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
        self.state.append_synthetic_observation(
            synthetic,
            timeline_kind="background_observation_event",
            content=payload.get("output_tail", "") or payload.get("stderr_tail", ""),
            metadata={"event": payload},
        )
        self._sync_todos_from_latest_observation()

    def _actor_from_run_context(self) -> dict[str, str]:
        extra = getattr(self.run_context, "extra", None)
        session_key = str(getattr(self.run_context, "session_id", "") or "")
        user_id = ""
        if isinstance(extra, dict):
            user_id = str(extra.get("actor_user_id", "") or "")
        user_id = user_id or session_key or "unknown"
        return {"user_id": user_id, "session_key": session_key or user_id}

    def _persist_state(self, stage: str, **metadata: Any) -> None:
        persist_agent_run_state(self.state, stage=stage, metadata=metadata)

    def _sync_todos_from_latest_observation(self) -> None:
        extra = getattr(self.run_context, "extra", None)
        if not isinstance(extra, dict) or not bool(extra.get("enable_agent_tools")):
            return
        if not self.state.observations:
            return
        actor = self._actor_from_run_context()
        observation = self.state.observations[-1]
        payload = {
            "ok": observation.ok,
            "status": observation.output.get("status", ""),
            "tool_name": observation.tool_name,
            "task_text": observation.task_text,
            "artifacts": list(observation.artifacts),
        }
        try:
            todo_list = update_todo_from_observation(
                user_id=actor["user_id"],
                session_key=actor["session_key"],
                observation=payload,
            )
        except Exception:
            return
        if todo_list is None:
            return
        self.state.append_timeline(
            role="system",
            kind="todo_sync",
            metadata={
                "step": self.state.step,
                "summary": todo_list.public_payload().get("summary", {}),
            },
        )

    def _cancelled_externally(self) -> bool:
        snapshot = get_agent_run_snapshot(self.state.run_id or self.state.trace_id)
        return isinstance(snapshot, dict) and str(snapshot.get("status", "")) == "cancelled"

    def _compress_context_if_needed(self) -> None:
        result = compress_agent_messages(
            self.state.messages,
            trace_id=self.state.trace_id,
        )
        if not result.changed:
            return
        self.state.messages = result.messages
        self.state.append_timeline(
            role="system",
            kind="context_compression",
            content=result.summary,
            metadata={
                "step": self.state.step,
                "before_tokens": result.before_tokens,
                "after_tokens": result.after_tokens,
                "compressed_tool_pairs": result.compressed_tool_pairs,
                "summarized_messages": result.summarized_messages,
            },
        )
        self._persist_state(
            "context_compressed",
            before_tokens=result.before_tokens,
            after_tokens=result.after_tokens,
            compressed_tool_pairs=result.compressed_tool_pairs,
            summarized_messages=result.summarized_messages,
        )

    def _compact_tool_result_for_context(
        self,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
    ) -> ToolResult:
        raw_output: Any = tool_result.output
        if not isinstance(raw_output, dict):
            raw_output = build_command_observation(
                ok=False,
                command_id="",
                rendered_command=str(tool_call.function.name or ""),
                matched_plugin=str(tool_call.function.name or ""),
                task_text="",
                error=normalize_message_text(str(raw_output or "")),
                retryable=False,
                trace_id=self.state.trace_id,
            )
        output = compact_tool_result_output(
            raw_output,
            trace_id=self.state.trace_id,
            source=f"tool_result:{tool_call.function.name}",
        )
        return ToolResult(
            output=output,
            display_content=summarize_artifact_text(
                str(tool_result.display_content or output.get("status", ""))
            ),
        )

    def _tool_result_for_model(
        self,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
    ) -> dict[str, Any]:
        if isinstance(tool_result.output, dict):
            return compact_tool_result_output(
                tool_result.output,
                trace_id=self.state.trace_id,
                source=f"model_payload:{tool_call.function.name}",
            )
        return compact_tool_result_output(
            build_command_observation(
                ok=False,
                command_id="",
                rendered_command=str(tool_call.function.name or ""),
                matched_plugin=str(tool_call.function.name or ""),
                task_text="",
                error=normalize_message_text(str(tool_result.output or "")),
                retryable=False,
                trace_id=self.state.trace_id,
            ),
            trace_id=self.state.trace_id,
            source=f"model_payload:{tool_call.function.name}",
        )

    def _normalize_command_tool_result(
        self,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
    ) -> ToolResult:
        output = tool_result.output
        if isinstance(output, dict) and (
            "messages_sent" in output
            or "remaining_task_hint" in output
            or output.get("status") in {"retrieved", "capability_candidates_retrieved"}
        ):
            return tool_result

        executable = self.state.tool_map.get(str(tool_call.function.name or ""))
        binding = getattr(executable, "binding", None)
        candidate = getattr(binding, "candidate", None)
        if candidate is None:
            return tool_result

        arguments = _parse_tool_arguments(str(tool_call.function.arguments or ""))
        task_text = ""
        if isinstance(arguments, dict):
            task_text = normalize_message_text(
                str(arguments.get(TASK_TEXT_FIELD) or "")
            )
        error = ""
        if isinstance(output, dict):
            error = normalize_message_text(
                str(output.get("message", "") or output.get("error", "") or output)
            )
        else:
            error = normalize_message_text(
                str(output or tool_result.display_content or "")
            )
        return ToolResult(
            output=build_command_observation(
                ok=False,
                command_id=getattr(binding, "command_id", ""),
                rendered_command=getattr(candidate.schema, "head", ""),
                matched_plugin=getattr(candidate, "plugin_name", ""),
                task_text=task_text,
                ambient_message=self.message_text,
                trace_id=self.state.trace_id,
                error=error or "命令工具执行失败。",
                retryable=False,
                plugin_module=getattr(candidate, "plugin_module", ""),
            ),
            display_content=tool_result.display_content,
        )

    def _exception_tool_result(
        self,
        tool_call: LLMToolCall,
        exc: Exception,
    ) -> ToolResult:
        executable = self.state.tool_map.get(str(tool_call.function.name or ""))
        binding = getattr(executable, "binding", None)
        candidate = getattr(binding, "candidate", None)
        arguments = _parse_tool_arguments(str(tool_call.function.arguments or ""))
        task_text = ""
        if isinstance(arguments, dict):
            task_text = normalize_message_text(
                str(arguments.get(TASK_TEXT_FIELD) or "")
            )
        if candidate is None:
            output = {
                "ok": False,
                "command_id": "",
                "rendered_command": "",
                "matched_plugin": "",
                "task_text": task_text,
                "error": f"工具执行异常：{type(exc).__name__}: {exc}",
                "messages_sent": [],
                "artifacts": [],
                "need_continue": True,
                "remaining_task_hint": task_text,
                "retryable": False,
                "status": "tool_execution_exception",
            }
        else:
            output = build_command_observation(
                ok=False,
                command_id=getattr(binding, "command_id", ""),
                rendered_command=getattr(candidate.schema, "head", ""),
                matched_plugin=getattr(candidate, "plugin_name", ""),
                task_text=task_text,
                ambient_message=self.message_text,
                trace_id=self.state.trace_id,
                error=f"工具执行异常：{type(exc).__name__}: {exc}",
                retryable=False,
                plugin_module=getattr(candidate, "plugin_module", ""),
            )
            output["status"] = "tool_execution_exception"
        return ToolResult(
            output=output,
            display_content=normalize_message_text(str(output.get("error", ""))),
        )

    def _record_tool_feedback(
        self,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
        *,
        latency_ms: int,
    ) -> None:
        output = tool_result.output if isinstance(tool_result.output, dict) else {}
        if not output.get("command_id"):
            return
        executable = self.state.tool_map.get(str(tool_call.function.name or ""))
        binding = getattr(executable, "binding", None)
        candidate = getattr(binding, "candidate", None)
        selected_rank = 0
        command_id = normalize_message_text(str(output.get("command_id", "")))
        if command_id:
            command_tools = [
                tool
                for tool in self.state.tool_map.values()
                if getattr(tool, "binding", None) is not None
            ]
            command_tools.sort(
                key=lambda tool: float(
                    getattr(
                        getattr(getattr(tool, "binding", None), "candidate", None),
                        "score",
                        0.0,
                    )
                    or 0.0
                ),
                reverse=True,
            )
            selected_rank = next(
                (
                    index
                    for index, tool in enumerate(command_tools, 1)
                    if normalize_message_text(
                        str(getattr(getattr(tool, "binding", None), "command_id", ""))
                    )
                    == command_id
                ),
                0,
            )
        selected_score = float(getattr(candidate, "score", 0.0) or 0.0)
        selected_reason = normalize_message_text(str(getattr(candidate, "reason", "")))
        try:
            record_command_observation_feedback(
                output=output,
                action="execute",
                session_id=self.state.session_key,
                latency_ms=latency_ms,
                selected_rank=selected_rank,
                selected_score=selected_score,
                selected_reason=selected_reason,
            )
        except Exception:
            return

    def _record_trajectory_once(
        self,
        *,
        started_at: float,
        latency_ms: int,
    ) -> None:
        if self._trajectory_recorded:
            return
        self._trajectory_recorded = True
        try:
            self.state.capture_budget(self.budget_controller)
            record_agent_trajectory(
                state=self.state,
                input_message=self.message_text,
                started_at=started_at,
                latency_ms=latency_ms,
                run_context_extra=dict(self.run_context.extra or {}),
            )
        except Exception:
            return


def _parse_tool_arguments(arguments: str) -> dict[str, Any] | str:
    text = str(arguments or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return text
    return value if isinstance(value, dict) else {"value": value}


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
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        summary = normalize_message_text(str(item.get("summary", "") or ""))
        artifact_type = normalize_message_text(str(item.get("type", "") or ""))
        artifact_id = normalize_message_text(str(item.get("artifact_id", "") or ""))
        result.append(
            {
                "artifact_id": artifact_id,
                "type": artifact_type,
                "summary": summary[:240],
            }
        )
    return result


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


def _approval_pause_reply(
    output: dict[str, Any],
    *,
    approval_ids: list[str],
) -> str:
    raw_approval = output.get("approval")
    approval: dict[str, Any] = (
        raw_approval if isinstance(raw_approval, dict) else {}
    )
    action = normalize_message_text(str(approval.get("action", "") or ""))
    reason = normalize_message_text(
        str(
            approval.get("reason", "")
            or _nested_get(output, "permission", "reason")
            or ""
        )
    )
    approval_text = "、".join(approval_ids) if approval_ids else "未知"
    parts = [f"这个操作需要确认，approval_id={approval_text}。"]
    if action:
        parts.append(f"操作：{action}。")
    if reason:
        parts.append(f"原因：{reason}。")
    parts.append("你确认后我会继续执行。")
    return "".join(parts)


def _estimate_prompt_tokens(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        if message.tool_calls:
            for call in message.tool_calls:
                function = getattr(call, "function", None)
                total += estimate_text_tokens(str(getattr(function, "name", "") or ""))
                total += estimate_text_tokens(
                    str(getattr(function, "arguments", "") or "")
                )
        content = message.content
        if isinstance(content, str):
            total += estimate_text_tokens(content)
            continue
        for part in content:
            if not isinstance(part, LLMContentPart):
                continue
            total += estimate_text_tokens(part.text or part.thought_text or "")
            if part.image_source:
                total += 48
    return total


__all__ = [
    "AgentRuntime",
    "AgentRuntimeResult",
    "AgentRuntimeTimelineItem",
]
