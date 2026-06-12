"""Agent state machine runtime for ChatInter native command tools.

The runtime owns the model/tool loop:
LLM -> tool_calls -> execute -> observations -> LLM ... -> final text.
"""

from __future__ import annotations

import json
import asyncio
import time
from typing import Any

from zhenxun.services.llm import AI, LLMMessage
from zhenxun.services.llm.tools import RunContext, ToolInvoker
from zhenxun.services.llm.types.models import (
    LLMResponse,
    LLMToolCall,
    ToolResult,
)
from zhenxun.services.llm.types.protocols import ToolExecutable

from .agent_run_store import (
    clear_agent_run_cancel_signal,
    get_agent_run_snapshot,
    is_agent_run_cancel_signaled,
    persist_agent_run_state,
)
from .config import get_fallback_models
from .provider_failover import request_with_failover
from .agent_runtime_chain import AgentRuntimeChain
from .agent_state import (
    AgentRunState,
    AgentRuntimeResult,
)
from .artifact_store import compact_tool_result_output, summarize_artifact_text
from .command_observation import build_command_observation
from .context_engine import get_context_engine
from .feedback import record_command_observation_feedback
from .provider_capability import (
    ProviderCapabilityAdapter,
)
from .route_text import normalize_message_text
from .task_frame import TASK_TEXT_FIELD
from .trajectory_store import record_agent_trajectory
from .turn_runtime import TurnBudgetController, estimate_text_tokens

_MAIN_STAGE = "main_request"
_PROGRESS_TASKS: set[asyncio.Task] = set()

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
        progress_hook: Any | None = None,
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
        self.chain = AgentRuntimeChain(
            state=state,
            run_context=run_context,
            message_text=self.message_text,
            model_name=model_name,
            generation_config=generation_config,
            timeout=timeout,
            provider_adapter=self.provider_adapter,
            sync_tools=self._sync_dynamic_tools,
            persist_state=self._persist_state,
        )
        self._trajectory_recorded = False
        self._progress_hook = progress_hook
        self._run_started_monotonic = 0.0
        self._last_progress_sent = 0.0

    # 进度回显(P0-2 最小落地):长任务期间向用户播报步数与当前工具,
    # 首次播报延迟 + 播报间隔节流,避免刷屏。
    _PROGRESS_FIRST_DELAY = 10.0
    _PROGRESS_MIN_INTERVAL = 12.0

    def _emit_progress(self) -> None:
        if self._progress_hook is None:
            return
        now = time.monotonic()
        elapsed = now - self._run_started_monotonic
        if elapsed < self._PROGRESS_FIRST_DELAY:
            return
        if now - self._last_progress_sent < self._PROGRESS_MIN_INTERVAL:
            return
        self._last_progress_sent = now
        last_tool = ""
        for observation in reversed(self.state.observations):
            last_tool = normalize_message_text(str(observation.tool_name or ""))
            if last_tool:
                break
        text = (
            f"⏳ 执行中 第{self.state.step}/{self.state.max_steps}步"
            + (f" · 上一步: {last_tool}" if last_tool else "")
        )

        async def _safe_send() -> None:
            try:
                await self._progress_hook(text)
            except Exception:
                pass

        task = asyncio.ensure_future(_safe_send())
        _PROGRESS_TASKS.add(task)
        task.add_done_callback(_PROGRESS_TASKS.discard)

    async def run(self) -> AgentRuntimeResult:
        started_at = time.time()
        started_perf = time.perf_counter()
        self._run_started_monotonic = time.monotonic()
        if self.state.status != "running":
            self.state.status = "running"
            self.state.paused_reason = ""
        clear_agent_run_cancel_signal(self.state.run_id or self.state.trace_id)
        self._persist_state("started")
        try:
            await self.chain.initialize()
            while self.state.step < self.state.max_steps:
                if self._cancelled_externally():
                    self.state.cancel(reason="cancelled_by_agent_run_cancel")
                    self._persist_state("cancelled")
                    return self.state.to_result()
                if self.state.token_budget_exceeded():
                    self._persist_state(
                        "guardrail",
                        reason="token_budget_exhausted",
                    )
                    await self._force_final_response(
                        reason="token_budget_exhausted"
                    )
                    return self.state.to_result()
                self.state.start_step()
                self._emit_progress()
                self._sync_dynamic_tools()
                self._persist_state("step_started")
                self._compress_context_if_needed()
                request_guardrail = self.chain.before_model_request()
                if request_guardrail is not None:
                    self.chain.apply_guardrail(request_guardrail, as_message=True)
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
                tool_choice = self.chain.tool_choice_for_request()
                tools_for_request = self.chain.tools_for_request(tool_choice)
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
                    final_decision = await self.chain.complete_if_acceptable(
                        final_text
                    )
                    if final_decision.action == "retry":
                        self._persist_state("final_acceptance_retry")
                        continue
                    self._persist_state("completed", reason=self.state.stop_reason)
                    return self.state.to_result()

                if not self._allow_tool_batch(len(tool_calls)):
                    budget_guardrail = self.chain.on_budget_exhausted(
                        call_count=len(tool_calls)
                    )
                    self.chain.apply_guardrail(budget_guardrail, as_message=True)
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
                force_final_reason = ""
                batch_all_read_only = bool(tool_calls)
                for tool_call in tool_calls:
                    tool_call_started = time.perf_counter()
                    pre_guardrail = (
                        self.chain.before_tool_call(tool_call)
                        if not force_final_reason
                        else self.chain.runtime_stopping_decision(tool_call)
                    )
                    if pre_guardrail is not None:
                        self.chain.apply_guardrail(pre_guardrail, as_message=False)
                        self._persist_state(
                            "guardrail",
                            reason=pre_guardrail.reason,
                            tool_name=str(tool_call.function.name or ""),
                        )
                        resolved_call = tool_call
                        tool_result = self.chain.tool_result_for_decision(
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
                    if batch_all_read_only and (
                        pre_guardrail is not None
                        or not self._is_read_only_tool(
                            str(resolved_call.function.name or "")
                        )
                    ):
                        batch_all_read_only = False
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
                        provider_adapter=self.provider_adapter,
                    )
                    self._persist_state(
                        "tool_observation",
                        tool_name=str(resolved_call.function.name or ""),
                    )
                    post_observation = await self.chain.after_tool_observation(
                        tool_call=resolved_call,
                        tool_result=tool_result,
                        pre_guardrail=pre_guardrail,
                    )
                    if post_observation.paused:
                        return self.state.to_result()
                    if post_observation.stop_reason:
                        force_final_reason = post_observation.stop_reason
                self._record_tool_batch(time.perf_counter() - started)
                self.chain.flush_post_batch_guardrails()
                if (
                    batch_all_read_only
                    and not force_final_reason
                    and self.state.refund_step()
                ):
                    self._persist_state(
                        "step_refunded", reason="read_only_tool_batch"
                    )
                if force_final_reason:
                    await self._force_final_response(reason=force_final_reason)
                    return self.state.to_result()
                if self.chain.should_finish_after_observation():
                    self.state.complete_final(
                        "",
                        reason="tool_completed_no_final_llm",
                    )
                    self._persist_state("completed", reason=self.state.stop_reason)
                    return self.state.to_result()

            max_steps_guardrail = self.chain.on_max_steps()
            self.chain.apply_guardrail(max_steps_guardrail, as_message=True)
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
        async def _do_request(model: str | None) -> LLMResponse:
            # 每次尝试都重新 prepare:上下文压缩(context_overflow 恢复)
            # 修改的是 state.messages,需要重新展开进请求。
            request = self.provider_adapter.prepare_model_request(
                messages=self.state.messages,
                tools=tools,
                tool_choice=tool_choice,
                required_tool_names=self.state.required_tool_names,
                generation_config=self.generation_config,
            )
            return await self.ai.generate_internal(
                request.messages,
                model=model,
                config=request.generation_config,
                tools=request.tools,
                tool_choice=request.tool_choice,
                timeout=self.timeout,
            )

        outcome = await request_with_failover(
            primary_model=self.model_name,
            fallback_models=get_fallback_models(),
            request_fn=_do_request,
            compress_fn=self._compress_context_if_needed,
            trace_id=self.state.trace_id,
        )
        if outcome.attempts:
            self.state.append_timeline(
                role="system",
                kind="provider_failover",
                metadata={
                    "used_model": outcome.used_model or "<default>",
                    "attempts": [
                        {
                            "model": item.model,
                            "kind": item.kind,
                            "error": item.error,
                        }
                        for item in outcome.attempts
                    ],
                },
            )
        return outcome.response

    async def _force_final_response(self, *, reason: str) -> None:
        self.state.transition_force_final(reason)
        self._compress_context_if_needed()
        self._persist_state("force_final_requested", reason=reason)
        response = await self._request_model(tools=None, tool_choice=None)
        final_text = normalize_message_text(str(response.text or ""))
        final_decision = await self.chain.complete_if_acceptable(
            final_text,
            allow_retry=False,
            fallback_reason=reason,
        )
        self._persist_state("completed", reason=final_decision.reason or reason)

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
        capability_registry = extra.get("capability_registry")
        if capability_registry is not None and hasattr(
            capability_registry, "executable_tool_map"
        ):
            merged_tools = capability_registry.executable_tool_map()
        else:
            merged_tools = self.base_tool_map
        self.state.tool_map = self.chain.filter_tool_map(merged_tools)

    def _uses_compact_command_schema(
        self,
        tools: dict[str, ToolExecutable] | None,
        tool_calls: list[LLMToolCall],
    ) -> bool:
        return self.provider_adapter.uses_compact_command_schema(
            request_tools=tools,
            base_tool_map=self.state.tool_map,
            tool_calls=tool_calls,
        )

    async def _resolve_compact_command_tool_calls(
        self,
        tool_calls: list[LLMToolCall],
    ) -> list[LLMToolCall]:
        selected = self.provider_adapter.selected_command_tools(
            self.state.tool_map,
            tool_calls,
        )
        if not selected:
            return tool_calls

        self.state.messages.append(
            LLMMessage.user(self.provider_adapter.compact_schema_upgrade_prompt())
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
            completion_decision = await self.chain.complete_if_acceptable(
                final_text,
                fallback_reason="compact_schema_direct_response",
            )
            if completion_decision.action == "retry":
                return []
        return []

    def _persist_state(self, stage: str, **metadata: Any) -> None:
        persist_agent_run_state(self.state, stage=stage, metadata=metadata)

    def _cancelled_externally(self) -> bool:
        run_key = self.state.run_id or self.state.trace_id
        if is_agent_run_cancel_signaled(run_key):
            return True
        snapshot = get_agent_run_snapshot(run_key)
        return isinstance(snapshot, dict) and str(snapshot.get("status", "")) == "cancelled"

    _READ_ONLY_TOOL_NAMES = frozenset(
        {
            "retrieve_plugin_commands",
        }
    )

    def _is_read_only_tool(self, tool_name: str) -> bool:
        """Read-only tools refund their loop step (P0-1, Hermes-style)."""
        if not tool_name:
            return False
        if tool_name in self._READ_ONLY_TOOL_NAMES:
            return True
        try:
            from .superuser_agent.registry import get_superuser_tool_card

            card = get_superuser_tool_card(tool_name)
        except Exception:
            return False
        return bool(card is not None and getattr(card, "read_only", False))

    def _compress_context_if_needed(self) -> None:
        result = get_context_engine().compress(
            self.state.messages,
            trace_id=self.state.trace_id,
            context_refs=self._context_refs(),
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
                "deduped_messages": result.deduped_messages,
                "pruned_tool_results": result.pruned_tool_results,
                "protected_messages": result.protected_messages,
                "compression_fingerprint": result.compression_fingerprint,
                "policy_trace": list(result.policy_trace),
            },
        )
        self._persist_state(
            "context_compressed",
            before_tokens=result.before_tokens,
            after_tokens=result.after_tokens,
            compressed_tool_pairs=result.compressed_tool_pairs,
            summarized_messages=result.summarized_messages,
            deduped_messages=result.deduped_messages,
            pruned_tool_results=result.pruned_tool_results,
            protected_messages=result.protected_messages,
            compression_fingerprint=result.compression_fingerprint,
            policy_trace=list(result.policy_trace),
        )

    def _context_refs(self) -> dict[str, Any]:
        refs: dict[str, Any] = {
            "run_id": self.state.run_id,
            "trace_id": self.state.trace_id,
            "waiting_approval_ids": list(self.state.waiting_approval_ids[-10:]),
            "background_task_ids": list(self.state.background_task_ids[-10:]),
            "observation_event_ids": list(self.state.observation_event_ids[-10:]),
            "artifact_refs": list(self.state.artifact_refs[-12:]),
            "pending_tasks": [
                {
                    "text": task.text,
                    "command_id": task.command_id,
                }
                for task in self.state.pending_tasks[-10:]
            ],
            "incomplete_task_goals": self.state.incomplete_task_goals()[-10:],
        }
        if self.state.task_graph is not None:
            refs["task_graph"] = {
                "graph_id": self.state.task_graph.graph_id,
                "incomplete_tasks": [
                    {
                        "task_id": task.task_id,
                        "goal": task.goal,
                    }
                    for task in self.state.task_graph.incomplete_tasks[:10]
                ],
            }
        if self.state.task_ledger is not None:
            refs["task_ledger"] = [
                {
                    "task_id": getattr(task, "task_id", ""),
                    "goal": getattr(task, "goal", ""),
                    "status": getattr(task, "status", ""),
                }
                for task in self.state.task_ledger.tasks
                if getattr(task, "status", "") not in {"completed", "unsupported"}
            ][:10]
        refs["observations"] = [
            {
                "tool_call_id": observation.tool_call_id,
                "command_id": observation.command_id,
                "task_text": observation.task_text,
                "output": observation.output or {},
            }
            for observation in self.state.observations[-12:]
        ]
        return refs

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


def _estimate_prompt_tokens(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        total += estimate_text_tokens(getattr(message, "content", ""))
    return total
