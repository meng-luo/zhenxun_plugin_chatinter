"""Agent state machine runtime for ChatInter native command tools.

The runtime owns the model/tool loop:
LLM -> tool_calls -> execute -> observations -> LLM ... -> final text.
"""

from __future__ import annotations

import json
import time
from typing import Any

from zhenxun.services.llm import AI, LLMContentPart
from zhenxun.services.llm.tools import RunContext, ToolInvoker
from zhenxun.services.llm.types.models import LLMResponse, LLMToolCall, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .agent_run_store import persist_agent_run_state
from .agent_state import AgentRunState, AgentRuntimeResult, AgentRuntimeTimelineItem
from .artifact_store import compact_tool_result_output, summarize_artifact_text
from .command_observation import build_command_observation
from .context_compression import compress_agent_messages
from .route_text import normalize_message_text
from .runtime_guardrails import RuntimeGuardrailDecision, RuntimeGuardrails
from .task_frame import TASK_TEXT_FIELD
from .task_coverage_judge import TaskCoverageJudge
from .turn_runtime import TurnBudgetController, estimate_text_tokens

_MAIN_STAGE = "main_request"
_DIRECT_ANSWER_INTERCEPT_LIMIT = 1
_FINAL_VALIDATION_INTERCEPT_LIMIT = 1
_TASK_COVERAGE_INTERCEPT_LIMIT = 2
_SAFE_NO_TOOL_RESULT_REPLY = (
    "我没有拿到真实工具执行结果，不能直接说已经完成。你可以换个说法，"
    "或稍后再让我试一次。"
)
_COMPLETION_CLAIM_MARKERS = (
    "已经完成",
    "已完成",
    "处理好了",
    "已经处理",
    "已处理",
    "已经做好",
    "已做好",
    "已经做成",
    "已做成",
    "已经生成",
    "已生成",
    "已经查到",
    "已查到",
    "已经执行",
    "已执行",
    "已经发送",
    "已发送",
    "已经发出",
    "已发出",
    "帮你完成",
    "帮你做好",
    "帮你做成",
    "帮你生成",
    "帮你查到",
    "帮你执行",
    "帮你发送",
    "帮你发出",
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
        self._force_required_tool_choice_once = False

    async def run(self) -> AgentRuntimeResult:
        self._persist_state("started")
        try:
            await self._seed_initial_coverage()
            for _ in range(self.state.max_steps):
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
                if not tool_calls:
                    final_text = normalize_message_text(str(response.text or ""))
                    direct_answer_action = self._direct_answer_intercept_action(
                        final_text
                    )
                    if direct_answer_action == "retry":
                        self._persist_state(
                            "tool_required_direct_answer_intercepted"
                        )
                        continue
                    if direct_answer_action == "safe_final":
                        self.state.complete_final(
                            _SAFE_NO_TOOL_RESULT_REPLY,
                            reason="tool_required_without_observation",
                        )
                        self._persist_state("completed", reason=self.state.stop_reason)
                        return self.state.to_result()
                    completion_action = self._unobserved_completion_action(final_text)
                    if completion_action == "retry":
                        self._persist_state("unobserved_completion_intercepted")
                        continue
                    if completion_action == "safe_final":
                        self.state.complete_final(
                            _SAFE_NO_TOOL_RESULT_REPLY,
                            reason="unobserved_completion_blocked",
                        )
                        self._persist_state("completed", reason=self.state.stop_reason)
                        return self.state.to_result()
                    coverage_action = await self._coverage_action(final_text)
                    if coverage_action == "retry":
                        self._persist_state("task_coverage_missing")
                        continue
                    if coverage_action == "partial_final":
                        self.state.complete_final(
                            self._partial_coverage_reply(),
                            reason="task_coverage_incomplete",
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

            max_steps_guardrail = self.guardrails.on_max_steps()
            self._apply_guardrail(max_steps_guardrail, as_message=True)
            self._persist_state("guardrail", reason=max_steps_guardrail.reason)
            await self._force_final_response(reason=max_steps_guardrail.reason)
            return self.state.to_result()
        except Exception as exc:
            self.state.stop_reason = f"runtime_exception:{type(exc).__name__}"
            self.state.recovery_action = (
                self.state.recovery_action or self.state.stop_reason
            )
            self._persist_state("failed", error=str(exc))
            raise

    async def _request_model(
        self,
        *,
        tools: dict[str, ToolExecutable] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> LLMResponse:
        return await self.ai.generate_internal(
            self.state.messages,
            model=self.model_name,
            config=self.generation_config,
            tools=tools,
            tool_choice=tool_choice,
            timeout=self.timeout,
        )

    async def _force_final_response(self, *, reason: str) -> None:
        self.state.transition_force_final(reason)
        self._compress_context_if_needed()
        self._persist_state("force_final_requested", reason=reason)
        response = await self._request_model(tools=None, tool_choice=None)
        final_text = normalize_message_text(str(response.text or ""))
        if self._should_block_unobserved_completion(final_text):
            final_text = _SAFE_NO_TOOL_RESULT_REPLY
            reason = f"{reason}:unobserved_completion_blocked"
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
        dynamic_tools = getattr(catalog_state, "tool_map", None)
        if dynamic_tools is None:
            return
        if callable(dynamic_tools):
            dynamic_tools = dynamic_tools()
        if not isinstance(dynamic_tools, dict):
            return
        self.state.tool_map = self.guardrails.filter_tool_map(
            {**self.base_tool_map, **dynamic_tools}
        )

    def _tool_choice_for_request(self) -> str | dict[str, Any] | None:
        if not self.state.tool_map:
            return None
        if self.state.tool_obligation == "none":
            return None
        if self._force_required_tool_choice_once:
            self._force_required_tool_choice_once = False
            return "required"
        if (
            self.state.tool_obligation == "required"
            and self._has_available_required_tools()
            and not self._has_command_observation()
        ):
            return "required"
        return "auto"

    def _tools_for_request(
        self,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, ToolExecutable] | None:
        if not self.state.tool_map:
            return None
        if self.state.tool_obligation == "none" and tool_choice is None:
            return None
        return self.state.tool_map

    async def _seed_initial_coverage(self) -> None:
        if not self._should_use_coverage_judge():
            return
        result = await self.coverage_judge.judge(
            original_message=self.message_text,
            observations=[],
            final_reply="",
            available_tools=self._available_tool_summaries(limit=32),
            pending_tasks=[],
            mode="initial_scan",
        )
        tasks = _normalized_tasks(result.missing_tasks)
        if not tasks and not result.unsupported_tasks:
            return
        self.state.replace_pending_tasks(tasks, source="coverage_initial")
        self.state.append_timeline(
            role="system",
            kind="task_coverage",
            metadata={
                "step": self.state.step,
                "mode": "initial_scan",
                "covered": result.covered,
                "missing_tasks": tasks,
                "unsupported_tasks": _normalized_tasks(result.unsupported_tasks),
                "reason": normalize_message_text(result.reason),
            },
        )
        self._persist_state("task_coverage_initial", missing=len(tasks))

    async def _coverage_action(self, final_text: str) -> str:
        if not self._should_use_coverage_judge():
            return ""
        result = await self.coverage_judge.judge(
            original_message=self.message_text,
            observations=self._observation_payloads(),
            final_reply=final_text,
            available_tools=self._available_tool_summaries(limit=40),
            pending_tasks=[task.text for task in self.state.pending_tasks],
            mode="final_check",
        )
        missing_tasks = _normalized_tasks(result.missing_tasks)
        unsupported_tasks = _normalized_tasks(result.unsupported_tasks)
        self.state.append_timeline(
            role="system",
            kind="task_coverage",
            metadata={
                "step": self.state.step,
                "mode": "final_check",
                "covered": result.covered,
                "missing_tasks": missing_tasks,
                "unsupported_tasks": unsupported_tasks,
                "reason": normalize_message_text(result.reason),
            },
        )
        if not missing_tasks:
            if unsupported_tasks:
                self.state.add_pending_tasks(
                    unsupported_tasks,
                    source="coverage_unsupported",
                )
            return ""
        if self.state.coverage_interceptions >= _TASK_COVERAGE_INTERCEPT_LIMIT:
            self.state.add_pending_tasks(missing_tasks, source="coverage_final")
            if unsupported_tasks:
                self.state.add_pending_tasks(
                    unsupported_tasks,
                    source="coverage_unsupported",
                )
            return "partial_final"
        self.state.coverage_interceptions += 1
        self.state.replace_pending_tasks(missing_tasks, source="coverage_final")
        if unsupported_tasks:
            self.state.add_pending_tasks(
                unsupported_tasks,
                source="coverage_unsupported",
            )
        self._force_required_tool_choice_once = self._has_actionable_command_tools()
        self.state.append_guardrail_observation(
            {
                "ok": False,
                "status": "runtime_task_coverage",
                "guardrail_reason": "task_coverage_missing",
                "reason": "task_coverage_missing",
                "covered": False,
                "missing_tasks": missing_tasks,
                "unsupported_tasks": unsupported_tasks,
                "available_tools": self._available_tool_summaries(limit=16),
                "instruction": (
                    "Continue from missing_tasks. Call matching real tools when "
                    "available; if a task is unsupported, explain that honestly."
                ),
                "need_continue": True,
                "retryable": True,
            },
            as_message=True,
        )
        return "retry"

    def _direct_answer_intercept_action(self, final_text: str) -> str:
        if (
            self.state.tool_obligation != "required"
            or not self._has_available_required_tools()
            or self._has_command_observation()
        ):
            return ""
        if self.state.direct_answer_interceptions >= _DIRECT_ANSWER_INTERCEPT_LIMIT:
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

    def _unobserved_completion_action(self, final_text: str) -> str:
        if not self._should_block_unobserved_completion(final_text):
            return ""
        if (
            self.state.final_validation_interceptions
            >= _FINAL_VALIDATION_INTERCEPT_LIMIT
        ):
            return "safe_final"
        self.state.final_validation_interceptions += 1
        self._force_required_tool_choice_once = self._has_available_required_tools()
        self.state.append_guardrail_observation(
            {
                "ok": False,
                "status": "runtime_final_validation",
                "guardrail_reason": "unobserved_completion_claim",
                "reason": "unobserved_completion_claim",
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

    def _should_block_unobserved_completion(self, final_text: str) -> bool:
        if self.state.tool_obligation == "none":
            return False
        if self._has_successful_command_observation():
            return False
        if not self._looks_like_completion_claim(final_text):
            return False
        return (
            self.state.tool_obligation == "required"
            or bool(self.state.observations or self.state.pending_tasks)
        )

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

    def _should_use_coverage_judge(self) -> bool:
        return (
            self.state.tool_obligation == "required"
            and self._has_actionable_command_tools()
        ) or bool(self.state.observations or self.state.pending_tasks)

    def _observation_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for observation in self.state.observations[-12:]:
            output = observation.output or {}
            messages_sent = output.get("messages_sent")
            artifacts = output.get("artifacts")
            payloads.append(
                {
                    "ok": observation.ok,
                    "command_id": observation.command_id,
                    "rendered_command": observation.rendered_command,
                    "matched_plugin": observation.matched_plugin,
                    "task_text": observation.task_text,
                    "error": observation.error,
                    "messages_sent": _compact_list(messages_sent, limit=4),
                    "artifacts": _compact_artifacts(artifacts),
                }
            )
        return payloads

    def _partial_coverage_reply(self) -> str:
        missing = [task.text for task in self.state.pending_tasks[:5]]
        if not missing:
            return "我只完成了部分任务，但没能确认剩余任务。"
        return "我只完成了部分任务，剩下这些还没确认完成：" + "；".join(missing)

    def _available_tool_summaries(self, *, limit: int = 16) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for name, tool in self.state.tool_map.items():
            binding = getattr(tool, "binding", None)
            if binding is None:
                continue
            candidate = getattr(binding, "candidate", None)
            schema = getattr(candidate, "schema", None)
            command_id = normalize_message_text(
                str(getattr(binding, "command_id", "") or "")
            )
            if not command_id:
                continue
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
                }
            )
            if len(items) >= max(1, limit):
                break
        return items

    def _has_actionable_command_tools(self) -> bool:
        return any(
            self._is_actionable_command_tool(tool)
            for tool in self.state.tool_map.values()
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

    @staticmethod
    def _looks_like_completion_claim(final_text: str) -> bool:
        normalized = normalize_message_text(final_text)
        return any(marker in normalized for marker in _COMPLETION_CLAIM_MARKERS)

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

    def _persist_state(self, stage: str, **metadata: Any) -> None:
        persist_agent_run_state(self.state, stage=stage, metadata=metadata)

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
        output = compact_tool_result_output(
            tool_result.output,
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
            {
                "ok": False,
                "command_id": "",
                "rendered_command": "",
                "matched_plugin": "",
                "task_text": "",
                "error": normalize_message_text(str(tool_result.output or "")),
                "messages_sent": [],
                "artifacts": [],
                "need_continue": False,
                "remaining_task_hint": "",
                "retryable": False,
            },
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
