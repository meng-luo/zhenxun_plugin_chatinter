"""Agent loop runtime for ChatInter native command tools.

The runtime owns the model/tool loop:
LLM -> tool_calls -> execute -> tool results -> LLM ... -> final text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any

from zhenxun.services.llm import AI, LLMContentPart, LLMMessage
from zhenxun.services.llm.tools import RunContext, ToolInvoker
from zhenxun.services.llm.types.models import LLMResponse, LLMToolCall, ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .command_observation import build_command_observation
from .route_text import normalize_message_text
from .task_frame import TASK_TEXT_FIELD, isolate_task_text
from .turn_runtime import TurnBudgetController, estimate_text_tokens

_DEFAULT_MAX_AGENT_STEPS = 5
_MAIN_STAGE = "main_request"


@dataclass(frozen=True)
class AgentRuntimeTimelineItem:
    role: str
    kind: str
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRuntimeResult:
    final_text: str
    tool_results: tuple[ToolResult, ...] = ()
    timeline: tuple[AgentRuntimeTimelineItem, ...] = ()
    messages: tuple[LLMMessage, ...] = ()
    stop_reason: str = "final_response"
    steps: int = 0


class AgentRuntime:
    """Small ReAct-style loop around the existing ChatInter command tools."""

    def __init__(
        self,
        *,
        session_key: str | None,
        messages: list[LLMMessage],
        tool_map: dict[str, ToolExecutable],
        run_context: RunContext,
        message_text: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
        budget_controller: TurnBudgetController | None = None,
        max_steps: int = _DEFAULT_MAX_AGENT_STEPS,
    ) -> None:
        self.session_key = session_key
        self.messages = list(messages)
        self.tool_map = dict(tool_map)
        self.run_context = run_context
        self.message_text = normalize_message_text(message_text)
        self.model_name = model_name
        self.generation_config = generation_config
        self.timeout = timeout
        self.budget_controller = budget_controller
        self.max_steps = max(1, int(max_steps or _DEFAULT_MAX_AGENT_STEPS))
        self.base_tool_map = dict(tool_map)
        self.ai = AI(session_id=f"chatinter-main:{session_key or 'global'}")
        self.invoker = ToolInvoker()

    async def run(self) -> AgentRuntimeResult:
        timeline: list[AgentRuntimeTimelineItem] = [
            AgentRuntimeTimelineItem(
                role="user",
                kind="current_user",
                content=self.message_text,
            )
        ]
        tool_results: list[ToolResult] = []

        for step in range(1, self.max_steps + 1):
            self._sync_dynamic_tools()
            self._record_prompt_use()
            response = await self._request_model(
                tools=self.tool_map or None,
                tool_choice="auto" if self.tool_map else None,
            )
            tool_calls = [
                self._isolate_tool_call_task_text(tool_call)
                for tool_call in list(response.tool_calls or [])
            ]
            if not tool_calls:
                final_text = normalize_message_text(str(response.text or ""))
                self.messages.append(LLMMessage.assistant_text_response(final_text))
                timeline.append(
                    AgentRuntimeTimelineItem(
                        role="assistant",
                        kind="assistant_text",
                        content=final_text,
                        metadata={"step": step},
                    )
                )
                return AgentRuntimeResult(
                    final_text=final_text,
                    tool_results=tuple(tool_results),
                    timeline=tuple(timeline),
                    messages=tuple(self.messages),
                    stop_reason="final_response",
                    steps=step,
                )

            if not self._allow_tool_batch(len(tool_calls)):
                final_text = await self._force_final_response(
                    reason="tool_budget_exhausted",
                    timeline=timeline,
                    step=step,
                )
                return AgentRuntimeResult(
                    final_text=final_text,
                    tool_results=tuple(tool_results),
                    timeline=tuple(timeline),
                    messages=tuple(self.messages),
                    stop_reason="tool_budget_exhausted",
                    steps=step,
                )

            timeline.extend(
                self._tool_call_timeline_item(tool_call, step=step)
                for tool_call in tool_calls
            )
            self.messages.append(
                LLMMessage.assistant_tool_calls(tool_calls, response.text or "")
            )

            started = time.perf_counter()
            for tool_call in tool_calls:
                resolved_call, tool_result = await self.invoker.execute_tool_call(
                    tool_call,
                    self.tool_map,
                    self.run_context,
                )
                tool_result = self._normalize_command_tool_result(
                    resolved_call,
                    tool_result,
                )
                self._sync_dynamic_tools()
                tool_results.append(tool_result)
                timeline.append(
                    self._tool_result_timeline_item(
                        resolved_call,
                        tool_result,
                        step=step,
                    )
                )
                self.messages.append(
                    LLMMessage.tool_response(
                        tool_call_id=resolved_call.id,
                        function_name=resolved_call.function.name,
                        result=self._tool_result_for_model(tool_result),
                    )
                )
            self._record_tool_batch(time.perf_counter() - started)

        final_text = await self._force_final_response(
            reason="max_agent_steps_reached",
            timeline=timeline,
            step=self.max_steps,
        )
        return AgentRuntimeResult(
            final_text=final_text,
            tool_results=tuple(tool_results),
            timeline=tuple(timeline),
            messages=tuple(self.messages),
            stop_reason="max_agent_steps_reached",
            steps=self.max_steps,
        )

    async def _request_model(
        self,
        *,
        tools: dict[str, ToolExecutable] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> LLMResponse:
        return await self.ai.generate_internal(
            self.messages,
            model=self.model_name,
            config=self.generation_config,
            tools=tools,
            tool_choice=tool_choice,
            timeout=self.timeout,
        )

    async def _force_final_response(
        self,
        *,
        reason: str,
        timeline: list[AgentRuntimeTimelineItem],
        step: int,
    ) -> str:
        self.messages.append(
            LLMMessage.user(
                "工具调用已经结束或达到上限。请不要再调用工具，"
                "根据已经完成的工具结果直接给用户一个简短最终回复。"
            )
        )
        response = await self._request_model(tools=None, tool_choice=None)
        final_text = normalize_message_text(str(response.text or ""))
        self.messages.append(LLMMessage.assistant_text_response(final_text))
        timeline.append(
            AgentRuntimeTimelineItem(
                role="assistant",
                kind="assistant_text",
                content=final_text,
                metadata={"step": step, "forced_final": reason},
            )
        )
        return final_text

    def _allow_tool_batch(self, call_count: int) -> bool:
        if self.budget_controller is None:
            return True
        return self.budget_controller.allow_tool_batch(
            call_count=call_count,
            batch_kind=_MAIN_STAGE,
        )

    def _record_tool_batch(self, duration: float) -> None:
        if self.budget_controller is None:
            return
        self.budget_controller.record_tool_batch(
            batch_kind=_MAIN_STAGE,
            duration=duration,
        )

    def _record_prompt_use(self) -> None:
        if self.budget_controller is None:
            return
        self.budget_controller.record_prompt_use(
            estimated_tokens=_estimate_prompt_tokens(self.messages)
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
        self.tool_map = {**self.base_tool_map, **dynamic_tools}

    def _isolate_tool_call_task_text(self, tool_call: LLMToolCall) -> LLMToolCall:
        tool_name = str(getattr(tool_call.function, "name", "") or "")
        executable = self.tool_map.get(tool_name)
        binding = getattr(executable, "binding", None)
        candidate = getattr(binding, "candidate", None)
        schema = getattr(candidate, "schema", None)
        if schema is None:
            return tool_call

        arguments = _parse_tool_arguments(str(tool_call.function.arguments or ""))
        if not isinstance(arguments, dict):
            return tool_call

        raw_task = arguments.get(TASK_TEXT_FIELD)
        if isinstance(raw_task, str) and raw_task.strip():
            task_text = isolate_task_text(raw_task)
        elif not schema.slots:
            task_text = isolate_task_text(
                _select_task_fragment_for_command(self.message_text, schema.head)
                or self.message_text,
                command_text=schema.head,
            )
        else:
            task_text = ""

        arguments[TASK_TEXT_FIELD] = task_text or None
        tool_call.function.arguments = json.dumps(arguments, ensure_ascii=False)
        return tool_call

    @staticmethod
    def _tool_call_timeline_item(
        tool_call: LLMToolCall,
        *,
        step: int,
    ) -> AgentRuntimeTimelineItem:
        return AgentRuntimeTimelineItem(
            role="assistant",
            kind="tool_call",
            tool_name=str(tool_call.function.name or ""),
            metadata={
                "step": step,
                "arguments": _parse_tool_arguments(tool_call.function.arguments),
            },
        )

    @staticmethod
    def _tool_result_timeline_item(
        tool_call: LLMToolCall,
        tool_result: ToolResult,
        *,
        step: int,
    ) -> AgentRuntimeTimelineItem:
        output = tool_result.output
        content = ""
        if isinstance(output, dict):
            messages_sent = output.get("messages_sent")
            if isinstance(messages_sent, list):
                content = "\n".join(
                    normalize_message_text(str(item or ""))
                    for item in messages_sent[:8]
                    if normalize_message_text(str(item or ""))
                )
            if not content:
                content = normalize_message_text(
                    str(
                        output.get("remaining_task_hint", "")
                        or output.get("error", "")
                        or ""
                    )
                )
        if not content:
            content = normalize_message_text(str(tool_result.display_content or ""))
        return AgentRuntimeTimelineItem(
            role="tool",
            kind="tool_result",
            content=content,
            tool_name=str(tool_call.function.name or ""),
            metadata={"step": step, "output": output},
        )

    @staticmethod
    def _tool_result_for_model(tool_result: ToolResult) -> dict[str, Any]:
        if isinstance(tool_result.output, dict):
            return dict(tool_result.output)
        return {
            "ok": False,
            "error": normalize_message_text(str(tool_result.output or "")),
            "messages_sent": [],
            "need_continue": False,
            "remaining_task_hint": "",
        }

    def _normalize_command_tool_result(
        self,
        tool_call: LLMToolCall,
        tool_result: ToolResult,
    ) -> ToolResult:
        output = tool_result.output
        if isinstance(output, dict) and (
            "messages_sent" in output
            or "remaining_task_hint" in output
            or output.get("status") == "retrieved"
        ):
            return tool_result

        executable = self.tool_map.get(str(tool_call.function.name or ""))
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
                error=error or "命令工具执行失败。",
                retryable=False,
                plugin_module=getattr(candidate, "plugin_module", ""),
            ),
            display_content=tool_result.display_content,
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


def _select_task_fragment_for_command(message_text: str, command_head: str) -> str:
    command = normalize_message_text(command_head)
    if not command:
        return ""
    splitter = re.compile(
        r"(?:，|,|。|；|;|\s)+(?:然后|接着|再|最后|顺便|并且|以及|还有|同时)\s*"
    )
    for part in splitter.split(normalize_message_text(message_text)):
        fragment = isolate_task_text(part)
        if command and command in fragment:
            return fragment
    return ""


def _estimate_prompt_tokens(messages: list[LLMMessage]) -> int:
    total = 0
    for message in messages:
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
