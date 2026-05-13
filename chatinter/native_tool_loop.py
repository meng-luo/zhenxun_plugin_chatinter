"""Native tool loop for ChatInter plugin command execution."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, cast

from zhenxun.services import logger
from zhenxun.services.llm import AI, LLMMessage
from zhenxun.services.llm.tools import RunContext, ToolInvoker
from zhenxun.services.llm.types import LLMToolCall
from zhenxun.services.llm.types.models import ToolResult
from zhenxun.services.llm.types.protocols import ToolExecutable

from .config import build_reasoning_generation_config, get_config_value, get_model_name
from .models.pydantic_models import PluginKnowledgeBase
from .native_command_tools import build_native_command_tools
from .native_executor import (
    ExecuteNativeRoute,
    NativeCommandExecutionContext,
    NativeToolExecutionResult,
)
from .prompt_guard import guard_prompt_sections
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
    NativeRouteResult,
    build_native_command_candidate_pool,
)
from .route_text import is_usage_question, normalize_message_text
from .turn_runtime import TurnBudgetController

_NATIVE_STAGE = "native_tool_loop"
_FINALIZE_TIMEOUT = 12.0
_NATIVE_TOOL_LOOP_RULES = """
<chatinter_native_tool_loop>
你正在同一次主请求中决定：直接聊天，还是调用候选插件工具。
候选工具会真实触发插件执行；调用后你会收到 tool_result。
每个工具都已经暴露完整 command schema；只根据工具 description 和参数 schema
选择工具并填写参数，不要猜 schema 外的参数名。
只有用户明确要执行候选插件能力、查询候选插件用法，或自然语言需求明显对应候选插件时才调用工具。
普通闲聊、泛泛讨论命令、表达感受、开玩笑、无法确认候选工具时，不要调用工具，直接自然回复。
工具执行后，用一句简短自然的话总结结果；如果插件已经发送了完整内容，不要复述整段内容。
工具返回失败且可重试时，可以根据错误再调用一次更合适的工具；否则直接说明需要用户补充的信息。
</chatinter_native_tool_loop>
""".strip()


@dataclass(frozen=True)
class NativeToolLoopResult:
    decision: NativeRouteDecision
    route_result: NativeRouteResult | None
    report: NativeRouteReport
    direct_reply: str = ""
    executions: tuple[NativeToolExecutionResult, ...] = ()

    @property
    def handled_by_tools(self) -> bool:
        return any(item.route_result is not None for item in self.executions)


async def run_native_tool_loop(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None = None,
    system_prompt: str = "",
    context_xml: str = "",
    route_executor: ExecuteNativeRoute,
    history_messages: list[LLMMessage] | None = None,
) -> NativeToolLoopResult | None:
    if not bool(get_config_value("NATIVE_TOOLS_ENABLED", True)):
        return None

    normalized_message = normalize_message_text(message_text)
    report = NativeRouteReport(helper_mode=is_usage_question(normalized_message))
    if not normalized_message:
        return None

    if budget_controller is not None and not budget_controller.allow_classifier(
        _NATIVE_STAGE
    ):
        return None

    started = time.perf_counter()
    try:
        return await _run_native_tool_loop(
            normalized_message,
            knowledge_base,
            session_key=session_key,
            budget_controller=budget_controller,
            has_reply=has_reply,
            command_tools=command_tools,
            system_prompt=system_prompt,
            context_xml=context_xml,
            history_messages=history_messages,
            route_executor=route_executor,
            report=report,
        )
    except Exception as exc:
        logger.debug(f"ChatInter native tool loop skipped: {exc}")
        return None
    finally:
        if budget_controller is not None:
            budget_controller.record_classifier(
                _NATIVE_STAGE,
                time.perf_counter() - started,
            )


async def _run_native_tool_loop(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    session_key: str | None,
    budget_controller: TurnBudgetController | None,
    has_reply: bool,
    command_tools: list[Any] | None,
    system_prompt: str,
    context_xml: str,
    history_messages: list[LLMMessage] | None,
    route_executor: ExecuteNativeRoute,
    report: NativeRouteReport,
) -> NativeToolLoopResult | None:
    candidates = build_native_command_candidate_pool(
        message_text,
        knowledge_base,
        session_key=session_key,
        command_tools=command_tools,
        limit=None,
        diversify=False,
        include_unscored=True,
    )
    if not candidates:
        return None

    native_candidates = candidates
    if not native_candidates:
        return None

    report.note_candidate_policy(
        reason="native_full_schema_exposure",
        limit=len(native_candidates),
    )
    report.candidate_total = max(report.candidate_total, len(native_candidates))
    report.note_tool_pool(len(native_candidates))
    report.note_prompt_exposure(native_candidates)

    tools = build_native_command_tools(native_candidates)
    if not tools:
        return None

    guarded = guard_prompt_sections(
        session_key=session_key or "global",
        stage=_NATIVE_STAGE,
        system_prompt=_build_system_prompt(system_prompt),
        context_text=context_xml,
        user_text=message_text,
        controller=budget_controller,
    )
    user_text = guarded.user_text
    if guarded.context_text:
        user_text = (
            f"{guarded.context_text}\n\n"
            f"<current_user_message>{guarded.user_text}</current_user_message>"
        )

    messages = [
        LLMMessage.system(guarded.system_prompt),
        *list(history_messages or []),
        LLMMessage.user(user_text),
    ]
    ai = AI(session_id=f"chatinter-native:{session_key or 'global'}")
    tool_map: dict[str, ToolExecutable] = {
        tool.binding.tool_name: cast(ToolExecutable, tool) for tool in tools
    }
    command_context = NativeCommandExecutionContext(
        candidates=native_candidates,
        has_reply=has_reply,
        report=report,
        route_executor=route_executor,
        message_text=message_text,
    )
    run_context = RunContext(
        session_id=session_key,
        extra={"native_command_context": command_context},
    )
    invoker = ToolInvoker()
    max_steps = max(int(get_config_value("NATIVE_TOOL_LOOP_STEPS", 0) or 0), 0)
    if max_steps <= 0:
        configured_steps = max(
            int(get_config_value("AGENT_MAX_TOOL_STEPS", 3) or 3),
            1,
        )
        max_steps = min(configured_steps, 4)
    max_steps = max(1, min(max_steps, 4))

    response = None
    had_tool_calls = False
    for _ in range(max_steps):
        response = await ai.generate_internal(
            messages,
            model=get_model_name(),
            config=build_reasoning_generation_config(),
            tools=tool_map,
            tool_choice="auto",
            timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
        )
        tool_calls = list(response.tool_calls or [])
        if not tool_calls:
            executions = list(command_context.executions)
            direct_reply = normalize_message_text(str(response.text or ""))
            if direct_reply:
                decision = NativeRouteDecision(
                    action="chat",
                    confidence=0.84 if not executions else 0.9,
                    reason=(
                        "native_tool_loop:final"
                        if executions
                        else "native_tool_loop:direct_chat"
                    ),
                )
                report.finalize(
                    reason=decision.reason or "native_tool_loop",
                    stage=_NATIVE_STAGE,
                )
                return NativeToolLoopResult(
                    decision=decision,
                    route_result=_first_route(executions),
                    report=report,
                    direct_reply=direct_reply,
                    executions=tuple(executions),
                )
            if executions:
                break
            if had_tool_calls:
                break
            return None

        had_tool_calls = True
        report.tool_choice_count += len(tool_calls)
        messages.append(
            LLMMessage.assistant_tool_calls(tool_calls, response.text or "")
        )
        tool_messages: list[LLMMessage] = []
        for tool_call in tool_calls:
            resolved_call, tool_result = await invoker.execute_tool_call(
                tool_call,
                tool_map,
                run_context,
            )
            tool_messages.append(_tool_result_to_message(resolved_call, tool_result))
        messages.extend(tool_messages)

    if response is not None:
        executions = list(command_context.executions)
        messages.append(
            LLMMessage.system(
                "工具调用阶段已结束。请基于 tool_result 直接给用户最终回复，"
                "不要继续调用工具。"
            )
        )
        try:
            final_response = await ai.generate_internal(
                messages,
                model=get_model_name(),
                config=build_reasoning_generation_config(),
                timeout=min(
                    float(get_config_value("INTENT_TIMEOUT", 20) or 20),
                    _FINALIZE_TIMEOUT,
                ),
            )
            final_reply = normalize_message_text(str(final_response.text or ""))
        except Exception as exc:
            logger.debug(f"ChatInter native finalization failed: {exc}")
            final_reply = ""
        if not final_reply:
            final_reply = _fallback_final_reply(executions)
        if final_reply:
            decision = NativeRouteDecision(
                action="chat",
                confidence=0.9,
                reason="native_tool_loop:tool_result_final",
            )
            report.finalize(
                reason=decision.reason or "native_tool_loop",
                stage=_NATIVE_STAGE,
            )
            return NativeToolLoopResult(
                decision=decision,
                route_result=_first_route(executions),
                report=report,
                direct_reply=final_reply,
                executions=tuple(executions),
            )
    return None


def _first_route(
    executions: list[NativeToolExecutionResult],
) -> NativeRouteResult | None:
    for execution in executions:
        if execution.route_result is not None:
            return execution.route_result
    return None


def _tool_result_to_message(
    tool_call: LLMToolCall,
    tool_result: ToolResult,
) -> LLMMessage:
    return LLMMessage.tool_response(
        tool_call_id=tool_call.id,
        function_name=tool_call.function.name or "unknown_tool",
        result=tool_result.output,
    )


def _fallback_final_reply(executions: list[NativeToolExecutionResult]) -> str:
    if not executions:
        return ""
    success_count = sum(1 for item in executions if item.success)
    latest = executions[-1]
    if latest.display_text:
        return latest.display_text
    if success_count:
        return "处理好了。"
    message = str(latest.output.get("message", "") or latest.reason or "").strip()
    return message or "这个暂时没处理成功。"


def _build_system_prompt(base_prompt: str) -> str:
    normalized = str(base_prompt or "").strip()
    if not normalized:
        return _NATIVE_TOOL_LOOP_RULES
    return f"{normalized}\n\n{_NATIVE_TOOL_LOOP_RULES}"


__all__ = [
    "NativeToolExecutionResult",
    "NativeToolLoopResult",
    "run_native_tool_loop",
]
