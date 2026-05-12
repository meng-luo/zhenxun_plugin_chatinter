"""Native tool loop for ChatInter plugin command execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import time
from typing import Any, cast

from zhenxun.services import logger
from zhenxun.services.llm import AI, LLMMessage
from zhenxun.services.llm.types.protocols import ToolExecutable

from .command_index import CommandCandidate
from .config import build_reasoning_generation_config, get_config_value, get_model_name
from .models.pydantic_models import PluginKnowledgeBase
from .native_command_tools import build_native_command_tools
from .native_validator import (
    NativeValidatedRoute,
    resolve_local_native_fallback,
    validate_native_tool_call_route,
)
from .prompt_guard import guard_prompt_sections
from .route_engine import (
    LLMRouterDecision,
    RouteAttemptReport,
    RouteResolveResult,
    _build_command_candidate_pool,
)
from .route_text import is_usage_question, normalize_message_text
from .turn_runtime import TurnBudgetController

_NATIVE_STAGE = "native_tool_loop"
_FINALIZE_TIMEOUT = 12.0
_NATIVE_TOOL_LOOP_RULES = """
<chatinter_native_tool_loop>
你正在同一次主请求中决定：直接聊天，还是调用候选插件工具。
候选工具会真实触发插件执行；调用后你会收到 tool_result。
只有用户明确要执行候选插件能力、查询候选插件用法，或自然语言需求明显对应候选插件时才调用工具。
普通闲聊、泛泛讨论命令、表达感受、开玩笑、无法确认候选工具时，不要调用工具，直接自然回复。
工具执行后，用一句简短自然的话总结结果；如果插件已经发送了完整内容，不要复述整段内容。
工具返回失败且可重试时，可以根据错误再调用一次更合适的工具；否则直接说明需要用户补充的信息。
</chatinter_native_tool_loop>
""".strip()


@dataclass(frozen=True)
class NativeToolExecutionResult:
    success: bool
    route_result: RouteResolveResult | None
    route_command: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    display_text: str = ""
    reason: str = ""


@dataclass(frozen=True)
class NativeToolLoopResult:
    decision: LLMRouterDecision
    route_result: RouteResolveResult | None
    report: RouteAttemptReport
    direct_reply: str = ""
    executions: tuple[NativeToolExecutionResult, ...] = ()

    @property
    def handled_by_tools(self) -> bool:
        return any(item.route_result is not None for item in self.executions)


ExecuteNativeRoute = Callable[
    [NativeValidatedRoute, RouteAttemptReport],
    Awaitable[NativeToolExecutionResult],
]


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
    execute_route: ExecuteNativeRoute,
) -> NativeToolLoopResult | None:
    if not bool(get_config_value("NATIVE_TOOLS_ENABLED", True)):
        return None

    normalized_message = normalize_message_text(message_text)
    report = RouteAttemptReport(helper_mode=is_usage_question(normalized_message))
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
            execute_route=execute_route,
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
    execute_route: ExecuteNativeRoute,
    report: RouteAttemptReport,
) -> NativeToolLoopResult | None:
    candidates = _build_command_candidate_pool(
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
        reason="native_full_exposure",
        limit=len(native_candidates),
    )
    report.candidate_total = max(report.candidate_total, len(native_candidates))
    report.note_tool_pool(len(native_candidates))
    report.note_prompt_exposure(native_candidates)

    tools, bindings = build_native_command_tools(native_candidates)
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
        LLMMessage.user(user_text),
    ]
    ai = AI(session_id=f"chatinter-native:{session_key or 'global'}")
    tool_map: dict[str, ToolExecutable] = {
        tool.binding.tool_name: cast(ToolExecutable, tool) for tool in tools
    }
    executions: list[NativeToolExecutionResult] = []
    max_steps = max(int(get_config_value("NATIVE_TOOL_LOOP_STEPS", 0) or 0), 0)
    if max_steps <= 0:
        configured_steps = max(
            int(get_config_value("AGENT_MAX_TOOL_STEPS", 3) or 3),
            1,
        )
        max_steps = min(configured_steps, 4)
    max_steps = max(1, min(max_steps, 4))

    response = None
    for step in range(max_steps):
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
            direct_reply = normalize_message_text(str(response.text or ""))
            if direct_reply:
                decision = LLMRouterDecision(
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
            local = _local_exact_fallback(
                message_text=message_text,
                candidates=native_candidates,
                has_reply=has_reply,
                report=report,
                execute_route=execute_route,
            )
            if local is not None:
                return await local
            return None

        report.tool_choice_count += len(tool_calls)
        messages.append(
            LLMMessage.assistant_tool_calls(tool_calls, response.text or "")
        )
        tool_messages: list[LLMMessage] = []
        for tool_call in tool_calls:
            tool_name = _tool_call_name(tool_call)
            validated = validate_native_tool_call_route(
                tool_call=tool_call,
                bindings=bindings,
                candidates=native_candidates,
                message_text=message_text,
                has_reply=has_reply,
            )
            if validated is None:
                tool_messages.append(
                    _tool_response(
                        tool_call=tool_call,
                        tool_name=tool_name,
                        payload={
                            "ok": False,
                            "status": "failed",
                            "error_type": "InvalidToolCall",
                            "message": (
                                "工具调用未通过本地校验，"
                                "请重新选择候选工具或直接聊天。"
                            ),
                            "is_retryable": True,
                        },
                    )
                )
                continue

            execution = await execute_route(validated, report)
            executions.append(execution)
            if execution.route_result is not None and report.final_reason == "init":
                report.finalize(
                    reason=validated.decision.reason or validated.reason,
                    stage=execution.route_result.stage,
                    plugin_name=execution.route_result.decision.plugin_name,
                    plugin_module=execution.route_result.decision.plugin_module,
                    command=execution.route_result.decision.command,
                )
            tool_messages.append(
                _tool_response(
                    tool_call=tool_call,
                    tool_name=tool_name,
                    payload=execution.output,
                )
            )
        messages.extend(tool_messages)

    if response is not None:
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
            decision = LLMRouterDecision(
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


def _local_exact_fallback(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    has_reply: bool,
    report: RouteAttemptReport,
    execute_route: ExecuteNativeRoute,
) -> Awaitable[NativeToolLoopResult] | None:
    validated = resolve_local_native_fallback(
        message_text=message_text,
        candidates=candidates,
        has_reply=has_reply,
        reason="native_tool_loop_no_tool_call",
    )
    if validated is None or validated.route_result is None:
        return None
    validated_route_result = validated.route_result

    async def _execute() -> NativeToolLoopResult:
        execution = await execute_route(validated, report)
        decision = LLMRouterDecision(
            action="chat",
            confidence=0.86,
            reason="native_tool_loop:local_exact_executed",
        )
        report.finalize(
            reason=decision.reason or "native_tool_loop:local_exact_executed",
            stage=validated_route_result.stage,
            plugin_name=validated_route_result.decision.plugin_name,
            plugin_module=validated_route_result.decision.plugin_module,
            command=validated_route_result.decision.command,
        )
        return NativeToolLoopResult(
            decision=decision,
            route_result=validated_route_result,
            report=report,
            direct_reply=_fallback_final_reply([execution]),
            executions=(execution,),
        )

    return _execute()


def _first_route(
    executions: list[NativeToolExecutionResult],
) -> RouteResolveResult | None:
    for execution in executions:
        if execution.route_result is not None:
            return execution.route_result
    return None


def _tool_call_name(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    return normalize_message_text(getattr(function, "name", "") or "") or "unknown_tool"


def _tool_call_id(tool_call: Any) -> str:
    return str(getattr(tool_call, "id", "") or f"call_{id(tool_call)}")


def _tool_response(
    *,
    tool_call: Any,
    tool_name: str,
    payload: dict[str, Any],
) -> LLMMessage:
    return LLMMessage.tool_response(
        tool_call_id=_tool_call_id(tool_call),
        function_name=tool_name or "unknown_tool",
        result=payload,
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
