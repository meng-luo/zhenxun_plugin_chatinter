import asyncio
from collections import defaultdict
import json
import time
from typing import Any

from nonebot.adapters import Bot, Event

from zhenxun.services.llm import AI, LLMMessage
from zhenxun.services.llm.tools import RunContext, ToolInvoker
from zhenxun.services.llm.types import LLMContentPart, LLMResponse, LLMToolCall
from zhenxun.services.llm.types.models import ToolResult
from zhenxun.services.llm.types.protocols import BaseCallbackHandler, ToolCallData
from zhenxun.services.log import logger

from .config import build_reasoning_generation_config, get_config_value
from .lifecycle import ToolLifecyclePayload, get_lifecycle_manager
from .runtime import get_runtime_scheduler
from .tool_registry import ChatInterToolRegistry, ToolSelectionContext

TOOL_EXEC_TIMEOUT = 8.0
LLM_TIMEOUT_MARGIN = 1.5
_ENABLE_TOOLS_ATTR = "_chatinter_enable_tools"
_DISABLE_TOOLS_ATTR = "_chatinter_disable_tools"
_TOOL_COMPLEXITY_HINTS = (
    "插件",
    "命令",
    "参数",
    "调用",
    "shell",
    "终端",
    "计算",
    "表达式",
    "代码",
)
_TOOL_CALL_REPEAT_LIMIT = 1
_AGENT_MIN_TIMEOUT = 5.0
_AGENT_FINALIZE_TIMEOUT_FLOOR = 1.2
_AGENT_MAX_EXPANDED_TOOLS = 16


def _response_tokens(response: LLMResponse) -> tuple[int, int, int]:
    usage = response.usage_info or {}
    if not isinstance(usage, dict):
        return 0, 0, 0

    def _int_val(key: str) -> int:
        value = usage.get(key, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = _int_val("prompt_tokens")
    completion_tokens = _int_val("completion_tokens")
    total_tokens = _int_val("total_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _resolve_tool_budget(message_text: str, tool_count: int) -> int:
    configured = max(int(get_config_value("AGENT_MAX_TOOL_STEPS", 4) or 4), 1)
    if tool_count <= 0:
        return 1
    normalized = str(message_text or "").lower()
    extra = 0
    if any(hint in normalized for hint in _TOOL_COMPLEXITY_HINTS):
        extra += 1
    if len(normalized) >= 40:
        extra += 1
    return max(1, min(configured + extra, 8))


def _resolve_agent_timeout(timeout: int) -> float:
    configured = int(get_config_value("AGENT_TOTAL_TIMEOUT", 0) or 0)
    if configured > 0:
        return float(max(configured, int(_AGENT_MIN_TIMEOUT)))
    base = max(int(timeout or 0), int(_AGENT_MIN_TIMEOUT))
    # 预留工具执行和收尾总结时间
    return float(base + 6)


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _step_timeout(
    deadline: float, fallback: float = _AGENT_FINALIZE_TIMEOUT_FLOOR
) -> float:
    remain = _remaining_seconds(deadline)
    if remain <= 0:
        return 0.0
    lower_bound = min(max(fallback, 0.2), remain)
    return max(min(remain, 30.0), lower_bound)


def _tool_message_payload(message: LLMMessage) -> Any:
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return content
    text = content.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def _is_tool_result_failed(payload: Any) -> bool:
    if isinstance(payload, dict):
        ok = payload.get("ok")
        if isinstance(ok, bool):
            return not ok
        status = str(payload.get("status", "")).strip().lower()
        if status in {"failed", "error"}:
            return True
        error_type = str(payload.get("error_type", "")).strip()
        if error_type:
            return True
        if "error" in payload and "result" not in payload:
            return True
    return False


def _tool_call_signature(tool_call: LLMToolCall) -> str:
    name = str(getattr(tool_call.function, "name", "") or "")
    args = getattr(tool_call.function, "arguments", None)
    if isinstance(args, str):
        args_text = args
    else:
        try:
            args_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except Exception:
            args_text = str(args)
    return f"{name}:{args_text}"


def _resolve_superuser(bot: Bot, user_id: str) -> bool:
    superusers = getattr(getattr(bot, "config", None), "superusers", set())
    return str(user_id) in {str(item) for item in superusers}


def _iter_runtime_overrides(event: Event, attr_name: str) -> set[str]:
    raw = getattr(event, attr_name, None)
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw.strip()} if raw.strip() else set()
    if isinstance(raw, list | tuple | set | frozenset):
        result: set[str] = set()
        for item in raw:
            value = str(item).strip()
            if value:
                result.add(value)
        return result
    return set()


async def _apply_runtime_tool_overrides(
    *,
    event: Event,
    session_key: str,
    group_id: str | None,
) -> None:
    await ChatInterToolRegistry.reset_dynamic_overrides(session_id=session_key)
    enable_names = _iter_runtime_overrides(event, _ENABLE_TOOLS_ATTR)
    disable_names = _iter_runtime_overrides(event, _DISABLE_TOOLS_ATTR)
    for name in enable_names:
        await ChatInterToolRegistry.set_tool_enabled(
            tool_name=name,
            enabled=True,
            session_id=session_key,
            group_id=group_id,
        )
    for name in disable_names:
        await ChatInterToolRegistry.set_tool_enabled(
            tool_name=name,
            enabled=False,
            session_id=session_key,
            group_id=group_id,
        )


class _ToolLifecycleCallback(BaseCallbackHandler):
    def __init__(
        self,
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
    ) -> None:
        self._session_id = session_id
        self._user_id = user_id
        self._group_id = group_id
        self._lifecycle = get_lifecycle_manager()

    async def on_tool_start(
        self,
        tool_call: LLMToolCall,
        data: ToolCallData,
        **kwargs: Any,
    ) -> ToolCallData | ToolResult | None:
        payload = ToolLifecyclePayload(
            session_id=self._session_id,
            user_id=self._user_id,
            group_id=self._group_id,
            tool_name=data.tool_name,
            tool_args=dict(data.tool_args),
            metadata={
                "phase": "start",
                "tool_call_id": tool_call.id,
            },
        )
        await self._lifecycle.dispatch("before_tool", payload)
        logger.debug(
            f"chatinter tool start: {payload.tool_name} session={self._session_id}"
        )
        data.tool_name = payload.tool_name
        data.tool_args = payload.tool_args
        return data

    async def on_tool_end(
        self,
        result: ToolResult | None,
        error: Exception | None,
        tool_call: LLMToolCall,
        duration: float,
        **kwargs: Any,
    ) -> None:
        payload = ToolLifecyclePayload(
            session_id=self._session_id,
            user_id=self._user_id,
            group_id=self._group_id,
            tool_name=tool_call.function.name,
            tool_args={},
            result=result.output if result else None,
            error=str(error) if error else None,
            duration=duration,
            metadata={
                "phase": "end",
                "tool_call_id": tool_call.id,
            },
        )
        await self._lifecycle.dispatch("after_tool", payload)
        logger.debug(
            "chatinter tool end: "
            f"{payload.tool_name} session={self._session_id} "
            f"duration={duration * 1000:.2f}ms error={payload.error or ''}"
        )


def _normalize_tool_calls(raw_calls: list[Any] | None) -> list[LLMToolCall]:
    if not raw_calls:
        return []
    normalized: list[LLMToolCall] = []
    for item in raw_calls:
        if isinstance(item, LLMToolCall):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            try:
                normalized.append(LLMToolCall(**item))
            except Exception:
                continue
    return normalized


def _assistant_message_from_response(response: LLMResponse) -> LLMMessage:
    if response.content_parts:
        return LLMMessage(
            role="assistant",
            content=response.content_parts,
            tool_calls=response.tool_calls,
        )
    return LLMMessage.assistant_tool_calls(
        response.tool_calls or [],
        response.text,
    )


def _record_usage(
    response: LLMResponse,
    *,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    total_tokens: int,
) -> tuple[int, int, int]:
    p_tokens, c_tokens, t_tokens = _response_tokens(response)
    return (
        total_prompt_tokens + p_tokens,
        total_completion_tokens + c_tokens,
        total_tokens + t_tokens,
    )


async def run_chatinter_agent(
    *,
    bot: Bot,
    event: Event,
    user_id: str,
    group_id: str | None,
    model: str | None,
    timeout: int,
    system_prompt: str,
    context_xml: str,
    message_text: str,
    image_parts: list[LLMContentPart] | None = None,
) -> LLMResponse:
    scheduler = get_runtime_scheduler()
    session_key = str(group_id or user_id)
    model_key = str(model or "default")
    selection_context = ToolSelectionContext(
        query=message_text,
        context_text=context_xml,
        session_id=session_key,
        user_id=str(user_id),
        group_id=str(group_id) if group_id else None,
        is_superuser=_resolve_superuser(bot, str(user_id)),
    )

    async def _run() -> LLMResponse:
        await _apply_runtime_tool_overrides(
            event=event,
            session_key=session_key,
            group_id=str(group_id) if group_id else None,
        )
        instruction = (
            f"{system_prompt}\n\n"
            "你是真寻 Bot 的 AI Agent。优先回答用户问题。"
            "当需要精确插件信息、计算或安全命令执行时，请调用工具。"
            "禁止输出危险 shell 建议；无法安全执行时直接拒绝。"
        )
        prompt = (
            "<context>\n"
            f"{context_xml}\n"
            "</context>\n"
            "<current_message>\n"
            f"{message_text}\n"
            "</current_message>"
        )

        user_content: str | list[LLMContentPart]
        if image_parts:
            user_content = [LLMContentPart.text_part(prompt), *image_parts]
        else:
            user_content = prompt

        ai = AI(session_id=f"chatinter-agent:{session_key}")
        reasoning_config = build_reasoning_generation_config()
        messages: list[LLMMessage] = [
            LLMMessage.system(instruction),
            LLMMessage.user(user_content),
        ]

        strict_tool_select = bool(get_config_value("AGENT_STRICT_TOOL_SELECT", True))
        expand_tools_step = max(
            int(get_config_value("AGENT_EXPAND_TOOLS_STEP", 2) or 2), 1
        )
        tool_failure_limit = max(
            int(get_config_value("AGENT_TOOL_FAILURE_LIMIT", 2) or 2), 1
        )
        failed_round_limit = max(
            int(get_config_value("AGENT_FAILED_ROUND_LIMIT", 2) or 2), 1
        )

        lower_message = str(message_text or "").lower()
        has_tool_intent_hint = any(
            hint in lower_message for hint in _TOOL_COMPLEXITY_HINTS
        )

        primary_tools = await ChatInterToolRegistry.get_tools_for_query(
            query=message_text,
            context_text=context_xml,
            max_tools=8,
            allow_fallback=not strict_tool_select,
            selection_context=selection_context,
        )
        expanded_tools = await ChatInterToolRegistry.get_tools(
            selection_context=selection_context
        )

        if primary_tools:
            active_tools: dict[str, Any] = dict(primary_tools)
        elif strict_tool_select and has_tool_intent_hint:
            # 严格筛选未命中但用户有明显工具意图，允许一个小窗口兜底工具池。
            fallback_tools = await ChatInterToolRegistry.get_tools_for_query(
                query=message_text,
                context_text=context_xml,
                max_tools=3,
                allow_fallback=True,
                selection_context=selection_context,
            )
            active_tools = dict(fallback_tools)
        elif strict_tool_select:
            active_tools = {}
        else:
            active_tools = dict(expanded_tools) if expanded_tools else {}

        if not active_tools:
            return await ai.generate_internal(
                messages,
                model=model,
                config=reasoning_config,
                timeout=max(timeout + LLM_TIMEOUT_MARGIN, _AGENT_MIN_TIMEOUT),
            )

        invoker = ToolInvoker(
            callbacks=[
                _ToolLifecycleCallback(
                    session_id=session_key,
                    user_id=str(user_id),
                    group_id=str(group_id) if group_id else None,
                )
            ]
        )
        context = RunContext(
            session_id=session_key,
            scope={"bot": bot, "event": event},
            extra={"user_id": user_id, "group_id": group_id},
        )
        max_tool_steps = _resolve_tool_budget(message_text, len(active_tools))
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        executed_tool_calls = 0
        seen_tool_signatures: set[tuple[str, ...]] = set()
        repeated_tool_calls = 0
        tool_failures: dict[str, int] = defaultdict(int)
        consecutive_failed_rounds = 0
        expanded_tools_activated = False
        forced_finalize_reason = ""

        deadline = time.monotonic() + _resolve_agent_timeout(timeout)
        llm_timeout = _step_timeout(deadline)
        if llm_timeout <= 0:
            llm_timeout = max(timeout + LLM_TIMEOUT_MARGIN, _AGENT_MIN_TIMEOUT)

        response = await ai.generate_internal(
            messages,
            model=model,
            config=reasoning_config,
            tools=active_tools,
            timeout=llm_timeout,
        )
        total_prompt_tokens, total_completion_tokens, total_tokens = _record_usage(
            response,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
        )

        for step in range(max_tool_steps):
            if _remaining_seconds(deadline) <= 0.2:
                forced_finalize_reason = "deadline_exceeded"
                break

            tool_calls = _normalize_tool_calls(response.tool_calls)
            if not tool_calls:
                logger.debug(
                    "chatinter agent complete: "
                    f"session={session_key} tools={len(active_tools)} "
                    f"calls={executed_tool_calls} steps={step + 1} "
                    "tokens="
                    f"{total_prompt_tokens}/{total_completion_tokens}/{total_tokens} "
                    f"failed_rounds={consecutive_failed_rounds}"
                )
                return response

            if (
                not expanded_tools_activated
                and step + 1 >= expand_tools_step
                and expanded_tools
                and len(expanded_tools) > len(active_tools)
            ):
                merged = dict(expanded_tools)
                for name, executable in active_tools.items():
                    merged[name] = executable
                if len(merged) > _AGENT_MAX_EXPANDED_TOOLS:
                    merged = dict(list(merged.items())[:_AGENT_MAX_EXPANDED_TOOLS])
                active_tools = merged
                expanded_tools_activated = True
                logger.debug(
                    "chatinter agent tool pool expanded: "
                    f"session={session_key} step={step + 1} size={len(active_tools)}"
                )

            executed_tool_calls += len(tool_calls)
            signature = tuple(sorted(_tool_call_signature(item) for item in tool_calls))
            if signature in seen_tool_signatures:
                repeated_tool_calls += 1
                if repeated_tool_calls > _TOOL_CALL_REPEAT_LIMIT:
                    forced_finalize_reason = "repeated_tool_calls"
                    break
            else:
                seen_tool_signatures.add(signature)

            exec_timeout = min(
                TOOL_EXEC_TIMEOUT, max(_step_timeout(deadline) - 0.2, 0.2)
            )
            if exec_timeout <= 0.2:
                forced_finalize_reason = "tool_budget_exhausted"
                break

            try:
                tool_messages = await asyncio.wait_for(
                    invoker.execute_batch(
                        tool_calls=tool_calls,
                        available_tools=active_tools,
                        context=context,
                    ),
                    timeout=exec_timeout,
                )
            except asyncio.TimeoutError:
                consecutive_failed_rounds += 1
                messages.append(_assistant_message_from_response(response))
                messages.append(
                    LLMMessage.system(
                        "工具执行超时。请根据已有信息尽量给出最终答复，必要时说明限制。"
                    )
                )
                if consecutive_failed_rounds >= failed_round_limit:
                    forced_finalize_reason = "tool_round_timeout"
                    break
                next_timeout = _step_timeout(deadline)
                if next_timeout <= 0:
                    forced_finalize_reason = "deadline_exceeded"
                    break
                response = await ai.generate_internal(
                    messages,
                    model=model,
                    config=reasoning_config,
                    tools=active_tools,
                    timeout=next_timeout,
                )
                total_prompt_tokens, total_completion_tokens, total_tokens = (
                    _record_usage(
                        response,
                        total_prompt_tokens=total_prompt_tokens,
                        total_completion_tokens=total_completion_tokens,
                        total_tokens=total_tokens,
                    )
                )
                continue

            messages.append(_assistant_message_from_response(response))
            messages.extend(tool_messages)

            round_successes = 0
            round_failures = 0
            failed_tools_this_round: set[str] = set()
            for call, tool_message in zip(tool_calls, tool_messages, strict=False):
                payload = _tool_message_payload(tool_message)
                tool_name = str(getattr(call.function, "name", "") or "")
                if _is_tool_result_failed(payload):
                    round_failures += 1
                    if tool_name:
                        failed_tools_this_round.add(tool_name)
                        tool_failures[tool_name] += 1
                else:
                    round_successes += 1
                    if tool_name in tool_failures:
                        tool_failures.pop(tool_name, None)

            if round_failures > 0 and round_successes == 0:
                consecutive_failed_rounds += 1
            else:
                consecutive_failed_rounds = 0

            if failed_tools_this_round and len(active_tools) > 1:
                disabled_tools: list[str] = []
                for name in failed_tools_this_round:
                    if tool_failures.get(name, 0) < tool_failure_limit:
                        continue
                    if name in active_tools:
                        active_tools.pop(name, None)
                        disabled_tools.append(name)
                if disabled_tools:
                    disabled_text = ",".join(sorted(disabled_tools))
                    logger.debug(
                        "chatinter agent circuit breaker disabled tools: "
                        f"session={session_key} tools={disabled_text}"
                    )

            if not active_tools:
                forced_finalize_reason = "all_tools_disabled"
                break

            if consecutive_failed_rounds >= failed_round_limit:
                forced_finalize_reason = "consecutive_failed_rounds"
                break

            next_timeout = _step_timeout(deadline)
            if next_timeout <= 0:
                forced_finalize_reason = "deadline_exceeded"
                break
            response = await ai.generate_internal(
                messages,
                model=model,
                config=reasoning_config,
                tools=active_tools,
                timeout=next_timeout,
            )
            total_prompt_tokens, total_completion_tokens, total_tokens = _record_usage(
                response,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
            )

        if not forced_finalize_reason:
            forced_finalize_reason = "tool_step_budget_exhausted"

        logger.debug(
            "chatinter agent finalize: "
            f"session={session_key} reason={forced_finalize_reason} "
            f"tools={len(active_tools)} calls={executed_tool_calls} "
            f"tokens={total_prompt_tokens}/{total_completion_tokens}/{total_tokens}"
        )

        messages.append(_assistant_message_from_response(response))
        messages.append(
            LLMMessage.system("请停止继续调用工具，基于当前信息直接输出最终答复。")
        )
        final_timeout = _step_timeout(deadline)
        if final_timeout > 0:
            final_response = await ai.generate_internal(
                messages,
                model=model,
                config=reasoning_config,
                timeout=final_timeout,
            )
            total_prompt_tokens, total_completion_tokens, total_tokens = _record_usage(
                final_response,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
            )
            logger.debug(
                "chatinter agent forced complete: "
                f"session={session_key} reason={forced_finalize_reason} "
                f"calls={executed_tool_calls} "
                f"tokens={total_prompt_tokens}/{total_completion_tokens}/{total_tokens}"
            )
            return final_response

        return response

    return await scheduler.run(
        session_key=session_key,
        model_key=model_key,
        runner=_run,
        interrupt_previous=True,
    )
