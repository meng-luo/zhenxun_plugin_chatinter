import asyncio
from typing import Any

from nonebot.adapters import Bot, Event

from zhenxun.services.llm import AI, LLMMessage
from zhenxun.services.llm.tools import RunContext, ToolInvoker
from zhenxun.services.llm.types import LLMContentPart, LLMResponse, LLMToolCall
from zhenxun.services.llm.types.models import ToolResult
from zhenxun.services.llm.types.protocols import BaseCallbackHandler, ToolCallData
from zhenxun.services.log import logger

from .config import build_reasoning_generation_config
from .lifecycle import ToolLifecyclePayload, get_lifecycle_manager
from .runtime import get_runtime_scheduler
from .tool_registry import ChatInterToolRegistry, ToolSelectionContext

MAX_TOOL_STEPS = 3
TOOL_EXEC_TIMEOUT = 8.0
LLM_TIMEOUT_MARGIN = 1.5
_ENABLE_TOOLS_ATTR = "_chatinter_enable_tools"
_DISABLE_TOOLS_ATTR = "_chatinter_disable_tools"


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
            "chatinter tool start: %s session=%s",
            payload.tool_name,
            self._session_id,
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
            "chatinter tool end: %s session=%s duration=%.2fms error=%s",
            payload.tool_name,
            self._session_id,
            duration * 1000,
            payload.error or "",
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

        tools = await ChatInterToolRegistry.get_tools_for_query(
            query=message_text,
            context_text=context_xml,
            max_tools=8,
            selection_context=selection_context,
        )
        if not tools:
            tools = await ChatInterToolRegistry.get_tools(
                selection_context=selection_context
            )

        if not tools:
            return await ai.generate_internal(
                messages,
                model=model,
                config=reasoning_config,
                timeout=max(timeout + LLM_TIMEOUT_MARGIN, 5),
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

        response = await ai.generate_internal(
            messages,
            model=model,
            config=reasoning_config,
            tools=tools,
            timeout=max(timeout + LLM_TIMEOUT_MARGIN, 5),
        )

        for _ in range(MAX_TOOL_STEPS):
            tool_calls = _normalize_tool_calls(response.tool_calls)
            if not tool_calls:
                return response

            tool_messages = await asyncio.wait_for(
                invoker.execute_batch(
                    tool_calls=tool_calls,
                    available_tools=tools,
                    context=context,
                ),
                timeout=TOOL_EXEC_TIMEOUT,
            )
            messages.append(_assistant_message_from_response(response))
            messages.extend(tool_messages)
            response = await ai.generate_internal(
                messages,
                model=model,
                config=reasoning_config,
                tools=tools,
                timeout=max(timeout + LLM_TIMEOUT_MARGIN, 5),
            )

        return response

    return await scheduler.run(
        session_key=session_key,
        model_key=model_key,
        runner=_run,
        interrupt_previous=True,
    )
