"""Prompt pipeline orchestration for ChatInter turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from .middleware import get_middleware_manager
from .pipeline_stages import (
    handle_pipeline_cancelled,
    handle_pipeline_error,
    stage_agent_run,
    stage_capability_hint,
    stage_current_user,
    stage_event_context,
    stage_identity,
    stage_memory,
    stage_persist,
    stage_scratchpad,
    stage_send,
    stage_thread_context,
)
from .turn_frame import TurnFrame


PostGateCallback = Callable[..., Awaitable[None]]


@dataclass
class PromptPipeline:
    """ChatLuna-style pipeline over a mutable ``TurnFrame``."""

    stage_names: tuple[str, ...] = field(
        default=(
            "identity",
            "event_context",
            "thread_context",
            "memory",
            "capability_hint",
            "current_user",
            "scratchpad",
            "agent_run",
            "persist",
            "send",
        )
    )

    async def bind_and_run(
        self,
        *,
        frame: TurnFrame,
        bot: Bot,
        event: Event,
        session: Uninfo,
        message,
        cached_plain_text: str | None,
        post_gate_callback: PostGateCallback,
    ) -> None:
        middleware = get_middleware_manager()
        frame.bind_runtime(
            bot=bot,
            event=event,
            session=session,
            message=message,
            cached_plain_text=cached_plain_text,
            middleware=middleware,
            post_gate_callback=post_gate_callback,
        )
        await self.run(frame)

    async def run(self, frame: TurnFrame) -> None:
        bot = _require(frame.bot, "bot")
        event = _require(frame.event, "event")
        session = _require(frame.session, "session")
        middleware = _require(frame.middleware, "middleware")
        middleware_state = _require(frame.middleware_state, "middleware_state")

        await stage_identity(
            frame=frame,
            event=event,
            middleware_state=middleware_state,
            middleware=middleware,
        )
        await stage_event_context(
            frame=frame,
            bot=bot,
            event=event,
            session=session,
            message=frame.message,
            cached_plain_text=frame.cached_plain_text,
        )
        await stage_thread_context(frame=frame)
        await stage_memory(frame=frame, bot=bot, event=event)
        await stage_capability_hint(
            frame=frame,
            bot=bot,
            event=event,
            middleware_state=middleware_state,
            middleware=middleware,
            cached_plain_text=frame.cached_plain_text,
        )
        await stage_current_user(frame=frame, message=frame.message)
        await stage_scratchpad(
            frame=frame,
            middleware_state=middleware_state,
            middleware=middleware,
        )
        await stage_agent_run(
            frame=frame,
            bot=bot,
            event=event,
            middleware_state=middleware_state,
            middleware=middleware,
        )
        await stage_persist(frame)
        await stage_send(frame)

    async def on_cancelled(self, frame: TurnFrame) -> None:
        await handle_pipeline_cancelled(frame)

    async def on_error(self, frame: TurnFrame, error: Exception) -> None:
        await handle_pipeline_error(frame, error)


def _require(value, name: str):
    if value is None:
        raise RuntimeError(f"missing pipeline runtime value: {name}")
    return value


__all__ = ["PromptPipeline"]
