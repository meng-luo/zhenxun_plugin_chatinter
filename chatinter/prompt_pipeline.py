"""Prompt pipeline orchestration for ChatInter turns.

One unified runtime shape: every turn (group or private) builds the full chat
context, then runs the unified agent which decides between plugin invocation
and plain reply inside a single model loop.  The old plugin-router /
chat-degrade split no longer exists.
"""

from __future__ import annotations

import time

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from .group_plugin_flow import (
    stage_group_capability_hint,
    stage_route_media_context,
)
from .gscore_adapter import get_gscore_adapter
from .pipeline_stages import (
    complete_suppressed_turn,
    handle_pipeline_cancelled,
    handle_pipeline_error,
    stage_chat_capability_hint,
    stage_current_user,
    stage_dialogue_state,
    stage_event_context,
    stage_identity,
    stage_memory,
    stage_persist,
    stage_scratchpad,
    stage_send,
    stage_thread_context,
)
from .turn_frame import PipelineStage, TurnFrame
from .unified_flow import stage_unified_run


class PromptPipeline:
    """ChatLuna-style pipeline over a mutable ``TurnFrame``."""

    async def bind_and_run(
        self,
        *,
        frame: TurnFrame,
        bot: Bot,
        event: Event,
        session: Uninfo,
        message,
        cached_plain_text: str | None,
    ) -> None:
        frame.bind_runtime(
            bot=bot,
            event=event,
            session=session,
            message=message,
            cached_plain_text=cached_plain_text,
        )
        await self.run(frame)

    async def run(self, frame: TurnFrame) -> None:
        bot = _require(frame.bot, "bot")
        event = _require(frame.event, "event")
        session = _require(frame.session, "session")

        await stage_identity(
            frame=frame,
            event=event,
        )
        await stage_event_context(
            frame=frame,
            bot=bot,
            event=event,
            session=session,
            message=frame.message,
            cached_plain_text=frame.cached_plain_text,
        )
        gscore_route_started = time.perf_counter()
        gscore_route = await get_gscore_adapter().route_turn(frame)
        frame.gscore_route_result = gscore_route
        frame.update_tags(
            gscore_route=gscore_route.disposition,
            gscore_matches=float(len(gscore_route.matches)),
            gscore_route_ms=(time.perf_counter() - gscore_route_started) * 1000,
        )
        frame.stage(PipelineStage.GSCORE_ROUTE)
        if gscore_route.suppress_chatinter:
            complete_suppressed_turn(
                frame,
                reason=f"gscore_{gscore_route.disposition}",
            )
            return
        await stage_thread_context(frame=frame, bot=bot)
        if frame.allow_plugin_tools:
            await stage_route_media_context(frame=frame, bot=bot, event=event)
            await stage_group_capability_hint(
                frame=frame,
                bot=bot,
                event=event,
                cached_plain_text=frame.cached_plain_text,
            )
        else:
            await stage_chat_capability_hint(
                frame=frame,
                bot=bot,
                event=event,
                cached_plain_text=frame.cached_plain_text,
            )
        await stage_dialogue_state(frame=frame)
        await stage_memory(frame=frame, bot=bot, event=event)
        await stage_current_user(frame=frame, message=frame.message)
        await stage_scratchpad(
            frame=frame,
        )
        await stage_unified_run(
            frame=frame,
            bot=bot,
            event=event,
        )
        await stage_send(frame)
        await stage_persist(frame)

    async def on_cancelled(self, frame: TurnFrame) -> None:
        await handle_pipeline_cancelled(frame)

    async def on_error(self, frame: TurnFrame, error: Exception) -> None:
        await handle_pipeline_error(frame, error)


def _require(value, name: str):
    if value is None:
        raise RuntimeError(f"missing pipeline runtime value: {name}")
    return value


__all__ = ["PromptPipeline"]
