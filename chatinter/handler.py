"""ChatInter event entrypoint."""

from __future__ import annotations

import asyncio

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.services import logger

from .config import get_config_value, get_model_name
from .event_signals import get_event_signal, set_event_signal
from .event_runtime import (
    get_nickname,
    is_already_handled,
    mark_as_handled,
    resolve_superuser,
)
from .prompt_pipeline import PromptPipeline
from .scenario_router import ScenarioRoute, resolve_chatinter_scenario
from .turn_frame import TurnFrame


async def handle_fallback(
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message=None,
    route_modules: set[str] | None = None,
    cached_plain_text: str | None = None,
    queued: bool = False,
    turn_messages: list[str] | None = None,
    pending_human_updates: list[str] | None = None,
    scenario_route: ScenarioRoute | None = None,
) -> None:
    """Handle one ChatInter fallback turn."""

    if not get_config_value("ENABLE_FALLBACK", True):
        logger.debug("ChatInter fallback disabled")
        return

    if not queued and is_already_handled(event):
        logger.debug("event already handled, skip ChatInter")
        return

    if route_modules:
        logger.debug("event already has route modules, skip ChatInter fallback")
        return

    user_id = str(session.user.id)
    group_id = str(session.group.id) if session.group else None
    if scenario_route is None:
        scenario_route = resolve_chatinter_scenario(
            bot=bot,
            event=event,
            raw_message=raw_message,
            user_id=user_id,
            group_id=group_id,
            route_modules=route_modules,
        )
    if not scenario_route.should_handle:
        logger.debug(f"ChatInter scenario skip: {scenario_route.reason}")
        return

    frame = TurnFrame.create(
        raw_message=raw_message,
        user_id=user_id,
        group_id=group_id,
        nickname=get_nickname(session),
        bot_id=str(bot.self_id) if hasattr(bot, "self_id") else None,
        model_name=get_model_name(),
        is_superuser=resolve_superuser(bot, user_id),
        scenario=scenario_route.scenario.value,
        allow_plugin_tools=scenario_route.allow_plugin_tools,
        allow_agent_tools=scenario_route.allow_agent_tools,
        message_id=str(getattr(event, "message_id", "")),
    )
    frame.update_tags(
        scenario=scenario_route.scenario.value,
        scenario_reason=scenario_route.reason,
    )
    frame.turn_messages = _coerce_text_list(
        turn_messages
        if turn_messages is not None
        else getattr(
            event,
            "_chatinter_turn_messages",
            [],
        )
    )
    pending_updates_value = (
        pending_human_updates
        if pending_human_updates is not None
        else get_event_signal(event, "_chatinter_pending_human_updates", [])
    )
    frame.pending_human_updates = (
        pending_updates_value
        if isinstance(pending_updates_value, list)
        else _coerce_text_list(pending_updates_value)
    )
    try:
        frame.turn_priority = int(
            get_event_signal(event, "_chatinter_turn_priority", 0) or 0
        )
    except (TypeError, ValueError):
        frame.turn_priority = 0
    if not queued:
        mark_as_handled(event)
    _attach_turn_runtime_attrs(
        event,
        turn_messages=turn_messages,
        pending_human_updates=pending_human_updates,
    )
    pipeline = PromptPipeline()

    async def _dispatch_post_gate(
        *,
        response_text: str | None = None,
        phase: str = "post_gate",
    ) -> None:
        if frame.post_gate_dispatched:
            return
        middleware = frame.middleware
        middleware_state = frame.middleware_state
        if middleware is None or middleware_state is None:
            return
        if response_text is not None:
            middleware_state.response_text = response_text
        middleware_state.metadata = {
            **middleware_state.metadata,
            "phase": phase,
        }
        await middleware.dispatch("post_gate", middleware_state)
        frame.post_gate_dispatched = True

    try:
        await pipeline.bind_and_run(
            frame=frame,
            bot=bot,
            event=event,
            session=session,
            message=message,
            cached_plain_text=cached_plain_text,
            post_gate_callback=_dispatch_post_gate,
        )
    except asyncio.CancelledError:
        await pipeline.on_cancelled(frame)
        return
    except Exception as exc:
        await pipeline.on_error(frame, exc)
        return


def _attach_turn_runtime_attrs(
    event: Event,
    *,
    turn_messages: list[str] | None,
    pending_human_updates: list[str] | None,
) -> None:
    if turn_messages is None and pending_human_updates is None:
        return
    if turn_messages is not None:
        set_event_signal(event, "_chatinter_turn_queued", True)
        set_event_signal(event, "_chatinter_turn_merged", len(turn_messages) > 1)
        set_event_signal(event, "_chatinter_turn_messages", list(turn_messages))
    if pending_human_updates is not None:
        set_event_signal(
            event,
            "_chatinter_pending_human_updates",
            pending_human_updates,
        )


def _coerce_text_list(value) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


__all__ = [
    "handle_fallback",
]
