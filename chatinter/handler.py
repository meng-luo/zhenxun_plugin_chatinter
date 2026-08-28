"""ChatInter event entrypoint."""

from __future__ import annotations

import asyncio

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.services import logger

from .config import chatinter_available, get_agent_model
from .event_runtime import (
    get_nickname,
    is_already_handled,
    mark_as_handled,
    resolve_superuser,
)
from .event_signals import get_event_signal
from .prompt_pipeline import PromptPipeline
from .scenario_router import (
    ChatInterScenario,
    ScenarioRoute,
    resolve_chatinter_scenario,
)
from .session_identity import conversation_session_key, legacy_session_key
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
    scenario_route: ScenarioRoute | None = None,
) -> None:
    """Handle one ChatInter fallback turn."""

    handled_session_key = conversation_session_key(session)
    if not queued and is_already_handled(
        event,
        session_key=handled_session_key,
    ):
        logger.debug("event already handled, skip ChatInter")
        return

    if route_modules:
        logger.debug("event already has route modules, skip ChatInter fallback")
        return

    user_id = str(session.user.id)
    group_id = str(session.group.id) if session.group else None
    if not chatinter_available(group_id):
        logger.debug("ChatInter 当前会话未启用")
        return
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
    if scenario_route.scenario is ChatInterScenario.SUPERUSER_AGENT:
        logger.debug("superuser agent must use its direct entry")
        return

    try:
        queue_wait_ms = max(
            float(
                get_event_signal(
                    event,
                    "_chatinter_turn_queue_wait_ms",
                    0.0,
                )
                or 0.0
            ),
            0.0,
        )
    except (TypeError, ValueError):
        queue_wait_ms = 0.0

    frame = TurnFrame.create(
        raw_message=raw_message,
        user_id=user_id,
        group_id=group_id,
        nickname=get_nickname(session),
        bot_id=str(bot.self_id) if hasattr(bot, "self_id") else None,
        model_name=get_agent_model("chat"),
        is_superuser=resolve_superuser(bot, user_id),
        scenario=scenario_route.scenario.value,
        allow_plugin_tools=scenario_route.allow_plugin_tools,
        message_id=str(getattr(event, "message_id", "")),
        session_key=conversation_session_key(session),
        legacy_session_key=legacy_session_key(session),
        queue_wait_ms=queue_wait_ms,
    )
    frame.turn_generation = int(
        get_event_signal(event, "_chatinter_turn_generation", 0) or 0
    )
    frame.current_turn_guard = get_event_signal(
        event,
        "_chatinter_turn_is_current",
        None,
    )
    frame.update_tags(
        scenario=scenario_route.scenario.value,
        scenario_reason=scenario_route.reason,
    )
    try:
        frame.turn_priority = int(
            get_event_signal(event, "_chatinter_turn_priority", 0) or 0
        )
    except (TypeError, ValueError):
        frame.turn_priority = 0
    if not queued:
        mark_as_handled(event, session_key=handled_session_key)
    pipeline = PromptPipeline()

    try:
        await pipeline.bind_and_run(
            frame=frame,
            bot=bot,
            event=event,
            session=session,
            message=message,
            cached_plain_text=cached_plain_text,
        )
    except asyncio.CancelledError:
        await pipeline.on_cancelled(frame)
        return
    except Exception as exc:
        await pipeline.on_error(frame, exc)
        return


__all__ = [
    "handle_fallback",
]
