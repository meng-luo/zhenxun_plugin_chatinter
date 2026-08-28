"""Direct entry for explicit superuser agent turns."""

from __future__ import annotations

from typing import Any

from nonebot.adapters import Bot, Event

from zhenxun.services import logger
from zhenxun.utils.message import MessageUtils

from ..config import get_reply_delivery_settings, reply_to_trigger_message_enabled
from ..history_policy import schedule_pending_history_summary_jobs
from ..reply_delivery import build_reply_delivery_plan
from ..route_text import normalize_message_text
from ..superuser_agent.runtime import (
    SuperuserSessionBusyError,
    run_superuser_agent_runtime,
)


async def handle_superuser_agent_turn(
    *,
    bot: Bot,
    event: Event,
    raw_message: str,
    session_key: str,
) -> None:
    """Run an active private Superuser Agent turn outside the chat queue."""

    normalized = normalize_message_text(raw_message)
    try:

        async def send_progress(text: str) -> None:
            await _send_message(bot=bot, event=event, text=text)

        try:
            result = await run_superuser_agent_runtime(
                message_text=normalized,
                session_key=session_key,
                progress_hook=send_progress,
                bot_id=str(getattr(bot, "self_id", "") or ""),
            )
        finally:
            schedule_pending_history_summary_jobs()
        if result.final_text:
            if getattr(result, "status", "completed") == "completed":
                await _send_final_answer(
                    bot=bot,
                    event=event,
                    text=result.final_text,
                    reply_to=reply_to_trigger_message_enabled(),
                )
            else:
                await _send_message(
                    bot=bot,
                    event=event,
                    text=result.final_text,
                    reply_to=reply_to_trigger_message_enabled(),
                )
    except SuperuserSessionBusyError:
        await _send_message(
            bot=bot,
            event=event,
            text="当前任务仍在执行，可回复 /状态 查看进度，或回复 /中断 停止。",
        )
    except Exception as exc:
        logger.error("ChatInter superuser agent failed", e=exc)
        await _send_message(
            bot=bot,
            event=event,
            text="Agent 任务执行失败，请稍后重试。",
        )


async def _send_final_answer(
    *,
    bot: Bot,
    event: Event,
    text: str,
    reply_to: bool,
) -> None:
    _, max_chars, _ = get_reply_delivery_settings()
    plan = build_reply_delivery_plan(
        text,
        conversational=False,
        hard_limit=max_chars,
    )
    for index, segment in enumerate(plan.segments):
        await _send_message(
            bot=bot,
            event=event,
            text=segment,
            reply_to=reply_to and index == 0,
        )


async def _send_message(
    *,
    bot: Bot,
    event: Event,
    text: str,
    reply_to: bool = False,
) -> None:
    kwargs: dict[str, Any] = {"target": event, "bot": bot}
    if reply_to:
        kwargs["reply_to"] = True
    await MessageUtils.build_message(text).send(**kwargs)


__all__ = ["handle_superuser_agent_turn"]
