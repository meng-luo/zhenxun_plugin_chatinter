"""Direct entry for explicit superuser agent turns."""

from __future__ import annotations

from zhenxun.services import logger
from zhenxun.utils.message import MessageUtils

from ..route_text import normalize_message_text
from ..superuser_agent.runtime import (
    SuperuserSessionBusyError,
    run_superuser_agent_runtime,
)


async def handle_superuser_agent_turn(
    *,
    raw_message: str,
    session_key: str,
) -> None:
    """Run an active private Superuser Agent turn outside the chat queue."""

    normalized = normalize_message_text(raw_message)
    try:

        async def send_progress(text: str) -> None:
            await MessageUtils.build_message(text).send()

        result = await run_superuser_agent_runtime(
            message_text=normalized,
            session_key=session_key,
            progress_hook=send_progress,
        )
        if result.final_text:
            message = MessageUtils.build_message(result.final_text)
            await message.send()
    except SuperuserSessionBusyError:
        await MessageUtils.build_message(
            "当前任务仍在执行，可回复 /状态 查看进度，或回复 /中断 停止。"
        ).send()
    except Exception as exc:
        logger.error("ChatInter superuser agent failed", e=exc)
        await MessageUtils.build_message("Agent 任务执行失败，请稍后重试。").send()


__all__ = ["handle_superuser_agent_turn"]
