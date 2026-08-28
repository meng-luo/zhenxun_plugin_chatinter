from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

_active_requests = 0
_idle_waiters: set[asyncio.Event] = set()


def foreground_llm_active() -> bool:
    return _active_requests > 0


def begin_foreground_llm_activity() -> None:
    global _active_requests
    _active_requests += 1


def end_foreground_llm_activity() -> None:
    global _active_requests
    _active_requests = max(_active_requests - 1, 0)
    if _active_requests != 0:
        return
    for waiter in tuple(_idle_waiters):
        waiter.set()


async def wait_for_foreground_llm_idle() -> None:
    while foreground_llm_active():
        waiter = asyncio.Event()
        _idle_waiters.add(waiter)
        try:
            if foreground_llm_active():
                await waiter.wait()
        finally:
            _idle_waiters.discard(waiter)


@asynccontextmanager
async def foreground_llm_activity() -> AsyncIterator[None]:
    begin_foreground_llm_activity()
    try:
        yield
    finally:
        end_foreground_llm_activity()


__all__ = [
    "begin_foreground_llm_activity",
    "end_foreground_llm_activity",
    "foreground_llm_active",
    "foreground_llm_activity",
    "wait_for_foreground_llm_idle",
]
