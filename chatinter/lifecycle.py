from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from zhenxun.services.log import logger

LifecycleStage = Literal[
    "pre_gate",
    "post_gate",
    "before_intent",
    "after_intent",
    "before_route",
    "after_route",
    "before_chat",
    "after_chat",
    "on_error",
]


@dataclass
class LifecyclePayload:
    user_id: str
    group_id: str | None
    message_text: str
    system_prompt: str
    context_xml: str
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    response_text: str | None = None


LifecycleHook = Callable[[Any], Awaitable[None]]


class ChatInterLifecycleManager:
    def __init__(self) -> None:
        self._hooks: dict[LifecycleStage, list[LifecycleHook]] = {
            "pre_gate": [],
            "post_gate": [],
            "before_intent": [],
            "after_intent": [],
            "before_route": [],
            "after_route": [],
            "before_chat": [],
            "after_chat": [],
            "on_error": [],
        }
        self._lock = asyncio.Lock()

    async def register(self, stage: LifecycleStage, hook: LifecycleHook) -> None:
        async with self._lock:
            self._hooks.setdefault(stage, []).append(hook)

    async def dispatch(self, stage: LifecycleStage, payload: Any) -> None:
        hooks = list(self._hooks.get(stage, []))
        for hook in hooks:
            try:
                await hook(payload)
            except Exception as exc:
                logger.warning(
                    f"chatinter lifecycle hook failed: stage={stage}, error={exc}"
                )


_lifecycle_manager = ChatInterLifecycleManager()
_startup_registered = False
_startup_lock = asyncio.Lock()


async def ensure_lifecycle_hooks_registered() -> None:
    global _startup_registered
    if _startup_registered:
        return
    async with _startup_lock:
        _startup_registered = True


def get_lifecycle_manager() -> ChatInterLifecycleManager:
    return _lifecycle_manager


__all__ = [
    "ChatInterLifecycleManager",
    "LifecyclePayload",
    "LifecycleStage",
    "ensure_lifecycle_hooks_registered",
    "get_lifecycle_manager",
]
