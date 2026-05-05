from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Literal

from .intent_classifier import IntentClassification
from .lifecycle import LifecyclePayload, get_lifecycle_manager
from .turn_runtime import TurnBudgetController

MiddlewareStage = Literal[
    "pre_gate",
    "before_intent",
    "after_intent",
    "before_route",
    "after_route",
    "before_chat",
    "after_chat",
    "before_agent",
    "after_agent",
    "post_gate",
    "on_error",
]


@dataclass
class TurnMiddlewareState:
    session_key: str
    user_id: str
    group_id: str | None
    message_text: str
    system_prompt: str
    context_xml: str
    model_name: str | None
    intent: IntentClassification | None = None
    route_message: str = ""
    response_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    budget_controller: TurnBudgetController | None = None


class ChatInterMiddlewareManager:
    async def dispatch(
        self, stage: MiddlewareStage, state: TurnMiddlewareState
    ) -> None:
        controller = state.budget_controller
        started = time.perf_counter()
        if controller is not None and not controller.allow_hook(stage):
            return
        try:
            await self._run_stage(stage, state)
        finally:
            if controller is not None:
                controller.record_hook(stage, time.perf_counter() - started)

    async def _run_stage(
        self, stage: MiddlewareStage, state: TurnMiddlewareState
    ) -> None:
        lifecycle = get_lifecycle_manager()
        phase = state.metadata.get("phase") or stage
        payload = LifecyclePayload(
            user_id=state.user_id,
            group_id=state.group_id,
            message_text=state.route_message or state.message_text,
            system_prompt=state.system_prompt,
            context_xml=state.context_xml,
            model_name=state.model_name,
            metadata={**state.metadata, "phase": phase, "middleware_stage": stage},
            response_text=state.response_text,
        )
        await lifecycle.dispatch(stage, payload)
        _apply_payload(state, payload)


def _apply_payload(state: TurnMiddlewareState, payload: LifecyclePayload) -> None:
    state.system_prompt = payload.system_prompt
    state.context_xml = payload.context_xml
    state.response_text = payload.response_text
    state.metadata.update(payload.metadata)


_middleware_manager = ChatInterMiddlewareManager()


def get_middleware_manager() -> ChatInterMiddlewareManager:
    return _middleware_manager


__all__ = [
    "ChatInterMiddlewareManager",
    "MiddlewareStage",
    "TurnMiddlewareState",
    "get_middleware_manager",
]
