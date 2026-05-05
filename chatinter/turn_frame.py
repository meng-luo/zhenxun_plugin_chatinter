from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .middleware import TurnMiddlewareState
from .trace import StageTrace
from .turn_runtime import TurnBudgetController


class PipelineStage(str, Enum):
    PRE_GATE = "pre_gate"
    KNOWLEDGE = "knowledge"
    EVENT_CONTEXT = "event_context"
    CONTEXT = "context"
    INTENT_BUDGET = "intent_budget"
    ROUTE_PREPARE = "route_prepare"
    ROUTE_SELECTION = "route_selection"
    INTENT = "intent"
    ROUTE = "route"
    MEDIA = "media"
    AGENT_BUDGET = "agent_budget"
    CHAT_FALLBACK = "chat_fallback"
    PERSIST = "persist"
    SEND = "send"
    NOTIFY = "notify"
    ERROR = "error"


@dataclass
class TurnFrame:
    """Mutable state for one ChatInter turn.

    `handler.py` still owns the orchestration, but all cross-stage state should
    live here so later stage extraction does not keep growing local variables.
    """

    raw_message: str
    user_id: str
    group_id: str | None
    nickname: str
    bot_id: str | None
    model_name: str | None
    session_key: str
    is_superuser: bool
    trace: StageTrace
    budget_controller: TurnBudgetController
    current_message: str = ""
    route_message: str = ""
    system_prompt: str = ""
    context_xml: str = ""
    enriched_context_xml: str = ""
    router_force_pure_chat: bool = False
    completion_disabled_force_chat: bool = False
    post_gate_dispatched: bool = False
    event_message: Any | None = None
    uni_msg: Any | None = None
    event_context: Any | None = None
    dialogue_context_pack: Any | None = None
    addressee_result: Any | None = None
    thread_context: Any | None = None
    intervention_decision: Any | None = None
    knowledge_base: Any | None = None
    selection_context: Any | None = None
    command_tools: list[Any] = field(default_factory=list)
    intent_profile: Any | None = None
    router_decision: Any | None = None
    route_result: Any | None = None
    route_report: Any | None = None
    dialogue_plan: Any | None = None
    mention_name_map: dict[str, str] = field(default_factory=dict)
    mention_profiles: dict[str, dict[str, str]] = field(default_factory=dict)
    reply_images_data: list[Any] = field(default_factory=list)
    reply_image_segments_for_reroute: list[Any] = field(default_factory=list)
    image_parts: list[Any] = field(default_factory=list)
    has_reply: bool = False
    reply_sender_id: str | None = None
    reply_image_count: int = 0

    @classmethod
    def create(
        cls,
        *,
        raw_message: str,
        user_id: str,
        group_id: str | None,
        nickname: str,
        bot_id: str | None,
        model_name: str | None,
        is_superuser: bool,
        message_id: str = "",
    ) -> "TurnFrame":
        session_key = str(group_id or user_id)
        trace = StageTrace(
            "chatinter",
            tags={
                "user": str(user_id),
                "group": str(group_id) if group_id else "private",
                "message_id": str(message_id or ""),
            },
        )
        return cls(
            raw_message=raw_message,
            user_id=str(user_id),
            group_id=str(group_id) if group_id else None,
            nickname=nickname,
            bot_id=bot_id,
            model_name=model_name,
            session_key=session_key,
            is_superuser=is_superuser,
            trace=trace,
            budget_controller=TurnBudgetController.for_session(session_key),
            current_message=raw_message,
        )

    def stage(self, stage: PipelineStage | str) -> None:
        label = stage.value if isinstance(stage, PipelineStage) else str(stage)
        self.trace.stage(label)

    def update_tags(self, **kwargs: str | float | None) -> None:
        self.trace.update_tags(**kwargs)

    def set_tag(self, key: str, value: str | float | None) -> None:
        self.trace.set_tag(key, value)

    def create_middleware_state(self) -> TurnMiddlewareState:
        return TurnMiddlewareState(
            session_key=self.session_key,
            user_id=self.user_id,
            group_id=self.group_id,
            message_text=self.current_message or self.raw_message,
            system_prompt=self.system_prompt,
            context_xml=self.context_xml,
            model_name=self.model_name,
            budget_controller=self.budget_controller,
            metadata={"phase": PipelineStage.PRE_GATE.value},
        )

    def sync_to_middleware(
        self,
        state: TurnMiddlewareState,
        *,
        phase: str,
        route_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        state.message_text = self.current_message
        state.system_prompt = self.system_prompt
        state.context_xml = self.context_xml
        if route_message is not None:
            state.route_message = route_message
        state.metadata = {"phase": phase, **(metadata or {})}

    def apply_prompt_state(self, state: TurnMiddlewareState) -> None:
        self.system_prompt = state.system_prompt
        self.context_xml = state.context_xml
        if state.route_message:
            self.route_message = state.route_message

    def set_context(
        self,
        *,
        system_prompt: str,
        context_xml: str,
        reply_images_data: list[Any],
    ) -> None:
        self.system_prompt = system_prompt
        self.context_xml = context_xml
        self.enriched_context_xml = context_xml
        self.reply_images_data = list(reply_images_data or [])

    def set_route_result(
        self,
        *,
        router_decision: Any,
        route_result: Any | None,
        route_report: Any | None,
    ) -> None:
        self.router_decision = router_decision
        self.route_result = route_result
        self.route_report = route_report


__all__ = [
    "PipelineStage",
    "TurnFrame",
]
