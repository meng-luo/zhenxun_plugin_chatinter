from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zhenxun.services.llm import LLMMessage

from .middleware import TurnMiddlewareState
from .trace import StageTrace
from .turn_runtime import TurnBudgetController


class PipelineStage(str, Enum):
    PRE_GATE = "pre_gate"
    IDENTITY = "identity"
    KNOWLEDGE = "knowledge"
    EVENT_CONTEXT = "event_context"
    THREAD_CONTEXT = "thread_context"
    DIALOGUE_STATE = "dialogue_state"
    CONTEXT = "context"
    MEMORY = "memory"
    CAPABILITY_HINT = "capability_hint"
    CURRENT_USER = "current_user"
    SCRATCHPAD = "scratchpad"
    AGENT_RUN = "agent_run"
    INTENT_BUDGET = "intent_budget"
    ROUTE_PREPARE = "route_prepare"
    ROUTE_SELECTION = "route_selection"
    INTENT = "intent"
    ROUTE = "route"
    MEDIA = "media"
    MAIN_REQUEST = "main_request"
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
    scenario: str
    allow_plugin_tools: bool
    allow_agent_tools: bool
    trace: StageTrace
    budget_controller: TurnBudgetController
    current_message: str = ""
    route_message: str = ""
    system_prompt: str = ""
    context_xml: str = ""
    enriched_context_xml: str = ""
    history_messages: list[LLMMessage] = field(default_factory=list)
    post_gate_dispatched: bool = False
    event_message: Any | None = None
    uni_msg: Any | None = None
    bot: Any | None = None
    event: Any | None = None
    session: Any | None = None
    message: Any | None = None
    cached_plain_text: str | None = None
    middleware: Any | None = None
    middleware_state: TurnMiddlewareState | None = None
    main_result: Any | None = None
    final_envelope: Any | None = None
    response_quality_result: Any | None = None
    chat_execution_frame: Any | None = None
    post_gate_callback: Any | None = None
    turn_finished: bool = False
    turn_messages: list[str] = field(default_factory=list)
    pending_human_updates: list[str] = field(default_factory=list)
    turn_priority: int = 0
    event_context: Any | None = None
    dialogue_context_pack: Any | None = None
    addressee_result: Any | None = None
    thread_context: Any | None = None
    intervention_decision: Any | None = None
    knowledge_base: Any | None = None
    selection_context: Any | None = None
    command_tools: list[Any] = field(default_factory=list)
    intent_profile: Any | None = None
    native_decision: Any | None = None
    route_result: Any | None = None
    route_report: Any | None = None
    dialogue_plan: Any | None = None
    dialogue_state: Any | None = None
    chat_runtime_profile: Any | None = None
    chat_tool_exposure_state: str = "unknown"
    previous_dialogue_state: Any | None = None
    chat_memory_layered: Any | None = None
    mention_name_map: dict[str, str] = field(default_factory=dict)
    mention_profiles: dict[str, dict[str, str]] = field(default_factory=dict)
    reply_images_data: list[Any] = field(default_factory=list)
    reply_image_segments_for_reroute: list[Any] = field(default_factory=list)
    image_parts: list[Any] = field(default_factory=list)
    agent_messages: list[LLMMessage] = field(default_factory=list)
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
        scenario: str = "group_plugin_selector",
        allow_plugin_tools: bool = True,
        allow_agent_tools: bool = False,
        message_id: str = "",
    ) -> "TurnFrame":
        session_key = str(group_id or user_id)
        trace = StageTrace(
            "chatinter",
            tags={
                "user": str(user_id),
                "group": str(group_id) if group_id else "private",
                "message_id": str(message_id or ""),
                "scenario": str(scenario or ""),
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
            scenario=str(scenario or "group_plugin_selector"),
            allow_plugin_tools=bool(allow_plugin_tools),
            allow_agent_tools=bool(allow_agent_tools),
            trace=trace,
            budget_controller=TurnBudgetController.for_session(session_key),
            current_message=raw_message,
        )

    def bind_runtime(
        self,
        *,
        bot: Any,
        event: Any,
        session: Any,
        message: Any | None,
        cached_plain_text: str | None,
        middleware: Any,
        post_gate_callback: Any | None = None,
    ) -> None:
        self.bot = bot
        self.event = event
        self.session = session
        self.message = message
        self.cached_plain_text = cached_plain_text
        self.middleware = middleware
        self.middleware_state = self.create_middleware_state()
        self.post_gate_callback = post_gate_callback

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
        history_messages: list[LLMMessage] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.context_xml = context_xml
        self.enriched_context_xml = context_xml
        self.reply_images_data = list(reply_images_data or [])
        self.history_messages = list(history_messages or [])

    def set_native_route(
        self,
        *,
        native_decision: Any,
        route_result: Any | None,
        route_report: Any | None,
    ) -> None:
        self.native_decision = native_decision
        self.route_result = route_result
        self.route_report = route_report


__all__ = [
    "PipelineStage",
    "TurnFrame",
]
