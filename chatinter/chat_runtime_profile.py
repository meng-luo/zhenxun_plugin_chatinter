"""Minimal chat-only profile used by ordinary ChatInter conversations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .chat_dialogue_planner import (
    ChatDialoguePlan,
    DialogueState,
    build_dialogue_state,
    load_dialogue_state,
    normalize_message_text,
    plan_chat_dialogue,
)
from .persona import PersonaSelection, resolve_persona

if TYPE_CHECKING:
    from .thread_resolver import ThreadContext


@dataclass(frozen=True)
class ChatRuntimeProfile:
    dialogue_plan: ChatDialoguePlan
    dialogue_state: DialogueState
    previous_state: DialogueState | None = None
    persona_selection: PersonaSelection | None = None
    reason: str = "chat_runtime_profile"

    def persona_prompt_fragment(self) -> str:
        if self.persona_selection is None:
            return ""
        return self.persona_selection.persona.prompt_fragment()


def build_chat_runtime_profile(
    *,
    session_key: str,
    user_id: str,
    group_id: str | None,
    message_text: str,
    scenario: str,
    has_images: bool = False,
    has_reply: bool = False,
    is_group: bool = False,
    intent: Any | None = None,
    previous_state: DialogueState | None = None,
    dialogue_context_pack: Any | None = None,
    thread_context: "ThreadContext | None" = None,
    legacy_session_key: str = "",
) -> ChatRuntimeProfile:
    loaded_previous = previous_state
    if loaded_previous is None:
        loaded_previous = load_dialogue_state(
            session_key=session_key,
            user_id=user_id,
            group_id=group_id,
            legacy_session_key=legacy_session_key,
        )
    relevant_people = tuple(getattr(dialogue_context_pack, "relevant_people", ()) or ())
    thread = thread_context or getattr(dialogue_context_pack, "thread", None)
    thread_topic = normalize_message_text(str(getattr(thread, "topic_key", "") or ""))
    plan = plan_chat_dialogue(
        message_text=message_text,
        intent=intent,
        has_images=has_images,
        has_reply=has_reply,
    )
    state = build_dialogue_state(
        message_text=message_text,
        plan=plan,
        intent=intent,
        scenario=scenario,
        has_images=has_images,
        has_reply=has_reply,
        previous_state=loaded_previous,
        is_group=is_group,
        relevant_people_count=len(relevant_people),
        thread_topic=thread_topic,
    )
    persona_selection = resolve_persona(
        session_key=session_key,
        user_id=user_id,
        group_id=group_id,
        scenario=scenario,
    )
    return ChatRuntimeProfile(
        dialogue_plan=plan,
        dialogue_state=state,
        previous_state=loaded_previous,
        persona_selection=persona_selection,
        reason=f"{plan.reason}:{state.reason}",
    )


__all__ = ["ChatRuntimeProfile", "build_chat_runtime_profile"]
