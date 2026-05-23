"""Independent chat-side runtime profile.

This module deliberately avoids plugin registry, command candidates and tool
obligation.  It can shape ordinary replies, memory recall and group tone, but
it must not decide whether a plugin/tool should run.
"""

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
from .group_chat_policy import GroupChatPolicyDecision, decide_group_chat_policy

if TYPE_CHECKING:
    from .thread_resolver import ThreadContext


@dataclass(frozen=True)
class ChatRuntimeProfile:
    """Chat-only state exposed to prompt/memory/quality layers."""

    dialogue_plan: ChatDialoguePlan
    dialogue_state: DialogueState
    previous_state: DialogueState | None = None
    person_facts: tuple[str, ...] = ()
    relationship_facts: tuple[str, ...] = ()
    preference_facts: tuple[str, ...] = ()
    thread_facts: tuple[str, ...] = ()
    group_policy: GroupChatPolicyDecision | None = None
    reason: str = "chat_runtime_profile"

    def to_context_xml(self) -> str:
        lines = [
            '<chat_runtime_profile scope="chat_response_only">',
            f"kind={self.dialogue_plan.kind}",
            f"style={self.dialogue_plan.style}",
            f"tone={self.dialogue_state.tone}",
            f"user_emotion={self.dialogue_state.user_emotion}",
            f"reply_posture={self.dialogue_state.reply_posture}",
            f"group_atmosphere={self.dialogue_state.group_atmosphere}",
            f"address_mode={self.dialogue_state.address_mode}",
            f"response_length={self.dialogue_state.response_length}",
            f"continuity={self.dialogue_state.continuity}",
            f"group_reply_policy={self.dialogue_state.group_reply_policy}",
        ]
        if self.dialogue_state.topic_hint:
            lines.append(f"topic_hint={_xml_escape(self.dialogue_state.topic_hint)}")
        if self.person_facts:
            lines.append("<person_facts>")
            lines.extend(_compact_lines(self.person_facts, limit=8))
            lines.append("</person_facts>")
        if self.relationship_facts:
            lines.append("<relationship_facts>")
            lines.extend(_compact_lines(self.relationship_facts, limit=6))
            lines.append("</relationship_facts>")
        if self.preference_facts:
            lines.append("<preference_facts>")
            lines.extend(_compact_lines(self.preference_facts, limit=6))
            lines.append("</preference_facts>")
        if self.thread_facts:
            lines.append("<thread_facts>")
            lines.extend(_compact_lines(self.thread_facts, limit=6))
            lines.append("</thread_facts>")
        if self.group_policy is not None:
            lines.append(self.group_policy.to_xml())
        lines.append(
            "rule=This profile is for direct chat wording only; it must not "
            "select tools, fill tool arguments, or override plugin execution."
        )
        lines.append("</chat_runtime_profile>")
        return "\n".join(lines)


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
    previous_state: DialogueState | None = None,
    dialogue_context_pack: Any | None = None,
    thread_context: "ThreadContext | None" = None,
) -> ChatRuntimeProfile:
    """Build chat-side state without consulting tool or plugin signals."""

    loaded_previous = previous_state
    if loaded_previous is None:
        loaded_previous = load_dialogue_state(
            session_key=session_key,
            user_id=user_id,
            group_id=group_id,
        )
    relevant_people = tuple(
        getattr(dialogue_context_pack, "relevant_people", ()) or ()
    )
    thread = thread_context or getattr(dialogue_context_pack, "thread", None)
    thread_topic = normalize_message_text(str(getattr(thread, "topic_key", "") or ""))
    plan = plan_chat_dialogue(
        message_text=message_text,
        intent=None,
        has_images=has_images,
        has_reply=has_reply,
    )
    state = build_dialogue_state(
        message_text=message_text,
        plan=plan,
        intent=None,
        scenario=scenario,
        has_images=has_images,
        has_reply=has_reply,
        previous_state=loaded_previous,
        is_group=is_group,
        relevant_people_count=len(relevant_people),
        thread_topic=thread_topic,
    )
    group_policy = decide_group_chat_policy(
        message_text=message_text,
        is_group=is_group,
        is_to_me=scenario in {"private_chat", "superuser_agent"},
        dialogue_state=state,
        thread_context=thread,
    )
    person_facts, relationship_facts, preference_facts = _person_fact_layers(
        dialogue_context_pack
    )
    return ChatRuntimeProfile(
        dialogue_plan=plan,
        dialogue_state=state,
        previous_state=loaded_previous,
        person_facts=tuple(person_facts),
        relationship_facts=tuple(relationship_facts),
        preference_facts=tuple(preference_facts),
        thread_facts=_thread_facts(thread),
        group_policy=group_policy,
        reason=f"{plan.reason}:{state.reason}",
    )


def _person_fact_layers(
    dialogue_context_pack: Any | None,
) -> tuple[list[str], list[str], list[str]]:
    person_facts: list[str] = []
    relationship_facts: list[str] = []
    preference_facts: list[str] = []
    seen: set[str] = set()

    def append(profile: Any, *, prefix: str) -> None:
        user_id = normalize_message_text(str(getattr(profile, "user_id", "") or ""))
        if not user_id or user_id in seen:
            return
        seen.add(user_id)
        person, relationship, preference = _format_profile_fact_layers(
            profile,
            prefix=prefix,
        )
        person_facts.extend(person)
        relationship_facts.extend(relationship)
        preference_facts.extend(preference)

    if dialogue_context_pack is None:
        return [], [], []
    append(getattr(dialogue_context_pack, "speaker_profile", None), prefix="speaker")
    relevant_people = tuple(
        getattr(dialogue_context_pack, "relevant_people", ()) or ()
    )
    for index, person in enumerate(relevant_people[:6], 1):
        profile = getattr(person, "profile", None)
        append(profile, prefix=f"person_{index}")
    return person_facts[:16], relationship_facts[:8], preference_facts[:8]


def _format_profile_fact_layers(
    profile: Any,
    *,
    prefix: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    label = normalize_message_text(prefix) or "person"
    person_facts: list[str] = []
    relationship_facts: list[str] = []
    preference_facts: list[str] = []
    display_name = _profile_display_name(profile)
    nickname = normalize_message_text(str(getattr(profile, "nickname", "") or ""))
    group_card = normalize_message_text(str(getattr(profile, "group_card", "") or ""))
    aliases = _profile_aliases(profile)
    known_facts = _profile_known_facts(profile)
    relationship = normalize_message_text(
        str(getattr(profile, "relationship", "") or "")
    )
    conflict_state = normalize_message_text(
        str(getattr(profile, "conflict_state", "") or "")
    )
    if display_name:
        person_facts.append(f"{label}_name={display_name}")
    if nickname:
        person_facts.append(f"{label}_nickname={nickname}")
    if group_card:
        person_facts.append(f"{label}_group_card={group_card}")
    if aliases:
        person_facts.append(f"{label}_aliases={'、'.join(aliases[:6])}")
    if known_facts:
        person_facts.append(f"{label}_known_facts={'；'.join(known_facts[:4])}")
    if conflict_state:
        person_facts.append(f"{label}_conflict_state={conflict_state}")
    if relationship:
        relationship_facts.append(f"{label}_relationship={relationship}")
    for fact in known_facts:
        if any(marker in fact for marker in ("喜欢", "不喜欢", "偏好", "讨厌")):
            preference_facts.append(f"{label}_preference={fact}")
    return (
        tuple(person_facts),
        tuple(relationship_facts),
        tuple(preference_facts[:4]),
    )


def _profile_display_name(profile: Any) -> str:
    display_name = getattr(profile, "display_name", "")
    if display_name:
        return normalize_message_text(str(display_name))
    for attr in ("group_card", "nickname", "user_id"):
        value = normalize_message_text(str(getattr(profile, attr, "") or ""))
        if value:
            return value
    return ""


def _profile_aliases(profile: Any) -> tuple[str, ...]:
    aliases = getattr(profile, "aliases", ()) or ()
    if isinstance(aliases, str):
        aliases = aliases.replace("，", "、").split("、")
    normalized = (normalize_message_text(str(value)) for value in aliases)
    return tuple(
        item
        for item in dict.fromkeys(normalized)
        if item
    )[:8]


def _profile_known_facts(profile: Any) -> tuple[str, ...]:
    facts = getattr(profile, "known_facts", ()) or ()
    if isinstance(facts, str):
        facts = facts.replace("；", "、").split("、")
    normalized = (normalize_message_text(str(value)) for value in facts)
    return tuple(
        item
        for item in dict.fromkeys(normalized)
        if item
    )[:8]


def _thread_facts(thread: Any | None) -> tuple[str, ...]:
    if thread is None:
        return ()
    facts: list[str] = []
    thread_id = normalize_message_text(str(getattr(thread, "thread_id", "") or ""))
    topic = normalize_message_text(str(getattr(thread, "topic_key", "") or ""))
    source = normalize_message_text(str(getattr(thread, "source", "") or ""))
    confidence = getattr(thread, "confidence", 0.0)
    if thread_id:
        facts.append(f"thread_id={thread_id}")
    if topic:
        facts.append(f"topic={topic}")
    if source:
        facts.append(f"source={source}")
    try:
        facts.append(f"confidence={float(confidence or 0.0):.2f}")
    except Exception:
        pass
    participants = tuple(getattr(thread, "related_user_ids", ()) or ())
    if participants:
        facts.append("participants=" + ",".join(str(item) for item in participants[:8]))
    pending = tuple(getattr(thread, "pending_entities", ()) or ())
    if pending:
        facts.append("pending_entities=" + "、".join(str(item) for item in pending[:6]))
    return tuple(facts)


def _compact_lines(values: tuple[str, ...] | list[str], *, limit: int) -> list[str]:
    lines: list[str] = []
    for value in values[: max(int(limit or 0), 0)]:
        text = normalize_message_text(str(value or ""))
        if text and text not in lines:
            lines.append(_xml_escape(text[:180]))
    return lines


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


__all__ = ["ChatRuntimeProfile", "build_chat_runtime_profile"]
