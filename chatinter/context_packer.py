from __future__ import annotations

from .addressee_resolver import AddresseeResult, format_addressee_xml
from .event_context import ChatInterEventContext
from .person_registry import PersonProfile, RelevantPerson, format_profile_lines
from .thread_resolver import ThreadContext, format_thread_xml


class DialogueContextPack:
    def __init__(
        self,
        *,
        event_context: ChatInterEventContext,
        speaker_profile: PersonProfile | None,
        addressee: AddresseeResult | None,
        thread: ThreadContext | None,
        relevant_people: tuple[RelevantPerson, ...] = (),
    ) -> None:
        self.event_context = event_context
        self.speaker_profile = speaker_profile
        self.addressee = addressee
        self.thread = thread
        self.relevant_people = relevant_people

    def to_context_xml(self) -> str:
        return build_group_dialogue_context(
            event_context=self.event_context,
            speaker_profile=self.speaker_profile,
            addressee=self.addressee,
            thread=self.thread,
            relevant_people=self.relevant_people,
        )


def append_group_dialogue_context(
    context_xml: str,
    *,
    event_context: ChatInterEventContext,
    speaker_profile: PersonProfile | None,
    addressee: AddresseeResult | None,
    thread: ThreadContext | None,
    relevant_people: tuple[RelevantPerson, ...] = (),
) -> str:
    packed = build_group_dialogue_context(
        event_context=event_context,
        speaker_profile=speaker_profile,
        addressee=addressee,
        thread=thread,
        relevant_people=relevant_people,
    )
    if not packed:
        return context_xml
    return f"{context_xml}\n{packed}"


def build_group_dialogue_context(
    *,
    event_context: ChatInterEventContext,
    speaker_profile: PersonProfile | None,
    addressee: AddresseeResult | None,
    thread: ThreadContext | None,
    relevant_people: tuple[RelevantPerson, ...] = (),
) -> str:
    lines: list[str] = []
    lines.extend(_event_lines(event_context))
    if speaker_profile is not None:
        lines.extend(_turn_identity_lines(event_context, speaker_profile))
    if speaker_profile is not None:
        lines.append("<speaker_profile>")
        lines.extend(format_profile_lines(speaker_profile, prefix="speaker"))
        lines.append("</speaker_profile>")
    if relevant_people:
        lines.extend(_relevant_people_lines(relevant_people))
    if addressee is not None:
        lines.extend(format_addressee_xml(addressee))
    if thread is not None:
        lines.extend(format_thread_xml(thread))
    if not lines:
        return ""
    return "\n".join(lines)


def _event_lines(event_context: ChatInterEventContext) -> list[str]:
    lines = ["<event_context>"]
    lines.append(f"adapter={_xml_escape(event_context.adapter)}")
    lines.append(f"chat_type={'private' if event_context.is_private else 'group'}")
    lines.append(f"user_id={_xml_escape(event_context.user_id)}")
    if event_context.group_id:
        lines.append(f"group_id={_xml_escape(event_context.group_id)}")
    if event_context.bot_id:
        lines.append(f"bot_id={_xml_escape(event_context.bot_id)}")
    if event_context.event_id:
        lines.append(f"event_id={_xml_escape(event_context.event_id)}")
    lines.append(f"is_to_me={int(event_context.is_to_me)}")
    if event_context.mentions:
        lines.append(
            "mentions="
            + ",".join(_xml_escape(item.user_id) for item in event_context.mentions)
        )
    if event_context.reply:
        if event_context.reply.message_id:
            lines.append(
                f"reply_message_id={_xml_escape(event_context.reply.message_id)}"
            )
        if event_context.reply.sender_id:
            lines.append(
                f"reply_sender_id={_xml_escape(event_context.reply.sender_id)}"
            )
    if event_context.images:
        lines.append(f"image_count={len(event_context.images)}")
    lines.append("</event_context>")
    return lines


def _turn_identity_lines(
    event_context: ChatInterEventContext,
    speaker_profile: PersonProfile,
) -> list[str]:
    lines = ["<turn_identity>"]
    lines.append(f"current_speaker_user_id={_xml_escape(speaker_profile.user_id)}")
    lines.append(
        f"current_speaker_display_name={_xml_escape(speaker_profile.display_name)}"
    )
    if speaker_profile.group_card:
        lines.append(
            f"current_speaker_group_card={_xml_escape(speaker_profile.group_card)}"
        )
    if speaker_profile.nickname:
        lines.append(
            f"current_speaker_nickname={_xml_escape(speaker_profile.nickname)}"
        )
    if speaker_profile.aliases:
        lines.append(
            "current_speaker_aliases="
            + _xml_escape("、".join(speaker_profile.aliases[:6]))
        )
    if event_context.group_id:
        lines.append(f"current_group_id={_xml_escape(event_context.group_id)}")
    lines.append(
        "identity_rule=称呼当前说话人时，只能使用当前 speaker 的"
        " display_name/明确自称；不要把其他群友的昵称或别名套给当前说话人。"
    )
    lines.append("</turn_identity>")
    return lines


def _relevant_people_lines(people: tuple[RelevantPerson, ...]) -> list[str]:
    lines = ["<relevant_people>"]
    for index, person in enumerate(people[:8], start=1):
        profile = person.profile
        fields = [
            f"index={index}",
            f"user_id={_xml_escape(profile.user_id)}",
            f"display_name={_xml_escape(profile.display_name)}",
            f"reason={_xml_escape(person.reason)}",
            f"confidence={person.confidence:.2f}",
            f"is_current_speaker={int(person.is_current_speaker)}",
        ]
        if profile.group_card:
            fields.append(f"group_card={_xml_escape(profile.group_card)}")
        if profile.nickname:
            fields.append(f"nickname={_xml_escape(profile.nickname)}")
        if profile.aliases:
            fields.append(f"aliases={_xml_escape('、'.join(profile.aliases[:6]))}")
        if person.matched_alias:
            fields.append(f"matched_alias={_xml_escape(person.matched_alias)}")
        if profile.conflict_state:
            fields.append(f"conflict_state={_xml_escape(profile.conflict_state)}")
        lines.append("; ".join(fields))
    lines.append("</relevant_people>")
    return lines


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


__all__ = [
    "DialogueContextPack",
    "append_group_dialogue_context",
    "build_group_dialogue_context",
]
