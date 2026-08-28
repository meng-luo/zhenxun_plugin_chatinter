from __future__ import annotations

from .addressee_resolver import AddresseeResult, format_addressee_xml
from .event_context import ChatInterEventContext
from .person_candidates import TurnPersonCandidateLedger
from .person_registry import PersonProfile, RelevantPerson
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
        person_candidate_ledger: TurnPersonCandidateLedger | None = None,
    ) -> None:
        self.event_context = event_context
        self.speaker_profile = speaker_profile
        self.addressee = addressee
        self.thread = thread
        self.relevant_people = relevant_people
        self.person_candidate_ledger = person_candidate_ledger
        visible_people = _visible_relevant_people(speaker_profile, relevant_people)
        if person_candidate_ledger is not None:
            person_candidate_ledger.bind_visible_people(
                visible_people,
                speaker_profile=speaker_profile,
            )

    def to_context_xml(self) -> str:
        return build_group_dialogue_context(
            event_context=self.event_context,
            speaker_profile=self.speaker_profile,
            addressee=self.addressee,
            thread=self.thread,
            relevant_people=self.relevant_people,
            target_refs=self.action_target_refs(),
        )

    def action_target_refs(self) -> dict[str, str]:
        if self.person_candidate_ledger is None:
            return {}
        return self.person_candidate_ledger.refs()


def append_group_dialogue_context(
    context_xml: str,
    *,
    event_context: ChatInterEventContext,
    speaker_profile: PersonProfile | None,
    addressee: AddresseeResult | None,
    thread: ThreadContext | None,
    relevant_people: tuple[RelevantPerson, ...] = (),
    target_refs: dict[str, str] | None = None,
) -> str:
    packed = build_group_dialogue_context(
        event_context=event_context,
        speaker_profile=speaker_profile,
        addressee=addressee,
        thread=thread,
        relevant_people=relevant_people,
        target_refs=target_refs,
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
    target_refs: dict[str, str] | None = None,
) -> str:
    lines: list[str] = []
    lines.extend(_event_lines(event_context))
    if speaker_profile is not None:
        lines.extend(
            _turn_identity_lines(
                event_context,
                speaker_profile,
                current_speaker_target_ref=_target_ref_for_user(
                    target_refs,
                    speaker_profile.user_id,
                ),
            )
        )
    visible_people = _visible_relevant_people(speaker_profile, relevant_people)
    if visible_people:
        lines.extend(
            _relevant_people_lines(
                visible_people,
                target_refs=target_refs,
            )
        )
    if addressee is not None:
        lines.extend(format_addressee_xml(addressee))
    if thread is not None:
        lines.extend(format_thread_xml(thread))
    if not lines:
        return ""
    return "\n".join(lines)


def _visible_relevant_people(
    speaker_profile: PersonProfile | None,
    relevant_people: tuple[RelevantPerson, ...],
) -> tuple[RelevantPerson, ...]:
    speaker_id = speaker_profile.user_id if speaker_profile is not None else ""
    return tuple(
        person
        for person in relevant_people
        if not person.is_current_speaker
        and (not speaker_id or person.profile.user_id != speaker_id)
    )


def _event_lines(event_context: ChatInterEventContext) -> list[str]:
    lines = ["<event_context>"]
    lines.append(f"adapter={_xml_escape(event_context.adapter)}")
    lines.append(f"chat_type={'private' if event_context.is_private else 'group'}")
    lines.append(f"user_id={_xml_escape(event_context.user_id)}")
    if event_context.group_id:
        lines.append(f"group_id={_xml_escape(event_context.group_id)}")
    if event_context.bot_id:
        lines.append(f"bot_id={_xml_escape(event_context.bot_id)}")
    lines.append(f"is_to_me={int(event_context.is_to_me)}")
    if event_context.mentions:
        lines.append(
            "mentions="
            + ",".join(_xml_escape(item.user_id) for item in event_context.mentions)
        )
    if event_context.reply:
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
    *,
    current_speaker_target_ref: str = "",
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
    if current_speaker_target_ref:
        lines.append(
            "current_speaker_target_ref="
            + _xml_escape(current_speaker_target_ref)
        )
    if event_context.group_id:
        lines.append(f"current_group_id={_xml_escape(event_context.group_id)}")
    lines.append("</turn_identity>")
    return lines


def _target_ref_for_user(
    target_refs: dict[str, str] | None,
    user_id: str,
) -> str:
    normalized = str(user_id or "").strip()
    return next(
        (
            target_ref
            for target_ref, candidate_user_id in (target_refs or {}).items()
            if candidate_user_id == normalized
        ),
        "",
    )


def _relevant_people_lines(
    people: tuple[RelevantPerson, ...],
    *,
    target_refs: dict[str, str] | None = None,
) -> list[str]:
    lines = ["<relevant_people>"]
    refs_by_user_id = {
        user_id: target_ref for target_ref, user_id in (target_refs or {}).items()
    }
    for index, person in enumerate(people[:8], start=1):
        profile = person.profile
        fields = [
            f"index={index}",
            f"display_name={_xml_escape(profile.display_name)}",
            f"is_current_speaker={int(person.is_current_speaker)}",
        ]
        target_ref = refs_by_user_id.get(profile.user_id)
        if target_ref:
            fields.insert(1, f"target_ref={_xml_escape(target_ref)}")
            fields.append("target_candidate=1")
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
