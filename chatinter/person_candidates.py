"""Turn-scoped group member discovery for model-grounded target selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import re

from .member_similarity import (
    MemberAliasEntry,
    build_member_alias_entries,
    normalize_member_alias,
    score_member_alias,
)
from .person_registry import PersonProfile, list_group_person_profiles
from .route_text import normalize_message_text

_SCRIPT_RUN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,31}|[\u4e00-\u9fff]{2,24}")
_MAX_IDENTITY_SPANS = 48
_MAX_TARGET_CANDIDATES = 4
_MIN_RECALL_SCORE = 0.82


@dataclass(frozen=True, slots=True)
class PersonCandidateEvidence:
    source: str
    score: float
    span: str = ""
    matched_alias: str = ""


@dataclass(frozen=True, slots=True)
class PersonCandidate:
    profile: PersonProfile
    score: float
    evidence: tuple[PersonCandidateEvidence, ...]
    matched_span: str = ""
    matched_alias: str = ""

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(item.source for item in self.evidence if item.source)
        )


@dataclass(frozen=True, slots=True)
class PersonCandidateSet:
    identity_spans: tuple[str, ...] = ()
    candidates: tuple[PersonCandidate, ...] = ()
    ambiguous: bool = False
    authorization_reason: str = "no_candidate"


@dataclass(slots=True)
class TurnPersonCandidateLedger:
    """Real member references exposed during one mixed-chat turn only."""

    candidate_set: PersonCandidateSet = field(default_factory=PersonCandidateSet)
    _refs: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _selected_ref: str = field(default="", init=False, repr=False)
    _validation_reason: str = field(default="", init=False, repr=False)

    @property
    def candidate_user_ids(self) -> tuple[str, ...]:
        if self.candidate_set.authorization_reason != "candidate_exposed":
            return ()
        return tuple(item.profile.user_id for item in self.candidate_set.candidates)

    def bind_visible_people(
        self,
        people: Sequence[object],
        *,
        speaker_profile: PersonProfile | None = None,
    ) -> dict[str, str]:
        available = {
            str(getattr(getattr(person, "profile", None), "user_id", "") or "").strip()
            for person in people
        }
        if speaker_profile is not None:
            available.add(str(speaker_profile.user_id or "").strip())
        refs: dict[str, str] = {}
        for candidate in self.candidate_set.candidates:
            user_id = str(candidate.profile.user_id or "").strip()
            if (
                self.candidate_set.authorization_reason == "candidate_exposed"
                and user_id
                and user_id in available
            ):
                refs[f"person:{len(refs) + 1}"] = user_id
        self._refs = refs
        return dict(refs)

    def ref_for_user(self, user_id: str) -> str:
        normalized = str(user_id or "").strip()
        return next(
            (
                target_ref
                for target_ref, candidate_user_id in self._refs.items()
                if candidate_user_id == normalized
            ),
            "",
        )

    def refs(self) -> dict[str, str]:
        return dict(self._refs)

    def trusted_identity_spans(self) -> tuple[str, ...]:
        """Return lexical spans backed by visible, turn-scoped member refs."""

        if self.candidate_set.authorization_reason != "candidate_exposed":
            return ()
        bound_user_ids = set(self._refs.values())
        return tuple(
            dict.fromkeys(
                span
                for candidate in self.candidate_set.candidates
                if candidate.profile.user_id in bound_user_ids
                if (span := normalize_message_text(candidate.matched_span))
            )
        )

    def validate(self, target_ref: str) -> str | None:
        resolved = self.validate_many((target_ref,))
        return resolved[0] if resolved else None

    def validate_many(self, target_refs: Sequence[str]) -> tuple[str, ...] | None:
        """Resolve a set of turn references atomically."""

        normalized_refs = tuple(
            normalize_message_text(target_ref).casefold()
            for target_ref in target_refs
            if normalize_message_text(target_ref)
        )
        resolved = tuple(
            self._refs.get(target_ref, "") for target_ref in normalized_refs
        )
        if not normalized_refs or any(not user_id for user_id in resolved):
            self._validation_reason = "unknown_target_ref"
            return None
        self._selected_ref = "|".join(normalized_refs)
        self._validation_reason = "candidate_exposed"
        return resolved

    def note_validation(self, reason: str) -> None:
        self._validation_reason = normalize_message_text(reason)

    def snapshot(self) -> dict[str, str | float]:
        return {
            "identity_spans": "|".join(self.candidate_set.identity_spans),
            "person_candidate_count": float(len(self.candidate_set.candidates)),
            "candidate_sources": "|".join(
                dict.fromkeys(
                    source
                    for candidate in self.candidate_set.candidates
                    for source in candidate.sources
                )
            ),
            "self_identity_candidate": float(
                any(
                    source.startswith("self_")
                    for candidate in self.candidate_set.candidates
                    for source in candidate.sources
                )
            ),
            "selected_target_ref": self._selected_ref,
            "target_resolution_mode": self.candidate_set.authorization_reason,
            "target_validation_reason": self._validation_reason,
        }


def extract_identity_spans(message_text: str) -> tuple[str, ...]:
    """Extract bounded script-consistent spans without action-word heuristics."""

    normalized = normalize_message_text(message_text)
    spans: list[str] = []
    for raw_run in _SCRIPT_RUN_PATTERN.findall(normalized):
        run = normalize_member_alias(raw_run)
        if not run:
            continue
        if run.isascii():
            spans.append(run)
            continue
        maximum = min(len(run), 8)
        for size in range(maximum, 1, -1):
            for start in range(0, len(run) - size + 1):
                spans.append(run[start : start + size])
                if len(spans) >= _MAX_IDENTITY_SPANS:
                    return tuple(dict.fromkeys(spans))
    return tuple(dict.fromkeys(spans))[:_MAX_IDENTITY_SPANS]


async def retrieve_person_candidates(
    *,
    group_id: str | None,
    message_text: str,
    roster_profiles: Sequence[Mapping[str, object]],
    current_user_id: str,
    bot_id: str | None,
    mention_user_ids: Sequence[str] = (),
    reply_sender_id: str | None = None,
    thread_user_ids: Sequence[str] = (),
    recent_user_ids: Sequence[str] = (),
    current_speaker_profile: PersonProfile | None = None,
    limit: int = _MAX_TARGET_CANDIDATES,
) -> PersonCandidateSet:
    if not group_id:
        return PersonCandidateSet()

    current_id = str(current_user_id or "").strip()
    excluded = {str(bot_id or "").strip()}
    roster = {
        str(item.get("user_id") or "").strip(): item
        for item in roster_profiles
        if str(item.get("user_id") or "").strip() not in excluded
    }
    if (
        current_speaker_profile is not None
        and current_id
        and current_id not in excluded
        and current_speaker_profile.user_id == current_id
        and current_speaker_profile.group_id == group_id
        and current_id not in roster
    ):
        roster[current_id] = _speaker_roster_profile(current_speaker_profile)
    if not roster:
        return PersonCandidateSet()

    stored_profiles = {
        profile.user_id: profile
        for profile in await list_group_person_profiles(group_id)
        if profile.user_id in roster
    }
    profiles = {
        user_id: _merge_roster_profile(
            group_id=group_id,
            roster_profile=roster_profile,
            stored_profile=stored_profiles.get(user_id),
        )
        for user_id, roster_profile in roster.items()
    }
    spans = extract_identity_spans(message_text)
    evidence_by_user: dict[str, list[PersonCandidateEvidence]] = {}

    def add(user_id: str, evidence: PersonCandidateEvidence) -> None:
        if user_id in profiles:
            evidence_by_user.setdefault(user_id, []).append(evidence)

    for user_id in dict.fromkeys(str(item or "").strip() for item in mention_user_ids):
        if user_id != current_id:
            add(user_id, PersonCandidateEvidence("mention", 1.0))
    reply_id = str(reply_sender_id or "").strip()
    if reply_id and reply_id != current_id:
        add(reply_id, PersonCandidateEvidence("reply", 0.91))

    recent_rank = {
        user_id: index
        for index, user_id in enumerate(
            dict.fromkeys(str(item or "").strip() for item in recent_user_ids)
        )
        if user_id in profiles
    }
    thread_ids = {
        str(item or "").strip()
        for item in thread_user_ids
        if str(item or "").strip() in profiles
    }

    for user_id, profile in profiles.items():
        best = _best_lexical_evidence(spans, profile)
        if best is not None:
            if user_id == current_id:
                best = PersonCandidateEvidence(
                    source=f"self_{best.source}",
                    score=best.score,
                    span=best.span,
                    matched_alias=best.matched_alias,
                )
            add(user_id, best)
        if user_id != current_id and user_id in thread_ids:
            add(user_id, PersonCandidateEvidence("thread_history", 0.68))
        if user_id != current_id and user_id in recent_rank:
            add(
                user_id,
                PersonCandidateEvidence(
                    "recent_history",
                    max(0.58, 0.66 - recent_rank[user_id] * 0.02),
                ),
            )

    ranked: list[PersonCandidate] = []
    for user_id, evidence in evidence_by_user.items():
        ordered = tuple(sorted(evidence, key=lambda item: item.score, reverse=True))
        lexical = next((item for item in ordered if item.span), None)
        combined = ordered[0].score + min(0.08, 0.02 * (len(ordered) - 1))
        ranked.append(
            PersonCandidate(
                profile=profiles[user_id],
                score=min(combined, 1.0),
                evidence=ordered,
                matched_span=lexical.span if lexical else "",
                matched_alias=lexical.matched_alias if lexical else "",
            )
        )
    ranked.sort(
        key=lambda item: (
            item.score,
            max((len(e.span) for e in item.evidence), default=0),
            item.profile.confidence,
            item.profile.user_id,
        ),
        reverse=True,
    )

    max_items = min(max(int(limit or 0), 0), _MAX_TARGET_CANDIDATES)
    lexical_ranked = [
        item
        for item in ranked
        if any(evidence.span for evidence in item.evidence)
    ]
    ambiguous = (
        len(lexical_ranked) > max_items
        and max_items > 0
        and lexical_ranked[max_items - 1].score
        - lexical_ranked[max_items].score
        < 0.04
    )
    if not max_items or ambiguous:
        return PersonCandidateSet(
            identity_spans=spans,
            candidates=tuple(ranked[:max_items]),
            ambiguous=ambiguous,
            authorization_reason=(
                "candidate_overflow" if ambiguous else "no_candidate"
            ),
        )
    candidates = tuple(ranked[:max_items])
    return PersonCandidateSet(
        identity_spans=spans,
        candidates=candidates,
        authorization_reason=("candidate_exposed" if candidates else "no_candidate"),
    )


def _best_lexical_evidence(
    spans: Iterable[str],
    profile: PersonProfile,
) -> PersonCandidateEvidence | None:
    aliases = _profile_alias_entries(profile)
    best: PersonCandidateEvidence | None = None
    for span in spans:
        for entry in aliases:
            score = score_member_alias(span, entry)
            source = _lexical_source(span, entry, score)
            if source is None:
                continue
            candidate = PersonCandidateEvidence(
                source=source,
                score=score,
                span=span,
                matched_alias=entry.source or entry.value,
            )
            if best is None or (candidate.score, len(candidate.span)) > (
                best.score,
                len(best.span),
            ):
                best = candidate
    return best


def _lexical_source(
    span: str,
    entry: MemberAliasEntry,
    score: float,
) -> str | None:
    normalized_span = normalize_member_alias(span)
    if not normalized_span or score < _MIN_RECALL_SCORE:
        return None
    if normalized_span == entry.value and entry.kind not in {"prefix", "suffix"}:
        return "exact_alias"
    if normalized_span in entry.value or entry.value in normalized_span:
        return "fragment_alias"
    return "similar_alias"


def _profile_alias_entries(profile: PersonProfile) -> tuple[MemberAliasEntry, ...]:
    return build_member_alias_entries(
        profile.display_name,
        profile.group_card,
        profile.nickname,
        *profile.aliases,
    )


def _merge_roster_profile(
    *,
    group_id: str,
    roster_profile: Mapping[str, object],
    stored_profile: PersonProfile | None,
) -> PersonProfile:
    user_id = str(roster_profile.get("user_id") or "").strip()
    display_name = normalize_message_text(
        str(roster_profile.get("display_name") or "")
    )
    nickname = normalize_message_text(str(roster_profile.get("nickname") or ""))
    group_card = normalize_message_text(
        str(roster_profile.get("user_name") or display_name or "")
    )
    aliases = tuple(
        dict.fromkeys(
            (
                *(stored_profile.aliases if stored_profile else ()),
                *(
                    item
                    for item in (
                        stored_profile.nickname if stored_profile else "",
                        stored_profile.group_card if stored_profile else "",
                    )
                    if item
                ),
                *(item for item in (display_name, nickname, group_card) if item),
            )
        )
    )[:12]
    return PersonProfile(
        user_id=user_id,
        group_id=group_id,
        nickname=nickname or (stored_profile.nickname if stored_profile else ""),
        group_card=group_card or (stored_profile.group_card if stored_profile else ""),
        aliases=aliases,
        alias_weights=stored_profile.alias_weights if stored_profile else (),
        alias_sources=stored_profile.alias_sources if stored_profile else (),
        known_facts=stored_profile.known_facts if stored_profile else (),
        relationship=stored_profile.relationship if stored_profile else "",
        conflict_state=stored_profile.conflict_state if stored_profile else "",
        confidence=max(stored_profile.confidence if stored_profile else 0.0, 0.75),
        last_seen=stored_profile.last_seen if stored_profile else None,
    )


def _speaker_roster_profile(profile: PersonProfile) -> dict[str, object]:
    return {
        "user_id": profile.user_id,
        "display_name": profile.display_name,
        "nickname": profile.nickname,
        "user_name": profile.group_card or profile.nickname,
        "membership_source": "current_event",
    }


__all__ = [
    "PersonCandidate",
    "PersonCandidateEvidence",
    "PersonCandidateSet",
    "TurnPersonCandidateLedger",
    "extract_identity_spans",
    "retrieve_person_candidates",
]
