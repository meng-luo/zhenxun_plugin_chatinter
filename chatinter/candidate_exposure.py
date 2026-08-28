"""Turn-scoped authorization for model-visible plugin candidates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .route_text import normalize_message_text


@dataclass(frozen=True, slots=True)
class CandidateExposureKey:
    source: str
    skill: str
    revision: str

    @classmethod
    def build(cls, *, source: str, skill: str, revision: str) -> CandidateExposureKey:
        return cls(
            source=normalize_message_text(source).casefold(),
            skill=normalize_message_text(skill).casefold(),
            revision=normalize_message_text(revision),
        )


@dataclass(slots=True)
class CandidateExposureLedger:
    """Authorize only candidate identities actually shown during one agent turn."""

    _exposed: dict[CandidateExposureKey, set[str]] = field(default_factory=dict)
    _pending: dict[CandidateExposureKey, set[str]] = field(default_factory=dict)
    _exact_identity_ids: set[str] = field(default_factory=set)
    _strict_identity_modes: dict[str, str] = field(default_factory=dict)
    _selected_skill: str = ""
    _discovery_source: str = ""
    _retrieval_query_count: int = 0
    _candidate_count: int = 0
    _candidate_displayed: int = 0
    _candidate_omitted: int = 0
    _selected_command_id: str = ""
    _selected_capability_id: str = ""
    _execution_validation_reason: str = ""

    def expose(
        self,
        key: CandidateExposureKey,
        identities: Iterable[object],
        *,
        discovery_source: str,
        exact_identity: bool = False,
        pending: bool = False,
        strict_identity_mode: str = "",
    ) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(
                identity
                for value in identities
                if (identity := normalize_message_text(str(value or "")))
            )
        )
        if normalized:
            target = self._pending if pending else self._exposed
            target.setdefault(key, set()).update(normalized)
            if exact_identity:
                self._exact_identity_ids.update(normalized)
                mode = normalize_message_text(strict_identity_mode)
                if mode:
                    self._strict_identity_modes.update(
                        {identity: mode for identity in normalized}
                    )
        self._selected_skill = key.skill
        self._discovery_source = normalize_message_text(discovery_source)
        return normalized

    def commit_pending(self) -> int:
        committed = 0
        for key, identities in self._pending.items():
            before = len(self._exposed.get(key, set()))
            self._exposed.setdefault(key, set()).update(identities)
            committed += len(self._exposed[key]) - before
        self._pending.clear()
        return committed

    def note_exact_identities(self, identities: Iterable[object]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(
                identity
                for value in identities
                if (identity := normalize_message_text(str(value or "")))
            )
        )
        self._exact_identity_ids.update(normalized)
        return normalized

    def discard_pending(self) -> None:
        self._pending.clear()

    def is_exposed(self, key: CandidateExposureKey, identity: object) -> bool:
        normalized = normalize_message_text(str(identity or ""))
        return bool(normalized and normalized in self._exposed.get(key, set()))

    def record_discovery(
        self,
        key: CandidateExposureKey,
        *,
        source: str,
        query_count: int,
        candidate_count: int,
        displayed_count: int,
        omitted_count: int,
    ) -> None:
        self._selected_skill = key.skill
        self._discovery_source = normalize_message_text(source)
        self._retrieval_query_count = max(int(query_count), 0)
        self._candidate_count = max(int(candidate_count), 0)
        self._candidate_displayed = max(int(displayed_count), 0)
        self._candidate_omitted = max(int(omitted_count), 0)

    def record_discovery_summary(
        self,
        *,
        skill: str,
        source: str,
        query_count: int,
        candidate_count: int,
        displayed_count: int,
        omitted_count: int,
    ) -> None:
        self._selected_skill = normalize_message_text(skill).casefold()
        self._discovery_source = normalize_message_text(source)
        self._retrieval_query_count = max(int(query_count), 0)
        self._candidate_count = max(int(candidate_count), 0)
        self._candidate_displayed = max(int(displayed_count), 0)
        self._candidate_omitted = max(int(omitted_count), 0)

    def record_execution(
        self,
        key: CandidateExposureKey,
        identity: object,
        *,
        valid: bool,
        reason: str,
    ) -> None:
        normalized = normalize_message_text(str(identity or ""))
        self._selected_skill = key.skill
        if key.source == "gscore":
            self._selected_capability_id = normalized
        else:
            self._selected_command_id = normalized
        self._execution_validation_reason = normalize_message_text(reason) or (
            "candidate_exposed" if valid else "candidate_identity_not_exposed"
        )

    @property
    def exposure_count(self) -> int:
        return sum(len(values) for values in self._exposed.values())

    @property
    def exposed_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    identity
                    for identities in self._exposed.values()
                    for identity in identities
                }
            )
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "exact_identity_ids": tuple(sorted(self._exact_identity_ids)),
            "strict_identity_match_modes": tuple(
                f"{identity}={mode}"
                for identity, mode in sorted(self._strict_identity_modes.items())
            ),
            "exposed_command_ids": self.exposed_ids,
            "selected_skill": self._selected_skill,
            "discovery_source": self._discovery_source,
            "retrieval_query_count": self._retrieval_query_count,
            "candidate_count": self._candidate_count,
            "candidate_displayed": self._candidate_displayed,
            "candidate_omitted": self._candidate_omitted,
            "candidate_exposure_count": self.exposure_count,
            "selected_command_id": self._selected_command_id,
            "selected_capability_id": self._selected_capability_id,
            "execution_validation_reason": self._execution_validation_reason,
        }


__all__ = ["CandidateExposureKey", "CandidateExposureLedger"]
