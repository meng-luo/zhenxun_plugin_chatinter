"""Strict, syntax-only command identity matching for mixed chat."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from .command_index import CommandCandidate
from .meta_tools import _candidate_from_snapshot
from .models.pydantic_models import CommandToolSnapshot
from .route_text import normalize_message_text, strip_bot_name_prefix

_BOUNDARY_CHARS = frozenset(" \t\r\n,，.。!！?？:：;；/|、()（）[]【】{}<>《》")


@dataclass(frozen=True, slots=True)
class _StrictIdentityMatch:
    candidate: CommandCandidate
    identity: str
    is_primary: bool


def resolve_strict_command_candidates(
    text: str,
    snapshots: Iterable[CommandToolSnapshot],
    *,
    trusted_person_spans: Iterable[object] = (),
) -> tuple[CommandCandidate, ...]:
    task = strip_bot_name_prefix(text)
    if not task:
        return ()
    normalized_person_spans = frozenset(
        normalized.casefold()
        for value in trusted_person_spans
        if (normalized := normalize_message_text(str(value or "")))
    )
    matched: list[_StrictIdentityMatch] = []
    seen: set[str] = set()
    for snapshot in snapshots:
        identity_match = _best_snapshot_identity_match(
            task,
            snapshot=snapshot,
            trusted_person_spans=normalized_person_spans,
        )
        if identity_match is None:
            continue
        identity, is_primary, match_mode = identity_match
        command_id = normalize_message_text(snapshot.command_id)
        if not command_id or command_id in seen:
            continue
        seen.add(command_id)
        matched.append(
            _StrictIdentityMatch(
                candidate=replace(
                    _candidate_from_snapshot(snapshot),
                    strict_identity_mode=match_mode,
                ),
                identity=identity,
                is_primary=is_primary,
            )
        )
    if not matched:
        return ()

    longest_identity = max(len(item.identity) for item in matched)
    matched = [item for item in matched if len(item.identity) == longest_identity]
    primary_identity_keys = {
        (
            normalize_message_text(item.candidate.plugin_module).casefold(),
            item.identity.casefold(),
        )
        for item in matched
        if item.is_primary
    }
    resolved = [
        item.candidate
        for item in matched
        if item.is_primary
        or (
            normalize_message_text(item.candidate.plugin_module).casefold(),
            item.identity.casefold(),
        )
        not in primary_identity_keys
    ]
    return tuple(
        sorted(
            resolved,
            key=lambda item: (
                normalize_message_text(item.plugin_module).casefold(),
                normalize_message_text(item.schema.command_id).casefold(),
            ),
        )
    )


def _best_snapshot_identity_match(
    text: str,
    *,
    snapshot: CommandToolSnapshot,
    trusted_person_spans: frozenset[str],
) -> tuple[str, bool, str] | None:
    matches: list[tuple[str, bool, str]] = []
    for index, value in enumerate((snapshot.head, *snapshot.aliases)):
        identity = normalize_message_text(str(value or ""))
        if not identity:
            continue
        mode = _literal_identity_at_start(
            text,
            identity,
            snapshot=snapshot,
            trusted_person_spans=trusted_person_spans,
        )
        if mode:
            matches.append((identity, index == 0, mode))
    if not matches:
        return None
    return max(
        matches,
        key=lambda item: (
            len(item[0]),
            item[1],
            item[0].casefold(),
        ),
    )


def _literal_identity_at_start(
    text: str,
    identity: object,
    *,
    snapshot: CommandToolSnapshot,
    trusted_person_spans: frozenset[str],
) -> str:
    normalized_text = normalize_message_text(text)
    normalized_identity = normalize_message_text(str(identity or ""))
    if not normalized_text or not normalized_identity:
        return ""
    folded_text = normalized_text.casefold()
    folded_identity = normalized_identity.casefold()
    if folded_text == folded_identity:
        return "boundary"
    if not folded_text.startswith(folded_identity):
        return ""
    following = normalized_text[len(normalized_identity)]
    if following in _BOUNDARY_CHARS:
        return "boundary"
    if snapshot.allow_sticky_arg:
        return "metadata_sticky"
    tail = normalize_message_text(normalized_text[len(normalized_identity) :])
    if (
        tail.casefold() in trusted_person_spans
        and _snapshot_accepts_person_target(snapshot)
    ):
        return "person_target_tail"
    return ""


def _snapshot_accepts_person_target(snapshot: CommandToolSnapshot) -> bool:
    return bool(
        snapshot.allow_at
        or snapshot.target_requirement != "none"
        or set(snapshot.target_sources) & {"at", "reply", "nickname", "self"}
        or any(slot.type == "at" for slot in snapshot.slots)
        or snapshot.entity_scope == "target_user"
    )


__all__ = ["resolve_strict_command_candidates"]
