from dataclasses import dataclass

from .route_text import normalize_message_text

_SELF_ONLY_SCOPE = "self_only"
_AT_SOURCE = "at"


@dataclass(frozen=True)
class CommandTargetPolicy:
    actor_scope: str
    target_requirement: str
    target_sources: frozenset[str]
    allow_at: bool


def resolve_command_target_policy(schema) -> CommandTargetPolicy:
    actor_scope = normalize_message_text(
        str(getattr(schema, "actor_scope", "") or "")
    ).lower()
    target_source_values: set[str] = set()
    for item in (getattr(schema, "target_sources", None) or []):
        value = normalize_message_text(str(item or "")).lower()
        if value:
            target_source_values.add(value)
    target_sources = frozenset(target_source_values)
    target_requirement = normalize_message_text(
        str(getattr(schema, "target_requirement", "") or "")
    ).lower()
    if target_requirement not in {"none", "optional", "required"}:
        target_requirement = "none"
    allow_at_raw = getattr(schema, "allow_at", None)
    allow_at = allow_at_raw is True or (
        allow_at_raw is not False and _AT_SOURCE in target_sources
    )
    if actor_scope == _SELF_ONLY_SCOPE:
        allow_at = False
        target_requirement = "none"
    return CommandTargetPolicy(
        actor_scope=actor_scope,
        target_requirement=target_requirement,
        target_sources=target_sources,
        allow_at=allow_at,
    )


def schema_allows_at(schema) -> bool:
    return resolve_command_target_policy(schema).allow_at


def schema_is_self_only(schema) -> bool:
    return resolve_command_target_policy(schema).actor_scope == _SELF_ONLY_SCOPE


def schema_accepts_at_target(schema) -> bool:
    policy = resolve_command_target_policy(schema)
    return policy.allow_at and policy.target_requirement != "none"
