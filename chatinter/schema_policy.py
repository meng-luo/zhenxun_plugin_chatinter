from dataclasses import dataclass, field

from .plugin_adapters import AdapterTargetPolicy
from .route_text import normalize_message_text

_SELF_ONLY_SCOPE = "self_only"
_AT_SOURCE = "at"


@dataclass(frozen=True)
class CommandTargetPolicy:
    actor_scope: str
    target_requirement: str
    target_sources: frozenset[str]
    allow_at: bool
    adapter_policy: AdapterTargetPolicy = field(default_factory=AdapterTargetPolicy)

    @property
    def media_related(self) -> bool:
        return self.adapter_policy.media_related

    @property
    def context_hints(self) -> tuple[str, ...]:
        return self.adapter_policy.context_hints

    @property
    def allow_at_as_target(self) -> bool:
        return self.allow_at and self.adapter_policy.allow_at_as_target

    @property
    def allow_image_as_target(self) -> bool:
        return self.adapter_policy.allow_image_as_target

    @property
    def allow_reply_image_as_target(self) -> bool:
        return self.adapter_policy.allow_reply_image_as_target

    @property
    def require_target_for_third_person(self) -> bool:
        return self.adapter_policy.require_target_for_third_person

    @property
    def target_missing_message(self) -> str:
        return self.adapter_policy.target_missing_message


def resolve_command_target_policy(
    schema,
    *,
    adapter_policy: AdapterTargetPolicy | None = None,
) -> CommandTargetPolicy:
    adapter = adapter_policy or AdapterTargetPolicy()
    actor_scope = normalize_message_text(
        str(getattr(schema, "actor_scope", "") or "")
    ).lower()
    target_source_values: set[str] = set()
    for item in getattr(schema, "target_sources", None) or []:
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
        adapter = AdapterTargetPolicy(
            family=adapter.family,
            context_hints=adapter.context_hints,
            media_related=adapter.media_related,
            allow_image_as_target=adapter.allow_image_as_target,
            allow_reply_image_as_target=adapter.allow_reply_image_as_target,
            target_missing_message=adapter.target_missing_message,
        )
    return CommandTargetPolicy(
        actor_scope=actor_scope,
        target_requirement=target_requirement,
        target_sources=target_sources,
        allow_at=allow_at,
        adapter_policy=adapter,
    )


def schema_is_self_only(schema) -> bool:
    return resolve_command_target_policy(schema).actor_scope == _SELF_ONLY_SCOPE
