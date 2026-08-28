from dataclasses import dataclass, field

from .route_text import normalize_message_text
from .target_policy import TargetPolicy

_SELF_ONLY_SCOPE = "self_only"
_AT_SOURCE = "at"
@dataclass(frozen=True)
class CommandTargetPolicy:
    actor_scope: str
    target_requirement: str
    target_sources: frozenset[str]
    allow_at: bool
    allow_image: bool = False
    allow_reply_image: bool = False
    media_related_value: bool = False
    adapter_policy: TargetPolicy = field(default_factory=TargetPolicy)

    @property
    def media_related(self) -> bool:
        return bool(self.media_related_value or self.adapter_policy.media_related)

    @property
    def context_hints(self) -> tuple[str, ...]:
        return self.adapter_policy.context_hints

    @property
    def allow_at_as_target(self) -> bool:
        return bool(self.allow_at)

    @property
    def allow_image_as_target(self) -> bool:
        return bool(self.allow_image or self.adapter_policy.allow_image_as_target)

    @property
    def allow_reply_image_as_target(self) -> bool:
        return bool(
            self.allow_reply_image or self.adapter_policy.allow_reply_image_as_target
        )

    @property
    def require_target_for_third_person(self) -> bool:
        return self.adapter_policy.require_target_for_third_person

def resolve_command_target_policy(
    schema,
    *,
    adapter_policy: TargetPolicy | None = None,
) -> CommandTargetPolicy:
    adapter = adapter_policy or TargetPolicy()
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
    try:
        image_min = max(int(getattr(schema, "image_min", 0) or 0), 0)
    except Exception:
        image_min = 0
    has_image_max = hasattr(schema, "image_max")
    image_max_raw = getattr(schema, "image_max", 0)
    try:
        image_max = int(image_max_raw) if image_max_raw is not None else None
    except Exception:
        image_max = None
    allow_at_raw = getattr(schema, "allow_at", None)
    requires = dict(getattr(schema, "requires", {}) or {})
    payload_policy = normalize_message_text(
        str(getattr(schema, "payload_policy", "") or "")
    ).lower()
    accepts_image_target = (
        image_min > 0
        or (has_image_max and image_max is None)
        or (image_max is not None and image_max > 0)
        or bool(requires.get("image"))
        or payload_policy in {"image_only", "text_or_image"}
    )
    slot_types = {
        normalize_message_text(str(getattr(slot, "type", "") or "")).lower()
        for slot in (getattr(schema, "slots", None) or [])
    }
    accepts_image_target = accepts_image_target or "image" in slot_types
    has_target_slot = "at" in slot_types
    if has_target_slot and target_requirement == "none":
        if any(
            bool(getattr(slot, "required", False))
            and normalize_message_text(str(getattr(slot, "type", "") or "")).lower()
            == "at"
            for slot in (getattr(schema, "slots", None) or [])
        ):
            target_requirement = "required"
        else:
            target_requirement = "optional"
    allow_at = allow_at_raw is True or (
        allow_at_raw is not False
        and (
            _AT_SOURCE in target_sources
            or "nickname" in target_sources
            or "reply" in target_sources
            or has_target_slot
            or target_requirement != "none"
        )
    )
    if not accepts_image_target:
        target_sources = frozenset(
            source
            for source in target_sources
            if allow_at or source not in {_AT_SOURCE, "reply", "nickname", "self"}
        )
        if target_requirement != "required":
            target_requirement = "optional" if allow_at else "none"
        adapter = _disable_media_target_adapter(adapter)
    if actor_scope == _SELF_ONLY_SCOPE:
        allow_at = False
        target_requirement = "none"
        adapter = TargetPolicy(
            family=adapter.family,
            context_hints=adapter.context_hints,
            media_related=adapter.media_related,
            allow_image_as_target=adapter.allow_image_as_target,
            allow_reply_image_as_target=adapter.allow_reply_image_as_target,
        )
    return CommandTargetPolicy(
        actor_scope=actor_scope,
        target_requirement=target_requirement,
        target_sources=target_sources,
        allow_at=allow_at,
        allow_image=accepts_image_target,
        allow_reply_image=accepts_image_target,
        media_related_value=accepts_image_target,
        adapter_policy=adapter,
    )


def schema_is_self_only(schema) -> bool:
    return resolve_command_target_policy(schema).actor_scope == _SELF_ONLY_SCOPE


def _disable_media_target_adapter(adapter: TargetPolicy) -> TargetPolicy:
    """Keep adapter metadata but disable media target rules for text commands."""

    return TargetPolicy(
        family=adapter.family,
        context_hints=adapter.context_hints,
        media_related=adapter.media_related,
        allow_at_as_target=False,
        allow_image_as_target=False,
        allow_reply_image_as_target=False,
        require_target_for_third_person=False,
    )
