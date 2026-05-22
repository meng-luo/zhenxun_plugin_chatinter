from dataclasses import dataclass, field

from .target_policy import TargetPolicy
from .route_text import normalize_message_text

_SELF_ONLY_SCOPE = "self_only"
_AT_SOURCE = "at"
_MEDIA_TERMS = (
    "图片",
    "图",
    "照片",
    "头像",
    "表情",
    "表情包",
    "梗图",
    "image",
    "img",
    "photo",
    "avatar",
    "meme",
)
_USER_TARGET_TERMS = (
    "@",
    "at",
    "qq",
    "user",
    "member",
    "target",
    "nickname",
    "用户",
    "成员",
    "群友",
    "目标",
    "对象",
    "昵称",
)


@dataclass(frozen=True)
class CommandTargetPolicy:
    actor_scope: str
    target_requirement: str
    target_sources: frozenset[str]
    allow_at: bool
    allow_image: bool = False
    allow_reply_image: bool = False
    media_related_value: bool = False
    target_missing_message_value: str = ""
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
            self.allow_reply_image
            or self.adapter_policy.allow_reply_image_as_target
        )

    @property
    def require_target_for_third_person(self) -> bool:
        return self.adapter_policy.require_target_for_third_person

    @property
    def target_missing_message(self) -> str:
        return self.target_missing_message_value or self.adapter_policy.target_missing_message


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
    schema_text = _schema_text(schema)
    media_related = _contains_any(schema_text, _MEDIA_TERMS)
    user_target_related = _contains_any(schema_text, _USER_TARGET_TERMS)
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
        or media_related
    )
    slot_types = {
        normalize_message_text(str(getattr(slot, "type", "") or "")).lower()
        for slot in (getattr(schema, "slots", None) or [])
    }
    has_target_slot = "at" in slot_types or user_target_related
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
            target_missing_message=adapter.target_missing_message,
        )
    return CommandTargetPolicy(
        actor_scope=actor_scope,
        target_requirement=target_requirement,
        target_sources=target_sources,
        allow_at=allow_at,
        allow_image=accepts_image_target,
        allow_reply_image=accepts_image_target
        or "reply" in target_sources
        or bool(requires.get("reply")),
        media_related_value=media_related or accepts_image_target,
        target_missing_message_value=_generic_target_missing_message(
            allow_at=allow_at,
            allow_image=accepts_image_target,
        ),
        adapter_policy=adapter,
    )


def schema_is_self_only(schema) -> bool:
    return resolve_command_target_policy(schema).actor_scope == _SELF_ONLY_SCOPE


def _disable_media_target_adapter(adapter: TargetPolicy) -> TargetPolicy:
    """Keep adapter metadata but prevent media target rules leaking to text-only commands."""

    return TargetPolicy(
        family=adapter.family,
        context_hints=adapter.context_hints,
        media_related=adapter.media_related,
        allow_at_as_target=False,
        allow_image_as_target=False,
        allow_reply_image_as_target=False,
        require_target_for_third_person=False,
        target_missing_message=adapter.target_missing_message,
    )


def _schema_text(schema) -> str:
    parts: list[str] = [
        str(getattr(schema, "command", "") or ""),
        str(getattr(schema, "head", "") or ""),
        str(getattr(schema, "description", "") or ""),
        str(getattr(schema, "render", "") or ""),
        str(getattr(schema, "payload_policy", "") or ""),
        str(getattr(schema, "command_role", "") or ""),
    ]
    parts.extend(str(item or "") for item in (getattr(schema, "params", None) or []))
    parts.extend(str(item or "") for item in (getattr(schema, "aliases", None) or []))
    parts.extend(
        str(item or "") for item in (getattr(schema, "retrieval_phrases", None) or [])
    )
    for slot in getattr(schema, "slots", None) or []:
        parts.append(str(getattr(slot, "name", "") or ""))
        parts.append(str(getattr(slot, "description", "") or ""))
        parts.extend(str(item or "") for item in (getattr(slot, "aliases", None) or []))
    return normalize_message_text(" ".join(parts)).casefold()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = normalize_message_text(text).casefold()
    return any(term.casefold() in lowered for term in terms)


def _generic_target_missing_message(*, allow_at: bool, allow_image: bool) -> str:
    if allow_at and allow_image:
        return "这个命令需要明确目标，请重新发送完整命令并@目标成员，或补充图片/回复图片。"
    if allow_at:
        return "这个命令需要明确目标，请重新发送完整命令并@目标成员或写清昵称。"
    if allow_image:
        return "这个命令需要图片上下文，请重新发送完整命令并附上图片，或回复一张图片。"
    return ""
