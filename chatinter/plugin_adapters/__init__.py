"""Plugin-specific ChatInter adapters.

The generic router should stay focused on command schemas, scoring and planning.
Adapters keep unavoidable plugin-specific knowledge in one place: schema overrides,
slot extractors, scoring hints and clarification policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..models.pydantic_models import (
    CommandCapability,
    CommandSlotSpec,
    PluginCommandSchema,
)
from ..route_text import normalize_message_text

if TYPE_CHECKING:
    from ..command_index import CommandCandidate


@dataclass(frozen=True)
class AdapterScoreHint:
    score: float
    reason: str


@dataclass(frozen=True)
class AdapterClarifyRoute:
    command_id: str
    missing: tuple[str, ...]
    confidence: float
    reason: str


@dataclass(frozen=True)
class AdapterTargetPolicy:
    family: str = "general"
    context_hints: tuple[str, ...] = ()
    media_related: bool = False
    allow_at_as_target: bool = False
    allow_image_as_target: bool = False
    allow_reply_image_as_target: bool = False
    require_target_for_third_person: bool = False
    target_missing_message: str = ""


@dataclass(frozen=True)
class AdapterNotificationPolicy:
    target_suffix: str = ""
    helper_heads: frozenset[str] = frozenset()
    default_templates: tuple[str, ...] = ()
    helper_templates: tuple[str, ...] = ()


SlotExtractor = Callable[[str, str], dict[str, Any]]
SchemaBuilder = Callable[[str, list[CommandCapability]], list[PluginCommandSchema]]
SemanticAliasProvider = Callable[[str, str, bool], list[str]]
ScoreHintProvider = Callable[[PluginCommandSchema, str, str], list[AdapterScoreHint]]
PromptScoreHintProvider = Callable[[PluginCommandSchema, str], list[AdapterScoreHint]]
ClarifyRouteResolver = Callable[
    [str, list["CommandCandidate"]], AdapterClarifyRoute | None
]


@dataclass(frozen=True)
class PluginCommandAdapter:
    modules: tuple[str, ...] = ()
    module_suffixes: tuple[str, ...] = ()
    family: str | None = None
    schemas: tuple[PluginCommandSchema, ...] = ()
    build_schemas: SchemaBuilder | None = None
    semantic_aliases: SemanticAliasProvider | None = None
    target_policy: AdapterTargetPolicy | None = None
    notification_policy: AdapterNotificationPolicy | None = None
    slot_extractors: dict[str, SlotExtractor] | None = None
    slot_extractor_prefixes: tuple[str, ...] = ()
    slot_extractor: SlotExtractor | None = None
    score_hints: ScoreHintProvider | None = None
    prompt_score_hints: PromptScoreHintProvider | None = None
    clarify_route: ClarifyRouteResolver | None = None


_ADAPTERS: list[PluginCommandAdapter] = []


def register_adapter(adapter: PluginCommandAdapter) -> None:
    _ADAPTERS.append(adapter)


def iter_adapters() -> tuple[PluginCommandAdapter, ...]:
    return tuple(_ADAPTERS)


def get_adapter_for_module(module: str) -> PluginCommandAdapter | None:
    module_key = normalize_message_text(module)
    for adapter in _ADAPTERS:
        if module_key in adapter.modules:
            return adapter
        if any(module_key.endswith(suffix) for suffix in adapter.module_suffixes):
            return adapter
    return None


def get_adapter_for_command_id(command_id: str) -> PluginCommandAdapter | None:
    command_key = normalize_message_text(command_id)
    if not command_key:
        return None
    for adapter in _ADAPTERS:
        if adapter.slot_extractors and command_key in adapter.slot_extractors:
            return adapter
        if adapter.slot_extractor is not None and any(
            command_key.startswith(prefix) for prefix in adapter.slot_extractor_prefixes
        ):
            return adapter
    return None


def get_adapter_for_route(
    *,
    plugin_module: str = "",
    command_id: str = "",
) -> PluginCommandAdapter | None:
    return get_adapter_for_module(plugin_module) or get_adapter_for_command_id(
        command_id
    )


def build_adapter_schemas(
    module: str,
    commands: list[CommandCapability],
) -> list[PluginCommandSchema] | None:
    adapter = get_adapter_for_module(module)
    if adapter is None:
        return None
    if adapter.build_schemas is not None:
        return adapter.build_schemas(normalize_message_text(module), commands)
    if adapter.schemas:
        return [schema.model_copy(deep=True) for schema in adapter.schemas]
    return None


def derive_adapter_semantic_aliases(
    head: object,
    *,
    module: str = "",
    image_required: bool = False,
) -> list[str]:
    normalized_head = normalize_message_text(str(head or ""))
    module_key = normalize_message_text(module)
    aliases: list[str] = []
    for adapter in iter_adapters():
        if adapter.semantic_aliases is None:
            continue
        if module_key and not (
            module_key in adapter.modules
            or any(module_key.endswith(suffix) for suffix in adapter.module_suffixes)
        ):
            continue
        for alias in adapter.semantic_aliases(
            normalized_head,
            module_key,
            image_required,
        ):
            text = normalize_message_text(alias)
            if text and text != normalized_head and text not in aliases:
                aliases.append(text)
    return aliases


def command_family_from_adapter(
    schema: PluginCommandSchema,
    *,
    plugin_module: str,
) -> str | None:
    adapter = get_adapter_for_module(plugin_module) or get_adapter_for_command_id(
        schema.command_id
    )
    return adapter.family if adapter is not None else None


def get_adapter_target_policy(
    *,
    plugin_module: str = "",
    plugin_name: str = "",
    command_id: str = "",
) -> AdapterTargetPolicy:
    adapter = get_adapter_for_route(
        plugin_module=plugin_module,
        command_id=command_id,
    )
    if adapter is not None and adapter.target_policy is not None:
        return adapter.target_policy
    module_text = normalize_message_text(plugin_module).casefold()
    name_text = normalize_message_text(plugin_name).casefold()
    return AdapterTargetPolicy(
        family=adapter.family if adapter is not None and adapter.family else "general",
        media_related=(
            "image" in module_text or "图片" in name_text or "图" == name_text
        ),
    )


def get_adapter_target_policy_for_schema(
    schema: object,
    *,
    plugin_module: str = "",
    plugin_name: str = "",
    command_id: str = "",
) -> AdapterTargetPolicy:
    return get_adapter_target_policy(
        plugin_module=plugin_module,
        plugin_name=plugin_name,
        command_id=command_id or str(getattr(schema, "command_id", "") or ""),
    )


def get_adapter_notification_policy(
    *,
    plugin_module: str = "",
    command_id: str = "",
) -> AdapterNotificationPolicy | None:
    adapter = get_adapter_for_route(
        plugin_module=plugin_module,
        command_id=command_id,
    )
    if adapter is None:
        return None
    return adapter.notification_policy


def extract_adapter_slots(
    command_id: str,
    message_text: str,
    arguments_text: str = "",
) -> dict[str, Any]:
    command_key = normalize_message_text(command_id)
    source = normalize_message_text(message_text)
    if not source and arguments_text:
        source = normalize_message_text(arguments_text)
    adapter = get_adapter_for_command_id(command_key)
    if adapter is None or not adapter.slot_extractors:
        if adapter is not None and adapter.slot_extractor is not None:
            return adapter.slot_extractor(command_key, source)
        return {}
    extractor = adapter.slot_extractors.get(command_key)
    if extractor is not None:
        return extractor(command_key, source)
    if adapter.slot_extractor is not None:
        return adapter.slot_extractor(command_key, source)
    return {}


def collect_score_hints(
    schema: PluginCommandSchema,
    *,
    lowered_query: str,
    stripped_lowered_query: str,
    plugin_module: str,
) -> list[AdapterScoreHint]:
    hints: list[AdapterScoreHint] = []
    for adapter in iter_adapters():
        if adapter.score_hints is None:
            continue
        hints.extend(adapter.score_hints(schema, lowered_query, stripped_lowered_query))
    return hints


def collect_prompt_score_hints(
    schema: PluginCommandSchema,
    *,
    normalized_query: str,
) -> list[AdapterScoreHint]:
    hints: list[AdapterScoreHint] = []
    for adapter in iter_adapters():
        if adapter.prompt_score_hints is None:
            continue
        hints.extend(adapter.prompt_score_hints(schema, normalized_query))
    return hints


def resolve_adapter_clarify_route(
    message_text: str,
    candidates: list["CommandCandidate"],
) -> AdapterClarifyRoute | None:
    for adapter in iter_adapters():
        if adapter.clarify_route is None:
            continue
        result = adapter.clarify_route(message_text, candidates)
        if result is not None:
            return result
    return None


def slot(
    name: str,
    slot_type: str = "text",
    *,
    required: bool = False,
    default: Any = None,
    aliases: list[str] | None = None,
    description: str = "",
) -> CommandSlotSpec:
    return CommandSlotSpec(
        name=name,
        type=slot_type,  # type: ignore[arg-type]
        required=required,
        default=default,
        aliases=list(aliases or []),
        description=description,
    )


def schema(
    command_id: str,
    head: str,
    *,
    aliases: list[str] | None = None,
    description: str = "",
    slots: list[CommandSlotSpec] | None = None,
    render: str | None = None,
    requires: dict[str, bool] | None = None,
    command_role: str = "execute",
    payload_policy: str = "none",
    extra_text_policy: str = "keep",
    source: str = "override",
    confidence: float = 0.85,
    matcher_key: str | None = None,
    retrieval_phrases: list[str] | None = None,
) -> PluginCommandSchema:
    normalized_head = normalize_message_text(head)
    normalized_aliases = [
        text
        for text in (normalize_message_text(alias) for alias in list(aliases or []))
        if text
    ]
    normalized_description = normalize_message_text(description)
    phrase_values = [
        normalized_head,
        *normalized_aliases,
        normalized_description,
        command_id,
        *(retrieval_phrases or []),
    ]
    phrases: list[str] = []
    for value in phrase_values:
        text = normalize_message_text(value)
        if text and text not in phrases:
            phrases.append(text)
    return PluginCommandSchema(
        command_id=command_id,
        head=normalized_head or head,
        aliases=list(dict.fromkeys(normalized_aliases)),
        description=normalized_description,
        slots=list(slots or []),
        render=render or head,
        requires={
            "text": False,
            "image": False,
            "reply": False,
            "at": False,
            **dict(requires or {}),
        },
        command_role=command_role,  # type: ignore[arg-type]
        payload_policy=payload_policy,  # type: ignore[arg-type]
        extra_text_policy=extra_text_policy,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
        matcher_key=matcher_key,
        retrieval_phrases=phrases,
    )


__all__ = [
    "AdapterClarifyRoute",
    "AdapterNotificationPolicy",
    "AdapterScoreHint",
    "AdapterTargetPolicy",
    "PluginCommandAdapter",
    "build_adapter_schemas",
    "collect_prompt_score_hints",
    "collect_score_hints",
    "command_family_from_adapter",
    "derive_adapter_semantic_aliases",
    "extract_adapter_slots",
    "get_adapter_for_command_id",
    "get_adapter_for_module",
    "get_adapter_for_route",
    "get_adapter_notification_policy",
    "get_adapter_target_policy",
    "get_adapter_target_policy_for_schema",
    "iter_adapters",
    "register_adapter",
    "resolve_adapter_clarify_route",
    "schema",
    "slot",
]

from . import builtin, memes  # noqa: F401
