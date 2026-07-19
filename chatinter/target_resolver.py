"""Command-aware target resolution and missing-context gate.

The pre-route pass can only guess target policy from the raw message.  Once a
native tool has selected a concrete command, this module re-checks the command
schema and resolves nickname targets again before rerouting to NoneBot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nonebot.adapters import Bot

from .native_route import NativeRouteResult
from .route_execution import (
    extract_at_tokens,
    extract_image_tokens,
    find_route_command_schema,
)
from .route_text import normalize_message_text
from .schema_policy import CommandTargetPolicy, resolve_command_target_policy
from .target_context import (
    enrich_route_message_with_fuzzy_target,
    extract_fuzzy_target_hint,
    needs_target_for_route,
)
from .target_policy import TargetPolicy, get_target_policy

TargetResolveStatus = Literal[
    "not_needed",
    "present",
    "resolved",
    "ambiguous",
    "missing",
]


@dataclass(frozen=True)
class TargetResolveResult:
    status: TargetResolveStatus
    message_text: str
    mention_profiles: dict[str, dict[str, str]] = field(default_factory=dict)
    prompt: str = ""
    target_hint: str = ""

    @property
    def blocked(self) -> bool:
        return self.status in {"ambiguous", "missing"} and bool(self.prompt)

    @property
    def resolved(self) -> bool:
        return self.status in {"present", "resolved"}


def _route_adapter_policy(route_result: NativeRouteResult) -> TargetPolicy:
    decision = route_result.decision
    return get_target_policy(
        plugin_module=decision.plugin_module,
        plugin_name=decision.plugin_name,
        command_id=route_result.command_id or "",
    )


def _route_command_policy(
    route_result: NativeRouteResult,
    knowledge_plugins,
) -> tuple[object | None, TargetPolicy, CommandTargetPolicy | None]:
    schema = find_route_command_schema(route_result, knowledge_plugins)
    adapter_policy = _route_adapter_policy(route_result)
    command_policy = (
        resolve_command_target_policy(schema, adapter_policy=adapter_policy)
        if schema is not None
        else None
    )
    return schema, adapter_policy, command_policy


def _schema_image_min(schema: object | None) -> int:
    if schema is None:
        return 0
    try:
        return max(int(getattr(schema, "image_min", 0) or 0), 0)
    except Exception:
        return 0


def _schema_accepts_image_context(schema: object | None) -> bool:
    if schema is None:
        return False
    try:
        policy = resolve_command_target_policy(schema)
        if policy.allow_image_as_target or policy.allow_reply_image_as_target:
            return True
    except Exception:
        pass
    if _schema_image_min(schema) > 0:
        return True
    if not hasattr(schema, "image_max"):
        return False
    image_max = getattr(schema, "image_max", 0)
    if image_max is None:
        return True
    try:
        return int(image_max) > 0
    except Exception:
        return False


def _schema_heads(schema: object | None, route_result: NativeRouteResult) -> set[str]:
    values: list[object] = [
        getattr(schema, "command", "") if schema is not None else "",
        getattr(schema, "head", "") if schema is not None else "",
        route_result.decision.command,
    ]
    if schema is not None:
        values.extend(getattr(schema, "aliases", None) or [])
    heads: set[str] = set()
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text:
            heads.add(text.split(" ", 1)[0])
    return heads


def _has_target_or_image_context(*messages: str) -> bool:
    for message in messages:
        if extract_at_tokens(message) or extract_image_tokens(message):
            return True
    return False


def _append_unique_context_tokens(message_text: str, *sources: str) -> str:
    message = normalize_message_text(message_text)
    existing = set(extract_at_tokens(message)) | set(extract_image_tokens(message))
    tokens: list[str] = []
    for source in sources:
        for token in [*extract_at_tokens(source), *extract_image_tokens(source)]:
            if token in existing:
                continue
            existing.add(token)
            tokens.append(token)
    if not tokens:
        return message
    return normalize_message_text(f"{message} {' '.join(tokens)}")


def _command_can_use_target(
    *,
    schema: object | None,
    adapter_policy: TargetPolicy,
    command_policy: CommandTargetPolicy | None,
) -> bool:
    if command_policy is not None:
        if (
            command_policy.allow_at_as_target
            or command_policy.allow_image_as_target
            or command_policy.allow_reply_image_as_target
            or command_policy.media_related
            or command_policy.target_requirement != "none"
        ):
            return True
    return (
        adapter_policy.allow_at_as_target
        or adapter_policy.allow_image_as_target
        or adapter_policy.allow_reply_image_as_target
        or adapter_policy.require_target_for_third_person
        or _schema_accepts_image_context(schema)
    )


async def resolve_pre_route_target(
    *,
    group_id: str | None,
    bot: Bot | None = None,
    original_message: str,
    route_message: str,
    mention_profiles: dict[str, dict[str, str]],
    target_policy: TargetPolicy,
    command_heads: set[str] | None = None,
) -> TargetResolveResult:
    (
        enriched_message,
        enriched_profiles,
        prompt,
    ) = await enrich_route_message_with_fuzzy_target(
        group_id=group_id,
        bot=bot,
        original_message=original_message,
        route_message=route_message,
        mention_profiles=mention_profiles,
        target_policy=target_policy,
        command_heads=command_heads,
    )
    if prompt:
        return TargetResolveResult(
            status="not_needed",
            message_text=route_message,
            mention_profiles=enriched_profiles,
            target_hint=extract_fuzzy_target_hint(route_message, command_heads),
        )
    if enriched_message != route_message:
        return TargetResolveResult(
            status="resolved",
            message_text=enriched_message,
            mention_profiles=enriched_profiles,
            target_hint=extract_fuzzy_target_hint(route_message, command_heads),
        )
    if _has_target_or_image_context(route_message):
        return TargetResolveResult(
            status="present",
            message_text=route_message,
            mention_profiles=enriched_profiles,
        )
    return TargetResolveResult(
        status="not_needed",
        message_text=route_message,
        mention_profiles=enriched_profiles,
    )


async def resolve_execution_target(
    *,
    group_id: str | None,
    bot: Bot | None = None,
    bot_id: str | None = None,
    route_result: NativeRouteResult,
    knowledge_plugins,
    task_message: str,
    ambient_message: str,
    target_hint: str = "",
    mention_profiles: dict[str, dict[str, str]] | None = None,
) -> TargetResolveResult:
    schema, adapter_policy, command_policy = _route_command_policy(
        route_result,
        knowledge_plugins,
    )
    mention_profiles = dict(mention_profiles or {})
    if not _command_can_use_target(
        schema=schema,
        adapter_policy=adapter_policy,
        command_policy=command_policy,
    ):
        return TargetResolveResult(
            status="not_needed",
            message_text=task_message,
            mention_profiles=mention_profiles,
        )

    explicit_target_hint = normalize_message_text(target_hint)
    if _has_target_or_image_context(explicit_target_hint):
        enriched_message = _append_unique_context_tokens(
            task_message,
            explicit_target_hint,
        )
        return TargetResolveResult(
            status="resolved" if enriched_message != task_message else "present",
            message_text=enriched_message,
            mention_profiles=mention_profiles,
            target_hint=explicit_target_hint,
        )

    if _has_target_or_image_context(task_message, ambient_message):
        return TargetResolveResult(
            status="present",
            message_text=task_message,
            mention_profiles=mention_profiles,
            target_hint=explicit_target_hint,
        )

    command_heads = _schema_heads(schema, route_result)
    if command_policy is not None and command_policy.allow_at_as_target and bot_id:
        self_target = await resolve_pre_route_target(
            group_id=group_id,
            bot=bot,
            original_message=task_message,
            route_message=task_message,
            mention_profiles=mention_profiles,
            target_policy=adapter_policy,
            command_heads=command_heads,
        )
        if self_target.status in {"resolved", "present"}:
            return self_target

    target_lookup_message = normalize_message_text(
        " ".join(item for item in (task_message, explicit_target_hint) if item)
    )
    extracted_target_hint = extract_fuzzy_target_hint(
        target_lookup_message or task_message,
        command_heads,
    )
    resolved_target_hint = explicit_target_hint or extracted_target_hint
    resolved = await resolve_pre_route_target(
        group_id=group_id,
        bot=bot,
        original_message=target_lookup_message or task_message,
        route_message=target_lookup_message or task_message,
        mention_profiles=mention_profiles,
        target_policy=adapter_policy,
        command_heads=command_heads,
    )
    if resolved.status in {"resolved", "ambiguous", "missing"}:
        if resolved.status == "resolved":
            resolved_message = _append_unique_context_tokens(
                task_message,
                resolved.message_text,
            )
            return TargetResolveResult(
                status="resolved",
                message_text=resolved_message,
                mention_profiles=resolved.mention_profiles,
                prompt=resolved.prompt,
                target_hint=resolved.target_hint or resolved_target_hint,
            )
        return resolved

    target_required = (
        command_policy is not None and command_policy.target_requirement == "required"
    )
    target_required = target_required or needs_target_for_route(
        task_message,
        task_message,
        target_policy=adapter_policy,
    )
    target_required = target_required or bool(
        resolved_target_hint
        and (
            _schema_accepts_image_context(schema)
            or (
                command_policy is not None
                and (
                    command_policy.allow_at_as_target
                    or command_policy.allow_image_as_target
                )
            )
        )
    )
    if target_required:
        return TargetResolveResult(
            status="missing",
            message_text=task_message,
            mention_profiles=mention_profiles,
            target_hint=resolved_target_hint,
        )

    return TargetResolveResult(
        status="not_needed",
        message_text=task_message,
        mention_profiles=mention_profiles,
        target_hint=resolved_target_hint,
    )


__all__ = [
    "TargetResolveResult",
    "resolve_execution_target",
    "resolve_pre_route_target",
]
