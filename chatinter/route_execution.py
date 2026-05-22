"""Safe command execution helpers for ChatInter."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from nonebot.adapters import Event

from .command_schema import normalize_schema_command_head
from .command_observation import build_command_observation
from .feedback_keys import (
    FEEDBACK_REASON_MISSING_PARAMS as _FEEDBACK_REASON_MISSING_PARAMS,
)
from .feedback_keys import (
    FEEDBACK_REASON_TARGET_REQUIRED as _FEEDBACK_REASON_TARGET_REQUIRED,
)
from .models.pydantic_models import PluginKnowledgeBase
from .native_route import NativeRouteResult
from .target_policy import TargetPolicy, get_target_policy
from .plugin_registry import PluginRegistry
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_placeholders,
    contains_any,
    is_usage_question,
    normalize_action_phrases,
    normalize_message_text,
)
from .schema_policy import resolve_command_target_policy


@dataclass(frozen=True)
class RouteExecutionPlan:
    command: str
    need_followup: bool = False
    followup_message: str | None = None
    feedback_reason: str | None = None
    image_missing: int = 0
    text_missing: int = 0
    allow_at: bool | None = None


@dataclass(frozen=True)
class RouteCommandSchemaView:
    command_id: str = ""
    command: str = ""
    aliases: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    description: str = ""
    payload_policy: str = "none"
    command_role: str = "execute"
    text_min: int = 0
    text_max: int | None = None
    image_min: int = 0
    image_max: int | None = None
    allow_at: bool | None = None
    actor_scope: str = "allow_other"
    target_requirement: str = "none"
    target_sources: tuple[str, ...] = ()
    requires_reply: bool = False
    requires_private: bool = False
    requires_to_me: bool = False
    allow_sticky_arg: bool = False
    access_level: str = "public"


def build_route_observation(
    *,
    route_result: NativeRouteResult,
    ok: bool,
    route_command: str = "",
    task_text: str = "",
    ambient_message: str = "",
    messages_sent: list[str] | tuple[str, ...] | None = None,
    artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    trace_id: str = "",
    error: str = "",
    missing: list[str] | tuple[str, ...] | None = None,
    retryable: bool = False,
    remaining_task_hint: str | None = None,
) -> dict[str, Any]:
    """Build the standard observation for a concrete route execution."""

    decision = route_result.decision
    return build_command_observation(
        ok=ok,
        command_id=route_result.command_id,
        rendered_command=route_command or decision.command,
        matched_plugin=decision.plugin_name,
        messages_sent=messages_sent,
        artifacts=artifacts,
        task_text=task_text,
        ambient_message=ambient_message,
        trace_id=trace_id,
        error=error,
        missing=missing,
        slots=dict(route_result.slots or {}),
        retryable=retryable,
        plugin_module=decision.plugin_module,
        remaining_task_hint=remaining_task_hint,
    )


def build_invalid_route_observation(
    *,
    decision,
    task_text: str,
    ambient_message: str,
    error: str,
    retryable: bool = True,
    missing: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the standard observation when route validation produced no route."""

    return build_command_observation(
        ok=False,
        command_id=getattr(decision, "command_id", "") or "",
        rendered_command=getattr(decision, "command", "") or "",
        matched_plugin=getattr(decision, "plugin_name", "") or "",
        task_text=task_text,
        ambient_message=ambient_message,
        error=error,
        missing=missing,
        retryable=retryable,
        plugin_module=getattr(decision, "plugin_module", "") or "",
    )


_SELF_ONLY_ACTION_KEYWORDS = ("\u7b7e\u5230", "\u6253\u5361", "\u8865\u7b7e")
_THIRD_PERSON_HINTS = (
    "\u4ed6",
    "\u5979",
    "ta",
    "\u5bf9\u65b9",
    "\u90a3\u4f4d",
    "\u8fd9\u4e2a\u4eba",
    "\u4e0a\u9762\u90a3\u4f4d",
)
_REPLY_REF_HINTS = (
    "\u56de\u590d",
    "\u5f15\u7528",
    "\u4e0a\u9762",
    "\u8fd9\u6761",
    "\u8fd9\u5f20",
    "\u8fd9\u56fe",
    "\u8fd9\u4e2a\u56fe",
    "\u8fd9\u5f20\u56fe",
    "\u7528\u8fd9\u5f20",
)
_AT_ID_TOKEN_PATTERN = re.compile(
    r"\[@([^\]\s]+)\]|(?<![0-9A-Za-z_])@(\d{5,20})(?=(?:\s|$|[\u7684\uff0c,\u3002.!！？?]))"
)
_PLACEHOLDER_SEGMENT_PATTERN = re.compile(r"\[@[^\]]+\]|\[image(?:#\d+)?\]")
_REPLY_TAG_PATTERN = re.compile(r"\[reply:[^\]]+\]", re.IGNORECASE)


def _is_self_only_action_message(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _SELF_ONLY_ACTION_KEYWORDS)


def _collect_target_capable_command_heads(knowledge_base) -> set[str]:
    heads: set[str] = set()
    plugins = getattr(knowledge_base, "plugins", None) or []
    for plugin in plugins:
        plugin_policy = _get_route_target_policy(
            plugin_module=getattr(plugin, "module", ""),
            plugin_name=getattr(plugin, "name", ""),
        )
        for meta in getattr(plugin, "command_meta", None) or []:
            policy = resolve_command_target_policy(
                meta,
                adapter_policy=plugin_policy,
            )
            image_min = int(getattr(meta, "image_min", 0) or 0)
            allow_sticky_arg = bool(getattr(meta, "allow_sticky_arg", False))
            if (
                not policy.allow_at
                and not policy.allow_image_as_target
                and not policy.allow_reply_image_as_target
                and image_min <= 0
                and policy.target_requirement == "none"
                and not allow_sticky_arg
            ):
                continue
            command_text = normalize_message_text(
                str(getattr(meta, "command", "") or "")
            )
            if command_text:
                heads.add(normalize_message_text(command_text.split(" ", 1)[0]))
            for alias in getattr(meta, "aliases", None) or []:
                alias_text = normalize_message_text(str(alias or ""))
                if alias_text:
                    heads.add(normalize_message_text(alias_text.split(" ", 1)[0]))
    return {head for head in heads if head}


def _get_route_target_policy(
    *,
    plugin_module: str = "",
    plugin_name: str = "",
    command_id: str = "",
) -> TargetPolicy:
    return get_target_policy(
        plugin_module=plugin_module,
        plugin_name=plugin_name,
        command_id=command_id,
    )


def _route_target_policy_from_result(
    route_result: NativeRouteResult,
) -> TargetPolicy:
    return _get_route_target_policy(
        plugin_module=route_result.decision.plugin_module,
        plugin_name=route_result.decision.plugin_name,
        command_id=route_result.command_id or "",
    )


def _has_adapter_context_hint(
    message_text: str,
    policy: TargetPolicy,
) -> bool:
    hints = tuple(policy.context_hints or ())
    if not hints:
        return False
    return contains_any(normalize_message_text(message_text or ""), hints)


def _build_target_modules(
    decision: NativeRouteResult,
    selection_plugins,
) -> set[str]:
    target_modules = {decision.decision.plugin_module}
    for plugin in selection_plugins:
        if plugin.name == decision.decision.plugin_name:
            target_modules.add(plugin.module)
    return target_modules


def _normalize_head(command_text: str) -> str:
    return normalize_schema_command_head(command_text)


def _schema_command_head(schema) -> str:
    return normalize_schema_command_head(
        str(getattr(schema, "command", None) or getattr(schema, "head", None) or "")
    )


def _view_from_command_meta(meta) -> RouteCommandSchemaView:
    return RouteCommandSchemaView(
        command_id=normalize_message_text(str(getattr(meta, "command_id", "") or "")),
        command=_schema_command_head(meta),
        aliases=tuple(
            alias
            for alias in (
                normalize_schema_command_head(str(item or ""))
                for item in (getattr(meta, "aliases", None) or [])
            )
            if alias
        ),
        prefixes=tuple(
            normalize_message_text(str(item or ""))
            for item in (getattr(meta, "prefixes", None) or [])
            if normalize_message_text(str(item or ""))
        ),
        params=tuple(
            normalize_message_text(str(item or ""))
            for item in (getattr(meta, "params", None) or [])
            if normalize_message_text(str(item or ""))
        ),
        description=normalize_message_text(str(getattr(meta, "description", "") or "")),
        payload_policy=normalize_message_text(
            str(getattr(meta, "payload_policy", "") or "none")
        ),
        command_role=normalize_message_text(
            str(getattr(meta, "command_role", "") or "execute")
        ),
        text_min=max(int(getattr(meta, "text_min", 0) or 0), 0),
        text_max=getattr(meta, "text_max", None),
        image_min=max(int(getattr(meta, "image_min", 0) or 0), 0),
        image_max=getattr(meta, "image_max", None),
        allow_at=getattr(meta, "allow_at", None),
        actor_scope=str(getattr(meta, "actor_scope", "") or "allow_other"),
        target_requirement=str(getattr(meta, "target_requirement", "") or "none"),
        target_sources=tuple(
            normalize_message_text(str(item or ""))
            for item in (getattr(meta, "target_sources", None) or [])
            if normalize_message_text(str(item or ""))
        ),
        requires_reply=bool(getattr(meta, "requires_reply", False)),
        requires_private=bool(getattr(meta, "requires_private", False)),
        requires_to_me=bool(getattr(meta, "requires_to_me", False)),
        allow_sticky_arg=bool(getattr(meta, "allow_sticky_arg", False)),
        access_level=str(getattr(meta, "access_level", "") or "public"),
    )


def _view_from_selected_candidate_schema(
    route_result: NativeRouteResult,
) -> RouteCommandSchemaView | None:
    candidate = getattr(route_result, "selected_candidate", None)
    schema = getattr(candidate, "schema", None)
    if schema is None:
        return None
    command_id = normalize_message_text(str(getattr(schema, "command_id", "") or ""))
    if (
        route_result.command_id
        and command_id
        and command_id.casefold() != normalize_message_text(route_result.command_id).casefold()
    ):
        return None
    return _view_from_plugin_command_schema(schema)


def _view_from_plugin_command_schema(schema) -> RouteCommandSchemaView:
    requires = getattr(schema, "requires", None) or {}
    slots = list(getattr(schema, "slots", None) or [])
    params = tuple(
        normalize_message_text(str(getattr(slot, "name", "") or ""))
        for slot in slots
        if normalize_message_text(str(getattr(slot, "name", "") or ""))
    )
    return RouteCommandSchemaView(
        command_id=normalize_message_text(str(getattr(schema, "command_id", "") or "")),
        command=_schema_command_head(schema),
        aliases=tuple(
            alias
            for alias in (
                normalize_schema_command_head(str(item or ""))
                for item in (getattr(schema, "aliases", None) or [])
            )
            if alias
        ),
        params=params,
        description=normalize_message_text(
            str(getattr(schema, "description", "") or "")
        ),
        payload_policy=normalize_message_text(
            str(getattr(schema, "payload_policy", "") or "none")
        ),
        command_role=normalize_message_text(
            str(getattr(schema, "command_role", "") or "execute")
        ),
        text_min=sum(
            1
            for slot in slots
            if str(getattr(slot, "type", "") or "") == "text"
            and bool(getattr(slot, "required", False))
        )
        or (0 if params else int(bool(requires.get("text")))),
        image_min=sum(
            1
            for slot in slots
            if str(getattr(slot, "type", "") or "") == "image"
            and bool(getattr(slot, "required", False))
        )
        or int(bool(requires.get("image"))),
        image_max=(
            None
            if any(str(getattr(slot, "type", "") or "") == "image" for slot in slots)
            else int(bool(requires.get("image")))
        ),
        allow_at=getattr(schema, "allow_at", None),
        actor_scope=str(getattr(schema, "actor_scope", "") or "allow_other"),
        target_requirement=str(getattr(schema, "target_requirement", "") or "none"),
        target_sources=tuple(
            normalize_message_text(str(item or ""))
            for item in (getattr(schema, "target_sources", None) or [])
            if normalize_message_text(str(item or ""))
        ),
        requires_reply=bool(requires.get("reply")),
        requires_private=bool(requires.get("private")),
        requires_to_me=bool(requires.get("to_me")),
        allow_sticky_arg=False,
        access_level="public",
    )


def _build_reference_schema_views(
    candidate_plugins,
) -> list[RouteCommandSchemaView]:
    if not candidate_plugins:
        return []
    knowledge_base = PluginKnowledgeBase(
        plugins=list(candidate_plugins),
        user_role="普通用户",
    )
    views: list[RouteCommandSchemaView] = []
    for reference in PluginRegistry.build_plugin_references(knowledge_base):
        for schema in reference.command_schemas:
            views.append(_view_from_plugin_command_schema(schema))
    return views


def _is_exact_command_head_match(head: str, schema: RouteCommandSchemaView) -> bool:
    normalized = normalize_schema_command_head(head).casefold()
    if not normalized:
        return False
    values = [schema.command, *schema.aliases]
    return any(normalized == value.casefold() for value in values if value)


def _is_public_command_meta(meta) -> bool:
    return (
        normalize_message_text(
            str(getattr(meta, "access_level", "public") or "public")
        ).lower()
        == "public"
    )


def _find_route_command_schema(route_result: NativeRouteResult, knowledge_plugins):
    selected_schema = _view_from_selected_candidate_schema(route_result)
    if selected_schema is not None:
        return selected_schema
    decision = route_result.decision
    head = _normalize_head(decision.command)
    command_id = normalize_message_text(route_result.command_id or "").casefold()
    if not head and not command_id:
        return None
    exact_module_plugins = [
        plugin
        for plugin in knowledge_plugins
        if plugin.module == decision.plugin_module
    ]
    candidate_plugins = exact_module_plugins or [
        plugin for plugin in knowledge_plugins if plugin.name == decision.plugin_name
    ]
    reference_views = _build_reference_schema_views(candidate_plugins)
    if command_id:
        for schema in reference_views:
            if schema.command_id.casefold() == command_id:
                return schema
        for plugin in candidate_plugins:
            for meta in plugin.command_meta:
                if not _is_public_command_meta(meta):
                    continue
                schema = _view_from_command_meta(meta)
                if schema.command_id.casefold() == command_id:
                    return schema
        return None
    for plugin in candidate_plugins:
        meta_views = [
            _view_from_command_meta(meta)
            for meta in plugin.command_meta
            if _is_public_command_meta(meta)
        ]
        for schema in meta_views:
            if _is_exact_command_head_match(head, schema):
                return schema
    for schema in reference_views:
        if _is_exact_command_head_match(head, schema):
            return schema
    return None


def _extract_at_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _AT_ID_TOKEN_PATTERN.finditer(text or ""):
        user_id = (match.group(1) or match.group(2) or "").strip()
        if not user_id:
            continue
        token = f"[@{user_id}]"
        if token not in tokens:
            tokens.append(token)
    return tokens


def _extract_image_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in collect_placeholders(text or ""):
        if token.lower().startswith("[image"):
            if token not in tokens:
                tokens.append(token)
    return tokens


def _contains_reply_reference_hint(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    if _REPLY_TAG_PATTERN.search(normalized):
        return True
    return any(hint in normalized for hint in _REPLY_REF_HINTS)


def _contains_third_person_reference(message_text: str) -> bool:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return False
    return any(hint in normalized for hint in _THIRD_PERSON_HINTS)


def _extract_reply_sender_id(event: Event) -> str | None:
    reply = getattr(event, "reply", None)
    if reply is None:
        return None
    sender = getattr(reply, "sender", None)
    if sender is None and isinstance(reply, dict):
        sender = reply.get("sender")
    if sender is None:
        return None
    user_id = None
    if isinstance(sender, dict):
        user_id = sender.get("user_id")
    else:
        user_id = getattr(sender, "user_id", None)
    if user_id is None:
        return None
    text = str(user_id).strip()
    return text if text.isdigit() else None


def _build_route_message_with_explicit_context(
    *,
    message_text: str,
    user_id: str,
    reply_image_count: int,
    reply_sender_id: str | None,
    target_policy: TargetPolicy | None = None,
) -> str:
    policy = target_policy or TargetPolicy()
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return normalized

    should_enrich = not is_usage_question(normalized) and (
        contains_any(normalized, ROUTE_ACTION_WORDS)
        or _has_adapter_context_hint(normalized, policy)
        or "[image" in normalized
        or "[@" in normalized
        or _contains_reply_reference_hint(normalized)
    )
    if not should_enrich:
        return normalized

    at_tokens = _extract_at_tokens(normalized)
    image_tokens = _extract_image_tokens(normalized)
    enriched = normalized
    policy_accepts_target = (
        policy.allow_at_as_target
        or policy.allow_image_as_target
        or policy.allow_reply_image_as_target
        or policy.require_target_for_third_person
    )

    if (
        policy_accepts_target
        and not at_tokens
        and _contains_strong_self_reference(normalized)
    ):
        enriched = normalize_message_text(f"{enriched} [@{user_id}]")
        at_tokens.append(f"[@{user_id}]")

    if (
        policy_accepts_target
        and not at_tokens
        and reply_sender_id
        and _contains_third_person_reference(normalized)
    ):
        enriched = normalize_message_text(f"{enriched} [@{reply_sender_id}]")
        at_tokens.append(f"[@{reply_sender_id}]")

    if (
        reply_image_count > 0
        and not image_tokens
        and (policy.allow_reply_image_as_target or policy.allow_image_as_target)
        and _contains_reply_reference_hint(normalized)
    ):
        suffix = " ".join("[image]" for _ in range(reply_image_count))
        enriched = normalize_message_text(f"{enriched} {suffix}")

    return enriched


def _select_adapter_policy_for_message(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> TargetPolicy:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return TargetPolicy()
    best_policy = TargetPolicy()
    best_score = 0
    for plugin in knowledge_base.plugins:
        score = 0
        command_texts: list[str] = []
        command_texts.extend(str(command or "") for command in plugin.commands)
        command_texts.extend(str(alias or "") for alias in plugin.aliases)
        command_policies = []
        for meta in getattr(plugin, "command_meta", None) or []:
            command_texts.append(str(getattr(meta, "command", "") or ""))
            command_texts.append(str(getattr(meta, "description", "") or ""))
            command_texts.extend(
                str(alias or "") for alias in getattr(meta, "aliases", None) or []
            )
            command_texts.extend(
                str(param or "") for param in getattr(meta, "params", None) or []
            )
            command_texts.extend(
                str(example or "") for example in getattr(meta, "examples", None) or []
            )
            command_policy = resolve_command_target_policy(meta)
            command_policies.append(command_policy)
            if command_policy.media_related:
                score += 1
            if command_policy.allow_at_as_target:
                score += 1
            if command_policy.allow_image_as_target:
                score += 1
        if any(
            text and text in normalized
            for text in (normalize_message_text(item) for item in command_texts)
        ):
            score += 3
        if score > best_score:
            best_policy = _merge_command_policies_to_adapter(command_policies)
            best_score = score
    return best_policy if best_score > 0 else TargetPolicy()


def _merge_command_policies_to_adapter(
    policies,
) -> TargetPolicy:
    context_hints: list[str] = []
    for hint in (
        "图片",
        "头像",
        "表情",
        "表情包",
        "照片",
        "回复",
        "@",
        "昵称",
        "目标",
        "对象",
    ):
        context_hints.append(hint)
    allow_at = any(policy.allow_at_as_target for policy in policies)
    allow_image = any(policy.allow_image_as_target for policy in policies)
    allow_reply_image = any(policy.allow_reply_image_as_target for policy in policies)
    target_required = any(policy.target_requirement == "required" for policy in policies)
    return TargetPolicy(
        family="general",
        context_hints=tuple(context_hints),
        media_related=allow_image,
        allow_at_as_target=allow_at,
        allow_image_as_target=allow_image,
        allow_reply_image_as_target=allow_reply_image,
        require_target_for_third_person=target_required or allow_at or allow_image,
    )


def _build_reply_image_segments_for_reroute(
    reply_images_data,
):
    if not reply_images_data:
        return []
    try:
        from nonebot.adapters.onebot.v11 import MessageSegment
    except Exception:
        return []

    segments = []
    seen_files: set[str] = set()
    for image in reply_images_data:
        file_id = str(getattr(image, "id", "") or "").strip()
        url = str(getattr(image, "url", "") or "").strip()
        path = getattr(image, "path", None)
        if not file_id and not url and not path:
            seg_type = getattr(image, "type", "")
            if seg_type == "image":
                seg_data = getattr(image, "data", {}) or {}
                file_id = str(seg_data.get("file", "") or "").strip()
                url = str(seg_data.get("url", "")).strip()
                path = seg_data.get("file")
        preferred_file_id = (
            file_id
            if file_id and not file_id.startswith(("http://", "https://", "base64://"))
            else ""
        )
        if preferred_file_id:
            key = f"id:{preferred_file_id}"
            if key in seen_files:
                continue
            try:
                if url:
                    segments.append(
                        MessageSegment(
                            "image",
                            {
                                "file": preferred_file_id,
                                "url": url,
                                "cache": "true",
                                "proxy": "true",
                            },
                        )
                    )
                else:
                    segments.append(MessageSegment.image(file=preferred_file_id))
                seen_files.add(key)
            except Exception:
                pass
            else:
                continue
        if url:
            key = f"url:{url}"
            if key in seen_files:
                continue
            try:
                segments.append(
                    MessageSegment(
                        "image",
                        {
                            "file": url,
                            "url": url,
                            "cache": "true",
                            "proxy": "true",
                        },
                    )
                )
                seen_files.add(key)
            except Exception:
                continue
            continue
        if path:
            path_text = str(path)
            key = f"path:{path_text}"
            if key in seen_files:
                continue
            try:
                segments.append(MessageSegment.image(file=path_text))
                seen_files.add(key)
            except Exception:
                continue
    return segments


def _extract_text_token_count(command_text: str) -> int:
    normalized = normalize_message_text(command_text)
    if not normalized:
        return 0
    parts = normalized.split(" ", 1)
    payload = parts[1] if len(parts) > 1 else ""
    payload = _PLACEHOLDER_SEGMENT_PATTERN.sub(" ", payload)
    payload = normalize_message_text(payload)
    if not payload:
        return 0
    return 1


def _fulfilled_text_slot_count(route_result: NativeRouteResult) -> int:
    """Count text slots that were explicitly filled by the selected tool call.

    The rendered command tail may contain multiple text slots, but plain text
    parsing cannot distinguish `cmd text1 text2` from one free-form text
    payload.  When the LLM selected a concrete command schema, the slot map is
    the source of truth for required text-slot satisfaction.
    """

    candidate = getattr(route_result, "selected_candidate", None)
    schema = getattr(candidate, "schema", None)
    slot_specs = list(getattr(schema, "slots", None) or [])
    route_slots = getattr(route_result, "slots", None) or {}
    if not slot_specs or not isinstance(route_slots, dict):
        return 0

    normalized_values = {
        normalize_message_text(str(key or "")): value
        for key, value in route_slots.items()
        if normalize_message_text(str(key or ""))
    }
    count = 0
    for slot in slot_specs:
        if str(getattr(slot, "type", "") or "") != "text":
            continue
        name = normalize_message_text(str(getattr(slot, "name", "") or ""))
        if not name:
            continue
        value = normalized_values.get(name)
        if value is None:
            continue
        if normalize_message_text(str(value or "")):
            count += 1
    return count


def _contains_self_reference(message_text: str) -> bool:
    normalized = normalize_message_text(normalize_action_phrases(message_text or ""))
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in ("我", "自己", "本人", "我的", "我自己", "自己的")
    )


def _contains_strong_self_reference(message_text: str) -> bool:
    normalized = normalize_message_text(normalize_action_phrases(message_text or ""))
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in ("我的", "我自己", "自己的", "本人", "本人的", "自己")
    )


def _build_followup_message(
    *,
    image_missing: int,
    text_missing: int,
    allow_at: bool,
) -> str:
    hints: list[str] = []
    if image_missing > 0:
        if allow_at:
            hints.append(f"还需要 {image_missing} 张图片（可发图或@目标）")
        else:
            hints.append(f"还需要 {image_missing} 张图片")
    if text_missing > 0:
        hints.append(f"还需要 {text_missing} 段文字")
    joined = "，".join(hints) if hints else "参数不足"
    return f"这个命令{joined}，请重新发送完整命令。"


def _build_target_required_message(schema) -> str:
    sources = {
        normalize_message_text(str(item or "")).lower()
        for item in (getattr(schema, "target_sources", None) or [])
    }
    hints: list[str] = []
    if "at" in sources:
        hints.append("直接@目标成员")
    if "reply" in sources:
        hints.append("回复对方消息并@")
    if "nickname" in sources:
        hints.append("补充完整昵称")
    if not hints:
        hints = ["补充目标成员（@或昵称）"]
    return "这个命令需要目标对象，请" + "、".join(hints) + "后重新发送完整命令。"


def _find_route_plugin_info(route_result: NativeRouteResult, knowledge_plugins):
    exact_module_plugins = [
        plugin
        for plugin in knowledge_plugins
        if plugin.module == route_result.decision.plugin_module
    ]
    if exact_module_plugins:
        return exact_module_plugins[0]
    for plugin in knowledge_plugins:
        if plugin.name == route_result.decision.plugin_name:
            return plugin
    return None


def _is_image_related_route(route_result: NativeRouteResult) -> bool:
    return _route_target_policy_from_result(route_result).media_related


def _append_unique_tokens(command: str, tokens: list[str]) -> str:
    normalized_command = normalize_message_text(command or "")
    if not normalized_command:
        return normalized_command
    merged: list[str] = []
    existing_placeholders = set(collect_placeholders(normalized_command))
    for token in tokens:
        text = normalize_message_text(token)
        if not text:
            continue
        if text in existing_placeholders:
            continue
        existing_placeholders.add(text)
        merged.append(text)
    if not merged:
        return normalized_command
    return normalize_message_text(f"{normalized_command} {' '.join(merged)}")


def _remove_tokens_from_command(command: str, tokens: list[str]) -> str:
    normalized_command = normalize_message_text(command or "")
    if not normalized_command or not tokens:
        return normalized_command
    parts = normalized_command.split(" ")
    head = normalize_message_text(parts[0] if parts else "")
    if not head:
        return ""
    token_set = {normalize_message_text(token) for token in tokens if token}
    payload = [
        token_text
        for token in parts[1:]
        if (token_text := normalize_message_text(token)) and token_text not in token_set
    ]
    if payload:
        return normalize_message_text(f"{head} {' '.join(payload)}")
    return head


def _prepare_route_execution_plan(
    *,
    route_result: NativeRouteResult,
    knowledge_plugins,
    current_message: str,
    ambient_message: str = "",
    user_id: str,
) -> RouteExecutionPlan:
    task_message = normalize_message_text(current_message or "")
    ambient_message = normalize_message_text(ambient_message or task_message)
    command = normalize_message_text(route_result.decision.command or "")
    if not command:
        return RouteExecutionPlan(command="")

    schema = _find_route_command_schema(route_result, knowledge_plugins)
    if schema is None:
        if _is_self_only_action_message(command):
            at_tokens = _extract_at_tokens(command)
            if at_tokens:
                command = _remove_tokens_from_command(command, at_tokens)
            return RouteExecutionPlan(command=command)
        if not _is_image_related_route(route_result):
            return RouteExecutionPlan(command=command)
        merged_at = _extract_at_tokens(task_message) or _extract_at_tokens(
            ambient_message
        )
        if not merged_at and _contains_self_reference(task_message or ambient_message):
            merged_at.append(f"[@{user_id}]")
        merged_images = _extract_image_tokens(task_message)
        for token in _extract_image_tokens(ambient_message):
            if token not in merged_images:
                merged_images.append(token)
        merged_tokens = [*merged_at, *merged_images]
        if merged_tokens:
            command = _append_unique_tokens(command, merged_tokens)
        return RouteExecutionPlan(command=command)

    schema_head = _schema_command_head(schema)
    command_head = _normalize_head(command)
    if schema_head and command_head and schema_head != command_head:
        tail = normalize_message_text(command[len(command_head) :].strip())
        command = (
            normalize_message_text(f"{schema_head} {tail}".strip())
            if tail
            else schema_head
        )

    image_min = max(int(getattr(schema, "image_min", 0) or 0), 0)
    image_max_value = getattr(schema, "image_max", None)
    try:
        image_max = (
            int(image_max_value)
            if image_max_value is not None
            else None
        )
    except (TypeError, ValueError):
        image_max = None
    text_min = max(int(getattr(schema, "text_min", 0) or 0), 0)
    policy = resolve_command_target_policy(
        schema,
        adapter_policy=_route_target_policy_from_result(route_result),
    )
    target_requirement = policy.target_requirement
    allow_at = policy.allow_at
    if allow_at:
        command_at = _extract_at_tokens(command)
    else:
        command_at = []
        disallowed_at = _extract_at_tokens(command)
        if disallowed_at:
            command = _remove_tokens_from_command(command, disallowed_at)
    command_images = _extract_image_tokens(command)
    message_images = _extract_image_tokens(task_message)
    for token in _extract_image_tokens(ambient_message):
        if token not in message_images:
            message_images.append(token)

    merged_at: list[str] = []
    if allow_at:
        merged_at = command_at[:]
        context_at_tokens = _extract_at_tokens(task_message)
        if not context_at_tokens:
            context_at_tokens = _extract_at_tokens(ambient_message)
        for token in context_at_tokens:
            if token not in merged_at:
                merged_at.append(token)
        if (
            target_requirement == "none"
            or (image_max is not None and image_max <= 0)
        ) and merged_at:
            command = _remove_tokens_from_command(command, merged_at)
            merged_at = []
    merged_images = command_images[:]
    for token in message_images:
        if token not in merged_images:
            merged_images.append(token)

    if (
        image_min > 0
        and allow_at
        and not merged_at
        and _contains_self_reference(task_message or ambient_message)
    ):
        self_at = f"[@{user_id}]"
        merged_at.append(self_at)

    if target_requirement == "required" and not (merged_at or merged_images):
        if allow_at and _contains_self_reference(task_message or ambient_message):
            merged_at.append(f"[@{user_id}]")
        else:
            return RouteExecutionPlan(
                command=_apply_route_command_prefixes(command, schema),
                need_followup=True,
                followup_message=(
                    policy.target_missing_message
                    or _build_target_required_message(schema)
                ),
                feedback_reason=_FEEDBACK_REASON_TARGET_REQUIRED,
                allow_at=allow_at,
            )

    if allow_at:
        image_count = len(merged_images) + len(merged_at)
    else:
        image_count = len(merged_images)
    text_count = max(
        _extract_text_token_count(command),
        _fulfilled_text_slot_count(route_result),
    )

    image_missing = max(image_min - image_count, 0)
    text_missing = max(text_min - text_count, 0)
    if image_missing > 0 or text_missing > 0:
        return RouteExecutionPlan(
            command=_apply_route_command_prefixes(command, schema),
            need_followup=True,
            followup_message=_build_followup_message(
                image_missing=image_missing,
                text_missing=text_missing,
                allow_at=allow_at,
            ),
            feedback_reason=_FEEDBACK_REASON_MISSING_PARAMS,
            image_missing=image_missing,
            text_missing=text_missing,
            allow_at=allow_at,
        )

    if allow_at and merged_at:
        command = _append_unique_tokens(command, merged_at)

    return RouteExecutionPlan(command=_apply_route_command_prefixes(command, schema))


def _apply_route_command_prefixes(command: str, schema) -> str:
    normalized = normalize_message_text(command)
    if not normalized or schema is None:
        return normalized
    raw_prefixes = getattr(schema, "prefixes", None) or []
    prefixes: list[str] = []
    for prefix in raw_prefixes:
        prefix_text = normalize_message_text(str(prefix or ""))
        if prefix_text and prefix_text not in prefixes:
            prefixes.append(prefix_text)
    if not prefixes:
        return normalized
    if any(normalized.startswith(prefix) for prefix in prefixes):
        return normalized
    return normalize_message_text(f"{prefixes[0]}{normalized}")


collect_target_capable_command_heads = _collect_target_capable_command_heads
has_adapter_context_hint = _has_adapter_context_hint
build_target_modules = _build_target_modules
extract_at_tokens = _extract_at_tokens
extract_image_tokens = _extract_image_tokens
contains_third_person_reference = _contains_third_person_reference
extract_reply_sender_id = _extract_reply_sender_id
build_route_message_with_explicit_context = _build_route_message_with_explicit_context
select_adapter_policy_for_message = _select_adapter_policy_for_message
build_reply_image_segments_for_reroute = _build_reply_image_segments_for_reroute
contains_self_reference = _contains_self_reference
prepare_route_execution_plan = _prepare_route_execution_plan
find_route_command_schema = _find_route_command_schema

__all__ = [
    "RouteExecutionPlan",
    "build_invalid_route_observation",
    "build_reply_image_segments_for_reroute",
    "build_route_observation",
    "build_route_message_with_explicit_context",
    "build_target_modules",
    "collect_target_capable_command_heads",
    "contains_self_reference",
    "contains_third_person_reference",
    "extract_at_tokens",
    "extract_image_tokens",
    "extract_reply_sender_id",
    "find_route_command_schema",
    "has_adapter_context_hint",
    "prepare_route_execution_plan",
    "select_adapter_policy_for_message",
]
