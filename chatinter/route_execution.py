"""Route planning and command execution helpers for ChatInter."""

from __future__ import annotations

from dataclasses import dataclass
import re

from nonebot.adapters import Event

from .command_planner import CommandPlanDecision, plan_command
from .feedback_keys import (
    FEEDBACK_REASON_MISSING_PARAMS as _FEEDBACK_REASON_MISSING_PARAMS,
    FEEDBACK_REASON_TARGET_REQUIRED as _FEEDBACK_REASON_TARGET_REQUIRED,
)
from .models.pydantic_models import PluginKnowledgeBase
from .native_route import NativeRouteResult
from .plugin_adapters import AdapterTargetPolicy, get_adapter_target_policy
from .plugin_registry import PluginRegistry
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_placeholders,
    contains_any,
    is_usage_question,
    match_command_head_canonical,
    normalize_action_phrases,
    normalize_message_text,
    parse_command_with_head,
)
from .schema_policy import resolve_command_target_policy
from .skill_registry import (
    SkillRouteDecision,
    _extract_explicit_value,
    _extract_schema_argument_tokens,
)

@dataclass(frozen=True)
class RouteExecutionPlan:
    command: str
    need_followup: bool = False
    followup_message: str | None = None
    feedback_reason: str | None = None
    image_missing: int = 0
    text_missing: int = 0
    allow_at: bool | None = None


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
) -> AdapterTargetPolicy:
    return get_adapter_target_policy(
        plugin_module=plugin_module,
        plugin_name=plugin_name,
        command_id=command_id,
    )


def _route_target_policy_from_result(
    route_result: NativeRouteResult,
) -> AdapterTargetPolicy:
    return _get_route_target_policy(
        plugin_module=route_result.decision.plugin_module,
        plugin_name=route_result.decision.plugin_name,
        command_id=route_result.command_id or "",
    )


def _has_adapter_context_hint(
    message_text: str,
    policy: AdapterTargetPolicy,
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
    normalized = normalize_message_text(command_text or "")
    if not normalized:
        return ""
    return normalize_message_text(normalized.split(" ", 1)[0])


def _iter_meta_aliases(meta) -> set[str]:
    aliases = getattr(meta, "aliases", None) or []
    values: set[str] = set()
    for alias in aliases:
        normalized = normalize_message_text(str(alias or ""))
        if normalized:
            values.add(normalized)
    return values


def _is_public_command_meta(meta) -> bool:
    return (
        normalize_message_text(
            str(getattr(meta, "access_level", "public") or "public")
        ).lower()
        == "public"
    )


def _find_route_command_schema(route_result: NativeRouteResult, knowledge_plugins):
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
    for plugin in candidate_plugins:
        if command_id:
            for meta in plugin.command_meta:
                meta_id = normalize_message_text(
                    str(getattr(meta, "command_id", "") or "")
                ).casefold()
                if meta_id and meta_id == command_id:
                    return meta
        plugin_aliases = {
            _normalize_head(alias).casefold()
            for alias in (getattr(plugin, "aliases", None) or [])
            if _normalize_head(alias)
        }
        for meta in plugin.command_meta:
            if not _is_public_command_meta(meta):
                continue
            command_head = normalize_message_text(getattr(meta, "command", ""))
            if not command_head:
                continue
            if match_command_head_canonical(head, command_head) or any(
                match_command_head_canonical(head, alias)
                for alias in _iter_meta_aliases(meta)
            ):
                return meta
        if head in plugin_aliases and len(plugin.command_meta) == 1:
            return plugin.command_meta[0]
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
    target_policy: AdapterTargetPolicy | None = None,
) -> str:
    policy = target_policy or AdapterTargetPolicy()
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

    if not at_tokens and _contains_strong_self_reference(normalized):
        enriched = normalize_message_text(f"{enriched} [@{user_id}]")
        at_tokens.append(f"[@{user_id}]")

    if (
        not at_tokens
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
) -> AdapterTargetPolicy:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return AdapterTargetPolicy()
    best_policy = AdapterTargetPolicy()
    best_score = 0
    for plugin in knowledge_base.plugins:
        policy = _get_route_target_policy(
            plugin_module=plugin.module,
            plugin_name=plugin.name,
        )
        if not policy.context_hints:
            continue
        score = 0
        if policy.media_related:
            score += 1
        if _has_adapter_context_hint(normalized, policy):
            score += 4
        command_texts: list[str] = []
        command_texts.extend(str(command or "") for command in plugin.commands)
        command_texts.extend(str(alias or "") for alias in plugin.aliases)
        for meta in getattr(plugin, "command_meta", None) or []:
            command_texts.append(str(getattr(meta, "command", "") or ""))
            command_texts.extend(
                str(alias or "") for alias in getattr(meta, "aliases", None) or []
            )
        if any(
            text and text in normalized
            for text in (normalize_message_text(item) for item in command_texts)
        ):
            score += 3
        if score > best_score:
            best_policy = policy
            best_score = score
    return best_policy if best_score > 0 else AdapterTargetPolicy()


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
    return len([token for token in payload.split(" ") if token])


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


def _build_planner_followup_message(missing: list[str]) -> str:
    labels: list[str] = []
    for item in missing:
        normalized = normalize_message_text(item).lower()
        if normalized in {"text", "文本", "文字", "参数", "内容"}:
            label = "要处理的文字"
        elif normalized in {"image", "图片", "图", "照片"}:
            label = "图片"
        elif normalized in {"reply", "回复", "引用"}:
            label = "回复上下文"
        else:
            label = item
        if label and label not in labels:
            labels.append(label)
    joined = "、".join(labels) if labels else "必要参数"
    return f"这个命令还需要{joined}，请补充后我再帮你执行。"


def _planner_missing_contains(missing: list[str], names: set[str]) -> bool:
    return any(normalize_message_text(item).lower() in names for item in missing)


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


def _extract_command_payload_tokens(command: str) -> list[str]:
    normalized_command = normalize_message_text(command or "")
    if not normalized_command:
        return []
    parts = normalized_command.split(" ", 1)
    if len(parts) < 2:
        return []
    tokens: list[str] = []
    for raw_token in parts[1].split(" "):
        token = normalize_message_text(raw_token)
        if not token:
            continue
        tokens.append(token)
    return tokens


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


def _clamp_command_text_tokens(command: str, text_max_raw) -> str:
    normalized_command = normalize_message_text(command or "")
    if not normalized_command:
        return normalized_command
    if text_max_raw is None:
        return normalized_command
    try:
        text_max = int(text_max_raw)
    except Exception:
        return normalized_command
    text_max = max(text_max, 0)

    parts = normalized_command.split(" ", 1)
    command_head = parts[0]
    if len(parts) < 2:
        return command_head

    kept_tokens: list[str] = []
    text_count = 0
    for raw_token in parts[1].split(" "):
        token = normalize_message_text(raw_token)
        if not token:
            continue
        if _PLACEHOLDER_SEGMENT_PATTERN.fullmatch(token):
            kept_tokens.append(token)
            continue
        if text_count < text_max:
            kept_tokens.append(token)
            text_count += 1

    if kept_tokens:
        return normalize_message_text(f"{command_head} {' '.join(kept_tokens)}")
    return command_head


def _schema_accepts_text_payload(schema) -> bool:
    text_min = max(int(getattr(schema, "text_min", 0) or 0), 0)
    if text_min > 0:
        return True
    text_max = getattr(schema, "text_max", None)
    if text_max is not None:
        try:
            return int(text_max) > 0
        except Exception:
            return False
    return bool(getattr(schema, "params", None))


def _prepare_route_execution_plan(
    *,
    route_result: NativeRouteResult,
    knowledge_plugins,
    current_message: str,
    user_id: str,
) -> RouteExecutionPlan:
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
        merged_at = _extract_at_tokens(current_message)
        if not merged_at and _contains_self_reference(current_message):
            merged_at.append(f"[@{user_id}]")
        merged_images = _extract_image_tokens(current_message)
        merged_tokens = [*merged_at, *merged_images]
        if merged_tokens:
            command = _append_unique_tokens(command, merged_tokens)
        return RouteExecutionPlan(command=command)

    schema_head = _normalize_head(getattr(schema, "command", ""))
    command_head = _normalize_head(command)
    if schema_head and command_head and schema_head != command_head:
        tail = normalize_message_text(command[len(command_head) :].strip())
        command = (
            normalize_message_text(f"{schema_head} {tail}".strip())
            if tail
            else schema_head
        )

    existing_payload_tokens = set(_extract_command_payload_tokens(command))
    payload_tokens: list[str] = []
    explicit_value = normalize_message_text(_extract_explicit_value(current_message))
    accepts_text_payload = _schema_accepts_text_payload(schema)
    if explicit_value and accepts_text_payload:
        payload_tokens.extend(
            token
            for token in explicit_value.split(" ")
            if token
            and token not in payload_tokens
            and token not in existing_payload_tokens
        )
    schema_tokens = _extract_schema_argument_tokens(current_message, schema)
    for token in schema_tokens:
        if (
            token
            and token not in payload_tokens
            and token not in existing_payload_tokens
        ):
            payload_tokens.append(token)
    if not payload_tokens and accepts_text_payload:
        parsed_payload = ""
        try:
            parsed = parse_command_with_head(
                current_message,
                schema_head or command_head,
                allow_sticky=bool(getattr(schema, "allow_sticky_arg", False)),
                max_prefix_len=16,
            )
            parsed_payload = normalize_message_text(
                (parsed.payload_text if parsed else "") or ""
            )
        except Exception:
            parsed_payload = ""
        if parsed_payload:
            for token in parsed_payload.split(" "):
                if (
                    token
                    and token not in payload_tokens
                    and token not in existing_payload_tokens
                ):
                    payload_tokens.append(token)
    if payload_tokens:
        command = _append_unique_tokens(command, payload_tokens)

    if not getattr(schema, "params", None):
        command = _clamp_command_text_tokens(command, getattr(schema, "text_max", None))

    image_min = max(int(getattr(schema, "image_min", 0) or 0), 0)
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
    message_images = _extract_image_tokens(current_message)

    merged_at: list[str] = []
    if allow_at:
        merged_at = command_at[:]
        for token in _extract_at_tokens(current_message):
            if token not in merged_at:
                merged_at.append(token)
        if target_requirement == "none" and merged_at:
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
        and _contains_self_reference(current_message)
    ):
        self_at = f"[@{user_id}]"
        merged_at.append(self_at)

    if target_requirement == "required" and not (merged_at or merged_images):
        if allow_at and _contains_self_reference(current_message):
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
    text_count = _extract_text_token_count(command)

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


def _plan_route_command(
    *,
    route_result: NativeRouteResult,
    knowledge_plugins,
    current_message: str,
    has_reply: bool,
    image_count: int,
) -> CommandPlanDecision:
    knowledge_base = PluginKnowledgeBase(
        plugins=list(knowledge_plugins),
        user_role="普通用户",
    )
    references = PluginRegistry.build_plugin_references(knowledge_base)
    decision = route_result.decision
    return plan_command(
        action="execute",
        plugin_module=decision.plugin_module,
        plugin_name=decision.plugin_name,
        command=decision.command,
        command_id=route_result.command_id,
        slots=route_result.slots,
        references=references,
        current_message=current_message,
        has_reply=has_reply,
        image_count=image_count,
        reason=f"route_stage:{route_result.stage}",
    )


def _apply_command_plan_to_route_result(
    route_result: NativeRouteResult,
    command_plan: CommandPlanDecision,
) -> NativeRouteResult:
    final_command = normalize_message_text(command_plan.final_command or "")
    final_command = final_command or normalize_message_text(
        route_result.decision.command
    )
    merged_slots = {**route_result.slots, **dict(command_plan.slots or {})}
    command_id = command_plan.command_id or route_result.command_id
    missing = tuple(command_plan.missing or route_result.missing)
    if not final_command and (
        command_id == route_result.command_id
        and merged_slots == route_result.slots
        and missing == route_result.missing
    ):
        return route_result
    decision = route_result.decision
    if (
        normalize_message_text(decision.command) == final_command
        and command_id == route_result.command_id
        and merged_slots == route_result.slots
        and missing == route_result.missing
    ):
        return route_result
    return NativeRouteResult(
        decision=SkillRouteDecision(
            plugin_name=decision.plugin_name,
            plugin_module=decision.plugin_module,
            command=final_command,
            source=decision.source,
            skill_kind=decision.skill_kind,
        ),
        stage=route_result.stage,
        report=route_result.report,
        command_id=command_id,
        slots=merged_slots,
        missing=missing,
        selected_rank=route_result.selected_rank,
        selected_score=route_result.selected_score,
        selected_reason=route_result.selected_reason,
    )


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
build_planner_followup_message = _build_planner_followup_message
planner_missing_contains = _planner_missing_contains
prepare_route_execution_plan = _prepare_route_execution_plan
plan_route_command = _plan_route_command
apply_command_plan_to_route_result = _apply_command_plan_to_route_result

__all__ = [
    "RouteExecutionPlan",
    "apply_command_plan_to_route_result",
    "build_planner_followup_message",
    "build_reply_image_segments_for_reroute",
    "build_route_message_with_explicit_context",
    "build_target_modules",
    "collect_target_capable_command_heads",
    "contains_self_reference",
    "contains_third_person_reference",
    "extract_at_tokens",
    "extract_image_tokens",
    "extract_reply_sender_id",
    "has_adapter_context_hint",
    "planner_missing_contains",
    "plan_route_command",
    "prepare_route_execution_plan",
    "select_adapter_policy_for_message",
]
