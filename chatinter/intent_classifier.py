from dataclasses import dataclass
import re
from typing import Literal

from .models.pydantic_models import PluginKnowledgeBase
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_placeholders,
    contains_any,
    has_negative_route_intent,
    is_usage_question,
    match_command_head,
    match_command_head_fuzzy,
    match_command_head_or_sticky,
    normalize_action_phrases,
    normalize_message_text,
    rewrite_command_with_head,
    strip_invoke_prefix,
)
from .schema_policy import resolve_command_target_policy
from .skill_registry import SkillCommandSchema, SkillSpec, get_skill_registry, skill_search

IntentKind = Literal["chat", "help", "execute", "execute_need_arg", "ambiguous"]
IntentSchemaState = Literal[
    "unknown",
    "ready",
    "missing_target",
    "missing_image",
    "missing_text",
]

_AT_TOKEN_PATTERN = re.compile(r"\[@\d{5,20}\]")
_IMAGE_TOKEN_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_SELF_REF_HINTS = ("我", "自己", "本人", "我自己", "自己的")
_TECHNICAL_HINTS = (
    "nonebot",
    "插件",
    "代码",
    "脚本",
    "函数",
    "类",
    "接口",
    "报错",
    "bug",
    "错误",
    "调试",
    "配置",
    "开发",
    "仓库",
    "git",
)
_WEAK_ROUTE_HINTS = (
    "帮我",
    "给我",
    "请",
    "麻烦",
    "查看",
    "看看",
    "看下",
    "查询",
    "使用",
    "发送",
)
_EXECUTE_NEED_ARG_HINTS = (
    "做个",
    "做一个",
    "做一张",
    "来个",
    "来一个",
    "来一张",
    "生成",
    "制作",
)


@dataclass(frozen=True)
class IntentClassification:
    kind: IntentKind
    reason: str
    explicit_command: bool = False
    plugin_name: str | None = None
    plugin_module: str | None = None
    command_head: str | None = None
    payload_text: str = ""
    schema: SkillCommandSchema | None = None
    confidence: float = 0.0
    schema_state: IntentSchemaState = "unknown"
    rewrite_command: str = ""


def _find_skill(
    knowledge_base: PluginKnowledgeBase,
    plugin_name: str,
    plugin_module: str,
) -> SkillSpec | None:
    registry = get_skill_registry(knowledge_base)
    for skill in registry.skills:
        if skill.plugin_name == plugin_name and skill.plugin_module == plugin_module:
            return skill
    return None


def _find_schema(skill: SkillSpec | None, command_head: str) -> SkillCommandSchema | None:
    if skill is None:
        return None
    normalized_head = normalize_message_text(command_head)
    if not normalized_head:
        return None
    for schema in skill.command_schemas:
        if normalize_message_text(schema.command) == normalized_head:
            return schema
        for alias in schema.aliases:
            if normalize_message_text(alias) == normalized_head:
                return schema
    return None


def _infer_usage_command(
    knowledge_base: PluginKnowledgeBase,
    normalized_message: str,
) -> tuple[str, str, str, SkillCommandSchema | None] | None:
    registry = get_skill_registry(knowledge_base)
    best: tuple[int, str, str, str, SkillCommandSchema | None] | None = None
    for skill in registry.skills:
        for schema in skill.command_schemas:
            candidates = [schema.command, *schema.aliases]
            for candidate in candidates:
                normalized_head = normalize_message_text(candidate)
                if not normalized_head:
                    continue
                if not normalized_message.startswith(normalized_head):
                    continue
                score = len(normalized_head)
                if best is None or score > best[0]:
                    best = (
                        score,
                        skill.plugin_name,
                        skill.plugin_module,
                        schema.command,
                        schema,
                    )
    if best is None:
        return None
    _, plugin_name, plugin_module, command_head, schema = best
    return plugin_name, plugin_module, command_head, schema


def _extract_payload_text(
    normalized_message: str,
    command_head: str,
    schema: SkillCommandSchema | None,
) -> str:
    normalized_head = normalize_message_text(command_head)
    if not normalized_message or not normalized_head:
        return ""

    payload = ""
    if match_command_head(normalized_message, normalized_head):
        parts = normalized_message.split(" ", 1)
        if len(parts) > 1 and normalize_message_text(parts[0]) == normalized_head:
            payload = normalize_message_text(parts[1])
    elif match_command_head_or_sticky(
        normalized_message,
        normalized_head,
        allow_sticky=bool(schema.allow_sticky_arg) if schema is not None else False,
    ):
        payload = normalize_message_text(normalized_message[len(normalized_head) :])

    if not payload:
        return ""

    placeholder_tokens = {
        normalize_message_text(token)
        for token in collect_placeholders(payload)
        if normalize_message_text(token)
    }
    text_tokens: list[str] = []
    for raw_token in payload.split(" "):
        token = normalize_message_text(raw_token)
        if not token or token in placeholder_tokens:
            continue
        text_tokens.append(token)
    return normalize_message_text(" ".join(text_tokens))


def _looks_like_chatty_sticky_payload(
    *,
    normalized_message: str,
    payload_text: str,
    schema: SkillCommandSchema | None,
) -> bool:
    if schema is None:
        return False
    if not bool(getattr(schema, "allow_sticky_arg", False)):
        return False
    if not payload_text or _has_structure_route_signal(normalized_message):
        return False
    if payload_text in _SELF_REF_HINTS:
        return False
    if normalized_message.endswith(("吗", "嘛", "么", "呢", "吧", "呀", "啦", "？", "?")):
        return True
    return bool(re.search(r"(怎么|如何|为什么|是不是|对不对)", normalized_message))


def _fallback_explicit_command(
    normalized_message: str,
    knowledge_base: PluginKnowledgeBase,
) -> tuple[str, str, str, SkillCommandSchema | None] | None:
    stripped = normalize_message_text(strip_invoke_prefix(normalized_message))
    best: tuple[float, str, str, str, SkillCommandSchema | None] | None = None
    for plugin in knowledge_base.plugins:
        for meta in plugin.command_meta or ():
            candidates = [meta.command, *(meta.aliases or ())]
            allow_sticky = bool(getattr(meta, "allow_sticky_arg", False))
            for candidate in candidates:
                head = normalize_message_text(candidate)
                if not head:
                    continue
                score = 0.0
                if match_command_head(stripped, head):
                    score = 240.0 + len(head)
                elif match_command_head(normalized_message, head):
                    score = 220.0 + len(head)
                elif match_command_head_or_sticky(
                    stripped,
                    head,
                    allow_sticky=allow_sticky,
                ):
                    score = 200.0 + len(head)
                elif match_command_head_or_sticky(
                    normalized_message,
                    head,
                    allow_sticky=allow_sticky,
                ):
                    score = 180.0 + len(head)
                elif match_command_head_fuzzy(
                    stripped,
                    head,
                    allow_sticky=allow_sticky,
                ):
                    score = 150.0 + len(head)
                elif match_command_head_fuzzy(
                    normalized_message,
                    head,
                    allow_sticky=allow_sticky,
                ):
                    score = 130.0 + len(head)
                if score <= 0:
                    continue
                candidate_result = (
                    score,
                    plugin.name,
                    plugin.module,
                    normalize_message_text(meta.command),
                    _find_schema(
                        _find_skill(knowledge_base, plugin.name, plugin.module),
                        normalize_message_text(meta.command),
                    )
                    or SkillCommandSchema(
                        command=normalize_message_text(meta.command),
                        aliases=tuple(
                            normalize_message_text(alias)
                            for alias in meta.aliases or ()
                            if normalize_message_text(alias)
                        ),
                        text_min=int(getattr(meta, "text_min", 0) or 0),
                        text_max=getattr(meta, "text_max", None),
                        image_min=int(getattr(meta, "image_min", 0) or 0),
                        image_max=getattr(meta, "image_max", None),
                        allow_at=bool(getattr(meta, "allow_at", False)),
                        actor_scope=str(
                            getattr(meta, "actor_scope", "allow_other")
                            or "allow_other"
                        ),
                        target_requirement=str(
                            getattr(meta, "target_requirement", "none") or "none"
                        ),
                        target_sources=tuple(
                            str(source).strip()
                            for source in getattr(meta, "target_sources", ()) or ()
                            if str(source).strip()
                        ),
                        allow_sticky_arg=allow_sticky,
                    ),
                )
                if best is None or candidate_result[0] > best[0]:
                    best = candidate_result
    if best is None:
        return None
    _, plugin_name, plugin_module, command_head, schema = best
    return plugin_name, plugin_module, command_head, schema


def _has_structure_route_signal(normalized_message: str) -> bool:
    return bool(_AT_TOKEN_PATTERN.search(normalized_message)) or bool(
        _IMAGE_TOKEN_PATTERN.search(normalized_message)
    )


def _contains_strong_route_action(normalized_message: str) -> bool:
    for word in ROUTE_ACTION_WORDS:
        if not word or word in _WEAK_ROUTE_HINTS:
            continue
        if word in normalized_message:
            return True
    return False


def _classify_non_explicit_intent(normalized_message: str) -> IntentClassification:
    has_structure_signal = _has_structure_route_signal(normalized_message)
    has_strong_action = _contains_strong_route_action(normalized_message)
    has_weak_action = contains_any(normalized_message, _WEAK_ROUTE_HINTS)
    has_technical_hint = contains_any(normalized_message, _TECHNICAL_HINTS)

    if has_technical_hint and not has_structure_signal and not has_strong_action:
        return IntentClassification(
            kind="chat",
            reason="technical_chat_request",
            confidence=0.96,
        )

    if has_structure_signal or has_strong_action:
        return IntentClassification(
            kind="ambiguous",
            reason="route_signal_without_command",
            confidence=0.62,
        )

    if has_weak_action:
        return IntentClassification(
            kind="chat",
            reason="weak_route_signal",
            confidence=0.72,
        )

    return IntentClassification(
        kind="chat",
        reason="no_route_signal",
        confidence=0.9,
    )


def _count_text_tokens(payload_text: str) -> int:
    normalized = normalize_message_text(payload_text)
    if not normalized:
        return 0
    return len([token for token in normalized.split(" ") if token])


def _classify_explicit_command(
    *,
    normalized_message: str,
    plugin_name: str,
    plugin_module: str,
    command_head: str,
    schema: SkillCommandSchema | None,
) -> IntentClassification:
    rewrite_command = rewrite_command_with_head(
        normalized_message,
        command_head,
        allow_sticky=bool(schema.allow_sticky_arg) if schema is not None else False,
    )
    has_at = bool(_AT_TOKEN_PATTERN.search(normalized_message))
    image_count = len(_IMAGE_TOKEN_PATTERN.findall(normalized_message))
    has_self_ref = contains_any(normalized_message, _SELF_REF_HINTS)
    payload_text = _extract_payload_text(
        rewrite_command or normalized_message,
        command_head,
        schema,
    )
    text_count = _count_text_tokens(payload_text)

    if _looks_like_chatty_sticky_payload(
        normalized_message=normalized_message,
        payload_text=payload_text,
        schema=schema,
    ):
        return IntentClassification(
            kind="chat",
            reason="sticky_payload_looks_chat",
            confidence=0.88,
            schema_state="unknown",
            rewrite_command=rewrite_command,
        )

    if schema is None:
        return IntentClassification(
            kind="execute",
            reason="explicit_command_without_schema",
            explicit_command=True,
            plugin_name=plugin_name,
            plugin_module=plugin_module,
            command_head=command_head,
            payload_text=payload_text,
            schema=None,
            confidence=0.8,
            schema_state="unknown",
            rewrite_command=rewrite_command,
        )

    policy = resolve_command_target_policy(schema)
    available_target_units = image_count
    if has_at and policy.allow_at:
        available_target_units += 1
    if not has_at and policy.allow_at and has_self_ref:
        available_target_units += 1

    if policy.target_requirement == "required" and available_target_units <= 0:
        return IntentClassification(
            kind="execute_need_arg",
            reason="explicit_missing_target",
            explicit_command=True,
            plugin_name=plugin_name,
            plugin_module=plugin_module,
            command_head=command_head,
            payload_text=payload_text,
            schema=schema,
            confidence=0.94,
            schema_state="missing_target",
            rewrite_command=rewrite_command,
        )

    image_min = max(int(schema.image_min or 0), 0)
    if image_min > available_target_units:
        return IntentClassification(
            kind="execute_need_arg",
            reason="explicit_missing_image_or_target",
            explicit_command=True,
            plugin_name=plugin_name,
            plugin_module=plugin_module,
            command_head=command_head,
            payload_text=payload_text,
            schema=schema,
            confidence=0.94,
            schema_state="missing_image",
            rewrite_command=rewrite_command,
        )

    text_min = max(int(schema.text_min or 0), 0)
    if text_min > text_count:
        return IntentClassification(
            kind="execute_need_arg",
            reason="explicit_missing_text",
            explicit_command=True,
            plugin_name=plugin_name,
            plugin_module=plugin_module,
            command_head=command_head,
            payload_text=payload_text,
            schema=schema,
            confidence=0.94,
            schema_state="missing_text",
            rewrite_command=rewrite_command,
        )

    return IntentClassification(
        kind="execute",
        reason="explicit_command_ready",
        explicit_command=True,
        plugin_name=plugin_name,
        plugin_module=plugin_module,
        command_head=command_head,
        payload_text=payload_text,
        schema=schema,
        confidence=0.98,
        schema_state="ready",
        rewrite_command=rewrite_command,
    )


def classify_message_intent(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> IntentClassification:
    normalized = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    if not normalized:
        return IntentClassification(
            kind="chat",
            reason="empty_message",
            confidence=1.0,
        )
    if has_negative_route_intent(normalized):
        return IntentClassification(
            kind="chat",
            reason="negative_route_intent",
            confidence=0.99,
        )

    search_result = skill_search(
        normalized,
        knowledge_base,
        include_usage=True,
        include_similarity=True,
    )
    explicit_command = search_result.fast_match
    fallback_explicit = None
    if explicit_command is None:
        fallback_explicit = _fallback_explicit_command(normalized, knowledge_base)
        if fallback_explicit is not None:
            explicit_command = fallback_explicit[:3]
    if is_usage_question(normalized):
        if explicit_command is not None:
            plugin_name, plugin_module, command_head = explicit_command
            skill = _find_skill(knowledge_base, plugin_name, plugin_module)
            fallback_schema = fallback_explicit[3] if fallback_explicit is not None else None
            return IntentClassification(
                kind="help",
                reason="usage_question_with_explicit_command",
                explicit_command=True,
                plugin_name=plugin_name,
                plugin_module=plugin_module,
                command_head=command_head,
                schema=_find_schema(skill, command_head) or fallback_schema,
                confidence=0.96,
                schema_state="ready",
                rewrite_command=rewrite_command_with_head(
                    normalized,
                    command_head,
                    allow_sticky=bool(
                        getattr(_find_schema(skill, command_head) or fallback_schema, "allow_sticky_arg", False)
                    ),
                ),
            )
        inferred_usage = _infer_usage_command(knowledge_base, normalized)
        if inferred_usage is not None:
            plugin_name, plugin_module, command_head, schema = inferred_usage
            return IntentClassification(
                kind="help",
                reason="usage_question_with_prefix_command",
                explicit_command=True,
                plugin_name=plugin_name,
                plugin_module=plugin_module,
                command_head=command_head,
                schema=schema,
                confidence=0.9,
                schema_state="ready",
                rewrite_command=rewrite_command_with_head(
                    normalized,
                    command_head,
                    allow_sticky=bool(getattr(schema, "allow_sticky_arg", False)),
                ),
            )
        return IntentClassification(
            kind="help",
            reason="usage_question",
            confidence=0.92,
        )

    if explicit_command is not None:
        plugin_name, plugin_module, command_head = explicit_command
        skill = _find_skill(knowledge_base, plugin_name, plugin_module)
        fallback_schema = fallback_explicit[3] if fallback_explicit is not None else None
        schema = _find_schema(skill, command_head) or fallback_schema
        return _classify_explicit_command(
            normalized_message=normalized,
            plugin_name=plugin_name,
            plugin_module=plugin_module,
            command_head=command_head,
            schema=schema,
        )

    fallback_intent = _classify_non_explicit_intent(normalized)
    if (
        fallback_intent.kind == "ambiguous"
        and contains_any(normalized, _EXECUTE_NEED_ARG_HINTS)
    ):
        return fallback_intent
    return fallback_intent


__all__ = [
    "IntentClassification",
    "IntentKind",
    "IntentSchemaState",
    "classify_message_intent",
]
