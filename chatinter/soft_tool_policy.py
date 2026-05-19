"""Soft command exposure policy.

Soft commands are low-context command tools that are easy for a model to call
accidentally: they need no user-provided payload, no media/reply target, and no
required slots.  They stay discoverable, but become executable only when the
turn has an explicit tool intent or the gate selected them.
"""

from __future__ import annotations

from typing import Any

from .route_text import normalize_message_text, strip_invoke_prefix

_EXPLICIT_ACTION_TERMS = (
    "帮我",
    "给我",
    "请",
    "麻烦",
    "来个",
    "来一",
    "发个",
    "发一",
    "发送",
    "抽个",
    "抽一",
    "抽张",
    "随机",
    "roll",
    "掷",
    "投",
    "选一个",
    "决定",
    "查一下",
    "查下",
    "查询",
    "查看",
    "看看",
    "看一下",
    "看下",
    "介绍",
    "打开",
    "调用",
    "执行",
    "使用",
    "生成",
    "制作",
    "签到",
    "打卡",
    "有哪些",
    "列表",
    "说明",
    "帮助",
    "版本",
    "文档",
    "项目地址",
)


def is_soft_command_candidate(candidate: Any) -> bool:
    """Return whether a recalled command should be explicit-request-only."""

    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    return is_soft_command_schema(schema, snapshot=snapshot)


def is_soft_command_schema(schema: Any, *, snapshot: Any | None = None) -> bool:
    """Classify low-context tools without using plugin-specific names."""

    if schema is None:
        return False
    if _has_required_user_input(schema, snapshot=snapshot):
        return False
    role = normalize_message_text(str(getattr(schema, "command_role", "") or ""))
    if role in {"catalog"}:
        return False
    return True


def soft_tool_policy_reason(candidate: Any) -> str:
    if not is_soft_command_candidate(candidate):
        return "normal_tool"
    schema = getattr(candidate, "schema", None)
    role = normalize_message_text(str(getattr(schema, "command_role", "") or ""))
    if role:
        return f"explicit_request_only:{role}:low_context"
    return "explicit_request_only:low_context"


def soft_tool_allowed_for_message(
    message_text: str,
    candidate: Any,
    *,
    selected_command_ids: set[str] | frozenset[str] | None = None,
) -> bool:
    """Return whether a soft candidate may be exposed as executable."""

    if not is_soft_command_candidate(candidate):
        return True
    schema = getattr(candidate, "schema", None)
    command_id = normalize_message_text(str(getattr(schema, "command_id", "") or ""))
    if command_id and selected_command_ids and command_id in selected_command_ids:
        return True
    return has_explicit_soft_tool_request(message_text, candidate)


def has_explicit_soft_tool_request(message_text: str, candidate: Any) -> bool:
    """Check generic explicit request signals for a specific soft command."""

    normalized = normalize_message_text(message_text)
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if not stripped:
        return False

    phrases = _candidate_phrases(candidate)
    if any(_matches_exact_command(stripped, phrase) for phrase in phrases):
        return True

    explicit_action = any(term in stripped for term in _EXPLICIT_ACTION_TERMS)
    if not explicit_action:
        return False

    if any(_phrase_mentions_command(stripped, phrase) for phrase in phrases):
        return True
    return float(getattr(candidate, "score", 0.0) or 0.0) > 0


def filter_soft_candidates(
    message_text: str,
    candidates: list[Any],
    *,
    selected_command_ids: set[str] | frozenset[str] | None = None,
) -> list[Any]:
    """Remove implicit soft candidates from an executable exposure set."""

    return [
        candidate
        for candidate in candidates
        if soft_tool_allowed_for_message(
            message_text,
            candidate,
            selected_command_ids=selected_command_ids,
        )
    ]


def _has_required_user_input(schema: Any, *, snapshot: Any | None = None) -> bool:
    requires = dict(getattr(schema, "requires", {}) or {})
    if snapshot is not None:
        requires.update(dict(getattr(snapshot, "requires", {}) or {}))
    hard_requires = ("text", "image", "reply", "at", "private", "to_me")
    if any(bool(requires.get(key)) for key in hard_requires):
        return True

    payload_policy = normalize_message_text(
        str(getattr(schema, "payload_policy", "") or "")
    )
    if payload_policy not in {"", "none"}:
        return True

    if any(bool(getattr(slot, "required", False)) for slot in _schema_slots(schema)):
        return True

    target_requirement = normalize_message_text(
        str(getattr(schema, "target_requirement", "") or "")
    )
    if target_requirement == "required":
        return True
    if bool(getattr(schema, "allow_at", False)):
        return True
    return False


def _schema_slots(schema: Any) -> list[Any]:
    slots = getattr(schema, "slots", []) or []
    return list(slots) if isinstance(slots, list | tuple) else []


def _candidate_phrases(candidate: Any) -> tuple[str, ...]:
    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    values: list[Any] = [
        getattr(schema, "head", ""),
        *list(getattr(schema, "aliases", []) or []),
    ]
    if snapshot is not None:
        values.extend(list(getattr(snapshot, "examples", []) or [])[:4])
    phrases: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text and text not in phrases:
            phrases.append(text)
    return tuple(phrases)


def _matches_exact_command(text: str, phrase: str) -> bool:
    normalized = normalize_message_text(text)
    target = normalize_message_text(phrase)
    if not normalized or not target:
        return False
    return normalized == target


def _phrase_mentions_command(text: str, phrase: str) -> bool:
    normalized = normalize_message_text(text)
    target = normalize_message_text(phrase)
    if not normalized or not target:
        return False
    if target in normalized:
        return True
    return False


__all__ = [
    "filter_soft_candidates",
    "has_explicit_soft_tool_request",
    "is_soft_command_candidate",
    "is_soft_command_schema",
    "soft_tool_allowed_for_message",
    "soft_tool_policy_reason",
]
