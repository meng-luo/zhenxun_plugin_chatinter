from dataclasses import dataclass
from typing import Literal

from .config import LLM_VERIFY_ALL_ROUTES
from .intent_classifier import IntentClassification
from .route_text import (
    ROUTE_ACTION_WORDS,
    contains_any,
    has_negative_route_intent,
    match_command_head,
    match_command_head_canonical,
    normalize_message_text,
    normalize_action_phrases,
    strip_invoke_prefix,
)
from .skill_registry import infer_command_role

RoutePolicyAction = Literal[
    "direct",
    "align",
    "usage",
    "chat",
]

_CHAT_DIALOGUE_SUBKINDS = {
    "recap",
    "identity_query",
    "memory_confirm",
    "explain_context",
}
_CREATE_HINTS = (
    "发",
    "塞",
    "创建",
    "新增",
    "生成",
    "制作",
    "上传",
    "设置",
    "绑定",
    "添加",
    "开启",
)
_OPEN_HINTS = (
    "开",
    "抢",
    "领取",
    "领",
    "抽",
    "抽签",
)
_RETURN_HINTS = (
    "退回",
    "退还",
    "删除",
    "取消",
    "解绑",
    "关闭",
)
_QUERY_HINTS = (
    "查",
    "搜",
    "搜索",
    "查询",
    "查看",
    "识别",
    "是什么",
    "今天",
    "今日",
    "本日",
    "当日",
    "看看",
    "看下",
)
_HELP_HINTS = (
    "帮助",
    "怎么用",
    "如何用",
    "怎样用",
    "怎么使用",
    "如何使用",
    "怎样使用",
    "用法",
    "教程",
    "参数",
    "说明",
    "示例",
    "例子",
    "详情",
)


@dataclass(frozen=True)
class RoutePolicyDecision:
    action: RoutePolicyAction
    reason: str
    message_role: str = "other"
    route_role: str = "other"


def infer_message_action_role(message_text: str) -> str:
    normalized = normalize_message_text(strip_invoke_prefix(message_text or ""))
    if not normalized:
        return "other"
    if contains_any(normalized, _HELP_HINTS):
        return "help"
    if contains_any(normalized, _RETURN_HINTS):
        return "return"
    if contains_any(normalized, _CREATE_HINTS):
        return "create"
    if contains_any(normalized, _OPEN_HINTS):
        return "open"
    if contains_any(normalized, _QUERY_HINTS):
        return "query"
    return "other"


def _route_command_has_payload(route_command: str) -> bool:
    normalized = normalize_message_text(route_command or "")
    if not normalized:
        return False
    parts = normalized.split(" ", 1)
    return len(parts) > 1 and bool(normalize_message_text(parts[1]))


def infer_route_action_role(route_command: str) -> str:
    command_head = normalize_message_text((route_command or "").split(" ", 1)[0])
    if not command_head:
        return "other"
    return infer_command_role(command_head, family="general")


def _route_command_head(route_command: str) -> str:
    normalized = normalize_message_text(route_command or "")
    if not normalized:
        return ""
    return normalize_message_text(normalized.split(" ", 1)[0])


def is_exact_standard_command_head(
    message_text: str,
    route_command: str,
) -> bool:
    normalized_message = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    command_head = _route_command_head(route_command)
    if not normalized_message or not command_head:
        return False
    return match_command_head(normalized_message, command_head)


def is_canonical_standard_command_head(
    message_text: str,
    route_command: str,
) -> bool:
    normalized_message = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    command_head = _route_command_head(route_command)
    if not normalized_message or not command_head:
        return False
    return match_command_head_canonical(normalized_message, command_head)


def _is_pure_command_message(message_text: str, route_command: str) -> bool:
    """Check if the message is exactly the command head with no trailing text.

    Returns True when the user typed just the command word (e.g. "签到"),
    meaning there is zero ambiguity and LLM verification can be skipped.
    """
    stripped = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    cmd_head = _route_command_head(route_command)
    if not stripped or not cmd_head:
        return False
    return stripped.casefold() == cmd_head.casefold()


def is_route_action_compatible(message_text: str, route_command: str) -> bool:
    message_role = infer_message_action_role(message_text)
    route_role = infer_route_action_role(route_command)
    if message_role == "other":
        return True
    if message_role == "help":
        return route_role == "help"
    if message_role == "create":
        return route_role in {"create", "other"}
    if message_role == "open":
        return route_role in {"open", "other"}
    if message_role == "return":
        return route_role in {"return", "other"}
    if message_role == "query":
        return route_role in {"query", "catalog", "other"}
    return True


def should_try_shortlist_alignment(
    *,
    message_text: str,
    intent_profile: IntentClassification,
) -> bool:
    if intent_profile.kind in {"execute", "execute_need_arg", "help"}:
        return True
    if intent_profile.reason in {
        "weak_route_signal",
        "family_route_signal_without_command",
        "usage_question_with_explicit_command",
    }:
        return True
    normalized = normalize_message_text(strip_invoke_prefix(message_text or ""))
    if not normalized:
        return False
    if has_negative_route_intent(normalized):
        return False
    if infer_message_action_role(normalized) != "other":
        return True
    return contains_any(normalized, ROUTE_ACTION_WORDS)


def decide_route_policy(
    *,
    message_text: str,
    intent_profile: IntentClassification,
    shortlist_route_result=None,
) -> RoutePolicyDecision:
    message_role = infer_message_action_role(message_text)

    if intent_profile.chat_subkind in _CHAT_DIALOGUE_SUBKINDS:
        return RoutePolicyDecision(
            action="chat",
            reason=f"dialogue_{intent_profile.chat_subkind}",
            message_role=message_role,
        )

    if intent_profile.reason == "weak_route_signal":
        return RoutePolicyDecision(
            action="chat",
            reason="weak_route_signal_chat",
            message_role=message_role,
        )

    if shortlist_route_result is not None:
        route_command = str(shortlist_route_result.decision.command or "")
        route_role = infer_route_action_role(route_command)
        exact_standard_head = is_exact_standard_command_head(
            message_text,
            route_command,
        )
        canonical_standard_head = is_canonical_standard_command_head(
            message_text,
            route_command,
        )
        if intent_profile.kind in {"help", "execute_need_arg"} or message_role == "help":
            return RoutePolicyDecision(
                action="usage",
                reason="explicit_usage",
                message_role=message_role,
                route_role=route_role,
            )
        if not exact_standard_head:
            if canonical_standard_head:
                if not is_route_action_compatible(message_text, route_command):
                    return RoutePolicyDecision(
                        action="align",
                        reason="shortlist_canonical_action_mismatch",
                        message_role=message_role,
                        route_role=route_role,
                    )
                if route_role == "create" and not _route_command_has_payload(
                    route_command
                ):
                    return RoutePolicyDecision(
                        action="usage",
                        reason="shortlist_create_needs_usage",
                        message_role=message_role,
                        route_role=route_role,
                    )
                if LLM_VERIFY_ALL_ROUTES and not _is_pure_command_message(
                    message_text, route_command
                ):
                    return RoutePolicyDecision(
                        action="align",
                        reason="llm_verify_canonical_route",
                        message_role=message_role,
                        route_role=route_role,
                    )
                return RoutePolicyDecision(
                    action="direct",
                    reason="shortlist_canonical_direct",
                    message_role=message_role,
                    route_role=route_role,
                )
            if should_try_shortlist_alignment(
                message_text=message_text,
                intent_profile=intent_profile,
            ):
                return RoutePolicyDecision(
                    action="align",
                    reason="shortlist_route_nonstandard_head",
                    message_role=message_role,
                    route_role=route_role,
                )
            return RoutePolicyDecision(
                action="chat",
                reason="shortlist_nonstandard_chat",
                message_role=message_role,
                route_role=route_role,
            )
        if not is_route_action_compatible(message_text, route_command):
            return RoutePolicyDecision(
                action="align",
                reason="shortlist_action_mismatch",
                message_role=message_role,
                route_role=route_role,
            )
        if route_role == "create" and not _route_command_has_payload(route_command):
            return RoutePolicyDecision(
                action="usage",
                reason="shortlist_create_needs_usage",
                message_role=message_role,
                route_role=route_role,
            )
        if LLM_VERIFY_ALL_ROUTES and not _is_pure_command_message(
            message_text, route_command
        ):
            return RoutePolicyDecision(
                action="align",
                reason="llm_verify_shortlist_route",
                message_role=message_role,
                route_role=route_role,
            )
        return RoutePolicyDecision(
            action="direct",
            reason="shortlist_direct",
            message_role=message_role,
            route_role=route_role,
        )

    exact_standard_head = bool(
        intent_profile.explicit_command
        and intent_profile.command_head
        and is_exact_standard_command_head(
            message_text,
            str(intent_profile.command_head),
        )
    )
    canonical_standard_head = bool(
        intent_profile.explicit_command
        and intent_profile.command_head
        and is_canonical_standard_command_head(
            message_text,
            str(intent_profile.command_head),
        )
    )

    if intent_profile.kind in {"help", "execute_need_arg"} and (
        intent_profile.explicit_command or intent_profile.command_head
    ):
        return RoutePolicyDecision(
            action="usage",
            reason="explicit_usage_no_shortlist",
            message_role=message_role,
        )

    if exact_standard_head and intent_profile.kind == "execute":
        return RoutePolicyDecision(
            action="direct",
            reason="explicit_exact_standard_head",
            message_role=message_role,
        )

    if canonical_standard_head and intent_profile.kind == "execute":
        return RoutePolicyDecision(
            action="direct",
            reason="explicit_canonical_standard_head",
            message_role=message_role,
        )

    if should_try_shortlist_alignment(
        message_text=message_text,
        intent_profile=intent_profile,
    ):
        return RoutePolicyDecision(
            action="align",
            reason="plugin_like_signal",
            message_role=message_role,
        )

    return RoutePolicyDecision(
        action="chat",
        reason="chat_default",
        message_role=message_role,
    )


__all__ = [
    "RoutePolicyAction",
    "RoutePolicyDecision",
    "decide_route_policy",
    "infer_message_action_role",
    "infer_route_action_role",
    "is_canonical_standard_command_head",
    "is_exact_standard_command_head",
    "is_route_action_compatible",
    "should_try_shortlist_alignment",
]
