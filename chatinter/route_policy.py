from .route_text import (
    contains_any,
    match_command_head,
    match_command_head_canonical,
    normalize_action_phrases,
    normalize_message_text,
    strip_invoke_prefix,
)
from .skill_registry import infer_command_role

_CREATE_HINTS = (
    "\u53d1",
    "\u585e",
    "\u521b\u5efa",
    "\u65b0\u589e",
    "\u751f\u6210",
    "\u5236\u4f5c",
    "\u4e0a\u4f20",
    "\u8bbe\u7f6e",
    "\u7ed1\u5b9a",
    "\u6dfb\u52a0",
    "\u5f00\u542f",
)
_OPEN_HINTS = ("\u5f00", "\u62a2", "\u9886\u53d6", "\u9886", "\u62bd", "\u62bd\u7b7e")
_RETURN_HINTS = (
    "\u9000\u56de",
    "\u9000\u8fd8",
    "\u5220\u9664",
    "\u53d6\u6d88",
    "\u89e3\u7ed1",
    "\u5173\u95ed",
)
_QUERY_HINTS = (
    "\u67e5",
    "\u641c",
    "\u641c\u7d22",
    "\u67e5\u8be2",
    "\u67e5\u770b",
    "\u8bc6\u522b",
    "\u662f\u4ec0\u4e48",
    "\u4eca\u5929",
    "\u4eca\u65e5",
    "\u672c\u65e5",
    "\u5f53\u65e5",
    "\u770b\u770b",
    "\u770b\u4e0b",
)
_HELP_HINTS = (
    "\u5e2e\u52a9",
    "\u600e\u4e48\u7528",
    "\u5982\u4f55\u7528",
    "\u600e\u6837\u7528",
    "\u600e\u4e48\u4f7f\u7528",
    "\u5982\u4f55\u4f7f\u7528",
    "\u600e\u6837\u4f7f\u7528",
    "\u7528\u6cd5",
    "\u6559\u7a0b",
    "\u53c2\u6570",
    "\u8bf4\u660e",
    "\u793a\u4f8b",
    "\u4f8b\u5b50",
    "\u8be6\u60c5",
)


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


def is_exact_standard_command_head(message_text: str, route_command: str) -> bool:
    normalized_message = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    command_head = _route_command_head(route_command)
    if not normalized_message or not command_head:
        return False
    return match_command_head(normalized_message, command_head)


def is_canonical_standard_command_head(message_text: str, route_command: str) -> bool:
    normalized_message = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text or ""))
    )
    command_head = _route_command_head(route_command)
    if not normalized_message or not command_head:
        return False
    return match_command_head_canonical(normalized_message, command_head)


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


__all__ = [
    "infer_message_action_role",
    "infer_route_action_role",
    "is_canonical_standard_command_head",
    "is_exact_standard_command_head",
    "is_route_action_compatible",
]
