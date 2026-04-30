"""
ChatInter 待续接路由。

这里不导入 NoneBot 和执行链，只把 TaskFrame 转成“可以继续尝试的路由意图”。
handler.py 负责把它包装成 RouteResolveResult 并交回原插件执行链。
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .route_text import collect_placeholders, contains_any, normalize_message_text
from .task_frame import TaskFrame, TaskFrameState, TaskFrameStore

_CONTEXT_TOKEN_PATTERN = re.compile(r"\[@[^\]]+\]|\[image(?:#\d+)?\]", re.IGNORECASE)
_FOLLOWUP_REUSE_HINTS = (
    "还是这个",
    "还是这条",
    "还是这一个",
    "还是上次那个",
    "继续",
    "照旧",
    "用上次那个",
    "用上次的",
    "上次那个",
    "上一个",
    "就这个",
    "就它",
    "再来一个",
    "再来一张",
    "再来个",
)
_FOLLOWUP_REWRITE_HINTS = (
    "改成",
    "改为",
    "换成",
    "变成",
    "替换成",
    "调整为",
    "设成",
    "设为",
)
_CANCEL_HINTS = ("算了", "不用了", "取消", "先不", "不要了", "停", "停止")
_TEXT_SLOT_NAMES = {"text", "文本", "文字", "参数", "内容", "message", "payload"}
_IMAGE_SLOT_NAMES = {"image", "图片", "图", "照片"}
_REPLY_SLOT_NAMES = {"reply", "回复", "引用"}


@dataclass(frozen=True)
class PendingRouteResolution:
    plugin_name: str
    plugin_module: str
    command: str
    route_message: str
    skill_kind: str
    reason: str
    state: TaskFrameState
    command_id: str | None = None
    slots: dict[str, object] | None = None


def _extract_context_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in collect_placeholders(text or ""):
        normalized = normalize_message_text(token)
        if normalized and normalized not in tokens:
            tokens.append(normalized)
    return tuple(tokens)


def _strip_context_tokens(text: str) -> str:
    normalized = normalize_message_text(text or "")
    if not normalized:
        return ""
    return normalize_message_text(_CONTEXT_TOKEN_PATTERN.sub(" ", normalized))


def _append_context_tokens(command: str, tokens: tuple[str, ...]) -> str:
    normalized = normalize_message_text(command)
    if not normalized or not tokens:
        return normalized
    existing = set(_extract_context_tokens(normalized))
    merged = [token for token in tokens if token and token not in existing]
    if not merged:
        return normalized
    return normalize_message_text(f"{normalized} {' '.join(merged)}")


def _missing_has(frame: TaskFrame, names: set[str]) -> bool:
    return any(
        normalize_message_text(slot).lower() in names for slot in frame.missing_slots
    )


def _extract_rewrite_payload(message_text: str) -> tuple[str, str]:
    normalized = normalize_message_text(message_text or "")
    if not normalized:
        return "", "empty"
    for marker in _FOLLOWUP_REWRITE_HINTS:
        if marker not in normalized:
            continue
        payload = normalize_message_text(normalized.split(marker, 1)[1])
        payload = _strip_context_tokens(payload)
        if payload:
            return payload, f"rewrite:{marker}"
        return "", f"rewrite:{marker}:empty"
    return "", "none"


def _build_resume_command(frame: TaskFrame, current_message: str) -> tuple[str, str]:
    normalized = normalize_message_text(current_message or "")
    if not normalized:
        return "", "empty"
    if contains_any(normalized, _CANCEL_HINTS):
        return "", "cancelled"

    context_tokens = (*frame.context_tokens, *_extract_context_tokens(normalized))
    rewrite_payload, rewrite_reason = _extract_rewrite_payload(normalized)
    if rewrite_payload:
        command = normalize_message_text(f"{frame.command_head} {rewrite_payload}")
        return _append_context_tokens(command, context_tokens), rewrite_reason

    if contains_any(normalized, _FOLLOWUP_REUSE_HINTS):
        return _append_context_tokens(frame.command, context_tokens), "reuse_previous"

    if frame.is_pending:
        payload = _strip_context_tokens(normalized)
        has_text_slot = _missing_has(frame, _TEXT_SLOT_NAMES)
        has_media_slot = _missing_has(frame, _IMAGE_SLOT_NAMES | _REPLY_SLOT_NAMES)
        if payload and (has_text_slot or not has_media_slot):
            command = normalize_message_text(f"{frame.command_head} {payload}")
        else:
            command = frame.command
        return _append_context_tokens(command, context_tokens), "pending_fill"

    if frame.is_executed and contains_any(normalized, _FOLLOWUP_REUSE_HINTS):
        return _append_context_tokens(frame.command, context_tokens), "reuse_previous"

    return "", "not_followup"


def remember_pending_route(
    *,
    session_id: str | None,
    plugin_module: str,
    plugin_name: str,
    command: str,
    command_head: str | None = None,
    command_id: str | None = None,
    source_message: str = "",
    skill_kind: str = "planner",
    missing_slots: tuple[str, ...] | list[str] = (),
    filled_slots: dict[str, object] | None = None,
    context_tokens: tuple[str, ...] | list[str] = (),
    state: TaskFrameState = "pending",
) -> TaskFrame | None:
    return TaskFrameStore.remember(
        session_id=session_id,
        plugin_module=plugin_module,
        plugin_name=plugin_name,
        command=command,
        command_head=command_head,
        command_id=command_id,
        source_message=source_message,
        skill_kind=skill_kind,
        missing_slots=missing_slots,
        filled_slots=filled_slots,
        context_tokens=context_tokens,
        state=state,
    )


def get_pending_route_frame(session_id: str | None) -> TaskFrame | None:
    return TaskFrameStore.get(session_id)


def discard_pending_route(session_id: str | None) -> TaskFrame | None:
    return TaskFrameStore.pop(session_id)


def try_resume_pending_route(
    *,
    session_id: str | None,
    current_message: str,
) -> PendingRouteResolution | None:
    frame = TaskFrameStore.get(session_id)
    if frame is None:
        return None
    command, reason = _build_resume_command(frame, current_message)
    if reason == "cancelled":
        TaskFrameStore.pop(session_id)
        return None
    if not command:
        return None

    if frame.is_pending:
        TaskFrameStore.pop(session_id)

    # Feed both the original request and the follow-up into schema slot completion.
    # This keeps "发4个红包" -> "20金币" on the same command_id with prior slots.
    route_message = normalize_message_text(
        f"{frame.source_message} {current_message} {' '.join(frame.context_tokens)}"
    )
    return PendingRouteResolution(
        plugin_name=frame.plugin_name,
        plugin_module=frame.plugin_module,
        command=command,
        route_message=route_message or current_message,
        skill_kind=frame.skill_kind or "pending",
        reason=reason,
        state=frame.state,
        command_id=frame.command_id,
        slots=dict(frame.filled_slots),
    )
