"""Result construction helpers for ChatInter main requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
from inspect import isawaitable
from typing import Any

from .llm_compat import ToolResult
from .main_request_models import (
    MainRequestOutput,
    MainRequestReplyHook,
    MainRequestResult,
    MainRequestRouteHook,
    MainRequestTimelineItem,
)
from .native_executor import NativeToolExecutionResult
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
    NativeRouteResult,
)
from .plugin_outcome import (
    PluginOutcome,
    plugin_results_have_visible_output,
    plugin_results_own_delivery,
    plugin_terminal_reply,
)
from .response_defaults import (
    EMPTY_REPLY_TEXT,
    PLUGIN_FAILURE_REPLY_TEXT,
    PLUGIN_SUCCESS_REPLY_TEXT,
)
from .route_text import normalize_message_text, normalize_reply_text

_MAIN_STAGE = "main_request"
_RECEIPT_OUTCOME_TEXT = {
    "executed": "已完成",
    "needs_input": "未执行",
    "not_executed": "未执行",
    "uncertain": "结果不确定",
}


def _fallback_result(
    *,
    report: NativeRouteReport,
    reason: str,
    reply: str,
    timeline: list[MainRequestTimelineItem] | None = None,
) -> MainRequestResult:
    decision = NativeRouteDecision(action="chat", confidence=0.0, reason=reason)
    report.finalize(reason=reason, stage=_MAIN_STAGE)
    return MainRequestResult(
        decision=decision,
        route_result=None,
        report=report,
        timeline=(
            *(timeline or []),
            MainRequestTimelineItem(
                role="system",
                kind="fallback",
                content=reason,
            ),
        ),
        output=MainRequestOutput(final_text=reply, memory_text=reply),
    )


async def _finalize_result(
    result: MainRequestResult,
    *,
    route_completed_hook: MainRequestRouteHook | None,
    reply_hook: MainRequestReplyHook | None,
) -> MainRequestResult:
    if route_completed_hook is not None:
        maybe_awaitable = route_completed_hook(result)
        if maybe_awaitable is not None:
            await maybe_awaitable

    output = result.output
    replay_as_assistant = _is_model_chat_result(result)
    if not output.should_send:
        terminal_results = _terminal_tool_results(result)
        if plugin_results_own_delivery(terminal_results):
            return result
        if plugin_results_have_visible_output(terminal_results):
            return result
        if output.outcome == "chat_completed" and not terminal_results:
            final_text = EMPTY_REPLY_TEXT
        else:
            final_text = plugin_terminal_reply(
                _terminal_plugin_outcome(result),
                terminal_results,
            )
        result = replace(
            result,
            output=replace(
                output,
                final_text=final_text,
                memory_text=normalize_message_text(output.memory_text) or final_text,
                should_send=True,
            ),
        )
        output = result.output

    final_text = normalize_reply_text(output.final_text)
    if not final_text and not output.nontext_delivery:
        final_text = _fallback_final_reply(list(result.executions)) or EMPTY_REPLY_TEXT
    if reply_hook is not None and final_text:
        maybe_reply = reply_hook(final_text)
        final_text = (
            await maybe_reply if isawaitable(maybe_reply) else str(maybe_reply or "")
        )
    final_text = normalize_reply_text(final_text)
    if not final_text and not output.nontext_delivery:
        final_text = EMPTY_REPLY_TEXT
    if not final_text and output.nontext_delivery:
        return replace(
            result,
            output=replace(
                output,
                final_text="",
                memory_text=normalize_message_text(output.memory_text),
                should_send=True,
            ),
        )
    final_timeline = _with_final_timeline(
        result.timeline,
        final_text=final_text,
        should_send=True,
        replay_as_assistant=replay_as_assistant,
    )
    memory_text = (
        final_text
        if replay_as_assistant
        else _timeline_memory_text(
            list(final_timeline),
            include_final_output=False,
        )
    )
    return replace(
        result,
        timeline=final_timeline,
        output=replace(
            output,
            final_text=final_text,
            memory_text=memory_text,
            should_send=True,
        ),
    )


def _first_route(
    executions: list[NativeToolExecutionResult],
) -> NativeRouteResult | None:
    for execution in executions:
        if execution.route_result is not None:
            return execution.route_result
    return None


def _terminal_tool_results(result: MainRequestResult) -> tuple[ToolResult, ...]:
    return result.tool_results


def _terminal_plugin_outcome(result: MainRequestResult) -> PluginOutcome:
    tool_outcome = normalize_message_text(result.output.tool_outcome).casefold()
    if tool_outcome == "needs_input":
        return PluginOutcome("needs_input")
    if tool_outcome == "uncertain":
        return PluginOutcome("uncertain")
    if tool_outcome == "executed" or result.output.outcome == "tool_completed":
        return PluginOutcome("executed")
    return PluginOutcome("not_executed", reason=tool_outcome or "not_executed")


def _fallback_final_reply(executions: list[NativeToolExecutionResult]) -> str:
    if not executions:
        return ""
    success_count = sum(1 for item in executions if item.success)
    latest = executions[-1]
    if latest.display_text:
        return latest.display_text
    if success_count:
        return PLUGIN_SUCCESS_REPLY_TEXT
    message = str(latest.output.get("error", "") or latest.reason or "").strip()
    return message or PLUGIN_FAILURE_REPLY_TEXT


def _timeline_memory_text(
    timeline: list[MainRequestTimelineItem] | tuple[MainRequestTimelineItem, ...],
    *,
    fallback: str = "",
    include_final_output: bool = True,
) -> str:
    lines = list(_timeline_action_receipts(timeline))
    if include_final_output:
        for item in timeline:
            if item.role != "assistant" or item.kind != "final_output":
                continue
            text = normalize_message_text(item.content)
            if text:
                lines.append(text)
        fallback_text = normalize_message_text(fallback)
        if fallback_text and fallback_text not in lines:
            lines.append(fallback_text)
    return "\n".join(dict.fromkeys(line for line in lines if line))[:4000]


def _is_model_chat_result(result: MainRequestResult) -> bool:
    if result.output.outcome != "chat_completed":
        return False
    if result.executions or not result.output.record_chat_feedback:
        return False
    if any(item.kind == "fallback" for item in result.timeline):
        return False
    return bool(normalize_message_text(result.output.final_text))


def _sync_visible_chat_result(
    result: MainRequestResult,
    *,
    final_text: str,
) -> MainRequestResult:
    if not _is_model_chat_result(result):
        return result
    visible_text = normalize_reply_text(final_text)
    if not visible_text:
        return result
    timeline = list(result.timeline)
    for index in range(len(timeline) - 1, -1, -1):
        item = timeline[index]
        if item.role == "assistant" and item.kind == "final_output":
            metadata = dict(item.metadata)
            metadata["assistant_history"] = True
            timeline[index] = replace(
                item,
                content=visible_text,
                metadata=metadata,
            )
            break
    else:
        timeline.extend(
            _with_final_timeline(
                (),
                final_text=visible_text,
                should_send=True,
                replay_as_assistant=True,
            )
        )
    return replace(
        result,
        timeline=tuple(timeline),
        output=replace(
            result.output,
            final_text=visible_text,
            memory_text=visible_text,
        ),
    )


def _timeline_action_receipts(
    timeline: Sequence[MainRequestTimelineItem | Mapping[str, Any]],
    *,
    requester: str = "",
) -> tuple[str, ...]:
    del requester
    receipts: list[str] = []
    for item in timeline:
        if _timeline_item_value(item, "kind") != "tool_result":
            continue
        metadata = _timeline_item_metadata(item)
        output_value = metadata.get("output")
        output = output_value if isinstance(output_value, Mapping) else {}
        execution_value = metadata.get("execution")
        execution = execution_value if isinstance(execution_value, Mapping) else {}
        if not _is_plugin_action_result(output, execution=execution):
            continue
        action = _receipt_action(output, execution=execution)
        target = _receipt_target(output, execution=execution)
        outcome = _receipt_outcome(execution=execution)
        receipts.append(f"用户请求执行{action}；目标：{target}；结果：{outcome}。")
    return tuple(receipts)


def _is_plugin_action_result(
    output: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
) -> bool:
    tool_kind = normalize_message_text(
        str(execution.get("tool_kind", "") or "")
    ).casefold()
    if tool_kind in {"native_command", "skill_dispatch"}:
        return True
    if "plugin_outcome" in execution:
        return True
    return any(
        key in output
        for key in (
            "matched_plugin",
            "command_id",
        )
    )


def _timeline_item_value(
    item: MainRequestTimelineItem | Mapping[str, Any],
    key: str,
) -> str:
    value = item.get(key, "") if isinstance(item, Mapping) else getattr(item, key, "")
    return normalize_message_text(str(value or ""))


def _timeline_item_metadata(
    item: MainRequestTimelineItem | Mapping[str, Any],
) -> Mapping[str, Any]:
    value = (
        item.get("metadata", {})
        if isinstance(item, Mapping)
        else getattr(item, "metadata", {})
    )
    return value if isinstance(value, Mapping) else {}


def _receipt_action(
    output: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
) -> str:
    plugin = _receipt_structured_text(output.get("matched_plugin"), limit=120)
    command_id = _receipt_structured_text(
        output.get("command_id"),
        limit=160,
    )
    if plugin and command_id:
        return f"插件“{plugin}”中的命令“{command_id}”"
    if plugin:
        return f"插件“{plugin}”"
    if command_id:
        return f"命令“{command_id}”"
    return "插件操作"


def _receipt_target(
    output: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
) -> str:
    value = execution.get("resolved_target") or output.get("resolved_target")
    identities = _resolved_target_identities(value)
    if not identities:
        return "未记录"
    return "、".join(identities)


def _resolved_target_identities(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list | tuple) else (value,)
    identities: list[str] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        raw_identity = item.get("user_id") or item.get("id")
        identity = _receipt_identity(raw_identity)
        if identity and identity not in identities:
            identities.append(identity)
    return tuple(identities)


def _receipt_outcome(
    *,
    execution: Mapping[str, Any],
) -> str:
    canonical = normalize_message_text(
        str(execution.get("plugin_outcome", "") or "")
    ).casefold()
    return _RECEIPT_OUTCOME_TEXT.get(canonical, "状态未记录")


def _receipt_structured_text(value: Any, *, limit: int) -> str:
    if value in (None, "") or isinstance(value, Mapping | list | tuple | set):
        return ""
    text = " ".join(normalize_message_text(str(value)).split())
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 1, 1)].rstrip()}…"


def _receipt_identity(value: Any) -> str:
    text = _receipt_structured_text(value, limit=240)
    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"已解析用户#{digest}"


def _user_timeline_item(message_text: str) -> MainRequestTimelineItem:
    return MainRequestTimelineItem(
        role="user",
        kind="current_user",
        content=message_text,
    )


def _with_final_timeline(
    timeline: tuple[MainRequestTimelineItem, ...],
    *,
    final_text: str,
    should_send: bool,
    replay_as_assistant: bool,
) -> tuple[MainRequestTimelineItem, ...]:
    if not final_text and not should_send:
        return timeline
    return (
        *timeline,
        MainRequestTimelineItem(
            role="assistant",
            kind="final_output",
            content=final_text,
            metadata={"assistant_history": replay_as_assistant},
        ),
    )


__all__ = [
    "_fallback_final_reply",
    "_fallback_result",
    "_finalize_result",
    "_is_model_chat_result",
    "_sync_visible_chat_result",
    "_timeline_action_receipts",
    "_timeline_memory_text",
    "_user_timeline_item",
]
