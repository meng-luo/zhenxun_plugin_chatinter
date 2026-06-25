"""Runtime-side approval confirmation for superuser Agent turns.

This keeps common approval confirmations out of the LLM loop:
tool -> approval_required -> user replies "确认/取消" -> runtime executes or
rejects the pending approval directly.  If the approval came from a paused
AgentRun, the runtime resumes that run with the approved action observation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types.models import ToolResult
from zhenxun.utils.message import MessageUtils

from ..artifact_store import compact_tool_result_output, summarize_artifact_text
from ..route_text import normalize_message_text
from .approval_store import (
    PendingApproval,
    consume_pending_approval,
    list_pending_approvals,
    reject_pending_approval,
)
from .toolsets.agent_run_tools import AgentRunResumeTool
from .toolsets.approval_tools import execute_approved_action

_APPROVAL_ID_RE = re.compile(r"\b[0-9a-fA-F]{8,16}\b")
_CONFIRM_PHRASES = {
    "y",
    "yes",
    "approve",
    "ok",
    "确认",
    "确认执行",
    "批准",
    "批准执行",
    "同意",
    "执行",
    "继续",
    "继续执行",
    "可以",
}
_REJECT_PHRASES = {
    "n",
    "no",
    "reject",
    "cancel",
    "取消",
    "拒绝",
    "不要",
    "不执行",
    "终止",
}


async def try_handle_runtime_approval(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
) -> bool:
    """Consume a pending approval confirmation without invoking the LLM."""

    user_id = str(session.user.id if session.user else "")
    session_key = str(session.group.id) if session.group else user_id
    if not user_id or not session_key:
        return False
    intent = _approval_intent(raw_message)
    if intent == "":
        return False

    approvals = list_pending_approvals(user_id=user_id, session_key=session_key)
    if not approvals:
        return False
    approval = _select_approval(raw_message, approvals)
    if approval is None:
        await _send_runtime_reply(
            bot=bot,
            event=event,
            session=session,
            text=(
                "当前有多个待确认操作，请带上 approval_id："
                + "、".join(item.approval_id for item in approvals[:6])
            ),
        )
        return True

    if intent == "reject":
        rejected = reject_pending_approval(
            approval_id=approval.approval_id,
            user_id=user_id,
            session_key=session_key,
        )
        await _send_runtime_reply(
            bot=bot,
            event=event,
            session=session,
            text=(
                "已取消待确认操作："
                + (
                    rejected.approval_id
                    if rejected is not None
                    else approval.approval_id
                )
            ),
        )
        return True

    consumed = consume_pending_approval(
        approval_id=approval.approval_id,
        user_id=user_id,
        session_key=session_key,
    )
    if consumed is None:
        await _send_runtime_reply(
            bot=bot,
            event=event,
            session=session,
            text="待确认操作已过期或不存在。",
        )
        return True

    actor = {"user_id": user_id, "session_key": session_key}
    action_result = await execute_approved_action(approval=consumed, actor=actor)
    run_id = _approval_run_id(consumed)
    if run_id and _should_resume_after_approval(action_result):
        resume_result = await _resume_agent_run(
            run_id=run_id,
            approval=consumed,
            action_result=action_result,
            actor=actor,
        )
        await _send_runtime_reply(
            bot=bot,
            event=event,
            session=session,
            text=_reply_from_resume_result(resume_result, fallback=action_result),
        )
        return True

    await _send_runtime_reply(
        bot=bot,
        event=event,
        session=session,
        text=_reply_from_tool_result(action_result),
    )
    return True


def has_runtime_approval_intent(raw_message: str) -> bool:
    return bool(_approval_intent(raw_message))


def _approval_intent(raw_message: str) -> str:
    text = normalize_message_text(raw_message).lower()
    if not text:
        return ""
    without_id = _APPROVAL_ID_RE.sub("", text).strip(" ：:，,。.")
    candidate = without_id or text
    if candidate in _CONFIRM_PHRASES:
        return "confirm"
    if candidate in _REJECT_PHRASES:
        return "reject"
    if any(candidate == f"{phrase}吧" for phrase in _CONFIRM_PHRASES):
        return "confirm"
    return ""


def _select_approval(
    raw_message: str,
    approvals: list[PendingApproval],
) -> PendingApproval | None:
    if len(approvals) == 1:
        return approvals[0]
    ids = {item.approval_id.lower(): item for item in approvals}
    for match in _APPROVAL_ID_RE.findall(raw_message):
        found = ids.get(match.lower())
        if found is not None:
            return found
    return None


def _approval_run_id(approval: PendingApproval) -> str:
    payload = approval.payload if isinstance(approval.payload, dict) else {}
    return normalize_message_text(
        str(payload.get("run_id") or payload.get("trace_id") or "")
    )


def _should_resume_after_approval(result: ToolResult) -> bool:
    output = result.output if isinstance(result.output, dict) else {}
    if bool(output.get("approval_required")):
        return False
    return True


async def _resume_agent_run(
    *,
    run_id: str,
    approval: PendingApproval,
    action_result: ToolResult,
    actor: dict[str, str],
) -> ToolResult:
    context = RunContext(
        session_id=actor["session_key"],
        extra={
            "actor_user_id": actor["user_id"],
            "agent_mode": "superuser_agent",
            "enable_agent_tools": True,
        },
    )
    return await AgentRunResumeTool().execute(
        context=context,
        run_id=run_id,
        resume_message=_resume_message(approval, action_result),
        max_steps=None,
        pre_resume_observation={
            "tool_call_id": f"approval:{approval.approval_id}",
            "tool_name": approval.action,
            "task_text": approval.reason or approval.action,
            "output": _compact_result_payload(action_result, trace_id=run_id),
        },
    )


def _resume_message(approval: PendingApproval, result: ToolResult) -> str:
    payload = _compact_result_payload(result, trace_id=_approval_run_id(approval))
    return (
        f"用户已确认 approval_id={approval.approval_id}，"
        f"已执行 pending action={approval.action}。"
        "以下是真实工具 observation，请基于它继续原任务：\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )


def _reply_from_resume_result(
    result: ToolResult,
    *,
    fallback: ToolResult,
) -> str:
    output = result.output if isinstance(result.output, dict) else {}
    final_text = normalize_message_text(str(output.get("final_text", "") or ""))
    if final_text:
        return final_text
    status = normalize_message_text(str(output.get("status", "") or ""))
    run_status = normalize_message_text(str(output.get("run_status", "") or ""))
    if status:
        return f"确认操作已执行，AgentRun 已继续：{run_status or status}。"
    return "确认操作已执行。\n" + _reply_from_tool_result(fallback)


def _reply_from_tool_result(result: ToolResult) -> str:
    payload = _compact_result_payload(result, trace_id="")
    status = normalize_message_text(str(payload.get("status", "") or ""))
    ok = bool(payload.get("ok"))
    parts = [f"确认操作已执行：{status or ('success' if ok else 'failed')}。"]
    summary = _payload_summary(payload)
    if summary:
        parts.append(summary)
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        ids = [
            normalize_message_text(str(item.get("artifact_id", "") or ""))
            for item in artifacts[:4]
            if isinstance(item, dict)
        ]
        ids = [item for item in ids if item]
        if ids:
            parts.append("大输出已保存为 artifact：" + "、".join(ids))
    return "\n".join(parts)


def _compact_result_payload(result: ToolResult, *, trace_id: str) -> dict[str, Any]:
    output = (
        result.output if isinstance(result.output, dict) else {"output": result.output}
    )
    return compact_tool_result_output(
        output,
        trace_id=trace_id,
        source="runtime_approval",
    )


def _payload_summary(payload: dict[str, Any]) -> str:
    for key in ("stdout", "stderr", "content", "error", "message"):
        value = payload.get(key)
        if not value:
            continue
        if isinstance(value, str):
            return summarize_artifact_text(value, limit=420)
    return ""


async def _send_runtime_reply(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    text: str,
) -> None:
    """Send approval replies even when handled from ChatInter's async queue."""

    try:
        await bot.send(event, text)
        return
    except Exception:
        pass
    try:
        await MessageUtils.build_message(text).send()
        return
    except Exception:
        pass
    group = session.group
    if group is not None:
        try:
            await bot.call_api(
                "send_msg",
                message_type="group",
                group_id=str(group.id),
                message=text,
            )
        except Exception:
            pass
        return
    user = session.user
    user_id = str(user.id if user else getattr(event, "user_id", "") or "")
    if not user_id:
        return
    try:
        await bot.call_api(
            "send_msg",
            message_type="private",
            user_id=user_id,
            message=text,
        )
    except Exception:
        pass


__all__ = ["has_runtime_approval_intent", "try_handle_runtime_approval"]
