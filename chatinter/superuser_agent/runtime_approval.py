"""Runtime-side approval confirmation for superuser Agent turns.

This keeps permission decisions out of the LLM loop:
tool -> approval_required -> user allows or rejects -> runtime executes or
rejects the pending approval directly. If the approval came from a paused
AgentRun, the runtime resumes that run with the approved action observation.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.utils.message import MessageUtils

from ..artifact_store import compact_tool_result_output, summarize_artifact_text
from ..config import (
    SUPERUSER_MODEL_TIMEOUT_SECONDS,
    build_superuser_generation_config,
    get_agent_model,
)
from ..event_runtime import event_is_private, resolve_superuser
from ..llm_compat import (
    LLMMessage,
    LLMToolCall,
    LLMToolFunction,
    RunContext,
    ToolResult,
)
from ..provider_capability import ProviderCapabilityAdapter
from ..route_text import normalize_message_text, normalize_reply_text
from .approval_store import (
    PendingApproval,
    consume_pending_approval,
    list_pending_approvals,
    reject_pending_approval,
)
from .approved_actions import execute_approved_action, validate_approved_action
from .permission_policy import grant_conversation_permission
from .progress import AgentProgressReporter
from .runtime import (
    AgentRuntime,
    SuperuserSessionBusyError,
    superuser_session_execution,
)
from .state import AgentObservation, repair_interrupted_tool_protocol
from .store import (
    clear_agent_run_cancel_signal,
    get_active_agent_run_id,
    is_agent_run_cancel_signaled,
    load_agent_run_state,
    persist_agent_run_state,
)
from .tools import build_superuser_tools
from .tools.common import tool_result


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
    if not event_is_private(event) or not resolve_superuser(bot, user_id):
        return False
    intent, rejection_reason = _approval_decision(raw_message)
    if intent == "":
        return False

    try:
        async with superuser_session_execution(session_key):
            return await _handle_runtime_approval_locked(
                bot=bot,
                event=event,
                session=session,
                user_id=user_id,
                session_key=session_key,
                intent=intent,
                rejection_reason=rejection_reason,
            )
    except SuperuserSessionBusyError:
        await _send_runtime_reply(
            bot=bot,
            event=event,
            session=session,
            text="当前任务仍在执行，请先回复 /中断 或稍后确认。",
        )
        return True


async def _handle_runtime_approval_locked(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    user_id: str,
    session_key: str,
    intent: str,
    rejection_reason: str,
) -> bool:
    active_run_id = get_active_agent_run_id(session_key)
    approvals = [
        approval
        for approval in list_pending_approvals(
            user_id=user_id,
            session_key=session_key,
        )
        if _approval_run_id(approval) == active_run_id
    ]
    if not approvals:
        return False
    approval = approvals[0]

    if intent == "reject":
        rejected = reject_pending_approval(
            approval_id=approval.approval_id,
            user_id=user_id,
            session_key=session_key,
        )
        run_id = _approval_run_id(approval)
        if rejected is not None and run_id:
            rejection_result = ToolResult(
                output={
                    "ok": False,
                    "status": "permission_rejected_by_user",
                    "approval_id": rejected.approval_id,
                    "action": rejected.action,
                    "reason": rejection_reason or "用户拒绝了该操作",
                    "error": rejection_reason or "用户拒绝了该操作",
                },
                is_error=True,
                is_retryable=False,
            )
            resume_result = await _resume_agent_run(
                run_id=run_id,
                approval=rejected,
                action_result=rejection_result,
                actor={"user_id": user_id, "session_key": session_key},
                decision="reject",
                decision_reason=rejection_reason,
            )
            await _send_runtime_reply(
                bot=bot,
                event=event,
                session=session,
                text=_reply_from_resume_result(
                    resume_result,
                    fallback=rejection_result,
                ),
            )
            return True
        await _send_runtime_reply(
            bot=bot,
            event=event,
            session=session,
            text="已拒绝该操作。"
            + (f"理由：{rejection_reason}" if rejection_reason else ""),
        )
        return True

    actor = {"user_id": user_id, "session_key": session_key}
    validation_error = validate_approved_action(approval=approval, actor=actor)
    if validation_error is not None:
        consume_pending_approval(
            approval_id=approval.approval_id,
            user_id=user_id,
            session_key=session_key,
        )
        await _send_runtime_reply(
            bot=bot,
            event=event,
            session=session,
            text=_reply_from_tool_result(validation_error),
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

    run_id = _approval_run_id(consumed)
    if intent == "allow_conversation":
        granted = grant_conversation_permission(
            run_id,
            section=consumed.permission_section,
            grant_key=consumed.permission_grant_key,
        )
        if not granted:
            intent = "allow_once"

    async def send_progress(text: str) -> None:
        await _send_runtime_reply(
            bot=bot,
            event=event,
            session=session,
            text=text,
        )

    progress = AgentProgressReporter(send_progress)
    progress.start()
    progress.tool_started(consumed.action)
    try:
        action_result, result_persisted = await _execute_approved_action_durably(
            approval=consumed,
            actor=actor,
            decision=intent,
        )
    finally:
        await progress.tool_finished(consumed.action)
        await progress.stop()
    if run_id and result_persisted and _should_resume_after_approval(action_result):
        resume_result = await _resume_agent_run(
            run_id=run_id,
            approval=consumed,
            action_result=action_result,
            actor=actor,
            decision=intent,
            result_persisted=True,
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
    return bool(_approval_decision(raw_message)[0])


def _approval_intent(raw_message: str) -> str:
    return _approval_decision(raw_message)[0]


def _approval_decision(raw_message: str) -> tuple[str, str]:
    text = normalize_message_text(raw_message)
    if text == "/允许":
        return "allow_once", ""
    if text == "/本对话允许":
        return "allow_conversation", ""
    if text == "/拒绝":
        return "reject", ""
    if text.startswith("/拒绝 "):
        return "reject", text.removeprefix("/拒绝 ").strip()
    return "", ""


def _approval_run_id(approval: PendingApproval) -> str:
    payload = approval.payload if isinstance(approval.payload, dict) else {}
    return normalize_message_text(
        str(payload.get("run_id") or payload.get("trace_id") or "")
    )


def _should_resume_after_approval(result: ToolResult) -> bool:
    output = result.output if isinstance(result.output, dict) else {}
    if bool(output.get("approval_required")):
        return False
    if output.get("status") == "tool_execution_cancelled":
        return False
    return True


async def _execute_approved_action_durably(
    *,
    approval: PendingApproval,
    actor: dict[str, str],
    decision: str,
) -> tuple[ToolResult, bool]:
    run_id = _approval_run_id(approval)
    state = load_agent_run_state(run_id, tool_map={}) if run_id else None
    fingerprint = str(approval.payload_fingerprint or "")
    if state is None or not fingerprint:
        return _approval_execution_not_started(approval), False

    tool_call = LLMToolCall(
        id=f"approval:{approval.approval_id}",
        function=LLMToolFunction(name=approval.action, arguments="{}"),
    )
    if state.unsettled_tool_execution(fingerprint) is not None:
        return _approval_execution_uncertain(approval), False

    state.start_tool_execution(tool_call, fingerprint=fingerprint)
    if not persist_agent_run_state(
        state,
        stage="tool_execution_started",
        metadata={
            "approval_id": approval.approval_id,
            "tool_name": approval.action,
            "call_fingerprint": fingerprint,
        },
    ):
        return _approval_execution_not_started(approval), False

    action_result = await _execute_approved_action_with_cancel(
        approval=approval,
        actor=actor,
        run_id=run_id,
    )
    compact_output = {
        **_compact_result_payload(action_result, trace_id=run_id),
        "approval_id": approval.approval_id,
    }
    persisted_result = ToolResult(
        output=compact_output,
        display_content=action_result.display_content,
        is_error=action_result.is_error,
        is_retryable=action_result.is_retryable,
    )
    state.messages.append(
        LLMMessage.user(
            _resume_message(
                approval,
                persisted_result,
                decision=decision,
            )
        )
    )
    state.pending_approval = ""
    state.append_synthetic_observation(
        _approval_observation(
            approval=approval,
            tool_call=tool_call,
            tool_result=persisted_result,
            step=state.step,
        ),
        timeline_kind="approved_action_result",
        content=str(compact_output.get("status", "") or ""),
        metadata={"source": "runtime_approval"},
    )
    result_status = str(compact_output.get("status", "") or "")
    execution_status = (
        "cancelled" if result_status == "tool_execution_cancelled" else "completed"
    )
    state.settle_tool_execution(
        fingerprint=fingerprint,
        status=execution_status,
        result_status=result_status,
    )
    if execution_status == "cancelled":
        state.cancel(reason="cancelled_by_runtime_control")
    if not persist_agent_run_state(
        state,
        stage="tool_execution_completed",
        metadata={
            "approval_id": approval.approval_id,
            "tool_name": approval.action,
            "call_fingerprint": fingerprint,
            "result_status": result_status,
        },
    ):
        return _approval_execution_uncertain(approval), False
    return action_result, True


async def _execute_approved_action_with_cancel(
    *,
    approval: PendingApproval,
    actor: dict[str, str],
    run_id: str,
) -> ToolResult:
    task = asyncio.create_task(execute_approved_action(approval=approval, actor=actor))
    while not task.done():
        if is_agent_run_cancel_signaled(run_id):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            return ToolResult(
                output={
                    "ok": False,
                    "status": "tool_execution_cancelled",
                    "approval_id": approval.approval_id,
                    "action": approval.action,
                    "error": "用户中断了当前操作。",
                },
                is_error=True,
                is_retryable=False,
            )
        await asyncio.wait({task}, timeout=0.2)
    return await task


def _approval_observation(
    *,
    approval: PendingApproval,
    tool_call: LLMToolCall,
    tool_result: ToolResult,
    step: int,
) -> AgentObservation:
    output = tool_result.output if isinstance(tool_result.output, dict) else {}
    artifacts = output.get("artifacts")
    return AgentObservation(
        tool_call_id=tool_call.id,
        tool_name=approval.action,
        task_text=approval.reason or approval.action,
        ok=bool(output.get("ok")),
        need_continue=bool(output.get("need_continue")),
        remaining_task_hint=normalize_message_text(
            str(output.get("remaining_task_hint", "") or "")
        ),
        error=normalize_message_text(str(output.get("error", "") or "")),
        artifacts=tuple(
            dict(item) for item in artifacts or () if isinstance(item, dict)
        )
        if isinstance(artifacts, list | tuple)
        else (),
        step=max(int(step or 0), 0),
        result=tool_result,
        output=dict(output),
    )


def _approval_execution_not_started(approval: PendingApproval) -> ToolResult:
    return ToolResult(
        output={
            "ok": False,
            "status": "approval_execution_not_started",
            "approval_id": approval.approval_id,
            "action": approval.action,
            "error": "审批执行状态无法持久化，操作未执行。",
        },
        is_error=True,
        is_retryable=True,
    )


def _approval_execution_uncertain(approval: PendingApproval) -> ToolResult:
    return ToolResult(
        output={
            "ok": False,
            "status": "tool_execution_uncertain",
            "approval_id": approval.approval_id,
            "action": approval.action,
            "error": "操作可能已执行，但完成结果未能持久化；不会自动重放。",
        },
        is_error=True,
        is_retryable=False,
    )


async def _resume_agent_run(
    *,
    run_id: str,
    approval: PendingApproval,
    action_result: ToolResult,
    actor: dict[str, str],
    decision: str,
    decision_reason: str = "",
    result_persisted: bool = False,
) -> ToolResult:
    tools = build_superuser_tools()
    state = load_agent_run_state(run_id, tool_map=tools)
    if state is None or state.session_key != actor["session_key"]:
        return tool_result(False, "agent_run_not_found", run_id=run_id)
    if state.status not in {"paused", "running"}:
        return tool_result(
            False,
            "agent_run_not_resumable",
            run_id=run_id,
            run_status=state.status,
        )

    model_name = get_agent_model("superuser")
    provider_adapter = ProviderCapabilityAdapter.for_model(model_name)
    repair = repair_interrupted_tool_protocol(
        state,
        provider_adapter=provider_adapter,
    )
    if any(repair.values()):
        persist_agent_run_state(
            state,
            stage="tool_protocol_repaired",
            metadata=repair,
        )
    if not result_persisted:
        resume_message = _resume_message(
            approval,
            action_result,
            decision=decision,
            decision_reason=decision_reason,
        )
        state.messages.append(LLMMessage.user(resume_message))
        state.pending_approval = ""
        state.append_synthetic_observation(
            _approval_observation(
                approval=approval,
                tool_call=LLMToolCall(
                    id=f"approval:{approval.approval_id}",
                    function=LLMToolFunction(name=approval.action, arguments="{}"),
                ),
                tool_result=action_result,
                step=state.step,
            ),
            timeline_kind="pre_resume_tool_observation",
            content=str(
                (action_result.output or {}).get("status", "")
                if isinstance(action_result.output, dict)
                else ""
            ),
            metadata={"source": "runtime_approval"},
        )

    state.tool_map = tools
    state.resume(reason="runtime_approval")
    if not persist_agent_run_state(
        state,
        stage="agent_run_resumed",
        metadata={"approval_id": approval.approval_id, "decision": decision},
    ):
        return tool_result(False, "agent_run_resume_persist_failed", run_id=run_id)
    clear_agent_run_cancel_signal(run_id)
    context = RunContext(
        session_id=actor["session_key"],
        extra={
            "actor_user_id": actor["user_id"],
            "agent_mode": "superuser_agent",
            "enable_agent_tools": True,
            "trace_id": state.trace_id,
            "run_id": state.run_id,
            "resumed_run_id": state.run_id,
            "artifact_refs": state.artifact_refs,
            "provider_capability": provider_adapter.profile.to_metadata(),
        },
    )
    result = await AgentRuntime(
        state=state,
        run_context=context,
        message_text="继续之前的任务。",
        model_name=model_name,
        generation_config=build_superuser_generation_config(),
        timeout=SUPERUSER_MODEL_TIMEOUT_SECONDS,
    ).run()
    return tool_result(
        result.status != "failed",
        "agent_run_resumed",
        run_id=result.run_id or run_id,
        run_status=result.status,
        paused_reason=result.paused_reason,
        stop_reason=result.stop_reason,
        final_text=result.final_text,
        steps=result.steps,
    )


def _resume_message(
    approval: PendingApproval,
    result: ToolResult,
    *,
    decision: str,
    decision_reason: str = "",
) -> str:
    payload = _compact_result_payload(result, trace_id=_approval_run_id(approval))
    if decision == "reject":
        return (
            f"用户拒绝了操作 {approval.action}，该操作未执行。"
            + (f"理由：{decision_reason}。" if decision_reason else "")
            + "请基于这个结果继续当前任务：\n"
            + json.dumps(payload, ensure_ascii=False, default=str)
        )
    scope = ""
    if decision == "allow_conversation":
        scope = (
            "，本对话内工作区普通命令后续不再提示，危险操作仍需确认"
            if approval.action == "shell_command"
            else "，本对话后续相同权限范围也已允许"
        )
    return f"用户允许执行 {approval.action}{scope}。" "执行结果：\n" + json.dumps(
        payload, ensure_ascii=False, default=str
    )


def _reply_from_resume_result(
    result: ToolResult,
    *,
    fallback: ToolResult,
) -> str:
    output = result.output if isinstance(result.output, dict) else {}
    final_text = normalize_reply_text(str(output.get("final_text", "") or ""))
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
    if status == "approval_execution_not_started":
        parts = ["操作未执行：执行状态无法持久化。"]
    elif status == "tool_execution_uncertain":
        parts = ["操作执行状态不确定，系统不会自动重放。"]
    elif status == "tool_execution_cancelled":
        parts = ["操作已中断。"]
    else:
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
    payload = compact_tool_result_output(
        output,
        trace_id=trace_id,
        source="runtime_approval",
    )
    return {
        key: value
        for key, value in payload.items()
        if key != "approval_id" and value not in (None, "", [], {})
    }


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
