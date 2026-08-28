"""User controls for the active Superuser Agent conversation."""

from __future__ import annotations

from copy import copy
from datetime import datetime
import time
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.utils.message import MessageUtils

from ..config import (
    SUPERUSER_MODEL_TIMEOUT_SECONDS,
    get_agent_context_window_tokens,
    get_agent_model,
    get_fallback_models,
    get_superuser_max_output_tokens,
)
from ..event_runtime import event_is_private, resolve_superuser
from ..host_llm import resolve_host_model_candidates
from ..llm_compat import AI
from ..provider_capability import ProviderCapabilityAdapter
from ..route_text import normalize_message_text
from ..web_access import tools_for_web_candidate
from .approval_store import list_pending_approvals, reject_pending_approval
from .compaction import compact_superuser_context, summarize_with_candidates
from .permission_policy import (
    clear_conversation_permissions,
    conversation_has_workspace_shell_grant,
    get_default_permission_mode,
    resolve_permission_mode,
)
from .progress import progress_phase
from .runtime import (
    SuperuserSessionBusyError,
    _calculate_tool_schema_metrics,
    _estimate_prompt_tokens,
    _model_visible_superuser_messages,
    _semantic_summary_generation_config,
    _superuser_prompt_cache_key,
    cancel_superuser_session_execution,
    record_superuser_model_usage,
    request_superuser_semantic_summary,
    superuser_session_execution,
    superuser_session_is_executing,
)
from .store import (
    activate_agent_session,
    archive_conversation,
    clear_agent_session_context,
    create_conversation,
    deactivate_agent_session,
    delete_conversation,
    get_active_conversation,
    get_agent_run_snapshot,
    get_agent_session,
    list_agent_run_activities,
    list_conversations,
    load_agent_run_state,
    persist_agent_run_state,
    rename_conversation,
    restore_conversation,
    set_conversation_permission_mode,
    switch_conversation,
    update_agent_run_status,
)
from .tools import build_superuser_tools
from .tools.shell_tools import (
    has_running_background_shell_tasks,
    stop_background_shell_tasks,
)
from .trajectory import record_agent_control_trajectory

_ACTIVE_STATUSES = {"running", "paused"}
_ACTIVE_BLOCKED_INTENTS = {
    "archive",
    "clear",
    "compact",
    "create_conversation",
    "delete_conversation",
    "exit",
    "open",
    "permission_mode",
    "switch_conversation",
}
_UNCERTAIN_BLOCKED_INTENTS = {"clear", "delete_conversation"}
_CONVERSATION_INTENTS = {
    "archive",
    "create_conversation",
    "current_conversation",
    "delete_conversation",
    "list_archived_conversations",
    "list_conversations",
    "permission_mode",
    "rename_conversation",
    "restore_conversation",
    "switch_conversation",
}
_PENDING_SWITCH_SECONDS = 60.0
_PENDING_SWITCH_UNTIL: dict[str, float] = {}
_AGENT_HELP_TEXT = (
    "Agent：/开启agent /退出agent /状态 /中断\n"
    "上下文：/清除上下文 /压缩上下文\n"
    "会话：/新增会话 [名称] /当前会话 /列出会话 "
    "/切换会话 ID/名称 /重命名会话 ID 新名称 "
    "/归档会话 [ID] /列出归档会话 /恢复会话 ID /删除会话 ID\n"
    "权限：/请求批准模式 /只读模式 /完全访问模式\n"
    "审批：/允许 /本对话允许 /拒绝 [理由] /中断"
)


async def try_handle_runtime_control(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
) -> bool:
    user_id = str(session.user.id if session.user else "")
    session_key = str(session.group.id) if session.group else user_id
    if not user_id or not session_key:
        return False
    if not event_is_private(event) or not resolve_superuser(bot, user_id):
        return False

    intent, argument = _control_request(raw_message)
    if not intent:
        selection = _pending_switch_selection(session_key, raw_message)
        if selection:
            intent, argument = "switch_conversation", selection
    if not intent:
        return False
    session_state = get_agent_session(session_key)
    run_id = str(session_state.get("run_id", "") or "")
    is_active = bool(session_state.get("active"))
    snapshot = get_agent_run_snapshot(run_id) if run_id else None
    is_executing = superuser_session_is_executing(session_key)
    is_running = (
        is_executing
        or has_running_background_shell_tasks(run_id)
        or bool(
            isinstance(snapshot, dict) and str(snapshot.get("status", "")) == "running"
        )
    )
    is_waiting_approval = (
        False
        if is_running
        else _has_pending_approval(
            snapshot,
            run_id=run_id,
            user_id=user_id,
            session_key=session_key,
        )
    )
    if intent in _ACTIVE_BLOCKED_INTENTS and is_running:
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text="当前任务仍在执行，请先回复 /中断，再操作会话。",
        )
        return True
    if intent in _ACTIVE_BLOCKED_INTENTS and is_waiting_approval:
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text="当前任务正在等待确认，请使用审批消息中的命令，或回复 /中断。",
        )
        return True
    if intent in _UNCERTAIN_BLOCKED_INTENTS and _has_uncertain_side_effect(snapshot):
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text="当前会话存在执行结果不确定的操作，不能清除或删除该会话。",
        )
        return True
    if intent in _CONVERSATION_INTENTS:
        conversations_before = {
            str(item.get("id", "") or ""): item
            for item in list_conversations(session_key, archived=None)
        }
        text = _handle_conversation_control(
            intent=intent,
            argument=argument,
            session_key=session_key,
            user_id=user_id,
        )
        conversations_after = {
            str(item.get("id", "") or ""): item
            for item in list_conversations(session_key, archived=None)
        }
        try:
            from .active_tasks import (
                delete_active_tasks_for_conversation,
                pause_active_tasks_for_conversation,
            )

            deleted_ids = conversations_before.keys() - conversations_after.keys()
            archived_ids = {
                conversation_id
                for conversation_id, item in conversations_after.items()
                if bool(item.get("archived"))
                and not bool(
                    conversations_before.get(conversation_id, {}).get("archived")
                )
            }
            for conversation_id in deleted_ids:
                await delete_active_tasks_for_conversation(
                    session_key,
                    conversation_id,
                )
            for conversation_id in archived_ids:
                await pause_active_tasks_for_conversation(
                    session_key,
                    conversation_id,
                )
        except Exception as exc:
            from zhenxun.services import logger

            logger.error("ChatInter 主动任务会话同步失败", e=exc)
            text += "\n关联主动任务同步失败，请检查任务列表。"
        await _send_runtime_reply(bot=bot, event=event, text=text)
        return True
    if intent == "help":
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text=_AGENT_HELP_TEXT,
        )
        return True
    if intent == "open":
        activate_agent_session(session_key, run_id=run_id)
        conversation = get_active_conversation(session_key)
        text = "Agent 已开启，直接发送任务即可。"
        if conversation is not None:
            text += (
                f"\n当前会话：{conversation['id']} {conversation['name']}"
                f"\n权限模式：{_permission_mode_label(conversation)}"
            )
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text=text,
        )
        return True
    if intent == "clear":
        previous = clear_agent_session_context(session_key)
        clear_conversation_permissions(previous)
        _discard_pending_approvals(
            previous,
            user_id=user_id,
            session_key=session_key,
        )
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text="当前对话上下文已清除。",
        )
        return True
    if intent == "exit":
        deactivate_agent_session(session_key)
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text="已退出 Agent；回复 /开启agent 可继续当前会话。",
        )
        return True
    if intent == "compact":
        try:
            async with superuser_session_execution(session_key):
                text = await _compact_conversation(run_id)
        except SuperuserSessionBusyError:
            text = "当前任务仍在执行，请先回复 /中断，再压缩上下文。"
        await _send_runtime_reply(bot=bot, event=event, text=text)
        return True

    if intent == "status":
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text=_status_reply(
                [snapshot] if isinstance(snapshot, dict) else [],
                [],
                conversation=get_active_conversation(session_key),
            ),
        )
        return True

    current = snapshot if isinstance(snapshot, dict) else None
    current_active = bool(
        is_executing
        or is_waiting_approval
        or (
            is_active
            and current is not None
            and str(current.get("status", "")) in _ACTIVE_STATUSES
        )
    )
    if intent == "stop":
        if not current_active:
            await _send_runtime_reply(
                bot=bot,
                event=event,
                text="当前没有正在执行的 Agent 任务。",
            )
            return True
        proactive_kind = ""
        if is_executing:
            from .proactive_tasks import proactive_session_execution_kind

            proactive_kind = proactive_session_execution_kind(session_key)
        if is_waiting_approval or proactive_kind not in {"notify", "script"}:
            update_agent_run_status(
                run_id,
                status="cancelled",
                reason="cancelled_by_runtime_control",
                metadata={"cancelled_by": user_id},
                clear_pending_approval=True,
            )
        _discard_pending_approvals(
            run_id,
            user_id=user_id,
            session_key=session_key,
        )
        cancel_superuser_session_execution(session_key)
        await stop_background_shell_tasks(run_id)
        await _send_runtime_reply(bot=bot, event=event, text="已中断当前任务。")
        return True
    return False


def has_runtime_control_intent(raw_message: str, *, session_key: str = "") -> bool:
    return bool(
        _control_request(raw_message)[0]
        or _pending_switch_selection(session_key, raw_message)
    )


def parse_runtime_control_command(raw_message: str) -> tuple[str, str]:
    return _control_request(raw_message)


def _control_request(raw_message: str) -> tuple[str, str]:
    return _parse_control_command(normalize_message_text(raw_message))


def _parse_control_command(command: str) -> tuple[str, str]:
    text = command.casefold()
    exact = {
        "/开启agent": ("open", ""),
        "/退出agent": ("exit", ""),
        "/agent帮助": ("help", ""),
        "/状态": ("status", ""),
        "/中断": ("stop", ""),
        "/清除上下文": ("clear", ""),
        "/压缩上下文": ("compact", ""),
        "/请求批准模式": "ask",
        "/只读模式": "read_only",
        "/完全访问模式": "full_access",
        "/当前会话": ("current_conversation", ""),
        "/列出会话": ("list_conversations", ""),
        "/列出归档会话": ("list_archived_conversations", ""),
    }
    matched = exact.get(text)
    if isinstance(matched, tuple):
        return matched
    if isinstance(matched, str):
        return "permission_mode", matched
    for prefix, intent in (
        ("/新增会话", "create_conversation"),
        ("/切换会话", "switch_conversation"),
        ("/重命名会话", "rename_conversation"),
        ("/归档会话", "archive"),
        ("/恢复会话", "restore_conversation"),
        ("/删除会话", "delete_conversation"),
    ):
        if command == prefix:
            return intent, ""
        marker = prefix + " "
        if command.startswith(marker):
            return intent, command[len(marker) :].strip()
    return "", ""


def _handle_conversation_control(
    *,
    intent: str,
    argument: str,
    session_key: str,
    user_id: str,
) -> str:
    if intent == "create_conversation":
        conversation = create_conversation(session_key, name=argument)
        if conversation is None:
            return "新建会话失败。"
        return (
            f"已新建并切换到会话 {conversation['id']}：{conversation['name']}。\n"
            f"权限模式：{_permission_mode_label(conversation)}"
        )

    if intent == "permission_mode":
        conversation = get_active_conversation(session_key)
        if conversation is None:
            return "当前没有选中的 Agent 会话。"
        updated = set_conversation_permission_mode(
            session_key,
            str(conversation["id"]),
            argument,
        )
        if updated is None:
            return "权限模式切换失败。"
        if argument in {"ask", "read_only"}:
            clear_conversation_permissions(str(conversation.get("run_id", "") or ""))
            return (
                f"权限模式已切换为：{_permission_mode_label(updated)}；"
                "本对话授权已清除。"
            )
        return f"权限模式已切换为：{_permission_mode_label(updated)}。"

    if intent == "current_conversation":
        conversation = get_active_conversation(session_key)
        if conversation is None:
            return "当前没有选中的 Agent 会话。"
        active = bool(get_agent_session(session_key).get("agent_mode_active"))
        return (
            f"当前会话：{conversation['id']} {conversation['name']}\n"
            f"Agent 模式：{'已开启' if active else '已退出'}\n"
            f"权限模式：{_permission_mode_label(conversation)}\n"
            f"本对话授权：{_conversation_grant_label(conversation)}"
        )

    if intent == "list_conversations":
        return _conversation_list_reply(
            list_conversations(session_key, archived=False),
            empty="当前没有可用的 Agent 会话。",
        )

    if intent == "list_archived_conversations":
        return _conversation_list_reply(
            list_conversations(session_key, archived=True),
            empty="当前没有已归档的 Agent 会话。",
        )

    if intent == "switch_conversation" and not argument:
        conversations = list_conversations(session_key, archived=False)
        if not conversations:
            return "当前没有可切换的 Agent 会话。"
        _PENDING_SWITCH_UNTIL[session_key] = time.monotonic() + _PENDING_SWITCH_SECONDS
        return "请选择要切换的会话\n" + _conversation_rows(conversations)

    if intent == "rename_conversation":
        selector, separator, name = argument.partition(" ")
        if not separator or not selector.strip() or not name.strip():
            return "用法：重命名会话 ID/名称 新名称"
        conversation, error = _resolve_conversation(session_key, selector)
        if conversation is None:
            return error
        renamed = rename_conversation(session_key, str(conversation["id"]), name)
        return (
            f"会话 {renamed['id']} 已重命名为：{renamed['name']}。"
            if renamed
            else "重命名会话失败。"
        )

    archived = intent == "restore_conversation"
    selector = argument
    if intent == "archive" and not selector:
        current = get_active_conversation(session_key)
        selector = str(current["id"]) if current else ""
    if not selector:
        return {
            "archive": "用法：归档会话 ID/名称",
            "delete_conversation": "用法：删除会话 ID/名称",
            "restore_conversation": "用法：恢复会话 ID/名称",
            "switch_conversation": "用法：切换会话 ID/名称",
        }.get(intent, "缺少会话 ID 或名称。")

    conversation, error = _resolve_conversation(
        session_key,
        selector,
        archived=archived if intent == "restore_conversation" else None,
    )
    if conversation is None:
        return error
    conversation_id = str(conversation["id"])
    if intent == "switch_conversation":
        _PENDING_SWITCH_UNTIL.pop(session_key, None)
        if conversation.get("archived"):
            return "该会话已归档，请先恢复会话。"
        switched = switch_conversation(session_key, conversation_id)
        return (
            f"已切换到会话 {switched['id']}：{switched['name']}。\n"
            f"权限模式：{_permission_mode_label(switched)}"
            if switched
            else "切换会话失败。"
        )
    if intent == "archive":
        result = archive_conversation(session_key, conversation_id)
        return f"会话 {result['id']} 已归档。" if result else "归档会话失败。"
    if intent == "restore_conversation":
        result = restore_conversation(session_key, conversation_id)
        return f"会话 {result['id']} 已恢复。" if result else "恢复会话失败。"
    if intent == "delete_conversation":
        target_run_id = str(conversation.get("run_id", "") or "")
        target_snapshot = (
            get_agent_run_snapshot(target_run_id) if target_run_id else None
        )
        if _has_uncertain_side_effect(target_snapshot):
            return "目标会话存在执行结果不确定的操作，不能删除该会话。"
        result = delete_conversation(session_key, conversation_id)
        if result is None:
            return "删除会话失败。"
        run_id = str(result.get("run_id", "") or "")
        clear_conversation_permissions(run_id)
        _discard_pending_approvals(
            run_id,
            user_id=user_id,
            session_key=session_key,
        )
        return f"会话 {result['id']} 已删除。"
    return "未知的会话操作。"


def _resolve_conversation(
    session_key: str,
    selector: str,
    *,
    archived: bool | None = None,
) -> tuple[dict[str, Any] | None, str]:
    target = normalize_message_text(selector)
    conversations = list_conversations(session_key, archived=archived)
    if target.isdigit():
        matches = [item for item in conversations if str(item.get("id")) == target]
    else:
        folded = target.casefold()
        matches = [
            item
            for item in conversations
            if str(item.get("name", "")).casefold() == folded
        ]
    if not matches:
        return None, f"未找到会话：{target}。"
    if len(matches) > 1:
        return None, f"存在多个同名会话“{target}”，请使用数字 ID。"
    return matches[0], ""


def _conversation_list_reply(
    conversations: list[dict[str, Any]],
    *,
    empty: str,
) -> str:
    if not conversations:
        return empty
    return _conversation_rows(conversations)


def _conversation_rows(conversations: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{item['id']} {item['name']} {_display_time(item.get('last_used_at'))}"
        for item in conversations
    )


def _display_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "时间未知"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text[:32]


def _pending_switch_selection(session_key: str, raw_message: str) -> str:
    key = str(session_key or "").strip()
    selection = normalize_message_text(raw_message).strip()
    if not key or not selection.isdigit():
        return ""
    deadline = _PENDING_SWITCH_UNTIL.get(key, 0.0)
    if deadline <= time.monotonic():
        _PENDING_SWITCH_UNTIL.pop(key, None)
        return ""
    return selection


def _discard_pending_approvals(
    run_id: str,
    *,
    user_id: str,
    session_key: str,
) -> None:
    for approval in list_pending_approvals(
        user_id=user_id,
        session_key=session_key,
    ):
        payload = approval.payload if isinstance(approval.payload, dict) else {}
        approval_run = str(payload.get("run_id") or payload.get("trace_id") or "")
        if approval_run == run_id:
            reject_pending_approval(
                approval_id=approval.approval_id,
                user_id=user_id,
                session_key=session_key,
            )


def _has_pending_approval(
    snapshot: dict[str, Any] | None,
    *,
    run_id: str,
    user_id: str,
    session_key: str,
) -> bool:
    if not run_id:
        return False
    approvals = list_pending_approvals(
        user_id=user_id,
        session_key=session_key,
    )
    pending_id = (
        str(snapshot.get("pending_approval", "") or "")
        if isinstance(snapshot, dict)
        else ""
    )
    if any(
        approval.approval_id == pending_id or _approval_run_id(approval) == run_id
        for approval in approvals
    ):
        return True
    if pending_id:
        update_agent_run_status(
            run_id,
            status="paused",
            reason="approval_expired",
            clear_pending_approval=True,
        )
        snapshot["pending_approval"] = ""
        snapshot.pop("waiting_approval_ids", None)
    return False


def _has_uncertain_side_effect(snapshot: dict[str, Any] | None) -> bool:
    if not isinstance(snapshot, dict):
        return False
    records = snapshot.get("tool_executions")
    if not isinstance(records, list | tuple):
        return False
    return any(
        isinstance(record, dict)
        and str(record.get("status", "")) in {"started", "uncertain"}
        for record in records
    )


def _approval_run_id(approval: Any) -> str:
    payload = getattr(approval, "payload", None)
    payload = payload if isinstance(payload, dict) else {}
    return str(payload.get("run_id") or payload.get("trace_id") or "")


async def _compact_conversation(run_id: str) -> str:
    from .context import (
        context_window_budget,
        estimate_prompt_tokens_with_baseline,
    )

    if not run_id:
        return "当前没有可精简的 Agent 对话。"
    started_at = time.time()
    started_perf = time.perf_counter()
    snapshot = get_agent_run_snapshot(run_id)
    if not isinstance(snapshot, dict):
        return "当前 Agent 对话不存在。"
    configured_model = get_agent_model("superuser")
    candidates = await resolve_host_model_candidates(
        configured_model,
        get_fallback_models(configured_model),
    )
    candidate = candidates[0]
    model_name = candidate.name
    configured_context_tokens = get_agent_context_window_tokens("superuser")
    max_input_tokens = candidate.context_window(configured_context_tokens)
    base_tools = build_superuser_tools()
    source_tools = tools_for_web_candidate(
        base_tools,
        candidate=candidate,
        scope="superuser",
    )
    provider_adapter = ProviderCapabilityAdapter.for_model(
        model_name,
        capabilities=candidate.capabilities,
        api_type=candidate.api_type,
    )
    tools = provider_adapter.prepare_tool_map_for_request(source_tools)
    state = load_agent_run_state(run_id, tool_map=source_tools)
    if state is None:
        return "当前 Agent 对话无法恢复。"
    budget = snapshot.get("budget")
    budget = budget if isinstance(budget, dict) else {}
    prompt_tokens_before = estimate_prompt_tokens_with_baseline(
        _model_visible_superuser_messages(state.messages),
        current_context_tokens=budget.get("current_context_tokens", 0),
        last_usage_message_count=budget.get("last_usage_message_count", 0),
        last_usage_schema_tokens=budget.get("last_usage_schema_tokens", 0),
        estimate=_estimate_prompt_tokens,
    )
    _, schema_tokens = await _calculate_tool_schema_metrics(tools)
    output_reserve_tokens = get_superuser_max_output_tokens()
    candidate_blocking_limits: list[int] = []
    for item in candidates:
        if item is candidate:
            item_schema_tokens = schema_tokens
        else:
            item_source_tools = tools_for_web_candidate(
                base_tools,
                candidate=item,
                scope="superuser",
            )
            item_adapter = ProviderCapabilityAdapter.for_model(
                item.name,
                capabilities=item.capabilities,
                api_type=item.api_type,
            )
            item_tools = item_adapter.prepare_tool_map_for_request(item_source_tools)
            _, item_schema_tokens = await _calculate_tool_schema_metrics(item_tools)
        candidate_blocking_limits.append(
            context_window_budget(
                max_input_tokens=item.context_window(configured_context_tokens),
                prompt_tokens=0,
                schema_tokens=item_schema_tokens,
                output_reserve_tokens=output_reserve_tokens,
            ).blocking_limit
        )
    ai = AI(session_id=f"chatinter-superuser:{run_id}")
    budget_before = (
        state.budget.run_input_tokens,
        state.budget.run_output_tokens,
        state.budget.model_calls,
    )

    async def invoke_summary(
        summary_candidate,
        request_messages,
        candidate_max_input_tokens,
        summary_token_target,
    ):
        prompt_cache_key = await _superuser_prompt_cache_key(
            run_id=run_id,
            model_name=summary_candidate.name,
            tool_schema_hash="",
            namespace="summary",
        )
        response = await request_superuser_semantic_summary(
            ai=ai,
            messages=request_messages,
            model_name=summary_candidate.name,
            candidate=summary_candidate,
            max_input_tokens=candidate_max_input_tokens,
            timeout=SUPERUSER_MODEL_TIMEOUT_SECONDS,
            prompt_cache_key=prompt_cache_key,
            summary_token_target=summary_token_target,
        )
        record_superuser_model_usage(
            state,
            response,
            estimated_input_tokens=_estimate_prompt_tokens(request_messages),
            schema_tokens=0,
            update_context=False,
            request_kind="summary",
            model_name=summary_candidate.name,
            tool_schema_hash="",
            request_message_count=len(request_messages),
            request_generation_config=_semantic_summary_generation_config(
                candidate_max_input_tokens,
                summary_token_target=summary_token_target,
                native_structured_output=(
                    getattr(response, "chatinter_summary_strategy", "native")
                    == "native"
                ),
            ),
            cache_phase="compaction_request",
            visible_messages=request_messages,
        )
        return response

    async def summarize(request_messages):
        return await summarize_with_candidates(
            request_messages,
            candidates=candidates,
            preferred_name=model_name,
            configured_context_tokens=configured_context_tokens,
            invoke=invoke_summary,
        )

    execution = await compact_superuser_context(
        state,
        max_input_tokens=max_input_tokens,
        schema_tokens=schema_tokens,
        output_reserve_tokens=output_reserve_tokens,
        prompt_tokens_before=prompt_tokens_before,
        summarize=summarize,
        persist=lambda stage, metadata: persist_agent_run_state(
            state,
            stage=stage,
            metadata=metadata,
        ),
        visible_messages=_model_visible_superuser_messages,
        trigger="manual",
        hard_required=True,
        summary_max_input_tokens=max(
            (
                candidate.context_window(configured_context_tokens)
                for candidate in candidates
            ),
            default=max_input_tokens,
        ),
    )
    result = execution.result
    usage_recorded = state.budget.model_calls > budget_before[2]
    fits_candidate_chain = any(
        result.after_tokens < blocking_limit
        for blocking_limit in candidate_blocking_limits
    )
    largest_candidate_limit = max(candidate_blocking_limits, default=0)
    if result.artifact_persistence_failed:
        return "上下文归档写入失败；原上下文已保留，请检查数据目录存储状态。"
    if not result.changed:
        if usage_recorded:
            _record_manual_compression_trajectory(
                state,
                started_at=started_at,
                started_perf=started_perf,
                budget_before=budget_before,
                outcome="failed" if result.failure_reason else "no_change",
                failure_reason=result.failure_reason or "no_safe_middle",
            )
        if not fits_candidate_chain:
            return (
                "固定系统上下文、工具定义和归档后的当前请求仍超过所有候选模型窗口。"
                "请缩短本次输入或配置更大上下文窗口的模型。"
            )
        return "当前没有可安全压缩的旧内容；原上下文未修改。"
    if execution.persistence_failed:
        if usage_recorded:
            _record_manual_compression_trajectory(
                state,
                started_at=started_at,
                started_perf=started_perf,
                budget_before=budget_before,
                outcome="failed",
                failure_reason="persistence_failed",
            )
        return "上下文压缩结果保存失败；原上下文已保留。"
    if usage_recorded:
        _record_manual_compression_trajectory(
            state,
            started_at=started_at,
            started_perf=started_perf,
            budget_before=budget_before,
            outcome="completed",
            failure_reason="",
        )
    if not fits_candidate_chain:
        return (
            f"上下文已压缩：{result.before_tokens} -> {result.after_tokens} tokens，"
            "但仍超过所有候选模型的可请求窗口"
            f"（{result.after_tokens} >= {largest_candidate_limit}）；"
            "请缩短本次输入或配置更大上下文窗口的模型。"
        )
    return f"上下文已压缩：{result.before_tokens} -> {result.after_tokens} tokens。"


def _record_manual_compression_trajectory(
    state: Any,
    *,
    started_at: float,
    started_perf: float,
    budget_before: tuple[int, int, int],
    outcome: str,
    failure_reason: str,
) -> None:
    try:
        projection = copy(state)
        projection.metrics = [
            metric
            for metric in state.metrics
            if metric.kind == "model_usage"
            and str(metric.metadata.get("request_kind", "")) == "summary"
        ]
        projection.tool_executions = []
        projection.final_text = ""
        projection.delivery_complete = False
        projection.final_source = ""
        projection.status = "failed" if outcome == "failed" else "completed"
        projection.budget = copy(state.budget)
        projection.budget.run_input_tokens = max(
            int(state.budget.run_input_tokens) - int(budget_before[0]),
            0,
        )
        projection.budget.run_output_tokens = max(
            int(state.budget.run_output_tokens) - int(budget_before[1]),
            0,
        )
        projection.budget.model_calls = max(
            int(state.budget.model_calls) - int(budget_before[2]),
            0,
        )
        record_agent_control_trajectory(
            state=projection,
            input_message="/压缩上下文",
            started_at=started_at,
            latency_ms=max(int((time.perf_counter() - started_perf) * 1000), 0),
            control_action="manual_context_compression",
            control_outcome=outcome,
            control_failure_reason=failure_reason,
        )
    except Exception:
        return


def _status_reply(
    active: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    conversation: dict[str, Any] | None = None,
) -> str:
    run = active[0] if active else (runs[0] if runs else None)
    if run is None:
        if conversation is None:
            return "当前没有 Agent 任务。\n" f"权限模式：{_permission_mode_label(None)}"
        return (
            f"当前会话：{conversation['id']} {conversation['name']}\n"
            "当前没有 Agent 任务。\n"
            f"权限模式：{_permission_mode_label(conversation)}\n"
            f"本对话授权：{_conversation_grant_label(conversation)}"
        )
    run_id = str(run.get("run_id") or run.get("trace_id") or "")
    status = str(run.get("status", "") or "unknown")
    waiting = (
        bool(run.get("pending_approval"))
        or str(run.get("paused_reason", "") or "") == "approval_required"
    )
    lines = []
    if conversation is not None:
        lines.append(f"当前会话：{conversation['id']} {conversation['name']}")
    lines.extend(
        (
            f"当前状态：{_status_label(status)}",
            f"当前动作：{_current_action(run_id, status=status, waiting=waiting)}",
            f"权限模式：{_permission_mode_label(conversation)}",
        )
    )
    if conversation is not None:
        lines.append(f"本对话授权：{_conversation_grant_label(conversation)}")
    lines.append(f"等待确认：{'是' if waiting else '否'}")
    return "\n".join(lines)


def _permission_mode_label(conversation: dict[str, Any] | None) -> str:
    raw_mode = (
        str(conversation.get("permission_mode", "") or "")
        if conversation is not None
        else ""
    )
    mode = (
        resolve_permission_mode(raw_mode) if raw_mode else get_default_permission_mode()
    )
    return {
        "ask": "请求批准",
        "read_only": "只读",
        "full_access": "完全访问",
    }[mode]


def _conversation_grant_label(conversation: dict[str, Any]) -> str:
    run_id = str(conversation.get("run_id", "") or "")
    return (
        "工作区普通命令已允许"
        if conversation_has_workspace_shell_grant(run_id)
        else "未授权"
    )


def _status_label(status: str) -> str:
    return {
        "running": "执行中",
        "paused": "已暂停",
        "completed": "已完成",
        "failed": "执行失败",
        "cancelled": "已中断",
    }.get(normalize_message_text(status), "未知")


def _current_action(run_id: str, *, status: str, waiting: bool) -> str:
    if waiting:
        return "等待用户确认"
    terminal_action = {
        "completed": "任务已完成",
        "failed": "任务执行失败",
        "cancelled": "任务已中断",
    }.get(normalize_message_text(status))
    if terminal_action:
        return terminal_action
    rows = list_agent_run_activities(run_id, limit=1) if run_id else []
    if rows:
        tool_name = normalize_message_text(str(rows[-1].get("tool_name", "") or ""))
        if tool_name:
            return progress_phase(tool_name) or "正在处理任务"
    return {
        "paused": "任务已暂停",
    }.get(normalize_message_text(status), "正在处理任务")


async def _send_runtime_reply(*, bot: Bot, event: Event, text: str) -> None:
    try:
        await bot.send(event, text)
        return
    except Exception:
        pass
    try:
        await MessageUtils.build_message(text).send()
    except Exception:
        pass


__all__ = [
    "has_runtime_control_intent",
    "parse_runtime_control_command",
    "try_handle_runtime_control",
]
