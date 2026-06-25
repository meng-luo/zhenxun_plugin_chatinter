"""Runtime controls consumed before the superuser Agent LLM path."""

from __future__ import annotations

from typing import Any

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.utils.message import MessageUtils

from ..agent_run_store import (
    get_agent_run_snapshot,
    list_agent_run_activities,
    list_agent_run_snapshots,
    update_agent_run_status,
)
from ..event_runtime import event_is_private, resolve_superuser
from ..route_text import normalize_message_text

_STOP_WORDS = frozenset(
    {
        "停止",
        "取消",
        "中断",
        "别继续",
        "不要继续",
        "停下",
        "stop",
        "cancel",
        "abort",
    }
)
_STATUS_WORDS = frozenset(
    {
        "状态",
        "进度",
        "现在到哪了",
        "现在在做什么",
        "执行到哪了",
        "到哪了",
        "agent状态",
    }
)
_ACTIVE_STATUSES = {"running", "paused"}


async def try_handle_runtime_control(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
) -> bool:
    """Consume stop/status controls without invoking the LLM."""

    user_id = str(session.user.id if session.user else "")
    session_key = str(session.group.id) if session.group else user_id
    if not user_id or not session_key:
        return False
    if not event_is_private(event) or not resolve_superuser(bot, user_id):
        return False

    intent = _control_intent(raw_message)
    if not intent:
        return False

    runs = list_agent_run_snapshots(session_key=session_key, limit=5)
    active = [run for run in runs if str(run.get("status", "")) in _ACTIVE_STATUSES]
    if intent == "stop":
        if not active:
            await _send_runtime_reply(
                bot=bot,
                event=event,
                text="当前没有运行中的 Agent。",
            )
            return True
        run_id = str(active[0].get("run_id") or active[0].get("trace_id") or "")
        update_agent_run_status(
            run_id,
            status="cancelled",
            reason="cancelled_by_runtime_control",
            metadata={"cancelled_by": user_id},
        )
        await _send_runtime_reply(
            bot=bot,
            event=event,
            text=f"已取消 AgentRun：{run_id}",
        )
        return True

    await _send_runtime_reply(bot=bot, event=event, text=_status_reply(active, runs))
    return True


def has_runtime_control_intent(raw_message: str) -> bool:
    return bool(_control_intent(raw_message))


def _control_intent(raw_message: str) -> str:
    text = normalize_message_text(raw_message).lower().strip(" ：:，,。.！!？?")
    if not text:
        return ""
    if text in _STOP_WORDS:
        return "stop"
    if text in _STATUS_WORDS:
        return "status"
    return ""


def _status_reply(active: list[dict[str, Any]], runs: list[dict[str, Any]]) -> str:
    run = active[0] if active else (runs[0] if runs else None)
    if run is None:
        return "当前没有 AgentRun 记录。"
    run_id = str(run.get("run_id") or run.get("trace_id") or "")
    status = str(run.get("status", "") or "unknown")
    step = run.get("step", 0)
    max_steps = run.get("max_steps", 0)
    updated = str(run.get("updated_at", "") or "")
    pending = int(run.get("pending_task_count", 0) or 0)
    completed = int(run.get("completed_task_count", 0) or 0)
    parts = [
        f"AgentRun：{run_id}",
        f"状态：{status}",
        f"步数：{step}/{max_steps}",
        f"任务：完成 {completed}，待处理 {pending}",
    ]
    if updated:
        parts.append(f"更新：{updated}")
    activities = _activity_lines(run_id)
    if activities:
        parts.append("最近动作：")
        parts.extend(activities)
    if active:
        parts.append("可回复“停止”取消。")
    return "\n".join(parts)


def _activity_lines(run_id: str) -> list[str]:
    rows = list_agent_run_activities(run_id, limit=10) if run_id else []
    if rows:
        return _activity_rows_to_lines(rows)
    snapshot = get_agent_run_snapshot(run_id) if run_id else None
    if not isinstance(snapshot, dict):
        return []
    legacy_rows: list[dict[str, Any]] = []
    for item in snapshot.get("tool_calls", []) or []:
        name = _nested_tool_name(item)
        if name:
            legacy_rows.append(
                {"step": int(item.get("step", 0) or 0), "tool_name": name}
            )
    for item in snapshot.get("observations", []) or []:
        if isinstance(item, dict) and item.get("tool_name"):
            legacy_rows.append(dict(item))
    return _activity_rows_to_lines(legacy_rows[-10:])


def _activity_rows_to_lines(items: list[dict[str, Any]]) -> list[str]:
    rows: list[tuple[int, str, bool | None]] = []
    for item in items:
        name = str(item.get("tool_name", "") or "")
        if name:
            rows.append(
                (
                    int(item.get("step", 0) or 0),
                    _activity_label(name),
                    item.get("ok") if isinstance(item.get("ok"), bool) else None,
                )
            )
    lines: list[str] = []
    for _step, label, ok in rows[-10:]:
        suffix = "" if ok is None else (" 完成" if ok else " 失败/等待")
        lines.append(f"- {label}{suffix}")
    return lines[-10:]


def _nested_tool_name(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    function = item.get("function")
    if isinstance(function, dict):
        return str(function.get("name", "") or "")
    return str(item.get("tool_name", "") or "")


def _activity_label(tool_name: str) -> str:
    name = normalize_message_text(tool_name)
    if name in {"read_file", "list_dir", "search_files", "artifact_read"}:
        return "读取文件"
    if name.startswith(("patch_", "write_", "append_", "replace_")):
        return "应用修改"
    if name.startswith(("engineering_eval_", "uv_", "python_")):
        return "跑验证"
    if name.endswith("_command") or name in {
        "shell_command",
        "git_command",
        "server_command",
    }:
        return "执行命令"
    if "approval" in name or name.startswith(("approve_", "reject_", "revoke_")):
        return "等待确认"
    if name.startswith(("plugin_dev_", "worktree_")):
        return "插件/工作区操作"
    return "处理任务"


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
    "try_handle_runtime_control",
]
