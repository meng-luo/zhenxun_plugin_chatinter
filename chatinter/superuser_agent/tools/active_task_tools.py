"""Persistent proactive task tools for the Superuser Agent."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any

from ...llm_compat import ToolDefinition, ToolResult
from ..active_tasks import (
    ActiveTask,
    active_task_next_run_time,
    build_script_identity,
    control_active_task,
    create_active_task,
    get_active_task,
    list_active_tasks,
    normalize_active_task_trigger,
    rotate_active_task_webhook_token,
    update_active_task,
    update_active_task_status,
)
from ..audit_log import record_audit_event
from ..permission_policy import (
    decide_active_task,
    file_path_deny,
)
from ..store import get_active_conversation
from .common import (
    actor_from_context,
    approval_required_result,
    permission_denied_result,
    tool_result,
)

_TASK_ID_PREFIX_LENGTH = 12
_LIST_DEFAULT_LIMIT = 50
_LIST_MAX_LIMIT = 100
_WEBHOOK_ROUTE_PREFIX = "/chatinter/active-task"
_TASK_KINDS = frozenset({"agent", "script", "notify"})
_CONTROL_ACTIONS = frozenset(
    {"pause", "resume", "delete", "run_now", "rotate_webhook"}
)
_CREATE_STORE_FIELDS = (
    "session_key",
    "user_id",
    "bot_id",
    "conversation_id",
    "name",
    "kind",
    "instruction",
    "trigger_type",
    "trigger_config",
    "entrypoint",
    "cwd",
    "args",
    "expected_entrypoint_sha256",
    "allow_network",
)


class ActiveTaskCreateTool:
    name = "active_task_create"
    read_only = False

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "创建可跨重启的主动任务。agent 在触发时以完全访问模式继续当前会话；"
                "notify 直接私聊固定文本；script 执行已批准且哈希不变的 Python 文件，"
                "触发事件 JSON 从 stdin 传入。script 在 Bot 宿主机执行，并非安全沙箱；"
                "agent 默认可使用完整工具和已配置的公网读取。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "maxLength": 160,
                        "description": "便于用户识别的任务名称。",
                    },
                    "kind": {
                        "type": "string",
                        "enum": sorted(_TASK_KINDS),
                        "description": "agent、notify 或 script。",
                    },
                    "instruction": {
                        "type": "string",
                        "maxLength": 20000,
                        "description": "触发后执行的固定指令或通知文本。",
                    },
                    "trigger_type": {
                        "type": "string",
                        "enum": ["date", "cron", "interval", "webhook"],
                    },
                    "trigger_config": {
                        "type": "object",
                        "description": (
                            "date 使用 run_date；cron 使用 cron 字段；interval 使用正数"
                            " weeks/days/hours/minutes/seconds；webhook 必须为空对象。"
                        ),
                        "properties": {
                            "run_date": {"type": "string"},
                            "year": {"type": ["integer", "string"]},
                            "month": {"type": ["integer", "string"]},
                            "day": {"type": ["integer", "string"]},
                            "week": {"type": ["integer", "string"]},
                            "day_of_week": {
                                "type": ["integer", "string"]
                            },
                            "hour": {"type": ["integer", "string"]},
                            "minute": {"type": ["integer", "string"]},
                            "second": {"type": ["integer", "string"]},
                            "weeks": {"type": "integer"},
                            "days": {"type": "integer"},
                            "hours": {"type": "integer"},
                            "minutes": {"type": "integer"},
                            "seconds": {"type": "integer"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "timezone": {"type": "string"},
                            "jitter": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    "entrypoint": {
                        "type": ["string", "null"],
                        "description": "script 的 .py 入口；其他 kind 省略。",
                    },
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "script 工作目录；入口必须位于其中。",
                    },
                    "args": {
                        "type": ["array", "null"],
                        "maxItems": 64,
                        "items": {"type": "string", "maxLength": 1024},
                        "description": "script 的固定命令行参数。",
                    },
                    "allow_network": {
                        "type": "boolean",
                        "description": (
                            "仅 agent 可用。是否允许触发后访问公网；默认 true。"
                        ),
                        "default": True,
                    },
                },
                "required": [
                    "name",
                    "kind",
                    "instruction",
                    "trigger_type",
                    "trigger_config",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        try:
            payload = prepare_active_task_create_payload(
                kwargs,
                context=context,
                actor=actor,
            )
        except (OSError, TypeError, ValueError) as exc:
            return tool_result(False, "invalid_active_task", error=str(exc))
        denied = validate_active_task_payload_paths(payload)
        if denied is not None:
            return permission_denied_result(
                actor=actor,
                action=self.name,
                payload=active_task_audit_payload(payload),
                permission=denied,
            )
        decision = decide_active_task("create")
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action=self.name,
                payload=active_task_audit_payload(payload),
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action=self.name,
                payload=payload,
                permission=decision,
                audit_payload=active_task_audit_payload(payload),
            )
        return await execute_active_task_create_payload(payload, actor=actor)


class ActiveTaskListTool:
    name = "active_task_list"
    read_only = True

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="列出当前超级用户拥有的主动任务及最近状态。",
            parameters={
                "type": "object",
                "properties": {
                    "offset": {"type": ["integer", "null"]},
                    "limit": {"type": ["integer", "null"]},
                    "task": {
                        "type": ["string", "null"],
                        "description": "可选；返回指定任务的详细信息。",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        reference = str(kwargs.get("task", "") or "").strip()
        if reference:
            task, error = resolve_active_task_reference(
                actor["session_key"],
                reference,
            )
            if task is None:
                return tool_result(False, "active_task_not_found", error=error)
            return tool_result(
                True,
                "active_task_detail",
                task=active_task_view(task, detailed=True),
            )
        offset = _bounded_int(kwargs.get("offset"), 0, 0, 100000)
        limit = _bounded_int(
            kwargs.get("limit"),
            _LIST_DEFAULT_LIMIT,
            1,
            _LIST_MAX_LIMIT,
        )
        tasks = list_active_tasks(actor["session_key"])
        page = tasks[offset : offset + limit]
        return tool_result(
            True,
            "active_tasks_listed",
            tasks=[active_task_view(task) for task in page],
            total=len(tasks),
            next_offset=offset + len(page) if offset + len(page) < len(tasks) else None,
        )


class ActiveTaskUpdateTool:
    name = "active_task_update"
    read_only = False

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "修改主动任务的名称、指令、调度、联网许可或已绑定脚本。"
                "任务类型及 webhook/定时触发类别不可互相转换。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "任务 ID、唯一 ID 前缀或精确任务名。",
                    },
                    "name": {"type": ["string", "null"], "maxLength": 160},
                    "instruction": {
                        "type": ["string", "null"],
                        "maxLength": 20000,
                    },
                    "trigger_type": {
                        "type": ["string", "null"],
                        "enum": ["date", "cron", "interval", "webhook", None],
                    },
                    "trigger_config": {
                        "type": ["object", "null"],
                        "properties": {
                            "run_date": {"type": "string"},
                            "year": {"type": ["integer", "string"]},
                            "month": {"type": ["integer", "string"]},
                            "day": {"type": ["integer", "string"]},
                            "week": {"type": ["integer", "string"]},
                            "day_of_week": {"type": ["integer", "string"]},
                            "hour": {"type": ["integer", "string"]},
                            "minute": {"type": ["integer", "string"]},
                            "second": {"type": ["integer", "string"]},
                            "weeks": {"type": "integer"},
                            "days": {"type": "integer"},
                            "hours": {"type": "integer"},
                            "minutes": {"type": "integer"},
                            "seconds": {"type": "integer"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "timezone": {"type": "string"},
                            "jitter": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    "entrypoint": {"type": ["string", "null"]},
                    "cwd": {"type": ["string", "null"]},
                    "args": {
                        "type": ["array", "null"],
                        "maxItems": 64,
                        "items": {"type": "string", "maxLength": 1024},
                    },
                    "allow_network": {"type": ["boolean", "null"]},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        try:
            payload = prepare_active_task_update_payload(kwargs, actor=actor)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return tool_result(False, "invalid_active_task_update", error=str(exc))
        denied = validate_active_task_payload_paths(payload)
        if denied is not None:
            return permission_denied_result(
                actor=actor,
                action=self.name,
                payload=active_task_audit_payload(payload),
                permission=denied,
            )
        decision = decide_active_task("update")
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action=self.name,
                payload=active_task_audit_payload(payload),
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action=self.name,
                payload=payload,
                permission=decision,
                audit_payload=active_task_audit_payload(payload),
            )
        return await execute_active_task_update_payload(payload, actor=actor)


class ActiveTaskControlTool:
    name = "active_task_control"
    read_only = False

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="暂停、恢复、删除、立即运行或轮换 Webhook 凭据。",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "列表返回的 ID、唯一 ID 前缀或精确任务名。",
                    },
                    "action": {
                        "type": "string",
                        "enum": sorted(_CONTROL_ACTIONS),
                    },
                },
                "required": ["task", "action"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        action = str(kwargs.get("action", "") or "").strip().casefold()
        if action not in _CONTROL_ACTIONS:
            return tool_result(False, "invalid_active_task_action")
        task, error = resolve_active_task_reference(
            actor["session_key"],
            str(kwargs.get("task", "") or ""),
        )
        if task is None:
            return tool_result(False, "active_task_not_found", error=error)
        payload = {
            "task_id": task.task_id,
            "action": action,
            "session_key": actor["session_key"],
        }
        decision = decide_active_task(action)
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action=self.name,
                payload={"task_id": _short_task_id(task.task_id), "action": action},
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action=self.name,
                payload=payload,
                permission=decision,
            )
        return await execute_active_task_control_payload(payload, actor=actor)


def prepare_active_task_create_payload(
    values: dict[str, Any],
    *,
    context: Any | None,
    actor: dict[str, str],
) -> dict[str, Any]:
    extra = getattr(context, "extra", None)
    bot_id = str(extra.get("bot_id", "") if isinstance(extra, dict) else "").strip()
    conversation_id = str(
        extra.get("conversation_id", "") if isinstance(extra, dict) else ""
    ).strip()
    if not bot_id or not conversation_id:
        raise ValueError("当前 Bot 或活动会话身份不可用")
    kind = str(values.get("kind", "") or "").strip().casefold()
    if kind not in _TASK_KINDS:
        raise ValueError("kind must be agent, notify, or script")
    trigger_type, trigger_config = normalize_active_task_trigger(
        str(values.get("trigger_type", "") or ""),
        values.get("trigger_config"),
    )
    entrypoint = str(values.get("entrypoint", "") or "").strip()
    cwd = str(values.get("cwd", "") or "").strip()
    args = values.get("args")
    allow_network = values.get("allow_network", kind == "agent")
    if not isinstance(allow_network, bool):
        raise TypeError("allow_network must be a boolean")
    if kind != "agent" and allow_network:
        raise ValueError("allow_network is only valid for agent tasks")
    expected_hash = ""
    if kind == "script":
        if not entrypoint:
            raise ValueError("script task requires entrypoint")
        identity = build_script_identity(entrypoint, cwd or Path.cwd())
        entrypoint = identity.entrypoint
        cwd = identity.cwd
        expected_hash = identity.sha256
    elif entrypoint or cwd or args:
        raise ValueError("entrypoint, cwd, and args are only valid for script tasks")
    return {
        "session_key": actor["session_key"],
        "user_id": actor["user_id"],
        "bot_id": bot_id,
        "conversation_id": conversation_id,
        "name": str(values.get("name", "") or ""),
        "kind": kind,
        "instruction": str(values.get("instruction", "") or ""),
        "trigger_type": trigger_type,
        "trigger_config": trigger_config,
        "entrypoint": entrypoint or None,
        "cwd": cwd or None,
        "args": list(args) if isinstance(args, list | tuple) else args,
        "expected_entrypoint_sha256": expected_hash or None,
        "allow_network": allow_network,
    }


def prepare_active_task_update_payload(
    values: dict[str, Any],
    *,
    actor: dict[str, str],
) -> dict[str, Any]:
    task, error = resolve_active_task_reference(
        actor["session_key"],
        str(values.get("task", "") or ""),
    )
    if task is None:
        raise KeyError(error)
    editable = {
        "name",
        "instruction",
        "trigger_type",
        "trigger_config",
        "entrypoint",
        "cwd",
        "args",
        "allow_network",
    }
    changes = {
        key: values[key]
        for key in editable
        if key in values and values[key] is not None
    }
    if not changes:
        raise ValueError("没有提供需要修改的字段")
    trigger_type = str(
        changes.get("trigger_type", task.trigger_type) or ""
    ).casefold()
    if "trigger_type" in changes or "trigger_config" in changes:
        normalized_type, normalized_config = normalize_active_task_trigger(
            trigger_type,
            changes.get("trigger_config", task.trigger_config),
        )
        changes["trigger_type"] = normalized_type
        changes["trigger_config"] = normalized_config
    allow_network = changes.get("allow_network", task.allow_network)
    if not isinstance(allow_network, bool):
        raise TypeError("allow_network must be a boolean")
    if task.kind != "agent" and allow_network:
        raise ValueError("allow_network is only valid for agent tasks")
    if task.kind == "script":
        identity = build_script_identity(
            str(changes.get("entrypoint", task.entrypoint) or ""),
            str(changes.get("cwd", task.cwd) or "") or None,
        )
        changes["entrypoint"] = identity.entrypoint
        changes["cwd"] = identity.cwd
        changes["expected_entrypoint_sha256"] = identity.sha256
    elif any(key in changes for key in ("entrypoint", "cwd", "args")):
        raise ValueError("脚本字段仅适用于 script 任务")
    return {
        "session_key": actor["session_key"],
        "task_id": task.task_id,
        "changes": changes,
    }


def validate_active_task_payload_paths(payload: dict[str, Any]):
    source = payload.get("changes")
    values = source if isinstance(source, dict) else payload
    for key in ("entrypoint", "cwd"):
        value = str(values.get(key, "") or "").strip()
        if value and (denied := file_path_deny(value)) is not None:
            return denied
    return None


def validate_active_task_approval_payload(
    *,
    action: str,
    payload: dict[str, Any],
    actor: dict[str, str],
) -> ToolResult | None:
    from ...config import active_tasks_enabled

    if not active_tasks_enabled():
        return tool_result(False, "active_tasks_disabled")
    if str(payload.get("session_key", "") or "") != actor["session_key"]:
        return tool_result(False, "approval_payload_invalid", error="任务 owner 不匹配")
    if action == "active_task_create":
        active = get_active_conversation(actor["session_key"])
        if active is None or str(active.get("id", "") or "") != str(
            payload.get("conversation_id", "") or ""
        ):
            return tool_result(
                False,
                "approval_payload_invalid",
                error="创建任务时绑定的会话已不再活动",
            )
        denied = validate_active_task_payload_paths(payload)
        if denied is not None:
            return permission_denied_result(
                actor=actor,
                action=action,
                payload=active_task_audit_payload(payload),
                permission=denied,
            )
        if str(payload.get("kind", "") or "") == "script":
            try:
                identity = build_script_identity(
                    str(payload.get("entrypoint", "") or ""),
                    str(payload.get("cwd", "") or "") or None,
                )
            except (OSError, TypeError, ValueError) as exc:
                return tool_result(False, "approval_payload_invalid", error=str(exc))
            expected = str(payload.get("expected_entrypoint_sha256", "") or "")
            if not expected or not hmac.compare_digest(identity.sha256, expected):
                return tool_result(
                    False,
                    "approval_payload_invalid",
                    error="脚本内容已在审批后变化，请重新创建任务并确认",
                )
        return None
    if action == "active_task_update":
        task = get_active_task(
            str(payload.get("task_id", "") or ""),
            actor["session_key"],
        )
        changes = payload.get("changes")
        if task is None or not isinstance(changes, dict) or not changes:
            return tool_result(False, "approval_payload_invalid")
        denied = validate_active_task_payload_paths(payload)
        if denied is not None:
            return permission_denied_result(
                actor=actor,
                action=action,
                payload=active_task_audit_payload(payload),
                permission=denied,
            )
        if task.kind == "script":
            try:
                identity = build_script_identity(
                    str(changes.get("entrypoint", task.entrypoint) or ""),
                    str(changes.get("cwd", task.cwd) or "") or None,
                )
            except (OSError, TypeError, ValueError) as exc:
                return tool_result(False, "approval_payload_invalid", error=str(exc))
            expected = str(
                changes.get("expected_entrypoint_sha256", "") or ""
            )
            if not expected or not hmac.compare_digest(identity.sha256, expected):
                return tool_result(
                    False,
                    "approval_payload_invalid",
                    error="脚本内容已在审批后变化，请重新编辑任务并确认",
                )
        return None
    task = get_active_task(str(payload.get("task_id", "") or ""), actor["session_key"])
    if task is None:
        return tool_result(False, "active_task_not_found")
    if str(payload.get("action", "") or "") not in _CONTROL_ACTIONS:
        return tool_result(False, "approval_payload_invalid")
    if (
        str(payload.get("action", "") or "") == "rotate_webhook"
        and task.trigger_type != "webhook"
    ):
        return tool_result(False, "approval_payload_invalid")
    return None


async def execute_active_task_create_payload(
    payload: dict[str, Any],
    *,
    actor: dict[str, str],
) -> ToolResult:
    from ..proactive_tasks import (
        deliver_active_task_webhook_credential,
        generate_active_task_webhook_token,
    )

    validation = validate_active_task_approval_payload(
        action="active_task_create",
        payload=payload,
        actor=actor,
    )
    if validation is not None:
        return validation
    token = (
        generate_active_task_webhook_token()
        if payload.get("trigger_type") == "webhook"
        else None
    )
    create_payload = {key: payload.get(key) for key in _CREATE_STORE_FIELDS}
    create_payload["allow_network"] = bool(
        payload.get("allow_network", payload.get("kind") == "agent")
    )
    create_payload["webhook_token_hash"] = token.token_hash if token else None
    try:
        task = await create_active_task(**create_payload)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return tool_result(False, "active_task_create_failed", error=str(exc))
    webhook_path = ""
    if token is not None:
        try:
            webhook_path = await deliver_active_task_webhook_credential(
                task,
                token.token,
            )
        except Exception as exc:
            cleanup_error = ""
            try:
                await control_active_task(
                    task.task_id,
                    "delete",
                    actor["session_key"],
                )
            except Exception as cleanup_exc:
                cleanup_error = f"；任务清理失败：{cleanup_exc}"
            return tool_result(
                False,
                "active_task_create_failed",
                error=f"Webhook 凭据投递失败：{exc}{cleanup_error}",
            )
    record_audit_event(
        event="operation_executed",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action="active_task_create",
        payload=active_task_audit_payload(payload),
        result={"ok": True, "task_id": task.task_id},
    )
    result: dict[str, Any] = {
        "task": active_task_view(task),
        "summary": f"主动任务已创建：{task.name}",
    }
    if token is not None:
        result["webhook"] = {
            "path": webhook_path or f"{_WEBHOOK_ROUTE_PREFIX}/{task.task_id}",
            "credential_delivered": True,
        }
    return tool_result(True, "active_task_created", **result)


async def execute_active_task_control_payload(
    payload: dict[str, Any],
    *,
    actor: dict[str, str],
) -> ToolResult:
    from ..proactive_tasks import (
        deliver_active_task_webhook_credential,
        generate_active_task_webhook_token,
        get_proactive_dispatcher,
    )

    validation = validate_active_task_approval_payload(
        action="active_task_control",
        payload=payload,
        actor=actor,
    )
    if validation is not None:
        return validation
    task_id = str(payload.get("task_id", "") or "")
    action = str(payload.get("action", "") or "")
    try:
        if action == "run_now":
            dispatch_status = await get_proactive_dispatcher().submit_manual(task_id)
            task = get_active_task(task_id, actor["session_key"])
        elif action == "rotate_webhook":
            token = generate_active_task_webhook_token()
            task = rotate_active_task_webhook_token(
                task_id,
                actor["session_key"],
                token.token_hash,
            )
            try:
                await deliver_active_task_webhook_credential(task, token.token)
            except Exception as exc:
                task_paused = False
                try:
                    await control_active_task(task_id, "pause", actor["session_key"])
                    task_paused = True
                except Exception:
                    pass
                update_active_task_status(
                    task_id,
                    session_key=actor["session_key"],
                    status="credential_delivery_failed",
                    error=str(exc),
                    increment_run_count=False,
                    execution_status="not_started",
                    delivery_status="failed",
                    touch_run_at=False,
                )
                return tool_result(
                    False,
                    "webhook_credential_delivery_failed",
                    error=str(exc),
                    task_paused=task_paused,
                )
            dispatch_status = "credential_rotated"
        else:
            task = await control_active_task(task_id, action, actor["session_key"])
            dispatch_status = ""
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return tool_result(False, "active_task_control_failed", error=str(exc))
    record_audit_event(
        event="operation_executed",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action="active_task_control",
        payload={"task_id": task_id, "action": action},
        result={"ok": True, "dispatch_status": dispatch_status},
    )
    return tool_result(
        True,
        "active_task_controlled",
        action=action,
        task=active_task_view(task) if task is not None else None,
        dispatch_status=dispatch_status or None,
        summary="主动任务操作已接受。",
    )


async def execute_active_task_update_payload(
    payload: dict[str, Any],
    *,
    actor: dict[str, str],
) -> ToolResult:
    validation = validate_active_task_approval_payload(
        action="active_task_update",
        payload=payload,
        actor=actor,
    )
    if validation is not None:
        return validation
    task_id = str(payload.get("task_id", "") or "")
    changes = payload.get("changes")
    try:
        task = await update_active_task(
            task_id,
            actor["session_key"],
            changes if isinstance(changes, dict) else {},
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return tool_result(False, "active_task_update_failed", error=str(exc))
    record_audit_event(
        event="operation_executed",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action="active_task_update",
        payload=active_task_audit_payload(payload),
        result={"ok": True, "task_id": task.task_id},
    )
    return tool_result(
        True,
        "active_task_updated",
        task=active_task_view(task, detailed=True),
        summary=f"主动任务已更新：{task.name}",
    )
def resolve_active_task_reference(
    session_key: str,
    reference: str,
) -> tuple[ActiveTask | None, str]:
    value = str(reference or "").strip()
    if not value:
        return None, "task reference is required"
    tasks = list_active_tasks(session_key)
    folded = value.casefold()
    matches = [
        task
        for task in tasks
        if task.task_id.casefold() == folded
        or task.task_id.casefold().startswith(folded)
        or task.name.casefold() == folded
    ]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, "任务名称或 ID 前缀存在歧义，请使用更长 ID"
    return None, "未找到该主动任务"


def active_task_view(
    task: ActiveTask,
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    view: dict[str, Any] = {
        "id": _short_task_id(task.task_id),
        "name": task.name,
        "kind": task.kind,
        "trigger": {"type": task.trigger_type, **task.trigger_config},
        "enabled": task.enabled,
        "conversation_id": task.conversation_id,
        "last_status": task.last_status or None,
        "last_execution_status": task.last_execution_status or None,
        "last_delivery_status": task.last_delivery_status or None,
        "last_error": str(task.last_error or "")[:500] or None,
        "last_run_at": task.last_run_at,
        "next_run_time": active_task_next_run_time(task),
        "run_count": task.run_count,
        "allow_network": task.allow_network,
        "script_sha256": task.entrypoint_sha256[:12]
        if task.entrypoint_sha256
        else None,
    }
    if detailed:
        instruction = str(task.instruction or "")
        view.update(
            {
                "full_id": task.task_id,
                "instruction": _bounded_detail_text(instruction),
                "instruction_chars": len(instruction),
                "entrypoint": task.entrypoint,
                "cwd": task.cwd,
                "args": list(task.args),
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            }
        )
    return view


def active_task_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    changes = payload.get("changes")
    source = changes if isinstance(changes, dict) else payload
    result = {
        key: source.get(key)
        for key in (
            "bot_id",
            "conversation_id",
            "name",
            "kind",
            "trigger_type",
            "trigger_config",
            "entrypoint",
            "cwd",
            "expected_entrypoint_sha256",
            "allow_network",
        )
        if source.get(key) not in (None, "", [], {})
    }
    for key in ("task_id", "action"):
        if payload.get(key) not in (None, ""):
            result[key] = payload.get(key)
    result["instruction_chars"] = len(str(source.get("instruction", "") or ""))
    args = source.get("args")
    result["args_count"] = len(args) if isinstance(args, list | tuple) else 0
    return result


def _short_task_id(task_id: str) -> str:
    return str(task_id or "")[:_TASK_ID_PREFIX_LENGTH]


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(default if value in (None, "") else value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _bounded_detail_text(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    head = max(limit - 1000, 1)
    return f"{value[:head]}\n...[truncated]...\n{value[-900:]}"


__all__ = [
    "ActiveTaskControlTool",
    "ActiveTaskCreateTool",
    "ActiveTaskListTool",
    "ActiveTaskUpdateTool",
    "active_task_audit_payload",
    "execute_active_task_control_payload",
    "execute_active_task_create_payload",
    "execute_active_task_update_payload",
    "prepare_active_task_create_payload",
    "prepare_active_task_update_payload",
    "validate_active_task_approval_payload",
]
