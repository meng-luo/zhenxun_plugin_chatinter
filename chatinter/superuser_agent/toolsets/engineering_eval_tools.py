"""Engineering eval tools for reliable superuser Agent code work."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..audit_log import record_audit_event
from ..engineering_eval import (
    build_failure_observation,
    create_engineering_eval,
    get_engineering_eval,
    list_engineering_evals,
    mark_eval_step,
    suggest_test_commands,
)
from ..permission_policy import decide_eval
from ..registry import register_superuser_tool
from ..workspace_isolation import resolve_cwd
from .common import (
    actor_from_context,
    approval_required_result,
    permission_denied_result,
    tool_result,
    worktree_id_from_context,
)
from .shell_tools import run_shell_command


class EngineeringEvalPlanTool:
    name = "engineering_eval_plan"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：为工程任务建立 eval 闭环计划，记录读代码、"
                "改代码、跑测试、必要时回滚的步骤。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "工程任务描述。"},
                    "files": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "涉及文件路径。",
                    },
                    "tests": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "验收命令，例如 py -3 -m compileall -q ...。",
                    },
                    "rollback_operation_id": {
                        "type": ["string", "null"],
                        "description": "可选 patch operation_id，失败时可回滚。",
                    },
                    "engineering_loop_id": {
                        "type": ["string", "null"],
                        "description": "可选工程闭环 loop_id；传入后自动绑定 eval gate。",
                    },
                },
                "required": [
                    "task",
                    "files",
                    "tests",
                    "rollback_operation_id",
                    "engineering_loop_id",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        decision = decide_eval("engineering_eval_plan")
        payload = _plan_payload(kwargs)
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action=self.name,
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action=self.name,
                payload=payload,
                permission=decision,
            )
        eval_run = create_engineering_eval(actor=actor, **payload)
        _bind_engineering_loop_eval(
            engineering_loop_id=str(kwargs.get("engineering_loop_id", "") or ""),
            eval_id=eval_run.eval_id,
        )
        return tool_result(
            True,
            "engineering_eval_planned",
            eval=eval_run.public_payload(),
            engineering_loop=_loop_payload(str(kwargs.get("engineering_loop_id", "") or "")),
            suggested_tests=eval_run.suggested_tests,
            instruction=(
                "按 eval steps 执行：先读代码，再修改，再用 engineering_eval_run "
                "运行测试并记录结果。"
            ),
        )


class EngineeringEvalRunTool:
    name = "engineering_eval_run"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：执行工程 eval 中的某个测试命令并记录结果。"
                "命令会经过 eval 权限策略；实际执行走 shell。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "eval_id": {"type": "string", "description": "eval_id。"},
                    "step_index": {
                        "type": ["integer", "null"],
                        "description": "要执行的 step 索引，从 0 开始；为空则执行下一个 run_test。",
                    },
                    "command": {
                        "type": ["string", "null"],
                        "description": "覆盖步骤里的命令；通常为空。",
                    },
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "工作目录，留空使用当前目录。",
                    },
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "description": "超时时间，默认 20 秒，最大 120 秒。",
                    },
                    "engineering_loop_id": {
                        "type": ["string", "null"],
                        "description": "可选工程闭环 loop_id；传入后更新 eval gate。",
                    },
                },
                "required": [
                    "eval_id",
                    "step_index",
                    "command",
                    "cwd",
                    "timeout_seconds",
                    "engineering_loop_id",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        return await run_engineering_eval_step(
            actor=actor,
            eval_id=str(kwargs.get("eval_id", "") or "").strip(),
            step_index=kwargs.get("step_index"),
            command=str(kwargs.get("command", "") or "").strip(),
            cwd=str(kwargs.get("cwd", "") or "") or None,
            timeout_seconds=kwargs.get("timeout_seconds"),
            worktree_id=worktree_id_from_context(context),
            engineering_loop_id=str(kwargs.get("engineering_loop_id", "") or ""),
        )


class EngineeringEvalStatusTool:
    name = "engineering_eval_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：查看工程 eval 状态，或列出最近 eval。",
            parameters={
                "type": "object",
                "properties": {
                    "eval_id": {
                        "type": ["string", "null"],
                        "description": "为空则列出最近 eval；否则查看指定 eval。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "列表数量，默认 20。",
                    },
                },
                "required": ["eval_id", "limit"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        eval_id = str(kwargs.get("eval_id", "") or "").strip()
        if eval_id:
            eval_run = get_engineering_eval(eval_id)
            if eval_run is None or eval_run.user_id != actor["user_id"] or eval_run.session_key != actor["session_key"]:
                return tool_result(False, "engineering_eval_not_found", eval_id=eval_id)
            return tool_result(True, "engineering_eval_status", eval=eval_run.public_payload())
        rows = list_engineering_evals(
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            limit=_coerce_limit(kwargs.get("limit")),
        )
        return tool_result(
            True,
            "engineering_evals_listed",
            evals=[item.public_payload() for item in rows],
            count=len(rows),
        )


def _plan_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    files = [str(item) for item in kwargs.get("files", []) or []]
    tests = [str(item) for item in kwargs.get("tests", []) or []]
    return {
        "task": str(kwargs.get("task", "") or ""),
        "files": files,
        "tests": tests or suggest_test_commands(files=files, task=str(kwargs.get("task", "") or "")),
        "rollback_operation_id": str(kwargs.get("rollback_operation_id", "") or ""),
    }


async def run_engineering_eval_step(
    *,
    actor: dict[str, str],
    eval_id: str,
    step_index: Any = None,
    command: str = "",
    cwd: str | None = None,
    timeout_seconds: Any = None,
    worktree_id: str = "",
    pre_resolved_cwd: bool = False,
    approval_id: str | None = None,
    engineering_loop_id: str = "",
) -> ToolResult:
    eval_run = get_engineering_eval(eval_id)
    if eval_run is None or eval_run.user_id != actor["user_id"] or eval_run.session_key != actor["session_key"]:
        return tool_result(False, "engineering_eval_not_found", eval_id=eval_id)
    resolved_index = _resolve_step_index(eval_run, step_index)
    if resolved_index < 0:
        return tool_result(
            False,
            "engineering_eval_no_runnable_step",
            eval=eval_run.public_payload(),
        )
    step = eval_run.steps[resolved_index]
    cwd, isolation = (
        (
            cwd or step.cwd,
            {
                "isolated": False,
                "pre_resolved": True,
                "resolved": cwd or step.cwd or "",
            },
        )
        if pre_resolved_cwd
        else resolve_cwd(cwd or step.cwd, actor=actor, worktree_id=worktree_id)
    )
    if isolation.get("invalid_worktree") or isolation.get("escaped_worktree"):
        return tool_result(False, "worktree_resolution_failed", cwd=cwd, isolation=isolation)
    command = str(command or step.command or "").strip()
    if not command:
        return tool_result(
            False,
            "engineering_eval_empty_command",
            eval_id=eval_id,
            step_index=resolved_index,
        )
    payload = {
        "eval_id": eval_id,
        "step_index": resolved_index,
        "command": command,
        "cwd": cwd,
        "isolation": isolation,
    }
    if not approval_id:
        decision = decide_eval("engineering_eval_run " + command)
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="engineering_eval_run",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="engineering_eval_run",
                payload={**payload, "timeout_seconds": timeout_seconds},
                permission=decision,
            )
    result = await run_shell_command(
        command=command,
        cwd=payload["cwd"],
        actor=actor,
        approval_id=approval_id,
        timeout_seconds=timeout_seconds,
        action="engineering_eval_run",
        isolation=isolation,
    )
    ok = bool(isinstance(result.output, dict) and result.output.get("ok"))
    command_result = dict(result.output) if isinstance(result.output, dict) else {}
    eval_run = mark_eval_step(
        eval_id=eval_id,
        step_index=resolved_index,
        status="passed" if ok else "failed",
        result=command_result,
    )
    _update_engineering_loop_from_eval(
        engineering_loop_id=engineering_loop_id,
        eval_id=eval_id,
    )
    failure_observation = (
        build_failure_observation(eval_run, step_index=resolved_index)
        if eval_run is not None and not ok
        else {}
    )
    if eval_run is not None and not ok:
        _diagnose_engineering_loop_failure(
            engineering_loop_id=engineering_loop_id,
            eval_id=eval_id,
            command=command,
        )
    record_audit_event(
        event="engineering_eval_step_ran",
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        action="engineering_eval_run",
        payload={**payload, "approval_id": approval_id},
        result={"ok": ok},
    )
    return tool_result(
        ok,
        "engineering_eval_step_passed" if ok else "engineering_eval_step_failed",
        eval=eval_run.public_payload() if eval_run is not None else None,
        engineering_loop=_loop_payload(engineering_loop_id),
        command_result=command_result,
        observation=failure_observation,
        recommended_next_action=(
            "run_next_eval_step"
            if ok
            else failure_observation.get("recommended_next_action", "inspect_output")
        ),
        retryable=not ok,
        need_continue=True,
        instruction=_eval_instruction(ok=ok, observation=failure_observation),
    )


def _resolve_step_index(eval_run: Any, raw_index: Any) -> int:
    if raw_index not in (None, ""):
        try:
            index = int(raw_index)
            return index if 0 <= index < len(eval_run.steps) else -1
        except (TypeError, ValueError):
            return -1
    for index, step in enumerate(eval_run.steps):
        if step.kind == "run_test" and step.status == "pending":
            return index
    return -1


def _coerce_limit(value: Any) -> int:
    try:
        return max(1, min(int(value or 20), 100))
    except (TypeError, ValueError):
        return 20


def _eval_instruction(*, ok: bool, observation: dict[str, Any]) -> str:
    if ok:
        return "测试通过。继续下一个 eval step；全部通过后再总结。"
    if not observation:
        return "测试失败。读取 command_result，判断是否需要重读代码、二次 patch 或 rollback。"
    actions: list[str] = []
    if observation.get("needs_reread"):
        actions.append("先重读相关代码")
    if observation.get("needs_second_patch"):
        actions.append("准备二次 patch")
    if observation.get("suggest_rollback"):
        actions.append("考虑 patch_rollback")
    return "测试失败；建议：" + "、".join(actions or ["检查输出后决定下一步"])


def _bind_engineering_loop_eval(
    *,
    engineering_loop_id: str,
    eval_id: str,
) -> None:
    if not engineering_loop_id or not eval_id:
        return
    try:
        from ..engineering_loop import bind_eval

        bind_eval(loop_id=engineering_loop_id, eval_id=eval_id)
    except Exception:
        return


def _update_engineering_loop_from_eval(
    *,
    engineering_loop_id: str,
    eval_id: str,
) -> None:
    if not engineering_loop_id or not eval_id:
        return
    try:
        from ..engineering_loop import update_loop_from_eval

        update_loop_from_eval(loop_id=engineering_loop_id, eval_id=eval_id)
    except Exception:
        return


def _diagnose_engineering_loop_failure(
    *,
    engineering_loop_id: str,
    eval_id: str,
    command: str,
) -> None:
    if not engineering_loop_id:
        return
    try:
        from ..engineering_loop import diagnose_eval_failure

        diagnose_eval_failure(
            loop_id=engineering_loop_id,
            eval_id=eval_id,
            notes=f"eval command failed: {command}",
        )
    except Exception:
        return


def _loop_payload(engineering_loop_id: str) -> dict[str, Any] | None:
    if not engineering_loop_id:
        return None
    try:
        from ..engineering_loop import get_engineering_loop

        loop = get_engineering_loop(engineering_loop_id)
        return loop.public_payload(include_events=False) if loop else None
    except Exception:
        return None


register_superuser_tool(EngineeringEvalPlanTool)
register_superuser_tool(EngineeringEvalRunTool)
register_superuser_tool(EngineeringEvalStatusTool)

__all__ = [
    "EngineeringEvalPlanTool",
    "EngineeringEvalRunTool",
    "EngineeringEvalStatusTool",
    "run_engineering_eval_step",
]
