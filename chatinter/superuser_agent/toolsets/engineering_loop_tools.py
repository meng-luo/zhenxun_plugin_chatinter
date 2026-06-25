"""Fixed engineering loop protocol tools."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..engineering_loop import (
    bind_eval,
    bind_patch_operation,
    build_semantic_patch_plan,
    complete_engineering_loop,
    create_engineering_loop,
    diagnose_eval_failure,
    eval_gate,
    get_engineering_loop,
    list_engineering_loops,
    read_code_symbols,
    record_lsp_read,
    record_semantic_patch_plan,
)
from ..registry import register_superuser_tool
from ..workspace_isolation import resolve_working_path
from .common import actor_from_context, tool_result, worktree_id_from_context


class EngineeringLoopStartTool:
    name = "engineering_loop_start"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：启动固定工程闭环协议。用于代码任务的 "
                "LSP 读代码 -> semantic patch -> checkpoint -> eval gate -> rollback。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "工程任务目标。"},
                    "files": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "已知相关文件；未知可为空。",
                    },
                },
                "required": ["task", "files"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        files = _text_list(kwargs.get("files"))
        loop = create_engineering_loop(
            actor=actor,
            task=str(kwargs.get("task", "") or ""),
            files=files,
        )
        return tool_result(
            True,
            "engineering_loop_started",
            loop=loop.public_payload(),
            next_tool="engineering_lsp_read",
            protocol=_protocol_steps(),
            instruction=(
                "进入固定工程闭环。下一步先用 engineering_lsp_read 读取相关代码；"
                "不要跳过读代码直接 patch。"
            ),
        )


class EngineeringLoopStatusTool:
    name = "engineering_loop_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：查看工程闭环状态，或列出最近闭环。",
            parameters={
                "type": "object",
                "properties": {
                    "loop_id": {
                        "type": ["string", "null"],
                        "description": "为空则列出最近闭环；否则查看指定闭环。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "列表数量，默认 20。",
                    },
                },
                "required": ["loop_id", "limit"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        loop_id = str(kwargs.get("loop_id", "") or "").strip()
        if loop_id:
            loop = get_engineering_loop(loop_id)
            if (
                loop is None
                or loop.user_id != actor["user_id"]
                or loop.session_key != actor["session_key"]
            ):
                return tool_result(False, "engineering_loop_not_found", loop_id=loop_id)
            return tool_result(
                True,
                "engineering_loop_status",
                loop=loop.public_payload(),
                gate=eval_gate(loop_id=loop_id),
            )
        rows = list_engineering_loops(
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            limit=_coerce_limit(kwargs.get("limit")),
        )
        return tool_result(
            True,
            "engineering_loops_listed",
            loops=[item.public_payload(include_events=False) for item in rows],
            count=len(rows),
        )


class EngineeringLspReadTool:
    name = "engineering_lsp_read"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：以轻量 LSP/符号索引方式读取代码结构，"
                "返回类、函数、引用数和诊断，并记录到工程闭环。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "loop_id": {
                        "type": "string",
                        "description": "engineering_loop_start 返回的 loop_id。",
                    },
                    "files": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "要读取的文件；为空则使用 loop 已知文件。",
                    },
                    "query": {
                        "type": ["string", "null"],
                        "description": "按任务/符号关键词筛选。",
                    },
                    "max_symbols": {
                        "type": ["integer", "null"],
                        "description": "最多返回符号数，默认 120。",
                    },
                },
                "required": ["loop_id", "files", "query", "max_symbols"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        loop_id = str(kwargs.get("loop_id", "") or "").strip()
        loop = get_engineering_loop(loop_id)
        if (
            loop is None
            or loop.user_id != actor["user_id"]
            or loop.session_key != actor["session_key"]
        ):
            return tool_result(False, "engineering_loop_not_found", loop_id=loop_id)
        requested_files = _text_list(kwargs.get("files")) or loop.files
        resolved_files, isolation = _resolve_files(
            requested_files,
            actor=actor,
            worktree_id=worktree_id_from_context(context),
        )
        if isolation.get("invalid_worktree") or isolation.get("escaped_worktree"):
            return tool_result(False, "worktree_resolution_failed", isolation=isolation)
        symbols, diagnostics = read_code_symbols(
            files=resolved_files,
            query=str(kwargs.get("query", "") or loop.task),
            max_symbols=_coerce_limit(
                kwargs.get("max_symbols"), default=120, max_value=300
            ),
        )
        loop = record_lsp_read(
            loop_id=loop_id,
            files=resolved_files,
            symbols=symbols,
            diagnostics=diagnostics,
        )
        return tool_result(
            True,
            "engineering_lsp_read",
            loop=loop.public_payload() if loop else None,
            symbols=[item.public_payload() for item in symbols],
            diagnostics=diagnostics,
            isolation=isolation,
            next_tool="semantic_patch_plan",
            instruction=(
                "已完成代码结构读取。下一步用 semantic_patch_plan 形成语义补丁计划，"
                "再用 patch_prepare 生成可审计 diff。"
            ),
        )


class SemanticPatchPlanTool:
    name = "semantic_patch_plan"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：基于 LSP 符号和任务目标生成语义补丁计划。"
                "它不写文件，只把目标符号、约束、后续 patch/eval 协议固定下来。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "loop_id": {"type": "string", "description": "工程闭环 loop_id。"},
                    "instructions": {
                        "type": ["string", "null"],
                        "description": "额外修改策略或边界。",
                    },
                },
                "required": ["loop_id", "instructions"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        loop_id = str(kwargs.get("loop_id", "") or "").strip()
        loop = get_engineering_loop(loop_id)
        if (
            loop is None
            or loop.user_id != actor["user_id"]
            or loop.session_key != actor["session_key"]
        ):
            return tool_result(False, "engineering_loop_not_found", loop_id=loop_id)
        if not loop.symbols:
            return tool_result(
                False,
                "engineering_lsp_read_required",
                loop=loop.public_payload(include_events=False),
                next_tool="engineering_lsp_read",
                instruction=(
                    "还没有读代码证据。先调用 engineering_lsp_read，"
                    "再生成 semantic patch plan。"
                ),
            )
        plan = build_semantic_patch_plan(
            task=loop.task,
            files=loop.files,
            symbols=loop.symbols,
            instructions=str(kwargs.get("instructions", "") or ""),
        )
        loop = record_semantic_patch_plan(loop_id=loop_id, plan=plan)
        return tool_result(
            True,
            "semantic_patch_planned",
            loop=loop.public_payload() if loop else None,
            plan=plan,
            next_tool="patch_prepare",
            required_patch_context={
                "engineering_loop_id": loop_id,
                "attempt": loop.attempt if loop else 0,
                "diagnosis": loop.diagnosis if loop else {},
            },
            instruction=(
                "语义补丁计划已记录。下一步用 patch_prepare 生成 diff；"
                "patch_apply 后要绑定 eval 并经过 engineering_eval_gate。"
            ),
        )


class EngineeringLoopBindTool:
    name = "engineering_loop_bind"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：把已有 patch operation 或 engineering eval "
                "回绑到工程闭环，用于恢复和统一验收。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "loop_id": {"type": "string", "description": "工程闭环 loop_id。"},
                    "operation_id": {
                        "type": ["string", "null"],
                        "description": "patch operation_id。",
                    },
                    "eval_id": {
                        "type": ["string", "null"],
                        "description": "engineering eval_id。",
                    },
                },
                "required": ["loop_id", "operation_id", "eval_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        loop_id = str(kwargs.get("loop_id", "") or "").strip()
        loop = get_engineering_loop(loop_id)
        if (
            loop is None
            or loop.user_id != actor["user_id"]
            or loop.session_key != actor["session_key"]
        ):
            return tool_result(False, "engineering_loop_not_found", loop_id=loop_id)
        operation_id = str(kwargs.get("operation_id", "") or "").strip()
        eval_id = str(kwargs.get("eval_id", "") or "").strip()
        if operation_id:
            loop = bind_patch_operation(loop_id=loop_id, operation_id=operation_id)
        if eval_id:
            loop = bind_eval(loop_id=loop_id, eval_id=eval_id)
        return tool_result(
            True,
            "engineering_loop_bound",
            loop=loop.public_payload() if loop else None,
            gate=eval_gate(loop_id=loop_id),
        )


class EngineeringEvalGateTool:
    name = "engineering_eval_gate"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：工程闭环最终验收门。只有绑定 eval 通过后才允许"
                "总结完成；失败时返回固定 recovery_plan。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "loop_id": {"type": "string", "description": "工程闭环 loop_id。"}
                },
                "required": ["loop_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        loop_id = str(kwargs.get("loop_id", "") or "").strip()
        loop = get_engineering_loop(loop_id)
        if (
            loop is None
            or loop.user_id != actor["user_id"]
            or loop.session_key != actor["session_key"]
        ):
            return tool_result(False, "engineering_loop_not_found", loop_id=loop_id)
        gate = eval_gate(loop_id=loop_id)
        return tool_result(
            bool(gate.get("ok")),
            "engineering_eval_gate_passed"
            if gate.get("ok")
            else "engineering_eval_gate_blocked",
            gate=gate,
            retryable=not bool(gate.get("ok")),
            need_continue=not bool(gate.get("ok")),
            instruction=(
                "验收通过，可以总结。"
                if gate.get("ok")
                else (
                    "验收未通过。按 gate.next_tools/recovery_plan 继续，"
                    "不要声称工程任务完成。"
                )
            ),
        )


class EngineeringFailureDiagnoseTool:
    name = "engineering_failure_diagnose"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：把工程 eval 失败固化成诊断和下一步协议。"
                "eval 失败后必须先调用它，再进行二次 patch、重跑测试或 rollback。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "loop_id": {"type": "string", "description": "工程闭环 loop_id。"},
                    "eval_id": {
                        "type": ["string", "null"],
                        "description": "可选 eval_id；为空使用 loop 绑定的 eval。",
                    },
                    "notes": {
                        "type": ["string", "null"],
                        "description": "可选人工/模型诊断补充，不作为完成证据。",
                    },
                },
                "required": ["loop_id", "eval_id", "notes"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        loop_id = str(kwargs.get("loop_id", "") or "").strip()
        loop = get_engineering_loop(loop_id)
        if (
            loop is None
            or loop.user_id != actor["user_id"]
            or loop.session_key != actor["session_key"]
        ):
            return tool_result(False, "engineering_loop_not_found", loop_id=loop_id)
        loop = diagnose_eval_failure(
            loop_id=loop_id,
            eval_id=str(kwargs.get("eval_id", "") or "").strip(),
            notes=str(kwargs.get("notes", "") or ""),
        )
        if loop is None:
            return tool_result(False, "engineering_loop_not_found", loop_id=loop_id)
        diagnosis = dict(loop.diagnosis or {})
        return tool_result(
            True,
            "engineering_failure_diagnosed",
            loop=loop.public_payload(),
            diagnosis=diagnosis,
            gate=eval_gate(loop_id=loop_id),
            next_tools=diagnosis.get("allowed_next_tools", []),
            retryable=True,
            need_continue=True,
            instruction=(
                "失败诊断已固化。按 diagnosis.allowed_next_tools 推进："
                "通常先 engineering_lsp_read 重读相关代码，再 semantic_patch_plan，"
                "然后 patch_prepare/patch_apply，最后 engineering_eval_run/gate。"
            ),
        )


class EngineeringLoopCompleteTool:
    name = "engineering_loop_complete"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：在 eval gate 通过后标记工程闭环完成。",
            parameters={
                "type": "object",
                "properties": {
                    "loop_id": {"type": "string", "description": "工程闭环 loop_id。"},
                    "reason": {
                        "type": ["string", "null"],
                        "description": "完成说明。",
                    },
                },
                "required": ["loop_id", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        loop_id = str(kwargs.get("loop_id", "") or "").strip()
        loop = get_engineering_loop(loop_id)
        if (
            loop is None
            or loop.user_id != actor["user_id"]
            or loop.session_key != actor["session_key"]
        ):
            return tool_result(False, "engineering_loop_not_found", loop_id=loop_id)
        loop = complete_engineering_loop(
            loop_id=loop_id,
            reason=str(kwargs.get("reason", "") or ""),
        )
        ok = bool(loop and loop.stage == "completed")
        return tool_result(
            ok,
            "engineering_loop_completed"
            if ok
            else "engineering_loop_completion_blocked",
            loop=loop.public_payload() if loop else None,
            gate=eval_gate(loop_id=loop_id),
            retryable=not ok,
            need_continue=not ok,
        )


def _resolve_files(
    files: list[str],
    *,
    actor: dict[str, str],
    worktree_id: str,
) -> tuple[list[str], dict[str, Any]]:
    resolved: list[str] = []
    payloads: list[dict[str, Any]] = []
    for path in files:
        resolved_path, payload = resolve_working_path(
            path,
            actor=actor,
            worktree_id=worktree_id,
        )
        payloads.append(payload)
        resolved.append(resolved_path)
    merged: dict[str, Any] = dict(payloads[0]) if payloads else {"isolated": False}
    merged["files"] = payloads
    merged["invalid_worktree"] = any(
        bool(item.get("invalid_worktree")) for item in payloads
    )
    merged["escaped_worktree"] = any(
        bool(item.get("escaped_worktree")) for item in payloads
    )
    return resolved, merged


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _coerce_limit(value: Any, *, default: int = 20, max_value: int = 100) -> int:
    try:
        return max(1, min(int(value or default), max_value))
    except (TypeError, ValueError):
        return default


def _protocol_steps() -> list[str]:
    return [
        "engineering_loop_start",
        "engineering_lsp_read",
        "semantic_patch_plan",
        "patch_prepare",
        "patch_apply",
        "engineering_eval_run",
        "engineering_eval_gate",
        "engineering_failure_diagnose",
        "engineering_loop_complete",
    ]


register_superuser_tool(
    EngineeringLoopStartTool,
    category="engineering_loop",
    risk="low",
    read_only=False,
    destructive=False,
    side_effect="mutate",
    tags=("engineering", "protocol", "loop"),
)
register_superuser_tool(
    EngineeringLoopStatusTool,
    category="engineering_loop",
    risk="low",
    read_only=True,
    tags=("engineering", "protocol", "status"),
)
register_superuser_tool(
    EngineeringLspReadTool,
    category="engineering_loop",
    risk="low",
    read_only=True,
    tags=("engineering", "lsp", "read_code"),
)
register_superuser_tool(
    SemanticPatchPlanTool,
    category="engineering_loop",
    risk="low",
    read_only=False,
    destructive=False,
    side_effect="mutate",
    tags=("engineering", "semantic_patch", "plan"),
)
register_superuser_tool(
    EngineeringLoopBindTool,
    category="engineering_loop",
    risk="low",
    read_only=False,
    destructive=False,
    side_effect="mutate",
    tags=("engineering", "protocol", "bind"),
)
register_superuser_tool(
    EngineeringEvalGateTool,
    category="engineering_loop",
    risk="low",
    read_only=True,
    tags=("engineering", "eval", "gate"),
)
register_superuser_tool(
    EngineeringFailureDiagnoseTool,
    category="engineering_loop",
    risk="low",
    read_only=False,
    destructive=False,
    side_effect="mutate",
    tags=("engineering", "eval", "diagnosis", "recovery"),
)
register_superuser_tool(
    EngineeringLoopCompleteTool,
    category="engineering_loop",
    risk="low",
    read_only=False,
    destructive=False,
    side_effect="mutate",
    tags=("engineering", "protocol", "complete"),
)

__all__ = [
    "EngineeringEvalGateTool",
    "EngineeringFailureDiagnoseTool",
    "EngineeringLoopBindTool",
    "EngineeringLoopCompleteTool",
    "EngineeringLoopStartTool",
    "EngineeringLoopStatusTool",
    "EngineeringLspReadTool",
    "SemanticPatchPlanTool",
]
