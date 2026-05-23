"""Worktree isolation tools for superuser engineering Agent tasks."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..permission_policy import decide_git
from ..registry import register_superuser_tool
from ..workspace_isolation import (
    create_worktree_session,
    list_worktree_sessions,
    remove_worktree_session,
    worktree_status,
)
from .common import (
    actor_from_context,
    approval_required_result,
    permission_denied_result,
    tool_result,
)


class WorktreeCreateTool:
    name = "worktree_create"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：为工程修改创建隔离 git worktree。之后文件、"
                "shell、git、patch、eval 工具默认在该 worktree 中执行，避免污染主工作区。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repo_root": {
                        "type": ["string", "null"],
                        "description": "仓库根目录，留空使用当前项目仓库。",
                    },
                    "base_ref": {
                        "type": ["string", "null"],
                        "description": "从哪个 ref 创建，默认 HEAD。",
                    },
                    "branch_name": {
                        "type": ["string", "null"],
                        "description": "可选分支名，默认自动生成 chatinter/...。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么需要隔离工作区。",
                    },
                },
                "required": ["repo_root", "base_ref", "branch_name", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        repo_root = str(kwargs.get("repo_root", "") or "") or None
        base_ref = str(kwargs.get("base_ref", "") or "HEAD") or "HEAD"
        branch_name = str(kwargs.get("branch_name", "") or "")
        reason = str(kwargs.get("reason", "") or "")
        command_preview = "git worktree add"
        decision = decide_git(command_preview)
        payload = {
            "repo_root": repo_root,
            "base_ref": base_ref,
            "branch_name": branch_name,
            "reason": reason,
        }
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
        try:
            session = create_worktree_session(
                actor=actor,
                repo_root=repo_root,
                base_ref=base_ref,
                branch_name=branch_name,
                reason=reason,
            )
        except Exception as exc:
            return tool_result(False, "worktree_create_failed", error=str(exc))
        return tool_result(
            True,
            "worktree_created",
            worktree=session.public_payload(),
            instruction=(
                "隔离 worktree 已启用。后续工程读写、patch、eval、shell/git 命令"
                "默认使用 worktree_path；仓库内绝对路径也会映射到隔离 worktree。"
            ),
        )


class WorktreeStatusTool:
    name = "worktree_status"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：查看当前或指定隔离 worktree 的状态和 diff 摘要。",
            parameters={
                "type": "object",
                "properties": {
                    "worktree_id": {
                        "type": ["string", "null"],
                        "description": "为空则查看当前会话最新 active worktree。",
                    }
                },
                "required": ["worktree_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        return tool_result(
            True,
            "worktree_status",
            **worktree_status(
                actor=actor,
                worktree_id=str(kwargs.get("worktree_id", "") or ""),
            ),
        )


class WorktreeListTool:
    name = "worktree_list"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="超级用户私聊专用：列出当前会话创建的隔离 worktree。",
            parameters={
                "type": "object",
                "properties": {
                    "include_removed": {
                        "type": ["boolean", "null"],
                        "description": "是否包含已移除 worktree，默认 false。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多返回数量，默认 20。",
                    },
                },
                "required": ["include_removed", "limit"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        rows = list_worktree_sessions(
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            include_removed=bool(kwargs.get("include_removed") or False),
            limit=_coerce_limit(kwargs.get("limit")),
        )
        return tool_result(
            True,
            "worktrees_listed",
            worktrees=[item.public_payload() for item in rows],
            count=len(rows),
        )


class WorktreeRemoveTool:
    name = "worktree_remove"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：移除隔离 worktree。默认不强制；有未提交改动时"
                "可能失败，除非 force=true。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "worktree_id": {"type": "string", "description": "worktree_id。"},
                    "force": {
                        "type": ["boolean", "null"],
                        "description": "是否强制移除，默认 false。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "移除原因。",
                    },
                },
                "required": ["worktree_id", "force", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        worktree_id = str(kwargs.get("worktree_id", "") or "").strip()
        force = bool(kwargs.get("force") or False)
        reason = str(kwargs.get("reason", "") or "")
        if not worktree_id:
            return tool_result(False, "worktree_id_required")
        decision = decide_git("git worktree remove --force" if force else "git worktree remove")
        payload = {"worktree_id": worktree_id, "force": force, "reason": reason}
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
        session = remove_worktree_session(
            actor=actor,
            worktree_id=worktree_id,
            force=force,
        )
        if session is None:
            return tool_result(False, "worktree_not_found", worktree_id=worktree_id)
        return tool_result(True, "worktree_removed", worktree=session.public_payload())


def _coerce_limit(value: Any) -> int:
    try:
        return max(1, min(int(value or 20), 100))
    except (TypeError, ValueError):
        return 20


register_superuser_tool(
    WorktreeCreateTool,
    category="worktree",
    risk="medium",
    approval_mode="policy",
    read_only=False,
    todo_relevant=True,
    tags=("worktree", "isolation", "engineering"),
)
register_superuser_tool(
    WorktreeStatusTool,
    category="worktree",
    risk="low",
    approval_mode="policy",
    read_only=True,
    todo_relevant=True,
    tags=("worktree", "isolation", "status"),
)
register_superuser_tool(
    WorktreeListTool,
    category="worktree",
    risk="low",
    approval_mode="policy",
    read_only=True,
    tags=("worktree", "isolation", "status"),
)
register_superuser_tool(
    WorktreeRemoveTool,
    category="worktree",
    risk="high",
    approval_mode="policy",
    read_only=False,
    tags=("worktree", "isolation", "cleanup", "approval_sensitive"),
)

__all__ = [
    "WorktreeCreateTool",
    "WorktreeListTool",
    "WorktreeRemoveTool",
    "WorktreeStatusTool",
]
