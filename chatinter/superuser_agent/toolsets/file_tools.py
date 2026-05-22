"""Filesystem tools for the superuser private Agent scenario."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..audit_log import record_audit_event
from ..patch_operations import FileChange, apply_changes_transaction
from ..permission_policy import decide_file_read, decide_file_write
from ..registry import register_superuser_tool
from .common import (
    actor_from_context,
    approval_required_result,
    audited_error_result,
    coerce_max_chars,
    compact_text,
    permission_denied_result,
    tool_result,
)


class ReadFileTool:
    name = "read_file"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：读取文本文件。受 file.allow_read/ask_read/deny "
                "权限策略控制；ask 会生成待确认操作。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径。"},
                    "max_chars": {
                        "type": ["integer", "null"],
                        "description": "最多返回字符数，默认 4000。",
                    },
                },
                "required": ["path", "max_chars"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "") or "").strip()
        max_chars = coerce_max_chars(kwargs.get("max_chars"))
        actor = actor_from_context(context)
        decision = decide_file_read(path)
        payload = {"path": path, "max_chars": max_chars}
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="read_file",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="read_file",
                payload=payload,
                permission=decision,
            )
        return await read_file(path=path, max_chars=max_chars, actor=actor)


class ListDirTool:
    name = "list_dir"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：列出目录内容。受 file.allow_read/ask_read/deny "
                "权限策略控制。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径。"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "") or ".").strip() or "."
        actor = actor_from_context(context)
        decision = decide_file_read(path)
        payload = {"path": path}
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="list_dir",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="list_dir",
                payload=payload,
                permission=decision,
            )
        return await list_dir(path=path, actor=actor)


class SearchFilesTool:
    name = "search_files"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：按 glob 和可选文本内容搜索文件。适合定位代码、"
                "日志和配置。受 file.allow_read/ask_read/deny 权限策略控制。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "搜索根目录。"},
                    "pattern": {
                        "type": ["string", "null"],
                        "description": "glob，例如 *.py、**/*.yaml；为空默认 **/*。",
                    },
                    "contains": {
                        "type": ["string", "null"],
                        "description": "可选文本过滤，只有包含该文本的文件返回。",
                    },
                    "max_results": {
                        "type": ["integer", "null"],
                        "description": "最多返回结果数，默认 50。",
                    },
                },
                "required": ["root", "pattern", "contains", "max_results"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        root = str(kwargs.get("root", "") or ".").strip() or "."
        pattern = str(kwargs.get("pattern", "") or "**/*").strip() or "**/*"
        contains = str(kwargs.get("contains", "") or "")
        try:
            max_results = max(1, min(int(kwargs.get("max_results") or 50), 200))
        except (TypeError, ValueError):
            max_results = 50
        actor = actor_from_context(context)
        decision = decide_file_read(root)
        payload = {
            "root": root,
            "pattern": pattern,
            "contains": contains,
            "max_results": max_results,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="search_files",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="search_files",
                payload=payload,
                permission=decision,
            )
        return await search_files(
            root=root,
            pattern=pattern,
            contains=contains,
            max_results=max_results,
            actor=actor,
        )


class WriteFileTool:
    name = "write_file"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：写入完整文本文件，覆盖原内容。受 "
                "file.allow_write/ask_write/deny 权限策略控制。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径。"},
                    "content": {"type": "string", "description": "完整文件内容。"},
                    "create_dirs": {
                        "type": ["boolean", "null"],
                        "description": "父目录不存在时是否创建，默认 false。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么要写入该文件。",
                    },
                },
                "required": ["path", "content", "create_dirs", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "") or "").strip()
        content = str(kwargs.get("content", "") or "")
        create_dirs = bool(kwargs.get("create_dirs") or False)
        reason = str(kwargs.get("reason", "") or "")
        actor = actor_from_context(context)
        decision = decide_file_write(path)
        payload = {
            "path": path,
            "content": content,
            "create_dirs": create_dirs,
            "reason": reason,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="write_file",
                payload=_safe_file_payload(payload),
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="write_file",
                payload=payload,
                permission=decision,
            )
        return await write_file(
            path=path,
            content=content,
            create_dirs=create_dirs,
            actor=actor,
            reason=reason,
        )


class AppendFileTool:
    name = "append_file"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：向文本文件末尾追加内容。受 "
                "file.allow_write/ask_write/deny 权限策略控制。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径。"},
                    "content": {"type": "string", "description": "要追加的文本。"},
                    "create_dirs": {
                        "type": ["boolean", "null"],
                        "description": "父目录不存在时是否创建，默认 false。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么要追加该文件。",
                    },
                },
                "required": ["path", "content", "create_dirs", "reason"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "") or "").strip()
        content = str(kwargs.get("content", "") or "")
        create_dirs = bool(kwargs.get("create_dirs") or False)
        reason = str(kwargs.get("reason", "") or "")
        actor = actor_from_context(context)
        decision = decide_file_write(path)
        payload = {
            "path": path,
            "content": content,
            "create_dirs": create_dirs,
            "reason": reason,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="append_file",
                payload=_safe_file_payload(payload),
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="append_file",
                payload=payload,
                permission=decision,
            )
        return await append_file(
            path=path,
            content=content,
            create_dirs=create_dirs,
            actor=actor,
            reason=reason,
        )


class ReplaceInFileTool:
    name = "replace_in_file"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：在文本文件中精确替换 old_text 为 new_text。"
                "适合小范围代码修改；受 file.allow_write/ask_write/deny 权限策略控制。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径。"},
                    "old_text": {"type": "string", "description": "要替换的原文。"},
                    "new_text": {"type": "string", "description": "替换后的文本。"},
                    "expected_replacements": {
                        "type": ["integer", "null"],
                        "description": "期望替换次数；为空则允许任意正次数。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么要修改该文件。",
                    },
                },
                "required": [
                    "path",
                    "old_text",
                    "new_text",
                    "expected_replacements",
                    "reason",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "") or "").strip()
        old_text = str(kwargs.get("old_text", "") or "")
        new_text = str(kwargs.get("new_text", "") or "")
        expected = kwargs.get("expected_replacements")
        expected_replacements = None
        if expected not in (None, ""):
            try:
                expected_replacements = max(1, int(expected))
            except (TypeError, ValueError):
                expected_replacements = None
        reason = str(kwargs.get("reason", "") or "")
        actor = actor_from_context(context)
        decision = decide_file_write(path)
        payload = {
            "path": path,
            "old_text": old_text,
            "new_text": new_text,
            "expected_replacements": expected_replacements,
            "reason": reason,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="replace_in_file",
                payload=_safe_file_payload(payload),
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="replace_in_file",
                payload=payload,
                permission=decision,
            )
        return await replace_in_file(
            path=path,
            old_text=old_text,
            new_text=new_text,
            expected_replacements=expected_replacements,
            actor=actor,
            reason=reason,
        )


async def read_file(
    *,
    path: str,
    max_chars: int,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        content = compact_text(text, max_chars=max_chars)
        result = {
            "path": path,
            "approval_id": approval_id,
            "content": content,
            "truncated": len(text) > len(content),
        }
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="read_file",
            payload={"path": path, "max_chars": max_chars, "approval_id": approval_id},
            result={"ok": True, "truncated": result["truncated"]},
        )
        return tool_result(True, "file_read", **result)
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="read_file",
            payload={"path": path, "max_chars": max_chars, "approval_id": approval_id},
            status="read_error",
            error=str(exc),
        )


async def list_dir(
    *,
    path: str,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    try:
        entries = []
        for item in sorted(Path(path).iterdir(), key=lambda p: p.name.lower())[:120]:
            entries.append(
                {
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="list_dir",
            payload={"path": path, "approval_id": approval_id},
            result={"ok": True, "entries": len(entries)},
        )
        return tool_result(
            True,
            "dir_listed",
            path=path,
            approval_id=approval_id,
            entries=entries,
        )
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="list_dir",
            payload={"path": path, "approval_id": approval_id},
            status="list_error",
            error=str(exc),
        )


async def search_files(
    *,
    root: str,
    pattern: str,
    contains: str,
    max_results: int,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    try:
        root_path = Path(root)
        results: list[dict[str, Any]] = []
        for item in root_path.rglob(pattern or "**/*"):
            if len(results) >= max_results:
                break
            if not item.is_file():
                continue
            match_info: dict[str, Any] = {
                "path": item.as_posix(),
                "size": item.stat().st_size,
            }
            if contains:
                try:
                    text = item.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                index = text.find(contains)
                if index < 0:
                    continue
                start = max(0, index - 80)
                end = min(len(text), index + len(contains) + 120)
                match_info["snippet"] = text[start:end].replace("\n", " ")[:260]
            results.append(match_info)
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="search_files",
            payload={
                "root": root,
                "pattern": pattern,
                "contains": contains,
                "max_results": max_results,
                "approval_id": approval_id,
            },
            result={"ok": True, "matches": len(results)},
        )
        return tool_result(
            True,
            "files_searched",
            root=root,
            pattern=pattern,
            contains=contains,
            approval_id=approval_id,
            results=results,
            truncated=len(results) >= max_results,
        )
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="search_files",
            payload={
                "root": root,
                "pattern": pattern,
                "contains": contains,
                "max_results": max_results,
                "approval_id": approval_id,
            },
            status="search_error",
            error=str(exc),
        )


async def write_file(
    *,
    path: str,
    content: str,
    create_dirs: bool,
    actor: dict[str, str],
    approval_id: str | None = None,
    reason: str = "",
) -> ToolResult:
    return apply_changes_transaction(
        actor=actor,
        action="write_file",
        reason=reason or "write full file content",
        approval_id=approval_id,
        changes=[
            FileChange(
                path=path,
                mode="write",
                content=content,
                create_dirs=create_dirs,
            )
        ],
    )


async def append_file(
    *,
    path: str,
    content: str,
    create_dirs: bool,
    actor: dict[str, str],
    approval_id: str | None = None,
    reason: str = "",
) -> ToolResult:
    return apply_changes_transaction(
        actor=actor,
        action="append_file",
        reason=reason or "append file content",
        approval_id=approval_id,
        changes=[
            FileChange(
                path=path,
                mode="append",
                content=content,
                create_dirs=create_dirs,
            )
        ],
    )


async def replace_in_file(
    *,
    path: str,
    old_text: str,
    new_text: str,
    expected_replacements: int | None,
    actor: dict[str, str],
    approval_id: str | None = None,
    reason: str = "",
) -> ToolResult:
    if not old_text:
        return tool_result(False, "replace_empty_old_text", path=path)
    return apply_changes_transaction(
        actor=actor,
        action="replace_in_file",
        reason=reason or "replace text in file",
        approval_id=approval_id,
        changes=[
            FileChange(
                path=path,
                mode="replace",
                old_text=old_text,
                new_text=new_text,
                expected_replacements=expected_replacements,
            )
        ],
    )


def _safe_file_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    for key in ("content", "old_text", "new_text"):
        if key in safe:
            safe[key] = compact_text(str(safe[key]), max_chars=240)
    return safe


register_superuser_tool(ReadFileTool)
register_superuser_tool(ListDirTool)
register_superuser_tool(SearchFilesTool)
register_superuser_tool(WriteFileTool)
register_superuser_tool(AppendFileTool)
register_superuser_tool(ReplaceInFileTool)

__all__ = [
    "AppendFileTool",
    "ListDirTool",
    "ReadFileTool",
    "ReplaceInFileTool",
    "SearchFilesTool",
    "WriteFileTool",
    "append_file",
    "list_dir",
    "read_file",
    "replace_in_file",
    "search_files",
    "write_file",
]
