"""Filesystem tools for the superuser private Agent scenario."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
from typing import Any

from ...llm_compat import ToolDefinition, ToolResult
from ..audit_log import record_audit_event
from ..patch_operations import FileChange, apply_changes_transaction
from ..patch_parser import parse_v4a_patch, patch_file_changes
from ..permission_policy import decide_file_read, decide_file_write
from ..process_control import (
    attach_process_tree,
    release_process_tree,
    subprocess_group_kwargs,
    terminate_process_tree,
)
from .common import (
    actor_from_context,
    approval_required_result,
    audited_error_result,
    coerce_max_chars,
    compact_text,
    decode,
    permission_denied_result,
    tool_result,
)


class ReadFileTool:
    name = "read_file"
    read_only = True

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "按行分页读取文本文件；content 中每行均以 `行号| ` 前缀返回，"
                "可将该行号用于定位和构造精确替换文本。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径。"},
                    "offset": {
                        "type": ["integer", "null"],
                        "description": "起始行号，从 1 开始，默认 1。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多返回行数，默认 200。",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "") or "").strip()
        read_args = _normalize_read_arguments(kwargs)
        actor = actor_from_context(context)
        path = str(Path(path or ".").resolve())
        decision = decide_file_read(path)
        payload = {"path": path, **read_args}
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
        return await read_file(
            path=path,
            actor=actor,
            **read_args,
        )


class ListDirTool:
    name = "list_dir"
    read_only = True

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="列出目录内容。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径。"},
                },
                "required": [],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "") or ".").strip() or "."
        actor = actor_from_context(context)
        path = str(Path(path).resolve())
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
    read_only = True

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "使用 ripgrep 搜索文件内容或按 glob 查找文件。"
                "query 为空时只返回文件路径。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": ["string", "null"],
                        "description": "内容搜索的正则表达式；为空时按 glob 查找文件。",
                    },
                    "path": {
                        "type": "string",
                        "description": "搜索目录或文件，默认当前工作目录。",
                    },
                    "glob": {
                        "type": ["string", "null"],
                        "description": "文件过滤 glob，例如 *.py、**/*.yaml。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多返回结果数，默认 50。",
                    },
                    "offset": {
                        "type": ["integer", "null"],
                        "description": "跳过前 N 个结果，默认 0。",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        search_args = _normalize_search_arguments(kwargs)
        actor = actor_from_context(context)
        path = str(Path(search_args["path"] or ".").resolve())
        search_args["path"] = path
        decision = decide_file_read(path)
        payload = {
            **search_args,
            "path": path,
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
            actor=actor,
            **search_args,
        )


class WriteFileTool:
    name = "write_file"
    read_only = False

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="写入完整文本文件并覆盖原内容。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径。"},
                    "content": {"type": "string", "description": "完整文件内容。"},
                    "create_dirs": {
                        "type": ["boolean", "null"],
                        "description": "父目录不存在时是否创建，默认 false。",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "") or "").strip()
        content = str(kwargs.get("content", "") or "")
        create_dirs = bool(kwargs.get("create_dirs") or False)
        reason = str(kwargs.get("reason", "") or "")
        actor = actor_from_context(context)
        path = str(Path(path or ".").resolve())
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


class ReplaceInFileTool:
    name = "replace_in_file"
    read_only = False

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "在一个或多个文本文件中精确替换 old_text 为 new_text。"
                "单文件使用 path/old_text/new_text，多文件使用 changes。"
                "每项默认必须且只能命中一处；"
                "多处替换必须显式设置 replace_all=true；"
                "优先使用完整上下文保持修改精确。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径。"},
                    "old_text": {"type": "string", "description": "要替换的原文。"},
                    "new_text": {"type": "string", "description": "替换后的文本。"},
                    "expected_replacements": {
                        "type": ["integer", "null"],
                        "description": "replace_all=true 时可断言实际替换次数。",
                    },
                    "replace_all": {
                        "type": ["boolean", "null"],
                        "description": "是否允许替换全部命中处，默认 false。",
                    },
                    "changes": {
                        "type": ["array", "null"],
                        "description": "需要作为一批执行的 1-20 个精确替换。",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "文件路径。"},
                                "old_text": {
                                    "type": "string",
                                    "description": "要替换的原文。",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "替换后的文本。",
                                },
                                "expected_replacements": {
                                    "type": ["integer", "null"],
                                    "description": "replace_all=true 时断言替换次数。",
                                },
                                "replace_all": {
                                    "type": ["boolean", "null"],
                                    "description": "是否替换全部命中处，默认 false。",
                                },
                            },
                            "required": ["path", "old_text", "new_text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [],
                "oneOf": [
                    {"required": ["path", "old_text", "new_text"]},
                    {"required": ["changes"]},
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        changes, error = build_replace_changes(kwargs)
        if error:
            return tool_result(False, "invalid_replace_changes", error=error)
        reason = str(kwargs.get("reason", "") or "")
        actor = actor_from_context(context)
        payload = _replace_payload(
            changes,
            batch=kwargs.get("changes") is not None,
            reason=reason,
        )
        pending_permission = None
        for change in changes:
            decision = decide_file_write(change.path)
            if decision.decision == "deny":
                return permission_denied_result(
                    actor=actor,
                    action="replace_in_file",
                    payload=_safe_file_payload(payload),
                    permission=decision,
                )
            if decision.decision == "ask" and pending_permission is None:
                pending_permission = decision
        if pending_permission is not None:
            return approval_required_result(
                actor=actor,
                action="replace_in_file",
                payload=payload,
                permission=pending_permission,
            )
        return await replace_files(
            changes=changes,
            actor=actor,
            reason=reason,
        )


class ApplyPatchTool:
    name = "apply_patch"
    read_only = False

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "原子应用一个或多个文件补丁，支持新增、更新、删除和移动。"
                "格式为 *** Begin Patch，随后使用 *** Add File、"
                "*** Update File、*** Delete File 或 *** Move to，"
                "最后以 *** End Patch 结束。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": "V4A 补丁文本。"},
                    "cwd": {
                        "type": ["string", "null"],
                        "description": "相对路径的基准目录，默认当前项目目录。",
                    },
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        patch = str(kwargs.get("patch", "") or "")
        cwd = str(kwargs.get("cwd", "") or "").strip() or None
        actor = actor_from_context(context)
        payload = {"patch": patch, "cwd": cwd}
        paths, error = patch_paths(patch, cwd=cwd)
        if error:
            return tool_result(False, "invalid_patch", error=error)
        pending_permission = None
        for path in paths:
            decision = decide_file_write(path)
            if decision.decision == "deny":
                return permission_denied_result(
                    actor=actor,
                    action="apply_patch",
                    payload={"paths": paths},
                    permission=decision,
                )
            if decision.decision == "ask" and pending_permission is None:
                pending_permission = decision
        if pending_permission is not None:
            return approval_required_result(
                actor=actor,
                action="apply_patch",
                payload=payload,
                permission=pending_permission,
            )
        return await apply_patch_text(patch=patch, cwd=cwd, actor=actor)


_FILE_PAGE_CHAR_LIMIT = 8000
_READ_DEFAULT_LINES = 200
_READ_MAX_LINES = 1000


def _normalize_read_arguments(values: dict[str, Any]) -> dict[str, int]:
    try:
        offset = max(int(values.get("offset") or 1), 1)
    except (TypeError, ValueError):
        offset = 1
    try:
        limit = max(
            1,
            min(int(values.get("limit") or _READ_DEFAULT_LINES), _READ_MAX_LINES),
        )
    except (TypeError, ValueError):
        limit = _READ_DEFAULT_LINES
    max_chars = _FILE_PAGE_CHAR_LIMIT
    if values.get("max_chars") not in (None, ""):
        max_chars = min(coerce_max_chars(values.get("max_chars")), max_chars)
    return {"offset": offset, "limit": limit, "max_chars": max_chars}


def _read_file_page(
    path: str, read_args: dict[str, int]
) -> tuple[str, int | None, int | None, str]:
    lines: list[str] = []
    chars = 0
    next_offset: int | None = None
    overflow_content = ""
    last_line: int | None = None
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number < read_args["offset"]:
                continue
            if len(lines) >= read_args["limit"]:
                next_offset = line_number
                break
            formatted = f"{line_number}| {raw_line.rstrip(chr(13) + chr(10))}"
            added_chars = len(formatted) + (1 if lines else 0)
            if chars + added_chars > read_args["max_chars"]:
                if lines:
                    next_offset = line_number
                    break
                visible_chars = read_args["max_chars"]
                lines.append(formatted[:visible_chars])
                overflow_content = formatted[visible_chars:]
                last_line = line_number
                next_offset = line_number + 1
                break
            lines.append(formatted)
            chars += added_chars
            last_line = line_number
    return "\n".join(lines), last_line, next_offset, overflow_content


async def read_file(
    *,
    path: str,
    actor: dict[str, str],
    offset: int = 1,
    limit: int = _READ_DEFAULT_LINES,
    max_chars: int | None = None,
    approval_id: str | None = None,
) -> ToolResult:
    read_args = _normalize_read_arguments(
        {"offset": offset, "limit": limit, "max_chars": max_chars}
    )
    try:
        content, last_line, next_offset, overflow_content = await asyncio.to_thread(
            _read_file_page, path, read_args
        )
        result = {
            "path": path,
            "approval_id": approval_id,
            "content": content,
            "offset": read_args["offset"],
            "limit": read_args["limit"],
            "end_line": last_line,
            "next_offset": next_offset,
            "truncated": next_offset is not None or bool(overflow_content),
        }
        if overflow_content:
            result["overflow_content"] = overflow_content
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="read_file",
            payload={"path": path, **read_args, "approval_id": approval_id},
            result={"ok": True, "truncated": result["truncated"]},
        )
        return tool_result(True, "file_read", **result)
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="read_file",
            payload={"path": path, **read_args, "approval_id": approval_id},
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


_SEARCH_TIMEOUT_SECONDS = 30.0
_SEARCH_EXCLUDE_GLOBS = (
    "!.git/**",
    "!.venv/**",
    "!data/chatinter_agent/**",
    "!data/chatinter_artifacts/**",
)


def _normalize_search_arguments(values: dict[str, Any]) -> dict[str, Any]:
    legacy_query = str(values.get("contains", "") or "")
    query_supplied = "query" in values and values.get("query") is not None
    query = str(values.get("query", "") or "") if query_supplied else legacy_query
    path = str(values.get("path") or values.get("root") or ".").strip() or "."
    glob = str(values.get("glob") or values.get("pattern") or "").strip()
    limit_value = values.get("limit")
    if limit_value in (None, ""):
        limit_value = values.get("max_results")
    try:
        limit = max(1, min(int(limit_value or 50), 200))
    except (TypeError, ValueError):
        limit = 50
    try:
        offset = max(int(values.get("offset") or 0), 0)
    except (TypeError, ValueError):
        offset = 0
    literal = bool(values.get("literal")) or (not query_supplied and bool(legacy_query))
    return {
        "query": query,
        "path": path,
        "glob": glob,
        "limit": limit,
        "offset": offset,
        "literal": literal,
    }


async def search_files(
    *,
    actor: dict[str, str],
    query: str | None = None,
    path: str | None = None,
    glob: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    literal: bool = False,
    root: str | None = None,
    pattern: str | None = None,
    contains: str | None = None,
    max_results: int | None = None,
    approval_id: str | None = None,
) -> ToolResult:
    search_args = _normalize_search_arguments(
        {
            "query": query,
            "path": path,
            "glob": glob,
            "limit": limit,
            "offset": offset,
            "literal": literal,
            "root": root,
            "pattern": pattern,
            "contains": contains,
            "max_results": max_results,
        }
    )
    payload = {
        **search_args,
        "approval_id": approval_id,
    }
    executable = shutil.which("rg")
    if not executable:
        return audited_error_result(
            actor=actor,
            action="search_files",
            payload=payload,
            status="search_backend_unavailable",
            error="ripgrep executable 'rg' was not found",
        )

    search_path = Path(search_args["path"])
    if not search_path.exists():
        return audited_error_result(
            actor=actor,
            action="search_files",
            payload=payload,
            status="search_path_not_found",
            error=f"path does not exist: {search_path}",
        )

    cwd = search_path if search_path.is_dir() else search_path.parent
    target = "." if search_path.is_dir() else search_path.name
    args = _rg_search_args(search_args, target=target)
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **subprocess_group_kwargs(),
        )
        attach_process_tree(process)
        lines, truncated = await asyncio.wait_for(
            _read_rg_lines(
                process,
                count=search_args["offset"] + search_args["limit"] + 1,
            ),
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
        if truncated and process.returncode is None:
            await terminate_process_tree(process)
        else:
            await process.wait()
            release_process_tree(process)
        stderr = decode(await process.stderr.read()) if process.stderr else ""
        if not truncated and process.returncode not in {0, 1}:
            return audited_error_result(
                actor=actor,
                action="search_files",
                payload=payload,
                status="search_error",
                error=stderr.strip() or f"rg exited with {process.returncode}",
            )
        page = lines[
            search_args["offset"] : search_args["offset"] + search_args["limit"]
        ]
        results = _parse_rg_results(
            page,
            cwd=cwd,
            content_mode=bool(search_args["query"]),
        )
        results, page_truncated = _bounded_search_results(results)
        truncated = truncated or page_truncated
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="search_files",
            payload=payload,
            result={"ok": True, "matches": len(results)},
        )
        return tool_result(
            True,
            "files_searched",
            query=search_args["query"],
            path=search_args["path"],
            glob=search_args["glob"],
            limit=search_args["limit"],
            offset=search_args["offset"],
            approval_id=approval_id,
            results=results,
            truncated=truncated,
            next_offset=(search_args["offset"] + len(results) if truncated else None),
        )
    except asyncio.TimeoutError:
        if process is not None:
            await terminate_process_tree(process)
        return audited_error_result(
            actor=actor,
            action="search_files",
            payload=payload,
            status="search_timeout",
        )
    except asyncio.CancelledError:
        if process is not None:
            await terminate_process_tree(process)
        raise
    except Exception as exc:
        if process is not None and process.returncode is None:
            await terminate_process_tree(process)
        return audited_error_result(
            actor=actor,
            action="search_files",
            payload=payload,
            status="search_error",
            error=str(exc),
        )


def _rg_search_args(search_args: dict[str, Any], *, target: str) -> list[str]:
    query = search_args["query"]
    if query:
        args = [
            "--line-number",
            "--no-heading",
            "--no-messages",
            "--color=never",
            "--max-columns=500",
        ]
        if search_args["literal"]:
            args.append("--fixed-strings")
    else:
        args = ["--files", "--no-messages"]
    if search_args["glob"]:
        args.extend(["--glob", search_args["glob"]])
    for excluded in _SEARCH_EXCLUDE_GLOBS:
        args.extend(["--glob", excluded])
    if query:
        args.extend(["-e", query])
    args.append(target)
    return args


async def _read_rg_lines(
    process: asyncio.subprocess.Process,
    *,
    count: int,
) -> tuple[list[str], bool]:
    if process.stdout is None:
        return [], False
    lines: list[str] = []
    while len(lines) < count:
        raw = await process.stdout.readline()
        if not raw:
            return lines, False
        lines.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
    return lines, True


def _parse_rg_results(
    lines: list[str],
    *,
    cwd: Path,
    content_mode: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw in lines:
        if content_mode:
            parts = raw.split(":", 2)
            if len(parts) != 3:
                continue
            raw_path, raw_line, text = parts
            try:
                line_number = int(raw_line)
            except ValueError:
                continue
            results.append(
                {
                    "path": _relative_search_path(cwd, raw_path),
                    "line": line_number,
                    "text": text,
                }
            )
        else:
            results.append({"path": _relative_search_path(cwd, raw)})
    return results


def _bounded_search_results(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    bounded: list[dict[str, Any]] = []
    used_chars = 0
    for item in results:
        item_chars = (
            len(str(item.get("path", ""))) + len(str(item.get("text", ""))) + 32
        )
        if bounded and used_chars + item_chars > _FILE_PAGE_CHAR_LIMIT:
            return bounded, True
        bounded.append(item)
        used_chars += item_chars
    return bounded, False


def _relative_search_path(cwd: Path, value: str) -> str:
    resolved = (cwd / value).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(cwd.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()


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


async def replace_in_file(
    *,
    path: str,
    old_text: str,
    new_text: str,
    expected_replacements: int | None,
    actor: dict[str, str],
    replace_all: bool = False,
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
                replace_all=replace_all,
            )
        ],
    )


async def replace_files(
    *,
    changes: list[FileChange],
    actor: dict[str, str],
    approval_id: str | None = None,
    reason: str = "",
) -> ToolResult:
    return apply_changes_transaction(
        actor=actor,
        action="replace_in_file",
        reason=reason or "replace text in files",
        approval_id=approval_id,
        changes=changes,
    )


async def apply_patch_text(
    *,
    patch: str,
    cwd: str | None,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    changes, error = patch_file_changes(patch, cwd=cwd)
    if error:
        return tool_result(False, "invalid_patch", error=error)
    return apply_changes_transaction(
        actor=actor,
        action="apply_patch",
        approval_id=approval_id,
        changes=changes,
    )


def patch_paths(patch: str, *, cwd: str | None) -> tuple[list[str], str]:
    operations, error = parse_v4a_patch(patch)
    if error:
        return [], error
    root = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    paths: list[str] = []
    for operation in operations:
        values = [operation.path]
        if operation.destination:
            values.append(operation.destination)
        for value in values:
            path = Path(value)
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            paths.append(str(resolved))
    if len(paths) > 20:
        return [], "patch exceeds 20 file changes"
    return paths, ""


def build_replace_changes(values: dict[str, Any]) -> tuple[list[FileChange], str]:
    allowed = {
        "path",
        "old_text",
        "new_text",
        "expected_replacements",
        "replace_all",
    }
    raw_changes = values.get("changes")
    if raw_changes is None:
        raw_changes = [{key: values[key] for key in allowed if key in values}]
    elif not isinstance(raw_changes, list):
        return [], "changes must be an array"
    if not raw_changes or len(raw_changes) > 20:
        return [], "changes must contain between 1 and 20 items"

    changes: list[FileChange] = []
    for index, item in enumerate(raw_changes, start=1):
        if not isinstance(item, dict):
            return [], f"changes[{index}] must be an object"
        unexpected = sorted(str(key) for key in item if key not in allowed)
        if unexpected:
            return [], (
                f"changes[{index}] contains unknown fields: {', '.join(unexpected)}"
            )
        path = str(item.get("path", "") or "").strip()
        old_text = str(item.get("old_text", "") or "")
        if not path:
            return [], f"changes[{index}].path is required"
        if not old_text:
            return [], f"changes[{index}].old_text is required"
        if "new_text" not in item:
            return [], f"changes[{index}].new_text is required"
        expected = item.get("expected_replacements")
        if expected in (None, ""):
            expected_replacements = None
        elif (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and expected > 0
        ):
            expected_replacements = expected
        else:
            return [], (
                f"changes[{index}].expected_replacements must be a positive integer"
            )
        changes.append(
            FileChange(
                path=str(Path(path).resolve()),
                mode="replace",
                old_text=old_text,
                new_text=str(item.get("new_text", "") or ""),
                expected_replacements=expected_replacements,
                replace_all=bool(item.get("replace_all") or False),
            )
        )
    return changes, ""


def _replace_payload(
    changes: list[FileChange],
    *,
    batch: bool,
    reason: str,
) -> dict[str, Any]:
    items = [
        {
            "path": change.path,
            "old_text": change.old_text,
            "new_text": change.new_text,
            "expected_replacements": change.expected_replacements,
            "replace_all": change.replace_all,
        }
        for change in changes
    ]
    if batch:
        return {"changes": items, "reason": reason}
    return {**items[0], "reason": reason}


def _safe_file_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    for key in ("content", "old_text", "new_text"):
        if key in safe:
            safe[key] = compact_text(str(safe[key]), max_chars=240)
    if isinstance(safe.get("changes"), list):
        safe["changes"] = [
            _safe_file_payload(item) if isinstance(item, dict) else item
            for item in safe["changes"]
        ]
    return safe


__all__ = [
    "ApplyPatchTool",
    "ListDirTool",
    "ReadFileTool",
    "ReplaceInFileTool",
    "SearchFilesTool",
    "WriteFileTool",
    "apply_patch_text",
    "build_replace_changes",
    "list_dir",
    "patch_paths",
    "read_file",
    "replace_files",
    "replace_in_file",
    "search_files",
    "write_file",
]
