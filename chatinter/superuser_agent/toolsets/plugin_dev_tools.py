"""Plugin development automation tools for the superuser private Agent scenario."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ..audit_log import record_audit_event
from ..patch_operations import FileChange, apply_changes_transaction
from ..permission_policy import decide_plugin_dev
from ..registry import register_superuser_tool
from .common import (
    actor_from_context,
    approval_required_result,
    audited_error_result,
    compact_text,
    permission_denied_result,
    project_root,
    tool_result,
)

_PLUGIN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_PLUGIN_ROOT = Path("zhenxun/plugins")


class PluginDevInspectTool:
    name = "plugin_dev_inspect"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：检查真寻插件目录结构、__init__.py 摘要和命令/元数据线索。"
                "用于改插件前先了解现状。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plugin_name": {
                        "type": ["string", "null"],
                        "description": "插件目录名；为空则列出插件根目录。",
                    },
                    "plugin_root": {
                        "type": ["string", "null"],
                        "description": "插件根目录，默认 zhenxun/plugins。",
                    },
                    "max_files": {
                        "type": ["integer", "null"],
                        "description": "最多返回文件数，默认 80。",
                    },
                },
                "required": ["plugin_name", "plugin_root", "max_files"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        plugin_name = str(kwargs.get("plugin_name", "") or "").strip()
        plugin_root = str(kwargs.get("plugin_root", "") or "").strip() or None
        max_files = _coerce_int(kwargs.get("max_files"), default=80, lower=1, upper=300)
        decision = decide_plugin_dev("plugin_dev_inspect " + plugin_name)
        payload = {
            "plugin_name": plugin_name,
            "plugin_root": plugin_root,
            "max_files": max_files,
        }
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="plugin_dev_inspect",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="plugin_dev_inspect",
                payload=payload,
                permission=decision,
            )
        return await inspect_plugin(
            plugin_name=plugin_name,
            plugin_root=plugin_root,
            max_files=max_files,
            actor=actor,
        )


class PluginDevScaffoldTool:
    name = "plugin_dev_scaffold"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：创建一个真寻插件包骨架，包括 __init__.py、"
                "PluginMetadata、PluginExtraData 和一个基础 on_alconna 命令。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plugin_name": {
                        "type": "string",
                        "description": "Python 包名，例如 my_plugin，只允许字母数字下划线。",
                    },
                    "display_name": {"type": "string", "description": "插件展示名。"},
                    "command": {"type": "string", "description": "主命令头，例如 今日运势。"},
                    "description": {"type": "string", "description": "插件描述。"},
                    "author": {
                        "type": ["string", "null"],
                        "description": "作者名，默认 ChatInter Agent。",
                    },
                    "menu_type": {
                        "type": ["string", "null"],
                        "description": "菜单分类，默认 功能。",
                    },
                    "plugin_root": {
                        "type": ["string", "null"],
                        "description": "插件根目录，默认 zhenxun/plugins。",
                    },
                    "overwrite": {
                        "type": ["boolean", "null"],
                        "description": "目录已存在时是否覆盖 __init__.py，默认 false。",
                    },
                },
                "required": [
                    "plugin_name",
                    "display_name",
                    "command",
                    "description",
                    "author",
                    "menu_type",
                    "plugin_root",
                    "overwrite",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        payload = {
            "plugin_name": str(kwargs.get("plugin_name", "") or "").strip(),
            "display_name": str(kwargs.get("display_name", "") or "").strip(),
            "command": str(kwargs.get("command", "") or "").strip(),
            "description": str(kwargs.get("description", "") or "").strip(),
            "author": str(kwargs.get("author", "") or "ChatInter Agent").strip()
            or "ChatInter Agent",
            "menu_type": str(kwargs.get("menu_type", "") or "功能").strip() or "功能",
            "plugin_root": str(kwargs.get("plugin_root", "") or "").strip() or None,
            "overwrite": bool(kwargs.get("overwrite") or False),
        }
        decision = decide_plugin_dev("plugin_dev_scaffold " + payload["plugin_name"])
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="plugin_dev_scaffold",
                payload=payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="plugin_dev_scaffold",
                payload=payload,
                permission=decision,
            )
        return await scaffold_plugin(actor=actor, **payload)


class PluginDevWriteFileTool:
    name = "plugin_dev_write_file"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：在指定插件目录内写入或覆盖文件。路径必须是插件内相对路径，"
                "适合自动制作/修改插件文件。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plugin_name": {
                        "type": "string",
                        "description": "插件目录名，例如 my_plugin。",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": "插件目录内相对路径，例如 data_source.py。",
                    },
                    "content": {"type": "string", "description": "完整文件内容。"},
                    "plugin_root": {
                        "type": ["string", "null"],
                        "description": "插件根目录，默认 zhenxun/plugins。",
                    },
                    "create_dirs": {
                        "type": ["boolean", "null"],
                        "description": "父目录不存在时是否创建，默认 true。",
                    },
                    "overwrite": {
                        "type": ["boolean", "null"],
                        "description": "文件已存在时是否覆盖，默认 true。",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "为什么需要写入该插件文件。",
                    },
                },
                "required": [
                    "plugin_name",
                    "relative_path",
                    "content",
                    "plugin_root",
                    "create_dirs",
                    "overwrite",
                    "reason",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        payload = {
            "plugin_name": str(kwargs.get("plugin_name", "") or "").strip(),
            "relative_path": str(kwargs.get("relative_path", "") or "").strip(),
            "content": str(kwargs.get("content", "") or ""),
            "plugin_root": str(kwargs.get("plugin_root", "") or "").strip() or None,
            "create_dirs": True if kwargs.get("create_dirs") is None else bool(kwargs.get("create_dirs")),
            "overwrite": True if kwargs.get("overwrite") is None else bool(kwargs.get("overwrite")),
            "reason": str(kwargs.get("reason", "") or ""),
        }
        decision = decide_plugin_dev(
            f"plugin_dev_write_file {payload['plugin_name']} {payload['relative_path']}"
        )
        safe_payload = {**payload, "content": compact_text(payload["content"], max_chars=240)}
        if decision.decision == "deny":
            return permission_denied_result(
                actor=actor,
                action="plugin_dev_write_file",
                payload=safe_payload,
                permission=decision,
            )
        if decision.decision == "ask":
            return approval_required_result(
                actor=actor,
                action="plugin_dev_write_file",
                payload=payload,
                permission=decision,
            )
        return await write_plugin_file(actor=actor, **payload)


async def inspect_plugin(
    *,
    plugin_name: str,
    plugin_root: str | None,
    max_files: int,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    try:
        root = _plugin_root(plugin_root)
        target = _plugin_dir(plugin_name, plugin_root) if plugin_name else root
        if not target.exists():
            return tool_result(False, "plugin_path_not_found", path=str(target))
        files: list[dict[str, Any]] = []
        candidates = [target] if target.is_file() else sorted(target.rglob("*"), key=lambda p: p.as_posix())
        for item in candidates:
            if len(files) >= max_files:
                break
            if "__pycache__" in item.parts:
                continue
            files.append(
                {
                    "path": item.relative_to(root).as_posix() if _is_relative_to(item, root) else item.as_posix(),
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
        init_path = target / "__init__.py" if target.is_dir() else target
        init_summary = ""
        commands: list[str] = []
        metadata_names: list[str] = []
        if init_path.exists() and init_path.is_file():
            text = init_path.read_text(encoding="utf-8", errors="replace")
            init_summary = compact_text(text, max_chars=2400)
            commands = sorted(set(re.findall(r"Alconna\((?:\n\s*)?[\"']([^\"']+)[\"']", text)))[:30]
            metadata_names = sorted(set(re.findall(r"name\s*=\s*[\"']([^\"']+)[\"']", text)))[:10]
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="plugin_dev_inspect",
            payload={"plugin_name": plugin_name, "plugin_root": plugin_root, "approval_id": approval_id},
            result={"ok": True, "files": len(files)},
        )
        return tool_result(
            True,
            "plugin_inspected",
            plugin_name=plugin_name,
            plugin_root=str(root),
            path=str(target),
            approval_id=approval_id,
            files=files,
            init_summary=init_summary,
            commands=commands,
            metadata_names=metadata_names,
            truncated=len(files) >= max_files,
        )
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="plugin_dev_inspect",
            payload={"plugin_name": plugin_name, "plugin_root": plugin_root, "approval_id": approval_id},
            status="plugin_inspect_error",
            error=str(exc),
        )


async def scaffold_plugin(
    *,
    plugin_name: str,
    display_name: str,
    command: str,
    description: str,
    author: str,
    menu_type: str,
    plugin_root: str | None,
    overwrite: bool,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    try:
        _validate_plugin_name(plugin_name)
        if not display_name:
            return tool_result(False, "plugin_display_name_required")
        if not command:
            return tool_result(False, "plugin_command_required")
        plugin_dir = _plugin_dir(plugin_name, plugin_root)
        init_path = plugin_dir / "__init__.py"
        if init_path.exists() and not overwrite:
            return tool_result(False, "plugin_already_exists", path=str(init_path))
        content = _render_plugin_init(
            display_name=display_name,
            command=command,
            description=description,
            author=author,
            menu_type=menu_type,
        )
        result = apply_changes_transaction(
            actor=actor,
            action="plugin_dev_scaffold",
            reason=f"scaffold plugin {plugin_name}",
            approval_id=approval_id,
            changes=[
                FileChange(
                    path=str(init_path),
                    mode="write",
                    content=content,
                    create_dirs=True,
                )
            ],
        )
        if not isinstance(result.output, dict) or not result.output.get("ok"):
            return result
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="plugin_dev_scaffold",
            payload={"plugin_name": plugin_name, "plugin_root": plugin_root, "approval_id": approval_id},
            result={"ok": True, "path": str(init_path)},
        )
        return tool_result(
            True,
            "plugin_scaffolded",
            plugin_name=plugin_name,
            path=str(init_path),
            approval_id=approval_id,
            patch_operation=result.output.get("operation") if isinstance(result.output, dict) else None,
            files=[str(init_path)],
            next_steps=[
                "根据需求补充业务逻辑和参数 schema。",
                "需要依赖时使用 uv_command 处理依赖。",
                "需要验证时使用 python_module 或 uv_command 运行定向检查。",
            ],
        )
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="plugin_dev_scaffold",
            payload={"plugin_name": plugin_name, "plugin_root": plugin_root, "approval_id": approval_id},
            status="plugin_scaffold_error",
            error=str(exc),
        )


async def write_plugin_file(
    *,
    plugin_name: str,
    relative_path: str,
    content: str,
    plugin_root: str | None,
    create_dirs: bool,
    overwrite: bool,
    reason: str,
    actor: dict[str, str],
    approval_id: str | None = None,
) -> ToolResult:
    try:
        _validate_plugin_name(plugin_name)
        if not relative_path:
            return tool_result(False, "plugin_relative_path_required")
        plugin_dir = _plugin_dir(plugin_name, plugin_root)
        target = _safe_child(plugin_dir, relative_path)
        if target.exists() and not overwrite:
            return tool_result(False, "plugin_file_exists", path=str(target))
        result = apply_changes_transaction(
            actor=actor,
            action="plugin_dev_write_file",
            reason=reason or f"write plugin file {plugin_name}/{relative_path}",
            approval_id=approval_id,
            changes=[
                FileChange(
                    path=str(target),
                    mode="write",
                    content=content,
                    create_dirs=create_dirs,
                )
            ],
        )
        if not isinstance(result.output, dict) or not result.output.get("ok"):
            return result
        record_audit_event(
            event="operation_executed",
            user_id=actor["user_id"],
            session_key=actor["session_key"],
            action="plugin_dev_write_file",
            payload={
                "plugin_name": plugin_name,
                "relative_path": relative_path,
                "reason": reason,
                "approval_id": approval_id,
            },
            result={"ok": True, "bytes_written": len(content.encode("utf-8"))},
        )
        return tool_result(
            True,
            "plugin_file_written",
            plugin_name=plugin_name,
            path=str(target),
            relative_path=relative_path,
            approval_id=approval_id,
            patch_operation=result.output.get("operation") if isinstance(result.output, dict) else None,
            bytes_written=len(content.encode("utf-8")),
        )
    except Exception as exc:
        return audited_error_result(
            actor=actor,
            action="plugin_dev_write_file",
            payload={
                "plugin_name": plugin_name,
                "relative_path": relative_path,
                "reason": reason,
                "approval_id": approval_id,
            },
            status="plugin_write_error",
            error=str(exc),
        )


def _plugin_root(plugin_root: str | None) -> Path:
    root = Path(plugin_root) if plugin_root else project_root() / _DEFAULT_PLUGIN_ROOT
    return root.resolve()


def _plugin_dir(plugin_name: str, plugin_root: str | None) -> Path:
    _validate_plugin_name(plugin_name)
    return (_plugin_root(plugin_root) / plugin_name).resolve()


def _validate_plugin_name(plugin_name: str) -> None:
    if not _PLUGIN_NAME_RE.fullmatch(plugin_name or ""):
        raise ValueError("plugin_name must be a valid Python package name")


def _safe_child(root: Path, relative_path: str) -> Path:
    target = (root / relative_path).resolve()
    root = root.resolve()
    if target == root or root not in target.parents:
        raise ValueError("relative_path must stay inside plugin directory")
    return target


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _render_plugin_init(
    *,
    display_name: str,
    command: str,
    description: str,
    author: str,
    menu_type: str,
) -> str:
    command_head = command.split()[0]
    return f'''from nonebot.plugin import PluginMetadata\nfrom nonebot_plugin_alconna import Alconna, Arparma, on_alconna\n\nfrom zhenxun.configs.utils import Command, PluginExtraData\nfrom zhenxun.services.log import logger\nfrom zhenxun.utils.message import MessageUtils\n\n__plugin_meta__ = PluginMetadata(\n    name={display_name!r},\n    description={description!r},\n    usage="""\n    {command}\n    """.strip(),\n    extra=PluginExtraData(\n        author={author!r},\n        version="0.1.0",\n        menu_type={menu_type!r},\n        commands=[Command(command={command!r})],\n    ).to_dict(),\n)\n\n_matcher = on_alconna(Alconna({command_head!r}), priority=5, block=True)\n\n\n@_matcher.handle()\nasync def _(arparma: Arparma):\n    await MessageUtils.build_message("插件已创建，请补充具体业务逻辑。").send(reply_to=True)\n    logger.info({display_name!r}, arparma.header_result)\n'''


def _coerce_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        return max(lower, min(int(value or default), upper))
    except (TypeError, ValueError):
        return default


register_superuser_tool(PluginDevInspectTool)
register_superuser_tool(PluginDevScaffoldTool)
register_superuser_tool(PluginDevWriteFileTool)

__all__ = [
    "PluginDevInspectTool",
    "PluginDevScaffoldTool",
    "PluginDevWriteFileTool",
    "inspect_plugin",
    "scaffold_plugin",
    "write_plugin_file",
]
