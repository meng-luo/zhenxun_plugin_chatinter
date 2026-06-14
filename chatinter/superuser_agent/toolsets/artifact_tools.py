"""Artifact inspection tools for superuser Agent turns."""

from __future__ import annotations

from typing import Any

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from ...artifact_store import get_artifact_store
from ..registry import register_superuser_tool
from .common import actor_from_context, tool_result


class ArtifactReadTool:
    name = "artifact_read"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：读取 ArtifactStore 中被压缩的长日志、diff、"
                "工具输出或文本片段。只接受 artifact_id。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "工具 observation 返回的 artifact_id。",
                    },
                    "max_chars": {
                        "type": ["integer", "null"],
                        "description": "最多返回字符数，默认 4000，最大 12000。",
                    },
                    "offset": {
                        "type": ["integer", "null"],
                        "description": "从第几个字符开始读取，默认 0。",
                    },
                },
                "required": ["artifact_id", "max_chars", "offset"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        artifact_id = str(kwargs.get("artifact_id", "") or "").strip()
        max_chars = _coerce_max_chars(kwargs.get("max_chars"))
        offset = _coerce_offset(kwargs.get("offset"))
        if not artifact_id:
            return tool_result(False, "artifact_id_required")
        result = get_artifact_store().read_text(
            artifact_id,
            max_chars=max_chars,
            offset=offset,
        )
        if result is None:
            return tool_result(False, "artifact_not_found", artifact_id=artifact_id)
        ref, content = result
        return tool_result(
            True,
            "artifact_read",
            artifact=ref.to_dict(),
            artifact_content=content,
            offset=offset,
            next_offset=offset + len(content)
            if offset + len(content) < ref.size
            else None,
            truncated=offset + len(content) < ref.size,
        )


class ArtifactListTool:
    name = "artifact_list"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "超级用户私聊专用：列出 ArtifactStore 最近保存的长日志、diff、"
                "后台任务输出和工具结果引用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多返回数量，默认 20，最大 100。",
                    },
                    "artifact_type": {
                        "type": ["string", "null"],
                        "description": (
                            "可选类型过滤，例如 " "log/plugin_output/text/image/file。"
                        ),
                    },
                    "source_contains": {
                        "type": ["string", "null"],
                        "description": (
                            "可选 source 关键词过滤，例如 " "background_task 或 patch。"
                        ),
                    },
                },
                "required": ["limit", "artifact_type", "source_contains"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor_from_context(context)
        refs = get_artifact_store().list_refs(
            limit=_coerce_limit(kwargs.get("limit")),
            artifact_type=str(kwargs.get("artifact_type", "") or ""),
            source_contains=str(kwargs.get("source_contains", "") or ""),
        )
        return tool_result(
            True,
            "artifact_list",
            artifacts=[ref.to_dict() for ref in refs],
            count=len(refs),
        )


def _coerce_max_chars(value: Any) -> int:
    try:
        return max(1, min(int(value or 4000), 12000))
    except (TypeError, ValueError):
        return 4000


def _coerce_offset(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_limit(value: Any) -> int:
    try:
        return max(1, min(int(value or 20), 100))
    except (TypeError, ValueError):
        return 20


register_superuser_tool(
    ArtifactReadTool, category="artifact", risk="low", read_only=True
)
register_superuser_tool(
    ArtifactListTool, category="artifact", risk="low", read_only=True
)

__all__ = ["ArtifactListTool", "ArtifactReadTool"]
