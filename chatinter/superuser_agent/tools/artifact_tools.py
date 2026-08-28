"""Artifact inspection tools for superuser Agent turns."""

from __future__ import annotations

from typing import Any

from ...artifact_store import get_artifact_store
from ...llm_compat import ToolDefinition, ToolResult
from .common import actor_from_context, tool_result


class ArtifactReadTool:
    name = "artifact_read"
    read_only = True

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "读取 ArtifactStore 中被压缩的长日志、diff、"
                "工具输出或文本片段；可分页读取或按 query 定位目标。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "工具结果返回的 artifact_id。",
                    },
                    "max_chars": {
                        "type": ["integer", "null"],
                        "description": "最多返回字符数，默认 4000，最大 12000。",
                    },
                    "offset": {
                        "type": ["integer", "null"],
                        "description": "从第几个字符开始读取，默认 0。",
                    },
                    "query": {
                        "type": ["string", "null"],
                        "description": "可选目标文本；提供后返回匹配位置和上下文片段。",
                    },
                    "max_matches": {
                        "type": ["integer", "null"],
                        "description": "查询最多返回的匹配数，默认 6，最大 20。",
                    },
                    "context_chars": {
                        "type": ["integer", "null"],
                        "description": "每个匹配前后保留字符数，默认 240。",
                    },
                    "scan_chars": {
                        "type": ["integer", "null"],
                        "description": "本次最多扫描字符数，默认 120000。",
                    },
                },
                "required": ["artifact_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        actor = actor_from_context(context)
        artifact_id = str(kwargs.get("artifact_id", "") or "").strip()
        max_chars = _coerce_max_chars(kwargs.get("max_chars"))
        offset = _coerce_offset(kwargs.get("offset"))
        query = str(kwargs.get("query", "") or "").strip()
        if not artifact_id:
            return tool_result(False, "artifact_id_required")
        if len(query) > 512:
            return tool_result(False, "artifact_query_too_long", max_chars=512)
        extra = getattr(context, "extra", None)
        artifact_refs = (
            extra.get("artifact_refs", ()) if isinstance(extra, dict) else ()
        )
        if not isinstance(artifact_refs, list | tuple | set):
            artifact_refs = ()
        if not actor["run_id"] or artifact_id not in artifact_refs:
            return tool_result(False, "artifact_not_found", artifact_id=artifact_id)
        store = get_artifact_store()
        if query:
            search_result = store.search_text(
                artifact_id,
                query,
                max_matches=_coerce_int(kwargs.get("max_matches"), 6, 1, 20),
                context_chars=_coerce_int(
                    kwargs.get("context_chars"),
                    240,
                    20,
                    2_000,
                ),
                scan_chars=_coerce_int(
                    kwargs.get("scan_chars"),
                    120_000,
                    1_000,
                    250_000,
                ),
                offset=offset,
            )
            if search_result is None:
                return tool_result(False, "artifact_not_found", artifact_id=artifact_id)
            ref, search = search_result
            return tool_result(
                True,
                "artifact_search",
                artifact=ref.to_dict(),
                **search,
            )
        result = store.read_text(
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


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(default if value is None else value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


__all__ = ["ArtifactReadTool"]
