"""Multi-view query expansion for ChatInter tool recall.

The router still executes only installed command schemas. This layer only
rewrites one user utterance into several retrieval views so fuzzy tool intent can
reach the reranker without broad, noisy hard-coded fallbacks.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from zhenxun.services.llm import generate_structured
from zhenxun.services.log import logger

from .config import get_config_value, get_model_name
from .models.pydantic_models import CommandToolSnapshot
from .route_text import has_chat_context_hint, normalize_message_text
from .tool_intent_probe import probe_tool_intent

_CATALOG_LIMIT = 80
_LOCAL_ACTION_VERBS = (
    "使用",
    "发送",
    "制作",
    "生成",
    "文本",
    "领取",
    "退回",
    "播放",
    "解析",
    "识别",
    "搜索",
    "添加",
    "解释",
    "随机",
    "查询",
)
_TOOL_TEXT_FIELDS = (
    "plugin_name",
    "head",
    "description",
    "capability_text",
    "family",
)


class ToolQueryDescriptor(BaseModel):
    action: Literal["expand", "chat", "no_available_tool"] = Field(
        default="expand",
        description="expand=生成工具检索视角；chat=普通对话",
    )
    capability_query: str = Field(
        default="",
        description="一句话描述用户想要的工具能力，不要发明未安装工具",
    )
    task_verbs: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


def _tool_catalog(tools: list[CommandToolSnapshot]) -> list[dict[str, object]]:
    return [
        {
            "command_id": tool.command_id,
            "plugin": tool.plugin_name,
            "head": tool.head,
            "family": tool.family,
            "capability": tool.capability_text or tool.description,
            "verbs": tool.task_verbs,
            "inputs": tool.input_requirements,
        }
        for tool in tools[:_CATALOG_LIMIT]
    ]


def _tool_text(tool: CommandToolSnapshot) -> str:
    parts = [str(getattr(tool, name, "") or "") for name in _TOOL_TEXT_FIELDS]
    parts.extend(tool.aliases)
    parts.extend(tool.retrieval_phrases)
    parts.extend(tool.task_verbs)
    parts.extend(tool.input_requirements)
    parts.extend(slot.name for slot in tool.slots)
    parts.extend(slot.description for slot in tool.slots)
    for slot in tool.slots:
        parts.extend(slot.aliases)
    return normalize_message_text(" ".join(parts)).casefold()


def _matched_tool_terms(
    normalized_message: str,
    tools: list[CommandToolSnapshot],
) -> tuple[list[str], list[str], list[str]]:
    query = normalize_message_text(normalized_message).casefold()
    if not query:
        return [], [], []
    task_verbs: list[str] = []
    objects: list[str] = []
    inputs: list[str] = []
    for tool in tools[:_CATALOG_LIMIT]:
        haystack = _tool_text(tool)
        if not haystack:
            continue
        command_terms = [tool.head, *tool.aliases, *tool.retrieval_phrases]
        matched = any(
            term and normalize_message_text(term).casefold() in query
            for term in command_terms
        )
        matched = matched or any(
            verb in query and verb in haystack for verb in _LOCAL_ACTION_VERBS
        )
        if not matched:
            continue
        for value in tool.task_verbs:
            text = normalize_message_text(value)
            if text and text not in task_verbs:
                task_verbs.append(text)
        for value in [tool.plugin_name, tool.head, tool.family, *tool.aliases]:
            text = normalize_message_text(value)
            if text and text not in objects:
                objects.append(text)
        for value in tool.input_requirements:
            text = normalize_message_text(value)
            if text and text not in inputs:
                inputs.append(text)
    return task_verbs, objects, inputs


def _local_descriptor(
    message_text: str,
    *,
    tools: list[CommandToolSnapshot],
    has_reply: bool,
) -> ToolQueryDescriptor | None:
    normalized = normalize_message_text(message_text)
    if not normalized:
        return None
    probe = probe_tool_intent(normalized, has_reply=has_reply)
    task_verbs, objects, inputs = _matched_tool_terms(normalized, tools)
    matched_tool = bool(task_verbs or objects or inputs)
    if not probe.has_tool_intent and not matched_tool:
        return None
    if matched_tool and not probe.has_tool_intent and has_chat_context_hint(normalized):
        return None
    capability_parts = [normalized, *task_verbs, *objects, *inputs]
    confidence = max(probe.confidence, 0.72 if matched_tool else 0.62)
    reason = f"local:{probe.reason};tool_hint={int(matched_tool)}"
    if matched_tool and not probe.has_tool_intent:
        reason = f"{reason};descriptor_only"
    return ToolQueryDescriptor(
        action="expand",
        capability_query=normalize_message_text(" ".join(capability_parts)),
        task_verbs=task_verbs,
        objects=objects,
        inputs=inputs,
        confidence=confidence,
        reason=reason,
    )


async def build_tool_query_descriptor(
    message_text: str,
    *,
    tools: list[CommandToolSnapshot],
    has_reply: bool = False,
) -> ToolQueryDescriptor | None:
    local = _local_descriptor(message_text, tools=tools, has_reply=has_reply)
    if not tools:
        return local
    timeout_raw = get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout = float(timeout_raw) if timeout_raw else None
    except (TypeError, ValueError):
        timeout = None
    payload = {
        "message": normalize_message_text(message_text),
        "installed_tools": _tool_catalog(tools),
        "task": (
            "将用户请求改写成工具检索描述。只描述已安装工具能力空间，"
            "不要输出具体不存在的命令；普通闲聊返回 chat。"
        ),
    }
    try:
        llm_result = await generate_structured(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ToolQueryDescriptor,
            model=get_model_name(),
            instruction=(
                "你是工具检索 query planner。输出 capability_query、动作词、对象、"
                "输入需求，用于召回候选工具。"
            ),
            timeout=timeout,
        )
    except Exception as exc:
        logger.debug(f"ChatInter tool query descriptor failed: {exc}")
        return local
    if llm_result.action != "expand":
        return local if local and local.confidence >= 0.7 else llm_result
    if not normalize_message_text(llm_result.capability_query):
        return local
    if local is None:
        return llm_result
    # LLM view first, local deterministic hints second; route_engine will merge.
    merged_objects = list(dict.fromkeys([*llm_result.objects, *local.objects]))
    merged_verbs = list(dict.fromkeys([*llm_result.task_verbs, *local.task_verbs]))
    return ToolQueryDescriptor(
        action="expand",
        capability_query=normalize_message_text(
            f"{llm_result.capability_query} {local.capability_query}"
        ),
        task_verbs=merged_verbs,
        objects=merged_objects,
        inputs=llm_result.inputs or local.inputs,
        negative_constraints=llm_result.negative_constraints,
        confidence=max(llm_result.confidence, local.confidence),
        reason=normalize_message_text(f"{llm_result.reason};{local.reason}"),
    )


def expand_tool_queries(
    message_text: str,
    descriptor: ToolQueryDescriptor | None,
) -> list[str]:
    queries: list[str] = []

    def add(value: object) -> None:
        text = normalize_message_text(str(value or ""))
        if text and text not in queries:
            queries.append(text)

    add(message_text)
    if descriptor is None or descriptor.action != "expand":
        return queries
    add(descriptor.capability_query)
    add(
        " ".join(
            [
                normalize_message_text(message_text),
                descriptor.capability_query,
                " ".join(descriptor.task_verbs),
                " ".join(descriptor.objects),
            ]
        )
    )
    add(" ".join([*descriptor.task_verbs, *descriptor.objects, *descriptor.inputs]))
    for value in [*descriptor.task_verbs, *descriptor.objects]:
        add(value)
    return queries


__all__ = [
    "ToolQueryDescriptor",
    "build_tool_query_descriptor",
    "expand_tool_queries",
]
