"""No-hit recovery for tool-intent messages.

This module never invents tools. It only rewrites a fuzzy user request into a
capability query, then asks the existing command index to search installed tools.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from zhenxun.services import generate_structured, logger

from .command_index import CommandCandidate, build_recovered_command_candidates
from .config import get_config_value, get_model_name
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .route_text import (
    collect_placeholders,
    contains_any,
    has_negative_route_intent,
    is_usage_question,
    normalize_message_text,
    strip_invoke_prefix,
)
from .speech_act import classify_speech_act

_ACTION_HINTS = (
    "帮我",
    "请",
    "麻烦",
    "给我",
    "替我",
    "把",
    "查",
    "查一下",
    "查询",
    "搜",
    "搜一下",
    "搜索",
    "找",
    "找一下",
    "看",
    "看一下",
    "识别",
    "解析",
    "生成",
    "制作",
    "做",
    "来",
    "来一",
    "发",
    "发送",
    "添加",
    "新增",
    "删除",
    "翻译",
    "播放",
    "打开",
    "关闭",
    "统计",
)
_DISCUSSION_HINTS = (
    "聊聊",
    "讨论",
    "觉得",
    "这个词",
    "这个概念",
    "作为",
    "优点",
    "缺点",
    "为什么",
    "什么意思",
)
_TOOL_INTENT_MIN_LEN = 4
_RECOVERY_MIN_SCORE = 96.0


class ToolIntentProbe(BaseModel):
    has_tool_intent: bool = Field(description="用户是否像是在请求调用某个工具")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")


class CapabilityRewriteResult(BaseModel):
    action: Literal["recover", "chat", "no_available_tool"] = Field(
        default="chat",
        description="recover=用 capability_query 检索工具；chat=普通聊天",
    )
    capability_query: str = Field(
        default="",
        description="面向工具检索的能力描述，不包含不存在的具体命令",
    )
    task_verbs: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")


def probe_tool_intent(message_text: str, *, has_reply: bool = False) -> ToolIntentProbe:
    normalized = normalize_message_text(strip_invoke_prefix(message_text or ""))
    if not normalized or len(normalized) < _TOOL_INTENT_MIN_LEN:
        return ToolIntentProbe(has_tool_intent=False, reason="too_short")
    if has_negative_route_intent(normalized):
        return ToolIntentProbe(has_tool_intent=False, reason="negative_route_intent")
    speech_act = classify_speech_act(normalized, has_reply=has_reply)
    if speech_act == "ask_usage":
        return ToolIntentProbe(has_tool_intent=True, confidence=0.72, reason="usage")
    if speech_act in {"discuss_command", "casual_chat"} and contains_any(
        normalized, _DISCUSSION_HINTS
    ):
        return ToolIntentProbe(has_tool_intent=False, reason=f"speech_act:{speech_act}")
    placeholders = collect_placeholders(normalized)
    has_image = any(token.startswith("[image") for token in placeholders)
    has_at = any(
        token.startswith("[@") or token.startswith("@") for token in placeholders
    )
    has_context = has_image or has_at or has_reply
    if has_context:
        if contains_any(normalized, _ACTION_HINTS):
            return ToolIntentProbe(
                has_tool_intent=True,
                confidence=0.78,
                reason="context_action",
            )
    if contains_any(normalized, _ACTION_HINTS):
        return ToolIntentProbe(has_tool_intent=True, confidence=0.68, reason="action")
    if is_usage_question(normalized):
        return ToolIntentProbe(has_tool_intent=True, confidence=0.62, reason="usage")
    return ToolIntentProbe(has_tool_intent=False, confidence=0.2, reason="no_signal")


def _tool_catalog(
    tools: list[CommandToolSnapshot],
    *,
    limit: int = 80,
) -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for tool in tools[: max(limit, 1)]:
        catalog.append(
            {
                "command_id": tool.command_id,
                "plugin": tool.plugin_name,
                "head": tool.head,
                "capability": tool.capability_text or tool.description,
                "verbs": tool.task_verbs,
                "inputs": tool.input_requirements,
            }
        )
    return catalog


async def rewrite_capability_query(
    message_text: str,
    *,
    tools: list[CommandToolSnapshot],
    timeout: float | None = None,
) -> CapabilityRewriteResult | None:
    payload = {
        "message": normalize_message_text(message_text),
        "installed_tools": _tool_catalog(tools),
        "task": (
            "如果用户像是在请求调用工具，把消息改写成适合检索已安装工具的能力查询。"
            "不要发明未列出的工具或命令；如果只是聊天/讨论，返回 chat。"
        ),
    }
    try:
        return await generate_structured(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            CapabilityRewriteResult,
            model=get_model_name(),
            instruction=(
                "你是工具检索 query rewriter。只根据 installed_tools 的能力空间改写，"
                "输出能力查询，不输出具体不存在的命令。"
            ),
            timeout=timeout,
        )
    except Exception as exc:
        logger.debug(f"ChatInter no-hit query rewrite failed: {exc}")
        return None


async def recover_no_hit_candidates(
    knowledge_base: PluginKnowledgeBase,
    message_text: str,
    *,
    tools: list[CommandToolSnapshot],
    session_id: str | None = None,
    has_reply: bool = False,
    limit: int = 24,
) -> tuple[CapabilityRewriteResult | None, list[CommandCandidate]]:
    probe = probe_tool_intent(message_text, has_reply=has_reply)
    if not probe.has_tool_intent:
        return None, []
    if not tools:
        return (
            CapabilityRewriteResult(
                action="no_available_tool",
                confidence=probe.confidence,
                reason="empty_tool_pool",
            ),
            [],
        )
    timeout_raw = get_config_value("INTENT_TIMEOUT", 20)
    try:
        timeout = float(timeout_raw) if timeout_raw else None
    except (TypeError, ValueError):
        timeout = None
    rewrite = await rewrite_capability_query(
        message_text,
        tools=tools,
        timeout=timeout,
    )
    if rewrite is None or rewrite.action != "recover":
        return rewrite, []
    capability_query = normalize_message_text(rewrite.capability_query)
    if not capability_query:
        return rewrite, []
    candidates = build_recovered_command_candidates(
        knowledge_base,
        original_query=message_text,
        capability_query=capability_query,
        limit=limit,
        session_id=session_id,
        tools=tools,
    )
    if not candidates:
        return rewrite, []
    top = candidates[0]
    if top.score < _RECOVERY_MIN_SCORE and not top.exact_protected:
        return rewrite, []
    return rewrite, candidates


__all__ = [
    "CapabilityRewriteResult",
    "ToolIntentProbe",
    "probe_tool_intent",
    "recover_no_hit_candidates",
    "rewrite_capability_query",
]
