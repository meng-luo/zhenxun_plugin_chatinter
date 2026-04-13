from __future__ import annotations

from dataclasses import dataclass

from .intent_classifier import IntentClassification
from .route_text import normalize_message_text

_AGENT_BLOCK_CHAT_SUBKINDS = {
    "general_chat",
    "recap",
    "identity_query",
    "memory_confirm",
    "explain_context",
}

_AGENT_DIRECT_TOOL_HINTS = (
    "插件",
    "命令",
    "工具",
    "mcp",
    "接口",
    "api",
    "shell",
    "终端",
    "命令行",
    "目录",
    "文件",
    "仓库",
    "代码库",
    "表达式",
    "计算",
    "算一下",
    "读取",
)

_AGENT_MULTI_STEP_HINTS = (
    "然后",
    "再",
    "接着",
    "并且",
    "同时",
    "依次",
    "分别",
    "一步步",
)


@dataclass(frozen=True)
class AgentGateDecision:
    enabled: bool
    reason: str


def decide_agent_gate(
    *,
    config_enabled: bool,
    intent: IntentClassification,
    message_text: str,
    has_images: bool = False,
    has_mcp_endpoints: bool = False,
) -> AgentGateDecision:
    if not config_enabled:
        return AgentGateDecision(False, "config_disabled")

    if intent.kind == "chat":
        if intent.chat_subkind in _AGENT_BLOCK_CHAT_SUBKINDS:
            return AgentGateDecision(False, f"chat_subkind:{intent.chat_subkind}")
        return AgentGateDecision(False, "chat_fast_path")

    if intent.explicit_command or intent.command_head or intent.plugin_module:
        return AgentGateDecision(False, "plugin_routing_preferred")

    if intent.kind in {"help", "execute_need_arg"}:
        return AgentGateDecision(False, f"intent:{intent.kind}")

    normalized = normalize_message_text(message_text)
    if not normalized:
        return AgentGateDecision(False, "empty_message")

    has_direct_tool_hint = any(hint in normalized for hint in _AGENT_DIRECT_TOOL_HINTS)
    has_multi_step_hint = any(hint in normalized for hint in _AGENT_MULTI_STEP_HINTS)

    if has_direct_tool_hint and (has_multi_step_hint or has_mcp_endpoints):
        return AgentGateDecision(True, "direct_tool_task")

    if has_direct_tool_hint and has_images:
        return AgentGateDecision(True, "multimodal_tool_task")

    if intent.kind == "ambiguous" and has_direct_tool_hint:
        return AgentGateDecision(True, "ambiguous_tool_task")

    return AgentGateDecision(False, "no_agent_signal")


__all__ = ["AgentGateDecision", "decide_agent_gate"]
