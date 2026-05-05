from __future__ import annotations

from dataclasses import dataclass

_MULTI_STEP_HINTS = (
    "然后",
    "再",
    "接着",
    "并且",
    "同时",
    "分别",
    "一步步",
)
_PLUGIN_RESEARCH_HINTS = (
    "插件",
    "命令",
    "功能",
    "怎么用",
    "用法",
    "参数",
)
_TOOL_OPERATOR_HINTS = (
    "shell",
    "终端",
    "命令行",
    "执行命令",
    "计算",
    "表达式",
)
_RETRIEVAL_HINTS = (
    "查",
    "搜",
    "搜索",
    "检索",
    "定位",
    "找出",
)


@dataclass(frozen=True)
class SubagentHandoffDecision:
    enabled: bool
    role: str
    reason: str
    tool_names: tuple[str, ...] = ()


def decide_subagent_handoff(
    *,
    message_text: str,
    tool_names: tuple[str, ...],
    has_images: bool = False,
) -> SubagentHandoffDecision:
    normalized = str(message_text or "").lower()
    if not normalized or not tool_names:
        return SubagentHandoffDecision(False, "none", "no_tools")

    has_multi_step = any(hint in normalized for hint in _MULTI_STEP_HINTS)
    if not has_multi_step and not has_images:
        return SubagentHandoffDecision(False, "none", "simple_task")

    if any(hint in normalized for hint in _PLUGIN_RESEARCH_HINTS):
        filtered = _filter_tool_names(tool_names, ("lookup", "plugin", "help"))
        return SubagentHandoffDecision(
            True,
            "plugin_researcher",
            "plugin_research_task",
            filtered or tool_names[:4],
        )

    if any(hint in normalized for hint in _TOOL_OPERATOR_HINTS):
        filtered = _filter_tool_names(tool_names, ("shell", "eval", "cmd"))
        return SubagentHandoffDecision(
            True,
            "tool_operator",
            "tool_operator_task",
            filtered or tool_names[:4],
        )

    if any(hint in normalized for hint in _RETRIEVAL_HINTS):
        filtered = _filter_tool_names(tool_names, ("lookup", "search", "query"))
        return SubagentHandoffDecision(
            True,
            "retrieval_worker",
            "retrieval_task",
            filtered or tool_names[:4],
        )

    return SubagentHandoffDecision(
        True, "workflow_planner", "multi_step_task", tool_names[:4]
    )


def build_subagent_instruction(
    *,
    base_instruction: str,
    decision: SubagentHandoffDecision,
) -> str:
    role_note = {
        "plugin_researcher": (
            "你是插件研究子代理，只负责确认最合适的插件和命令，不要闲聊。"
        ),
        "tool_operator": (
            "你是工具执行子代理，只负责安全地规划和调用工具，不要扩展闲聊。"
        ),
        "retrieval_worker": "你是检索子代理，只负责快速检索并给出最小结论。",
        "workflow_planner": (
            "你是工作流规划子代理，只负责把复杂任务拆成最少步骤并执行。"
        ),
    }.get(decision.role, "你是子代理，只负责当前交付，不要偏离任务。")
    return (
        f"{base_instruction}\n\n{role_note}\n"
        "若能直接完成就直接完成，不能完成时说明限制。"
    )


def _filter_tool_names(
    tool_names: tuple[str, ...],
    keywords: tuple[str, ...],
) -> tuple[str, ...]:
    selected = tuple(
        name
        for name in tool_names
        if any(keyword in name.lower() for keyword in keywords)
    )
    return selected


__all__ = [
    "SubagentHandoffDecision",
    "build_subagent_instruction",
    "decide_subagent_handoff",
]
