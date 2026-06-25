"""Complexity routing for the superuser private Agent.

Lightweight requests should not pay the planner-executor-verifier tax.  This
gate is intentionally local and metadata-driven: it only decides whether the
runtime should enable the durable planner/verifier layer, not which tool to use.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from .route_text import normalize_message_text

AgentComplexityMode = Literal[
    "chat", "readonly_fast", "single_tool_fast", "multi_tool_fast", "complex_pev"
]

_MULTI_STEP_HINTS = (
    "然后",
    "最后",
    "顺便",
    "接着",
    "再帮",
    "再看",
    "同时",
    "并且",
    "以及",
    "全部",
    "完整",
    "自动",
    "长期",
    "持续",
    "后台",
    "部署",
    "修复",
    "重构",
    "实现",
    "开发",
    "制作插件",
    "写插件",
    "改代码",
    "跑测试",
    "验收",
    "回滚",
)
_ACTION_HINTS = (
    "查",
    "看",
    "读",
    "列",
    "搜索",
    "执行",
    "运行",
    "写",
    "改",
    "删",
    "创建",
    "生成",
    "安装",
    "启动",
    "停止",
    "重启",
    "提交",
    "回滚",
    "测试",
)
_HIGH_RISK_HINTS = (
    "删除",
    "清空",
    "覆盖",
    "迁移",
    "部署",
    "kill",
    "rm ",
    "del ",
    "drop",
    "reset",
    "checkout",
)
_CONNECTOR_PATTERN = re.compile(r"(?:然后|最后|顺便|接着|以及|同时|并且|再帮|再看)")
_READONLY_FAST_HINTS = (
    "查状态",
    "看状态",
    "查看状态",
    "读日志",
    "看日志",
    "查看日志",
    "读文件",
    "看文件",
    "查看文件",
    "列目录",
    "看目录",
)
_READONLY_FAST_COMMANDS = (
    "git status",
    "git diff",
    "git log",
    "docker ps",
    "docker info",
    "df",
    "free",
    "uptime",
    "whoami",
    "hostname",
)
_READONLY_FAST_BLOCKERS = (
    "写",
    "改",
    "删",
    "删除",
    "创建",
    "生成",
    "安装",
    "启动",
    "停止",
    "重启",
    "提交",
    "回滚",
    "修复",
    "实现",
    "开发",
    "制作",
    "发布",
    "部署",
    "为什么",
    "原因",
    "分析",
    "定位",
    "解决",
    "怎么办",
    "怎么改",
    "优化",
    "审计",
    "测试",
    "验收",
)
_PLUGIN_SCAFFOLD_HINTS = ("生成插件", "创建插件", "制作插件", "写插件")
_COMPLEX_PLUGIN_HINTS = (
    "复杂",
    "完整",
    "多文件",
    "数据库",
    "权限",
    "测试",
    "验收",
    "发布",
    "同步",
    "修复",
    "改造",
    "重构",
    "性能",
    "压测",
    "迁移",
    "后台",
    "定时",
    "webui",
    "api",
)


@dataclass(frozen=True)
class AgentComplexityDecision:
    mode: AgentComplexityMode
    reason: str
    score: float = 0.0
    expected_tool_budget: int = 0
    enable_planner: bool = False
    enable_verifier: bool = False

    @property
    def is_complex(self) -> bool:
        return self.mode == "complex_pev"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "score": round(float(self.score or 0.0), 3),
            "expected_tool_budget": self.expected_tool_budget,
            "enable_planner": self.enable_planner,
            "enable_verifier": self.enable_verifier,
        }


def route_agent_complexity(
    *,
    message_text: str,
    tool_map: dict[str, Any],
    enable_agent_tools: bool,
    resumed_run: bool = False,
    local_task_count: int = 0,
) -> AgentComplexityDecision:
    if not enable_agent_tools:
        return AgentComplexityDecision(
            mode="chat",
            reason="agent_tools_disabled",
            expected_tool_budget=0,
        )
    if resumed_run:
        return AgentComplexityDecision(
            mode="complex_pev",
            reason="resumed_run_requires_stateful_verification",
            score=4.0,
            expected_tool_budget=4,
            enable_planner=True,
            enable_verifier=True,
        )

    text = normalize_message_text(message_text)
    if not text:
        return AgentComplexityDecision(mode="chat", reason="empty_message")

    if _looks_like_readonly_fast_request(text):
        return AgentComplexityDecision(
            mode="readonly_fast",
            reason="readonly_fast_request",
            score=0.5,
            expected_tool_budget=1,
            enable_planner=False,
            enable_verifier=False,
        )

    if local_task_count >= 2 and not any(
        hint in text.casefold() for hint in _HIGH_RISK_HINTS
    ):
        return AgentComplexityDecision(
            mode="multi_tool_fast",
            reason=f"local_task_ledger={local_task_count}",
            score=1.8,
            expected_tool_budget=max(int(local_task_count) + 1, 3),
            enable_planner=False,
            enable_verifier=False,
        )

    if _looks_like_simple_plugin_scaffold(text):
        return AgentComplexityDecision(
            mode="single_tool_fast",
            reason="simple_plugin_scaffold_request",
            score=1.0,
            expected_tool_budget=1,
            enable_planner=False,
            enable_verifier=False,
        )

    score = 0.0
    reasons: list[str] = []
    lowered = text.casefold()
    if len(text) >= 120:
        score += 1.0
        reasons.append("long_request")
    connector_count = len(_CONNECTOR_PATTERN.findall(text))
    if connector_count:
        score += min(connector_count, 3) * 0.8
        reasons.append(f"multi_clause={connector_count}")
    action_count = sum(1 for hint in _ACTION_HINTS if hint in lowered)
    if action_count >= 2:
        score += 0.8
        reasons.append(f"multiple_action_hints={action_count}")
    if any(hint in lowered for hint in _MULTI_STEP_HINTS):
        score += 0.8
        reasons.append("multi_step_hint")
    if any(hint in lowered for hint in _HIGH_RISK_HINTS):
        score += 1.0
        reasons.append("risk_hint")

    available_tools = len(tool_map)
    if available_tools > 20 and action_count >= 1:
        score += 0.3
        reasons.append("broad_agent_toolset")

    if score >= 2.4:
        return AgentComplexityDecision(
            mode="complex_pev",
            reason=";".join(reasons) or "complex_request",
            score=score,
            expected_tool_budget=5,
            enable_planner=True,
            enable_verifier=True,
        )
    if action_count > 0:
        return AgentComplexityDecision(
            mode="single_tool_fast",
            reason=";".join(reasons) or "single_action_request",
            score=score,
            expected_tool_budget=1,
            enable_planner=False,
            enable_verifier=False,
        )
    return AgentComplexityDecision(
        mode="chat",
        reason="no_action_hint",
        score=score,
        expected_tool_budget=0,
        enable_planner=False,
        enable_verifier=False,
    )


def _looks_like_readonly_fast_request(text: str) -> bool:
    lowered = text.casefold()
    if _CONNECTOR_PATTERN.search(text):
        return False
    if any(hint in lowered for hint in _HIGH_RISK_HINTS):
        return False
    if any(hint in lowered for hint in _READONLY_FAST_BLOCKERS):
        return False
    if any(command in lowered for command in _READONLY_FAST_COMMANDS):
        return True
    return any(hint in lowered for hint in _READONLY_FAST_HINTS)


def _looks_like_simple_plugin_scaffold(text: str) -> bool:
    lowered = text.casefold()
    if _CONNECTOR_PATTERN.search(text):
        return False
    if not any(hint in lowered for hint in _PLUGIN_SCAFFOLD_HINTS):
        return False
    return not any(hint in lowered for hint in _COMPLEX_PLUGIN_HINTS)


@dataclass(frozen=True)
class AgentRunBudget:
    """Resolved per-run budget: steps + cumulative estimated prompt tokens."""

    max_steps: int
    max_total_tokens: int
    max_step_refunds: int
    scenario: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "max_steps": self.max_steps,
            "max_total_tokens": self.max_total_tokens,
            "max_step_refunds": self.max_step_refunds,
        }


def resolve_agent_run_budget(
    *,
    mode: AgentComplexityMode | str,
    enable_agent_tools: bool,
    enable_plugin_tools: bool,
) -> AgentRunBudget:
    """Resolve max_steps / token budget by scenario + complexity mode.

    Scenario precedence mirrors scenario_router: superuser agent >
    group plugin selector > private chat.
    """
    from .config import (
        AGENT_STEP_BUDGETS,
        AGENT_STEP_REFUND_RATIO,
        AGENT_TOKEN_BUDGETS,
    )

    if enable_agent_tools:
        scenario = "superuser_agent"
    elif enable_plugin_tools:
        scenario = "group_plugin_selector"
    else:
        scenario = "private_chat"
    steps_by_mode = AGENT_STEP_BUDGETS.get(scenario) or {}
    normalized_mode = str(mode or "chat")
    max_steps = int(
        steps_by_mode.get(normalized_mode) or steps_by_mode.get("chat") or 5
    )
    max_total_tokens = int(AGENT_TOKEN_BUDGETS.get(scenario) or 0)
    max_step_refunds = max(int(max_steps * AGENT_STEP_REFUND_RATIO), 0)
    return AgentRunBudget(
        max_steps=max(max_steps, 1),
        max_total_tokens=max(max_total_tokens, 0),
        max_step_refunds=max_step_refunds,
        scenario=scenario,
    )


__all__ = [
    "AgentComplexityDecision",
    "AgentComplexityMode",
    "AgentRunBudget",
    "resolve_agent_run_budget",
    "route_agent_complexity",
]
