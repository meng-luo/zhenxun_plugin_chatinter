"""Complexity routing for the superuser private Agent.

Lightweight requests should not pay the planner-executor-verifier tax.  This
gate is intentionally local and metadata-driven: it only decides whether the
runtime should enable the durable planner/verifier layer, not which tool to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import re

from .route_text import normalize_message_text

AgentComplexityMode = Literal["chat", "single_tool_fast", "complex_pev"]

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


__all__ = [
    "AgentComplexityDecision",
    "AgentComplexityMode",
    "route_agent_complexity",
]
