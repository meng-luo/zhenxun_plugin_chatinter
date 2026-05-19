"""LLM-backed tool intent gate for ChatInter.

The local command index is a recall layer only.  This gate decides whether the
current turn should hide tools, expose tools as optional, or require a real
tool call before a final answer.
"""

from __future__ import annotations

from typing import Any, Literal
import json

from pydantic import BaseModel, Field

from zhenxun.services import logger
from zhenxun.services.llm import AI

from .route_text import normalize_message_text
from .soft_tool_policy import is_soft_command_candidate, soft_tool_policy_reason

ToolIntentKind = Literal[
    "chat",
    "plugin_optional",
    "plugin_required",
    "unsupported",
]

_MAX_GATE_CANDIDATES = 36
_MAX_TEXT_CHARS = 420


class ToolIntentGateResult(BaseModel):
    """Structured result for tool-obligation policy."""

    intent: ToolIntentKind = Field(default="chat")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    required_command_ids: list[str] = Field(default_factory=list)
    allowed_command_ids: list[str] = Field(default_factory=list)
    needs_real_execution: bool = False


class ToolIntentGate:
    """Semantic gate that prevents recall scores from becoming hard decisions."""

    def __init__(
        self,
        *,
        trace_id: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
    ) -> None:
        self.ai = AI(session_id=f"chatinter-tool-intent:{trace_id}")
        self.model_name = model_name
        self.generation_config = generation_config
        self.timeout = max(4.0, min(float(timeout or 12.0), 18.0))

    async def judge(
        self,
        *,
        message_text: str,
        candidates: list[Any],
        command_tool_count: int,
    ) -> ToolIntentGateResult:
        payload = {
            "message": normalize_message_text(message_text),
            "command_tool_count": max(int(command_tool_count or 0), 0),
            "candidate_count": len(candidates),
            "candidates": [
                _candidate_summary(candidate, index=index)
                for index, candidate in enumerate(
                    candidates[:_MAX_GATE_CANDIDATES],
                    1,
                )
            ],
        }
        try:
            result = await self.ai.generate_structured(
                json.dumps(payload, ensure_ascii=False),
                ToolIntentGateResult,
                model=self.model_name,
                config=self.generation_config,
                instruction=_TOOL_INTENT_GATE_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                auto_thinking=False,
            )
            return _normalize_result(result, candidates=candidates)
        except Exception as exc:
            logger.warning(f"[ChatInter] tool intent gate failed: {exc}")
            return _fallback_result(candidates)


def _normalize_result(
    result: ToolIntentGateResult,
    *,
    candidates: list[Any],
) -> ToolIntentGateResult:
    known_ids = {
        normalize_message_text(str(getattr(getattr(item, "schema", None), "command_id", "") or ""))
        for item in candidates
    }
    known_ids.discard("")
    required_ids = _normalize_command_ids(result.required_command_ids, known_ids)
    allowed_ids = _normalize_command_ids(result.allowed_command_ids, known_ids)
    intent = result.intent
    if intent == "plugin_required" and not known_ids:
        intent = "chat"
    return ToolIntentGateResult(
        intent=intent,
        confidence=max(0.0, min(float(result.confidence or 0.0), 1.0)),
        reason=normalize_message_text(result.reason),
        required_command_ids=required_ids,
        allowed_command_ids=allowed_ids,
        needs_real_execution=bool(result.needs_real_execution)
        or intent == "plugin_required",
    )


def _normalize_command_ids(
    values: list[str] | tuple[str, ...],
    known_ids: set[str],
) -> list[str]:
    result: list[str] = []
    for value in values:
        command_id = normalize_message_text(str(value or ""))
        if not command_id or command_id in result:
            continue
        if known_ids and command_id not in known_ids:
            continue
        result.append(command_id)
    return result[:12]


def _fallback_result(candidates: list[Any]) -> ToolIntentGateResult:
    has_positive_candidate = any(
        float(getattr(candidate, "score", 0.0) or 0.0) > 0 for candidate in candidates
    )
    if not has_positive_candidate:
        return ToolIntentGateResult(
            intent="chat",
            confidence=0.35,
            reason="gate_failed_no_positive_candidate",
        )
    return ToolIntentGateResult(
        intent="plugin_optional",
        confidence=0.35,
        reason="gate_failed_conservative_optional",
    )


def _candidate_summary(candidate: Any, *, index: int) -> dict[str, Any]:
    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    features = getattr(candidate, "features", None)
    aliases = list(getattr(schema, "aliases", []) or [])[:6]
    slots = list(getattr(schema, "slots", []) or [])
    requires = {
        key: bool(value)
        for key, value in dict(getattr(schema, "requires", {}) or {}).items()
        if value
    }
    return {
        "rank": index,
        "score": round(float(getattr(candidate, "score", 0.0) or 0.0), 2),
        "reason": _clip(getattr(candidate, "reason", "")),
        "reasons": list(getattr(candidate, "reasons", ()) or ())[:8],
        "exact_protected": bool(getattr(candidate, "exact_protected", False)),
        "plugin_name": _clip(getattr(candidate, "plugin_name", "")),
        "plugin_module": _clip(getattr(candidate, "plugin_module", "")),
        "command_id": _clip(getattr(schema, "command_id", "")),
        "head": _clip(getattr(schema, "head", "")),
        "aliases": [_clip(alias, limit=80) for alias in aliases if _clip(alias)],
        "description": _clip(getattr(schema, "description", "")),
        "capability": _clip(getattr(snapshot, "capability_text", "")),
        "usage": _clip(getattr(snapshot, "usage", ""), limit=240),
        "examples": [
            _clip(example, limit=120)
            for example in list(getattr(snapshot, "examples", []) or [])[:3]
            if _clip(example)
        ],
        "role": _clip(getattr(schema, "command_role", "")),
        "payload_policy": _clip(getattr(schema, "payload_policy", "")),
        "target_requirement": _clip(getattr(schema, "target_requirement", "")),
        "requires": requires,
        "slots": [
            {
                "name": _clip(getattr(slot, "name", ""), limit=80),
                "type": _clip(getattr(slot, "type", ""), limit=40),
                "required": bool(getattr(slot, "required", False)),
                "description": _clip(getattr(slot, "description", ""), limit=120),
                "aliases": [
                    _clip(alias, limit=60)
                    for alias in list(getattr(slot, "aliases", []) or [])[:4]
                    if _clip(alias)
                ],
            }
            for slot in slots[:8]
        ],
        "features": {
            "exact": round(float(getattr(features, "exact_score", 0.0) or 0.0), 2),
            "lexical": round(float(getattr(features, "lexical_score", 0.0) or 0.0), 2),
            "semantic": round(float(getattr(features, "semantic_score", 0.0) or 0.0), 2),
            "context": round(float(getattr(features, "context_score", 0.0) or 0.0), 2),
            "feedback": round(float(getattr(features, "feedback_score", 0.0) or 0.0), 2),
        },
        "soft_tool_policy": {
            "is_soft_tool": is_soft_command_candidate(candidate),
            "execution_policy": soft_tool_policy_reason(candidate),
        },
    }


def _clip(value: Any, *, limit: int = _MAX_TEXT_CHARS) -> str:
    text = normalize_message_text(str(value or ""))
    return text[: max(1, int(limit or _MAX_TEXT_CHARS))]


_TOOL_INTENT_GATE_INSTRUCTION = """
你是 ChatInter 的工具意图门控器。你只判断“这条消息是否需要真实插件工具执行”，不执行任何任务。

分类：
- chat：寒暄、闲聊、安慰、讨论概念、解释词义、常识回答、写作建议、对插件/命令的泛泛讨论；不需要真实插件结果。
- plugin_optional：候选工具可能有帮助，但用户意图不够明确，允许主模型看到工具并自行决定。
- plugin_required：用户明确要求机器人执行、查询、生成、翻译、抽取、制作、发送、签到、查看个人/群状态，或自然语言需求明显只能由候选插件给出真实结果。
- unsupported：用户有工具/插件意图，但候选命令都不合适。

关键规则：
- 本地候选分数和 rank 只是召回信号，不是执行决策。
- 软内容插件（语录、鸡汤、roll、关于/介绍等）只有在用户明确要求“来一条/抽一个/执行/查看/调用/介绍机器人功能”等真实插件行为时才是 plugin_required。
- candidates 里标记 is_soft_tool=true 的低上下文工具，默认不要升为 plugin_required；只有用户明确点名执行、请求随机/生成/查看其结果时才升为 plugin_required。
- 如果用户只是问某个词、命令名、插件能力的含义，或在普通聊天中提到候选词，通常是 chat。
- 如果用户说“帮我/给我/查/查询/看一下/生成/制作/抽/翻译/签到/发/调用”，且有匹配候选，通常是 plugin_required。
- 如果需要图片、文件、外部状态、群成员状态、随机抽取或插件内部数据，不能直接代答；应判为 plugin_required。
- required_command_ids 只填最匹配、应真实执行的 command_id；不确定时留空并用 plugin_optional。

只返回 JSON：
{
  "intent": "chat",
  "confidence": 0.0,
  "reason": "",
  "required_command_ids": [],
  "allowed_command_ids": [],
  "needs_real_execution": false
}
""".strip()


__all__ = [
    "ToolIntentGate",
    "ToolIntentGateResult",
    "ToolIntentKind",
]
