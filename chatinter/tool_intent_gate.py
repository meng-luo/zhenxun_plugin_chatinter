"""LLM-backed tool intent gate for ChatInter.

The local command index is a recall layer only.  This gate decides whether the
current turn should hide tools, expose tools as optional, or require a real
tool call before a final answer.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from zhenxun.services import logger
from zhenxun.services.llm import AI

from .route_text import normalize_message_text
from .soft_tool_policy import (
    is_soft_command_candidate,
    soft_tool_policy_reason,
)

ToolIntentType = Literal[
    "chat",
    "plugin_optional",
    "plugin_required",
    "unsupported",
]
RequestStrengthKind = Literal["none", "weak", "medium", "strong", "explicit"]
ToolObligationKind = Literal["none", "auto", "required"]

_MAX_GATE_CANDIDATES = 36
_MAX_TEXT_CHARS = 420


class ToolIntentGateResult(BaseModel):
    """Structured result for tool-obligation policy."""

    intent_type: ToolIntentType = Field(default="chat")
    request_strength: RequestStrengthKind = Field(default="none")
    mention_only: bool = False
    needs_real_execution: bool = False
    obligation: ToolObligationKind = Field(default="none")
    reason: str = ""
    required_command_ids: list[str] = Field(default_factory=list)
    allowed_command_ids: list[str] = Field(default_factory=list)
    candidate_intent_types: list[str] = Field(default_factory=list)


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
        normalize_message_text(
            str(getattr(getattr(item, "schema", None), "command_id", "") or "")
        )
        for item in candidates
    }
    known_ids.discard("")
    required_ids = _normalize_command_ids(result.required_command_ids, known_ids)
    allowed_ids = _normalize_command_ids(result.allowed_command_ids, known_ids)
    intent_type = _normalize_intent_type(result.intent_type)
    request_strength = _normalize_request_strength(result.request_strength)
    mention_only = bool(result.mention_only)
    if intent_type == "plugin_required" and not known_ids:
        intent_type = "chat"
    if mention_only and intent_type == "plugin_required":
        intent_type = "plugin_optional" if known_ids else "chat"
    if mention_only and request_strength in {"strong", "explicit"}:
        request_strength = "weak"
    obligation = _normalize_obligation(
        result.obligation,
        intent_type=intent_type,
        request_strength=request_strength,
        mention_only=mention_only,
        candidates=candidates,
        required_ids=required_ids,
        allowed_ids=allowed_ids,
    )
    return ToolIntentGateResult(
        intent_type=intent_type,
        request_strength=request_strength,
        mention_only=mention_only,
        needs_real_execution=(
            (bool(result.needs_real_execution) and not mention_only)
            or obligation == "required"
        ),
        obligation=obligation,
        reason=normalize_message_text(result.reason),
        required_command_ids=required_ids,
        allowed_command_ids=allowed_ids,
        candidate_intent_types=_normalize_intent_types(result.candidate_intent_types),
    )


def _normalize_obligation(
    value: str,
    *,
    intent_type: ToolIntentType,
    request_strength: RequestStrengthKind,
    mention_only: bool,
    candidates: list[Any],
    required_ids: list[str],
    allowed_ids: list[str],
) -> ToolObligationKind:
    normalized = normalize_message_text(str(value or "")).lower()
    if normalized in {"none", "auto", "required"}:
        obligation: ToolObligationKind = normalized  # type: ignore[assignment]
    elif intent_type == "plugin_required":
        obligation = "required"
    elif intent_type == "plugin_optional":
        obligation = "auto"
    else:
        obligation = "none"
    if mention_only:
        return "auto" if candidates and intent_type != "chat" else "none"
    if obligation == "required" and request_strength not in {"strong", "explicit"}:
        obligation = "auto"
    if obligation == "required" and not candidates:
        return "none"
    if obligation == "required" and not (required_ids or allowed_ids):
        strong_candidates = [
            candidate
            for candidate in candidates
            if _candidate_can_satisfy_required_tool(candidate)
        ]
        if not strong_candidates:
            return "auto"
    if intent_type == "chat":
        return "none"
    if intent_type == "unsupported":
        return "auto" if candidates else "none"
    return obligation


def _candidate_can_satisfy_required_tool(candidate: Any) -> bool:
    if bool(getattr(candidate, "exact_protected", False)):
        return True
    snapshot = getattr(candidate, "tool", None)
    if snapshot is None:
        return float(getattr(candidate, "score", 0.0) or 0.0) >= 120.0
    policy = normalize_message_text(
        str(getattr(snapshot, "execution_policy", "") or "normal")
    )
    if bool(getattr(snapshot, "soft_tool", False)) and policy == "explicit_only":
        return False
    risk_level = normalize_message_text(str(getattr(snapshot, "risk_level", "") or ""))
    side_effect = normalize_message_text(
        str(getattr(snapshot, "side_effect", "") or "")
    )
    output_mode = normalize_message_text(
        str(getattr(snapshot, "output_mode", "") or "")
    )
    source_of_truth = normalize_message_text(
        str(getattr(snapshot, "source_of_truth", "") or "")
    )
    schema_quality = float(getattr(snapshot, "schema_quality", 0.5) or 0.5)
    reliability = float(getattr(snapshot, "reliability", 0.5) or 0.5)
    intent_types = {
        normalize_message_text(str(intent or "")).lower()
        for intent in list(getattr(snapshot, "intent_types", []) or [])
        if normalize_message_text(str(intent or ""))
    }
    score = float(getattr(candidate, "score", 0.0) or 0.0)
    return (
        (score >= 120.0 and reliability >= 0.38 and schema_quality >= 0.35)
        or risk_level in {"medium", "high"}
        or side_effect in {"query", "send", "mutate"}
        or output_mode in {"image", "file", "action"}
        or source_of_truth in {"bot_state", "external_service", "local_state"}
        or bool(getattr(snapshot, "requires_real_tool", False))
        or bool(getattr(snapshot, "requires_real_result", False))
        or bool(getattr(snapshot, "generative", False))
        or policy in {"strong_intent", "confirmation_required"}
        or bool(
            intent_types & {"query", "status", "generate", "media", "random", "play"}
        )
    )


def _normalize_intent_types(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or "")).lower()
        if text and text not in result:
            result.append(text[:48])
    return result[:8]


def _normalize_intent_type(value: str) -> ToolIntentType:
    normalized = normalize_message_text(str(value or "")).lower()
    if normalized in {"chat", "plugin_optional", "plugin_required", "unsupported"}:
        return normalized  # type: ignore[return-value]
    return "chat"


def _normalize_request_strength(value: str) -> RequestStrengthKind:
    normalized = normalize_message_text(str(value or "")).lower()
    if normalized in {"none", "weak", "medium", "strong", "explicit"}:
        return normalized  # type: ignore[return-value]
    return "none"


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
            intent_type="chat",
            request_strength="none",
            mention_only=False,
            needs_real_execution=False,
            obligation="none",
            reason="gate_failed_no_positive_candidate",
        )
    return ToolIntentGateResult(
        intent_type="plugin_optional",
        request_strength="weak",
        mention_only=False,
        needs_real_execution=False,
        obligation="auto",
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
        "capability_card": _capability_card_summary(snapshot),
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
            "semantic": round(
                float(getattr(features, "semantic_score", 0.0) or 0.0), 2
            ),
            "context": round(float(getattr(features, "context_score", 0.0) or 0.0), 2),
            "feedback": round(
                float(getattr(features, "feedback_score", 0.0) or 0.0), 2
            ),
            "schema": round(float(getattr(features, "schema_score", 0.0) or 0.0), 2),
            "reliability": round(
                float(getattr(features, "reliability_score", 0.0) or 0.0), 2
            ),
            "false_trigger": round(
                float(getattr(features, "false_trigger_score", 0.0) or 0.0), 2
            ),
            "param_failure": round(
                float(getattr(features, "param_failure_score", 0.0) or 0.0), 2
            ),
            "latency": round(float(getattr(features, "latency_score", 0.0) or 0.0), 2),
        },
        "request_strength": _request_strength_summary(candidate),
        "soft_tool_policy": {
            "is_soft_tool": is_soft_command_candidate(candidate),
            "execution_policy": soft_tool_policy_reason(candidate),
        },
    }


def _clip(value: Any, *, limit: int = _MAX_TEXT_CHARS) -> str:
    text = normalize_message_text(str(value or ""))
    return text[: max(1, int(limit or _MAX_TEXT_CHARS))]


_TOOL_INTENT_GATE_INSTRUCTION = """
你是 ChatInter 的工具意图门控器。
你只判断“这条消息是否需要真实插件工具执行”，不执行任何任务。

分类：
- chat：寒暄、闲聊、安慰、讨论概念、解释词义、常识回答。
- chat 还包括写作建议、对插件/命令的泛泛讨论；不需要真实插件结果。
- plugin_optional：候选工具可能有帮助，但用户意图不够明确。
- plugin_optional 时允许主模型看到工具并自行决定。
- plugin_required：用户明确要求机器人执行、查询、生成、翻译。
- plugin_required 还包括抽取、制作、发送、签到、查看个人/群状态。
- 如果自然语言需求明显只能由候选插件给出真实结果，也用 plugin_required。
- unsupported：用户有工具/插件意图，但候选命令都不合适。

同时输出 request_strength：
- none：没有工具请求，只是聊天/解释/讨论。
- weak：只隐约相关，或只是提到能力词。
- medium：有可执行倾向，但目标或上下文还不够明确。
- strong：明确要执行/查询/生成真实结果。
- explicit：直接命令、精确命令头、或明确说“调用/执行这个插件”。

同时输出 mention_only：
- true：用户只是提到命令词、插件名、能力名，或讨论它们。
- true 表示不是让机器人执行。
- false：用户在请求真实动作、真实查询、真实生成。
- false 也用于候选能力与当前任务直接相关的情况。

同时输出 obligation：
- none：纯聊天，不暴露/不要求命令工具。
- auto：可以暴露工具，但模型可聊天也可调用。
- required：必须获取真实工具结果，不能直接代答。

关键规则：
- 本地候选分数和 rank 只是召回信号，不是执行决策。
- explicit_only 的低上下文或生成工具，默认不要升为 plugin_required。
- 只有用户明确点名执行、请求随机/生成/查看真实结果时才升为 required。
- capability_card 描述工具的 source_of_truth、requires_real_tool。
- capability_card 还描述 output_mode、entity_scope、risk、reliability。
- capability_card 也描述 schema_quality、intent_types 和适用/不适用场景。
- 优先按这些属性判断，不按插件名判断。
- 如果用户只是问词义、命令含义、插件能力，通常是 chat。
- 普通聊天中提到候选词，也通常是 chat。
- 如果用户说“帮我/给我/查/查询/看一下/生成/制作/抽/翻译/签到/发/调用”。
- 且有匹配候选，通常是 plugin_required。
- 如果需要图片、文件、外部状态、群成员状态、随机抽取。
- 或需要插件内部数据，不能直接代答；应判为 plugin_required。
- mention_only=true 时 obligation 通常不能是 required。
- needs_real_execution=true 表示回答“完成/查到/生成了”前必须有 observation。
- required_command_ids 只填最匹配、应真实执行的 command_id。
- 不确定时留空并用 plugin_optional。
- candidate_intent_types 填通用能力类型。
- 例如 chat/query/generate/image/template/translate/random/status/help，不要填插件名。

只返回 JSON：
{
  "intent_type": "chat",
  "request_strength": "none",
  "mention_only": false,
  "needs_real_execution": false,
  "obligation": "none",
  "reason": "",
  "required_command_ids": [],
  "allowed_command_ids": [],
  "candidate_intent_types": []
}
""".strip()


def _capability_card_summary(snapshot: Any | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    payload = {
        "use_cases": list(getattr(snapshot, "use_cases", []) or [])[:4],
        "anti_use_cases": list(getattr(snapshot, "anti_use_cases", []) or [])[:4],
        "source_of_truth": _clip(getattr(snapshot, "source_of_truth", ""), limit=80),
        "requires_real_tool": bool(getattr(snapshot, "requires_real_tool", True)),
        "output_mode": _clip(getattr(snapshot, "output_mode", ""), limit=80),
        "entity_scope": _clip(getattr(snapshot, "entity_scope", ""), limit=80),
        "side_effect": _clip(getattr(snapshot, "side_effect", ""), limit=80),
        "risk": _clip(
            getattr(snapshot, "risk", "") or getattr(snapshot, "risk_level", ""),
            limit=80,
        ),
        "reliability": round(float(getattr(snapshot, "reliability", 0.0) or 0.0), 3),
        "schema_quality": round(
            float(getattr(snapshot, "schema_quality", 0.0) or 0.0),
            3,
        ),
        "soft_tool": bool(getattr(snapshot, "soft_tool", False)),
        "intent_types": list(getattr(snapshot, "intent_types", []) or [])[:8],
        "requires_real_result": bool(getattr(snapshot, "requires_real_result", True)),
        "generative": bool(getattr(snapshot, "generative", False)),
        "execution_policy": _clip(
            getattr(snapshot, "execution_policy", ""),
            limit=80,
        ),
    }
    return {
        key: value for key, value in payload.items() if value not in ("", [], {}, None)
    }


def _request_strength_summary(candidate: Any) -> dict[str, Any]:
    features = getattr(candidate, "features", None)
    return {
        "recall_score": round(float(getattr(candidate, "score", 0.0) or 0.0), 2),
        "exact_command": bool(getattr(candidate, "exact_protected", False)),
        "context_score": round(
            float(getattr(features, "context_score", 0.0) or 0.0),
            2,
        ),
    }


__all__ = [
    "RequestStrengthKind",
    "ToolIntentGate",
    "ToolIntentGateResult",
    "ToolIntentType",
    "ToolObligationKind",
]
