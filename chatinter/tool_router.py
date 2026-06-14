"""Structured top-k tool router for ChatInter.

The router is a decision helper only.  It receives a small candidate set from
the retriever, asks the model for a structured choice constrained to those
tool names, then validates the result locally.  It never executes tools.
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, create_model

from .command_index import CommandCandidate
from .route_text import normalize_message_text

ToolRouterAction = Literal["select", "clarify", "none"]

_MAX_ROUTER_CANDIDATES = 12
_MAX_TEXT_CHARS = 420


class ToolRouterDecision(BaseModel):
    """Stable router output after local validation."""

    action: ToolRouterAction = Field(default="none")
    tool_name: str = ""
    command_id: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_question: str = ""
    reason: str = ""


class ToolRouter:
    """LLM-backed router constrained to a turn-local top-k enum."""

    def __init__(
        self,
        *,
        trace_id: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
    ) -> None:
        self.ai: Any = _create_ai(session_id=f"chatinter-tool-router:{trace_id}")
        self.model_name = model_name
        self.generation_config = generation_config
        self.timeout = max(4.0, min(float(timeout or 12.0), 18.0))

    async def route(
        self,
        *,
        message_text: str,
        candidates: list[CommandCandidate],
        tool_names_by_command_id: dict[str, str] | None = None,
    ) -> ToolRouterDecision:
        options = build_tool_router_options(
            candidates,
            tool_names_by_command_id=tool_names_by_command_id,
        )
        if not options:
            return ToolRouterDecision(
                action="none",
                reason="no_candidate_options",
            )

        response_model = build_tool_router_response_model(
            [option["tool_name"] for option in options],
        )
        payload = {
            "message": normalize_message_text(message_text),
            "selection_policy": (
                "tool_name 必须来自 candidate_options；rank/score 只用于参考，"
                "不能替代 schema 和用户意图判断。"
            ),
            "candidate_options": options,
        }
        try:
            result = await self.ai.generate_structured(
                json.dumps(payload, ensure_ascii=False),
                response_model,
                model=self.model_name,
                config=self.generation_config,
                instruction=_TOOL_ROUTER_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                auto_thinking=False,
            )
            return normalize_tool_router_result(
                result,
                candidates=candidates,
                tool_names_by_command_id=tool_names_by_command_id,
            )
        except Exception as exc:
            _log_warning(f"[ChatInter] tool router failed: {exc}")
            return fallback_tool_router_decision(candidates)


def build_tool_router_options(
    candidates: list[CommandCandidate],
    *,
    tool_names_by_command_id: dict[str, str] | None = None,
    limit: int = _MAX_ROUTER_CANDIDATES,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen_tools: set[str] = set()
    for index, candidate in enumerate(candidates[: max(1, int(limit or 1))], 1):
        command_id = normalize_message_text(candidate.schema.command_id)
        if not command_id:
            continue
        tool_name = _tool_name_for_candidate(
            candidate,
            tool_names_by_command_id=tool_names_by_command_id,
        )
        if not tool_name or tool_name in seen_tools:
            continue
        seen_tools.add(tool_name)
        options.append(_candidate_option(candidate, index=index, tool_name=tool_name))
    return options


def build_tool_router_response_model(tool_names: list[str]) -> type[BaseModel]:
    """Create a structured response schema with tool_name constrained to top-k."""

    enum_values = [
        normalize_message_text(name)
        for name in dict.fromkeys(tool_names)
        if normalize_message_text(name)
    ]
    return create_model(
        "ToolRouterResponse",
        action=(
            str,
            Field(
                default="none",
                description="select/clarify/none",
                json_schema_extra=cast(Any, {"enum": ["select", "clarify", "none"]}),
            ),
        ),
        tool_name=(
            str,
            Field(
                default="",
                description="选择的工具名；select 时必须来自 enum",
                json_schema_extra=cast(Any, {"enum": enum_values}),
            ),
        ),
        arguments=(
            dict[str, Any],
            Field(default_factory=dict, description="按候选 schema 抽取的参数"),
        ),
        confidence=(float, Field(default=0.0, ge=0.0, le=1.0)),
        needs_clarification=(bool, Field(default=False)),
        clarification_question=(str, Field(default="")),
        reason=(str, Field(default="")),
    )


def normalize_tool_router_result(
    result: Any,
    *,
    candidates: list[CommandCandidate],
    tool_names_by_command_id: dict[str, str] | None = None,
) -> ToolRouterDecision:
    options = build_tool_router_options(
        candidates,
        tool_names_by_command_id=tool_names_by_command_id,
    )
    allowed_tools = {option["tool_name"] for option in options}
    command_by_tool = {
        option["tool_name"]: normalize_message_text(str(option["command_id"]))
        for option in options
    }

    action = _normalize_action(getattr(result, "action", "none"))
    tool_name = normalize_message_text(str(getattr(result, "tool_name", "") or ""))
    needs_clarification = bool(getattr(result, "needs_clarification", False))
    if action == "select" and tool_name not in allowed_tools:
        action = "clarify" if needs_clarification else "none"
        tool_name = ""
    if action == "clarify":
        tool_name = ""
        needs_clarification = True
    if action == "none":
        tool_name = ""
        needs_clarification = False

    arguments = getattr(result, "arguments", {}) or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolRouterDecision(
        action=action,
        tool_name=tool_name,
        command_id=command_by_tool.get(tool_name, ""),
        arguments={
            normalize_message_text(str(key)): value
            for key, value in arguments.items()
            if normalize_message_text(str(key))
        },
        confidence=_coerce_confidence(getattr(result, "confidence", 0.0)),
        needs_clarification=needs_clarification,
        clarification_question=_clip(
            getattr(result, "clarification_question", ""),
            limit=160,
        ),
        reason=_clip(getattr(result, "reason", ""), limit=240),
    )


def fallback_tool_router_decision(
    candidates: list[CommandCandidate],
) -> ToolRouterDecision:
    if not candidates:
        return ToolRouterDecision(action="none", reason="router_failed_no_candidate")
    exact = [candidate for candidate in candidates if candidate.exact_protected]
    if len(exact) == 1:
        tool_name = _tool_name_for_candidate(exact[0], tool_names_by_command_id=None)
        return ToolRouterDecision(
            action="select",
            tool_name=tool_name,
            command_id=normalize_message_text(exact[0].schema.command_id),
            confidence=0.72,
            reason="router_failed_exact_single_fallback",
        )
    return ToolRouterDecision(
        action="clarify",
        needs_clarification=True,
        clarification_question="你想让我使用哪个插件命令？",
        reason="router_failed_needs_clarification",
    )


def _candidate_option(
    candidate: CommandCandidate,
    *,
    index: int,
    tool_name: str,
) -> dict[str, Any]:
    schema = candidate.schema
    tool = candidate.tool
    return {
        "rank": index,
        "tool_name": tool_name,
        "command_id": normalize_message_text(schema.command_id),
        "plugin_name": _clip(candidate.plugin_name),
        "plugin_module": _clip(candidate.plugin_module),
        "head": _clip(schema.head),
        "aliases": [_clip(alias, limit=80) for alias in list(schema.aliases)[:6]],
        "description": _clip(schema.description),
        "payload_policy": normalize_message_text(schema.payload_policy),
        "command_role": normalize_message_text(schema.command_role),
        "score": round(float(candidate.score or 0.0), 2),
        "exact_protected": bool(candidate.exact_protected),
        "reason": _clip(candidate.reason, limit=160),
        "capability_text": _clip(getattr(tool, "capability_text", ""), limit=260),
        "intent_types": list(getattr(tool, "intent_types", []) or [])[:8],
        "output_mode": _clip(getattr(tool, "output_mode", ""), limit=80),
        "side_effect": _clip(getattr(tool, "side_effect", ""), limit=80),
        "requires_real_tool": bool(getattr(tool, "requires_real_tool", True)),
        "requires_real_result": bool(getattr(tool, "requires_real_result", True)),
        "slots": [
            {
                "name": _clip(getattr(slot, "name", ""), limit=80),
                "type": _clip(getattr(slot, "type", ""), limit=40),
                "required": bool(getattr(slot, "required", False)),
                "aliases": [
                    _clip(alias, limit=60)
                    for alias in list(getattr(slot, "aliases", []) or [])[:4]
                ],
                "description": _clip(
                    getattr(slot, "description", ""),
                    limit=120,
                ),
            }
            for slot in list(schema.slots or [])[:8]
        ],
    }


def _tool_name_for_candidate(
    candidate: CommandCandidate,
    *,
    tool_names_by_command_id: dict[str, str] | None,
) -> str:
    command_id = normalize_message_text(candidate.schema.command_id)
    mapped = normalize_message_text(
        str((tool_names_by_command_id or {}).get(command_id, "") or "")
    )
    if mapped:
        return mapped
    return f"tool_{command_id}" if command_id else ""


def _normalize_action(value: Any) -> ToolRouterAction:
    normalized = normalize_message_text(str(value or "")).lower()
    if normalized in {"select", "clarify", "none"}:
        return cast(ToolRouterAction, normalized)
    return "none"


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _clip(value: Any, *, limit: int = _MAX_TEXT_CHARS) -> str:
    text = normalize_message_text(str(value or ""))
    return text[: max(1, int(limit or _MAX_TEXT_CHARS))]


def _create_ai(*, session_id: str) -> Any:
    from zhenxun.services.llm import AI

    return AI(session_id=session_id)


def _log_warning(message: str) -> None:
    try:
        from zhenxun.services import logger

        logger.warning(message)
    except Exception:
        return


_TOOL_ROUTER_INSTRUCTION = """
你是 ChatInter 的结构化工具路由器。
你只在给定 candidate_options 中选择工具，不执行工具，不发最终回答。

规则：
- tool_name 必须严格来自 candidate_options 的 tool_name enum。
- 如果没有明确匹配，action=none。
- 如果用户想使用工具但参数/目标不清楚，action=clarify。
- 只有当前消息明确需要某个候选工具真实执行时，action=select。
- rank/score/reason 只是召回参考；最终按用户意图、schema、slots 和能力描述判断。
- 不要选择 enum 外工具名，不要编造工具。
- arguments 只放能从用户消息中直接确定的参数；不确定就不要填。
- 对 required slot 缺失且无法从上下文确定时，action=clarify。
- needs_clarification=true 时必须给 clarification_question。

只返回 JSON：
{
  "action": "select",
  "tool_name": "",
  "arguments": {},
  "confidence": 0.0,
  "needs_clarification": false,
  "clarification_question": "",
  "reason": ""
}
""".strip()


__all__ = [
    "ToolRouter",
    "ToolRouterAction",
    "ToolRouterDecision",
    "build_tool_router_options",
    "build_tool_router_response_model",
    "fallback_tool_router_decision",
    "normalize_tool_router_result",
]
