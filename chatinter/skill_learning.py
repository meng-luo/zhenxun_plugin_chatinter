"""Background skill extraction for successful superuser Agent runs."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from zhenxun.services import logger
from zhenxun.services.llm import AI

from .config import build_reasoning_generation_config, get_config_value, get_model_name
from .feedback import FeedbackStore
from .route_text import normalize_message_text
from .skill_store import SkillCandidate, SkillRecord, upsert_skill

_SKILL_LEARNING_TASKS: set[asyncio.Task[None]] = set()
_MAX_RECORD_TEXT = 1800
_MAX_PAYLOAD_TEXT = 6000
_FAILURE_STOP_MARKERS = (
    "exception",
    "failed",
    "cancel",
    "timeout",
    "max_steps",
    "budget_exhausted",
)


class _ExtractedSkill(BaseModel):
    should_store: bool = Field(default=False)
    title: str = ""
    pattern: str = ""
    summary: str = ""
    steps: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    confidence: float = 0.55


def schedule_skill_learning(record: dict[str, Any]) -> None:
    """Extract a reusable skill in the background after a successful run."""

    if not _should_learn(record):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(_learn_safely(dict(record)))
    _SKILL_LEARNING_TASKS.add(task)
    task.add_done_callback(_SKILL_LEARNING_TASKS.discard)


async def learn_skill_from_trajectory_record(
    record: dict[str, Any],
) -> SkillRecord | None:
    if not _should_learn(record):
        return None
    candidate = await _extract_skill_with_llm(record)
    if candidate is None:
        candidate = _fallback_skill_candidate(record)
    if candidate is None:
        return None
    return upsert_skill(candidate)


async def _learn_safely(record: dict[str, Any]) -> None:
    try:
        skill = await learn_skill_from_trajectory_record(record)
        if skill is not None:
            logger.debug(f"[ChatInter] learned skill {skill.skill_id}: {skill.title}")
    except Exception as exc:
        logger.debug(f"[ChatInter] skill learning skipped: {type(exc).__name__}: {exc}")


async def _extract_skill_with_llm(record: dict[str, Any]) -> SkillCandidate | None:
    payload = _skill_prompt_payload(record)
    if not payload:
        return None
    trace_id = normalize_message_text(str(record.get("trace_id", "") or "unknown"))
    try:
        result = await AI(session_id=f"chatinter-skill:{trace_id}").generate_structured(
            json.dumps(payload, ensure_ascii=False),
            _ExtractedSkill,
            model=get_model_name(),
            config=build_reasoning_generation_config(),
            instruction=_SKILL_EXTRACT_INSTRUCTION,
            timeout=_skill_learning_timeout(),
            max_validation_retries=0,
            auto_thinking=False,
        )
    except Exception as exc:
        logger.debug(
            "[ChatInter] LLM skill extraction failed for "
            f"{trace_id}: {type(exc).__name__}: {exc}"
        )
        return None
    if not result.should_store:
        return None
    confidence = _merge_confidence(float(result.confidence or 0.0), record)
    return SkillCandidate(
        title=result.title,
        pattern=result.pattern,
        summary=result.summary,
        steps=tuple(result.steps),
        tools=tuple(result.tools),
        cautions=tuple(result.cautions),
        tags=("superuser_agent", *tuple(result.tags)),
        examples=tuple(result.examples),
        markdown="",
        trace_id=trace_id,
        confidence=confidence,
    )


def _fallback_skill_candidate(record: dict[str, Any]) -> SkillCandidate | None:
    tools = _selected_tools(record)
    observations = _observations(record)
    ok_observations = [item for item in observations if item.get("ok")]
    completed = _completed_task_texts(record)
    steps = _dedupe_texts(
        [
            *completed,
            *[
                str(item.get("task_text") or item.get("status") or "")
                for item in ok_observations
            ],
            *[f"调用工具 {tool}" for tool in tools],
        ],
        limit=10,
    )
    if not tools and not steps:
        return None
    input_message = normalize_message_text(str(record.get("input_message", "") or ""))
    final_reply = normalize_message_text(str(record.get("final_reply", "") or ""))
    title = _title_from_input(input_message, tools)
    summary_parts = []
    if tools:
        summary_parts.append("常用工具序列: " + ", ".join(tools[:10]))
    if final_reply:
        summary_parts.append("成功答复摘要: " + final_reply[:240])
    summary = "；".join(summary_parts) or "从成功超级用户任务轨迹提炼出的复用流程。"
    cautions = ["必须以真实工具 observation 为依据，失败工具结果不能当作完成证据。"]
    return SkillCandidate(
        title=title,
        pattern=input_message[:240] or title,
        summary=summary[:500],
        steps=tuple(steps),
        tools=tuple(tools),
        cautions=tuple(cautions),
        tags=("superuser_agent", *tuple(tools[:8])),
        examples=(input_message[:500],) if input_message else (),
        trace_id=normalize_message_text(str(record.get("trace_id", "") or "")),
        confidence=_merge_confidence(0.5, record),
    )


def _should_learn(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    agent_mode = normalize_message_text(str(record.get("agent_mode", "") or ""))
    scenario = normalize_message_text(str(record.get("scenario", "") or ""))
    if agent_mode == "superuser_subagent":
        return False
    if agent_mode != "superuser_agent" and scenario != "superuser_agent":
        return False
    if normalize_message_text(str(record.get("status", "") or "")) != "completed":
        return False
    stop_reason = normalize_message_text(str(record.get("stop_reason", "") or ""))
    if any(marker in stop_reason for marker in _FAILURE_STOP_MARKERS):
        return False
    if _selected_tools(record):
        return True
    if _completed_task_texts(record):
        return True
    return any(item.get("ok") for item in _observations(record))


def _skill_prompt_payload(record: dict[str, Any]) -> dict[str, Any]:
    selected_tools = _selected_tools(record)
    observations = _observations(record)
    if not selected_tools and not observations:
        return {}
    payload = {
        "input_message": _clip(record.get("input_message"), limit=1000),
        "selected_tools": selected_tools[:20],
        "tool_calls": _clip_jsonable(record.get("tool_calls"), limit=1600),
        "observations": [
            {
                "tool_name": _clip(item.get("tool_name"), limit=120),
                "task_text": _clip(item.get("task_text"), limit=240),
                "ok": bool(item.get("ok")),
                "status": _clip(item.get("status"), limit=120),
                "error": _clip(item.get("error"), limit=200),
                "messages_sent_summary": _clip_jsonable(
                    item.get("messages_sent_summary"), limit=300
                ),
            }
            for item in observations[:20]
        ],
        "completed_tasks": _completed_task_texts(record)[:12],
        "final_reply": _clip(record.get("final_reply"), limit=800),
        "evaluation": _clip_jsonable(record.get("evaluation"), limit=800),
        "feedback_profiles": _feedback_profiles(selected_tools),
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded) <= _MAX_PAYLOAD_TEXT:
        return payload
    payload["tool_calls"] = []
    payload["observations"] = payload["observations"][:8]
    payload["final_reply"] = _clip(payload.get("final_reply"), limit=400)
    return payload


def _feedback_profiles(tools: list[str]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for tool in tools[:12]:
        profile = FeedbackStore.command_feedback_profile(command_id=tool)
        if profile.total_count <= 0:
            continue
        profiles.append(
            {
                "tool": tool,
                "success_rate": profile.success_rate,
                "total_count": profile.total_count,
                "feedback_score": profile.feedback_score,
            }
        )
    return profiles


def _merge_confidence(base: float, record: dict[str, Any]) -> float:
    tools = _selected_tools(record)
    rates: list[float] = []
    for tool in tools[:8]:
        profile = FeedbackStore.command_feedback_profile(command_id=tool)
        if profile.total_count > 0 and profile.success_rate > 0:
            rates.append(float(profile.success_rate))
    feedback_boost = (sum(rates) / len(rates) - 0.5) * 0.16 if rates else 0.0
    ok_count = sum(1 for item in _observations(record) if item.get("ok"))
    evidence_boost = min(ok_count, 4) * 0.025
    return max(0.25, min(float(base or 0.55) + feedback_boost + evidence_boost, 0.95))


def _selected_tools(record: dict[str, Any]) -> list[str]:
    return _dedupe_texts(record.get("selected_tools"), limit=24)


def _observations(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("observations")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _completed_task_texts(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    completed = record.get("completed_tasks")
    if isinstance(completed, list):
        for item in completed:
            if isinstance(item, dict):
                values.append(str(item.get("text") or item.get("goal") or ""))
            else:
                values.append(str(item or ""))
    ledger = record.get("task_ledger")
    tasks = ledger.get("tasks") if isinstance(ledger, dict) else None
    if isinstance(tasks, list):
        for item in tasks:
            if isinstance(item, dict) and item.get("status") in {
                "completed",
                "unsupported",
            }:
                values.append(str(item.get("goal") or item.get("text") or ""))
    return _dedupe_texts(values, limit=16)


def _dedupe_texts(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list | tuple | set):
        items = list(value)
    else:
        items = []
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = normalize_message_text(str(item or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text[:500])
        if len(result) >= limit:
            break
    return result


def _title_from_input(input_message: str, tools: list[str]) -> str:
    text = normalize_message_text(input_message)
    if text:
        return (text[:48] + "...") if len(text) > 48 else text
    if tools:
        return "使用 " + ", ".join(tools[:3]) + " 的任务流程"
    return "超级用户任务流程"


def _skill_learning_timeout() -> float:
    try:
        configured = float(get_config_value("INTENT_TIMEOUT", 20) or 20)
    except (TypeError, ValueError):
        configured = 20.0
    return max(6.0, min(configured, 18.0))


def _clip_jsonable(value: Any, *, limit: int) -> Any:
    if isinstance(value, str):
        return _clip(value, limit=limit)
    if isinstance(value, dict):
        return {
            str(key): _clip_jsonable(item, limit=max(120, limit // 2))
            for key, item in list(value.items())[:16]
        }
    if isinstance(value, list):
        return [_clip_jsonable(item, limit=max(120, limit // 2)) for item in value[:16]]
    return value


def _clip(value: Any, *, limit: int = _MAX_RECORD_TEXT) -> str:
    text = normalize_message_text(str(value or ""))
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


_SKILL_EXTRACT_INSTRUCTION = """
你是 ChatInter 超级用户 Agent 的技能提炼器。你的任务是从一次成功轨迹中提炼可复用技能。

提炼规则：
- 只在轨迹展示了可复用的任务模式、工具序列或注意事项时 should_store=true。
- 不要保存一次性闲聊、没有真实工具证据、失败/取消/超时的流程。
- pattern 描述“什么类型的用户任务适用”，不要复制完整原文。
- steps 写可执行流程，强调先检查、再修改、再验证。
- tools 只填轨迹中真实调用过的工具名。
- cautions 写风险、验收、不要做的事情。
- 技能只是下次任务的提示，不是强制路由规则。

只返回 JSON：
{
  "should_store": true,
  "title": "",
  "pattern": "",
  "summary": "",
  "steps": [],
  "tools": [],
  "cautions": [],
  "tags": [],
  "examples": [],
  "confidence": 0.55
}
""".strip()


__all__ = [
    "learn_skill_from_trajectory_record",
    "schedule_skill_learning",
]
