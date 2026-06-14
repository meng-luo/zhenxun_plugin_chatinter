"""Lightweight final reply quality checks for chat turns.

The judge is deliberately small and runtime-oriented.  It does not replace the
model or add plugin-specific routing; it only catches generic failure modes
before a message is sent.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from zhenxun.services import logger
from zhenxun.services.llm import AI

from .chat_dialogue_planner import DialogueState
from .config import build_reasoning_generation_config, get_config_value, get_model_name
from .persistence import append_jsonl, state_path, utc_now_iso
from .route_text import normalize_message_text
from .trajectory_eval import record_response_quality_eval

QualityAction = Literal["ok", "revise", "block"]
ShadowQualityVerdict = Literal["ok", "minor_issue", "bad", "unsafe"]

_STIFF_PHRASES = (
    "尊敬的用户",
    "您好，关于您的问题",
    "作为一个人工智能",
    "我无法提供情感",
)
_QUESTION_MARKERS = ("吗", "么", "嘛", "？", "?", "怎么", "为什么", "是什么", "是啥")
_GENERIC_FILLER_REPLIES = frozenset(
    {
        "好的",
        "嗯嗯",
        "可以",
        "收到",
        "明白了",
        "我知道了",
        "了解",
    }
)
_REFUSAL_MARKERS = (
    "我不能",
    "无法帮助",
    "无法提供",
    "不能协助",
    "不能回答",
    "不方便回答",
)
_UNFOUNDED_CERTAINTY = (
    "一定",
    "绝对",
    "肯定是",
    "毫无疑问",
    "百分百",
)
_SOURCE_REQUEST_MARKERS = (
    "来源",
    "依据",
    "资料",
    "引用",
    "链接",
    "文档",
    "出处",
)
_HALLUCINATION_MARKERS = (
    "根据上面的图片",
    "如图所示",
    "我已经执行",
    "已经为你完成",
    "我查到了",
)
_MAX_SHADOW_TASKS = 16
_SHADOW_TASKS: set[asyncio.Task[None]] = set()
_SHADOW_LOG_PATH = state_path("quality", "shadow_judge.jsonl")


@dataclass(frozen=True)
class ResponseQualityResult:
    action: QualityAction
    reason: str = ""
    revised_text: str = ""
    instruction: str = ""
    severity: str = "info"
    rule_hits: tuple[str, ...] = ()
    shadow_verdict: str = ""
    shadow_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.action == "ok"


class ResponseQualityJudge:
    @staticmethod
    def judge_chat_only(
        *,
        final_text: str,
        original_message: str,
        dialogue_state: DialogueState | None,
    ) -> ResponseQualityResult:
        """Conversational quality check without tool/execution dependencies."""

        reply = normalize_message_text(final_text)
        message = normalize_message_text(original_message)
        if not reply:
            return ResponseQualityResult(
                action="block",
                reason="empty_final_reply",
                revised_text="我暂时没想好怎么回答你。",
                severity="error",
                rule_hits=("empty_final_reply",),
            )
        rule_hits = _quality_rule_hits(
            message=message,
            reply=reply,
            state=dialogue_state,
        )
        if "unsafe_refusal_needed" in rule_hits:
            return ResponseQualityResult(
                action="block",
                reason="unsafe_refusal_needed",
                revised_text="这个我不能帮你完成，但可以换个安全的方向一起看。",
                severity="error",
                rule_hits=tuple(rule_hits),
            )
        if _looks_stiff(reply, dialogue_state):
            return ResponseQualityResult(
                action="revise",
                reason="too_stiff_for_dialogue_state",
                revised_text=_soften_reply(reply),
                severity="warn",
                rule_hits=(*rule_hits, "too_stiff_for_dialogue_state"),
            )
        if _looks_off_topic(message, reply, dialogue_state):
            return ResponseQualityResult(
                action="revise",
                reason="possible_off_topic",
                instruction=(
                    "Reply should answer the user's current message directly; "
                    "avoid generic filler."
                ),
                severity="warn",
                rule_hits=(*rule_hits, "possible_off_topic"),
            )
        if "too_long_for_short_state" in rule_hits:
            return ResponseQualityResult(
                action="revise",
                reason="too_long_for_short_state",
                instruction="Compress the reply; keep conclusion first.",
                severity="warn",
                rule_hits=tuple(rule_hits),
            )
        if rule_hits:
            return ResponseQualityResult(
                action="ok",
                reason=rule_hits[0],
                severity="info",
                rule_hits=tuple(rule_hits),
            )
        return ResponseQualityResult(action="ok")


class ShadowQualityResult(BaseModel):
    verdict: ShadowQualityVerdict = Field(default="ok")
    reason: str = ""
    should_revise_next_time: bool = False
    tags: list[str] = Field(default_factory=list)


def schedule_shadow_quality_judge(
    *,
    final_text: str,
    original_message: str,
    scenario: str,
    session_key: str,
    trace_id: str = "",
    quality_result: ResponseQualityResult | None = None,
) -> None:
    if not _shadow_enabled():
        return
    if len(_SHADOW_TASKS) >= _MAX_SHADOW_TASKS:
        return
    if not _should_sample_shadow(
        final_text=final_text,
        original_message=original_message,
        quality_result=quality_result,
    ):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(
        _shadow_quality_judge_safely(
            final_text=final_text,
            original_message=original_message,
            scenario=scenario,
            session_key=session_key,
            trace_id=trace_id,
            quality_result=quality_result,
        )
    )
    _SHADOW_TASKS.add(task)
    task.add_done_callback(_SHADOW_TASKS.discard)


def result_to_record(result: ResponseQualityResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    payload = asdict(result)
    payload["ok"] = result.ok
    return payload


def _looks_stiff(reply: str, state: DialogueState | None) -> bool:
    if not any(phrase in reply for phrase in _STIFF_PHRASES):
        return False
    if state is None:
        return True
    return state.tone in {"casual", "warm", "playful", "empathetic"}


def _soften_reply(reply: str) -> str:
    softened = reply
    for phrase in _STIFF_PHRASES:
        softened = softened.replace(phrase, "")
    return normalize_message_text(softened).lstrip("，,。.!！？?") or reply


def _looks_off_topic(
    message: str,
    reply: str,
    state: DialogueState | None,
) -> bool:
    if not message or not reply:
        return False
    if state is not None and state.dialogue_purpose in {"chat", "support"}:
        return False
    if len(reply) <= 8 and any(marker in message for marker in _QUESTION_MARKERS):
        return True
    if reply in _GENERIC_FILLER_REPLIES and len(message) >= 12:
        return True
    return False


def _quality_rule_hits(
    *,
    message: str,
    reply: str,
    state: DialogueState | None,
) -> list[str]:
    hits: list[str] = []
    if len(reply) <= 2 and len(message) >= 8:
        hits.append("too_short_for_context")
    if _generic_filler(message, reply):
        hits.append("generic_filler")
    if _unhelpful_refusal(message, reply):
        hits.append("unhelpful_refusal")
    if _unsafe_request(message) and not _has_refusal(reply):
        hits.append("unsafe_refusal_needed")
    if _unsupported_grounding_claim(message, reply):
        hits.append("unsupported_grounding_claim")
    if _overconfident_without_basis(message, reply):
        hits.append("overconfident_without_basis")
    if _too_long_for_state(reply, state):
        hits.append("too_long_for_short_state")
    return hits


def _generic_filler(message: str, reply: str) -> bool:
    return reply in _GENERIC_FILLER_REPLIES and len(message) >= 10


def _unhelpful_refusal(message: str, reply: str) -> bool:
    if not any(marker in reply for marker in _REFUSAL_MARKERS):
        return False
    return "为什么" in message or "怎么" in message or len(message) >= 10


def _unsafe_request(message: str) -> bool:
    unsafe_markers = (
        "盗号",
        "破解",
        "绕过验证",
        "窃取",
        "木马",
        "炸群",
        "刷屏攻击",
    )
    return any(marker in message for marker in unsafe_markers)


def _has_refusal(reply: str) -> bool:
    return any(marker in reply for marker in _REFUSAL_MARKERS) or "不能" in reply


def _unsupported_grounding_claim(message: str, reply: str) -> bool:
    if any(marker in message for marker in _SOURCE_REQUEST_MARKERS):
        return False
    return any(marker in reply for marker in _HALLUCINATION_MARKERS)


def _overconfident_without_basis(message: str, reply: str) -> bool:
    if any(marker in message for marker in _SOURCE_REQUEST_MARKERS):
        return False
    if "不确定" in reply or "可能" in reply or "看起来" in reply:
        return False
    return any(marker in reply for marker in _UNFOUNDED_CERTAINTY)


def _too_long_for_state(reply: str, state: DialogueState | None) -> bool:
    if state is None or state.response_length != "short":
        return False
    return len(reply) > 360 and state.dialogue_purpose in {"chat", "support"}


def _shadow_enabled() -> bool:
    return _shadow_sample_rate() > 0.0


def _shadow_sample_rate() -> float:
    try:
        value = float(get_config_value("QUALITY_SHADOW_SAMPLE_RATE", 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(value, 1.0))


def _should_sample_shadow(
    *,
    final_text: str,
    original_message: str,
    quality_result: ResponseQualityResult | None,
) -> bool:
    if quality_result is not None and quality_result.rule_hits:
        return True
    sample_rate = _shadow_sample_rate()
    if sample_rate <= 0:
        return False
    key = f"{original_message}\n---\n{final_text}"
    digest = hashlib.blake2b(key.encode("utf-8", "ignore"), digest_size=8).digest()
    stable = int.from_bytes(digest, "big") / float(2**64 - 1)
    return stable < sample_rate


async def _shadow_quality_judge_safely(
    *,
    final_text: str,
    original_message: str,
    scenario: str,
    session_key: str,
    trace_id: str,
    quality_result: ResponseQualityResult | None,
) -> None:
    started = asyncio.get_running_loop().time()
    try:
        result = await _run_shadow_quality_judge(
            final_text=final_text,
            original_message=original_message,
            scenario=scenario,
        )
        latency_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        append_jsonl(
            _SHADOW_LOG_PATH,
            {
                "schema_version": "chatinter.quality_shadow.v1",
                "created_at": utc_now_iso(),
                "trace_id": trace_id,
                "session_key": session_key,
                "scenario": scenario,
                "verdict": result.verdict,
                "reason": result.reason,
                "should_revise_next_time": result.should_revise_next_time,
                "tags": result.tags[:12],
                "local_quality": result_to_record(quality_result),
                "latency_ms": latency_ms,
            },
        )
        await _record_shadow_eval_projection(
            result=result,
            final_text=final_text,
            original_message=original_message,
            scenario=scenario,
            session_key=session_key,
            trace_id=trace_id,
        )
    except Exception as exc:
        logger.debug(f"[ChatInter] shadow quality judge skipped: {exc}")


async def _run_shadow_quality_judge(
    *,
    final_text: str,
    original_message: str,
    scenario: str,
) -> ShadowQualityResult:
    prompt = {
        "scenario": normalize_message_text(scenario),
        "user_message": normalize_message_text(original_message)[:1200],
        "assistant_reply": normalize_message_text(final_text)[:1800],
    }
    return await AI(session_id="chatinter-quality-shadow").generate_structured(
        json.dumps(prompt, ensure_ascii=False),
        ShadowQualityResult,
        model=get_model_name(),
        config=build_reasoning_generation_config(),
        instruction=_SHADOW_QUALITY_INSTRUCTION,
        timeout=_shadow_timeout(),
        max_validation_retries=0,
        auto_thinking=False,
    )


def _shadow_timeout() -> float:
    try:
        configured = float(get_config_value("INTENT_TIMEOUT", 20) or 20)
    except (TypeError, ValueError):
        configured = 20.0
    return max(4.0, min(configured, 12.0))


async def _record_shadow_eval_projection(
    *,
    result: ShadowQualityResult,
    final_text: str,
    original_message: str,
    scenario: str,
    session_key: str,
    trace_id: str,
) -> None:
    try:
        await asyncio.to_thread(
            record_response_quality_eval,
            quality_result=None,
            final_text=final_text,
            original_message=original_message,
            scenario=scenario,
            session_key=session_key,
            trace_id=trace_id,
            shadow_verdict=result.verdict,
            shadow_reason=result.reason,
        )
    except Exception:
        return


_SHADOW_QUALITY_INSTRUCTION = """
你是 ChatInter 的回复质量抽检器。只做事后评估,不要改写回复。

判定维度:
- 是否直接回应用户当前消息。
- 是否有空泛敷衍、客服腔、过度拒绝。
- 是否声称做了未观察到的事情。
- 是否在没有依据时过度确定。
- 是否存在明显不安全回复。

只返回 JSON:
{
  "verdict": "ok",
  "reason": "",
  "should_revise_next_time": false,
  "tags": []
}
""".strip()


__all__ = [
    "ResponseQualityJudge",
    "ResponseQualityResult",
    "ShadowQualityResult",
    "result_to_record",
    "schedule_shadow_quality_judge",
]
