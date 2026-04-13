import re
import asyncio
import time
from typing import ClassVar

from .feedback_keys import FEEDBACK_REASON_ROUTE_SUCCESS
from .route_text import contains_any, normalize_message_text

_SESSION_PREF_TTL = 2 * 60 * 60
_SESSION_PREF_KEEP = 48
_SESSION_PREF_PRUNE = 24
_SESSION_PREF_MIN_SCORE = 0.04
_SESSION_REASON_MIN_SCORE = 0.03
_SESSION_FEEDBACK_LOG_KEEP = 64
_SESSION_SLOT_KEYS = ("command_head", "target", "image", "text")
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_RAG_STOPWORDS = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "她们",
    "这",
    "那",
    "这个",
    "那个",
    "最近",
    "现在",
    "一下",
    "一下下",
    "帮我",
    "请",
    "麻烦",
    "什么",
    "怎么",
    "如何",
    "怎样",
    "可以",
    "能不能",
    "群里",
    "群",
    "消息",
    "聊天",
    "说什么",
    "说了什么",
}


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(text or ""):
        lower = token.lower()
        if lower in _RAG_STOPWORDS:
            continue
        tokens.append(lower)
        if any("\u4e00" <= char <= "\u9fff" for char in lower):
            chars = [char for char in lower if "\u4e00" <= char <= "\u9fff"]
            if len(lower) <= 3:
                tokens.extend(char for char in chars if char not in _RAG_STOPWORDS)
            if len(lower) >= 2:
                tokens.extend(
                    ngram
                    for ngram in (lower[i : i + 2] for i in range(len(lower) - 1))
                    if ngram not in _RAG_STOPWORDS
                )
            if len(lower) >= 3:
                tokens.extend(
                    ngram
                    for ngram in (lower[i : i + 3] for i in range(len(lower) - 2))
                    if ngram not in _RAG_STOPWORDS
                )
    return tokens


class PluginRAGFeedbackMixin:
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _session_preference: ClassVar[dict[str, dict[str, float]]] = {}
    _session_preference_time: ClassVar[dict[str, float]] = {}
    _session_reason_penalty: ClassVar[dict[str, dict[str, float]]] = {}
    _session_slot_feedback: ClassVar[dict[str, dict[str, dict[str, float]]]] = {}
    _session_feedback_journal: ClassVar[dict[str, list[dict[str, object]]]] = {}

    @classmethod
    def _prune_session_preference(cls, now: float) -> None:
        expired = [
            session_id
            for session_id, updated_at in cls._session_preference_time.items()
            if now - updated_at >= _SESSION_PREF_TTL
        ]
        for session_id in expired:
            cls._session_preference.pop(session_id, None)
            cls._session_preference_time.pop(session_id, None)
            cls._session_reason_penalty.pop(session_id, None)
            cls._session_slot_feedback.pop(session_id, None)
            cls._session_feedback_journal.pop(session_id, None)

    @classmethod
    async def update_session_feedback(
        cls,
        session_id: str | None,
        modules: set[str] | list[str],
        reward: float = 1.0,
        reason: str | None = None,
        slot_feedback: dict[str, float] | None = None,
    ) -> None:
        if not session_id:
            return
        normalized_modules = [module for module in modules if module]
        if not normalized_modules:
            return
        now = time.monotonic()
        async with cls._lock:
            cls._prune_session_preference(now)
            pref = cls._session_preference.get(session_id, {})
            if pref:
                for module in list(pref):
                    pref[module] *= 0.94
                    if abs(pref[module]) < _SESSION_PREF_MIN_SCORE:
                        pref.pop(module, None)
            for module in normalized_modules:
                pref[module] = pref.get(module, 0.0) + reward
            if len(pref) > _SESSION_PREF_KEEP:
                ranked = sorted(pref.items(), key=lambda item: item[1], reverse=True)[
                    :_SESSION_PREF_PRUNE
                ]
                pref = dict(ranked)
            cls._session_preference[session_id] = pref

            reason_penalty = cls._session_reason_penalty.get(session_id, {})
            if reason_penalty:
                for module in list(reason_penalty):
                    reason_penalty[module] *= 0.90
                    if reason_penalty[module] < _SESSION_REASON_MIN_SCORE:
                        reason_penalty.pop(module, None)
            normalized_reason = normalize_message_text(str(reason or "")).lower()
            if (
                normalized_reason
                and normalized_reason != FEEDBACK_REASON_ROUTE_SUCCESS
            ):
                penalty_step = max(abs(min(float(reward), 0.0)), 0.15)
                for module in normalized_modules:
                    reason_penalty[module] = reason_penalty.get(module, 0.0) + penalty_step
            elif normalized_reason == FEEDBACK_REASON_ROUTE_SUCCESS:
                for module in normalized_modules:
                    restored = max(0.0, reason_penalty.get(module, 0.0) - 0.08)
                    if restored < _SESSION_REASON_MIN_SCORE:
                        reason_penalty.pop(module, None)
                    else:
                        reason_penalty[module] = restored
            cls._session_reason_penalty[session_id] = reason_penalty

            slot_store = cls._session_slot_feedback.get(session_id, {})
            normalized_slot_feedback: dict[str, float] = {}
            for slot, value in (slot_feedback or {}).items():
                slot_name = normalize_message_text(str(slot or "")).lower()
                if slot_name not in _SESSION_SLOT_KEYS:
                    continue
                try:
                    numeric = float(value)
                except Exception:
                    continue
                if abs(numeric) <= 1e-6:
                    continue
                normalized_slot_feedback[slot_name] = numeric
            if normalized_slot_feedback:
                for module in normalized_modules:
                    module_slots = slot_store.get(module, {})
                    for slot_name in list(module_slots):
                        module_slots[slot_name] *= 0.92
                        if abs(module_slots[slot_name]) < 0.02:
                            module_slots.pop(slot_name, None)
                    for slot_name, delta in normalized_slot_feedback.items():
                        module_slots[slot_name] = module_slots.get(slot_name, 0.0) + delta
                    if module_slots:
                        slot_store[module] = module_slots
                    else:
                        slot_store.pop(module, None)
            cls._session_slot_feedback[session_id] = slot_store

            journal = cls._session_feedback_journal.get(session_id, [])
            journal.append(
                {
                    "ts": int(time.time()),
                    "modules": normalized_modules,
                    "reward": float(reward),
                    "reason": normalized_reason,
                    "slot_feedback": normalized_slot_feedback,
                }
            )
            if len(journal) > _SESSION_FEEDBACK_LOG_KEEP:
                journal = journal[-_SESSION_FEEDBACK_LOG_KEEP :]
            cls._session_feedback_journal[session_id] = journal
            cls._session_preference_time[session_id] = now
            cls._clear_query_cache()

    @classmethod
    def _session_pref_scores(cls, session_id: str | None) -> dict[str, float]:
        if not session_id:
            return {}
        now = time.monotonic()
        updated_at = cls._session_preference_time.get(session_id)
        if not updated_at:
            return {}
        if now - updated_at >= _SESSION_PREF_TTL:
            cls._session_preference.pop(session_id, None)
            cls._session_preference_time.pop(session_id, None)
            return {}
        pref = cls._session_preference.get(session_id, {})
        if not pref:
            return {}
        max_abs_score = max(abs(score) for score in pref.values()) or 1.0
        return {
            module: max(-1.0, min(score / max_abs_score, 1.0))
            for module, score in pref.items()
        }

    @classmethod
    def _session_reason_penalty_scores(
        cls, session_id: str | None
    ) -> dict[str, float]:
        if not session_id:
            return {}
        reason_penalty = cls._session_reason_penalty.get(session_id, {})
        if not reason_penalty:
            return {}
        return {
            module: max(0.0, min(score, 1.0))
            for module, score in reason_penalty.items()
        }

    @classmethod
    def _session_slot_scores(
        cls,
        session_id: str | None,
        query: str,
        context_text: str,
    ) -> dict[str, float]:
        if not session_id:
            return {}
        now = time.monotonic()
        updated_at = cls._session_preference_time.get(session_id)
        if not updated_at or now - updated_at >= _SESSION_PREF_TTL:
            return {}
        slot_store = cls._session_slot_feedback.get(session_id, {})
        if not slot_store:
            return {}
        merged_query = normalize_message_text(f"{query} {context_text}")
        active_slot_weights = {"command_head": 1.0}
        if (
            "[@" in merged_query
            or contains_any(merged_query, ("给", "帮", "让", "他", "她", "ta", "@"))
        ):
            active_slot_weights["target"] = 1.0
        if "[image" in merged_query or contains_any(
            merged_query, ("图", "图片", "头像", "表情")
        ):
            active_slot_weights["image"] = 1.0
        if len(_tokenize(merged_query)) >= 3:
            active_slot_weights["text"] = 0.6

        raw_scores: dict[str, float] = {}
        for module, module_slots in slot_store.items():
            score = 0.0
            for slot_name, weight in active_slot_weights.items():
                score += float(module_slots.get(slot_name, 0.0)) * weight
            if score:
                raw_scores[module] = score
        if not raw_scores:
            return {}
        max_abs = max(abs(score) for score in raw_scores.values()) or 1.0
        return {
            module: max(-1.0, min(score / max_abs, 1.0))
            for module, score in raw_scores.items()
        }
