from __future__ import annotations

from dataclasses import dataclass
import re

from .chat_memory_store import ChatMemoryStore, extract_memory_candidates
from .group_memory_digest import build_group_memory_digest
from .memory_policy import MemoryPolicyDecision, decide_memory_policy
from .person_registry import upsert_person_alias
from .reflection_observer import record_reflection_observation
from .route_text import normalize_message_text

_AT_ALIAS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?P<alias>[A-Za-z0-9\u4e00-\u9fff]{1,24}?)"
        r"[（(]\s*\[@(?P<user_id>[^\]\s]+)\]\s*[）)]"
    ),
    re.compile(
        r"\[@(?P<user_id>[^\]\s]+)\]\s*"
        r"(?:叫|是|就是|昵称是|群昵称是)"
        r"(?P<alias>[A-Za-z0-9\u4e00-\u9fff]{1,24})"
    ),
)
_SELF_ALIAS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:以后)?(?:叫我|喊我|称呼我|我叫|你忘了我是|还记得我是)"
        r"\s*(?P<alias>[A-Za-z0-9\u4e00-\u9fff]{1,24}?)(?:了嘛|了吗|了么|吗|嘛|么|吧|呀|啊|$)"
    ),
    re.compile(
        r"(?:^|[，,。.!！？?])我是"
        r"\s*(?P<alias>[A-Za-z0-9\u4e00-\u9fff]{1,24}?)(?:了嘛|了吗|了么|吗|嘛|么|吧|呀|啊|$)"
    ),
)
_QUESTION_SUFFIXES = ("吗", "么", "嘛", "？", "?")
_SELF_ALIAS_NEGATIVE_PREFIXES = ("不是", "并不是", "应该不是", "不是说")
_SELF_ALIAS_UNSAFE_SUFFIXES = (
    "第一次",
    "今天",
    "刚",
    "听说",
    "看到",
    "发现",
    "觉得",
    "来",
    "去",
    "想",
    "要",
    "在",
    "从",
)


@dataclass(frozen=True)
class MemoryWriteContext:
    session_id: str
    user_id: str
    group_id: str | None
    message_text: str
    response_text: str = ""
    source_dialog_id: int | None = None
    thread_id: str | None = None
    topic_key: str = ""
    participants: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryWriteResult:
    written: int
    policy: MemoryPolicyDecision
    candidate_count: int = 0


@dataclass(frozen=True)
class PersonAliasCandidate:
    user_id: str
    alias: str
    source: str
    confidence: float


class MemoryWriter:
    @classmethod
    async def write_from_dialog(cls, context: MemoryWriteContext) -> MemoryWriteResult:
        message_text = normalize_message_text(context.message_text)
        response_text = normalize_message_text(context.response_text)
        candidates = extract_memory_candidates(message_text)
        person_aliases = extract_person_alias_candidates(
            message_text=message_text,
            current_user_id=context.user_id,
        )
        policy = decide_memory_policy(
            message_text=message_text,
            response_text=response_text,
            group_id=context.group_id,
            thread_id=context.thread_id,
            memory_candidate_count=len(candidates) + len(person_aliases),
        )
        if not policy.should_write:
            _record_reflection(
                context,
                action="memory_skip",
                policy=policy,
                written=0,
                candidate_count=len(candidates) + len(person_aliases),
            )
            return MemoryWriteResult(
                written=0,
                policy=policy,
                candidate_count=len(candidates) + len(person_aliases),
            )

        written = await ChatMemoryStore.record_candidates(
            session_id=context.session_id,
            user_id=context.user_id,
            group_id=context.group_id,
            candidates=candidates,
            source_dialog_id=context.source_dialog_id,
            source_message=message_text,
            scope="user",
            thread_id=context.thread_id,
            topic_key=context.topic_key,
            participants=context.participants,
        )
        person_alias_written = await _record_person_aliases(
            context=context,
            candidates=person_aliases,
        )
        written += person_alias_written
        action = "memory_write"
        if policy.action == "digest":
            digest = build_group_memory_digest(
                session_id=context.session_id,
                user_id=context.user_id,
                group_id=context.group_id,
                thread_id=context.thread_id,
                topic_key=context.topic_key,
                participants=context.participants,
                message_text=message_text,
                response_text=response_text,
            )
            if digest is not None:
                written += await ChatMemoryStore.record_group_digest(digest)
                action = "memory_digest"

        _record_reflection(
            context,
            action=action,
            policy=policy,
            written=written,
            candidate_count=len(candidates) + len(person_aliases),
        )
        return MemoryWriteResult(
            written=written,
            policy=policy,
            candidate_count=len(candidates) + len(person_aliases),
        )


def extract_person_alias_candidates(
    *,
    message_text: str,
    current_user_id: str,
) -> list[PersonAliasCandidate]:
    text = normalize_message_text(message_text)
    if not text:
        return []
    candidates: list[PersonAliasCandidate] = []
    seen: set[tuple[str, str]] = set()

    def add(user_id: str, alias: str, source: str, confidence: float) -> None:
        normalized_user_id = str(user_id or "").strip()
        normalized_alias = _clean_alias(alias)
        if not normalized_user_id or not normalized_alias:
            return
        key = (normalized_user_id, normalized_alias)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            PersonAliasCandidate(
                user_id=normalized_user_id,
                alias=normalized_alias,
                source=source,
                confidence=confidence,
            )
        )

    for pattern in _AT_ALIAS_PATTERNS:
        for match in pattern.finditer(text):
            add(
                match.group("user_id"),
                match.group("alias"),
                "explicit_at_alias",
                0.88,
            )

    if current_user_id:
        for pattern in _SELF_ALIAS_PATTERNS:
            for match in pattern.finditer(text):
                if not _is_safe_self_alias_match(text, match):
                    continue
                add(current_user_id, match.group("alias"), "self_alias", 0.82)

    return candidates[:6]


async def _record_person_aliases(
    *,
    context: MemoryWriteContext,
    candidates: list[PersonAliasCandidate],
) -> int:
    written = 0
    for candidate in candidates:
        if await upsert_person_alias(
            user_id=candidate.user_id,
            group_id=context.group_id,
            alias=candidate.alias,
            source=candidate.source,
            confidence=candidate.confidence,
        ):
            written += 1
    return written


def _clean_alias(value: str) -> str:
    alias = normalize_message_text(value).strip(" 　，,。.!！？?（）()[]【】")
    alias = re.sub(r"^(?:以后看到|看到|以后叫|叫|喊|称呼)", "", alias).strip()
    if not alias:
        return ""
    if alias in {"我", "自己", "本人", "这个", "那个", "他", "她", "它", "ta"}:
        return ""
    return alias[:24]


def _is_safe_self_alias_match(text: str, match: re.Match[str]) -> bool:
    start = max(int(match.start()), 0)
    prefix = normalize_message_text(text[max(0, start - 6) : start])
    if any(prefix.endswith(item) for item in _SELF_ALIAS_NEGATIVE_PREFIXES):
        return False

    alias = _clean_alias(match.group("alias") or "")
    if not alias:
        return False
    if len(alias) > 12:
        return False
    return not any(alias.startswith(item) for item in _SELF_ALIAS_UNSAFE_SUFFIXES)


def _record_reflection(
    context: MemoryWriteContext,
    *,
    action: str,
    policy: MemoryPolicyDecision,
    written: int,
    candidate_count: int,
) -> None:
    record_reflection_observation(
        action=action,
        session_id=context.session_id,
        user_id=context.user_id,
        group_id=context.group_id,
        thread_id=context.thread_id,
        reason=policy.reason,
        written=written,
        candidate_count=candidate_count,
        message_text=context.message_text,
    )


__all__ = [
    "MemoryWriteContext",
    "MemoryWriteResult",
    "MemoryWriter",
    "PersonAliasCandidate",
    "extract_person_alias_candidates",
]
