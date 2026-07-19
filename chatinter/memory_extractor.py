from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from .chat_memory_store import MemoryCandidate
from .llm_compat import AI
from .route_text import normalize_message_text

_EXTRACT_EVERY_N_TURNS = 8
_EXTRACT_HISTORY_TURNS = 8
_EXTRACT_TIMEOUT_SECONDS = 18.0
_EXTRACT_HISTORY_MAX_CHARS = 2400
_IMAGE_CONTEXT_PATTERN = re.compile(
    r"<image_context>.*?</image_context>",
    re.IGNORECASE | re.DOTALL,
)
_MEDIA_PLACEHOLDER_PATTERN = re.compile(r"\[image(?:#\d+)?\]", re.IGNORECASE)
_ALLOWED_MEMORY_TYPES = {
    "nickname",
    "preference",
    "relationship",
    "person_profile_summary",
    "thread_digest",
}
_turn_counts: dict[str, int] = defaultdict(int)
_tasks: set[asyncio.Task[None]] = set()
_inflight_keys: set[str] = set()


class _ExtractedMemory(BaseModel):
    memory_type: Literal[
        "nickname",
        "preference",
        "relationship",
        "person_profile_summary",
        "thread_digest",
    ] = "person_profile_summary"
    key: str = ""
    value: str = ""
    confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    supersedes: bool = False


class _MemoryExtractionResult(BaseModel):
    memories: list[_ExtractedMemory] = Field(default_factory=list)


@dataclass(frozen=True)
class MemoryExtractionRequest:
    session_id: str
    user_id: str
    group_id: str | None
    message_text: str
    source_dialog_id: int | None = None
    thread_id: str | None = None
    topic_key: str = ""
    participants: tuple[str, ...] = ()


async def extract_stable_memory_candidates(
    request: MemoryExtractionRequest,
) -> list[MemoryCandidate]:
    from .config import build_agent_generation_config, get_agent_model

    history_text = await _load_recent_plain_history(
        request,
        limit=_EXTRACT_HISTORY_TURNS,
    )
    if not history_text:
        return []
    allow_thread_digest = bool(request.group_id and request.thread_id)
    try:
        result = await AI(
            session_id=f"chatinter-memory:{request.session_id or request.user_id}"
        ).generate_structured(
            json.dumps(
                {
                    "target_user_id": request.user_id,
                    "history": history_text,
                    "allow_thread_digest": allow_thread_digest,
                },
                ensure_ascii=False,
            ),
            _MemoryExtractionResult,
            model=get_agent_model("chat"),
            instruction=_EXTRACTOR_INSTRUCTION,
            timeout=_EXTRACT_TIMEOUT_SECONDS,
            config=build_agent_generation_config("chat"),
            max_validation_retries=0,
        )
    except Exception:
        return []
    return _normalize_extracted_memories(
        result,
        allow_thread_digest=allow_thread_digest,
    )


async def _load_recent_plain_history(
    request: MemoryExtractionRequest,
    *,
    limit: int,
) -> str:
    rows = []
    try:
        from .models.chat_history import ChatInterChatHistory

        rows = await ChatInterChatHistory.get_recent_dialogs(
            request.session_id,
            limit=max(int(limit or 0), 1),
            user_id=request.user_id,
        )
    except Exception:
        rows = []

    lines: list[str] = []
    target_user_id = normalize_message_text(request.user_id)
    for row in rows:
        row_user_id = normalize_message_text(getattr(row, "user_id", ""))
        if not target_user_id or row_user_id != target_user_id:
            continue
        user_text = _plain_chat_text(getattr(row, "user_message", ""))
        if user_text:
            timestamp = _history_timestamp(getattr(row, "create_time", None))
            lines.append(f"[{timestamp}] {target_user_id}: {user_text}")

    if not lines:
        user_text = _plain_chat_text(request.message_text)
        if user_text:
            lines.append(
                f"[{_history_timestamp(None)}] {target_user_id}: {user_text}"
            )

    text = "\n".join(lines)
    return text[-_EXTRACT_HISTORY_MAX_CHARS:]


def _history_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def _plain_chat_text(value: object) -> str:
    text = normalize_message_text(str(value or ""))
    if not text:
        return ""
    text = _IMAGE_CONTEXT_PATTERN.sub(" ", text)
    text = _MEDIA_PLACEHOLDER_PATTERN.sub(" ", text)
    text = re.sub(r"<[^>]{1,32}>", " ", text)
    return normalize_message_text(text)


def _normalize_extracted_memories(
    result: _MemoryExtractionResult,
    *,
    allow_thread_digest: bool = False,
) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for item in result.memories[:6]:
        memory_type = normalize_message_text(item.memory_type)
        key = normalize_message_text(item.key)[:64] or memory_type
        value = normalize_message_text(item.value)[:80]
        if memory_type not in _ALLOWED_MEMORY_TYPES or not value:
            continue
        if memory_type == "thread_digest" and not allow_thread_digest:
            continue
        dedupe_key = (memory_type, key, value)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(
            MemoryCandidate(
                memory_type=memory_type,
                key=key,
                value=value,
                confidence=max(0.0, min(float(item.confidence or 0.0), 1.0)),
                supersedes=bool(item.supersedes),
            )
        )
    return candidates


def schedule_memory_extraction(
    request: MemoryExtractionRequest,
    writer: Callable[[list[MemoryCandidate]], Awaitable[int]],
) -> bool:
    key = _request_key(request)
    if not key or key in _inflight_keys:
        return False
    if not _should_extract(request, key=key):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    _inflight_keys.add(key)
    task = loop.create_task(_run_extraction(request, writer, key=key))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True


def _request_key(request: MemoryExtractionRequest) -> str:
    session_id = normalize_message_text(request.session_id)
    user_id = normalize_message_text(request.user_id)
    return f"{session_id or user_id}:{user_id}" if user_id else session_id


def _should_extract(
    request: MemoryExtractionRequest,
    *,
    key: str | None = None,
) -> bool:
    text = normalize_message_text(request.message_text)
    if not text or text.startswith("<image_context>"):
        return False
    key = key or _request_key(request)
    if not key:
        return False
    if request.source_dialog_id is not None and request.source_dialog_id > 0:
        return request.source_dialog_id % _EXTRACT_EVERY_N_TURNS == 0
    _turn_counts[key] += 1
    return _turn_counts[key] % _EXTRACT_EVERY_N_TURNS == 0


async def drain_memory_extraction_tasks(timeout: float = 5.0) -> None:
    _tasks.difference_update(task for task in _tasks if task.done())
    pending = [task for task in _tasks if not task.done()]
    if not pending:
        return
    done, still_pending = await asyncio.wait(pending, timeout=max(timeout, 0.0))
    for task in still_pending:
        task.cancel()
    if still_pending:
        await asyncio.gather(*still_pending, return_exceptions=True)
    for task in done:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    _tasks.difference_update(pending)


async def _run_extraction(
    request: MemoryExtractionRequest,
    writer: Callable[[list[MemoryCandidate]], Awaitable[int]],
    *,
    key: str,
) -> None:
    try:
        candidates = await extract_stable_memory_candidates(request)
        if candidates:
            await writer(candidates)
    except Exception:
        return
    finally:
        _inflight_keys.discard(key)


_EXTRACTOR_INSTRUCTION = """
你是聊天长期记忆抽取器。输入中的 target_user_id 是唯一目标主体；
只能抽取该用户明确表达的原子事实，不得把其他用户或助手的信息归到目标用户。
事实必须可跨会话复用，并且离开当前上下文也能独立理解。

只保留：
- nickname: 用户明确要求的稳定称呼/昵称。
- preference: 用户长期偏好/厌恶。
- relationship: 用户与重要人物/宠物/群友的稳定关系。
- person_profile_summary: 用户身份、职业、长期习惯、长期兴趣。
- thread_digest: 仅当 allow_thread_digest=true 时，抽取群聊话题中的稳定决定/约定。

忽略：
- 寒暄、临时情绪、一次性任务、当前对话过程、技术排错过程。
- 图片、图片描述、<image_context>、[image]、视觉内容。
- 推断、猜测、泛化、助手总结、未由用户直接表达的内容。
- 不确定、不能独立理解、离开上下文就含糊的内容。

要求：
- 大多数对话应返回空列表。
- value 必须是简短、独立、声明式中文事实；必要时补足主体或对象。
- 只有最新用户原话明确纠正或替换了同一事实时，supersedes 才为 true。
- 普通新增事实的 supersedes 必须为 false。
- 更正项沿用被更正事实的稳定 key，使旧值可以被准确替换。
- 不要编造，不要总结助手说过的话。
""".strip()


__all__ = [
    "MemoryExtractionRequest",
    "extract_stable_memory_candidates",
    "schedule_memory_extraction",
]
