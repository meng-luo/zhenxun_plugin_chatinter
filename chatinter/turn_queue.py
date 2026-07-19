"""Turn queue for ChatInter group/private conversations.

The queue merges short same-user bursts into one stable turn and serializes each
conversation so multiple ChatInter runs do not compete with each other.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import re
import time
from typing import Any
import unicodedata
import uuid

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.services.log import logger

from .event_runtime import get_nickname, is_already_handled, mark_as_handled
from .event_signals import set_event_signal
from .group_turn_context import record_group_turn_message
from .route_text import normalize_message_text
from .session_identity import conversation_session_key

COLLECT_WINDOW_SECONDS = 1.2
MAX_PENDING_TURNS_PER_GROUP = 8
MAX_MESSAGES_PER_TURN = 5
MAX_TURN_TEXT_CHARS = 1200
_PRIORITY_COLLECT_SECONDS = 0.25
_COMMAND_COLLECT_SECONDS = 0.45
_IDLE_WORKER_EXIT_SECONDS = 30.0
_DEDUPE_CACHE_SIZE = 2048
_DEDUPE_CACHE_AGE_SECONDS = 180.0
_MEDIA_PLACEHOLDER_PATTERN = re.compile(
    r"\[image(?:#[0-9]+|:[^\]]*)?\]",
    re.IGNORECASE,
)

TurnProcessor = Callable[..., Awaitable[None]]


@dataclass
class QueuedMessage:
    bot: Bot
    event: Event
    session: Uninfo
    raw_message: str
    message: Any | None
    route_modules: set[str] | None
    cached_plain_text: str | None
    user_id: str
    group_id: str | None
    conversation_key: str
    message_id: str
    priority: int
    must_keep: bool
    source: str = "message"
    dedupe_key: str = ""
    created_at: float = field(default_factory=time.monotonic)
    pending_human_update: bool = False
    reply_to_message_id: str = ""
    generation: int = 0


@dataclass
class QueuedTurn:
    conversation_key: str
    user_id: str
    group_id: str | None
    messages: list[QueuedMessage]
    pending_human_updates: list[str] = field(default_factory=list)

    @property
    def latest(self) -> QueuedMessage:
        return self.messages[-1]

    @property
    def priority(self) -> int:
        return max((item.priority for item in self.messages), default=0)

    @property
    def must_keep(self) -> bool:
        return any(item.must_keep for item in self.messages)

    @property
    def raw_message(self) -> str:
        return _compose_turn_text([item.raw_message for item in self.messages])

    @property
    def route_modules(self) -> set[str] | None:
        modules: set[str] = set()
        for item in self.messages:
            if item.route_modules:
                modules.update(item.route_modules)
        return modules or None


@dataclass
class _ConversationState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    current_by_user: dict[str, QueuedTurn] = field(default_factory=dict)
    pending: deque[QueuedTurn] = field(default_factory=deque)
    flush_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    worker_task: asyncio.Task | None = None
    active_turn: QueuedTurn | None = None
    active_updates: list[str] = field(default_factory=list)
    active_task: asyncio.Task[None] | None = None
    generation: int = 0


class TurnQueue:
    def __init__(self) -> None:
        self._states: dict[str, _ConversationState] = {}
        self._dedupe_cache: deque[tuple[str, float]] = deque()
        self._dedupe_keys: set[str] = set()

    async def submit(
        self,
        *,
        bot: Bot,
        event: Event,
        session: Uninfo,
        raw_message: str,
        message: Any | None,
        route_modules: set[str] | None,
        cached_plain_text: str | None,
        processor: TurnProcessor,
        priority_override: int | None = None,
        must_keep_override: bool | None = None,
        source: str = "message",
        dedupe_key: str | None = None,
        allow_handled_event: bool = False,
        mark_event_handled: bool = True,
    ) -> bool:
        if not allow_handled_event and is_already_handled(event):
            logger.debug("ChatInter TurnQueue 跳过重复消息")
            return False

        item = _build_queued_message(
            bot=bot,
            event=event,
            session=session,
            raw_message=raw_message,
            message=message,
            route_modules=route_modules,
            cached_plain_text=cached_plain_text,
            priority_override=priority_override,
            must_keep_override=must_keep_override,
            source=source,
            dedupe_key=dedupe_key,
        )
        if not self._remember_dedupe_key(item.dedupe_key):
            logger.debug(
                "ChatInter TurnQueue 跳过重复 continuation: "
                f"source={item.source} key={item.dedupe_key[:64]}"
            )
            return False
        record_group_turn_message(
            group_id=item.group_id,
            user_id=item.user_id,
            nickname=get_nickname(session),
            text=item.raw_message,
            message_id=item.message_id,
        )
        state = self._states.setdefault(item.conversation_key, _ConversationState())
        async with state.lock:
            if _has_dedupe_locked(state, item.dedupe_key):
                logger.debug(
                    "ChatInter TurnQueue 队列内已存在 continuation: "
                    f"source={item.source} key={item.dedupe_key[:64]}"
                )
                return False
            if self._handle_low_priority_locked(state, item, processor):
                if mark_event_handled:
                    mark_as_handled(event)
                return True
            if item.priority <= 0:
                logger.debug(
                    "ChatInter TurnQueue 忽略低优先级闲聊: "
                    f"group={item.group_id or 'private'} user={item.user_id}"
                )
                return False
            if mark_event_handled:
                mark_as_handled(event)
            if item.group_id is None:
                state.generation += 1
                item.generation = state.generation
            if item.group_id is None and state.active_turn is not None:
                self._supersede_private_turn_locked(state, item, processor)
                return True
            if (
                state.active_turn is not None
                and state.active_turn.user_id == item.user_id
                and item.source == "message"
                and _is_followup_for_active_turn(state.active_turn, item)
            ):
                self._record_active_update_locked(state, item)
            self._add_to_current_or_queue_locked(state, item)
            self._schedule_flush_locked(
                state,
                item.conversation_key,
                item.user_id,
                processor,
                delay=_collect_delay(item),
            )
            self._ensure_worker_locked(state, item.conversation_key, processor)
        return True

    def _supersede_private_turn_locked(
        self,
        state: _ConversationState,
        item: QueuedMessage,
        processor: TurnProcessor,
    ) -> None:
        active_task = state.active_task
        if active_task is not None and not active_task.done():
            active_task.cancel()
        for task in state.flush_tasks.values():
            if not task.done():
                task.cancel()
        state.flush_tasks.clear()
        state.current_by_user.clear()
        state.pending.clear()
        state.active_updates.clear()
        state.current_by_user[item.user_id] = QueuedTurn(
            conversation_key=item.conversation_key,
            user_id=item.user_id,
            group_id=None,
            messages=[item],
        )
        self._schedule_flush_locked(
            state,
            item.conversation_key,
            item.user_id,
            processor,
            delay=_collect_delay(item),
        )

    def _remember_dedupe_key(self, key: str) -> bool:
        now = time.monotonic()
        while self._dedupe_cache and (
            len(self._dedupe_cache) > _DEDUPE_CACHE_SIZE
            or now - self._dedupe_cache[0][1] > _DEDUPE_CACHE_AGE_SECONDS
        ):
            old_key, _ = self._dedupe_cache.popleft()
            self._dedupe_keys.discard(old_key)
        if not key:
            return True
        if key in self._dedupe_keys:
            return False
        self._dedupe_keys.add(key)
        self._dedupe_cache.append((key, now))
        return True

    def _handle_low_priority_locked(
        self,
        state: _ConversationState,
        item: QueuedMessage,
        processor: TurnProcessor,
    ) -> bool:
        if item.priority > 0:
            return False
        if state.active_turn is not None and state.active_turn.user_id == item.user_id:


            self._record_active_update_locked(state, item)
            return True
        current = state.current_by_user.get(item.user_id)
        if current is not None and _can_merge(current, item):
            current.messages.append(item)
            self._schedule_flush_locked(
                state,
                item.conversation_key,
                item.user_id,
                processor,
                delay=_collect_delay(item),
            )
            self._ensure_worker_locked(state, item.conversation_key, processor)
            return True
        return False

    def _add_to_current_or_queue_locked(
        self,
        state: _ConversationState,
        item: QueuedMessage,
    ) -> None:
        current = state.current_by_user.get(item.user_id)
        if current is not None and _can_merge(current, item):
            current.messages.append(item)
            return
        if current is not None:
            self._enqueue_turn_locked(state, current)
        state.current_by_user[item.user_id] = QueuedTurn(
            conversation_key=item.conversation_key,
            user_id=item.user_id,
            group_id=item.group_id,
            messages=[item],
            pending_human_updates=list(state.active_updates)
            if item.pending_human_update
            else [],
        )

    def _record_active_update_locked(
        self,
        state: _ConversationState,
        item: QueuedMessage,
    ) -> None:
        item.pending_human_update = True
        text = normalize_message_text(item.raw_message)
        if text and text not in state.active_updates:
            state.active_updates.append(text)

    def _schedule_flush_locked(
        self,
        state: _ConversationState,
        conversation_key: str,
        user_id: str,
        processor: TurnProcessor,
        *,
        delay: float,
    ) -> None:
        previous_task = state.flush_tasks.get(user_id)
        if previous_task is not None and not previous_task.done():
            previous_task.cancel()
        state.flush_tasks[user_id] = asyncio.create_task(
            self._flush_later(conversation_key, user_id, processor, delay=delay)
        )

    async def _flush_later(
        self,
        conversation_key: str,
        user_id: str,
        processor: TurnProcessor,
        *,
        delay: float,
    ) -> None:
        try:
            await asyncio.sleep(max(delay, 0.0))
            state = self._states.get(conversation_key)
            if state is None:
                return
            async with state.lock:
                self._flush_current_locked(state, user_id)
                task = state.flush_tasks.get(user_id)
                if task is asyncio.current_task():
                    state.flush_tasks.pop(user_id, None)
                self._ensure_worker_locked(state, conversation_key, processor)
        except asyncio.CancelledError:
            return

    def _flush_current_locked(self, state: _ConversationState, user_id: str) -> None:
        current = state.current_by_user.pop(user_id, None)
        if current is None:
            return
        self._enqueue_turn_locked(state, current)

    def _enqueue_turn_locked(
        self,
        state: _ConversationState,
        turn: QueuedTurn,
    ) -> None:
        if turn.group_id is None:
            state.pending.clear()
            state.pending.append(turn)
            return
        if not self._ensure_queue_capacity_locked(state, turn):
            logger.warning(
                "ChatInter TurnQueue 丢弃低优先级闲聊 turn: "
                f"group={turn.group_id or 'private'} user={turn.user_id}"
            )
            return
        state.pending.append(turn)

    def _ensure_queue_capacity_locked(
        self,
        state: _ConversationState,
        new_turn: QueuedTurn,
    ) -> bool:
        if len(state.pending) < MAX_PENDING_TURNS_PER_GROUP:
            return True
        for index, turn in enumerate(state.pending):
            if not turn.must_keep:
                _drop_pending_at(state.pending, index)
                return True
        if new_turn.must_keep and state.pending:
            state.pending.popleft()
            return True
        return False

    def _ensure_worker_locked(
        self,
        state: _ConversationState,
        conversation_key: str,
        processor: TurnProcessor,
    ) -> None:
        if state.worker_task is not None and not state.worker_task.done():
            return
        state.worker_task = asyncio.create_task(
            self._worker(conversation_key, processor)
        )

    async def _worker(
        self,
        conversation_key: str,
        processor: TurnProcessor,
    ) -> None:
        state = self._states.get(conversation_key)
        if state is None:
            return
        idle_started = time.monotonic()
        while True:
            async with state.lock:
                if state.pending:
                    turn = state.pending.popleft()
                    state.active_turn = turn
                    state.active_updates.clear()
                    state.active_updates.extend(turn.pending_human_updates)
                    process_task = asyncio.create_task(
                        self._process_turn(
                            turn,
                            processor,
                            state.active_updates,
                            is_current=lambda generation=turn.latest.generation: (
                                not generation or state.generation == generation
                            ),
                        )
                    )
                    state.active_task = process_task
                else:
                    turn = None
                    process_task = None
                    if state.current_by_user:
                        idle_started = time.monotonic()
                    elif time.monotonic() - idle_started >= _IDLE_WORKER_EXIT_SECONDS:
                        for task in state.flush_tasks.values():
                            if not task.done():
                                task.cancel()
                        state.flush_tasks.clear()
                        state.worker_task = None
                        if self._states.get(conversation_key) is state:
                            self._states.pop(conversation_key, None)
                        return
            if turn is None:
                await asyncio.sleep(0.2)
                continue
            try:
                if process_task is not None:
                    await process_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(
                    "ChatInter TurnQueue turn 执行失败: "
                    f"group={turn.group_id or 'private'} user={turn.user_id}",
                    e=exc,
                )
            finally:
                async with state.lock:
                    if state.active_turn is turn:
                        state.active_turn = None
                    if state.active_task is process_task:
                        state.active_task = None
                    state.active_updates.clear()
                    idle_started = time.monotonic()

    async def _process_turn(
        self,
        turn: QueuedTurn,
        processor: TurnProcessor,
        active_updates: list[str],
        *,
        is_current: Callable[[], bool],
    ) -> None:
        latest = turn.latest
        raw_message = turn.raw_message
        pending_updates = _unique_normalized_texts(
            [
                *turn.pending_human_updates,
                *(msg.raw_message for msg in turn.messages if msg.pending_human_update),
            ]
        )
        active_updates[:] = pending_updates
        _attach_turn_metadata(
            latest.event,
            turn_messages=[item.raw_message for item in turn.messages],
            turn_message_sources=[
                item.message for item in turn.messages if item.message is not None
            ],
            pending_human_updates=active_updates,
            priority=turn.priority,
            generation=latest.generation,
            is_current=is_current,
        )
        logger.debug(
            "ChatInter TurnQueue flush: "
            f"group={turn.group_id or 'private'} user={turn.user_id} "
            f"messages={len(turn.messages)} priority={turn.priority}"
        )
        await processor(
            latest.bot,
            latest.event,
            latest.session,
            raw_message,
            latest.message,
            route_modules=turn.route_modules,
            cached_plain_text=raw_message,
            queued=True,
            turn_messages=[item.raw_message for item in turn.messages],
            pending_human_updates=active_updates,
        )


def _build_queued_message(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message: Any | None,
    route_modules: set[str] | None,
    cached_plain_text: str | None,
    priority_override: int | None = None,
    must_keep_override: bool | None = None,
    source: str = "message",
    dedupe_key: str | None = None,
) -> QueuedMessage:
    user_id = str(session.user.id)
    group_id = str(session.group.id) if session.group else None
    message_id = str(
        getattr(event, "message_id", "")
        or getattr(event, "event_id", "")
        or getattr(event, "id", "")
        or ""
    )
    priority = _message_priority(
        bot=bot,
        event=event,
        raw_message=raw_message,
        route_modules=route_modules,
        group_id=group_id,
    )
    if priority_override is not None:
        priority = max(0, int(priority_override))
    conversation_key = conversation_session_key(session)
    normalized_source = normalize_message_text(source) or "message"
    normalized_raw = normalize_message_text(raw_message)
    final_dedupe_key = normalize_message_text(str(dedupe_key or ""))
    if not final_dedupe_key:
        final_dedupe_key = _default_dedupe_key(
            conversation_key=conversation_key,
            user_id=user_id,
            message_id=message_id,
            raw_message=normalized_raw,
            source=normalized_source,
        )
    return QueuedMessage(
        bot=bot,
        event=event,
        session=session,
        raw_message=normalized_raw,
        message=message,
        route_modules=set(route_modules) if route_modules else None,
        cached_plain_text=cached_plain_text,
        user_id=user_id,
        group_id=group_id,
        conversation_key=conversation_key,
        message_id=message_id,
        source=normalized_source,
        dedupe_key=final_dedupe_key,
        priority=priority,
        must_keep=(
            bool(must_keep_override)
            if must_keep_override is not None
            else priority >= 1
        ),
        reply_to_message_id=_reply_message_id(event),
    )


def _default_dedupe_key(
    *,
    conversation_key: str,
    user_id: str,
    message_id: str,
    raw_message: str,
    source: str,
) -> str:
    if message_id:
        return f"{source}:{conversation_key}:{message_id}"
    return f"{source}:{conversation_key}:{user_id}:local:{uuid.uuid4().hex}"


def _has_dedupe_locked(state: _ConversationState, dedupe_key: str) -> bool:
    if not dedupe_key:
        return False
    if state.active_turn is not None and _turn_has_dedupe(
        state.active_turn,
        dedupe_key,
    ):
        return True
    if any(
        _turn_has_dedupe(turn, dedupe_key) for turn in state.current_by_user.values()
    ):
        return True
    return any(_turn_has_dedupe(turn, dedupe_key) for turn in state.pending)


def _turn_has_dedupe(turn: QueuedTurn, dedupe_key: str) -> bool:
    return any(item.dedupe_key == dedupe_key for item in turn.messages)


def _message_priority(
    *,
    bot: Bot,
    event: Event,
    raw_message: str,
    route_modules: set[str] | None,
    group_id: str | None,
) -> int:
    if route_modules:
        return 3
    if group_id is None:
        return 2
    if bool(getattr(event, "to_me", False)) or _is_reply_to_bot(bot, event):
        return 2
    if _looks_like_command(raw_message):
        return 1
    return 0


def _is_reply_to_bot(bot: Bot, event: Event) -> bool:
    reply = getattr(event, "reply", None)
    if reply is None:
        return False
    sender = getattr(reply, "sender", None)
    if sender is None and isinstance(reply, dict):
        sender = reply.get("sender")
    if isinstance(sender, dict):
        sender_id = sender.get("user_id")
    else:
        sender_id = getattr(sender, "user_id", None)
    return bool(
        sender_id is not None and str(sender_id) == str(getattr(bot, "self_id", ""))
    )


def _reply_message_id(event: Event) -> str:
    reply = getattr(event, "reply", None)
    if reply is None:
        return ""
    if isinstance(reply, dict):
        return normalize_message_text(
            str(reply.get("message_id") or reply.get("id") or "")
        )
    return normalize_message_text(
        str(getattr(reply, "message_id", "") or getattr(reply, "id", "") or "")
    )


def _is_followup_for_active_turn(
    active_turn: QueuedTurn,
    item: QueuedMessage,
) -> bool:
    if item.reply_to_message_id:
        return any(
            item.reply_to_message_id == message.message_id
            for message in active_turn.messages
            if message.message_id
        )
    if item.must_keep:
        return True
    if not active_turn.messages:
        return False
    return time.monotonic() - active_turn.messages[-1].created_at <= (
        COLLECT_WINDOW_SECONDS * 2
    )


def _looks_like_command(text: str) -> bool:
    normalized = normalize_message_text(text)
    if not normalized:
        return False
    if not normalized.startswith(("/", "!", "！", ".", "。", "#")):
        return False
    remainder = _MEDIA_PLACEHOLDER_PATTERN.sub("", normalized[1:])
    return any(
        not char.isspace() and not unicodedata.category(char).startswith("P")
        for char in remainder
    )


def _can_merge(turn: QueuedTurn, item: QueuedMessage) -> bool:
    if turn.user_id != item.user_id:
        return False
    if item.source != "message" or any(
        msg.source != "message" for msg in turn.messages
    ):
        return False
    if _turn_has_dedupe(turn, item.dedupe_key):
        return False
    if len(turn.messages) >= MAX_MESSAGES_PER_TURN:
        return False
    if time.monotonic() - turn.messages[-1].created_at > COLLECT_WINDOW_SECONDS:
        return False
    next_text = _compose_turn_text(
        [*(msg.raw_message for msg in turn.messages), item.raw_message]
    )
    return len(next_text) <= MAX_TURN_TEXT_CHARS


def _drop_pending_at(pending: deque[QueuedTurn], index: int) -> None:
    kept = [
        turn for current_index, turn in enumerate(pending) if current_index != index
    ]
    pending.clear()
    pending.extend(kept)


def _collect_delay(item: QueuedMessage) -> float:
    if item.priority >= 2:
        return _PRIORITY_COLLECT_SECONDS
    if item.priority == 1:
        return _COMMAND_COLLECT_SECONDS
    return COLLECT_WINDOW_SECONDS


def _compose_turn_text(messages: list[str]) -> str:
    values: list[str] = []
    for message in messages:
        text = normalize_message_text(message)
        if text:
            values.append(text)
    merged = "；".join(values)
    if len(merged) <= MAX_TURN_TEXT_CHARS:
        return merged
    return merged[:MAX_TURN_TEXT_CHARS].rstrip()


def _unique_normalized_texts(messages: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for message in messages:
        text = normalize_message_text(message)
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _attach_turn_metadata(
    event: Event,
    *,
    turn_messages: list[str],
    turn_message_sources: list[Any],
    pending_human_updates: list[str],
    priority: int,
    generation: int = 0,
    is_current: Callable[[], bool] | None = None,
) -> None:
    set_event_signal(event, "_chatinter_turn_queued", True)
    set_event_signal(event, "_chatinter_turn_merged", len(turn_messages) > 1)
    set_event_signal(event, "_chatinter_turn_messages", list(turn_messages))
    set_event_signal(
        event,
        "_chatinter_turn_message_sources",
        list(turn_message_sources),
    )
    set_event_signal(event, "_chatinter_pending_human_updates", pending_human_updates)
    set_event_signal(event, "_chatinter_turn_priority", int(priority))
    set_event_signal(event, "_chatinter_turn_generation", int(generation))
    if is_current is not None:
        set_event_signal(event, "_chatinter_turn_is_current", is_current)


_TURN_QUEUE = TurnQueue()


def get_turn_queue() -> TurnQueue:
    return _TURN_QUEUE


__all__ = [
    "COLLECT_WINDOW_SECONDS",
    "MAX_MESSAGES_PER_TURN",
    "MAX_PENDING_TURNS_PER_GROUP",
    "MAX_TURN_TEXT_CHARS",
    "QueuedTurn",
    "TurnQueue",
    "get_turn_queue",
]
