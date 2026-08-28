"""Single-message serialized turn queue for ChatInter conversations."""

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
from nonebot.matcher import current_bot, current_event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.services.log import logger

from .event_runtime import get_nickname, is_already_handled, mark_as_handled
from .event_signals import set_event_signal
from .group_turn_context import record_group_turn_message, remove_group_turn_message
from .mode_gate import MixedTurnLease
from .route_text import normalize_message_text
from .session_identity import conversation_session_key

MAX_PENDING_TURNS_PER_GROUP = 8
_QUEUE_REPLACED_MESSAGE = "消息排队较多，这条请求已被后续消息替代，请重新发送。"
_QUEUE_REJECTED_MESSAGE = "当前消息队列已满，这条请求未能处理，请稍后重试。"
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
    mode_lease: MixedTurnLease | None = None
    source: str = "message"
    dedupe_key: str = ""
    group_context_record_id: str = ""
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class QueuedTurn:
    conversation_key: str
    user_id: str
    group_id: str | None
    messages: list[QueuedMessage]

    @property
    def latest(self) -> QueuedMessage:
        return self.messages[-1]

    @property
    def priority(self) -> int:
        return self.latest.priority

    @property
    def must_keep(self) -> bool:
        return self.latest.must_keep

    @property
    def raw_message(self) -> str:
        return self.latest.raw_message

    @property
    def route_modules(self) -> set[str] | None:
        return self.latest.route_modules


@dataclass
class _ConversationState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    wake_event: asyncio.Event = field(default_factory=asyncio.Event)
    pending: deque[QueuedTurn] = field(default_factory=deque)
    worker_task: asyncio.Task | None = None
    active_turn: QueuedTurn | None = None
    active_task: asyncio.Task[None] | None = None


class TurnQueue:
    def __init__(self) -> None:
        self._states: dict[str, _ConversationState] = {}
        self._states_lock = asyncio.Lock()
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
        mode_lease: MixedTurnLease | None = None,
        priority_override: int | None = None,
        must_keep_override: bool | None = None,
        source: str = "message",
        dedupe_key: str | None = None,
        allow_handled_event: bool = False,
        mark_event_handled: bool = True,
    ) -> bool:
        lease_transferred = False
        dropped_turn: QueuedTurn | None = None
        rejected_turn: QueuedTurn | None = None
        try:
            handled_session_key = conversation_session_key(session)
            if not allow_handled_event and is_already_handled(
                event,
                session_key=handled_session_key,
            ):
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
                mode_lease=mode_lease,
                priority_override=priority_override,
                must_keep_override=must_keep_override,
                source=source,
                dedupe_key=dedupe_key,
            )
            if item.priority <= 0:
                return False
            if not self._remember_dedupe_key(item.dedupe_key):
                logger.debug(
                    "ChatInter TurnQueue 跳过重复消息: "
                    f"source={item.source} key={item.dedupe_key[:64]}"
                )
                return False

            async with self._states_lock:
                state = self._states.setdefault(
                    item.conversation_key,
                    _ConversationState(),
                )
                async with state.lock:
                    if _has_dedupe_locked(state, item.dedupe_key):
                        return False
                    turn = QueuedTurn(
                        conversation_key=item.conversation_key,
                        user_id=item.user_id,
                        group_id=item.group_id,
                        messages=[item],
                    )
                    accepted, dropped_turn = self._enqueue_turn_locked(state, turn)
                    if not accepted:
                        rejected_turn = turn
                        logger.warning(
                            "ChatInter TurnQueue 队列已满，丢弃消息: "
                            f"group={item.group_id or 'private'} user={item.user_id}"
                        )
                        return False
                    item.group_context_record_id = record_group_turn_message(
                        group_id=item.group_id,
                        user_id=item.user_id,
                        nickname=get_nickname(session),
                        text=item.raw_message,
                        message_id=item.message_id,
                        record_id=item.dedupe_key,
                    )
                    if dropped_turn is not None:
                        _rollback_turn_group_context(dropped_turn)
                    if mark_event_handled:
                        mark_as_handled(event, session_key=handled_session_key)
                    state.wake_event.set()
                    try:
                        self._ensure_worker_locked(
                            state,
                            item.conversation_key,
                            processor,
                        )
                    except BaseException:
                        _remove_pending_turn_locked(state.pending, turn)
                        _rollback_turn_group_context(turn)
                        rejected_turn = turn
                        raise
                    lease_transferred = True
            return True
        finally:
            try:
                if dropped_turn is not None:
                    await asyncio.shield(_release_turn_mode_leases(dropped_turn))
            finally:
                try:
                    if dropped_turn is not None:
                        await asyncio.shield(
                            _notify_turn_terminal(dropped_turn, reason="replaced")
                        )
                    if rejected_turn is not None:
                        await asyncio.shield(
                            _notify_turn_terminal(rejected_turn, reason="queue_full")
                        )
                finally:
                    if mode_lease is not None and not lease_transferred:
                        await asyncio.shield(mode_lease.release())

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

    def _enqueue_turn_locked(
        self,
        state: _ConversationState,
        turn: QueuedTurn,
    ) -> tuple[bool, QueuedTurn | None]:
        if len(state.pending) < MAX_PENDING_TURNS_PER_GROUP:
            state.pending.append(turn)
            return True, None
        for index, pending_turn in enumerate(state.pending):
            if not pending_turn.must_keep:
                dropped = _drop_pending_at(state.pending, index)
                state.pending.append(turn)
                return True, dropped
        if turn.must_keep and state.pending:
            dropped = state.pending.popleft()
            state.pending.append(turn)
            return True, dropped
        return False, None

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
        while True:
            async with state.lock:
                if state.pending:
                    turn = state.pending.popleft()
                    queue_wait_ms = max(
                        (time.monotonic() - turn.latest.enqueued_at) * 1000,
                        0.0,
                    )
                    if not state.pending:
                        state.wake_event.clear()
                    state.active_turn = turn
                    process_task = asyncio.create_task(
                        self._process_turn(turn, processor, queue_wait_ms)
                    )
                    state.active_task = process_task
                else:
                    state.wake_event.clear()
                    turn = None
                    process_task = None
            if turn is None:
                try:
                    await asyncio.wait_for(
                        state.wake_event.wait(),
                        timeout=_IDLE_WORKER_EXIT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    if await self._retire_idle_worker(conversation_key, state):
                        return
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

    async def _retire_idle_worker(
        self,
        conversation_key: str,
        state: _ConversationState,
    ) -> bool:
        async with self._states_lock:
            if self._states.get(conversation_key) is not state:
                return True
            async with state.lock:
                if state.pending or state.wake_event.is_set():
                    return False
                if state.worker_task is not asyncio.current_task():
                    return True
                state.worker_task = None
                self._states.pop(conversation_key, None)
                return True

    async def _process_turn(
        self,
        turn: QueuedTurn,
        processor: TurnProcessor,
        queue_wait_ms: float,
    ) -> None:
        item = turn.latest
        current_text = (
            item.cached_plain_text
            if item.cached_plain_text is not None
            else item.raw_message
        )
        bot_token = current_bot.set(item.bot)
        event_token = current_event.set(item.event)
        try:
            _attach_turn_metadata(
                item.event,
                priority=item.priority,
                queue_wait_ms=queue_wait_ms,
                is_current=lambda: True,
                group_context_record_id=item.group_context_record_id,
            )
            logger.debug(
                "ChatInter TurnQueue flush: "
                f"group={turn.group_id or 'private'} user={turn.user_id} messages=1 "
                f"priority={turn.priority}"
            )
            await processor(
                item.bot,
                item.event,
                item.session,
                current_text,
                item.message,
                route_modules=item.route_modules,
                cached_plain_text=current_text,
                queued=True,
            )
        finally:
            current_event.reset(event_token)
            current_bot.reset(bot_token)
            await asyncio.shield(_release_turn_mode_leases(turn))


def _build_queued_message(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message: Any | None,
    route_modules: set[str] | None,
    cached_plain_text: str | None,
    mode_lease: MixedTurnLease | None = None,
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
        mode_lease=mode_lease,
    )


def _default_dedupe_key(
    *,
    conversation_key: str,
    user_id: str,
    message_id: str,
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


def _drop_pending_at(pending: deque[QueuedTurn], index: int) -> QueuedTurn:
    dropped = pending[index]
    kept = [
        turn for current_index, turn in enumerate(pending) if current_index != index
    ]
    pending.clear()
    pending.extend(kept)
    return dropped


def _remove_pending_turn_locked(
    pending: deque[QueuedTurn],
    target: QueuedTurn,
) -> None:
    kept = [turn for turn in pending if turn is not target]
    pending.clear()
    pending.extend(kept)


async def _release_turn_mode_leases(turn: QueuedTurn) -> None:
    for item in turn.messages:
        lease = item.mode_lease
        if lease is None:
            continue
        item.mode_lease = None
        await lease.release()


def _rollback_turn_group_context(turn: QueuedTurn) -> None:
    for item in turn.messages:
        if not item.group_context_record_id:
            continue
        remove_group_turn_message(item.group_id, item.group_context_record_id)
        item.group_context_record_id = ""


async def _notify_turn_terminal(turn: QueuedTurn, *, reason: str) -> None:
    text = _QUEUE_REPLACED_MESSAGE if reason == "replaced" else _QUEUE_REJECTED_MESSAGE
    for item in turn.messages:
        mark_as_handled(item.event, session_key=item.conversation_key)
        set_event_signal(item.event, "_chatinter_queue_terminal", reason)
        try:
            await item.bot.send(item.event, text)
        except Exception as exc:
            logger.warning(
                "ChatInter TurnQueue terminal notification failed: "
                f"reason={reason} group={item.group_id or 'private'} "
                f"user={item.user_id} error={type(exc).__name__}"
            )


def _attach_turn_metadata(
    event: Event,
    *,
    priority: int,
    queue_wait_ms: float = 0.0,
    generation: int = 0,
    is_current: Callable[[], bool] | None = None,
    group_context_record_id: str = "",
) -> None:
    set_event_signal(event, "_chatinter_turn_queued", True)
    set_event_signal(event, "_chatinter_turn_priority", int(priority))
    set_event_signal(
        event,
        "_chatinter_turn_queue_wait_ms",
        max(float(queue_wait_ms), 0.0),
    )
    set_event_signal(event, "_chatinter_turn_generation", int(generation))
    set_event_signal(
        event,
        "_chatinter_group_context_record_id",
        normalize_message_text(group_context_record_id),
    )
    if is_current is not None:
        set_event_signal(event, "_chatinter_turn_is_current", is_current)


_TURN_QUEUE = TurnQueue()


def get_turn_queue() -> TurnQueue:
    return _TURN_QUEUE


__all__ = [
    "MAX_PENDING_TURNS_PER_GROUP",
    "QueuedTurn",
    "TurnQueue",
    "get_turn_queue",
]
