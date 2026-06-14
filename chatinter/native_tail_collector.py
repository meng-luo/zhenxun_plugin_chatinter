"""Non-blocking collector for native-command heads with follow-up tasks."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import re
import time
from typing import Any
import weakref

from nonebot import get_driver, get_loaded_plugins
from nonebot.adapters import Bot, Event
from nonebot.matcher import matchers
from nonebot_plugin_uninfo import Uninfo
from pydantic import BaseModel, Field

from zhenxun.configs.config import BotConfig
from zhenxun.configs.utils import PluginExtraData
from zhenxun.services import logger
from zhenxun.services.llm import AI

from .config import build_reasoning_generation_config, get_config_value, get_model_name
from .event_signals import get_event_signal, set_event_signal
from .route_text import normalize_message_text
from .scenario_router import ChatInterScenario, ScenarioRoute
from .turn_queue import TurnProcessor, get_turn_queue

_COLLECT_DELAY_MIN_SECONDS = 0.8
_COLLECT_DELAY_BASE_SECONDS = 1.2
_COLLECT_DELAY_MAX_SECONDS = 3.2
_NATIVE_OUTPUT_WAIT_SECONDS = 0.8
_MAX_CACHE_SIZE = 512
_MAX_CACHE_AGE_SECONDS = 120.0
_MAX_TAIL_TASKS = 5
_MESSAGE_RATE_WINDOW_SECONDS = 12.0
_MESSAGE_RATE_CACHE_SIZE = 512
_MESSAGE_CACHE: deque[tuple[str, float]] = deque()
_MESSAGE_KEYS: set[str] = set()
_CONTINUATION_CACHE: deque[tuple[str, float]] = deque()
_CONTINUATION_KEYS: set[str] = set()
_MESSAGE_RATE_CACHE: dict[str, deque[float]] = {}
_MESSAGE_RATE_LAST_SEEN: deque[tuple[str, float]] = deque()
_TAIL_FOLLOWUP_TASKS: set[asyncio.Task[None]] = set()
_CONTINUATION_HINTS = (
    "然后",
    "最后",
    "顺便",
    "接着",
    "以及",
    "还有",
    "同时",
    "并且",
    "再帮",
    "再看",
    "再来",
    "再抽",
)
_CONTINUATION_PUNCTUATION = ("，", ",", "。", "；", ";")
_COMMAND_BOUNDARY_CHARS = frozenset(" \t\r\n，,。.!！?？；;：:/|）)]】}》>、")
_LEADING_COMMAND_PUNCTUATION_PATTERN = re.compile(r"^[\s,，:：;；.!！?？、]+")
_LEADING_PLACEHOLDER_PATTERN = re.compile(r"^(?:\s*(?:\[[^\]]*]|\<[^>]*>))+\s*")
_PARAM_TOKEN_PATTERN = re.compile(r"\s*[?*+]?\[[^\]]+\]")
_PARAM_BRACKET_PATTERN = re.compile(r"\s*[?*+]?[<(｟][^>)｠]+[>)｠]")
_TRAILING_TEMPLATE_MARKER_PATTERN = re.compile(r"(?:\s+[?*]+|[?*]+)$")
_REGEX_LIKE_COMMAND_TOKENS = ("\\", "(", ")", "[", "]", "|", "^", "$", "re:")
_CONTINUATION_PREFIX_PATTERN = re.compile(
    r"^\s*(?:然后|最后|顺便|接着|以及|还有|同时|并且|再帮(?:我)?|再看(?:看|一下|下)?|再来|再抽|再发)\s*",
    re.IGNORECASE,
)
_NATIVE_ROUTE_CACHE_SIGNATURE: tuple[tuple[str, ...], ...] | None = None
_NATIVE_ROUTE_PREFIX_MAP: dict[str, tuple["_NativeCommandEntry", ...]] = {}
_SENT_OBSERVER_REGISTERED = False


@dataclass(frozen=True)
class _NativeCommandEntry:
    command: str
    modules: frozenset[str]


class NativeTailTaskResult(BaseModel):
    should_continue: bool = Field(default=False)
    remaining_tasks: list[str] = Field(default_factory=list)
    reason: str = ""


def should_collect_native_tail(
    *,
    raw_message: str,
    route_modules: set[str] | None,
) -> bool:
    if not route_modules:
        return False
    message = normalize_message_text(raw_message)
    if len(message) < 8:
        return False
    if any(hint in message for hint in _CONTINUATION_HINTS):
        return True
    return sum(message.count(item) for item in _CONTINUATION_PUNCTUATION) >= 2


def resolve_native_tail_route_modules(raw_message: str) -> set[str]:
    """Best-effort native route detection without importing hook plugins.

    The auth checker owns the authoritative route cache, but importing it here
    during event dispatch can dynamically load hooks and cancel the native
    matcher that should process the leading command.  This local scanner only
    inspects already loaded plugin metadata and is therefore side-effect free.
    """

    candidates = _message_probe_candidates(raw_message)
    if not candidates:
        return set()
    _ensure_native_route_index()
    for message in candidates:
        entries = _NATIVE_ROUTE_PREFIX_MAP.get(message[0].casefold())
        if not entries:
            continue
        matched_modules: set[str] = set()
        for entry in entries:
            if _command_matches(message, entry.command):
                matched_modules.update(entry.modules)
        if matched_modules:
            return matched_modules
    return set()


def schedule_native_tail_followup(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message: Any | None,
    route_modules: set[str] | None,
    cached_plain_text: str | None,
    processor: TurnProcessor,
) -> bool:
    key = _message_key(event, raw_message)
    if key in _MESSAGE_KEYS:
        return False
    route_modules = route_modules or resolve_native_tail_route_modules(raw_message)
    if not should_collect_native_tail(
        raw_message=raw_message,
        route_modules=route_modules,
    ):
        return False
    if not _remember_message(key):
        return False
    collect_delay = _record_and_resolve_collect_delay(event=event, session=session)
    _mark_native_tail_pending(event, raw_message)
    _ensure_sent_observer_registered()
    task = asyncio.create_task(
        _run_native_tail_followup(
            bot=bot,
            event=event,
            session=session,
            raw_message=raw_message,
            message=message,
            route_modules=route_modules,
            cached_plain_text=cached_plain_text,
            processor=processor,
            collect_delay=collect_delay,
        )
    )
    _TAIL_FOLLOWUP_TASKS.add(task)
    task.add_done_callback(_TAIL_FOLLOWUP_TASKS.discard)
    return True


async def _run_native_tail_followup(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message: Any | None,
    route_modules: set[str],
    cached_plain_text: str | None,
    processor: TurnProcessor,
    collect_delay: float,
) -> None:
    try:
        await asyncio.sleep(collect_delay)
        if not _native_command_has_completed(event):
            await _wait_native_command_output(event)
        result = await _judge_native_tail(
            raw_message=raw_message,
            route_modules=route_modules,
            native_completed=_native_command_has_completed(event),
        )
    except Exception as exc:
        logger.warning(f"[ChatInter] native-tail collector failed: {exc}")
        _clear_native_tail_pending(event)
        return
    remaining_tasks = _normalize_tasks(result.remaining_tasks)
    if not result.should_continue or not remaining_tasks:
        logger.debug(
            "ChatInter native-tail collector 无剩余任务: "
            f"reason={result.reason or 'none'}"
        )
        _clear_native_tail_pending(event)
        return
    remaining_tasks = _drop_native_head_echoes(
        remaining_tasks,
        raw_message=raw_message,
    )
    if not remaining_tasks:
        logger.debug(
            "ChatInter native-tail collector 剩余任务仅包含原生命令回声，已忽略"
        )
        _clear_native_tail_pending(event)
        return
    tail_message = "；".join(remaining_tasks)
    continuation_key = _continuation_key(
        event=event,
        raw_message=raw_message,
        tail_message=tail_message,
    )
    if not _remember_continuation(continuation_key):
        logger.debug(
            "[ChatInter] 原生命令后续任务重复，已跳过：" f"{tail_message[:80]}..."
        )
        _clear_native_tail_pending(event)
        return
    scenario_route = ScenarioRoute(
        scenario=ChatInterScenario.GROUP_PLUGIN_SELECTOR,
        reason="native_command_tail_followup",
        allow_plugin_tools=True,
    )

    async def _processor(
        bot_arg: Bot,
        event_arg: Event,
        session_arg: Uninfo,
        raw_arg: str,
        message_arg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        await processor(
            bot_arg,
            event_arg,
            session_arg,
            raw_arg,
            message_arg,
            scenario_route=scenario_route,
            **kwargs,
        )

    try:
        accepted = await get_turn_queue().submit(
            bot=bot,
            event=event,
            session=session,
            raw_message=tail_message,
            message=message,
            route_modules=None,
            cached_plain_text=tail_message,
            processor=_processor,
            priority_override=2,
            must_keep_override=True,
            source="native_tail",
            dedupe_key=continuation_key,
            allow_handled_event=True,
            mark_event_handled=False,
        )
    except Exception as exc:
        logger.warning(f"[ChatInter] native-tail enqueue failed: {exc}")
        _clear_native_tail_pending(event)
        return
    if accepted:
        logger.info(
            "[ChatInter] 原生命令后续任务已进入队列：" f"{tail_message[:80]}..."
        )
    _clear_native_tail_pending(event)


async def _judge_native_tail(
    *,
    raw_message: str,
    route_modules: set[str],
    native_completed: bool,
) -> NativeTailTaskResult:
    payload = {
        "original_message": normalize_message_text(raw_message),
        "native_route_modules": sorted(route_modules),
        "native_command_status": (
            "completed_or_observed"
            if native_completed
            else "handed_to_original_plugin_but_output_not_observed"
        ),
        "continuation_policy": (
            "The leading native command has priority. ChatInter may only handle "
            "independent follow-up tasks after that leading command; never repeat "
            "or reinterpret the leading command."
        ),
    }
    digest = hashlib.blake2s(
        normalize_message_text(raw_message).encode("utf-8", errors="ignore"),
        digest_size=6,
    ).hexdigest()
    ai = AI(session_id=f"chatinter-native-tail:{digest}")
    try:
        return await ai.generate_structured(
            json.dumps(payload, ensure_ascii=False),
            NativeTailTaskResult,
            model=get_model_name(),
            config=build_reasoning_generation_config(),
            instruction=_NATIVE_TAIL_INSTRUCTION,
            timeout=float(get_config_value("INTENT_TIMEOUT", 20) or 20),
            max_validation_retries=0,
            auto_thinking=False,
        )
    except Exception as exc:
        fallback = _fallback_native_tail_result(
            raw_message=raw_message,
            route_modules=route_modules,
            native_completed=native_completed,
            reason=f"judge_failed:{type(exc).__name__}",
        )
        if fallback.should_continue:
            logger.debug(
                "[ChatInter] native-tail judge failed, fallback accepted: "
                f"{fallback.remaining_tasks[:3]}"
            )
        else:
            logger.debug(f"[ChatInter] native-tail judge failed: {exc}")
        return fallback


def _record_and_resolve_collect_delay(*, event: Event, session: Uninfo) -> float:
    scope_key = _message_rate_scope_key(event=event, session=session)
    now = time.monotonic()
    _cleanup_message_rate_cache(now)
    bucket = _MESSAGE_RATE_CACHE.setdefault(scope_key, deque())
    bucket.append(now)
    while bucket and now - bucket[0] > _MESSAGE_RATE_WINDOW_SECONDS:
        bucket.popleft()
    _MESSAGE_RATE_LAST_SEEN.append((scope_key, now))
    message_count = len(bucket)
    if message_count <= 1:
        return _COLLECT_DELAY_BASE_SECONDS
    if message_count <= 3:
        delay = _COLLECT_DELAY_BASE_SECONDS + 0.25 * (message_count - 1)
    else:
        delay = _COLLECT_DELAY_BASE_SECONDS + 0.5 + 0.15 * min(message_count - 3, 10)
    return max(
        _COLLECT_DELAY_MIN_SECONDS,
        min(_COLLECT_DELAY_MAX_SECONDS, round(delay, 2)),
    )


def _message_rate_scope_key(*, event: Event, session: Uninfo) -> str:
    session_id = normalize_message_text(str(getattr(session, "id", "") or ""))
    if session_id:
        return f"session:{session_id}"
    for attr in ("group_id", "guild_id", "channel_id", "user_id"):
        value = normalize_message_text(str(getattr(event, attr, "") or ""))
        if value:
            return f"{attr}:{value}"
    return f"event:{type(event).__module__}.{type(event).__name__}"


def _cleanup_message_rate_cache(now: float) -> None:
    while _MESSAGE_RATE_LAST_SEEN and (
        len(_MESSAGE_RATE_LAST_SEEN) > _MESSAGE_RATE_CACHE_SIZE
        or now - _MESSAGE_RATE_LAST_SEEN[0][1] > _MAX_CACHE_AGE_SECONDS
    ):
        scope_key, _ = _MESSAGE_RATE_LAST_SEEN.popleft()
        bucket = _MESSAGE_RATE_CACHE.get(scope_key)
        if bucket is None:
            continue
        while bucket and now - bucket[0] > _MESSAGE_RATE_WINDOW_SECONDS:
            bucket.popleft()
        if not bucket:
            _MESSAGE_RATE_CACHE.pop(scope_key, None)


def _fallback_native_tail_result(
    *,
    raw_message: str,
    route_modules: set[str],
    native_completed: bool,
    reason: str,
) -> NativeTailTaskResult:
    """Deterministic conservative fallback when the LLM judge is unavailable."""

    if not native_completed:
        return NativeTailTaskResult(
            should_continue=False,
            reason=f"{reason}:native_not_observed",
        )
    tasks = _extract_fallback_tail_tasks(
        raw_message=raw_message,
        route_modules=route_modules,
    )
    if not tasks:
        return NativeTailTaskResult(should_continue=False, reason=reason)
    return NativeTailTaskResult(
        should_continue=True,
        remaining_tasks=tasks,
        reason=f"{reason}:heuristic_tail",
    )


def _extract_fallback_tail_tasks(
    *,
    raw_message: str,
    route_modules: set[str],
) -> list[str]:
    message = normalize_message_text(raw_message)
    if not message or not route_modules:
        return []
    head = _matched_native_head(message)
    if not head:
        return []
    tail = normalize_message_text(message[len(head) :])
    tail = _LEADING_COMMAND_PUNCTUATION_PATTERN.sub("", tail)
    if not any(hint in tail for hint in _CONTINUATION_HINTS):
        return []
    result: list[str] = []
    for part in _split_fallback_tail(tail):
        normalized = _strip_continuation_prefix(part)
        if not normalized or _looks_like_native_head_argument(normalized):
            continue
        if normalized not in result:
            result.append(normalized)
        if len(result) >= _MAX_TAIL_TASKS:
            break
    return result


def _matched_native_head(message: str) -> str:
    candidates = _message_probe_candidates(message)
    _ensure_native_route_index()
    for candidate in candidates:
        entries = (
            _NATIVE_ROUTE_PREFIX_MAP.get(candidate[0].casefold()) if candidate else None
        )
        if not entries:
            continue
        for entry in entries:
            if _command_matches(candidate, entry.command):
                return entry.command
    return ""


def _split_fallback_tail(text: str) -> list[str]:
    tail = normalize_message_text(text)
    if not tail:
        return []
    parts = re.split(r"[，,。；;]+", tail)
    return [
        normalize_message_text(part) for part in parts if normalize_message_text(part)
    ]


def _looks_like_native_head_argument(text: str) -> bool:
    normalized = normalize_message_text(text)
    if not normalized:
        return True
    if len(normalized) <= 2 and not any(
        hint in normalized for hint in _CONTINUATION_HINTS
    ):
        return True
    return False


def _message_key(event: Event, raw_message: str) -> str:
    message_id = (
        getattr(event, "message_id", None)
        or getattr(event, "event_id", None)
        or getattr(event, "id", None)
        or ""
    )
    return f"{message_id}:{normalize_message_text(raw_message)}"


def _continuation_key(
    *,
    event: Event,
    raw_message: str,
    tail_message: str,
) -> str:
    digest = hashlib.blake2s(
        normalize_message_text(tail_message).encode("utf-8", errors="ignore"),
        digest_size=8,
    ).hexdigest()
    return f"native_tail:{_message_key(event, raw_message)}:{digest}"


def _mark_native_tail_pending(event: Event, raw_message: str) -> None:
    set_event_signal(event, "_chatinter_native_tail_pending", True)
    set_event_signal(
        event, "_chatinter_native_tail_source", normalize_message_text(raw_message)
    )


def _clear_native_tail_pending(event: Event) -> None:
    set_event_signal(event, "_chatinter_native_tail_pending", False)


def _native_command_has_completed(event: Event) -> bool:
    return bool(
        get_event_signal(event, "_chatinter_native_tail_observed_output", False)
    )


async def _wait_native_command_output(event: Event) -> None:
    deadline = time.monotonic() + _NATIVE_OUTPUT_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _native_command_has_completed(event):
            return
        await asyncio.sleep(0.05)


def _ensure_sent_observer_registered() -> None:
    global _SENT_OBSERVER_REGISTERED
    if _SENT_OBSERVER_REGISTERED:
        return
    try:
        from nonebot.message import event_postprocessor
    except Exception:
        return

    @event_postprocessor
    async def _chatinter_native_tail_output_observer(event: Event) -> None:
        if not bool(
            get_event_signal(event, "_chatinter_native_tail_pending", False)
        ):
            return
        set_event_signal(event, "_chatinter_native_tail_observed_output", True)

    _SENT_OBSERVER_REGISTERED = True


def _remember_message(key: str) -> bool:
    return _remember_key(
        key,
        cache=_MESSAGE_CACHE,
        keys=_MESSAGE_KEYS,
    )


def _remember_continuation(key: str) -> bool:
    return _remember_key(
        key,
        cache=_CONTINUATION_CACHE,
        keys=_CONTINUATION_KEYS,
    )


def _remember_key(
    key: str,
    *,
    cache: deque[tuple[str, float]],
    keys: set[str],
) -> bool:
    now = time.monotonic()
    while cache and (
        len(cache) > _MAX_CACHE_SIZE or now - cache[0][1] > _MAX_CACHE_AGE_SECONDS
    ):
        old_key, _ = cache.popleft()
        keys.discard(old_key)
    if key in keys:
        return False
    keys.add(key)
    cache.append((key, now))
    return True


def _normalize_tasks(tasks: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for task in tasks:
        normalized = normalize_message_text(task)
        if normalized and normalized not in result:
            result.append(normalized)
        if len(result) >= _MAX_TAIL_TASKS:
            break
    return result


def _drop_native_head_echoes(
    tasks: list[str],
    *,
    raw_message: str,
) -> list[str]:
    original = normalize_message_text(raw_message)
    result: list[str] = []
    for task in tasks:
        normalized = _strip_continuation_prefix(task)
        if not normalized:
            continue
        if normalized == original:
            continue
        if original.startswith(normalized):
            continue
        if normalized not in result:
            result.append(normalized)
    return result[:_MAX_TAIL_TASKS]


def _strip_continuation_prefix(text: str) -> str:
    normalized = normalize_message_text(text)
    previous = ""
    while normalized and previous != normalized:
        previous = normalized
        normalized = normalize_message_text(
            _CONTINUATION_PREFIX_PATTERN.sub("", normalized)
        )
    return normalized


def _ensure_native_route_index() -> None:
    global _NATIVE_ROUTE_CACHE_SIGNATURE

    signature = _native_route_index_signature()
    if signature == _NATIVE_ROUTE_CACHE_SIGNATURE:
        return
    prefix_map: dict[str, list[_NativeCommandEntry]] = {}
    for plugin in get_loaded_plugins():
        modules = _plugin_modules(plugin)
        if not modules:
            continue
        for command in _plugin_command_heads(plugin):
            entry = _NativeCommandEntry(
                command=command,
                modules=frozenset(modules),
            )
            prefix_map.setdefault(command[0].casefold(), []).append(entry)
    _NATIVE_ROUTE_PREFIX_MAP.clear()
    _NATIVE_ROUTE_PREFIX_MAP.update(
        {
            key: tuple(
                sorted(
                    values,
                    key=lambda item: (len(item.command), item.command),
                    reverse=True,
                )
            )
            for key, values in prefix_map.items()
        }
    )
    _NATIVE_ROUTE_CACHE_SIGNATURE = signature


def _native_route_index_signature() -> tuple[tuple[str, ...], ...]:
    plugin_items = [
        (
            "plugin",
            str(getattr(plugin, "name", "") or ""),
            str(getattr(plugin, "module_name", "") or ""),
        )
        for plugin in get_loaded_plugins()
    ]
    matcher_items: list[tuple[str, ...]] = []
    for priority, matcher_list in matchers.items():
        for matcher_cls in list(matcher_list):
            plugin = getattr(matcher_cls, "plugin", None)
            matcher_items.append(
                (
                    "matcher",
                    str(priority),
                    str(id(matcher_cls)),
                    str(getattr(matcher_cls, "plugin_name", "") or ""),
                    str(getattr(plugin, "module_name", "") or ""),
                )
            )
    return tuple(sorted([*plugin_items, *matcher_items]))


def _plugin_modules(plugin: Any) -> set[str]:
    modules = {
        normalize_message_text(str(getattr(plugin, "name", "") or "")),
        normalize_message_text(str(getattr(plugin, "module_name", "") or "")),
    }
    return {module for module in modules if module}


def _plugin_command_heads(plugin: Any) -> tuple[str, ...]:
    metadata = getattr(plugin, "metadata", None)
    commands: list[str] = []
    try:
        extra_data = (
            PluginExtraData(**(metadata.extra or {})) if metadata is not None else None
        )
    except Exception:
        extra_data = None
    if extra_data is not None:
        for command in extra_data.commands or []:
            commands.extend(
                _split_command_variants(getattr(command, "command", "") or "")
            )
        commands.extend(str(alias) for alias in (extra_data.aliases or set()) if alias)
    commands.extend(_matcher_command_heads_for_plugin(plugin))
    normalized: list[str] = []
    seen: set[str] = set()
    for command in commands:
        head = _normalize_metadata_command(command)
        if not head or head in seen or _is_ambiguous_route_command(head):
            continue
        seen.add(head)
        normalized.append(head)
    return tuple(normalized)


def _matcher_command_heads_for_plugin(plugin: Any) -> tuple[str, ...]:
    plugin_name = str(getattr(plugin, "name", "") or "")
    module_name = str(getattr(plugin, "module_name", "") or "")
    commands: set[str] = set()
    for matcher_list in matchers.values():
        for matcher_cls in list(matcher_list):
            matcher_plugin = getattr(matcher_cls, "plugin", None)
            matcher_plugin_name = str(getattr(matcher_cls, "plugin_name", "") or "")
            matcher_module = str(
                getattr(getattr(matcher_cls, "module", None), "__name__", "") or ""
            )
            if (
                matcher_plugin is not plugin
                and matcher_plugin_name != plugin_name
                and matcher_module != module_name
            ):
                continue
            _collect_command_literals(getattr(matcher_cls, "command", None), commands)
            rule = getattr(matcher_cls, "rule", None)
            for checker in getattr(rule, "checkers", ()) or ():
                call = getattr(checker, "call", None)
                if call is None:
                    continue
                for attr in ("cmds", "command", "commands", "cmd"):
                    _collect_command_literals(getattr(call, attr, None), commands)
    return tuple(commands)


def _collect_command_literals(
    value: Any,
    target: set[str],
    depth: int = 0,
) -> None:
    if value is None or depth > 4:
        return
    if isinstance(value, str):
        text = normalize_message_text(value)
        if text:
            target.add(text)
        return
    if isinstance(value, weakref.ReferenceType):
        resolved = value()
        if resolved is not None and resolved is not value:
            _collect_command_literals(resolved, target, depth + 1)
        return
    if isinstance(value, list | tuple | set | frozenset):
        for item in value:
            _collect_command_literals(item, target, depth + 1)
        return
    if callable(value) and getattr(value, "__self__", None) is not None:
        try:
            resolved = value()
        except Exception:
            resolved = None
        if resolved is not None and resolved is not value:
            _collect_command_literals(resolved, target, depth + 1)
            return
    for attr in (
        "name",
        "path",
        "aliases",
        "header_display",
        "command",
        "commands",
        "cmd",
        "cmds",
    ):
        nested = getattr(value, attr, None)
        if nested is not None and nested is not value:
            _collect_command_literals(nested, target, depth + 1)


def _split_command_variants(command: str) -> tuple[str, ...]:
    text = normalize_message_text(command)
    if not text:
        return ()
    if text.startswith("/"):
        return (text,)
    if "/" in text and " " not in text:
        parts = tuple(part.strip() for part in text.split("/") if part.strip())
        if parts:
            return parts
    return (text,)


def _normalize_metadata_command(command: str) -> str:
    text = normalize_message_text(command)
    if not text:
        return ""
    text = _PARAM_TOKEN_PATTERN.split(text, maxsplit=1)[0]
    text = _PARAM_BRACKET_PATTERN.split(text, maxsplit=1)[0]
    text = _TRAILING_TEMPLATE_MARKER_PATTERN.sub("", normalize_message_text(text))
    return normalize_message_text(text)


def _is_ambiguous_route_command(command: str) -> bool:
    text = normalize_message_text(command)
    if not text:
        return True
    if any(token in text for token in _REGEX_LIKE_COMMAND_TOKENS):
        return True
    if "xx" in text.casefold():
        return True
    return False


def _message_probe_candidates(raw_message: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    def _append(value: str) -> None:
        normalized = normalize_message_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            values.append(normalized)

    normalized = _LEADING_PLACEHOLDER_PATTERN.sub(
        "",
        normalize_message_text(raw_message),
    )
    _append(normalized)
    _append(_strip_bot_nickname_prefix(normalized))
    return values


def _strip_bot_nickname_prefix(text: str) -> str:
    stripped = normalize_message_text(text)
    for nickname in _bot_nickname_variants():
        if not nickname:
            continue
        if stripped.casefold().startswith(nickname.casefold()):
            return normalize_message_text(
                _LEADING_COMMAND_PUNCTUATION_PATTERN.sub(
                    "",
                    stripped[len(nickname) :],
                )
            )
    return stripped


def _bot_nickname_variants() -> tuple[str, ...]:
    nicknames: set[str] = set()
    configured_self = normalize_message_text(
        getattr(BotConfig, "self_nickname", "") or ""
    )
    if configured_self:
        nicknames.add(configured_self)
    try:
        for item in get_driver().config.nickname:
            nickname = normalize_message_text(str(item or ""))
            if nickname:
                nicknames.add(nickname)
    except Exception:
        pass
    return tuple(sorted(nicknames, key=len, reverse=True))


def _command_matches(text: str, command: str) -> bool:
    normalized_text = normalize_message_text(text).casefold()
    normalized_command = normalize_message_text(command).casefold()
    if not normalized_text or not normalized_command:
        return False
    if normalized_text == normalized_command:
        return True
    if not normalized_text.startswith(normalized_command):
        return False
    if len(normalized_text) == len(normalized_command):
        return True
    next_char = normalized_text[len(normalized_command)]
    if next_char.isspace() or next_char in _COMMAND_BOUNDARY_CHARS:
        return True
    return not normalized_command[-1].isascii()


_NATIVE_TAIL_INSTRUCTION = """
你是 ChatInter 的原生命令后续任务判定器。

目标：
- 原消息开头已经被 NoneBot 原生插件处理，不要把这部分加入 remaining_tasks。
- 只找“开头原生命令之外”的独立后续任务。
- 不按连接词机械切分；按语义判断是否真的有第二/第三个任务。
- 如果只是一个命令的长参数、说明、数量、目标、修饰语，不算后续任务。
- 如果没有明确后续任务，should_continue=false。

输出 JSON：
{
  "should_continue": false,
  "remaining_tasks": [],
  "reason": ""
}
""".strip()


__all__ = [
    "resolve_native_tail_route_modules",
    "schedule_native_tail_followup",
    "should_collect_native_tail",
]
