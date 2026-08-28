"""
ChatInter - 聊天响应处理

实现聊天意图处理和消息响应生成。
"""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
import re
import time
from typing import Any
import uuid

from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import (
    Bot as OneBotV11Bot,
)
from nonebot.adapters.onebot.v11 import (
    GroupMessageEvent,
    Message,
    MessageSegment,
    PrivateMessageEvent,
)
from nonebot.message import handle_event as handle_nonebot_event
from nonebot.plugin import get_loaded_plugins

from zhenxun.services import logger
from zhenxun.services.send_queue import (
    SendObservation,
    observe_send_trace,
    pop_send_observations,
)

from .artifact_store import get_artifact_store
from .event_signals import get_event_signal, set_event_signal
from .route_text import normalize_message_text

_REROUTE_TASKS: set[asyncio.Task] = set()
_REROUTE_CANCEL_GRACE_SECONDS = 1.0
_REROUTE_TOKEN_PATTERN = re.compile(
    r"\[@(?:[^\]\s]+|所有人)\]|\[image(?:#\d+)?\]|(?<![0-9A-Za-z_])@\d{5,20}(?=(?:\s|$|[的，,。.!！？?]))",
    re.IGNORECASE,
)
_IMAGE_INDEX_PATTERN = re.compile(r"\[image#(\d+)\]", re.IGNORECASE)
_MD_FENCED_CODE_PATTERN = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_MD_INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_MD_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_MD_HEADING_LINE_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_QUOTE_LINE_PATTERN = re.compile(r"(?m)^\s{0,3}>\s?")
_MD_BULLET_LINE_PATTERN = re.compile(r"(?m)^\s*[-*+]\s+")
_MD_ORDERED_LINE_PATTERN = re.compile(r"(?m)^\s*(\d+)[.)]\s+")
_MD_RULE_LINE_PATTERN = re.compile(r"(?m)^\s*([-*_]\s*){3,}\s*$")
_MD_BOLD_PATTERN = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
_MD_STRIKE_PATTERN = re.compile(r"~~(.+?)~~", re.DOTALL)
_MD_EXCESSIVE_LINE_BREAKS_PATTERN = re.compile(r"\n{3,}")
_AT_ID_TOKEN_PATTERN = re.compile(
    r"\[@([^\]\s]+)\]|(?<![0-9A-Za-z_])@(\d{5,20})(?=(?:\s|$|[的，,。.!！？?]))"
)
_UNRESOLVED_IMAGE_PLACEHOLDER_PATTERN = re.compile(
    r"\[image(?:#\d+)?\]",
    re.IGNORECASE,
)
_OBSERVED_IMAGE_OUTPUT_PATTERN = re.compile(
    r"(?:\[image\b|\[CQ:image\b|type=['\"]?image)",
    re.IGNORECASE,
)
_FORWARD_SEND_APIS = frozenset(
    {
        "send_forward_msg",
        "send_group_forward_msg",
        "send_private_forward_msg",
    }
)
_FORWARD_SEND_CAPTURE: ContextVar[tuple[str, Bot, list[SendObservation]] | None] = (
    ContextVar("chatinter_forward_send_capture", default=None)
)


async def _dispatch_rerouted_event(bot: Bot, event: Event) -> None:
    if isinstance(bot, OneBotV11Bot):
        await handle_nonebot_event(bot, event)
        return
    await bot.handle_event(event)


@dataclass(frozen=True)
class RerouteExecutionResult:
    success: bool
    command: str
    trace_id: str
    outputs: list[SendObservation] = field(default_factory=list)
    error: str = ""
    timed_out: bool = False
    cancelled: bool = False
    execution_uncertain: bool = False
    execution_started: bool = False
    task_stopped: bool = True
    dispatched: bool = False

    @property
    def observed_text(self) -> str:
        return "\n".join(item.text for item in self.outputs if item.text).strip()


_REROUTE_CANCELLATION_RECEIPT = ContextVar(
    "chatinter_reroute_cancellation_receipt",
    default=None,
)


def consume_reroute_cancellation_receipt() -> RerouteExecutionResult | None:
    receipt = _REROUTE_CANCELLATION_RECEIPT.get()
    _REROUTE_CANCELLATION_RECEIPT.set(None)
    return receipt


async def _cancel_and_wait_reroute_task(
    task: asyncio.Task[Any],
    *,
    trace_id: str,
) -> bool:
    if not task.done():
        task.cancel()
    done, _pending = await asyncio.wait(
        {task},
        timeout=_REROUTE_CANCEL_GRACE_SECONDS,
    )
    if task in done:
        await asyncio.gather(task, return_exceptions=True)
        return True
    task.add_done_callback(lambda _done_task: pop_send_observations(trace_id))
    return False


def _captured_send_count(bot: Bot) -> int | None:
    records = getattr(bot, "sent_messages", None)
    return len(records) if isinstance(records, list) else None


def _reroute_execution_observed(
    event: Event,
) -> bool:
    return bool(get_event_signal(event, "_ai_plugin_execution_started", False))


def _reroute_execution_failed(event: Event) -> bool:
    return bool(get_event_signal(event, "_ai_plugin_execution_failed", False))


def _captured_send_observations(
    bot: Bot,
    *,
    trace_id: str,
    start_index: int | None,
) -> list[SendObservation]:
    records = getattr(bot, "sent_messages", None)
    if start_index is None or not isinstance(records, list):
        return []
    outputs: list[SendObservation] = []
    for item in records[start_index : start_index + 12]:
        if isinstance(item, tuple) and len(item) >= 2:
            api, data = item[0], item[1]
        else:
            api, data = "captured_send", item
        payload = data if isinstance(data, dict) else {"message": data}
        raw = _message_payload_to_text(payload.get("message"))
        if not raw and isinstance(payload, dict):
            for key in ("messages", "message", "raw_message"):
                raw = _message_payload_to_text(payload.get(key))
                if raw:
                    break
        if (
            not raw
            and isinstance(payload, dict)
            and str(api or "")
            in {
                "send_msg",
                "send_group_msg",
                "send_private_msg",
            }
        ):
            raw = "[plugin visible send]"
        outputs.append(
            SendObservation(
                trace_id=trace_id,
                api=str(api or "captured_send"),
                text=normalize_message_text(raw)[:900],
                raw_message=raw[:900],
                result={},
                timestamp=time.time(),
            )
        )
    return outputs


def _message_payload_to_text(message: Any) -> str:
    if message is None:
        return ""
    if hasattr(message, "extract_plain_text"):
        try:
            plain = str(message.extract_plain_text())
            if plain.strip():
                return plain
        except Exception:
            pass
    if isinstance(message, list | tuple):
        parts = [_message_payload_to_text(item) for item in message]
        return " ".join(part for part in parts if part).strip()
    if isinstance(message, dict):
        segment_type = str(message.get("type", "") or "")
        data = message.get("data")
        if segment_type:
            if segment_type == "text" and isinstance(data, dict):
                return str(data.get("text", "") or "")
            if segment_type == "image":
                return "[plugin image output]"
            if segment_type == "music":
                source = ""
                music_id = ""
                if isinstance(data, dict):
                    source = str(data.get("type", "") or "")
                    music_id = str(data.get("id", "") or "")
                suffix = " ".join(item for item in (source, music_id) if item)
                return f"[plugin music output{': ' + suffix if suffix else ''}]"
            if segment_type in {"record", "voice"}:
                return "[plugin voice output]"
            if segment_type == "video":
                return "[plugin video output]"
            if segment_type in {"node", "forward"}:
                return "[plugin forward message]"
            return f"[plugin {segment_type} output]"
        for key in ("message", "messages", "raw_message", "content"):
            raw = _message_payload_to_text(message.get(key))
            if raw:
                return raw
    try:
        return str(message)
    except Exception:
        return ""


async def _observe_forward_send_api(
    bot: Bot,
    exception: Exception | None,
    api: str,
    data: dict[str, Any],
    result: Any,
) -> None:
    capture = _FORWARD_SEND_CAPTURE.get()
    if (
        capture is None
        or capture[1] is not bot
        or api not in _FORWARD_SEND_APIS
        or len(capture[2]) >= 12
    ):
        return
    raw = _message_payload_to_text(data.get("message"))
    if not raw:
        raw = _message_payload_to_text(data.get("messages"))
    if not raw:
        raw = "[plugin forward message]"
    capture[2].append(
        SendObservation(
            trace_id=capture[0],
            api=api,
            text=normalize_message_text(raw)[:900],
            raw_message=raw[:900],
            result=(
                {"ok": False, "error": str(exception)}
                if exception is not None
                else result
            ),
            timestamp=time.time(),
        )
    )


Bot.on_called_api(_observe_forward_send_api)


def _merge_captured_outputs_if_empty(
    outputs: list[SendObservation],
    *,
    bot: Bot,
    trace_id: str,
    start_index: int | None,
    additional_outputs: list[SendObservation] | None = None,
) -> list[SendObservation]:
    merged = [*outputs, *(additional_outputs or ())]
    if merged:
        return merged[:12]
    return _captured_send_observations(
        bot,
        trace_id=trace_id,
        start_index=start_index,
    )


def artifacts_from_send_observations(
    outputs: list[SendObservation],
    *,
    trace_id: str,
) -> list[dict[str, Any]]:
    """Convert plugin sends into compact artifact refs for model observations."""

    store = get_artifact_store()
    artifacts: list[dict[str, Any]] = []
    for index, output in enumerate(_successful_send_observations(outputs), 1):
        raw = str(output.raw_message or "")
        text = str(output.text or "")
        if _OBSERVED_IMAGE_OUTPUT_PATTERN.search(raw):
            artifacts.append(
                store.store_reference(
                    artifact_type="image",
                    summary=f"plugin image output #{index}",
                    trace_id=trace_id,
                    source=output.api,
                    size=len(raw),
                ).to_dict()
            )
            if raw and raw != text:
                ref = store.store_text(
                    raw,
                    artifact_type="plugin_output",
                    trace_id=trace_id,
                    source=f"{output.api}:raw_message",
                    force_file=len(raw) > 240,
                )
                if ref is not None:
                    artifacts.append(ref.to_dict())
            continue

        if raw and raw != text and len(raw) > len(text):
            ref = store.store_text(
                raw,
                artifact_type="plugin_output",
                trace_id=trace_id,
                source=f"{output.api}:raw_message",
                force_file=len(raw) > 240,
            )
            if ref is not None:
                artifacts.append(ref.to_dict())
            continue

    return artifacts


def messages_summary_from_send_observations(
    outputs: list[SendObservation],
) -> list[str]:
    """Summarize real plugin sends; image-only sends still count as visible."""

    summaries: list[str] = []
    for index, output in enumerate(_successful_send_observations(outputs), 1):
        text = normalize_message_text(str(output.text or ""))
        raw = str(output.raw_message or "")
        if text:
            summaries.append(text[:260])
        elif _OBSERVED_IMAGE_OUTPUT_PATTERN.search(raw):
            summaries.append(f"[plugin image output #{index}]")
        elif raw:
            summaries.append(normalize_message_text(raw)[:260])
    return summaries[:8]


def _successful_send_observations(
    outputs: list[SendObservation],
) -> list[SendObservation]:
    return [output for output in outputs if not _send_result_failed(output.result)]


def _send_result_failed(result: Any) -> bool:
    if isinstance(result, BaseException):
        return True
    if not isinstance(result, dict):
        return False
    if result.get("ok") is False:
        return True
    status = normalize_message_text(str(result.get("status", "") or "")).casefold()
    if status in {"failed", "failure", "error"}:
        return True
    retcode = result.get("retcode")
    if retcode is None:
        return False
    try:
        return int(retcode) != 0
    except (TypeError, ValueError):
        return True


async def reroute_to_plugin(
    bot: Bot,
    event: Event,
    command: str,
    target_modules: set[str] | None = None,
    extra_image_segments: list[MessageSegment] | None = None,
) -> bool:
    result = await reroute_to_plugin_with_result(
        bot,
        event,
        command,
        target_modules=target_modules,
        extra_image_segments=extra_image_segments,
        wait=False,
    )
    return result.success


async def reroute_to_plugin_with_result(
    bot: Bot,
    event: Event,
    command: str,
    target_modules: set[str] | None = None,
    extra_image_segments: list[MessageSegment] | None = None,
    *,
    trace_id: str | None = None,
    wait: bool = True,
    timeout: float = 10.0,
) -> RerouteExecutionResult:
    _REROUTE_CANCELLATION_RECEIPT.set(None)
    trace_key = trace_id or uuid.uuid4().hex
    command_text = command.strip()
    try:
        import time

        captured_send_start = _captured_send_count(bot)
        event_data = event.model_dump()
        bot_self_id = str(getattr(bot, "self_id", "")) or None
        new_message = _build_reroute_message(
            command_text,
            event,
            bot_self_id,
            extra_images=extra_image_segments,
        )
        unresolved_plain_text = ""
        image_segment_count = 0
        for segment in new_message:
            if segment.type == "image":
                image_segment_count += 1
                continue
            if segment.type == "text":
                unresolved_plain_text += str(segment.data.get("text", ""))
        if image_segment_count <= 0 and _UNRESOLVED_IMAGE_PLACEHOLDER_PATTERN.search(
            unresolved_plain_text
        ):
            logger.warning(
                "重路由消息仍包含未解析的 [image] 占位符，"
                f"取消重投以避免下游插件解析失败：{command_text}"
            )
            return RerouteExecutionResult(
                success=False,
                command=command_text,
                trace_id=trace_key,
                error="unresolved image placeholder",
            )

        rendered_plain_text = new_message.extract_plain_text()
        event_data["message"] = new_message
        event_data["raw_message"] = rendered_plain_text
        event_data["plain_text"] = rendered_plain_text
        if getattr(event, "reply", None) is not None:
            event_data["reply"] = getattr(event, "reply")

        if hasattr(bot, "self_id"):
            event_data["self_id"] = bot.self_id

        event_data["message_id"] = int(time.time() * 1000)
        event_data["time"] = int(time.time())

        logger.debug(
            f"构造重路由消息：'{new_message.extract_plain_text()}', "
            f"self_id={event_data.get('self_id')}, "
            f"images={sum(1 for seg in new_message if seg.type == 'image')}, "
            f"ats={sum(1 for seg in new_message if seg.type == 'at')}"
        )

        if isinstance(event, GroupMessageEvent):
            new_event = GroupMessageEvent(**event_data)
        elif isinstance(event, PrivateMessageEvent):
            new_event = PrivateMessageEvent(**event_data)
        else:
            logger.warning(f"不支持的事件类型：{type(event)}")
            return RerouteExecutionResult(
                success=False,
                command=command_text,
                trace_id=trace_key,
                error=f"unsupported event type: {type(event)}",
            )

        set_event_signal(new_event, "_ai_triggered", True)
        set_event_signal(new_event, "_ai_trace_id", trace_key)
        expanded_target_modules = _expand_reroute_target_modules(target_modules)
        if expanded_target_modules:
            set_event_signal(
                new_event, "_ai_route_modules", frozenset(expanded_target_modules)
            )
        route_heads = _extract_reroute_heads(command_text)
        if route_heads:
            set_event_signal(new_event, "_ai_route_heads", frozenset(route_heads))

        forward_send_outputs: list[SendObservation] = []
        if wait:
            capture_token = _FORWARD_SEND_CAPTURE.set(
                (trace_key, bot, forward_send_outputs)
            )
            try:
                with observe_send_trace(trace_key):
                    task = asyncio.create_task(_dispatch_rerouted_event(bot, new_event))
            finally:
                _FORWARD_SEND_CAPTURE.reset(capture_token)
        else:
            task = asyncio.create_task(_dispatch_rerouted_event(bot, new_event))
        _REROUTE_TASKS.add(task)
        task.add_done_callback(lambda done_task: _REROUTE_TASKS.discard(done_task))
        if wait:
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=max(float(timeout), 0.5),
                )
            except asyncio.TimeoutError:
                task_stopped = await _cancel_and_wait_reroute_task(
                    task,
                    trace_id=trace_key,
                )
                outputs = _merge_captured_outputs_if_empty(
                    pop_send_observations(trace_key),
                    bot=bot,
                    trace_id=trace_key,
                    start_index=captured_send_start,
                    additional_outputs=forward_send_outputs,
                )
                execution_started = _reroute_execution_observed(new_event)
                logger.warning(f"消息重路由等待超时，执行结果不确定：{command_text}")
                return RerouteExecutionResult(
                    success=False,
                    command=command_text,
                    trace_id=trace_key,
                    outputs=outputs,
                    error="reroute timeout",
                    timed_out=True,
                    execution_uncertain=True,
                    execution_started=execution_started,
                    task_stopped=task_stopped,
                    dispatched=True,
                )
            except asyncio.CancelledError:
                task_stopped = await _cancel_and_wait_reroute_task(
                    task,
                    trace_id=trace_key,
                )
                outputs = _merge_captured_outputs_if_empty(
                    pop_send_observations(trace_key),
                    bot=bot,
                    trace_id=trace_key,
                    start_index=captured_send_start,
                    additional_outputs=forward_send_outputs,
                )
                execution_started = _reroute_execution_observed(new_event)
                cancellation_result = RerouteExecutionResult(
                    success=False,
                    command=command_text,
                    trace_id=trace_key,
                    outputs=outputs,
                    error="reroute cancelled",
                    cancelled=True,
                    execution_uncertain=execution_started or not task_stopped,
                    execution_started=execution_started,
                    task_stopped=task_stopped,
                    dispatched=True,
                )
                if cancellation_result.execution_uncertain:
                    setattr(task, "_chatinter_reroute_receipt", cancellation_result)
                    _REROUTE_CANCELLATION_RECEIPT.set(cancellation_result)
                logger.warning(
                    "消息重路由取消收据："
                    f"trace_id={trace_key}, task_stopped={task_stopped}, "
                    f"execution_uncertain={cancellation_result.execution_uncertain}"
                )
                raise
            except Exception as exc:
                outputs = _merge_captured_outputs_if_empty(
                    pop_send_observations(trace_key),
                    bot=bot,
                    trace_id=trace_key,
                    start_index=captured_send_start,
                    additional_outputs=forward_send_outputs,
                )
                execution_started = _reroute_execution_observed(new_event)
                logger.warning(f"消息重路由执行异常：{command_text}, error={exc}")
                return RerouteExecutionResult(
                    success=False,
                    command=command_text,
                    trace_id=trace_key,
                    outputs=outputs,
                    error=str(exc),
                    execution_uncertain=True,
                    execution_started=execution_started,
                    dispatched=True,
                )
        outputs = _merge_captured_outputs_if_empty(
            pop_send_observations(trace_key),
            bot=bot,
            trace_id=trace_key,
            start_index=captured_send_start,
            additional_outputs=forward_send_outputs,
        )
        execution_started = _reroute_execution_observed(new_event)
        execution_failed = _reroute_execution_failed(new_event)
        if execution_started and not execution_failed:
            logger.info(f"消息重路由执行完成：{command_text}")
        elif execution_failed:
            logger.info(f"消息重路由目标插件执行异常：{command_text}")
        else:
            logger.info(f"消息重路由未观察到目标插件执行：{command_text}")
        return RerouteExecutionResult(
            success=execution_started and not execution_failed,
            command=command_text,
            trace_id=trace_key,
            outputs=outputs,
            error=(
                "plugin_execution_failed"
                if execution_failed
                else ""
                if execution_started
                else "plugin_not_observed"
            ),
            execution_uncertain=execution_failed,
            execution_started=execution_started,
            dispatched=True,
        )

    except Exception as e:
        logger.error(f"消息重路由失败：{e}")
        return RerouteExecutionResult(
            success=False,
            command=command_text,
            trace_id=trace_key,
            outputs=pop_send_observations(trace_key),
            error=str(e),
        )


def _parse_at_target(token: str) -> str | None:
    token = token.strip()
    if token.startswith("[@") and token.endswith("]"):
        target = token[2:-1].strip()
    elif token.startswith("@"):
        target = token[1:].strip()
    else:
        return None
    if not target:
        return None
    if target in {"所有人", "all"}:
        return "all"
    return target


def _expand_reroute_target_modules(target_modules: set[str] | None) -> set[str]:
    if not target_modules:
        return set()

    expanded = {
        item.strip()
        for item in target_modules
        if isinstance(item, str) and item.strip()
    }
    if not expanded:
        return set()

    for plugin in get_loaded_plugins():
        plugin_name = str(getattr(plugin, "name", "") or "").strip()
        module_name = str(getattr(plugin, "module_name", "") or "").strip()
        if not plugin_name and not module_name:
            continue
        if module_name and module_name in expanded:
            expanded.add(plugin_name)
            continue
        if plugin_name and plugin_name in expanded and module_name:
            expanded.add(module_name)
    return expanded


def _extract_reroute_heads(command_text: str) -> set[str]:
    normalized = str(command_text or "").strip()
    if not normalized:
        return set()
    head = normalized.split(" ", 1)[0].strip()
    if not head:
        return set()
    return {head, head.lower(), head.casefold()}


def _extract_source_images(event: Event) -> list[MessageSegment]:
    try:
        source_message = event.get_message()
    except Exception:
        source_message = getattr(event, "message", None)
    if not isinstance(source_message, Message):
        return []
    images: list[MessageSegment] = []
    for seg in source_message:
        if seg.type == "image":
            images.append(seg)
    return images


def _extract_source_mentions(
    event: Event,
    bot_self_id: str | None,
) -> list[MessageSegment]:
    try:
        source_message = event.get_message()
    except Exception:
        source_message = getattr(event, "message", None)
    if not isinstance(source_message, Message):
        return []
    mentions: list[MessageSegment] = []
    for seg in source_message:
        if seg.type != "at":
            continue
        qq_value = str(seg.data.get("qq", "")).strip()
        if not qq_value:
            continue
        if bot_self_id and qq_value == str(bot_self_id):
            continue
        mentions.append(seg)
    return mentions


def _build_reroute_message(
    command_text: str,
    event: Event,
    bot_self_id: str | None = None,
    extra_images: list[MessageSegment] | None = None,
) -> Message:
    if not command_text:
        return Message("")

    source_images = _extract_source_images(event)
    if extra_images:
        for image in extra_images:
            if not isinstance(image, MessageSegment):
                continue
            if image.type != "image":
                continue
            source_images.append(image)
    source_mentions = _extract_source_mentions(event, bot_self_id)
    has_explicit_image_token = False
    has_explicit_at_token = False
    result = Message()
    cursor = 0
    for match in _REROUTE_TOKEN_PATTERN.finditer(command_text):
        if match.start() > cursor:
            result += MessageSegment.text(command_text[cursor : match.start()])

        token = match.group(0)
        lower_token = token.lower()
        if lower_token.startswith("[image"):
            has_explicit_image_token = True
            image_index = 0
            index_match = _IMAGE_INDEX_PATTERN.fullmatch(token)
            if index_match:
                parsed_index = int(index_match.group(1))
                image_index = max(parsed_index - 1, 0)
            if source_images:
                chosen_index = min(image_index, len(source_images) - 1)
                result += source_images[chosen_index]
            else:
                result += MessageSegment.text(token)
        else:
            target = _parse_at_target(token)
            if target == "all":
                has_explicit_at_token = True
                result += MessageSegment.at("all")
            elif target:
                has_explicit_at_token = True
                result += MessageSegment.at(target)
            else:
                result += MessageSegment.text(token)
        cursor = match.end()

    if cursor < len(command_text):
        result += MessageSegment.text(command_text[cursor:])

    if not has_explicit_at_token and source_mentions:
        result += MessageSegment.text(" ")
        for mention in source_mentions:
            result += mention

    if not has_explicit_image_token and source_images:
        result += MessageSegment.text(" ")
        result += source_images[0]

    if not result:
        return Message(command_text)
    return result


def _looks_like_markdown(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    markdown_signals = (
        _MD_LINK_PATTERN.search(text) is not None,
        _MD_IMAGE_PATTERN.search(text) is not None,
        _MD_INLINE_CODE_PATTERN.search(text) is not None,
        _MD_HEADING_LINE_PATTERN.search(text) is not None,
        _MD_QUOTE_LINE_PATTERN.search(text) is not None,
        _MD_BULLET_LINE_PATTERN.search(text) is not None,
        _MD_ORDERED_LINE_PATTERN.search(text) is not None,
        _MD_BOLD_PATTERN.search(text) is not None,
        _MD_STRIKE_PATTERN.search(text) is not None,
    )
    return any(markdown_signals)


def _has_code_markdown(text: str) -> bool:
    return "```" in text or _MD_FENCED_CODE_PATTERN.search(text) is not None


def normalize_ai_reply_text(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return normalized

    if _has_code_markdown(normalized):
        return normalized
    if not _looks_like_markdown(normalized):
        return normalized

    converted = normalized
    converted = _MD_IMAGE_PATTERN.sub(
        lambda match: (
            f"{match.group(1)} ({match.group(2)})"
            if match.group(1).strip()
            else match.group(2)
        ),
        converted,
    )
    converted = _MD_LINK_PATTERN.sub(
        lambda match: f"{match.group(1)} ({match.group(2)})",
        converted,
    )
    converted = _MD_HEADING_LINE_PATTERN.sub("", converted)
    converted = _MD_QUOTE_LINE_PATTERN.sub("", converted)
    converted = _MD_BULLET_LINE_PATTERN.sub("• ", converted)
    converted = _MD_ORDERED_LINE_PATTERN.sub(r"\1. ", converted)
    converted = _MD_RULE_LINE_PATTERN.sub("", converted)
    converted = _MD_BOLD_PATTERN.sub(r"\2", converted)
    converted = _MD_STRIKE_PATTERN.sub(r"\1", converted)
    converted = "\n".join(line.rstrip() for line in converted.splitlines())
    converted = _MD_EXCESSIVE_LINE_BREAKS_PATTERN.sub("\n\n", converted).strip()
    return converted or normalized


def replace_mention_ids_with_names(
    text: str,
    mention_name_map: dict[str, str] | None = None,
) -> str:
    normalized = (text or "").strip()
    if not normalized or not mention_name_map:
        return normalized

    def _replace(match: re.Match[str]) -> str:
        user_id = (match.group(1) or match.group(2) or "").strip()
        if not user_id:
            return match.group(0)
        nickname = mention_name_map.get(user_id)
        if not nickname:
            return match.group(0)
        return f"@{nickname}"

    return _AT_ID_TOKEN_PATTERN.sub(_replace, normalized)


__all__ = [
    "RerouteExecutionResult",
    "artifacts_from_send_observations",
    "messages_summary_from_send_observations",
    "normalize_ai_reply_text",
    "replace_mention_ids_with_names",
    "reroute_to_plugin",
    "reroute_to_plugin_with_result",
]
