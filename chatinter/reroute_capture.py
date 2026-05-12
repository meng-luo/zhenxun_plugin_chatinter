"""Trace-scoped capture of messages produced by ChatInter reroute tasks."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Any, ClassVar

from nonebot.adapters import Bot

from zhenxun.services.log import logger

_SEND_APIS = {"send_msg", "send_group_msg", "send_private_msg"}
_CURRENT_TRACE_ID: ContextVar[str | None] = ContextVar(
    "chatinter_reroute_trace_id",
    default=None,
)
_MAX_RECORDS_PER_TRACE = 12
_MAX_TEXT_LEN = 900


@dataclass(frozen=True)
class CapturedRerouteOutput:
    trace_id: str
    api: str
    text: str
    raw_message: str
    result: Any
    timestamp: float


class RerouteOutputCapture:
    _records: ClassVar[dict[str, list[CapturedRerouteOutput]]] = defaultdict(list)
    _patched_target: ClassVar[Any | None] = None

    @classmethod
    def ensure_patched(cls) -> None:
        current = Bot.call_api
        if getattr(current, "_chatinter_capture_wrapper", False):
            return

        original = current

        async def _capturing_call_api(self: Bot, api: str, **data: Any):
            trace_id = _CURRENT_TRACE_ID.get()
            try:
                result = await original(self, api, **data)
            except Exception as exc:
                if trace_id and api in _SEND_APIS:
                    cls.record(
                        trace_id=trace_id,
                        api=api,
                        data=data,
                        result={"ok": False, "error": str(exc)},
                    )
                raise
            if trace_id and api in _SEND_APIS:
                cls.record(trace_id=trace_id, api=api, data=data, result=result)
            return result

        setattr(_capturing_call_api, "_chatinter_capture_wrapper", True)
        setattr(_capturing_call_api, "_chatinter_capture_original", original)
        Bot.call_api = _capturing_call_api  # type: ignore[assignment]
        cls._patched_target = original

    @classmethod
    def record(
        cls,
        *,
        trace_id: str,
        api: str,
        data: dict[str, Any],
        result: Any,
    ) -> None:
        trace_key = str(trace_id or "").strip()
        if not trace_key:
            return
        raw_message = _message_to_text(data.get("message"))
        text = _compact_text(raw_message)
        target = cls._records[trace_key]
        if len(target) >= _MAX_RECORDS_PER_TRACE:
            return
        target.append(
            CapturedRerouteOutput(
                trace_id=trace_key,
                api=api,
                text=text,
                raw_message=raw_message[:_MAX_TEXT_LEN],
                result=result,
                timestamp=time.time(),
            )
        )

    @classmethod
    @contextmanager
    def activate(cls, trace_id: str):
        cls.ensure_patched()
        trace_key = str(trace_id or "").strip()
        token = _CURRENT_TRACE_ID.set(trace_key or None)
        try:
            yield
        finally:
            _CURRENT_TRACE_ID.reset(token)

    @classmethod
    def pop_outputs(cls, trace_id: str) -> list[CapturedRerouteOutput]:
        return cls._records.pop(str(trace_id or "").strip(), [])


def _message_to_text(message: Any) -> str:
    if message is None:
        return ""
    if hasattr(message, "extract_plain_text"):
        try:
            text = str(message.extract_plain_text())
            if text.strip():
                return text
        except Exception:
            pass
    try:
        return str(message)
    except Exception as exc:
        logger.debug(f"ChatInter reroute output stringify failed: {exc}")
        return ""


def _compact_text(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= _MAX_TEXT_LEN:
        return normalized
    return normalized[: _MAX_TEXT_LEN - 1].rstrip() + "…"


__all__ = [
    "CapturedRerouteOutput",
    "RerouteOutputCapture",
]
