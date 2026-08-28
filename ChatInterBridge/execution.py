"""Execution and delivery tracking for ChatInterBridge requests."""

from __future__ import annotations

import time
import asyncio
from copy import deepcopy
from typing import Literal, Callable, Awaitable, TypedDict, NotRequired
from collections import OrderedDict
from dataclasses import dataclass

from gsuid_core.bot import Bot, _Bot
from gsuid_core.models import Event, Message

ExecutionState = Literal["preparing", "accepted", "running", "succeeded", "failed", "rejected", "unknown"]
DeliveryState = Literal[
    "not_attempted",
    "pending",
    "observed",
    "partial",
    "unobserved",
    "failed",
    "no_output",
]
TerminalExecutionState = Literal["succeeded", "failed", "rejected", "unknown"]
TargetType = Literal["group", "direct", "channel", "sub_channel"]
JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
OutboundMessage = Message | list[Message] | list[str] | str | bytes


class ExecuteResponse(TypedDict):
    disposition: str
    duplicate: bool
    revision: NotRequired[str]
    reason: NotRequired[str]
    request_id: NotRequired[str]
    task_id: NotRequired[str]


class ExecutionSnapshot(TypedDict):
    request_id: str
    task_id: str | None
    execution_state: ExecutionState
    execution_error: str | None
    delivery_state: DeliveryState
    delivery_observed: bool
    message_ids: list[str]
    delivery_attempts: int
    delivery_confirmed_attempts: int
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class ClaimCreated:
    pass


@dataclass(frozen=True)
class ClaimExisting:
    response: ExecuteResponse


@dataclass(frozen=True)
class ClaimConflict:
    response: ExecuteResponse


ClaimResult = ClaimCreated | ClaimExisting | ClaimConflict


@dataclass
class ExecutionRecord:
    request_id: str
    fingerprint: str
    response: ExecuteResponse
    execution_state: ExecutionState
    delivery_state: DeliveryState
    created_at: float
    updated_at: float
    task_id: str | None = None
    execution_error: str | None = None
    delivery_attempts: int = 0
    delivery_confirmed_attempts: int = 0
    delivery_inflight: int = 0
    delivery_failed_attempts: int = 0
    message_ids: list[str] | None = None
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        if self.message_ids is None:
            self.message_ids = []


class ExecutionStore:
    """Event-loop-local TTL store for idempotency and execution receipts."""

    def __init__(
        self,
        ttl_seconds: float = 600.0,
        max_entries: int = 4096,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._wall_clock = wall_clock
        self._records: OrderedDict[str, ExecutionRecord] = OrderedDict()

    def claim(self, request_id: str, fingerprint: str) -> ClaimResult:
        self._prune()
        existing = self._records[request_id] if request_id in self._records else None
        if existing is not None:
            if existing.fingerprint != fingerprint:
                return ClaimConflict(
                    {
                        "disposition": "conflict",
                        "reason": "idempotency_fingerprint_mismatch",
                        "duplicate": False,
                    }
                )
            response = deepcopy(existing.response)
            response["duplicate"] = True
            return ClaimExisting(response)

        while len(self._records) >= self._max_entries:
            self._records.popitem(last=False)
        now = self._wall_clock()
        expires_at = self._clock() + self._ttl_seconds
        self._records[request_id] = ExecutionRecord(
            request_id=request_id,
            fingerprint=fingerprint,
            response={"disposition": "unknown", "reason": "submission_in_progress", "duplicate": False},
            execution_state="preparing",
            delivery_state="not_attempted",
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )
        return ClaimCreated()

    def finish_preparation(
        self,
        request_id: str,
        response: ExecuteResponse,
        state: TerminalExecutionState,
    ) -> None:
        record = self._record(request_id)
        if record is None:
            return
        record.response = deepcopy(response)
        record.execution_state = state
        record.execution_error = response["reason"] if "reason" in response else None
        self._touch(record)

    def mark_accepted(self, request_id: str, task_id: str, response: ExecuteResponse) -> None:
        record = self._record(request_id)
        if record is None:
            return
        record.response = deepcopy(response)
        record.task_id = task_id
        record.execution_state = "accepted"
        self._touch(record)

    def mark_running(self, request_id: str) -> None:
        record = self._record(request_id)
        if record is None:
            return
        record.execution_state = "running"
        self._touch(record)

    def mark_succeeded(self, request_id: str) -> None:
        record = self._record(request_id)
        if record is None:
            return
        record.execution_state = "succeeded"
        if record.delivery_attempts == 0:
            record.delivery_state = "no_output"
        self._touch(record)

    def mark_failed(self, request_id: str, error: str) -> None:
        record = self._record(request_id)
        if record is None:
            return
        record.execution_state = "failed"
        record.execution_error = error
        self._touch(record)

    def delivery_started(self, request_id: str) -> None:
        record = self._record(request_id)
        if record is None:
            return
        record.delivery_attempts += 1
        record.delivery_inflight += 1
        record.delivery_state = "pending"
        self._touch(record)

    def delivery_finished(self, request_id: str, message_ids: list[str] | None) -> None:
        record = self._record(request_id)
        if record is None:
            return
        record.delivery_inflight = max(0, record.delivery_inflight - 1)
        if message_ids:
            record.delivery_confirmed_attempts += 1
            known_ids = record.message_ids
            if known_ids is not None:
                for message_id in message_ids:
                    if message_id not in known_ids:
                        known_ids.append(message_id)
        self._refresh_delivery_state(record)
        self._touch(record)

    def delivery_failed(self, request_id: str) -> None:
        record = self._record(request_id)
        if record is None:
            return
        record.delivery_inflight = max(0, record.delivery_inflight - 1)
        record.delivery_failed_attempts += 1
        self._refresh_delivery_state(record)
        self._touch(record)

    def snapshot(self, request_id: str) -> ExecutionSnapshot | None:
        self._prune()
        record = self._records[request_id] if request_id in self._records else None
        if record is None:
            return None
        message_ids = record.message_ids if record.message_ids is not None else []
        return {
            "request_id": record.request_id,
            "task_id": record.task_id,
            "execution_state": record.execution_state,
            "execution_error": record.execution_error,
            "delivery_state": record.delivery_state,
            "delivery_observed": bool(message_ids),
            "message_ids": list(message_ids),
            "delivery_attempts": record.delivery_attempts,
            "delivery_confirmed_attempts": record.delivery_confirmed_attempts,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def reset(self) -> None:
        self._records.clear()

    def _record(self, request_id: str) -> ExecutionRecord | None:
        return self._records[request_id] if request_id in self._records else None

    def _touch(self, record: ExecutionRecord) -> None:
        record.updated_at = self._wall_clock()
        record.expires_at = self._clock() + self._ttl_seconds

    def _prune(self) -> None:
        now = self._clock()
        expired = [request_id for request_id, record in self._records.items() if now > record.expires_at]
        for request_id in expired:
            del self._records[request_id]

    @staticmethod
    def _refresh_delivery_state(record: ExecutionRecord) -> None:
        if record.delivery_inflight:
            record.delivery_state = "pending"
        elif record.delivery_confirmed_attempts == record.delivery_attempts:
            record.delivery_state = "observed"
        elif record.delivery_confirmed_attempts:
            record.delivery_state = "partial"
        elif record.delivery_failed_attempts == record.delivery_attempts:
            record.delivery_state = "failed"
        else:
            record.delivery_state = "unobserved"


class BridgeBot(Bot):
    """Bot used only by bridge executions to request adapter delivery receipts."""

    def __init__(self, bot: _Bot, event: Event, request_id: str, records: ExecutionStore) -> None:
        super().__init__(bot, event)
        self._bridge_request_id = request_id
        self._bridge_records = records

    async def _observe_delivery(self, delivery: Awaitable[list[str] | None]) -> list[str] | None:
        self._bridge_records.delivery_started(self._bridge_request_id)
        try:
            message_ids = await delivery
        except asyncio.CancelledError:
            self._bridge_records.delivery_failed(self._bridge_request_id)
            raise
        except Exception:
            self._bridge_records.delivery_failed(self._bridge_request_id)
            raise
        self._bridge_records.delivery_finished(self._bridge_request_id, message_ids)
        return message_ids

    async def send(
        self,
        message: OutboundMessage,
        at_sender: bool = False,
        extra_metadata: dict[str, JsonValue] | None = None,
        wait_recall: bool = False,
    ) -> list[str] | None:
        delivery = super().send(
            message,
            at_sender,
            extra_metadata=extra_metadata,
            wait_recall=True,
        )
        message_ids = await self._observe_delivery(delivery)
        return message_ids if wait_recall else None

    async def target_send(
        self,
        message: OutboundMessage,
        target_type: TargetType,
        target_id: str | None,
        at_sender: bool = False,
        sender_id: str = "",
        send_source_group: str | None = None,
        wait_recall: bool = False,
    ) -> list[str] | None:
        delivery = super().target_send(
            message,
            target_type,
            target_id,
            at_sender,
            sender_id,
            send_source_group,
            wait_recall=True,
        )
        message_ids = await self._observe_delivery(delivery)
        return message_ids if wait_recall else None


async def track_execution(
    records: ExecutionStore,
    request_id: str,
    execution: Awaitable[object],
) -> object:
    records.mark_running(request_id)
    try:
        result = await execution
    except asyncio.CancelledError:
        records.mark_failed(request_id, "CancelledError")
        raise
    except Exception as exc:
        records.mark_failed(request_id, type(exc).__name__)
        raise
    records.mark_succeeded(request_id)
    return result
