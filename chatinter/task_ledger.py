"""Structured task/capability ledgers for ChatInter agent coverage.

The LLM owns semantic task listing.  Local code only records what capabilities
were available/used and validates whether observations cover the listed tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid

from .route_text import normalize_message_text

TASK_STATUS_PENDING = "pending"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_UNSUPPORTED = "unsupported"
TASK_STATUS_BLOCKED = "blocked"
TASK_STATUS_FAILED = "failed"

TERMINAL_TASK_STATUSES = {
    TASK_STATUS_COMPLETED,
    TASK_STATUS_UNSUPPORTED,
    TASK_STATUS_BLOCKED,
    TASK_STATUS_FAILED,
}


@dataclass
class CapabilityLedgerEntry:
    tool_name: str
    command_id: str = ""
    plugin: str = ""
    head: str = ""
    role: str = ""
    output_mode: str = ""
    requires_real_tool: bool = False
    source_of_truth: str = ""
    used_count: int = 0
    last_ok: bool | None = None
    last_task_id: str = ""
    last_error: str = ""

    @classmethod
    def from_tool_summary(cls, payload: dict[str, Any]) -> "CapabilityLedgerEntry":
        return cls(
            tool_name=normalize_message_text(str(payload.get("tool", "") or "")),
            command_id=normalize_message_text(str(payload.get("command_id", "") or "")),
            plugin=normalize_message_text(str(payload.get("plugin", "") or "")),
            head=normalize_message_text(str(payload.get("head", "") or "")),
            role=normalize_message_text(str(payload.get("role", "") or "")),
            output_mode=normalize_message_text(
                str(payload.get("output_mode", "") or "")
            ),
            requires_real_tool=bool(payload.get("requires_real_tool")),
            source_of_truth=normalize_message_text(
                str(payload.get("source_of_truth", "") or "")
            ),
        )

    def record_observation(
        self,
        *,
        ok: bool,
        task_id: str = "",
        error: str = "",
    ) -> None:
        self.used_count += 1
        self.last_ok = bool(ok)
        self.last_task_id = normalize_message_text(task_id)
        self.last_error = normalize_message_text(error)[:300]

    def to_public_payload(self) -> dict[str, Any]:
        payload = {
            "tool_name": self.tool_name,
            "command_id": self.command_id,
            "plugin": self.plugin,
            "head": self.head,
            "role": self.role,
            "output_mode": self.output_mode,
            "requires_real_tool": self.requires_real_tool,
            "source_of_truth": self.source_of_truth,
            "used_count": self.used_count,
            "last_ok": self.last_ok,
            "last_task_id": self.last_task_id,
            "last_error": self.last_error,
        }
        return {key: value for key, value in payload.items() if value not in ("", None)}


@dataclass
class CapabilityLedger:
    entries: dict[str, CapabilityLedgerEntry] = field(default_factory=dict)

    @classmethod
    def from_tool_summaries(
        cls,
        tools: list[dict[str, Any]],
    ) -> "CapabilityLedger":
        ledger = cls()
        ledger.refresh_tools(tools)
        return ledger

    @classmethod
    def from_payload(cls, payload: Any) -> "CapabilityLedger":
        ledger = cls()
        if not isinstance(payload, dict):
            return ledger
        for item in payload.get("entries", []) or []:
            if not isinstance(item, dict):
                continue
            entry = CapabilityLedgerEntry(
                tool_name=normalize_message_text(
                    str(item.get("tool_name", "") or item.get("tool", "") or "")
                ),
                command_id=normalize_message_text(
                    str(item.get("command_id", "") or "")
                ),
                plugin=normalize_message_text(str(item.get("plugin", "") or "")),
                head=normalize_message_text(str(item.get("head", "") or "")),
                role=normalize_message_text(str(item.get("role", "") or "")),
                output_mode=normalize_message_text(
                    str(item.get("output_mode", "") or "")
                ),
                requires_real_tool=bool(item.get("requires_real_tool")),
                source_of_truth=normalize_message_text(
                    str(item.get("source_of_truth", "") or "")
                ),
                used_count=int(item.get("used_count", 0) or 0),
                last_ok=item.get("last_ok")
                if isinstance(item.get("last_ok"), bool)
                else None,
                last_task_id=normalize_message_text(
                    str(item.get("last_task_id", "") or "")
                ),
                last_error=normalize_message_text(
                    str(item.get("last_error", "") or "")
                ),
            )
            if entry.tool_name:
                ledger.entries[entry.tool_name] = entry
        return ledger

    def refresh_tools(self, tools: list[dict[str, Any]]) -> None:
        for payload in tools:
            if not isinstance(payload, dict):
                continue
            entry = CapabilityLedgerEntry.from_tool_summary(payload)
            if not entry.tool_name:
                continue
            existing = self.entries.get(entry.tool_name)
            if existing is None:
                self.entries[entry.tool_name] = entry
                continue
            existing.command_id = entry.command_id or existing.command_id
            existing.plugin = entry.plugin or existing.plugin
            existing.head = entry.head or existing.head
            existing.role = entry.role or existing.role
            existing.output_mode = entry.output_mode or existing.output_mode
            existing.requires_real_tool = (
                entry.requires_real_tool or existing.requires_real_tool
            )
            existing.source_of_truth = entry.source_of_truth or existing.source_of_truth

    def record_observation(
        self,
        *,
        tool_name: str,
        command_id: str = "",
        plugin: str = "",
        ok: bool,
        task_id: str = "",
        error: str = "",
    ) -> None:
        normalized_tool = normalize_message_text(tool_name)
        if not normalized_tool:
            return
        entry = self.entries.get(normalized_tool)
        if entry is None:
            entry = CapabilityLedgerEntry(
                tool_name=normalized_tool,
                command_id=normalize_message_text(command_id),
                plugin=normalize_message_text(plugin),
            )
            self.entries[normalized_tool] = entry
        entry.command_id = normalize_message_text(command_id) or entry.command_id
        entry.plugin = normalize_message_text(plugin) or entry.plugin
        entry.record_observation(ok=ok, task_id=task_id, error=error)

    def by_command_id(self, command_id: str) -> CapabilityLedgerEntry | None:
        normalized = normalize_message_text(command_id)
        if not normalized:
            return None
        for entry in self.entries.values():
            if entry.command_id == normalized:
                return entry
        return None

    def public_entries(self, *, limit: int = 40) -> list[dict[str, Any]]:
        max_items = int(limit or 0)
        entries = sorted(
            self.entries.values(),
            key=lambda item: (-(item.used_count or 0), item.tool_name),
        )
        if max_items > 0:
            entries = entries[:max_items]
        return [entry.to_public_payload() for entry in entries]

    def to_payload(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_public_payload() for entry in self.entries.values()]
        }


@dataclass
class TaskLedgerEntry:
    task_id: str
    goal: str
    intent_type: str = "unknown"
    requires_real_tool: bool = False
    expected_capabilities: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    status: str = TASK_STATUS_PENDING
    covered_by: list[str] = field(default_factory=list)
    unsupported_reason: str = ""
    reason: str = ""

    def to_public_payload(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "goal": self.goal,
            "intent_type": self.intent_type,
            "requires_real_tool": self.requires_real_tool,
            "expected_capabilities": list(self.expected_capabilities),
            "acceptance_criteria": list(self.acceptance_criteria),
            "status": self.status,
            "covered_by": list(self.covered_by),
            "unsupported_reason": self.unsupported_reason,
            "reason": self.reason,
        }
        return {
            key: value for key, value in payload.items() if value not in ("", [], None)
        }


@dataclass
class TaskLedger:
    original_message: str
    tasks: list[TaskLedgerEntry] = field(default_factory=list)
    ledger_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    reason: str = ""

    @classmethod
    def create(
        cls,
        *,
        original_message: str,
        tasks: list[TaskLedgerEntry] | None = None,
        reason: str = "",
    ) -> "TaskLedger":
        ledger = cls(
            original_message=normalize_message_text(original_message),
            tasks=list(tasks or []),
            reason=normalize_message_text(reason),
        )
        ledger.dedupe()
        return ledger

    @classmethod
    def from_payload(cls, payload: Any) -> "TaskLedger | None":
        if not isinstance(payload, dict):
            return None
        tasks: list[TaskLedgerEntry] = []
        for item in payload.get("tasks", []) or []:
            if not isinstance(item, dict):
                continue
            goal = normalize_message_text(str(item.get("goal", "") or ""))
            if not goal:
                continue
            tasks.append(
                TaskLedgerEntry(
                    task_id=normalize_message_text(str(item.get("task_id", "") or ""))
                    or f"task_{len(tasks) + 1}",
                    goal=goal,
                    intent_type=normalize_message_text(
                        str(item.get("intent_type", "") or "unknown")
                    ),
                    requires_real_tool=bool(item.get("requires_real_tool")),
                    expected_capabilities=_text_list(item.get("expected_capabilities")),
                    acceptance_criteria=_text_list(item.get("acceptance_criteria")),
                    status=_normalize_task_status(str(item.get("status", "") or "")),
                    covered_by=_text_list(item.get("covered_by")),
                    unsupported_reason=normalize_message_text(
                        str(item.get("unsupported_reason", "") or "")
                    ),
                    reason=normalize_message_text(str(item.get("reason", "") or "")),
                )
            )
        ledger = cls(
            ledger_id=normalize_message_text(str(payload.get("ledger_id", "") or ""))
            or uuid.uuid4().hex[:10],
            original_message=normalize_message_text(
                str(payload.get("original_message", "") or "")
            ),
            tasks=tasks,
            reason=normalize_message_text(str(payload.get("reason", "") or "")),
        )
        ledger.dedupe()
        return ledger

    @property
    def pending_tasks(self) -> list[TaskLedgerEntry]:
        return [
            task for task in self.tasks if task.status not in TERMINAL_TASK_STATUSES
        ]

    @property
    def incomplete_goals(self) -> list[str]:
        return [task.goal for task in self.pending_tasks if task.goal]

    def dedupe(self) -> None:
        result: list[TaskLedgerEntry] = []
        seen: set[str] = set()
        for index, task in enumerate(self.tasks, 1):
            task.goal = normalize_message_text(task.goal)
            if not task.goal:
                continue
            key = task.goal.casefold()
            if key in seen:
                continue
            seen.add(key)
            if not normalize_message_text(task.task_id):
                task.task_id = f"task_{index}"
            task.task_id = normalize_message_text(task.task_id)
            task.intent_type = normalize_message_text(task.intent_type) or "unknown"
            task.expected_capabilities = _text_list(task.expected_capabilities)
            task.acceptance_criteria = _text_list(task.acceptance_criteria)
            task.covered_by = _text_list(task.covered_by)
            task.status = _normalize_task_status(task.status)
            result.append(task)
        self.tasks = result[:12]

    def apply_coverage(
        self,
        *,
        covered_task_ids: list[str],
        unsupported_tasks: list[dict[str, Any]] | list[str],
        failed_tasks: list[dict[str, Any]] | list[str] | None = None,
        reason: str = "",
    ) -> None:
        covered = {normalize_message_text(item) for item in covered_task_ids}
        unsupported = _task_reason_map(unsupported_tasks)
        failed = _task_reason_map(failed_tasks or [])
        for task in self.tasks:
            if task.task_id in covered:
                task.status = TASK_STATUS_COMPLETED
                if "observation" not in task.covered_by:
                    task.covered_by.append("observation")
            elif task.task_id in unsupported:
                task.status = TASK_STATUS_UNSUPPORTED
                task.unsupported_reason = unsupported[task.task_id]
            elif task.task_id in failed:
                task.status = TASK_STATUS_FAILED
                task.reason = failed[task.task_id]
        if reason:
            self.reason = normalize_message_text(reason)

    def to_payload(self) -> dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "original_message": self.original_message,
            "reason": self.reason,
            "tasks": [task.to_public_payload() for task in self.tasks],
        }

    def to_public_payload(self) -> dict[str, Any]:
        return self.to_payload()


def task_ledger_from_payload(payload: Any) -> TaskLedger | None:
    return TaskLedger.from_payload(payload)


def capability_ledger_from_payload(payload: Any) -> CapabilityLedger:
    return CapabilityLedger.from_payload(payload)


def _text_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[str] = []
    for item in value:
        text = normalize_message_text(str(item or ""))
        if text and text not in result:
            result.append(text[:240])
    return result[: max(1, int(limit or 12))]


def _task_reason_map(value: list[dict[str, Any]] | list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(value, list | tuple):
        return result
    for item in value:
        if isinstance(item, dict):
            task_id = normalize_message_text(str(item.get("task_id", "") or ""))
            reason = normalize_message_text(str(item.get("reason", "") or ""))
        else:
            task_id = normalize_message_text(str(item or ""))
            reason = ""
        if task_id:
            result[task_id] = reason
    return result


def _normalize_task_status(status: str) -> str:
    normalized = normalize_message_text(status)
    if normalized in {
        TASK_STATUS_PENDING,
        TASK_STATUS_COMPLETED,
        TASK_STATUS_UNSUPPORTED,
        TASK_STATUS_BLOCKED,
        TASK_STATUS_FAILED,
    }:
        return normalized
    return TASK_STATUS_PENDING


__all__ = [
    "TASK_STATUS_BLOCKED",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_FAILED",
    "TASK_STATUS_PENDING",
    "TASK_STATUS_UNSUPPORTED",
    "TERMINAL_TASK_STATUSES",
    "CapabilityLedger",
    "CapabilityLedgerEntry",
    "TaskLedger",
    "TaskLedgerEntry",
    "capability_ledger_from_payload",
    "task_ledger_from_payload",
]
