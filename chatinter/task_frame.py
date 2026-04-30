"""
ChatInter 会话任务帧。

TaskFrame 是短生命周期的内存状态，只服务于“上一轮缺参数，下一轮补参数”
和“继续/再来一个”这类续接，不进入数据库热路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import ClassVar, Literal

from .route_text import normalize_message_text

TaskFrameState = Literal["pending", "ready", "executed", "expired"]


@dataclass(frozen=True)
class TaskFrame:
    session_id: str
    plugin_module: str
    plugin_name: str
    command: str
    command_head: str
    source_message: str
    command_id: str | None = None
    skill_kind: str = "planner"
    missing_slots: tuple[str, ...] = ()
    filled_slots: dict[str, object] = field(default_factory=dict)
    context_tokens: tuple[str, ...] = ()
    state: TaskFrameState = "pending"
    created_at: float = 0.0
    updated_at: float = 0.0
    expires_at: float = 0.0

    @property
    def is_pending(self) -> bool:
        return self.state in {"pending", "ready"}

    @property
    def is_executed(self) -> bool:
        return self.state == "executed"


class TaskFrameStore:
    _frames: ClassVar[dict[str, TaskFrame]] = {}
    _ttl: ClassVar[float] = 10 * 60.0
    _max_size: ClassVar[int] = 512

    @classmethod
    def configure(
        cls,
        *,
        ttl: float | None = None,
        max_size: int | None = None,
    ) -> None:
        if ttl is not None:
            cls._ttl = max(float(ttl), 1.0)
        if max_size is not None:
            cls._max_size = max(int(max_size), 1)
        cls.prune()

    @classmethod
    def remember(
        cls,
        *,
        session_id: str | None,
        plugin_module: str,
        plugin_name: str,
        command: str,
        command_head: str | None = None,
        command_id: str | None = None,
        source_message: str = "",
        skill_kind: str = "planner",
        missing_slots: tuple[str, ...] | list[str] = (),
        filled_slots: dict[str, object] | None = None,
        context_tokens: tuple[str, ...] | list[str] = (),
        state: TaskFrameState = "pending",
        ttl: float | None = None,
    ) -> TaskFrame | None:
        sid = normalize_message_text(session_id or "")
        normalized_command = normalize_message_text(command)
        if not sid or not normalized_command:
            return None
        now = time.monotonic()
        ttl_value = max(float(ttl if ttl is not None else cls._ttl), 1.0)
        head = normalize_message_text(command_head or "")
        if not head:
            head = normalize_message_text(normalized_command.split(" ", 1)[0])
        frame = TaskFrame(
            session_id=sid,
            plugin_module=normalize_message_text(plugin_module),
            plugin_name=normalize_message_text(plugin_name),
            command=normalized_command,
            command_head=head,
            source_message=normalize_message_text(source_message),
            command_id=normalize_message_text(command_id or "") or None,
            skill_kind=normalize_message_text(skill_kind) or "planner",
            missing_slots=tuple(
                item
                for item in (
                    normalize_message_text(str(slot or "")) for slot in missing_slots
                )
                if item
            ),
            filled_slots=dict(filled_slots or {}),
            context_tokens=tuple(
                item
                for item in (
                    normalize_message_text(str(token or "")) for token in context_tokens
                )
                if item
            ),
            state=state,
            created_at=now,
            updated_at=now,
            expires_at=now + ttl_value,
        )
        cls._frames[sid] = frame
        cls.prune(now=now)
        return frame

    @classmethod
    def get(cls, session_id: str | None) -> TaskFrame | None:
        sid = normalize_message_text(session_id or "")
        if not sid:
            return None
        now = time.monotonic()
        cls.prune(now=now)
        frame = cls._frames.get(sid)
        if frame is None:
            return None
        if frame.expires_at <= now:
            cls._frames.pop(sid, None)
            return None
        return frame

    @classmethod
    def pop(cls, session_id: str | None) -> TaskFrame | None:
        sid = normalize_message_text(session_id or "")
        if not sid:
            return None
        return cls._frames.pop(sid, None)

    @classmethod
    def prune(cls, *, now: float | None = None) -> None:
        if not cls._frames:
            return
        now_value = now if now is not None else time.monotonic()
        expired = [
            session_id
            for session_id, frame in cls._frames.items()
            if frame.expires_at <= now_value or frame.state == "expired"
        ]
        for session_id in expired:
            cls._frames.pop(session_id, None)

        if len(cls._frames) <= cls._max_size:
            return
        overflow = len(cls._frames) - cls._max_size
        oldest = sorted(cls._frames.items(), key=lambda item: item[1].updated_at)
        for session_id, _frame in oldest[:overflow]:
            cls._frames.pop(session_id, None)

    @classmethod
    def clear(cls) -> None:
        cls._frames.clear()

    @classmethod
    def size(cls) -> int:
        cls.prune()
        return len(cls._frames)
