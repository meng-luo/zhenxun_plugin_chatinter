"""Durable Todo state for the superuser Agent engineering loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
import uuid
from typing import Any, Literal

from ..persistence import read_json, state_path, write_json
from ..route_text import normalize_message_text
from .audit_log import record_audit_event

TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
TodoPriority = Literal["low", "medium", "high"]

_TODOS_PATH = state_path("superuser_todos.json")
_TODOS: dict[str, "AgentTodoList"] = {}
_LOADED = False


@dataclass
class AgentTodo:
    todo_id: str
    content: str
    status: TodoStatus = "pending"
    priority: TodoPriority = "medium"
    active_form: str = ""
    related_tools: list[str] = field(default_factory=list)
    related_artifacts: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def public_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = int(self.created_at)
        payload["updated_at"] = int(self.updated_at)
        return payload


@dataclass
class AgentTodoList:
    user_id: str
    session_key: str
    todos: list[AgentTodo] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def public_payload(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_key": self.session_key,
            "updated_at": int(self.updated_at),
            "summary": todo_summary(self.todos),
            "todos": [todo.public_payload() for todo in self.todos],
        }

    def to_record(self) -> dict[str, Any]:
        return self.public_payload()


def read_todos(*, user_id: str, session_key: str) -> AgentTodoList:
    _ensure_loaded()
    key = _todo_key(user_id, session_key)
    todo_list = _TODOS.get(key)
    if todo_list is None:
        todo_list = AgentTodoList(
            user_id=str(user_id or ""),
            session_key=str(session_key or ""),
        )
        _TODOS[key] = todo_list
        _save_todos()
    return todo_list


def write_todos(
    *,
    user_id: str,
    session_key: str,
    todos: list[dict[str, Any]],
    replace: bool = True,
) -> AgentTodoList:
    _ensure_loaded()
    old_list = read_todos(user_id=user_id, session_key=session_key)
    if replace:
        next_todos = [_todo_from_input(item) for item in todos]
    else:
        next_todos = list(old_list.todos)
        index_by_id = {todo.todo_id: index for index, todo in enumerate(next_todos)}
        for raw in todos:
            todo = _todo_from_input(raw)
            old_index = index_by_id.get(todo.todo_id)
            if old_index is None:
                index_by_id[todo.todo_id] = len(next_todos)
                next_todos.append(todo)
            else:
                created_at = next_todos[old_index].created_at
                todo.created_at = created_at
                next_todos[old_index] = todo
    todo_list = AgentTodoList(
        user_id=str(user_id or ""),
        session_key=str(session_key or ""),
        todos=_dedupe_todos(next_todos),
        updated_at=time.time(),
    )
    _TODOS[_todo_key(user_id, session_key)] = todo_list
    _save_todos()
    record_audit_event(
        event="agent_todos_updated",
        user_id=todo_list.user_id,
        session_key=todo_list.session_key,
        action="todo_write",
        payload={
            "replace": replace,
            "count": len(todo_list.todos),
            "summary": todo_summary(todo_list.todos),
        },
        result={"ok": True},
    )
    return todo_list


def update_todo_from_observation(
    *,
    user_id: str,
    session_key: str,
    observation: dict[str, Any],
) -> AgentTodoList | None:
    """Conservatively mark matching in-progress todos from tool evidence."""

    _ensure_loaded()
    todo_list = _TODOS.get(_todo_key(user_id, session_key))
    if todo_list is None:
        return None
    if not todo_list.todos:
        return todo_list
    task_text = normalize_message_text(str(observation.get("task_text", "") or ""))
    status = normalize_message_text(str(observation.get("status", "") or ""))
    ok = bool(observation.get("ok"))
    tool_name = normalize_message_text(str(observation.get("tool_name", "") or ""))
    artifacts = [
        normalize_message_text(str(item.get("artifact_id", "") or ""))
        for item in observation.get("artifacts", []) or []
        if isinstance(item, dict)
    ]
    changed = False
    for todo in todo_list.todos:
        if todo.status == "completed":
            continue
        if not _observation_matches_todo(todo, task_text, tool_name):
            continue
        if tool_name and tool_name not in todo.related_tools:
            todo.related_tools.append(tool_name)
        for artifact_id in artifacts:
            if artifact_id and artifact_id not in todo.related_artifacts:
                todo.related_artifacts.append(artifact_id)
        if ok and status not in {"approval_required", "background_task_started"}:
            todo.status = "completed"
        elif todo.status == "pending":
            todo.status = "in_progress"
        todo.updated_at = time.time()
        changed = True
    if changed:
        todo_list.updated_at = time.time()
        _save_todos()
    return todo_list


def todo_summary(todos: list[AgentTodo]) -> dict[str, int]:
    summary = {"pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}
    for todo in todos:
        summary[todo.status] = summary.get(todo.status, 0) + 1
    return summary


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    raw = read_json(_TODOS_PATH, {})
    if not isinstance(raw, dict):
        return
    for key, payload in raw.items():
        item = _todo_list_from_payload(key, payload)
        if item is not None:
            _TODOS[_todo_key(item.user_id, item.session_key)] = item


def _save_todos() -> None:
    write_json(
        _TODOS_PATH,
        {key: todo_list.to_record() for key, todo_list in sorted(_TODOS.items())},
    )


def _todo_list_from_payload(
    key: object,
    payload: object,
) -> AgentTodoList | None:
    if not isinstance(payload, dict):
        return None
    user_id = str(payload.get("user_id", "") or "")
    session_key = str(payload.get("session_key", "") or "")
    if not user_id and isinstance(key, str) and "|" in key:
        user_id, session_key = key.split("|", 1)
    try:
        return AgentTodoList(
            user_id=user_id,
            session_key=session_key,
            todos=[
                _todo_from_payload(item)
                for item in payload.get("todos", []) or []
                if isinstance(item, dict)
            ],
            updated_at=float(payload.get("updated_at") or time.time()),
        )
    except Exception:
        return None


def _todo_from_payload(payload: dict[str, Any]) -> AgentTodo:
    return AgentTodo(
        todo_id=normalize_message_text(str(payload.get("todo_id", "") or ""))
        or uuid.uuid4().hex[:10],
        content=normalize_message_text(str(payload.get("content", "") or "")),
        status=_normalize_status(str(payload.get("status", "") or "")),
        priority=_normalize_priority(str(payload.get("priority", "") or "")),
        active_form=normalize_message_text(str(payload.get("active_form", "") or "")),
        related_tools=[
            normalize_message_text(str(item or ""))
            for item in payload.get("related_tools", []) or []
            if normalize_message_text(str(item or ""))
        ][:12],
        related_artifacts=[
            normalize_message_text(str(item or ""))
            for item in payload.get("related_artifacts", []) or []
            if normalize_message_text(str(item or ""))
        ][:24],
        created_at=float(payload.get("created_at") or time.time()),
        updated_at=float(payload.get("updated_at") or time.time()),
    )


def _todo_from_input(raw: dict[str, Any]) -> AgentTodo:
    content = normalize_message_text(str(raw.get("content", "") or ""))
    if not content:
        raise ValueError("todo.content is required")
    now = time.time()
    return AgentTodo(
        todo_id=normalize_message_text(str(raw.get("todo_id", "") or ""))
        or uuid.uuid4().hex[:10],
        content=content,
        status=_normalize_status(str(raw.get("status", "") or "")),
        priority=_normalize_priority(str(raw.get("priority", "") or "")),
        active_form=normalize_message_text(str(raw.get("active_form", "") or "")),
        related_tools=_text_list(raw.get("related_tools"), limit=12),
        related_artifacts=_text_list(raw.get("related_artifacts"), limit=24),
        created_at=now,
        updated_at=now,
    )


def _dedupe_todos(todos: list[AgentTodo]) -> list[AgentTodo]:
    result: list[AgentTodo] = []
    seen: set[str] = set()
    for todo in todos[:80]:
        key = todo.todo_id or todo.content
        if not todo.content or key in seen:
            continue
        seen.add(key)
        result.append(todo)
    return result


def _observation_matches_todo(todo: AgentTodo, task_text: str, tool_name: str) -> bool:
    haystack = " ".join([todo.content, todo.active_form, *todo.related_tools])
    normalized = normalize_message_text(haystack).lower()
    if tool_name and tool_name.lower() in normalized:
        return True
    if task_text:
        lowered = task_text.lower()
        return bool(lowered in normalized or normalized in lowered)
    return todo.status == "in_progress"


def _text_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[str] = []
    for item in value:
        text = normalize_message_text(str(item or ""))
        if text and text not in result:
            result.append(text)
    return result[: max(1, limit)]


def _normalize_status(status: str) -> TodoStatus:
    normalized = normalize_message_text(status)
    if normalized in {"pending", "in_progress", "completed", "cancelled"}:
        return normalized  # type: ignore[return-value]
    return "pending"


def _normalize_priority(priority: str) -> TodoPriority:
    normalized = normalize_message_text(priority)
    if normalized in {"low", "medium", "high"}:
        return normalized  # type: ignore[return-value]
    return "medium"


def _todo_key(user_id: str, session_key: str) -> str:
    return "|".join([str(user_id or ""), str(session_key or "")])


__all__ = [
    "AgentTodo",
    "AgentTodoList",
    "TodoPriority",
    "TodoStatus",
    "read_todos",
    "todo_summary",
    "update_todo_from_observation",
    "write_todos",
]
