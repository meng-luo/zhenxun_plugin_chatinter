from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Literal

from tortoise import Tortoise
from tortoise.expressions import Q

from zhenxun.services.llm.types.models import ToolDefinition, ToolResult

from .models.chat_history import ChatInterChatHistory
from .route_text import normalize_message_text

_FTS_TABLE = "chatinter_session_search_fts"
_META_TABLE = "chatinter_session_search_meta"
_INDEX_BATCH_SIZE = 500
_MAX_INDEX_BATCHES_PER_SEARCH = 20
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,8}", re.IGNORECASE)
_DDL_LOCK = asyncio.Lock()
_SYNC_LOCK = asyncio.Lock()
_FTS_READY: bool | None = None

SearchMode = Literal["discovery", "scroll", "browse"]


@dataclass(frozen=True)
class SessionSearchHit:
    id: int
    session_id: str
    user_id: str
    group_id: str | None
    nickname: str
    create_time: str
    user_message: str
    ai_response: str
    snippet: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "nickname": self.nickname,
            "create_time": self.create_time,
            "user_message": self.user_message,
            "ai_response": self.ai_response,
            "snippet": self.snippet,
        }
        return {key: value for key, value in payload.items() if value not in ("", None)}


class SessionSearchTool:
    name = "session_search"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "检索当前 ChatInter 会话的历史记录。它不调用模型，只读取本地"
                "会话历史。支持 discovery=关键词发现、scroll=围绕历史 id "
                "查看上下文、browse=按时间浏览。默认只搜索当前 session。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["discovery", "scroll", "browse"],
                        "description": "检索模式。",
                    },
                    "query": {
                        "type": ["string", "null"],
                        "description": "discovery 模式的关键词或自然语言查询。",
                    },
                    "session_id": {
                        "type": ["string", "null"],
                        "description": (
                            "要检索的会话 id；留空时使用当前会话，避免跨群/私聊泄露。"
                        ),
                    },
                    "anchor_id": {
                        "type": ["integer", "null"],
                        "description": "scroll/browse 的锚点历史 id。",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "最多返回条数，默认 8，最大 30。",
                    },
                    "before": {
                        "type": ["integer", "null"],
                        "description": "scroll 模式锚点前条数，默认 4。",
                    },
                    "after": {
                        "type": ["integer", "null"],
                        "description": "scroll 模式锚点后条数，默认 4。",
                    },
                    "direction": {
                        "type": ["string", "null"],
                        "enum": ["backward", "forward", None],
                        "description": "browse 模式翻页方向，默认 backward。",
                    },
                },
                "required": [
                    "mode",
                    "query",
                    "session_id",
                    "anchor_id",
                    "limit",
                    "before",
                    "after",
                    "direction",
                ],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        current_session_id = _context_session_id(context)
        requested_session_id = normalize_message_text(
            str(kwargs.get("session_id", "") or "")
        )
        session_id = requested_session_id or current_session_id
        if not session_id:
            return _tool_result(
                ok=False,
                status="session_id_required",
                display="缺少当前会话 id，无法检索历史。",
            )
        mode = _coerce_mode(kwargs.get("mode"))
        limit = _coerce_int(kwargs.get("limit"), default=8, minimum=1, maximum=30)
        query = normalize_message_text(str(kwargs.get("query", "") or ""))
        anchor_id = _coerce_optional_int(kwargs.get("anchor_id"))

        if mode == "discovery":
            if not query:
                return _tool_result(
                    ok=False,
                    status="query_required",
                    mode=mode,
                    session_id=session_id,
                    display="缺少检索关键词。",
                )
            result = await search_session_history(
                session_id=session_id,
                query=query,
                limit=limit,
            )
        elif mode == "scroll":
            if not anchor_id:
                return _tool_result(
                    ok=False,
                    status="anchor_id_required",
                    mode=mode,
                    session_id=session_id,
                    display="scroll 模式需要 anchor_id。",
                )
            result = await scroll_session_history(
                session_id=session_id,
                anchor_id=anchor_id,
                before=_coerce_int(
                    kwargs.get("before"), default=4, minimum=0, maximum=15
                ),
                after=_coerce_int(
                    kwargs.get("after"), default=4, minimum=0, maximum=15
                ),
            )
        else:
            result = await browse_session_history(
                session_id=session_id,
                anchor_id=anchor_id,
                limit=limit,
                direction=_coerce_direction(kwargs.get("direction")),
            )

        return _tool_result(
            ok=True,
            status="session_history_retrieved",
            mode=mode,
            session_id=session_id,
            query=query or None,
            anchor_id=anchor_id,
            count=len(result),
            results=[item.to_payload() for item in result],
            display=f"检索到 {len(result)} 条会话历史。",
        )


async def upsert_session_search_dialog(dialog: ChatInterChatHistory) -> None:
    if not await _ensure_fts_ready():
        return
    try:
        db = Tortoise.get_connection("default")
        await db.execute_query(
            f"INSERT OR REPLACE INTO {_FTS_TABLE} "
            "(rowid, content, session_id, user_id, group_id, nickname, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                int(dialog.id),
                _index_content(dialog),
                str(dialog.session_id or ""),
                str(dialog.user_id or ""),
                str(dialog.group_id or ""),
                str(dialog.nickname or ""),
                _format_time(dialog.create_time),
            ],
        )
    except Exception:
        return


async def search_session_history(
    *,
    session_id: str,
    query: str,
    limit: int = 8,
) -> list[SessionSearchHit]:
    normalized_query = normalize_message_text(query)
    if not session_id or not normalized_query:
        return []
    if await _ensure_fts_ready():
        await _sync_fts_index()
        hits = await _search_fts(
            session_id=session_id,
            query=normalized_query,
            limit=limit,
        )
        if hits:
            return hits
    return await _search_fallback(
        session_id=session_id,
        query=normalized_query,
        limit=limit,
    )


async def scroll_session_history(
    *,
    session_id: str,
    anchor_id: int,
    before: int = 4,
    after: int = 4,
) -> list[SessionSearchHit]:
    if not session_id or anchor_id <= 0:
        return []
    anchor = await ChatInterChatHistory.filter(
        id=anchor_id,
        session_id=session_id,
        reset=False,
    ).first()
    if anchor is None:
        return []
    before_rows = (
        await ChatInterChatHistory.filter(
            session_id=session_id,
            reset=False,
            id__lt=anchor_id,
        )
        .order_by("-id")
        .limit(max(before, 0))
    )
    after_rows = (
        await ChatInterChatHistory.filter(
            session_id=session_id,
            reset=False,
            id__gt=anchor_id,
        )
        .order_by("id")
        .limit(max(after, 0))
    )
    rows = [*reversed(before_rows), anchor, *after_rows]
    return [_hit_from_dialog(row) for row in rows]


async def browse_session_history(
    *,
    session_id: str,
    anchor_id: int | None = None,
    limit: int = 8,
    direction: Literal["backward", "forward"] = "backward",
) -> list[SessionSearchHit]:
    if not session_id:
        return []
    query = ChatInterChatHistory.filter(session_id=session_id, reset=False)
    if anchor_id and anchor_id > 0:
        query = (
            query.filter(id__gt=anchor_id)
            if direction == "forward"
            else query.filter(id__lt=anchor_id)
        )
    query = query.order_by("id" if direction == "forward" else "-id").limit(limit)
    rows = await query
    if direction == "backward":
        rows = list(reversed(rows))
    return [_hit_from_dialog(row) for row in rows]


async def _ensure_fts_ready() -> bool:
    global _FTS_READY
    if _FTS_READY is not None:
        return _FTS_READY
    async with _DDL_LOCK:
        if _FTS_READY is not None:
            return _FTS_READY
        if _connection_dialect() != "sqlite":
            _FTS_READY = False
            return False
        try:
            db = Tortoise.get_connection("default")
            await db.execute_query(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} "
                "USING fts5("
                "content, "
                "session_id UNINDEXED, "
                "user_id UNINDEXED, "
                "group_id UNINDEXED, "
                "nickname UNINDEXED, "
                "created_at UNINDEXED, "
                "tokenize='unicode61'"
                ")"
            )
            await db.execute_query(
                f"CREATE TABLE IF NOT EXISTS {_META_TABLE} "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            _FTS_READY = True
        except Exception:
            _FTS_READY = False
        return _FTS_READY


async def _sync_fts_index() -> None:
    if not await _ensure_fts_ready():
        return
    async with _SYNC_LOCK:
        last_id = await _get_last_indexed_id()
        max_seen = last_id
        for _ in range(_MAX_INDEX_BATCHES_PER_SEARCH):
            rows = (
                await ChatInterChatHistory.filter(id__gt=max_seen, reset=False)
                .order_by("id")
                .limit(_INDEX_BATCH_SIZE)
            )
            if not rows:
                break
            db = Tortoise.get_connection("default")
            for row in rows:
                await db.execute_query(
                    f"INSERT OR REPLACE INTO {_FTS_TABLE} "
                    "(rowid, content, session_id, user_id, group_id, nickname, "
                    "created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        int(row.id),
                        _index_content(row),
                        str(row.session_id or ""),
                        str(row.user_id or ""),
                        str(row.group_id or ""),
                        str(row.nickname or ""),
                        _format_time(row.create_time),
                    ],
                )
                max_seen = max(max_seen, int(row.id))
            await _set_last_indexed_id(max_seen)
            if len(rows) < _INDEX_BATCH_SIZE:
                break


async def _search_fts(
    *,
    session_id: str,
    query: str,
    limit: int,
) -> list[SessionSearchHit]:
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []
    try:
        db = Tortoise.get_connection("default")
        rows = await db.execute_query_dict(
            f"""
            SELECT
                h.id,
                h.session_id,
                h.user_id,
                h.group_id,
                h.nickname,
                h.user_message,
                h.ai_response,
                h.create_time,
                snippet({_FTS_TABLE}, 0, '[', ']', '...', 18) AS snippet,
                bm25({_FTS_TABLE}) AS rank
            FROM {_FTS_TABLE}
            JOIN chatinter_chat_history h ON h.id = {_FTS_TABLE}.rowid
            WHERE {_FTS_TABLE} MATCH ?
              AND {_FTS_TABLE}.session_id = ?
              AND h.reset = 0
            ORDER BY rank, h.id DESC
            LIMIT ?
            """,
            [fts_query, session_id, max(limit, 1)],
        )
    except Exception:
        return []
    return [_hit_from_row(row) for row in rows]


async def _search_fallback(
    *,
    session_id: str,
    query: str,
    limit: int,
) -> list[SessionSearchHit]:
    rows = (
        await ChatInterChatHistory.filter(session_id=session_id, reset=False)
        .filter(
            Q(user_message__icontains=query)
            | Q(ai_response__icontains=query)
            | Q(timeline__icontains=query)
        )
        .order_by("-id")
        .limit(max(limit, 1))
    )
    return [_hit_from_dialog(row, snippet=_make_snippet(row, query)) for row in rows]


async def _get_last_indexed_id() -> int:
    try:
        db = Tortoise.get_connection("default")
        rows = await db.execute_query_dict(
            f"SELECT value FROM {_META_TABLE} WHERE key = ?",
            ["last_indexed_id"],
        )
        if rows:
            return max(int(rows[0].get("value", 0) or 0), 0)
    except Exception:
        return 0
    return 0


async def _set_last_indexed_id(value: int) -> None:
    try:
        db = Tortoise.get_connection("default")
        await db.execute_query(
            f"INSERT OR REPLACE INTO {_META_TABLE} (key, value) VALUES (?, ?)",
            ["last_indexed_id", str(max(int(value), 0))],
        )
    except Exception:
        return


def _index_content(dialog: ChatInterChatHistory) -> str:
    text = " ".join(
        part
        for part in (
            str(dialog.user_message or ""),
            str(dialog.ai_response or ""),
            _timeline_text(dialog),
        )
        if part
    )
    terms = " ".join(_search_terms(text))
    return f"{text}\n{terms}".strip()


def _timeline_text(dialog: ChatInterChatHistory) -> str:
    parts: list[str] = []
    for item in dialog.get_timeline():
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(part) for part in content if isinstance(part, str))
    return " ".join(parts)


def _search_terms(text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(normalize_message_text(text)):
        lowered = token.casefold()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        result.append(lowered)
        chars = "".join(char for char in lowered if "\u4e00" <= char <= "\u9fff")
        max_size = min(len(chars), 4)
        for size in range(2, max_size + 1):
            for start in range(0, len(chars) - size + 1):
                gram = chars[start : start + size]
                if gram not in seen:
                    seen.add(gram)
                    result.append(gram)
    return result


def _build_fts_query(query: str) -> str:
    terms = _search_terms(query)[:16]
    if not terms:
        return ""
    return " OR ".join(
        f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms
    )


def _hit_from_dialog(
    dialog: ChatInterChatHistory,
    *,
    snippet: str = "",
) -> SessionSearchHit:
    return SessionSearchHit(
        id=int(dialog.id),
        session_id=str(dialog.session_id or ""),
        user_id=str(dialog.user_id or ""),
        group_id=str(dialog.group_id or "") or None,
        nickname=str(dialog.nickname or ""),
        create_time=_format_time(dialog.create_time),
        user_message=_compact_text(str(dialog.user_message or ""), 260),
        ai_response=_compact_text(str(dialog.ai_response or ""), 260),
        snippet=snippet,
    )


def _hit_from_row(row: dict[str, Any]) -> SessionSearchHit:
    return SessionSearchHit(
        id=int(row.get("id", 0) or 0),
        session_id=str(row.get("session_id", "") or ""),
        user_id=str(row.get("user_id", "") or ""),
        group_id=str(row.get("group_id") or "") or None,
        nickname=str(row.get("nickname", "") or ""),
        create_time=_format_time(row.get("create_time")),
        user_message=_compact_text(str(row.get("user_message", "") or ""), 260),
        ai_response=_compact_text(str(row.get("ai_response", "") or ""), 260),
        snippet=_compact_text(str(row.get("snippet", "") or ""), 320),
    )


def _make_snippet(dialog: ChatInterChatHistory, query: str) -> str:
    text = " ".join(
        part
        for part in (
            str(dialog.user_message or ""),
            str(dialog.ai_response or ""),
            _timeline_text(dialog),
        )
        if part
    )
    normalized = normalize_message_text(text)
    needle = normalize_message_text(query)
    index = normalized.casefold().find(needle.casefold()) if needle else -1
    if index < 0:
        return _compact_text(normalized, 260)
    start = max(index - 80, 0)
    end = min(index + len(needle) + 120, len(normalized))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(normalized) else ""
    return f"{prefix}{normalized[start:end]}{suffix}"


def _context_session_id(context: Any | None) -> str:
    return normalize_message_text(str(getattr(context, "session_id", "") or ""))


def _connection_dialect() -> str:
    try:
        connection = Tortoise.get_connection("default")
        capabilities = getattr(connection, "capabilities", None)
        return str(getattr(capabilities, "dialect", "") or "").lower()
    except Exception:
        return ""


def _coerce_mode(value: Any) -> SearchMode:
    text = normalize_message_text(str(value or "")).lower()
    if text in {"scroll", "browse"}:
        return text  # pyright: ignore[reportReturnType]
    return "discovery"


def _coerce_direction(value: Any) -> Literal["backward", "forward"]:
    text = normalize_message_text(str(value or "")).lower()
    return "forward" if text == "forward" else "backward"


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _coerce_optional_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _format_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    text = str(value or "").strip()
    return text


def _compact_text(text: str, limit: int) -> str:
    normalized = normalize_message_text(text)
    max_len = max(int(limit or 0), 0)
    if max_len <= 0 or len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


def _tool_result(
    ok: bool,
    status: str,
    *,
    display: str,
    **payload: Any,
) -> ToolResult:
    output = {"ok": ok, "status": status}
    output.update({key: value for key, value in payload.items() if value is not None})
    return ToolResult(output=output, display_content=display)


__all__ = [
    "SessionSearchHit",
    "SessionSearchTool",
    "browse_session_history",
    "scroll_session_history",
    "search_session_history",
    "upsert_session_search_dialog",
]
