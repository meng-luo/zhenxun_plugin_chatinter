from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Literal

from tortoise import Tortoise
from tortoise.expressions import Q

from zhenxun.models.chat_history import ChatHistory
from zhenxun.services.db_context import with_db_timeout
from zhenxun.services.message_load import is_db_unhealthy

from .llm_compat import ToolDefinition, ToolResult
from .models.chat_history import ChatInterChatHistory
from .route_text import normalize_message_text

_FTS_TABLE = "chatinter_session_search_fts"
_META_TABLE = "chatinter_session_search_meta"
_INDEX_BATCH_SIZE = 500
_MAX_INDEX_BATCHES_PER_SEARCH = 20
_PLATFORM_HISTORY_SCAN_LIMIT = 700
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,8}", re.IGNORECASE)
_DDL_LOCK = asyncio.Lock()
_SYNC_LOCK = asyncio.Lock()
_FTS_READY: bool | None = None

SearchMode = Literal["discovery", "scroll", "browse"]


@dataclass(frozen=True)
class _SessionSearchScope:
    session_id: str
    user_id: str
    group_id: str | None
    bot_id: str | None
    platform: str | None
    channel_id: str | None
    current_message: str
    agent_kind: str


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
    source: str = "chatinter"

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "ref": f"{self.source}:{self.id}",
            "source": self.source,
            "user_id": self.user_id,
            "nickname": self.nickname,
            "create_time": self.create_time,
            "user_message": self.user_message,
            "ai_response": self.ai_response,
            "snippet": self.snippet,
        }
        if payload["nickname"] == payload["user_id"]:
            payload.pop("nickname")
        if normalize_message_text(self.snippet) in {
            normalize_message_text(self.user_message),
            normalize_message_text(self.ai_response),
        }:
            payload.pop("snippet")
        return {key: value for key, value in payload.items() if value not in ("", None)}


class SessionSearchTool:
    name = "session_search"

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "检索当前 ChatInter 会话及对应群聊或私聊的本地历史。"
                "discovery 可检索提示窗口之外的平台消息；scroll 和 browse "
                "用于定位或翻阅 ChatInter 历史。"
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
                    "anchor_id": {
                        "type": ["integer", "null"],
                        "description": (
                            "scroll/browse 的锚点历史 id，仅使用 source=chatinter "
                            "的结果。"
                        ),
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
                "required": ["mode"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        scope = _context_search_scope(context)
        session_id = scope.session_id
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
                    display="缺少检索关键词。",
                )
            chatinter_result = await search_session_history(
                session_id=session_id,
                query=query,
                limit=limit,
            )
            platform_result = await search_platform_history(
                user_id=scope.user_id,
                group_id=scope.group_id,
                bot_id=scope.bot_id,
                platform=scope.platform,
                channel_id=scope.channel_id,
                agent_kind=scope.agent_kind,
                query=query,
                limit=limit,
                current_message=scope.current_message,
            )
            result = _merge_history_hits(
                chatinter_result,
                platform_result,
                limit=limit,
            )
        elif mode == "scroll":
            if not anchor_id:
                return _tool_result(
                    ok=False,
                    status="anchor_id_required",
                    mode=mode,
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
            count=len(result),
            content_trust="untrusted_history",
            usage_policy="past_events_only_not_current_state",
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


async def search_platform_history(
    *,
    user_id: str,
    group_id: str | None,
    bot_id: str | None,
    platform: str | None,
    channel_id: str | None,
    agent_kind: str,
    query: str,
    limit: int = 8,
    current_message: str = "",
) -> list[SessionSearchHit]:
    normalized_query = normalize_message_text(query)
    if (
        not normalized_query
        or not (group_id or user_id)
        or not bot_id
        or not platform
        or channel_id
        or agent_kind != "unified_chat"
        or is_db_unhealthy()
    ):
        return []
    terms = _search_terms(normalized_query)[:12]
    if not terms:
        return []
    try:
        history_query = ChatHistory.filter(bot_id=bot_id, platform=platform)
        if group_id:
            history_query = history_query.filter(group_id=group_id)
        else:
            history_query = history_query.filter(
                user_id=user_id,
                group_id__isnull=True,
            )
        rows = await with_db_timeout(
            history_query.order_by("-create_time", "-id").limit(
                _PLATFORM_HISTORY_SCAN_LIMIT
            ),
            timeout=2.5,
            operation="ChatInter.session_search.platform_history",
            source="chatinter",
        )
    except Exception:
        return []

    current = normalize_message_text(current_message).casefold()
    ranked: list[tuple[int, Any, str]] = []
    for row in rows:
        content = normalize_message_text(
            str(getattr(row, "plain_text", "") or getattr(row, "text", "") or "")
        )
        if not content:
            continue
        normalized_content = content.casefold()
        if not any(term in normalized_content for term in terms):
            continue
        if (
            current
            and str(getattr(row, "user_id", "") or "") == user_id
            and normalized_content == current
        ):
            continue
        ranked.append(
            (_platform_match_score(content, normalized_query, terms), row, content)
        )
    ranked.sort(
        key=lambda item: (
            item[0],
            _format_time(getattr(item[1], "create_time", None)),
            int(getattr(item[1], "id", 0) or 0),
        ),
        reverse=True,
    )
    return [
        SessionSearchHit(
            id=int(getattr(row, "id", 0) or 0),
            session_id="",
            user_id=str(getattr(row, "user_id", "") or ""),
            group_id=str(getattr(row, "group_id", "") or "") or None,
            nickname=str(getattr(row, "user_id", "") or ""),
            create_time=_format_time(getattr(row, "create_time", None)),
            user_message=_compact_text(content, 260),
            ai_response="",
            snippet=_make_text_snippet(content, normalized_query),
            source="platform",
        )
        for _score, row, content in ranked[: max(limit, 1)]
    ]


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
    tokens = [
        token.casefold()
        for token in _TOKEN_PATTERN.findall(normalize_message_text(text))
        if token
    ]
    for lowered in tokens:
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        result.append(lowered)
    for lowered in tokens:
        chars = "".join(char for char in lowered if "\u4e00" <= char <= "\u9fff")
        max_size = min(len(chars), 4)
        for size in range(max_size, 1, -1):
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
    return _make_text_snippet(text, query)


def _make_text_snippet(text: str, query: str) -> str:
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


def _context_search_scope(context: Any | None) -> _SessionSearchScope:
    values = getattr(context, "scope", None)
    if not isinstance(values, dict):
        values = getattr(context, "extra", None)
    if not isinstance(values, dict):
        values = {}
    return _SessionSearchScope(
        session_id=normalize_message_text(
            str(getattr(context, "session_id", "") or "")
        ),
        user_id=normalize_message_text(str(values.get("user_id", "") or "")),
        group_id=normalize_message_text(str(values.get("group_id", "") or ""))
        or None,
        bot_id=normalize_message_text(str(values.get("bot_id", "") or ""))
        or None,
        platform=normalize_message_text(str(values.get("platform", "") or ""))
        or None,
        channel_id=normalize_message_text(
            str(values.get("channel_id", "") or "")
        )
        or None,
        current_message=normalize_message_text(
            str(values.get("current_message", "") or "")
        ),
        agent_kind=normalize_message_text(str(values.get("agent_kind", "") or "")),
    )


def _merge_history_hits(
    *sources: list[SessionSearchHit],
    limit: int,
) -> list[SessionSearchHit]:
    ranked = [
        (rank, item)
        for source in sources
        for rank, item in enumerate(source)
    ]
    ranked.sort(
        key=lambda pair: (
            pair[1].create_time,
            pair[1].source == "chatinter",
            pair[1].id,
        ),
        reverse=True,
    )
    ranked.sort(key=lambda pair: pair[0])
    ordered = [item for _rank, item in ranked]
    deduplicated = _deduplicate_history_hits(ordered)
    return deduplicated[: max(limit, 1)]


def _deduplicate_history_hits(
    items: list[SessionSearchHit],
) -> list[SessionSearchHit]:
    unique: list[SessionSearchHit] = []
    seen_refs: set[tuple[str, int]] = set()
    matched_chatinter: set[int] = set()
    drop_platform: set[int] = set()
    chatinter_items = [item for item in items if item.source == "chatinter"]
    platform_items = [item for item in items if item.source == "platform"]
    for platform_item in platform_items:
        matches = [
            (index, candidate)
            for index, candidate in enumerate(chatinter_items)
            if index not in matched_chatinter
            and _same_cross_source_message(candidate, platform_item)
        ]
        if not matches:
            continue
        match_index, _candidate = min(
            matches,
            key=lambda pair: _history_time_distance(pair[1], platform_item),
        )
        matched_chatinter.add(match_index)
        drop_platform.add(id(platform_item))
    for item in items:
        ref = (item.source, item.id)
        if ref in seen_refs or id(item) in drop_platform:
            continue
        seen_refs.add(ref)
        unique.append(item)
    return unique


def _same_cross_source_message(
    first: SessionSearchHit,
    second: SessionSearchHit,
) -> bool:
    if first.source == second.source or first.user_id != second.user_id:
        return False
    first_text = normalize_message_text(first.user_message).casefold()
    second_text = normalize_message_text(second.user_message).casefold()
    return bool(
        first_text
        and first_text == second_text
        and _history_time_distance(first, second) <= 300
    )


def _history_time_distance(
    first: SessionSearchHit,
    second: SessionSearchHit,
) -> float:
    try:
        first_time = datetime.fromisoformat(first.create_time)
        second_time = datetime.fromisoformat(second.create_time)
    except ValueError:
        return float("inf")
    return abs((first_time - second_time).total_seconds())


def _platform_match_score(text: str, query: str, terms: list[str]) -> int:
    normalized = normalize_message_text(text).casefold()
    needle = normalize_message_text(query).casefold()
    score = 10_000 if needle and needle in normalized else 0
    return score + sum(len(term) * len(term) for term in terms if term in normalized)


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
    "search_platform_history",
    "search_session_history",
    "upsert_session_search_dialog",
]
