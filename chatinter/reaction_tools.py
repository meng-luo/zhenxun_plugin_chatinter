"""Optional reaction discovery and reply tools for unified mixed chat."""

from __future__ import annotations

from html import escape
from typing import Any

from .llm_compat import ToolDefinition, ToolResult
from .reaction_models import ReactionAction, ReactionTurnState
from .reaction_runtime import (
    reaction_settings,
    reaction_store,
)
from .reaction_search import ReactionSearchIndex
from .route_text import normalize_message_text, normalize_reply_text

REACTION_SEARCH_TOOL_NAME = "reaction_search"
REACTION_REPLY_TOOL_NAME = "reaction_reply"
REACTION_TOOL_KIND = "reaction_image"


async def build_reaction_tools(
    *,
    session_id: str,
    recent_reactions: tuple[Any, ...] = (),
) -> tuple[ReactionTurnState | None, dict[str, Any]]:
    settings = reaction_settings()
    if not settings.enabled:
        return None, {}
    store = reaction_store(settings)
    records = tuple(await store.records())
    if not records:
        return None, {}
    descriptions: dict[str, str] = {}
    for record in sorted(records, key=lambda item: item.reaction_id):
        category = str(record.category or "").strip()
        if not category:
            continue
        description = " ".join(str(record.category_description or "").split())[:160]
        if category not in descriptions or (not descriptions[category] and description):
            descriptions[category] = description
    category_catalog = tuple(sorted(descriptions.items()))
    state = ReactionTurnState(
        settings=settings,
        store=store,
        session_id=str(session_id or ""),
        category_catalog=category_catalog,
        records_snapshot=records,
        recent_reactions=tuple(recent_reactions),
    )
    tools = (
        ReactionSearchTool(state),
        ReactionReplyTool(state),
    )
    return state, {tool.name: tool for tool in tools}


class ReactionSearchTool:
    name = REACTION_SEARCH_TOOL_NAME
    execution_side = "client"
    chatinter_tool_kind = REACTION_TOOL_KIND
    chatinter_tool_group = "reaction_images"
    chatinter_tool_group_atomic = True
    chatinter_required_tool = True
    read_only = True

    def __init__(self, state: ReactionTurnState) -> None:
        self.state = state

    async def get_definition(self) -> ToolDefinition:
        category_ids = [
            category for category, _description in self.state.category_catalog
        ]
        catalog = "；".join(
            f"{category}: {description or category}"
            for category, description in self.state.category_catalog
        )
        properties: dict[str, Any] = {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 512,
                "description": (
                    "准备表达的语气、潜台词和适用情境，" "不要只复制用户原话。"
                ),
            },
            "retrieval_queries": {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                },
                "maxItems": 5,
                "description": (
                    "可选的等价情绪、动作、潜台词或中英文类别表达。"
                    "仅用于扩大本地稀疏召回，不能作为候选 ID。"
                ),
            },
        }
        if category_ids:
            properties["category_hints"] = {
                "type": ["array", "null"],
                "items": {"type": "string", "enum": category_ids},
                "maxItems": 3,
                "description": (
                    "可选的真实图库分类提示，只扩大召回而不筛除其他候选，"
                    "也不能直接发送图片。分类用途：" + catalog
                ),
            }
        return ToolDefinition(
            name=self.name,
            description=(
                "主动搜索一张可作为机器人非语言表达的本地聊天表情。"
                "低风险社交互动中，只要图片能为文字增加自然的非语言价值，"
                "通常应先搜索一次，不要求用户明确索要图片。"
                "query 应描述机器人准备表达的回复意图、情绪方向、语气强度和互动情景，"
                "而不是复述用户原话。与规划明确一致的候选通常应选用；"
                "方向、强度或 Persona 不合时仍可放弃。"
                "这不是制作表情包工具，也不能代替用户明确请求的插件操作。"
            ),
            parameters={
                "type": "object",
                "properties": properties,
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        del context
        query = normalize_message_text(str(kwargs.get("query") or ""))[:512]
        if not query:
            return _error("query_required", "表情搜索词不能为空。")
        if self.state.search_payload is not None:
            return ToolResult(
                output=dict(self.state.search_payload), is_retryable=False
            )
        records = self.state.records_snapshot or await self.state.store.records()
        results = await ReactionSearchIndex.search(
            self.state.settings.root,
            records,
            query,
            semantic_enabled=self.state.settings.semantic_search,
            retrieval_queries=kwargs.get("retrieval_queries"),
            category_hints=kwargs.get("category_hints"),
        )
        self.state.search_query = query
        self.state.candidates = {
            record.reaction_id: record for record, _score in results
        }
        recent_turns = {
            fact.reaction_id: fact.turns_ago for fact in self.state.recent_reactions
        }
        candidates = [
            record.public_candidate(
                score=score,
                recently_used=record.reaction_id in recent_turns,
                turns_ago=recent_turns.get(record.reaction_id),
            )
            for record, score in results
        ]
        payload = {
            "ok": True,
            "status": "reaction_candidates" if candidates else "reaction_empty",
            "candidates": candidates,
            "max_selectable": 1,
            "instruction": (
                "候选合适时调用 reaction_reply；没有合适候选时直接正常文字回复。"
            ),
        }
        self.state.search_payload = payload
        return ToolResult(output=payload, is_retryable=False)


class ReactionReplyTool:
    name = REACTION_REPLY_TOOL_NAME
    execution_side = "client"
    chatinter_tool_kind = REACTION_TOOL_KIND
    chatinter_tool_group = "reaction_images"
    chatinter_tool_group_atomic = True
    chatinter_required_tool = True
    read_only = False

    def __init__(self, state: ReactionTurnState) -> None:
        self.state = state

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "使用 reaction_search 本轮返回的一张本地表情完成回复。"
                "append 表示先发送 reply_text 再附图，only 表示只发图片；"
                "reply_text 在 only 模式下作为图片失败时的自然回退。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reaction_id": {
                        "type": "string",
                        "description": "本轮 reaction_search 候选中的完整 id。",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["append", "only"],
                    },
                    "reply_text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 3500,
                        "description": (
                            "符合当前 Persona 的最终回复；"
                            "only 模式下仅在图片失败时显示；"
                            "不得包含候选 ID、候选描述、标签、工具名或调用过程。"
                        ),
                    },
                },
                "required": ["reaction_id", "mode", "reply_text"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        del context
        if self.state.terminal_reply_text is not None:
            return _error("reaction_already_selected", "本轮已经选择过表情。")
        reaction_id = normalize_message_text(str(kwargs.get("reaction_id") or ""))
        mode = normalize_message_text(str(kwargs.get("mode") or "")).casefold()
        reply_text = normalize_reply_text(str(kwargs.get("reply_text") or ""))
        if mode not in {"append", "only"} or not reply_text:
            return _error("invalid_reaction_reply", "表情回复参数无效。")
        record = self.state.candidates.get(reaction_id)
        if record is None:
            return _error(
                "reaction_not_in_current_candidates",
                "只能选择本轮表情搜索返回的候选。",
            )
        path = await self.state.store.resolve(record)
        if path is None:
            return _error("reaction_file_changed", "候选图片已经变化或不存在。")
        intent = normalize_message_text(self.state.search_query)[:160]
        reaction_memory = (
            "<reaction_history>此前回复发送了一张本地表情图片"
            f"；表达意图：{escape(intent, quote=False)}"
            "</reaction_history>"
            if intent
            else "<reaction_history>此前回复发送了一张本地表情图片</reaction_history>"
        )
        terminal_memory = (
            f"{reply_text}\n{reaction_memory}" if mode == "append" else reaction_memory
        )
        self.state.action = ReactionAction(
            reaction_id=record.reaction_id,
            content_sha256=record.content_sha256,
            path=path,
            root=self.state.settings.root,
            mode=mode,
            reply_text=reply_text,
            fallback_text=reply_text,
            memory_text=terminal_memory,
            category=record.category,
            search_intent=intent,
        )
        self.state.terminal_reply_text = "" if mode == "only" else reply_text
        self.state.terminal_memory_text = terminal_memory
        return ToolResult(
            output={
                "ok": True,
                "status": "reaction_reply_completed",
                "reaction_id": reaction_id,
                "mode": mode,
                "attached": True,
                "reply_text": self.state.terminal_reply_text,
                "memory_text": self.state.terminal_memory_text,
                "nontext_delivery": mode == "only",
            },
            is_retryable=False,
        )


def _error(status: str, message: str) -> ToolResult:
    return ToolResult(
        output={"ok": False, "status": status, "error": message},
        display_content=message,
        is_error=True,
        is_retryable=False,
    )


__all__ = [
    "REACTION_REPLY_TOOL_NAME",
    "REACTION_SEARCH_TOOL_NAME",
    "REACTION_TOOL_KIND",
    "ReactionReplyTool",
    "ReactionSearchTool",
    "build_reaction_tools",
]
