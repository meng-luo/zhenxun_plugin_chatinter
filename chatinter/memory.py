"""
ChatInter - 聊天记忆管理

实现 Astr 风格请求上下文：
1. System: 系统设定
2. History: 最近多轮对话作为独立 role messages 注入
3. Context: 语境层（XML 标签包裹）
   - <event_context>/<turn_identity>: 当前事件与说话人
   - <quoted_message>: 当前消息引用的历史内容
   - <chatroom_history>/<long_term_memory>: 群聊背景与长期记忆
4. Current: 当前用户消息

使用 UniMessage 统一处理消息。
"""

import asyncio
from dataclasses import dataclass
from html import escape as _xml_escape
import re
from typing import TYPE_CHECKING, Protocol

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import Image, UniMessage
from nonebot_plugin_alconna.uniseg.tools import reply_fetch

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger
from zhenxun.services.cache import BoundedTTLCache
from zhenxun.services.db_context import with_db_timeout
from zhenxun.services.message_load import is_db_unhealthy

from .chat_memory_store import ChatMemoryStore, LayeredMemoryRecall
from .config import (
    MAX_REPLY_LAYERS,
    USE_SIGN_IN_IMPRESSION,
    get_chat_history_limit,
)
from .context_budget import (
    ChatContextBundle,
    ChatContextSection,
    context_sections_from_lines,
    trim_context_lines,
)
from .event_signals import get_event_signal
from .llm_compat import LLMMessage
from .memory_recall_context import MemoryRecallContext
from .models.chat_history import ChatInterChatHistory
from .prompt_text import build_chat_base_prompt, build_global_attitude_prompt
from .reaction_models import RecentReactionFact
from .utils.cache import get_user_impression_with_cache
from .utils.multimodal import MAX_CHAT_IMAGE_PARTS
from .utils.unimsg_utils import (
    extract_reply_from_message,
    remove_reply_segment,
    uni_to_text_with_tags,
)

if TYPE_CHECKING:
    from .chat_dialogue_planner import DialogueState
    from .event_context import ReplyContext
    from .persona import PersonaSelection

_MEMORY_RECALL_HINTS = (
    "还记得",
    "记得我",
    "之前",
    "上次",
    "上回",
    "我们聊过",
    "聊过",
    "我喜欢",
    "我不喜欢",
    "叫我",
    "喊我",
    "称呼我",
    "我是谁",
    "我叫",
)
_CASUAL_EXACT_MESSAGES = {
    "你好",
    "您好",
    "嗨",
    "hi",
    "hello",
    "哈喽",
    "在吗",
    "在不在",
    "早",
    "早上好",
    "晚上好",
    "午安",
    "晚安",
    "哈哈",
    "哈哈哈",
}
_SAME_THREAD_SOURCES = {"reply", "reply_store", "topic_store", "pending_entity_store"}
_MEMORY_REWRITE_FOLLOWUP_MARKERS = (
    "那个",
    "这个",
    "这事",
    "那件事",
    "它",
    "后来",
    "然后呢",
    "知道吗",
)


def _render_recent_reactions_context(
    facts: tuple[RecentReactionFact, ...],
) -> tuple[str, ...]:
    if not facts:
        return ()
    lines = ["<recent_reactions>"]
    for fact in facts:
        attributes = [
            f'turns_ago="{max(int(fact.turns_ago), 1)}"',
            f'id="{_xml_escape(fact.reaction_id, quote=True)}"',
            f'mode="{_xml_escape(fact.mode, quote=True)}"',
        ]
        if fact.category:
            attributes.append(f'category="{_xml_escape(fact.category, quote=True)}"')
        if fact.search_intent:
            attributes.append(f'intent="{_xml_escape(fact.search_intent, quote=True)}"')
        lines.append(f"<reaction {' '.join(attributes)}/>")
    lines.append("</recent_reactions>")
    return tuple(lines)


_CONTEXT_TOTAL_TOKEN_BUDGET = 3000
_FORWARD_PLACEHOLDER_PATTERN = re.compile(
    r"^(?:[\(\[]?[^\]:\)]*[\)\]]?\s*:\s*)?"
    r"\[(?:forward(?: message)?|reference|转发消息|合并转发)\]$",
    re.IGNORECASE,
)
_QUOTED_MESSAGE_POLICY = "policy=reference_only_current_user_message_has_priority"
_NICKNAME_CACHE_TTL_SECONDS = 30 * 60
_NICKNAME_CACHE_MAX_ITEMS = 1024


class DialogueContextPack(Protocol):
    @property
    def thread(self) -> object | None: ...

    def to_context_xml(self) -> str: ...


@dataclass(frozen=True)
class MemoryRecallRequest:
    search_text: str = ""
    max_candidates: int = 0
    inject_limit: int = 0


class ChatMemory:
    """聊天记忆管理"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._bot_nickname: str | None = None
        self._user_nickname_cache = BoundedTTLCache[tuple[str, str], str](
            "chatinter_user_nicknames",
            ttl_seconds=_NICKNAME_CACHE_TTL_SECONDS,
            max_items=_NICKNAME_CACHE_MAX_ITEMS,
        )
        self._migrated_session_ids: set[tuple[str, str]] = set()

    async def _migrate_legacy_session(self, legacy: str, current: str) -> None:
        pair = (str(legacy or ""), str(current or ""))
        if not pair[0] or pair[0] == pair[1] or pair in self._migrated_session_ids:
            return
        self._migrated_session_ids.add(pair)
        await ChatInterChatHistory.migrate_session_id(*pair)
        from .history_policy import migrate_history_policy_state

        migrate_history_policy_state(*pair)

    @staticmethod
    def _normalize_context_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _append_context_section(
        sections: list[ChatContextSection],
        name: str,
        lines: list[str] | tuple[str, ...],
    ) -> None:
        sections.extend(context_sections_from_lines(name, lines))

    @staticmethod
    def _render_budgeted_context(
        sections: list[ChatContextSection | tuple[str, list[str] | tuple[str, ...]]],
        token_budget: int | None = None,
    ) -> list[str]:
        budget = (
            _CONTEXT_TOTAL_TOKEN_BUDGET
            if token_budget is None
            else max(int(token_budget or 0), 0)
        )
        bundle = ChatContextBundle.from_named_sections(sections)
        return list(bundle.render_lines(budget))

    @staticmethod
    def _trim_context_lines(lines: list[str], token_budget: int) -> list[str]:
        return list(trim_context_lines(lines, token_budget))

    @staticmethod
    def _build_layered_memory_xml(
        layered_memory: LayeredMemoryRecall,
        *,
        limit: int,
    ) -> list[str]:
        return ChatMemory._limit_layered_memory(layered_memory, limit).to_xml_lines()

    @staticmethod
    def _limit_layered_memory(
        layered_memory: LayeredMemoryRecall,
        limit: int,
    ) -> LayeredMemoryRecall:
        remaining = max(int(limit or 0), 0)

        def take(values: tuple[str, ...]) -> tuple[str, ...]:
            nonlocal remaining
            if remaining <= 0:
                return ()
            selected = values[:remaining]
            remaining -= len(selected)
            return selected

        return LayeredMemoryRecall(
            person_facts=take(layered_memory.person_facts),
            preference_facts=take(layered_memory.preference_facts),
            relationship_facts=take(layered_memory.relationship_facts),
            recent_thread_facts=take(layered_memory.recent_thread_facts),
            other_facts=take(layered_memory.other_facts),
        )

    @classmethod
    def _build_memory_recall_request(
        cls,
        current_message_text: str,
        *,
        group_id: str | None,
        dialogue_context: DialogueContextPack | None,
        dialogue_state: "DialogueState | None",
        thread: object | None,
    ) -> MemoryRecallRequest:
        text = cls._normalize_context_text(current_message_text)
        if cls._should_skip_memory_recall(
            text,
            dialogue_context=dialogue_context,
            dialogue_state=dialogue_state,
            thread=thread,
        ):
            return MemoryRecallRequest()
        search_text = cls._memory_search_text(
            text,
            dialogue_context=dialogue_context,
            dialogue_state=dialogue_state,
            thread=thread,
        )
        has_context_match = cls._matches_memory_context(
            text,
            dialogue_context=dialogue_context,
            dialogue_state=dialogue_state,
            thread=thread,
        )
        explicit_recall = cls._has_memory_recall_hint(text)
        return MemoryRecallRequest(
            search_text=search_text or text,
            max_candidates=8 if (explicit_recall or has_context_match) else 4,
            inject_limit=4 if not group_id else 3,
        )

    @classmethod
    def _should_skip_memory_recall(
        cls,
        text: str,
        *,
        dialogue_context: DialogueContextPack | None,
        dialogue_state: "DialogueState | None",
        thread: object | None,
    ) -> bool:
        compact = "".join(cls._normalize_context_text(text).split()).casefold()
        if not compact:
            return True
        if compact in _CASUAL_EXACT_MESSAGES:
            return True
        if not any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in compact):
            return True
        if len(compact) <= 2 and not cls._matches_memory_context(
            text,
            dialogue_context=dialogue_context,
            dialogue_state=dialogue_state,
            thread=thread,
        ):
            return True
        return False

    @classmethod
    def _memory_search_text(
        cls,
        current_message_text: str,
        *,
        dialogue_context: DialogueContextPack | None,
        dialogue_state: "DialogueState | None",
        thread: object | None,
    ) -> str:
        parts = [current_message_text]
        if dialogue_state is not None and dialogue_state.topic_hint:
            parts.append(str(dialogue_state.topic_hint))
        if thread is not None:
            for attr in ("topic_key", "entity_hints", "pending_entities"):
                value = getattr(thread, attr, "") or ""
                if isinstance(value, str):
                    parts.append(value)
                else:
                    parts.extend(str(item) for item in value or ())
        parts.extend(cls._context_person_terms(dialogue_context))
        unique_parts = dict.fromkeys(cls._normalize_context_text(p) for p in parts if p)
        return " ".join(unique_parts)

    @classmethod
    def _rewrite_memory_search_text(
        cls,
        current_message_text: str,
        *,
        current_search_text: str,
        dialogue_context: DialogueContextPack | None,
        dialogue_state: "DialogueState | None",
        thread: object | None,
    ) -> str:
        text = cls._normalize_context_text(current_message_text)
        if not cls._should_rewrite_memory_query(
            text,
            dialogue_context=dialogue_context,
            dialogue_state=dialogue_state,
            thread=thread,
        ):
            return ""
        parts = [text]
        if dialogue_state is not None:
            parts.extend(
                (
                    dialogue_state.topic_hint,
                    dialogue_state.last_user_message,
                    dialogue_state.last_reply_summary,
                )
            )
        if thread is not None:
            for attr in ("topic_key", "entity_hints", "pending_entities"):
                value = getattr(thread, attr, "") or ""
                if isinstance(value, str):
                    parts.append(value)
                else:
                    parts.extend(str(item) for item in value or ())
        parts.extend(cls._context_person_terms(dialogue_context))
        unique_parts = [
            item
            for item in dict.fromkeys(cls._normalize_context_text(p) for p in parts)
            if item
        ]
        rewritten = " ".join(unique_parts)
        if rewritten == cls._normalize_context_text(current_search_text):
            return ""
        return rewritten

    @classmethod
    def _should_rewrite_memory_query(
        cls,
        text: str,
        *,
        dialogue_context: DialogueContextPack | None,
        dialogue_state: "DialogueState | None",
        thread: object | None,
    ) -> bool:
        compact = "".join(cls._normalize_context_text(text).split()).casefold()
        if not compact or compact in _CASUAL_EXACT_MESSAGES:
            return False
        has_followup_state = (
            dialogue_state is not None
            and dialogue_state.continuity in {"same_topic", "followup"}
            and bool(
                dialogue_state.topic_hint
                or dialogue_state.last_user_message
                or dialogue_state.last_reply_summary
            )
        )
        has_context = has_followup_state or cls._matches_memory_context(
            text,
            dialogue_context=dialogue_context,
            dialogue_state=dialogue_state,
            thread=thread,
        )
        if cls._looks_like_personal_memory_query(text) and len(compact) <= 24:
            return True
        if len(compact) > 12 or not has_context:
            return False
        return compact.endswith(("呢", "吗", "嘛", "？", "?")) or any(
            marker in compact for marker in _MEMORY_REWRITE_FOLLOWUP_MARKERS
        )

    @classmethod
    def _context_person_terms(
        cls,
        dialogue_context: DialogueContextPack | None,
    ) -> list[str]:
        if dialogue_context is None:
            return []
        terms: list[str] = []
        for person in getattr(dialogue_context, "relevant_people", ()) or ():
            profile = getattr(person, "profile", None)
            terms.append(str(getattr(person, "matched_alias", "") or ""))
            for attr in ("display_name", "nickname", "group_card"):
                terms.append(str(getattr(profile, attr, "") or ""))
            terms.extend(str(item) for item in getattr(profile, "aliases", ()) or ())
        return [
            item for item in dict.fromkeys(terms) if cls._normalize_context_text(item)
        ]

    @staticmethod
    def _looks_like_personal_memory_query(text: str) -> bool:
        normalized = ChatMemory._normalize_context_text(text)
        if "我" not in normalized:
            return False
        return normalized.endswith(("?", "？", "吗", "么", "嘛", "呢")) or any(
            marker in normalized
            for marker in ("什么", "哪个", "哪种", "谁", "多少", "来着")
        )

    @staticmethod
    def _has_memory_recall_hint(text: str) -> bool:
        return any(hint in text for hint in _MEMORY_RECALL_HINTS)

    @classmethod
    def _matches_memory_context(
        cls,
        text: str,
        *,
        dialogue_context: DialogueContextPack | None,
        dialogue_state: "DialogueState | None",
        thread: object | None,
    ) -> bool:
        if cls._is_same_thread_context(thread):
            return True
        if cls._message_matches_topic(text, dialogue_state, thread):
            return True
        return cls._message_mentions_context_person(text, dialogue_context)

    @staticmethod
    def _is_same_thread_context(thread: object | None) -> bool:
        if thread is None:
            return False
        source = str(getattr(thread, "source", "") or "")
        try:
            confidence = float(getattr(thread, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return source in _SAME_THREAD_SOURCES and confidence >= 0.62

    @classmethod
    def _message_matches_topic(
        cls,
        text: str,
        dialogue_state: "DialogueState | None",
        thread: object | None,
    ) -> bool:
        terms: list[str] = []
        if dialogue_state is not None and dialogue_state.continuity in {
            "same_topic",
            "followup",
        }:
            terms.extend(cls._split_recall_terms(dialogue_state.topic_hint))
        if cls._is_same_thread_context(thread):
            terms.extend(
                cls._split_recall_terms(str(getattr(thread, "topic_key", "") or ""))
            )
            terms.extend(
                cls._split_recall_terms(
                    " ".join(
                        str(item) for item in getattr(thread, "entity_hints", ()) or ()
                    )
                )
            )
        return cls._text_matches_terms(text, terms)

    @classmethod
    def _message_mentions_context_person(
        cls,
        text: str,
        dialogue_context: DialogueContextPack | None,
    ) -> bool:
        if dialogue_context is None:
            return False
        terms: list[str] = []
        for person in getattr(dialogue_context, "relevant_people", ()) or ():
            matched_alias = str(getattr(person, "matched_alias", "") or "")
            is_current_speaker = bool(getattr(person, "is_current_speaker", False))
            if is_current_speaker and not matched_alias:
                continue
            profile = getattr(person, "profile", None)
            terms.append(matched_alias)
            for attr in ("display_name", "nickname", "group_card"):
                terms.append(str(getattr(profile, attr, "") or ""))
            terms.extend(str(item) for item in getattr(profile, "aliases", ()) or ())
        return cls._text_matches_terms(text, terms)

    @staticmethod
    def _split_recall_terms(value: str) -> list[str]:
        return [
            item
            for item in re.split(r"[\s,，。.!！？?、_/|;；：:（）()\[\]【】]+", value)
            if len(item) >= 2
        ]

    @classmethod
    def _text_matches_terms(cls, text: str, terms: list[str]) -> bool:
        normalized = cls._normalize_context_text(text).casefold()
        for term in terms:
            candidate = cls._normalize_context_text(term).casefold()
            if len(candidate) >= 2 and candidate in normalized:
                return True
        return False

    @staticmethod
    def _extract_http_url(value: object) -> str:
        text = str(value or "").strip()
        if text.startswith(("http://", "https://")):
            return text
        return ""

    @classmethod
    def _extract_url_from_get_image_result(cls, payload: object) -> str:
        if not isinstance(payload, dict):
            return ""
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("url", "src", "file"):
                if url := cls._extract_http_url(data.get(key)):
                    return url
        for key in ("url", "src", "file"):
            if url := cls._extract_http_url(payload.get(key)):
                return url
        return ""

    async def _resolve_onebot_image_url(self, bot: Bot | None, file_id: str) -> str:
        if not bot:
            return ""
        file_text = str(file_id or "").strip()
        if not file_text or file_text.startswith(("http://", "https://", "base64://")):
            return ""
        try:
            result = await bot.get_image(file=file_text)
        except Exception as e:
            logger.debug(f"Reply 图片 URL 解析失败，file={file_text}, err={e}")
            return ""
        return self._extract_url_from_get_image_result(result)

    async def _build_reply_image_segment(
        self,
        *,
        bot: Bot | None,
        file_value: str = "",
        url_value: str = "",
        path_value: str = "",
    ) -> Image | None:
        file_text = str(file_value or "").strip()
        url_text = str(url_value or "").strip()
        path_text = str(path_value or "").strip()

        if (
            not url_text
            and file_text
            and not file_text.startswith(("http://", "https://", "base64://"))
        ):
            url_text = await self._resolve_onebot_image_url(bot, file_text)

        if file_text.startswith(("http://", "https://")):
            url_text = url_text or file_text
            file_text = ""

        if file_text:
            return Image(id=file_text, url=url_text or None)
        if url_text:
            return Image(url=url_text)
        if path_text:
            return Image(path=path_text)
        return None

    async def _fetch_user_nickname(
        self, user_id: str, group_id: str | None
    ) -> str | None:
        """获取用户昵称（带缓存）

        参数:
            user_id: 用户 ID
            group_id: 群组 ID

        返回:
            昵称，如果未找到返回 None
        """
        cache_key = (str(group_id or ""), str(user_id))
        if nickname := await self._user_nickname_cache.get(cache_key):
            return nickname

        if group_id:
            if is_db_unhealthy():
                return None
            from zhenxun.models.group_member_info import GroupInfoUser

            try:
                member = await with_db_timeout(
                    GroupInfoUser.filter(group_id=group_id, user_id=user_id).first(),
                    timeout=2.0,
                    operation="ChatInter.fetch_user_nickname",
                    source="chatinter",
                )
            except TimeoutError:
                return None
            except Exception:
                return None
            if member:
                nick = str(getattr(member, "nickname", "") or member.user_name or "")
                if nick:
                    await self._user_nickname_cache.set(cache_key, nick)
                    return nick

        return None

    def set_bot_nickname(self, nickname: str):
        """设置 bot 昵称"""
        self._bot_nickname = nickname

    def get_session_id(self, user_id: str, group_id: str | None) -> str:
        """获取用于数据库存储的 session_id"""
        return group_id if group_id else user_id

    async def add_timeline(
        self,
        user_id: str,
        group_id: str | None,
        nickname: str,
        user_message: str | UniMessage,
        response_summary: str,
        timeline: list[dict],
        bot_id: str | None = None,
        session_id: str | None = None,
    ) -> ChatInterChatHistory | None:
        """添加一次完整 ChatInter message timeline。"""
        session_id = session_id or self.get_session_id(user_id, group_id)

        formatted_user_message = uni_to_text_with_tags(user_message)
        formatted_response_summary = uni_to_text_with_tags(response_summary)
        from .history_policy import freeze_timeline_sender_label

        frozen_timeline = await freeze_timeline_sender_label(
            timeline,
            user_id=user_id,
            group_id=group_id,
            fallback_name=nickname,
        )

        async with self._lock:
            dialog = await ChatInterChatHistory.add_timeline(
                session_id=session_id,
                user_id=user_id,
                group_id=group_id,
                nickname=nickname,
                user_message=formatted_user_message,
                ai_response=formatted_response_summary,
                timeline=frozen_timeline,
                bot_id=bot_id,
            )
        return dialog

    async def build_full_context(
        self,
        user_id: str,
        group_id: str | None,
        nickname: str,
        raw_message: str | UniMessage,
        bot: Bot | None = None,
        bot_id: str | None = None,
        event: Event | None = None,
        dialogue_context: DialogueContextPack | None = None,
        dialogue_state: "DialogueState | None" = None,
        scenario: str = "",
        persona_selection: "PersonaSelection | None" = None,
        session_id: str | None = None,
        legacy_session_id: str | None = None,
        reply_context: "ReplyContext | None" = None,
        context_sections_out: list[ChatContextSection] | None = None,
        recent_reactions_out: list[RecentReactionFact] | None = None,
    ) -> tuple[str, str, list[Image], list[LLMMessage]]:
        """构建完整的上下文（System + Context + Current + History Messages）

        参数:
            user_id: 用户 ID
            group_id: 群组 ID
            nickname: 用户昵称
            raw_message: 原始用户消息（UniMessage 或字符串）
            bot: Bot 实例（用于获取消息）
            bot_id: Bot ID
            event: Event 实例（用于获取回复 ID）

        返回:
            tuple: (system_prompt, context_xml, reply_images, history_messages)
            - system_prompt: 系统提示词
            - context_xml: XML 格式的上下文
            - reply_images: 回复链中的图片 Image Segment 列表（用于多模态处理）
            - history_messages: Astr 风格的独立 role 历史消息
        """
        context_sections: list[ChatContextSection] = []
        reply_images: list[Image] = []
        history_messages: list[LLMMessage] = []
        current_message_text = ""
        inject_chat_memory = scenario != "superuser_agent"

        try:
            if isinstance(raw_message, UniMessage):
                normalized_current_msg = remove_reply_segment(raw_message)
                current_message_text = uni_to_text_with_tags(normalized_current_msg)
            else:
                current_message_text = uni_to_text_with_tags(str(raw_message or ""))
        except Exception:
            current_message_text = str(raw_message or "")
        current_message_text = self._normalize_context_text(current_message_text)

        if dialogue_context is not None:
            packed_context = dialogue_context.to_context_xml()
            if packed_context:
                self._append_context_section(
                    context_sections,
                    "event",
                    packed_context.splitlines(),
                )
        thread = getattr(dialogue_context, "thread", None)
        addressee = getattr(dialogue_context, "addressee", None)
        thread_id = str(getattr(thread, "thread_id", "") or "").strip()
        thread_user_ids = tuple(
            str(item)
            for item in getattr(thread, "related_user_ids", ()) or ()
            if str(item)
        )
        addressee_user_id = str(getattr(addressee, "target_user_id", "") or "")

        session_id = session_id or self.get_session_id(user_id, group_id)
        if legacy_session_id and legacy_session_id != session_id:
            await self._migrate_legacy_session(legacy_session_id, session_id)
        from .history_policy import (
            append_chatroom_history_context,
            build_astr_history_payload,
        )

        history_limit = get_chat_history_limit()
        history_payload = await build_astr_history_payload(
            session_id=session_id,
            user_id=user_id,
            current_message_text=current_message_text,
            current_message_id=str(
                get_event_signal(
                    event,
                    "_chatinter_group_context_record_id",
                    "",
                )
                or getattr(event, "message_id", "")
                or getattr(event, "event_id", "")
                or getattr(event, "id", "")
                or ""
            ),
            group_id=group_id,
            bot_id=bot_id,
            dialog_limit=history_limit,
            chatroom_limit=history_limit,
        )
        history_messages = list(history_payload.messages)
        recent_reactions = tuple(getattr(history_payload, "recent_reactions", ()) or ())
        if recent_reactions_out is not None:
            recent_reactions_out.clear()
            recent_reactions_out.extend(recent_reactions)
        if inject_chat_memory and recent_reactions:
            self._append_context_section(
                context_sections,
                "recent_reactions",
                _render_recent_reactions_context(recent_reactions),
            )
        chatroom_context_lines: list[str] = []
        append_chatroom_history_context(
            chatroom_context_lines,
            history_payload.chatroom_lines,
        )
        self._append_context_section(
            context_sections,
            "chatroom",
            chatroom_context_lines,
        )

        recall_request = self._build_memory_recall_request(
            current_message_text,
            group_id=group_id,
            dialogue_context=dialogue_context,
            dialogue_state=dialogue_state,
            thread=thread,
        )
        recall_context = MemoryRecallContext.build(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            thread_id=thread_id or None,
            topic_key=str(getattr(thread, "topic_key", "") or ""),
            participants=thread_user_ids,
            addressee_user_id=addressee_user_id or None,
            query=recall_request.search_text or current_message_text,
        )
        if not inject_chat_memory or recall_request.max_candidates <= 0:
            layered_memory = LayeredMemoryRecall()
        else:
            layered_memory = await ChatMemoryStore.recall_layered(
                session_id=session_id,
                user_id=user_id,
                group_id=group_id,
                query=recall_request.search_text,
                limit=recall_request.max_candidates,
                recall_context=recall_context,
            )
            if layered_memory.is_empty:
                rewrite_search_text = self._rewrite_memory_search_text(
                    current_message_text,
                    current_search_text=recall_request.search_text,
                    dialogue_context=dialogue_context,
                    dialogue_state=dialogue_state,
                    thread=thread,
                )
                if rewrite_search_text:
                    rewrite_request = MemoryRecallRequest(
                        search_text=rewrite_search_text,
                        max_candidates=max(recall_request.max_candidates, 8),
                        inject_limit=recall_request.inject_limit,
                    )
                    rewrite_context = MemoryRecallContext.build(
                        session_id=session_id,
                        user_id=user_id,
                        group_id=group_id,
                        thread_id=thread_id or None,
                        topic_key=str(getattr(thread, "topic_key", "") or ""),
                        participants=thread_user_ids,
                        addressee_user_id=addressee_user_id or None,
                        query=rewrite_search_text,
                    )
                    rewritten_memory = await ChatMemoryStore.recall_layered(
                        session_id=session_id,
                        user_id=user_id,
                        group_id=group_id,
                        query=rewrite_search_text,
                        limit=rewrite_request.max_candidates,
                        recall_context=rewrite_context,
                    )
                    if not rewritten_memory.is_empty:
                        layered_memory = rewritten_memory
                        recall_request = rewrite_request
        if dialogue_context is not None:
            try:
                setattr(dialogue_context, "layered_memory", layered_memory)
            except Exception:
                pass
        memory_lines = (
            self._build_layered_memory_xml(
                layered_memory,
                limit=recall_request.inject_limit,
            )
            if inject_chat_memory
            else []
        )
        if memory_lines:
            self._append_context_section(
                context_sections,
                "memory",
                ["<long_term_memory>", *memory_lines, "</long_term_memory>"],
            )

        (
            current_message_layers_lines,
            reply_images,
        ) = await self._build_current_message_layers(
            group_id,
            raw_message,
            nickname,
            bot_id,
            bot,
            event,
            reply_context=reply_context,
        )
        if current_message_layers_lines:
            self._append_context_section(
                context_sections,
                "reply_layers",
                [
                    "<quoted_message>",
                    *current_message_layers_lines,
                    "</quoted_message>",
                ],
            )

        if inject_chat_memory and USE_SIGN_IN_IMPRESSION:
            impression, attitude = await self.get_user_impression(user_id)
            impression_rule = build_global_attitude_prompt(
                impression,
                attitude,
            ).strip()
            if impression_rule:
                self._append_context_section(
                    context_sections,
                    "relationship",
                    [
                        "<relationship>"
                        f"{_xml_escape(impression_rule, quote=False)}"
                        "</relationship>",
                    ],
                )

        context_bundle = ChatContextBundle(tuple(context_sections))
        if context_sections_out is not None:
            context_sections_out.extend(context_bundle.sections)
        context_xml = context_bundle.render()
        system_prompt = (
            self._build_system_prompt(
                current_message_text=current_message_text,
                persona_selection=persona_selection,
            )
            if inject_chat_memory
            else ""
        )

        return system_prompt, context_xml, reply_images, history_messages

    async def _build_current_message_layers(
        self,
        group_id: str | None,
        raw_message: str | UniMessage,
        nickname: str,
        bot_id: str | None = None,
        bot: Bot | None = None,
        event: Event | None = None,
        reply_context: "ReplyContext | None" = None,
    ) -> tuple[list[str], list[Image]]:
        """构建当前消息所引用的历史内容。"""
        reply_images: list[Image] = []
        embedded_text = self._normalize_context_text(
            str(getattr(reply_context, "text", "") or "")
        )
        embedded_sender_id = str(getattr(reply_context, "sender_id", "") or "").strip()
        reply_id = str(getattr(reply_context, "message_id", "") or "").strip()

        if embedded_text and not self._is_forward_placeholder_only(embedded_text):
            sender = await self._resolve_reply_sender(
                embedded_sender_id,
                group_id=group_id,
                bot_id=bot_id,
            )
            if "[image" in embedded_text.casefold() and bot and reply_id:
                try:
                    msg_data = await bot.get_msg(message_id=reply_id)
                    payload = self._unwrap_reply_payload(msg_data)
                    _, fetched_images, _ = await self._parse_reply_message(
                        payload.get("message", payload.get("raw_message", "")),
                        bot=bot,
                        image_limit=MAX_CHAT_IMAGE_PARTS,
                    )
                    reply_images.extend(fetched_images)
                except Exception as e:
                    logger.debug(f"获取内嵌回复图片失败：{e}")
            return [
                _QUOTED_MESSAGE_POLICY,
                self._format_quoted_message_line(1, sender, embedded_text),
            ], reply_images

        if not reply_id and isinstance(raw_message, UniMessage):
            reply_id = str(extract_reply_from_message(raw_message) or "").strip()

        if not reply_id and event and bot:
            try:
                reply_seg = await reply_fetch(event, bot)
                if reply_seg and hasattr(reply_seg, "id") and reply_seg.id:
                    reply_id = str(reply_seg.id).strip()
            except Exception as e:
                logger.debug(f"从 reply_fetch 获取回复 ID 失败：{e}")

        if not reply_id or not bot:
            if not embedded_text:
                return [], []
            sender = await self._resolve_reply_sender(
                embedded_sender_id,
                group_id=group_id,
                bot_id=bot_id,
            )
            return [
                _QUOTED_MESSAGE_POLICY,
                self._format_quoted_message_line(1, sender, embedded_text),
            ], []

        lines: list[str] = []
        seen_ids: set[str] = set()
        current_reply_id = reply_id

        for layer in range(1, MAX_REPLY_LAYERS + 1):
            if current_reply_id in seen_ids:
                break
            seen_ids.add(current_reply_id)

            try:
                msg_data = await bot.get_msg(message_id=current_reply_id)
                if not msg_data:
                    break
                payload = self._unwrap_reply_payload(msg_data)
                sender_data = payload.get("sender", {})
                msg_user_id = str(
                    payload.get("user_id", "")
                    or (
                        sender_data.get("user_id", "")
                        if isinstance(sender_data, dict)
                        else getattr(sender_data, "user_id", "")
                    )
                    or ""
                ).strip()
                raw_msg = payload.get("message", payload.get("raw_message", ""))
                content, images, next_reply_id = await self._parse_reply_message(
                    raw_msg,
                    bot=bot,
                    image_limit=max(MAX_CHAT_IMAGE_PARTS - len(reply_images), 0),
                )
                reply_images.extend(images)
                sender = await self._resolve_reply_sender(
                    msg_user_id,
                    group_id=group_id,
                    bot_id=bot_id,
                )
                lines.append(self._format_quoted_message_line(layer, sender, content))

                if not next_reply_id:
                    break
                current_reply_id = str(next_reply_id).strip()
                if not current_reply_id:
                    break
            except Exception as e:
                logger.error(f"获取回复失败 layer={layer}: {e}")
                break

        if not lines and embedded_text:
            sender = await self._resolve_reply_sender(
                embedded_sender_id,
                group_id=group_id,
                bot_id=bot_id,
            )
            lines.append(self._format_quoted_message_line(1, sender, embedded_text))
        if lines:
            lines.insert(0, _QUOTED_MESSAGE_POLICY)
        return lines, reply_images

    @staticmethod
    def _is_forward_placeholder_only(text: str) -> bool:
        values = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        return bool(values) and all(
            _FORWARD_PLACEHOLDER_PATTERN.match(line) for line in values
        )

    @staticmethod
    def _unwrap_reply_payload(msg_data: object) -> dict:
        if not isinstance(msg_data, dict):
            return {}
        nested = msg_data.get("data")
        if (
            isinstance(nested, dict)
            and "message" not in msg_data
            and "raw_message" not in msg_data
        ):
            return nested
        return msg_data

    async def _resolve_reply_sender(
        self,
        user_id: str,
        *,
        group_id: str | None,
        bot_id: str | None,
    ) -> str:
        normalized_user_id = str(user_id or "").strip()
        if bot_id and normalized_user_id == str(bot_id):
            return f"[{self._bot_nickname or BotConfig.self_nickname}]"
        if normalized_user_id:
            nickname = await self._fetch_user_nickname(normalized_user_id, group_id)
            if nickname:
                return f"[{nickname}]"
            return f"[QQ:{normalized_user_id}]"
        return "[unknown]"

    @staticmethod
    def _format_quoted_message_line(layer: int, sender: str, content: str) -> str:
        normalized_content = ChatMemory._normalize_context_text(content) or "(空消息)"
        return (
            f"[Layer {layer}][reply][from:"
            f"{_xml_escape(sender, quote=False)}] "
            f"{_xml_escape(normalized_content, quote=False)}"
        )

    @staticmethod
    def _reply_message_segments(raw_message: object) -> list[object]:
        if isinstance(raw_message, list):
            return list(raw_message)
        if isinstance(raw_message, UniMessage):
            return list(raw_message)
        if isinstance(raw_message, str):
            try:
                from nonebot.adapters.onebot.v11 import Message as OBMessage

                parsed = OBMessage(raw_message)
                return list(parsed) if parsed else []
            except Exception:
                return []
        if raw_message is None or isinstance(raw_message, dict):
            return [raw_message] if isinstance(raw_message, dict) else []
        try:
            return list(raw_message)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return []

    async def _parse_reply_message(
        self,
        raw_message: object,
        *,
        bot: Bot | None,
        image_limit: int,
    ) -> tuple[str, list[Image], str | None]:
        parts: list[str] = []
        images: list[Image] = []
        next_reply_id: str | None = None
        segments = self._reply_message_segments(raw_message)
        if not segments and raw_message:
            return self._normalize_context_text(str(raw_message)), [], None

        for segment in segments:
            if isinstance(segment, dict):
                segment_type = str(segment.get("type", "") or "").casefold()
                data = segment.get("data", {})
                segment_data = data if isinstance(data, dict) else {}
            else:
                segment_type = str(
                    getattr(segment, "type", "") or type(segment).__name__
                ).casefold()
                data = getattr(segment, "data", {})
                segment_data = data if isinstance(data, dict) else {}

            if segment_type in {"text", "plain"}:
                text = segment_data.get("text") if segment_data else None
                parts.append(
                    str(text if text is not None else getattr(segment, "text", ""))
                )
            elif segment_type == "at":
                target = (
                    segment_data.get("qq")
                    or segment_data.get("target")
                    or getattr(segment, "target", "")
                )
                parts.append(f"[@{target}]" if target else "[@]")
            elif segment_type == "image":
                parts.append("[image]")
                if len(images) >= image_limit:
                    continue
                if isinstance(segment, Image):
                    image_segment = segment
                else:
                    image_segment = await self._build_reply_image_segment(
                        bot=bot,
                        file_value=str(segment_data.get("file", "") or ""),
                        url_value=str(segment_data.get("url", "") or ""),
                        path_value=str(segment_data.get("path", "") or ""),
                    )
                if image_segment is not None:
                    images.append(image_segment)
            elif segment_type in {"reply"}:
                value = (
                    segment_data.get("id")
                    or segment_data.get("message_id")
                    or getattr(segment, "id", "")
                )
                if value and next_reply_id is None:
                    next_reply_id = str(value)
            elif segment_type == "file":
                raw_name = str(
                    segment_data.get("name")
                    or segment_data.get("file")
                    or getattr(segment, "name", "")
                    or ""
                )
                file_name = raw_name.replace("\\", "/").rsplit("/", 1)[-1][:80]
                parts.append(f"[file:{file_name}]" if file_name else "[file]")
            elif segment_type in {"record", "voice", "audio"}:
                parts.append("[voice]")
            elif segment_type == "video":
                parts.append("[video]")
            elif segment_type in {"forward", "reference", "node", "nodes"}:
                parts.append("[forward]")
            elif segment_type:
                parts.append(f"[{segment_type}]")

        content = self._normalize_context_text("".join(parts)) or "(空消息)"
        return content, images, next_reply_id

    def _build_system_prompt(
        self,
        current_message_text: str = "",
        persona_selection: "PersonaSelection | None" = None,
    ) -> str:
        """构建系统提示词。

        产出必须在会话内逐字节稳定（供应商按前缀缓存 prompt）：
        任何会随轮次变化的内容（好感度、召回、时间）都放 context_xml，不放这里。
        """
        persona = persona_selection.persona if persona_selection is not None else None
        _ = current_message_text

        base = build_chat_base_prompt(
            self._bot_nickname or BotConfig.self_nickname,
        )

        custom_prompt = persona.prompt_fragment() if persona is not None else ""
        persona_prompt = (
            "<persona_config>\n"
            "当前人格设定（来自配置）：\n"
            f"{custom_prompt}\n"
            "</persona_config>"
            if custom_prompt
            else ""
        )

        return "\n\n".join(
            part
            for part in (
                persona_prompt,
                base,
            )
            if part
        )

    async def get_user_impression(self, user_id: str) -> tuple[float, str]:
        """获取用户好感度"""
        return await get_user_impression_with_cache(user_id)

    async def reset_session_history(
        self,
        user_id: str,
        group_id: str | None,
        *,
        session_id: str | None = None,
        legacy_session_id: str | None = None,
    ) -> int:
        """重置会话历史（软删除，标记 reset=True）

        参数:
            user_id: 用户 ID
            group_id: 群组 ID

        返回:
            int: 被重置的对话数量
        """
        session_id = session_id or self.get_session_id(user_id, group_id)
        if legacy_session_id and legacy_session_id != session_id:
            await self._migrate_legacy_session(legacy_session_id, session_id)
        reset_count = await ChatInterChatHistory.reset_session(session_id)
        from .history_policy import reset_history_policy_state

        reset_history_policy_state(session_id)
        return reset_count


_chat_memory = ChatMemory()
