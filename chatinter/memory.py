"""
ChatInter - 聊天记忆管理

实现 Astr 风格请求上下文：
1. System: 系统设定
2. History: 最近多轮对话作为独立 role messages 注入
3. Context: 语境层（XML 标签包裹）
   - <qq_context>: QQ 上下文元数据
   - <context_layers>: 回复链追溯 + 群聊历史
   - <chatroom_history>: 群聊最近消息背景
4. Current: 当前用户消息

使用 UniMessage 统一处理消息。
"""

import asyncio
import re
import time
from typing import TYPE_CHECKING, Protocol, cast

from nonebot.adapters import Bot, Event, Message
from nonebot_plugin_alconna.uniseg import Image, UniMessage
from nonebot_plugin_alconna.uniseg.tools import reply_fetch

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger
from zhenxun.services.db_context import with_db_timeout
from zhenxun.services.llm import LLMMessage
from zhenxun.services.message_load import is_db_unhealthy

from .chat_memory_store import ChatMemoryStore, LayeredMemoryRecall
from .config import (
    CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX,
    MAX_REPLY_LAYERS,
    SESSION_CONTEXT_LIMIT,
    USE_SIGN_IN_IMPRESSION,
)
from .memory_recall_context import MemoryRecallContext
from .models.chat_history import ChatInterChatHistory
from .persona import resolve_persona
from .prompt_text import build_chat_base_prompt, build_global_attitude_prompt
from .utils.cache import get_user_impression_with_cache
from .utils.unimsg_utils import (
    extract_reply_from_message,
    remove_reply_segment,
    uni_to_text_with_tags,
)

if TYPE_CHECKING:
    from .chat_dialogue_planner import DialogueState

_CONTEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,8}", re.IGNORECASE)
_CONTEXT_STOPWORDS = {
    "这个",
    "那个",
    "然后",
    "就是",
    "一下",
    "一个",
    "我们",
    "你们",
    "他们",
    "自己",
    "可以",
    "怎么",
    "如何",
}
_COMPLEX_QUERY_HINTS = (
    "代码",
    "插件",
    "报错",
    "错误",
    "异常",
    "调试",
    "排查",
    "实现",
    "方案",
    "步骤",
    "配置",
    "脚本",
    "traceback",
    "exception",
    "nonebot",
    "python",
    "api",
)


class DialogueContextPack(Protocol):
    @property
    def thread(self) -> object | None: ...

    def to_context_xml(self) -> str: ...


class ChatMemory:
    """聊天记忆管理"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._bot_nickname: str | None = None
        self._user_nickname_cache: dict[str, str] = {}
        self._nickname_cache_time: dict[str, float] = {}
        self._nickname_ttl = 30 * 60

    @staticmethod
    def _normalize_context_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @classmethod
    def _tokenize_context_text(cls, text: str) -> list[str]:
        normalized = cls._normalize_context_text(text).lower()
        if not normalized:
            return []
        return [
            token
            for token in _CONTEXT_TOKEN_PATTERN.findall(normalized)
            if token and token not in _CONTEXT_STOPWORDS
        ]

    @classmethod
    def _is_complex_query(cls, text: str) -> bool:
        normalized = cls._normalize_context_text(text).lower()
        if not normalized:
            return False
        if "```" in normalized or "\n" in normalized:
            return True
        if len(normalized) >= 36:
            return True
        return any(hint in normalized for hint in _COMPLEX_QUERY_HINTS)

    def _build_conversation_focus(
        self,
        current_message_text: str,
        dialogue_state: "DialogueState | None" = None,
    ) -> list[str]:
        normalized = self._normalize_context_text(current_message_text)
        if not normalized:
            return []

        tokens = self._tokenize_context_text(normalized)
        top_keywords: list[str] = []
        for token in tokens:
            if token not in top_keywords:
                top_keywords.append(token)
            if len(top_keywords) >= 6:
                break
        if not top_keywords:
            top_keywords = ["无"]

        response_mode = (
            str(dialogue_state.response_length)
            if dialogue_state is not None
            else "detailed"
            if self._is_complex_query(normalized)
            else "concise"
        )
        lines = [
            "<conversation_focus>",
            f"query_keywords={','.join(top_keywords)}",
            f"response_mode={response_mode}",
        ]
        if dialogue_state is not None:
            lines.extend(
                (
                    f"tone={dialogue_state.tone}",
                    f"user_emotion={dialogue_state.user_emotion}",
                    f"dialogue_purpose={dialogue_state.dialogue_purpose}",
                    f"need_followup={int(dialogue_state.need_followup)}",
                    f"continuity={dialogue_state.continuity}",
                    f"group_reply_policy={dialogue_state.group_reply_policy}",
                    f"reply_posture={dialogue_state.reply_posture}",
                    f"group_atmosphere={dialogue_state.group_atmosphere}",
                    f"address_mode={dialogue_state.address_mode}",
                )
            )
            if dialogue_state.last_user_message:
                lines.append(
                    "previous_user_message="
                    f"{self._normalize_context_text(dialogue_state.last_user_message)[:120]}"
                )
            if dialogue_state.last_reply_summary:
                lines.append(
                    "previous_reply_summary="
                    f"{self._normalize_context_text(dialogue_state.last_reply_summary)[:120]}"
                )
        lines.append("</conversation_focus>")
        return lines

    @staticmethod
    def _build_layered_memory_xml(
        layered_memory: LayeredMemoryRecall,
        *,
        person_fact_lines: list[str] | tuple[str, ...] = (),
        relationship_fact_lines: list[str] | tuple[str, ...] = (),
        preference_fact_lines: list[str] | tuple[str, ...] = (),
    ) -> list[str]:
        person_facts = tuple(
            dict.fromkeys(
                (
                    *person_fact_lines,
                    *layered_memory.person_facts,
                )
            )
        )
        relationship_facts = tuple(
            dict.fromkeys(
                (
                    *relationship_fact_lines,
                    *layered_memory.relationship_facts,
                )
            )
        )
        preference_facts = tuple(
            dict.fromkeys(
                (
                    *preference_fact_lines,
                    *layered_memory.preference_facts,
                )
            )
        )
        merged = LayeredMemoryRecall(
            person_facts=person_facts,
            relationship_facts=relationship_facts,
            preference_facts=preference_facts,
            recent_thread_facts=layered_memory.recent_thread_facts,
            other_facts=layered_memory.other_facts,
        )
        return merged.to_xml_lines()

    @staticmethod
    def _profile_fact_lines(
        dialogue_context: DialogueContextPack | None,
    ) -> tuple[
        list[str],
        list[str],
        list[str],
    ]:
        if dialogue_context is None:
            return [], [], []
        try:
            from .person_registry import format_person_fact_layers
        except Exception:
            return [], [], []

        person_facts: list[str] = []
        relationship_facts: list[str] = []
        preference_facts: list[str] = []
        seen_user_ids: set[str] = set()

        def append_profile(profile, *, prefix: str) -> None:
            user_id = str(getattr(profile, "user_id", "") or "")
            if not user_id or user_id in seen_user_ids:
                return
            seen_user_ids.add(user_id)
            layers = format_person_fact_layers(profile, prefix=prefix)
            person_facts.extend(layers.person_facts)
            relationship_facts.extend(layers.relationship_facts)
            preference_facts.extend(layers.preference_facts)

        append_profile(
            getattr(dialogue_context, "speaker_profile", None),
            prefix="speaker",
        )
        for index, person in enumerate(
            getattr(dialogue_context, "relevant_people", ()) or (),
            start=1,
        ):
            append_profile(getattr(person, "profile", None), prefix=f"person_{index}")
        return person_facts[:24], relationship_facts[:12], preference_facts[:12]

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

    def _is_nickname_cached(self, user_id: str) -> bool:
        """检查昵称是否在缓存中且未过期"""
        if user_id not in self._user_nickname_cache:
            return False
        cache_time = self._nickname_cache_time.get(user_id, 0)
        return time.time() - cache_time < self._nickname_ttl

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
        if self._is_nickname_cached(user_id):
            return self._user_nickname_cache.get(user_id)

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
                    self._user_nickname_cache[user_id] = nick
                    self._nickname_cache_time[user_id] = time.time()
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
    ) -> ChatInterChatHistory | None:
        """添加一次完整 ChatInter message timeline。"""
        session_id = self.get_session_id(user_id, group_id)

        formatted_user_message = uni_to_text_with_tags(user_message)
        formatted_response_summary = uni_to_text_with_tags(response_summary)

        async with self._lock:
            dialog = await ChatInterChatHistory.add_timeline(
                session_id=session_id,
                user_id=user_id,
                group_id=group_id,
                nickname=nickname,
                user_message=formatted_user_message,
                ai_response=formatted_response_summary,
                timeline=timeline,
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
        lines: list[str] = []
        reply_images: list[Image] = []
        history_messages: list[LLMMessage] = []
        current_message_text = ""

        try:
            if isinstance(raw_message, UniMessage):
                normalized_current_msg = remove_reply_segment(raw_message)
                current_message_text = uni_to_text_with_tags(normalized_current_msg)
            else:
                current_message_text = uni_to_text_with_tags(str(raw_message or ""))
        except Exception:
            current_message_text = str(raw_message or "")
        current_message_text = self._normalize_context_text(current_message_text)

        # 1. QQ Context 元数据
        qq_context_lines = [
            "<qq_context>",
            f"chatType={'group' if group_id else 'direct'}",
            f"userId={user_id}",
        ]
        if group_id:
            group_name = group_id
            if bot:
                try:
                    group_info = await bot.get_group_info(group_id=group_id)
                    if group_info and group_info.get("group_name"):
                        group_name = group_info.get("group_name")
                except Exception as e:
                    logger.debug(f"获取群聊名称失败：{e}")
            qq_context_lines.extend(
                [
                    f"groupId={group_id}",
                    f"groupName={group_name}",
                ]
            )
        qq_context_lines.extend(
            [
                f"senderName={nickname}",
                f"botName={self._bot_nickname or BotConfig.self_nickname}",
                f"botId={bot_id or 'unknown'}",
                "</qq_context>",
            ]
        )
        lines.extend(qq_context_lines)
        if dialogue_context is not None:
            packed_context = dialogue_context.to_context_xml()
            if packed_context:
                lines.append(packed_context)
        lines.extend(
            self._build_conversation_focus(
                current_message_text,
                dialogue_state=dialogue_state,
            )
        )
        if dialogue_state is not None:
            lines.append(dialogue_state.to_xml())
        thread = getattr(dialogue_context, "thread", None)
        addressee = getattr(dialogue_context, "addressee", None)
        thread_id = str(getattr(thread, "thread_id", "") or "").strip()
        thread_user_ids = tuple(
            str(item)
            for item in getattr(thread, "related_user_ids", ()) or ()
            if str(item)
        )
        addressee_user_id = str(getattr(addressee, "target_user_id", "") or "")

        session_id = self.get_session_id(user_id, group_id)
        from .history_policy import (
            append_chatroom_history_context,
            build_astr_history_payload,
        )

        history_payload = await build_astr_history_payload(
            session_id=session_id,
            user_id=user_id,
            current_message_text=current_message_text,
            group_id=group_id,
            bot_id=bot_id,
            dialog_limit=min(max(int(SESSION_CONTEXT_LIMIT), 1), 12),
            chatroom_limit=min(max(int(SESSION_CONTEXT_LIMIT), 1), 16),
        )
        history_messages = list(history_payload.messages)
        append_chatroom_history_context(lines, history_payload.chatroom_lines)

        recall_context = MemoryRecallContext.build(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            thread_id=thread_id or None,
            topic_key=str(getattr(thread, "topic_key", "") or ""),
            participants=thread_user_ids,
            addressee_user_id=addressee_user_id or None,
            query=current_message_text,
        )
        layered_memory = await ChatMemoryStore.recall_layered(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            query=current_message_text,
            recall_context=recall_context,
        )
        if dialogue_context is not None:
            try:
                setattr(dialogue_context, "layered_memory", layered_memory)
            except Exception:
                pass
        (
            person_fact_lines,
            relationship_fact_lines,
            preference_fact_lines,
        ) = self._profile_fact_lines(dialogue_context)
        memory_lines = self._build_layered_memory_xml(
            layered_memory,
            person_fact_lines=person_fact_lines,
            relationship_fact_lines=relationship_fact_lines,
            preference_fact_lines=preference_fact_lines,
        )
        if memory_lines:
            lines.append("<long_term_memory>")
            lines.extend(memory_lines)
            lines.append("</long_term_memory>")

        # 4. 当前消息层（Layer 0 + 回复链追溯）
        (
            current_message_layers_lines,
            reply_images,
        ) = await self._build_current_message_layers(
            group_id, raw_message, nickname, bot_id, bot, event
        )
        if current_message_layers_lines:
            lines.append("<current_message_layers>")
            lines.extend(current_message_layers_lines)
            lines.append("</current_message_layers>")

        # 5. 好感度信息
        impression = 0.0
        attitude = "一般"
        if USE_SIGN_IN_IMPRESSION:
            impression, attitude = await self.get_user_impression(user_id)
            lines.append("<user_state>")
            lines.append(f"impression={impression:.0f}")
            lines.append(f"attitude={attitude}")
            lines.append("</user_state>")

        context_xml = "\n".join(lines)
        system_prompt = self._build_system_prompt(
            impression,
            attitude,
            current_message_text=current_message_text,
            user_id=user_id,
            group_id=group_id,
            scenario=scenario,
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
    ) -> tuple[list[str], list[Image]]:
        """构建当前消息层 XML（Layer 0 + 回复链追溯）

        结构：
        - Layer 0: 当前消息
        - Layer 1-N: 回复链追溯（如果有）

        返回:
            tuple: (xml_lines, reply_images)
            - xml_lines: XML 行列表
            - reply_images: 回复链中的图片 Image Segment 列表
        """
        lines: list[str] = []
        reply_images: list[Image] = []

        try:
            if isinstance(raw_message, UniMessage):
                uni_msg = raw_message
            else:
                uni_msg = UniMessage.of(cast(Message, raw_message))
        except Exception:
            uni_msg = None

        # Layer 0: 当前消息
        if uni_msg:
            current_msg = remove_reply_segment(uni_msg)
            current_message = uni_to_text_with_tags(current_msg)
        else:
            current_message = str(raw_message)

        lines.append(f"[Layer 0][current] [{nickname}]: {current_message}")

        # Layer 1-N: 回复链追溯
        if group_id and bot:
            max_layers = MAX_REPLY_LAYERS
            reply_id = None

            # 从多个来源提取回复 ID（优先级：UniMessage > event > 字符串）
            if isinstance(raw_message, UniMessage):
                reply_id = extract_reply_from_message(raw_message)

            if not reply_id and event and bot:
                try:
                    reply_seg = await reply_fetch(event, bot)
                    if reply_seg and hasattr(reply_seg, "id") and reply_seg.id:
                        reply_id = str(reply_seg.id)
                except Exception as e:
                    logger.debug(f"从 reply_fetch 获取回复 ID 失败：{e}")

            if reply_id:
                seen_ids: set[str] = set()
                current_reply_id = reply_id

                for layer in range(1, max_layers + 1):
                    if current_reply_id in seen_ids:
                        break
                    seen_ids.add(current_reply_id)

                    try:
                        msg_data = await bot.get_msg(message_id=current_reply_id)
                        if not msg_data:
                            break

                        msg_user_id = str(msg_data.get("user_id", ""))
                        raw_msg = msg_data.get("message", "")

                        try:
                            from nonebot.adapters.onebot.v11 import Message as OBMessage

                            if isinstance(raw_msg, list):
                                from nonebot_plugin_alconna.uniseg import At, Text

                                uni_msg_layer = UniMessage()
                                for seg in raw_msg:
                                    seg_type = seg.get("type", "")
                                    seg_data = seg.get("data", {})
                                    if seg_type == "text":
                                        uni_msg_layer.append(
                                            Text(seg_data.get("text", ""))
                                        )
                                    elif seg_type == "at":
                                        qq = seg_data.get("qq", "")
                                        uni_msg_layer.append(At(target=qq, flag="user"))
                                    elif seg_type == "image":
                                        file = str(seg_data.get("file", "")).strip()
                                        url = str(seg_data.get("url", "")).strip()
                                        image_segment = (
                                            await self._build_reply_image_segment(
                                                bot=bot,
                                                file_value=file,
                                                url_value=url,
                                            )
                                        )
                                        if image_segment:
                                            uni_msg_layer.append(image_segment)
                                            reply_images.append(image_segment)
                                    elif seg_type == "reply":
                                        pass
                            elif isinstance(raw_msg, str):
                                parsed_msg = OBMessage(raw_msg)
                                if parsed_msg:
                                    from nonebot_plugin_alconna.uniseg import At, Text

                                    uni_msg_layer = UniMessage()
                                    for seg in parsed_msg:
                                        seg_type = getattr(seg, "type", "")
                                        seg_data = getattr(seg, "data", {}) or {}
                                        if seg_type == "text":
                                            uni_msg_layer.append(
                                                Text(str(seg_data.get("text", "")))
                                            )
                                        elif seg_type == "at":
                                            qq = str(seg_data.get("qq", "")).strip()
                                            if qq:
                                                uni_msg_layer.append(
                                                    At(target=qq, flag="user")
                                                )
                                        elif seg_type == "image":
                                            file = str(seg_data.get("file", "")).strip()
                                            url = str(seg_data.get("url", "")).strip()
                                            image_segment = (
                                                await self._build_reply_image_segment(
                                                    bot=bot,
                                                    file_value=file,
                                                    url_value=url,
                                                )
                                            )
                                            if image_segment:
                                                uni_msg_layer.append(image_segment)
                                                reply_images.append(image_segment)
                                        elif seg_type == "reply":
                                            pass
                                else:
                                    uni_msg_layer = UniMessage.text(str(raw_msg))
                            else:
                                uni_msg_layer = UniMessage.text(str(raw_msg))
                        except Exception as e:
                            logger.debug(f"转换消息为 UniMessage 失败：{e}")
                            uni_msg_layer = None

                        plain_text = (
                            uni_msg_layer.extract_plain_text()
                            if uni_msg_layer
                            else str(raw_msg)
                        )
                        is_bot_msg = bot_id and msg_user_id == str(bot_id)

                        if not is_bot_msg:
                            cached_nick = await self._fetch_user_nickname(
                                msg_user_id, group_id
                            )
                            if cached_nick:
                                self._user_nickname_cache[msg_user_id] = cached_nick

                        if is_bot_msg:
                            sender = (
                                f"[{self._bot_nickname or BotConfig.self_nickname}]"
                            )
                        else:
                            cached_nick = self._user_nickname_cache.get(msg_user_id)
                            sender = (
                                f"[{cached_nick}]"
                                if cached_nick
                                else f"[QQ:{msg_user_id}]"
                            )

                        content = (
                            uni_to_text_with_tags(uni_msg_layer)
                            if uni_msg_layer
                            else plain_text
                        )
                        content = content or "(空消息)"

                        lines.append(f"[Layer {layer}][reply][from:{sender}] {content}")

                        next_reply_id = None
                        if isinstance(raw_msg, list):
                            for seg in raw_msg:
                                if seg.get("type") == "reply":
                                    next_reply_id = seg.get("data", {}).get("id")
                                    break

                        if not next_reply_id:
                            if uni_msg_layer is not None:
                                next_reply_id = extract_reply_from_message(
                                    uni_msg_layer
                                )

                        if not next_reply_id:
                            break
                        current_reply_id = next_reply_id

                    except Exception as e:
                        logger.error(f"获取回复失败 layer={layer}: {e}")
                        break

        return lines, reply_images

    def _build_system_prompt(
        self,
        impression: float,
        attitude: str,
        current_message_text: str = "",
        user_id: str = "",
        group_id: str | None = None,
        scenario: str = "",
    ) -> str:
        """构建系统提示词"""
        persona = resolve_persona(
            user_id=user_id,
            group_id=group_id,
            scenario=scenario,
        ).persona
        chat_style = persona.style
        if CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX and self._is_complex_query(
            current_message_text
        ):
            length_rule = (
                "当前问题偏复杂（如代码/排错/实现类），允许详细分点回答并给出可执行步骤，"
                "不受80字限制。"
            )
        else:
            length_rule = "默认控制在80字以内，除非用户明确要求详细步骤。"

        base = build_chat_base_prompt(
            self._bot_nickname or BotConfig.self_nickname,
            chat_style,
            length_rule,
        )
        if USE_SIGN_IN_IMPRESSION:
            impression_rule = build_global_attitude_prompt(impression, attitude)
        else:
            impression_rule = ""

        custom_prompt = persona.prompt_fragment()
        custom_prompt_text = f"\n额外人格设定：{custom_prompt}" if custom_prompt else ""

        return base + impression_rule + custom_prompt_text

    async def get_user_impression(self, user_id: str) -> tuple[float, str]:
        """获取用户好感度"""
        return await get_user_impression_with_cache(user_id)

    async def reset_session_history(self, user_id: str, group_id: str | None) -> int:
        """重置会话历史（软删除，标记 reset=True）

        参数:
            user_id: 用户 ID
            group_id: 群组 ID

        返回:
            int: 被重置的对话数量
        """
        session_id = self.get_session_id(user_id, group_id)
        return await ChatInterChatHistory.reset_session(session_id)


_chat_memory = ChatMemory()
