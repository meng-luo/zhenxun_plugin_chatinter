"""
ChatInter - 聊天记忆管理

实现 3 层上下文结构：
1. System: 系统设定
2. Context: 语境层（XML 标签包裹）
   - <qq_context>: QQ 上下文元数据
   - <context_layers>: 回复链追溯 + 群聊历史
   - <history>: 群聊历史记录
3. Current: 当前用户消息

使用 UniMessage 统一处理消息。
"""

import asyncio
from collections import Counter
import re
import time

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import Image, UniMessage
from nonebot_plugin_alconna.uniseg.tools import reply_fetch

from zhenxun.configs.config import BotConfig
from zhenxun.models.chat_history import ChatHistory
from zhenxun.services import logger

from .config import get_config_value
from .models.chat_history import ChatInterChatHistory
from .utils.cache import get_user_impression_with_cache
from .utils.unimsg_utils import (
    extract_reply_from_message,
    remove_reply_segment,
    uni_to_text_with_tags,
)

_CONTEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,8}", re.IGNORECASE)
_LOW_VALUE_ONLY_SYMBOLS = re.compile(r"^[\W_]+$", re.UNICODE)
_COMMAND_LIKE_PREFIX = ("/", ".", "!", "。", "！")
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
_FOLLOWUP_QUERY_HINTS = (
    "然后",
    "接着",
    "继续",
    "再来",
    "再说",
    "展开",
    "详细",
    "具体",
    "这个",
    "那个",
    "它",
    "他",
    "她",
    "刚才",
    "上面",
    "上一个",
    "前面",
)


class ChatMemory:
    """聊天记忆管理"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._bot_nickname: str | None = None
        self._user_nickname_cache: dict[str, str] = {}
        self._nickname_cache_time: dict[str, float] = {}
        self._nickname_ttl = 30 * 60
        self._compression_fetch_factor = 3
        self._compression_summary_cap = 12

    @staticmethod
    def _clip_context_line(text: str, limit: int = 140) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: max(24, limit - 1)].rstrip()}…"

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
    def _similarity_score(cls, query_tokens: list[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = cls._tokenize_context_text(text)
        if not text_tokens:
            return 0.0

        query_counter = Counter(query_tokens)
        text_counter = Counter(text_tokens)
        overlap = sum(
            min(count, text_counter.get(token, 0))
            for token, count in query_counter.items()
        )
        if overlap <= 0:
            return 0.0

        precision = overlap / max(sum(text_counter.values()), 1)
        recall = overlap / max(sum(query_counter.values()), 1)
        if precision + recall <= 0:
            base_score = 0.0
        else:
            base_score = 2 * precision * recall / (precision + recall)

        query_text = "".join(query_tokens)
        normalized_text = cls._normalize_context_text(text).lower()
        phrase_bonus = 0.0
        if len(query_text) >= 4 and query_text in normalized_text:
            phrase_bonus = 0.12
        return min(base_score + phrase_bonus, 1.0)

    @classmethod
    def _is_low_value_background_message(cls, text: str) -> bool:
        normalized = cls._normalize_context_text(text)
        if not normalized:
            return True
        if len(normalized) <= 1:
            return True
        if normalized.startswith(_COMMAND_LIKE_PREFIX):
            return True
        if normalized.lower().startswith(("http://", "https://")) and len(
            normalized
        ) <= 64:
            return True
        if _LOW_VALUE_ONLY_SYMBOLS.fullmatch(normalized):
            return True
        return False

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

    @classmethod
    def _is_followup_query(cls, text: str) -> bool:
        normalized = cls._normalize_context_text(text).lower()
        if not normalized:
            return False
        if len(normalized) <= 4:
            return any(hint in normalized for hint in _FOLLOWUP_QUERY_HINTS)
        if len(normalized) <= 10 and any(
            hint in normalized for hint in _FOLLOWUP_QUERY_HINTS
        ):
            return True
        return normalized.startswith(("那", "再", "继续", "然后"))

    async def _should_isolate_context(
        self,
        *,
        session_id: str,
        group_id: str | None,
        current_message_text: str,
    ) -> tuple[bool, str]:
        if not bool(get_config_value("ENABLE_CONTEXT_RELEVANCE_GATE", True)):
            return False, "disabled"

        normalized_query = self._normalize_context_text(current_message_text)
        if not normalized_query:
            return False, "empty_query"

        if self._is_followup_query(normalized_query):
            return False, "followup_query"

        query_tokens = self._tokenize_context_text(normalized_query)
        min_query_tokens = max(
            int(get_config_value("CONTEXT_RELEVANCE_MIN_QUERY_TOKENS", 1) or 1), 1
        )
        if len(query_tokens) < min_query_tokens:
            return False, "insufficient_query_tokens"

        sample_limit = max(
            int(get_config_value("CONTEXT_RELEVANCE_SAMPLE_LIMIT", 18) or 18), 4
        )
        threshold = float(get_config_value("CONTEXT_RELEVANCE_THRESHOLD", 0.11) or 0.11)

        sample_texts: list[str] = []
        dialogs = await ChatInterChatHistory.get_recent_dialogs(
            session_id,
            sample_limit,
        )
        for dialog in dialogs:
            user_text = self._strip_non_final_channel_text(
                uni_to_text_with_tags(dialog.user_message)
            )
            ai_text = self._strip_non_final_channel_text(
                uni_to_text_with_tags(dialog.ai_response or "")
            )
            merged = self._normalize_context_text(f"{user_text} {ai_text}")
            if merged:
                sample_texts.append(merged)

        if group_id:
            group_msgs = (
                await ChatHistory.filter(group_id=group_id)
                .order_by("-create_time", "-id")
                .limit(sample_limit)
            )
            for msg in group_msgs:
                text = self._normalize_context_text(
                    uni_to_text_with_tags(msg.plain_text or msg.text or "")
                )
                if not text or self._is_low_value_background_message(text):
                    continue
                sample_texts.append(text)

        if not sample_texts:
            return False, "no_samples"

        max_score = 0.0
        for text in sample_texts:
            score = self._similarity_score(query_tokens, text)
            if score > max_score:
                max_score = score

        if max_score < threshold:
            return True, f"max_score={max_score:.3f}<threshold={threshold:.3f}"
        return False, f"max_score={max_score:.3f}"

    def _build_conversation_focus(
        self,
        current_message_text: str,
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

        response_mode = "detailed" if self._is_complex_query(normalized) else "concise"
        return [
            "<conversation_focus>",
            f"query_keywords={','.join(top_keywords)}",
            f"response_mode={response_mode}",
            "</conversation_focus>",
        ]

    @staticmethod
    def _strip_non_final_channel_text(text: str) -> str:
        normalized = str(text or "")
        if not normalized:
            return ""
        # 兼容历史遗留的通道标记，避免 analysis/commentary 污染后续上下文。
        normalized = re.sub(r"(?i)\[(analysis|commentary)\]\s*", "", normalized)
        normalized = re.sub(
            r"(?im)^\s*(analysis|commentary)\s*[:：]\s*",
            "",
            normalized,
        )
        return normalized.strip()

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

        if not url_text and file_text and not file_text.startswith(
            ("http://", "https://", "base64://")
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

    def _summarize_old_dialogs(self, dialogs: list["ChatInterChatHistory"]) -> str:
        if not dialogs:
            return ""

        token_pattern = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)
        all_text = " ".join(
            (
                f"{uni_to_text_with_tags(item.user_message)} "
                f"{uni_to_text_with_tags(item.ai_response or '')}"
            )
            for item in dialogs[-self._compression_summary_cap :]
        )
        tokens = [
            token.lower()
            for token in token_pattern.findall(all_text.lower())
            if token and token not in {"一个", "这个", "那个", "然后", "就是", "功能"}
        ]
        topics = ", ".join(word for word, _ in Counter(tokens).most_common(6))
        if not topics:
            topics = "无明显主题"
        return f"更早{len(dialogs)}轮对话摘要: 主要主题[{topics}]"

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
            from zhenxun.models.group_member_info import GroupInfoUser

            member = await GroupInfoUser.filter(
                group_id=group_id, user_id=user_id
            ).first()
            if member:
                nick = member.nickname or member.user_name
                if nick:
                    self._user_nickname_cache[user_id] = nick
                    self._nickname_cache_time[user_id] = time.time()
                    return nick

        return None

    async def _preload_nicknames_for_group(
        self, user_ids: set[str], group_id: str | None
    ):
        """批量预加载群成员昵称

        参数:
            user_ids: 需要获取昵称的用户 ID 集合
            group_id: 群组 ID
        """
        if not group_id:
            return

        user_ids_to_fetch = {
            uid for uid in user_ids if not self._is_nickname_cached(uid)
        }

        if not user_ids_to_fetch:
            return

        from zhenxun.models.group_member_info import GroupInfoUser

        members = await GroupInfoUser.filter(
            group_id=group_id, user_id__in=list(user_ids_to_fetch)
        ).all()

        for member in members:
            nick = member.nickname or member.user_name
            if nick:
                self._user_nickname_cache[member.user_id] = nick
                self._nickname_cache_time[member.user_id] = time.time()

    def set_bot_nickname(self, nickname: str):
        """设置 bot 昵称"""
        self._bot_nickname = nickname

    def get_session_id(self, user_id: str, group_id: str | None) -> str:
        """获取用于数据库存储的 session_id"""
        return group_id if group_id else user_id

    async def add_dialog(
        self,
        user_id: str,
        group_id: str | None,
        nickname: str,
        user_message: str | UniMessage,
        ai_response: str | UniMessage,
        bot_id: str | None = None,
    ):
        """添加一轮对话到数据库（一问一答）"""
        session_id = self.get_session_id(user_id, group_id)

        formatted_user_message = uni_to_text_with_tags(user_message)
        formatted_ai_response = uni_to_text_with_tags(ai_response)

        async with self._lock:
            await ChatInterChatHistory.add_dialog(
                session_id=session_id,
                user_id=user_id,
                group_id=group_id,
                nickname=nickname,
                user_message=formatted_user_message,
                ai_response=formatted_ai_response,
                bot_id=bot_id,
            )

    async def build_full_context(
        self,
        user_id: str,
        group_id: str | None,
        nickname: str,
        raw_message: str | UniMessage,
        bot: Bot | None = None,
        bot_id: str | None = None,
        event: Event | None = None,
    ) -> tuple[str, str, list[Image]]:
        """构建完整的上下文（System + Context + Current）

        参数:
            user_id: 用户 ID
            group_id: 群组 ID
            nickname: 用户昵称
            raw_message: 原始用户消息（UniMessage 或字符串）
            bot: Bot 实例（用于获取消息）
            bot_id: Bot ID
            event: Event 实例（用于获取回复 ID）

        返回:
            tuple: (system_prompt, context_xml, reply_images)
            - system_prompt: 系统提示词
            - context_xml: XML 格式的上下文
            - reply_images: 回复链中的图片 Image Segment 列表（用于多模态处理）
        """
        lines: list[str] = []
        reply_images: list[Image] = []
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
                    group_info = await bot.get_group_info(group_id=int(group_id))
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
        lines.extend(self._build_conversation_focus(current_message_text))

        session_id = self.get_session_id(user_id, group_id)
        isolate_context, isolate_reason = await self._should_isolate_context(
            session_id=session_id,
            group_id=group_id,
            current_message_text=current_message_text,
        )
        if isolate_context:
            logger.debug(
                "ChatInter 上下文门控命中，按单轮对话处理: "
                f"session={session_id}, reason={isolate_reason}"
            )
            lines.extend(
                [
                    "<context_mode>",
                    "mode=single_turn",
                    f"reason={isolate_reason}",
                    "</context_mode>",
                ]
            )

        # 2. 对话历史（来自 ChatInterChatHistory）
        if not isolate_context:
            history_context_lines = await self._build_history_context(
                user_id,
                group_id,
                nickname,
                bot_id,
                current_message_text=current_message_text,
            )
            if history_context_lines:
                lines.append("<history_context>")
                lines.extend(history_context_lines)
                lines.append("</history_context>")

        # 3. 群聊背景（最近 5 条群消息，来自 ChatHistory）
        if not isolate_context:
            group_background_lines = await self._build_group_background_xml(
                user_id,
                group_id,
                bot_id,
                current_message_text=current_message_text,
            )
            if group_background_lines:
                lines.append("<history>")
                lines.extend(group_background_lines)
                lines.append("</history>")

        # 4. 当前消息层（Layer 0 + 回复链追溯）
        (
            current_message_layers_lines,
            reply_images,
        ) = await self._build_current_message_layers(
            user_id, group_id, raw_message, nickname, bot_id, bot, event
        )
        if current_message_layers_lines:
            lines.append("<current_message_layers>")
            lines.extend(current_message_layers_lines)
            lines.append("</current_message_layers>")

        # 5. 好感度信息
        use_sign_in_impression = get_config_value("USE_SIGN_IN_IMPRESSION", True)
        impression = 0.0
        attitude = "一般"
        if use_sign_in_impression:
            impression, attitude = await self.get_user_impression(user_id)
            lines.append("<user_state>")
            lines.append(f"impression={impression:.0f}")
            lines.append(f"attitude={attitude}")
            lines.append("</user_state>")

        context_xml = "\n".join(lines)
        system_prompt = self._build_system_prompt(
            group_id,
            impression,
            attitude,
            current_message_text=current_message_text,
        )

        return system_prompt, context_xml, reply_images

    async def _build_current_message_layers(
        self,
        user_id: str,
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
                uni_msg = UniMessage.of(raw_message)
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
            max_layers = get_config_value("MAX_REPLY_LAYERS", 3)
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

            if not reply_id:
                reply_id = (
                    extract_reply_from_message(raw_message)
                    if isinstance(raw_message, str)
                    else None
                )

            if reply_id:
                seen_ids: set[str] = set()
                current_reply_id = reply_id

                for layer in range(1, max_layers + 1):
                    if current_reply_id in seen_ids:
                        break
                    seen_ids.add(current_reply_id)

                    try:
                        msg_data = await bot.get_msg(message_id=int(current_reply_id))
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
                            next_reply_id = extract_reply_from_message(
                                uni_msg_layer or plain_text
                            )

                        if not next_reply_id:
                            break
                        current_reply_id = next_reply_id

                    except Exception as e:
                        logger.error(f"获取回复失败 layer={layer}: {e}")
                        break

        return lines, reply_images

    async def _build_history_context(
        self,
        user_id: str,
        group_id: str | None,
        nickname: str,
        bot_id: str | None = None,
        current_message_text: str = "",
    ) -> list[str]:
        """构建对话历史 XML（来自 ChatInterChatHistory）

        格式：[MM-DD HH:MM:SS][发送者]: 内容

        参数:
            user_id: 用户 ID
            group_id: 群组 ID
            nickname: 当前用户昵称
            bot_id: Bot ID

        返回:
            list[str]: XML 行列表
        """
        max_context = max(int(get_config_value("SESSION_CONTEXT_LIMIT", 20) or 20), 1)
        session_id = self.get_session_id(user_id, group_id)
        recall_candidate_limit = max(
            int(get_config_value("HISTORY_RECALL_CANDIDATE_LIMIT", 60) or 60),
            max_context,
        )
        fetch_limit = max(
            max_context * self._compression_fetch_factor,
            recall_candidate_limit,
        )
        dialogs = await ChatInterChatHistory.get_recent_dialogs(session_id, fetch_limit)

        if not dialogs:
            return []

        history_summary = ""
        recalled_dialogs: list[ChatInterChatHistory] = []
        query_tokens = self._tokenize_context_text(current_message_text)
        recall_limit = max(int(get_config_value("HISTORY_RECALL_LIMIT", 4) or 4), 0)
        recall_min_score = float(
            get_config_value("HISTORY_RECALL_MIN_SCORE", 0.18) or 0.18
        )
        if len(dialogs) > max_context:
            old_dialogs = dialogs[:-max_context]
            dialogs = dialogs[-max_context:]
            history_summary = self._summarize_old_dialogs(old_dialogs)
            if old_dialogs and query_tokens and recall_limit > 0:
                recall_candidates: list[tuple[float, ChatInterChatHistory]] = []
                for dialog in old_dialogs:
                    merged_text = self._strip_non_final_channel_text(
                        f"{uni_to_text_with_tags(dialog.user_message)} "
                        f"{uni_to_text_with_tags(dialog.ai_response or '')}"
                    )
                    score = self._similarity_score(query_tokens, merged_text)
                    if score >= recall_min_score:
                        recall_candidates.append((score, dialog))
                recall_candidates.sort(
                    key=lambda item: (item[0], int(item[1].id or 0)),
                    reverse=True,
                )
                recalled_dialogs = [
                    dialog for _, dialog in recall_candidates[:recall_limit]
                ]
                recalled_dialogs.sort(key=lambda item: int(item.id or 0))

        if group_id:
            display_dialogs = [*dialogs, *recalled_dialogs]
            user_ids_to_fetch = {
                dlg.user_id
                for dlg in display_dialogs
                if not self._is_nickname_cached(dlg.user_id)
            }
            await self._preload_nicknames_for_group(user_ids_to_fetch, group_id)

        history_lines: list[str] = []
        if history_summary:
            history_lines.append(history_summary)
        if recalled_dialogs:
            history_lines.append(f"相关历史记忆({len(recalled_dialogs)}条):")
            for dlg in recalled_dialogs:
                if dlg.create_time:
                    timestamp = dlg.create_time.strftime("%m-%d %H:%M:%S")
                else:
                    timestamp = "??:??:??"
                if group_id:
                    cached_nick = self._user_nickname_cache.get(dlg.user_id)
                    sender = (
                        f"[{cached_nick}]"
                        if cached_nick
                        else f"[QQ:{dlg.user_id}]"
                    )
                else:
                    sender = f"[{dlg.nickname}]"

                recalled_user_msg = self._clip_context_line(
                    self._strip_non_final_channel_text(
                        uni_to_text_with_tags(dlg.user_message)
                    ),
                    120,
                )
                history_lines.append(f"[{timestamp}] {sender}: {recalled_user_msg}")

                if dlg.ai_response:
                    ai_sender = f"[{self._bot_nickname or BotConfig.self_nickname}]"
                    recalled_ai_msg = self._clip_context_line(
                        self._strip_non_final_channel_text(
                            uni_to_text_with_tags(dlg.ai_response)
                        ),
                        120,
                    )
                    history_lines.append(
                        f"[{timestamp}] {ai_sender}: {recalled_ai_msg}"
                    )

        for dlg in dialogs:
            if dlg.create_time:
                timestamp = dlg.create_time.strftime("%m-%d %H:%M:%S")
            else:
                timestamp = "??:??:??"

            if group_id:
                cached_nick = self._user_nickname_cache.get(dlg.user_id)
                sender = f"[{cached_nick}]" if cached_nick else f"[QQ:{dlg.user_id}]"
            else:
                sender = f"[{dlg.nickname}]"

            user_msg = self._strip_non_final_channel_text(
                uni_to_text_with_tags(dlg.user_message)
            )
            history_lines.append(f"[{timestamp}] {sender}: {user_msg}")

            if dlg.ai_response:
                ai_sender = f"[{self._bot_nickname or BotConfig.self_nickname}]"
                ai_msg = self._strip_non_final_channel_text(
                    uni_to_text_with_tags(dlg.ai_response)
                )
                history_lines.append(f"[{timestamp}] {ai_sender}: {ai_msg}")

        return history_lines

    async def _build_group_background_xml(
        self,
        user_id: str,
        group_id: str | None,
        bot_id: str | None = None,
        current_message_text: str = "",
    ) -> list[str]:
        """构建群聊背景 XML（最近 5 条群消息，来自 ChatHistory）

        格式：[时间][发送者]: 内容

        参数:
            user_id: 用户 ID
            group_id: 群组 ID
            bot_id: Bot ID

        返回:
            list[str]: XML 行列表
        """
        lines: list[str] = []
        prefix_size = max(int(get_config_value("CONTEXT_PREFIX_SIZE", 5) or 5), 1)
        fetch_multiplier = max(
            int(get_config_value("GROUP_BACKGROUND_FETCH_MULTIPLIER", 3) or 3),
            1,
        )
        relevant_limit = max(
            int(get_config_value("GROUP_BACKGROUND_RELEVANT_LIMIT", 3) or 3),
            0,
        )
        relevant_min_score = float(
            get_config_value("GROUP_BACKGROUND_MIN_SCORE", 0.16) or 0.16
        )

        if not group_id:
            return lines

        fetch_limit = max(prefix_size, prefix_size * fetch_multiplier)
        chat_history_msgs = (
            await ChatHistory.filter(group_id=group_id)
            .order_by("-create_time", "-id")
            .limit(fetch_limit)
        )
        chat_history_msgs = list(reversed(chat_history_msgs))

        if not chat_history_msgs:
            return lines

        query_tokens = self._tokenize_context_text(current_message_text)
        user_ids_to_fetch = set()
        for msg in chat_history_msgs:
            is_bot_msg = bot_id and msg.user_id == bot_id
            if not is_bot_msg and not self._is_nickname_cached(msg.user_id):
                user_ids_to_fetch.add(msg.user_id)

        await self._preload_nicknames_for_group(user_ids_to_fetch, group_id)

        scored_msgs: list[tuple[ChatHistory, str, float]] = []
        for msg in chat_history_msgs:
            content = uni_to_text_with_tags(msg.plain_text or msg.text or "")
            content = content or "(空消息)"
            if self._is_low_value_background_message(content):
                continue
            score = (
                self._similarity_score(query_tokens, content)
                if query_tokens
                else 0.0
            )
            scored_msgs.append((msg, content, score))

        if not scored_msgs:
            return lines

        baseline_keep = max(prefix_size - relevant_limit, 1)
        selected: list[tuple[ChatHistory, str, float]] = []
        selected_ids: set[int] = set()
        for item in scored_msgs[-baseline_keep:]:
            selected.append(item)
            selected_ids.add(int(item[0].id))

        if query_tokens and relevant_limit > 0:
            related_candidates = [
                item
                for item in scored_msgs
                if int(item[0].id) not in selected_ids and item[2] >= relevant_min_score
            ]
            related_candidates.sort(
                key=lambda item: (item[2], int(item[0].id)),
                reverse=True,
            )
            for item in related_candidates[:relevant_limit]:
                selected.append(item)
                selected_ids.add(int(item[0].id))

        selected.sort(key=lambda item: int(item[0].id))
        if len(selected) > prefix_size:
            selected = selected[-prefix_size:]

        for msg, content, _score in selected:
            if msg.create_time:
                timestamp = msg.create_time.strftime("%m-%d %H:%M:%S")
            else:
                timestamp = "??:??:??"
            is_bot_msg = bot_id and msg.user_id == bot_id
            if is_bot_msg:
                sender = f"[{self._bot_nickname or BotConfig.self_nickname}]"
            else:
                cached_nick = self._user_nickname_cache.get(msg.user_id)
                sender = f"[{cached_nick}]" if cached_nick else f"[QQ:{msg.user_id}]"

            lines.append(f"[{timestamp}] {sender}: {content}")

        return lines

    def _build_system_prompt(
        self,
        group_id: str | None,
        impression: float,
        attitude: str,
        current_message_text: str = "",
    ) -> str:
        """构建系统提示词"""
        chat_style = get_config_value("CHAT_STYLE", "")
        use_sign_in_impression = get_config_value("USE_SIGN_IN_IMPRESSION", True)
        allow_long_for_complex = bool(
            get_config_value("CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX", True)
        )
        if allow_long_for_complex and self._is_complex_query(current_message_text):
            length_rule = (
                "当前问题偏复杂（如代码/排错/实现类），允许详细分点回答并给出可执行步骤，"
                "不受80字限制。"
            )
        else:
            length_rule = "默认控制在80字以内，除非用户明确要求详细步骤。"

        if chat_style:
            base = (
                f"你是{self._bot_nickname or BotConfig.self_nickname}，"
                f"一个{chat_style}机器人助手。回复简洁自然，优先使用中文。"
                "语气偏日式二次元、软萌中带一点傲娇，避免生硬正式。"
                "可适度使用“好啦、诶嘿、唔、哼哼、欸”等口吻词，但不要堆叠。"
                f"{length_rule}"
                "上下文不足时先追问一个关键澄清问题，不要凭空猜测。"
                "结构化任务或命令输出时不要加入口癖修饰。"
            )
        else:
            base = (
                f"你是{self._bot_nickname or BotConfig.self_nickname}，"
                "一个日式二次元、软萌中带一点傲娇的机器人助手。"
                "回复简洁自然，优先使用中文，避免生硬正式。"
                "可适度使用“好啦、诶嘿、唔、哼哼、欸”等口吻词，但不要堆叠。"
                f"{length_rule}"
                "上下文不足时先追问一个关键澄清问题，不要凭空猜测。"
                "结构化任务或命令输出时不要加入口癖修饰。"
            )

        impression_rule = ""
        if use_sign_in_impression:
            impression_rule = (
                f"\n用户好感度：{impression:.0f}，态度：{attitude}。"
                "排斥/警惕→冷淡简短；一般/可以交流→正常友好；好朋友/是个好人→热情；亲密/恋人→亲密关心。回复风格符合用户好感度态度，即使对方好感度很低，你对他态度再差，也要温柔对待他，言语不能含有攻击性"
            )

        custom_prompt = get_config_value("CUSTOM_PROMPT", "")
        custom_prompt_text = f"\n额外设定：{custom_prompt}" if custom_prompt else ""

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

    async def clear_session_history(self, user_id: str, group_id: str | None):
        """清空会话历史（硬删除，包括已重置的）"""
        session_id = self.get_session_id(user_id, group_id)
        await ChatInterChatHistory.clear_session(session_id)


_chat_memory = ChatMemory()
