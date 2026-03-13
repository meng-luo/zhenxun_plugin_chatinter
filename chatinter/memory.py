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
import time

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import Image, UniMessage

from zhenxun.configs.config import BotConfig
from zhenxun.models.chat_history import ChatHistory
from zhenxun.services import logger

from .config import get_config_value
from .utils.cache import get_user_impression_with_cache
from .utils.unimsg_utils import (
    uni_to_text_with_tags,
    extract_reply_from_message,
    remove_reply_segment,
)
from nonebot_plugin_alconna.uniseg.tools import reply_fetch
from .models.chat_history import ChatInterChatHistory


class ChatMemory:
    """聊天记忆管理"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._bot_nickname: str | None = None
        self._user_nickname_cache: dict[str, str] = {}
        self._nickname_cache_time: dict[str, float] = {}
        self._nickname_ttl = 30 * 60

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
                group_id=group_id,
                user_id=user_id
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
            uid for uid in user_ids
            if not self._is_nickname_cached(uid)
        }

        if not user_ids_to_fetch:
            return

        from zhenxun.models.group_member_info import GroupInfoUser
        members = await GroupInfoUser.filter(
            group_id=group_id,
            user_id__in=list(user_ids_to_fetch)
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
                    if group_info and group_info.get('group_name'):
                        group_name = group_info.get('group_name')
                except Exception as e:
                    logger.debug(f"获取群聊名称失败：{e}")
            qq_context_lines.extend([
                f"groupId={group_id}",
                f"groupName={group_name}",
            ])
        qq_context_lines.extend([
            f"senderName={nickname}",
            f"botName={self._bot_nickname or BotConfig.self_nickname}",
            f"botId={bot_id or 'unknown'}",
            "</qq_context>",
        ])
        lines.extend(qq_context_lines)

        # 2. 对话历史（来自 ChatInterChatHistory）
        history_context_lines = await self._build_history_context(
            user_id, group_id, nickname, bot_id
        )
        if history_context_lines:
            lines.append("<history_context>")
            lines.extend(history_context_lines)
            lines.append("</history_context>")

        # 3. 群聊背景（最近 5 条群消息，来自 ChatHistory）
        group_background_lines = await self._build_group_background_xml(user_id, group_id, bot_id)
        if group_background_lines:
            lines.append("<history>")
            lines.extend(group_background_lines)
            lines.append("</history>")

        # 4. 当前消息层（Layer 0 + 回复链追溯）
        current_message_layers_lines, reply_images = await self._build_current_message_layers(
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
        system_prompt = self._build_system_prompt(group_id, impression, attitude)

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
                    if reply_seg and hasattr(reply_seg, 'id') and reply_seg.id:
                        reply_id = str(reply_seg.id)
                except Exception as e:
                    logger.debug(f"从 reply_fetch 获取回复 ID 失败：{e}")

            if not reply_id:
                reply_id = extract_reply_from_message(raw_message) if isinstance(raw_message, str) else None

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

                        msg_user_id = str(msg_data.get('user_id', ''))
                        raw_msg = msg_data.get('message', '')

                        try:
                            if isinstance(raw_msg, list):
                                from nonebot_plugin_alconna.uniseg import At, Image, Text
                                uni_msg_layer = UniMessage()
                                for seg in raw_msg:
                                    seg_type = seg.get('type', '')
                                    seg_data = seg.get('data', {})
                                    if seg_type == 'text':
                                        uni_msg_layer.append(Text(seg_data.get('text', '')))
                                    elif seg_type == 'at':
                                        qq = seg_data.get('qq', '')
                                        uni_msg_layer.append(At(target=qq, flag='user'))
                                    elif seg_type == 'image':
                                        file = seg_data.get('file', '')
                                        url = seg_data.get('url', '')
                                        if file.startswith('http'):
                                            uni_msg_layer.append(Image(url=file))
                                            reply_images.append(Image(url=file))
                                        elif url:
                                            uni_msg_layer.append(Image(url=url))
                                            reply_images.append(Image(url=url))
                                        else:
                                            uni_msg_layer.append(Image(path=file))
                                            reply_images.append(Image(path=file))
                                    elif seg_type == 'reply':
                                        pass
                            else:
                                uni_msg_layer = UniMessage.text(str(raw_msg))
                        except Exception as e:
                            logger.debug(f"转换消息为 UniMessage 失败：{e}")
                            uni_msg_layer = None

                        plain_text = uni_msg_layer.extract_plain_text() if uni_msg_layer else str(raw_msg)
                        is_bot_msg = bot_id and msg_user_id == str(bot_id)

                        if not is_bot_msg:
                            cached_nick = await self._fetch_user_nickname(msg_user_id, group_id)
                            if cached_nick:
                                self._user_nickname_cache[msg_user_id] = cached_nick

                        if is_bot_msg:
                            sender = f"[{self._bot_nickname or BotConfig.self_nickname}]"
                        else:
                            cached_nick = self._user_nickname_cache.get(msg_user_id)
                            sender = f"[{cached_nick}]" if cached_nick else f"[QQ:{msg_user_id}]"

                        content = uni_to_text_with_tags(uni_msg_layer) if uni_msg_layer else plain_text
                        content = content or "(空消息)"

                        lines.append(f"[Layer {layer}][reply][from:{sender}] {content}")

                        next_reply_id = None
                        if isinstance(raw_msg, list):
                            for seg in raw_msg:
                                if seg.get('type') == 'reply':
                                    next_reply_id = seg.get('data', {}).get('id')
                                    break

                        if not next_reply_id:
                            next_reply_id = extract_reply_from_message(uni_msg_layer or plain_text)

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
        max_context = get_config_value("SESSION_CONTEXT_LIMIT", 20)
        session_id = self.get_session_id(user_id, group_id)
        dialogs = await ChatInterChatHistory.get_recent_dialogs(session_id, max_context)

        if not dialogs:
            return []

        if group_id:
            user_ids_to_fetch = {
                dlg.user_id for dlg in dialogs if not self._is_nickname_cached(dlg.user_id)
            }
            await self._preload_nicknames_for_group(user_ids_to_fetch, group_id)

        history_lines: list[str] = []
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

            user_msg = uni_to_text_with_tags(dlg.user_message)
            history_lines.append(f"[{timestamp}] {sender}: {user_msg}")

            if dlg.ai_response:
                ai_sender = f"[{self._bot_nickname or BotConfig.self_nickname}]"
                ai_msg = uni_to_text_with_tags(dlg.ai_response)
                history_lines.append(f"[{timestamp}] {ai_sender}: {ai_msg}")

        return history_lines

    async def _build_group_background_xml(
        self,
        user_id: str,
        group_id: str | None,
        bot_id: str | None = None,
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
        prefix_size = get_config_value("CONTEXT_PREFIX_SIZE", 5)

        if not group_id:
            return lines

        chat_history_msgs = (
            await ChatHistory.filter(group_id=group_id)
            .order_by("-create_time", "-id")
            .limit(prefix_size)
        )
        chat_history_msgs = list(reversed(chat_history_msgs))

        if not chat_history_msgs:
            return lines

        user_ids_to_fetch = set()
        for msg in chat_history_msgs:
            is_bot_msg = bot_id and msg.user_id == bot_id
            if not is_bot_msg and not self._is_nickname_cached(msg.user_id):
                user_ids_to_fetch.add(msg.user_id)

        await self._preload_nicknames_for_group(user_ids_to_fetch, group_id)

        for msg in chat_history_msgs:
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

            content = uni_to_text_with_tags(msg.plain_text or msg.text or "")
            content = content or "(空消息)"
            lines.append(f"[{timestamp}] {sender}: {content}")

        return lines

    def _build_system_prompt(
        self,
        group_id: str | None,
        impression: float,
        attitude: str,
    ) -> str:
        """构建系统提示词"""
        chat_style = get_config_value("CHAT_STYLE", "")
        use_sign_in_impression = get_config_value("USE_SIGN_IN_IMPRESSION", True)

        if chat_style:
            base = f"你是{self._bot_nickname or BotConfig.self_nickname}，一个{chat_style}机器人助手。回复简洁自然，优先使用中文。"
        else:
            base = f"你是{self._bot_nickname or BotConfig.self_nickname}，一个友好、热情的机器人助手。回复简洁自然，优先使用中文。"

        impression_rule = ""
        if use_sign_in_impression:
            impression_rule = (
                f"\n用户好感度：{impression:.0f}，态度：{attitude}。"
                "排斥/警惕→冷淡简短；一般/可以交流→正常友好；好朋友/是个好人→热情；亲密/恋人→亲密关心。回复风格符合用户好感度态度，即使对方好感度很低，你对他态度再差，也要温柔对待他，言语不能含有攻击性"
            )

        return base + impression_rule

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
