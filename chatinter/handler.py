"""
ChatInter - 主处理器

实现消息处理流程，支持多模态输入（图片识别）。
使用 UniMessage 统一处理消息。
"""

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_uninfo import Uninfo

from zhenxun.services import logger
from zhenxun.utils.message import MessageUtils

from .config import get_config_value
from .intent_analyzer import analyze_intent_safe
from .memory import _chat_memory
from .plugin_registry import get_user_plugin_knowledge
from .utils.unimsg_utils import uni_to_text_with_tags, remove_reply_segment
from .utils.multimodal import extract_images_from_message
from .chat_handler import handle_chat_intent, handle_chat_message, reroute_to_plugin


_HANDLED_MESSAGE_IDS: set[str] = set()
_MAX_HANDLED_CACHE = 1000


def _is_already_handled(event: Event) -> bool:
    """检查消息是否已被本插件处理过"""
    message_id = getattr(event, "message_id", None)
    if not message_id:
        return False
    return str(message_id) in _HANDLED_MESSAGE_IDS


def _mark_as_handled(event: Event):
    """标记消息已被处理"""
    message_id = getattr(event, "message_id", None)
    if not message_id:
        return
    if len(_HANDLED_MESSAGE_IDS) >= _MAX_HANDLED_CACHE:
        _HANDLED_MESSAGE_IDS.clear()
    _HANDLED_MESSAGE_IDS.add(str(message_id))


def _get_nickname(session: Uninfo) -> str:
    """获取用户昵称"""
    if session.user and hasattr(session.user, "display_name") and session.user.display_name:
        return session.user.display_name
    if session.user and hasattr(session.user, "name") and session.user.name:
        return session.user.name
    return "用户"


async def handle_fallback(
    bot: Bot, event: Event, session: Uninfo, raw_message: str, message=None
) -> bool:
    """消息处理器

    当消息未被其他插件处理时，使用 AI 分析用户意图并响应。

    参数:
        bot: Bot 实例
        event: 事件对象
        session: Uninfo 会话信息
        raw_message: 原始消息文本
        message: 原始消息对象（可选）

    返回:
        bool: 是否处理成功
    """
    if not get_config_value("ENABLE_FALLBACK", True):
        logger.debug("ChatInter 功能已禁用")
        return

    if _is_already_handled(event):
        logger.debug("消息已被处理，跳过")
        return

    _mark_as_handled(event)

    user_id = session.user.id
    group_id = session.group.id if session.group else None
    nickname = _get_nickname(session)
    bot_id = str(bot.self_id) if hasattr(bot, "self_id") else None

    try:
        knowledge_base = await get_user_plugin_knowledge()

        # 尝试使用 UniMessage 传递
        uni_msg = None
        if message:
            try:
                uni_msg = UniMessage.of(message)
            except Exception:
                pass

        system_prompt, context_xml, reply_images_data = await _chat_memory.build_full_context(
            user_id, group_id, nickname, uni_msg or raw_message, bot, bot_id, event
        )

        # 优先使用 UniMessage 处理
        if uni_msg:
            current_msg = remove_reply_segment(uni_msg)
            current_message = uni_to_text_with_tags(current_msg)
        else:
            # 无法解析为 UniMessage 时，使用原始消息文本
            current_message = raw_message.strip()

        # 提取图片（多模态处理）
        image_parts = await extract_images_from_message(bot, event, uni_msg or message or raw_message)
        if image_parts:
            logger.debug(f"当前消息中包含 {len(image_parts)} 张图片")

        # 提取回复链中的图片（直接使用 Image Segment 处理）
        if reply_images_data:
            from .utils.multimodal import _process_image_segment
            for img_seg in reply_images_data:
                image_part = await _process_image_segment(img_seg)
                if image_part:
                    image_parts.append(image_part)
            if reply_images_data:
                logger.debug(f"回复链中包含 {len(reply_images_data)} 张图片")

        # 分析意图
        result = await analyze_intent_safe(
            current_message,
            knowledge_base,
            system_prompt,
            context_xml,
            user_id,
            nickname,
            group_id,
            image_parts,
        )

        logger.info(f"意图分析结果：action={result.action}")

        if result.action == "call_plugin" and result.plugin_intent:
            command = result.plugin_intent.command
            response_text = result.plugin_intent.response
            logger.info(
                f"识别为插件调用：plugin={result.plugin_intent.plugin_name}, "
                f"command={command}, confidence={result.plugin_intent.confidence:.2f}"
            )

            await _chat_memory.add_dialog(
                user_id=user_id,
                group_id=group_id,
                nickname=nickname,
                user_message=uni_msg or raw_message,
                ai_response=response_text,
                bot_id=bot_id,
            )

            await MessageUtils.build_message(response_text).send()

            success = await reroute_to_plugin(bot, event, command, session)
            if success:
                return
            else:
                logger.warning("重路由失败，降级为聊天处理")
                reply = await handle_chat_message(raw_message, user_id, group_id, nickname)
                await MessageUtils.build_message(reply).send()
                return

        else:
            logger.info(f"识别为普通聊天 (action={result.action})")
            reply = await handle_chat_intent(raw_message, user_id, group_id, result, nickname)
            await MessageUtils.build_message(reply).send()
            return

    except Exception as e:
        logger.error(f"ChatInter 处理失败：{e}")
        await MessageUtils.build_failure_message().send()
        return


__all__ = [
    "handle_fallback",
]
