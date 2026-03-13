"""
ChatInter - 聊天响应处理

实现聊天意图处理和消息响应生成。
"""

import asyncio

from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message, PrivateMessageEvent
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_uninfo import Uninfo

from zhenxun.services import chat, logger
from zhenxun.utils.message import MessageUtils

from .config import get_config_value
from .memory import _chat_memory
from .models.pydantic_models import IntentAnalysisResult


async def handle_chat_intent(
    message: str,
    user_id: str,
    group_id: str | None,
    analysis_result: IntentAnalysisResult,
    nickname: str = "用户",
) -> str | UniMessage:
    """处理普通聊天意图"""
    if analysis_result.chat_intent and analysis_result.chat_intent.response:
        await _chat_memory.add_dialog(
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            user_message=message,
            ai_response=analysis_result.chat_intent.response,
        )
        return analysis_result.chat_intent.response

    return await handle_chat_message(message, user_id, group_id, nickname)


async def handle_chat_message(
    message: str, user_id: str, group_id: str | None = None, nickname: str = "用户"
) -> str | UniMessage:
    """处理聊天消息"""
    chat_style = get_config_value("CHAT_STYLE", "")

    system_prompt = await build_chat_system_prompt(
        user_id=user_id,
        nickname=nickname,
        group_id=group_id,
        chat_style=chat_style,
    )

    logger.debug(f"系统提示词：{system_prompt[:500]}...")

    try:
        response = await chat(
            message=message,
            instruction=system_prompt,
            model=get_config_value("INTENT_MODEL", None),
        )

        reply_text = response.text if response else "抱歉，我现在有点累，稍后再聊吧~"
        await _chat_memory.add_dialog(
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            user_message=message,
            ai_response=reply_text,
        )
        return reply_text

    except Exception as e:
        logger.error(f"聊天处理失败：{e}")
        return MessageUtils.build_failure_message()


async def build_chat_system_prompt(
    user_id: str,
    nickname: str,
    group_id: str | None = None,
    group_name: str | None = None,
    chat_style: str = "",
) -> str:
    """构建聊天系统提示词

    参数:
        user_id: 用户 ID
        nickname: 用户昵称
        group_id: 群组 ID
        group_name: 群组名称
        chat_style: 聊天风格

    返回:
        str: 系统提示词
    """
    use_sign_in_impression = get_config_value("USE_SIGN_IN_IMPRESSION", True)

    impression_prompt = ""
    if use_sign_in_impression:
        impression, attitude = await _chat_memory.get_user_impression(user_id)
        impression_prompt = (
            f"\n\n用户：{nickname} | 好感度：{impression:.0f} | 态度：{attitude}\n"
            f"按态度回复：排斥/警惕→冷淡简短；一般/可以交流→正常友好；好朋友/是个好人→热情；亲密/恋人→亲密关心"
        )

    if chat_style:
        base_prompt = f"你是{chat_style}风格的机器人助手。回复简洁自然，像人类一样随性，优先使用中文。"
    else:
        base_prompt = "你是友好、热情的机器人助手。回复简洁自然，像人类一样随性，优先使用中文。"

    if group_id and group_name:
        group_prompt = f"\n群组：{group_name}({group_id})"
    elif group_id:
        group_prompt = f"\n群组 ID：{group_id}"
    else:
        group_prompt = ""

    return base_prompt + impression_prompt + group_prompt


async def reroute_to_plugin(
    bot: Bot, event: Event, command: str, session: Uninfo
) -> bool:
    """将消息重路由到目标插件

    参数:
        bot: Bot 实例
        event: 事件对象
        command: 命令文本
        session: 会话信息

    返回:
        bool: 是否重路由成功
    """
    try:
        import time

        event_data = event.model_dump()
        command_text = command.strip()
        new_message = Message(command_text)

        event_data["message"] = new_message
        event_data["raw_message"] = command_text
        event_data["plain_text"] = command_text

        if hasattr(bot, "self_id"):
            event_data["self_id"] = bot.self_id

        event_data["message_id"] = int(time.time() * 1000)
        event_data["time"] = int(time.time())

        logger.debug(f"构造重路由消息：'{command_text}', self_id={event_data.get('self_id')}")

        if isinstance(event, GroupMessageEvent):
            new_event = GroupMessageEvent(**event_data)
        elif isinstance(event, PrivateMessageEvent):
            new_event = PrivateMessageEvent(**event_data)
        else:
            logger.warning(f"不支持的事件类型：{type(event)}")
            return False

        setattr(new_event, "_ai_triggered", True)
        asyncio.create_task(bot.handle_event(new_event))
        logger.info(f"消息重路由成功：{command_text}")
        return True

    except Exception as e:
        logger.error(f"消息重路由失败：{e}")
        return False


__all__ = [
    "handle_chat_intent",
    "handle_chat_message",
    "build_chat_system_prompt",
    "reroute_to_plugin",
]
