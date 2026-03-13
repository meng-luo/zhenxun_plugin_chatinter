"""
ChatInter - AI 意图识别插件

当用户消息未被其他插件匹配时，使用 AI 分析用户意图：
- 功能调用意图 -> 重路由到对应插件
- 普通聊天意图 -> 进行正常对话回复

使用 UniMessage 统一处理消息，支持多模态输入。
"""

from nonebot import get_driver, on_message
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata
from nonebot.rule import to_me
from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import GroupMessageEvent, PrivateMessageEvent
from nonebot_plugin_alconna import Alconna, on_alconna
from nonebot_plugin_alconna.uniseg import UniMsg
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig
from zhenxun.configs.utils import Command, PluginExtraData, RegisterConfig
from zhenxun.services.log import logger
from zhenxun.utils.message import MessageUtils

from .data_source import handle_fallback, _chat_memory
from .plugin_registry import PluginRegistry
from .models.chat_history import ChatInterChatHistory  # noqa: F401

driver = get_driver()


__plugin_meta__ = PluginMetadata(
    name="ChatInter",
    description="当消息未被其他插件处理时，使用 AI 分析用户意图并智能响应",
    usage="""
    ChatInter 功能，自动识别用户意图
    """.strip(),
    extra=PluginExtraData(
        author="meng-luo",
        version="1.0.0",
        menu_type="其他",
        superuser_help="""
- `重置会话`
        """.strip(),
        configs=[
            RegisterConfig(
                key="ENABLE_FALLBACK",
                value=True,
                help="是否启用 ChatInter 功能",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                key="INTENT_MODEL",
                value=None,
                help="ChatInter 使用的模型，具体参考 AI 模块",
                default_value=None,
                type=str | None,
            ),
            RegisterConfig(
                key="INTENT_TIMEOUT",
                value=30,
                help="意图识别超时时间（秒）",
                default_value=30,
                type=int,
            ),
            RegisterConfig(
                key="CONFIDENCE_THRESHOLD",
                value=0.7,
                help="插件调用置信度阈值，低于此值时按普通聊天处理",
                default_value=0.7,
                type=float,
            ),
            RegisterConfig(
                key="CHAT_STYLE",
                value="",
                help="聊天回复风格，为空时使用默认风格",
                default_value="",
                type=str,
            ),
            RegisterConfig(
                key="USE_SIGN_IN_IMPRESSION",
                value=True,
                help="是否使用签到好感度，禁用时好感度提示词将被禁用",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                key="CONTEXT_PREFIX_SIZE",
                value=5,
                help="语境前缀消息数（从数据库加载的最近 N 条群消息作为背景语境）",
                default_value=5,
                type=int,
            ),
            RegisterConfig(
                key="SESSION_CONTEXT_LIMIT",
                value=20,
                help="单会话上下文上限（对话历史的最大消息数，超出时舍弃最早的）",
                default_value=20,
                type=int,
            ),
            RegisterConfig(
                key="MAX_REPLY_LAYERS",
                value=3,
                help="回复链追溯最大层数（递归获取被回复消息，构建多层对话链条）",
                default_value=3,
                type=int,
            ),
        ],
    ).to_dict(),
)


_fallback_matcher = on_message(
    priority=999,
    block=True,
    rule=to_me(),
)


@_fallback_matcher.handle()
async def _handle_fallback(
    bot: Bot,
    event: Event,
    session: Uninfo,
    msg: UniMsg,
):
    """消息处理器

    当消息未被其他插件处理时，使用 AI 分析用户意图并响应
    """
    try:
        raw_message = str(msg)
    except Exception as e:
        logger.error(f"获取消息内容失败：{e}")
        return

    if not raw_message:
        logger.debug("消息为空，跳过处理")
        return

    logger.info(f"[ChatInter] 收到消息：{raw_message[:50]}...")
    await handle_fallback(bot, event, session, raw_message, msg)


_reset_matcher = on_alconna(
    Alconna("重置会话"),
    permission=SUPERUSER,
    block=True,
    priority=1,
    rule=to_me(),
)


@_reset_matcher.handle()
async def _handle_reset_by_alconna(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent, session: Uninfo):
    """重置当前会话历史（仅超级用户）"""
    user_id = session.user.id if session.user else ""
    group_id = session.group.id if session.group else None

    reset_count = await _chat_memory.reset_session_history(user_id, group_id)

    chat_type = "群聊" if group_id else "私聊"
    logger.info(f"超级用户 {user_id} 重置了{chat_type}会话，共 {reset_count} 条对话被标记为重置")
    await MessageUtils.build_message(f"✅ 会话已重置，共 {reset_count} 条对话记录已被归档").send()


@driver.on_startup
async def _on_startup():
    """插件启动初始化"""
    from zhenxun.configs.config import BotConfig

    logger.info("ChatInter 插件已加载")
    _chat_memory.set_bot_nickname(BotConfig.self_nickname)
    await PluginRegistry.preload_cache()
