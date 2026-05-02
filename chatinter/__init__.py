"""
ChatInter - AI 意图识别插件

当用户消息未被其他插件匹配时，使用 AI 分析用户意图：
- 功能调用意图 -> 重路由到对应插件
- 普通聊天意图 -> 进行正常对话回复

使用 UniMessage 统一处理消息，支持多模态输入。
"""

import asyncio

from nonebot import get_driver, on_message
from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message, PrivateMessageEvent
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata
from nonebot.rule import to_me
from nonebot.typing import T_State
from nonebot_plugin_alconna import Alconna, on_alconna
from nonebot_plugin_alconna.uniseg import UniMsg
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.utils import Command, PluginExtraData, RegisterConfig
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType
from zhenxun.utils.message import MessageUtils

from . import models as _models  # noqa: F401
from .execution_observer import render_execution_observer_summary
from .handler import handle_fallback
from .lifecycle import ensure_lifecycle_hooks_registered
from .memory import _chat_memory
from .plugin_registry import PluginRegistry
from .turn_metrics import render_route_observer_summary
from .utils.unimsg_utils import uni_to_text_with_tags

driver = get_driver()
_DYNAMIC_MATCHER_RESCAN_DELAYS = (2, 8, 20)
_dynamic_rescan_task: asyncio.Task | None = None


__plugin_meta__ = PluginMetadata(
    name="ChatInter",
    description="当消息未被其他插件处理时，使用 AI 分析用户意图并智能响应",
    usage="""
    ChatInter 功能，自动识别用户意图
    """.strip(),
    extra=PluginExtraData(
        author="Copaan & meng-luo",
        version="1.3.0",
        plugin_type=PluginType.DEPENDANT,
        menu_type="其他",
        ignore_prompt=True,
        ignore_statistics=True,
        configs=[
            RegisterConfig(
                module="chatinter",
                key="ENABLE_FALLBACK",
                value=True,
                help="是否启用 ChatInter 兜底对话能力",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                module="chatinter",
                key="ENABLE_AGENT_MODE",
                value=True,
                help="是否启用 ChatInter Agent（工具调用）模式",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                module="chatinter",
                key="INTENT_TIMEOUT",
                value=20,
                help=(
                    "ChatInter 推理超时时间（秒），"
                    "<=0 时复用 AI.CLIENT_SETTINGS.timeout"
                ),
                default_value=20,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="AGENT_MAX_TOOL_STEPS",
                value=4,
                help="Agent 工具调用最大迭代步数（复杂请求会在此基础上自动小幅上调）",
                default_value=4,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="AGENT_TOOL_FAILURE_LIMIT",
                value=2,
                help="单工具连续失败达到阈值后自动熔断禁用",
                default_value=2,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="AGENT_FAILED_ROUND_LIMIT",
                value=2,
                help="连续失败回合阈值，达到后停止工具并直接总结",
                default_value=2,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="CHAT_STYLE",
                value="",
                help="ChatInter 对话风格补充设定，留空使用默认风格",
                default_value="",
                type=str,
            ),
            RegisterConfig(
                module="chatinter",
                key="CUSTOM_PROMPT",
                value="",
                help="ChatInter 自定义系统提示词补充，会追加到系统提示词末尾",
                default_value="",
                type=str,
            ),
            RegisterConfig(
                module="chatinter",
                key="MCP_ENDPOINTS",
                value="",
                help=(
                    "MCP 工具服务地址列表，使用英文逗号分隔。"
                    "示例: http://127.0.0.1:9001,http://127.0.0.1:9002"
                ),
                default_value="",
                type=str,
            ),
            RegisterConfig(
                module="chatinter",
                key="REASONING_EFFORT",
                value="MEDIUM",
                help=("强制推理强度，可选 MEDIUM 或 HIGH。留空表示不强制设置。"),
                default_value="MEDIUM",
                type=str,
            ),
        ],
        commands=[
            Command(
                command="重置会话",
                description="重置当前会话历史（超级用户）",
            ),
            Command(
                command="chatinter统计",
                description="查看最近 ChatInter 路由统计（超级用户）",
            ),
        ],
        superuser_help="""
- `重置会话`
- `chatinter统计`
        """.strip(),
    ).to_dict(),
)


_fallback_matcher = on_message(
    priority=999,
    block=True,
    rule=to_me(),
)


def _is_private_text_only_message(
    event: Event,
    event_message: object,
) -> bool:
    if not isinstance(event, PrivateMessageEvent):
        return True
    if not isinstance(event_message, Message):
        return False
    has_text = False
    for seg in event_message:
        seg_type = str(getattr(seg, "type", "") or "")
        if seg_type == "text":
            text = str(getattr(seg, "data", {}).get("text", "")).strip()
            if text:
                has_text = True
            continue
        if seg_type == "reply":
            continue
        return False
    return has_text


@_fallback_matcher.handle()
async def _handle_fallback(
    bot: Bot,
    event: Event,
    session: Uninfo,
    msg: UniMsg,
    state: T_State,
):
    """消息处理器

    当消息未被其他插件处理时，使用 AI 分析用户意图并响应
    """
    if getattr(event, "_ai_triggered", False):
        return

    state_plain_text = state.get("_zx_plain_text")
    if not isinstance(state_plain_text, str):
        state_plain_text = None

    try:
        event_message = event.get_message()
    except Exception:
        event_message = None
    if not _is_private_text_only_message(event, event_message):
        logger.debug("ChatInter 私聊仅文本策略：忽略非文本消息")
        return

    try:
        tagged_message = (
            uni_to_text_with_tags(event_message)
            if event_message is not None
            else uni_to_text_with_tags(msg)
        )
        raw_message = tagged_message or state_plain_text or str(msg)
    except Exception as e:
        logger.error(f"获取消息内容失败：{e}")
        return

    if not isinstance(raw_message, str):
        raw_message = str(raw_message)

    raw_message = raw_message.strip()
    if not raw_message:
        logger.debug("消息为空，跳过处理")
        return

    logger.info(f"[ChatInter] 收到消息：{raw_message[:50]}...")
    route_modules = state.get("_zx_route_modules")
    if not isinstance(route_modules, set):
        route_modules = None
    await handle_fallback(
        bot,
        event,
        session,
        raw_message,
        msg,
        route_modules=route_modules,
        cached_plain_text=state_plain_text,
    )


_reset_matcher = on_alconna(
    Alconna("重置会话"),
    permission=SUPERUSER,
    block=True,
    priority=1,
    rule=to_me(),
)

_stats_matcher = on_alconna(
    Alconna("chatinter统计"),
    permission=SUPERUSER,
    block=True,
    priority=1,
    rule=to_me(),
)


@_reset_matcher.handle()
async def _handle_reset_by_alconna(
    _bot: Bot, _event: GroupMessageEvent | PrivateMessageEvent, session: Uninfo
):
    """重置当前会话历史（仅超级用户）"""
    user_id = session.user.id if session.user else ""
    group_id = session.group.id if session.group else None

    reset_count = await _chat_memory.reset_session_history(user_id, group_id)

    chat_type = "群聊" if group_id else "私聊"
    logger.info(
        f"超级用户 {user_id} 重置了{chat_type}会话，共 {reset_count} 条对话被标记为重置"
    )
    await MessageUtils.build_message(
        f"✅ 会话已重置，共 {reset_count} 条对话记录已被归档"
    ).send()


@_stats_matcher.handle()
async def _handle_stats_by_alconna():
    await MessageUtils.build_message(
        render_route_observer_summary() + "\n\n" + render_execution_observer_summary()
    ).send()


@driver.on_startup
async def _on_startup():
    """插件启动初始化"""
    global _dynamic_rescan_task

    from zhenxun.configs.config import BotConfig

    logger.info("ChatInter 插件已加载")
    await ensure_lifecycle_hooks_registered()
    _chat_memory.set_bot_nickname(BotConfig.self_nickname)
    await PluginRegistry.preload_cache()
    _dynamic_rescan_task = asyncio.create_task(_rescan_dynamic_matchers_after_startup())


async def _rescan_dynamic_matchers_after_startup():
    """等其它插件 startup 动态 matcher 创建完成后，分批重建知识库。"""
    for delay_seconds in _DYNAMIC_MATCHER_RESCAN_DELAYS:
        await asyncio.sleep(delay_seconds)
        await PluginRegistry.preload_cache(force_refresh=True)
        logger.info(
            "ChatInter 已完成 startup 后动态 matcher 补扫：" f"delay={delay_seconds}s"
        )
