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

from zhenxun.configs.utils import Command, PluginExtraData
from zhenxun.models.chat_history import ChatHistory as _ChatHistory  # noqa: F401
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType
from zhenxun.utils.message import MessageUtils

from .config import CHATINTER_REGISTER_CONFIGS
from .event_runtime import (
    event_is_private,
    is_already_handled,
    mark_as_handled,
    resolve_superuser,
)
from .event_signals import get_event_signal
from .execution_observer import render_execution_observer_summary
from .handler import handle_fallback
from .lifecycle import ensure_lifecycle_hooks_registered
from .memory import _chat_memory
from .models import chat_history as _chatinter_models  # noqa: F401
from .plugin_registry import PluginRegistry
from .reflection_observer import render_reflection_observer_summary
from .scenario_router import ChatInterScenario, resolve_chatinter_scenario
from .session_identity import conversation_session_key, legacy_session_key
from .turn_metrics import render_route_observer_summary
from .turn_queue import get_turn_queue
from .utils.unimsg_utils import uni_to_text_with_tags

driver = get_driver()
_DYNAMIC_MATCHER_RESCAN_DELAYS = (8,)
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
        configs=list(CHATINTER_REGISTER_CONFIGS),
        commands=[
            Command(
                command="重置会话",
                description="重置当前会话历史（超级用户）",
            ),
            Command(
                command="chatinter统计",
                description="查看最近 ChatInter 路由统计（超级用户）",
            ),
            Command(
                command="重建插件索引",
                description="重建 ChatInter 插件知识库索引（超级用户）",
            ),
            Command(
                command="/开启agent",
                description="开启 Superuser Agent（超级用户私聊）",
            ),
            Command(
                command="/退出agent",
                description="退出 Superuser Agent，保留当前会话（超级用户私聊）",
            ),
            Command(
                command="/agent帮助",
                description="查看 Superuser Agent 命令（超级用户私聊）",
            ),
            Command(
                command="/状态",
                description="查看当前 Agent 状态（超级用户私聊）",
            ),
            Command(
                command="/中断",
                description="中断当前任务或待审批操作（超级用户私聊）",
            ),
            Command(
                command="/清除上下文",
                description="清除当前会话上下文（超级用户私聊）",
            ),
            Command(
                command="/压缩上下文",
                description="压缩当前会话上下文（超级用户私聊）",
            ),
            Command(
                command="/请求批准模式",
                description="切换为请求批准权限模式（超级用户私聊）",
            ),
            Command(
                command="/只读模式",
                description="切换为只读权限模式（超级用户私聊）",
            ),
            Command(
                command="/完全访问模式",
                description="切换为完全访问权限模式（超级用户私聊）",
            ),
            Command(
                command="/新增会话",
                params=["[名称]"],
                description="新增并切换 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/当前会话",
                description="查看当前 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/列出会话",
                description="列出可用 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/切换会话",
                params=["[ID/名称]"],
                description="选择或切换 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/重命名会话",
                params=["ID/名称", "新名称"],
                description="重命名 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/归档会话",
                params=["[ID/名称]"],
                description="归档 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/列出归档会话",
                description="列出已归档 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/恢复会话",
                params=["ID/名称"],
                description="恢复已归档 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/删除会话",
                params=["ID/名称"],
                description="删除 Agent 会话（超级用户私聊）",
            ),
            Command(
                command="/允许",
                description="允许执行一次待审批操作（超级用户私聊）",
            ),
            Command(
                command="/本对话允许",
                description="允许当前对话后续相同权限范围（超级用户私聊）",
            ),
            Command(
                command="/拒绝",
                params=["[理由]"],
                description="拒绝待审批操作（超级用户私聊）",
            ),
        ],
        superuser_help="""
- `重置会话`
- `chatinter统计`
- `重建插件索引`
- Agent：`/开启agent` `/退出agent` `/agent帮助` `/状态` `/中断`
- 上下文：`/清除上下文` `/压缩上下文`
- 权限：`/请求批准模式` `/只读模式` `/完全访问模式`
- 会话：`/新增会话 [名称]` `/当前会话` `/列出会话`
  `/切换会话 [ID/名称]` `/重命名会话 ID/名称 新名称`
  `/归档会话 [ID/名称]` `/列出归档会话`
  `/恢复会话 ID/名称` `/删除会话 ID/名称`
- 审批：`/允许` `/本对话允许` `/拒绝 [理由]` `/中断`
        """.strip(),
    ).to_dict(),
)


_fallback_matcher = on_message(
    priority=999,
    block=True,
    rule=to_me(),
)

_turn_followup_matcher = on_message(
    priority=998,
    block=False,
)


def _is_supported_private_message(
    event: Event,
    event_message: object,
) -> bool:
    if not isinstance(event, PrivateMessageEvent):
        return True
    if not isinstance(event_message, Message):
        return False
    has_content = False
    for seg in event_message:
        seg_type = str(getattr(seg, "type", "") or "")
        if seg_type == "text":
            text = str(getattr(seg, "data", {}).get("text", "")).strip()
            if text:
                has_content = True
            continue
        if seg_type in {"reply", "image"}:
            has_content = True
            continue
        return False
    return has_content


def _state_plain_text(state: T_State) -> str | None:
    state_plain_text = state.get("_zx_plain_text")
    return state_plain_text if isinstance(state_plain_text, str) else None


def _event_route_modules(state: T_State) -> set[str] | None:
    route_modules = state.get("_zx_route_modules")
    return route_modules if isinstance(route_modules, set) else None


def _resolve_entry_scenario(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    route_modules: set[str] | None,
):
    user_id = str(session.user.id)
    group_id = str(session.group.id) if session.group else None
    return resolve_chatinter_scenario(
        bot=bot,
        event=event,
        raw_message=raw_message,
        user_id=user_id,
        group_id=group_id,
        route_modules=route_modules,
    )


def _extract_raw_message(
    event: Event,
    msg: UniMsg,
    state_plain_text: str | None,
) -> str | None:
    try:
        event_message = event.get_message()
    except Exception:
        event_message = None
    if not _is_supported_private_message(event, event_message):
        logger.debug("ChatInter 私聊媒体策略：忽略不支持的消息段")
        return None

    try:
        tagged_message = (
            uni_to_text_with_tags(event_message)
            if event_message is not None
            else uni_to_text_with_tags(msg)
        )
        raw_message = tagged_message or state_plain_text or str(msg)
    except Exception as e:
        logger.error(f"获取消息内容失败：{e}")
        return None

    if not isinstance(raw_message, str):
        raw_message = str(raw_message)
    raw_message = raw_message.strip()
    if not raw_message:
        logger.debug("消息为空，跳过处理")
        return None
    return raw_message


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
    if get_event_signal(event, "_ai_triggered", False):
        return

    state_plain_text = _state_plain_text(state)
    raw_message = _extract_raw_message(event, msg, state_plain_text)
    if raw_message is None:
        return

    if is_already_handled(event):
        logger.debug("event already handled, skip ChatInter fallback")
        return

    if await _try_runtime_approval_before_queue(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
    ):
        return

    if await _try_runtime_control_before_queue(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
    ):
        return

    route_modules = _event_route_modules(state)
    if route_modules:
        logger.debug("event already has route modules, skip ChatInter fallback")
        return

    scenario = _resolve_entry_scenario(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
        route_modules=route_modules,
    )
    if not scenario.should_handle:
        logger.debug(f"ChatInter scenario skip: {scenario.reason}")
        return
    if scenario.scenario is ChatInterScenario.SUPERUSER_AGENT:
        from .agents.superuser_entry import handle_superuser_agent_turn

        mark_as_handled(event)
        await handle_superuser_agent_turn(
            raw_message=raw_message,
            session_key=str(session.group.id)
            if session.group
            else str(session.user.id),
        )
        return
    accepted = await get_turn_queue().submit(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
        message=msg,
        route_modules=route_modules,
        cached_plain_text=state_plain_text,
        processor=handle_fallback,
    )
    if accepted:
        logger.info(f"[ChatInter] 收到消息：{raw_message[:50]}...")


@_turn_followup_matcher.handle()
async def _handle_turn_followup(
    bot: Bot,
    event: Event,
    session: Uninfo,
    msg: UniMsg,
    state: T_State,
):
    """Non-blocking collector for short follow-up messages in an active turn."""

    if get_event_signal(event, "_ai_triggered", False) or bool(
        getattr(event, "to_me", False)
    ):
        return
    route_modules = _event_route_modules(state)
    if route_modules:
        return
    state_plain_text = _state_plain_text(state)
    raw_message = _extract_raw_message(event, msg, state_plain_text)
    if raw_message is None:
        return
    scenario = _resolve_entry_scenario(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
        route_modules=route_modules,
    )
    if not scenario.should_handle:
        return
    accepted = await get_turn_queue().submit(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
        message=msg,
        route_modules=None,
        cached_plain_text=state_plain_text,
        processor=handle_fallback,
        priority_override=0,
    )
    if accepted:
        logger.debug(f"[ChatInter] 收到连续 turn 补充：{raw_message[:50]}...")


async def _try_runtime_control_before_queue(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
) -> bool:
    user_id = str(session.user.id if session.user else "")
    if not event_is_private(event) or not resolve_superuser(bot, user_id):
        return False
    from .superuser_agent.runtime_control import (
        has_runtime_control_intent,
        try_handle_runtime_control,
    )

    session_key = str(session.group.id) if session.group else user_id
    if not has_runtime_control_intent(raw_message, session_key=session_key):
        return False
    if not await try_handle_runtime_control(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
    ):
        return False
    mark_as_handled(event)
    return True


async def _try_runtime_approval_before_queue(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
) -> bool:
    user_id = str(session.user.id if session.user else "")
    if not event_is_private(event) or not resolve_superuser(bot, user_id):
        return False
    from .superuser_agent.runtime_approval import (
        has_runtime_approval_intent,
        try_handle_runtime_approval,
    )

    if not has_runtime_approval_intent(raw_message):
        return False
    if not await try_handle_runtime_approval(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
    ):
        return False
    mark_as_handled(event)
    return True


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

_rebuild_plugin_index_matcher = on_alconna(
    Alconna("重建插件索引"),
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

    reset_count = await _chat_memory.reset_session_history(
        user_id,
        group_id,
        session_id=conversation_session_key(session),
        legacy_session_id=legacy_session_key(session),
    )

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
        render_route_observer_summary()
        + "\n\n"
        + render_execution_observer_summary()
        + "\n\n"
        + render_reflection_observer_summary()
    ).send()


@_rebuild_plugin_index_matcher.handle()
async def _handle_rebuild_plugin_index():
    try:
        knowledge_base = await PluginRegistry.get_plugin_knowledge_base(
            force_refresh=True
        )
    except Exception as exc:
        logger.error(f"ChatInter 插件索引重建失败：{exc}")
        await MessageUtils.build_message(f"插件索引重建失败：{exc}").send()
        return
    await MessageUtils.build_message(
        f"插件索引已重建，共 {len(knowledge_base.plugins)} 个插件。"
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


@driver.on_shutdown
async def _on_shutdown():
    """Release long-lived ChatInter runtime resources."""

    from .mcp_runtime import get_mcp_runtime_manager
    from .memory_extractor import drain_memory_extraction_tasks

    await drain_memory_extraction_tasks()
    await get_mcp_runtime_manager().shutdown()


async def _rescan_dynamic_matchers_after_startup():
    """等其它插件 startup 动态 matcher 创建完成后，重建一次知识库。

    大部分插件（包括 nonebot_plugin_memes）在导入期已注册完 matcher；保留
    一次延迟补扫主要覆盖 parser-lite 这类 startup 阶段动态注册 matcher 的插件。
    插件开启/关闭状态变化由 PluginInfoMemoryCache refresh 版本驱动缓存失效。
    """
    for delay_seconds in _DYNAMIC_MATCHER_RESCAN_DELAYS:
        await asyncio.sleep(delay_seconds)
        await PluginRegistry.preload_cache(force_refresh=True)
        logger.info(
            "ChatInter 已完成 startup 后动态 matcher 补扫：" f"delay={delay_seconds}s"
        )
