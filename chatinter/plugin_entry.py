"""
ChatInter - AI 意图识别插件

当用户消息未被其他插件匹配时，使用 AI 分析用户意图：
- 功能调用意图 -> 重路由到对应插件
- 普通聊天意图 -> 进行正常对话回复

使用 UniMessage 统一处理消息，支持多模态输入。
"""

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from nonebot import on_message
from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import (
    Bot as OneBotV11Bot,
)
from nonebot.adapters.onebot.v11 import (
    GroupMessageEvent,
    Message,
    PrivateMessageEvent,
)
from nonebot.matcher import Matcher
from nonebot.message import run_postprocessor
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
from zhenxun.utils.manager.priority_manager import PriorityLifecycle
from zhenxun.utils.message import MessageUtils

from .config import CHATINTER_REGISTER_CONFIGS, chatinter_available
from .event_runtime import (
    event_is_private,
    is_already_handled,
    mark_as_handled,
    resolve_superuser,
)
from .event_signals import get_event_signal, set_event_signal
from .execution_observer import render_execution_observer_summary
from .handler import handle_fallback
from .history_policy import (
    history_foreground_arrived,
    schedule_pending_history_summary_jobs,
    shutdown_history_summary_tasks,
)
from .memory import _chat_memory
from .mode_gate import MixedTurnAdmission, get_mode_gate
from .models import chat_history as _chatinter_models  # noqa: F401
from .plugin_registry import PluginRegistry
from .reflection_observer import render_reflection_observer_summary
from .scenario_router import ChatInterScenario, resolve_chatinter_scenario
from .session_identity import conversation_session_key, legacy_session_key
from .turn_metrics import render_route_observer_summary
from .turn_queue import get_turn_queue
from .utils.unimsg_utils import uni_to_text_with_tags

_DYNAMIC_MATCHER_RESCAN_DELAY_SECONDS = 10.0
_dynamic_rescan_task: asyncio.Task | None = None


def _event_observing_sender(
    sender: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    @wraps(sender)
    async def observed(bot: Bot, event: Event, message: Any, **kwargs: Any) -> Any:
        result = await sender(bot, event, message, **kwargs)
        set_event_signal(event, "_zx_visible_output_sent", True)
        return result

    setattr(observed, "_chatinter_event_send_observer", True)
    return observed


def _patch_onebot_event_send_observer() -> None:
    current = OneBotV11Bot.send
    if getattr(current, "_chatinter_event_send_observer", False):
        return
    OneBotV11Bot.send = _event_observing_sender(current)  # type: ignore[method-assign]


_patch_onebot_event_send_observer()


@run_postprocessor
async def _observe_rerouted_plugin_execution(
    matcher: Matcher,
    event: Event,
    exception: Exception | None = None,
) -> None:
    target_modules = get_event_signal(event, "_ai_route_modules", frozenset())
    if not isinstance(target_modules, set | frozenset) or not target_modules:
        return
    plugin = matcher.plugin
    identifiers = {
        str(getattr(plugin, key, "") or "").strip()
        for key in ("name", "module_name")
        if str(getattr(plugin, key, "") or "").strip()
    }
    if identifiers.intersection(target_modules):
        set_event_signal(event, "_ai_plugin_execution_started", True)
        if exception is not None:
            set_event_signal(event, "_ai_plugin_execution_failed", True)


__plugin_meta__ = PluginMetadata(
    name="ChatInter",
    description="当消息未被其他插件处理时，使用 AI 分析用户意图并智能响应",
    usage="""
    ChatInter 功能，自动识别用户意图
    """.strip(),
    extra=PluginExtraData(
        author="Copaan & meng-luo",
        version="1.5.0",
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
_reaction_observer = on_message(priority=1, block=False)


@_reaction_observer.handle()
async def _observe_group_reaction_images(
    bot: Bot,
    event: Event,
    session: Uninfo,
    msg: UniMsg,
) -> None:
    if session.group is None or session.user is None:
        return
    group_id = str(session.group.id)
    sender_id = str(session.user.id)
    if not group_id or not sender_id or sender_id == str(bot.self_id):
        return
    if not chatinter_available(group_id):
        return
    from .reaction_runtime import reaction_settings, schedule_reaction_observation

    settings = reaction_settings()
    if not settings.enabled or not settings.auto_discovery:
        return
    event_id = str(getattr(event, "message_id", "") or "")
    if not event_id:
        try:
            event_id = str(event.get_event_id())
        except Exception:
            event_id = ""
    schedule_reaction_observation(
        group_id=group_id,
        sender_id=sender_id,
        message_id=event_id,
        message=msg,
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


async def _process_queued_fallback(
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
    message=None,
    route_modules: set[str] | None = None,
    cached_plain_text: str | None = None,
    queued: bool = False,
) -> None:
    try:
        await handle_fallback(
            bot,
            event,
            session,
            raw_message,
            message,
            route_modules=route_modules,
            cached_plain_text=cached_plain_text,
            queued=queued,
        )
    finally:
        schedule_pending_history_summary_jobs()


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
    group_id = str(session.group.id) if session.group else None
    if not chatinter_available(group_id):
        logger.debug("ChatInter 当前会话未启用")
        return
    if get_event_signal(event, "_ai_triggered", False):
        return
    if get_event_signal(event, "_zx_visible_output_sent", False):
        logger.debug("earlier matcher produced visible output, skip ChatInter fallback")
        return

    state_plain_text = _state_plain_text(state)
    raw_message = _extract_raw_message(event, msg, state_plain_text)
    if raw_message is None:
        return

    handled_session_key = conversation_session_key(session)
    if is_already_handled(event, session_key=handled_session_key):
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

        mark_as_handled(event, session_key=handled_session_key)
        await handle_superuser_agent_turn(
            bot=bot,
            event=event,
            raw_message=raw_message,
            session_key=str(session.group.id)
            if session.group
            else str(session.user.id),
        )
        return
    mode_admission = await _acquire_private_superuser_mixed_turn(
        bot=bot,
        event=event,
        session=session,
    )
    if mode_admission is not None and not mode_admission.accepted:
        if mode_admission.blocked_by == "agent_active":
            from .agents.superuser_entry import handle_superuser_agent_turn

            mark_as_handled(event, session_key=handled_session_key)
            await handle_superuser_agent_turn(
                bot=bot,
                event=event,
                raw_message=raw_message,
                session_key=str(session.user.id),
            )
            return
        mark_as_handled(event, session_key=handled_session_key)
        await _send_mode_gate_reply(
            bot=bot,
            event=event,
            text="Agent 模式正在切换，请稍后重试。",
        )
        return
    mode_lease = mode_admission.lease if mode_admission is not None else None
    summary_session_id = conversation_session_key(session)
    history_foreground_arrived(summary_session_id)
    accepted = await get_turn_queue().submit(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
        message=msg,
        route_modules=route_modules,
        cached_plain_text=state_plain_text,
        processor=_process_queued_fallback,
        mode_lease=mode_lease,
    )
    if not accepted:
        schedule_pending_history_summary_jobs()
    if accepted:
        logger.info(f"[ChatInter] 收到消息：{raw_message[:50]}...")


async def _try_runtime_control_before_queue(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
) -> bool:
    user_id = str(session.user.id if session.user else "")
    handled_session_key = conversation_session_key(session)
    if not event_is_private(event) or not resolve_superuser(bot, user_id):
        return False
    from .superuser_agent.runtime_control import (
        has_runtime_control_intent,
        parse_runtime_control_command,
        try_handle_runtime_control,
    )

    session_key = str(session.group.id) if session.group else user_id
    if not has_runtime_control_intent(raw_message, session_key=session_key):
        return False
    intent, _ = parse_runtime_control_command(raw_message)
    gate = get_mode_gate()

    def active_source() -> bool:
        return _stored_agent_mode_active(session_key)

    transition = None
    if intent == "open":
        transition, blocked_by = await gate.try_begin_agent_transition(
            session_key,
            agent_active=active_source,
        )
        if transition is None:
            text = (
                "当前仍有消息正在处理或排队，请等待完成后再回复 /开启agent。"
                if blocked_by == "mixed_busy"
                else "Agent 模式正在切换，请稍后重试。"
            )
            await _send_mode_gate_reply(bot=bot, event=event, text=text)
            mark_as_handled(event, session_key=handled_session_key)
            return True
    else:
        await gate.sync_agent_active(session_key, active=active_source)
    try:
        handled = await try_handle_runtime_control(
            bot=bot,
            event=event,
            session=session,
            raw_message=raw_message,
        )
    finally:
        if transition is not None:
            await asyncio.shield(transition.finish(agent_active=active_source))
        else:
            await asyncio.shield(
                gate.sync_agent_active(session_key, active=active_source)
            )
    if not handled:
        return False
    mark_as_handled(event, session_key=handled_session_key)
    return True


async def _acquire_private_superuser_mixed_turn(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
) -> MixedTurnAdmission | None:
    user_id = str(session.user.id if session.user else "")
    if (
        not user_id
        or session.group is not None
        or not event_is_private(event)
        or not resolve_superuser(bot, user_id)
    ):
        return None
    return await get_mode_gate().try_acquire_mixed_turn(
        user_id,
        agent_active=lambda: _stored_agent_mode_active(user_id),
    )


def _stored_agent_mode_active(session_key: str) -> bool:
    from .superuser_agent.store import agent_session_is_active

    return bool(agent_session_is_active(session_key))


async def _send_mode_gate_reply(*, bot: Bot, event: Event, text: str) -> None:
    try:
        await bot.send(event, text)
        return
    except Exception:
        pass
    try:
        await MessageUtils.build_message(text).send()
    except Exception:
        pass


async def _try_runtime_approval_before_queue(
    *,
    bot: Bot,
    event: Event,
    session: Uninfo,
    raw_message: str,
) -> bool:
    user_id = str(session.user.id if session.user else "")
    handled_session_key = conversation_session_key(session)
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
    mark_as_handled(event, session_key=handled_session_key)
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


@PriorityLifecycle.on_startup(priority=60)
async def _on_startup():
    """插件启动初始化"""
    global _dynamic_rescan_task

    if not chatinter_available():
        logger.info("ChatInter 插件已关闭")
        return

    from zhenxun.configs.config import BotConfig

    from .persona import ensure_persona_file

    logger.info("ChatInter 插件已加载")
    ensure_persona_file()
    _chat_memory.set_bot_nickname(BotConfig.self_nickname)
    await PluginRegistry.preload_cache()
    from .reaction_runtime import start_reaction_runtime

    await start_reaction_runtime()
    _dynamic_rescan_task = asyncio.create_task(_rescan_dynamic_matchers_after_startup())


@PriorityLifecycle.on_startup(priority=100)
async def _on_active_tasks_startup():
    from .config import active_tasks_enabled

    if not active_tasks_enabled():
        logger.info("ChatInter 主动任务已关闭")
        return
    registered = 0
    failed = 0
    try:
        from .superuser_agent.active_tasks import initialize_active_task_schedules

        registered, failed = await initialize_active_task_schedules(
            _dispatch_scheduled_active_task
        )
    except Exception as exc:
        failed += 1
        logger.error("ChatInter 主动任务调度初始化失败", e=exc)
    try:
        from .superuser_agent.proactive_tasks import (
            install_active_task_webhook_route,
        )

        webhook_installed = install_active_task_webhook_route()
    except Exception as exc:
        webhook_installed = False
        logger.error("ChatInter 主动任务 Webhook 初始化失败", e=exc)
    logger.info(
        "ChatInter 主动任务已初始化："
        f"调度 {registered}，失败 {failed}，Webhook {webhook_installed}"
    )


@PriorityLifecycle.on_shutdown(priority=40)
async def _on_shutdown():
    """Release long-lived ChatInter runtime resources."""

    global _dynamic_rescan_task

    if _dynamic_rescan_task is not None and not _dynamic_rescan_task.done():
        _dynamic_rescan_task.cancel()
        await asyncio.gather(_dynamic_rescan_task, return_exceptions=True)
    _dynamic_rescan_task = None
    from .reaction_runtime import shutdown_reaction_runtime

    await shutdown_reaction_runtime()
    await PluginRegistry.shutdown()
    from .superuser_agent.proactive_tasks import shutdown_proactive_tasks

    await shutdown_proactive_tasks()
    await shutdown_history_summary_tasks()

    from .mcp_runtime import get_mcp_runtime_manager
    from .memory_extractor import drain_memory_extraction_tasks

    await drain_memory_extraction_tasks()
    await get_mcp_runtime_manager().shutdown()

    from .gscore_adapter import get_gscore_adapter

    await get_gscore_adapter().close()


async def _dispatch_scheduled_active_task(task, _bot, _context) -> None:
    from .superuser_agent.proactive_tasks import get_proactive_dispatcher

    await get_proactive_dispatcher().dispatch(
        task.task_id,
        {"event": "scheduled_trigger"},
        source="scheduler",
        claimed_task=task,
    )


async def _rescan_dynamic_matchers_after_startup():
    """等其它插件 startup 动态 matcher 创建完成后，重建一次知识库。

    这次补扫生成运行期固定快照；之后只在活动插件集合变化或显式重建时更新。
    """
    await asyncio.sleep(_DYNAMIC_MATCHER_RESCAN_DELAY_SECONDS)
    await PluginRegistry.preload_cache(force_refresh=True)
    logger.info("ChatInter 已完成启动后 10 秒插件知识快照")
