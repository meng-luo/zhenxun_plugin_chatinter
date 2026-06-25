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
from nonebot_plugin_alconna import Alconna, Args, Match, on_alconna
from nonebot_plugin_alconna.uniseg import UniMsg
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.utils import Command, PluginExtraData, RegisterConfig
from zhenxun.models.chat_history import ChatHistory as _ChatHistory  # noqa: F401
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType
from zhenxun.utils.message import MessageUtils

from .event_runtime import mark_as_handled
from .event_signals import get_event_signal
from .execution_observer import render_execution_observer_summary
from .handler import handle_fallback
from .lifecycle import ensure_lifecycle_hooks_registered
from .memory import _chat_memory
from .models import chat_history as _chatinter_models  # noqa: F401
from .native_tail_collector import (
    resolve_native_tail_route_modules,
    schedule_native_tail_followup,
)
from .plugin_registry import PluginRegistry
from .reflection_observer import render_reflection_observer_summary
from .scenario_router import resolve_chatinter_scenario
from .superuser_agent.permission_policy import (
    clear_session_permission_mode,
    get_session_permission_mode,
    set_session_permission_mode,
)
from .superuser_agent.runtime_approval import (
    has_runtime_approval_intent,
    try_handle_runtime_approval,
)
from .superuser_agent.runtime_control import (
    has_runtime_control_intent,
    try_handle_runtime_control,
)
from .superuser_agent.tool_preset import (
    get_session_tool_preset,
    set_session_tool_preset,
    tool_preset_label,
)
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
                key="NATIVE_REROUTE_TIMEOUT",
                value=10,
                help="等待插件重路由执行并观测发送输出的超时时间（秒）",
                default_value=10,
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
                key="PERSONA_FILE",
                value="",
                help=(
                    "ChatInter Persona JSON 文件路径，留空使用 "
                    "data/chatinter_agent/personas.json。"
                    "未配置文件时自动兼容 CHAT_STYLE/CUSTOM_PROMPT。"
                ),
                default_value="",
                type=str,
            ),
            RegisterConfig(
                module="chatinter",
                key="QUALITY_SHADOW_SAMPLE_RATE",
                value=0.0,
                help=(
                    "ChatInter 回复质量轻模型事后抽检比例，0 表示关闭。"
                    "抽检异步执行，不阻塞主回复。"
                ),
                default_value=0.0,
                type=float,
            ),
            RegisterConfig(
                module="chatinter",
                key="REASONING_EFFORT",
                value="MEDIUM",
                help=("强制推理强度，可选 MEDIUM 或 HIGH。留空表示不强制设置。"),
                default_value="MEDIUM",
                type=str,
            ),
            RegisterConfig(
                module="chatinter",
                key="FALLBACK_MODELS",
                value="",
                help=(
                    "主模型请求失败时的降级模型链，逗号分隔，按顺序尝试。"
                    "留空表示不降级。"
                ),
                default_value="",
                type=str,
            ),
            RegisterConfig(
                module="chatinter",
                key="SUPERUSER_PERMISSION_MODE",
                value="default",
                help=(
                    "超级用户 Agent 权限模式：default=按权限策略，"
                    "ask_all=全部确认，auto_readonly=只读自动通过/其他确认，"
                    "bypass=跳过确认"
                ),
                default_value="default",
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
            Command(
                command="重建插件索引",
                description="重建 ChatInter 插件知识库索引（超级用户）",
            ),
            Command(
                command="Agent权限模式 [default|ask_all|auto_readonly|bypass|clear]",
                description="设置当前会话的超级用户 Agent 权限模式（内存生效）",
            ),
            Command(
                command=(
                    "Agent工具模式 "
                    "[default|read_only|code_edit|plugin_dev|server_ops|clear]"
                ),
                description="设置当前会话的超级用户 Agent 工具预设（内存生效）",
            ),
        ],
        superuser_help="""
- `重置会话`
- `chatinter统计`
- `重建插件索引`
- `Agent权限模式 [default|ask_all|auto_readonly|bypass|clear]`
- `Agent工具模式 [default|read_only|code_edit|plugin_dev|server_ops|clear]`
        """.strip(),
    ).to_dict(),
)


_fallback_matcher = on_message(
    priority=999,
    block=True,
    rule=to_me(),
)
setattr(_fallback_matcher, "_zx_dispatch_lane", "fallback_ai")

_turn_followup_matcher = on_message(
    priority=998,
    block=False,
)
setattr(_turn_followup_matcher, "_zx_dispatch_lane", "passive_light")

_native_tail_collector_matcher = on_message(
    priority=4,
    block=False,
)
setattr(_native_tail_collector_matcher, "_zx_dispatch_lane", "passive_light")


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
    if not _is_private_text_only_message(event, event_message):
        logger.debug("ChatInter 私聊仅文本策略：忽略非文本消息")
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
    if not has_runtime_control_intent(raw_message):
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


@_native_tail_collector_matcher.handle()
async def _handle_native_tail_collector(
    bot: Bot,
    event: Event,
    session: Uninfo,
    msg: UniMsg,
    state: T_State,
):
    """Collect native-command messages that contain independent follow-up tasks."""

    if get_event_signal(event, "_ai_triggered", False):
        return
    route_modules = _event_route_modules(state)
    state_plain_text = _state_plain_text(state)
    raw_message = _extract_raw_message(event, msg, state_plain_text)
    if raw_message is None:
        return
    if not route_modules:
        route_modules = resolve_native_tail_route_modules(raw_message)
    if not route_modules:
        return
    scheduled = schedule_native_tail_followup(
        bot=bot,
        event=event,
        session=session,
        raw_message=raw_message,
        message=msg,
        route_modules=route_modules,
        cached_plain_text=state_plain_text,
        processor=handle_fallback,
    )
    if scheduled:
        logger.debug(
            "[ChatInter] 原生命令后续任务旁路收集已挂起：" f"{raw_message[:80]}..."
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

_rebuild_plugin_index_matcher = on_alconna(
    Alconna("重建插件索引"),
    permission=SUPERUSER,
    block=True,
    priority=1,
    rule=to_me(),
)

_permission_mode_matcher = on_alconna(
    Alconna(
        "Agent权限模式",
        Args["mode?", ["default", "ask_all", "auto_readonly", "bypass", "clear"]],
    ),
    permission=SUPERUSER,
    block=True,
    priority=1,
    rule=to_me(),
)

_tool_preset_matcher = on_alconna(
    Alconna(
        "Agent工具模式",
        Args[
            "preset?",
            [
                "default",
                "read_only",
                "code_edit",
                "plugin_dev",
                "server_ops",
                "clear",
                "只读模式",
                "改代码模式",
                "插件开发模式",
                "服务器排查模式",
            ],
        ],
    ),
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


@_permission_mode_matcher.handle()
async def _handle_permission_mode(session: Uninfo, mode: Match[str]):
    user_id = str(session.user.id if session.user else "")
    session_key = str(session.group.id) if session.group else user_id
    if not session_key:
        await MessageUtils.build_message("无法识别当前会话。").send()
        return
    current = get_session_permission_mode(session_key)
    if not mode.available:
        await MessageUtils.build_message(
            "当前会话权限模式："
            f"{current or '未设置，使用全局 SUPERUSER_PERMISSION_MODE'}"
        ).send()
        return
    value = str(mode.result or "").strip().lower()
    if value == "clear":
        clear_session_permission_mode(session_key)
        await MessageUtils.build_message(
            "已清除当前会话权限模式，恢复全局配置。"
        ).send()
        return
    applied = set_session_permission_mode(session_key, value)
    await MessageUtils.build_message(f"当前会话权限模式已设为：{applied}").send()


@_tool_preset_matcher.handle()
async def _handle_tool_preset(session: Uninfo, preset: Match[str]):
    user_id = str(session.user.id if session.user else "")
    session_key = str(session.group.id) if session.group else user_id
    if not session_key:
        await MessageUtils.build_message("无法识别当前会话。").send()
        return
    current = get_session_tool_preset(session_key)
    if not preset.available:
        await MessageUtils.build_message(
            "当前会话工具模式：" f"{tool_preset_label(current)}"
        ).send()
        return
    value = str(preset.result or "").strip()
    if value == "clear":
        value = "default"
    applied = set_session_tool_preset(session_key, value)
    await MessageUtils.build_message(
        "当前会话工具模式已设为："
        f"{tool_preset_label(applied)}"
        + ("；权限模式已联动调整。" if applied != "default" else "。")
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
