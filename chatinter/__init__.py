"""
ChatInter - AI 意图识别插件

当用户消息未被其他插件匹配时，使用 AI 分析用户意图：
- 功能调用意图 -> 重路由到对应插件
- 普通聊天意图 -> 进行正常对话回复

使用 UniMessage 统一处理消息，支持多模态输入。
"""

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

from .data_source import _chat_memory, handle_fallback
from .handler import (
    build_pending_followup_message,
    build_pending_target_followup_message,
    claim_pending_image_followup,
    clear_pending_target_followup,
    get_pending_target_followup,
    has_pending_image_followup,
    remember_target_resolution,
    resolve_pending_target_followup_user_id,
)
from .lifecycle import ensure_lifecycle_hooks_registered
from .models.chat_history import ChatInterChatHistory  # noqa: F401
from .plugin_registry import PluginRegistry
from .utils.unimsg_utils import uni_to_text_with_tags

driver = get_driver()


__plugin_meta__ = PluginMetadata(
    name="ChatInter",
    description="当消息未被其他插件处理时，使用 AI 分析用户意图并智能响应",
    usage="""
    ChatInter 功能，自动识别用户意图
    """.strip(),
    extra=PluginExtraData(
        author="Copaan & meng-luo",
        version="1.2.1",
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
                key="INTENT_MODEL",
                value="",
                help=(
                    "ChatInter 使用的模型名称 (格式: ProviderName/ModelName)，"
                    "留空时复用 AI.DEFAULT_MODEL_NAME"
                ),
                default_value="",
                type=str,
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
                key="AGENT_TOTAL_TIMEOUT",
                value=0,
                help="Agent 全链路预算秒数（0=自动跟随 INTENT_TIMEOUT）",
                default_value=0,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="AGENT_EXPAND_TOOLS_STEP",
                value=2,
                help="Agent 第 N 轮后从精准工具池扩展到完整工具池",
                default_value=2,
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
                key="AGENT_STRICT_TOOL_SELECT",
                value=True,
                help="是否启用严格工具选择（无匹配时不回退全量工具）",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                module="chatinter",
                key="CONFIDENCE_THRESHOLD",
                value=0.72,
                help="插件意图置信度阈值，低于该值时降级为普通聊天",
                default_value=0.72,
                type=float,
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
            RegisterConfig(
                module="chatinter",
                key="HISTORY_RECALL_LIMIT",
                value=4,
                help="上下文中额外召回的历史相关对话片段数量",
                default_value=4,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="HISTORY_RECALL_MIN_SCORE",
                value=0.18,
                help="历史片段召回最低相关度阈值（0-1）",
                default_value=0.18,
                type=float,
            ),
            RegisterConfig(
                module="chatinter",
                key="HISTORY_RECALL_CANDIDATE_LIMIT",
                value=60,
                help="历史片段召回候选池大小（从最近历史中取前N条评估）",
                default_value=60,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="GROUP_BACKGROUND_FETCH_MULTIPLIER",
                value=3,
                help="群聊背景候选抓取倍数（最终输出数量=CONTEXT_PREFIX_SIZE）",
                default_value=3,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="GROUP_BACKGROUND_RELEVANT_LIMIT",
                value=3,
                help="群聊背景中按相关度补充的消息上限",
                default_value=3,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="GROUP_BACKGROUND_MIN_SCORE",
                value=0.16,
                help="群聊背景相关补充的最低相关度阈值（0-1）",
                default_value=0.16,
                type=float,
            ),
            RegisterConfig(
                module="chatinter",
                key="CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX",
                value=True,
                help="复杂问题自动放宽对话长度限制（代码/排错/方案类请求）",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                module="chatinter",
                key="ENABLE_CONTEXT_RELEVANCE_GATE",
                value=True,
                help="开启上下文关联门控：新话题与历史低关联时按单轮对话处理",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                module="chatinter",
                key="CONTEXT_RELEVANCE_THRESHOLD",
                value=0.11,
                help="上下文关联阈值（0-1），低于该值时隔离历史上下文",
                default_value=0.11,
                type=float,
            ),
            RegisterConfig(
                module="chatinter",
                key="CONTEXT_RELEVANCE_SAMPLE_LIMIT",
                value=18,
                help="上下文关联评估采样条数（会话+群背景）",
                default_value=18,
                type=int,
            ),
            RegisterConfig(
                module="chatinter",
                key="CONTEXT_RELEVANCE_MIN_QUERY_TOKENS",
                value=1,
                help="触发上下文关联评估所需的最少查询关键词数量",
                default_value=1,
                type=int,
            ),
        ],
        commands=[
            Command(
                command="重置会话",
                description="重置当前会话历史（超级用户）",
            )
        ],
        superuser_help="""
- `重置会话`
        """.strip(),
    ).to_dict(),
)


_fallback_matcher = on_message(
    priority=999,
    block=True,
    rule=to_me(),
)

_pending_image_matcher = on_message(
    priority=998,
    block=False,
)


def _is_private_text_only_message(
    event: Event,
    event_message: Message | None,
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


@_pending_image_matcher.handle()
async def _handle_pending_image_followup(
    bot: Bot,
    event: Event,
    session: Uninfo,
    msg: UniMsg,
    state: T_State,
):
    if getattr(event, "_ai_triggered", False):
        return
    if getattr(event, "_chatinter_pending_consumed", False):
        return
    user_id = str(getattr(session.user, "id", "") or "")
    group_id = str(getattr(session.group, "id", "") or "") or None
    if not user_id or not has_pending_image_followup(user_id, group_id):
        return

    try:
        event_message = event.get_message()
    except Exception:
        event_message = None
    if not _is_private_text_only_message(event, event_message):
        logger.debug("ChatInter 私聊仅文本策略：忽略非文本消息")
        return
    tagged_message = (
        uni_to_text_with_tags(event_message)
        if event_message is not None
        else uni_to_text_with_tags(msg)
    )

    pending_target = get_pending_target_followup(user_id, group_id)
    if pending_target is not None:
        resolved_target_user_id = await resolve_pending_target_followup_user_id(
            pending_target,
            tagged_message,
            group_id,
        )
        if resolved_target_user_id:
            synthetic_message = build_pending_target_followup_message(
                pending_target.original_message,
                tagged_message,
                resolved_target_user_id,
            )
            clear_pending_target_followup(user_id, group_id)
            remember_target_resolution(
                group_id,
                pending_target.target_hint,
                resolved_target_user_id,
            )
            setattr(event, "_chatinter_pending_consumed", True)
            setattr(event, "_ai_triggered", True)
            logger.info(
                "[ChatInter] 命中待补目标会话，继续处理："
                "user="
                f"{user_id}, "
                f"group={group_id or 'private'}, "
                f"target={resolved_target_user_id}"
            )
            await handle_fallback(
                bot,
                event,
                session,
                synthetic_message,
                msg,
                route_modules=None,
                cached_plain_text=state.get("_zx_plain_text"),
                current_message_override=synthetic_message,
                from_pending_followup=True,
            )
            return
        if getattr(event, "to_me", False):
            setattr(event, "_chatinter_pending_consumed", True)
            setattr(event, "_ai_triggered", True)
            await MessageUtils.build_message(
                "我还没确定是群里的哪位，回复我并直接@对方，我就继续。"
            ).send()
            return

    if "[image" not in tagged_message:
        return

    pending = claim_pending_image_followup(user_id, group_id)
    if pending is None:
        return

    synthetic_message = build_pending_followup_message(
        pending.original_message,
        tagged_message,
    )
    setattr(event, "_chatinter_pending_consumed", True)
    setattr(event, "_ai_triggered", True)
    logger.info(
        "[ChatInter] 命中待补图会话，继续处理："
        f"user={user_id}, group={group_id or 'private'}"
    )
    await handle_fallback(
        bot,
        event,
        session,
        synthetic_message,
        msg,
        route_modules=None,
        cached_plain_text=state.get("_zx_plain_text"),
        current_message_override=synthetic_message,
        from_pending_followup=True,
    )


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


@_reset_matcher.handle()
async def _handle_reset_by_alconna(
    bot: Bot, event: GroupMessageEvent | PrivateMessageEvent, session: Uninfo
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


@driver.on_startup
async def _on_startup():
    """插件启动初始化"""
    from zhenxun.configs.config import BotConfig

    logger.info("ChatInter 插件已加载")
    await ensure_lifecycle_hooks_registered()
    _chat_memory.set_bot_nickname(BotConfig.self_nickname)
    await PluginRegistry.preload_cache()
