"""
ChatInter - 聊天响应处理

实现聊天意图处理和消息响应生成。
"""

import asyncio
import re

from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import (
    GroupMessageEvent,
    Message,
    MessageSegment,
    PrivateMessageEvent,
)
from nonebot.plugin import get_loaded_plugins
from nonebot_plugin_alconna.uniseg import UniMessage

from zhenxun.services import chat, logger
from zhenxun.utils.message import MessageUtils

from .config import build_reasoning_generation_config, get_config_value
from .memory import _chat_memory

_REROUTE_TASKS: set[asyncio.Task] = set()
_REROUTE_TOKEN_PATTERN = re.compile(
    r"\[@(?:\d+|所有人)\]|\[image(?:#\d+)?\]|(?<![0-9A-Za-z_])@\d{5,20}(?=(?:\s|$|[的，,。.!！？?]))",
    re.IGNORECASE,
)
_IMAGE_INDEX_PATTERN = re.compile(r"\[image#(\d+)\]", re.IGNORECASE)
_MD_FENCED_CODE_PATTERN = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_MD_INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_MD_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_MD_HEADING_LINE_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_QUOTE_LINE_PATTERN = re.compile(r"(?m)^\s{0,3}>\s?")
_MD_BULLET_LINE_PATTERN = re.compile(r"(?m)^\s*[-*+]\s+")
_MD_ORDERED_LINE_PATTERN = re.compile(r"(?m)^\s*(\d+)[.)]\s+")
_MD_RULE_LINE_PATTERN = re.compile(r"(?m)^\s*([-*_]\s*){3,}\s*$")
_MD_BOLD_PATTERN = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
_MD_STRIKE_PATTERN = re.compile(r"~~(.+?)~~", re.DOTALL)
_MD_EXCESSIVE_LINE_BREAKS_PATTERN = re.compile(r"\n{3,}")
_AT_ID_TOKEN_PATTERN = re.compile(
    r"\[@(\d{5,20})\]|(?<![0-9A-Za-z_])@(\d{5,20})(?=(?:\s|$|[的，,。.!！？?]))"
)
_COMPLEX_QUERY_HINTS = (
    "代码",
    "插件",
    "报错",
    "错误",
    "异常",
    "调试",
    "排查",
    "怎么",
    "如何",
    "实现",
    "步骤",
    "配置",
    "脚本",
    "方案",
    "traceback",
    "exception",
    "nonebot",
    "python",
    "api",
)


def _is_complex_query(message_text: str) -> bool:
    normalized = str(message_text or "").strip().lower()
    if not normalized:
        return False
    if "```" in normalized or "\n" in normalized:
        return True
    if len(normalized) >= 36:
        return True
    return any(hint in normalized for hint in _COMPLEX_QUERY_HINTS)


async def handle_chat_message(
    message: str,
    user_id: str,
    group_id: str | None = None,
    nickname: str = "用户",
    mention_name_map: dict[str, str] | None = None,
) -> str | UniMessage:
    chat_style = get_config_value("CHAT_STYLE", "")

    system_prompt = await build_chat_system_prompt(
        user_id=user_id,
        nickname=nickname,
        group_id=group_id,
        chat_style=chat_style,
        message_text=message,
    )

    logger.debug(f"系统提示词：{system_prompt[:500]}...")

    try:
        response = await chat(
            message=message,
            instruction=system_prompt,
            model=get_config_value("INTENT_MODEL", None),
            config=build_reasoning_generation_config(),
        )

        reply_text = normalize_ai_reply_text(
            response.text if response else "抱歉，我现在有点累，稍后再聊吧~"
        )
        reply_text = replace_mention_ids_with_names(reply_text, mention_name_map)
        return reply_text

    except Exception as e:
        logger.error(f"聊天处理失败：{e}")
        return MessageUtils.build_failure_message()


async def build_chat_system_prompt(
    user_id: str,
    nickname: str,
    group_id: str | None = None,
    chat_style: str = "",
    message_text: str = "",
) -> str:
    use_sign_in_impression = get_config_value("USE_SIGN_IN_IMPRESSION", True)

    impression_prompt = ""
    if use_sign_in_impression:
        impression, attitude = await _chat_memory.get_user_impression(user_id)
        impression_prompt = (
            f"\n\n用户：{nickname} | 好感度：{impression:.0f} | 态度：{attitude}\n"
            f"按态度回复：排斥/警惕→冷淡简短；一般/可以交流→正常友好；好朋友/是个好人→热情；亲密/恋人→亲密关心"
        )

    style_text = (
        f"{chat_style}风格的"
        if chat_style
        else "日式二次元、软萌中带一点傲娇的"
    )
    allow_long_for_complex = bool(
        get_config_value("CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX", True)
    )
    is_complex_query = _is_complex_query(message_text)
    if allow_long_for_complex and is_complex_query:
        length_rule = (
            "当前问题偏复杂（如代码/排错/实现类），允许使用分点和步骤化说明，"
            "优先给出可执行结论，不受80字限制。"
        )
    else:
        length_rule = "默认控制在80字以内，除非用户明确要求详细步骤。"

    base_prompt = (
        f"你是{style_text}机器人助手，优先使用中文。"
        "语气要可爱自然，避免生硬、严肃、正式的客服腔。"
        "可以适度使用“好啦、诶嘿、唔、哼哼、欸”等口吻词，但不要每句都堆叠。"
        f"{length_rule}"
        "若上下文信息不足，先问一个最关键的澄清问题，不要凭空猜测。"
        "如果是结构化输出或插件命令，不要加入口癖修饰。"
    )
    group_prompt = f"\n群组 ID：{group_id}" if group_id else ""

    custom_prompt = get_config_value("CUSTOM_PROMPT", "")
    custom_prompt_text = f"\n额外设定：{custom_prompt}" if custom_prompt else ""

    return base_prompt + impression_prompt + group_prompt + custom_prompt_text


async def reroute_to_plugin(
    bot: Bot,
    event: Event,
    command: str,
    target_modules: set[str] | None = None,
    extra_image_segments: list[MessageSegment] | None = None,
) -> bool:
    try:
        import time

        event_data = event.model_dump()
        command_text = command.strip()
        bot_self_id = str(getattr(bot, "self_id", "")) or None
        new_message = _build_reroute_message(
            command_text,
            event,
            bot_self_id,
            extra_images=extra_image_segments,
        )

        event_data["message"] = new_message
        event_data["raw_message"] = command_text
        event_data["plain_text"] = command_text
        event_data["reply"] = None

        if hasattr(bot, "self_id"):
            event_data["self_id"] = bot.self_id

        event_data["message_id"] = int(time.time() * 1000)
        event_data["time"] = int(time.time())

        logger.debug(
            f"构造重路由消息：'{new_message.extract_plain_text()}', "
            f"self_id={event_data.get('self_id')}, "
            f"images={sum(1 for seg in new_message if seg.type == 'image')}, "
            f"ats={sum(1 for seg in new_message if seg.type == 'at')}"
        )

        if isinstance(event, GroupMessageEvent):
            new_event = GroupMessageEvent(**event_data)
        elif isinstance(event, PrivateMessageEvent):
            new_event = PrivateMessageEvent(**event_data)
        else:
            logger.warning(f"不支持的事件类型：{type(event)}")
            return False

        setattr(new_event, "_ai_triggered", True)
        expanded_target_modules = _expand_reroute_target_modules(target_modules)
        if expanded_target_modules:
            setattr(new_event, "_ai_route_modules", frozenset(expanded_target_modules))
        route_heads = _extract_reroute_heads(command_text)
        if route_heads:
            setattr(new_event, "_ai_route_heads", frozenset(route_heads))

        task = asyncio.create_task(bot.handle_event(new_event))
        _REROUTE_TASKS.add(task)
        task.add_done_callback(_REROUTE_TASKS.discard)
        logger.info(f"消息重路由成功：{command_text}")
        return True

    except Exception as e:
        logger.error(f"消息重路由失败：{e}")
        return False


def _parse_at_target(token: str) -> str | None:
    token = token.strip()
    if token.startswith("[@") and token.endswith("]"):
        target = token[2:-1].strip()
    elif token.startswith("@"):
        target = token[1:].strip()
    else:
        return None
    if not target:
        return None
    if target in {"所有人", "all"}:
        return "all"
    if target.isdigit():
        return target
    return None


def _expand_reroute_target_modules(target_modules: set[str] | None) -> set[str]:
    if not target_modules:
        return set()

    expanded = {
        item.strip()
        for item in target_modules
        if isinstance(item, str) and item.strip()
    }
    if not expanded:
        return set()

    for plugin in get_loaded_plugins():
        plugin_name = str(getattr(plugin, "name", "") or "").strip()
        module_name = str(getattr(plugin, "module_name", "") or "").strip()
        if not plugin_name and not module_name:
            continue
        if module_name and module_name in expanded:
            expanded.add(plugin_name)
            continue
        if plugin_name and plugin_name in expanded and module_name:
            expanded.add(module_name)
    return expanded


def _extract_reroute_heads(command_text: str) -> set[str]:
    normalized = str(command_text or "").strip()
    if not normalized:
        return set()
    head = normalized.split(" ", 1)[0].strip()
    if not head:
        return set()
    return {head, head.lower(), head.casefold()}


def _extract_source_images(event: Event) -> list[MessageSegment]:
    try:
        source_message = event.get_message()
    except Exception:
        source_message = getattr(event, "message", None)
    if not isinstance(source_message, Message):
        return []
    images: list[MessageSegment] = []
    for seg in source_message:
        if seg.type == "image":
            images.append(seg)
    return images


def _extract_source_mentions(
    event: Event,
    bot_self_id: str | None,
) -> list[MessageSegment]:
    try:
        source_message = event.get_message()
    except Exception:
        source_message = getattr(event, "message", None)
    if not isinstance(source_message, Message):
        return []
    mentions: list[MessageSegment] = []
    for seg in source_message:
        if seg.type != "at":
            continue
        qq_value = str(seg.data.get("qq", "")).strip()
        if not qq_value:
            continue
        if bot_self_id and qq_value == str(bot_self_id):
            continue
        mentions.append(seg)
    return mentions


def _build_reroute_message(
    command_text: str,
    event: Event,
    bot_self_id: str | None = None,
    extra_images: list[MessageSegment] | None = None,
) -> Message:
    if not command_text:
        return Message("")

    source_images = _extract_source_images(event)
    if extra_images:
        for image in extra_images:
            if not isinstance(image, MessageSegment):
                continue
            if image.type != "image":
                continue
            source_images.append(image)
    source_mentions = _extract_source_mentions(event, bot_self_id)
    has_explicit_image_token = False
    has_explicit_at_token = False
    result = Message()
    cursor = 0

    for match in _REROUTE_TOKEN_PATTERN.finditer(command_text):
        if match.start() > cursor:
            result += MessageSegment.text(command_text[cursor : match.start()])

        token = match.group(0)
        lower_token = token.lower()
        if lower_token.startswith("[image"):
            has_explicit_image_token = True
            image_index = 0
            index_match = _IMAGE_INDEX_PATTERN.fullmatch(token)
            if index_match:
                parsed_index = int(index_match.group(1))
                image_index = max(parsed_index - 1, 0)
            if source_images:
                chosen_index = min(image_index, len(source_images) - 1)
                result += source_images[chosen_index]
            else:
                result += MessageSegment.text(token)
        else:
            target = _parse_at_target(token)
            if target == "all":
                has_explicit_at_token = True
                result += MessageSegment.at("all")
            elif target and target.isdigit():
                has_explicit_at_token = True
                result += MessageSegment.at(int(target))
            else:
                result += MessageSegment.text(token)
        cursor = match.end()

    if cursor < len(command_text):
        result += MessageSegment.text(command_text[cursor:])

    if not has_explicit_at_token and source_mentions:
        result += MessageSegment.text(" ")
        for mention in source_mentions:
            result += mention

    if not has_explicit_image_token and source_images:
        result += MessageSegment.text(" ")
        result += source_images[0]

    if not result:
        return Message(command_text)
    return result


def _looks_like_markdown(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    markdown_signals = (
        _MD_LINK_PATTERN.search(text) is not None,
        _MD_IMAGE_PATTERN.search(text) is not None,
        _MD_INLINE_CODE_PATTERN.search(text) is not None,
        _MD_HEADING_LINE_PATTERN.search(text) is not None,
        _MD_QUOTE_LINE_PATTERN.search(text) is not None,
        _MD_BULLET_LINE_PATTERN.search(text) is not None,
        _MD_ORDERED_LINE_PATTERN.search(text) is not None,
        _MD_BOLD_PATTERN.search(text) is not None,
        _MD_STRIKE_PATTERN.search(text) is not None,
    )
    return any(markdown_signals)


def _has_code_markdown(text: str) -> bool:
    return "```" in text or _MD_FENCED_CODE_PATTERN.search(text) is not None


def normalize_ai_reply_text(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return normalized
    # 代码回复保留 Markdown，避免代码块/语言标记被剥离。
    if _has_code_markdown(normalized):
        return normalized
    if not _looks_like_markdown(normalized):
        return normalized

    converted = normalized
    converted = _MD_IMAGE_PATTERN.sub(
        lambda match: (
            f"{match.group(1)} ({match.group(2)})"
            if match.group(1).strip()
            else match.group(2)
        ),
        converted,
    )
    converted = _MD_LINK_PATTERN.sub(
        lambda match: f"{match.group(1)} ({match.group(2)})",
        converted,
    )
    converted = _MD_HEADING_LINE_PATTERN.sub("", converted)
    converted = _MD_QUOTE_LINE_PATTERN.sub("", converted)
    converted = _MD_BULLET_LINE_PATTERN.sub("• ", converted)
    converted = _MD_ORDERED_LINE_PATTERN.sub(r"\1. ", converted)
    converted = _MD_RULE_LINE_PATTERN.sub("", converted)
    converted = _MD_BOLD_PATTERN.sub(r"\2", converted)
    converted = _MD_STRIKE_PATTERN.sub(r"\1", converted)
    converted = "\n".join(line.rstrip() for line in converted.splitlines())
    converted = _MD_EXCESSIVE_LINE_BREAKS_PATTERN.sub("\n\n", converted).strip()
    return converted or normalized


def replace_mention_ids_with_names(
    text: str,
    mention_name_map: dict[str, str] | None = None,
) -> str:
    normalized = (text or "").strip()
    if not normalized or not mention_name_map:
        return normalized

    def _replace(match: re.Match[str]) -> str:
        user_id = (match.group(1) or match.group(2) or "").strip()
        if not user_id:
            return match.group(0)
        nickname = mention_name_map.get(user_id)
        if not nickname:
            return match.group(0)
        return f"@{nickname}"

    return _AT_ID_TOKEN_PATTERN.sub(_replace, normalized)


__all__ = [
    "build_chat_system_prompt",
    "handle_chat_message",
    "normalize_ai_reply_text",
    "replace_mention_ids_with_names",
    "reroute_to_plugin",
]
