"""
ChatInter - UniMessage 消息处理工具

使用 nonebot_plugin_alconna 的 UniMessage 统一处理消息。
"""

from nonebot.adapters import Message
from nonebot_plugin_alconna.uniseg import (
    At,
    Image,
    Reply,
    Segment,
    Text,
    UniMessage,
    Video,
    Voice,
)


def uni_to_text_with_tags(
    message: UniMessage | Message | str, keep_at_id: bool = True
) -> str:
    """将 UniMessage 转换为带标签的文本表示

    格式化规则：
    - 文本：直接输出
    - @：[@user_id] 或 [@]
    - 图片：[image]
    - 回复：[reply]
    - 表情：[face]
    - 视频：[video]
    - 语音：[voice]
    - 其他：[segment 类型]

    参数:
        message: UniMessage、Message 或字符串
        keep_at_id: 是否保留@的用户 ID

    返回:
        格式化后的文本
    """
    if isinstance(message, str):
        return message

    if isinstance(message, Message):
        return _adapter_message_to_text(message, keep_at_id)

    if not isinstance(message, UniMessage):
        return str(message)

    parts: list[str] = []
    for seg in message:
        text = _segment_to_text(seg, keep_at_id)
        parts.append(text)

    return "".join(parts)


def _segment_to_text(seg: Segment, keep_at_id: bool = True) -> str:
    """将单个消息段转换为文本表示"""
    if isinstance(seg, Text):
        return seg.text

    if isinstance(seg, At):
        if seg.flag == "all":
            return "[@所有人]"
        if keep_at_id and seg.target:
            return f"[@{seg.target}]"
        return "[@]"

    if isinstance(seg, Image):
        return "[image]"

    if isinstance(seg, Reply):
        return "[reply]"

    if isinstance(seg, Video):
        return "[video]"

    if isinstance(seg, Voice):
        return "[voice]"

    seg_type = type(seg).__name__.lower()
    return f"[{seg_type}]"


def _adapter_message_to_text(message: Message, keep_at_id: bool = True) -> str:
    parts: list[str] = []
    for seg in message:
        seg_type = getattr(seg, "type", "")
        seg_data = getattr(seg, "data", {}) or {}

        if seg_type == "text":
            parts.append(str(seg_data.get("text", "")))
            continue

        if seg_type == "at":
            qq_value = str(seg_data.get("qq", "")).strip()
            if qq_value == "all":
                parts.append("[@所有人]")
            elif keep_at_id and qq_value:
                parts.append(f"[@{qq_value}]")
            else:
                parts.append("[@]")
            continue

        if seg_type == "image":
            parts.append("[image]")
            continue

        if seg_type == "reply":
            parts.append("[reply]")
            continue

        if seg_type == "video":
            parts.append("[video]")
            continue

        if seg_type in {"record", "voice"}:
            parts.append("[voice]")
            continue

        if seg_type:
            parts.append(f"[{seg_type}]")
        else:
            parts.append(str(seg))

    return "".join(parts)


def extract_reply_from_message(message: UniMessage | Message) -> str | None:
    """从消息中提取回复 ID

    参数:
        message: UniMessage 或 Message

    返回:
        回复 ID，如果未找到返回 None
    """
    if isinstance(message, UniMessage):
        reply_seg = message.get(Reply, 0)
        if reply_seg:
            return reply_seg.id if hasattr(reply_seg, "id") and reply_seg.id else None
        return None

    if isinstance(message, Message):
        for seg in message:
            if getattr(seg, "type", "") != "reply":
                continue
            seg_data = getattr(seg, "data", {}) or {}
            reply_id = seg_data.get("id") or seg_data.get("message_id")
            if reply_id:
                return str(reply_id)
        return None

    return None


def remove_reply_segment(message: UniMessage) -> UniMessage:
    """从消息中移除回复 Segment

    参数:
        message: UniMessage

    返回:
        移除回复后的消息
    """
    return message.exclude(Reply)
