"""
ChatInter - 多模态消息处理工具

支持从消息中提取图片等多媒体内容，转换为 LLM 可识别的格式。
"""

import base64
from pathlib import Path

import aiofiles
from nonebot.adapters import Message as AdapterMessage
from nonebot_plugin_alconna.uniseg import Image, UniMessage

from zhenxun.services.llm.types import LLMContentPart
from zhenxun.utils.http_utils import AsyncHttpx


def _detect_image_mime(data: bytes, hint: str = "") -> str:
    """按文件头魔数探测图片 MIME(补强2:取代硬编码 image/png)。

    优先看二进制魔数,其次看路径/URL 扩展名,最终兜底 image/png。
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    lowered = (hint or "").lower()
    for ext, mime in (
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".gif", "image/gif"),
        (".webp", "image/webp"),
        (".bmp", "image/bmp"),
    ):
        if ext in lowered:
            return mime
    return "image/png"


def _image_part_from_bytes(content: bytes, hint: str = "") -> LLMContentPart:
    b64_data = base64.b64encode(content).decode("utf-8")
    return LLMContentPart.image_base64_part(b64_data, _detect_image_mime(content, hint))


async def extract_images_from_message(
    raw_message: str | UniMessage | AdapterMessage,
) -> list[LLMContentPart]:
    """从消息中提取图片，转换为 LLM 可识别的 Base64 格式

    参数:
        raw_message: 原始消息（字符串、Message 或 UniMessage）

    返回:
        list[LLMContentPart]: 图片内容列表
    """
    images: list[LLMContentPart] = []

    try:
        if isinstance(raw_message, UniMessage):
            uni_msg = raw_message
            for seg in uni_msg:
                if isinstance(seg, Image):
                    image_part = await _process_image_segment(seg)
                    if image_part:
                        images.append(image_part)
            return images

        if isinstance(raw_message, AdapterMessage):
            for seg in raw_message:
                if getattr(seg, "type", "") != "image":
                    continue
                image_part = await _process_adapter_image_segment(seg)
                if image_part:
                    images.append(image_part)
            return images

        uni_msg = _safe_to_unimessage(raw_message)
        if uni_msg is None:
            return images
    except Exception:
        return images

    for seg in uni_msg:
        if isinstance(seg, Image):
            image_part = await _process_image_segment(seg)
            if image_part:
                images.append(image_part)

    return images


def _safe_to_unimessage(raw_message) -> UniMessage | None:
    if isinstance(raw_message, UniMessage):
        return raw_message

    of_method = getattr(UniMessage, "of", None)
    if callable(of_method):
        try:
            value = of_method(raw_message)
            if isinstance(value, UniMessage):
                return value
        except Exception:
            pass

    generate_method = getattr(UniMessage, "generate", None)
    if callable(generate_method):
        try:
            generated = generate_method(message=raw_message)
            if isinstance(generated, UniMessage):
                return generated
        except Exception:
            pass

    return None


async def _process_adapter_image_segment(seg) -> LLMContentPart | None:
    seg_data = getattr(seg, "data", {}) or {}

    url = seg_data.get("url")
    if url:
        try:
            media_bytes = await AsyncHttpx.get_content(str(url))
            return _image_part_from_bytes(media_bytes, str(url))
        except Exception:
            pass

    file_value = seg_data.get("file")
    if file_value:
        try:
            path = Path(str(file_value))
            if path.exists() and path.is_file():
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                return _image_part_from_bytes(content, str(file_value))
        except Exception:
            pass

    return None


async def _process_image_segment(seg: Image) -> LLMContentPart | None:
    """处理 Alconna Image Segment

    参数:
        seg: Image Segment

    返回:
        LLMContentPart | None: Base64 格式的图片内容
    """
    if hasattr(seg, "raw") and seg.raw:
        if isinstance(seg.raw, bytes):
            return _image_part_from_bytes(seg.raw)

    if getattr(seg, "path", None):
        try:
            path = Path(str(seg.path))
            if path.exists():
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                return _image_part_from_bytes(content, str(seg.path))
        except Exception:
            pass

    if getattr(seg, "url", None):
        try:
            media_bytes = await AsyncHttpx.get_content(str(seg.url))
            return _image_part_from_bytes(media_bytes, str(seg.url))
        except Exception:
            pass

    return None


async def extract_images_from_reply_chain(
    reply_images: list[Image],
) -> list[LLMContentPart]:
    """从回复链图片中提取图片，转换为 LLM 可识别的 Base64 格式

    参数:
        reply_images: 回复链中的图片 Image Segment 列表

    返回:
        list[LLMContentPart]: 图片内容列表
    """
    images: list[LLMContentPart] = []

    for img_seg in reply_images:
        image_part = await _process_image_segment(img_seg)
        if image_part:
            images.append(image_part)

    return images


__all__ = [
    "extract_images_from_message",
    "extract_images_from_reply_chain",
]
