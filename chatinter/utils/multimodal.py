"""
ChatInter - 多模态消息处理工具

支持从消息中提取图片等多媒体内容，转换为 LLM 可识别的格式。
"""

import base64
from dataclasses import dataclass
from html import escape
from pathlib import Path

import aiofiles
from nonebot.adapters import Message as AdapterMessage
from nonebot_plugin_alconna.uniseg import Image, UniMessage

from zhenxun.utils.http_utils import AsyncHttpx

from ..config import build_agent_generation_config
from ..llm_compat import LLMContentPart

MAX_CHAT_IMAGE_PARTS = 3
MAX_CHAT_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_TOO_LARGE_CONTEXT = "<image_context>图片过大，未传入视觉模型</image_context>"


@dataclass(frozen=True)
class ChatImageExtraction:
    image_parts: list[LLMContentPart]
    context_xml: str = ""
    original_count: int = 0
    skipped_oversized: int = 0


@dataclass(frozen=True)
class _ImagePartResult:
    part: LLMContentPart | None = None
    oversized: bool = False


@dataclass(frozen=True)
class ChatImageRouting:
    image_parts: list[LLMContentPart]
    context_xml: str = ""
    mode: str = "none"
    vision_model: str | None = None
    original_count: int = 0


def _detect_image_mime(data: bytes, hint: str = "") -> str:
    """按文件头魔数探测图片 MIME。

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


def _image_part_from_bytes(content: bytes, hint: str = "") -> LLMContentPart | None:
    return _image_part_result_from_bytes(content, hint).part


def _image_part_result_from_bytes(content: bytes, hint: str = "") -> _ImagePartResult:
    if len(content) > MAX_CHAT_IMAGE_BYTES:
        return _ImagePartResult(oversized=True)
    b64_data = base64.b64encode(content).decode("utf-8")
    return _ImagePartResult(
        part=LLMContentPart.image_base64_part(
            b64_data,
            _detect_image_mime(content, hint),
        )
    )


async def extract_images_from_message(
    raw_message: str | UniMessage | AdapterMessage,
) -> list[LLMContentPart]:
    """从消息中提取图片，转换为 LLM 可识别的 Base64 格式

    参数:
        raw_message: 原始消息（字符串、Message 或 UniMessage）

    返回:
        list[LLMContentPart]: 图片内容列表
    """
    return (await extract_chat_images_from_message(raw_message)).image_parts


async def extract_chat_images_from_message(
    raw_message: str | UniMessage | AdapterMessage,
) -> ChatImageExtraction:
    images: list[LLMContentPart] = []
    skipped_oversized = 0

    try:
        if isinstance(raw_message, UniMessage):
            uni_msg = raw_message
            for seg in uni_msg:
                if isinstance(seg, Image):
                    result = await _process_image_segment_result(seg)
                    if result.part:
                        images.append(result.part)
                    skipped_oversized += int(result.oversized)
            return _image_extraction(images, skipped_oversized)

        if isinstance(raw_message, AdapterMessage):
            for seg in raw_message:
                if getattr(seg, "type", "") != "image":
                    continue
                result = await _process_adapter_image_segment_result(seg)
                if result.part:
                    images.append(result.part)
                skipped_oversized += int(result.oversized)
            return _image_extraction(images, skipped_oversized)

        uni_msg = _safe_to_unimessage(raw_message)
        if uni_msg is None:
            return _image_extraction(images, skipped_oversized)
    except Exception:
        return _image_extraction(images, skipped_oversized)

    for seg in uni_msg:
        if isinstance(seg, Image):
            result = await _process_image_segment_result(seg)
            if result.part:
                images.append(result.part)
            skipped_oversized += int(result.oversized)

    return _image_extraction(images, skipped_oversized)


def _image_extraction(
    images: list[LLMContentPart],
    skipped_oversized: int,
) -> ChatImageExtraction:
    return ChatImageExtraction(
        image_parts=images,
        context_xml=IMAGE_TOO_LARGE_CONTEXT if skipped_oversized else "",
        original_count=len(images) + skipped_oversized,
        skipped_oversized=skipped_oversized,
    )


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
    return (await _process_adapter_image_segment_result(seg)).part


async def _process_adapter_image_segment_result(seg) -> _ImagePartResult:
    seg_data = getattr(seg, "data", {}) or {}

    url = seg_data.get("url")
    if url:
        try:
            media_bytes = await AsyncHttpx.get_content(str(url))
            return _image_part_result_from_bytes(media_bytes, str(url))
        except Exception:
            pass

    file_value = seg_data.get("file")
    if file_value:
        try:
            path = Path(str(file_value))
            if path.exists() and path.is_file():
                if path.stat().st_size > MAX_CHAT_IMAGE_BYTES:
                    return _ImagePartResult(oversized=True)
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                return _image_part_result_from_bytes(content, str(file_value))
        except Exception:
            pass

    return _ImagePartResult()


async def _process_image_segment(seg: Image) -> LLMContentPart | None:
    return (await _process_image_segment_result(seg)).part


async def _process_image_segment_result(seg: Image) -> _ImagePartResult:
    """处理 Alconna Image Segment

    参数:
        seg: Image Segment

    返回:
        LLMContentPart | None: Base64 格式的图片内容
    """
    if hasattr(seg, "raw") and seg.raw:
        if isinstance(seg.raw, bytes):
            return _image_part_result_from_bytes(seg.raw)

    if getattr(seg, "path", None):
        try:
            path = Path(str(seg.path))
            if path.exists():
                if path.stat().st_size > MAX_CHAT_IMAGE_BYTES:
                    return _ImagePartResult(oversized=True)
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                return _image_part_result_from_bytes(content, str(seg.path))
        except Exception:
            pass

    if getattr(seg, "url", None):
        try:
            media_bytes = await AsyncHttpx.get_content(str(seg.url))
            return _image_part_result_from_bytes(media_bytes, str(seg.url))
        except Exception:
            pass

    return _ImagePartResult()


def model_supports_image_input(model_name: str | None) -> bool:
    if not str(model_name or "").strip():
        return False
    try:
        from ..provider_capability import ProviderCapabilityAdapter

        return bool(
            ProviderCapabilityAdapter.for_model(model_name).profile.supports_image_input
        )
    except Exception:
        return False


def select_vision_model(
    *,
    primary_model: str | None,
    fallback_models: tuple[str, ...] = (),
) -> str | None:
    candidates = list(fallback_models)
    try:
        from zhenxun.services.ai.config.manager import get_ai_config

        default_models = get_ai_config().get("default_models", {}) or {}
        if isinstance(default_models, dict):
            candidates.append(str(default_models.get("image") or ""))
    except Exception:
        pass
    for model_name in candidates:
        model_name = str(model_name or "").strip()
        if (
            model_name
            and model_name != primary_model
            and model_supports_image_input(model_name)
        ):
            return model_name
    return None


def image_placeholder(count: int) -> str:
    count = max(int(count or 0), 1)
    return (
        "<image_context>"
        f"当前消息包含 {count} 张图片，但当前聊天模型不支持直接看图。"
        "</image_context>"
    )


def build_labeled_image_user_content(
    text: str,
    image_parts: list[LLMContentPart],
) -> list[LLMContentPart]:
    content: list[LLMContentPart] = []
    for index, image_part in enumerate(image_parts[:MAX_CHAT_IMAGE_PARTS], 1):
        content.append(LLMContentPart.text_part(f"Image {index}:"))
        content.append(image_part)
    if text:
        content.append(LLMContentPart.text_part(text))
    return content


async def caption_images_for_chat(
    parts: list[LLMContentPart],
    *,
    text: str,
    model_name: str,
    timeout: float = 20.0,
) -> str:
    images = list(parts[:MAX_CHAT_IMAGE_PARTS])
    if not images:
        return ""
    try:
        from ..llm_compat import AI, LLMMessage

        prompt = (
            "只回答当前文字问题相关的可见信息。不要猜身份，不要扩写，不要输出列表。"
            "除非用户明确要求比较、关联、区别或哪张更好，否则不要推断多图关系。\n"
            f"当前消息：{text or '(无文字)'}"
        )
        response = await AI().generate_internal(
            [LLMMessage.user([LLMContentPart.text_part(prompt), *images])],
            model=model_name,
            config=build_agent_generation_config("chat"),
            timeout=timeout,
        )
        return str(getattr(response, "text", "") or "").strip()[:160]
    except Exception:
        return ""


async def route_images_for_chat(
    parts: list[LLMContentPart],
    *,
    text: str,
    model_name: str | None,
    fallback_models: tuple[str, ...] = (),
    timeout: float = 20.0,
) -> ChatImageRouting:
    original_count = len(parts)
    limited = list(parts[:MAX_CHAT_IMAGE_PARTS])
    if not limited:
        return ChatImageRouting(image_parts=[], original_count=original_count)
    if model_supports_image_input(model_name):
        context_xml = ""
        if original_count > len(limited):
            context_xml = (
                "<image_context>"
                f"当前消息包含 {original_count} 张图片，已只传入前 {len(limited)} 张。"
                "</image_context>"
            )
        return ChatImageRouting(
            image_parts=limited,
            context_xml=context_xml,
            mode="direct",
            vision_model=model_name,
            original_count=original_count,
        )
    vision_model = select_vision_model(
        primary_model=model_name,
        fallback_models=fallback_models,
    )
    if vision_model:
        caption = await caption_images_for_chat(
            limited,
            text=text,
            model_name=vision_model,
            timeout=timeout,
        )
        if caption:
            return ChatImageRouting(
                image_parts=[],
                context_xml=f"<image_context>{escape(caption)}</image_context>",
                mode="caption",
                vision_model=vision_model,
                original_count=original_count,
            )
    return ChatImageRouting(
        image_parts=[],
        context_xml=image_placeholder(original_count),
        mode="placeholder",
        vision_model=vision_model,
        original_count=original_count,
    )


async def extract_images_from_reply_chain(
    reply_images: list[Image],
) -> list[LLMContentPart]:
    """从回复链图片中提取图片，转换为 LLM 可识别的 Base64 格式

    参数:
        reply_images: 回复链中的图片 Image Segment 列表

    返回:
        list[LLMContentPart]: 图片内容列表
    """
    return (await extract_chat_images_from_reply_chain(reply_images)).image_parts


async def extract_chat_images_from_reply_chain(
    reply_images: list[Image],
) -> ChatImageExtraction:
    images: list[LLMContentPart] = []
    skipped_oversized = 0

    for img_seg in reply_images:
        result = await _process_image_segment_result(img_seg)
        if result.part:
            images.append(result.part)
        skipped_oversized += int(result.oversized)

    return _image_extraction(images, skipped_oversized)


__all__ = [
    "IMAGE_TOO_LARGE_CONTEXT",
    "MAX_CHAT_IMAGE_BYTES",
    "MAX_CHAT_IMAGE_PARTS",
    "ChatImageExtraction",
    "ChatImageRouting",
    "build_labeled_image_user_content",
    "caption_images_for_chat",
    "extract_chat_images_from_message",
    "extract_chat_images_from_reply_chain",
    "extract_images_from_message",
    "extract_images_from_reply_chain",
    "image_placeholder",
    "model_supports_image_input",
    "route_images_for_chat",
    "select_vision_model",
]
