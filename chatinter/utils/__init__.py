"""
ChatInter - 工具函数包

使用 UniMessage 统一处理消息。
"""

from .cache import (
    clear_impression_cache,
    get_user_impression_with_cache,
)
from .impression_provider import (
    ImpressionProvider,
    SignUserImpressionProvider,
    get_impression_provider,
    set_impression_provider,
)
from .multimodal import (
    extract_chat_images_from_message,
    extract_chat_images_from_reply_chain,
    extract_images_from_message,
    extract_images_from_reply_chain,
)
from .unimsg_utils import (
    extract_reply_from_message,
    remove_reply_segment,
    uni_to_text_with_tags,
)

__all__ = [
    "ImpressionProvider",
    "SignUserImpressionProvider",
    "clear_impression_cache",
    "extract_chat_images_from_message",
    "extract_chat_images_from_reply_chain",
    "extract_images_from_message",
    "extract_images_from_reply_chain",
    "extract_reply_from_message",
    "get_impression_provider",
    "get_user_impression_with_cache",
    "remove_reply_segment",
    "set_impression_provider",
    "uni_to_text_with_tags",
]
