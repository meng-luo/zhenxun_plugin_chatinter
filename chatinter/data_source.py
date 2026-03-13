"""
ChatInter - 数据源导出

整合各子模块的导出，保持向后兼容。
"""

# 聊天记忆管理
from .memory import ChatMemory, _chat_memory

# 聊天响应处理
from .chat_handler import (
    handle_chat_intent,
    handle_chat_message,
    build_chat_system_prompt,
    reroute_to_plugin,
)

# 主处理器
from .handler import handle_fallback

__all__ = [
    "ChatMemory",
    "_chat_memory",
    "handle_chat_intent",
    "handle_chat_message",
    "build_chat_system_prompt",
    "reroute_to_plugin",
    "handle_fallback",
]
