"""
ChatInter - 数据源导出

整合各子模块的导出，保持向后兼容。
"""

# 聊天记忆管理
# 聊天响应处理
from .chat_handler import (
    build_chat_system_prompt,
    handle_chat_message,
    reroute_to_plugin,
)

# 主处理器
from .handler import handle_fallback
from .memory import ChatMemory, _chat_memory

__all__ = [
    "ChatMemory",
    "_chat_memory",
    "build_chat_system_prompt",
    "handle_chat_message",
    "handle_fallback",
    "reroute_to_plugin",
]
