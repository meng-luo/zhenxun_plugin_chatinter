"""Compatibility export for the renamed chat reply agent."""

from .chat_reply_agent import ChatReplyAgent

PrivateChatAgent = ChatReplyAgent

__all__ = ["PrivateChatAgent"]
