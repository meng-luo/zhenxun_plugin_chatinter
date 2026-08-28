"""Persona-neutral terminal replies generated without a model call."""

EMPTY_REPLY_TEXT = "刚才没能正常回复，请再说一次。"
AGENT_ERROR_REPLY_TEXT = "刚才处理时出了点问题。"
PLUGIN_SUCCESS_REPLY_TEXT = "已经处理好了。"
PLUGIN_FAILURE_REPLY_TEXT = "这次没有执行，我先不乱操作。"
PLUGIN_SELECTION_REPLY_TEXT = "我没能确定该用哪个功能，所以没有贸然执行。"
WEB_SEARCH_UNAVAILABLE_REPLY_TEXT = "暂时没有找到可靠的公开信息。"

__all__ = [
    "AGENT_ERROR_REPLY_TEXT",
    "EMPTY_REPLY_TEXT",
    "PLUGIN_FAILURE_REPLY_TEXT",
    "PLUGIN_SELECTION_REPLY_TEXT",
    "PLUGIN_SUCCESS_REPLY_TEXT",
    "WEB_SEARCH_UNAVAILABLE_REPLY_TEXT",
]
