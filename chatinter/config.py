"""
ChatInter - 配置项获取

配置项已在 __init__.py 的 PluginExtraData 中注册，
此处仅提供便捷的获取函数。
"""

from zhenxun.configs.config import Config

# 配置默认值
DEFAULTS = {
    "ENABLE_FALLBACK": True,
    "INTENT_MODEL": None,
    "INTENT_TIMEOUT": 30,
    "CONFIDENCE_THRESHOLD": 0.7,
    "CHAT_STYLE": "",
    "USE_SIGN_IN_IMPRESSION": True,
    "CONTEXT_PREFIX_SIZE": 5,
    "SESSION_CONTEXT_LIMIT": 20,
    "MAX_REPLY_LAYERS": 3,
}


def get_config_value(key: str, default=None):
    """
    获取配置值

    参数:
        key: 配置键
        default: 默认值（如果为 None 则使用 DEFAULTS 中的默认值）

    返回:
        配置值
    """
    module = "chatinter"
    if default is None:
        default = DEFAULTS.get(key)
    return Config.get_config(module, key, default)

