from typing import Any

from zhenxun.configs.config import Config
from zhenxun.services.llm.config.generation import (
    LLMGenerationConfig,
    ReasoningConfig,
    ReasoningEffort,
)

CHATINTER_GROUP = "chatinter"
AI_GROUP = "AI"

# 固定策略参数：不再注册为插件配置项，避免配置面过大。
CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX = True
MAX_REPLY_LAYERS = 3
ROUTE_CANDIDATE_EXPAND_STEP = 1
ROUTE_CANDIDATE_INITIAL_LIMIT = 24
ROUTE_CANDIDATE_MAX_LIMIT = 30
ROUTE_CANDIDATE_MIN_SCORE = 0.35
ROUTE_DEFERRED_NAMESPACE_ENABLED = False
LLM_VERIFY_ALL_ROUTES = True
ROUTE_OBSERVER_MAX_RECORDS = 400
SESSION_CONTEXT_LIMIT = 20
USE_SIGN_IN_IMPRESSION = True

DEFAULTS = {
    "ENABLE_FALLBACK": True,
    "INTENT_TIMEOUT": 20,
    "CHAT_STYLE": "",
    "CUSTOM_PROMPT": "",
    "REASONING_EFFORT": "MEDIUM",
}


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_reasoning_effort(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"MEDIUM", "HIGH"}:
        return text
    return ""


def get_model_name() -> str | None:
    model_name = Config.get_config(AI_GROUP, "DEFAULT_MODEL_NAME", "")
    model_name = str(model_name or "").strip()
    return model_name or None


def _fallback_timeout() -> int:
    client_settings = Config.get_config(AI_GROUP, "CLIENT_SETTINGS", None)
    timeout = getattr(client_settings, "timeout", None)
    if isinstance(timeout, int) and timeout > 0:
        return timeout
    return int(DEFAULTS["INTENT_TIMEOUT"])


def get_config_value(key: str, default: Any = None):
    key = key.upper()
    if default is None:
        default = DEFAULTS.get(key)

    raw_value = Config.get_config(CHATINTER_GROUP, key, default)

    if key == "ENABLE_FALLBACK":
        return _parse_bool(raw_value, bool(default))

    if key == "INTENT_TIMEOUT":
        try:
            timeout = int(raw_value)
        except (TypeError, ValueError):
            timeout = 0
        if timeout > 0:
            return timeout
        return _fallback_timeout()

    if key in {"CHAT_STYLE", "CUSTOM_PROMPT"}:
        return str(raw_value or "").strip()

    if key == "REASONING_EFFORT":
        return _normalize_reasoning_effort(raw_value)

    return raw_value


def build_reasoning_generation_config() -> LLMGenerationConfig | None:
    effort_text = get_config_value("REASONING_EFFORT", "")
    if not effort_text:
        return None
    effort = ReasoningEffort.MEDIUM if effort_text == "MEDIUM" else ReasoningEffort.HIGH
    return LLMGenerationConfig(
        reasoning=ReasoningConfig(
            effort=effort,
            show_thoughts=False,
        )
    )
