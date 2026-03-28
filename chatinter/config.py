from typing import Any

from zhenxun.configs.config import Config
from zhenxun.services.llm.config.generation import (
    LLMGenerationConfig,
    ReasoningConfig,
    ReasoningEffort,
)

CHATINTER_GROUP = "chatinter"
AI_GROUP = "AI"

DEFAULTS = {
    "ENABLE_FALLBACK": True,
    "ENABLE_AGENT_MODE": True,
    "AGENT_MAX_TOOL_STEPS": 4,
    "AGENT_TOTAL_TIMEOUT": 0,
    "AGENT_EXPAND_TOOLS_STEP": 2,
    "AGENT_TOOL_FAILURE_LIMIT": 2,
    "AGENT_FAILED_ROUND_LIMIT": 2,
    "AGENT_STRICT_TOOL_SELECT": True,
    "INTENT_MODEL": "",
    "INTENT_TIMEOUT": 20,
    "CONFIDENCE_THRESHOLD": 0.72,
    "CHAT_STYLE": "",
    "CUSTOM_PROMPT": "",
    "MCP_ENDPOINTS": "",
    "REASONING_EFFORT": "MEDIUM",
    "USE_SIGN_IN_IMPRESSION": True,
    "CONTEXT_PREFIX_SIZE": 5,
    "SESSION_CONTEXT_LIMIT": 20,
    "MAX_REPLY_LAYERS": 3,
    "HISTORY_RECALL_LIMIT": 4,
    "HISTORY_RECALL_MIN_SCORE": 0.18,
    "HISTORY_RECALL_CANDIDATE_LIMIT": 60,
    "GROUP_BACKGROUND_FETCH_MULTIPLIER": 3,
    "GROUP_BACKGROUND_RELEVANT_LIMIT": 3,
    "GROUP_BACKGROUND_MIN_SCORE": 0.16,
    "CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX": True,
    "ENABLE_CONTEXT_RELEVANCE_GATE": True,
    "CONTEXT_RELEVANCE_THRESHOLD": 0.11,
    "CONTEXT_RELEVANCE_SAMPLE_LIMIT": 18,
    "CONTEXT_RELEVANCE_MIN_QUERY_TOKENS": 1,
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


def _fallback_model_name() -> str | None:
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

    if key in {
        "ENABLE_FALLBACK",
        "ENABLE_AGENT_MODE",
        "AGENT_STRICT_TOOL_SELECT",
        "CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX",
        "ENABLE_CONTEXT_RELEVANCE_GATE",
    }:
        return _parse_bool(raw_value, bool(default))

    if key == "INTENT_MODEL":
        model_text = str(raw_value or "").strip()
        if model_text:
            return model_text
        return _fallback_model_name()

    if key == "INTENT_TIMEOUT":
        try:
            timeout = int(raw_value)
        except (TypeError, ValueError):
            timeout = 0
        if timeout > 0:
            return timeout
        return _fallback_timeout()

    if key == "CONFIDENCE_THRESHOLD":
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return float(default)

    if key in {
        "HISTORY_RECALL_MIN_SCORE",
        "GROUP_BACKGROUND_MIN_SCORE",
        "CONTEXT_RELEVANCE_THRESHOLD",
    }:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return float(default)

    if key in {"CHAT_STYLE", "CUSTOM_PROMPT", "MCP_ENDPOINTS"}:
        return str(raw_value or "").strip()

    if key == "REASONING_EFFORT":
        return _normalize_reasoning_effort(raw_value)

    if key in {
        "USE_SIGN_IN_IMPRESSION",
    }:
        return _parse_bool(raw_value, bool(default))

    if key in {
        "AGENT_MAX_TOOL_STEPS",
        "AGENT_TOTAL_TIMEOUT",
        "AGENT_EXPAND_TOOLS_STEP",
        "AGENT_TOOL_FAILURE_LIMIT",
        "AGENT_FAILED_ROUND_LIMIT",
        "CONTEXT_PREFIX_SIZE",
        "SESSION_CONTEXT_LIMIT",
        "MAX_REPLY_LAYERS",
        "HISTORY_RECALL_LIMIT",
        "HISTORY_RECALL_CANDIDATE_LIMIT",
        "GROUP_BACKGROUND_FETCH_MULTIPLIER",
        "GROUP_BACKGROUND_RELEVANT_LIMIT",
        "CONTEXT_RELEVANCE_SAMPLE_LIMIT",
        "CONTEXT_RELEVANCE_MIN_QUERY_TOKENS",
    }:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return int(default)

    return raw_value


def get_mcp_endpoints() -> list[str]:
    raw = get_config_value("MCP_ENDPOINTS", "")
    if not raw:
        return []
    return [item.strip().rstrip("/") for item in str(raw).split(",") if item.strip()]


def build_reasoning_generation_config() -> LLMGenerationConfig | None:
    effort_text = get_config_value("REASONING_EFFORT", "")
    if not effort_text:
        return None
    effort = (
        ReasoningEffort.MEDIUM
        if effort_text == "MEDIUM"
        else ReasoningEffort.HIGH
    )
    return LLMGenerationConfig(
        reasoning=ReasoningConfig(
            effort=effort,
            show_thoughts=False,
        )
    )
