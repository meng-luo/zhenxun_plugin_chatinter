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
AGENT_ABORT_PARALLEL_ON_FAILURE = True
AGENT_DIMINISHING_TOKEN_DELTA = 220
AGENT_EXPAND_TOOLS_STEP = 2
AGENT_PARALLEL_SAFE_TOOLS = 4
AGENT_STRICT_TOOL_SELECT = True
AGENT_TOKEN_BUDGET = 0
CHAT_ALLOW_LONG_RESPONSE_FOR_COMPLEX = True
CONTEXT_PREFIX_SIZE = 5
CONTEXT_RELEVANCE_MIN_QUERY_TOKENS = 1
CONTEXT_RELEVANCE_SAMPLE_LIMIT = 18
CONTEXT_RELEVANCE_THRESHOLD = 0.11
CONTEXT_TOKEN_BUDGET = 0
ENABLE_CONTEXT_RELEVANCE_GATE = True
GROUP_BACKGROUND_FETCH_MULTIPLIER = 3
GROUP_BACKGROUND_MIN_SCORE = 0.16
GROUP_BACKGROUND_RELEVANT_LIMIT = 3
HISTORY_RECALL_CANDIDATE_LIMIT = 60
HISTORY_RECALL_LIMIT = 4
HISTORY_RECALL_MIN_SCORE = 0.18
HISTORY_SELECTOR_CANDIDATE_LIMIT = 12
HISTORY_SELECTOR_ENABLED = True
HISTORY_SELECTOR_MIN_CANDIDATES = 8
HISTORY_SELECTOR_TIMEOUT = 6
MAX_REPLY_LAYERS = 3
ROUTE_CANDIDATE_EXPAND_STEP = 4
ROUTE_CANDIDATE_INITIAL_LIMIT = 10
ROUTE_CANDIDATE_MAX_LIMIT = 20
ROUTE_CANDIDATE_MIN_SCORE = 0.35
ROUTE_DEFERRED_NAMESPACE_ENABLED = True
ROUTE_OBSERVER_MAX_RECORDS = 400
ROUTE_PROMPT_TOKEN_BUDGET = 2600
ROUTE_TOOL_PLANNER_ENABLED = True
ROUTE_TOOL_PLANNER_MAX_TOOLS = 10
SESSION_CONTEXT_LIMIT = 20
USE_SIGN_IN_IMPRESSION = True

DEFAULTS = {
    "ENABLE_FALLBACK": True,
    "ENABLE_AGENT_MODE": True,
    "AGENT_MAX_TOOL_STEPS": 4,
    "AGENT_TOOL_FAILURE_LIMIT": 2,
    "AGENT_FAILED_ROUND_LIMIT": 2,
    "INTENT_TIMEOUT": 20,
    "CHAT_STYLE": "",
    "CUSTOM_PROMPT": "",
    "MCP_ENDPOINTS": "",
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

    if key in {"ENABLE_FALLBACK", "ENABLE_AGENT_MODE"}:
        return _parse_bool(raw_value, bool(default))

    if key == "INTENT_TIMEOUT":
        try:
            timeout = int(raw_value)
        except (TypeError, ValueError):
            timeout = 0
        if timeout > 0:
            return timeout
        return _fallback_timeout()

    if key in {"CHAT_STYLE", "CUSTOM_PROMPT", "MCP_ENDPOINTS"}:
        return str(raw_value or "").strip()

    if key == "REASONING_EFFORT":
        return _normalize_reasoning_effort(raw_value)

    if key in {
        "AGENT_MAX_TOOL_STEPS",
        "AGENT_TOOL_FAILURE_LIMIT",
        "AGENT_FAILED_ROUND_LIMIT",
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
