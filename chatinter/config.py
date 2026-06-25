from typing import Any

from zhenxun.configs.config import Config
from zhenxun.services.llm.config.generation import (
    LLMGenerationConfig,
    ReasoningConfig,
    ReasoningEffort,
    ToolConfig,
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

# Agent 循环预算(P0-1):max_steps 按场景与复杂度动态解析;
# token 预算为整个 run 的累计估算 prompt token 上限(0 表示不限)。
AGENT_STEP_BUDGETS: dict[str, dict[str, int]] = {
    "superuser_agent": {
        "chat": 8,
        "readonly_fast": 4,
        "single_tool_fast": 12,
        "multi_tool_fast": 16,
        "complex_pev": 40,
    },
    "group_plugin_selector": {
        "chat": 6,
        "single_tool_fast": 8,
        "multi_tool_fast": 10,
        "complex_pev": 10,
    },
    "private_chat": {
        "chat": 5,
        "single_tool_fast": 6,
        "multi_tool_fast": 8,
        "complex_pev": 8,
    },
}
AGENT_TOKEN_BUDGETS: dict[str, int] = {
    "superuser_agent": 240_000,
    "group_plugin_selector": 80_000,
    "private_chat": 60_000,
}
# 读类工具步数退款上限系数:refund 总数 <= max_steps * 系数
AGENT_STEP_REFUND_RATIO = 0.5

# Schema 质量门(P0-4):fallback 档(头部命令反射失败后的兜底推断)
# 可信度过低,默认不暴露给 LLM 工具池;确需暴露的插件加入白名单。
EXPOSE_FALLBACK_SCHEMAS = False
SCHEMA_FALLBACK_ALLOWLIST: frozenset[str] = frozenset()

# P1-1 两阶段工具选择:候选池过大时首轮只暴露轻量 capability card,
# 由模型先选能力,再二次请求完整 schema。这样保留插件零改造调用能力,
# 同时避免一次性塞入大量完整参数 schema。
COMMAND_TWO_STAGE_THRESHOLD = 30
COMMAND_INITIAL_EXPOSURE_CAP = 40
COMMAND_TWO_STAGE_PLUGIN_CAP = 4

# P1-2 向量底座:0 表示不再硬限制记忆向量条数;如需压内存可设软上限,
# 淘汰顺序按重要度+LRU,不是旧版纯 oldest 1024 硬截断。
MEMORY_VECTOR_MAX_ITEMS = 0

DEFAULTS = {
    "ENABLE_FALLBACK": True,
    "INTENT_TIMEOUT": 20,
    "CHAT_STYLE": "",
    "CUSTOM_PROMPT": "",
    "PERSONA_FILE": "",
    "QUALITY_SHADOW_SAMPLE_RATE": 0.0,
    "REASONING_EFFORT": "MEDIUM",
    "FALLBACK_MODELS": "",
    "SUPERUSER_PERMISSION_MODE": "default",
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


def get_fallback_models() -> tuple[str, ...]:
    """P0-3:主模型失败后的降级模型链(chatinter.FALLBACK_MODELS,逗号分隔)。"""
    raw = Config.get_config(CHATINTER_GROUP, "FALLBACK_MODELS", "")
    names = [part.strip() for part in str(raw or "").split(",")]
    primary = get_model_name()
    return tuple(name for name in names if name and name != primary)


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

    if key == "QUALITY_SHADOW_SAMPLE_RATE":
        try:
            return max(0.0, min(float(raw_value or 0.0), 1.0))
        except (TypeError, ValueError):
            return float(default or 0.0)

    if key in {"CHAT_STYLE", "CUSTOM_PROMPT", "PERSONA_FILE"}:
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


def build_tool_generation_config(
    *,
    tool_choice: str | dict[str, Any] | None,
    base: LLMGenerationConfig | None = None,
) -> LLMGenerationConfig | None:
    """Build a per-request config whose tool mode matches runtime policy.

    Some providers/adapters infer a default tool mode from generation config.
    Keeping it explicit makes the request stable for cache-friendly payloads and
    avoids accidental extra tool forcing when runtime decides `none`/`auto`.
    """

    base = base or build_reasoning_generation_config()
    mode = "AUTO"
    if isinstance(tool_choice, dict):
        mode = "ANY"
    elif tool_choice == "required":
        mode = "ANY"
    elif tool_choice is None or tool_choice == "none":
        mode = "NONE"
    tool_config = LLMGenerationConfig(tool_config=ToolConfig(mode=mode))
    return base.merge_with(tool_config) if base is not None else tool_config
