from typing import Any, Literal

from zhenxun.configs.config import Config
from zhenxun.configs.utils import RegisterConfig

from .llm_compat import (
    LLMGenerationConfig,
    ReasoningConfig,
    ToolConfig,
)

CHATINTER_GROUP = "chatinter"

INTENT_TIMEOUT_SECONDS = 20
CHAT_RESPONSE_TIMEOUT_SECONDS = 120
NATIVE_REROUTE_TIMEOUT_SECONDS = 10
AgentRole = Literal["chat", "plugin", "superuser"]
DEFAULT_MODELS_SOURCE = "DEFAULT_MODELS"
_REASONING_EFFORTS = frozenset(
    {"DEFAULT", "NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH", "MAX"}
)
_DEFAULT_AGENTS: dict[AgentRole, dict[str, Any]] = {
    "chat": {
        "model": DEFAULT_MODELS_SOURCE,
        "context_window_tokens": 64_000,
        "max_output_tokens": 12_000,
        "reasoning_effort": "MEDIUM",
    },
    "plugin": {
        "model": DEFAULT_MODELS_SOURCE,
        "context_window_tokens": 16_000,
        "max_output_tokens": 2_048,
        "reasoning_effort": "MEDIUM",
    },
    "superuser": {
        "model": DEFAULT_MODELS_SOURCE,
        "context_window_tokens": 200_000,
        "max_output_tokens": 32_000,
        "reasoning_effort": "HIGH",
    },
}
_DEFAULT_PERMISSIONS = {
    "preset": "python",
    "default_mode": "ask",
    "dangerous_policy": "ask",
}

CHATINTER_REGISTER_CONFIGS = (
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="ENABLED",
        value=True,
        help="是否启用 ChatInter",
        default_value=True,
        type=bool,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="AGENTS",
        value={key: dict(value) for key, value in _DEFAULT_AGENTS.items()},
        help=(
            "聊天、插件调用和 Superuser Agent 的模型、输入窗口、输出上限和思考等级；"
            "DEFAULT_MODELS 表示读取 AI.DEFAULT_MODELS.chat"
        ),
        default_value={key: dict(value) for key, value in _DEFAULT_AGENTS.items()},
        type=dict,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="FALLBACK_MODELS",
        value=[],
        help="主模型失败时按顺序尝试的降级模型列表",
        default_value=[],
        type=list[str],
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="PERMISSIONS",
        value=dict(_DEFAULT_PERMISSIONS),
        help="Superuser Agent 权限模式与命令护栏配置",
        default_value=dict(_DEFAULT_PERMISSIONS),
        type=dict,
    ),
)


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



AGENT_STEP_BUDGETS: dict[str, dict[str, int]] = {
    "superuser_agent": {
        "chat": 3,
        "standard": 90,
    },
    "group_plugin_selector": {
        "chat": 6,
        "standard": 10,
    },
    "private_chat": {
        "chat": 5,
        "standard": 8,
    },
}
AGENT_COST_CHECKPOINT_TOKENS: dict[str, int] = {
    "superuser_agent": 32_000,
    "group_plugin_selector": 80_000,
    "private_chat": 60_000,
}


EXPOSE_FALLBACK_SCHEMAS = False
SCHEMA_FALLBACK_ALLOWLIST: frozenset[str] = frozenset()




COMMAND_TWO_STAGE_THRESHOLD = 30
COMMAND_INITIAL_EXPOSURE_CAP = 40
COMMAND_TWO_STAGE_PLUGIN_CAP = 4



MEMORY_VECTOR_MAX_ITEMS = 0

SUPERUSER_MODEL_TIMEOUT_SECONDS = 120.0


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_reasoning_effort(value: Any, *, default: str = "DEFAULT") -> str:
    text = str(value or "").strip().upper()
    return text if text in _REASONING_EFFORTS else default


def _agent_settings(role: AgentRole) -> dict[str, Any]:
    defaults = _DEFAULT_AGENTS[role]
    raw = Config.get_config(CHATINTER_GROUP, "AGENTS", None)
    if isinstance(raw, dict):
        configured = raw.get(role)
        if isinstance(configured, dict):
            return {**defaults, **configured}



    if role == "superuser":
        legacy_context = Config.get_config(
            CHATINTER_GROUP,
            "SUPERUSER_CONTEXT_WINDOW_TOKENS",
            defaults["context_window_tokens"],
        )
        legacy_reasoning = Config.get_config(
            CHATINTER_GROUP,
            "REASONING_EFFORT",
            defaults["reasoning_effort"],
        )
        return {
            **defaults,
            "context_window_tokens": legacy_context,
            "reasoning_effort": legacy_reasoning,
        }
    return dict(defaults)


def get_agent_model(role: AgentRole) -> str:
    configured = str(_agent_settings(role).get("model") or "").strip()
    source = configured or str(_DEFAULT_AGENTS[role]["model"])
    if source.upper() != DEFAULT_MODELS_SOURCE:
        return source

    from zhenxun.services.ai.llm.manager import get_default_model

    model_name = str(get_default_model("chat") or "").strip()
    if not model_name:
        raise RuntimeError("AI.DEFAULT_MODELS.chat 未配置")
    return model_name


def chatinter_enabled() -> bool:
    raw = Config.get_config(CHATINTER_GROUP, "ENABLED", True)
    return _parse_bool(raw, True)


def get_fallback_models(primary_model: str | None = None) -> tuple[str, ...]:
    raw = Config.get_config(CHATINTER_GROUP, "FALLBACK_MODELS", [])
    values = raw if isinstance(raw, list | tuple | set) else str(raw or "").split(",")
    names = [str(part or "").strip() for part in values]
    primary = str(primary_model or get_agent_model("chat")).strip()
    return tuple(name for name in names if name and name != primary)


def get_agent_context_window_tokens(role: AgentRole) -> int:
    default = int(_DEFAULT_AGENTS[role]["context_window_tokens"])
    raw = _agent_settings(role).get("context_window_tokens", default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def get_superuser_context_window_tokens() -> int:
    """Compatibility alias for the persisted Superuser context policy."""

    return get_agent_context_window_tokens("superuser")


def get_agent_max_output_tokens(role: AgentRole) -> int:
    default = int(_DEFAULT_AGENTS[role]["max_output_tokens"])
    raw = _agent_settings(role).get("max_output_tokens", default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def resolve_agent_context_window_tokens(
    role: AgentRole,
    model_name: str | None = None,
) -> int:
    from zhenxun.services.ai.llm.system.capabilities import get_model_capabilities

    configured = get_agent_context_window_tokens(role)
    declared = int(
        get_model_capabilities(model_name or get_agent_model(role)).max_input_tokens
        or 0
    )
    return min(configured, declared) if declared > 0 else configured


def get_superuser_max_output_tokens() -> int:
    return get_agent_max_output_tokens("superuser")


def get_permission_policy() -> dict[str, Any]:
    raw = Config.get_config(CHATINTER_GROUP, "PERMISSIONS", _DEFAULT_PERMISSIONS)
    policy = dict(raw) if isinstance(raw, dict) else {}
    return {**_DEFAULT_PERMISSIONS, **policy}


def build_reasoning_generation_config(
    role: AgentRole = "superuser",
) -> LLMGenerationConfig | None:
    raw = _agent_settings(role).get("reasoning_effort", "")
    effort_text = _normalize_reasoning_effort(
        raw,
        default=str(_DEFAULT_AGENTS[role]["reasoning_effort"]),
    )
    if effort_text == "DEFAULT":
        return None
    return LLMGenerationConfig(
        reasoning=ReasoningConfig(
            effort=effort_text,
            show_thoughts=False,
        )
    )


def build_agent_generation_config(
    role: AgentRole,
    *,
    max_output_tokens: int | None = None,
) -> LLMGenerationConfig:
    config = build_reasoning_generation_config(role) or LLMGenerationConfig()
    output_tokens = (
        get_agent_max_output_tokens(role)
        if max_output_tokens is None
        else max(int(max_output_tokens), 1)
    )
    return config.model_copy(update={"max_tokens": output_tokens})


def build_superuser_generation_config() -> LLMGenerationConfig:
    return build_agent_generation_config("superuser")


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

    base = base or build_agent_generation_config("superuser")
    mode = "AUTO"
    if isinstance(tool_choice, dict):
        mode = "ANY"
    elif tool_choice == "required":
        mode = "ANY"
    elif tool_choice is None or tool_choice == "none":
        mode = "NONE"
    tool_config = LLMGenerationConfig(tool_config=ToolConfig(mode=mode))
    return base.merge_with(tool_config) if base is not None else tool_config
