import math
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
NATIVE_REROUTE_TIMEOUT_SECONDS = 60
AgentRole = Literal["chat", "plugin", "superuser"]
WebAccessMode = Literal["off", "agent", "all"]
ChatInterGroupMode = Literal["whitelist", "blacklist"]
ReplyDeliveryMode = Literal["streaming", "whole"]
ChatWebSearchProvider = Literal[
    "baidu",
    "bocha",
    "brave",
    "exa",
    "firecrawl",
    "tavily",
]
DEFAULT_MODELS_SOURCE = "DEFAULT_MODELS"
DEFAULT_CHAT_WEB_SEARCH_API_URL = "DEFAULT"
_REASONING_EFFORTS = frozenset(
    {"DEFAULT", "NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH", "MAX"}
)
_DEFAULT_AGENTS: dict[AgentRole, dict[str, Any]] = {
    "chat": {
        "model": DEFAULT_MODELS_SOURCE,
        "context_window_tokens": 128_000,
        "max_output_tokens": 8_192,
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
_DEFAULT_GROUP_ACCESS = {
    "mode": "blacklist",
    "enabled_groups": [],
    "disabled_groups": [],
}
_DEFAULT_GSCORE_BRIDGE = {
    "enabled": False,
    "url": "",
    "secret": "",
}
_DEFAULT_REPLY_DELIVERY = {
    "mode": "streaming",
    "max_chars": 3_500,
    "max_segments": 6,
    "interval_method": "random",
    "interval": "1.5,3.5",
    "log_base": 2.6,
}
_DEFAULT_REACTION_IMAGES = {
    "enabled": False,
    "directory": "data/chatinter/reactions",
    "import_directory": "data/chatinter/reaction_import",
    "semantic_search": True,
    "auto_caption": True,
    "auto_discovery": False,
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
        key="GROUP_ACCESS",
        value=dict(_DEFAULT_GROUP_ACCESS),
        help=(
            "群聊启用范围；mode 为 whitelist/blacklist，enabled_groups 和 "
            "disabled_groups 填写群号，冲突时禁用优先"
        ),
        default_value=dict(_DEFAULT_GROUP_ACCESS),
        type=dict,
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
        value=["", ""],
        help="主模型失败时按顺序尝试的降级模型列表",
        default_value=["", ""],
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
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="ACTIVE_TASKS_ENABLED",
        value=True,
        help="是否启用 Superuser Agent 主动任务、调度和 Webhook",
        default_value=True,
        type=bool,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="PRIVATE_PLUGIN_TOOLS",
        value=True,
        help="私聊是否启用插件调用（统一混合模式）",
        default_value=True,
        type=bool,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="REPLY_TO_TRIGGER_MESSAGE",
        value=False,
        help="ChatInter 最终回复是否引用本轮触发消息（仅首段）",
        default_value=False,
        type=bool,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="REPLY_DELIVERY",
        value=dict(_DEFAULT_REPLY_DELIVERY),
        help=(
            "回复投递设置；streaming 为完整生成后按句逐段发送，whole 为整段发送；"
            "interval_method 支持 random/log，interval 为随机等待秒数范围，"
            "log_base 为按下一段长度计算等待时使用的对数底数"
        ),
        default_value=dict(_DEFAULT_REPLY_DELIVERY),
        type=dict,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="REACTION_IMAGES",
        value=dict(_DEFAULT_REACTION_IMAGES),
        help=(
            "混合聊天本地表情能力；enabled 开启后注入语义发现与发送工具，"
            "directory 为表情库目录，import_directory 为启动时扫描的导入目录，"
            "semantic_search/auto_caption 控制语义索引，auto_discovery 控制群聊"
            "重复图片发现"
        ),
        default_value=dict(_DEFAULT_REACTION_IMAGES),
        type=dict,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="CHAT_HISTORY_LIMIT",
        value=100,
        help="混合聊天每个会话缓存的最近对话数量（1-1000）",
        default_value=100,
        type=int,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="MIXED_CHAT_SKIP_PREFIXES",
        value=["", ""],
        help="混合聊天消息以列表中任一关键词开头时静默跳过；仅支持列表，空列表表示不过滤",
        default_value=["", ""],
        type=list[str],
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="UNIFIED_MAX_TOOL_STEPS",
        value=4,
        help="统一混合模式单回合最多工具循环步数（1-8）",
        default_value=4,
        type=int,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="WEB_ACCESS_MODE",
        value="agent",
        help="只读联网能力范围：off=关闭，agent=仅 Superuser，all=全部聊天",
        default_value="agent",
        type=str,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="CHAT_WEB_SEARCH_ENABLED",
        value=True,
        help=(
            "混合聊天在模型无原生搜索时是否启用搜索 API 回退；" "需 WEB_ACCESS_MODE=all"
        ),
        default_value=True,
        type=bool,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="CHAT_WEB_SEARCH_PROVIDER",
        value="baidu",
        help="混合聊天搜索协议：baidu/tavily/bocha/brave/firecrawl/exa",
        default_value="baidu",
        type=str,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="CHAT_WEB_SEARCH_API_URL",
        value=DEFAULT_CHAT_WEB_SEARCH_API_URL,
        help="搜索 API 地址；DEFAULT 使用所选协议的官方端点",
        default_value=DEFAULT_CHAT_WEB_SEARCH_API_URL,
        type=str,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="CHAT_WEB_SEARCH_API_KEY",
        value="",
        help="混合聊天搜索 API Key；留空时不暴露本地搜索工具",
        default_value="",
        type=str,
    ),
    RegisterConfig(
        module=CHATINTER_GROUP,
        key="GSCORE_BRIDGE",
        value=dict(_DEFAULT_GSCORE_BRIDGE),
        help=(
            "GScore 原生命令协调与语义兜底；enabled 控制启用，url 留空时从 "
            "gsuid_core_host/port 推导，secret 与 ChatInterBridge 插件一致"
        ),
        default_value=dict(_DEFAULT_GSCORE_BRIDGE),
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


def _normalize_group_ids(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    elif isinstance(value, list | tuple | set | frozenset):
        values = value
    else:
        values = ()
    return frozenset(
        str(item or "").strip() for item in values if str(item or "").strip()
    )


def get_chatinter_group_access() -> (
    tuple[
        ChatInterGroupMode,
        frozenset[str],
        frozenset[str],
    ]
):
    raw = Config.get_config(CHATINTER_GROUP, "GROUP_ACCESS", _DEFAULT_GROUP_ACCESS)
    configured = raw if isinstance(raw, dict) else {}
    mode_value = str(configured.get("mode", "blacklist") or "").strip().casefold()
    mode: ChatInterGroupMode = "whitelist" if mode_value == "whitelist" else "blacklist"
    enabled = _normalize_group_ids(configured.get("enabled_groups", ()))
    disabled = _normalize_group_ids(configured.get("disabled_groups", ()))
    return mode, enabled, disabled


def chatinter_available(group_id: str | int | None = None) -> bool:
    if not chatinter_enabled():
        return False
    if group_id is None:
        return True
    normalized = str(group_id).strip()
    if not normalized:
        return True
    mode, enabled, disabled = get_chatinter_group_access()
    if normalized in disabled:
        return False
    if normalized in enabled:
        return True
    return mode == "blacklist"


def private_plugin_tools_enabled() -> bool:
    raw = Config.get_config(CHATINTER_GROUP, "PRIVATE_PLUGIN_TOOLS", True)
    return _parse_bool(raw, True)


def active_tasks_enabled() -> bool:
    if not chatinter_enabled():
        return False
    raw = Config.get_config(CHATINTER_GROUP, "ACTIVE_TASKS_ENABLED", True)
    return _parse_bool(raw, True)


def reply_to_trigger_message_enabled() -> bool:
    raw = Config.get_config(CHATINTER_GROUP, "REPLY_TO_TRIGGER_MESSAGE", False)
    return _parse_bool(raw, False)


def get_reply_delivery_settings() -> tuple[ReplyDeliveryMode, int, int]:
    raw = Config.get_config(
        CHATINTER_GROUP,
        "REPLY_DELIVERY",
        _DEFAULT_REPLY_DELIVERY,
    )
    configured = raw if isinstance(raw, dict) else {}
    mode_value = str(configured.get("mode") or "").strip().casefold()
    mode: ReplyDeliveryMode = "whole" if mode_value == "whole" else "streaming"
    try:
        max_chars = int(configured.get("max_chars", 3_500))
    except (TypeError, ValueError):
        max_chars = 3_500
    if max_chars <= 0:
        max_chars = 3_500
    max_chars = min(max(max_chars, 256), 3_500)
    try:
        max_segments = int(configured.get("max_segments", 6))
    except (TypeError, ValueError):
        max_segments = 6
    if max_segments < 0:
        max_segments = 6
    return mode, max_chars, max_segments


def get_reply_delivery_interval_settings() -> tuple[str, tuple[float, float], float]:
    raw = Config.get_config(
        CHATINTER_GROUP,
        "REPLY_DELIVERY",
        _DEFAULT_REPLY_DELIVERY,
    )
    configured = raw if isinstance(raw, dict) else {}
    method = (
        "log"
        if str(configured.get("interval_method") or "").strip().casefold() == "log"
        else "random"
    )
    interval_raw = configured.get("interval", _DEFAULT_REPLY_DELIVERY["interval"])
    if isinstance(interval_raw, str):
        interval_parts = interval_raw.replace(" ", "").split(",")
    elif isinstance(interval_raw, list | tuple):
        interval_parts = list(interval_raw)
    else:
        interval_parts = []
    try:
        interval_values = [float(value) for value in interval_parts]
    except (TypeError, ValueError):
        interval_values = []
    if len(interval_values) != 2 or not all(
        math.isfinite(value) and value >= 0 for value in interval_values
    ):
        interval = (1.5, 3.5)
    else:
        lower, upper = sorted(
            (min(interval_values[0], 10.0), min(interval_values[1], 10.0))
        )
        interval = (lower, upper)
    try:
        log_base = float(configured.get("log_base", 2.6))
    except (TypeError, ValueError):
        log_base = 2.6
    if not math.isfinite(log_base) or not 1.0 < log_base <= 10.0:
        log_base = 2.6
    return method, interval, log_base


def get_reaction_image_settings() -> dict[str, Any]:
    raw = Config.get_config(
        CHATINTER_GROUP,
        "REACTION_IMAGES",
        _DEFAULT_REACTION_IMAGES,
    )
    configured = raw if isinstance(raw, dict) else {}
    directory = str(
        configured.get("directory") or _DEFAULT_REACTION_IMAGES["directory"]
    ).strip()
    if not directory:
        directory = str(_DEFAULT_REACTION_IMAGES["directory"])
    import_directory = str(
        configured.get("import_directory")
        or _DEFAULT_REACTION_IMAGES["import_directory"]
    ).strip()
    if not import_directory:
        import_directory = str(_DEFAULT_REACTION_IMAGES["import_directory"])
    return {
        "enabled": _parse_bool(configured.get("enabled"), False),
        "directory": directory,
        "import_directory": import_directory,
        "semantic_search": _parse_bool(
            configured.get("semantic_search"),
            True,
        ),
        "auto_caption": _parse_bool(configured.get("auto_caption"), True),
        "auto_discovery": _parse_bool(
            configured.get(
                "auto_discovery",
                configured.get("auto_collect", False),
            ),
            False,
        ),
    }


def reaction_images_enabled() -> bool:
    return bool(get_reaction_image_settings()["enabled"])


def get_chat_history_limit() -> int:
    raw = Config.get_config(CHATINTER_GROUP, "CHAT_HISTORY_LIMIT", 100)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 100
    return min(value, 1000) if value > 0 else 100


def get_mixed_chat_skip_prefixes() -> tuple[str, ...]:
    raw = Config.get_config(CHATINTER_GROUP, "MIXED_CHAT_SKIP_PREFIXES", ["", ""])
    values = raw if isinstance(raw, list) else ()
    return tuple(
        dict.fromkeys(prefix for item in values if (prefix := str(item or "").strip()))
    )


def mixed_chat_message_should_skip(message: str) -> bool:
    normalized = str(message or "").strip()
    return bool(normalized) and any(
        normalized.startswith(prefix) for prefix in get_mixed_chat_skip_prefixes()
    )


def get_unified_max_tool_steps() -> int:
    raw = Config.get_config(CHATINTER_GROUP, "UNIFIED_MAX_TOOL_STEPS", 4)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 4
    return max(1, min(value, 8))


def get_web_access_mode() -> WebAccessMode:
    raw = str(Config.get_config(CHATINTER_GROUP, "WEB_ACCESS_MODE", "agent") or "")
    value = raw.strip().casefold()
    return value if value in {"off", "agent", "all"} else "agent"


def chat_web_search_enabled() -> bool:
    raw = Config.get_config(CHATINTER_GROUP, "CHAT_WEB_SEARCH_ENABLED", True)
    return _parse_bool(raw, True)


def get_chat_web_search_provider() -> ChatWebSearchProvider:
    raw = Config.get_config(CHATINTER_GROUP, "CHAT_WEB_SEARCH_PROVIDER", "baidu")
    value = str(raw or "").strip().casefold().replace("-", "_")
    aliases = {
        "baidu_ai_search": "baidu",
        "bocha_ai": "bocha",
    }
    value = aliases.get(value, value)
    if value in {"baidu", "bocha", "brave", "exa", "firecrawl", "tavily"}:
        return value
    return "baidu"


def get_chat_web_search_api_url() -> str:
    raw = Config.get_config(
        CHATINTER_GROUP,
        "CHAT_WEB_SEARCH_API_URL",
        DEFAULT_CHAT_WEB_SEARCH_API_URL,
    )
    return str(raw or "").strip()


def get_chat_web_search_api_key() -> str:
    raw = Config.get_config(CHATINTER_GROUP, "CHAT_WEB_SEARCH_API_KEY", "")
    return str(raw or "").strip()


def get_gscore_bridge_config() -> dict[str, Any]:
    raw = Config.get_config(CHATINTER_GROUP, "GSCORE_BRIDGE", _DEFAULT_GSCORE_BRIDGE)
    configured = raw if isinstance(raw, dict) else {}
    enabled = _parse_bool(configured.get("enabled"), False)
    url = str(configured.get("url") or "").strip()
    secret = str(configured.get("secret") or "").strip()
    if enabled and not url:
        try:
            from nonebot import get_driver

            driver_config = get_driver().config
            host = str(getattr(driver_config, "gsuid_core_host", "localhost") or "")
            port = str(getattr(driver_config, "gsuid_core_port", "8765") or "")
            scheme = (
                "https"
                if _parse_bool(
                    getattr(driver_config, "gsuid_core_https", False),
                    False,
                )
                else "http"
            )
            if host and port:
                url = f"{scheme}://{host}:{port}"
        except Exception:
            url = ""
    return {
        "enabled": enabled,
        "url": url.rstrip("/"),
        "secret": secret,
    }


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
    validation_policy = dict(config.validation_policy or {})
    validation_policy["chatinter_reasoning_transport_policy"] = "capability_gated"
    output_tokens = (
        get_agent_max_output_tokens(role)
        if max_output_tokens is None
        else max(int(max_output_tokens), 1)
    )
    return config.model_copy(
        update={
            "max_tokens": output_tokens,
            "validation_policy": validation_policy,
        }
    )


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
