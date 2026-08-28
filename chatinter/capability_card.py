"""Capability cards for command-level tool policy.

CapabilityCard is the durable semantic layer between raw NoneBot command
metadata and runtime policy.  It deliberately avoids plugin-name decisions:
callability, risk, output and soft-tool behavior are derived from schema
properties, command roles, slots, requirements and metadata text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .route_text import normalize_message_text

CapabilityKind = Literal["plugin_command", "catalog", "superuser_tool", "runtime_tool"]
CapabilityOutputMode = Literal["text", "image", "file", "plugin_output", "action"]
CapabilitySideEffect = Literal["none", "query", "send", "mutate"]
CapabilityRiskLevel = Literal["low", "medium", "high"]
CapabilitySourceOfTruth = Literal[
    "model_knowledge",
    "plugin_runtime",
    "bot_state",
    "external_service",
    "local_state",
    "user_provided",
    "unknown",
]
CapabilityEntityScope = Literal[
    "none",
    "self_bot",
    "actor_user",
    "target_user",
    "group",
    "global",
    "external",
]
CapabilityExecutionPolicy = Literal[
    "normal",
    "explicit_only",
    "strong_intent",
    "confirmation_required",
]

_DESTRUCTIVE_TERMS = (
    "删除",
    "清空",
    "禁用",
    "关闭",
    "重启",
    "封禁",
    "拉黑",
    "退群",
    "踢",
    "解绑",
    "移除",
    "delete",
    "remove",
    "ban",
    "disable",
    "restart",
)
_CURRENCY_TERMS = ("金币", "余额", "花费", "消耗", "红包", "抽奖", "购买", "积分")
_FILE_TERMS = ("文件", "日志", "导出", "下载", "上传", "file", "log")
_IMAGE_TERMS = ("图片", "图", "照片", "头像", "表情", "image", "meme")
_LOOKUP_TERMS = (
    "解释",
    "展开",
    "缩写",
    "简称",
    "黑话",
    "术语",
    "翻译",
    "转换",
    "转成",
    "识别",
    "解析",
)
_QUERY_TERMS = (
    "查询",
    "查看",
    "搜索",
    "详情",
    "识别",
    "解析",
    "排行",
    "统计",
    "翻译",
    "帮助",
    *_LOOKUP_TERMS,
)
_MUTATE_TERMS = ("添加", "新增", "创建", "设置", "绑定", "修改", "删除", "退回")
_GENERATE_TERMS = (
    "生成",
    "制作",
    "绘制",
    "画",
    "随机",
    "抽",
    "roll",
)
_RANDOM_TERMS = ("随机", "抽", "roll", "掷", "投", "选择", "决定")
_TRANSFORM_TERMS = ("翻译", "转换", "转成", "识别", "解析", "压缩", *_LOOKUP_TERMS)
_STATUS_TERMS = ("信息", "状态", "记录", "排行", "统计", "余额", "积分", "签到", "打卡")
_HELP_TERMS = ("帮助", "说明", "用法", "教程", "参数", "示例", "文档", "列表", "有哪些")
_SEND_TERMS = ("发送", "发", "推送", "通知")
_PLAY_TERMS = (
    "播放",
    "点播",
    "点歌",
    "点一首",
    "点首",
    "播一首",
    "播首",
    "放一首",
    "放首",
    "听歌",
    "play",
)
_BOT_STATE_TERMS = (
    "机器人",
    "bot",
    "真寻",
    "小真寻",
    "项目",
    "版本",
    "作者",
    "仓库",
    "开源",
    "文档",
    "简介",
)
_MODEL_KNOWLEDGE_TERMS = (
    "概念",
    "理论",
    "原理",
    "为什么",
    "怎么看",
    "建议",
    "分析",
    "写作",
    "闲聊",
    "聊天",
)
_PLUGIN_RUNTIME_TERMS = (
    "插件",
    "命令",
    "模板",
    "语录",
    "一言",
    "热评",
    "鸡汤",
    "签到",
    "抽卡",
    "塔罗",
    "词云",
    "封面",
    "表情",
    "缩写",
    "简称",
    "黑话",
)
_EXTERNAL_TERMS = (
    "搜索",
    "识别",
    "解析",
    "翻译",
    "下载",
    "上传",
    "链接",
    "url",
    "http",
    "bv",
    "av",
    "api",
)
_GROUP_SCOPE_TERMS = ("群", "排行", "统计", "词云", "成员", "群友", "全体")


@dataclass(frozen=True)
class CapabilityCard:
    """A compact semantic card for one executable command capability."""

    command_id: str
    plugin_module: str
    plugin_name: str
    family: str
    description: str = ""
    use_cases: tuple[str, ...] = ()
    anti_use_cases: tuple[str, ...] = ()
    task_verbs: tuple[str, ...] = ()
    input_requirements: tuple[str, ...] = ()
    output_mode: CapabilityOutputMode = "plugin_output"
    side_effect: CapabilitySideEffect = "send"
    risk_level: CapabilityRiskLevel = "low"
    risk: CapabilityRiskLevel = "low"
    source_of_truth: CapabilitySourceOfTruth = "plugin_runtime"
    requires_real_tool: bool = True
    entity_scope: CapabilityEntityScope = "global"
    reliability: float = 0.5
    schema_quality: float = 0.5
    soft_tool: bool = False
    intent_types: tuple[str, ...] = ()
    requires_real_result: bool = True
    generative: bool = False
    execution_policy: CapabilityExecutionPolicy = "normal"
    risk_tags: tuple[str, ...] = ()
    permission_tags: tuple[str, ...] = ("public",)
    search_text: str = ""
    capability_kind: CapabilityKind = "plugin_command"
    tool_name: str = ""
    available: bool = True
    availability_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command_id": self.command_id,
            "capability_kind": self.capability_kind,
            "tool_name": self.tool_name,
            "family": self.family,
            "description": self.description,
            "use_cases": list(self.use_cases),
            "anti_use_cases": list(self.anti_use_cases),
            "task_verbs": list(self.task_verbs),
            "input_requirements": list(self.input_requirements),
            "output_mode": self.output_mode,
            "side_effect": self.side_effect,
            "risk_level": self.risk_level,
            "risk": self.risk,
            "source_of_truth": self.source_of_truth,
            "requires_real_tool": self.requires_real_tool,
            "entity_scope": self.entity_scope,
            "reliability": round(self.reliability, 3),
            "schema_quality": round(self.schema_quality, 3),
            "soft_tool": self.soft_tool,
            "intent_types": list(self.intent_types),
            "requires_real_result": self.requires_real_result,
            "generative": self.generative,
            "execution_policy": self.execution_policy,
            "risk_tags": list(self.risk_tags),
            "permission_tags": list(self.permission_tags),
            "available": self.available,
            "availability_reason": self.availability_reason,
        }
        return {
            key: value
            for key, value in payload.items()
            if value not in ("", [], {}, None)
        }


def build_capability_card(
    *,
    tool: Any,
    family: str,
    description: str = "",
    semantic_text: str | None = None,
) -> CapabilityCard:
    """Build a capability card from a command snapshot-like object."""

    search_text = _search_text(tool, description=description)
    text = normalize_message_text(semantic_text or "") or search_text
    task_verbs = _tuple_texts(getattr(tool, "task_verbs", None) or ())
    input_requirements = _tuple_texts(getattr(tool, "input_requirements", None) or ())
    output_mode = _infer_output_mode(
        tool, text=text, input_requirements=input_requirements
    )
    side_effect = _infer_side_effect(tool, text=text, output_mode=output_mode)
    risk_tags = _risk_tags(
        tool, text=text, output_mode=output_mode, side_effect=side_effect
    )
    permission_tags = _permission_tags(tool, text=text)
    risk_level = _risk_level(risk_tags=risk_tags, side_effect=side_effect)
    intent_types = _infer_intent_types(
        tool,
        text=text,
        task_verbs=task_verbs,
        output_mode=output_mode,
        side_effect=side_effect,
    )
    soft_tool = _is_soft_tool(
        tool,
        output_mode=output_mode,
        side_effect=side_effect,
        risk_level=risk_level,
    )
    generative = _is_generative(
        tool,
        text=text,
        intent_types=intent_types,
        output_mode=output_mode,
    )
    requires_real_result = _requires_real_result(
        intent_types=intent_types,
        output_mode=output_mode,
        side_effect=side_effect,
    )
    source_of_truth = _source_of_truth(
        tool,
        text=text,
        output_mode=output_mode,
        side_effect=side_effect,
        intent_types=intent_types,
    )
    requires_real_tool = _requires_real_tool(
        source_of_truth=source_of_truth,
        requires_real_result=requires_real_result,
        output_mode=output_mode,
        side_effect=side_effect,
        intent_types=intent_types,
    )
    entity_scope = _entity_scope(
        tool,
        text=text,
        source_of_truth=source_of_truth,
        intent_types=intent_types,
    )
    reliability = _initial_reliability(tool)
    schema_quality = _schema_quality(tool)
    execution_policy = _execution_policy(
        tool,
        soft_tool=soft_tool,
        generative=generative,
        side_effect=side_effect,
        risk_level=risk_level,
    )
    use_cases = _use_cases(
        tool,
        task_verbs=task_verbs,
        input_requirements=input_requirements,
        intent_types=intent_types,
        output_mode=output_mode,
        side_effect=side_effect,
    )
    anti_use_cases = _anti_use_cases(
        soft_tool=soft_tool,
        generative=generative,
        execution_policy=execution_policy,
    )
    return CapabilityCard(
        command_id=normalize_message_text(str(getattr(tool, "command_id", "") or "")),
        plugin_module=normalize_message_text(
            str(getattr(tool, "plugin_module", "") or "")
        ),
        plugin_name=normalize_message_text(str(getattr(tool, "plugin_name", "") or "")),
        family=normalize_message_text(family or "general") or "general",
        description=normalize_message_text(description),
        use_cases=use_cases,
        anti_use_cases=anti_use_cases,
        task_verbs=task_verbs,
        input_requirements=input_requirements,
        output_mode=output_mode,
        side_effect=side_effect,
        risk_level=risk_level,
        risk=risk_level,
        source_of_truth=source_of_truth,
        requires_real_tool=requires_real_tool,
        entity_scope=entity_scope,
        reliability=reliability,
        schema_quality=schema_quality,
        soft_tool=soft_tool,
        intent_types=intent_types,
        requires_real_result=requires_real_result,
        generative=generative,
        execution_policy=execution_policy,
        risk_tags=risk_tags,
        permission_tags=permission_tags,
        search_text=search_text,
    )


def _tuple_texts(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or ():
        text = normalize_message_text(str(value or ""))
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _search_text(tool: Any, *, description: str) -> str:
    parts = [
        getattr(tool, "plugin_name", ""),
        getattr(tool, "head", ""),
        " ".join(getattr(tool, "aliases", None) or []),
        description,
        getattr(tool, "usage", "") or "",
        " ".join(getattr(tool, "examples", None) or []),
        " ".join(getattr(tool, "retrieval_phrases", None) or []),
        getattr(tool, "capability_text", "") or "",
        " ".join(getattr(tool, "task_verbs", None) or []),
        " ".join(getattr(tool, "input_requirements", None) or []),
        " ".join(
            " ".join(
                [
                    str(getattr(slot, "name", "") or ""),
                    str(getattr(slot, "description", "") or ""),
                    " ".join(getattr(slot, "aliases", None) or []),
                ]
            )
            for slot in (getattr(tool, "slots", None) or [])
        ),
        getattr(tool, "command_role", "") or "",
        getattr(tool, "payload_policy", "") or "",
    ]
    return normalize_message_text(" ".join(str(part or "") for part in parts if part))


def _infer_output_mode(
    tool: Any,
    *,
    text: str,
    input_requirements: tuple[str, ...],
) -> CapabilityOutputMode:
    payload_policy = normalize_message_text(
        str(getattr(tool, "payload_policy", "") or "")
    )
    role = normalize_message_text(str(getattr(tool, "command_role", "") or ""))
    requirements = " ".join(input_requirements)
    lowered = text.casefold()
    explicit_image_output = any(
        phrase in lowered
        for phrase in (
            "输出图片",
            "返回图片",
            "生成图片",
            "制作图片",
            "生成表情",
            "制作表情",
            "做表情",
            "表情包",
        )
    )
    media_generation = any(
        term.casefold() in lowered for term in _GENERATE_TERMS
    ) and any(term.casefold() in lowered for term in _IMAGE_TERMS)
    if role == "template" or explicit_image_output or media_generation:
        return "image"
    if any(term.casefold() in lowered for term in _FILE_TERMS):
        return "file"
    if role in {"helper", "usage", "catalog"}:
        return "text"
    if payload_policy in {"image_only", "text_or_image"}:
        return "plugin_output"
    if "图片" in requirements or "image" in requirements.casefold():
        return "plugin_output"
    return "plugin_output"


def _infer_side_effect(
    tool: Any,
    *,
    text: str,
    output_mode: CapabilityOutputMode,
) -> CapabilitySideEffect:
    role = normalize_message_text(str(getattr(tool, "command_role", "") or ""))
    lowered = text.casefold()
    if any(term.casefold() in lowered for term in _DESTRUCTIVE_TERMS):
        return "mutate"
    if any(term.casefold() in lowered for term in _MUTATE_TERMS):
        return "mutate"
    if role in {"helper", "usage", "catalog"}:
        return "query"
    if any(term.casefold() in lowered for term in _QUERY_TERMS):
        return "query"
    if output_mode in {"image", "plugin_output", "file"}:
        return "send"
    return "none"


def _infer_intent_types(
    tool: Any,
    *,
    text: str,
    task_verbs: tuple[str, ...],
    output_mode: CapabilityOutputMode,
    side_effect: CapabilitySideEffect,
) -> tuple[str, ...]:
    """Infer generic capability intent types from schema, not plugin names."""

    role = normalize_message_text(str(getattr(tool, "command_role", "") or ""))
    lowered = text.casefold()
    joined_verbs = " ".join(task_verbs)
    intents: list[str] = []

    def add(value: str) -> None:
        if value and value not in intents:
            intents.append(value)

    if role in {"helper", "usage", "catalog"} or "帮助" in task_verbs:
        add("help")
    if role == "random" or any(term.casefold() in lowered for term in _RANDOM_TERMS):
        add("random")
    if role == "template" or any(
        term.casefold() in lowered for term in _GENERATE_TERMS
    ):
        add("generate")
    if output_mode in {"image", "file"} or any(
        term.casefold() in lowered for term in _IMAGE_TERMS + _FILE_TERMS
    ):
        add("media")
    if (
        any(term.casefold() in lowered for term in _TRANSFORM_TERMS)
        or "翻译" in joined_verbs
    ):
        add("transform")
    if side_effect == "query" or any(
        term.casefold() in lowered for term in _QUERY_TERMS
    ):
        add("query")
    if any(term.casefold() in lowered for term in _STATUS_TERMS):
        add("status")
    if side_effect == "mutate":
        add("mutate")
    if side_effect == "send" or any(term.casefold() in lowered for term in _SEND_TERMS):
        add("send")
    if any(term.casefold() in lowered for term in _PLAY_TERMS) or any(
        term in joined_verbs for term in ("播放", "点歌", "点播")
    ):
        add("play")
    if not intents:
        add("execute")
    return tuple(intents)


def _is_generative(
    tool: Any,
    *,
    text: str,
    intent_types: tuple[str, ...],
    output_mode: CapabilityOutputMode,
) -> bool:
    role = normalize_message_text(str(getattr(tool, "command_role", "") or ""))
    if role in {"template", "random"}:
        return True
    if output_mode == "image":
        return True
    if any(intent in intent_types for intent in ("generate", "random")):
        return True
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in _GENERATE_TERMS)


def _requires_real_result(
    *,
    intent_types: tuple[str, ...],
    output_mode: CapabilityOutputMode,
    side_effect: CapabilitySideEffect,
) -> bool:
    if output_mode in {"image", "file", "plugin_output", "action"}:
        return True
    if side_effect in {"query", "send", "mutate"}:
        return True
    return any(
        intent in intent_types
        for intent in {
            "query",
            "generate",
            "media",
            "status",
            "random",
            "transform",
            "mutate",
            "send",
            "play",
            "help",
        }
    )


def _requires_real_tool(
    *,
    source_of_truth: CapabilitySourceOfTruth,
    requires_real_result: bool,
    output_mode: CapabilityOutputMode,
    side_effect: CapabilitySideEffect,
    intent_types: tuple[str, ...],
) -> bool:
    if not requires_real_result:
        return False
    if source_of_truth in {
        "plugin_runtime",
        "bot_state",
        "external_service",
        "local_state",
    }:
        return True
    if output_mode in {"image", "file", "plugin_output", "action"}:
        return True
    if side_effect in {"query", "send", "mutate"}:
        return True
    return any(
        intent in intent_types
        for intent in {
            "query",
            "status",
            "generate",
            "media",
            "random",
            "transform",
            "play",
        }
    )


def _source_of_truth(
    tool: Any,
    *,
    text: str,
    output_mode: CapabilityOutputMode,
    side_effect: CapabilitySideEffect,
    intent_types: tuple[str, ...],
) -> CapabilitySourceOfTruth:
    explicit = normalize_message_text(str(getattr(tool, "source_of_truth", "") or ""))
    if explicit in {
        "model_knowledge",
        "plugin_runtime",
        "bot_state",
        "external_service",
        "local_state",
        "user_provided",
        "unknown",
    }:
        return explicit  # type: ignore[return-value]

    lowered = text.casefold()
    if any(term.casefold() in lowered for term in _MODEL_KNOWLEDGE_TERMS) and (
        output_mode == "text"
        and side_effect == "none"
        and not {"query", "status", "random", "generate", "media", "transform"}
        & set(intent_types)
    ):
        return "model_knowledge"
    if any(term.casefold() in lowered for term in _BOT_STATE_TERMS) and (
        {"help", "status", "query"} & set(intent_types) or side_effect == "query"
    ):
        return "bot_state"
    if side_effect == "mutate":
        return "local_state"
    if any(term.casefold() in lowered for term in _EXTERNAL_TERMS):
        return "external_service"
    if any(term.casefold() in lowered for term in _PLUGIN_RUNTIME_TERMS):
        return "plugin_runtime"
    if output_mode in {"image", "file", "plugin_output", "action"}:
        return "plugin_runtime"
    if side_effect in {"query", "send"}:
        return "plugin_runtime"
    role = normalize_message_text(str(getattr(tool, "command_role", "") or ""))
    if role in {"helper", "usage", "catalog"}:
        return "plugin_runtime"
    return "model_knowledge"


def _entity_scope(
    tool: Any,
    *,
    text: str,
    source_of_truth: CapabilitySourceOfTruth,
    intent_types: tuple[str, ...],
) -> CapabilityEntityScope:
    explicit = normalize_message_text(str(getattr(tool, "entity_scope", "") or ""))
    if explicit in {
        "none",
        "self_bot",
        "actor_user",
        "target_user",
        "group",
        "global",
        "external",
    }:
        return explicit  # type: ignore[return-value]
    if source_of_truth == "bot_state":
        return "self_bot"
    if source_of_truth == "external_service":
        return "external"
    if normalize_message_text(str(getattr(tool, "target_requirement", "") or "")) in {
        "optional",
        "required",
    } or bool(getattr(tool, "allow_at", False)):
        return "target_user"
    requires = dict(getattr(tool, "requires", {}) or {})
    if requires.get("private"):
        return "actor_user"
    lowered = text.casefold()
    if any(term.casefold() in lowered for term in _GROUP_SCOPE_TERMS):
        return "group"
    if "status" in intent_types and not requires.get("at"):
        return "actor_user"
    return "global"


def _initial_reliability(tool: Any) -> float:
    explicit = getattr(tool, "reliability", None)
    if explicit is not None:
        return _clamp01(explicit)
    source = normalize_message_text(str(getattr(tool, "source", "") or ""))
    confidence = _clamp01(getattr(tool, "confidence", 0.5))
    source_bonus = {
        "explicit": 0.2,
        "override": 0.18,
        "metadata": 0.12,
        "matcher": 0.02,
        "fallback": -0.12,
    }.get(source, 0.0)
    score = 0.35 + confidence * 0.45 + source_bonus
    if normalize_message_text(str(getattr(tool, "source_signature", "") or "")):
        score += 0.03
    if getattr(tool, "retrieval_phrases", None):
        score += 0.03
    if getattr(tool, "capability_text", None):
        score += 0.02
    return _clamp01(score)


def _schema_quality(tool: Any) -> float:
    explicit = getattr(tool, "schema_quality", None)
    if explicit is not None:
        return _clamp01(explicit)
    score = 0.0
    if normalize_message_text(str(getattr(tool, "command_id", "") or "")):
        score += 0.1
    if normalize_message_text(str(getattr(tool, "head", "") or "")):
        score += 0.14
    if normalize_message_text(str(getattr(tool, "description", "") or "")):
        score += 0.14
    if normalize_message_text(str(getattr(tool, "render", "") or "")):
        score += 0.14
    if getattr(tool, "aliases", None):
        score += 0.08
    if getattr(tool, "retrieval_phrases", None):
        score += 0.08
    if getattr(tool, "capability_text", None):
        score += 0.08
    if getattr(tool, "use_cases", None):
        score += 0.06
    if getattr(tool, "anti_use_cases", None):
        score += 0.05
    slots = list(getattr(tool, "slots", None) or [])
    if slots:
        described = sum(
            1
            for slot in slots
            if normalize_message_text(str(getattr(slot, "description", "") or ""))
            or getattr(slot, "aliases", None)
        )
        score += 0.07 + min(described / max(len(slots), 1), 1.0) * 0.06
    requires = dict(getattr(tool, "requires", {}) or {})
    if any(bool(requires.get(key)) for key in ("text", "image", "reply", "at")):
        score += 0.04
    source = normalize_message_text(str(getattr(tool, "source", "") or ""))
    if source in {"explicit", "override", "metadata"}:
        score += 0.08
    score += _clamp01(getattr(tool, "confidence", 0.5)) * 0.08
    return _clamp01(score)


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(number, 1.0))


def _execution_policy(
    tool: Any,
    *,
    soft_tool: bool,
    generative: bool,
    side_effect: CapabilitySideEffect,
    risk_level: CapabilityRiskLevel,
) -> CapabilityExecutionPolicy:
    if risk_level == "high" or side_effect == "mutate":
        return "confirmation_required"
    if soft_tool:
        return "explicit_only"
    if generative and not _has_required_user_input(tool):
        return "explicit_only"
    if risk_level == "medium":
        return "strong_intent"
    return "normal"


def _is_soft_tool(
    tool: Any,
    *,
    output_mode: CapabilityOutputMode,
    side_effect: CapabilitySideEffect,
    risk_level: CapabilityRiskLevel,
) -> bool:
    if risk_level != "low":
        return False
    if output_mode not in {"text", "plugin_output"}:
        return False
    if side_effect not in {"none", "query", "send"}:
        return False
    if _has_required_user_input(tool):
        return False
    role = normalize_message_text(str(getattr(tool, "command_role", "") or ""))
    return role in {"", "execute", "helper", "random"} or side_effect in {
        "none",
        "query",
    }


def _has_required_user_input(tool: Any) -> bool:
    requires = dict(getattr(tool, "requires", {}) or {})
    if any(
        bool(requires.get(key))
        for key in ("text", "image", "reply", "at", "private", "to_me")
    ):
        return True
    payload_policy = normalize_message_text(
        str(getattr(tool, "payload_policy", "") or "")
    )
    if payload_policy not in {"", "none"}:
        return True
    slots = list(getattr(tool, "slots", None) or [])
    if any(bool(getattr(slot, "required", False)) for slot in slots):
        return True
    target_requirement = normalize_message_text(
        str(getattr(tool, "target_requirement", "") or "")
    )
    if target_requirement == "required":
        return True
    return bool(getattr(tool, "allow_at", False))


def _risk_tags(
    tool: Any,
    *,
    text: str,
    output_mode: CapabilityOutputMode,
    side_effect: CapabilitySideEffect,
) -> tuple[str, ...]:
    lowered = text.casefold()
    tags: list[str] = []
    if any(term.casefold() in lowered for term in _DESTRUCTIVE_TERMS):
        tags.append("destructive")
    if any(term.casefold() in lowered for term in _CURRENCY_TERMS):
        tags.append("cost_or_currency")
    if output_mode in {"image", "file"}:
        tags.append("media" if output_mode == "image" else "file")
    if side_effect == "mutate":
        tags.append("state_mutation")
    if normalize_message_text(
        str(getattr(tool, "target_requirement", "") or "")
    ) != "none" or bool(getattr(tool, "allow_at", False)):
        tags.append("targeted")
    return tuple(dict.fromkeys(tags))


def _permission_tags(tool: Any, *, text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    tags = ["public"]
    if any(
        term in lowered for term in ("admin", "superuser", "管理员", "超级用户", "群管")
    ):
        tags.append("sensitive_permission")
    requires = dict(getattr(tool, "requires", {}) or {})
    if requires.get("private"):
        tags.append("private_only")
    if requires.get("to_me"):
        tags.append("addressed_to_bot")
    return tuple(tags)


def _risk_level(
    *,
    risk_tags: tuple[str, ...],
    side_effect: CapabilitySideEffect,
) -> CapabilityRiskLevel:
    if "destructive" in risk_tags or side_effect == "mutate":
        return "high"
    if "cost_or_currency" in risk_tags or "file" in risk_tags:
        return "medium"
    return "low"


def _use_cases(
    tool: Any,
    *,
    task_verbs: tuple[str, ...],
    input_requirements: tuple[str, ...],
    intent_types: tuple[str, ...],
    output_mode: CapabilityOutputMode,
    side_effect: CapabilitySideEffect,
) -> tuple[str, ...]:
    cases: list[str] = []
    head = normalize_message_text(str(getattr(tool, "head", "") or ""))
    description = normalize_message_text(str(getattr(tool, "description", "") or ""))
    if description:
        cases.append(description)
    elif head:
        cases.append(f"用户明确请求执行“{head}”能力")
    if task_verbs:
        cases.append("动作: " + "/".join(task_verbs))
    if input_requirements:
        cases.append("输入: " + "/".join(input_requirements))
    if intent_types:
        cases.append("能力类型: " + "/".join(intent_types))
    cases.append(f"输出: {output_mode}; 副作用: {side_effect}")
    return tuple(dict.fromkeys(cases))


def _anti_use_cases(
    *,
    soft_tool: bool,
    generative: bool,
    execution_policy: CapabilityExecutionPolicy,
) -> tuple[str, ...]:
    del soft_tool, execution_policy
    cases: list[str] = []
    if generative:
        cases.append("没有明确生成、制作、随机或发送请求时不要调用")
    return tuple(dict.fromkeys(cases))


__all__ = [
    "CapabilityCard",
    "CapabilityEntityScope",
    "CapabilityExecutionPolicy",
    "CapabilityKind",
    "CapabilityOutputMode",
    "CapabilityRiskLevel",
    "CapabilitySideEffect",
    "CapabilitySourceOfTruth",
    "build_capability_card",
]
