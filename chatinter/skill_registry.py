from dataclasses import dataclass
import json
import re
from typing import Any

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .route_text import (
    ROUTE_ACTION_WORDS,
    collect_placeholders,
    contains_any,
    has_negative_route_intent,
    has_template_route_context,
    is_usage_question,
    match_command_head,
    match_command_head_or_sticky,
    parse_command_with_head,
    normalize_action_phrases,
    normalize_message_text,
    sanitize_template_tail,
    strip_invoke_prefix,
)

_CACHE: dict[str, "SkillRegistry"] = {}
_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_PLACEHOLDER_PATTERN = re.compile(
    r"\s*(?:\[[^\]]+\]|<[^>]+>|\{[^}]+\}|(?<![0-9A-Za-z_])@\d{5,20}(?=(?:\s|$|[的，,。.!！？?]))|\([^)]*(?:参数|类型|默认|可选|说明)[^)]*\))\s*"
)
_COMMAND_TAIL_PATTERN = re.compile(r"(?:\s+[?*+]+)+\s*$")
_COMMAND_INLINE_PATTERN = re.compile(r"\s+[?*+]+(?=\s|$)")
_HELPER_KEYWORDS = (
    "帮助",
    "说明",
    "详情",
    "用法",
    "教程",
    "配置",
    "参数",
    "示例",
    "例子",
    "搜索",
    "查询",
    "列表",
    "查看",
)
_EXPLICIT_HELPER_HINTS = (
    "怎么用",
    "如何用",
    "怎样用",
    "怎么使用",
    "如何使用",
    "怎样使用",
    "用法",
    "教程",
    "参数",
    "说明",
    "示例",
    "例子",
    "帮助",
    "详情",
    "怎么触发",
    "如何触发",
    "怎样触发",
    "怎么调用",
    "如何调用",
    "怎样调用",
)
_MESSAGE_CREATE_HINTS = (
    "发",
    "塞",
    "创建",
    "新增",
    "生成",
    "制作",
    "上传",
    "设置",
    "绑定",
    "添加",
    "开启",
)
_MESSAGE_OPEN_HINTS = (
    "开",
    "抢",
    "领取",
    "领",
    "抽",
    "抽签",
)
_MESSAGE_RETURN_HINTS = (
    "退回",
    "退还",
    "删除",
    "取消",
    "解绑",
    "关闭",
)
_MESSAGE_QUERY_HINTS = (
    "查",
    "搜",
    "搜索",
    "查询",
    "查看",
    "识别",
    "是什么",
    "今天",
    "今日",
    "本日",
    "当日",
    "看看",
    "看下",
)
_TEXT_LABELS = (
    "内容是",
    "内容为",
    "文案是",
    "文案为",
    "文字是",
    "文字为",
    "文本是",
    "文本为",
    "标题是",
    "标题为",
    "歌名是",
    "城市是",
    "地点是",
)
_GENERIC_STOPWORDS = {
    "真寻",
    "小真寻",
    "zhenxun",
    "nonebot",
    "plugin",
    "plugins",
    "机器人",
    "bot",
    "帮我",
    "帮忙",
    "麻烦",
    "请",
    "给我",
    "一下",
    "一个",
    "一张",
    "一首",
    "这个",
    "那个",
    "什么",
    "怎么",
    "如何",
    "怎样",
    "可以",
    "使用",
    "执行",
    "调用",
    "打开",
    "关闭",
    "开启",
    "禁用",
    "看看",
    "看下",
    "查询",
    "查看",
    "生成",
    "制作",
    "发送",
}
_ROUTE_NOISE_WORDS = {
    "帮我",
    "帮他",
    "帮她",
    "替我",
    "替他",
    "替她",
    "给我",
    "给他",
    "给她",
    "我",
    "他",
    "她",
    "真寻",
    "小真寻",
    "机器人",
    "bot",
    "请",
    "麻烦",
    "执行",
    "调用",
    "使用",
    "打开",
    "关闭",
    "开启",
    "禁用",
    "查看",
    "看看",
    "看下",
    "查询",
    "设置",
    "生成",
    "制作",
    "做个",
    "做一个",
    "做一张",
    "做张",
    "来个",
    "来一个",
    "来一张",
    "再来个",
    "再来一个",
    "再来一张",
    "点歌",
    "播放",
    "签到",
    "启动",
    "表情",
    "表情包",
    "梗图",
    "图片",
    "图",
    "模板",
    "一个",
    "一张",
    "一首",
    "个",
    "张",
    "的",
}
_TEMPLATE_KIND_HINTS = (
    "表情",
    "表情包",
    "梗图",
    "meme",
    "模板",
    "头像",
    "图片",
    "文本",
    "文字",
    "内容",
    "标题",
)
_SEARCH_QUERY_HINTS = (
    "找",
    "搜",
    "查找",
    "查询",
    "搜索",
    "寻找",
)
_SEARCH_QUERY_TIME_HINTS = (
    "今天",
    "今日",
    "本日",
    "当日",
)
_TEMPLATE_QUERY_HINTS = (
    "表情",
    "表情包",
    "梗图",
    "模板",
    "头像",
    "图片",
    "文字",
    "文本",
    "内容",
    "标题",
    "生成",
    "制作",
    "做一张",
    "做个",
    "做一个",
)
_TRANSACTION_QUERY_HINTS = (
    "红包",
    "金币",
    "金额",
    "总额",
    "总计",
    "总共",
    "合计",
    "转账",
    "支付",
    "打赏",
)
_SELF_QUERY_HINTS = (
    "签到",
    "自我介绍",
    "我的信息",
    "我自己",
    "本人",
    "自己",
)
_UTILITY_QUERY_HINTS = (
    "抠图",
    "超分",
    "识图",
    "词云",
    "关于",
    "消息排行",
    "消息统计",
    "统计",
    "管理",
    "admin",
)
_INLINE_AT_TOKEN_PATTERN = re.compile(
    r"\[@\d{5,20}\]|(?<![0-9A-Za-z_])@\d{5,20}(?=(?:\s|$|[的，,。.!！？?]))"
)
_NUMERIC_TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_AMOUNT_HINT_PATTERN = re.compile(
    r"(?:总额|总金|总计|总共|金额|合计|一共|共)\s*(?:为|是|=|:|：)?\s*(\d+(?:\.\d+)?)"
)
_COUNT_HINT_PATTERN = re.compile(
    r"(?:红包数|数量|个数|份数|数目|个)\s*(?:为|是|=|:|：)?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:个|份|次|张|人)\b"
)
_NUMERIC_PARAM_HINTS = (
    "amount",
    "num",
    "count",
    "total",
    "money",
    "gold",
    "金币数",
    "price",
    "quantity",
    "number",
    "金额",
    "总额",
    "总金",
    "总计",
    "总共",
    "合计",
    "红包数",
    "数量",
    "个数",
    "份数",
    "数目",
)
_TARGET_PARAM_HINTS = ("user", "target", "at", "nickname", "self")


@dataclass(frozen=True)
class SkillCommandSchema:
    command: str
    aliases: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    text_min: int | None = None
    text_max: int | None = None
    image_min: int | None = None
    image_max: int | None = None
    allow_at: bool | None = None
    actor_scope: str = "allow_other"
    target_requirement: str = "none"
    target_sources: tuple[str, ...] = ()
    allow_sticky_arg: bool = False


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    kind: str
    plugin_name: str
    plugin_module: str
    description: str
    commands: tuple[str, ...]
    aliases: tuple[str, ...]
    examples: tuple[str, ...]
    usage: str | None
    hint: str
    action_commands: tuple[str, ...]
    helper_commands: tuple[str, ...]
    tokens: tuple[str, ...]
    supports_placeholders: bool
    supports_text_payload: bool
    command_schemas: tuple[SkillCommandSchema, ...]


@dataclass(frozen=True)
class SkillRegistry:
    skills: tuple[SkillSpec, ...]
    signature: str


@dataclass(frozen=True)
class SkillRouteDecision:
    plugin_name: str
    plugin_module: str
    command: str
    source: str
    skill_kind: str


@dataclass(frozen=True)
class SkillRankedCandidate:
    skill: SkillSpec
    score: float
    matched_command: str | None
    exact_head_hit: bool
    inline_hit_count: int
    alias_hit_count: int
    name_hit: bool
    token_overlap: int
    route_intent: bool


@dataclass(frozen=True)
class SkillSearchResult:
    query: str
    registry: SkillRegistry
    fast_match: tuple[str, str, str] | None
    ranked_candidates: tuple[SkillRankedCandidate, ...]
    is_usage: bool
    include_usage: bool
    include_similarity: bool


def _signature(knowledge_base: PluginKnowledgeBase) -> str:
    return "\n".join(
        (
            f"{plugin.module}|{plugin.name}|"
            f"{','.join(plugin.commands)}|"
            f"{','.join(plugin.aliases)}|"
            f"{plugin.usage or ''}|"
            f"{';'.join(_command_meta_signature(meta) for meta in plugin.command_meta)}"
        )
        for plugin in knowledge_base.plugins
    )


def _command_meta_signature(meta: PluginInfo.PluginCommandMeta) -> str:
    return "|".join(
        [
            str(getattr(meta, "command", "") or "").strip(),
            ",".join(
                str(alias).strip()
                for alias in getattr(meta, "aliases", ()) or ()
                if str(alias).strip()
            ),
            str(getattr(meta, "text_min", "") or ""),
            str(getattr(meta, "text_max", "") or ""),
            str(getattr(meta, "image_min", "") or ""),
            str(getattr(meta, "image_max", "") or ""),
            str(int(bool(getattr(meta, "allow_at", False)))),
            str(getattr(meta, "actor_scope", "") or ""),
            str(getattr(meta, "target_requirement", "") or ""),
            ",".join(
                str(source).strip()
                for source in getattr(meta, "target_sources", ()) or ()
                if str(source).strip()
            ),
            str(int(bool(getattr(meta, "allow_sticky_arg", False)))),
            str(
                normalize_message_text(
                    str(getattr(meta, "access_level", "") or "")
                ).lower()
            ),
        ]
    )


def _normalize_skill_phrase(text: str | None) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    normalized = _PLACEHOLDER_PATTERN.sub(" ", normalized)
    normalized = _COMMAND_INLINE_PATTERN.sub(" ", normalized)
    normalized = _COMMAND_TAIL_PATTERN.sub("", normalized)
    normalized = normalized.replace("`", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip(
        " ：:，,。.!！?？-[](){}<>【】（）《》「」『』"
    )
    return normalized


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall((text or "").lower()):
        normalized = token.strip()
        if not normalized or normalized in _GENERIC_STOPWORDS:
            continue
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_command_schemas(
    plugin: PluginInfo,
    commands: tuple[str, ...],
) -> tuple[SkillCommandSchema, ...]:
    metas: dict[str, SkillCommandSchema] = {}
    for raw in plugin.command_meta:
        if (
            normalize_message_text(
                str(getattr(raw, "access_level", "public") or "public")
            ).lower()
            != "public"
        ):
            continue
        command = _normalize_skill_phrase(getattr(raw, "command", ""))
        if not command:
            continue
        aliases = tuple(
            alias
            for alias in (
                _normalize_skill_phrase(item)
                for item in (getattr(raw, "aliases", None) or [])
            )
            if alias
        )
        metas[command] = SkillCommandSchema(
            command=command,
            aliases=aliases,
            params=tuple(
                param
                for param in (
                    _normalize_skill_phrase(str(item or ""))
                    for item in (getattr(raw, "params", None) or [])
                )
                if param
            ),
            text_min=_safe_int(getattr(raw, "text_min", None)),
            text_max=_safe_int(getattr(raw, "text_max", None)),
            image_min=_safe_int(getattr(raw, "image_min", None)),
            image_max=_safe_int(getattr(raw, "image_max", None)),
            allow_at=getattr(raw, "allow_at", None),
            actor_scope=normalize_message_text(
                str(getattr(raw, "actor_scope", "allow_other") or "allow_other")
            ).lower()
            or "allow_other",
            target_requirement=normalize_message_text(
                str(getattr(raw, "target_requirement", "none") or "none")
            ).lower()
            or "none",
            target_sources=tuple(
                source
                for source in (
                    normalize_message_text(str(item or "")).lower()
                    for item in (getattr(raw, "target_sources", None) or [])
                )
                if source in {"at", "reply", "nickname", "self"}
            ),
            allow_sticky_arg=bool(getattr(raw, "allow_sticky_arg", False)),
        )
    for command in commands:
        metas.setdefault(command, SkillCommandSchema(command=command))
    return tuple(metas.values())


def _helper_commands(commands: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        command
        for command in commands
        if any(keyword in command for keyword in _HELPER_KEYWORDS)
    )


def _action_commands(
    commands: tuple[str, ...],
    helper_commands: tuple[str, ...],
) -> tuple[str, ...]:
    helper_set = set(helper_commands)
    values = [command for command in commands if command not in helper_set]
    if not values:
        values = list(commands)
    return tuple(dict.fromkeys(values))


def _infer_kind(
    plugin: PluginInfo,
    commands: tuple[str, ...],
    helper_commands: tuple[str, ...],
    examples: tuple[str, ...],
) -> str:
    haystack = " ".join(
        [
            plugin.name,
            plugin.module,
            plugin.description,
            " ".join(commands),
            plugin.usage or "",
            " ".join(examples),
        ]
    ).lower()
    has_template_hint = any(marker in haystack for marker in _TEMPLATE_KIND_HINTS)
    helper_text = " ".join(helper_commands)
    has_catalog_helper = bool(helper_commands) and any(
        marker in helper_text for marker in ("搜索", "详情", "列表")
    )

    if has_template_hint:
        return "template"
    if has_catalog_helper:
        return "catalog"
    return "command"


def infer_query_families(text: str) -> tuple[str, ...]:
    normalized = normalize_message_text(normalize_action_phrases(text)).lower()
    if not normalized:
        return ("general",)

    families: list[str] = []
    if contains_any(normalized, _TEMPLATE_QUERY_HINTS):
        families.append("template")
    if contains_any(normalized, _TRANSACTION_QUERY_HINTS):
        families.append("transaction")
    if contains_any(normalized, _SEARCH_QUERY_HINTS) or (
        "抽" in normalized
        and (
            contains_any(normalized, _SEARCH_QUERY_TIME_HINTS)
            or "猪" in normalized
            or "小猪" in normalized
        )
    ):
        families.append("search")
    if contains_any(normalized, _SELF_QUERY_HINTS):
        families.append("self")
    if contains_any(normalized, _UTILITY_QUERY_HINTS):
        families.append("utility")

    if not families:
        families.append("general")
    return tuple(dict.fromkeys(families))


def infer_query_family(text: str) -> str:
    return infer_query_families(text)[0]


def infer_skill_family(
    plugin: Any,
    commands: tuple[str, ...],
    helper_commands: tuple[str, ...],
    examples: tuple[str, ...],
) -> str:
    plugin_name = normalize_message_text(
        str(getattr(plugin, "name", getattr(plugin, "plugin_name", "")) or "")
    )
    plugin_module = normalize_message_text(
        str(getattr(plugin, "module", getattr(plugin, "plugin_module", "")) or "")
    )
    plugin_description = normalize_message_text(
        str(getattr(plugin, "description", "") or "")
    )
    plugin_usage = normalize_message_text(str(getattr(plugin, "usage", "") or ""))
    skill_kind = normalize_message_text(str(getattr(plugin, "kind", "") or "")).lower()
    command_text = " ".join(commands).lower()
    helper_text = " ".join(helper_commands).lower()
    haystack = " ".join(
        [
            plugin_name,
            plugin_module,
            plugin_description,
            " ".join(commands),
            plugin_usage,
            " ".join(examples),
        ]
    ).lower()

    if skill_kind == "template":
        return "template"
    if skill_kind == "catalog":
        return "search"

    if "meme" in plugin_module or "表情" in plugin_name:
        return "template"
    if any(marker in haystack for marker in _TRANSACTION_QUERY_HINTS):
        return "transaction"
    if (
        any(marker in haystack for marker in _SEARCH_QUERY_HINTS)
        or any(marker in command_text for marker in _SEARCH_QUERY_HINTS)
        or any(marker in helper_text for marker in _SEARCH_QUERY_HINTS)
    ):
        return "search"
    if any(marker in haystack for marker in _SELF_QUERY_HINTS):
        return "self"
    if any(marker in haystack for marker in _UTILITY_QUERY_HINTS):
        return "utility"
    return "general"


def infer_command_role(command: str, *, family: str = "general") -> str:
    normalized = normalize_message_text(command).lower()
    if not normalized:
        return "other"
    if any(token in normalized for token in ("帮助", "help", "用法", "说明", "详情", "参数")):
        return "help"
    if any(
        token in normalized
        for token in ("发", "塞", "创建", "新增", "生成", "制作", "上传", "设置", "绑定", "添加", "开启")
    ):
        return "create"
    if any(token in normalized for token in ("开", "抢", "领取", "领", "抽签")):
        return "open"
    if any(token in normalized for token in ("退回", "退还", "删除", "取消", "解绑", "关闭")):
        return "return"
    if any(
        token in normalized
        for token in ("查", "搜", "搜索", "查询", "查看", "识别", "是什么", "今天", "今日", "本日", "当日")
    ):
        return "query"
    if any(token in normalized for token in ("排行", "统计", "列表", "菜单")):
        return "catalog"
    if family == "transaction":
        return "create"
    if family == "search":
        return "query"
    if family == "template":
        return "create"
    return "other"


def _family_alignment_bonus(
    skill_family: str,
    query_families: tuple[str, ...],
    *,
    normalized_query: str,
    commands: tuple[str, ...],
    aliases: tuple[str, ...],
) -> float:
    if not query_families or not skill_family:
        return 0.0

    bonus = 0.0
    primary_family = query_families[0]
    if primary_family == "general":
        # "general" means we do not have a reliable family signal yet.
        # Do not let every generic plugin inherit the same +5 baseline,
        # otherwise shortlist/local parse degenerates into registry-order picks.
        if skill_family == "general":
            bonus += 0.0
        elif skill_family in query_families[1:]:
            bonus += 1.0
    elif skill_family == primary_family:
        bonus += 5.0
    elif skill_family in query_families[1:]:
        bonus += 3.0
    elif primary_family != "general" and skill_family == "general":
        bonus += 0.5

    family_blob = " ".join([*commands, *aliases]).lower()
    if skill_family == "search":
        if any(token in normalized_query for token in _SEARCH_QUERY_TIME_HINTS):
            if any(token in family_blob for token in _SEARCH_QUERY_TIME_HINTS):
                bonus += 2.8
        if any(token in normalized_query for token in _SEARCH_QUERY_HINTS):
            bonus += 1.6
    elif skill_family == "template":
        if contains_any(normalized_query, _TEMPLATE_QUERY_HINTS):
            bonus += 2.4
    elif skill_family == "transaction":
        if contains_any(normalized_query, _TRANSACTION_QUERY_HINTS):
            bonus += 2.4
    elif skill_family == "self":
        if contains_any(normalized_query, _SELF_QUERY_HINTS):
            bonus += 1.8
    elif skill_family == "utility":
        if contains_any(normalized_query, _UTILITY_QUERY_HINTS):
            bonus += 1.5

    return bonus


def _build_examples(plugin: PluginInfo) -> tuple[str, ...]:
    values: list[str] = []
    for meta in plugin.command_meta:
        for example in getattr(meta, "examples", [])[:2]:
            text = _normalize_skill_phrase(example)
            if text and text not in values:
                values.append(text)
    usage = _normalize_skill_phrase(plugin.usage)
    if usage and usage not in values:
        values.append(usage)
    return tuple(values[:4])


def _build_tokens(
    plugin: PluginInfo,
    commands: tuple[str, ...],
    aliases: tuple[str, ...],
    examples: tuple[str, ...],
) -> tuple[str, ...]:
    values = [
        plugin.name,
        plugin.description,
        " ".join(commands),
        " ".join(aliases),
        plugin.usage or "",
        " ".join(examples),
    ]
    tokens: list[str] = []
    for value in values:
        for token in _tokenize(value):
            if token not in tokens:
                tokens.append(token)
    return tuple(tokens[:120])


def _build_hint(
    plugin: PluginInfo,
    action_commands: tuple[str, ...],
    helper_commands: tuple[str, ...],
) -> str:
    parts: list[str] = []
    if plugin.description:
        parts.append(plugin.description.strip())
    if action_commands:
        parts.append(f"执行命令: {', '.join(action_commands[:3])}")
    if helper_commands:
        parts.append(f"辅助命令: {', '.join(helper_commands[:2])}")
    return "；".join(part for part in parts if part) or "执行插件命令"


def get_skill_registry(knowledge_base: PluginKnowledgeBase) -> SkillRegistry:
    signature = _signature(knowledge_base)
    cached = _CACHE.get(signature)
    if cached is not None:
        return cached

    skills: list[SkillSpec] = []
    for plugin in knowledge_base.plugins:
        commands = tuple(
            command
            for command in (_normalize_skill_phrase(item) for item in plugin.commands)
            if command
        )
        aliases = tuple(
            alias
            for alias in (_normalize_skill_phrase(item) for item in plugin.aliases)
            if alias
        )
        examples = _build_examples(plugin)
        helper_commands = _helper_commands(commands)
        action_commands = _action_commands(commands, helper_commands)
        command_schemas = _extract_command_schemas(plugin, commands)
        kind = _infer_kind(plugin, commands, helper_commands, examples)
        text_space = " ".join(
            [plugin.usage or "", " ".join(examples), " ".join(commands)]
        )
        supports_placeholders = any(
            marker in text_space for marker in ("[image]", "@", "图片", "头像", "自己")
        )
        supports_text_payload = any(
            marker in text_space
            for marker in ("文字", "文本", "内容", "标题", "歌名", "城市", "地点")
        )
        skills.append(
            SkillSpec(
                skill_id=f"{kind}:{plugin.module}",
                kind=kind,
                plugin_name=plugin.name,
                plugin_module=plugin.module,
                description=plugin.description or "暂无描述",
                commands=commands,
                aliases=aliases,
                examples=examples,
                usage=plugin.usage,
                hint=_build_hint(plugin, action_commands, helper_commands),
                action_commands=action_commands,
                helper_commands=helper_commands,
                tokens=_build_tokens(plugin, commands, aliases, examples),
                supports_placeholders=supports_placeholders,
                supports_text_payload=supports_text_payload,
                command_schemas=command_schemas,
            )
        )

    kind_order = {"catalog": 0, "template": 1, "command": 2}
    skills.sort(key=lambda item: (kind_order.get(item.kind, 99), item.plugin_module))
    registry = SkillRegistry(skills=tuple(skills), signature=signature)
    _CACHE[signature] = registry
    if len(_CACHE) > 16:
        for key in list(_CACHE)[:-8]:
            _CACHE.pop(key, None)
    return registry


def _prepare_query(text: str) -> tuple[str, str, str, tuple[str, ...]]:
    normalized = normalize_message_text(normalize_action_phrases(text))
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    lowered = normalized.lower()
    tokens = tuple(_tokenize(normalized))
    return normalized, stripped or normalized, lowered, tokens


def infer_message_action_role(message_text: str) -> str:
    normalized = normalize_message_text(strip_invoke_prefix(message_text or "")).lower()
    if not normalized:
        return "other"
    if contains_any(normalized, _EXPLICIT_HELPER_HINTS):
        return "help"
    if contains_any(normalized, _MESSAGE_RETURN_HINTS):
        return "return"
    if contains_any(normalized, _MESSAGE_CREATE_HINTS):
        return "create"
    if contains_any(normalized, _MESSAGE_OPEN_HINTS):
        return "open"
    if contains_any(normalized, _MESSAGE_QUERY_HINTS):
        return "query"
    return "other"


def _command_role_alignment_bonus(message_role: str, command_role: str) -> float:
    if not message_role or message_role == "other" or not command_role:
        return 0.0
    if message_role == command_role:
        return 14.0
    if message_role == "help":
        return 8.0 if command_role == "help" else -4.0
    if message_role == "query":
        if command_role in {"query", "catalog"}:
            return 10.0
        return 0.0
    if message_role == "create":
        if command_role == "create":
            return 14.0
        if command_role in {"open", "return"}:
            return -8.0
        return 0.0
    if message_role == "open":
        if command_role == "open":
            return 14.0
        if command_role in {"create", "return"}:
            return -8.0
        return 0.0
    if message_role == "return":
        if command_role == "return":
            return 14.0
        if command_role in {"create", "open"}:
            return -8.0
        return 0.0
    return 0.0


def _match_score_for_command(
    command: str,
    *,
    normalized: str,
    stripped: str,
    lowered: str,
    ) -> tuple[float, bool, bool]:
    command_text = command.strip()
    if not command_text:
        return 0.0, False, False
    if match_command_head(stripped, command_text) or match_command_head(
        normalized, command_text
    ):
        return 32.0 + len(command_text) / 50.0, True, False
    command_lower = command_text.lower()
    if command_lower in lowered:
        return 18.0 + len(command_text) / 100.0, False, True
    return 0.0, False, False


def _build_ranked_candidate(
    skill: SkillSpec,
    query: str,
    *,
    include_usage: bool,
    include_similarity: bool,
) -> SkillRankedCandidate:
    normalized, stripped, lowered, query_tokens = _prepare_query(query)
    query_families = infer_query_families(normalized)
    message_role = infer_message_action_role(normalized)
    if not normalized:
        return SkillRankedCandidate(
            skill=skill,
            score=0.0,
            matched_command=None,
            exact_head_hit=False,
            inline_hit_count=0,
            alias_hit_count=0,
            name_hit=False,
            token_overlap=0,
            route_intent=False,
        )

    score = 0.0
    matched_command: str | None = None
    exact_head_hit = False
    inline_hit_count = 0
    alias_hit_count = 0

    skill_family = infer_skill_family(
        skill,
        skill.commands,
        skill.helper_commands,
        skill.examples,
    )

    for command in skill.action_commands:
        command_score, head_hit, inline_hit = _match_score_for_command(
            command,
            normalized=normalized,
            stripped=stripped,
            lowered=lowered,
        )
        command_role = infer_command_role(command, family=skill_family)
        command_score += _command_role_alignment_bonus(message_role, command_role)
        if command_score <= 0:
            continue
        score += command_score
        if matched_command is None or command_score > 30:
            matched_command = command
        exact_head_hit = exact_head_hit or head_hit
        if inline_hit:
            inline_hit_count += 1

    for alias in skill.aliases:
        alias_score, alias_head_hit, alias_inline_hit = _match_score_for_command(
            alias,
            normalized=normalized,
            stripped=stripped,
            lowered=lowered,
        )
        alias_role = infer_command_role(alias, family=skill_family)
        alias_score += _command_role_alignment_bonus(message_role, alias_role)
        if alias_score <= 0:
            continue
        score += alias_score * 0.75
        if alias_head_hit and matched_command is None:
            matched_command = alias
        exact_head_hit = exact_head_hit or alias_head_hit
        if alias_inline_hit:
            alias_hit_count += 1

    name_hit = False
    plugin_name_lower = skill.plugin_name.lower()
    if plugin_name_lower and plugin_name_lower in lowered:
        name_hit = True
        score += 9.0

    for alias in skill.aliases:
        if alias and alias.lower() in lowered:
            name_hit = True
            score += 5.0
            break

    token_overlap = 0
    if include_similarity and query_tokens and skill.tokens:
        token_overlap = len(set(query_tokens) & set(skill.tokens))
        score += min(token_overlap * 1.2, 10.0)

    if include_usage and is_usage_question(normalized) and skill.helper_commands:
        score += 6.0

    if skill.supports_placeholders and ("[@" in normalized or "[image" in normalized):
        score += 1.8

    route_intent = (stripped != normalized) or contains_any(
        normalized, ROUTE_ACTION_WORDS
    )
    if route_intent:
        score += 1.5
    if skill.kind == "template" and contains_any(
        normalized,
        ("表情", "表情包", "梗图", "模板", "头像", "图片"),
    ):
        score += 8.0
    if skill.kind == "template" and ("[@" in normalized or "[image" in normalized):
        score += 2.0
    score += _family_alignment_bonus(
        skill_family,
        query_families,
        normalized_query=normalized,
        commands=skill.commands,
        aliases=skill.aliases,
    )
    if has_negative_route_intent(normalized):
        score -= 12.0

    return SkillRankedCandidate(
        skill=skill,
        score=score,
        matched_command=matched_command,
        exact_head_hit=exact_head_hit,
        inline_hit_count=inline_hit_count,
        alias_hit_count=alias_hit_count,
        name_hit=name_hit,
        token_overlap=token_overlap,
        route_intent=route_intent,
    )


def _rank_skills(
    registry: SkillRegistry,
    query: str,
    *,
    include_usage: bool,
    include_similarity: bool,
) -> list[SkillRankedCandidate]:
    ranked = [
        _build_ranked_candidate(
            skill,
            query,
            include_usage=include_usage,
            include_similarity=include_similarity,
        )
        for skill in registry.skills
    ]
    ranked.sort(
        key=lambda item: (
            item.score,
            item.exact_head_hit,
            item.inline_hit_count,
            item.alias_hit_count,
            item.name_hit,
            item.token_overlap,
            len(item.skill.plugin_name),
        ),
        reverse=True,
    )
    return ranked


def _find_skill_by_identity(
    registry: SkillRegistry,
    plugin_name: str | None,
    plugin_module: str | None,
) -> SkillSpec | None:
    module_text = normalize_message_text(plugin_module or "")
    if module_text:
        for skill in registry.skills:
            if skill.plugin_module == module_text:
                return skill

    name_text = normalize_message_text(plugin_name or "").lower()
    if not name_text:
        return None
    for skill in registry.skills:
        if skill.plugin_name.lower() == name_text:
            return skill
        if name_text in {alias.lower() for alias in skill.aliases}:
            return skill
    return None


def _resolve_skill_command_schema(
    skill: SkillSpec,
    command_head: str,
) -> SkillCommandSchema | None:
    normalized_head = _normalize_skill_phrase(command_head)
    if not normalized_head:
        return None
    for schema in skill.command_schemas:
        if schema.command == normalized_head:
            return schema
        if normalized_head in schema.aliases:
            return schema
    return None


def _pick_helper_command(skill: SkillSpec, message_text: str) -> str | None:
    if not _has_explicit_helper_intent(message_text):
        return None
    if not skill.helper_commands:
        return None
    normalized = normalize_message_text(message_text).lower()
    ranked = sorted(
        skill.helper_commands,
        key=lambda item: (
            item.lower() not in normalized,
            "详情" not in item,
            "帮助" not in item,
            "说明" not in item,
            len(item),
        ),
    )
    return ranked[0] if ranked else None


def _has_explicit_helper_intent(message_text: str) -> bool:
    normalized = normalize_message_text(message_text).lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _EXPLICIT_HELPER_HINTS)


def _pick_command_by_evidence(
    skill: SkillSpec,
    *,
    message_text: str,
    suggested_command: str | None = None,
    prefer_helper: bool = False,
    allow_fallback: bool = False,
) -> str | None:
    pool = skill.helper_commands if prefer_helper else skill.action_commands
    if not pool:
        pool = skill.commands
    pool_values: list[str] = []
    for command in pool:
        normalized_command = normalize_message_text(command)
        if normalized_command and normalized_command not in pool_values:
            pool_values.append(normalized_command)
    for alias in skill.aliases:
        normalized_alias = normalize_message_text(alias)
        if normalized_alias and normalized_alias not in pool_values:
            pool_values.append(normalized_alias)

    normalized, stripped, lowered, query_tokens = _prepare_query(message_text)
    query_families = infer_query_families(normalized)
    message_role = infer_message_action_role(normalized)
    skill_family = infer_skill_family(
        skill,
        skill.commands,
        skill.helper_commands,
        skill.examples,
    )
    suggested = normalize_message_text(suggested_command or "")
    suggested_lower = suggested.lower()

    best: tuple[float, str] | None = None
    for command in pool_values:
        score = 0.0
        has_command_evidence = False
        if suggested and (
            match_command_head(suggested, command)
            or command.lower() in suggested_lower
        ):
            score += 30.0
            has_command_evidence = True
        command_score, _, _ = _match_score_for_command(
            command,
            normalized=normalized,
            stripped=stripped,
            lowered=lowered,
        )
        command_role = infer_command_role(command, family=skill_family)
        role_bonus = _command_role_alignment_bonus(message_role, command_role)
        command_score += role_bonus
        score += command_score

        overlap = [token for token in _tokenize(command) if token in query_tokens]
        if overlap:
            score += min(len(overlap) * 2.0, 8.0)
            has_command_evidence = True
        if command_score > 0:
            has_command_evidence = True
        if role_bonus > 0:
            has_command_evidence = True

        family_bonus = _family_alignment_bonus(
            skill_family,
            query_families,
            normalized_query=normalized,
            commands=skill.commands,
            aliases=skill.aliases,
        )
        score += family_bonus

        # Family alignment can reorder already-matched candidates, but it must not
        # create a command match out of thin air.
        if not has_command_evidence:
            continue

        if score <= 0:
            continue
        candidate = (score, command)
        if best is None or candidate > best:
            best = candidate

    if best is not None:
        return best[1]

    if allow_fallback and len(pool_values) == 1:
        if skill.plugin_name.lower() in lowered or any(
            alias.lower() in lowered for alias in skill.aliases
        ):
            return pool_values[0]
    return None


def _extract_explicit_value(message_text: str) -> str:
    normalized = normalize_message_text(strip_invoke_prefix(message_text))
    if not normalized:
        return ""
    for label in _TEXT_LABELS:
        if label not in normalized:
            continue
        value = normalized.split(label, 1)[1].lstrip(" :：")
        return normalize_message_text(value).strip("`\"'“”‘’")
    return ""


def _strip_route_noise(text: str) -> str:
    normalized = normalize_message_text(text)
    if not normalized:
        return ""
    tokens = [token for token in normalized.split(" ") if token]
    cleaned = [token for token in tokens if token not in _ROUTE_NOISE_WORDS]
    if cleaned:
        return normalize_message_text(" ".join(cleaned))
    return ""


def _is_numeric_param_name(param_name: str) -> bool:
    normalized = normalize_message_text(param_name).lower()
    if not normalized:
        return False
    return any(hint in normalized for hint in _NUMERIC_PARAM_HINTS)


def _is_target_param_name(param_name: str) -> bool:
    normalized = normalize_message_text(param_name).lower()
    if not normalized:
        return False
    return any(hint in normalized for hint in _TARGET_PARAM_HINTS)


def _schema_supports_payload(schema: SkillCommandSchema | None) -> bool:
    if schema is None:
        return False
    if schema.params:
        return True
    if (schema.text_min or 0) > 0 or (schema.text_max or 0) > 0:
        return True
    if (schema.image_min or 0) > 0 or (schema.image_max or 0) > 0:
        return True
    if schema.allow_at is not None:
        return True
    if schema.target_requirement and schema.target_requirement != "none":
        return True
    return bool(schema.target_sources)


def _score_schema_payload_fit(
    schema: SkillCommandSchema,
    *,
    head_text: str,
    argument_text: str,
) -> float:
    score = 0.0
    normalized_head = normalize_message_text(head_text).lower()
    normalized_argument = normalize_message_text(argument_text).lower()
    schema_blob = " ".join(
        [
            schema.command,
            " ".join(schema.aliases),
            " ".join(schema.params),
        ]
    ).lower()
    if not normalized_head and not normalized_argument:
        return 0.0

    if "金币" in normalized_head and "金币" in schema_blob:
        score += 12.0
    if "红包" in normalized_head and "红包" in schema_blob:
        score += 8.0
    if any(token in normalized_head for token in ("amount", "money", "gold", "金额", "总额", "总金", "总计", "总共", "合计")) and any(
        token in schema_blob for token in ("amount", "money", "gold", "金币数", "金币", "金额", "总额", "总金", "总计", "总共", "合计")
    ):
        score += 10.0
    if any(token in normalized_head for token in ("num", "count", "quantity", "number", "红包数", "数量", "个数", "份数", "数目")) and any(
        token in schema_blob for token in ("num", "count", "quantity", "number", "红包数", "数量", "个数", "份数", "数目")
    ):
        score += 10.0

    schema_tokens = _extract_schema_argument_tokens(argument_text, schema)
    if schema_tokens:
        score += 8.0 + len(schema_tokens) * 4.0
    else:
        argument_tokens = _parse_argument_tokens(argument_text)
        if argument_tokens:
            score += min(len(argument_tokens) * 2.0, 8.0)

    if _NUMERIC_TOKEN_PATTERN.search(normalized_argument):
        score += 2.0
    if _INLINE_AT_TOKEN_PATTERN.search(normalized_argument):
        score += 2.0
    if "image" in normalized_argument:
        score += 2.0
    return score


def _message_has_payload_signals(text: str) -> bool:
    normalized = normalize_message_text(text).lower()
    if not normalized:
        return False
    if _NUMERIC_TOKEN_PATTERN.search(normalized):
        return True
    if _INLINE_AT_TOKEN_PATTERN.search(normalized):
        return True
    if contains_any(normalized, _TEXT_LABELS):
        return True
    if contains_any(
        normalized,
        (
            "总额",
            "总金",
            "总计",
            "总共",
            "金额",
            "合计",
            "个数",
            "数量",
            "份数",
            "数目",
            "红包数",
            "金币数",
            "文字",
            "文本",
            "内容",
            "标题",
        ),
    ):
        return True
    return False


def _select_payload_schema(
    skill: SkillSpec,
    *,
    current_schema: SkillCommandSchema | None,
    head_text: str,
    argument_text: str,
    message_text: str,
) -> SkillCommandSchema | None:
    if current_schema is None:
        return None
    if _schema_supports_payload(current_schema):
        return current_schema

    probe_text = normalize_message_text(argument_text or message_text)
    if not probe_text or not _message_has_payload_signals(probe_text):
        return current_schema

    best_schema = current_schema
    best_score = _score_schema_payload_fit(
        current_schema,
        head_text=head_text,
        argument_text=probe_text,
    )
    for candidate_schema in skill.command_schemas:
        if candidate_schema is current_schema:
            continue
        if not _schema_supports_payload(candidate_schema):
            continue
        candidate_score = _score_schema_payload_fit(
            candidate_schema,
            head_text=head_text,
            argument_text=probe_text,
        )
        if candidate_score > best_score:
            best_schema = candidate_schema
            best_score = candidate_score
    return best_schema


def _extract_labeled_number(raw_text: str, pattern: re.Pattern[str]) -> str:
    compact = normalize_message_text(raw_text).replace(" ", "")
    if not compact:
        return ""
    match = pattern.search(compact)
    if not match:
        return ""
    for group in match.groups():
        if group:
            return group
    return ""


def _extract_schema_argument_tokens(
    argument_text: str,
    schema: SkillCommandSchema | None,
) -> list[str]:
    raw = normalize_message_text(argument_text)
    if not raw or schema is None:
        return []
    param_names = tuple(schema.params or ())
    if not param_names:
        return []

    numeric_param_count = sum(
        1 for param in param_names if _is_numeric_param_name(param)
    )
    if numeric_param_count < 2:
        return []
    if (schema.text_min or 0) > 0:
        return []

    all_numbers = [token for token in _NUMERIC_TOKEN_PATTERN.findall(raw) if token]
    if not all_numbers:
        return []

    amount_value = _extract_labeled_number(raw, _AMOUNT_HINT_PATTERN)
    count_value = _extract_labeled_number(raw, _COUNT_HINT_PATTERN)

    ordered_tokens: list[str] = []
    number_index = 0

    def _consume_next_number() -> str:
        nonlocal number_index
        if number_index >= len(all_numbers):
            return ""
        token = all_numbers[number_index]
        number_index += 1
        return token

    for param_name in param_names:
        if _is_target_param_name(param_name):
            continue
        if not _is_numeric_param_name(param_name):
            continue
        param_l = normalize_message_text(param_name).lower()
        value = ""
        if amount_value and any(
            hint in param_l
            for hint in (
                "amount",
                "money",
                "gold",
                "金币数",
                "金币",
                "price",
                "总额",
                "金额",
                "总金",
                "总计",
                "总共",
                "合计",
                "共",
            )
        ):
            value = amount_value
        elif count_value and any(
            hint in param_l for hint in ("num", "count", "quantity", "number", "红包数", "数量", "个数", "份数", "数目")
        ):
            value = count_value
        if not value:
            value = _consume_next_number()
        if value:
            ordered_tokens.append(value)

    if not ordered_tokens:
        return []
    if len(ordered_tokens) < numeric_param_count:
        for token in all_numbers:
            ordered_tokens.append(token)
            if len(ordered_tokens) >= numeric_param_count:
                break
    return ordered_tokens


def _extract_argument_around_head(message_text: str, command_head: str) -> str:
    normalized = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text))
    )
    if not normalized or not command_head:
        return ""

    explicit_value = _extract_explicit_value(normalized)
    if explicit_value:
        return explicit_value

    if match_command_head(normalized, command_head):
        return normalize_message_text(
            normalized[len(command_head) :].strip(" ：:，,。.!！?？")
        )

    if normalized.startswith(command_head) and len(normalized) > len(command_head):
        sticky_tail = normalize_message_text(
            normalized[len(command_head) :].strip(" ：:，,。.!！?？")
        )
        if sticky_tail:
            return sticky_tail

    index = normalized.find(command_head)
    if index < 0:
        return ""

    before = normalize_message_text(normalized[:index].strip(" ：:，,。.!！?？"))
    after = normalize_message_text(
        normalized[index + len(command_head) :].strip(" ：:，,。.!！?？")
    )
    before_cleaned = _strip_route_noise(before)
    after_cleaned = _strip_route_noise(after)

    if before_cleaned and after_cleaned:
        combined = normalize_message_text(f"{before_cleaned} {after_cleaned}")
        if _message_has_payload_signals(combined):
            return combined

    if after_cleaned:
        return after_cleaned

    if before_cleaned and len(before_cleaned) <= 30:
        return before_cleaned
    return ""


def _extract_command_context(
    message_text: str,
    command_head: str,
    *,
    schema: SkillCommandSchema | None,
) -> str:
    parsed = parse_command_with_head(
        message_text,
        command_head,
        allow_sticky=bool(getattr(schema, "allow_sticky_arg", False)),
    )
    if parsed is not None:
        fragments = [parsed.prefix_text, parsed.payload_text]
        combined = normalize_message_text(" ".join(part for part in fragments if part))
        if combined:
            return combined
    return _extract_argument_around_head(message_text, command_head)


def _strip_skill_terms(argument: str, skill: SkillSpec) -> str:
    cleaned = normalize_message_text(argument)
    if not cleaned:
        return ""

    removable: list[str] = [
        skill.plugin_name,
        *skill.aliases,
        *skill.commands,
        *skill.helper_commands,
        *skill.action_commands,
        "表情",
        "表情包",
        "梗图",
        "模板",
        "图片",
        "图",
        "功能",
        "命令",
    ]
    removable = sorted(
        {
            normalize_message_text(item)
            for item in removable
            if normalize_message_text(item)
        },
        key=len,
        reverse=True,
    )
    lowered = cleaned.lower()
    for token in removable:
        token_l = token.lower()
        if lowered == token_l:
            return ""
        prefix = f"{token_l} "
        if lowered.startswith(prefix):
            cleaned = normalize_message_text(cleaned[len(token) :])
            lowered = cleaned.lower()
        suffix = f" {token_l}"
        if lowered.endswith(suffix):
            cleaned = normalize_message_text(cleaned[: -len(token)].strip())
            lowered = cleaned.lower()
    return cleaned


def _extract_template_keyword(message_text: str, skill: SkillSpec) -> str:
    normalized = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text))
    )
    if not normalized:
        return ""

    direct_commands = sorted(
        {
            normalize_message_text(command)
            for command in skill.commands
            if command
            and normalize_message_text(command)
            and normalize_message_text(command) not in set(skill.helper_commands)
        },
        key=len,
        reverse=True,
    )
    for command in direct_commands:
        if command in normalized:
            return command

    pattern = re.compile(
        r"(?:做(?:一张|个|一个)?|来(?:一张|个|一个)?|生成(?:一张|个|一个)?|制作(?:一张|个|一个)?|做张)(.+?)(?:表情包|表情|梗图|模板|图片)",
        re.IGNORECASE,
    )
    keyword = ""
    match = pattern.search(normalized)
    if match:
        fragment = sanitize_template_tail(match.group(1))
        fragment = re.sub(r"^(?:我|你|他|她|ta)(?:自己)?(?:的)?", "", fragment)
        fragment = re.sub(r"^(?:自己|本人|头像|图片|图)", "", fragment)
        fragment = sanitize_template_tail(fragment)
        if fragment:
            keyword = fragment

    if not keyword:
        cleaned = normalize_message_text(_PLACEHOLDER_PATTERN.sub(" ", normalized))
        cleaned = _strip_route_noise(cleaned)
        cleaned = _strip_skill_terms(cleaned, skill)
        tokens = [
            token
            for token in cleaned.split(" ")
            if token and token not in _ROUTE_NOISE_WORDS
        ]
        if tokens:
            keyword = tokens[-1]

    keyword = _INLINE_AT_TOKEN_PATTERN.sub(" ", keyword)
    keyword = normalize_message_text(keyword)
    keyword = re.sub(r"^的+", "", keyword)
    keyword = re.sub(r"^(?:\d{5,20})(?:的)?", "", keyword)
    if keyword.startswith(("我", "你", "他", "她")) and "的" in keyword:
        keyword = keyword.rsplit("的", 1)[-1]
    keyword = re.sub(r"^(?:我|你|他|她|ta)(?:自己)?(?:的)?", "", keyword)
    keyword = re.sub(r"^自己(?:的)?", "", keyword)
    keyword = sanitize_template_tail(keyword)
    if not keyword:
        return ""

    if keyword.endswith("表情包"):
        keyword = keyword[: -len("表情包")]
    elif keyword.endswith("表情"):
        keyword = keyword[: -len("表情")]
    elif keyword.endswith("梗图"):
        keyword = keyword[: -len("梗图")]
    elif keyword.endswith("模板"):
        keyword = keyword[: -len("模板")]
    keyword = sanitize_template_tail(keyword)
    return keyword if len(keyword) >= 1 else ""


def _is_dynamic_template_skill(skill: SkillSpec) -> bool:
    if skill.kind != "template":
        return False
    commands_text = " ".join(skill.commands)
    has_catalog = "搜索" in commands_text and "详情" in commands_text
    return has_catalog


def _parse_argument_tokens(argument_text: str) -> list[str]:
    argument = normalize_message_text(argument_text)
    if not argument:
        return []
    return [token for token in argument.split(" ") if token]


def _normalize_at_placeholder(token: str) -> str:
    text = normalize_message_text(token or "")
    if not text:
        return ""
    if text.startswith("[@") and text.endswith("]"):
        return text
    match = _INLINE_AT_TOKEN_PATTERN.fullmatch(text)
    if not match:
        return ""
    return f"[{text}]"


def _sanitize_usage_argument(argument_text: str) -> str:
    argument = normalize_message_text(argument_text)
    if not argument:
        return ""
    for marker in (
        "怎么用",
        "如何用",
        "怎样用",
        "怎么使用",
        "如何使用",
        "怎样使用",
        "用法",
        "说明",
        "教程",
        "是什么",
        "什么意思",
    ):
        index = argument.find(marker)
        if index >= 0:
            argument = argument[:index].strip(" ：:，,。.!！?？")
            break
    argument = re.sub(r"(这个|该)?(表情|模板|功能)$", "", argument).strip(
        " ：:，,。.!！?？"
    )
    argument = re.sub(r"^(这个|该)", "", argument).strip(" ：:，,。.!！?？")
    return sanitize_template_tail(argument)


def _apply_command_schema(
    command_head: str,
    *,
    argument_text: str,
    placeholders: list[str],
    schema: SkillCommandSchema | None,
) -> str:
    schema_tokens = _extract_schema_argument_tokens(argument_text, schema)
    text_tokens = (
        schema_tokens if schema_tokens else _parse_argument_tokens(argument_text)
    )
    at_tokens = []
    for token in placeholders:
        normalized_at = _normalize_at_placeholder(token)
        if normalized_at and normalized_at not in at_tokens:
            at_tokens.append(normalized_at)
    image_tokens = [token for token in placeholders if token.startswith("[image")]

    if schema is not None:
        allow_at = schema.allow_at
        if allow_at is False:
            at_tokens = []

        if schema.image_max is not None:
            image_tokens = image_tokens[: max(schema.image_max, 0)]
        if schema.image_max == 0:
            image_tokens = []

        if schema.text_max is not None:
            text_max = max(schema.text_max, 0)
            if text_max == 0:
                text_tokens = []
            elif text_max == 1 and len(text_tokens) > 1:
                text_tokens = [" ".join(text_tokens)]
            elif len(text_tokens) > text_max:
                text_tokens = text_tokens[:text_max]

    payload_tokens = [token for token in text_tokens if token]
    payload_tokens.extend(at_tokens)
    payload_tokens.extend(image_tokens)
    return normalize_message_text(" ".join([command_head, *payload_tokens]).strip())


def _compose_skill_command(
    skill: SkillSpec,
    command_head: str,
    *,
    message_text: str,
    suggested_command: str | None = None,
) -> str:
    head = _normalize_skill_phrase(command_head)
    if not head:
        return ""

    schema = _resolve_skill_command_schema(skill, head)
    normalized_message = normalize_message_text(
        normalize_action_phrases(strip_invoke_prefix(message_text))
    )
    command_at_head = match_command_head(normalized_message, head)
    placeholders = collect_placeholders(message_text)
    parsed_context = parse_command_with_head(
        message_text,
        head,
        allow_sticky=bool(getattr(schema, "allow_sticky_arg", False)),
    )
    if parsed_context is None and suggested_command:
        parsed_context = parse_command_with_head(
            suggested_command,
            head,
            allow_sticky=bool(getattr(schema, "allow_sticky_arg", False)),
        )
    if parsed_context is not None:
        fragments = [parsed_context.prefix_text, parsed_context.payload_text]
        argument_text = normalize_message_text(
            " ".join(part for part in fragments if part)
        )
    else:
        argument_text = _extract_command_context(
            message_text,
            head,
            schema=schema,
        )

    if not argument_text and suggested_command:
        argument_text = _extract_command_context(
            suggested_command,
            head,
            schema=schema,
        )
    argument_text = _strip_skill_terms(argument_text, skill)

    best_schema = _select_payload_schema(
        skill,
        current_schema=schema,
        head_text=head,
        argument_text=argument_text,
        message_text=message_text,
    )
    if best_schema is not None and best_schema is not schema:
        schema = best_schema
        head = _normalize_skill_phrase(best_schema.command) or head

    if (
        parsed_context is None
        and not command_at_head
        and not (schema is not None and schema.allow_sticky_arg and argument_text)
        and not _schema_supports_payload(schema)
        and not _message_has_payload_signals(message_text)
    ):
        argument_text = ""

    if head in skill.helper_commands:
        argument_text = _sanitize_usage_argument(argument_text)

    if not argument_text and skill.kind in {"template", "catalog"}:
        keyword = _extract_template_keyword(message_text, skill)
        if head in skill.helper_commands:
            keyword = _sanitize_usage_argument(keyword)
        if keyword:
            if (
                keyword != head
                and keyword not in _ROUTE_NOISE_WORDS
                and keyword not in skill.helper_commands
            ):
                argument_text = keyword

    command = _apply_command_schema(
        head,
        argument_text=argument_text,
        placeholders=placeholders,
        schema=schema,
    )
    if (
        normalize_message_text(command) == head
        and _message_has_payload_signals(message_text)
    ):
        rescued_argument = _strip_skill_terms(
            _extract_argument_around_head(message_text, head),
            skill,
        )
        if rescued_argument and rescued_argument != argument_text:
            rescued_schema = _select_payload_schema(
                skill,
                current_schema=schema,
                head_text=head,
                argument_text=rescued_argument,
                message_text=message_text,
            )
            if rescued_schema is not None and rescued_schema is not schema:
                schema = rescued_schema
                head = _normalize_skill_phrase(rescued_schema.command) or head
            command = _apply_command_schema(
                head,
                argument_text=rescued_argument,
                placeholders=placeholders,
                schema=schema,
            )
    return command


def _passes_conservative_guard(
    skill: SkillSpec,
    command: str,
    *,
    message_text: str,
    is_usage: bool,
) -> bool:
    normalized_command = normalize_message_text(command)
    if not normalized_command:
        return False
    if has_negative_route_intent(message_text):
        return False

    command_head = normalize_message_text(normalized_command.split(" ", 1)[0])
    if is_usage and skill.helper_commands:
        if command_head not in skill.helper_commands and not any(
            match_command_head(command_head, item) for item in skill.helper_commands
        ):
            return False

    if skill.kind == "catalog" and not is_usage:
        if command_head in skill.helper_commands and not contains_any(
            message_text, ROUTE_ACTION_WORDS
        ):
            return False

    if skill.kind == "template" and is_usage:
        return True

    allowed_heads = set(skill.commands) | set(skill.aliases)
    if allowed_heads and command_head not in allowed_heads:
        return False

    return True


def match_skill_command_fast(
    query: str,
    knowledge_base: PluginKnowledgeBase,
) -> tuple[str, str, str] | None:
    registry = get_skill_registry(knowledge_base)
    normalized, stripped, lowered, _ = _prepare_query(query)
    message_role = infer_message_action_role(normalized)
    if not normalized:
        return None

    is_usage = is_usage_question(normalized)
    query_families = infer_query_families(normalized)
    best: tuple[float, SkillSpec, str] | None = None
    for skill in registry.skills:
        pool = skill.helper_commands if is_usage else skill.action_commands
        if not pool:
            pool = skill.commands
        pool_values: list[str] = []
        for command in pool:
            normalized_command = normalize_message_text(command)
            if normalized_command and normalized_command not in pool_values:
                pool_values.append(normalized_command)
        for alias in skill.aliases:
            normalized_alias = normalize_message_text(alias)
            if normalized_alias and normalized_alias not in pool_values:
                pool_values.append(normalized_alias)
        alias_set = {
            normalize_message_text(alias)
            for alias in skill.aliases
            if normalize_message_text(alias)
        }

        skill_family = infer_skill_family(
            skill,
            skill.commands,
            skill.helper_commands,
            skill.examples,
        )

        for command in pool_values:
            normalized_command = command
            schema = None
            for current_schema in skill.command_schemas:
                if normalize_message_text(current_schema.command) == normalized_command:
                    schema = current_schema
                    break
                if any(
                    normalize_message_text(alias) == normalized_command
                    for alias in current_schema.aliases
                ):
                    schema = current_schema
                    break
            allow_sticky = bool(schema.allow_sticky_arg) if schema is not None else False
            score = 0.0
            if match_command_head(stripped, command):
                score += 100.0 + len(command)
            elif match_command_head(normalized, command):
                score += 85.0 + len(command)
            elif match_command_head_or_sticky(
                stripped,
                command,
                allow_sticky=allow_sticky,
            ):
                score += 72.0 + len(command)
            elif match_command_head_or_sticky(
                normalized,
                command,
                allow_sticky=allow_sticky,
            ):
                score += 60.0 + len(command)
            elif len(command) >= 2 and command.lower() in lowered:
                score += 30.0 + len(command) * 0.1
            if command in alias_set:
                score += 1.0
            command_role = infer_command_role(command, family=skill_family)
            score += _command_role_alignment_bonus(message_role, command_role)
            score += _family_alignment_bonus(
                skill_family,
                query_families,
                normalized_query=normalized,
                commands=skill.commands,
                aliases=skill.aliases,
            )
            if score <= 0:
                continue
            candidate = (score, skill, command)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return None
    if best[0] < 8.0:
        return None
    _, skill, command = best
    return (skill.plugin_name, skill.plugin_module, command)


def skill_search(
    query: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    include_usage: bool = True,
    include_similarity: bool = True,
) -> SkillSearchResult:
    registry = get_skill_registry(knowledge_base)
    normalized = normalize_message_text(query)
    usage = is_usage_question(normalized)
    fast_match = match_skill_command_fast(normalized, knowledge_base)
    ranked = _rank_skills(
        registry,
        normalized,
        include_usage=include_usage,
        include_similarity=include_similarity,
    )
    return SkillSearchResult(
        query=normalized,
        registry=registry,
        fast_match=fast_match,
        ranked_candidates=tuple(ranked),
        is_usage=usage,
        include_usage=include_usage,
        include_similarity=include_similarity,
    )


def _build_route_decision(
    skill: SkillSpec,
    command: str,
    *,
    source: str,
) -> SkillRouteDecision:
    return SkillRouteDecision(
        plugin_name=skill.plugin_name,
        plugin_module=skill.plugin_module,
        command=command,
        source=source,
        skill_kind=skill.kind,
    )


def skill_execute(
    search_result: SkillSearchResult,
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> SkillRouteDecision | None:
    registry = search_result.registry
    is_usage = search_result.is_usage
    has_explicit_helper_intent = _has_explicit_helper_intent(message_text)

    if search_result.fast_match is not None:
        plugin_name, plugin_module, command_head = search_result.fast_match
        skill = _find_skill_by_identity(registry, plugin_name, plugin_module)
        if skill is not None:
            if (
                not is_usage
                and skill.kind == "template"
                and has_template_route_context(message_text, knowledge_base)
            ):
                template_keyword = _extract_template_keyword(message_text, skill)
                if template_keyword and template_keyword in skill.commands:
                    command_head = template_keyword
            command = _compose_skill_command(
                skill,
                command_head,
                message_text=message_text,
                suggested_command=command_head,
            )
            command_head_normalized = normalize_message_text(
                command.split(" ", 1)[0] if command else ""
            )
            if (
                command_head_normalized
                and command_head_normalized in skill.helper_commands
                and not has_explicit_helper_intent
            ):
                command = ""
            if _passes_conservative_guard(
                skill,
                command,
                message_text=message_text,
                is_usage=is_usage,
            ):
                return _build_route_decision(skill, command, source="fast")

    for index, candidate in enumerate(search_result.ranked_candidates):
        if candidate.score <= 0:
            continue
        if index > 0 and candidate.score < 3.0:
            continue
        skill = candidate.skill
        template_keyword = ""
        if (
            not is_usage
            and skill.kind == "template"
            and has_template_route_context(message_text, knowledge_base)
        ):
            template_keyword = _extract_template_keyword(message_text, skill)
        command_head = candidate.matched_command
        if template_keyword and template_keyword in skill.commands:
            command_head = template_keyword
        if command_head is None:
            command_head = _pick_command_by_evidence(
                skill,
                message_text=message_text,
                prefer_helper=is_usage,
                allow_fallback=index == 0,
            )
            if template_keyword and template_keyword in skill.commands:
                command_head = template_keyword

        if command_head is None and is_usage:
            command_head = _pick_helper_command(skill, message_text)

        if command_head is None and template_keyword:
            if template_keyword in skill.commands:
                command_head = template_keyword
            elif _is_dynamic_template_skill(skill) and is_usage:
                helper = _pick_helper_command(skill, message_text)
                if helper:
                    command_head = helper

        if command_head is None:
            continue

        command = _compose_skill_command(
            skill,
            command_head,
            message_text=message_text,
            suggested_command=candidate.matched_command or command_head,
        )
        command_head_normalized = normalize_message_text(
            command.split(" ", 1)[0] if command else ""
        )
        if (
            command_head_normalized
            and command_head_normalized in skill.helper_commands
            and not has_explicit_helper_intent
        ):
            continue
        if not _passes_conservative_guard(
            skill,
            command,
            message_text=message_text,
            is_usage=is_usage,
        ):
            continue
        source = "rank_exact" if candidate.exact_head_hit else "rank"
        return _build_route_decision(skill, command, source=source)

    return None


def select_relevant_skills(
    registry: SkillRegistry,
    query: str,
    *,
    limit: int = 6,
) -> tuple[SkillSpec, ...]:
    if not registry.skills:
        return ()
    ranked = _rank_skills(
        registry,
        query,
        include_usage=True,
        include_similarity=True,
    )
    selected: list[SkillSpec] = []
    for candidate in ranked:
        if candidate.score <= 0:
            continue
        selected.append(candidate.skill)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _render_command_schema(schema: SkillCommandSchema) -> dict:
    result: dict[str, object] = {"command": schema.command}
    if schema.aliases:
        result["aliases"] = list(schema.aliases)
    if schema.text_min is not None:
        result["text_min"] = schema.text_min
    if schema.text_max is not None:
        result["text_max"] = schema.text_max
    if schema.image_min is not None:
        result["image_min"] = schema.image_min
    if schema.image_max is not None:
        result["image_max"] = schema.image_max
    if schema.allow_at is not None:
        result["allow_at"] = schema.allow_at
    if schema.actor_scope and schema.actor_scope != "allow_other":
        result["actor_scope"] = schema.actor_scope
    if schema.target_requirement and schema.target_requirement != "none":
        result["target_requirement"] = schema.target_requirement
    if schema.target_sources:
        result["target_sources"] = list(schema.target_sources)
    if schema.allow_sticky_arg:
        result["allow_sticky_arg"] = True
    return result


def _select_prompt_commands(
    commands: tuple[str, ...],
    query: str,
    *,
    limit: int,
    family: str = "general",
) -> list[str]:
    if not commands:
        return []
    normalized = normalize_message_text(normalize_action_phrases(query or ""))
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    query_tokens = tuple(_tokenize(normalized))
    message_role = infer_message_action_role(normalized)
    query_has_today_hint = any(token in normalized for token in _SEARCH_QUERY_TIME_HINTS)
    query_has_pig_hint = any(token in normalized for token in ("猪", "小猪"))

    scored: list[tuple[float, str]] = []
    for command in commands:
        command_text = normalize_message_text(command)
        if not command_text:
            continue
        score = 0.0
        command_role = infer_command_role(command_text, family=family)
        if normalized:
            if match_command_head(stripped, command_text) or match_command_head(
                normalized, command_text
            ):
                score += 200.0
            elif command_text.lower() in normalized.lower():
                score += 120.0

        overlap = len(set(_tokenize(command_text)) & set(query_tokens))
        if overlap > 0:
            score += overlap * 12.0
        role_bonus = _command_role_alignment_bonus(message_role, command_role)
        score += role_bonus
        if family == "search":
            if query_has_today_hint:
                if any(token in command_text for token in ("今天", "今日")):
                    score += 12.0
                elif any(token in command_text for token in ("本日", "当日")):
                    score += 6.0
            if query_has_pig_hint and any(token in command_text for token in ("猪", "小猪")):
                score += 4.0

        scored.append((score, command_text))

    scored.sort(key=lambda item: (item[0], -len(item[1]), item[1]), reverse=True)
    selected: list[str] = []
    for _, command_text in scored:
        if command_text not in selected:
            selected.append(command_text)
        if len(selected) >= limit:
            break
    return selected


def _select_prompt_schemas(
    skill: SkillSpec,
    selected_commands: list[str],
    *,
    limit: int,
    fallback_all: bool = True,
) -> list[SkillCommandSchema]:
    if not skill.command_schemas:
        return []
    selected_heads = {
        normalize_message_text(command).casefold()
        for command in selected_commands
        if normalize_message_text(command)
    }
    picked: list[SkillCommandSchema] = []
    for schema in skill.command_schemas:
        head = normalize_message_text(schema.command).casefold()
        if head and head in selected_heads:
            picked.append(schema)
            if len(picked) >= limit:
                return picked
    if picked:
        return picked
    if not fallback_all:
        return []
    return list(skill.command_schemas[:limit])


def render_skill_namespace(
    knowledge_base: PluginKnowledgeBase,
    *,
    query: str = "",
    limit: int = 10,
    preferred_modules: list[str] | tuple[str, ...] | None = None,
    include_helpers: bool = True,
    mask_module: bool = False,
) -> str:
    registry = get_skill_registry(knowledge_base)
    _ = query
    skills = list(registry.skills[: max(limit, 1)])

    if preferred_modules:
        preferred = [
            normalize_message_text(module)
            for module in preferred_modules
            if normalize_message_text(module)
        ]
        if preferred:
            module_to_skill = {skill.plugin_module: skill for skill in registry.skills}
            merged: list[SkillSpec] = []
            for module in preferred:
                skill = module_to_skill.get(module)
                if skill and skill not in merged:
                    merged.append(skill)
            for skill in skills:
                if skill not in merged:
                    merged.append(skill)
            skills = merged[:limit]

    payload: list[dict] = []
    for skill in skills:
        family = infer_skill_family(
            skill,
            skill.commands,
            skill.helper_commands,
            skill.examples,
        )
        selected_actions = _select_prompt_commands(
            skill.action_commands or skill.commands,
            query,
            limit=3,
            family=family,
        )
        if not selected_actions:
            selected_actions = list((skill.action_commands or skill.commands)[:3])
        selected_helpers: list[str] = []
        if include_helpers:
            selected_helpers = _select_prompt_commands(
                skill.helper_commands or skill.commands,
                query,
                limit=2,
                family=family,
            )
            if not selected_helpers:
                selected_helpers = list((skill.helper_commands or skill.commands)[:2])
        schema_selection = _select_prompt_schemas(
            skill,
            [*selected_actions, *selected_helpers],
            limit=6,
            fallback_all=True,
        )
        payload.append(
            {
                "skill": skill.plugin_name,
                "module": (
                    skill.plugin_module.rsplit(".", 1)[-1]
                    if mask_module
                    else skill.plugin_module
                ),
                "kind": skill.kind,
                "family": family,
                "action_commands": selected_actions,
                "action_roles": [
                    {
                        "command": command,
                        "role": infer_command_role(command, family=family),
                    }
                    for command in selected_actions
                ],
                "helper_commands": selected_helpers,
                "aliases": list(skill.aliases[:3]),
                "schemas": [
                    _render_command_schema(schema)
                    for schema in schema_selection
                ],
            }
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "SkillCommandSchema",
    "SkillRegistry",
    "SkillRouteDecision",
    "SkillSearchResult",
    "SkillSpec",
    "infer_command_role",
    "infer_message_action_role",
    "infer_query_family",
    "infer_query_families",
    "infer_skill_family",
    "get_skill_registry",
    "match_skill_command_fast",
    "render_skill_namespace",
    "select_relevant_skills",
    "skill_execute",
    "skill_search",
]
