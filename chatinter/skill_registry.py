from dataclasses import dataclass
import re
from typing import Any

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .route_text import (
    ROUTE_ACTION_WORDS,
    _find_short_noise_head_boundary,
    contains_any,
    match_command_head,
    normalize_action_phrases,
    normalize_message_text,
    parse_command_with_head,
    strip_invoke_prefix,
)

_CACHE: dict[str, "SkillRegistry"] = {}
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
_ROUTE_NOISE_WORDS = {
    "帮我",
    "帮他",
    "帮她",
    "帮忙",
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
    "一个",
    "一张",
    "一首",
    "个",
    "张",
    "的",
    "一下",
    "一下子",
    "一下下",
    "吧",
    "嘛",
    "呀",
    "啊",
    "哦",
    "呢",
    "啦",
    "了",
    "那个",
    "这个",
    "什么",
    "怎么",
    "如何",
    "怎样",
    "可以",
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
    r"\[@[^\]\s]+\]|(?<![0-9A-Za-z_])@\d{5,20}(?=(?:\s|$|[的，,。.!！？?]))"
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
    prefixes: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    text_min: int | None = None
    text_max: int | None = None
    image_min: int | None = None
    image_max: int | None = None
    allow_at: bool | None = None
    actor_scope: str = "allow_other"
    target_requirement: str = "none"
    target_sources: tuple[str, ...] = ()
    requires_reply: bool = False
    requires_private: bool = False
    requires_to_me: bool = False
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
class SkillSearchResult:
    fast_match: tuple[str, str, str] | None


def _score_explicit_head(
    *,
    normalized: str,
    stripped: str,
    head: str,
    allow_sticky: bool,
) -> float:
    score = 0.0
    if match_command_head(stripped, head):
        score = max(score, 240.0 + len(head))
    if match_command_head(normalized, head):
        score = max(score, 220.0 + len(head))
    if parse_command_with_head(stripped, head, allow_sticky=allow_sticky):
        score = max(score, 200.0 + len(head))
    if parse_command_with_head(normalized, head, allow_sticky=allow_sticky):
        score = max(score, 180.0 + len(head))
    return score


def skill_search(
    query: str,
    knowledge_base: PluginKnowledgeBase,
    *,
    include_usage: bool = True,
    include_similarity: bool = True,
) -> SkillSearchResult:
    _ = include_usage, include_similarity
    registry = get_skill_registry(knowledge_base)
    normalized = normalize_message_text(normalize_action_phrases(query or ""))
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if not normalized:
        return SkillSearchResult(fast_match=None)

    best: tuple[float, str, str, str] | None = None
    for skill in registry.skills:
        for schema in skill.command_schemas:
            for term in (schema.command, *schema.aliases):
                head = normalize_message_text(term)
                if not head:
                    continue
                score = _score_explicit_head(
                    normalized=normalized,
                    stripped=stripped,
                    head=head,
                    allow_sticky=bool(schema.allow_sticky_arg),
                )
                if score <= 0:
                    continue
                candidate = (
                    score,
                    skill.plugin_name,
                    skill.plugin_module,
                    normalize_message_text(schema.command),
                )
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        return SkillSearchResult(fast_match=None)
    _, plugin_name, plugin_module, command_head = best
    return SkillSearchResult(fast_match=(plugin_name, plugin_module, command_head))


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
            ",".join(
                str(prefix).strip()
                for prefix in getattr(meta, "prefixes", ()) or ()
                if str(prefix).strip()
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
            str(int(bool(getattr(meta, "requires_reply", False)))),
            str(int(bool(getattr(meta, "requires_private", False)))),
            str(int(bool(getattr(meta, "requires_to_me", False)))),
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
        " ：:，,。.!！?？[](){}<>【】（）《》「」『』"
    )
    return normalized


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
            prefixes=tuple(
                prefix
                for prefix in (
                    _normalize_skill_phrase(str(item or ""))
                    for item in (getattr(raw, "prefixes", None) or [])
                )
                if prefix
            ),
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
            requires_reply=bool(getattr(raw, "requires_reply", False)),
            requires_private=bool(getattr(raw, "requires_private", False)),
            requires_to_me=bool(getattr(raw, "requires_to_me", False)),
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


def infer_command_role(command: str, *, family: str = "general") -> str:
    normalized = normalize_message_text(command).lower()
    if not normalized:
        return "other"
    if any(
        token in normalized
        for token in ("帮助", "help", "用法", "说明", "详情", "参数")
    ):
        return "help"
    if any(
        token in normalized
        for token in (
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
    ):
        return "create"
    if any(token in normalized for token in ("开", "抢", "领取", "领", "抽签")):
        return "open"
    if any(
        token in normalized
        for token in ("退回", "退还", "删除", "取消", "解绑", "关闭")
    ):
        return "return"
    if any(
        token in normalized
        for token in (
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
        )
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
            hint in param_l
            for hint in (
                "num",
                "count",
                "quantity",
                "number",
                "红包数",
                "数量",
                "个数",
                "份数",
                "数目",
            )
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

    fuzzy_boundary = _find_short_noise_head_boundary(normalized, command_head)
    if fuzzy_boundary is not None and fuzzy_boundary < len(normalized):
        fuzzy_tail = normalize_message_text(
            normalized[fuzzy_boundary:].strip(" ：:，,。.!！?？")
        )
        fuzzy_tail = _strip_route_noise(fuzzy_tail)
        if fuzzy_tail:
            return fuzzy_tail

    index = normalized.find(command_head)
    if index < 0:
        return ""

    before = normalize_message_text(normalized[:index].strip(" ：:，,。.!！?？"))
    after = normalize_message_text(
        normalized[index + len(command_head) :].strip(" ：:，,。.!！?？")
    )
    before_cleaned = _strip_route_noise(before)
    after_cleaned = _strip_route_noise(after)

    def _is_usable_argument_fragment(fragment: str) -> bool:
        cleaned = normalize_message_text(fragment)
        if not cleaned:
            return False
        if _message_has_payload_signals(cleaned):
            return True
        if contains_any(cleaned, ROUTE_ACTION_WORDS):
            return False
        if cleaned in _ROUTE_NOISE_WORDS:
            return False
        return len(cleaned) > 2

    if before_cleaned and not _is_usable_argument_fragment(before_cleaned):
        before_cleaned = ""
    if after_cleaned and not _is_usable_argument_fragment(after_cleaned):
        after_cleaned = ""

    if before_cleaned and after_cleaned:
        combined = normalize_message_text(f"{before_cleaned} {after_cleaned}")
        if _message_has_payload_signals(combined):
            return combined

    if after_cleaned:
        return after_cleaned

    if before_cleaned and len(before_cleaned) <= 30:
        return before_cleaned
    return ""


__all__ = [
    "SkillCommandSchema",
    "SkillRegistry",
    "SkillRouteDecision",
    "SkillSearchResult",
    "SkillSpec",
    "get_skill_registry",
    "infer_command_role",
    "infer_query_families",
    "skill_search",
]
