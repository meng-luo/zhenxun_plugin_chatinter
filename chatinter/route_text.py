import re

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase

_WHITESPACE_PATTERN = re.compile(r"\s+")
_PLACEHOLDER_PATTERN = re.compile(
    r"\[@(?:\d{5,20}|所有人)\]|\[image(?:#\d+)?\]|(?<![0-9A-Za-z_])@\d{5,20}(?=(?:\s|$|[的，,。.!！？?]))",
    re.IGNORECASE,
)

INVOKE_PREFIXES = (
    "小真寻",
    "真寻",
    "机器人",
    "bot",
    "帮我",
    "帮忙",
    "麻烦",
    "给我",
    "请",
    "制作一个",
    "制作一张",
    "做一个",
    "做一张",
    "做个",
    "来一个",
    "来一张",
    "来个",
    "再来一个",
    "再来一张",
    "再来个",
)

QUESTION_WORDS = (
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
    "怎么触发",
    "如何触发",
    "怎样触发",
    "怎么调用",
    "如何调用",
    "怎样调用",
    "怎么做",
    "如何做",
    "怎样做",
    "是什么",
    "是啥",
    "什么意思",
    "?",
    "？",
)

USAGE_WORDS = (
    "配置",
    "怎么配置",
    "如何配置",
    "怎样配置",
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
    "怎么触发",
    "如何触发",
    "怎样触发",
    "怎么调用",
    "如何调用",
    "怎样调用",
    "怎么做",
    "如何做",
    "怎样做",
    "是什么",
    "是啥",
    "什么意思",
)

_USAGE_CONTEXT_HINTS = (
    "命令",
    "插件",
    "功能",
    "用法",
    "参数",
    "说明",
    "教程",
    "触发",
    "调用",
    "配置",
    "详情",
    "搜索",
    "列表",
)

_GENERIC_QUESTION_WORDS = (
    "怎么",
    "如何",
    "怎样",
    "啥",
    "什么",
    "能否",
    "能不能",
    "可以吗",
    "为什么",
    "?",
    "？",
)

EXECUTE_WORDS = (
    "帮我",
    "请",
    "麻烦",
    "执行",
    "调用",
    "打开",
    "关闭",
    "开启",
    "禁用",
    "设置",
    "看看",
    "看下",
    "生成",
    "制作",
    "点歌",
    "播放",
    "签到",
    "启动",
    "来个",
    "来一张",
    "做个",
    "做一张",
    "做一个",
    "发送",
)

STRONG_EXECUTE_WORDS = (
    "帮我",
    "请",
    "麻烦",
    "执行",
    "调用",
    "给我",
    "来个",
    "来一张",
    "来一个",
    "再来个",
    "再来一张",
    "再来一个",
    "做个",
    "做一张",
    "做一个",
    "生成",
    "制作",
    "发送",
    "启动",
    "打开",
    "关闭",
    "开启",
    "禁用",
    "设置",
    "查",
    "查询",
    "看看",
    "看下",
    "点播",
)

ROUTE_ACTION_WORDS = (
    "帮我",
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
    "点歌",
    "播放",
    "签到",
    "启动",
    "来个",
    "做个",
    "做一张",
    "做一个",
)

ROUTE_NEGATIVE_HINT_WORDS = (
    "不是在让你执行",
    "不是让你执行",
    "不是让你",
    "不是叫你",
    "不是命令",
    "别执行",
    "只是提到",
    "只是说说",
    "只是聊",
    "只是问问",
    "只是讨论",
    "我在讨论",
    "我在聊",
    "我在想",
    "我只是问",
    "只是问一下",
    "只是想知道",
    "这个词",
    "听起来怎么样",
    "写到签名里",
    "写在签名里",
    "有人在讨论",
    "听到别人念了句",
    "看到有人在讨论",
    "什么意思",
    "是什么意思",
    "看到有人说",
    "听到有人说",
)

TEMPLATE_ROUTE_HINT_WORDS = (
    "表情",
    "表情包",
    "梗图",
    "meme",
    "模板",
    "头像",
    "贴图",
    "做图",
    "制图",
    "图片操作",
    "表情详情",
    "表情搜索",
    "随机表情",
    "[image",
    "[@",
)

MEME_TRIGGER_WORDS = (
    "表情包",
    "表情",
    "梗图",
    "meme",
    "做一张",
    "生成一张",
    "来一张",
    "来一个",
    "再来一张",
    "再来一个",
    "制作一张",
    "做个",
    "做一个",
    "做张",
    "制作一个",
    "来个",
    "再来个",
    "做图",
    "生成图",
    "制作",
    "启动",
)

KNOWLEDGE_REFRESH_WORDS = (
    "表情",
    "梗图",
    "meme",
    "命令",
    "插件",
    "调用",
    "怎么用",
    "如何用",
    "生成",
    "制作",
    "点歌",
    "签到",
    "启动",
    "开关",
)

ACTION_REWRITES = (
    ("点一首", "点歌 "),
    ("点首", "点歌 "),
    ("来一首", "点歌 "),
    ("来首", "点歌 "),
    ("播一首", "点歌 "),
    ("播首", "点歌 "),
    ("签个到", "签到"),
    ("签一下到", "签到"),
    ("签个", "签到"),
)

_TEMPLATE_TAIL_NOISE_WORDS = (
    "表情包",
    "表情",
    "梗图",
    "meme",
    "图片",
    "模板",
)

_TEMPLATE_TAIL_LEADING_NOISE_WORDS = (
    "的",
    "一个",
    "一张",
    "个",
    "张",
    "这个",
    "这张",
    "那个",
    "那张",
)


def normalize_message_text(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", (text or "").strip())


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = normalize_message_text(text).lower()
    if not normalized:
        return False
    return any(keyword.lower() in normalized for keyword in keywords)


def match_command_head(text: str, command: str) -> bool:
    normalized_text = normalize_message_text(text)
    normalized_command = normalize_message_text(command)
    if not normalized_text or not normalized_command:
        return False
    text_fold = normalized_text.casefold()
    command_fold = normalized_command.casefold()
    if text_fold == command_fold:
        return True
    if text_fold.startswith(command_fold):
        if len(normalized_text) == len(normalized_command):
            return True
        next_char = normalized_text[len(normalized_command)]
        return next_char.isspace()
    return False


def strip_invoke_prefix(text: str) -> str:
    stripped = normalize_message_text(text)
    while stripped:
        matched = next(
            (
                prefix
                for prefix in INVOKE_PREFIXES
                if stripped.lower().startswith(prefix.lower())
            ),
            None,
        )
        if matched is None:
            return stripped
        stripped = normalize_message_text(stripped[len(matched) :])
    return stripped


def normalize_action_phrases(text: str) -> str:
    normalized = normalize_message_text(text)
    if not normalized:
        return normalized
    for source, replacement in ACTION_REWRITES:
        normalized = normalized.replace(source, replacement)
    return normalize_message_text(normalized)


def is_usage_question(text: str) -> bool:
    normalized = normalize_message_text(text)
    if not normalized:
        return False
    has_question_hint = contains_any(normalized, QUESTION_WORDS)
    has_explicit_usage_hint = contains_any(normalized, USAGE_WORDS)
    if not has_question_hint and not has_explicit_usage_hint:
        return False
    if has_explicit_usage_hint:
        return True
    # 仅出现“什么/怎么/？”等泛问句，不视为插件帮助请求。
    if contains_any(normalized, _GENERIC_QUESTION_WORDS) and not contains_any(
        normalized, _USAGE_CONTEXT_HINTS
    ):
        return False
    if contains_any(normalized, STRONG_EXECUTE_WORDS):
        return False
    if contains_any(normalized, EXECUTE_WORDS):
        return False
    return has_question_hint


def collect_placeholders(text: str) -> list[str]:
    deduplicated: list[str] = []
    for match in _PLACEHOLDER_PATTERN.finditer(text or ""):
        token = match.group(0).strip()
        if token and token not in deduplicated:
            deduplicated.append(token)
    return deduplicated


def has_negative_route_intent(text: str) -> bool:
    return contains_any(text, ROUTE_NEGATIVE_HINT_WORDS)


def _is_meme_plugin(plugin: PluginInfo) -> bool:
    module_l = plugin.module.lower()
    name_l = plugin.name.lower()
    if "meme" in module_l:
        return True
    if "表情" in name_l:
        return True
    command_text = " ".join(str(command).lower() for command in plugin.commands)
    if "表情搜索" in command_text and "表情详情" in command_text:
        return True
    return False


def _is_template_like_plugin(plugin: PluginInfo) -> bool:
    if _is_meme_plugin(plugin):
        return True
    command_text = " ".join(normalize_message_text(command) for command in plugin.commands)
    usage_text = normalize_message_text(plugin.usage or "")
    hint_text = f"{plugin.name} {plugin.description} {command_text} {usage_text}".lower()
    has_catalog_hint = any(word in hint_text for word in ("搜索", "详情", "列表"))
    has_template_hint = any(
        word in hint_text
        for word in (
            "模板",
            "表情",
            "梗图",
            "图片",
            "头像",
            "文字",
            "文本",
            "内容",
            "标题",
        )
    )
    return has_catalog_hint and has_template_hint


def has_template_route_context(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> bool:
    normalized = normalize_message_text(normalize_action_phrases(message_text))
    if not normalized:
        return False
    if has_negative_route_intent(normalized):
        return False

    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    has_template_hint = contains_any(normalized, TEMPLATE_ROUTE_HINT_WORDS)
    has_placeholder = "[image" in normalized or "[@" in normalized
    has_route_action = contains_any(normalized, ROUTE_ACTION_WORDS)
    has_meme_trigger = contains_any(normalized, MEME_TRIGGER_WORDS)
    matched_template = False
    matched_non_template = False

    for plugin in knowledge_base.plugins:
        template_plugin = _is_template_like_plugin(plugin)
        for command in plugin.commands:
            cmd = normalize_message_text(command)
            if len(cmd) < 2:
                continue
            if match_command_head(stripped, cmd) or match_command_head(normalized, cmd):
                if template_plugin:
                    matched_template = True
                else:
                    matched_non_template = True

    if not (has_template_hint or has_placeholder or has_meme_trigger or matched_template):
        return False
    if matched_non_template and not (has_template_hint or has_placeholder or has_meme_trigger):
        return False

    if has_template_hint or has_placeholder or has_meme_trigger:
        return has_route_action or has_placeholder or matched_template
    return matched_template


def sanitize_template_tail(tail: str) -> str:
    cleaned = normalize_message_text(_PLACEHOLDER_PATTERN.sub(" ", tail or ""))
    cleaned = cleaned.strip(" ：:，,。.!！?？;；-")
    if not cleaned:
        return ""

    leading_tokens = _TEMPLATE_TAIL_LEADING_NOISE_WORDS + _TEMPLATE_TAIL_NOISE_WORDS
    while cleaned:
        updated = cleaned
        for token in leading_tokens:
            if cleaned == token:
                return ""
            prefix = f"{token} "
            if cleaned.startswith(prefix):
                updated = cleaned[len(prefix) :].strip(" ：:，,。.!！?？;；-")
                break
        if updated == cleaned:
            break
        cleaned = updated

    while cleaned:
        updated = cleaned
        for token in _TEMPLATE_TAIL_NOISE_WORDS:
            if cleaned == token:
                return ""
            suffix = f" {token}"
            if cleaned.endswith(suffix):
                updated = cleaned[: -len(suffix)].strip(" ：:，,。.!！?？;；-")
                break
        if updated == cleaned:
            break
        cleaned = updated

    return cleaned


def _is_explicit_command_request(text: str, commands: list[str]) -> bool:
    if not text or not commands:
        return False
    normalized = normalize_message_text(normalize_action_phrases(text))
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    lowered = normalized.lower()
    for command in commands:
        cmd = normalize_message_text(command)
        if not cmd:
            continue
        if match_command_head(stripped, cmd) or match_command_head(normalized, cmd):
            return True
        if len(cmd) >= 2 and cmd.lower() in lowered:
            return True
    return False


def should_force_knowledge_refresh(
    message_text: str,
    knowledge_base: PluginKnowledgeBase,
) -> bool:
    normalized_message = normalize_message_text(message_text)
    if not normalized_message:
        return False

    plugin_count = len(knowledge_base.plugins)
    if plugin_count <= 2 and contains_any(normalized_message, KNOWLEDGE_REFRESH_WORDS):
        return True

    if contains_any(normalized_message, MEME_TRIGGER_WORDS) and not any(
        _is_template_like_plugin(plugin) for plugin in knowledge_base.plugins
    ):
        return True

    if plugin_count <= 8 and contains_any(normalized_message, STRONG_EXECUTE_WORDS):
        all_commands = [
            cmd.strip()
            for plugin in knowledge_base.plugins
            for cmd in plugin.commands
            if cmd and cmd.strip()
        ]
        if all_commands and not _is_explicit_command_request(
            normalized_message, all_commands
        ):
            return True

    return False


__all__ = [
    "ROUTE_ACTION_WORDS",
    "collect_placeholders",
    "contains_any",
    "has_negative_route_intent",
    "has_template_route_context",
    "is_usage_question",
    "match_command_head",
    "normalize_action_phrases",
    "normalize_message_text",
    "sanitize_template_tail",
    "should_force_knowledge_refresh",
    "strip_invoke_prefix",
]
