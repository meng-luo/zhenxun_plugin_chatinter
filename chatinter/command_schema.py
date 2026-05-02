"""Command-level schemas for ChatInter routing.

插件命令最终仍走原 NoneBot matcher；这里仅把自然语言意图转换为稳定的
command_id + slots，再确定性渲染回原命令文本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .command_alias import derive_semantic_aliases, is_shadowed_meme_head
from .models.pydantic_models import (
    CommandCapability,
    CommandSlotSpec,
    PluginCommandSchema,
)
from .route_text import normalize_message_text, parse_command_with_head
from .slot_extractors import extract_builtin_slots

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUMBER_TEXT = r"\d+|[零〇一二两俩三四五六七八九十百千]+"
_TEXT_PLACEHOLDER_PATTERN = re.compile(r"\{(?P<name>[A-Za-z_][0-9A-Za-z_]*)\}")
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+")
_URL_PAYLOAD_PATTERN = re.compile(
    r"https?://\S+|(?:BV|AV|av)[0-9A-Za-z]+",
    re.IGNORECASE,
)
_TEXT_TAIL_PREFIX_PATTERN = re.compile(
    r"^(?:这句话|这段话|这句|内容|文本|参数|链接|地址|是|为|叫|名称|名字|"
    r"：|:|-|，|,|。)+"
)


@dataclass(frozen=True)
class CommandSchemaSelection:
    """A scored schema choice.

    The selector keeps command choice deterministic when several schemas share the
    same head, while still accepting LLM-provided command_id as the strongest hint.
    """

    schema: PluginCommandSchema
    score: float
    reason: str
    slots: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()


def _slot(
    name: str,
    slot_type: str = "text",
    *,
    required: bool = False,
    default: Any = None,
    aliases: list[str] | None = None,
    description: str = "",
) -> CommandSlotSpec:
    return CommandSlotSpec(
        name=name,
        type=slot_type,  # type: ignore[arg-type]
        required=required,
        default=default,
        aliases=list(aliases or []),
        description=description,
    )


def _schema(
    command_id: str,
    head: str,
    *,
    aliases: list[str] | None = None,
    description: str = "",
    slots: list[CommandSlotSpec] | None = None,
    render: str | None = None,
    requires: dict[str, bool] | None = None,
    command_role: str = "execute",
    payload_policy: str = "none",
    extra_text_policy: str = "keep",
    source: str = "override",
    confidence: float = 0.85,
    matcher_key: str | None = None,
    retrieval_phrases: list[str] | None = None,
) -> PluginCommandSchema:
    normalized_head = normalize_message_text(head)
    normalized_aliases = [
        text
        for text in (normalize_message_text(alias) for alias in list(aliases or []))
        if text
    ]
    normalized_description = normalize_message_text(description)
    phrase_values = [
        normalized_head,
        *normalized_aliases,
        normalized_description,
        command_id,
        *(retrieval_phrases or []),
    ]
    phrases: list[str] = []
    for value in phrase_values:
        text = normalize_message_text(value)
        if text and text not in phrases:
            phrases.append(text)
    return PluginCommandSchema(
        command_id=command_id,
        head=normalized_head or head,
        aliases=list(dict.fromkeys(normalized_aliases)),
        description=normalized_description,
        slots=list(slots or []),
        render=render or head,
        requires={
            "text": False,
            "image": False,
            "reply": False,
            "at": False,
            **dict(requires or {}),
        },
        command_role=command_role,  # type: ignore[arg-type]
        payload_policy=payload_policy,  # type: ignore[arg-type]
        extra_text_policy=extra_text_policy,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
        matcher_key=matcher_key,
        retrieval_phrases=phrases,
    )


_SCHEMA_OVERRIDES: dict[str, list[PluginCommandSchema]] = {
    "zhenxun.plugins.gold_redbag": [
        _schema(
            "gold_redbag.send",
            "塞红包",
            aliases=[
                "金币红包",
                "发红包",
                "塞金币红包",
                "给群里发红包",
                "发金币红包",
                "给大家发红包",
            ],
            description=("发送金币红包；用于发/塞红包，amount=总金币，num=红包个数"),
            slots=[
                _slot(
                    "amount",
                    "int",
                    required=True,
                    aliases=["金额", "金币", "总额"],
                    description="红包总金币数",
                ),
                _slot(
                    "num",
                    "int",
                    default=5,
                    aliases=["数量", "红包数", "个", "份"],
                    description="红包个数，默认 5",
                ),
            ],
            render="塞红包 {amount} {num}",
            requires={"text": True},
            payload_policy="slots",
            extra_text_policy="slot_only",
        ),
        _schema(
            "gold_redbag.open",
            "开",
            aliases=["抢", "开红包", "抢红包", "我想抢红包", "领红包"],
            description="打开/抢/领取当前群可领取的红包；不发送新红包",
            render="开",
            extra_text_policy="discard",
        ),
        _schema(
            "gold_redbag.return",
            "退回红包",
            aliases=["退还红包", "红包退回", "没领完的红包退回", "退回没领完红包"],
            description="退回自己发出且未领取完的红包；不是抢红包",
            render="退回红包",
            extra_text_policy="discard",
        ),
    ],
    "zhenxun.plugins.roll": [
        _schema(
            "roll.choose",
            "roll",
            aliases=[
                "随机选",
                "帮我选",
                "从里面选",
                "选择困难",
                "二选一",
                "选一个",
                "挑一个",
                "做个选择",
                "帮我决定",
            ],
            description="从给定多个候选项中随机选择一个；需要 options",
            slots=[
                _slot(
                    "options",
                    "text",
                    required=True,
                    aliases=["选项", "候选"],
                    description="用空格分隔的候选项",
                )
            ],
            render="roll {options}",
            requires={"text": True},
            payload_policy="slots",
            extra_text_policy="slot_only",
        ),
        _schema(
            "roll.number",
            "roll",
            aliases=["随机数字", "掷骰子", "roll点", "随机一个数字", "投个随机数字"],
            description="随机生成数字/骰子点数；不需要候选项文本",
            render="roll",
            extra_text_policy="discard",
        ),
    ],
    "zhenxun.plugins.poetry": [
        _schema(
            "poetry.random",
            "古诗",
            aliases=[
                "念诗",
                "来首诗",
                "念首诗",
                "给我念一首诗",
                "来一首古诗",
                "来首古诗",
                "诗词",
            ],
            description="随机发送一首古诗词",
            render="念诗",
        )
    ],
    "zhenxun.plugins.cover": [
        _schema(
            "cover.bilibili",
            "b封面",
            aliases=["B站封面", "视频封面", "查视频封面"],
            description="获取 B 站视频或直播封面",
            slots=[
                _slot(
                    "target",
                    "text",
                    required=True,
                    aliases=["链接", "BV号", "av号", "直播id"],
                )
            ],
            render="b封面 {target}",
            requires={"text": True},
            payload_policy="slots",
            extra_text_policy="slot_only",
        )
    ],
    "zhenxun.plugins.translate": [
        _schema(
            "translate.text",
            "翻译",
            aliases=["翻译一下", "翻成中文", "翻译成中文", "帮我翻译", "用中文说一下"],
            description="翻译给定文本；需要 text，不用于查看支持语种",
            slots=[_slot("text", "text", required=True, aliases=["文本", "内容"])],
            render="翻译 {text}",
            requires={"text": True},
            payload_policy="text",
            extra_text_policy="slot_only",
        ),
        _schema(
            "translate.langs",
            "翻译语种",
            aliases=["翻译语种", "支持哪些语言", "翻译支持什么语言"],
            description="查看翻译插件支持的语言列表；不是执行翻译",
            render="翻译语种",
            command_role="helper",
            extra_text_policy="discard",
        ),
    ],
    "zhenxun.plugins.luxun": [
        _schema(
            "luxun.say",
            "鲁迅说",
            aliases=["鲁迅风格", "来张鲁迅说", "让鲁迅说"],
            description="生成鲁迅说图片",
            slots=[_slot("text", "text", required=True, aliases=["内容", "文本"])],
            render="鲁迅说 {text}",
            requires={"text": True},
            payload_policy="text",
            extra_text_policy="slot_only",
        )
    ],
    "zhenxun.plugins.nbnhhsh": [
        _schema(
            "nbnhhsh.expand",
            "能不能好好说话",
            aliases=["nbnhhsh", "解释缩写", "缩写是什么意思"],
            description="解释网络缩写",
            slots=[_slot("text", "text", required=True, aliases=["缩写", "文本"])],
            render="能不能好好说话 {text}",
            requires={"text": True},
            payload_policy="text",
            extra_text_policy="slot_only",
        )
    ],
    "zhenxun.plugins.quotations": [
        _schema(
            "quotations.hitokoto",
            "语录",
            aliases=["来一句语录", "一言"],
            description="随机发送一句语录",
            render="语录",
        ),
        _schema(
            "quotations.acg",
            "二次元",
            aliases=["二次元语录", "来一句二次元语录"],
            description="随机发送一句二次元语录",
            render="二次元",
        ),
    ],
    "zhenxun.builtin_plugins.about": [
        _schema(
            "about.info",
            "关于",
            aliases=[
                "about",
                "真寻信息",
                "小真寻信息",
                "小真寻的信息",
                "了解小真寻",
                "想了解小真寻",
                "机器人信息",
                "bot信息",
                "项目介绍",
                "项目说明",
                "介绍真寻",
            ],
            description="查看真寻项目、版本和帮助入口",
            render="关于",
            command_role="helper",
            extra_text_policy="discard",
        )
    ],
}


def _command_id(module: str, head: str) -> str:
    safe_module = re.sub(r"[^0-9A-Za-z_]+", "_", module.rsplit(".", 1)[-1])
    safe_head = re.sub(r"\s+", "_", normalize_message_text(head))
    return f"{safe_module}.{safe_head or 'command'}"


def _requires_from_capability(command: CommandCapability) -> dict[str, bool]:
    requirement = command.requirement
    params = [
        normalize_message_text(str(param or "")).lower()
        for param in requirement.params
        if normalize_message_text(str(param or ""))
    ]
    internal_media_params = {"meme_params", "img", "image", "images", "图片"}
    params_require_text = bool(params) and not (
        requirement.text_min <= 0
        and requirement.image_min > 0
        and all(param in internal_media_params for param in params)
    )
    return {
        "text": bool(requirement.text_min > 0 or params_require_text),
        "image": bool(requirement.image_min > 0),
        "reply": bool(requirement.requires_reply),
        "private": bool(requirement.requires_private),
        "to_me": bool(requirement.requires_to_me),
        "at": bool(
            requirement.allow_at
            or "at" in requirement.target_sources
            or requirement.target_requirement == "required"
        ),
    }


def _is_internal_media_param(name: str, requirement: Any) -> bool:
    normalized = normalize_message_text(name).lower()
    if normalized not in {"meme_params", "img", "image", "images", "图片"}:
        return False
    return (
        max(int(getattr(requirement, "text_min", 0) or 0), 0) <= 0
        and max(int(getattr(requirement, "image_min", 0) or 0), 0) > 0
    )


def _payload_policy_from_capability(command: CommandCapability) -> tuple[str, str]:
    requirement = command.requirement
    if requirement.image_min > 0 and requirement.text_min <= 0:
        return "image_only", "discard"
    if requirement.text_min > 0:
        return "text", "slot_only"
    if requirement.params:
        return "slots", "slot_only"
    return "none", "keep"


def _slot_type_from_name(name: str) -> str:
    normalized = normalize_message_text(name).lower()
    if any(
        token in normalized
        for token in (
            "num",
            "count",
            "amount",
            "金币",
            "数量",
            "金额",
            "次数",
            "份",
            "个数",
        )
    ):
        return "int"
    if any(token in normalized for token in ("image", "图片", "图", "照片")):
        return "image"
    if any(token in normalized for token in ("at", "user", "用户", "目标", "对象")):
        return "at"
    return "text"


def _slot_aliases_from_name(name: str) -> list[str]:
    normalized = normalize_message_text(name)
    alias_map = {
        "amount": ["金额", "金币", "总额"],
        "num": ["数量", "个数", "份数"],
        "count": ["数量", "次数", "个数"],
        "text": ["文本", "内容"],
        "content": ["文本", "内容"],
        "target": ["目标", "对象"],
        "image": ["图片", "图"],
    }
    aliases = [normalized] if normalized else []
    for key, values in alias_map.items():
        if key in normalized.lower() or normalized in values:
            aliases.extend(values)
    result: list[str] = []
    for alias in aliases:
        text = normalize_message_text(alias)
        if text and text not in result:
            result.append(text)
    return result


def _slot_description(name: str, slot_type: str) -> str:
    normalized = normalize_message_text(name)
    if not normalized:
        return ""
    if slot_type == "int":
        return f"{normalized}，通常填写数字"
    if slot_type == "image":
        return f"{normalized}，需要图片上下文"
    if slot_type == "at":
        return f"{normalized}，需要@、回复或昵称目标"
    return f"{normalized}文本"


def _command_description(command: CommandCapability, head: str) -> str:
    parts: list[str] = []
    examples = [
        normalize_message_text(example)
        for example in command.examples[:2]
        if normalize_message_text(example)
    ]
    if examples:
        parts.append("示例: " + " / ".join(examples))
    requirement = command.requirement
    requirement_parts: list[str] = []
    if requirement.params:
        requirement_parts.append("参数: " + " ".join(requirement.params[:4]))
    if requirement.text_min > 0:
        requirement_parts.append(f"至少{requirement.text_min}段文本")
    if requirement.image_min > 0:
        requirement_parts.append(f"至少{requirement.image_min}张图片")
    if requirement.requires_reply:
        requirement_parts.append("需要回复上下文")
    if requirement.target_requirement == "required":
        requirement_parts.append("需要明确目标")
    if requirement_parts:
        parts.append("；".join(requirement_parts))
    if not parts:
        parts.append(f"执行“{head}”命令")
    description = "；".join(parts)
    return description[:120].rstrip()


def _schema_from_capability(
    module: str,
    command: CommandCapability,
) -> PluginCommandSchema | None:
    head = normalize_message_text(command.command)
    if not head:
        return None
    slots: list[CommandSlotSpec] = []
    requirement = command.requirement
    raw_params = [
        normalize_message_text(str(param or ""))
        for param in requirement.params
        if normalize_message_text(str(param or ""))
    ]
    raw_params = [
        param
        for param in raw_params
        if not _is_internal_media_param(param, requirement)
    ]
    if not raw_params and requirement.text_min > 0:
        raw_params = ["text"]
    for index, slot_name in enumerate(raw_params[:4]):
        slot_type = _slot_type_from_name(slot_name)
        slots.append(
            _slot(
                slot_name,
                slot_type,
                required=requirement.text_min > index,
                aliases=_slot_aliases_from_name(slot_name),
                description=_slot_description(slot_name, slot_type),
            )
        )
    render = head
    if slots:
        render = " ".join([head, *[f"{{{slot.name}}}" for slot in slots]])
    payload_policy, extra_text_policy = _payload_policy_from_capability(command)
    aliases = [
        *command.aliases,
        *derive_semantic_aliases(
            head,
            module=module,
            image_required=requirement.image_min > 0,
        ),
    ]
    return _schema(
        _command_id(module, head),
        head,
        aliases=list(dict.fromkeys(alias for alias in aliases if alias)),
        description=_command_description(command, head),
        slots=slots,
        render=render,
        requires=_requires_from_capability(command),
        command_role="template" if requirement.image_min > 0 else "execute",
        payload_policy=payload_policy,
        extra_text_policy=extra_text_policy,
        source="matcher",
        confidence=0.68,
        matcher_key=f"{module}:{head}",
    )


def _build_meme_schemas(
    module: str,
    commands: list[CommandCapability],
) -> list[PluginCommandSchema]:
    schemas: list[PluginCommandSchema] = [
        _schema(
            "memes.list",
            "表情包制作",
            aliases=[
                "表情列表",
                "表情包列表",
                "头像表情包",
                "文字表情包",
                "有哪些表情包",
            ],
            description="查看可制作的表情包列表；列表/有哪些/打开表情包时执行",
            render="表情包制作",
            command_role="catalog",
            payload_policy="none",
            extra_text_policy="discard",
        ),
        _schema(
            "memes.search",
            "表情搜索",
            aliases=["搜索表情", "找表情", "查找表情", "找相关表情"],
            description="按关键词搜索相关表情包模板",
            slots=[
                _slot(
                    "keyword",
                    "text",
                    required=True,
                    aliases=["关键词", "表情名"],
                    description="要搜索的表情关键词",
                )
            ],
            render="表情搜索 {keyword}",
            requires={"text": True},
            command_role="helper",
            payload_policy="slots",
            extra_text_policy="slot_only",
        ),
        _schema(
            "memes.info",
            "表情详情",
            aliases=["表情用法", "表情参数", "这个表情怎么用"],
            description="查看某个表情的参数、预览和用法",
            slots=[
                _slot(
                    "keyword",
                    "text",
                    required=True,
                    aliases=["关键词", "表情名"],
                    description="要查看详情的表情关键词",
                )
            ],
            render="表情详情 {keyword}",
            requires={"text": True},
            command_role="usage",
            payload_policy="slots",
            extra_text_policy="slot_only",
        ),
        _schema(
            "memes.random",
            "随机表情",
            aliases=["随机做个表情", "随机表情包", "随便做个表情"],
            description="使用当前图片/文字随机制作一个表情包",
            render="随机表情",
            requires={"image": False, "text": False},
            command_role="random",
            payload_policy="text_or_image",
            extra_text_policy="discard",
        ),
    ]
    seen = {schema.command_id for schema in schemas}
    for command in commands:
        head = normalize_message_text(command.command)
        if not head or is_shadowed_meme_head(head):
            continue
        schema = _schema_from_capability(module, command)
        if schema is None or schema.command_id in seen:
            continue
        seen.add(schema.command_id)
        schemas.append(schema)
    return schemas


def build_command_schemas(
    module: str,
    commands: list[CommandCapability],
) -> list[PluginCommandSchema]:
    module_key = normalize_message_text(module)
    overrides = _SCHEMA_OVERRIDES.get(module_key)
    if overrides:
        return [schema.model_copy(deep=True) for schema in overrides]
    if module_key.endswith("nonebot_plugin_memes"):
        return _build_meme_schemas(module_key, commands)

    schemas: list[PluginCommandSchema] = []
    seen: set[str] = set()
    for command in commands:
        schema = _schema_from_capability(module_key, command)
        if schema is None or schema.command_id in seen:
            continue
        seen.add(schema.command_id)
        schemas.append(schema)
    return schemas


def _command_head(command: str | None) -> str:
    normalized = normalize_message_text(command or "")
    return normalize_message_text(normalized.split(" ", 1)[0]) if normalized else ""


def _command_tail(command: str | None) -> str:
    normalized = normalize_message_text(command or "")
    if not normalized or " " not in normalized:
        return ""
    return normalize_message_text(normalized.split(" ", 1)[1])


def _tokenize(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(normalize_message_text(text))
        if token
    }


def _schema_phrases(schema: PluginCommandSchema) -> list[str]:
    values = [
        schema.command_id,
        schema.head,
        *schema.aliases,
        schema.description,
        *schema.retrieval_phrases,
    ]
    for slot in schema.slots:
        values.extend([slot.name, slot.description, *slot.aliases])
    result: list[str] = []
    for value in values:
        text = normalize_message_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _schema_render_slots(schema: PluginCommandSchema) -> set[str]:
    return {
        normalize_message_text(match.group("name"))
        for match in _TEXT_PLACEHOLDER_PATTERN.finditer(schema.render or "")
        if normalize_message_text(match.group("name"))
    }


def _message_has_text_payload(text: str) -> bool:
    normalized = normalize_message_text(text)
    if not normalized:
        return False
    if re.search(r"\d", normalized):
        return True
    if len(normalized) >= 4:
        return True
    return bool(_tokenize(normalized))


def _score_schema(
    schema: PluginCommandSchema,
    *,
    command_id: str | None,
    command: str | None,
    message_text: str,
    arguments_text: str,
    slots: dict[str, Any] | None,
    action: str | None,
) -> CommandSchemaSelection:
    normalized_id = normalize_message_text(command_id or "").casefold()
    schema_id = normalize_message_text(schema.command_id).casefold()
    command_text = normalize_message_text(command or "")
    command_head = _command_head(command_text).casefold()
    command_tail = _command_tail(command_text)
    message = normalize_message_text(" ".join([message_text, arguments_text]))
    message_fold = message.casefold()
    action_text = normalize_message_text(action or "").casefold()
    score = 0.0
    reasons: list[str] = []

    if normalized_id:
        if normalized_id == schema_id:
            score += 1000.0
            reasons.append("command_id")
        else:
            score -= 24.0

    head = normalize_message_text(schema.head).casefold()
    aliases = [
        normalize_message_text(alias).casefold()
        for alias in schema.aliases
        if normalize_message_text(alias)
    ]
    if command_head:
        if command_head == head:
            score += 420.0
            reasons.append("head")
        elif command_head in aliases:
            score += 380.0
            reasons.append("alias_head")
        elif head and command_text.casefold().startswith(head):
            score += 260.0
            reasons.append("head_prefix")
        elif any(
            alias and command_text.casefold().startswith(alias) for alias in aliases
        ):
            score += 220.0
            reasons.append("alias_prefix")

    if message_fold:
        for alias in aliases:
            if alias and alias in message_fold:
                score += 130.0 + min(len(alias), 16)
                reasons.append("message_alias")
        if head and head in message_fold:
            score += 84.0 + min(len(head), 12)
            reasons.append("message_head")

        message_tokens = _tokenize(message)
        phrase_tokens: set[str] = set()
        for phrase in _schema_phrases(schema):
            phrase_tokens.update(_tokenize(phrase))
        overlap = len(message_tokens & phrase_tokens)
        if overlap:
            score += min(overlap * 10.0, 60.0)
            reasons.append("token_overlap")

    completed_slots, missing = complete_slots(
        schema,
        slots=slots,
        message_text=message_text,
        arguments_text=arguments_text or command_tail,
    )
    provided_slot_names = {
        normalize_message_text(str(name or ""))
        for name, value in dict(slots or {}).items()
        if normalize_message_text(str(name or "")) and value is not None
    }
    schema_slot_names = {normalize_message_text(slot.name) for slot in schema.slots}
    matched_provided = len(provided_slot_names & schema_slot_names)
    if matched_provided:
        score += matched_provided * 72.0
        reasons.append("slots")

    render_slots = _schema_render_slots(schema)
    completed_render_slots = {
        name
        for name in render_slots
        if name in completed_slots and completed_slots.get(name) is not None
    }
    if completed_render_slots:
        score += len(completed_render_slots) * 28.0
        reasons.append("completed_slots")

    if missing:
        penalty = 18.0 if action_text == "clarify" else 86.0
        score -= len(missing) * penalty
        reasons.append("missing")

    requires = schema.requires or {}
    has_payload = _message_has_text_payload(
        " ".join([command_tail, arguments_text, message_text])
    )
    if requires.get("text") and has_payload:
        score += 18.0
    elif not requires.get("text") and command_tail:
        score -= 16.0

    if not schema.slots and not requires.get("text") and action_text == "execute":
        score += 3.0

    return CommandSchemaSelection(
        schema=schema,
        score=score,
        reason=",".join(dict.fromkeys(reasons)) or "fallback",
        slots=completed_slots,
        missing=tuple(missing),
    )


def select_command_schema(
    schemas: list[PluginCommandSchema],
    *,
    command_id: str | None = None,
    command: str | None = None,
    message_text: str = "",
    arguments_text: str = "",
    slots: dict[str, Any] | None = None,
    action: str | None = None,
) -> CommandSchemaSelection | None:
    if not schemas:
        return None
    has_hint = any(
        normalize_message_text(value or "")
        for value in (command_id, command, message_text, arguments_text)
    ) or bool(slots)
    if not has_hint:
        return None

    selections = [
        _score_schema(
            schema,
            command_id=command_id,
            command=command,
            message_text=message_text,
            arguments_text=arguments_text,
            slots=slots,
            action=action,
        )
        for schema in schemas
    ]
    selections.sort(
        key=lambda item: (
            item.score,
            -len(item.missing),
            len(item.schema.slots),
            -len(item.schema.head),
        ),
        reverse=True,
    )
    best = selections[0]
    if best.score <= 0:
        return None
    return best


def find_command_schema(
    schemas: list[PluginCommandSchema],
    *,
    command_id: str | None = None,
    command: str | None = None,
) -> PluginCommandSchema | None:
    selection = select_command_schema(
        schemas,
        command_id=command_id,
        command=command,
    )
    return selection.schema if selection is not None else None


def _parse_int_token(value: str) -> int | None:
    text = normalize_message_text(value)
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total = 0
    current = 0
    for char in text:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
            continue
        unit = _CN_UNITS.get(char)
        if unit is None:
            return None
        if current == 0:
            current = 1
        total += current * unit
        current = 0
    return total + current if total or current else None


def _extract_number(pattern: str, text: str, group: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return _parse_int_token(match.group(group))


def _extract_redbag_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    slots: dict[str, Any] = {}
    amount = _extract_number(
        rf"总额\s*(?P<amount>{_NUMBER_TEXT})",
        text,
        "amount",
    )
    num = _extract_number(
        rf"分\s*(?P<num>{_NUMBER_TEXT})\s*[份个]",
        text,
        "num",
    )
    if amount is not None:
        slots["amount"] = amount
    if num is not None:
        slots["num"] = num

    pair = re.search(
        rf"(?P<num>{_NUMBER_TEXT})\s*[个份]\s*(?P<amount>{_NUMBER_TEXT})\s*金币",
        text,
        re.IGNORECASE,
    )
    if pair:
        parsed_num = _parse_int_token(pair.group("num"))
        parsed_amount = _parse_int_token(pair.group("amount"))
        if parsed_num is not None:
            slots["num"] = parsed_num
        if parsed_amount is not None:
            slots["amount"] = parsed_amount

    separated_pair = re.search(
        rf"(?P<num>{_NUMBER_TEXT})\s*[个份]\s*红包.*?(?P<amount>{_NUMBER_TEXT})\s*金币",
        text,
        re.IGNORECASE,
    )
    if separated_pair:
        parsed_num = _parse_int_token(separated_pair.group("num"))
        parsed_amount = _parse_int_token(separated_pair.group("amount"))
        if parsed_num is not None:
            slots["num"] = parsed_num
        if parsed_amount is not None:
            slots["amount"] = parsed_amount

    amount_first_pair = re.search(
        rf"(?P<amount>{_NUMBER_TEXT})\s*金币.*?(?P<num>{_NUMBER_TEXT})\s*[个份]\s*红包",
        text,
        re.IGNORECASE,
    )
    if amount_first_pair:
        parsed_num = _parse_int_token(amount_first_pair.group("num"))
        parsed_amount = _parse_int_token(amount_first_pair.group("amount"))
        if parsed_num is not None:
            slots["num"] = parsed_num
        if parsed_amount is not None:
            slots["amount"] = parsed_amount

    if "num" not in slots:
        num_only = _extract_number(
            rf"(?P<num>{_NUMBER_TEXT})\s*[个份]\s*红包",
            text,
            "num",
        )
        if num_only is not None:
            slots["num"] = num_only

    if "amount" not in slots:
        amount = _extract_number(rf"(?P<amount>{_NUMBER_TEXT})\s*金币", text, "amount")
        if amount is not None:
            slots["amount"] = amount
    return slots


def _extract_translate_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    for pattern in (
        r"把\s*(?P<text>.+?)\s*翻(?:译)?成(?:中文|英文|日文|韩文)",
        r"翻译一下\s*(?P<text>.+)",
        r"翻译\s*(?P<text>.+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = normalize_message_text(match.group("text"))
            if value:
                return {"text": value}
    return {}


def _extract_roll_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    match = re.search(r"从\s*(?P<options>.+?)\s*(?:里|中)?\s*选", text)
    if not match:
        return {}
    options = normalize_message_text(match.group("options"))
    if not options:
        return {}
    if " " not in options and len(options) <= 6:
        options = " ".join(options)
    return {"options": options}


def _extract_luxun_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    match = re.search(r"鲁迅风格(?:写一句|写一段|说)?\s*(?P<text>.+)", text)
    if match:
        value = normalize_message_text(match.group("text"))
        return {"text": value} if value else {}
    match = re.search(r"文字是\s*(?P<text>.+)", text)
    if match:
        value = normalize_message_text(match.group("text"))
        return {"text": value} if value else {}
    match = re.search(r"内容是\s*(?P<text>.+)", text)
    if match:
        value = normalize_message_text(match.group("text"))
        return {"text": value} if value else {}
    match = re.search(r"说\s*(?P<text>.+)", text)
    if not match:
        return {}
    value = normalize_message_text(match.group("text"))
    return {"text": value} if value else {}


def _extract_cover_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    match = re.search(r"(https?://\S+|BV[0-9A-Za-z]+|av\d+|cv\d+)", text)
    return {"target": match.group(1)} if match else {}


def _extract_nbnhhsh_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    patterns = (
        r"(?P<text>[0-9A-Za-z_]{2,16})\s*(?:是)?(?:什么|啥|哪个)?缩写",
        r"(?:缩写|解释一下缩写|解释)\s*(?P<text>[0-9A-Za-z_]{2,16})",
        r"(?P<text>[0-9A-Za-z_]{2,16}).{0,4}(?:展开|啥意思|什么意思|是什么意思)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {"text": match.group("text")}
    return {}


def _infer_builtin_slots(
    schema: PluginCommandSchema,
    message_text: str,
) -> dict[str, Any]:
    command_id = normalize_message_text(schema.command_id)
    normalized = normalize_message_text(message_text)
    if not command_id:
        return {}
    if command_id == "translate.text" and normalized in {"帮我翻译一下", "翻译一下"}:
        return {}
    if command_id != "translate.text" and not (
        command_id == "gold_redbag.send"
        or command_id == "nbnhhsh.expand"
        or command_id.startswith("nonebot_plugin_memes.")
        or command_id in {"memes.search", "memes.info"}
        or any(char.isdigit() for char in normalized)
        or any(char in normalized for char in _CN_DIGITS)
        or any(char in normalized for char in _CN_UNITS)
    ):
        return {}
    extracted = extract_builtin_slots(command_id, message_text)
    if extracted:
        return extracted
    if command_id == "luxun.say":
        return _extract_luxun_slots(message_text)
    if command_id == "cover.bilibili":
        return _extract_cover_slots(message_text)
    if command_id == "nbnhhsh.expand":
        return _extract_nbnhhsh_slots(message_text)
    return {}


def _clean_text_payload(value: str) -> str:
    payload = normalize_message_text(value)
    while payload:
        cleaned = normalize_message_text(_TEXT_TAIL_PREFIX_PATTERN.sub("", payload))
        if cleaned == payload:
            break
        payload = cleaned
    return payload


def _extract_command_tail_payload(
    schema: PluginCommandSchema,
    message_text: str,
) -> str:
    heads = [schema.head, *schema.aliases]
    for head in heads:
        normalized_head = normalize_message_text(head)
        if not normalized_head:
            continue
        parsed = parse_command_with_head(
            message_text,
            normalized_head,
            allow_sticky=True,
            max_prefix_len=12,
        )
        if parsed is None:
            continue
        payload = _clean_text_payload(parsed.payload_text)
        if payload:
            return payload
    return ""


def _slot_accepts_url(slot: CommandSlotSpec) -> bool:
    text = normalize_message_text(
        " ".join([slot.name, slot.description, *slot.aliases])
    ).casefold()
    return any(
        token in text
        for token in ("链接", "地址", "url", "bv", "av", "视频", "link")
    )


def _extract_url_payload(message_text: str) -> str:
    match = _URL_PAYLOAD_PATTERN.search(normalize_message_text(message_text))
    return match.group(0) if match else ""


def _fill_slots_from_payload(
    merged: dict[str, Any],
    schema: PluginCommandSchema,
    payload: str,
) -> None:
    argument_payload = _clean_text_payload(payload)
    if not argument_payload:
        return
    payload_tokens = [token for token in argument_payload.split(" ") if token]
    token_index = 0
    for slot in schema.slots:
        if slot.name in merged or slot.type == "text":
            continue
        if token_index >= len(payload_tokens):
            break
        value: Any = payload_tokens[token_index]
        token_index += 1
        if slot.type == "int":
            parsed_value = _parse_int_token(value)
            if parsed_value is None:
                continue
            value = parsed_value
        merged[slot.name] = value
    for slot in schema.slots:
        if slot.name in merged or slot.type != "text":
            continue
        merged[slot.name] = argument_payload
        break


def _fill_link_slots_from_message(
    merged: dict[str, Any],
    schema: PluginCommandSchema,
    message_text: str,
) -> None:
    url_payload = _extract_url_payload(message_text)
    if not url_payload:
        return
    for slot in schema.slots:
        if slot.name in merged or slot.type != "text":
            continue
        if not _slot_accepts_url(slot):
            continue
        merged[slot.name] = url_payload
        return


def complete_slots(
    schema: PluginCommandSchema,
    *,
    slots: dict[str, Any] | None = None,
    message_text: str = "",
    arguments_text: str = "",
) -> tuple[dict[str, Any], list[str]]:
    merged: dict[str, Any] = {}
    for key, value in dict(slots or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not normalize_message_text(value):
            continue
        merged[key] = value
    # 手写 extractor 只覆盖少量高频命令，语义更贴近实际 matcher 参数顺序；
    # 它作为确定性修正层，避免模型把“4个20金币红包”误当成总额 80。
    inferred = _infer_builtin_slots(schema, normalize_message_text(message_text))
    if not inferred and arguments_text:
        inferred = _infer_builtin_slots(schema, normalize_message_text(arguments_text))
    merged.update(
        inferred,
    )
    _fill_link_slots_from_message(merged, schema, message_text)
    argument_payload = normalize_message_text(arguments_text)
    if not argument_payload:
        argument_payload = _extract_command_tail_payload(schema, message_text)
    if argument_payload:
        _fill_slots_from_payload(merged, schema, argument_payload)

    missing: list[str] = []
    for slot in schema.slots:
        if slot.name not in merged and slot.default is not None:
            merged[slot.name] = slot.default
        if slot.required and slot.name not in merged:
            missing.append(slot.name)
    return merged, missing


def render_command(
    schema: PluginCommandSchema,
    *,
    slots: dict[str, Any] | None = None,
    message_text: str = "",
    arguments_text: str = "",
) -> tuple[str, list[str]]:
    completed, missing = complete_slots(
        schema,
        slots=slots,
        message_text=message_text,
        arguments_text=arguments_text,
    )
    if missing:
        return schema.head, missing
    values = {
        slot.name: normalize_message_text(str(completed.get(slot.name, "")))
        for slot in schema.slots
    }
    try:
        rendered = schema.render.format_map(values)
    except Exception:
        rendered = schema.head
    return normalize_message_text(rendered), []
