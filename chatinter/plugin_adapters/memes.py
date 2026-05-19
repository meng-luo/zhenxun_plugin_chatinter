"""Adapter for nonebot-plugin-memes style commands."""

from __future__ import annotations

from ..models.pydantic_models import CommandCapability, PluginCommandSchema
from ..route_text import normalize_message_text
from . import (
    AdapterNotificationPolicy,
    AdapterTargetPolicy,
    PluginCommandAdapter,
    register_adapter,
    schema,
    slot,
)

_CANONICAL_ALIAS_MAP: dict[str, str] = {
    "摸摸": "摸",
    "摸头": "摸",
    "摸摸头": "摸",
    "亲亲": "亲",
    "亲一下": "亲",
    "拍拍": "拍",
    "拍一下": "拍",
    "吃掉": "吃",
    "吃掉表情": "吃",
    "丢出去": "丢",
    "扔出去": "丢",
    "丢出": "丢",
    "扔出": "丢",
    "天使": "小天使",
    "天使头像": "小天使",
    "小天使头像": "小天使",
}

_SHORT_ACTION_SUFFIXES = (
    "一下",
    "一把",
    "一张",
    "表情",
    "表情包",
    "梗图",
)

_TARGET_CONTEXT_HINTS = ("表情", "表情包", "梗图", "头像")

_NOTIFY_TEMPLATES = (
    "好、好啦，真寻这就做{target}。",
    "收到啦，马上给你做{target}。",
    "诶嘿，开工咯，这就做{target}。",
    "等我一下下，这就把{target}做出来。",
    "哼，这个我超会，立刻做{target}。",
    "软乎乎开工中，马上给你{target}。",
)

_HELPER_NOTIFY_TEMPLATES = (
    "好、好啦，这就给你{target}。",
    "收到啦，我马上帮你{target}。",
    "唔，知道啦，这就去{target}。",
    "安排上啦，立刻给你{target}。",
    "等一下下，这就帮你{target}。",
)

_HELPER_HEADS = frozenset({"表情搜索", "表情详情", "启用表情", "更新表情"})
_ADAPTER_OWNED_HEADS = frozenset({"表情包制作", "随机表情", *_HELPER_HEADS})


def _meme_semantic_aliases(
    head: str,
    _module: str,
    image_required: bool,
) -> list[str]:
    if not head or not image_required:
        return []
    aliases: list[str] = []

    def add(value: object) -> None:
        text = normalize_message_text(str(value or ""))
        if text and text != head and text not in aliases:
            aliases.append(text)

    for alias, canonical in _CANONICAL_ALIAS_MAP.items():
        if canonical == head:
            add(alias)
    if len(head) <= 2:
        for suffix in _SHORT_ACTION_SUFFIXES:
            add(f"{head}{suffix}")
    if head == "小天使":
        add("天使")
        add("天使头像")
        add("小天使头像")
    return aliases


def _is_shadowed_meme_head(value: str) -> bool:
    normalized = normalize_message_text(value)
    canonical = _CANONICAL_ALIAS_MAP.get(normalized)
    return bool(canonical and canonical != normalized)


def _schema_from_capability(
    module: str,
    command: CommandCapability,
) -> PluginCommandSchema | None:
    # Imported lazily to avoid command_schema <-> adapter import cycles.
    from ..command_schema import schema_from_capability

    return schema_from_capability(module, command)


def _build_meme_schemas(
    module: str,
    commands: list[CommandCapability],
) -> list[PluginCommandSchema]:
    schemas: list[PluginCommandSchema] = [
        schema(
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
        schema(
            "memes.search",
            "表情搜索",
            aliases=["搜索表情", "找表情", "查找表情", "找相关表情"],
            description="按关键词搜索相关表情包模板",
            slots=[
                slot(
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
        schema(
            "memes.info",
            "表情详情",
            aliases=["表情用法", "表情参数", "这个表情怎么用"],
            description="查看某个表情的参数、预览和用法",
            slots=[
                slot(
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
        schema(
            "memes.random",
            "随机表情",
            aliases=[
                "随机做个表情",
                "随机表情包",
                "随便做个表情",
                "随便做表情",
                "随便把图整成表情",
            ],
            description="使用当前图片/文字随机制作一个表情包",
            render="随机表情",
            requires={"image": False, "text": False},
            command_role="random",
            payload_policy="text_or_image",
            extra_text_policy="discard",
            retrieval_phrases=[
                "随便 表情",
                "随机 表情",
                "随便 把 图 整成 表情",
            ],
        ),
    ]
    seen = {item.command_id for item in schemas}
    for command in commands:
        head = normalize_message_text(command.command)
        if not head or head in _ADAPTER_OWNED_HEADS or _is_shadowed_meme_head(head):
            continue
        command_schema = _schema_from_capability(module, command)
        if command_schema is None or command_schema.command_id in seen:
            continue
        seen.add(command_schema.command_id)
        schemas.append(command_schema)
    return schemas


register_adapter(
    PluginCommandAdapter(
        module_suffixes=("nonebot_plugin_memes",),
        family="meme",
        build_schemas=_build_meme_schemas,
        semantic_aliases=_meme_semantic_aliases,
        target_policy=AdapterTargetPolicy(
            family="meme",
            context_hints=_TARGET_CONTEXT_HINTS,
            media_related=True,
            allow_at_as_target=True,
            allow_image_as_target=True,
            allow_reply_image_as_target=True,
            require_target_for_third_person=True,
            target_missing_message=(
                "要帮别人制作的话，请补充完整昵称、直接@对方，或者发对方头像。"
            ),
        ),
        notification_policy=AdapterNotificationPolicy(
            target_suffix="表情",
            helper_heads=_HELPER_HEADS,
            default_templates=_NOTIFY_TEMPLATES,
            helper_templates=_HELPER_NOTIFY_TEMPLATES,
        ),
    )
)
