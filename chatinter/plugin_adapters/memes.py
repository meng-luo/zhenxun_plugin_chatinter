"""Adapter for nonebot-plugin-memes style commands."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..models.pydantic_models import CommandCapability, PluginCommandSchema
from ..route_text import contains_any, has_chat_context_hint, normalize_message_text
from . import (
    AdapterClarifyRoute,
    AdapterNotificationPolicy,
    AdapterScoreHint,
    AdapterTargetPolicy,
    PluginCommandAdapter,
    register_adapter,
    schema,
    slot,
)

if TYPE_CHECKING:
    from ..command_index import CommandCandidate

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
    return aliases


def _is_shadowed_meme_head(value: str) -> bool:
    normalized = normalize_message_text(value)
    canonical = _CANONICAL_ALIAS_MAP.get(normalized)
    return bool(canonical and canonical != normalized)


def _split_text_parts(raw: str, *, max_parts: int) -> list[str]:
    text = normalize_message_text(raw)
    if not text:
        return []
    text = re.sub(r"^(?:文字|内容|文本)(?:是|为)?\s*", "", text)
    text = re.sub(r"^(?:四格|四段|四句)\s*", "", text)
    parts = [
        normalize_message_text(part)
        for part in re.split(r"\s*[，,、/|；;]\s*|\s{2,}", text)
        if normalize_message_text(part)
    ]
    if len(parts) >= max_parts:
        return parts[:max_parts]
    spaced = [part for part in text.split(" ") if part]
    if len(spaced) >= max_parts:
        return spaced[:max_parts]
    return parts


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
            aliases=["随机做个表情", "随机表情包", "随便做个表情"],
            description="使用当前图片/文字随机制作一个表情包",
            render="随机表情",
            requires={"image": False, "text": False},
            command_role="random",
            payload_policy="text_or_image",
            extra_text_policy="discard",
        ),
    ]
    seen = {item.command_id for item in schemas}
    for command in commands:
        head = normalize_message_text(command.command)
        if not head or _is_shadowed_meme_head(head):
            continue
        command_schema = _schema_from_capability(module, command)
        if command_schema is None or command_schema.command_id in seen:
            continue
        seen.add(command_schema.command_id)
        schemas.append(command_schema)
    return schemas


def _meme_slot_extractor(command_id: str, source: str) -> dict[str, Any]:
    normalized_id = normalize_message_text(command_id)
    text = normalize_message_text(source)
    if normalized_id == "memes.search":
        patterns = (
            r"(?:查找|搜索|找|搜|查)(?:一下)?\s*(?P<keyword>.+?)(?:相关)?(?:的)?表情",
            r"表情搜索\s*(?P<keyword>.+)",
        )
    elif normalized_id == "memes.info":
        patterns = (
            r"(?:查|看|了解)(?:一下)?\s*(?P<keyword>.+?)(?:这个)?表情(?:怎么用|详情|用法)?",
            r"表情详情\s*(?P<keyword>.+)",
        )
    else:
        if normalized_id.endswith(".王境泽"):
            if "王境泽" not in text:
                return {}
            match = re.search(r"(?:文字|内容|台词)(?:是|为)?\s*(?P<text>.+)", text)
            source_text = match.group("text") if match else text
            parts = _split_text_parts(source_text, max_parts=4)
            if len(parts) >= 4:
                return {
                    "文本1": parts[0],
                    "文本2": parts[1],
                    "文本3": parts[2],
                    "文本4": parts[3],
                }
        if normalized_id.endswith(".5000兆"):
            if "5000" not in text and not ("上" in text and "下" in text):
                return {}
            match = re.search(
                r"上(?:面|方)?(?:写|文字是)?\s*(?P<top>.+?)"
                r"\s*下(?:面|方)?(?:写|文字是)?\s*(?P<bottom>.+)",
                text,
            )
            if match:
                return {
                    "上文字": normalize_message_text(match.group("top")),
                    "下文字": normalize_message_text(match.group("bottom")),
                }
            parts = _split_text_parts(text, max_parts=2)
            if len(parts) >= 2:
                return {"上文字": parts[0], "下文字": parts[1]}
        return {}
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        keyword = normalize_message_text(match.group("keyword"))
        keyword = re.sub(r"(?:这个|的|怎么用|用法|详情)$", "", keyword).strip()
        if keyword:
            return {"keyword": keyword}
    return {}


def _meme_score_hints(
    schema_value: PluginCommandSchema,
    lowered_query: str,
    _stripped_lowered_query: str,
) -> list[AdapterScoreHint]:
    command_id = schema_value.command_id
    hints: list[AdapterScoreHint] = []
    if command_id == "memes.list" and any(
        token in lowered_query for token in ("表情", "表情包", "梗图")
    ):
        if any(token in lowered_query for token in ("列表", "有哪些", "有什么")):
            hints.append(AdapterScoreHint(180.0, "meme_catalog_intent"))
        elif any(
            token in lowered_query
            for token in ("没说模板", "没选模板", "不知道模板", "哪个模板", "什么模板")
        ):
            hints.append(AdapterScoreHint(180.0, "meme_template_missing"))
        elif any(
            token in lowered_query for token in ("做", "制作", "生成", "来个", "来张")
        ):
            hints.append(AdapterScoreHint(80.0, "meme_catalog_intent"))
    if schema_value.command_role == "template" and any(
        token in lowered_query for token in ("表情", "表情包", "梗图", "头像")
    ):
        hints.append(AdapterScoreHint(38.0, "template"))
    if command_id == "memes.search" and any(
        token in lowered_query for token in ("相关表情", "找一下", "搜一下", "搜索")
    ):
        hints.append(AdapterScoreHint(260.0, "meme_search_intent"))
    if command_id == "memes.info" and any(
        token in lowered_query for token in ("怎么用", "用法", "详情")
    ):
        hints.append(AdapterScoreHint(220.0, "meme_info_intent"))
    return hints


def _is_generic_meme_creation_request(
    message_text: str,
    candidates: list["CommandCandidate"],
) -> bool:
    normalized = normalize_message_text(message_text).casefold()
    if not normalized:
        return False
    if has_chat_context_hint(normalized):
        return False
    if contains_any(
        normalized,
        (
            "表情管理系统",
            "表情包系统",
            "架构",
            "怎么设计",
            "如何设计",
            "系统设计",
            "设计方案",
        ),
    ):
        return False
    meme_candidates = [item for item in candidates if item.family == "meme"]
    if not meme_candidates:
        return False
    if "随机" in normalized:
        return False
    if not any(token in normalized for token in ("表情", "表情包", "梗图", "头像")):
        return False
    if not any(
        token in normalized
        for token in ("做", "制作", "生成", "整", "来个", "来一个", "来张", "来一张")
    ):
        return False
    if any(
        candidate.schema.command_id == "memes.list"
        and candidate.reason == "meme_template_missing"
        for candidate in meme_candidates
    ):
        return True
    for candidate in meme_candidates:
        schema_value = candidate.schema
        if schema_value.command_role not in {"template", "random"}:
            continue
        phrases = [schema_value.head, *schema_value.aliases]
        if any(
            normalize_message_text(phrase).casefold()
            and normalize_message_text(phrase).casefold() in normalized
            for phrase in phrases
        ):
            return False
    return True


def _pick_meme_clarify_candidate(
    candidates: list["CommandCandidate"],
) -> "CommandCandidate | None":
    meme_candidates = [item for item in candidates if item.family == "meme"]
    for candidate in candidates:
        if candidate.schema.command_id == "memes.list":
            return candidate
    for candidate in meme_candidates:
        if candidate.schema.command_role in {"catalog", "helper"}:
            return candidate
    return meme_candidates[0] if meme_candidates else None


def _meme_clarify_route(
    message_text: str,
    candidates: list["CommandCandidate"],
) -> AdapterClarifyRoute | None:
    if not _is_generic_meme_creation_request(message_text, candidates):
        return None
    candidate = _pick_meme_clarify_candidate(candidates)
    if candidate is None:
        return None
    return AdapterClarifyRoute(
        command_id=candidate.schema.command_id,
        missing=("具体表情模板",),
        confidence=0.86,
        reason="generic_meme_template_missing",
    )


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
        slot_extractors={
            "memes.search": _meme_slot_extractor,
            "memes.info": _meme_slot_extractor,
        },
        slot_extractor_prefixes=("nonebot_plugin_memes.",),
        slot_extractor=lambda command_id, source: _meme_slot_extractor(
            command_id, source
        )
        if command_id.startswith("nonebot_plugin_memes.")
        else {},
        score_hints=_meme_score_hints,
        prompt_score_hints=lambda schema_value, normalized_query: _meme_score_hints(
            schema_value, normalized_query, normalized_query
        ),
        clarify_route=_meme_clarify_route,
    )
)
