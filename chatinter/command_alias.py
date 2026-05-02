"""Semantic alias helpers for command-level routing.

The router should not need one-off fixes for every plugin, but short Chinese
template commands often have natural spoken variants ("摸摸头" -> "摸").
This module keeps those expansions deterministic and reusable across the
candidate index, schema selector and planner.
"""

from __future__ import annotations

from .route_text import normalize_message_text

_MEME_CANONICAL_ALIAS_MAP: dict[str, str] = {
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

_MUSIC_HEADS = {"点歌", "搜歌", "播放音乐", "音乐搜索"}
_MUSIC_ALIASES = (
    "点一首歌",
    "点首歌",
    "点一首",
    "点首",
    "播一首歌",
    "播首歌",
    "来一首歌",
    "来首歌",
    "放一首歌",
    "听歌",
    "搜歌",
    "找歌",
)
_SIGN_IN_HEADS = {"签到", "打卡", "补签"}
_SIGN_IN_ALIASES = (
    "打卡",
    "签个到",
    "签一下到",
    "今日签到",
    "今天签到",
)
_POETRY_HEADS = {"念诗", "来首诗", "念首诗", "古诗", "诗词"}
_POETRY_ALIASES = (
    "古诗",
    "诗词",
    "来首诗",
    "念首诗",
    "来一首古诗",
    "来首古诗",
)


def normalize_alias(value: object) -> str:
    return normalize_message_text(str(value or "")).casefold()


def canonical_meme_head(value: object) -> str:
    normalized = normalize_message_text(str(value or ""))
    return _MEME_CANONICAL_ALIAS_MAP.get(normalized, normalized)


def is_shadowed_meme_head(value: object) -> bool:
    normalized = normalize_message_text(str(value or ""))
    canonical = _MEME_CANONICAL_ALIAS_MAP.get(normalized)
    return bool(canonical and canonical != normalized)


def derive_semantic_aliases(
    head: object,
    *,
    module: str = "",
    image_required: bool = False,
) -> list[str]:
    normalized = normalize_message_text(str(head or ""))
    if not normalized:
        return []

    aliases: list[str] = []

    def add(value: object) -> None:
        text = normalize_message_text(str(value or ""))
        if text and text != normalized and text not in aliases:
            aliases.append(text)

    module_l = normalize_message_text(module).lower()
    if "meme" in module_l or image_required:
        for alias, canonical in _MEME_CANONICAL_ALIAS_MAP.items():
            if canonical == normalized:
                add(alias)
        if len(normalized) <= 2:
            for suffix in _SHORT_ACTION_SUFFIXES:
                add(f"{normalized}{suffix}")

    if normalized in _MUSIC_HEADS:
        for alias in _MUSIC_ALIASES:
            add(alias)

    if normalized in _SIGN_IN_HEADS:
        for alias in _SIGN_IN_ALIASES:
            add(alias)

    if normalized in _POETRY_HEADS:
        for alias in _POETRY_ALIASES:
            add(alias)

    return aliases


__all__ = [
    "canonical_meme_head",
    "derive_semantic_aliases",
    "is_shadowed_meme_head",
    "normalize_alias",
]
