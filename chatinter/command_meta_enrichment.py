from __future__ import annotations

"""Plugin-specific enrichment hooks for ChatInter's auto-discovered command
metadata.

Some plugins expose rich, structured retrieval signal (e.g. tags/keywords)
through a public runtime API that :class:`AutoMetadataBuilder` cannot see
because it only reflects on matcher/parser objects. This module provides a
small, generic extension point: a registry of ``module suffix -> enricher``
callables that ``PluginRegistry`` can invoke right after
``AutoMetadataBuilder.build`` produces its raw command dict payload and
before that payload is converted into ``PluginCommandMeta``.

Enrichment must never be able to break plugin/command registration: all
lookups and enricher invocations are wrapped so failures degrade silently
(a single debug log line) and the original payload is returned unchanged.

This module intentionally does not import anything from the meme plugin
packages themselves (``nonebot_plugin_memes`` / ``nonebot_plugin_memes_api``)
- it only talks to the third-party ``meme_generator`` library's public API,
which both of those plugins also depend on. If that library isn't
installed/importable, enrichment becomes a silent no-op.
"""

from collections.abc import Callable
from typing import Any

from .log_compat import logger

_MAX_PHRASES_PER_COMMAND = 8
_MAX_PHRASE_LENGTH = 16


def _truncate_phrase(text: object) -> str:
    normalized = str(text or "").strip()
    return normalized[:_MAX_PHRASE_LENGTH]


def _dedup_limited(
    phrases: list[str],
    *,
    limit: int = _MAX_PHRASES_PER_COMMAND,
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for phrase in phrases:
        text = _truncate_phrase(phrase)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _build_meme_keyword_index() -> dict[str, Any]:
    """Build a keyword/shortcut -> ``MemeInfo`` lookup.

    Uses only the public ``meme_generator`` API (``get_memes``). Read-only;
    never imports the meme plugin packages. Returns an empty dict if the
    library is unavailable or the call fails for any reason.
    """

    index: dict[str, Any] = {}
    try:
        import meme_generator  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug(f"ChatInter meme 标签富化跳过：meme_generator 不可用: {exc}")
        return index

    try:
        memes = meme_generator.get_memes()
    except Exception as exc:  # pragma: no cover - runtime/env dependent
        logger.debug(f"ChatInter meme 标签富化跳过：get_memes 调用失败: {exc}")
        return index

    for meme in memes or []:
        try:
            info = meme.info
        except Exception:
            continue
        for keyword in getattr(info, "keywords", None) or []:
            key = str(keyword or "").strip()
            if key and key not in index:
                index[key] = info
        for shortcut in getattr(info, "shortcuts", None) or []:
            for attr_name in ("pattern", "humanized"):
                value = str(getattr(shortcut, attr_name, "") or "").strip()
                if value and value not in index:
                    index[value] = info
    return index


_MEME_KEYWORD_INDEX: dict[str, Any] | None = None
_MEME_KEYWORD_INDEX_LOADED = False


def _get_meme_keyword_index() -> dict[str, Any]:
    global _MEME_KEYWORD_INDEX, _MEME_KEYWORD_INDEX_LOADED
    if not _MEME_KEYWORD_INDEX_LOADED:
        _MEME_KEYWORD_INDEX = _build_meme_keyword_index()
        _MEME_KEYWORD_INDEX_LOADED = True
    return _MEME_KEYWORD_INDEX or {}


def reset_meme_keyword_index_cache() -> None:
    """Testing/debug helper: force the keyword index to be rebuilt."""

    global _MEME_KEYWORD_INDEX, _MEME_KEYWORD_INDEX_LOADED
    _MEME_KEYWORD_INDEX = None
    _MEME_KEYWORD_INDEX_LOADED = False


def _phrases_for_head(head: str, aliases: list[str]) -> list[str]:
    index = _get_meme_keyword_index()
    if not index:
        return []
    info = None
    for candidate in (head, *aliases):
        candidate_text = str(candidate or "").strip()
        if candidate_text and candidate_text in index:
            info = index[candidate_text]
            break
    if info is None:
        return []
    phrases: list[str] = []
    try:
        for tag in sorted(str(tag) for tag in (getattr(info, "tags", None) or [])):
            phrases.append(tag)
        for keyword in getattr(info, "keywords", None) or []:
            text = str(keyword or "").strip()
            if text and text != head:
                phrases.append(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"ChatInter meme 标签富化解析失败: {exc}")
        return []
    return _dedup_limited(phrases)


def enrich_memes_command_payload(
    payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach meme tags/keywords to matching command dicts' ``examples``.

    ``examples`` is chosen (rather than a new field) because
    ``command_schema.py`` already folds ``command.examples`` into
    ``retrieval_phrases`` when building the command schema, so no schema
    changes are required downstream.
    """

    if not payload:
        return payload
    for item in payload:
        if not isinstance(item, dict):
            continue
        head = str(item.get("command") or item.get("head") or "").strip()
        if not head:
            continue
        aliases = item.get("aliases")
        alias_list = list(aliases) if isinstance(aliases, list | tuple) else []
        extra_phrases = _phrases_for_head(head, alias_list)
        if not extra_phrases:
            continue
        examples = item.get("examples")
        existing = list(examples) if isinstance(examples, list | tuple) else []
        item["examples"] = _dedup_limited([*existing, *extra_phrases])
    return payload


# Generic extension point: additional plugins can register their own
# enricher here, keyed by the plugin's module suffix (the last dotted
# segment(s) of its module name, matched via exact match or ``.suffix``).
_COMMAND_META_ENRICHERS: dict[
    str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
] = {
    "nonebot_plugin_memes_api": enrich_memes_command_payload,
    "nonebot_plugin_memes": enrich_memes_command_payload,
}


def enrich_command_meta_payload(
    module_name: str,
    payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply an optional plugin-specific enrichment hook to raw command dicts.

    Looked up by ``module_name`` suffix against ``_COMMAND_META_ENRICHERS``.
    Any failure (missing dependency, unexpected shape, etc.) is caught and
    debug-logged; the original, unmodified payload is returned so a broken
    enricher can never block command registration.
    """

    if not module_name or not payload:
        return payload
    normalized = str(module_name).strip()
    enricher = None
    for suffix, candidate in _COMMAND_META_ENRICHERS.items():
        if normalized == suffix or normalized.endswith(f".{suffix}"):
            enricher = candidate
            break
    if enricher is None:
        return payload
    try:
        return enricher(payload)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            f"ChatInter 命令元数据富化钩子失败: module={module_name}, error={exc}"
        )
        return payload


__all__ = [
    "enrich_command_meta_payload",
    "enrich_memes_command_payload",
    "reset_meme_keyword_index_cache",
]
