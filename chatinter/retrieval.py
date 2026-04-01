from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
import re
from types import MappingProxyType
from typing import Any

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .route_text import ROUTE_ACTION_WORDS, contains_any, normalize_message_text
from .skill_registry import get_skill_registry, select_relevant_skills

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)
_REQUIRED_TOKEN_PATTERN = re.compile(r"(?<!\w)\+([a-z0-9_]+|[\u4e00-\u9fff]{1,6})")
_SELECT_PATTERN = re.compile(
    r"(?:^|\s)(?:select|选择插件|指定插件|模块)\s*[:：]\s*([A-Za-z0-9_.\-\u4e00-\u9fff]+)",
    re.IGNORECASE,
)

_STOPWORDS = {
    "zhenxun",
    "nonebot",
    "plugin",
    "plugins",
    "功能",
    "插件",
    "一下",
    "一个",
    "这个",
    "那个",
    "什么",
    "怎么",
    "如何",
    "please",
    "help",
}


def _normalize_sequence(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text:
            normalized.append(text)
    return tuple(normalized)


def _plugin_meta_signature(meta: PluginInfo.PluginCommandMeta) -> tuple[Any, ...]:
    return (
        normalize_message_text(meta.command),
        _normalize_sequence(meta.aliases or ()),
        _normalize_sequence(meta.params or ()),
        _normalize_sequence(meta.examples or ()),
        int(getattr(meta, "image_min", 0) or 0),
        int(getattr(meta, "image_max", 0) or 0),
        getattr(meta, "allow_at", None),
        normalize_message_text(str(getattr(meta, "actor_scope", "") or "")).lower(),
        normalize_message_text(
            str(getattr(meta, "target_requirement", "") or "")
        ).lower(),
        _normalize_sequence(getattr(meta, "target_sources", None) or ()),
    )


def _plugin_cache_key(plugin: PluginInfo) -> tuple[Any, ...]:
    return (
        normalize_message_text(plugin.name),
        normalize_message_text(plugin.module),
        normalize_message_text(plugin.description),
        normalize_message_text(plugin.usage or ""),
        _normalize_sequence(plugin.commands or ()),
        _normalize_sequence(plugin.aliases or ()),
        tuple(_plugin_meta_signature(meta) for meta in plugin.command_meta),
    )


@lru_cache(maxsize=2048)
def _cached_plugin_terms(cache_key: tuple[Any, ...]) -> frozenset[str]:
    name, module, description, usage, commands, aliases, meta_signatures = cache_key
    command_meta_text = " ".join(
        " ".join(
            (
                command,
                " ".join(meta_aliases),
                " ".join(params),
                " ".join(examples),
            )
        ).strip()
        for command, meta_aliases, params, examples, *_ in meta_signatures
    )
    haystack = (
        f"{name} {module} {description} "
        f"{' '.join(commands)} {' '.join(aliases)} {command_meta_text} {usage}"
    )
    terms = set(_tokenize(haystack))
    if name:
        terms.add(name.lower())
    module_tail = _module_tail(module).lower()
    if module_tail:
        terms.add(module_tail)
    for command in commands:
        if command:
            terms.add(command.lower())
    for alias in aliases:
        if alias:
            terms.add(alias.lower())
    for command, meta_aliases, *_ in meta_signatures:
        if command:
            terms.add(command.lower())
        for alias in meta_aliases:
            if alias:
                terms.add(alias.lower())
    return frozenset(term for term in terms if term)


@lru_cache(maxsize=2048)
def _cached_plugin_haystack(cache_key: tuple[Any, ...]) -> str:
    name, module, description, usage, commands, aliases, meta_signatures = cache_key
    command_meta_text = " ".join(
        " ".join(
            (
                command,
                " ".join(meta_aliases),
                " ".join(params),
                " ".join(examples),
            )
        ).strip()
        for command, meta_aliases, params, examples, *_ in meta_signatures
    )
    return normalize_message_text(
        (
            f"{name} {module} {description} "
            f"{' '.join(commands)} {' '.join(aliases)} {command_meta_text} {usage}"
        )
    ).lower()


@lru_cache(maxsize=2048)
def _cached_plugin_metadata(
    cache_key: tuple[Any, ...],
) -> MappingProxyType[str, bool]:
    _, _, _, _, commands, _, meta_signatures = cache_key
    commands_text = normalize_message_text(" ".join(commands)).lower()
    has_helper = any(
        keyword in commands_text
        for keyword in ("帮助", "详情", "搜索", "用法", "参数", "说明", "列表")
    )
    target_capable = False
    image_capable = False
    self_only = False
    for (
        _command,
        _aliases,
        _params,
        _examples,
        image_min,
        image_max,
        allow_at_raw,
        actor_scope,
        target_requirement,
        target_sources,
    ) in meta_signatures:
        normalized_sources = {
            normalize_message_text(source).lower()
            for source in target_sources
            if normalize_message_text(source)
        }
        normalized_requirement = normalize_message_text(target_requirement).lower()
        if normalized_requirement not in {"none", "optional", "required"}:
            normalized_requirement = "none"
        normalized_scope = normalize_message_text(actor_scope).lower()
        allow_at = allow_at_raw is True or (
            allow_at_raw is not False and "at" in normalized_sources
        )
        if normalized_scope == "self_only":
            allow_at = False
            normalized_requirement = "none"
        if allow_at or bool(normalized_sources & {"at", "reply", "nickname"}):
            target_capable = True
        if normalized_scope == "self_only":
            self_only = True
        if int(image_min) > 0 or int(image_max) > 0:
            image_capable = True
    return MappingProxyType(
        {
            "has_helper": has_helper,
            "target_capable": target_capable,
            "image_capable": image_capable,
            "self_only": self_only,
        }
    )


@dataclass
class _PluginCandidate:
    plugin: PluginInfo
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, value: float, reason: str):
        if value <= 0:
            return
        self.score += value
        if reason and reason not in self.reasons:
            self.reasons.append(reason)


def _tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text.lower())]
    return [token for token in tokens if token and token not in _STOPWORDS]


def _required_tokens(text: str) -> set[str]:
    return {
        normalize_message_text(item).lower()
        for item in _REQUIRED_TOKEN_PATTERN.findall(text or "")
        if normalize_message_text(item)
    }


def _extract_select_key(text: str) -> str:
    matched = _SELECT_PATTERN.search(text or "")
    if not matched:
        return ""
    return normalize_message_text(matched.group(1)).lower()


def _module_tail(module: str) -> str:
    normalized = normalize_message_text(module)
    if "." not in normalized:
        return normalized
    return normalized.rsplit(".", 1)[-1]


def _plugin_terms(
    plugin: PluginInfo,
    *,
    cache_key: tuple[Any, ...] | None = None,
) -> frozenset[str]:
    return _cached_plugin_terms(cache_key or _plugin_cache_key(plugin))


def _plugin_metadata(
    plugin: PluginInfo,
    *,
    cache_key: tuple[Any, ...] | None = None,
) -> MappingProxyType[str, bool]:
    return _cached_plugin_metadata(cache_key or _plugin_cache_key(plugin))


def _plugin_haystack(
    plugin: PluginInfo,
    *,
    cache_key: tuple[Any, ...] | None = None,
) -> str:
    return _cached_plugin_haystack(cache_key or _plugin_cache_key(plugin))


def _satisfies_filters(
    plugin: PluginInfo,
    metadata_filters: dict[str, bool] | None,
    *,
    cache_key: tuple[Any, ...] | None = None,
) -> bool:
    if not metadata_filters:
        return True
    metadata = _plugin_metadata(plugin, cache_key=cache_key)
    for key, expected in metadata_filters.items():
        if metadata.get(key) is not bool(expected):
            return False
    return True


def _match_select_key(plugin: PluginInfo, key: str) -> bool:
    if not key:
        return False
    module = normalize_message_text(plugin.module).lower()
    module_tail = _module_tail(plugin.module).lower()
    name = normalize_message_text(plugin.name).lower()
    if key in {module, module_tail, name}:
        return True
    if module.endswith(f".{key}") or module.startswith(f"{key}."):
        return True
    for command in plugin.commands or []:
        if normalize_message_text(command).lower() == key:
            return True
    for alias in plugin.aliases or []:
        if normalize_message_text(alias).lower() == key:
            return True
    return False


def _iter_command_heads(plugin: PluginInfo) -> Iterable[str]:
    yielded: set[str] = set()
    for command in plugin.commands or []:
        text = normalize_message_text(command).lower()
        if text and text not in yielded:
            yielded.add(text)
            yield text
    for meta in plugin.command_meta:
        head = normalize_message_text(meta.command).lower()
        if head and head not in yielded:
            yielded.add(head)
            yield head
        for alias in meta.aliases or []:
            text = normalize_message_text(alias).lower()
            if text and text not in yielded:
                yielded.add(text)
                yield text


def _build_candidate(
    *,
    plugin: PluginInfo,
    query: str,
    query_tokens: set[str],
    required_tokens: set[str],
    has_execute_intent: bool,
    plugin_cache_key: tuple[Any, ...] | None = None,
) -> _PluginCandidate | None:
    candidate = _PluginCandidate(plugin=plugin)
    normalized_query = normalize_message_text(query).lower()
    haystack = _plugin_haystack(plugin, cache_key=plugin_cache_key)
    terms = _plugin_terms(plugin, cache_key=plugin_cache_key)

    if required_tokens and not required_tokens.issubset(terms):
        return None

    name = normalize_message_text(plugin.name).lower()
    module_tail = _module_tail(plugin.module).lower()
    if name and name in normalized_query:
        candidate.add(2.0, "name_hit")
    if module_tail and module_tail in normalized_query:
        candidate.add(1.4, "module_hit")

    command_heads = list(_iter_command_heads(plugin))
    for head in command_heads:
        if normalized_query.startswith(head):
            candidate.add(3.0, "head_prefix")
            break
        if head and head in normalized_query:
            candidate.add(1.5, "head_contains")
            break

    for token in query_tokens:
        if token in terms:
            candidate.add(0.9, f"token:{token}")
        elif token in haystack:
            candidate.add(0.5, f"fuzzy:{token}")
        if name.startswith(token):
            candidate.add(0.8, "name_prefix")
        if module_tail.startswith(token):
            candidate.add(0.4, "module_prefix")

    if has_execute_intent and any(
        head and (normalized_query.startswith(head) or head in normalized_query)
        for head in command_heads
    ):
        candidate.add(0.8, "intent_command_alignment")

    return candidate if candidate.score > 0 else None


def rank_route_candidates(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    preferred_modules: list[str] | tuple[str, ...] | None = None,
    metadata_filters: dict[str, bool] | None = None,
    min_score: float = 0.0,
) -> list[PluginInfo]:
    plugins = knowledge_base.plugins or []
    if not plugins:
        return []

    normalized_query = normalize_message_text(query)
    if not normalized_query:
        filtered_plugins = [
            plugin
            for plugin in plugins
            if _satisfies_filters(
                plugin,
                metadata_filters,
                cache_key=_plugin_cache_key(plugin),
            )
        ]
        if not filtered_plugins:
            return []

        normalized_pref = [
            normalize_message_text(module).lower()
            for module in preferred_modules or ()
            if normalize_message_text(module)
        ]
        if not normalized_pref:
            return filtered_plugins

        pref_order = {
            module: idx for idx, module in enumerate(normalized_pref)
        }
        fallback_index = len(pref_order)
        filtered_plugins.sort(
            key=lambda plugin: pref_order.get(
                normalize_message_text(plugin.module).lower(),
                fallback_index,
            )
        )
        return filtered_plugins

    query_tokens = set(_tokenize(normalized_query))
    required_tokens = _required_tokens(normalized_query)
    has_execute_intent = contains_any(normalized_query, ROUTE_ACTION_WORDS)
    select_key = _extract_select_key(normalized_query)

    score_map: dict[str, _PluginCandidate] = {}
    for plugin in plugins:
        cache_key = _plugin_cache_key(plugin)
        if not _satisfies_filters(
            plugin,
            metadata_filters,
            cache_key=cache_key,
        ):
            continue
        candidate = _build_candidate(
            plugin=plugin,
            query=normalized_query,
            query_tokens=query_tokens,
            required_tokens=required_tokens,
            has_execute_intent=has_execute_intent,
            plugin_cache_key=cache_key,
        )
        if candidate is None:
            continue
        score_map[plugin.module] = candidate

    if not score_map:
        return []

    registry = get_skill_registry(knowledge_base)
    skill_ranked = select_relevant_skills(registry, normalized_query, limit=10)
    for idx, skill in enumerate(skill_ranked):
        candidate = score_map.get(skill.plugin_module)
        if candidate is None:
            continue
        boost = max(1.4 - idx * 0.14, 0.2)
        candidate.add(boost, "skill_rank")

    normalized_pref = [
        normalize_message_text(module).lower()
        for module in preferred_modules or ()
        if normalize_message_text(module)
    ]
    for idx, module in enumerate(normalized_pref):
        candidate = score_map.get(module)
        if candidate is None:
            continue
        boost = max(1.1 - idx * 0.08, 0.15)
        candidate.add(boost, "vector_pref")

    scored = list(score_map.values())
    scored.sort(
        key=lambda item: (
            item.score,
            len(normalize_message_text(item.plugin.commands[0]).strip())
            if item.plugin.commands
            else 0,
        ),
        reverse=True,
    )
    ranked_plugins = [
        item.plugin
        for item in scored
        if item.score >= max(min_score, 0.0)
    ]

    if select_key:
        direct = [
            plugin
            for plugin in ranked_plugins
            if _match_select_key(plugin, select_key)
        ]
        if direct:
            remain = [plugin for plugin in ranked_plugins if plugin not in direct]
            ranked_plugins = [*direct, *remain]

    if ranked_plugins:
        return ranked_plugins
    return []


def select_route_candidates(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    limit: int = 10,
    preferred_modules: list[str] | tuple[str, ...] | None = None,
    metadata_filters: dict[str, bool] | None = None,
    min_score: float = 0.0,
) -> list[PluginInfo]:
    ranked = rank_route_candidates(
        knowledge_base,
        query,
        preferred_modules=preferred_modules,
        metadata_filters=metadata_filters,
        min_score=min_score,
    )
    resolved_limit = max(int(limit or 0), 1)
    return ranked[:resolved_limit]


def select_relevant_plugins(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    limit: int = 8,
) -> PluginKnowledgeBase:
    selected = select_route_candidates(knowledge_base, query, limit=limit)
    if not selected:
        selected = list(knowledge_base.plugins[:limit])
    return PluginKnowledgeBase(
        plugins=selected,
        user_role=knowledge_base.user_role,
    )


__all__ = [
    "rank_route_candidates",
    "select_relevant_plugins",
    "select_route_candidates",
]
