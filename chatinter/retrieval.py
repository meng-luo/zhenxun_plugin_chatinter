from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
import re
from types import MappingProxyType
from typing import Any

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .route_text import (
    ROUTE_ACTION_WORDS,
    contains_any,
    match_command_head_fuzzy,
    match_command_head_or_sticky,
    normalize_message_text,
)
from .skill_registry import get_skill_registry, select_relevant_skills
from .skill_registry import infer_query_families

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
_FAMILY_SEARCH_HINTS = (
    "找",
    "搜",
    "查",
    "查询",
    "搜索",
    "寻找",
    "列表",
    "今天",
    "今日",
    "本日",
    "当日",
)
_FAMILY_TEMPLATE_HINTS = (
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
)
_FAMILY_TRANSACTION_HINTS = (
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
_FAMILY_SELF_HINTS = (
    "签到",
    "自我介绍",
    "我的信息",
    "我自己",
    "本人",
    "自己",
)
_FAMILY_UTILITY_HINTS = (
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


def _infer_plugin_family(plugin: PluginInfo) -> str:
    command_heads = [
        normalize_message_text(command).lower()
        for command in plugin.commands or ()
        if normalize_message_text(command)
    ]
    alias_heads = [
        normalize_message_text(alias).lower()
        for alias in plugin.aliases or ()
        if normalize_message_text(alias)
    ]
    meta_heads = [
        normalize_message_text(getattr(meta, "command", "")).lower()
        for meta in plugin.command_meta or ()
        if normalize_message_text(getattr(meta, "command", ""))
    ]
    meta_aliases = [
        normalize_message_text(alias).lower()
        for meta in plugin.command_meta or ()
        for alias in getattr(meta, "aliases", ()) or ()
        if normalize_message_text(alias)
    ]
    haystack = " ".join(
        [
            normalize_message_text(plugin.name),
            normalize_message_text(plugin.module),
            normalize_message_text(plugin.description),
            normalize_message_text(plugin.usage or ""),
            " ".join(command_heads),
            " ".join(alias_heads),
            " ".join(meta_heads),
            " ".join(meta_aliases),
        ]
    ).lower()

    if any(marker in haystack for marker in _FAMILY_TEMPLATE_HINTS):
        return "template"
    if any(marker in haystack for marker in _FAMILY_TRANSACTION_HINTS):
        return "transaction"
    if any(marker in haystack for marker in _FAMILY_SEARCH_HINTS):
        return "search"
    if any(marker in haystack for marker in _FAMILY_SELF_HINTS):
        return "self"
    if any(marker in haystack for marker in _FAMILY_UTILITY_HINTS):
        return "utility"
    return "general"


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
    best_head_len: int = 0

    def add(self, value: float, reason: str):
        if value <= 0:
            return
        self.score += value
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def note_head_match(self, head: str):
        self.best_head_len = max(self.best_head_len, len(head or ""))


@dataclass(frozen=True)
class RouteCandidateScore:
    plugin: PluginInfo
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CommandEntryMatch:
    score: float
    reason: str
    head: str


def _score_command_entry_match(
    normalized_query: str,
    head: str,
    *,
    allow_sticky: bool,
    exact_score: float,
    prefix_score: float,
    sticky_score: float,
    contains_score: float,
    reason_prefix: str,
) -> _CommandEntryMatch | None:
    if not head:
        return None

    head_bonus = min(len(head) * 0.08, 0.8)
    if normalized_query == head:
        return _CommandEntryMatch(
            score=exact_score + head_bonus,
            reason=f"{reason_prefix}_exact",
            head=head,
        )
    if normalized_query.startswith(head):
        return _CommandEntryMatch(
            score=prefix_score + head_bonus,
            reason=f"{reason_prefix}_prefix",
            head=head,
        )
    if match_command_head_or_sticky(
        normalized_query,
        head,
        allow_sticky=allow_sticky,
    ):
        return _CommandEntryMatch(
            score=sticky_score + head_bonus,
            reason=f"{reason_prefix}_sticky",
            head=head,
        )
    if match_command_head_fuzzy(
        normalized_query,
        head,
        allow_sticky=allow_sticky,
    ):
        return _CommandEntryMatch(
            score=contains_score + 0.6 + head_bonus,
            reason=f"{reason_prefix}_fuzzy",
            head=head,
        )
    if len(head) >= 3 and head in normalized_query:
        return _CommandEntryMatch(
            score=contains_score + head_bonus,
            reason=f"{reason_prefix}_contains",
            head=head,
        )
    return None


def _best_command_entry_match(
    normalized_query: str,
    command_entries: Iterable[tuple[str, bool]],
    *,
    exact_score: float,
    prefix_score: float,
    sticky_score: float,
    contains_score: float,
    reason_prefix: str,
) -> _CommandEntryMatch | None:
    best: _CommandEntryMatch | None = None
    for head, allow_sticky in command_entries:
        matched = _score_command_entry_match(
            normalized_query,
            head,
            allow_sticky=allow_sticky,
            exact_score=exact_score,
            prefix_score=prefix_score,
            sticky_score=sticky_score,
            contains_score=contains_score,
            reason_prefix=reason_prefix,
        )
        if matched is None:
            continue
        if best is None or (matched.score, len(matched.head)) > (
            best.score,
            len(best.head),
        ):
            best = matched
    return best


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


def _iter_command_entries(plugin: PluginInfo) -> Iterable[tuple[str, bool]]:
    yielded: set[tuple[str, bool]] = set()
    for meta in plugin.command_meta:
        allow_sticky = bool(getattr(meta, "allow_sticky_arg", False))
        head = normalize_message_text(getattr(meta, "command", "")).lower()
        if head:
            entry = (head, allow_sticky)
            if entry not in yielded:
                yielded.add(entry)
                yield entry
        for alias in meta.aliases or []:
            text = normalize_message_text(alias).lower()
            if text:
                entry = (text, allow_sticky)
                if entry not in yielded:
                    yielded.add(entry)
                    yield entry
    for head in _iter_command_heads(plugin):
        entry = (head, False)
        if entry not in yielded:
            yielded.add(entry)
            yield entry


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
    query_families = infer_query_families(normalized_query)
    haystack = _plugin_haystack(plugin, cache_key=plugin_cache_key)
    terms = _plugin_terms(plugin, cache_key=plugin_cache_key)
    family = _infer_plugin_family(plugin)

    if required_tokens and not required_tokens.issubset(terms):
        return None

    name = normalize_message_text(plugin.name).lower()
    module_tail = _module_tail(plugin.module).lower()
    if name and name in normalized_query:
        candidate.add(2.0, "name_hit")
    if module_tail and module_tail in normalized_query:
        candidate.add(1.4, "module_hit")

    command_entries = list(_iter_command_entries(plugin))
    head_match = _best_command_entry_match(
        normalized_query,
        command_entries,
        exact_score=4.6,
        prefix_score=3.3,
        sticky_score=2.9,
        contains_score=1.5,
        reason_prefix="head",
    )
    if head_match is not None:
        candidate.add(head_match.score, head_match.reason)
        candidate.note_head_match(head_match.head)

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
        head
        and (
            normalized_query.startswith(head)
            or match_command_head_or_sticky(
                normalized_query,
                head,
                allow_sticky=allow_sticky,
            )
            or match_command_head_fuzzy(
                normalized_query,
                head,
                allow_sticky=allow_sticky,
            )
            or head in normalized_query
        )
        for head, allow_sticky in command_entries
    ):
        candidate.add(0.8, "intent_command_alignment")

    if family == query_families[0]:
        candidate.add(5.0, f"family:{family}")
    elif family in query_families[1:]:
        candidate.add(3.0, f"family:{family}")
    elif query_families[0] != "general" and family == "general":
        candidate.add(0.5, "family:general")

    if family == "search":
        if contains_any(normalized_query, _FAMILY_SEARCH_HINTS):
            candidate.add(1.8, "family_search_hint")
        if any(marker in normalized_query for marker in _FAMILY_SEARCH_HINTS) and any(
            marker in haystack for marker in ("今日", "今天", "本日", "当日", "找", "搜", "查")
        ):
            candidate.add(1.6, "family_search_time")
    elif family == "template" and contains_any(normalized_query, _FAMILY_TEMPLATE_HINTS):
        candidate.add(2.0, "family_template_hint")
    elif family == "transaction" and contains_any(normalized_query, _FAMILY_TRANSACTION_HINTS):
        candidate.add(2.0, "family_transaction_hint")
    elif family == "self" and contains_any(normalized_query, _FAMILY_SELF_HINTS):
        candidate.add(1.4, "family_self_hint")
    elif family == "utility" and contains_any(normalized_query, _FAMILY_UTILITY_HINTS):
        candidate.add(1.2, "family_utility_hint")

    return candidate if candidate.score > 0 else None


def collect_direct_head_candidates(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    metadata_filters: dict[str, bool] | None = None,
) -> list[RouteCandidateScore]:
    plugins = knowledge_base.plugins or []
    if not plugins:
        return []

    normalized_query = normalize_message_text(query).lower()
    if not normalized_query:
        return []

    has_execute_intent = contains_any(normalized_query, ROUTE_ACTION_WORDS)
    scored: list[RouteCandidateScore] = []
    for plugin in plugins:
        cache_key = _plugin_cache_key(plugin)
        if not _satisfies_filters(
            plugin,
            metadata_filters,
            cache_key=cache_key,
        ):
            continue

        candidate = _PluginCandidate(plugin=plugin)
        head_match = _best_command_entry_match(
            normalized_query,
            _iter_command_entries(plugin),
            exact_score=6.0,
            prefix_score=4.2,
            sticky_score=3.5,
            contains_score=2.2,
            reason_prefix="direct_head",
        )
        if head_match is not None:
            candidate.add(head_match.score, head_match.reason)
            candidate.note_head_match(head_match.head)
        if candidate.score <= 0:
            continue
        if has_execute_intent:
            candidate.add(0.6, "intent_alignment")
        scored.append(
            RouteCandidateScore(
                plugin=plugin,
                score=float(candidate.score),
                reasons=tuple(candidate.reasons),
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def rank_route_candidates(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    preferred_modules: list[str] | tuple[str, ...] | None = None,
    metadata_filters: dict[str, bool] | None = None,
    min_score: float = 0.0,
) -> list[PluginInfo]:
    ranked_plugins, _ = _rank_route_candidates_internal(
        knowledge_base,
        query,
        preferred_modules=preferred_modules,
        metadata_filters=metadata_filters,
        min_score=min_score,
    )
    return ranked_plugins


def rank_route_candidates_with_scores(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    preferred_modules: list[str] | tuple[str, ...] | None = None,
    metadata_filters: dict[str, bool] | None = None,
    min_score: float = 0.0,
) -> list[RouteCandidateScore]:
    _, scored = _rank_route_candidates_internal(
        knowledge_base,
        query,
        preferred_modules=preferred_modules,
        metadata_filters=metadata_filters,
        min_score=min_score,
    )
    return scored


def _rank_route_candidates_internal(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    *,
    preferred_modules: list[str] | tuple[str, ...] | None = None,
    metadata_filters: dict[str, bool] | None = None,
    min_score: float = 0.0,
) -> tuple[list[PluginInfo], list[RouteCandidateScore]]:
    plugins = knowledge_base.plugins or []
    if not plugins:
        return [], []

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
            return [], []

        normalized_pref = [
            normalize_message_text(module).lower()
            for module in preferred_modules or ()
            if normalize_message_text(module)
        ]
        if not normalized_pref:
            return filtered_plugins, [
                RouteCandidateScore(
                    plugin=plugin,
                    score=0.0,
                    reasons=("query_empty",),
                )
                for plugin in filtered_plugins
            ]

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
        return filtered_plugins, [
            RouteCandidateScore(
                plugin=plugin,
                score=0.0,
                reasons=("query_empty",),
            )
            for plugin in filtered_plugins
        ]

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
        return [], []

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
            item.best_head_len,
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
        candidate_by_module = {item.plugin.module: item for item in scored}
        scored_rows: list[RouteCandidateScore] = []
        for plugin in ranked_plugins:
            candidate = candidate_by_module.get(plugin.module)
            if candidate is None:
                continue
            scored_rows.append(
                RouteCandidateScore(
                    plugin=plugin,
                    score=float(candidate.score),
                    reasons=tuple(candidate.reasons),
                )
            )
        return ranked_plugins, scored_rows
    return [], []


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
    "collect_direct_head_candidates",
    "rank_route_candidates",
    "rank_route_candidates_with_scores",
    "select_relevant_plugins",
    "select_route_candidates",
]
