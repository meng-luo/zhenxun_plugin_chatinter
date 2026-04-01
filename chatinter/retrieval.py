from collections.abc import Iterable
from dataclasses import dataclass, field
import re

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .route_text import ROUTE_ACTION_WORDS, contains_any, normalize_message_text
from .schema_policy import resolve_command_target_policy
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


def _plugin_text(plugin: PluginInfo) -> str:
    joined_commands = " ".join(plugin.commands or [])
    aliases = " ".join(plugin.aliases or [])
    command_meta_text = " ".join(
        " ".join(
            [
                meta.command,
                " ".join(meta.aliases),
                " ".join(meta.params),
                " ".join(meta.examples),
            ]
        )
        for meta in plugin.command_meta
    )
    usage = plugin.usage or ""
    return (
        f"{plugin.name} {plugin.module} {plugin.description} "
        f"{joined_commands} {aliases} {command_meta_text} {usage}"
    )


def _plugin_terms(plugin: PluginInfo) -> set[str]:
    terms = set(_tokenize(_plugin_text(plugin)))
    terms.add(normalize_message_text(plugin.name).lower())
    terms.add(_module_tail(plugin.module).lower())
    for command in plugin.commands or []:
        text = normalize_message_text(command).lower()
        if text:
            terms.add(text)
    for alias in plugin.aliases or []:
        text = normalize_message_text(alias).lower()
        if text:
            terms.add(text)
    for meta in plugin.command_meta:
        command = normalize_message_text(meta.command).lower()
        if command:
            terms.add(command)
        for alias in meta.aliases or []:
            text = normalize_message_text(alias).lower()
            if text:
                terms.add(text)
    return {term for term in terms if term}


def _plugin_metadata(plugin: PluginInfo) -> dict[str, bool]:
    has_helper = any(
        keyword in normalize_message_text(" ".join(plugin.commands or []))
        for keyword in ("帮助", "详情", "搜索", "用法", "参数", "说明", "列表")
    )
    target_capable = False
    image_capable = False
    self_only = False
    for meta in plugin.command_meta:
        policy = resolve_command_target_policy(meta)
        if policy.allow_at or bool(policy.target_sources & {"at", "reply", "nickname"}):
            target_capable = True
        if policy.actor_scope == "self_only":
            self_only = True
        image_min = int(getattr(meta, "image_min", 0) or 0)
        image_max = int(getattr(meta, "image_max", 0) or 0)
        if image_min > 0 or image_max > 0:
            image_capable = True
    return {
        "has_helper": has_helper,
        "target_capable": target_capable,
        "image_capable": image_capable,
        "self_only": self_only,
    }


def _satisfies_filters(
    plugin: PluginInfo,
    metadata_filters: dict[str, bool] | None,
) -> bool:
    if not metadata_filters:
        return True
    metadata = _plugin_metadata(plugin)
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
) -> _PluginCandidate | None:
    candidate = _PluginCandidate(plugin=plugin)
    normalized_query = normalize_message_text(query).lower()
    haystack = normalize_message_text(_plugin_text(plugin)).lower()
    terms = _plugin_terms(plugin)

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
        return plugins

    query_tokens = set(_tokenize(normalized_query))
    required_tokens = _required_tokens(normalized_query)
    has_execute_intent = contains_any(normalized_query, ROUTE_ACTION_WORDS)
    select_key = _extract_select_key(normalized_query)

    score_map: dict[str, _PluginCandidate] = {}
    for plugin in plugins:
        if not _satisfies_filters(plugin, metadata_filters):
            continue
        candidate = _build_candidate(
            plugin=plugin,
            query=normalized_query,
            query_tokens=query_tokens,
            required_tokens=required_tokens,
            has_execute_intent=has_execute_intent,
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
