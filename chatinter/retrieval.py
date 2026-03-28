import re

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .skill_registry import (
    get_skill_registry,
    select_relevant_skills,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)
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
def _tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text.lower())]
    return [token for token in tokens if token and token not in _STOPWORDS]


def _plugin_text(plugin: PluginInfo) -> str:
    joined_commands = " ".join(plugin.commands)
    aliases = " ".join(plugin.aliases)
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
        f"{plugin.name} {plugin.description} "
        f"{joined_commands} {aliases} {command_meta_text} {usage}"
    )


def _pin_plugins(
    selected: list[PluginInfo],
    knowledge_base: PluginKnowledgeBase,
    query: str,
    limit: int,
) -> list[PluginInfo]:
    pinned: list[PluginInfo] = []
    pinned_modules: set[str] = set()

    registry = get_skill_registry(knowledge_base)
    for skill in select_relevant_skills(registry, query, limit=4):
        plugin = next(
            (
                item
                for item in knowledge_base.plugins
                if item.module == skill.plugin_module
            ),
            None,
        )
        if plugin is None or plugin.module in pinned_modules:
            continue
        pinned_modules.add(plugin.module)
        pinned.append(plugin)

    lowered_query = query.lower()
    for plugin in knowledge_base.plugins:
        if plugin.module in pinned_modules:
            continue
        if any(
            command and command.lower() in lowered_query
            for command in plugin.commands
        ):
            pinned_modules.add(plugin.module)
            pinned.append(plugin)
            if len(pinned) >= 3:
                break

    merged: list[PluginInfo] = []
    merged_modules: set[str] = set()
    for plugin in [*pinned, *selected]:
        if plugin.module in merged_modules:
            continue
        merged_modules.add(plugin.module)
        merged.append(plugin)
        if len(merged) >= limit:
            break
    return merged


def _score_plugin(query_tokens: set[str], plugin: PluginInfo) -> float:
    if not query_tokens:
        return 0.0
    haystack = _plugin_text(plugin).lower()
    score = 0.0
    alias_tokens = {alias.lower() for alias in plugin.aliases}
    command_tokens = {command.lower() for command in plugin.commands}
    for token in query_tokens:
        if token in haystack:
            score += 1.0
            if plugin.name.lower().startswith(token):
                score += 1.0
            if any(cmd.lower().startswith(token) for cmd in plugin.commands):
                score += 0.8
            if token in alias_tokens:
                score += 0.8
            if token in command_tokens:
                score += 1.0
    return score


def select_relevant_plugins(
    knowledge_base: PluginKnowledgeBase,
    query: str,
    limit: int = 8,
) -> PluginKnowledgeBase:
    if not knowledge_base.plugins:
        return knowledge_base

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return knowledge_base

    scored = [
        (plugin, _score_plugin(query_tokens, plugin))
        for plugin in knowledge_base.plugins
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    filtered = [plugin for plugin, score in scored if score > 0]
    if not filtered:
        filtered = knowledge_base.plugins[:limit]
    filtered = _pin_plugins(filtered, knowledge_base, query, limit)
    return PluginKnowledgeBase(
        plugins=filtered[:limit],
        user_role=knowledge_base.user_role,
    )
