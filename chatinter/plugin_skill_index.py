from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json

from zhenxun.services.log import logger

from .models.pydantic_models import (
    CommandToolSnapshot,
    PluginInfo,
    PluginKnowledgeBase,
    SemanticToolContract,
)
from .route_text import normalize_message_text

_DESCRIPTION_LIMIT = 180
_INDEX_CACHE_LIMIT = 16
_SOURCE_INDEX_CACHE_LIMIT = 16
_DEBUG_CACHE_LIMIT = 16
_INDEX_CACHE: OrderedDict[str, "PluginSkillIndex"] = OrderedDict()
_SOURCE_INDEX_CACHE: OrderedDict[
    tuple[int, tuple[int, ...]],
    tuple[
        PluginKnowledgeBase,
        tuple[CommandToolSnapshot, ...],
        "PluginSkillIndex",
    ],
] = OrderedDict()
_DEBUGGED_FINGERPRINTS: OrderedDict[str, None] = OrderedDict()


@dataclass(frozen=True, slots=True)
class PluginSkill:
    skill_id: str
    plugin_module: str
    plugin_name: str
    description: str
    command_ids: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    usage: str = ""
    introduction: str = ""
    precautions: tuple[str, ...] = ()
    semantic_tools: tuple[SemanticToolContract, ...] = ()
    input_types: tuple[str, ...] = ()
    command_count: int = 0


@dataclass(frozen=True, slots=True)
class PluginSkillIndex:
    fingerprint: str
    skills: tuple[PluginSkill, ...]
    command_count: int


def build_plugin_skill_index(
    knowledge_base: PluginKnowledgeBase,
    command_tools: list[CommandToolSnapshot],
) -> PluginSkillIndex:
    source_key = (id(knowledge_base), tuple(id(tool) for tool in command_tools))
    source_cached = _SOURCE_INDEX_CACHE.get(source_key)
    if source_cached is not None:
        cached_knowledge, cached_tools, cached_index = source_cached
        if cached_knowledge is knowledge_base and len(cached_tools) == len(
            command_tools
        ) and all(
            cached is current
            for cached, current in zip(cached_tools, command_tools, strict=True)
        ):
            _SOURCE_INDEX_CACHE.move_to_end(source_key)
            return cached_index
        _SOURCE_INDEX_CACHE.pop(source_key, None)

    plugins_by_module = {
        _module_key(plugin.module): plugin
        for plugin in knowledge_base.plugins
        if _module_key(plugin.module)
    }
    tools_by_module: dict[str, list[CommandToolSnapshot]] = {}
    modules: dict[str, str] = {}
    for tool in command_tools:
        module = normalize_message_text(tool.plugin_module)
        key = _module_key(module)
        if not key or key not in plugins_by_module:
            continue
        modules.setdefault(key, module)
        tools_by_module.setdefault(key, []).append(tool)

    raw_skills: list[tuple[str, str, PluginInfo, tuple[str, ...]]] = []
    for key in sorted(tools_by_module):
        plugin = plugins_by_module[key]
        module = modules[key]
        command_ids = tuple(
            sorted(
                {
                    normalize_message_text(tool.command_id)
                    for tool in tools_by_module[key]
                    if normalize_message_text(tool.command_id)
                },
                key=lambda value: (value.casefold(), value),
            )
        )
        if command_ids:
            raw_skills.append((_base_skill_id(module), module, plugin, command_ids))

    duplicate_ids = {
        skill_id
        for skill_id in {item[0] for item in raw_skills}
        if sum(1 for item in raw_skills if item[0] == skill_id) > 1
    }
    skills = tuple(
        sorted(
            (
                PluginSkill(
                    skill_id=(
                        _qualified_skill_id(module)
                        if base_skill_id in duplicate_ids
                        else base_skill_id
                    ),
                    plugin_module=module,
                    plugin_name=_single_line(plugin.name) or base_skill_id,
                    description=_skill_description(
                        plugin, tools_by_module[_module_key(module)]
                    ),
                    command_ids=command_ids,
                    aliases=_stable_values(plugin.aliases),
                    usage=normalize_message_text(str(plugin.usage or "")),
                    introduction=normalize_message_text(
                        str(plugin.introduction or "")
                    ),
                    precautions=_stable_values(plugin.precautions),
                    semantic_tools=tuple(
                        sorted(
                            plugin.semantic_tools,
                            key=lambda item: (item.name.casefold(), item.name),
                        )
                    ),
                    input_types=_skill_input_types(
                        tools_by_module[_module_key(module)]
                    ),
                    command_count=len(command_ids),
                )
                for base_skill_id, module, plugin, command_ids in raw_skills
            ),
            key=lambda skill: (
                skill.skill_id.casefold(),
                skill.plugin_module.casefold(),
            ),
        )
    )
    fingerprint = _index_fingerprint(skills)
    cached = _INDEX_CACHE.get(fingerprint)
    if cached is not None:
        _INDEX_CACHE.move_to_end(fingerprint)
        _remember_source_index(source_key, knowledge_base, command_tools, cached)
        return cached

    index = PluginSkillIndex(
        fingerprint=fingerprint,
        skills=skills,
        command_count=sum(len(skill.command_ids) for skill in skills),
    )
    _INDEX_CACHE[fingerprint] = index
    while len(_INDEX_CACHE) > _INDEX_CACHE_LIMIT:
        _INDEX_CACHE.popitem(last=False)
    _remember_source_index(source_key, knowledge_base, command_tools, index)
    return index


def _remember_source_index(
    source_key: tuple[int, tuple[int, ...]],
    knowledge_base: PluginKnowledgeBase,
    command_tools: list[CommandToolSnapshot],
    index: PluginSkillIndex,
) -> None:
    _SOURCE_INDEX_CACHE[source_key] = (
        knowledge_base,
        tuple(command_tools),
        index,
    )
    _SOURCE_INDEX_CACHE.move_to_end(source_key)
    while len(_SOURCE_INDEX_CACHE) > _SOURCE_INDEX_CACHE_LIMIT:
        _SOURCE_INDEX_CACHE.popitem(last=False)


def skills_by_id(index: PluginSkillIndex) -> dict[str, PluginSkill]:
    return {skill.skill_id.casefold(): skill for skill in index.skills}


def render_skill_debug_listing(index: PluginSkillIndex) -> str:
    lines = ["x = 模型可见（已通过权限和基础设施过滤）"]
    lines.extend(
        f"[{position:3d}][x] {skill.skill_id} - {skill.description}"
        for position, skill in enumerate(index.skills, start=1)
    )
    if not index.skills:
        lines.append("允许列表为空，模型看不到任何插件技能。")
    return "\n".join(lines)


def log_skill_debug_once(index: PluginSkillIndex) -> None:
    if index.fingerprint in _DEBUGGED_FINGERPRINTS:
        _DEBUGGED_FINGERPRINTS.move_to_end(index.fingerprint)
        return
    _DEBUGGED_FINGERPRINTS[index.fingerprint] = None
    while len(_DEBUGGED_FINGERPRINTS) > _DEBUG_CACHE_LIMIT:
        _DEBUGGED_FINGERPRINTS.popitem(last=False)
    logger.debug(
        "ChatInter 固定 Plugin Skill 索引：\n" + render_skill_debug_listing(index)
    )


def _base_skill_id(module: str) -> str:
    normalized = _module_key(module)
    for prefix in ("zhenxun.plugins.", "plugins."):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized.rsplit(".", 1)[-1]


def _qualified_skill_id(module: str) -> str:
    normalized = _module_key(module)
    for prefix in (
        "zhenxun.builtin_plugins.",
        "zhenxun.plugins.",
        "plugins.",
    ):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _module_key(module: str) -> str:
    return normalize_message_text(module).casefold()


def _skill_description(
    plugin: PluginInfo,
    tools: list[CommandToolSnapshot],
) -> str:
    description = _single_line(plugin.description)
    if not description:
        description = next(
            (
                _single_line(tool.description)
                for tool in sorted(tools, key=lambda item: item.command_id)
                if _single_line(tool.description)
            ),
            "",
        )
    return _clip(description or _single_line(plugin.name) or "插件功能")


def _single_line(value: object) -> str:
    return " ".join(normalize_message_text(str(value or "")).split())


def _stable_values(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        source = (values,)
    else:
        try:
            source = tuple(values or ())
        except TypeError:
            source = ()
    return tuple(
        sorted(
            {
                normalize_message_text(str(value or ""))
                for value in source
                if normalize_message_text(str(value or ""))
            },
            key=lambda value: (value.casefold(), value),
        )
    )


def _skill_input_types(
    tools: list[CommandToolSnapshot],
) -> tuple[str, ...]:
    values: list[str] = []
    for tool in tools:
        values.extend(tool.input_requirements)
        values.extend(slot.type for slot in tool.slots)
        values.extend(key for key, required in tool.requires.items() if required)
    return _stable_values(values)


def _clip(text: str) -> str:
    if len(text) <= _DESCRIPTION_LIMIT:
        return text
    return text[: _DESCRIPTION_LIMIT - 1].rstrip() + "…"


def _index_fingerprint(skills: tuple[PluginSkill, ...]) -> str:
    digest = hashlib.blake2s(digest_size=16)
    for skill in skills:
        payload = {
            "skill_id": skill.skill_id,
            "plugin_module": skill.plugin_module,
            "plugin_name": skill.plugin_name,
            "description": skill.description,
            "command_ids": skill.command_ids,
            "aliases": skill.aliases,
            "usage": skill.usage,
            "introduction": skill.introduction,
            "precautions": skill.precautions,
            "semantic_tools": [
                item.model_dump(mode="json") for item in skill.semantic_tools
            ],
            "input_types": skill.input_types,
            "command_count": skill.command_count,
        }
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8", "ignore")
        )
        digest.update(b"\x1e")
    return digest.hexdigest()


__all__ = [
    "PluginSkill",
    "PluginSkillIndex",
    "build_plugin_skill_index",
    "log_skill_debug_once",
    "render_skill_debug_listing",
    "skills_by_id",
]
