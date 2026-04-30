"""
ChatInter 插件引用视图。

CapabilityGraph 是完整结构；PluginReference 是 Router / Planner 可直接使用的
轻量视图，避免在提示词中塞入过多无关字段。
"""

from __future__ import annotations

import re

from .command_schema import build_command_schemas
from .models.pydantic_models import (
    CapabilityGraphSnapshot,
    CommandCapability,
    PluginCapability,
    PluginCommandSchema,
    PluginReference,
)
from .route_text import normalize_message_text

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,6}", re.IGNORECASE)


def _append_unique(target: list[str], value: object) -> None:
    text = normalize_message_text(str(value or ""))
    if text and text not in target:
        target.append(text)


def _summarize_requirement(command: CommandCapability) -> dict[str, bool]:
    requirement = command.requirement
    return {
        "text": bool(requirement.text_min > 0 or requirement.params),
        "image": bool(requirement.image_min > 0),
        "reply": bool(requirement.requires_reply),
        "at": bool(
            requirement.allow_at
            or "at" in requirement.target_sources
            or requirement.target_requirement == "required"
        ),
    }


def _merge_requires(left: dict[str, bool], right: dict[str, bool]) -> dict[str, bool]:
    keys = {"text", "image", "reply", "at"}
    return {key: bool(left.get(key) or right.get(key)) for key in keys}


def _summarize_schema_requires(
    schemas: list[PluginCommandSchema],
) -> dict[str, bool]:
    # 插件级 requires 只保留“所有命令都必须”的条件，避免某个命令需要参数
    # 导致整个插件被误判为必须 reply/文本。
    keys = {"text", "image", "reply", "at"}
    if not schemas:
        return {key: False for key in keys}
    return {
        key: all(bool(schema.requires.get(key)) for schema in schemas) for key in keys
    }


def build_plugin_reference(plugin: PluginCapability) -> PluginReference:
    commands: list[str] = []
    aliases: list[str] = []
    examples: list[str] = []
    requires = {
        "text": False,
        "image": False,
        "reply": False,
        "at": False,
    }
    for command in plugin.commands:
        _append_unique(commands, command.command)
        for alias in command.aliases:
            _append_unique(aliases, alias)
        for example in command.examples:
            _append_unique(examples, example)
        requires = _merge_requires(requires, _summarize_requirement(command))

    for alias in plugin.aliases:
        _append_unique(aliases, alias)

    does = normalize_message_text(plugin.description or "")
    if does == "暂无描述" and plugin.usage:
        does = normalize_message_text(plugin.usage)
    if len(does) > 96:
        does = does[:96].rstrip() + "..."

    command_schemas = build_command_schemas(plugin.module, plugin.commands)
    if command_schemas:
        requires = _summarize_schema_requires(command_schemas)

    return PluginReference(
        module=plugin.module,
        name=plugin.name,
        does=does,
        commands=commands,
        aliases=aliases,
        examples=examples,
        requires=requires,
        command_schemas=command_schemas,
    )


def build_plugin_references(
    graph: CapabilityGraphSnapshot,
    *,
    limit: int | None = None,
) -> list[PluginReference]:
    references: list[PluginReference] = []
    max_items = max(int(limit), 0) if limit is not None else None
    if max_items == 0:
        return references

    for plugin in graph.plugins:
        references.append(build_plugin_reference(plugin))
        if max_items is not None and len(references) >= max_items:
            break
    return references


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(normalize_message_text(text))
        if token
    }


def _score_schema_for_query(schema: PluginCommandSchema, query: str) -> float:
    normalized = normalize_message_text(query).casefold()
    if not normalized:
        return 0.0
    score = 0.0
    head = normalize_message_text(schema.head).casefold()
    aliases = [
        normalize_message_text(alias).casefold()
        for alias in schema.aliases
        if normalize_message_text(alias)
    ]
    if head and normalized.startswith(head):
        score += 240.0
    elif head and head in normalized:
        score += 110.0 + min(len(head), 8)
    for alias in aliases:
        if alias and normalized.startswith(alias):
            score += 220.0
        elif alias and alias in normalized:
            score += 140.0 + min(len(alias), 12)
    text = normalize_message_text(
        " ".join(
            [
                schema.command_id,
                schema.head,
                " ".join(schema.aliases),
                schema.description,
                schema.command_role,
                schema.payload_policy,
            ]
        )
    )
    overlap = len(_tokens(normalized) & _tokens(text))
    if overlap:
        score += min(overlap * 14.0, 84.0)
    if schema.command_role == "catalog" and any(
        token in normalized for token in ("列表", "有哪些", "打开", "头像表情")
    ):
        score += 80.0
    if schema.command_role == "helper" and any(
        token in normalized for token in ("搜索", "找", "查找")
    ):
        score += 70.0
    if schema.command_role == "helper" and any(
        token in normalized
        for token in ("支持哪些", "哪些语言", "语种", "支持什么语言")
    ):
        score += 220.0
    if schema.command_role == "random" and "随机" in normalized:
        score += 100.0
    if schema.command_id == "memes.search" and any(
        token in normalized for token in ("相关表情", "找一下", "搜一下", "搜索")
    ):
        score += 260.0
    if schema.command_id == "memes.info" and any(
        token in normalized for token in ("怎么用", "用法", "详情")
    ):
        score += 220.0
    if schema.requires.get("text") and any(
        token in normalized
        for token in ("支持哪些", "哪些语言", "语种", "支持什么语言")
    ):
        score -= 360.0
    return score


def _select_prompt_schemas(
    schemas: list[PluginCommandSchema],
    query: str,
    *,
    limit: int = 8,
) -> list[PluginCommandSchema]:
    if not schemas:
        return []
    scored = [
        (_score_schema_for_query(schema, query), index, schema)
        for index, schema in enumerate(schemas)
    ]
    if not normalize_message_text(query):
        return [schema for _score, _index, schema in scored[:limit]]
    scored.sort(
        key=lambda item: (
            item[0],
            item[2].command_role in {"catalog", "helper", "random"},
            -item[1],
        ),
        reverse=True,
    )
    selected = [schema for score, _index, schema in scored if score > 0][:limit]
    if selected:
        return selected
    return [schema for _score, _index, schema in scored[:limit]]


def _dump_schema_for_prompt(schema: PluginCommandSchema) -> dict[str, object]:
    payload: dict[str, object] = {
        "command_id": schema.command_id,
        "head": schema.head,
        "role": schema.command_role,
        "payload_policy": schema.payload_policy,
        "extra_text_policy": schema.extra_text_policy,
    }
    if schema.aliases:
        payload["aliases"] = schema.aliases[:8]
    if schema.description:
        payload["description"] = schema.description
    if schema.slots:
        payload["slots"] = [
            {
                key: value
                for key, value in {
                    "name": slot.name,
                    "type": slot.type,
                    "required": slot.required or None,
                    "default": slot.default,
                    "aliases": slot.aliases[:5] or None,
                    "description": slot.description or None,
                }.items()
                if value is not None
            }
            for slot in schema.slots[:4]
        ]
    if schema.render and schema.render != schema.head:
        payload["render"] = schema.render
    true_requires = {
        key: value for key, value in (schema.requires or {}).items() if value
    }
    if true_requires:
        payload["requires"] = true_requires
    return payload


def build_router_cards_from_graph(
    graph: CapabilityGraphSnapshot,
    *,
    limit: int | None = None,
    query: str = "",
) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    references = build_plugin_references(graph, limit=None)
    if normalize_message_text(query):
        references.sort(
            key=lambda reference: (
                max(
                    (
                        _score_schema_for_query(schema, query)
                        for schema in reference.command_schemas
                    ),
                    default=0.0,
                ),
                bool(reference.command_schemas),
            ),
            reverse=True,
        )
    if limit is not None:
        references = references[: max(int(limit), 0)]
    for reference in references:
        selected_schemas = _select_prompt_schemas(
            reference.command_schemas,
            query,
            limit=8,
        )
        cards.append(
            {
                "module": reference.module,
                "name": reference.name,
                "commands": reference.commands[:10],
                "aliases": reference.aliases[:6],
                "does": reference.does,
                "examples": reference.examples[:4],
                "requires": reference.requires,
                "command_schemas": [
                    _dump_schema_for_prompt(schema) for schema in selected_schemas
                ],
            }
        )
    return cards
