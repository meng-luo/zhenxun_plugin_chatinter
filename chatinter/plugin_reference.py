"""
ChatInter 插件引用视图。

CapabilityGraph 是完整结构；PluginReference 是 Native tools / Planner 可直接使用的
轻量视图，避免在提示词中塞入过多无关字段。
"""

from __future__ import annotations

from hashlib import sha1
import re
from typing import Any

from zhenxun.utils.pydantic_compat import model_dump

from .capability_card import build_capability_card
from .command_schema import build_command_schemas
from .models.pydantic_models import (
    CapabilityGraphSnapshot,
    CommandCapability,
    CommandToolSnapshot,
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


def _command_family(schema: PluginCommandSchema, *, plugin_module: str) -> str:
    module = normalize_message_text(plugin_module).casefold()
    if schema.command_role in {"catalog", "helper", "usage"}:
        return schema.command_role
    return module.rsplit(".", 1)[-1] or "general"


def _infer_task_verbs(
    schema: PluginCommandSchema,
    reference: PluginReference,
) -> list[str]:
    text = normalize_message_text(
        " ".join(
            [
                schema.head,
                " ".join(schema.aliases),
                schema.description,
                reference.does,
                schema.command_role,
                schema.payload_policy,
            ]
        )
    )
    verb_groups = {
        "查询": ("查", "查询", "查看", "搜索", "搜", "找", "识别", "解析"),
        "生成": ("生成", "制作", "做", "来", "发送", "随机"),
        "添加": ("添加", "新增", "创建", "设置", "绑定"),
        "删除": ("删除", "移除", "取消", "关闭", "退回", "解绑"),
        "翻译": ("翻译", "语种", "缩写", "简称", "黑话", "解释", "展开"),
        "播放": ("播放", "点播", "音乐", "歌曲", "点歌", "搜歌"),
        "统计": ("统计", "排行", "词云", "报告"),
        "帮助": ("用法", "教程", "帮助", "说明", "参数", "示例"),
    }
    verbs: list[str] = []
    for verb, keywords in verb_groups.items():
        if any(keyword in text for keyword in keywords):
            _append_unique(verbs, verb)
    if not verbs and schema.command_role in {"helper", "usage", "catalog"}:
        _append_unique(verbs, "帮助")
    if not verbs:
        _append_unique(verbs, "执行")
    return verbs


def _input_requirements(schema: PluginCommandSchema) -> list[str]:
    requirements: list[str] = []
    requires = schema.requires or {}
    has_text_slot = any(slot.type == "text" for slot in schema.slots)
    if requires.get("text") or schema.payload_policy in {"text", "free_tail"}:
        _append_unique(requirements, "文本")
    elif schema.payload_policy == "slots" and has_text_slot:
        _append_unique(requirements, "文本")
    if requires.get("image") or schema.payload_policy in {
        "image_only",
        "text_or_image",
    }:
        _append_unique(requirements, "图片")
    if requires.get("reply"):
        _append_unique(requirements, "回复")
    if requires.get("at"):
        _append_unique(requirements, "@目标")
    slot_text = normalize_message_text(
        " ".join(
            " ".join([slot.name, slot.description, *slot.aliases])
            for slot in schema.slots
        )
    ).casefold()
    if any(token in slot_text for token in ("url", "链接", "地址", "bv", "av")):
        _append_unique(requirements, "链接")
    if any(token in slot_text for token in ("num", "count", "amount", "数量", "金额")):
        _append_unique(requirements, "数字")
    return requirements


def _capability_text(
    *,
    reference: PluginReference,
    schema: PluginCommandSchema,
    task_verbs: list[str],
    input_requirements: list[str],
) -> str:
    parts = [
        reference.name,
        schema.head,
        schema.description or reference.does,
        "动作:" + "/".join(task_verbs) if task_verbs else "",
        "输入:" + "/".join(input_requirements) if input_requirements else "",
    ]
    return normalize_message_text("；".join(part for part in parts if part))


def _schema_signature(
    *,
    module: str,
    name: str,
    usage: str | None,
    schema: PluginCommandSchema,
) -> str:
    payload = {
        "module": module,
        "name": name,
        "usage": usage or "",
        "schema": model_dump(schema),
    }
    return sha1(repr(payload).encode("utf-8")).hexdigest()


def _schema_meta(schema: PluginCommandSchema) -> dict[str, object]:
    requires = schema.requires or {}
    text_slots = [slot for slot in schema.slots if slot.type == "text"]
    image_slots = [slot for slot in schema.slots if slot.type == "image"]
    shortcut_renders = getattr(schema, "shortcut_renders", None)
    meta = {
        "text_min": sum(1 for slot in text_slots if slot.required),
        "text_max": len(text_slots)
        if text_slots
        else (1 if requires.get("text") else 0),
        "image_min": 1 if requires.get("image") else 0,
        "image_max": len(image_slots)
        if image_slots
        else (1 if requires.get("image") else 0),
        "target_accepts_at": bool(requires.get("at") or schema.allow_at),
    }
    if isinstance(shortcut_renders, list) and shortcut_renders:
        meta["shortcut_renders"] = shortcut_renders[:24]
    slot_choices = {
        slot.name: list(slot.choices)
        for slot in schema.slots
        if getattr(slot, "choices", None)
    }
    if slot_choices:
        meta["slot_choices"] = slot_choices
    return meta


def _summarize_requirement(command: CommandCapability) -> dict[str, bool]:
    requirement = command.requirement
    return {
        "text": bool(requirement.text_min > 0 or requirement.params),
        "image": bool(requirement.image_min > 0),
        "reply": bool(requirement.requires_reply),
        "private": bool(requirement.requires_private),
        "to_me": bool(requirement.requires_to_me),
        "at": bool(
            requirement.allow_at
            or "at" in requirement.target_sources
            or requirement.target_requirement == "required"
        ),
    }


def _merge_requires(left: dict[str, bool], right: dict[str, bool]) -> dict[str, bool]:
    keys = {"text", "image", "reply", "at", "private", "to_me"}
    return {key: bool(left.get(key) or right.get(key)) for key in keys}


def _summarize_schema_requires(
    schemas: list[PluginCommandSchema],
) -> dict[str, bool]:


    keys = {"text", "image", "reply", "at", "private", "to_me"}
    if not schemas:
        return {key: False for key in keys}
    return {
        key: any(bool(schema.requires.get(key)) for schema in schemas) for key in keys
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
        _append_unique(examples, command.description)
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

    command_schemas = build_command_schemas(
        plugin.module,
        plugin.commands,
        usage=plugin.usage,
        plugin_description=plugin.description,
    )
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


def _schema_blocked_by_quality_gate(schema: Any, *, module: str) -> bool:
    """Keep fallback-source schemas out of the tool pool unless allowlisted."""
    from .config import EXPOSE_FALLBACK_SCHEMAS, SCHEMA_FALLBACK_ALLOWLIST

    if str(getattr(schema, "source", "") or "") != "fallback":
        return False
    if EXPOSE_FALLBACK_SCHEMAS:
        return False
    return module not in SCHEMA_FALLBACK_ALLOWLIST


def build_command_tool_snapshots(
    graph: CapabilityGraphSnapshot,
    *,
    limit: int | None = None,
) -> list[CommandToolSnapshot]:
    """Flatten safe plugin capabilities into command-level tools."""

    snapshots: list[CommandToolSnapshot] = []
    max_items = max(int(limit), 0) if limit is not None else None
    if max_items == 0:
        return snapshots

    for plugin in graph.plugins:
        reference = build_plugin_reference(plugin)
        plugin_usage = normalize_message_text(plugin.usage or "") or None
        multi_command = len(reference.command_schemas) > 1
        for schema in reference.command_schemas:
            if _schema_blocked_by_quality_gate(schema, module=reference.module):
                continue
            family = _command_family(schema, plugin_module=reference.module)
            task_verbs = _infer_task_verbs(schema, reference)
            input_requirements = _input_requirements(schema)
            capability_text = _capability_text(
                reference=reference,
                schema=schema,
                task_verbs=task_verbs,
                input_requirements=input_requirements,
            )
            global_phrases = (
                []
                if multi_command
                else [reference.does, plugin_usage or "", *reference.examples]
            )
            phrases: list[str] = []
            for value in [
                schema.command_id,
                schema.head,
                *schema.aliases,
                schema.description,
                capability_text,
                " ".join(task_verbs),
                " ".join(input_requirements),
                *global_phrases,
                *schema.retrieval_phrases,
            ]:
                _append_unique(phrases, value)
            snapshot = CommandToolSnapshot(
                command_id=schema.command_id,
                plugin_module=reference.module,
                plugin_name=reference.name,
                head=schema.head,
                aliases=list(schema.aliases),
                description=schema.description
                or ("" if multi_command else reference.does),
                usage=None if multi_command else plugin_usage,
                examples=[] if multi_command else list(reference.examples),
                slots=list(schema.slots),
                requires=dict(schema.requires or {}),
                allow_at=schema.allow_at,
                actor_scope=schema.actor_scope,
                target_requirement=schema.target_requirement,
                target_sources=list(schema.target_sources),
                render=schema.render,
                payload_policy=schema.payload_policy,
                extra_text_policy=schema.extra_text_policy,
                command_role=schema.command_role,
                family=family,
                retrieval_phrases=phrases,
                capability_text=capability_text,
                task_verbs=task_verbs,
                input_requirements=input_requirements,
                source=schema.source,
                confidence=schema.confidence,
                matcher_key=schema.matcher_key,
                source_signature=_schema_signature(
                    module=reference.module,
                    name=reference.name,
                    usage=plugin_usage,
                    schema=schema,
                ),
                meta=_schema_meta(schema),
            )
            card = build_capability_card(
                tool=snapshot,
                family=family,
                description=snapshot.description or capability_text,
            )
            snapshot = snapshot.model_copy(
                update={
                    "use_cases": list(card.use_cases),
                    "anti_use_cases": list(card.anti_use_cases),
                    "output_mode": card.output_mode,
                    "side_effect": card.side_effect,
                    "risk_level": card.risk_level,
                    "risk": card.risk,
                    "source_of_truth": card.source_of_truth,
                    "requires_real_tool": card.requires_real_tool,
                    "entity_scope": card.entity_scope,
                    "reliability": card.reliability,
                    "schema_quality": card.schema_quality,
                    "soft_tool": card.soft_tool,
                    "intent_types": list(card.intent_types),
                    "requires_real_result": card.requires_real_result,
                    "generative": card.generative,
                    "execution_policy": card.execution_policy,
                }
            )
            snapshots.append(snapshot)
            if max_items is not None and len(snapshots) >= max_items:
                return snapshots
    return snapshots
