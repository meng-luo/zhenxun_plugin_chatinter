"""Reviewable command metadata overrides for the capability catalog.

离线富化脚本（scripts/chatinter_enrich_commands.py）批量生成、人工可审查的
每命令元数据：标准化一行能力描述、口语化别名/示例、以及"语义天然含糊"命令的
显式排除标记。运行时只做数据合并——任何插件语义知识都必须以这里的数据形式
存在，禁止写进代码。

文件格式（data/chatinter/command_metadata_overrides.json）：
{
  "schema_version": 2,
  "commands": {
    "<command_id>": {
      "description": "一行能力描述（可选，非空则替换）",
      "aliases": ["口语别名", ...],
      "examples": ["口语示例", ...],
      "use_cases": ["适用场景", ...],
      "anti_use_cases": ["不适用场景", ...],
      "side_effect": "none" | "query" | "send" | "mutate",
      "source_of_truth": "plugin_runtime" | "bot_state" | "external_service" | ...,
      "requires_real_tool": true,
      "intent_types": ["query", "status", ...],
      "requires_real_result": true,
      "execution_policy": "normal" | "explicit_only" | "strong_intent"
                          | "confirmation_required",
      "exclude": false,
      "exclude_reason": "",
      "source": "llm_enrichment" | "manual",
      "confidence": 0.0
    }
  }
}
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from zhenxun.services.log import logger

from .models.pydantic_models import CommandToolSnapshot
from .route_text import normalize_message_text

OVERRIDES_PATH = Path("data/chatinter/command_metadata_overrides.json")
_ALIAS_LIMIT = 12
_EXAMPLE_LIMIT = 6
_CASE_LIMIT = 8
_ENUM_FIELDS: dict[str, set[str]] = {
    "output_mode": {"text", "image", "file", "plugin_output", "action"},
    "side_effect": {"none", "query", "send", "mutate"},
    "risk": {"low", "medium", "high"},
    "source_of_truth": {
        "model_knowledge",
        "plugin_runtime",
        "bot_state",
        "external_service",
        "local_state",
        "user_provided",
        "unknown",
    },
    "entity_scope": {
        "none",
        "self_bot",
        "actor_user",
        "target_user",
        "group",
        "global",
        "external",
    },
    "execution_policy": {
        "normal",
        "explicit_only",
        "strong_intent",
        "confirmation_required",
    },
}
_BOOL_FIELDS = {"requires_real_tool", "requires_real_result"}


@dataclass(frozen=True)
class CommandMetadataOverrides:
    version: str
    commands: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def excluded_command_ids(self) -> set[str]:
        return {
            command_id
            for command_id, item in self.commands.items()
            if bool(item.get("exclude"))
        }

    def apply(
        self,
        snapshots: list[CommandToolSnapshot],
    ) -> list[CommandToolSnapshot]:
        if not self.commands:
            return snapshots
        result: list[CommandToolSnapshot] = []
        for snapshot in snapshots:
            override = self.commands.get(
                normalize_message_text(snapshot.command_id)
            )
            if override is None:
                result.append(snapshot)
                continue
            if bool(override.get("exclude")):
                continue
            result.append(_apply_one(snapshot, override))
        return result


_EMPTY_OVERRIDES = CommandMetadataOverrides(version="none")
_cached: CommandMetadataOverrides | None = None
_cached_stat: tuple[float, int] | None = None


def load_command_overrides(
    path: Path = OVERRIDES_PATH,
) -> CommandMetadataOverrides:
    global _cached, _cached_stat
    try:
        stat = path.stat()
    except OSError:
        _cached = _EMPTY_OVERRIDES
        _cached_stat = None
        return _cached
    stat_key = (stat.st_mtime, stat.st_size)
    if _cached is not None and _cached_stat == stat_key:
        return _cached
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        commands_raw = raw.get("commands")
        commands = {
            normalize_message_text(command_id): dict(item)
            for command_id, item in (commands_raw or {}).items()
            if normalize_message_text(command_id) and isinstance(item, dict)
        }
        _cached = CommandMetadataOverrides(
            version=f"{stat.st_mtime:.0f}:{stat.st_size}",
            commands=commands,
        )
    except Exception as exc:
        _has_valid = _cached is not None and _cached is not _EMPTY_OVERRIDES
        logger.warning(
            f"ChatInter 命令元数据 overrides 加载失败：{exc}。"
            + ("沿用上次有效版本。" if _has_valid else "使用空覆盖。")
        )
        if _cached is None:
            _cached = _EMPTY_OVERRIDES
    _cached_stat = stat_key
    return _cached


def _apply_one(
    snapshot: CommandToolSnapshot,
    override: dict[str, Any],
) -> CommandToolSnapshot:
    update: dict[str, Any] = {}
    description = normalize_message_text(str(override.get("description", "") or ""))
    if description:
        update["description"] = description
    extra_aliases = _clean_texts(override.get("aliases"))
    if extra_aliases:
        update["aliases"] = _merge_texts(
            snapshot.aliases,
            extra_aliases,
            limit=_ALIAS_LIMIT,
        )
    extra_examples = _clean_texts(override.get("examples"))
    if extra_examples:
        update["examples"] = _merge_texts(
            snapshot.examples,
            extra_examples,
            limit=_EXAMPLE_LIMIT,
        )
    for field_name in ("use_cases", "anti_use_cases"):
        values = _clean_texts(override.get(field_name))
        if values:
            update[field_name] = _merge_texts(
                list(getattr(snapshot, field_name)),
                values,
                limit=_CASE_LIMIT,
            )
    intent_types = _clean_texts(override.get("intent_types"))
    if intent_types:
        update["intent_types"] = _merge_texts(
            snapshot.intent_types,
            intent_types,
            limit=_CASE_LIMIT,
        )
    for field_name, allowed in _ENUM_FIELDS.items():
        value = normalize_message_text(str(override.get(field_name, "") or ""))
        if value in allowed:
            update[field_name] = value
    if "risk" in update:
        update["risk_level"] = update["risk"]
    for field_name in _BOOL_FIELDS:
        value = override.get(field_name)
        if isinstance(value, bool):
            update[field_name] = value
    if not update:
        return snapshot
    update["source"] = "override"
    return snapshot.model_copy(update=update)


def _clean_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = normalize_message_text(str(item or ""))
        if text and text not in result:
            result.append(text)
    return result


def _merge_texts(
    original: list[str],
    extra: list[str],
    *,
    limit: int,
) -> list[str]:
    merged: list[str] = []
    for text in [*original, *extra]:
        normalized = normalize_message_text(str(text or ""))
        if normalized and normalized not in merged:
            merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


__all__ = [
    "OVERRIDES_PATH",
    "CommandMetadataOverrides",
    "load_command_overrides",
]
