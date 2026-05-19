"""Capability registry for ChatInter command discovery.

The registry owns the complete, safe command capability set.  Local code only
recalls a small candidate set from it; the model still chooses the executable
command tool from injected full schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_graph import build_capability_graph_snapshot
from .command_index import (
    CommandCandidate,
    dump_candidate_for_prompt,
    retrieve_command_candidates,
)
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .plugin_reference import build_command_tool_snapshots
from .route_text import normalize_message_text

_ADMIN_KEYWORDS = (
    "admin",
    "superuser",
    "管理员",
    "超级用户",
    "群管",
)
_DESTRUCTIVE_KEYWORDS = (
    "删除",
    "清空",
    "禁用",
    "关闭",
    "重启",
    "封禁",
    "拉黑",
    "退群",
    "踢",
)
_COST_KEYWORDS = ("金币", "余额", "花费", "消耗", "红包", "抽奖", "购买")


@dataclass(frozen=True)
class CapabilityRecord:
    """One discoverable plugin command capability."""

    command_id: str
    plugin_module: str
    plugin_name: str
    family: str
    tool: CommandToolSnapshot
    description: str = ""
    risk_tags: tuple[str, ...] = ()
    permission_tags: tuple[str, ...] = ()
    search_text: str = ""

    def to_prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command_id": self.command_id,
            "plugin_module": self.plugin_module,
            "plugin_name": self.plugin_name,
            "family": self.family,
            "description": self.description,
            "risk_tags": list(self.risk_tags),
            "permission_tags": list(self.permission_tags),
            "head": self.tool.head,
            "aliases": list(self.tool.aliases),
            "payload_policy": self.tool.payload_policy,
            "target_policy": {
                "requirement": self.tool.target_requirement,
                "sources": list(self.tool.target_sources),
                "allow_at": self.tool.allow_at,
                "actor_scope": self.tool.actor_scope,
            },
            "slots": [
                {
                    "name": slot.name,
                    "type": slot.type,
                    "required": slot.required,
                    "aliases": list(slot.aliases),
                    "description": slot.description,
                }
                for slot in self.tool.slots
            ],
        }
        return {key: value for key, value in payload.items() if value not in ("", [])}


class CapabilityRegistry:
    """Registry of all safe command capabilities for one turn/session."""

    def __init__(
        self,
        *,
        knowledge_base: PluginKnowledgeBase,
        tools: list[CommandToolSnapshot],
        session_id: str | None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.tools = list(tools)
        self.session_id = session_id
        self.records = {
            normalize_message_text(tool.command_id): _record_from_tool(tool)
            for tool in self.tools
            if normalize_message_text(tool.command_id)
        }

    @classmethod
    def from_knowledge_base(
        cls,
        knowledge_base: PluginKnowledgeBase,
        *,
        session_id: str | None,
        tools: list[Any] | None = None,
    ) -> "CapabilityRegistry":
        command_tools = _ensure_command_tools(knowledge_base, tools)
        return cls(
            knowledge_base=knowledge_base,
            tools=command_tools,
            session_id=session_id,
        )

    @property
    def total_commands(self) -> int:
        return len(self.tools)

    def recall(
        self,
        query: str,
        *,
        limit: int,
        diversify: bool = True,
    ) -> list[CommandCandidate]:
        """Return recall candidates only; never treat ranking as final choice."""

        candidates = retrieve_command_candidates(
            self.knowledge_base,
            query,
            limit=limit,
            session_id=self.session_id,
            diversify=diversify,
            tools=self.tools,
        )
        return [
            _mark_registry_recall(candidate)
            for candidate in candidates
            if self.record_for(candidate.schema.command_id) is not None
        ]

    def record_for(self, command_id: str) -> CapabilityRecord | None:
        return self.records.get(normalize_message_text(command_id))

    def candidate_payload(
        self,
        candidate: CommandCandidate,
        *,
        index: int,
    ) -> dict[str, Any]:
        payload = dump_candidate_for_prompt(candidate, index=index)
        record = self.record_for(candidate.schema.command_id)
        if record is None:
            return payload
        payload["selection_policy"] = (
            "local_recall_only; choose by executable schema, not by rank"
        )
        payload["risk_tags"] = list(record.risk_tags)
        payload["permission_tags"] = list(record.permission_tags)
        payload["capability"] = record.to_prompt_payload()
        return payload


def _ensure_command_tools(
    knowledge_base: PluginKnowledgeBase,
    tools: list[Any] | None,
) -> list[CommandToolSnapshot]:
    if tools is not None:
        return [tool for tool in tools if isinstance(tool, CommandToolSnapshot)]
    graph = build_capability_graph_snapshot(knowledge_base)
    return list(build_command_tool_snapshots(graph))


def _record_from_tool(tool: CommandToolSnapshot) -> CapabilityRecord:
    description = normalize_message_text(tool.description or tool.capability_text)
    return CapabilityRecord(
        command_id=normalize_message_text(tool.command_id),
        plugin_module=normalize_message_text(tool.plugin_module),
        plugin_name=normalize_message_text(tool.plugin_name),
        family=normalize_message_text(tool.family or "general"),
        tool=tool,
        description=description,
        risk_tags=tuple(_risk_tags(tool)),
        permission_tags=tuple(_permission_tags(tool)),
        search_text=_search_text(tool),
    )


def _mark_registry_recall(candidate: CommandCandidate) -> CommandCandidate:
    reasons = tuple(
        f"recall:{reason}" if not reason.startswith("recall:") else reason
        for reason in candidate.reasons
    )
    reason = ",".join(reasons) or "recall"
    return CommandCandidate(
        plugin_module=candidate.plugin_module,
        plugin_name=candidate.plugin_name,
        schema=candidate.schema,
        score=candidate.score,
        reason=reason,
        family=candidate.family,
        tool=candidate.tool,
        reasons=reasons,
        exact_protected=candidate.exact_protected,
        features=candidate.features,
    )


def _search_text(tool: CommandToolSnapshot) -> str:
    parts = [
        tool.command_id,
        tool.plugin_module,
        tool.plugin_name,
        tool.head,
        " ".join(tool.aliases),
        tool.description,
        tool.usage or "",
        " ".join(tool.examples),
        " ".join(tool.retrieval_phrases),
        tool.capability_text,
        " ".join(tool.task_verbs),
        " ".join(tool.input_requirements),
    ]
    return normalize_message_text(" ".join(part for part in parts if part))


def _risk_tags(tool: CommandToolSnapshot) -> list[str]:
    text = _search_text(tool).casefold()
    tags: list[str] = []
    if any(keyword.casefold() in text for keyword in _DESTRUCTIVE_KEYWORDS):
        tags.append("destructive")
    if any(keyword.casefold() in text for keyword in _COST_KEYWORDS):
        tags.append("cost_or_currency")
    if tool.payload_policy in {"image_only", "text_or_image"} or tool.requires.get(
        "image"
    ):
        tags.append("media")
    if tool.target_requirement != "none" or tool.allow_at:
        tags.append("targeted")
    return tags


def _permission_tags(tool: CommandToolSnapshot) -> list[str]:
    text = _search_text(tool).casefold()
    tags = ["public"]
    if any(keyword.casefold() in text for keyword in _ADMIN_KEYWORDS):
        tags.append("sensitive_permission")
    if tool.requires.get("private"):
        tags.append("private_only")
    if tool.requires.get("to_me"):
        tags.append("addressed_to_bot")
    return tags


__all__ = [
    "CapabilityRecord",
    "CapabilityRegistry",
]
