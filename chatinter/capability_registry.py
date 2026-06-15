"""Capability registry for ChatInter command discovery.

The registry owns the complete, safe command capability set.  Local code only
recalls a small candidate set from it; the model still chooses the executable
command tool from injected full schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, cast

from .capability_card import (
    CapabilityCard,
    CapabilityEntityScope,
    CapabilityExecutionPolicy,
    CapabilityOutputMode,
    CapabilityRiskLevel,
    CapabilitySideEffect,
    CapabilitySourceOfTruth,
    build_capability_card,
)
from .capability_graph import build_capability_graph_snapshot
from .command_index import (
    CommandCandidate,
    dump_candidate_for_prompt,
    retrieve_command_candidates,
)
from .feedback import get_command_feedback_profile
from .llm_compat import ToolDefinition, ToolExecutable, ToolResult
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .native_command_tools import build_native_command_tools
from .plugin_reference import build_command_tool_snapshots
from .route_text import normalize_message_text


@dataclass(frozen=True)
class CapabilityRecord:
    """One discoverable plugin command capability."""

    command_id: str
    plugin_module: str
    plugin_name: str
    family: str
    tool: CommandToolSnapshot
    card: CapabilityCard
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
            "capability_card": self.card.to_prompt_payload(),
            "head": self.tool.head,
            "aliases": list(self.tool.aliases),
            "payload_policy": self.tool.payload_policy,
            "intent_types": list(self.tool.intent_types),
            "output_mode": self.tool.output_mode,
            "side_effect": self.tool.side_effect,
            "risk_level": self.tool.risk_level,
            "risk": self.tool.risk,
            "source_of_truth": self.tool.source_of_truth,
            "requires_real_tool": self.tool.requires_real_tool,
            "entity_scope": self.tool.entity_scope,
            "reliability": self.tool.reliability,
            "schema_quality": self.tool.schema_quality,
            "soft_tool": self.tool.soft_tool,
            "requires_real_result": self.tool.requires_real_result,
            "generative": self.tool.generative,
            "execution_policy": self.tool.execution_policy,
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


@dataclass(frozen=True)
class ToolCapabilityRecord:
    """One executable runtime tool attached to a capability card.

    This is the unified turn-local source of truth for tools exposed to the
    LLM.  Plugin commands, catalog tools and superuser tools all pass through
    the same metadata surface before becoming executable tools.
    """

    capability_id: str
    tool_name: str
    kind: str
    executable: ToolExecutable
    card: CapabilityCard
    command_record: CapabilityRecord | None = None
    raw_card: Any | None = None
    available: bool = True
    availability_reason: str = ""
    feedback_profile: dict[str, Any] | None = None

    @property
    def risk(self) -> str:
        return self.card.risk or self.card.risk_level

    @property
    def reliability(self) -> float:
        return float(self.card.reliability or 0.0)

    @property
    def schema_quality(self) -> float:
        return float(self.card.schema_quality or 0.0)

    def to_prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "capability_id": self.capability_id,
            "tool_name": self.tool_name,
            "kind": self.kind,
            "available": self.available,
            "availability_reason": self.availability_reason,
            "source_of_truth": self.card.source_of_truth,
            "requires_real_tool": self.card.requires_real_tool,
            "output_mode": self.card.output_mode,
            "entity_scope": self.card.entity_scope,
            "risk": self.risk,
            "reliability": round(self.reliability, 3),
            "schema_quality": round(self.schema_quality, 3),
            "soft_tool": self.card.soft_tool,
            "execution_policy": self.card.execution_policy,
            "intent_types": list(self.card.intent_types),
            "permission_tags": list(self.card.permission_tags),
        }
        if self.command_record is not None:
            payload["plugin_module"] = self.command_record.plugin_module
            payload["plugin_name"] = self.command_record.plugin_name
            payload["command_id"] = self.command_record.command_id
        if self.feedback_profile:
            payload["feedback_profile"] = dict(self.feedback_profile)
        return {
            key: value
            for key, value in payload.items()
            if value not in ("", [], {}, None)
        }


@dataclass(frozen=True)
class _ImmutableCommandToolIndex:
    fingerprint: str
    tools: tuple[CommandToolSnapshot, ...]
    records: dict[str, CapabilityRecord]
    tools_by_plugin: dict[str, tuple[CommandToolSnapshot, ...]]


_COMMAND_TOOL_INDEX_CACHE: dict[str, _ImmutableCommandToolIndex] = {}
_COMMAND_TOOL_INDEX_CACHE_ORDER: list[str] = []
_COMMAND_TOOL_INDEX_CACHE_MAX = 16


class CapabilityRegistry:
    """Registry of all safe capabilities and executable tools for one turn."""

    def __init__(
        self,
        *,
        knowledge_base: PluginKnowledgeBase,
        tools: list[CommandToolSnapshot],
        session_id: str | None,
    ) -> None:
        index = _get_command_tool_index(tools)
        self.knowledge_base = knowledge_base
        self.tools = list(index.tools)
        self.session_id = session_id
        self.records = dict(index.records)
        self._tools_by_plugin = {
            key: list(items) for key, items in index.tools_by_plugin.items()
        }
        self.tool_records: dict[str, ToolCapabilityRecord] = {}
        self.generation = 0

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

    def command_tools_by_plugin(self) -> dict[str, list[CommandToolSnapshot]]:
        """Return command snapshots grouped by their plugin identity."""

        return {key: list(items) for key, items in self._tools_by_plugin.items()}

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
        payload["capability"] = record.to_prompt_payload()
        profile = get_command_feedback_profile(
            command_id=record.command_id,
            session_id=self.session_id,
            plugin_module=record.plugin_module,
        )
        reliability = _effective_reliability(
            base=record.tool.reliability,
            feedback_score=profile.reliability_score,
            false_trigger_score=profile.false_trigger_score,
            param_failure_score=profile.param_failure_score,
            latency_score=profile.latency_score,
        )
        payload["feedback_profile"] = {
            key: value
            for key, value in {
                "success_rate": round(profile.success_rate, 3),
                "false_trigger_rate": round(profile.false_trigger_rate, 3),
                "param_failure_rate": round(profile.param_failure_rate, 3),
                "avg_latency_ms": round(profile.avg_latency_ms, 1),
                "low_reliability": profile.low_reliability,
                "high_reliability": profile.high_reliability,
                "exposure_score": round(profile.exposure_score, 3),
            }.items()
            if value not in (0, 0.0, False)
        }
        payload["capability_policy"] = {
            key: value
            for key, value in {
                "source_of_truth": record.tool.source_of_truth,
                "requires_real_tool": record.tool.requires_real_tool,
                "output_mode": record.tool.output_mode,
                "entity_scope": record.tool.entity_scope,
                "soft_tool": record.tool.soft_tool,
                "risk": record.tool.risk or record.tool.risk_level,
                "schema_quality": round(record.tool.schema_quality, 3),
                "reliability": round(reliability, 3),
                "catalog_only_hint": _catalog_only_hint(
                    soft_tool=record.tool.soft_tool,
                    schema_quality=record.tool.schema_quality,
                    reliability=reliability,
                    false_trigger_rate=profile.false_trigger_rate,
                    param_failure_rate=profile.param_failure_rate,
                ),
            }.items()
            if value not in ("", None, False)
        }
        return payload

    def register_catalog_tool(
        self,
        *,
        tool_name: str,
        executable: ToolExecutable,
    ) -> None:
        normalized_name = normalize_message_text(tool_name)
        if not normalized_name:
            return
        card = CapabilityCard(
            command_id=normalized_name,
            plugin_module="chatinter.catalog",
            plugin_name="ChatInter Capability Catalog",
            family="catalog",
            description="检索当前可用的真寻插件命令能力并注入可执行 schema。",
            use_cases=("首轮候选没有合适命令时补查长尾插件能力",),
            output_mode="text",
            side_effect="query",
            risk_level="low",
            risk="low",
            source_of_truth="local_state",
            requires_real_tool=True,
            entity_scope="global",
            reliability=0.78,
            schema_quality=0.86,
            soft_tool=False,
            intent_types=("catalog", "query"),
            requires_real_result=True,
            execution_policy="normal",
            permission_tags=("public", "runtime_catalog"),
            capability_kind="catalog",
            tool_name=normalized_name,
        )
        self.register_executable_tool(
            tool_name=normalized_name,
            executable=executable,
            kind="catalog",
            card=card,
        )

    def register_session_search_tool(self) -> None:
        """Expose current-session history search as a read-only runtime tool."""

        from .session_search import SessionSearchTool

        tool = SessionSearchTool()
        normalized_name = normalize_message_text(tool.name)
        card = CapabilityCard(
            command_id=normalized_name,
            plugin_module="chatinter.session_search",
            plugin_name="ChatInter Session Search",
            family="memory",
            description="检索当前 ChatInter 会话历史，支持关键词、锚点滑窗和时序浏览。",
            use_cases=(
                "用户询问上次聊到哪里、之前提到的报错、历史上下文时使用",
                "需要围绕某条历史消息继续查看上下文时使用",
            ),
            output_mode="text",
            side_effect="query",
            risk_level="low",
            risk="low",
            source_of_truth="local_state",
            requires_real_tool=True,
            entity_scope="global",
            reliability=0.82,
            schema_quality=0.86,
            soft_tool=False,
            intent_types=("memory", "history", "query"),
            requires_real_result=True,
            execution_policy="normal",
            permission_tags=("public", "current_session"),
            capability_kind="runtime_tool",
            tool_name=normalized_name,
        )
        self.register_executable_tool(
            tool_name=normalized_name,
            executable=cast(ToolExecutable, tool),
            kind="runtime_tool",
            card=card,
        )

    def register_superuser_tools(
        self,
        tools: dict[str, ToolExecutable],
        *,
        cards: tuple[Any, ...] | list[Any] = (),
    ) -> None:
        cards_by_name = {
            normalize_message_text(str(getattr(card, "name", "") or "")): card
            for card in cards
        }
        for name, executable in tools.items():
            tool_name = normalize_message_text(str(name or ""))
            if not tool_name:
                continue
            raw_card = cards_by_name.get(tool_name)
            card = _capability_card_from_superuser_tool(
                tool_name=tool_name,
                executable=executable,
                raw_card=raw_card,
            )
            self.register_executable_tool(
                tool_name=tool_name,
                executable=executable,
                kind="superuser_tool",
                card=card,
                raw_card=raw_card,
            )

    def register_available_superuser_tools(self) -> None:
        """Load and register currently available superuser tools.

        Callers should not import the superuser registry directly for runtime
        exposure; this keeps availability, risk and metadata under this unified
        registry.
        """

        from .superuser_agent.registry import build_superuser_agent_tool_bundle

        bundle = build_superuser_agent_tool_bundle()
        self.register_superuser_tools(bundle.tools, cards=bundle.cards)

    def register_external_tools(
        self,
        tools: dict[str, ToolExecutable],
        *,
        provider_adapter: Any,
        namespace: str = "external",
        source: str = "mcp",
    ) -> None:
        """Register MCP/external tools through the unified capability layer."""

        normalized_namespace = normalize_message_text(namespace) or "external"
        normalized_source = normalize_message_text(source) or "mcp"
        for name, executable in tools.items():
            tool_name = provider_adapter.sanitize_external_tool_name(
                str(name or ""),
                namespace=normalized_namespace,
            )
            if not tool_name:
                continue
            wrapped = ProviderExternalTool(
                name=tool_name,
                executable=executable,
                adapter=provider_adapter,
            )
            card = _capability_card_from_external_tool(
                tool_name=tool_name,
                executable=executable,
                namespace=normalized_namespace,
                source=normalized_source,
                provider_adapter=provider_adapter,
            )
            self.register_executable_tool(
                tool_name=tool_name,
                executable=cast(ToolExecutable, wrapped),
                kind="runtime_tool",
                card=card,
                raw_card={
                    "namespace": normalized_namespace,
                    "source": normalized_source,
                    "provider_protocol": provider_adapter.mcp_metadata(),
                },
            )

    async def register_available_mcp_tools(
        self,
        *,
        provider_adapter: Any,
        force: bool = False,
    ) -> dict[str, Any]:
        """Discover configured MCP servers and register available tools.

        MCP is superuser-agent scoped by callers, but the discovered tools
        still flow through this unified capability registry so availability,
        provider schema normalization and risk metadata are not split into a
        side channel.
        """

        from .mcp_runtime import get_mcp_runtime_manager

        result = await get_mcp_runtime_manager().discover_tools(force=force)
        for server_name, tools in result.tools_by_server.items():
            self.register_external_tools(
                tools,
                provider_adapter=provider_adapter,
                namespace=f"mcp_{server_name}",
                source="mcp",
            )
        return result.to_payload()

    def sync_command_tools(
        self,
        candidates: list[CommandCandidate],
        tool_map: dict[str, ToolExecutable],
    ) -> None:
        """Replace runtime plugin-command tool records with current injections."""

        self._remove_tool_kind("plugin_command")
        command_by_id = {
            normalize_message_text(candidate.schema.command_id): candidate
            for candidate in candidates
            if normalize_message_text(candidate.schema.command_id)
        }
        for tool_name, executable in tool_map.items():
            binding = getattr(executable, "binding", None)
            candidate = getattr(binding, "candidate", None)
            command_id = normalize_message_text(
                str(getattr(binding, "command_id", "") or "")
            )
            if candidate is None and command_id:
                candidate = command_by_id.get(command_id)
            if candidate is None:
                continue
            command_id = normalize_message_text(candidate.schema.command_id)
            command_record = self.record_for(command_id)
            if command_record is None:
                continue
            card = replace(
                command_record.card,
                capability_kind="plugin_command",
                tool_name=normalize_message_text(str(tool_name or "")),
                available=True,
                availability_reason="",
            )
            feedback = _feedback_payload(
                command_record=command_record,
                session_id=self.session_id,
            )
            self.register_executable_tool(
                tool_name=tool_name,
                executable=executable,
                kind="plugin_command",
                card=card,
                command_record=command_record,
                feedback_profile=feedback,
            )

    def inject_plugin_command_candidates(
        self,
        candidates: list[CommandCandidate],
        *,
        max_command_tools: int,
    ) -> list[ToolExecutable]:
        """Replace executable plugin command tools from selected candidates.

        This is the only path that turns command recall results into runtime
        tools.  Catalog and first-turn exposure both call this method so the
        runtime never sees command tools that are not represented in the
        registry.
        """

        selected = self._trim_command_candidates(
            candidates,
            max_command_tools=max_command_tools,
        )
        tools = build_native_command_tools(selected)
        tool_map = {
            normalize_message_text(tool.binding.tool_name): cast(ToolExecutable, tool)
            for tool in tools
            if normalize_message_text(tool.binding.tool_name)
        }
        self.sync_command_tools(selected, tool_map)
        return list(tool_map.values())

    def clear_plugin_command_tools(self) -> None:
        """Remove executable plugin commands while keeping discoverable records."""

        self._remove_tool_kind("plugin_command")

    def register_executable_tool(
        self,
        *,
        tool_name: str,
        executable: ToolExecutable,
        kind: str,
        card: CapabilityCard,
        command_record: CapabilityRecord | None = None,
        raw_card: Any | None = None,
        feedback_profile: dict[str, Any] | None = None,
        available: bool = True,
        availability_reason: str = "",
    ) -> None:
        normalized_name = normalize_message_text(str(tool_name or ""))
        if not normalized_name:
            return
        capability_id = _capability_id(kind=kind, tool_name=normalized_name, card=card)
        self.tool_records[normalized_name] = ToolCapabilityRecord(
            capability_id=capability_id,
            tool_name=normalized_name,
            kind=kind,
            executable=executable,
            card=replace(
                card,
                capability_kind=_capability_kind(kind),
                tool_name=normalized_name,
                available=bool(available),
                availability_reason=normalize_message_text(availability_reason),
            ),
            command_record=command_record,
            raw_card=raw_card,
            available=bool(available),
            availability_reason=normalize_message_text(availability_reason),
            feedback_profile=feedback_profile,
        )
        self.generation += 1

    def executable_tool_map(
        self,
        *,
        include_unavailable: bool = False,
    ) -> dict[str, ToolExecutable]:
        return {
            record.tool_name: record.executable
            for record in self._sorted_tool_records(
                include_unavailable=include_unavailable
            )
        }

    def executable_tool_count(
        self,
        *,
        kind: str | None = None,
        include_unavailable: bool = False,
    ) -> int:
        normalized_kind = normalize_message_text(kind or "")
        return sum(
            1
            for record in self.tool_records.values()
            if (not normalized_kind or record.kind == normalized_kind)
            and (include_unavailable or record.available)
        )

    def executable_tool_map_by_kind(
        self,
        kind: str,
        *,
        include_unavailable: bool = False,
    ) -> dict[str, ToolExecutable]:
        normalized_kind = normalize_message_text(kind)
        return {
            record.tool_name: record.executable
            for record in self._sorted_tool_records(
                include_unavailable=include_unavailable
            )
            if record.kind == normalized_kind
        }

    def tool_record_for(self, tool_name: str) -> ToolCapabilityRecord | None:
        return self.tool_records.get(normalize_message_text(tool_name))

    def remove_executable_tool(
        self,
        tool_name: str,
        *,
        kind: str | None = None,
    ) -> bool:
        normalized_name = normalize_message_text(tool_name)
        if not normalized_name:
            return False
        record = self.tool_records.get(normalized_name)
        if record is None:
            return False
        normalized_kind = normalize_message_text(kind or "")
        if normalized_kind and record.kind != normalized_kind:
            return False
        self.tool_records.pop(normalized_name, None)
        self.generation += 1
        return True

    def tool_payloads(
        self,
        *,
        kind: str | None = None,
        include_unavailable: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        records = [
            record
            for record in self.tool_records.values()
            if (kind is None or record.kind == kind)
            and (include_unavailable or record.available)
        ]
        records.sort(key=_tool_record_sort_key)
        return tuple(record.to_prompt_payload() for record in records)

    def _sorted_tool_records(
        self,
        *,
        include_unavailable: bool,
    ) -> list[ToolCapabilityRecord]:
        records = [
            record
            for record in self.tool_records.values()
            if include_unavailable or record.available
        ]
        records.sort(key=_tool_record_sort_key)
        return records

    def _remove_tool_kind(self, kind: str) -> None:
        removed = False
        for name, record in list(self.tool_records.items()):
            if record.kind == kind:
                self.tool_records.pop(name, None)
                removed = True
        if removed:
            self.generation += 1

    @staticmethod
    def _trim_command_candidates(
        candidates: list[CommandCandidate],
        *,
        max_command_tools: int,
    ) -> list[CommandCandidate]:
        cap = max(1, int(max_command_tools or 1))
        by_id: dict[str, CommandCandidate] = {}
        for candidate in candidates:
            command_id = normalize_message_text(candidate.schema.command_id)
            if not command_id:
                continue
            previous = by_id.get(command_id)
            if previous is None or _command_candidate_rank_key(
                candidate
            ) > _command_candidate_rank_key(previous):
                by_id[command_id] = candidate
        return sorted(
            by_id.values(),
            key=_command_candidate_rank_key,
            reverse=True,
        )[:cap]


def _ensure_command_tools(
    knowledge_base: PluginKnowledgeBase,
    tools: list[Any] | None,
) -> list[CommandToolSnapshot]:
    if tools is not None:
        return [tool for tool in tools if isinstance(tool, CommandToolSnapshot)]
    graph = build_capability_graph_snapshot(knowledge_base)
    return list(build_command_tool_snapshots(graph))


def _get_command_tool_index(
    tools: list[CommandToolSnapshot],
) -> _ImmutableCommandToolIndex:
    fingerprint = _command_tool_index_fingerprint(tools)
    cached = _COMMAND_TOOL_INDEX_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    records = {
        normalize_message_text(tool.command_id): _record_from_tool(tool)
        for tool in tools
        if normalize_message_text(tool.command_id)
    }
    grouped: dict[str, list[CommandToolSnapshot]] = {}
    for tool in tools:
        key = normalize_message_text(tool.plugin_module or tool.plugin_name)
        if not key:
            key = "unknown"
        grouped.setdefault(key, []).append(tool)

    index = _ImmutableCommandToolIndex(
        fingerprint=fingerprint,
        tools=tuple(tools),
        records=records,
        tools_by_plugin={key: tuple(items) for key, items in grouped.items()},
    )
    _COMMAND_TOOL_INDEX_CACHE[fingerprint] = index
    _COMMAND_TOOL_INDEX_CACHE_ORDER.append(fingerprint)
    while len(_COMMAND_TOOL_INDEX_CACHE_ORDER) > _COMMAND_TOOL_INDEX_CACHE_MAX:
        old = _COMMAND_TOOL_INDEX_CACHE_ORDER.pop(0)
        _COMMAND_TOOL_INDEX_CACHE.pop(old, None)
    return index


def _command_tool_index_fingerprint(tools: list[CommandToolSnapshot]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for tool in tools:
        digest.update(tool.model_dump_json().encode("utf-8", "ignore"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _record_from_tool(tool: CommandToolSnapshot) -> CapabilityRecord:
    description = normalize_message_text(tool.description or tool.capability_text)
    card = build_capability_card(
        tool=tool,
        family=tool.family or "general",
        description=description,
    )
    return CapabilityRecord(
        command_id=normalize_message_text(tool.command_id),
        plugin_module=normalize_message_text(tool.plugin_module),
        plugin_name=normalize_message_text(tool.plugin_name),
        family=normalize_message_text(tool.family or "general"),
        tool=tool,
        card=card,
        description=description,
        risk_tags=card.risk_tags,
        permission_tags=card.permission_tags,
        search_text=card.search_text,
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


def _effective_reliability(
    *,
    base: float,
    feedback_score: float,
    false_trigger_score: float,
    param_failure_score: float,
    latency_score: float,
) -> float:
    score = float(base or 0.5)
    score += max(min(float(feedback_score or 0.0), 18.0), -24.0) / 120.0
    score += max(min(float(false_trigger_score or 0.0), 0.0), -24.0) / 100.0
    score += max(min(float(param_failure_score or 0.0), 0.0), -18.0) / 120.0
    score += max(min(float(latency_score or 0.0), 6.0), -8.0) / 180.0
    return max(0.0, min(score, 1.0))


def _feedback_payload(
    *,
    command_record: CapabilityRecord,
    session_id: str | None,
) -> dict[str, Any]:
    profile = get_command_feedback_profile(
        command_id=command_record.command_id,
        session_id=session_id,
        plugin_module=command_record.plugin_module,
    )
    return {
        key: value
        for key, value in {
            "success_rate": round(profile.success_rate, 3),
            "false_trigger_rate": round(profile.false_trigger_rate, 3),
            "param_failure_rate": round(profile.param_failure_rate, 3),
            "avg_latency_ms": round(profile.avg_latency_ms, 1),
            "low_reliability": profile.low_reliability,
            "high_reliability": profile.high_reliability,
            "exposure_score": round(profile.exposure_score, 3),
        }.items()
        if value not in (0, 0.0, False)
    }


def _capability_card_from_superuser_tool(
    *,
    tool_name: str,
    executable: ToolExecutable,
    raw_card: Any | None,
) -> CapabilityCard:
    category = normalize_message_text(str(getattr(raw_card, "category", "") or "agent"))
    if not category:
        category = "agent"
    risk = _normalize_risk(getattr(raw_card, "risk", "medium"))
    read_only = bool(getattr(raw_card, "read_only", False))
    description = normalize_message_text(
        str(
            getattr(raw_card, "description", "")
            or getattr(executable, "description", "")
            or ""
        )
    )
    approval_mode = normalize_message_text(
        str(getattr(raw_card, "approval_mode", "") or "policy")
    )
    output_mode = _normalize_output_mode(
        getattr(raw_card, "output_mode", "plugin_output")
    )
    entity_scope = _normalize_entity_scope(getattr(raw_card, "entity_scope", "global"))
    source_of_truth = _normalize_source_of_truth(
        getattr(raw_card, "source_of_truth", "local_state")
    )
    side_effect = "query" if read_only else "mutate"
    execution_policy = "normal"
    if approval_mode == "deny":
        execution_policy = "confirmation_required"
    elif approval_mode == "ask" or risk == "high" or not read_only:
        execution_policy = (
            "confirmation_required" if risk == "high" else "strong_intent"
        )
    tags = tuple(
        normalize_message_text(str(tag or ""))
        for tag in tuple(getattr(raw_card, "tags", ()) or ())
        if normalize_message_text(str(tag or ""))
    )
    permission_tags = tuple(
        dict.fromkeys(
            (
                "superuser",
                f"approval:{approval_mode or 'policy'}",
                f"category:{category}",
            )
        )
    )
    use_cases = tuple(
        item
        for item in (
            description or f"超级用户工具 {tool_name}",
            f"类别: {category}",
            f"风险: {risk}",
        )
        if item
    )
    return CapabilityCard(
        command_id=tool_name,
        plugin_module="chatinter.superuser_agent",
        plugin_name=f"superuser:{category}",
        family=f"superuser:{category}",
        description=description,
        use_cases=use_cases,
        anti_use_cases=("普通群聊和普通私聊不可调用", "非超级用户私聊不可调用"),
        task_verbs=(category,),
        input_requirements=("superuser_private_chat",),
        output_mode=cast(CapabilityOutputMode, output_mode),
        side_effect=cast(CapabilitySideEffect, side_effect),
        risk_level=cast(CapabilityRiskLevel, risk),
        risk=cast(CapabilityRiskLevel, risk),
        source_of_truth=cast(CapabilitySourceOfTruth, source_of_truth),
        requires_real_tool=bool(getattr(raw_card, "requires_real_tool", True)),
        entity_scope=cast(CapabilityEntityScope, entity_scope),
        reliability=_clamp01(getattr(raw_card, "reliability", 0.7)),
        schema_quality=_clamp01(getattr(raw_card, "schema_quality", 0.65)),
        soft_tool=bool(getattr(raw_card, "soft_tool", False)),
        intent_types=(category, "agent"),
        requires_real_result=True,
        generative=False,
        execution_policy=cast(CapabilityExecutionPolicy, execution_policy),
        risk_tags=tuple(dict.fromkeys((risk, *tags))),
        permission_tags=permission_tags,
        search_text=normalize_message_text(
            " ".join([tool_name, category, description, " ".join(tags)])
        ),
        capability_kind="superuser_tool",
        tool_name=tool_name,
    )


class ProviderExternalTool:
    """Provider-normalized wrapper for MCP/external ToolExecutable objects."""

    def __init__(
        self,
        *,
        name: str,
        executable: ToolExecutable,
        adapter: Any,
    ) -> None:
        self.name = normalize_message_text(name)
        self.executable = executable
        self.adapter = adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self.executable, name)

    async def get_definition(self) -> ToolDefinition:
        definition = await self.executable.get_definition()
        sanitized = self.adapter.sanitize_tool_definition(definition)
        return ToolDefinition(
            name=self.name or sanitized.name,
            description=sanitized.description,
            parameters=sanitized.parameters,
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        result = await self.executable.execute(context=context, **kwargs)
        return ToolResult(
            output=self.adapter.tool_result_payload(result.output),
            display_content=result.display_content,
        )


def _capability_card_from_external_tool(
    *,
    tool_name: str,
    executable: ToolExecutable,
    namespace: str,
    source: str,
    provider_adapter: Any,
) -> CapabilityCard:
    description = normalize_message_text(
        str(getattr(executable, "description", "") or "")
    )
    source_of_truth = (
        "external_service" if source in {"mcp", "external"} else "local_state"
    )
    return CapabilityCard(
        command_id=tool_name,
        plugin_module=f"chatinter.{source}",
        plugin_name=f"{source}:{namespace}",
        family=f"{source}:{namespace}",
        description=description or f"External runtime tool {tool_name}",
        use_cases=(description or f"调用外部工具 {tool_name}",),
        anti_use_cases=("未被当前场景暴露时不可调用",),
        task_verbs=(namespace, source),
        input_requirements=("tool_schema",),
        output_mode="plugin_output",
        side_effect="query",
        risk_level="medium",
        risk="medium",
        source_of_truth=cast(CapabilitySourceOfTruth, source_of_truth),
        requires_real_tool=True,
        entity_scope="external",
        reliability=0.55,
        schema_quality=0.6,
        soft_tool=False,
        intent_types=("external_tool", source),
        requires_real_result=True,
        generative=False,
        execution_policy="strong_intent",
        risk_tags=("external", source),
        permission_tags=(
            "runtime",
            source,
            f"provider:{provider_adapter.profile.family}",
        ),
        search_text=normalize_message_text(
            " ".join([tool_name, namespace, source, description])
        ),
        capability_kind="runtime_tool",
        tool_name=tool_name,
        metadata={"provider_protocol": provider_adapter.mcp_metadata()},
    )


def _capability_id(*, kind: str, tool_name: str, card: CapabilityCard) -> str:
    if kind == "plugin_command" and card.command_id:
        return f"plugin_command:{card.command_id}"
    return f"{kind}:{tool_name}"


def _capability_kind(kind: str) -> str:
    if kind in {"plugin_command", "catalog", "superuser_tool", "runtime_tool"}:
        return kind
    return "runtime_tool"


def _kind_rank(kind: str) -> int:
    return {
        "catalog": 0,
        "superuser_tool": 1,
        "plugin_command": 2,
        "runtime_tool": 3,
    }.get(kind, 9)


def _tool_record_sort_key(
    record: ToolCapabilityRecord,
) -> tuple[int, int, float, float, str]:
    score = 0.0
    if record.command_record is not None:
        score = _tool_score(record.executable)
    return (
        _kind_rank(record.kind),
        -int(score * 100),
        -record.reliability,
        -record.schema_quality,
        record.tool_name,
    )


def _tool_score(executable: ToolExecutable) -> float:
    binding = getattr(executable, "binding", None)
    candidate = getattr(binding, "candidate", None)
    try:
        return float(getattr(candidate, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _command_candidate_rank_key(
    candidate: CommandCandidate,
) -> tuple[int, float, str]:
    return (
        1 if candidate.exact_protected else 0,
        float(candidate.score or 0.0),
        normalize_message_text(candidate.schema.command_id),
    )


def _normalize_risk(value: Any) -> str:
    risk = normalize_message_text(str(value or "")).lower()
    if risk == "critical":
        return "high"
    if risk in {"low", "medium", "high"}:
        return risk
    return "medium"


def _normalize_output_mode(value: Any) -> str:
    mode = normalize_message_text(str(value or "")).lower()
    if mode in {"text", "image", "file", "plugin_output", "action"}:
        return mode
    return "plugin_output"


def _normalize_entity_scope(value: Any) -> str:
    scope = normalize_message_text(str(value or "")).lower()
    if scope in {
        "none",
        "self_bot",
        "actor_user",
        "target_user",
        "group",
        "global",
        "external",
    }:
        return scope
    return "global"


def _normalize_source_of_truth(value: Any) -> str:
    source = normalize_message_text(str(value or "")).lower()
    if source in {
        "model_knowledge",
        "plugin_runtime",
        "bot_state",
        "external_service",
        "local_state",
        "user_provided",
        "unknown",
    }:
        return source
    return "local_state"


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(number, 1.0))


def _catalog_only_hint(
    *,
    soft_tool: bool,
    schema_quality: float,
    reliability: float,
    false_trigger_rate: float,
    param_failure_rate: float,
) -> str:
    reasons: list[str] = []
    if soft_tool:
        reasons.append("soft_tool")
    if schema_quality < 0.35:
        reasons.append("low_schema_quality")
    if reliability < 0.35:
        reasons.append("low_reliability")
    if false_trigger_rate >= 0.22:
        reasons.append("false_trigger_history")
    if param_failure_rate >= 0.45:
        reasons.append("param_failure_history")
    return ",".join(reasons)


__all__ = [
    "CapabilityCard",
    "CapabilityRecord",
    "CapabilityRegistry",
    "ProviderExternalTool",
    "ToolCapabilityRecord",
]
