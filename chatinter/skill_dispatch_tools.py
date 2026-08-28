"""Stable plugin-scoped dispatch tools for the ChatInter mixed-chat agent."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any
import unicodedata

from .candidate_exposure import CandidateExposureKey, CandidateExposureLedger
from .command_index import CommandCandidate, _schema_from_tool_snapshot
from .llm_compat import ToolDefinition, ToolExecutable, ToolResult
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .native_command_tools import NativeCommandToolBinding
from .native_executor import NativeCommandExecutionContext
from .plugin_skill_index import PluginSkill
from .route_text import normalize_message_text
from .sparse_retrieval import fuse_sparse_rankings, normalize_retrieval_queries
from .task_frame import (
    PAYLOAD_HINT_FIELD,
    TARGET_HINT_FIELD,
    TARGET_REF_FIELD,
    TARGET_REF_SCHEMA_DESCRIPTION,
    TARGET_REFS_FIELD,
    TARGET_REFS_SCHEMA_DESCRIPTION,
    TASK_TEXT_FIELD,
)
from .token_compat import estimate_text_tokens
from .tool_cards import project_command_card
from .tool_retriever import CommandToolRetriever

_TOOL_NAME_PREFIX = "ci_skill_"
_TOOL_NAME_SLUG_LIMIT = 40
_TOOL_NAME_PART_PATTERN = re.compile(r"[^a-z0-9_]+")
_ALIASES_TEXT_LIMIT = 480
_PRECAUTIONS_TEXT_LIMIT = 720
_CAPABILITY_VALUES_TEXT_LIMIT = 360
_SEMANTIC_CONTRACTS_TEXT_LIMIT = 1600
_RETRIEVAL_QUERIES_FIELD = "retrieval_queries"


@dataclass(frozen=True)
class _SelectionPlan:
    snapshots: list[CommandToolSnapshot]
    reason: str
    full_listing_fallback: bool
    recall: str = ""
    query_count: int = 0


class PluginSkillDispatchTool:
    chatinter_plugin_tool_kind = "skill_dispatch"
    chatinter_ignore_unknown_top_level_arguments = True

    def __init__(
        self,
        skill: PluginSkill,
        *,
        known_commands: Sequence[CommandToolSnapshot],
        available_commands: Sequence[CommandToolSnapshot],
        knowledge_base: PluginKnowledgeBase,
        session_id: str | None,
        command_context: NativeCommandExecutionContext,
        exposure_ledger: CandidateExposureLedger | None = None,
        revision: str = "",
        ambiguity_token_budget: int | None = None,
        result_char_budget: int | None = None,
        tool_name: str | None = None,
    ) -> None:
        self.skill = skill
        self.name = normalize_message_text(tool_name or "") or skill_dispatch_tool_name(
            skill.plugin_module
        )
        self._skill_command_keys = frozenset(
            _command_key(command_id) for command_id in skill.command_ids
        )
        # Module-scoped known lookup: keeps known-but-unavailable commands
        # reachable for the unavailable_in_context diagnostic branch even
        # though skill.command_ids only exposes available commands.
        self._known_by_key = _skill_snapshots_by_key(
            skill,
            known_commands,
            restrict_to_skill=False,
        )
        self._available_by_key = {
            key: snapshot
            for key, snapshot in _skill_snapshots_by_key(
                skill,
                available_commands,
            ).items()
            if key in self._known_by_key
        }
        self._command_context = command_context
        self._ambiguity_token_budget = (
            max(int(ambiguity_token_budget), 1)
            if ambiguity_token_budget is not None
            else None
        )
        self._result_char_budget = (
            max(int(result_char_budget), 1) if result_char_budget is not None else None
        )
        self._exposure_ledger = exposure_ledger or CandidateExposureLedger()
        self._exposure_key = CandidateExposureKey.build(
            source="local",
            skill=skill.skill_id,
            revision=revision,
        )
        self._retriever = CommandToolRetriever(
            _plugin_knowledge_base(skill, knowledge_base),
            session_id=session_id,
            tools=list(self._available_by_key.values()),
        )

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=_tool_description(self.skill),
            parameters={
                "type": "object",
                "properties": {
                    TASK_TEXT_FIELD: {
                        "type": "string",
                        "minLength": 1,
                        "description": "当前工具调用对应的用户原话或任务片段",
                    },
                    _RETRIEVAL_QUERIES_FIELD: {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 256},
                        "maxItems": 6,
                        "description": (
                            "可选的本插件内检索改写。填写保持用户原意的能力名、"
                            "操作名或领域表达；仅用于查找候选，不会作为命令执行"
                        ),
                    },
                    "command_id": {
                        "type": ["string", "null"],
                        "description": (
                            "仅在候选中已知具体命令时填写该 Skill 内的 command_id；"
                            "尚未选择时填写 null，由 Skill 内部检索"
                        ),
                    },
                    TARGET_REF_FIELD: {
                        "type": ["string", "null"],
                        "description": TARGET_REF_SCHEMA_DESCRIPTION,
                    },
                    TARGET_REFS_FIELD: {
                        "type": ["array", "null"],
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 2,
                        "maxItems": 4,
                        "uniqueItems": True,
                        "description": TARGET_REFS_SCHEMA_DESCRIPTION,
                    },
                    "slots": {
                        "type": "object",
                        "description": "命令参数槽位，键为槽位名",
                        "additionalProperties": True,
                    },
                },
                "required": [TASK_TEXT_FIELD],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        del context
        task_text = normalize_message_text(str(kwargs.get(TASK_TEXT_FIELD, "") or ""))
        if not task_text:
            return self._not_executed(
                "invalid",
                error="task_text 不能为空",
                reason="missing_task_text",
            )
        command_id = normalize_message_text(str(kwargs.get("command_id", "") or ""))
        # JSON null 的常见字符串形式等同于未提供 command_id，由 Skill 负责检索。
        if command_id.casefold() in {"null", "none", "nil", "undefined", ""}:
            command_id = ""
        if command_id:
            return await self._dispatch_command(
                command_id=command_id,
                task_text=task_text,
                kwargs=kwargs,
            )

        return self._selection_result(
            task_text=task_text,
            retrieval_queries=kwargs.get(_RETRIEVAL_QUERIES_FIELD),
            defer_candidate_exposure=bool(kwargs.get("_defer_candidate_exposure")),
        )

    def _selection_result(
        self,
        *,
        task_text: str,
        retrieval_queries: object = None,
        requested_command_id: str = "",
        reason: str = "",
        response_policy: str = "",
        defer_candidate_exposure: bool = False,
    ) -> ToolResult:
        prepared = self._prepare_selection(
            task_text=task_text,
            retrieval_queries=retrieval_queries,
            requested_command_id=requested_command_id,
            reason=reason,
        )
        if isinstance(prepared, ToolResult):
            return prepared
        return self._render_selection(
            prepared,
            requested_command_id=requested_command_id,
            response_policy=response_policy,
            defer_candidate_exposure=defer_candidate_exposure,
        )

    def _prepare_selection(
        self,
        *,
        task_text: str,
        retrieval_queries: object,
        requested_command_id: str,
        reason: str,
    ) -> _SelectionPlan | ToolResult:
        query_rewrites = list(
            retrieval_queries
            if isinstance(retrieval_queries, list | tuple)
            else ()
        )
        requested_key = _command_key(requested_command_id)
        if requested_key in self._available_by_key:
            query_rewrites.append(self._available_by_key[requested_key].command_id)
        queries = normalize_retrieval_queries(task_text, query_rewrites)
        rankings: list[list[str]] = []
        exact_ids: set[str] = set()
        for query in queries:
            retrieval = self._retriever.retrieve(
                query,
                limit=max(
                    int(
                        getattr(
                            self._retriever,
                            "total_commands",
                            len(self._available_by_key),
                        )
                    ),
                    1,
                ),
                context={
                    "exhaustive_sparse": True,
                    "pure_sparse": True,
                },
            )
            rankings.append(
                [candidate.schema.command_id for candidate in retrieval.candidates]
            )
            exact_ids.update(
                candidate.schema.command_id
                for candidate in retrieval.candidates
                if candidate.exact_protected
            )
        fused = fuse_sparse_rankings(queries, rankings, exact_ids=exact_ids)
        direct = [
            snapshot
            for command_id in fused.ranked_ids
            if (snapshot := self._available_by_key.get(_command_key(command_id)))
            is not None
        ]
        snapshots = direct
        full_listing_fallback = False
        if not snapshots:
            # 静态 BM25 零召回不能证明 Skill 没有适用命令；可用命令全集仍作为候选。
            fallback_snapshots = sorted(
                self._available_by_key.values(),
                key=_snapshot_stable_key,
            )
            if not fallback_snapshots:
                payload: dict[str, Any] = {
                    "error": "该插件内没有匹配当前任务的命令",
                    "query": _clip_text(task_text, 1_000),
                }
                if requested_command_id:
                    payload["requested_command_id"] = _clip_text(
                        requested_command_id,
                        256,
                    )
                if reason:
                    payload["reason"] = _clip_text(reason, 128)
                return self._not_executed(
                    "not_found",
                    **payload,
                )
            snapshots = fallback_snapshots
            reason = reason or "fallback_full_listing"
            full_listing_fallback = True
        return _SelectionPlan(
            snapshots=list(snapshots),
            reason=reason,
            full_listing_fallback=full_listing_fallback,
            recall=(
                "skill_full_listing"
                if full_listing_fallback
                else "skill_sparse_multi_query"
                if len(queries) > 1
                else "skill_sparse"
            ),
            query_count=len(queries),
        )

    def _render_selection(
        self,
        plan: _SelectionPlan,
        *,
        requested_command_id: str,
        response_policy: str,
        defer_candidate_exposure: bool,
    ) -> ToolResult:
        snapshots = plan.snapshots
        requested_command_id = _clip_text(requested_command_id, 256)
        reason = _clip_text(plan.reason, 128)
        payload: dict[str, Any] = {
            "candidates": [],
            "candidate_count": len(snapshots),
            "displayed_candidate_count": 0,
            "omitted_candidate_count": len(snapshots),
            "truncated": bool(snapshots),
        }
        if requested_command_id:
            payload["requested_command_id"] = requested_command_id
        if reason:
            payload["reason"] = reason
        if response_policy:
            payload["response_policy"] = response_policy
        if plan.full_listing_fallback:
            payload["recall"] = "skill_full_listing"
            payload["note"] = (
                "本次稀疏检索未命中，以下是该插件内完整可用命令列表。"
                "仅在存在明确匹配项时调用，否则按普通聊天回复。"
            )
        elif plan.recall:
            payload["recall"] = plan.recall
            payload["note"] = (
                "以下是真实元数据的本地稀疏检索候选。请确认 command_id 与用户目标"
                "一致后再调用；若都不匹配，按普通聊天回复。"
            )

        # Candidate projection already accounts for its compact selection envelope.
        # Reserve only the additional bytes/tokens used by the real Skill result.
        empty_output = {
            "status": "selection_required",
            "plugin_execution": False,
            "executed": False,
            "skill_id": self.skill.skill_id,
            "plugin_module": self.skill.plugin_module,
            **payload,
        }
        empty_serialized = json.dumps(
            empty_output,
            ensure_ascii=False,
            default=str,
        )
        compact_empty_tokens = _selection_payload_tokens(
            [],
            requested_command_id=requested_command_id,
            reason=reason,
            truncated=True,
        )
        compact_empty_chars = _selection_payload_chars(
            [],
            requested_command_id=requested_command_id,
            reason=reason,
            truncated=True,
        )
        projection_token_budget = (
            max(
                self._ambiguity_token_budget
                - max(
                    estimate_text_tokens(empty_serialized) - compact_empty_tokens,
                    0,
                ),
                1,
            )
            if self._ambiguity_token_budget is not None
            else None
        )
        projection_char_budget = (
            max(
                self._result_char_budget
                - max(len(empty_serialized) - compact_empty_chars, 0),
                1,
            )
            if self._result_char_budget is not None
            else None
        )
        cards = _project_ambiguous_cards(
            snapshots,
            token_budget=projection_token_budget,
            char_budget=projection_char_budget,
            requested_command_id=requested_command_id,
            reason=reason,
        )
        if plan.full_listing_fallback and len(cards) < len(snapshots):
            cards = []
        payload["candidates"] = cards
        payload["displayed_candidate_count"] = len(cards)
        payload["omitted_candidate_count"] = max(len(snapshots) - len(cards), 0)
        payload["truncated"] = len(cards) < len(snapshots)
        if plan.full_listing_fallback:
            if cards:
                payload["recall"] = "skill_full_listing"
            else:
                payload["recall"] = "skill_no_recall"
                payload["note"] = (
                    "本次稀疏检索未命中，且完整命令列表超出结果预算，因此没有返回"
                    "有偏的部分列表。可使用更准确的检索改写再次调用，或按普通聊天回复。"
                )
        while cards and not self._selection_output_fits(payload):
            cards.pop()
            payload["displayed_candidate_count"] = len(cards)
            payload["omitted_candidate_count"] = len(snapshots) - len(cards)
            payload["truncated"] = True
        if not defer_candidate_exposure:
            exposed_ids = [card.get("command_id") for card in cards]
            self.expose_candidates(
                exposed_ids,
                source=str(payload.get("recall") or plan.recall or "skill_no_recall"),
                pending=True,
            )
            self._exposure_ledger.record_discovery(
                self._exposure_key,
                source=str(payload.get("recall") or plan.recall or "skill_no_recall"),
                query_count=plan.query_count,
                candidate_count=len(snapshots),
                displayed_count=len(cards),
                omitted_count=max(len(snapshots) - len(cards), 0),
            )
        return self._not_executed("selection_required", **payload)

    def _selection_output_fits(self, payload: dict[str, Any]) -> bool:
        output = {
            "status": "selection_required",
            "plugin_execution": False,
            "executed": False,
            "skill_id": self.skill.skill_id,
            "plugin_module": self.skill.plugin_module,
            **payload,
        }
        serialized = json.dumps(output, ensure_ascii=False, default=str)
        return (
            self._ambiguity_token_budget is None
            or estimate_text_tokens(serialized) <= self._ambiguity_token_budget
        ) and (
            self._result_char_budget is None
            or len(serialized) <= self._result_char_budget
        )

    async def _dispatch_command(
        self,
        *,
        command_id: str,
        task_text: str,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        key = _command_key(command_id)
        if key not in self._skill_command_keys and key not in self._known_by_key:
            return self._selection_result(
                task_text=task_text,
                retrieval_queries=kwargs.get(_RETRIEVAL_QUERIES_FIELD),
                requested_command_id=command_id,
                reason="command_out_of_skill",
                defer_candidate_exposure=bool(
                    kwargs.get("_defer_candidate_exposure")
                ),
            )
        known = self._known_by_key.get(key)
        if known is None:
            return self._not_executed(
                "not_found",
                command_id=command_id,
                error="当前 Skill 缺少该命令的可用定义",
                reason="command_snapshot_missing",
            )
        snapshot = self._available_by_key.get(key)
        if snapshot is None:
            return self._not_executed(
                "unavailable_in_context",
                command_id=known.command_id,
                error="当前会话条件不满足，或该命令在当前场景不可用",
                command_schema=project_command_card(known),
            )
        if not self._exposure_ledger.is_exposed(
            self._exposure_key,
            snapshot.command_id,
        ):
            self._exposure_ledger.record_execution(
                self._exposure_key,
                snapshot.command_id,
                valid=False,
                reason="candidate_identity_not_exposed",
            )
            return self._selection_result(
                task_text=task_text,
                retrieval_queries=kwargs.get(_RETRIEVAL_QUERIES_FIELD),
                requested_command_id=command_id,
                reason="candidate_identity_not_exposed",
                defer_candidate_exposure=bool(
                    kwargs.get("_defer_candidate_exposure")
                ),
            )
        self._exposure_ledger.record_execution(
            self._exposure_key,
            snapshot.command_id,
            valid=True,
            reason="candidate_exposed",
        )
        return await self._execute_snapshot(snapshot, kwargs=kwargs)

    @property
    def exposure_key(self) -> CandidateExposureKey:
        return self._exposure_key

    def owns_candidate_identity(self, identity: object) -> bool:
        return _command_key(identity) in self._available_by_key

    def is_candidate_exposed(self, identity: object) -> bool:
        return self._exposure_ledger.is_exposed(self._exposure_key, identity)

    def expose_candidates(
        self,
        identities: Iterable[object],
        *,
        source: str,
        pending: bool,
        exact_identity: bool = False,
        strict_identity_mode: str = "",
    ) -> tuple[str, ...]:
        valid = (
            self._available_by_key[_command_key(identity)].command_id
            for identity in identities
            if _command_key(identity) in self._available_by_key
        )
        return self._exposure_ledger.expose(
            self._exposure_key,
            valid,
            discovery_source=source,
            exact_identity=exact_identity,
            pending=pending,
            strict_identity_mode=strict_identity_mode,
        )

    def record_discovery(
        self,
        *,
        source: str,
        query_count: int,
        candidate_count: int,
        displayed_count: int,
        omitted_count: int,
    ) -> None:
        self._exposure_ledger.record_discovery(
            self._exposure_key,
            source=source,
            query_count=query_count,
            candidate_count=candidate_count,
            displayed_count=displayed_count,
            omitted_count=omitted_count,
        )

    async def _execute_snapshot(
        self,
        snapshot: CommandToolSnapshot,
        *,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        candidate = _candidate_from_snapshot(snapshot, skill=self.skill)
        binding = NativeCommandToolBinding(
            tool_name=self.name,
            candidate=candidate,
        )
        if all(
            _command_key(item.schema.command_id) != _command_key(snapshot.command_id)
            for item in self._command_context.candidates
        ):
            self._command_context.candidates.append(candidate)
        execution_count = len(self._command_context.executions)
        result = await self._command_context.execute_tool(
            binding=binding,
            raw_slots=_build_raw_slots(kwargs),
        )
        execution = (
            self._command_context.executions[-1]
            if len(self._command_context.executions) > execution_count
            else None
        )
        execution_started = bool(
            execution is not None and (execution.execution_started or execution.success)
        )
        executed = bool(execution is not None and execution.success)
        output = (
            dict(result.output)
            if isinstance(result.output, dict)
            else {"value": result.output}
        )
        output.setdefault("status", "executed" if executed else "not_executed")
        output["plugin_execution"] = execution_started
        output["executed"] = executed
        output.setdefault("skill_id", self.skill.skill_id)
        output.setdefault("plugin_module", self.skill.plugin_module)
        output.setdefault("command_id", snapshot.command_id)
        return ToolResult(
            output=output,
            display_content=result.display_content,
            is_error=result.is_error,
            is_retryable=result.is_retryable,
        )

    def _not_executed(self, status: str, **payload: Any) -> ToolResult:
        output = {
            "status": status,
            "plugin_execution": False,
            "executed": False,
            "skill_id": self.skill.skill_id,
            "plugin_module": self.skill.plugin_module,
            **payload,
        }
        return ToolResult(
            output=output,
            display_content=f"{self.name}: {status}",
            is_retryable=status in {"ambiguous", "selection_required"},
        )


def _snapshot_stable_key(snapshot: CommandToolSnapshot) -> tuple[str, str]:
    command_id = normalize_message_text(snapshot.command_id)
    return command_id.casefold(), command_id


def skill_dispatch_tool_name(plugin_module: str) -> str:
    module = normalize_message_text(plugin_module).casefold()
    if not module:
        raise ValueError("plugin_module cannot be empty")
    digest = hashlib.blake2s(module.encode("utf-8"), digest_size=4).hexdigest()
    tail = module.rsplit(".", 1)[-1]
    ascii_tail = (
        unicodedata.normalize("NFKD", tail).encode("ascii", "ignore").decode("ascii")
    )
    slug = _TOOL_NAME_PART_PATTERN.sub("_", ascii_tail).strip("_") or "plugin"
    slug = slug[:_TOOL_NAME_SLUG_LIMIT].rstrip("_") or "plugin"
    return f"{_TOOL_NAME_PREFIX}{slug}_{digest}"


def build_plugin_skill_dispatch_tools(
    *,
    skills: Iterable[PluginSkill],
    known_commands: Sequence[CommandToolSnapshot],
    available_commands: Sequence[CommandToolSnapshot],
    knowledge_base: PluginKnowledgeBase,
    session_id: str | None,
    command_context: NativeCommandExecutionContext,
    exposure_ledger: CandidateExposureLedger | None = None,
    revision: str = "",
    ambiguity_token_budget: int | None = None,
    result_char_budget: int | None = None,
) -> dict[str, ToolExecutable]:
    result: dict[str, ToolExecutable] = {}
    ordered_skills = sorted(
        skills,
        key=lambda item: (
            normalize_message_text(item.plugin_module).casefold(),
            normalize_message_text(item.skill_id).casefold(),
        ),
    )
    for skill in ordered_skills:
        tool = PluginSkillDispatchTool(
            skill,
            known_commands=known_commands,
            available_commands=available_commands,
            knowledge_base=knowledge_base,
            session_id=session_id,
            command_context=command_context,
            exposure_ledger=exposure_ledger,
            revision=revision,
            ambiguity_token_budget=ambiguity_token_budget,
            result_char_budget=result_char_budget,
        )
        if tool.name in result:
            raise ValueError(f"duplicate plugin Skill tool name: {tool.name}")
        result[tool.name] = tool
    return result


def _skill_snapshots_by_key(
    skill: PluginSkill,
    snapshots: Sequence[CommandToolSnapshot],
    *,
    restrict_to_skill: bool = True,
) -> dict[str, CommandToolSnapshot]:
    module_key = normalize_message_text(skill.plugin_module).casefold()
    allowed = (
        {_command_key(command_id) for command_id in skill.command_ids}
        if restrict_to_skill
        else None
    )
    result: dict[str, CommandToolSnapshot] = {}
    for snapshot in sorted(
        snapshots,
        key=lambda item: (
            _command_key(item.command_id),
            normalize_message_text(item.command_id),
        ),
    ):
        key = _command_key(snapshot.command_id)
        if (
            not key
            or (allowed is not None and key not in allowed)
            or normalize_message_text(snapshot.plugin_module).casefold() != module_key
        ):
            continue
        result.setdefault(key, snapshot)
    return result


def _plugin_knowledge_base(
    skill: PluginSkill,
    knowledge_base: PluginKnowledgeBase,
) -> PluginKnowledgeBase:
    module_key = normalize_message_text(skill.plugin_module).casefold()
    return PluginKnowledgeBase(
        plugins=[
            plugin
            for plugin in knowledge_base.plugins
            if normalize_message_text(plugin.module).casefold() == module_key
        ],
        user_role=knowledge_base.user_role,
    )


def _candidate_from_snapshot(
    snapshot: CommandToolSnapshot,
    *,
    skill: PluginSkill,
) -> CommandCandidate:
    return CommandCandidate(
        plugin_module=snapshot.plugin_module,
        plugin_name=snapshot.plugin_name,
        schema=_schema_from_tool_snapshot(snapshot),
        score=0.0,
        reason=f"skill_dispatch:{skill.skill_id}",
        family=snapshot.family,
        tool=snapshot,
    )


def _build_raw_slots(kwargs: dict[str, Any]) -> dict[str, Any]:
    slots = kwargs.get("slots")
    raw_slots = dict(slots) if isinstance(slots, dict) else {}
    raw_slots[TASK_TEXT_FIELD] = str(kwargs.get(TASK_TEXT_FIELD, "") or "")
    for field in (TARGET_HINT_FIELD, TARGET_REF_FIELD, PAYLOAD_HINT_FIELD):
        value = str(kwargs.get(field, "") or "")
        if value:
            raw_slots[field] = value
    target_refs = kwargs.get(TARGET_REFS_FIELD)
    if isinstance(target_refs, list | tuple):
        raw_slots[TARGET_REFS_FIELD] = list(target_refs)
    return raw_slots


def _project_ambiguous_cards(
    snapshots: Sequence[CommandToolSnapshot],
    *,
    token_budget: int | None,
    char_budget: int | None,
    requested_command_id: str,
    reason: str,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for snapshot in snapshots:
        card = project_command_card(snapshot)
        remaining_tokens = (
            token_budget
            - _selection_payload_tokens(
                cards,
                requested_command_id=requested_command_id,
                reason=reason,
                truncated=True,
            )
            if token_budget is not None
            else None
        )
        remaining_chars = (
            char_budget
            - _selection_payload_chars(
                cards,
                requested_command_id=requested_command_id,
                reason=reason,
                truncated=True,
            )
            if char_budget is not None
            else None
        )
        fitted = (
            _fit_ambiguous_card(
                card,
                token_budget=remaining_tokens,
                char_budget=remaining_chars,
            )
            if token_budget is not None or char_budget is not None
            else card
        )
        if fitted is None:
            continue
        trial_cards = [*cards, fitted]
        trial_tokens = _selection_payload_tokens(
            trial_cards,
            requested_command_id=requested_command_id,
            reason=reason,
            truncated=len(trial_cards) < len(snapshots),
        )
        trial_chars = _selection_payload_chars(
            trial_cards,
            requested_command_id=requested_command_id,
            reason=reason,
            truncated=len(trial_cards) < len(snapshots),
        )
        if token_budget is not None and trial_tokens > token_budget:
            break
        if char_budget is not None and trial_chars > char_budget:
            break
        cards = trial_cards
    return cards


def _fit_ambiguous_card(
    card: dict[str, Any],
    *,
    token_budget: int | None,
    char_budget: int | None,
) -> dict[str, Any] | None:
    if token_budget is not None and token_budget <= 0:
        return None
    if char_budget is not None and char_budget <= 0:
        return None
    if _card_fits(card, token_budget=token_budget, char_budget=char_budget):
        return card

    command_id = str(card.get("command_id", "") or "")
    if not command_id:
        return None
    compact: dict[str, Any] = {"command_id": command_id}
    if not _card_fits(
        compact,
        token_budget=token_budget,
        char_budget=char_budget,
    ):
        return None

    priority = (
        "head",
        "usage",
        "plugin",
        "description",
        "render",
        "slots",
        "accepted_inputs",
        "required_context",
        "aliases",
        "examples",
        "use_cases",
        "anti_use_cases",
        "output_mode",
        "side_effect",
        "execution_policy",
        "source_of_truth",
        "requires_real_result",
    )
    for key in priority:
        if key not in card:
            continue
        _add_card_value(
            compact,
            key,
            card[key],
            token_budget=token_budget,
            char_budget=char_budget,
        )
    return compact


def _add_card_value(
    target: dict[str, Any],
    key: str,
    value: Any,
    *,
    token_budget: int | None,
    char_budget: int | None,
) -> None:
    if isinstance(value, list):
        accepted: list[Any] = []
        for item in value:
            candidate_item = _compact_card_list_item(item)
            trial = {**target, key: [*accepted, candidate_item]}
            if not _card_fits(
                trial,
                token_budget=token_budget,
                char_budget=char_budget,
            ):
                break
            accepted.append(candidate_item)
        if accepted:
            target[key] = accepted
        return

    trial = {**target, key: value}
    if _card_fits(
        trial,
        token_budget=token_budget,
        char_budget=char_budget,
    ):
        target[key] = value
        return
    if not isinstance(value, str):
        return

    low = 0
    high = len(value)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        clipped = value[:middle].rstrip()
        if middle < len(value) and clipped:
            clipped += "…"
        if clipped and _card_fits(
            {**target, key: clipped},
            token_budget=token_budget,
            char_budget=char_budget,
        ):
            best = clipped
            low = middle + 1
        else:
            high = middle - 1
    if best:
        target[key] = best


def _compact_card_list_item(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    keys = (
        "name",
        "type",
        "required",
        "description",
        "for",
        "any_of",
    )
    return {key: value[key] for key in keys if key in value}


def _card_fits(
    card: dict[str, Any],
    *,
    token_budget: int | None,
    char_budget: int | None,
) -> bool:
    serialized = _compact_json(card)
    return (
        token_budget is None or estimate_text_tokens(serialized) <= token_budget
    ) and (char_budget is None or len(serialized) <= char_budget)


def _selection_payload(
    cards: list[dict[str, Any]],
    *,
    requested_command_id: str,
    reason: str,
    truncated: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "selection_required",
        "plugin_execution": False,
        "executed": False,
        "candidates": cards,
        "truncated": truncated,
    }
    if requested_command_id:
        payload["requested_command_id"] = requested_command_id
    if reason:
        payload["reason"] = reason
    return payload


def _selection_payload_tokens(
    cards: list[dict[str, Any]],
    *,
    requested_command_id: str,
    reason: str,
    truncated: bool,
) -> int:
    return estimate_text_tokens(
        _compact_json(
            _selection_payload(
                cards,
                requested_command_id=requested_command_id,
                reason=reason,
                truncated=truncated,
            )
        )
    )


def _selection_payload_chars(
    cards: list[dict[str, Any]],
    *,
    requested_command_id: str,
    reason: str,
    truncated: bool,
) -> int:
    return len(
        _compact_json(
            _selection_payload(
                cards,
                requested_command_id=requested_command_id,
                reason=reason,
                truncated=truncated,
            )
        )
    )


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _tool_description(skill: PluginSkill) -> str:
    metadata = {
        "plugin": _clip_text(skill.plugin_name, 160),
        "description": _clip_text(skill.description, 360),
        "aliases": _bounded_values(
            skill.aliases,
            item_limit=120,
            total_limit=_ALIASES_TEXT_LIMIT,
        ),
        "usage": _clip_text(skill.usage, 600),
        "introduction": _clip_text(skill.introduction, 480),
        "precautions": _bounded_values(
            skill.precautions,
            item_limit=240,
            total_limit=_PRECAUTIONS_TEXT_LIMIT,
        ),
        "command_count": max(int(skill.command_count), 0),
        "input_types": _bounded_values(
            skill.input_types,
            item_limit=80,
            total_limit=_CAPABILITY_VALUES_TEXT_LIMIT,
        ),
        "semantic_tools": _semantic_contracts(skill),
    }
    compact = {key: value for key, value in metadata.items() if value}
    payload = json.dumps(
        compact,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"插件级能力契约：{payload}\n"
        "已有明确 command_id 时可直接指定；否则仅在本插件内部检索具体命令。"
    )


def _semantic_contracts(skill: PluginSkill) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    used = 0
    contracts = sorted(
        skill.semantic_tools,
        key=lambda item: (
            _single_line(item.name).casefold(),
            _single_line(item.name),
        ),
    )
    for contract in contracts:
        item = {
            "name": _clip_text(contract.name, 120),
            "description": _clip_text(contract.description, 320),
            "use_cases": _bounded_values(
                contract.use_cases,
                item_limit=180,
                total_limit=360,
            ),
            "anti_use_cases": _bounded_values(
                contract.anti_use_cases,
                item_limit=180,
                total_limit=360,
            ),
            "output_mode": contract.output_mode,
            "side_effect": contract.side_effect,
            "execution_policy": contract.execution_policy,
            "requires_real_result": contract.requires_real_result,
        }
        compact = {
            key: value
            for key, value in item.items()
            if value is not None and value != []
        }
        size = len(
            json.dumps(
                compact,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        if result and used + size > _SEMANTIC_CONTRACTS_TEXT_LIMIT:
            break
        result.append(compact)
        used += size
    return result


def _bounded_values(
    values: Iterable[object],
    *,
    item_limit: int,
    total_limit: int,
) -> list[str]:
    normalized = sorted(
        {_single_line(value) for value in values if _single_line(value)},
        key=lambda value: (value.casefold(), value),
    )
    result: list[str] = []
    used = 0
    for value in normalized:
        clipped = _clip_text(value, item_limit)
        if result and used + len(clipped) > total_limit:
            break
        if not result and len(clipped) > total_limit:
            clipped = _clip_text(clipped, total_limit)
        result.append(clipped)
        used += len(clipped)
    return result


def _clip_text(value: object, limit: int) -> str:
    text = _single_line(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _single_line(value: object) -> str:
    return " ".join(normalize_message_text(str(value or "")).split())


def _command_key(command_id: object) -> str:
    return normalize_message_text(str(command_id or "")).casefold()


__all__ = [
    "PluginSkillDispatchTool",
    "build_plugin_skill_dispatch_tools",
    "skill_dispatch_tool_name",
]
