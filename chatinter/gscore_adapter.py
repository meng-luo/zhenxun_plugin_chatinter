"""GScore bridge client for ChatInter mixed-chat routing and execution."""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
import time
from typing import Any, Literal

import aiohttp

from zhenxun.services.log import logger

from .candidate_exposure import CandidateExposureKey, CandidateExposureLedger
from .config import get_gscore_bridge_config
from .llm_compat import RunContext, ToolDefinition, ToolResult
from .route_text import normalize_message_text
from .sparse_retrieval import fuse_sparse_rankings, normalize_retrieval_queries
from .token_compat import estimate_text_tokens

_API_PREFIX = "/api/chatinter-bridge/v1"
_ROUTE_TIMEOUT_SECONDS = 2.0
_CAPABILITY_TIMEOUT_SECONDS = 3.0
_TRANSPORT_BACKOFF_SECONDS = (5.0, 10.0, 20.0, 40.0, 60.0)
_EXECUTE_TIMEOUT_SECONDS = 4.0
_EXECUTION_STATUS_TIMEOUT_SECONDS = 4.0
_EXECUTION_STATUS_WAIT_SECONDS = 60.0
_EXECUTION_STATUS_INITIAL_INTERVAL_SECONDS = 0.25
_EXECUTION_STATUS_MAX_INTERVAL_SECONDS = 2.0
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_CAPABILITY_INDEX_CACHE_LIMIT = 8
_CAPABILITY_INDEX_VERSION = 1
_GSCORE_SKILL_TOOL_PREFIX = "ci_gscore_skill_"
_CAPABILITY_FIELD_WEIGHTS = {
    "trigger": 6.0,
    "alias": 5.0,
    "service": 4.5,
    "name": 4.0,
    "example": 3.5,
    "context_tag": 3.5,
    "summary": 5.5,
    "fallback": 2.0,
    "domain": 3.0,
    "plugin": 2.0,
    "schema": 1.0,
}
_ASCII_SEARCH_TERM_PATTERN = re.compile(r"[0-9a-z][0-9a-z_.:/-]*", re.IGNORECASE)
_CJK_SEARCH_CHUNK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
_CAPABILITY_SEARCH_STOP_CHARS = frozenset(
    "我你他她它的了呢啊呀吧吗嘛请用这张这个那个一下一个一条给看查帮寻真"
)
_LEGAL_TRIGGER_TYPES = frozenset(
    {"command", "fullmatch", "keyword", "prefix", "regex", "suffix"}
)

GScoreRouteDisposition = Literal[
    "claimed",
    "unmatched",
    "interactive",
    "blocked",
    "unknown",
    "disabled",
]


@dataclass(frozen=True, slots=True)
class GScoreTriggerPattern:
    trigger_type: str
    keyword: str
    prefix: str = ""
    to_me: bool = False

    @property
    def command(self) -> str:
        return f"{self.prefix}{self.keyword}"


@dataclass(frozen=True, slots=True)
class GScoreCapability:
    capability_id: str
    name: str
    description: str = ""
    plugin: str = ""
    service: str = ""
    retrieval_summary: str = ""
    metadata_sources: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    parameters: dict[str, Any] | None = None
    context_tags: tuple[str, ...] = ()
    capability_domain: str = ""
    trigger_patterns: tuple[GScoreTriggerPattern, ...] = ()
    trigger_type: str = ""
    trigger_keyword: str = ""
    trigger_prefix: str = ""
    trigger_to_me: bool = False
    command_starts: tuple[str, ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        command_starts: tuple[str, ...] = (),
    ) -> GScoreCapability | None:
        if not isinstance(payload, dict):
            return None
        capability_id = normalize_message_text(
            str(payload.get("capability_id") or payload.get("id") or "")
        )
        if not capability_id:
            return None
        plugin_payload = payload.get("plugin")
        plugin_mapping = plugin_payload if isinstance(plugin_payload, dict) else {}
        service_payload = payload.get("service")
        service_mapping = service_payload if isinstance(service_payload, dict) else {}
        trigger_payload = payload.get("trigger")
        trigger_payload = trigger_payload if isinstance(trigger_payload, dict) else {}
        trigger_patterns = _trigger_patterns_from_payload(trigger_payload)
        primary_trigger = (
            trigger_patterns[0] if trigger_patterns else GScoreTriggerPattern("", "")
        )
        plugin_name = normalize_message_text(
            str(
                plugin_mapping.get("name")
                or (plugin_payload if isinstance(plugin_payload, str) else "")
            )
        )
        service_name = normalize_message_text(
            str(
                service_mapping.get("name")
                or (service_payload if isinstance(service_payload, str) else "")
            )
        )
        parameters = payload.get("input_schema") or payload.get("parameters")
        metadata_sources = _accepted_metadata_sources(
            payload,
            has_trigger=bool(trigger_patterns),
        )
        raw_description = str(payload.get("description") or service_name or "")
        return cls(
            capability_id=capability_id,
            name=normalize_message_text(
                str(
                    payload.get("name")
                    or service_name
                    or primary_trigger.keyword
                    or capability_id
                )
            ),
            description=normalize_message_text(raw_description),
            plugin=plugin_name,
            service=service_name,
            retrieval_summary=_payload_retrieval_summary(
                payload,
                raw_description,
            ),
            metadata_sources=metadata_sources,
            aliases=_string_tuple(
                payload.get("aliases") or plugin_mapping.get("aliases")
            ),
            examples=_string_tuple(payload.get("examples")),
            parameters=dict(parameters) if isinstance(parameters, dict) else None,
            context_tags=_string_tuple(
                payload.get("context_tags") or payload.get("tags")
            ),
            capability_domain=normalize_message_text(
                str(payload.get("capability_domain") or payload.get("domain") or "")
            ),
            trigger_patterns=trigger_patterns,
            trigger_type=primary_trigger.trigger_type,
            trigger_keyword=primary_trigger.keyword,
            trigger_prefix=primary_trigger.prefix,
            trigger_to_me=primary_trigger.to_me,
            command_starts=command_starts,
        )


@dataclass(frozen=True, slots=True)
class _CapabilitySearchDocument:
    capability: GScoreCapability
    field_counts: dict[str, dict[str, int]]
    field_lengths: dict[str, int]
    identities: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class _CapabilitySearchIndex:
    documents: tuple[_CapabilitySearchDocument, ...]
    idf: dict[str, float]
    average_field_lengths: dict[str, float]
    identity_document_frequency: dict[str, int]

    def rank(
        self,
        query: str,
        *,
        context_text: str = "",
    ) -> list[tuple[float, GScoreCapability]]:
        primary_terms = set(_capability_search_terms(query))
        query_term_weights = {term: 1.0 for term in primary_terms}
        for term in _capability_search_terms(context_text):
            query_term_weights.setdefault(term, 0.3)
        query_identity = _search_normalize(query)
        if not query_term_weights and not query_identity:
            return []
        ranked: list[tuple[float, GScoreCapability]] = []
        document_count = max(len(self.documents), 1)
        for document in self.documents:
            score = 0.0
            primary_evidence = 0.0
            for field, counts in document.field_counts.items():
                if not counts:
                    continue
                field_score = 0.0
                primary_field_score = 0.0
                field_length = document.field_lengths.get(field, 0)
                average_length = max(
                    self.average_field_lengths.get(field, 0.0),
                    1.0,
                )
                for term, query_weight in query_term_weights.items():
                    term_count = counts.get(term, 0)
                    if term_count <= 0:
                        continue
                    term_score = (
                        self.idf.get(term, 0.0)
                        * (term_count * 2.2)
                        / (
                            term_count
                            + 1.2 * (0.45 + 0.55 * field_length / average_length)
                        )
                    )
                    field_score += query_weight * term_score
                    if term in primary_terms:
                        primary_field_score += term_score
                field_weight = _CAPABILITY_FIELD_WEIGHTS.get(field, 1.0)
                score += field_score * field_weight
                primary_evidence += primary_field_score * field_weight
            for identity, weight in document.identities:
                if not identity or not query_identity:
                    continue
                frequency = self.identity_document_frequency.get(identity, 1)
                rarity = math.log1p(document_count / max(frequency, 1))
                if identity == query_identity:
                    identity_score = weight * rarity * 8.0
                    score += identity_score
                    primary_evidence += identity_score
            if score > 0 and primary_evidence > 0:
                ranked.append((score, document.capability))
        ranked.sort(key=lambda item: (-item[0], item[1].capability_id))
        return ranked


_CAPABILITY_INDEX_CACHE: OrderedDict[str, _CapabilitySearchIndex] = OrderedDict()


@dataclass(frozen=True, slots=True)
class GScoreRouteResult:
    disposition: GScoreRouteDisposition
    revision: str = ""
    matches: tuple[str, ...] = ()
    reason: str = ""

    @property
    def suppress_chatinter(self) -> bool:
        return self.disposition in {
            "claimed",
            "interactive",
            "blocked",
        }


class GScoreBridgeError(RuntimeError):
    def __init__(self, message: str, *, uncertain: bool = False) -> None:
        super().__init__(message)
        self.uncertain = uncertain


class GScoreExecutionTool:
    name = "gscore_execute"
    chatinter_plugin_tool_kind = "gscore"

    def __init__(
        self,
        adapter: GScoreAdapter,
        capabilities: tuple[GScoreCapability, ...],
        message_payload: dict[str, Any],
        ws_bot_id: str,
        revision: str,
        source_request_id: str,
    ) -> None:
        self._adapter = adapter
        self._capabilities = _merge_capabilities_preserving_order(capabilities)
        self._capabilities_by_id = {
            item.capability_id: item for item in self._capabilities
        }
        self._capability_ids = frozenset(
            item.capability_id for item in self._capabilities
        )
        self._message_payload = message_payload
        self._ws_bot_id = ws_bot_id
        self._revision = revision
        self._source_request_id = source_request_id

    @property
    def capability_count(self) -> int:
        return len(self._capabilities)

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "执行本轮候选列表中的 GScore 外部插件能力。"
                "仅在用户明确需要候选能力时调用，不要把普通聊天改写成插件操作。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "capability_id": {
                        "type": "string",
                        "description": "本轮 GScore 候选能力卡中的稳定 capability_id",
                    },
                    "command_text": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "按能力卡触发器构造的完整 GScore 命令文本；"
                            "必须能由所选 trigger 重新匹配"
                        ),
                    },
                },
                "required": ["capability_id", "command_text"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        del context
        capability_id = normalize_message_text(str(kwargs.get("capability_id") or ""))
        command_text = normalize_message_text(str(kwargs.get("command_text") or ""))
        capability = self._capabilities_by_id.get(capability_id)
        if (
            capability is None
            or not command_text
            or not _command_matches_capability(
                capability,
                command_text,
                message=self._message_payload,
            )
        ):
            return ToolResult(
                output={
                    "status": "invalid_arguments",
                    "plugin_execution": False,
                    "executed": False,
                },
                is_error=True,
                is_retryable=False,
            )
        execute_payload = {
            "request_id": _execution_request_id(
                self._source_request_id,
                capability_id,
                command_text,
            ),
            "ws_bot_id": self._ws_bot_id,
            "message": self._message_payload,
            "capability_id": capability_id,
            "revision": self._revision,
            "command_text": command_text,
        }
        request_id = str(execute_payload["request_id"])
        log_context = {
            "request_id": request_id,
            "capability_id": capability_id,
            "message_id": normalize_message_text(
                str(self._message_payload.get("msg_id") or "")
            ),
        }
        try:
            response = await self._adapter.execute(execute_payload)
        except asyncio.TimeoutError as exc:
            _log_execution_event(
                "warning",
                "execute_unknown",
                **log_context,
                error_type=type(exc).__name__,
            )
            return await self._poll_execution_status(
                request_id=request_id,
                capability_id=capability_id,
                message_id=log_context["message_id"],
                submission_confirmed=False,
            )
        except GScoreBridgeError as exc:
            _log_execution_event(
                "warning",
                "execute_failed",
                **log_context,
                error_type=type(exc).__name__,
                execution_uncertain=exc.uncertain,
            )
            if exc.uncertain:
                return await self._poll_execution_status(
                    request_id=request_id,
                    capability_id=capability_id,
                    message_id=log_context["message_id"],
                    submission_confirmed=False,
                )
            return ToolResult(
                output={
                    "status": "unavailable",
                    "plugin_execution": False,
                    "executed": False,
                },
                is_error=True,
                is_retryable=False,
            )

        disposition = normalize_message_text(
            str(response.get("disposition") or response.get("status") or "unknown")
        ).casefold()
        if disposition in {"accepted", "duplicate"}:
            delivery_state, delivery_observed = _delivery_observation(response)
            _log_execution_event(
                "info",
                "execute_accepted",
                **log_context,
                disposition=disposition,
                delivery_state=delivery_state,
                delivery_observed=delivery_observed,
            )
            if delivery_observed:
                return _external_delivery_result(
                    disposition,
                    submitted=True,
                    uncertain=False,
                    delivery_state=delivery_state,
                    delivery_observed=True,
                )
            return await self._poll_execution_status(
                request_id=request_id,
                capability_id=capability_id,
                message_id=log_context["message_id"],
            )
        if disposition == "unknown":
            _log_execution_event(
                "warning",
                "execute_unknown",
                **log_context,
                disposition=disposition,
                execution_uncertain=True,
            )
            return _external_delivery_result("unknown", uncertain=True)
        _log_execution_event(
            "warning",
            "execute_rejected",
            **log_context,
            disposition=disposition or "rejected",
            execution_uncertain=False,
        )
        return ToolResult(
            output={
                "status": disposition or "rejected",
                "plugin_execution": False,
                "executed": False,
            },
            is_error=disposition not in {"rejected", "blocked", "unavailable"},
            is_retryable=False,
        )

    async def _poll_execution_status(
        self,
        *,
        request_id: str,
        capability_id: str,
        message_id: str,
        submission_confirmed: bool = True,
    ) -> ToolResult:
        started_at = time.monotonic()
        deadline = started_at + _EXECUTION_STATUS_WAIT_SECONDS
        interval = _EXECUTION_STATUS_INITIAL_INTERVAL_SECONDS
        attempts = 0
        last_error_type = ""
        status_observed = submission_confirmed

        while (remaining := deadline - time.monotonic()) > 0:
            attempts += 1
            try:
                response = await asyncio.wait_for(
                    self._adapter.execution_status(request_id),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as exc:
                last_error_type = type(exc).__name__
            except GScoreBridgeError as exc:
                last_error_type = type(exc).__name__
            else:
                status_observed = True
                execution_status = _execution_status(response)
                delivery_state, delivery_observed = _delivery_observation(response)
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                if delivery_observed:
                    _log_execution_event(
                        "info",
                        "execution_delivered",
                        request_id=request_id,
                        capability_id=capability_id,
                        message_id=message_id,
                        execution_status=execution_status,
                        delivery_state=delivery_state,
                        poll_attempts=attempts,
                        elapsed_ms=elapsed_ms,
                    )
                    return _external_delivery_result(
                        execution_status or "succeeded",
                        submitted=True,
                        uncertain=False,
                        delivery_state=delivery_state,
                        delivery_observed=True,
                    )
                if execution_status == "succeeded" and _execution_has_no_output(
                    response,
                    delivery_state=delivery_state,
                ):
                    _log_execution_event(
                        "info",
                        "execution_succeeded_no_output",
                        request_id=request_id,
                        capability_id=capability_id,
                        message_id=message_id,
                        execution_status=execution_status,
                        delivery_state="no_output",
                        poll_attempts=attempts,
                        elapsed_ms=elapsed_ms,
                    )
                    return _execution_no_output_result()
                if execution_status == "failed":
                    _log_execution_event(
                        "warning",
                        "execution_failed",
                        request_id=request_id,
                        capability_id=capability_id,
                        message_id=message_id,
                        execution_status=execution_status,
                        delivery_state=delivery_state,
                        poll_attempts=attempts,
                        elapsed_ms=elapsed_ms,
                    )
                    return _execution_failed_result(delivery_state=delivery_state)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(interval, remaining))
            interval = min(
                interval * 2,
                _EXECUTION_STATUS_MAX_INTERVAL_SECONDS,
            )

        _log_execution_event(
            "warning",
            "execution_status_unknown",
            request_id=request_id,
            capability_id=capability_id,
            message_id=message_id,
            poll_attempts=attempts,
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
            error_type=last_error_type or "StatusDeadlineExceeded",
            execution_uncertain=True,
        )
        return _external_delivery_result(
            "unknown",
            submitted=status_observed,
            uncertain=True,
        )


class GScoreSkillDispatchTool:
    """Stable plugin-scoped discovery tool backed by the GScore executor."""

    chatinter_plugin_tool_kind = "gscore"
    chatinter_ignore_unknown_top_level_arguments = True

    def __init__(
        self,
        *,
        plugin_key: str,
        capabilities: tuple[GScoreCapability, ...],
        executor: GScoreExecutionTool,
        revision: str,
        exposure_ledger: CandidateExposureLedger | None = None,
        result_token_budget: int = 4_096,
        result_char_budget: int = 12_000,
    ) -> None:
        self.plugin_key = plugin_key
        self.name = _gscore_skill_tool_name(plugin_key)
        self._capabilities = tuple(
            sorted(capabilities, key=lambda item: item.capability_id)
        )
        self._capabilities_by_id = {
            item.capability_id: item for item in self._capabilities
        }
        self._executor = executor
        self._revision = revision or _capability_index_fingerprint(self._capabilities)
        self._exposure_ledger = exposure_ledger or CandidateExposureLedger()
        self._exposure_key = CandidateExposureKey.build(
            source="gscore",
            skill=plugin_key,
            revision=self._revision,
        )
        self._result_token_budget = max(int(result_token_budget), 1)
        self._result_char_budget = max(int(result_char_budget), 1)

    @property
    def capability_count(self) -> int:
        return len(self._capabilities)

    def with_result_budget(
        self,
        *,
        token_budget: int,
        char_budget: int,
    ) -> GScoreSkillDispatchTool:
        return GScoreSkillDispatchTool(
            plugin_key=self.plugin_key,
            capabilities=self._capabilities,
            executor=self._executor,
            revision=self._revision,
            exposure_ledger=self._exposure_ledger,
            result_token_budget=token_budget,
            result_char_budget=char_budget,
        )

    async def get_definition(self) -> ToolDefinition:
        metadata = _gscore_skill_metadata(self.plugin_key, self._capabilities)
        return ToolDefinition(
            name=self.name,
            description=(
                "GScore 插件级能力契约："
                + json.dumps(
                    metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "。未知具体能力时先用 task_text 和 retrieval_queries 检索；"
                "只有 capability_id 来自返回候选时才能执行。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_text": {
                        "type": "string",
                        "minLength": 1,
                        "description": "当前工具调用对应的用户原话或任务片段",
                    },
                    "retrieval_queries": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 256},
                        "maxItems": 6,
                        "description": (
                            "可选的本插件内检索改写，仅用于查找候选，不会作为命令执行"
                        ),
                    },
                    "capability_id": {
                        "type": ["string", "null"],
                        "description": "仅填写本工具此前返回的真实 capability_id",
                    },
                    "command_text": {
                        "type": ["string", "null"],
                        "description": (
                            "选择能力后，按候选 command_forms 构造完整实际命令；"
                            "literal_head/literal_prefix/literal_suffix 均须原样保留"
                        ),
                    },
                },
                "required": ["task_text"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        task_text = normalize_message_text(str(kwargs.get("task_text") or ""))
        if not task_text:
            return self._not_executed("invalid_arguments", reason="missing_task_text")
        capability_id = normalize_message_text(str(kwargs.get("capability_id") or ""))
        if capability_id.casefold() in {"null", "none", "nil", "undefined"}:
            capability_id = ""
        if capability_id:
            if capability_id not in self._capabilities_by_id:
                return self._selection_result(
                    task_text,
                    kwargs.get("retrieval_queries"),
                    reason="capability_out_of_skill",
                    defer_candidate_exposure=bool(
                        kwargs.get("_defer_candidate_exposure")
                    ),
                )
            if not self._exposure_ledger.is_exposed(
                self._exposure_key,
                capability_id,
            ):
                self._exposure_ledger.record_execution(
                    self._exposure_key,
                    capability_id,
                    valid=False,
                    reason="candidate_identity_not_exposed",
                )
                return self._selection_result(
                    task_text,
                    _append_retrieval_identity(
                        kwargs.get("retrieval_queries"),
                        capability_id,
                    ),
                    reason="candidate_identity_not_exposed",
                    defer_candidate_exposure=bool(
                        kwargs.get("_defer_candidate_exposure")
                    ),
                )
            command_text = normalize_message_text(str(kwargs.get("command_text") or ""))
            if not command_text:
                return self._not_executed(
                    "invalid_arguments",
                    capability_id=capability_id,
                    reason="missing_command_text",
                )
            self._exposure_ledger.record_execution(
                self._exposure_key,
                capability_id,
                valid=True,
                reason="candidate_exposed",
            )
            return await self._executor.execute(
                context,
                capability_id=capability_id,
                command_text=command_text,
            )
        return self._selection_result(
            task_text,
            kwargs.get("retrieval_queries"),
            defer_candidate_exposure=bool(kwargs.get("_defer_candidate_exposure")),
        )

    def _selection_result(
        self,
        task_text: str,
        retrieval_queries: object,
        *,
        reason: str = "",
        defer_candidate_exposure: bool = False,
    ) -> ToolResult:
        queries = normalize_retrieval_queries(task_text, retrieval_queries)
        index = _capability_search_index(self._capabilities, revision=self._revision)
        rankings = [
            [item.capability_id for _score, item in index.rank(query)]
            for query in queries
        ]
        exact_ids = {
            capability_id
            for query in queries
            for capability_id in _matched_capability_identity_ids(
                self._capabilities,
                query,
            )
        }
        fused = fuse_sparse_rankings(queries, rankings, exact_ids=exact_ids)
        ranked = [
            self._capabilities_by_id[capability_id]
            for capability_id in fused.ranked_ids
            if capability_id in self._capabilities_by_id
        ]
        full_listing = not ranked
        candidate_count = len(ranked) if ranked else len(self._capabilities)
        recall = (
            "gscore_full_listing"
            if full_listing
            else "gscore_sparse_multi_query"
            if len(queries) > 1
            else "gscore_sparse"
        )
        base_payload = {
            "candidate_count": candidate_count,
            "recall": recall,
            "reason": reason,
        }
        selected = self._fit_candidates(
            list(self._capabilities) if full_listing else ranked,
            base_payload=base_payload,
            candidate_count=candidate_count,
        )
        if full_listing and len(selected) < len(self._capabilities):
            selected = []
            recall = "gscore_no_recall"
        if not defer_candidate_exposure:
            self.expose_candidates(
                (item.capability_id for item in selected),
                source=recall,
                pending=True,
            )
            self._exposure_ledger.record_discovery(
                self._exposure_key,
                source=recall,
                query_count=len(queries),
                candidate_count=candidate_count,
                displayed_count=len(selected),
                omitted_count=max(candidate_count - len(selected), 0),
            )
        return self._not_executed(
            "selection_required",
            candidates=[_capability_candidate_card(item) for item in selected],
            candidate_count=candidate_count,
            displayed_candidate_count=len(selected),
            omitted_candidate_count=max(candidate_count - len(selected), 0),
            truncated=len(selected) < candidate_count,
            recall=recall,
            reason=reason,
        )

    @property
    def exposure_key(self) -> CandidateExposureKey:
        return self._exposure_key

    def owns_candidate_identity(self, identity: object) -> bool:
        return normalize_message_text(str(identity or "")) in self._capabilities_by_id

    def is_candidate_exposed(self, identity: object) -> bool:
        return self._exposure_ledger.is_exposed(self._exposure_key, identity)

    def expose_candidates(
        self,
        identities: Any,
        *,
        source: str,
        pending: bool,
        exact_identity: bool = False,
    ) -> tuple[str, ...]:
        valid = (
            identity
            for value in identities
            if (
                identity := normalize_message_text(str(value or ""))
            ) in self._capabilities_by_id
        )
        return self._exposure_ledger.expose(
            self._exposure_key,
            valid,
            discovery_source=source,
            exact_identity=exact_identity,
            pending=pending,
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

    def _fit_candidates(
        self,
        capabilities: list[GScoreCapability],
        *,
        base_payload: dict[str, Any],
        candidate_count: int,
    ) -> list[GScoreCapability]:
        selected: list[GScoreCapability] = []
        cards: list[dict[str, Any]] = []
        for capability in capabilities:
            trial_cards = [*cards, _capability_candidate_card(capability)]
            output = {
                "status": "selection_required",
                "plugin_execution": False,
                "executed": False,
                "gscore_plugin": self.plugin_key,
                **base_payload,
                "candidates": trial_cards,
                "displayed_candidate_count": len(trial_cards),
                "omitted_candidate_count": max(
                    candidate_count - len(trial_cards),
                    0,
                ),
                "truncated": len(trial_cards) < candidate_count,
            }
            serialized = json.dumps(
                {
                    key: value
                    for key, value in output.items()
                    if value not in (None, "")
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
            if (
                estimate_text_tokens(serialized) > self._result_token_budget
                or len(serialized) > self._result_char_budget
            ):
                continue
            selected.append(capability)
            cards = trial_cards
        return selected

    def _not_executed(self, status: str, **payload: Any) -> ToolResult:
        return ToolResult(
            output={
                "status": status,
                "plugin_execution": False,
                "executed": False,
                "gscore_plugin": self.plugin_key,
                **{
                    key: value
                    for key, value in payload.items()
                    if value not in (None, "")
                },
            },
            display_content=f"{self.name}: {status}",
            is_retryable=status == "selection_required",
        )


class GScoreAdapter:
    def __init__(self) -> None:
        self._capabilities: tuple[GScoreCapability, ...] = ()
        self._capabilities_loaded = False
        self._revision = ""
        self._capability_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._http_session: aiohttp.ClientSession | None = None
        self._revision_epoch = 0
        self._config_fingerprint = ""
        self._transport_failure_count = 0
        self._transport_backoff_until = 0.0

    @property
    def enabled(self) -> bool:
        config = get_gscore_bridge_config()
        return bool(config["enabled"] and config["url"] and config["secret"])

    def _sync_configuration(self) -> tuple[bool, bool]:
        config = get_gscore_bridge_config()
        enabled = bool(config["enabled"] and config["url"] and config["secret"])
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "enabled": enabled,
                    "url": config["url"],
                    "secret": config["secret"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        changed = fingerprint != self._config_fingerprint
        if changed:
            self._config_fingerprint = fingerprint
            self._capabilities = ()
            self._capabilities_loaded = False
            self._revision = ""
            self._revision_epoch += 1
            self._clear_transport_backoff()
        return enabled, changed

    async def _prepare(self) -> bool:
        enabled, changed = self._sync_configuration()
        if changed:
            await self.close()
        return enabled

    async def route_turn(self, frame: Any) -> GScoreRouteResult:
        if not await self._prepare() or not _mixed_tools_allowed(frame):
            return GScoreRouteResult("disabled")
        if not self._transport_probe_allowed():
            return GScoreRouteResult("unknown", reason="transport_backoff")
        message_payload = build_gscore_event_payload(frame)
        if not message_payload:
            return GScoreRouteResult("disabled")
        payload = {
            "request_id": _route_request_id(
                frame,
                str(message_payload.get("msg_id", "") or ""),
            ),
            "ws_bot_id": _gscore_ws_bot_id(),
            "message": message_payload,
        }
        try:
            response = await self._request_json(
                "POST",
                "/route",
                payload,
                timeout_seconds=_ROUTE_TIMEOUT_SECONDS,
            )
        except (GScoreBridgeError, asyncio.TimeoutError) as exc:
            self._record_transport_failure()
            logger.warning(f"ChatInter GScore route result unknown: {exc}")
            return GScoreRouteResult("unknown", reason=type(exc).__name__)

        self._clear_transport_backoff()

        disposition = normalize_message_text(
            str(response.get("disposition") or "unknown")
        ).casefold()
        if disposition not in {
            "claimed",
            "unmatched",
            "interactive",
            "blocked",
            "unknown",
        }:
            disposition = "unknown"
        revision = normalize_message_text(str(response.get("revision") or ""))
        matches = _string_tuple(response.get("matches"))
        self._observe_revision(revision)
        return GScoreRouteResult(
            disposition=disposition,
            revision=revision,
            matches=matches,
            reason=normalize_message_text(str(response.get("reason") or "")),
        )

    async def build_tools(
        self,
        frame: Any,
        *,
        route_result: GScoreRouteResult,
        exposure_ledger: CandidateExposureLedger | None = None,
    ) -> dict[str, GScoreSkillDispatchTool]:
        if route_result.disposition != "unmatched":
            return {}
        if not await self._prepare() or not _mixed_tools_allowed(frame):
            return {}
        message_payload = build_gscore_event_payload(frame)
        if not message_payload:
            return {}
        capabilities = await self.get_capabilities()
        if not capabilities:
            return {}
        source_request_id = _route_request_id(
            frame,
            str(message_payload.get("msg_id", "") or ""),
        )
        tools: dict[str, GScoreSkillDispatchTool] = {}
        for plugin_key, grouped in _group_gscore_capabilities(capabilities):
            executor = GScoreExecutionTool(
                self,
                grouped,
                message_payload,
                _gscore_ws_bot_id(),
                self._revision,
                source_request_id,
            )
            tool = GScoreSkillDispatchTool(
                plugin_key=plugin_key,
                capabilities=grouped,
                executor=executor,
                revision=self._revision,
                exposure_ledger=exposure_ledger,
            )
            tools[tool.name] = tool
        return dict(sorted(tools.items()))

    async def get_capabilities(self) -> tuple[GScoreCapability, ...]:
        if not await self._prepare():
            return ()
        if self._capabilities_loaded:
            return self._capabilities
        async with self._capability_lock:
            if self._capabilities_loaded:
                return self._capabilities
            if not self._transport_probe_allowed():
                return ()
            for _attempt in range(2):
                observed_epoch = self._revision_epoch
                try:
                    response = await self._request_json(
                        "GET",
                        "/capabilities",
                        None,
                        timeout_seconds=_CAPABILITY_TIMEOUT_SECONDS,
                    )
                except (GScoreBridgeError, asyncio.TimeoutError) as exc:
                    self._record_transport_failure()
                    logger.warning(
                        f"ChatInter GScore capability discovery failed: {exc}"
                    )
                    return ()
                self._clear_transport_backoff()
                revision = normalize_message_text(str(response.get("revision") or ""))
                if (
                    self._revision_epoch != observed_epoch
                    and self._revision
                    and revision != self._revision
                ):
                    continue
                items = response.get("capabilities")
                parsed = tuple(
                    item
                    for payload in items or ()
                    if (item := GScoreCapability.from_payload(payload)) is not None
                )
                capabilities = _merge_capabilities(parsed)
                self._observe_revision(revision)
                self._capabilities = capabilities
                self._capabilities_loaded = True
                return self._capabilities
        return ()

    def _transport_probe_allowed(self) -> bool:
        return time.monotonic() >= self._transport_backoff_until

    def _record_transport_failure(self) -> None:
        self._transport_failure_count = min(
            self._transport_failure_count + 1,
            len(_TRANSPORT_BACKOFF_SECONDS),
        )
        delay = _TRANSPORT_BACKOFF_SECONDS[self._transport_failure_count - 1]
        self._transport_backoff_until = time.monotonic() + delay

    def _clear_transport_backoff(self) -> None:
        self._transport_failure_count = 0
        self._transport_backoff_until = 0.0

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            "/execute",
            payload,
            timeout_seconds=_EXECUTE_TIMEOUT_SECONDS,
        )

    async def execution_status(self, request_id: str) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            f"/executions/{request_id}",
            None,
            timeout_seconds=_EXECUTION_STATUS_TIMEOUT_SECONDS,
        )

    def _observe_revision(self, revision: str) -> None:
        if revision and self._revision and revision != self._revision:
            self._capabilities = ()
            self._capabilities_loaded = False
        if revision and revision != self._revision:
            self._revision_epoch += 1
        if revision:
            self._revision = revision

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        config = get_gscore_bridge_config()
        if not config["enabled"] or not config["url"] or not config["secret"]:
            raise GScoreBridgeError("bridge is not configured")
        body = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if payload is not None
            else b""
        )
        timestamp = str(int(time.time()))
        signature = hmac.new(
            str(config["secret"]).encode(),
            timestamp.encode() + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "Accept": "application/json",
            "X-ChatInter-Timestamp": timestamp,
            "X-ChatInter-Signature": signature,
        }
        if body:
            headers["Content-Type"] = "application/json"
        url = f"{config['url']}{_API_PREFIX}{path}"
        side_effecting = method == "POST" and path == "/execute"
        try:
            session = await self._get_http_session()
            async with session.request(
                method,
                url,
                data=body if body else None,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as response:
                raw = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if len(raw) + len(chunk) > _MAX_RESPONSE_BYTES:
                        raise GScoreBridgeError(
                            "bridge response is too large",
                            uncertain=side_effecting,
                        )
                    raw.extend(chunk)
                if response.status >= 400:
                    raise GScoreBridgeError(
                        f"bridge returned HTTP {response.status}",
                        uncertain=side_effecting and response.status >= 500,
                    )
        except asyncio.TimeoutError:
            raise
        except aiohttp.ClientConnectorError as exc:
            raise GScoreBridgeError(str(exc)) from exc
        except aiohttp.ServerDisconnectedError as exc:
            raise GScoreBridgeError(
                str(exc),
                uncertain=side_effecting,
            ) from exc
        except aiohttp.ClientError as exc:
            raise GScoreBridgeError(
                str(exc),
                uncertain=side_effecting,
            ) from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GScoreBridgeError(
                "bridge returned invalid JSON",
                uncertain=side_effecting,
            ) from exc
        if not isinstance(decoded, dict):
            raise GScoreBridgeError(
                "bridge response must be an object",
                uncertain=side_effecting,
            )
        data = decoded.get("data") if "data" in decoded else decoded
        if not isinstance(data, dict):
            raise GScoreBridgeError(
                "bridge response data must be an object",
                uncertain=side_effecting,
            )
        return data

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is not None and not self._http_session.closed:
            return self._http_session
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                self._http_session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=8, ttl_dns_cache=300),
                )
            return self._http_session

    async def close(self) -> None:
        session = self._http_session
        self._http_session = None
        if session is not None and not session.closed:
            await session.close()


def build_gscore_event_payload(frame: Any) -> dict[str, Any]:
    event_context = getattr(frame, "event_context", None)
    if event_context is None:
        return {}
    event = getattr(frame, "event", None)
    event_id = normalize_message_text(str(getattr(event_context, "event_id", "") or ""))
    payload = {
        "bot_id": _event_adapter_id(event),
        "bot_self_id": str(getattr(event_context, "bot_id", "") or ""),
        "msg_id": event_id,
        "user_type": "group" if getattr(event_context, "group_id", None) else "direct",
        "group_id": getattr(event_context, "group_id", None),
        "user_id": str(getattr(event_context, "user_id", "") or ""),
        "sender": _event_sender(event),
        "user_pm": _event_user_pm(frame, event),
        "content": _event_content(frame),
    }
    return (
        payload
        if payload["bot_id"] and payload["user_id"] and payload["content"]
        else {}
    )


def get_gscore_adapter() -> GScoreAdapter:
    return _GSCORE_ADAPTER


def _external_delivery_result(
    status: str,
    *,
    submitted: bool = False,
    uncertain: bool = True,
    delivery_state: str = "unknown",
    delivery_observed: bool = False,
) -> ToolResult:
    output: dict[str, Any] = {
        "status": status,
        "plugin_execution": True,
        "submitted": submitted,
        "executed": delivery_observed,
        "execution_uncertain": uncertain,
        "external_delivery": delivery_observed,
        "delivery_observed": delivery_observed,
        "delivery_state": delivery_state,
    }
    if delivery_observed:
        output["delivery_owner"] = "gscore"
    return ToolResult(
        output=output,
        is_error=not submitted,
        is_retryable=False,
    )


def _execution_no_output_result() -> ToolResult:
    return ToolResult(
        output={
            "status": "succeeded",
            "plugin_execution": True,
            "submitted": True,
            "executed": True,
            "execution_uncertain": False,
            "external_delivery": False,
            "delivery_observed": False,
            "delivery_state": "no_output",
        },
        display_content="操作已执行完成。",
        is_error=False,
        is_retryable=False,
    )


def _execution_failed_result(*, delivery_state: str) -> ToolResult:
    return ToolResult(
        output={
            "status": "failed",
            "plugin_execution": True,
            "submitted": True,
            "executed": False,
            "execution_uncertain": False,
            "external_delivery": False,
            "delivery_observed": False,
            "delivery_state": delivery_state,
        },
        display_content="外部插件执行失败。",
        is_error=True,
        is_retryable=False,
    )


def _execution_status(response: dict[str, Any]) -> str:
    status = normalize_message_text(
        str(
            response.get("execution_state")
            or response.get("execution_status")
            or response.get("status")
            or response.get("disposition")
            or ""
        )
    ).casefold()
    if status in {"success", "succeeded", "complete", "completed"}:
        return "succeeded"
    if status in {"error", "failed", "rejected", "blocked", "cancelled"}:
        return "failed"
    if status in {"accepted", "pending", "queued", "running", "processing"}:
        return "pending"
    return status


def _execution_has_no_output(
    response: dict[str, Any],
    *,
    delivery_state: str,
) -> bool:
    output_state = normalize_message_text(
        str(response.get("output_state") or "")
    ).casefold()
    return bool(
        response.get("no_output") is True
        or delivery_state == "no_output"
        or output_state == "no_output"
    )


def _log_execution_event(level: str, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        **{key: value for key, value in fields.items() if value not in (None, "")},
    }
    message = "ChatInter GScore execution " + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    if level == "warning":
        logger.warning(message)
    else:
        logger.info(message)


def _delivery_observation(response: dict[str, Any]) -> tuple[str, bool]:
    state = normalize_message_text(str(response.get("delivery_state") or "unknown"))
    state = state.casefold()
    observed = response.get("delivery_observed") is True or state in {
        "complete",
        "completed",
        "delivered",
        "observed",
        "sent",
    }
    return (state if state else "unknown"), observed


def _capability_card(capability: GScoreCapability) -> str:
    return json.dumps(
        _capability_candidate_card(capability),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _capability_candidate_card(capability: GScoreCapability) -> dict[str, Any]:
    display_name = _capability_display_name(capability)
    regex_identities = _regex_trigger_identities(capability)
    aliases = [value for value in capability.aliases if value not in regex_identities]
    card: dict[str, Any] = {
        "capability_id": capability.capability_id,
        "plugin": capability.plugin,
        "name": display_name,
        "description": capability.description,
        "aliases": aliases,
        "examples": list(capability.examples),
        "domain": capability.capability_domain,
        "context_tags": list(capability.context_tags),
        "parameters": capability.parameters or {},
        "triggers": _project_trigger_groups(capability.trigger_patterns),
        "command_forms": _project_command_forms(capability.trigger_patterns),
    }
    return {
        key: value for key, value in card.items() if value not in (None, "", [], (), {})
    }


def _group_gscore_capabilities(
    capabilities: tuple[GScoreCapability, ...],
) -> tuple[tuple[str, tuple[GScoreCapability, ...]], ...]:
    grouped: dict[str, list[GScoreCapability]] = {}
    labels: dict[str, str] = {}
    for capability in capabilities:
        label = normalize_message_text(
            capability.plugin
            or capability.capability_domain
            or capability.service
            or "GScore"
        )
        key = label.casefold()
        labels.setdefault(key, label)
        grouped.setdefault(key, []).append(capability)
    return tuple(
        (
            labels[key],
            tuple(sorted(grouped[key], key=lambda item: item.capability_id)),
        )
        for key in sorted(grouped)
    )


def _gscore_skill_tool_name(plugin_key: str) -> str:
    normalized = normalize_message_text(plugin_key).casefold()
    digest = hashlib.blake2s(normalized.encode("utf-8"), digest_size=5).hexdigest()
    slug = re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")[:32]
    return f"{_GSCORE_SKILL_TOOL_PREFIX}{slug or 'plugin'}_{digest}"


def _gscore_skill_metadata(
    plugin_key: str,
    capabilities: tuple[GScoreCapability, ...],
) -> dict[str, Any]:
    domains = _bounded_unique_texts(
        (item.capability_domain for item in capabilities),
        total_chars=320,
    )
    summaries = _bounded_unique_texts(
        (
            item.retrieval_summary or item.description or item.service
            for item in capabilities
        ),
        total_chars=720,
    )
    metadata: dict[str, Any] = {
        "plugin": plugin_key,
        "capability_count": len(capabilities),
        "domains": domains,
        "summaries": summaries,
    }
    return {
        key: value for key, value in metadata.items() if value not in (None, "", [])
    }


def _bounded_unique_texts(
    values: Any,
    *,
    total_chars: int,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    used = 0
    for value in values:
        text = normalize_message_text(str(value or ""))
        key = text.casefold()
        if not text or key in seen:
            continue
        remaining = total_chars - used
        if remaining <= 0:
            break
        clipped = text[:remaining]
        result.append(clipped)
        seen.add(key)
        used += len(clipped)
    return result


def _project_trigger_groups(
    patterns: tuple[GScoreTriggerPattern, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, bool], list[str]] = {}
    for pattern in patterns:
        key = (pattern.trigger_type, pattern.keyword, pattern.to_me)
        prefixes = grouped.setdefault(key, [])
        if pattern.prefix not in prefixes:
            prefixes.append(pattern.prefix)
    return [
        {
            "type": trigger_type,
            "keyword": keyword,
            "prefixes": prefixes,
            **({"to_me": True} if to_me else {}),
        }
        for (trigger_type, keyword, to_me), prefixes in grouped.items()
    ]


def _project_command_forms(
    patterns: tuple[GScoreTriggerPattern, ...],
) -> list[dict[str, Any]]:
    """Project trigger metadata into complete, copyable command constraints."""
    forms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in patterns:
        if pattern.trigger_type == "fullmatch":
            form: dict[str, Any] = {
                "type": "fullmatch",
                "command_text": pattern.command,
            }
        elif pattern.trigger_type in {"command", "prefix"}:
            form = {
                "type": pattern.trigger_type,
                "literal_head": pattern.command,
                "arguments": (
                    "required_after_head"
                    if pattern.trigger_type == "prefix"
                    else "optional_after_head"
                ),
            }
        elif pattern.trigger_type == "keyword":
            form = {
                "type": "keyword",
                "required_keyword": pattern.keyword,
                **(
                    {"literal_prefix": pattern.prefix}
                    if pattern.prefix
                    else {}
                ),
            }
        elif pattern.trigger_type == "suffix":
            form = {
                "type": "suffix",
                "literal_suffix": pattern.keyword,
                "arguments": "required_before_suffix",
                **(
                    {"literal_prefix": pattern.prefix}
                    if pattern.prefix
                    else {}
                ),
            }
        else:
            form = {
                "type": "regex",
                "pattern": pattern.keyword,
                **(
                    {"literal_prefix": pattern.prefix}
                    if pattern.prefix
                    else {}
                ),
            }
        if pattern.to_me:
            form["to_me"] = True
        identity = json.dumps(
            form,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if identity not in seen:
            seen.add(identity)
            forms.append(form)
    return forms


def _matched_capability_identity_ids(
    capabilities: tuple[GScoreCapability, ...],
    message_text: str,
) -> frozenset[str]:
    query_identity = _search_normalize(message_text)
    if not query_identity:
        return frozenset()
    return frozenset(
        capability.capability_id
        for capability in capabilities
        if any(
            identity == query_identity
            for identity, _weight in _capability_search_identities(capability)
            if identity
        )
    )


def _append_retrieval_identity(
    retrieval_queries: object,
    identity: str,
) -> list[object]:
    values = list(
        retrieval_queries
        if isinstance(retrieval_queries, list | tuple)
        else ()
    )
    values.append(identity)
    return values


def _capability_search_index(
    capabilities: tuple[GScoreCapability, ...],
    *,
    revision: str = "",
) -> _CapabilitySearchIndex:
    content_fingerprint = _capability_index_fingerprint(capabilities)
    cache_key = f"{_CAPABILITY_INDEX_VERSION}:{revision}:{content_fingerprint}"
    cached = _CAPABILITY_INDEX_CACHE.get(cache_key)
    if cached is not None:
        _CAPABILITY_INDEX_CACHE.move_to_end(cache_key)
        return cached

    raw_documents: list[
        tuple[
            GScoreCapability,
            dict[str, list[str]],
            tuple[tuple[str, float], ...],
        ]
    ] = []
    document_frequency: Counter[str] = Counter()
    identity_document_frequency: Counter[str] = Counter()
    field_length_totals: Counter[str] = Counter()
    field_document_counts: Counter[str] = Counter()
    for capability in capabilities:
        fields = _capability_search_fields(capability)
        identities = _capability_search_identities(capability)
        raw_documents.append((capability, fields, identities))
        document_frequency.update({term for terms in fields.values() for term in terms})
        identity_document_frequency.update({value for value, _weight in identities})
        for field, terms in fields.items():
            if terms:
                field_length_totals[field] += len(terms)
                field_document_counts[field] += 1

    document_count = max(len(raw_documents), 1)
    idf = {
        term: math.log1p((document_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    documents = tuple(
        _CapabilitySearchDocument(
            capability=capability,
            field_counts={
                field: dict(Counter(terms)) for field, terms in fields.items()
            },
            field_lengths={field: len(terms) for field, terms in fields.items()},
            identities=identities,
        )
        for capability, fields, identities in raw_documents
    )
    index = _CapabilitySearchIndex(
        documents=documents,
        idf=idf,
        average_field_lengths={
            field: field_length_totals[field] / count
            for field, count in field_document_counts.items()
            if count > 0
        },
        identity_document_frequency=dict(identity_document_frequency),
    )
    _CAPABILITY_INDEX_CACHE[cache_key] = index
    _CAPABILITY_INDEX_CACHE.move_to_end(cache_key)
    while len(_CAPABILITY_INDEX_CACHE) > _CAPABILITY_INDEX_CACHE_LIMIT:
        _CAPABILITY_INDEX_CACHE.popitem(last=False)
    return index


def _capability_search_fields(
    capability: GScoreCapability,
) -> dict[str, list[str]]:
    trigger_values = [
        value
        for pattern in capability.trigger_patterns
        if pattern.trigger_type != "regex"
        for value in (pattern.keyword, pattern.command)
    ]
    aliases = [
        value
        for value in capability.aliases
        if value not in _regex_trigger_identities(capability)
    ]
    name_values = [_capability_display_name(capability)]
    summary_field = "summary" if "to_ai" in capability.metadata_sources else "fallback"
    raw_fields: dict[str, list[str]] = {
        "trigger": trigger_values,
        "alias": aliases,
        "service": [capability.service],
        "name": name_values,
        "example": list(capability.examples),
        "context_tag": list(capability.context_tags),
        summary_field: [capability.retrieval_summary],
        "domain": [capability.capability_domain],
        "plugin": [capability.plugin],
        "schema": list(_schema_search_fields(capability.parameters)),
    }
    return {
        field: [term for value in values for term in _capability_search_terms(value)]
        for field, values in raw_fields.items()
    }


def _capability_search_identities(
    capability: GScoreCapability,
) -> tuple[tuple[str, float], ...]:
    weighted_values = [
        (capability.capability_id, 7.0),
        *(
            (pattern.keyword, 6.0)
            for pattern in capability.trigger_patterns
            if pattern.trigger_type != "regex"
        ),
        *(
            (pattern.command, 6.0)
            for pattern in capability.trigger_patterns
            if pattern.trigger_type != "regex"
        ),
        *((value, 5.0) for value in capability.aliases),
        (capability.service, 4.5),
        (_capability_display_name(capability), 4.0),
    ]
    identities: dict[str, float] = {}
    regex_identities = _regex_trigger_identities(capability)
    for value, weight in weighted_values:
        if value in regex_identities:
            continue
        identity = _search_normalize(value)
        if identity:
            identities[identity] = max(identities.get(identity, 0.0), weight)
    return tuple(sorted(identities.items()))


def _capability_search_terms(value: object) -> list[str]:
    normalized = normalize_message_text(str(value or "")).casefold()
    terms: list[str] = []
    for match in _ASCII_SEARCH_TERM_PATTERN.findall(normalized):
        lowered = match.casefold()
        terms.append(lowered)
        terms.extend(
            part for part in re.split(r"[_.:/-]+", lowered) if part and part != lowered
        )
    for chunk in _CJK_SEARCH_CHUNK_PATTERN.findall(normalized):
        chunk = "".join(
            char for char in chunk if char not in _CAPABILITY_SEARCH_STOP_CHARS
        )
        if not chunk:
            continue
        if len(chunk) == 1:
            terms.append(chunk)
            continue
        max_size = min(len(chunk), 4)
        for size in range(2, max_size + 1):
            terms.extend(
                chunk[start : start + size] for start in range(len(chunk) - size + 1)
            )
    return terms


def _regex_trigger_identities(capability: GScoreCapability) -> frozenset[str]:
    return frozenset(
        value
        for pattern in capability.trigger_patterns
        if pattern.trigger_type == "regex"
        for value in (pattern.keyword, pattern.command)
        if value
    )


def _capability_name_contains_regex(capability: GScoreCapability) -> bool:
    return any(
        pattern and pattern in capability.name
        for pattern in _regex_trigger_identities(capability)
    )


def _capability_display_name(capability: GScoreCapability) -> str:
    if not _capability_name_contains_regex(capability):
        return capability.name
    if capability.service:
        return capability.service
    name = capability.name
    for pattern in _regex_trigger_identities(capability):
        name = name.replace(pattern, " ")
    normalized = normalize_message_text(name).strip(" -:：")
    return normalized or capability.plugin


def _capability_index_fingerprint(
    capabilities: tuple[GScoreCapability, ...],
) -> str:
    digest = hashlib.blake2s(digest_size=16)
    for capability in capabilities:
        payload = {
            "capability_id": capability.capability_id,
            "fields": _capability_search_fields(capability),
            "identities": _capability_search_identities(capability),
        }
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _search_normalize(value: object) -> str:
    return "".join(
        char.casefold()
        for char in normalize_message_text(str(value or ""))
        if char.isalnum()
    )


def _message_text(message_payload: dict[str, Any]) -> str:
    content = message_payload.get("content")
    if not isinstance(content, list):
        return ""
    return " ".join(
        normalize_message_text(str(item.get("data") or ""))
        for item in content
        if isinstance(item, dict) and str(item.get("type") or "") == "text"
    )


def _frame_retrieval_context(frame: Any) -> str:
    thread = getattr(frame, "thread_context", None)
    if thread is None:
        return ""
    values: list[object] = [getattr(thread, "topic_key", "")]
    values.extend(getattr(thread, "entity_hints", ()) or ())
    values.extend(getattr(thread, "pending_entities", ()) or ())
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_message_text(str(value or ""))
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            normalized.append(text)
    return " ".join(normalized)


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list | tuple | set | frozenset):
        values = value
    else:
        values = ()
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = normalize_message_text(str(item or ""))
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return tuple(result)


def _payload_retrieval_summary(
    payload: dict[str, Any],
    description: str,
) -> str:
    explicit = normalize_message_text(str(payload.get("retrieval_summary") or ""))
    if explicit:
        return explicit
    normalized = str(description or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        lines.append(stripped)
    return normalize_message_text(" ".join(lines))


def _exact_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list | tuple | set | frozenset):
        values = value
    else:
        values = ()
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = normalize_message_text(str(item or ""))
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _accepted_metadata_sources(
    payload: dict[str, Any],
    *,
    has_trigger: bool,
) -> tuple[str, ...]:
    raw_sources = _merge_string_tuples(
        (
            _string_tuple(payload.get("source") or payload.get("metadata_source")),
            _string_tuple(payload.get("metadata_sources")),
        )
    )
    accepted = _string_tuple(
        tuple(
            normalized
            for source in raw_sources
            if (normalized := _canonical_metadata_source(source))
        )
    )
    if not raw_sources and has_trigger:
        return ("trigger",)
    return accepted


def _canonical_metadata_source(source: str) -> str:
    normalized = source.casefold().replace("-", "_")
    if normalized in {"to_ai", "toai"}:
        return "to_ai"
    if "trigger" in normalized:
        return "trigger"
    return ""


def _trigger_patterns_from_payload(
    trigger: dict[str, Any],
) -> tuple[GScoreTriggerPattern, ...]:
    patterns: list[GScoreTriggerPattern] = []

    def add(mapping: dict[str, Any]) -> None:
        trigger_type = normalize_message_text(
            str(mapping.get("type") or trigger.get("type") or "")
        ).casefold()
        keyword = normalize_message_text(str(mapping.get("keyword") or ""))
        prefix = normalize_message_text(str(mapping.get("prefix") or ""))
        if trigger_type in _LEGAL_TRIGGER_TYPES and keyword:
            patterns.append(
                GScoreTriggerPattern(
                    trigger_type=trigger_type,
                    keyword=keyword,
                    prefix=prefix,
                    to_me=bool(mapping.get("to_me", trigger.get("to_me", False))),
                )
            )

    add(trigger)
    for key in ("patterns", "routes"):
        nested = trigger.get(key)
        if isinstance(nested, list | tuple):
            for item in nested:
                if isinstance(item, dict):
                    add(item)

    base_type = normalize_message_text(str(trigger.get("type") or "")).casefold()
    base_keyword = normalize_message_text(str(trigger.get("keyword") or ""))
    for prefix in _exact_string_tuple(trigger.get("prefixes")):
        add({"type": base_type, "keyword": base_keyword, "prefix": prefix})
    for command in _exact_string_tuple(trigger.get("commands")):
        if base_type == "regex":
            add({"type": base_type, "keyword": command})
            continue
        if base_keyword and command.endswith(base_keyword):
            add(
                {
                    "type": base_type,
                    "keyword": base_keyword,
                    "prefix": command[: -len(base_keyword)],
                }
            )
        else:
            add({"type": base_type, "keyword": command})
    return _merge_trigger_patterns(patterns)


def _merge_trigger_patterns(
    patterns: Any,
) -> tuple[GScoreTriggerPattern, ...]:
    unique = {
        (
            pattern.trigger_type,
            pattern.prefix,
            pattern.keyword,
            pattern.to_me,
        ): pattern
        for pattern in patterns
        if pattern.trigger_type in _LEGAL_TRIGGER_TYPES and pattern.keyword
    }
    return tuple(unique[key] for key in sorted(unique))


def _trigger_pattern_projection(pattern: GScoreTriggerPattern) -> str:
    if pattern.trigger_type == "regex":
        return (
            f"regex(prefix={pattern.prefix},pattern={pattern.keyword},"
            f"to_me={str(pattern.to_me).lower()})"
        )
    return (
        f"{pattern.trigger_type}(pattern={pattern.command},"
        f"to_me={str(pattern.to_me).lower()})"
    )


def _schema_search_fields(schema: dict[str, Any] | None) -> tuple[str, ...]:
    if not schema:
        return ()
    values: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str):
            text = normalize_message_text(value)
            if text:
                values.append(text)

    def walk(value: object) -> None:
        if isinstance(value, dict):
            add(value.get("title"))
            add(value.get("description"))
            properties = value.get("properties")
            if isinstance(properties, dict):
                for name, definition in properties.items():
                    add(name)
                    walk(definition)
            required = value.get("required")
            if isinstance(required, list | tuple):
                for item in required:
                    add(item)
            enum = value.get("enum")
            if isinstance(enum, list | tuple):
                for item in enum:
                    add(item)
            for key in ("items", "anyOf", "oneOf", "allOf"):
                nested = value.get(key)
                if isinstance(nested, list | tuple):
                    for item in nested:
                        walk(item)
                elif isinstance(nested, dict):
                    walk(nested)

    walk(schema)
    return _string_tuple(values)


def _merge_capabilities(
    capabilities: tuple[GScoreCapability, ...],
) -> tuple[GScoreCapability, ...]:
    grouped: dict[str, list[GScoreCapability]] = {}
    for capability in capabilities:
        grouped.setdefault(capability.capability_id, []).append(capability)
    merged = (
        _merge_capability_group(grouped[capability_id])
        for capability_id in sorted(grouped)
    )
    return tuple(
        capability
        for capability in merged
        if capability.metadata_sources and capability.trigger_patterns
    )


def _merge_capabilities_preserving_order(
    capabilities: tuple[GScoreCapability, ...],
) -> tuple[GScoreCapability, ...]:
    grouped: dict[str, list[GScoreCapability]] = {}
    for capability in capabilities:
        grouped.setdefault(capability.capability_id, []).append(capability)
    return tuple(
        merged
        for group in grouped.values()
        if (merged := _merge_capability_group(group)).metadata_sources
        and merged.trigger_patterns
    )


def _merge_capability_group(
    capabilities: list[GScoreCapability],
) -> GScoreCapability:
    ranked = sorted(
        capabilities,
        key=lambda item: -_capability_source_priority(item),
    )

    def first_text(field: str) -> str:
        return next(
            (value for item in ranked if (value := str(getattr(item, field) or ""))),
            "",
        )

    trigger_sources = tuple(
        item
        for item in ranked
        if "trigger" in item.metadata_sources and item.trigger_patterns
    )
    if not trigger_sources:
        trigger_sources = tuple(
            item
            for item in ranked
            if "to_ai" in item.metadata_sources and item.trigger_patterns
        )
    trigger_patterns = _merge_trigger_patterns(
        pattern for item in trigger_sources for pattern in item.trigger_patterns
    )
    primary_trigger = (
        trigger_patterns[0] if trigger_patterns else GScoreTriggerPattern("", "")
    )
    parameters = next(
        (item.parameters for item in ranked if item.parameters is not None),
        None,
    )
    return GScoreCapability(
        capability_id=ranked[0].capability_id,
        name=first_text("name"),
        description=first_text("description"),
        plugin=first_text("plugin"),
        service=first_text("service"),
        retrieval_summary=first_text("retrieval_summary"),
        metadata_sources=_merge_string_tuples(item.metadata_sources for item in ranked),
        aliases=_merge_string_tuples(item.aliases for item in ranked),
        examples=_merge_string_tuples(item.examples for item in ranked),
        parameters=parameters,
        context_tags=_merge_string_tuples(item.context_tags for item in ranked),
        capability_domain=first_text("capability_domain"),
        trigger_patterns=trigger_patterns,
        trigger_type=primary_trigger.trigger_type,
        trigger_keyword=primary_trigger.keyword,
        trigger_prefix=primary_trigger.prefix,
        trigger_to_me=primary_trigger.to_me,
        command_starts=_merge_string_tuples(item.command_starts for item in ranked),
    )


def _merge_string_tuples(values: Any) -> tuple[str, ...]:
    return _string_tuple(tuple(item for group in values for item in group))


def _metadata_source_priority(source: str) -> int:
    normalized = source.casefold().replace("-", "_")
    if normalized in {"to_ai", "toai"}:
        return 20
    if "trigger" in normalized:
        return 10
    return 0


def _capability_source_priority(capability: GScoreCapability) -> int:
    return max(
        (_metadata_source_priority(source) for source in capability.metadata_sources),
        default=0,
    )


def _route_request_id(frame: Any, event_id: str) -> str:
    parts = [
        str(getattr(frame, "bot_id", "") or ""),
        str(getattr(frame, "group_id", "") or ""),
        str(getattr(frame, "user_id", "") or ""),
        event_id,
    ]
    if not event_id:
        parts.extend(
            (
                str(getattr(frame, "turn_generation", 0) or 0),
                f"{float(getattr(frame, 'started_at', 0.0) or 0.0):.9f}",
            )
        )
    stable = "|".join(parts)
    return hashlib.sha256(stable.encode()).hexdigest()[:32]


def _execution_request_id(
    source_request_id: str,
    capability_id: str,
    command_text: str,
) -> str:
    stable = json.dumps(
        {
            "source_request_id": source_request_id,
            "capability_id": capability_id,
            "command_text": command_text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(stable.encode()).hexdigest()[:32]


def _gscore_ws_bot_id() -> str:
    try:
        from nonebot import get_driver

        return str(getattr(get_driver().config, "gsuid_core_botid", "NoneBot2") or "")
    except Exception:
        return "NoneBot2"


def _mixed_tools_allowed(frame: Any) -> bool:
    return bool(
        getattr(frame, "allow_plugin_tools", False)
        and str(getattr(frame, "scenario", "") or "") != "superuser_agent"
    )


def _event_adapter_id(event: Any) -> str:
    module = str(getattr(getattr(event, "__class__", None), "__module__", "") or "")
    class_name = str(getattr(getattr(event, "__class__", None), "__name__", "") or "")
    if ".onebot.v12." in module:
        return "onebot_v12"
    if ".onebot.v11." in module:
        return "onebot"
    if ".qq." in module:
        if class_name in {
            "C2CMessageCreateEvent",
            "GroupAtMessageCreateEvent",
            "GroupMessageCreateEvent",
        }:
            return "qqgroup"
        return "qqguild"
    parts = module.split(".")
    if len(parts) > 3 and parts[:2] == ["nonebot", "adapters"]:
        return parts[2]
    return ""


def _event_sender(event: Any) -> dict[str, Any]:
    sender = getattr(event, "sender", None)
    if sender is None:
        return {}
    if isinstance(sender, dict):
        source = sender
    elif hasattr(sender, "model_dump"):
        source = sender.model_dump()
    elif hasattr(sender, "dict"):
        source = sender.dict()
    else:
        return {}
    allowed = {"user_id", "nickname", "card", "role", "sex", "age"}
    return {key: value for key, value in source.items() if key in allowed}


def _event_user_pm(frame: Any, event: Any) -> int:
    if bool(getattr(frame, "is_superuser", False)):
        return 1
    sender = _event_sender(event)
    role = str(sender.get("role", "") or "").casefold()
    if role == "owner":
        return 2
    if role in {"admin", "administrator"}:
        return 3
    return 6


def _host_command_starts() -> tuple[str, ...]:
    try:
        from nonebot import get_driver

        raw = getattr(get_driver().config, "command_start", ())
    except Exception:
        return ()
    values = raw if isinstance(raw, str | list | tuple | set | frozenset) else ()
    if isinstance(values, str):
        values = (values,)
    return tuple(str(item) for item in values if str(item))


def _strip_host_command_start(text: Any) -> Any:
    value = str(text or "")
    stripped = value.strip()
    for start in _host_command_starts():
        if stripped.startswith(start):
            return stripped[len(start) :]
    return text


def _event_content(frame: Any) -> list[dict[str, Any]]:
    event = getattr(frame, "event", None)
    try:
        message = event.get_message() if event is not None else None
    except Exception:
        message = None
    content: list[dict[str, Any]] = []
    if message is not None:
        for index, segment in enumerate(message):
            segment_type = str(getattr(segment, "type", "") or "")
            data = getattr(segment, "data", {})
            data = data if isinstance(data, dict) else {}
            value: Any = None
            if segment_type == "text":
                value = data.get("text")
                if index in {0, 1}:
                    value = _strip_host_command_start(value)
            elif segment_type == "at":
                value = data.get("qq") or data.get("user_id") or data.get("target")
            elif segment_type in {"image", "record", "video"}:
                value = data.get("url") or data.get("file")
            elif segment_type == "reply":
                value = data.get("id") or data.get("message_id")
            elif segment_type == "file":
                file_name = data.get("name") or data.get("file_name") or "file"
                file_value = data.get("url") or data.get("file") or data.get("id")
                value = f"{file_name}|{file_value}" if file_value else None
            if value is not None and str(value).strip():
                content.append({"type": segment_type, "data": value})
    for item in _reply_image_content(event):
        if item not in content:
            content.append(item)
    if not content:
        text = normalize_message_text(
            str(
                getattr(frame, "route_message", "")
                or getattr(frame, "current_message", "")
            )
        )
        if text:
            content.append({"type": "text", "data": text})
    return content


def _reply_image_content(event: Any) -> list[dict[str, Any]]:
    reply = getattr(event, "reply", None)
    message = getattr(reply, "message", None)
    if message is None:
        return []
    result: list[dict[str, Any]] = []
    for segment in message:
        if str(getattr(segment, "type", "") or "") != "image":
            continue
        data = getattr(segment, "data", {})
        data = data if isinstance(data, dict) else {}
        value = data.get("url") or data.get("file")
        if value is not None and str(value).strip():
            result.append({"type": "image", "data": value})
    return result


def _cached_native_match(
    capabilities: tuple[GScoreCapability, ...],
    message: dict[str, Any],
) -> bool:
    text = "".join(
        str(item.get("data") or "").strip()
        for item in message.get("content", ())
        if isinstance(item, dict) and item.get("type") == "text"
    )
    if not text:
        return False
    return any(
        _command_matches_capability(capability, text, message=message)
        for capability in capabilities
    )


def _command_matches_capability(
    capability: GScoreCapability,
    command_text: str,
    *,
    message: dict[str, Any],
) -> bool:
    text = normalize_message_text(command_text)
    if not text:
        return False
    is_tome = message.get("user_type") == "direct" or any(
        isinstance(item, dict)
        and item.get("type") == "at"
        and str(item.get("data") or "") == str(message.get("bot_self_id") or "")
        for item in message.get("content", ())
    )
    for pattern in capability.trigger_patterns:
        if pattern.to_me and not is_tome:
            continue
        prefix = pattern.prefix
        keyword = pattern.keyword
        head = pattern.command
        trigger_type = pattern.trigger_type
        if trigger_type == "fullmatch" and text == head:
            return True
        if trigger_type == "command" and text.startswith(head):
            return True
        if trigger_type == "prefix" and text.startswith(head) and text != head:
            return True
        if trigger_type == "keyword" and text.startswith(prefix) and keyword in text:
            return True
        if (
            trigger_type == "suffix"
            and text.startswith(prefix)
            and text.endswith(keyword)
            and text != head
        ):
            return True
        if trigger_type == "regex" and text.startswith(prefix):
            try:
                if re.search(keyword, text[len(prefix) :]):
                    return True
            except re.error:
                continue
    return False


_GSCORE_ADAPTER = GScoreAdapter()


__all__ = [
    "GScoreAdapter",
    "GScoreBridgeError",
    "GScoreCapability",
    "GScoreExecutionTool",
    "GScoreRouteResult",
    "GScoreSkillDispatchTool",
    "build_gscore_event_payload",
    "get_gscore_adapter",
]
