"""Structured top-k tool router for ChatInter.

The router is a decision helper only.  It receives a small candidate set from
the retriever, asks the model for a structured choice constrained to those
tool names, then validates the result locally.  It never executes tools.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, create_model

from .command_index import CommandCandidate
from .route_text import normalize_message_text

ToolRouterAction = Literal["select", "none"]

_MAX_ROUTER_CANDIDATES = 12
_EXPANDED_ROUTER_CANDIDATES = 16
_MAX_TEXT_CHARS = 420


class ToolRouterDecision(BaseModel):
    """Stable router output after local validation."""

    action: ToolRouterAction = Field(default="none")
    tool_name: str = ""
    command_id: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class ToolRouterSelection(BaseModel):
    """One selected command for one task in a batch router call."""

    task_id: str = ""
    tool_name: str = ""
    command_id: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ToolRouterBatchDecision(BaseModel):
    """Stable batch router output after local validation."""

    action: ToolRouterAction = Field(default="none")
    selections: list[ToolRouterSelection] = Field(default_factory=list)
    reason: str = ""


class ToolRouter:
    """LLM-backed router constrained to a turn-local top-k enum."""

    def __init__(
        self,
        *,
        trace_id: str,
        model_name: str | None,
        generation_config: Any,
        timeout: float,
        usage_callback: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.ai: Any = _create_ai(session_id=f"chatinter-tool-router:{trace_id}")
        self.model_name = model_name
        self.generation_config = generation_config
        self.timeout = max(4.0, min(float(timeout or 12.0), 18.0))
        self.usage_callback = usage_callback

    async def route(
        self,
        *,
        message_text: str,
        candidates: list[CommandCandidate],
        tool_names_by_command_id: dict[str, str] | None = None,
    ) -> ToolRouterDecision:
        """Deprecated compatibility path; group plugin routing uses route_tasks()."""

        candidate_limit = _router_candidate_limit(candidates)
        options = build_tool_router_options(
            candidates,
            tool_names_by_command_id=tool_names_by_command_id,
            limit=candidate_limit,
        )
        if not options:
            return ToolRouterDecision(
                action="none",
                reason="no_candidate_options",
            )

        while options:
            response_model = build_tool_router_response_model(
                [option["tool_name"] for option in options],
            )
            payload: dict[str, Any] = {
                "message": normalize_message_text(message_text),
                "candidate_options": options,
            }
            payload_json = json.dumps(payload, ensure_ascii=False)
            if _router_request_fits(
                payload_json,
                response_model=response_model,
                instruction=_TOOL_ROUTER_INSTRUCTION,
                model_name=self.model_name,
            ):
                break
            options.pop()
        if not options:
            return ToolRouterDecision(
                action="none",
                reason="router_context_exhausted",
            )
        _log_router_payload_size(
            "single",
            payload_json,
            option_count=len(options),
            task_count=1,
            candidate_limit=candidate_limit,
        )
        try:
            result = await self.ai.generate_structured(
                payload_json,
                response_model,
                model=self.model_name,
                config=self.generation_config,
                instruction=_TOOL_ROUTER_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                usage_callback=self.usage_callback,
            )
            return normalize_tool_router_result(
                result,
                candidates=candidates,
                tool_names_by_command_id=tool_names_by_command_id,
                limit=len(options),
            )
        except Exception as exc:
            _log_warning(f"[ChatInter] tool router failed: {exc}")
            return fallback_tool_router_decision(candidates)

    async def route_tasks(
        self,
        *,
        tasks: list[dict[str, Any]],
        candidates: list[CommandCandidate],
        tool_names_by_command_id: dict[str, str] | None = None,
        router_context: dict[str, Any] | None = None,
    ) -> ToolRouterBatchDecision:
        context = _router_context(router_context)
        candidate_limit = _router_candidate_limit(candidates, router_context=context)
        options = build_tool_router_options(
            candidates,
            tool_names_by_command_id=tool_names_by_command_id,
            limit=candidate_limit,
        )
        task_options = _task_options(tasks)
        if not options or not task_options:
            return ToolRouterBatchDecision(
                action="none",
                reason="no_task_or_candidate_options",
            )

        while options:
            response_model = build_tool_router_batch_response_model(
                [option["tool_name"] for option in options],
                [task["task_id"] for task in task_options],
            )
            payload = {
                "tasks": task_options,
                "candidate_options": options,
            }
            if context:
                payload["context"] = context
            payload_json = json.dumps(payload, ensure_ascii=False)
            if _router_request_fits(
                payload_json,
                response_model=response_model,
                instruction=_TOOL_ROUTER_BATCH_INSTRUCTION,
                model_name=self.model_name,
            ):
                break
            options.pop()
        if not options:
            return ToolRouterBatchDecision(
                action="none",
                reason="router_context_exhausted",
            )
        _log_router_payload_size(
            "batch",
            payload_json,
            option_count=len(options),
            task_count=len(task_options),
            candidate_limit=candidate_limit,
        )
        try:
            result = await self.ai.generate_structured(
                payload_json,
                response_model,
                model=self.model_name,
                config=self.generation_config,
                instruction=_TOOL_ROUTER_BATCH_INSTRUCTION,
                timeout=self.timeout,
                max_validation_retries=0,
                usage_callback=self.usage_callback,
            )
            return normalize_tool_router_batch_result(
                result,
                tasks=task_options,
                candidates=candidates,
                tool_names_by_command_id=tool_names_by_command_id,
                limit=len(options),
            )
        except Exception as exc:
            _log_warning(f"[ChatInter] batch tool router failed: {exc}")
            return ToolRouterBatchDecision(
                action="none",
                reason="batch_router_failed_no_selection",
            )


def build_tool_router_options(
    candidates: list[CommandCandidate],
    *,
    tool_names_by_command_id: dict[str, str] | None = None,
    limit: int = _MAX_ROUTER_CANDIDATES,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen_tools: set[str] = set()
    for candidate in candidates[: max(1, int(limit or 1))]:
        command_id = normalize_message_text(candidate.schema.command_id)
        if not command_id:
            continue
        tool_name = _tool_name_for_candidate(
            candidate,
            tool_names_by_command_id=tool_names_by_command_id,
        )
        if not tool_name or tool_name in seen_tools:
            continue
        seen_tools.add(tool_name)
        options.append(_candidate_option(candidate, tool_name=tool_name))
    return options


def build_tool_router_response_model(tool_names: list[str]) -> type[BaseModel]:
    """Create a structured response schema with tool_name constrained to top-k."""

    enum_values = [
        normalize_message_text(name)
        for name in dict.fromkeys(tool_names)
        if normalize_message_text(name)
    ]
    return create_model(
        "ToolRouterResponse",
        action=(
            str,
            Field(
                default="none",
                description="select/none",
                json_schema_extra=cast(Any, {"enum": ["select", "none"]}),
            ),
        ),
        tool_name=(
            str,
            Field(
                default="",
                description="选择的工具名；select 时必须来自 enum",
                json_schema_extra=cast(Any, {"enum": enum_values}),
            ),
        ),
        arguments=(
            dict[str, Any],
            Field(default_factory=dict, description="按候选 schema 抽取的参数"),
        ),
        confidence=(float, Field(default=0.0, ge=0.0, le=1.0)),
    )


def build_tool_router_batch_response_model(
    tool_names: list[str],
    task_ids: list[str],
) -> type[BaseModel]:
    """Create a structured response schema for one-call multi-task selection."""

    tool_enum = [
        normalize_message_text(name)
        for name in dict.fromkeys(tool_names)
        if normalize_message_text(name)
    ]
    task_enum = [
        normalize_message_text(task_id)
        for task_id in dict.fromkeys(task_ids)
        if normalize_message_text(task_id)
    ]
    selection_model = create_model(
        "ToolRouterBatchSelection",
        task_id=(
            str,
            Field(
                default="",
                description="对应 tasks 中的 task_id",
                json_schema_extra=cast(Any, {"enum": task_enum}),
            ),
        ),
        tool_name=(
            str,
            Field(
                default="",
                description="选择的工具名；必须来自 candidate_options",
                json_schema_extra=cast(Any, {"enum": tool_enum}),
            ),
        ),
        arguments=(
            dict[str, Any],
            Field(default_factory=dict, description="按候选 schema 抽取的参数"),
        ),
        confidence=(float, Field(default=0.0, ge=0.0, le=1.0)),
    )
    selection_list_type = list[selection_model]  # type: ignore[valid-type]
    return create_model(
        "ToolRouterBatchResponse",
        action=(
            str,
            Field(
                default="none",
                description="select/none",
                json_schema_extra=cast(Any, {"enum": ["select", "none"]}),
            ),
        ),
        selections=(
            selection_list_type,
            Field(default_factory=list, description="按 tasks 顺序选择 0 个或多个工具"),
        ),
    )


def normalize_tool_router_result(
    result: Any,
    *,
    candidates: list[CommandCandidate],
    tool_names_by_command_id: dict[str, str] | None = None,
    limit: int = _MAX_ROUTER_CANDIDATES,
) -> ToolRouterDecision:
    options = build_tool_router_options(
        candidates,
        tool_names_by_command_id=tool_names_by_command_id,
        limit=limit,
    )
    allowed_tools = {option["tool_name"] for option in options}
    command_by_tool = {
        option["tool_name"]: normalize_message_text(str(option["command_id"]))
        for option in options
    }

    action = _normalize_action(getattr(result, "action", "none"))
    tool_name = normalize_message_text(str(getattr(result, "tool_name", "") or ""))
    if action == "select" and tool_name not in allowed_tools:
        action = "none"
        tool_name = ""
    if action == "none":
        tool_name = ""

    arguments = getattr(result, "arguments", {}) or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolRouterDecision(
        action=action,
        tool_name=tool_name,
        command_id=command_by_tool.get(tool_name, ""),
        arguments={
            normalize_message_text(str(key)): value
            for key, value in arguments.items()
            if normalize_message_text(str(key))
        },
        confidence=_coerce_confidence(getattr(result, "confidence", 0.0)),
        reason=_clip(getattr(result, "reason", ""), limit=240),
    )


def normalize_tool_router_batch_result(
    result: Any,
    *,
    tasks: list[dict[str, Any]],
    candidates: list[CommandCandidate],
    tool_names_by_command_id: dict[str, str] | None = None,
    limit: int = _MAX_ROUTER_CANDIDATES,
) -> ToolRouterBatchDecision:
    options = build_tool_router_options(
        candidates,
        tool_names_by_command_id=tool_names_by_command_id,
        limit=limit,
    )
    allowed_tools = {option["tool_name"] for option in options}
    command_by_tool = {
        option["tool_name"]: normalize_message_text(str(option["command_id"]))
        for option in options
    }
    task_ids = {
        normalize_message_text(str(task.get("task_id", "") or "")) for task in tasks
    }
    task_ids.discard("")

    action = _normalize_action(getattr(result, "action", "none"))
    raw_selections = getattr(result, "selections", []) or []
    if not isinstance(raw_selections, list | tuple):
        raw_selections = []

    selections: list[ToolRouterSelection] = []
    seen_tasks: set[str] = set()
    for item in raw_selections:
        task_id = normalize_message_text(str(getattr(item, "task_id", "") or ""))
        tool_name = normalize_message_text(str(getattr(item, "tool_name", "") or ""))
        if not task_id or task_id not in task_ids or task_id in seen_tasks:
            continue
        if not tool_name or tool_name not in allowed_tools:
            continue
        arguments = getattr(item, "arguments", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        seen_tasks.add(task_id)
        selections.append(
            ToolRouterSelection(
                task_id=task_id,
                tool_name=tool_name,
                command_id=command_by_tool.get(tool_name, ""),
                arguments={
                    normalize_message_text(str(key)): value
                    for key, value in arguments.items()
                    if normalize_message_text(str(key))
                },
                confidence=_coerce_confidence(getattr(item, "confidence", 0.0)),
            )
        )

    if action != "select" or not selections:
        return ToolRouterBatchDecision(
            action="none",
            selections=[],
            reason=_clip(getattr(result, "reason", ""), limit=240)
            or "batch_router:no_selection",
        )
    return ToolRouterBatchDecision(
        action="select",
        selections=selections,
        reason=_clip(getattr(result, "reason", ""), limit=240),
    )


def fallback_tool_router_decision(
    candidates: list[CommandCandidate],
) -> ToolRouterDecision:
    if not candidates:
        return ToolRouterDecision(action="none", reason="router_failed_no_candidate")
    exact = [candidate for candidate in candidates if candidate.exact_protected]
    if len(exact) == 1:
        tool_name = _tool_name_for_candidate(exact[0], tool_names_by_command_id=None)
        return ToolRouterDecision(
            action="select",
            tool_name=tool_name,
            command_id=normalize_message_text(exact[0].schema.command_id),
            confidence=0.72,
            reason="router_failed_exact_single_fallback",
        )
    return ToolRouterDecision(
        action="none",
        reason="router_failed_no_confident_selection",
    )


def _task_options(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, task in enumerate(tasks, 1):
        task_id = normalize_message_text(str(task.get("task_id", "") or ""))
        text = _clip(task.get("text", ""), limit=220)
        if not task_id or not text or task_id in seen:
            continue
        seen.add(task_id)
        options.append(
            {
                "task_id": task_id,
                "text": text,
                "order": int(task.get("order", index) or index),
            }
        )
        if len(options) >= 8:
            break
    return options


def _router_context(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("has_reply", "has_image", "has_at"):
        if key in value:
            result[key] = bool(value[key])
    avatar_inputs = ["sender_avatar"]
    if result.get("has_at"):
        avatar_inputs.append("at_target_avatar")
    if result.get("has_reply"):
        avatar_inputs.append("reply_target_avatar")
    if len(avatar_inputs) > 1:
        result["avatar_inputs"] = avatar_inputs
        result["avatar_input_count"] = len(avatar_inputs)
    if "reply_image_count" in value:
        try:
            result["reply_image_count"] = max(0, int(value.get("reply_image_count", 0)))
        except (TypeError, ValueError):
            pass
    target_resolution = _clip(value.get("target_resolution", ""), limit=40)
    if target_resolution:
        result["target_resolution"] = target_resolution
    return result


def _router_candidate_limit(
    candidates: list[CommandCandidate],
    *,
    router_context: dict[str, Any] | None = None,
) -> int:
    if len(candidates) <= _MAX_ROUTER_CANDIDATES:
        return _MAX_ROUTER_CANDIDATES
    context = _router_context(router_context)
    if (
        context.get("has_reply")
        or context.get("has_image")
        or int(context.get("reply_image_count", 0) or 0) > 0
    ):
        return _EXPANDED_ROUTER_CANDIDATES

    boundary_score = _candidate_score(candidates[_MAX_ROUTER_CANDIDATES - 1])
    next_score = _candidate_score(candidates[_MAX_ROUTER_CANDIDATES])
    if boundary_score > 0 and (
        next_score >= boundary_score * 0.9
        or abs(boundary_score - next_score) <= 20.0
    ):
        return _EXPANDED_ROUTER_CANDIDATES
    return _MAX_ROUTER_CANDIDATES


def _router_request_fits(
    payload_json: str,
    *,
    response_model: type[BaseModel],
    instruction: str,
    model_name: str | None,
) -> bool:
    from .config import get_agent_max_output_tokens
    from .turn_runtime import estimate_text_tokens

    window = _plugin_context_window(model_name)
    limit = max(window - get_agent_max_output_tokens("plugin"), 1)
    schema = json.dumps(
        response_model.model_json_schema(),
        ensure_ascii=False,
        default=str,
    )
    return estimate_text_tokens(f"{instruction}\n{payload_json}\n{schema}") <= limit


def _plugin_context_window(model_name: str | None) -> int:
    from .config import resolve_agent_context_window_tokens

    return resolve_agent_context_window_tokens("plugin", model_name)


def _candidate_score(candidate: CommandCandidate) -> float:
    try:
        return max(float(candidate.score or 0.0), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_option(
    candidate: CommandCandidate,
    *,
    tool_name: str,
) -> dict[str, Any]:
    schema = candidate.schema
    tool = candidate.tool
    description = _clip(schema.description, limit=120)
    payload_policy = normalize_message_text(schema.payload_policy)
    required_slots = _required_slots(schema)
    optional_slots = _optional_text_slots(schema)
    payload = {
        "tool_name": tool_name,
        "command_id": normalize_message_text(schema.command_id),
        "head": _clip(schema.head),
        "aliases": [_clip(alias, limit=60) for alias in list(schema.aliases)[:4]],
        "description": description,
        "required_slots": required_slots,
    }
    if payload_policy not in {"", "none"}:
        payload["payload_policy"] = payload_policy
    target_requirement = normalize_message_text(
        getattr(schema, "target_requirement", "")
        or getattr(tool, "target_requirement", "")
    )
    if target_requirement not in {"", "none"}:
        payload["target_requirement"] = target_requirement
    if payload_policy in {"slots", "text"} and optional_slots:
        payload["optional_slots"] = optional_slots
    evidence = _metadata_evidence(candidate)
    if evidence:
        payload["evidence"] = evidence
    render = _clip(getattr(tool, "render", ""), limit=120)
    if required_slots or (
        payload_policy != "none" and not _has_render_placeholder(render)
    ):
        payload["render"] = render
    return {key: value for key, value in payload.items() if value not in ("", [], {})}


def _has_render_placeholder(text: str) -> bool:
    return "{" in text and "}" in text


def _metadata_evidence(candidate: CommandCandidate) -> list[str]:
    schema = candidate.schema
    tool = candidate.tool
    evidence: list[str] = []
    description = normalize_message_text(schema.description)

    def add(label: str, value: Any, *, limit: int = 64) -> None:
        if len(evidence) >= 3:
            return
        text = _clip(value, limit=limit)
        if text and f"{label}: {text}" not in evidence:
            evidence.append(f"{label}: {text}")

    add("head", schema.head)
    aliases = [
        _clip(alias, limit=24)
        for alias in list(getattr(schema, "aliases", []) or [])[:3]
        if _clip(alias, limit=24) and _clip(alias, limit=24) != _clip(schema.head)
    ]
    if aliases:
        add("alias", "/".join(aliases), limit=72)
    shortcuts: list[str] = []
    for item in list(getattr(schema, "shortcut_renders", []) or [])[:3]:
        if not isinstance(item, dict):
            continue
        alias = _clip(item.get("alias"), limit=24)
        if alias:
            shortcuts.append(alias)
    if shortcuts:
        add("shortcut", "/".join(shortcuts), limit=72)
    required_slot_names = [
        item["name"] for item in _required_slots(schema) if item.get("name")
    ]
    if required_slot_names:
        add("required_slot", "/".join(required_slot_names), limit=72)
    optional_slot_names = [
        item["name"] for item in _optional_text_slots(schema) if item.get("name")
    ]
    if optional_slot_names:
        add("optional_slot", "/".join(optional_slot_names), limit=72)
    usage = getattr(tool, "usage", "") if tool is not None else ""
    if usage and not _description_covers_metadata(description, usage):
        add("usage", usage, limit=72)
    examples = list(getattr(tool, "examples", []) or []) if tool is not None else []
    if examples and not _description_has_examples(description, examples):
        add("example", " / ".join(str(item) for item in examples[:2]), limit=72)
    return evidence[:3]


def _description_covers_metadata(description: str, value: Any) -> bool:
    description_text = normalize_message_text(description).casefold()
    value_text = normalize_message_text(str(value or "")).casefold()
    return bool(value_text and value_text in description_text)


def _description_has_examples(description: str, examples: list[Any]) -> bool:
    description_text = normalize_message_text(description).casefold()
    if "示例" in description_text or "example" in description_text:
        return True
    return any(
        _description_covers_metadata(description_text, item) for item in examples
    )


def _required_slots(schema: Any) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for slot in list(getattr(schema, "slots", []) or []):
        if not bool(getattr(slot, "required", False)):
            continue
        payload = {
            "name": _clip(getattr(slot, "name", ""), limit=48),
            "type": _clip(getattr(slot, "type", ""), limit=16),
            "aliases": [
                _clip(alias, limit=16)
                for alias in list(getattr(slot, "aliases", []) or [])[:2]
                if _clip(alias, limit=16)
            ],
            "description": _clip(getattr(slot, "description", ""), limit=40),
        }
        slots.append(
            {key: value for key, value in payload.items() if value not in ("", [])}
        )
        if len(slots) >= 4:
            break
    return slots


def _optional_text_slots(schema: Any) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for slot in list(getattr(schema, "slots", []) or []):
        if bool(getattr(slot, "required", False)):
            continue
        if normalize_message_text(getattr(slot, "type", "")) != "text":
            continue
        name = _clip(getattr(slot, "name", ""), limit=32)
        if not _is_clean_optional_slot_name(name):
            continue
        payload = {
            "name": name,
            "type": "text",
            "aliases": [
                _clip(alias, limit=16)
                for alias in list(getattr(slot, "aliases", []) or [])[:2]
                if _is_clean_optional_slot_name(_clip(alias, limit=16))
            ],
            "description": _clip(getattr(slot, "description", ""), limit=40),
        }
        slots.append(
            {key: value for key, value in payload.items() if value not in ("", [])}
        )
        if len(slots) >= 2:
            break
    return slots


def _is_clean_optional_slot_name(name: str) -> bool:
    text = normalize_message_text(name)
    if not text or len(text) > 24:
        return False
    if any(mark in text for mark in " ，,。；;：:/\\()（）[]【】"):
        return False
    return not any(term in text for term in ("默认", "无参数", "分隔", "参数"))


def _tool_name_for_candidate(
    candidate: CommandCandidate,
    *,
    tool_names_by_command_id: dict[str, str] | None,
) -> str:
    command_id = normalize_message_text(candidate.schema.command_id)
    mapped = normalize_message_text(
        str((tool_names_by_command_id or {}).get(command_id, "") or "")
    )
    if mapped:
        return mapped
    return f"tool_{command_id}" if command_id else ""


def _normalize_action(value: Any) -> ToolRouterAction:
    normalized = normalize_message_text(str(value or "")).lower()
    if normalized in {"select", "none"}:
        return cast(ToolRouterAction, normalized)
    return "none"


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _clip(value: Any, *, limit: int = _MAX_TEXT_CHARS) -> str:
    text = normalize_message_text(str(value or ""))
    return text[: max(1, int(limit or _MAX_TEXT_CHARS))]


def _create_ai(*, session_id: str) -> Any:
    from .llm_compat import AI

    return AI(session_id=session_id)


def _log_warning(message: str) -> None:
    try:
        from zhenxun.services import logger

        logger.warning(message)
    except Exception:
        return


def _log_debug(message: str) -> None:
    try:
        from zhenxun.services import logger

        logger.debug(message)
    except Exception:
        return


def _log_router_payload_size(
    kind: str,
    payload_json: str,
    *,
    option_count: int,
    task_count: int,
    candidate_limit: int,
) -> None:
    _log_debug(
        "[ChatInter] tool router payload "
        f"kind={kind} chars={len(payload_json)} "
        f"options={option_count} tasks={task_count} limit={candidate_limit}"
    )


_TOOL_ROUTER_INSTRUCTION = """
你是 ChatInter 的结构化工具路由器。
你只在给定 candidate_options 中选择工具，不执行工具，不发最终回答。

规则：
- tool_name 必须严格来自 candidate_options 的 tool_name enum。
- 如果没有明确匹配，action=none。
- 只有当前消息明确需要某个候选工具真实执行时，action=select。
- 不要选择 enum 外工具名，不要编造工具。
- arguments 只放能从用户消息中直接确定的参数；不确定就不要填。
- 对 required slot 缺失且无法从上下文确定时，action=none。

只返回 JSON：
{
  "action": "select",
  "tool_name": "",
  "arguments": {},
  "confidence": 0.0
}
""".strip()


_TOOL_ROUTER_BATCH_INSTRUCTION = """
你是插件命令选择器，只选择，不执行，不回复用户。
只允许 selection.task_id 来自 tasks，tool_name 来自 candidate_options。
上下文中的 avatar_inputs / avatar_input_count 可计入头像类候选所需图片。
没有明确匹配，或参数/目标/图片/回复上下文不确定，就不要选择。
arguments 只填能从用户消息和上下文直接确定的值。只返回 JSON。
""".strip()


__all__ = [
    "ToolRouter",
    "ToolRouterAction",
    "ToolRouterBatchDecision",
    "ToolRouterDecision",
    "ToolRouterSelection",
    "build_tool_router_batch_response_model",
    "build_tool_router_options",
    "build_tool_router_response_model",
    "fallback_tool_router_decision",
    "normalize_tool_router_batch_result",
    "normalize_tool_router_result",
]
