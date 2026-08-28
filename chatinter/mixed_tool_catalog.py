"""Provider-specific plugin tool assembly for the unified chat agent."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import TYPE_CHECKING, Any

from .candidate_exposure import CandidateExposureKey, CandidateExposureLedger
from .command_index import CommandCandidate
from .meta_tools import (
    _candidate_from_snapshot,
    render_command_candidate_context,
)
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .native_command_tools import build_native_command_tools
from .native_executor import NativeCommandExecutionContext
from .plugin_skill_index import PluginSkill, PluginSkillIndex
from .route_text import normalize_message_text
from .skill_dispatch_tools import (
    PluginSkillDispatchTool,
    build_plugin_skill_dispatch_tools,
)
from .skill_overflow import OVERFLOW_SKILL_TOOL_NAME, SkillOverflowTool
from .turn_runtime import estimate_text_tokens
from .web_access import (
    candidate_web_search_kind,
    tools_for_web_candidate,
)

if TYPE_CHECKING:
    from .host_llm import HostModelCandidate
    from .provider_capability import ProviderCapabilityAdapter

_DETAIL_PROTOCOL_MARGIN_TOKENS = 4_096
_AMBIGUITY_RESULT_TOKEN_LIMIT = 16_000
_CANDIDATE_CONTEXT_TOKEN_BASELINE = 4_096


@dataclass(frozen=True, slots=True)
class MixedToolCatalog:
    skill_index: PluginSkillIndex | None
    known_commands: tuple[CommandToolSnapshot, ...]
    available_commands: tuple[CommandToolSnapshot, ...]
    initial_candidates: tuple[CommandCandidate, ...]
    knowledge_base: PluginKnowledgeBase | None
    session_id: str | None
    command_context: NativeCommandExecutionContext | None
    exposure_ledger: CandidateExposureLedger = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class MixedToolView:
    tools: dict[str, Any]
    command_candidate_text: str
    native_command_ids: tuple[str, ...]
    indexed_command_ids: tuple[str, ...]
    skill_tool_names: tuple[str, ...] = ()
    tool_priority_names: tuple[str, ...] = ()
    required_tool_names: tuple[str, ...] = ()
    native_tool_bindings: tuple[tuple[str, str], ...] = ()
    indexed_tool_bindings: tuple[tuple[str, str], ...] = ()
    initial_candidates: tuple[CommandCandidate, ...] = ()
    strict_candidate_exposures: tuple[
        tuple[CandidateExposureKey, str, str], ...
    ] = ()
    base_candidate_contexts: tuple[tuple[str, str], ...] = ()
    candidate_token_budget: int = 0
    schema_tokens: int = 0
    schema_omitted_names: tuple[str, ...] = ()
    skill_delegates: tuple[tuple[str, Any], ...] = field(
        default=(),
        compare=False,
        repr=False,
    )
    indexed_delegate_bindings: tuple[tuple[str, str], ...] = ()
    native_fallback_delegates: tuple[tuple[str, Any], ...] = field(
        default=(),
        compare=False,
        repr=False,
    )
    native_fallback_bindings: tuple[
        tuple[str, str, str, CandidateExposureKey], ...
    ] = ()
    native_fallback_candidates: tuple[CommandCandidate, ...] = ()
    exposure_ledger: CandidateExposureLedger | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    overflow_token_budget: int = 0
    overflow_char_budget: int = 0
    plugin_capacity_degraded: bool = False


@dataclass(frozen=True, slots=True)
class ToolSchemaSelection:
    tools: dict[str, Any]
    schema_tokens: int
    omitted_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateContextProjection:
    text: str
    displayed_command_ids: tuple[str, ...] = ()


def build_mixed_tool_catalog(
    *,
    skill_index: PluginSkillIndex | None,
    known_commands: list[CommandToolSnapshot],
    available_commands: list[CommandToolSnapshot],
    initial_candidates: list[CommandCandidate],
    knowledge_base: PluginKnowledgeBase | None,
    session_id: str | None,
    command_context: NativeCommandExecutionContext | None,
    exposure_ledger: CandidateExposureLedger | None = None,
) -> MixedToolCatalog:
    return MixedToolCatalog(
        skill_index=skill_index,
        known_commands=tuple(_stable_snapshots(known_commands)),
        available_commands=tuple(_stable_snapshots(available_commands)),
        initial_candidates=tuple(initial_candidates),
        knowledge_base=knowledge_base,
        session_id=session_id,
        command_context=command_context,
        exposure_ledger=exposure_ledger or CandidateExposureLedger(),
    )


def assemble_candidate_tool_view(
    catalog: MixedToolCatalog,
    *,
    adapter: ProviderCapabilityAdapter,
    candidate: HostModelCandidate,
    context_window_tokens: int,
    output_reserve_tokens: int,
    base_prompt_tokens: int,
    base_tools: dict[str, Any] | None = None,
) -> MixedToolView:
    profile = adapter.profile
    if not profile.supports_tools or adapter.max_tools <= 0:
        return MixedToolView(
            tools={},
            command_candidate_text="",
            native_command_ids=(),
            indexed_command_ids=(),
            skill_tool_names=(),
            plugin_capacity_degraded=True,
        )

    skill_base_count = sum(
        _is_skill_delegate(tool) for tool in (base_tools or {}).values()
    )
    non_skill_base_count = sum(
        not _is_skill_delegate(tool) for tool in (base_tools or {}).values()
    )
    semantic = sorted(
        (
            snapshot
            for snapshot in catalog.available_commands
            if _is_semantic_snapshot(snapshot)
        ),
        key=_native_snapshot_sort_key,
    )
    semantic_ids = {
        normalize_message_text(snapshot.command_id) for snapshot in semantic
    }
    indexed_skill_modules = {
        _module_key(skill.plugin_module)
        for skill in (catalog.skill_index.skills if catalog.skill_index else ())
    }
    nonsemantic_skill_count = len(
        {
            _module_key(snapshot.plugin_module)
            for snapshot in catalog.available_commands
            if _module_key(snapshot.plugin_module) in indexed_skill_modules
            and normalize_message_text(snapshot.command_id) not in semantic_ids
        }
    )
    plugin_tool_demand = skill_base_count + len(semantic) + nonsemantic_skill_count
    web_available = (
        candidate_web_search_kind(
            candidate,
            scope="chat",
            has_client_tools=True,
        )
        is not None
    )
    reserve_web = int(
        web_available and not (adapter.max_tools == 1 and plugin_tool_demand > 0)
    )
    client_capacity = max(adapter.max_tools - reserve_web, 0)
    plugin_capacity_without_overflow = max(
        client_capacity - non_skill_base_count,
        0,
    )
    overflow_needed = (
        client_capacity > 0
        and plugin_tool_demand > plugin_capacity_without_overflow
    )
    regular_capacity = max(client_capacity - int(overflow_needed), 0)
    base_items = _select_base_tool_items(base_tools or {}, regular_capacity)
    plugin_capacity = max(regular_capacity - len(base_items), 0)
    selected_native = (
        semantic if nonsemantic_skill_count + len(semantic) <= plugin_capacity else []
    )
    native_ids = {
        normalize_message_text(snapshot.command_id) for snapshot in selected_native
    }
    native_semantic_names: dict[str, set[str]] = {}
    for snapshot in selected_native:
        semantic_name = normalize_message_text(
            str(snapshot.meta.get("semantic_tool_name") or "")
        ).casefold()
        if semantic_name:
            native_semantic_names.setdefault(
                _module_key(snapshot.plugin_module),
                set(),
            ).add(semantic_name)
    indexed_available = [
        snapshot
        for snapshot in catalog.available_commands
        if normalize_message_text(snapshot.command_id) not in native_ids
    ]
    dispatch_skills = (
        _available_dispatch_skills(
            catalog.skill_index,
            indexed_available,
            known_commands=list(catalog.known_commands),
            native_semantic_names=native_semantic_names,
        )
        if catalog.skill_index is not None
        else ()
    )
    detail_token_budget = max(
        int(context_window_tokens)
        - max(int(output_reserve_tokens), 0)
        - max(int(base_prompt_tokens), 0)
        - _DETAIL_PROTOCOL_MARGIN_TOKENS,
        1,
    )
    ambiguity_token_budget = min(
        max(detail_token_budget // 2, 1),
        _AMBIGUITY_RESULT_TOKEN_LIMIT,
    )
    base_items = [
        (
            name,
            _bind_tool_result_budget(
                tool,
                token_budget=ambiguity_token_budget,
                char_budget=_adapter_tool_result_char_budget(adapter),
            ),
        )
        for name, tool in base_items
    ]
    dispatch_tools = (
        build_plugin_skill_dispatch_tools(
            skills=dispatch_skills,
            known_commands=list(catalog.known_commands),
            available_commands=indexed_available,
            knowledge_base=catalog.knowledge_base,
            session_id=catalog.session_id,
            command_context=catalog.command_context,
            exposure_ledger=catalog.exposure_ledger,
            revision=(catalog.skill_index.fingerprint if catalog.skill_index else ""),
            ambiguity_token_budget=ambiguity_token_budget,
            result_char_budget=_adapter_tool_result_char_budget(adapter),
        )
        if dispatch_skills
        and catalog.knowledge_base is not None
        and catalog.command_context is not None
        else {}
    )
    native_fallback_skills = (
        _available_dispatch_skills(
            catalog.skill_index,
            semantic,
            known_commands=list(catalog.known_commands),
            native_semantic_names={},
        )
        if selected_native
        and catalog.skill_index is not None
        and catalog.knowledge_base is not None
        and catalog.command_context is not None
        else ()
    )
    native_fallback_tools: dict[str, PluginSkillDispatchTool] = {}
    for skill in native_fallback_skills:
        fallback_name = _semantic_fallback_delegate_name(skill.plugin_module)
        native_fallback_tools[fallback_name] = PluginSkillDispatchTool(
            skill,
            known_commands=list(catalog.known_commands),
            available_commands=semantic,
            knowledge_base=catalog.knowledge_base,
            session_id=catalog.session_id,
            command_context=catalog.command_context,
            exposure_ledger=catalog.exposure_ledger,
            revision=(catalog.skill_index.fingerprint if catalog.skill_index else ""),
            ambiguity_token_budget=ambiguity_token_budget,
            result_char_budget=_adapter_tool_result_char_budget(adapter),
            tool_name=fallback_name,
        )
    dispatch_capacity = max(plugin_capacity - len(selected_native), 0)
    dispatch_items = _select_dispatch_items(
        dispatch_tools,
        capacity=dispatch_capacity,
    )
    all_skill_delegates = {
        name: _bind_tool_result_budget(
            tool,
            token_budget=ambiguity_token_budget,
            char_budget=_adapter_tool_result_char_budget(adapter),
        )
        for name, tool in {
            **{
                name: tool
                for name, tool in (base_tools or {}).items()
                if _is_skill_delegate(tool)
            },
            **dispatch_tools,
        }.items()
    }
    selected_skill_delegate_names = {
        name for name, tool in base_items if _is_skill_delegate(tool)
    } | {name for name, _tool in dispatch_items}
    overflow_delegates = {
        name: tool
        for name, tool in all_skill_delegates.items()
        if name not in selected_skill_delegate_names
    }
    overflow_tool = None
    if overflow_needed and overflow_delegates:
        overflow_tool = SkillOverflowTool(
            overflow_delegates,
            result_token_budget=ambiguity_token_budget,
            result_char_budget=_adapter_tool_result_char_budget(adapter),
            exposure_ledger=catalog.exposure_ledger,
        )
    capacity_omitted_names = _stable_unique(
        (
            *(name for name in (base_tools or {}) if name not in dict(base_items)),
            *(name for name in dispatch_tools if name not in dict(dispatch_items)),
        )
    )
    selected_dispatch_names = {name for name, _tool in dispatch_items}
    overflow_dispatch_names = set(overflow_delegates) & set(dispatch_tools)
    skill_tools_by_command_id = {
        normalize_message_text(command_id): (
            name
            if name in selected_dispatch_names
            else OVERFLOW_SKILL_TOOL_NAME
        )
        for name, tool in dispatch_tools.items()
        if name in selected_dispatch_names or name in overflow_dispatch_names
        for command_id in tool.skill.command_ids
    }
    exposed_skill_ids = set(skill_tools_by_command_id)
    delegate_by_command_id = {
        normalize_message_text(command_id): tool
        for tool in dispatch_tools.values()
        for command_id in tool.skill.command_ids
    }
    native_candidates = [_candidate_from_snapshot(item) for item in selected_native]
    native_tools = build_native_command_tools(
        native_candidates,
        execution_context=catalog.command_context,
    )
    combined = [
        *base_items,
        *((tool.binding.tool_name, tool) for tool in native_tools),
        *dispatch_items,
        *(
            ((OVERFLOW_SKILL_TOOL_NAME, overflow_tool),)
            if overflow_tool
            else ()
        ),
    ]
    combined_names = [name for name, _tool in combined]
    if len(combined_names) != len(set(combined_names)):
        raise ValueError("duplicate mixed tool name")
    tools = dict(sorted(combined, key=lambda item: item[0]))

    base_candidate_context_list: list[tuple[str, str]] = []
    for name, tool in base_items:
        text = str(getattr(tool, "candidate_context", "") or "").strip()
        if text:
            base_candidate_context_list.append((name, text))
    base_candidate_contexts = tuple(base_candidate_context_list)
    candidate_available = max(detail_token_budget - ambiguity_token_budget, 1)
    candidate_token_budget = min(
        candidate_available,
        _CANDIDATE_CONTEXT_TOKEN_BASELINE,
    )
    initial_candidates = [
        item
        for item in catalog.initial_candidates
        if normalize_message_text(item.schema.command_id) in exposed_skill_ids
    ]
    strict_mode_by_command_id = {
        normalize_message_text(item.schema.command_id): normalize_message_text(
            item.strict_identity_mode
        )
        for item in initial_candidates
    }
    projection = _render_bounded_candidate_context(
        base_candidate_contexts=base_candidate_contexts,
        initial_candidates=tuple(initial_candidates),
        skill_tools_by_command_id=skill_tools_by_command_id,
        token_budget=candidate_token_budget,
    )
    with_web = (
        tools_for_web_candidate(
            tools,
            candidate=candidate,
            scope="chat",
        )
        if reserve_web
        else tools
    )
    final_tools = dict(with_web or {})
    final_skill_names = tuple(
        name for name, tool in final_tools.items() if _is_skill_delegate(tool)
    )
    base_names = tuple(name for name, _tool in base_items if name in final_tools)
    required_base_names = tuple(
        name
        for name, tool in base_items
        if name in final_tools and bool(getattr(tool, "chatinter_required_tool", False))
    )
    native_names = tuple(tool.binding.tool_name for tool in native_tools)
    dispatch_names = tuple(name for name, _tool in dispatch_items)
    overflow_names = (
        (OVERFLOW_SKILL_TOOL_NAME,)
        if OVERFLOW_SKILL_TOOL_NAME in final_tools
        else ()
    )
    local_names = {*base_names, *native_names, *dispatch_names, *overflow_names}
    web_names = tuple(name for name in final_tools if name not in local_names)
    native_fallback_by_module = {
        _module_key(tool.skill.plugin_module): (name, tool)
        for name, tool in native_fallback_tools.items()
    }
    native_fallback_bindings = tuple(
        (
            snapshot.command_id,
            native_tool.binding.tool_name,
            fallback_name,
            fallback_tool.exposure_key,
        )
        for snapshot, native_tool in zip(
            selected_native,
            native_tools,
            strict=True,
        )
        if (
            fallback := native_fallback_by_module.get(
                _module_key(snapshot.plugin_module)
            )
        )
        is not None
        for fallback_name, fallback_tool in (fallback,)
    )
    semantic_ids = {
        normalize_message_text(snapshot.command_id) for snapshot in selected_native
    }
    tool_priority_names = _stable_unique(
        (
            *base_names,
            *native_names,
            *dispatch_names,
            *overflow_names,
            *web_names,
        )
    )
    return MixedToolView(
        tools=final_tools,
        command_candidate_text=projection.text,
        native_command_ids=tuple(snapshot.command_id for snapshot in selected_native),
        indexed_command_ids=tuple(
            snapshot.command_id
            for snapshot in indexed_available
            if normalize_message_text(snapshot.command_id) in exposed_skill_ids
        ),
        skill_tool_names=final_skill_names,
        tool_priority_names=tool_priority_names,
        required_tool_names=_stable_unique(
            (*required_base_names, *overflow_names)
        ),
        native_tool_bindings=tuple(
            (snapshot.command_id, tool.binding.tool_name)
            for snapshot, tool in zip(selected_native, native_tools, strict=True)
        ),
        indexed_tool_bindings=tuple(
            (
                snapshot.command_id,
                skill_tools_by_command_id[normalize_message_text(snapshot.command_id)],
            )
            for snapshot in indexed_available
            if normalize_message_text(snapshot.command_id) in skill_tools_by_command_id
        ),
        initial_candidates=tuple(initial_candidates),
        strict_candidate_exposures=tuple(
            (
                delegate.exposure_key,
                command_id,
                strict_mode_by_command_id.get(command_id, "boundary"),
            )
            for command_id in projection.displayed_command_ids
            if (delegate := delegate_by_command_id.get(command_id)) is not None
        ),
        base_candidate_contexts=base_candidate_contexts,
        candidate_token_budget=candidate_token_budget,
        schema_omitted_names=capacity_omitted_names,
        skill_delegates=tuple(sorted(all_skill_delegates.items())),
        indexed_delegate_bindings=tuple(
            (
                snapshot.command_id,
                delegate.name,
            )
            for snapshot in indexed_available
            if (
                delegate := delegate_by_command_id.get(
                    normalize_message_text(snapshot.command_id)
                )
            )
            is not None
        ),
        native_fallback_delegates=tuple(sorted(native_fallback_tools.items())),
        native_fallback_bindings=native_fallback_bindings,
        native_fallback_candidates=tuple(
            candidate
            for candidate in catalog.initial_candidates
            if normalize_message_text(candidate.schema.command_id) in semantic_ids
        ),
        exposure_ledger=catalog.exposure_ledger,
        overflow_token_budget=ambiguity_token_budget,
        overflow_char_budget=_adapter_tool_result_char_budget(adapter),
    )


def _select_base_tool_items(
    tools: dict[str, Any],
    capacity: int,
) -> list[tuple[str, Any]]:
    available = sorted(tools.items())
    selected: list[tuple[str, Any]] = []
    consumed_groups: set[str] = set()
    for name, tool in available:
        if len(selected) >= capacity:
            break
        group = str(getattr(tool, "chatinter_tool_group", "") or "").strip()
        atomic = bool(getattr(tool, "chatinter_tool_group_atomic", False))
        if not group or not atomic:
            selected.append((name, tool))
            continue
        if group in consumed_groups:
            continue
        consumed_groups.add(group)
        members = [
            item
            for item in available
            if str(getattr(item[1], "chatinter_tool_group", "") or "").strip() == group
            and bool(getattr(item[1], "chatinter_tool_group_atomic", False))
        ]
        if len(selected) + len(members) <= capacity:
            selected.extend(members)
    return selected


def _is_skill_delegate(tool: Any) -> bool:
    return str(getattr(tool, "chatinter_plugin_tool_kind", "") or "") in {
        "skill_dispatch",
        "gscore",
    }


async def bound_candidate_tool_view_schema(
    view: MixedToolView,
    *,
    token_budget: int,
) -> MixedToolView:
    required_names = view.required_tool_names
    try:
        selection = await select_tools_within_schema_budget(
            view.tools,
            token_budget=token_budget,
            priority_names=view.tool_priority_names,
            required_names=required_names,
        )
    except ValueError:
        selection = await select_tools_within_schema_budget(
            view.tools,
            token_budget=token_budget,
            priority_names=view.tool_priority_names,
        )
    delegate_tools = dict(view.skill_delegates)
    native_fallback_tools = dict(view.native_fallback_delegates)

    def omitted_delegates(selected: set[str]) -> dict[str, Any]:
        omitted = {
            name: tool
            for name, tool in delegate_tools.items()
            if name not in selected
        }
        for _command_id, native_name, fallback_name, _key in (
            view.native_fallback_bindings
        ):
            if native_name not in selected and fallback_name in native_fallback_tools:
                omitted[fallback_name] = native_fallback_tools[fallback_name]
        return omitted

    selected_names = set(selection.tools)
    omitted_delegate_tools = omitted_delegates(selected_names)
    if omitted_delegate_tools:
        tools_with_overflow = dict(view.tools)
        tools_with_overflow[OVERFLOW_SKILL_TOOL_NAME] = SkillOverflowTool(
            omitted_delegate_tools,
            result_token_budget=max(view.overflow_token_budget, 1),
            result_char_budget=max(view.overflow_char_budget, 1),
            exposure_ledger=view.exposure_ledger,
        )
        required_names = _stable_unique(
            (*required_names, OVERFLOW_SKILL_TOOL_NAME)
        )
        priority_names = _stable_unique(
            (*view.tool_priority_names, OVERFLOW_SKILL_TOOL_NAME)
        )
        try:
            selection = await select_tools_within_schema_budget(
                tools_with_overflow,
                token_budget=token_budget,
                priority_names=priority_names,
                required_names=required_names,
            )
        except ValueError:
            selection = await select_tools_within_schema_budget(
                tools_with_overflow,
                token_budget=token_budget,
                priority_names=priority_names,
            )
        selected_names = set(selection.tools)
        omitted_delegate_tools = omitted_delegates(selected_names)
        if (
            OVERFLOW_SKILL_TOOL_NAME in selection.tools
            and omitted_delegate_tools
        ):
            selected_tools = dict(selection.tools)
            selected_tools[OVERFLOW_SKILL_TOOL_NAME] = SkillOverflowTool(
                omitted_delegate_tools,
                result_token_budget=max(view.overflow_token_budget, 1),
                result_char_budget=max(view.overflow_char_budget, 1),
                exposure_ledger=view.exposure_ledger,
            )
            selection = replace(selection, tools=selected_tools)
            selected_names = set(selection.tools)
    plugin_capacity_degraded = False
    if omitted_delegate_tools and OVERFLOW_SKILL_TOOL_NAME not in selected_names:
        native_names = {
            tool_name for _command_id, tool_name in view.native_tool_bindings
        }
        plugin_names = {
            *delegate_tools,
            *native_names,
            OVERFLOW_SKILL_TOOL_NAME,
        }
        plain_tools = {
            name: tool
            for name, tool in selection.tools.items()
            if name not in plugin_names and not _is_skill_delegate(tool)
        }
        selection = ToolSchemaSelection(
            tools=dict(sorted(plain_tools.items())),
            schema_tokens=await tool_schema_tokens(plain_tools),
            omitted_names=_stable_unique(
                (*selection.omitted_names, *plugin_names)
            ),
        )
        selected_names = set(selection.tools)
        plugin_capacity_degraded = True
    omitted_delegate_names = set(omitted_delegate_tools)
    skill_tool_names = tuple(
        name for name in view.skill_tool_names if name in selected_names
    )
    if (
        OVERFLOW_SKILL_TOOL_NAME in selected_names
        and OVERFLOW_SKILL_TOOL_NAME not in skill_tool_names
    ):
        skill_tool_names = (*skill_tool_names, OVERFLOW_SKILL_TOOL_NAME)
    native_command_ids = tuple(
        command_id
        for command_id, tool_name in view.native_tool_bindings
        if tool_name in selected_names
    )
    selected_skill_by_command: dict[str, str] = {}
    for command_id, delegate_name in view.indexed_delegate_bindings:
        if delegate_name in selected_names:
            selected_skill_by_command[command_id] = delegate_name
        elif (
            OVERFLOW_SKILL_TOOL_NAME in selected_names
            and delegate_name in omitted_delegate_names
        ):
            selected_skill_by_command[command_id] = OVERFLOW_SKILL_TOOL_NAME
    native_fallback_keys: dict[str, CandidateExposureKey] = {}
    for command_id, native_name, fallback_name, exposure_key in (
        view.native_fallback_bindings
    ):
        if (
            native_name not in selected_names
            and OVERFLOW_SKILL_TOOL_NAME in selected_names
            and fallback_name in omitted_delegate_names
        ):
            selected_skill_by_command[command_id] = OVERFLOW_SKILL_TOOL_NAME
            native_fallback_keys[command_id] = exposure_key
    indexed_command_ids = tuple(selected_skill_by_command)
    active_fallback_candidates = tuple(
        candidate
        for candidate in view.native_fallback_candidates
        if normalize_message_text(candidate.schema.command_id)
        in native_fallback_keys
    )
    initial_candidates = tuple(
        item
        for item in (*view.initial_candidates, *active_fallback_candidates)
        if normalize_message_text(item.schema.command_id) in selected_skill_by_command
    )
    projection = _render_bounded_candidate_context(
        base_candidate_contexts=tuple(
            (tool_name, text)
            for tool_name, text in view.base_candidate_contexts
            if tool_name in selected_names
        ),
        initial_candidates=initial_candidates,
        skill_tools_by_command_id=selected_skill_by_command,
        token_budget=min(
            max(view.candidate_token_budget, 0),
            max(int(token_budget) - selection.schema_tokens, 0),
        ),
    )
    strict_mode_by_command_id = {
        normalize_message_text(candidate.schema.command_id): normalize_message_text(
            candidate.strict_identity_mode
        )
        for candidate in initial_candidates
        if normalize_message_text(candidate.strict_identity_mode)
    }
    strict_mode_by_command_id.update({
        command_id: match_mode
        for _key, command_id, match_mode in view.strict_candidate_exposures
    })
    return replace(
        view,
        tools=selection.tools,
        command_candidate_text=projection.text,
        native_command_ids=native_command_ids,
        indexed_command_ids=indexed_command_ids,
        indexed_tool_bindings=tuple(selected_skill_by_command.items()),
        skill_tool_names=skill_tool_names,
        initial_candidates=initial_candidates,
        strict_candidate_exposures=(
            *(
                (key, command_id, match_mode)
                for key, command_id, match_mode in view.strict_candidate_exposures
                if command_id in projection.displayed_command_ids
            ),
            *(
                (
                    key,
                    command_id,
                    strict_mode_by_command_id.get(command_id, "boundary"),
                )
                for command_id, key in native_fallback_keys.items()
                if command_id in projection.displayed_command_ids
            ),
        ),
        schema_tokens=selection.schema_tokens,
        schema_omitted_names=_stable_unique(
            (*view.schema_omitted_names, *selection.omitted_names)
        ),
        required_tool_names=tuple(
            name for name in required_names if name in selection.tools
        ),
        plugin_capacity_degraded=plugin_capacity_degraded,
    )


async def select_tools_within_schema_budget(
    tools: dict[str, Any] | None,
    *,
    token_budget: int,
    priority_names: tuple[str, ...] = (),
    required_names: tuple[str, ...] = (),
) -> ToolSchemaSelection:
    available = dict(tools or {})
    if not available:
        return ToolSchemaSelection({}, 0, ())

    required = tuple(name for name in required_names if name in available)
    ordered = _stable_unique(
        (
            *required,
            *(name for name in priority_names if name in available),
            *sorted(available),
        )
    )
    selected_names: list[str] = []
    schema_payloads = {
        name: await _tool_schema_payload(name, available[name]) for name in ordered
    }
    schema_tokens = 0
    budget = max(int(token_budget), 0)
    consumed: set[str] = set()
    required_set = set(required)
    for name in ordered:
        if name in consumed:
            continue
        group = str(
            getattr(available[name], "chatinter_tool_group", "") or ""
        ).strip()
        atomic = bool(
            group and getattr(available[name], "chatinter_tool_group_atomic", False)
        )
        unit = (
            tuple(
                item
                for item in ordered
                if str(
                    getattr(available[item], "chatinter_tool_group", "") or ""
                ).strip()
                == group
                and bool(
                    getattr(
                        available[item],
                        "chatinter_tool_group_atomic",
                        False,
                    )
                )
            )
            if atomic
            else (name,)
        )
        consumed.update(unit)
        trial_names = (*selected_names, *unit)
        trial_tokens = _schema_payload_tokens(
            [schema_payloads[item] for item in trial_names]
        )
        if trial_tokens <= budget:
            selected_names.extend(unit)
            schema_tokens = trial_tokens
            continue
        if required_set.intersection(unit):
            raise ValueError(
                "required tool schema exceeds available prompt budget: "
                f"tool={','.join(unit)} required={trial_tokens} available={budget}"
            )
        omitted = tuple(item for item in ordered if item not in selected_names)
        return ToolSchemaSelection(
            tools={name: available[name] for name in sorted(selected_names)},
            schema_tokens=schema_tokens,
            omitted_names=omitted,
        )

    return ToolSchemaSelection(
        tools={name: available[name] for name in sorted(selected_names)},
        schema_tokens=schema_tokens,
        omitted_names=(),
    )


async def tool_schema_tokens(tools: dict[str, Any] | None) -> int:
    schemas = [
        await _tool_schema_payload(name, (tools or {})[name])
        for name in sorted(tools or {})
    ]
    return _schema_payload_tokens(schemas)


async def _tool_schema_payload(name: str, tool: Any) -> dict[str, Any]:
    definition = await tool.get_definition()
    payload = (
        definition.model_dump(mode="json")
        if hasattr(definition, "model_dump")
        else {
            "name": str(getattr(definition, "name", name) or name),
            "description": str(getattr(definition, "description", "") or ""),
            "parameters": getattr(definition, "parameters", {}) or {},
        }
    )
    return {"name": name, "schema": payload}


def _schema_payload_tokens(schemas: list[dict[str, Any]]) -> int:
    if not schemas:
        return 0
    return estimate_text_tokens(
        json.dumps(
            schemas,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
    )


def _render_bounded_candidate_context(
    *,
    base_candidate_contexts: tuple[tuple[str, str], ...],
    initial_candidates: tuple[CommandCandidate, ...] = (),
    skill_tools_by_command_id: dict[str, str] | None = None,
    token_budget: int,
) -> CandidateContextProjection:
    budget = max(int(token_budget), 0)
    if budget <= 0:
        return CandidateContextProjection("")
    sections: list[str] = []
    for _tool_name, text in base_candidate_contexts:
        candidate = "\n".join((*sections, text))
        if estimate_text_tokens(candidate) > budget:
            break
        sections.append(text)
    used_tokens = estimate_text_tokens("\n".join(sections)) if sections else 0
    separator_tokens = estimate_text_tokens("\n") if sections else 0
    command_budget = max(budget - used_tokens - separator_tokens, 0)
    command_context = render_command_candidate_context(
        list(initial_candidates),
        token_budget=command_budget,
        skill_tools_by_command_id=skill_tools_by_command_id,
    )
    if command_context:
        candidate = "\n".join((*sections, command_context))
        if estimate_text_tokens(candidate) <= budget:
            sections.append(command_context)
    displayed_command_ids: tuple[str, ...] = ()
    if command_context and command_context in sections:
        try:
            parsed = json.loads(command_context)
            displayed_command_ids = tuple(
                normalize_message_text(str(card.get("command_id") or ""))
                for card in parsed.get("commands", ())
                if isinstance(card, dict)
                and normalize_message_text(str(card.get("command_id") or ""))
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            displayed_command_ids = ()
    return CandidateContextProjection(
        text="\n".join(sections),
        displayed_command_ids=displayed_command_ids,
    )


def expose_candidate_tool_view(view: MixedToolView) -> tuple[str, ...]:
    if view.exposure_ledger is not None:
        exposed: list[str] = []
        for key, command_id, match_mode in view.strict_candidate_exposures:
            exposed.extend(
                view.exposure_ledger.expose(
                    key,
                    (command_id,),
                    discovery_source="strict_identity",
                    pending=False,
                    exact_identity=True,
                    strict_identity_mode=match_mode,
                )
            )
        return tuple(dict.fromkeys(exposed))

    exposed: list[str] = []
    for key, command_id, match_mode in view.strict_candidate_exposures:
        # All catalog-local keys share one turn ledger. Resolve it through the
        # dispatch tool carrying the same key so no authorization state enters
        # the immutable model-visible view.
        for tool in view.tools.values():
            if getattr(tool, "exposure_key", None) != key:
                continue
            values = tool.expose_candidates(
                (command_id,),
                source="strict_identity",
                pending=False,
                exact_identity=True,
                strict_identity_mode=match_mode,
            )
            exposed.extend(values)
            break
        else:
            overflow = view.tools.get(OVERFLOW_SKILL_TOOL_NAME)
            delegates = getattr(overflow, "_delegates", {}) if overflow else {}
            for tool in delegates.values():
                if getattr(tool, "exposure_key", None) != key:
                    continue
                exposed.extend(
                    tool.expose_candidates(
                        (command_id,),
                        source="strict_identity",
                        pending=False,
                        exact_identity=True,
                        strict_identity_mode=match_mode,
                    )
                )
                break
    return tuple(dict.fromkeys(exposed))


def _semantic_fallback_delegate_name(plugin_module: str) -> str:
    module = normalize_message_text(plugin_module).casefold()
    digest = hashlib.blake2s(module.encode("utf-8"), digest_size=6).hexdigest()
    return f"ci_hidden_semantic_{digest}"


def _adapter_tool_result_char_budget(adapter: ProviderCapabilityAdapter) -> int:
    direct = getattr(adapter, "max_tool_result_chars", None)
    if direct is not None:
        return max(int(direct or 0), 1)
    profile = getattr(adapter, "profile", None)
    protocol = getattr(profile, "protocol", None)
    mcp = getattr(protocol, "mcp", None) or getattr(profile, "mcp", None)
    return max(int(getattr(mcp, "max_result_chars", 12_000) or 12_000), 1)


def _bind_tool_result_budget(
    tool: Any,
    *,
    token_budget: int,
    char_budget: int,
) -> Any:
    binder = getattr(tool, "with_result_budget", None)
    if not callable(binder):
        return tool
    return binder(token_budget=token_budget, char_budget=char_budget)


def _stable_snapshots(
    snapshots: list[CommandToolSnapshot],
) -> list[CommandToolSnapshot]:
    by_id: dict[str, CommandToolSnapshot] = {}
    for snapshot in snapshots:
        command_id = normalize_message_text(snapshot.command_id)
        if command_id and command_id not in by_id:
            by_id[command_id] = snapshot
    return sorted(by_id.values(), key=_native_snapshot_sort_key)


def _available_dispatch_skills(
    skill_index: PluginSkillIndex,
    available_commands: list[CommandToolSnapshot],
    *,
    known_commands: list[CommandToolSnapshot],
    native_semantic_names: dict[str, set[str]],
) -> tuple[PluginSkill, ...]:
    available_by_module: dict[str, dict[str, CommandToolSnapshot]] = {}
    for snapshot in available_commands:
        module_key = _module_key(snapshot.plugin_module)
        command_id = normalize_message_text(snapshot.command_id)
        if module_key and command_id:
            available_by_module.setdefault(module_key, {})[command_id] = snapshot

    known_by_module: dict[str, dict[str, CommandToolSnapshot]] = {}
    for snapshot in known_commands:
        module_key = _module_key(snapshot.plugin_module)
        command_id = normalize_message_text(snapshot.command_id)
        if module_key and command_id:
            known_by_module.setdefault(module_key, {})[command_id] = snapshot

    projected: list[PluginSkill] = []
    for skill in skill_index.skills:
        module_key = _module_key(skill.plugin_module)
        snapshots_by_id = available_by_module.get(module_key, {})
        known_by_id = known_by_module.get(module_key, {})
        # Expose only currently-available commands on the skill; known-but-
        # unavailable ids stay reachable inside the dispatch tool via its
        # module-scoped known lookup (unavailable_in_context branch).
        command_ids = tuple(
            command_id
            for command_id in skill.command_ids
            if normalize_message_text(command_id) in snapshots_by_id
            and normalize_message_text(command_id) in known_by_id
        )
        if not command_ids:
            continue
        # Input metadata is derived from available snapshots only; inferred
        # semantic classifications stay internal to execution.
        snapshots = [
            snapshots_by_id[normalize_message_text(command_id)]
            for command_id in command_ids
        ]
        semantic_names = {
            normalize_message_text(
                str(snapshot.meta.get("semantic_tool_name") or "")
            ).casefold()
            for snapshot in snapshots
            if isinstance(snapshot.meta, dict)
            and normalize_message_text(
                str(snapshot.meta.get("semantic_tool_name") or "")
            )
        }
        semantic_names.difference_update(native_semantic_names.get(module_key, set()))
        projected.append(
            replace(
                skill,
                command_ids=command_ids,
                command_count=len(command_ids),
                semantic_tools=tuple(
                    contract
                    for contract in skill.semantic_tools
                    if normalize_message_text(contract.name).casefold()
                    in semantic_names
                ),
                input_types=_skill_input_types(snapshots),
            )
        )
    return tuple(projected)


def _skill_input_types(
    snapshots: list[CommandToolSnapshot],
) -> tuple[str, ...]:
    values: list[str] = []
    for snapshot in snapshots:
        values.extend(snapshot.input_requirements)
        values.extend(slot.type for slot in snapshot.slots)
        values.extend(key for key, required in snapshot.requires.items() if required)
    return _stable_values(values)


def _stable_values(values: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                normalized
                for value in values
                if (normalized := normalize_message_text(str(value or "")))
            },
            key=lambda value: (value.casefold(), value),
        )
    )


def _select_dispatch_items(
    dispatch_tools: dict[str, Any],
    *,
    capacity: int,
) -> list[tuple[str, Any]]:
    if capacity <= 0:
        return []
    return sorted(dispatch_tools.items(), key=lambda item: item[0])[:capacity]


def _native_snapshot_sort_key(
    snapshot: CommandToolSnapshot,
) -> tuple[str, str, str]:
    semantic_name = ""
    if isinstance(snapshot.meta, dict):
        semantic_name = normalize_message_text(
            str(snapshot.meta.get("semantic_tool_name") or "")
        )
    return (
        semantic_name.casefold(),
        _module_key(snapshot.plugin_module),
        normalize_message_text(snapshot.command_id).casefold(),
    )


def _is_semantic_snapshot(snapshot: CommandToolSnapshot) -> bool:
    return bool(
        isinstance(snapshot.meta, dict)
        and snapshot.meta.get("semantic_tool_name")
        and isinstance(snapshot.meta.get("semantic_contract"), dict)
    )


def _module_key(value: str) -> str:
    return normalize_message_text(value).casefold()


def _stable_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = [
    "MixedToolCatalog",
    "MixedToolView",
    "ToolSchemaSelection",
    "assemble_candidate_tool_view",
    "bound_candidate_tool_view_schema",
    "build_mixed_tool_catalog",
    "expose_candidate_tool_view",
    "select_tools_within_schema_budget",
    "tool_schema_tokens",
]
