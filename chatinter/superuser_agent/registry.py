"""Registry for superuser-only ChatInter agent tools.

The registry is intentionally the single source of truth for superuser Agent
tool availability, metadata and safety defaults.  Tool implementations still
own their concrete permission checks, while the registry provides Hermes-like
check_fn caching and Claude-like capability grouping for the runtime prompt and
tool exposure layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import time
from typing import Any, Literal

from zhenxun.services.llm.types.protocols import ToolExecutable

ToolFactory = Callable[[], ToolExecutable]
ToolCheck = Callable[[], bool]
ToolRisk = Literal["low", "medium", "high", "critical"]
ApprovalMode = Literal["allow", "ask", "deny", "policy"]

_CHECK_CACHE_TTL_SECONDS = 30.0
_CHECK_CACHE: dict[ToolCheck, tuple[float, bool]] = {}


@dataclass(frozen=True)
class SuperuserToolCard:
    """Metadata attached to every superuser tool.

    `approval_mode="policy"` means the tool's execute() method performs the
    concrete allow/ask/deny check using the configured permission policy.
    """

    name: str
    category: str = "general"
    risk: ToolRisk = "medium"
    approval_mode: ApprovalMode = "policy"
    check_fn: ToolCheck | None = None
    cache_ttl_seconds: float = _CHECK_CACHE_TTL_SECONDS
    approval_scope: str = ""
    background_capable: bool = False
    produces_artifacts: bool = False
    read_only: bool = False
    todo_relevant: bool = False
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_of_truth: str = "local_state"
    requires_real_tool: bool = True
    output_mode: str = "plugin_output"
    entity_scope: str = "global"
    reliability: float = 0.7
    schema_quality: float = 0.65
    soft_tool: bool = False
    availability_reason: str = ""

    def public_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("check_fn", None)
        payload["tags"] = list(self.tags)
        return payload


@dataclass(frozen=True)
class RegisteredSuperuserTool:
    factory: ToolFactory
    card: SuperuserToolCard


@dataclass(frozen=True)
class SuperuserToolBundle:
    """Available superuser tools plus their registry metadata."""

    tools: dict[str, ToolExecutable]
    cards: tuple[SuperuserToolCard, ...]

    @property
    def cards_by_name(self) -> dict[str, SuperuserToolCard]:
        return {card.name: card for card in self.cards}


class SuperuserToolRegistry:
    """Explicit registry so superuser tools can be composed by toolset."""

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredSuperuserTool] = {}

    def register(
        self,
        factory: ToolFactory,
        *,
        category: str = "",
        risk: ToolRisk | None = None,
        approval_mode: ApprovalMode = "policy",
        check_fn: ToolCheck | None = None,
        approval_scope: str = "",
        background_capable: bool = False,
        produces_artifacts: bool = False,
        read_only: bool | None = None,
        todo_relevant: bool = False,
        description: str = "",
        tags: tuple[str, ...] | list[str] = (),
        source_of_truth: str = "",
        requires_real_tool: bool | None = None,
        output_mode: str = "",
        entity_scope: str = "",
        reliability: float | None = None,
        schema_quality: float | None = None,
        soft_tool: bool | None = None,
    ) -> ToolFactory:
        tool = factory()
        name = tool_name(tool)
        if not name:
            raise ValueError("superuser agent tool must expose a non-empty name")
        inferred = infer_tool_card(tool)
        card = SuperuserToolCard(
            name=name,
            category=category or inferred.category,
            risk=risk or inferred.risk,
            approval_mode=approval_mode or inferred.approval_mode,
            check_fn=check_fn or inferred.check_fn,
            approval_scope=approval_scope or inferred.approval_scope,
            background_capable=background_capable or inferred.background_capable,
            produces_artifacts=produces_artifacts or inferred.produces_artifacts,
            read_only=inferred.read_only if read_only is None else bool(read_only),
            todo_relevant=todo_relevant or inferred.todo_relevant,
            description=description or inferred.description,
            tags=tuple(tags or inferred.tags),
            source_of_truth=source_of_truth or inferred.source_of_truth,
            requires_real_tool=(
                inferred.requires_real_tool
                if requires_real_tool is None
                else bool(requires_real_tool)
            ),
            output_mode=output_mode or inferred.output_mode,
            entity_scope=entity_scope or inferred.entity_scope,
            reliability=(
                inferred.reliability if reliability is None else _clamp01(reliability)
            ),
            schema_quality=(
                inferred.schema_quality
                if schema_quality is None
                else _clamp01(schema_quality)
            ),
            soft_tool=inferred.soft_tool if soft_tool is None else bool(soft_tool),
        )
        self._entries[name] = RegisteredSuperuserTool(factory=factory, card=card)
        return factory

    def build_tools(self) -> dict[str, ToolExecutable]:
        return {
            name: entry.factory()
            for name, entry in self._entries.items()
            if self._card_available(entry.card)
        }

    def build_bundle(self) -> SuperuserToolBundle:
        tools: dict[str, ToolExecutable] = {}
        cards: list[SuperuserToolCard] = []
        for name, entry in self._entries.items():
            if not self._card_available(entry.card):
                continue
            tools[name] = entry.factory()
            cards.append(entry.card)
        return SuperuserToolBundle(tools=tools, cards=tuple(cards))

    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def available_tool_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, entry in self._entries.items()
            if self._card_available(entry.card)
        )

    def tool_cards(
        self, *, available_only: bool = False
    ) -> tuple[SuperuserToolCard, ...]:
        cards: list[SuperuserToolCard] = []
        for entry in self._entries.values():
            if available_only and not self._card_available(entry.card):
                continue
            cards.append(entry.card)
        return tuple(cards)

    def tool_card(self, name: str) -> SuperuserToolCard | None:
        entry = self._entries.get(str(name or "").strip())
        return entry.card if entry else None

    def invalidate_check_cache(self) -> None:
        _CHECK_CACHE.clear()

    def _card_available(self, card: SuperuserToolCard) -> bool:
        if card.approval_mode == "deny":
            return False
        if card.check_fn is None:
            return True
        return _cached_check(card.check_fn, ttl=card.cache_ttl_seconds)


def tool_name(tool: ToolExecutable) -> str:
    return str(getattr(tool, "name", "") or "").strip()


def infer_tool_card(tool: ToolExecutable) -> SuperuserToolCard:
    name = tool_name(tool)
    category = _category_from_name(name)
    risk = _risk_from_name(name)
    read_only = _read_only_from_name(name)
    return SuperuserToolCard(
        name=name,
        category=category,
        risk=risk,
        approval_scope=category,
        background_capable=name.startswith("background_task_start"),
        produces_artifacts=name.startswith(
            (
                "artifact_",
                "background_",
                "worktree_",
                "patch_",
                "engineering_eval_",
                "engineering_loop_",
                "engineering_lsp_",
                "semantic_patch_",
                "python_",
                "uv_",
                "git_",
                "shell_",
                "server_",
            )
        ),
        read_only=read_only,
        todo_relevant=category
        in {"patch", "eval", "engineering_loop", "plugin_dev", "agent_run", "todo"},
        description=str(getattr(tool, "description", "") or ""),
        tags=tuple(_tags_from_name(name)),
        source_of_truth=_source_of_truth_from_category(category),
        requires_real_tool=True,
        output_mode=_output_mode_from_name(
            name, category=category, read_only=read_only
        ),
        entity_scope=_entity_scope_from_category(category),
        reliability=_initial_reliability_from_name(
            name, risk=risk, read_only=read_only
        ),
        schema_quality=_schema_quality_from_name(name, category=category),
        soft_tool=False,
    )


def _cached_check(check_fn: ToolCheck, *, ttl: float) -> bool:
    now = time.monotonic()
    cached = _CHECK_CACHE.get(check_fn)
    if cached is not None:
        timestamp, value = cached
        if now - timestamp <= max(float(ttl or _CHECK_CACHE_TTL_SECONDS), 0.1):
            return value
    try:
        value = bool(check_fn())
    except Exception:
        value = False
    _CHECK_CACHE[check_fn] = (now, value)
    return value


def _category_from_name(name: str) -> str:
    if name.startswith("agent_run_"):
        return "agent_run"
    if name.startswith(("approve_", "reject_", "revoke_", "list_pending_")):
        return "approval"
    if name.startswith("artifact_"):
        return "artifact"
    if name.startswith("runtime_event_"):
        return "runtime"
    if name.startswith("audit_"):
        return "audit"
    if name.startswith("todo_"):
        return "todo"
    if name.startswith("background_"):
        return "background"
    if name.startswith("engineering_eval_"):
        return "eval"
    if name.startswith(("engineering_loop_", "engineering_lsp_", "semantic_patch_")):
        return "engineering_loop"
    if name.startswith(("patch_",)):
        return "patch"
    if name.startswith("plugin_dev_"):
        return "plugin_dev"
    if name.startswith(
        (
            "read_file",
            "list_dir",
            "search_files",
            "write_file",
            "append_file",
            "replace_in_file",
        )
    ):
        return "file"
    if name.startswith("git_"):
        return "git"
    if name.startswith("python_"):
        return "python"
    if name.startswith(("server_", "process_")):
        return "server"
    if name.startswith("shell_"):
        return "shell"
    if name.startswith("uv_"):
        return "uv"
    if name.startswith("worktree_"):
        return "worktree"
    return "general"


def _risk_from_name(name: str) -> ToolRisk:
    if name in {
        "patch_apply",
        "patch_rollback",
        "write_file",
        "append_file",
        "replace_in_file",
        "plugin_dev_scaffold",
        "plugin_dev_write_file",
        "server_command",
        "shell_command",
        "python_exec",
        "background_task_start",
        "background_task_cancel",
        "worktree_remove",
    }:
        return "high"
    if name in {
        "git_command",
        "uv_command",
        "python_module",
        "engineering_eval_run",
        "worktree_create",
    }:
        return "medium"
    return "low"


def _read_only_from_name(name: str) -> bool:
    return name in {
        "agent_run_status",
        "artifact_read",
        "artifact_list",
        "runtime_event_list",
        "runtime_event_read",
        "audit_log_query",
        "background_observation_list",
        "background_observation_wait",
        "background_task_status",
        "engineering_eval_gate",
        "engineering_eval_status",
        "engineering_loop_status",
        "engineering_lsp_read",
        "list_dir",
        "list_pending_approvals",
        "patch_show",
        "plugin_dev_inspect",
        "process_list",
        "read_file",
        "search_files",
        "server_status",
        "todo_read",
        "worktree_list",
        "worktree_status",
    }


def _tags_from_name(name: str) -> list[str]:
    category = _category_from_name(name)
    tags = [category]
    if not _read_only_from_name(name):
        tags.append("mutating")
    if _risk_from_name(name) in {"high", "critical"}:
        tags.append("approval_sensitive")
    return tags


def _source_of_truth_from_category(category: str) -> str:
    if category in {
        "shell",
        "python",
        "uv",
        "git",
        "server",
        "file",
        "patch",
        "worktree",
        "engineering_loop",
    }:
        return "local_state"
    if category in {"background", "agent_run", "todo", "artifact", "audit", "runtime"}:
        return "local_state"
    if category == "plugin_dev":
        return "local_state"
    return "unknown"


def _output_mode_from_name(name: str, *, category: str, read_only: bool) -> str:
    if category in {"artifact", "file"}:
        return "file" if not read_only else "text"
    if category == "runtime":
        return "text"
    if category in {
        "shell",
        "python",
        "uv",
        "git",
        "server",
        "background",
        "eval",
        "engineering_loop",
        "worktree",
    }:
        return "plugin_output"
    if category in {"patch", "plugin_dev"}:
        return "action"
    if read_only or name.endswith(("_status", "_list", "_read")):
        return "text"
    return "plugin_output"


def _entity_scope_from_category(category: str) -> str:
    if category in {
        "shell",
        "python",
        "uv",
        "git",
        "server",
        "file",
        "patch",
        "worktree",
        "engineering_loop",
    }:
        return "global"
    if category in {"approval", "agent_run", "todo"}:
        return "actor_user"
    return "global"


def _initial_reliability_from_name(
    name: str,
    *,
    risk: ToolRisk,
    read_only: bool,
) -> float:
    score = 0.72
    if read_only:
        score += 0.08
    if risk == "medium":
        score -= 0.06
    elif risk in {"high", "critical"}:
        score -= 0.12
    if name.startswith(
        (
            "patch_",
            "engineering_eval_",
            "engineering_loop_",
            "engineering_lsp_",
            "semantic_patch_",
            "agent_run_",
            "artifact_",
        )
    ):
        score += 0.04
    return _clamp01(score)


def _schema_quality_from_name(name: str, *, category: str) -> float:
    score = 0.62
    if "_" in name:
        score += 0.05
    if category != "general":
        score += 0.08
    if name.startswith(
        (
            "patch_",
            "engineering_eval_",
            "engineering_loop_",
            "engineering_lsp_",
            "semantic_patch_",
            "agent_run_",
            "background_",
        )
    ):
        score += 0.05
    return _clamp01(score)


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(number, 1.0))


_REGISTRY = SuperuserToolRegistry()


def register_superuser_tool(
    factory: ToolFactory | None = None,
    **metadata: Any,
) -> ToolFactory | Callable[[ToolFactory], ToolFactory]:
    def decorator(inner: ToolFactory) -> ToolFactory:
        return _REGISTRY.register(inner, **metadata)

    if factory is None:
        return decorator
    return decorator(factory)


def build_superuser_agent_tool_bundle() -> SuperuserToolBundle:
    from . import toolsets as _toolsets  # noqa: F401  # import registers toolsets

    return _REGISTRY.build_bundle()


def registered_superuser_tool_names() -> tuple[str, ...]:
    from . import toolsets as _toolsets  # noqa: F401  # import registers toolsets

    return _REGISTRY.tool_names()


def available_superuser_tool_names() -> tuple[str, ...]:
    from . import toolsets as _toolsets  # noqa: F401  # import registers toolsets

    return _REGISTRY.available_tool_names()


def get_superuser_tool_card(name: str) -> SuperuserToolCard | None:
    from . import toolsets as _toolsets  # noqa: F401  # import registers toolsets

    return _REGISTRY.tool_card(name)


def superuser_tool_cards(
    *, available_only: bool = False
) -> tuple[SuperuserToolCard, ...]:
    from . import toolsets as _toolsets  # noqa: F401  # import registers toolsets

    return _REGISTRY.tool_cards(available_only=available_only)


def invalidate_superuser_tool_check_cache() -> None:
    _REGISTRY.invalidate_check_cache()


__all__ = [
    "ApprovalMode",
    "RegisteredSuperuserTool",
    "SuperuserToolBundle",
    "SuperuserToolCard",
    "SuperuserToolRegistry",
    "ToolCheck",
    "ToolRisk",
    "available_superuser_tool_names",
    "build_superuser_agent_tool_bundle",
    "get_superuser_tool_card",
    "infer_tool_card",
    "invalidate_superuser_tool_check_cache",
    "register_superuser_tool",
    "registered_superuser_tool_names",
    "superuser_tool_cards",
]
