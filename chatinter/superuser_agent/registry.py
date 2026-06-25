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
_CARD_DESCRIPTION_LIMIT = 96
_TOOL_SCHEMA_DESCRIPTION_LIMIT = 180
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
    destructive: bool = False
    side_effect: str = "query"
    always_load: bool = False
    defer_load: bool = False
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
        destructive: bool | None = None,
        side_effect: str = "",
        always_load: bool | None = None,
        defer_load: bool | None = None,
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
        _validate_explicit_safety_metadata(
            name=name,
            read_only=inferred.read_only if read_only is None else bool(read_only),
            risk=risk,
            destructive=destructive,
            side_effect=side_effect,
        )
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
            destructive=(
                inferred.destructive if destructive is None else bool(destructive)
            ),
            side_effect=side_effect or inferred.side_effect,
            always_load=(
                inferred.always_load if always_load is None else bool(always_load)
            ),
            defer_load=inferred.defer_load if defer_load is None else bool(defer_load),
            todo_relevant=todo_relevant or inferred.todo_relevant,
            description=_compact_description(description or inferred.description),
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

    def build_tools(
        self,
        *,
        message_text: str = "",
        limit: int | None = None,
        include_deferred: bool = False,
    ) -> dict[str, ToolExecutable]:
        return {
            name: _CardSummaryTool(entry.factory(), entry.card)
            for name, entry in self._selected_entries(
                message_text=message_text,
                limit=limit,
                include_deferred=include_deferred,
            )
        }

    def build_bundle(
        self,
        *,
        message_text: str = "",
        limit: int | None = None,
        include_deferred: bool = False,
    ) -> SuperuserToolBundle:
        tools: dict[str, ToolExecutable] = {}
        cards: list[SuperuserToolCard] = []
        for name, entry in self._selected_entries(
            message_text=message_text,
            limit=limit,
            include_deferred=include_deferred,
        ):
            tools[name] = _CardSummaryTool(entry.factory(), entry.card)
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

    def _selected_entries(
        self,
        *,
        message_text: str,
        limit: int | None,
        include_deferred: bool,
    ) -> tuple[tuple[str, RegisteredSuperuserTool], ...]:
        available = [
            (name, entry)
            for name, entry in self._entries.items()
            if self._card_available(entry.card)
        ]
        query = _normalize_query(message_text)

        selected: dict[str, RegisteredSuperuserTool] = {}
        for name, entry in available:
            if entry.card.always_load:
                selected[name] = entry

        scored: list[tuple[float, str, RegisteredSuperuserTool]] = []
        matched_count = 0
        for name, entry in available:
            if name in selected:
                continue
            score = _tool_relevance_score(entry.card, query=query)
            if entry.card.defer_load and not include_deferred and score < 3.0:
                continue
            if score <= 0:
                continue
            matched_count += 1
            scored.append((score, name, entry))
        scored.sort(
            key=lambda item: (
                item[0],
                item[2].card.reliability,
                item[2].card.schema_quality,
                -_risk_rank(item[2].card.risk),
            ),
            reverse=True,
        )

        cap = _selection_limit(query=query, explicit=limit)
        for _score, name, entry in scored:
            if len(selected) >= cap:
                break
            selected[name] = entry

        if not selected:
            return ()

        ordered = [(name, entry) for name, entry in available if name in selected]
        return tuple(ordered)


class _CardSummaryTool:
    """Use short registry text for prompts; keep full parameters for selected tools."""

    def __init__(self, tool: ToolExecutable, card: SuperuserToolCard) -> None:
        self._tool = tool
        self._card = card

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    async def get_definition(self) -> Any:
        definition = await self._tool.get_definition()
        summary = _tool_schema_summary(
            self._card,
            fallback=str(getattr(definition, "description", "") or ""),
        )
        if hasattr(definition, "model_copy"):
            return definition.model_copy(update={"description": summary})
        return type(definition)(
            name=getattr(definition, "name", self._card.name),
            description=summary,
            parameters=getattr(definition, "parameters", {}) or {},
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> Any:
        return await self._tool.execute(context=context, **kwargs)


def tool_name(tool: ToolExecutable) -> str:
    return str(getattr(tool, "name", "") or "").strip()


def _validate_explicit_safety_metadata(
    *,
    name: str,
    read_only: bool | None,
    risk: ToolRisk | None,
    destructive: bool | None,
    side_effect: str,
) -> None:
    if read_only is not False:
        return
    missing = [
        field
        for field, value in {
            "risk": risk,
            "destructive": destructive,
            "side_effect": side_effect,
        }.items()
        if value in (None, "")
    ]
    if missing:
        raise ValueError(
            f"non-readonly superuser tool {name!r} must explicitly declare "
            + ", ".join(missing)
        )


def infer_tool_card(tool: ToolExecutable) -> SuperuserToolCard:
    name = tool_name(tool)
    category = _category_from_name(name)
    risk = _risk_from_name(name)
    read_only = _read_only_from_name(name)
    side_effect = _side_effect_from_name(name, category=category, read_only=read_only)
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
        destructive=_destructive_from_name(name, risk=risk, read_only=read_only),
        side_effect=side_effect,
        always_load=_always_load_from_name(name),
        defer_load=_defer_load_from_name(name),
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
        "plugin_dev_publish",
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


def _destructive_from_name(name: str, *, risk: ToolRisk, read_only: bool) -> bool:
    if read_only:
        return False
    if risk in {"high", "critical"}:
        return True
    return any(
        marker in name
        for marker in (
            "write",
            "append",
            "replace",
            "remove",
            "cancel",
            "rollback",
            "apply",
            "command",
            "exec",
        )
    )


def _side_effect_from_name(
    name: str,
    *,
    category: str,
    read_only: bool,
) -> str:
    if read_only:
        return "query"
    if category in {"shell", "python", "uv", "git", "server", "background"}:
        return "execute"
    if category in {"file", "patch", "plugin_dev", "worktree", "todo"}:
        return "mutate"
    if category == "approval":
        return "control"
    if name.endswith(("_cancel", "_remove", "_rollback")):
        return "destructive"
    return "mutate"


def _always_load_from_name(name: str) -> bool:
    # ponytail: runtime control/status is handled before LLM; keep tools query-only.
    return False


def _defer_load_from_name(name: str) -> bool:
    return name in {
        "approve_pending_action",
        "delegate_task",
        "mcp_runtime_reload",
        "runtime_event_index_rebuild",
        "worktree_remove",
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


def _compact_description(value: str, *, limit: int = _CARD_DESCRIPTION_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    for prefix in ("超级用户私聊专用：", "超级用户私聊专用:"):
        text = text.removeprefix(prefix)
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)].rstrip(" ，,。;；") + "…"


def _tool_schema_summary(card: SuperuserToolCard, *, fallback: str = "") -> str:
    flags = [card.category, f"risk={card.risk}", f"effect={card.side_effect}"]
    if card.read_only:
        flags.append("read_only")
    if card.destructive:
        flags.append("destructive")
    if card.background_capable:
        flags.append("background")
    if card.produces_artifacts:
        flags.append("artifact")
    description = _compact_description(card.description or fallback)
    summary = f"{card.name}: {'; '.join(flag for flag in flags if flag)}"
    if description:
        summary = f"{summary}; {description}"
    return _compact_description(summary, limit=_TOOL_SCHEMA_DESCRIPTION_LIMIT)


_CATEGORY_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "agent_run": ("继续", "恢复", "暂停", "状态", "run", "agentrun", "任务"),
    "approval": ("确认", "批准", "拒绝", "取消", "待确认", "approval", "approve"),
    "artifact": ("artifact", "大输出", "长日志", "日志", "diff", "原文", "查看结果"),
    "audit": ("审计", "audit", "历史"),
    "background": ("后台", "长任务", "长期", "压测", "等待", "运行中", "三小时"),
    "delegate": ("并行", "子任务", "子代理", "delegate", "多方向"),
    "engineering_loop": (
        "代码",
        "修复",
        "改造",
        "重构",
        "lsp",
        "语义",
        "工程闭环",
        "验证",
        "测试",
    ),
    "eval": ("测试", "验证", "回归", "验收", "pyright", "ruff", "pytest", "eval"),
    "file": (
        "文件",
        "目录",
        "读取",
        "查看",
        "搜索",
        "代码",
        "日志",
        "配置",
        "grep",
        "rg",
    ),
    "git": ("git", "提交", "commit", "push", "分支", "diff", "status", "仓库"),
    "mcp": ("mcp", "外部工具", "服务器工具"),
    "patch": ("修改", "补丁", "patch", "diff", "应用", "回滚", "修复", "改代码"),
    "plugin_dev": ("插件", "生成插件", "创建插件", "插件开发", "scaffold", "publish"),
    "python": ("python", "脚本", "运行代码", "pyright", "pytest", "编译"),
    "registry": (
        "工具",
        "能力",
        "注册表",
        "tool",
        "可用工具",
        "搜索工具",
        "查工具",
        "注入工具",
        "长尾工具",
    ),
    "runtime": ("事件", "runtime", "轨迹", "observation", "状态投影"),
    "server": ("服务", "进程", "端口", "服务器", "状态", "process"),
    "shell": ("shell", "命令", "终端", "powershell", "执行", "运行", "cmd"),
    "todo": ("todo", "计划", "清单", "步骤", "任务"),
    "uv": ("uv", "依赖", "安装", "sync", "run", "pip"),
    "worktree": ("worktree", "隔离", "工作区", "分支", "沙箱"),
}

_COMPLEX_ENGINEERING_TERMS = (
    "代码",
    "修复",
    "改造",
    "重构",
    "测试",
    "验证",
    "回归",
    "插件",
    "多文件",
    "性能",
    "压测",
    "pyright",
    "ruff",
    "pytest",
)


def _normalize_query(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _tool_relevance_score(card: SuperuserToolCard, *, query: str) -> float:
    if not query:
        return 0.0
    name = card.name.lower()
    category = card.category.lower()
    tags = " ".join(card.tags).lower()
    description = card.description.lower()
    haystack = " ".join([name, category, tags, description])
    score = 0.0
    if name and name in query:
        score += 5.0
    if category and category in query:
        score += 2.0
    for part in name.split("_"):
        if len(part) >= 3 and part in query:
            score += 0.7
    for term in _CATEGORY_QUERY_TERMS.get(category, ()):
        if term and term.lower() in query:
            score += 1.25
    for tag in card.tags:
        lowered = tag.lower()
        if lowered and lowered in query:
            score += 0.8
    for token in _query_tokens(query):
        if len(token) >= 2 and token in haystack:
            score += 0.35
    if score <= 0:
        return 0.0
    if card.todo_relevant and any(term in query for term in ("计划", "步骤", "复杂")):
        score += 0.5
    if card.background_capable and any(
        term in query for term in ("后台", "长期", "等待")
    ):
        score += 1.0
    if card.produces_artifacts and any(
        term in query for term in ("日志", "输出", "diff")
    ):
        score += 0.5
    if card.destructive and not any(
        term in query for term in ("修改", "写", "删除", "执行", "应用", "回滚", "取消")
    ):
        score -= 0.4
    return score


def _query_tokens(query: str) -> tuple[str, ...]:
    normalized = query.replace("_", " ").replace("-", " ")
    return tuple(token for token in normalized.split() if token)


def _selection_limit(*, query: str, explicit: int | None) -> int:
    if explicit is not None:
        return max(4, min(int(explicit or 0), 40))
    if any(term in query for term in _COMPLEX_ENGINEERING_TERMS):
        return 12
    return 8


def _risk_rank(risk: ToolRisk) -> int:
    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(risk, 1)


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


def build_superuser_agent_tool_bundle(
    *,
    message_text: str = "",
    limit: int | None = None,
    include_deferred: bool = False,
) -> SuperuserToolBundle:
    from . import toolsets as _toolsets  # noqa: F401  # import registers toolsets

    return _REGISTRY.build_bundle(
        message_text=message_text,
        limit=limit,
        include_deferred=include_deferred,
    )


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
