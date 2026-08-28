"""Small operation guardrail for the superuser Agent."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from fnmatch import fnmatchcase
import os
from pathlib import Path
import re
from typing import Any, Literal

from ..persistence import read_json, state_path, write_json

Decision = Literal["allow", "ask", "deny"]
PermissionMode = Literal["ask", "read_only", "full_access"]
_PERMISSION_MODES = frozenset({"ask", "read_only", "full_access"})
_WINDOWS_CASE_INSENSITIVE = os.name == "nt"
_CURRENT_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chatinter_superuser_permission_run",
    default="",
)
_CURRENT_MODE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chatinter_superuser_permission_mode",
    default="",
)
_CONVERSATION_GRANTS_PATH = state_path("agent_conversation_grants.json")
_WORKSPACE_SHELL_GRANT = "workspace_non_dangerous_shell"
_GRANTS_CACHE_PATH: Path | None = None
_GRANTS_CACHE_SIGNATURE: tuple[int, int] | None = None
_GRANTS_CACHE: dict[str, list[dict[str, str]]] = {}
_EFFECTIVE_POLICY_SOURCE: dict[str, Any] | None = None
_EFFECTIVE_POLICY_CACHE: dict[str, Any] | None = None

_DEFAULT_POLICY: dict[str, Any] = {
    "preset": "python",
    "default_mode": "ask",
    "dangerous_policy": "ask",
    "allow": [],
    "ask": [],
    "dangerous": [],
    "deny": [],
}
_PYTHON_PRESET: dict[str, tuple[str, ...]] = {
    "allow": (
        "File(@workspace/**)",
        "Shell(pwd)",
        "Shell(cd)",
        "Shell(whoami)",
        "Shell(hostname)",
        "Shell(git status)",
        "Shell(git status *)",
        "Shell(git diff)",
        "Shell(git diff *)",
        "Shell(git branch)",
        "Shell(git branch --list)",
        "Shell(git branch --show-current)",
    ),
    "ask": (
        "Shell(pytest*)",
        "Shell(python -m pytest*)",
        "Shell(py -m pytest*)",
        "Shell(ruff*)",
        "Shell(mypy*)",
        "Shell(uv run*)",
        "Shell(poetry run*)",
    ),
    "dangerous": (
        "Shell(rm *)",
        "Shell(del *)",
        "Shell(rmdir *)",
        "Shell(remove-item *)",
        "Shell(git reset *)",
        "Shell(git clean *)",
        "Shell(git push *)",
        "Shell(pip install*)",
        "Shell(python -m pip install*)",
        "Shell(py -m pip install*)",
        "Shell(uv add*)",
        "Shell(uv lock*)",
        "Shell(poetry add*)",
        "Shell(curl *)",
        "Shell(wget *)",
        "Shell(docker run *)",
        "Shell(docker rm *)",
        "Shell(docker system prune*)",
        "Shell(kill *)",
        "Shell(taskkill *)",
        "Shell(systemctl *)",
        "Shell(service *)",
        "Shell(sc *)",
    ),
    "deny": (
        "File(**/.env*)",
        "File(**/.git)",
        "File(**/.git/**)",
        "File(**/.ssh)",
        "File(**/.ssh/**)",
        "File(**/.aws)",
        "File(**/.aws/**)",
        "File(**/.azure)",
        "File(**/.azure/**)",
        "File(**/.config/gcloud)",
        "File(**/.config/gcloud/**)",
        "File(**/AppData/Roaming/gcloud)",
        "File(**/AppData/Roaming/gcloud/**)",
        "File(**/.kube)",
        "File(**/.kube/**)",
        "File(**/.docker/config.json)",
        "File(**/secrets)",
        "File(**/secrets/**)",
        "File(**/.secrets)",
        "File(**/.secrets/**)",
        "Shell(shutdown*)",
        "Shell(reboot*)",
        "Shell(poweroff*)",
        "Shell(halt*)",
        "Shell(mkfs*)",
        "Shell(format *)",
    ),
}
_PRESETS = {"python": _PYTHON_PRESET, "none": {}}


@dataclass(frozen=True)
class PermissionResult:
    decision: Decision
    reason: str
    matched_pattern: str = ""
    section: str = ""
    grant_key: str = ""


def set_current_permission_run(run_id: str | None) -> contextvars.Token[str]:
    return _CURRENT_RUN_ID.set(str(run_id or ""))


def reset_current_permission_run(token: contextvars.Token[str]) -> None:
    _CURRENT_RUN_ID.reset(token)


def set_current_permission_mode(mode: str | None) -> contextvars.Token[str]:
    return _CURRENT_MODE.set(str(mode or ""))


def reset_current_permission_mode(token: contextvars.Token[str]) -> None:
    _CURRENT_MODE.reset(token)


def get_current_permission_mode() -> PermissionMode:
    return _resolve_permission_mode()


def get_default_permission_mode() -> PermissionMode:
    policy = _effective_policy()
    return _resolve_permission_mode(
        str(policy.get("default_mode", "ask") or "ask"),
        policy=policy,
    )


def resolve_permission_mode(mode: str | None) -> PermissionMode:
    return _resolve_permission_mode(mode)


def grant_conversation_permission(
    run_id: str,
    *,
    section: str,
    grant_key: str,
) -> bool:
    run = str(run_id or "").strip()
    scope = str(section or "").strip()
    key = str(grant_key or "").strip()
    if not run or not scope or not key:
        return False
    grants = _conversation_grants()
    values = grants.get(run)
    entries = list(values) if isinstance(values, list) else []
    item = {"section": scope, "grant_key": key}
    if item not in entries:
        entries.append(item)
    grants[run] = entries[-100:]
    _save_conversation_grants(grants)
    return True


def clear_conversation_permissions(run_id: str) -> None:
    run = str(run_id or "").strip()
    if not run:
        return
    grants = _conversation_grants()
    if run not in grants:
        return
    grants.pop(run, None)
    _save_conversation_grants(grants)


def conversation_has_workspace_shell_grant(run_id: str) -> bool:
    run = str(run_id or "").strip()
    if not run:
        return False
    grants = _conversation_grants()
    entries = grants.get(run)
    return isinstance(entries, list) and {
        "section": "shell",
        "grant_key": _WORKSPACE_SHELL_GRANT,
    } in entries


def permission_reason_text(result: PermissionResult) -> str:
    if result.decision == "deny":
        if result.reason == "hard_floor_deny":
            return "该操作可能造成不可恢复的系统破坏，已被安全边界拒绝"
        if result.reason == "read_only_mode_deny":
            return "当前处于只读模式，该操作不会执行"
        if result.reason == "opaque_shell_wrapper_deny":
            return "该命令包含无法检查的编码 Shell 内容，已被拒绝"
        if result.reason == "dangerous_policy_deny":
            return "该命令属于危险操作，当前配置禁止执行"
        if result.reason == "default_deny":
            return "该操作不在允许范围内"
        return "该操作命中了当前禁止规则"
    if result.decision == "ask":
        if result.reason == "dangerous_operation":
            return "该命令可能删除、发布或改变系统状态，需要你的确认"
        if result.reason == "active_task_requires_approval":
            return "该操作会创建或更改可在未来主动运行的任务，需要你的确认"
        return "该操作会修改文件、进程或外部状态，需要你的确认"
    return "当前权限模式允许该操作"


def decide_shell(
    command: str,
    *,
    cwd: str | None = None,
    mode: str | None = None,
) -> PermissionResult:
    policy = _effective_policy()
    value = _normalize_text(command)
    permission_mode = _resolve_permission_mode(mode, policy=policy)

    # Deny rules (hard floor, opaque wrappers, explicit deny, dangerous=deny)
    # are evaluated before any mode short-circuit: full_access must never
    # permit an unrecoverable-destruction command.
    denied = _shell_command_deny(command, policy=policy)
    if denied is not None:
        return denied

    if permission_mode == "full_access":
        return PermissionResult(
            "allow", "full_access_mode_allow", section="shell", grant_key=value
        )

    cwd_is_safe = _cwd_is_within_workspace(cwd)
    is_readonly = cwd_is_safe and _is_builtin_readonly_shell_command(command)
    if permission_mode == "read_only":
        if is_readonly:
            return PermissionResult(
                "allow", "builtin_readonly_allow", section="shell", grant_key=value
            )
        return PermissionResult("deny", "read_only_mode_deny", section="shell")

    dangerous = _matched_shell_rule(command, policy.get("dangerous", ()))
    if dangerous:
        if str(policy.get("dangerous_policy", "ask")).strip().lower() == "deny":
            return PermissionResult("deny", "dangerous_policy_deny", dangerous)
        return PermissionResult("ask", "dangerous_operation", dangerous)

    allowed = _matched_shell_rule(command, policy.get("allow", ()))
    if cwd_is_safe and allowed and _is_simple_shell_command(command):
        if not _allowed_shell_has_side_effect_flag(command):
            return PermissionResult(
                "allow", "matched_allow", allowed, "shell", allowed
            )
    if is_readonly:
        return PermissionResult(
            "allow", "builtin_readonly_allow", section="shell", grant_key=value
        )

    ask_pattern = _matched_shell_rule(command, policy.get("ask", ()))
    pending = PermissionResult(
        "ask",
        "matched_ask" if ask_pattern else "shell_requires_approval",
        ask_pattern,
        "shell",
        ask_pattern or _WORKSPACE_SHELL_GRANT,
    )
    if not cwd_is_safe:
        return pending
    return _apply_conversation_grant(pending)


def decide_file_read(path: str, *, mode: str | None = None) -> PermissionResult:
    policy = _effective_policy()
    permission_mode = _resolve_permission_mode(mode, policy=policy)
    denied = _file_path_deny(path, policy=policy)
    if denied is not None:
        return denied
    if permission_mode == "full_access":
        return PermissionResult("allow", "full_access_mode_allow")
    matched = _matched_file_rule(path, policy.get("allow", ()))
    if matched:
        return PermissionResult("allow", "matched_allow", matched)
    return PermissionResult("deny", "default_deny")


def decide_file_write(path: str, *, mode: str | None = None) -> PermissionResult:
    policy = _effective_policy()
    value = _normalize_path(path)
    permission_mode = _resolve_permission_mode(mode, policy=policy)
    denied = _file_path_deny(path, policy=policy)
    if denied is not None:
        return denied
    if permission_mode == "full_access":
        return PermissionResult(
            "allow", "full_access_mode_allow", section="file", grant_key=value
        )
    if permission_mode == "read_only":
        return PermissionResult(
            "deny", "read_only_mode_deny", section="file", grant_key=value
        )
    matched = _matched_file_rule(path, policy.get("allow", ()))
    if matched:
        return PermissionResult("allow", "matched_allow", matched, "file", matched)
    ask_pattern = _matched_file_rule(path, policy.get("ask", ()))
    pending = PermissionResult(
        "ask",
        "matched_ask" if ask_pattern else "default_ask",
        ask_pattern,
        "file",
        ask_pattern or value,
    )
    return _apply_conversation_grant(pending)


def decide_active_task(
    action: str,
    *,
    mode: str | None = None,
) -> PermissionResult:
    normalized_action = _normalize_text(action).casefold()
    if normalized_action == "list":
        return PermissionResult("allow", "read_only_operation")
    if normalized_action not in {
        "create",
        "update",
        "pause",
        "resume",
        "delete",
        "run_now",
        "rotate_webhook",
    }:
        return PermissionResult("deny", "default_deny", section="active_task")
    permission_mode = _resolve_permission_mode(mode)
    if permission_mode == "full_access":
        return PermissionResult(
            "allow",
            "full_access_mode_allow",
            section="active_task",
        )
    if permission_mode == "read_only":
        return PermissionResult(
            "deny",
            "read_only_mode_deny",
            section="active_task",
        )
    return PermissionResult(
        "ask",
        "active_task_requires_approval",
        section="active_task",
    )


def _resolve_permission_mode(
    requested: str | None = None,
    *,
    policy: dict[str, Any] | None = None,
) -> PermissionMode:
    current_policy = policy or _effective_policy()
    configured_default = str(
        current_policy.get("default_mode", "ask") or "ask"
    ).strip().lower()
    if configured_default not in _PERMISSION_MODES:
        configured_default = "ask"
    # Precedence: explicit request > active session mode > configured default.
    # A full_access default must not override an explicit /只读模式 switch.
    raw = requested
    if raw is None or not str(raw).strip():
        raw = _CURRENT_MODE.get().strip() or configured_default
    normalized = str(raw or "").strip().lower()
    return normalized if normalized in _PERMISSION_MODES else "ask"  # type: ignore[return-value]


def _apply_conversation_grant(result: PermissionResult) -> PermissionResult:
    if result.decision != "ask" or not result.section or not result.grant_key:
        return result
    run_id = _CURRENT_RUN_ID.get().strip()
    if not run_id:
        return result
    entries = _conversation_grants().get(run_id)
    expected = {"section": result.section, "grant_key": result.grant_key}
    if not isinstance(entries, list) or expected not in entries:
        return result
    return PermissionResult(
        "allow",
        "conversation_grant",
        result.matched_pattern,
        result.section,
        result.grant_key,
    )


def expand_shell_command_candidates(command: str) -> tuple[str, ...]:
    """Return original, compound, unwrapped, and canonical command forms."""

    root = _normalize_text(command)
    if not root:
        return ()
    pending = [root]
    candidates: list[str] = []
    seen: set[str] = set()
    while pending:
        candidate = _normalize_text(pending.pop(0))
        identity = candidate.casefold()
        if not candidate or identity in seen:
            continue
        seen.add(identity)
        candidates.append(candidate)
        canonical = _canonical_command(candidate)
        if canonical and canonical.casefold() not in seen:
            pending.append(canonical)
        if wrapped := _unwrap_shell_command(candidate):
            pending.append(wrapped)
        segments = [
            _strip_outer_quotes(part.lstrip(" \t([{"))
            for part in re.split(r"[;&|\r\n]+", candidate)
            if part.strip()
        ]
        pending.extend(segment for segment in segments if segment != candidate)
    return tuple(candidates)


def hard_floor_command_deny(command: str) -> PermissionResult | None:
    value = _normalize_text(command).lower()
    if not value:
        return None
    for candidate in expand_shell_command_candidates(command):
        if _matches_hard_floor(candidate.lower()):
            return PermissionResult("deny", "hard_floor_deny", value[:120])
    return None


def shell_command_deny(command: str) -> PermissionResult | None:
    return _shell_command_deny(command, policy=_effective_policy())


def file_path_deny(path: str) -> PermissionResult | None:
    return _file_path_deny(path, policy=_effective_policy())


def _file_path_deny(path: str, *, policy: dict[str, Any]) -> PermissionResult | None:
    matched = _matched_file_rule(path, policy.get("deny", ()))
    return PermissionResult("deny", "matched_deny", matched) if matched else None


def _shell_command_deny(
    command: str,
    *,
    policy: dict[str, Any],
) -> PermissionResult | None:
    candidates = expand_shell_command_candidates(command)
    for candidate in candidates:
        if _is_encoded_powershell_command(candidate):
            return PermissionResult(
                "deny",
                "opaque_shell_wrapper_deny",
                candidate[:120],
                section="shell",
            )
    hard_floor = hard_floor_command_deny(command)
    if hard_floor is not None:
        return hard_floor
    matched = _matched_shell_rule(command, policy.get("deny", ()))
    if matched:
        return PermissionResult("deny", "matched_deny", matched)
    dangerous = _matched_shell_rule(command, policy.get("dangerous", ()))
    dangerous_policy = str(policy.get("dangerous_policy", "ask")).strip().lower()
    if dangerous and dangerous_policy == "deny":
        return PermissionResult("deny", "dangerous_policy_deny", dangerous)
    return None


def _matched_file_rule(path: str, rules: Any) -> str:
    value = _normalize_path(path)
    compared_value = value.casefold() if _WINDOWS_CASE_INSENSITIVE else value
    for rule in _patterns(rules):
        kind, pattern = _parse_rule(rule)
        if kind != "file":
            continue
        expanded = _expand_workspace_pattern(pattern)
        compared = expanded.casefold() if _WINDOWS_CASE_INSENSITIVE else expanded
        if _match(compared_value, compared):
            return rule
    return ""


def _matched_shell_rule(command: str, rules: Any) -> str:
    values = tuple(item.casefold() for item in expand_shell_command_candidates(command))
    for rule in _patterns(rules):
        kind, pattern = _parse_rule(rule)
        if kind != "shell":
            continue
        compared = _normalize_text(pattern).casefold()
        if any(_match(value, compared) for value in values):
            return rule
    return ""


def _parse_rule(rule: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*(File|Shell)\((.*)\)\s*", rule, flags=re.IGNORECASE)
    if not match:
        return "", ""
    return match.group(1).casefold(), match.group(2).strip()


def _expand_workspace_pattern(pattern: str) -> str:
    marker = "@workspace"
    if pattern.casefold().startswith(marker):
        suffix = pattern[len(marker) :].replace("\\", "/")
        return Path.cwd().resolve().as_posix().rstrip("/") + suffix
    return pattern.replace("\\", "/")


def _canonical_command(value: str) -> str:
    match = re.match(r'^\s*("[^"]+"|\'[^\']+\'|\S+)(.*)$', value)
    if not match:
        return ""
    executable = (
        _strip_outer_quotes(match.group(1)).replace("\\", "/").rsplit("/", 1)[-1]
    )
    executable = re.sub(r"\.(?:exe|cmd|bat|com)$", "", executable, flags=re.IGNORECASE)
    rest = match.group(2).strip()
    return _normalize_text(f"{executable} {rest}" if rest else executable)


def _unwrap_shell_command(value: str) -> str:
    cmd_match = re.match(
        r'^"?cmd(?:\.exe)?"?\s+(?:/[a-z](?::\S+)?\s+)*?/[ck]'
        r'(?:\s+|(?=["\']))(.+)$',
        value,
        flags=re.IGNORECASE,
    )
    if cmd_match:
        return _strip_outer_quotes(cmd_match.group(1).strip())
    powershell_match = re.match(
        r'^"?(?:powershell|pwsh)(?:\.exe)?"?\s+.*?'
        r"(?:-|/)(?:command|c)(?:\s+|:)(.+)$",
        value,
        flags=re.IGNORECASE,
    )
    if powershell_match:
        return _strip_outer_quotes(powershell_match.group(1).strip())
    posix_match = re.match(
        r'^"?(?:sh|bash|zsh)"?\s+-[a-z]*c[a-z]*\s+(.+)$',
        value,
        flags=re.IGNORECASE,
    )
    if posix_match:
        return _strip_outer_quotes(posix_match.group(1).strip())
    return ""


def _is_encoded_powershell_command(value: str) -> bool:
    if not re.match(
        r'^\s*"?(?:powershell|pwsh)(?:\.exe)?"?(?:\s|$)',
        value,
        flags=re.IGNORECASE,
    ):
        return False
    options = re.findall(
        r'(?:^|\s)["\']?[-/]([a-z]+)(?=["\']?(?:\s|[:=]|$))',
        value,
        flags=re.IGNORECASE,
    )
    return any("encodedcommand".startswith(option.casefold()) for option in options)


def _strip_outer_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def _matches_hard_floor(value: str) -> bool:
    if "sudo -s" in value and "|" in value:
        return True
    if value.startswith(("shutdown", "reboot", "poweroff", "halt")):
        return True
    if value.startswith(("mkfs", "format ")):
        return True
    if value.startswith("rm ") and "-rf" in value and "/" in value.split():
        return True
    if value.startswith("dd ") and _dd_writes_disk(value):
        return True
    return False


def _dd_writes_disk(value: str) -> bool:
    return any(
        part.startswith("of=/dev/")
        or part.startswith("of=\\\\.\\")
        or part.startswith("of=//./")
        for part in value.split()
    )


def _is_builtin_readonly_shell_command(command: str) -> bool:
    if not _is_simple_shell_command(command):
        return False
    value = _canonical_command(_normalize_text(command)).casefold()
    parts = value.split()
    if not parts:
        return False
    if parts[0] in {"pwd", "cd", "whoami", "hostname", "df", "free", "uptime"}:
        return len(parts) == 1 or parts[0] in {"df", "free", "uptime"}
    if parts[0] != "git" or len(parts) < 2:
        return False
    if parts[1:] in (
        ["remote", "-v"],
        ["branch"],
        ["branch", "--list"],
        ["branch", "--show-current"],
    ):
        return True
    if parts[1] == "status":
        return True
    return parts[1] == "diff" and not _allowed_shell_has_side_effect_flag(value)


def _is_simple_shell_command(command: str) -> bool:
    value = _normalize_text(command)
    return bool(value) and not any(
        token in value for token in (";", "&", "|", ">", "<", "\n", "\r")
    ) and not _unwrap_shell_command(value)


def _allowed_shell_has_side_effect_flag(command: str) -> bool:
    value = _normalize_text(command).casefold()
    return any(token in value for token in ("--output", "--ext-diff", "--fix"))


def _cwd_is_within_workspace(cwd: str | None) -> bool:
    if not cwd:
        return True
    try:
        return Path(cwd).resolve().is_relative_to(Path.cwd().resolve())
    except (OSError, RuntimeError):
        return False


def _effective_policy() -> dict[str, Any]:
    global _EFFECTIVE_POLICY_CACHE, _EFFECTIVE_POLICY_SOURCE

    raw = _load_policy()
    if raw == _EFFECTIVE_POLICY_SOURCE and _EFFECTIVE_POLICY_CACHE is not None:
        return _EFFECTIVE_POLICY_CACHE
    preset_name = str(raw.get("preset", "python") or "python").strip().lower()
    preset = _PRESETS.get(preset_name, _PYTHON_PRESET)
    result = {
        "preset": preset_name,
        "default_mode": str(raw.get("default_mode", "ask") or "ask"),
        "dangerous_policy": str(raw.get("dangerous_policy", "ask") or "ask"),
    }
    for key in ("allow", "ask", "dangerous", "deny"):
        result[key] = [*_patterns(preset.get(key, ())), *_patterns(raw.get(key, ()))]
    _EFFECTIVE_POLICY_SOURCE = raw
    _EFFECTIVE_POLICY_CACHE = result
    return result


def _conversation_grants() -> dict[str, list[dict[str, str]]]:
    global _GRANTS_CACHE, _GRANTS_CACHE_PATH, _GRANTS_CACHE_SIGNATURE

    path = _CONVERSATION_GRANTS_PATH
    signature = _file_signature(path)
    if path == _GRANTS_CACHE_PATH and signature == _GRANTS_CACHE_SIGNATURE:
        return {
            run_id: [dict(item) for item in entries]
            for run_id, entries in _GRANTS_CACHE.items()
        }
    raw = read_json(path, {})
    source = raw if isinstance(raw, dict) else {}
    grants = {
        str(run_id): [
            {
                "section": str(item.get("section", "")),
                "grant_key": str(item.get("grant_key", "")),
            }
            for item in entries
            if isinstance(item, dict)
        ]
        for run_id, entries in source.items()
        if isinstance(entries, list)
    }
    _GRANTS_CACHE_PATH = path
    _GRANTS_CACHE_SIGNATURE = signature
    _GRANTS_CACHE = grants
    return {
        run_id: [dict(item) for item in entries]
        for run_id, entries in grants.items()
    }


def _save_conversation_grants(
    grants: dict[str, list[dict[str, str]]],
) -> None:
    global _GRANTS_CACHE, _GRANTS_CACHE_PATH, _GRANTS_CACHE_SIGNATURE

    write_json(_CONVERSATION_GRANTS_PATH, grants)
    _GRANTS_CACHE_PATH = _CONVERSATION_GRANTS_PATH
    _GRANTS_CACHE_SIGNATURE = _file_signature(_CONVERSATION_GRANTS_PATH)
    _GRANTS_CACHE = {
        run_id: [dict(item) for item in entries]
        for run_id, entries in grants.items()
    }


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _load_policy() -> dict[str, Any]:
    from ..config import get_permission_policy

    return {**_DEFAULT_POLICY, **get_permission_policy()}


def _patterns(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list | tuple | set):
        return []
    return [_normalize_text(str(item or "")) for item in values if str(item or "")]


def _match(value: str, pattern: str) -> bool:
    return bool(pattern and fnmatchcase(value, pattern))


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_path(value: str) -> str:
    try:
        return Path(value).resolve().as_posix()
    except Exception:
        return str(value or "").replace("\\", "/")


__all__ = [
    "PermissionMode",
    "PermissionResult",
    "clear_conversation_permissions",
    "conversation_has_workspace_shell_grant",
    "decide_active_task",
    "decide_file_read",
    "decide_file_write",
    "decide_shell",
    "expand_shell_command_candidates",
    "file_path_deny",
    "get_current_permission_mode",
    "get_default_permission_mode",
    "grant_conversation_permission",
    "hard_floor_command_deny",
    "permission_reason_text",
    "reset_current_permission_mode",
    "reset_current_permission_run",
    "resolve_permission_mode",
    "set_current_permission_mode",
    "set_current_permission_run",
    "shell_command_deny",
]
