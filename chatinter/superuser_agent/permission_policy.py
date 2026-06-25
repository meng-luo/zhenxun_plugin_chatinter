"""Configurable allow/ask/deny policy for superuser agent tools."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal, cast

from loguru import logger

Decision = Literal["allow", "ask", "deny"]
PermissionMode = Literal["default", "ask_all", "auto_readonly", "bypass"]

_CONFIG_PATH = Path("data/configs/chatinter_agent_permissions.yaml")
_VALID_PERMISSION_MODES = frozenset({"default", "ask_all", "auto_readonly", "bypass"})
_CURRENT_SESSION_KEY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chatinter_superuser_permission_session",
    default="",
)
# ponytail: process-memory override; add TTL only if stale sessions become measurable.
_SESSION_PERMISSION_MODES: dict[str, PermissionMode] = {}
_DEFAULT_POLICY: dict[str, Any] = {
    "superuser_agent": {
        "shell": {
            "allow": [
                "pwd",
                "cd",
                "dir",
                "ls",
                "echo*",
                "type *",
                "cat *",
                "docker ps",
                "docker ps *",
                "docker info",
                "docker info *",
                "docker stats --no-stream",
                "docker stats --no-stream *",
                "df",
                "df *",
                "free",
                "free *",
                "uptime",
                "uptime *",
                "whoami",
                "hostname",
                "hostname *",
            ],
            "ask": [],
            "deny": [
                "git *",
                "uv *",
                "uvx *",
                "python*",
                "pip *",
                "systemctl *",
                "service *",
                "pm2 *",
                "screen *",
                "taskkill *",
                "kill *",
                "netstat *",
                "ss *",
                "lsof *",
                "ps *",
                "top*",
                "rm -rf*",
                "del /s*",
                "rmdir /s*",
                "format *",
                "shutdown*",
                "reboot*",
                "poweroff*",
            ],
        },
        "git": {
            "allow": [
                "git status*",
                "git diff*",
                "git log*",
                "git branch*",
                "git worktree list*",
                "git show*",
                "git remote -v*",
                "git rev-parse*",
                "git ls-files*",
            ],
            "ask": [
                "git worktree add*",
                "git worktree remove*",
                "git worktree prune*",
                "git add*",
                "git commit*",
                "git push*",
                "git pull*",
                "git fetch*",
                "git checkout*",
                "git switch*",
                "git merge*",
                "git rebase*",
                "git stash*",
                "git restore*",
                "git clean -n*",
            ],
            "deny": [
                "git reset --hard*",
                "git clean -f*",
                "git clean -fd*",
                "git filter-branch*",
                "git gc --prune=now*",
            ],
        },
        "server": {
            "allow": [
                "server_status",
                "mcp_runtime_status*",
                "mcp_runtime_refresh*",
                "process_list*",
                "disk_usage*",
                "systemctl status*",
                "service * status*",
                "pm2 status*",
                "pm2 list*",
                "screen -ls*",
                "tasklist*",
                "ps *",
                "netstat *",
                "ss *",
                "lsof *",
                "df *",
                "free *",
                "uptime*",
            ],
            "ask": [
                "mcp_runtime_reload*",
                "systemctl restart*",
                "systemctl start*",
                "systemctl stop*",
                "service * restart*",
                "service * start*",
                "service * stop*",
                "pm2 restart*",
                "pm2 start*",
                "pm2 stop*",
                "screen -S * -X *",
                "taskkill *",
                "kill *",
            ],
            "deny": [
                "shutdown*",
                "reboot*",
                "poweroff*",
                "format *",
                "rm -rf*",
                "del /s*",
                "rmdir /s*",
                "kill -9*",
                "taskkill /f*",
            ],
        },
        "plugin_dev": {
            "allow": [
                "plugin_dev_inspect*",
                "plugin_dev_validate_name*",
            ],
            "ask": [
                "plugin_dev_scaffold*",
                "plugin_dev_write_file*",
                "plugin_dev_publish*",
            ],
            "deny": [
                "plugin_dev_delete*",
                "plugin_dev_remove*",
                "plugin_dev_overwrite_builtin*",
            ],
        },
        "patch": {
            "allow": [
                "patch_prepare*",
                "patch_show*",
            ],
            "ask": [
                "patch_apply*",
                "patch_rollback*",
            ],
            "deny": [],
        },
        "background": {
            "allow": [
                "background_task_status*",
                "background_task_list*",
            ],
            "ask": [
                "background_task_start*",
                "background_task_cancel*",
            ],
            "deny": [],
        },
        "uv": {
            "allow": [
                "uv --version*",
                "uv tree*",
                "uv pip list*",
                "uv lock*",
                "uv run ruff*",
                "uv run pyright*",
            ],
            "ask": [
                "uv sync*",
                "uv add*",
                "uv remove*",
                "uv pip install*",
                "uv pip uninstall*",
                "uv run*",
                "uvx*",
            ],
            "deny": [],
        },
        "python": {
            "allow": [
                "python --version*",
                "python -V*",
                "python -m py_compile*",
            ],
            "ask": [
                "python_exec*",
                "python_module*",
                "python *",
            ],
            "deny": [],
        },
        "eval": {
            "allow": [
                "engineering_loop_start*",
                "engineering_loop_status*",
                "engineering_lsp_read*",
                "semantic_patch_plan*",
                "engineering_loop_bind*",
                "engineering_failure_diagnose*",
                "engineering_eval_gate*",
                "engineering_loop_complete*",
                "engineering_eval_plan*",
                "engineering_eval_status*",
            ],
            "ask": [
                "engineering_eval_run*",
                "engineering_eval_rollback*",
            ],
            "deny": [],
        },
        "file": {
            "allow_read": ["C:/zhenxun/**"],
            "ask_read": [],
            "allow_write": [],
            "ask_write": ["C:/zhenxun/**"],
            "deny": [
                "C:/Windows/**",
                "C:/Program Files/**",
                "C:/Program Files (x86)/**",
                "C:/Users/*/.ssh/**",
                "C:/Users/*/AppData/Roaming/Microsoft/Windows/PowerShell/**",
            ],
        },
    }
}


@dataclass(frozen=True)
class PermissionResult:
    decision: Decision
    reason: str
    matched_pattern: str = ""


def set_current_permission_session(
    session_key: str | None,
) -> contextvars.Token[str]:
    """Bind the active superuser permission session for this task."""

    return _CURRENT_SESSION_KEY.set(str(session_key or ""))


def reset_current_permission_session(token: contextvars.Token[str]) -> None:
    _CURRENT_SESSION_KEY.reset(token)


def set_session_permission_mode(
    session_key: str,
    mode: str,
) -> PermissionMode:
    """Override permission mode for one session until process restart/clear."""

    normalized = _coerce_permission_mode(mode)
    if normalized is None:
        raise ValueError(f"invalid permission mode: {mode}")
    key = str(session_key or "").strip()
    if not key:
        raise ValueError("session_key is required")
    _SESSION_PERMISSION_MODES[key] = normalized
    return normalized


def clear_session_permission_mode(session_key: str) -> None:
    _SESSION_PERMISSION_MODES.pop(str(session_key or "").strip(), None)


def get_session_permission_mode(
    session_key: str | None = None,
) -> PermissionMode | None:
    key = str(session_key if session_key is not None else _CURRENT_SESSION_KEY.get())
    key = key.strip()
    return _SESSION_PERMISSION_MODES.get(key) if key else None


def decide_shell(command: str) -> PermissionResult:
    return _decide_command("shell", command, default="ask")


def decide_git(command: str) -> PermissionResult:
    return _decide_command("git", command, default="ask")


def decide_server(command: str) -> PermissionResult:
    return _decide_command("server", command, default="ask")


def decide_plugin_dev(command: str) -> PermissionResult:
    return _decide_command("plugin_dev", command, default="ask")


def decide_patch(command: str) -> PermissionResult:
    return _decide_command("patch", command, default="ask")


def decide_background(command: str) -> PermissionResult:
    return _decide_command("background", command, default="ask")


def decide_uv(command: str) -> PermissionResult:
    return _decide_command("uv", command, default="ask")


def decide_python(command: str) -> PermissionResult:
    return _decide_command("python", command, default="ask")


def decide_eval(command: str) -> PermissionResult:
    return _decide_command("eval", command, default="ask")


def decide_file_read(path: str) -> PermissionResult:
    policy = _policy_section("file")
    result = _decide_by_patterns(
        value=_normalize_path(path),
        allow=policy.get("allow_read", []),
        ask=policy.get("ask_read", []),
        deny=policy.get("deny", []),
        default="ask",
    )
    return _apply_permission_mode(result, read_only=True)


def decide_file_write(path: str) -> PermissionResult:
    policy = _policy_section("file")
    result = _decide_by_patterns(
        value=_normalize_path(path),
        allow=policy.get("allow_write", []),
        ask=policy.get("ask_write", []),
        deny=policy.get("deny", []),
        default="ask",
    )
    return _apply_permission_mode(result, read_only=False)


def _decide_command(
    section: str, command: str, *, default: Decision
) -> PermissionResult:
    hard_floor = _hard_floor_command_deny(command)
    if hard_floor is not None:
        return hard_floor
    policy = _policy_section(section)
    result = _decide_by_patterns(
        value=_normalize_text(command),
        allow=policy.get("allow", []),
        ask=policy.get("ask", []),
        deny=policy.get("deny", []),
        default=default,
    )
    if (
        section == "shell"
        and result.reason == "default_ask"
        and _is_builtin_readonly_shell_command(command)
    ):
        result = PermissionResult("allow", "builtin_readonly_allow")
    return _apply_permission_mode(
        result,
        read_only=_is_read_only_command(section, command, result),
    )


def _policy_section(name: str) -> dict[str, Any]:
    section = _load_policy().get("superuser_agent", {}).get(name, {})
    return section if isinstance(section, dict) else {}


def _decide_by_patterns(
    *,
    value: str,
    allow: Any,
    ask: Any,
    deny: Any,
    default: Decision,
) -> PermissionResult:
    for pattern in _patterns(deny):
        if _match(value, pattern):
            return PermissionResult("deny", "matched_deny", pattern)
    for pattern in _patterns(allow):
        if _match(value, pattern):
            return PermissionResult("allow", "matched_allow", pattern)
    for pattern in _patterns(ask):
        if _match(value, pattern):
            return PermissionResult("ask", "matched_ask", pattern)
    return PermissionResult(default, "default_" + default)


def _hard_floor_command_deny(command: str) -> PermissionResult | None:
    value = _normalize_text(command).lower()
    if not value:
        return None
    if _matches_hard_floor(value):
        return PermissionResult("deny", "hard_floor_deny", value[:120])
    return None


def _matches_hard_floor(value: str) -> bool:
    # ponytail: tiny disaster floor; policy config handles ordinary risk.
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
    parts = value.split()
    return any(
        part.startswith("of=/dev/")
        or part.startswith("of=\\\\.\\")
        or part.startswith("of=//./")
        for part in parts
    )


def _permission_mode() -> PermissionMode:
    override = get_session_permission_mode()
    if override is not None:
        return override
    from ..config import get_config_value

    return (
        _coerce_permission_mode(
            str(get_config_value("SUPERUSER_PERMISSION_MODE", "default") or "default")
        )
        or "default"
    )


def _coerce_permission_mode(value: str) -> PermissionMode | None:
    normalized = str(value or "").strip().lower()
    if normalized in _VALID_PERMISSION_MODES:
        return cast(PermissionMode, normalized)
    return None


def _apply_permission_mode(
    result: PermissionResult,
    *,
    read_only: bool,
) -> PermissionResult:
    if result.decision == "deny":
        return result
    mode = _permission_mode()
    if mode == "default":
        return result
    if mode == "ask_all":
        return PermissionResult("ask", "mode_ask_all", result.matched_pattern)
    if mode == "bypass":
        return PermissionResult("allow", "mode_bypass", result.matched_pattern)
    if mode == "auto_readonly":
        decision: Decision = "allow" if read_only else "ask"
        return PermissionResult(decision, "mode_auto_readonly", result.matched_pattern)
    return result


def _is_read_only_command(
    section: str,
    command: str,
    result: PermissionResult,
) -> bool:
    if result.decision != "allow":
        return False
    value = _normalize_text(command).lower()
    if section == "shell":
        return value.startswith(("pwd", "cd", "dir", "ls", "type ", "cat ")) or (
            _is_builtin_readonly_shell_command(value)
        )
    if section == "git":
        return True
    if section == "server":
        return not value.startswith("mcp_runtime_refresh")
    if section == "plugin_dev":
        return value.startswith(("plugin_dev_inspect", "plugin_dev_validate_name"))
    if section == "patch":
        return value.startswith("patch_show")
    if section == "background":
        return value.startswith(("background_task_status", "background_task_list"))
    if section == "uv":
        return value.startswith(("uv --version", "uv tree", "uv pip list"))
    if section == "python":
        return value.startswith(("python --version", "python -v"))
    if section == "eval":
        return value.startswith(
            (
                "engineering_eval_gate",
                "engineering_eval_status",
                "engineering_loop_status",
                "engineering_lsp_read",
                "engineering_failure_diagnose",
                "semantic_patch_plan",
            )
        )
    return False


def _is_builtin_readonly_shell_command(command: str) -> bool:
    value = _normalize_text(command).lower()
    if not value or any(token in value for token in (";", "&&", "||", "|", "\n")):
        return False
    parts = value.split()
    if not parts:
        return False
    if parts[0] in {"df", "free", "uptime"}:
        return True
    if parts[0] == "whoami":
        return len(parts) == 1
    if parts[0] == "hostname":
        return True
    if len(parts) < 2 or parts[0] != "docker":
        return False
    if parts[1] in {"ps", "info"}:
        return True
    return parts[1] == "stats" and "--no-stream" in parts[2:]


def _load_policy() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return _DEFAULT_POLICY
    try:
        import yaml

        data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"ChatInter agent 权限配置读取失败，使用默认策略: {exc}")
        return _DEFAULT_POLICY
    if not isinstance(data, dict):
        return _DEFAULT_POLICY
    return _deep_merge(_DEFAULT_POLICY, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        old_value = result.get(key)
        if isinstance(old_value, dict) and isinstance(value, dict):
            result[key] = _deep_merge(old_value, value)
        else:
            result[key] = value
    return result


def _patterns(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
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
    "clear_session_permission_mode",
    "decide_background",
    "decide_eval",
    "decide_file_read",
    "decide_file_write",
    "decide_git",
    "decide_patch",
    "decide_plugin_dev",
    "decide_python",
    "decide_server",
    "decide_shell",
    "decide_uv",
    "get_session_permission_mode",
    "reset_current_permission_session",
    "set_current_permission_session",
    "set_session_permission_mode",
]
