"""Configurable allow/ask/deny policy for superuser agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

from zhenxun.services import logger

Decision = Literal["allow", "ask", "deny"]

_CONFIG_PATH = Path("data/configs/chatinter_agent_permissions.yaml")
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
    return _decide_by_patterns(
        value=_normalize_path(path),
        allow=policy.get("allow_read", []),
        ask=policy.get("ask_read", []),
        deny=policy.get("deny", []),
        default="ask",
    )


def decide_file_write(path: str) -> PermissionResult:
    policy = _policy_section("file")
    return _decide_by_patterns(
        value=_normalize_path(path),
        allow=policy.get("allow_write", []),
        ask=policy.get("ask_write", []),
        deny=policy.get("deny", []),
        default="ask",
    )


def _decide_command(
    section: str, command: str, *, default: Decision
) -> PermissionResult:
    policy = _policy_section(section)
    return _decide_by_patterns(
        value=_normalize_text(command),
        allow=policy.get("allow", []),
        ask=policy.get("ask", []),
        deny=policy.get("deny", []),
        default=default,
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
    "PermissionResult",
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
]
