"""Session-local superuser tool presets."""

from __future__ import annotations

from typing import Literal, cast

from ..route_text import normalize_message_text
from .permission_policy import (
    clear_session_permission_mode,
    set_session_permission_mode,
)

ToolPreset = Literal[
    "default", "read_only", "code_edit", "plugin_dev", "server_ops"
]

_VALID_PRESETS = frozenset(
    {"default", "read_only", "code_edit", "plugin_dev", "server_ops"}
)
_SESSION_PRESETS: dict[str, ToolPreset] = {}

_PRESET_CATEGORIES: dict[ToolPreset, frozenset[str]] = {
    "default": frozenset(),
    "read_only": frozenset(
        {
            "agent_run",
            "artifact",
            "audit",
            "eval",
            "file",
            "git",
            "registry",
            "runtime",
            "server",
            "todo",
            "worktree",
        }
    ),
    "code_edit": frozenset(
        {
            "agent_run",
            "artifact",
            "eval",
            "file",
            "git",
            "patch",
            "python",
            "registry",
            "uv",
            "worktree",
        }
    ),
    "plugin_dev": frozenset(
        {
            "agent_run",
            "artifact",
            "eval",
            "file",
            "git",
            "patch",
            "plugin_dev",
            "python",
            "registry",
            "uv",
            "worktree",
        }
    ),
    "server_ops": frozenset(
        {
            "agent_run",
            "artifact",
            "background",
            "python",
            "registry",
            "runtime",
            "server",
            "shell",
        }
    ),
}

_PRESET_PERMISSION_MODE = {
    "read_only": "auto_readonly",
    "code_edit": "auto_readonly",
    "plugin_dev": "auto_readonly",
    "server_ops": "ask_all",
}

_PRESET_LABELS = {
    "default": "默认模式",
    "read_only": "只读模式",
    "code_edit": "改代码模式",
    "plugin_dev": "插件开发模式",
    "server_ops": "服务器排查模式",
}


def set_session_tool_preset(session_key: str, preset: str) -> ToolPreset:
    key = str(session_key or "").strip()
    if not key:
        raise ValueError("session_key is required")
    normalized = _coerce_preset(preset)
    if normalized is None:
        raise ValueError(f"invalid tool preset: {preset}")
    if normalized == "default":
        clear_session_tool_preset(key)
        clear_session_permission_mode(key)
        return normalized
    _SESSION_PRESETS[key] = normalized
    mode = _PRESET_PERMISSION_MODE.get(normalized)
    if mode:
        set_session_permission_mode(key, mode)
    return normalized


def clear_session_tool_preset(session_key: str) -> None:
    _SESSION_PRESETS.pop(str(session_key or "").strip(), None)


def get_session_tool_preset(session_key: str | None) -> ToolPreset:
    key = str(session_key or "").strip()
    return _SESSION_PRESETS.get(key, "default")


def tool_preset_label(preset: str) -> str:
    normalized = _coerce_preset(preset) or "default"
    return _PRESET_LABELS[normalized]


def preset_allows_card(preset: str, card: object) -> bool:
    normalized = _coerce_preset(preset) or "default"
    if normalized == "default":
        return True
    category = normalize_message_text(str(getattr(card, "category", "") or ""))
    if category not in _PRESET_CATEGORIES[normalized]:
        return False
    if normalized == "read_only" and not bool(getattr(card, "read_only", False)):
        return False
    return True


def _coerce_preset(value: str) -> ToolPreset | None:
    normalized = normalize_message_text(value).lower()
    alias = {
        "只读": "read_only",
        "只读模式": "read_only",
        "改代码": "code_edit",
        "改代码模式": "code_edit",
        "插件开发": "plugin_dev",
        "插件开发模式": "plugin_dev",
        "服务器排查": "server_ops",
        "服务器排查模式": "server_ops",
        "server": "server_ops",
        "clear": "default",
        "默认": "default",
        "默认模式": "default",
    }.get(normalized, normalized)
    if alias in _VALID_PRESETS:
        return cast(ToolPreset, alias)
    return None


__all__ = [
    "ToolPreset",
    "clear_session_tool_preset",
    "get_session_tool_preset",
    "preset_allows_card",
    "set_session_tool_preset",
    "tool_preset_label",
]
