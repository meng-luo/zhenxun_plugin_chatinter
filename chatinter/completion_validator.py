"""Lightweight final reply validator for tool-backed turns.

This module does not decide which plugin to call.  It only protects the final
reply invariant: if a turn required real tools, the assistant must not present
tool/action completion unless successful observations exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .route_text import normalize_message_text

_ACTION_OUTPUT_MODES = {"image", "file", "plugin_output", "action"}
_ACTION_SIDE_EFFECTS = {"query", "send", "mutate"}
_COMPLETION_VERBS = (
    "完成",
    "处理",
    "生成",
    "制作",
    "做成",
    "做出",
    "做好",
    "弄好",
    "搞定",
    "查到",
    "查询",
    "执行",
    "发送",
    "发出",
    "抽",
    "翻译",
    "识别",
)


@dataclass(frozen=True)
class CompletionValidationResult:
    ok: bool
    reason: str = ""
    requires_observation: bool = False
    successful_observations: int = 0
    action_like_tools: int = 0


def validate_final_reply(
    *,
    final_text: str,
    tool_obligation: str,
    observations: list[Any] | tuple[Any, ...],
    pending_tasks: list[Any] | tuple[Any, ...],
    tool_map: dict[str, Any],
) -> CompletionValidationResult:
    """Validate final reply against observed tool facts."""

    normalized = normalize_message_text(final_text)
    successful = [
        observation
        for observation in observations
        if bool(getattr(observation, "ok", False))
        and normalize_message_text(str(getattr(observation, "command_id", "") or ""))
    ]
    action_tools = _action_like_tool_count(tool_map)
    claims_action_completion = _looks_like_action_completion(normalized)
    requires_observation = (
        normalize_message_text(tool_obligation) == "required"
        or bool(pending_tasks)
        or bool(action_tools and observations)
        or bool(action_tools and claims_action_completion)
    )
    if not requires_observation:
        return CompletionValidationResult(
            ok=True,
            successful_observations=len(successful),
            action_like_tools=action_tools,
        )
    if successful:
        return CompletionValidationResult(
            ok=True,
            requires_observation=True,
            successful_observations=len(successful),
            action_like_tools=action_tools,
        )
    if claims_action_completion:
        return CompletionValidationResult(
            ok=False,
            reason="final_claim_without_successful_observation",
            requires_observation=True,
            successful_observations=0,
            action_like_tools=action_tools,
        )
    return CompletionValidationResult(
        ok=True,
        requires_observation=requires_observation,
        successful_observations=0,
        action_like_tools=action_tools,
    )


def _action_like_tool_count(tool_map: dict[str, Any]) -> int:
    count = 0
    for tool in tool_map.values():
        binding = getattr(tool, "binding", None)
        candidate = getattr(binding, "candidate", None)
        snapshot = getattr(candidate, "tool", None)
        if snapshot is None:
            continue
        output_mode = normalize_message_text(str(getattr(snapshot, "output_mode", "") or ""))
        side_effect = normalize_message_text(str(getattr(snapshot, "side_effect", "") or ""))
        if output_mode in _ACTION_OUTPUT_MODES or side_effect in _ACTION_SIDE_EFFECTS:
            count += 1
    return count


def _looks_like_action_completion(text: str) -> bool:
    if not text:
        return False
    if any(verb in text for verb in _COMPLETION_VERBS):
        return True
    return False


__all__ = [
    "CompletionValidationResult",
    "validate_final_reply",
]
