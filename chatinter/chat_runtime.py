"""Chat-only runtime boundary for ChatInter.

This module owns dialogue state, chat memory layers, group reply posture and
chat final quality.  It intentionally does not import tool registries, command
schemas, tool obligation, AgentRuntime or route execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chat_dialogue_planner import (
    DialogueState,
    persist_dialogue_state,
)
from .chat_runtime_profile import ChatRuntimeProfile, build_chat_runtime_profile
from .chat_strategy import build_chat_strategy_prompt, build_dialogue_state_prompt
from .response_quality_judge import ResponseQualityJudge, ResponseQualityResult
from .route_text import normalize_message_text


@dataclass(frozen=True)
class ChatIsolationDecision:
    """Whether chat-only state may be exposed to the current model request."""

    allow_prompt_profile: bool
    allow_memory_profile: bool
    allow_quality_judge: bool
    allow_state_persist: bool
    reason: str = "default"


@dataclass(frozen=True)
class ChatPromptContext:
    system_fragments: tuple[str, ...] = ()
    context_xml: str = ""
    tags: dict[str, str] | None = None


class ChatRuntime:
    """Facade for all chat-side evolution.

    Tool/plugin turns can still build the profile for metrics, but prompt
    injection, quality judging and persistence are gated here so plugin
    execution and chat personality never fight over the same prompt.
    """

    @staticmethod
    def isolation_for_frame(frame: Any) -> ChatIsolationDecision:
        scenario = normalize_message_text(str(getattr(frame, "scenario", "") or ""))
        allow_plugin_tools = bool(getattr(frame, "allow_plugin_tools", False))
        allow_agent_tools = bool(getattr(frame, "allow_agent_tools", False))
        exposure_state = normalize_message_text(
            str(getattr(frame, "chat_tool_exposure_state", "") or "unknown")
        )
        if scenario == "private_chat" and not allow_agent_tools:
            return ChatIsolationDecision(
                allow_prompt_profile=True,
                allow_memory_profile=True,
                allow_quality_judge=True,
                allow_state_persist=True,
                reason="private_chat",
            )
        if scenario == "group_plugin_selector":
            if allow_plugin_tools and exposure_state in {"unknown", "plugin_tools_exposed"}:
                return ChatIsolationDecision(
                    allow_prompt_profile=False,
                    allow_memory_profile=False,
                    allow_quality_judge=False,
                    allow_state_persist=False,
                    reason=exposure_state or "plugin_tools_exposed",
                )
            return ChatIsolationDecision(
                allow_prompt_profile=True,
                allow_memory_profile=True,
                allow_quality_judge=True,
                allow_state_persist=True,
                reason="group_chat_no_tool_exposure",
            )
        return ChatIsolationDecision(
            allow_prompt_profile=False,
            allow_memory_profile=False,
            allow_quality_judge=False,
            allow_state_persist=False,
            reason="agent_or_tool_scenario",
        )

    @staticmethod
    def build_profile(frame: Any) -> ChatRuntimeProfile | None:
        try:
            return build_chat_runtime_profile(
                session_key=getattr(frame, "session_key", ""),
                user_id=getattr(frame, "user_id", ""),
                group_id=getattr(frame, "group_id", None),
                message_text=(
                    getattr(frame, "current_message", "")
                    or getattr(frame, "raw_message", "")
                ),
                scenario=getattr(frame, "scenario", ""),
                has_images=bool(
                    getattr(getattr(frame, "event_context", None), "images", ()) or ()
                ),
                has_reply=bool(
                    getattr(
                        getattr(getattr(frame, "event_context", None), "reply", None),
                        "sender_id",
                        "",
                    )
                ),
                is_group=bool(getattr(frame, "group_id", None)),
                previous_state=getattr(frame, "previous_dialogue_state", None),
                dialogue_context_pack=getattr(frame, "dialogue_context_pack", None),
                thread_context=getattr(frame, "thread_context", None),
            )
        except Exception:
            return None

    @classmethod
    def attach_profile(cls, frame: Any) -> ChatRuntimeProfile | None:
        profile = getattr(frame, "chat_runtime_profile", None) or cls.build_profile(frame)
        if profile is None:
            return None
        frame.chat_runtime_profile = profile
        frame.dialogue_plan = profile.dialogue_plan
        frame.dialogue_state = profile.dialogue_state
        frame.previous_dialogue_state = profile.previous_state
        return profile

    @classmethod
    def memory_dialogue_state(cls, frame: Any) -> DialogueState | None:
        decision = cls.isolation_for_frame(frame)
        if not decision.allow_memory_profile:
            return None
        profile = cls.attach_profile(frame)
        return profile.dialogue_state if profile is not None else None

    @classmethod
    def build_prompt_context(
        cls,
        frame: Any,
        *,
        base_context_xml: str,
    ) -> ChatPromptContext:
        decision = cls.isolation_for_frame(frame)
        if not decision.allow_prompt_profile:
            return ChatPromptContext(
                context_xml=strip_dialogue_state_context(base_context_xml),
                tags={"chat_isolation": decision.reason},
            )
        profile = cls.attach_profile(frame)
        if profile is None:
            return ChatPromptContext(
                context_xml=base_context_xml,
                tags={"chat_isolation": "profile_unavailable"},
            )
        fragments: list[str] = []
        strategy_prompt = build_chat_strategy_prompt(profile.dialogue_plan)
        if strategy_prompt:
            fragments.append(strategy_prompt)
        state_prompt = build_dialogue_state_prompt(profile.dialogue_state)
        if state_prompt:
            fragments.append(state_prompt)
        fragments.append(profile.to_context_xml())
        return ChatPromptContext(
            system_fragments=tuple(fragments),
            context_xml=replace_dialogue_state_context(
                base_context_xml,
                profile.dialogue_state,
            ),
            tags={
                "chat_isolation": decision.reason,
                "chat_kind": profile.dialogue_plan.kind,
                "chat_style": profile.dialogue_plan.style,
                "dialogue_tone": profile.dialogue_state.tone,
                "dialogue_emotion": profile.dialogue_state.user_emotion,
                "dialogue_purpose": profile.dialogue_state.dialogue_purpose,
                "reply_posture": profile.dialogue_state.reply_posture,
                "group_atmosphere": profile.dialogue_state.group_atmosphere,
            },
        )

    @classmethod
    def judge_final_reply(
        cls,
        *,
        frame: Any,
        final_text: str,
    ) -> ResponseQualityResult:
        decision = cls.isolation_for_frame(frame)
        if not decision.allow_quality_judge:
            return ResponseQualityResult(action="ok", reason=decision.reason)
        return ResponseQualityJudge.judge_chat_only(
            final_text=final_text,
            original_message=getattr(frame, "current_message", ""),
            dialogue_state=getattr(frame, "dialogue_state", None),
        )

    @classmethod
    def should_persist_dialogue_state(cls, frame: Any) -> bool:
        decision = cls.isolation_for_frame(frame)
        if not decision.allow_state_persist:
            return False
        main_result = getattr(frame, "main_result", None)
        if main_result is None or getattr(frame, "dialogue_state", None) is None:
            return False
        output = getattr(main_result, "output", None)
        if output is None or not bool(getattr(output, "record_chat_feedback", False)):
            return False
        if normalize_message_text(str(getattr(output, "outcome", "") or "")) != "chat_completed":
            return False
        if bool(getattr(main_result, "handled_by_tools", False)):
            return False
        if bool(getattr(main_result, "tool_results", ()) or ()):
            return False
        if bool(getattr(main_result, "executions", ()) or ()):
            return False
        return True

    @classmethod
    def persist_dialogue_state(cls, *, frame: Any, reply_text: str) -> bool:
        if not cls.should_persist_dialogue_state(frame):
            return False
        persist_dialogue_state(
            session_key=getattr(frame, "session_key", ""),
            user_id=getattr(frame, "user_id", ""),
            group_id=getattr(frame, "group_id", None),
            message_text=getattr(frame, "current_message", ""),
            state=getattr(frame, "dialogue_state"),
            outcome=getattr(getattr(frame, "main_result", None), "output").outcome,
            reply_text=reply_text,
        )
        return True


def replace_dialogue_state_context(context_xml: str, dialogue_state: DialogueState) -> str:
    block = dialogue_state.to_xml()
    base = _strip_dialogue_state_block(context_xml)
    return f"{base}\n{block}".strip() if base else block


def strip_dialogue_state_context(context_xml: str) -> str:
    return _strip_dialogue_state_block(context_xml)


def _strip_dialogue_state_block(context_xml: str) -> str:
    import re

    pattern = re.compile(r"\n?<dialogue_state>.*?</dialogue_state>", re.DOTALL)
    return pattern.sub("", str(context_xml or "")).strip()


__all__ = [
    "ChatIsolationDecision",
    "ChatPromptContext",
    "ChatRuntime",
    "replace_dialogue_state_context",
    "strip_dialogue_state_context",
]
