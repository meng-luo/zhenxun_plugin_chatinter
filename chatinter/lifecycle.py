import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any, Literal

from zhenxun.services.log import logger

from .config import CONTEXT_TOKEN_BUDGET

LifecycleStage = Literal[
    "pre_gate",
    "post_gate",
    "before_intent",
    "after_intent",
    "before_route",
    "after_route",
    "before_chat",
    "after_chat",
    "before_agent",
    "after_agent",
    "before_tool",
    "after_tool",
    "on_error",
]

_TOKEN_BUDGET = 2200
_SYSTEM_PROMPT_BUDGET = 800
_SECTION_TRIM_ORDER = (
    "history",
    "history_context",
    "retrieval_knowledge",
    "current_message_layers",
)
_HARD_DROP_ORDER = (
    "history",
    "history_context",
    "retrieval_knowledge",
)
_SMALL_SECTION_TOKEN_THRESHOLD = 48
_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
_HIGH_CONTEXT_BUDGET = 6000
_MIDDLE_CONTEXT_BUDGET = 3500
_BUDGET_ALERT_RATIO = 0.90

try:
    import tiktoken
except Exception:
    tiktoken = None


@lru_cache(maxsize=16)
def _resolve_tokenizer(model_name: str | None):
    if not model_name or tiktoken is None:
        return None
    name = model_name.split("/", 1)[-1].strip()
    if not name:
        return None
    try:
        return tiktoken.encoding_for_model(name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _resolve_budget(model_name: str | None) -> int:
    if CONTEXT_TOKEN_BUDGET > 0:
        return CONTEXT_TOKEN_BUDGET
    if not model_name:
        return _TOKEN_BUDGET
    lower = model_name.lower()
    if any(key in lower for key in ("gpt-5", "128k", "200k", "gemini-2.5")):
        return _HIGH_CONTEXT_BUDGET
    if any(key in lower for key in ("32k", "64k", "qwen-max", "gpt-4.1")):
        return _MIDDLE_CONTEXT_BUDGET
    return _TOKEN_BUDGET


def _estimate_tokens(text: str, model_name: str | None = None) -> int:
    if not text:
        return 0
    tokenizer = _resolve_tokenizer(model_name)
    if tokenizer is not None:
        try:
            encoded = tokenizer.encode(text, disallowed_special=())
            if encoded:
                return len(encoded)
        except Exception:
            pass
    token_hits = len(_TOKEN_PATTERN.findall(text))
    return max(1, int(token_hits * 0.9))


def _trim_text_by_budget(text: str, budget: int, model_name: str | None = None) -> str:
    if budget <= 0 or not text:
        return ""
    current_tokens = _estimate_tokens(text, model_name)
    if current_tokens <= budget:
        return text
    ratio = max(0.15, min(1.0, budget / max(current_tokens, 1)))
    cut = max(120, int(len(text) * ratio))
    return text[:cut].rstrip()


def _split_sections(context_xml: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    if not context_xml.strip():
        return sections
    lines = context_xml.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("<") and line.endswith(">") and not line.startswith("</"):
            tag = line[1:-1].strip()
            end_tag = f"</{tag}>"
            section_lines = [lines[index]]
            index += 1
            while index < len(lines):
                section_lines.append(lines[index])
                if lines[index].strip() == end_tag:
                    break
                index += 1
            sections[tag] = section_lines
        index += 1
    return sections


def _compose_sections(sections: dict[str, list[str]]) -> str:
    ordered = []
    for tag in (
        "qq_context",
        "history_context",
        "history",
        "current_message_layers",
        "user_state",
        "retrieval_knowledge",
    ):
        section = sections.get(tag)
        if section:
            ordered.extend(section)
    for tag, section in sections.items():
        if tag in {
            "qq_context",
            "history_context",
            "history",
            "current_message_layers",
            "user_state",
            "retrieval_knowledge",
        }:
            continue
        ordered.extend(section)
    return "\n".join(ordered)


def _drop_sections_to_budget(
    context_xml: str,
    *,
    token_budget: int,
    model_name: str | None = None,
) -> tuple[str, list[str]]:
    sections = _split_sections(context_xml)
    if not sections:
        return _trim_text_by_budget(context_xml, token_budget, model_name), []

    composed = _compose_sections(sections)
    current_tokens = _estimate_tokens(composed, model_name)
    if current_tokens <= token_budget:
        return composed, []

    overflow_tokens = max(current_tokens - token_budget, 0)
    dynamic_threshold = max(
        _SMALL_SECTION_TOKEN_THRESHOLD,
        int(overflow_tokens * 0.25),
    )
    large_sections: list[str] = []
    small_sections: list[str] = []
    for tag in _HARD_DROP_ORDER:
        section = sections.get(tag)
        if not section:
            continue
        section_tokens = _estimate_tokens("\n".join(section), model_name)
        if section_tokens >= dynamic_threshold:
            large_sections.append(tag)
        else:
            small_sections.append(tag)

    dropped: list[str] = []
    for tag in [*large_sections, *small_sections]:
        section = sections.pop(tag, None)
        if not section:
            continue
        dropped.append(tag)
        composed = _compose_sections(sections)
        if _estimate_tokens(composed, model_name) <= token_budget:
            return composed, dropped

    composed = _compose_sections(sections)
    if _estimate_tokens(composed, model_name) > token_budget:
        composed = _trim_text_by_budget(composed, token_budget, model_name)
    return composed, dropped


def _trim_section_content(section_lines: list[str]) -> list[str]:
    if len(section_lines) <= 4:
        return section_lines
    head = section_lines[0]
    tail = section_lines[-1]
    body = section_lines[1:-1]
    keep = max(4, int(len(body) * 0.65))
    return [head, *body[-keep:], tail]


def _compress_context(
    system_prompt: str,
    context_xml: str,
    model_name: str | None = None,
) -> tuple[str, str]:
    token_budget = _resolve_budget(model_name)
    trimmed_system_prompt = _trim_text_by_budget(
        system_prompt,
        _SYSTEM_PROMPT_BUDGET,
        model_name,
    )
    if (
        _estimate_tokens(trimmed_system_prompt + "\n" + context_xml, model_name)
        <= token_budget
    ):
        return trimmed_system_prompt, context_xml

    sections = _split_sections(context_xml)
    if not sections:
        return trimmed_system_prompt, _trim_text_by_budget(
            context_xml,
            token_budget,
            model_name,
        )

    for _ in range(8):
        current_xml = _compose_sections(sections)
        current_tokens = _estimate_tokens(
            trimmed_system_prompt + "\n" + current_xml,
            model_name,
        )
        if current_tokens <= token_budget:
            return trimmed_system_prompt, current_xml
        reduced = False
        for tag in _SECTION_TRIM_ORDER:
            section = sections.get(tag)
            if not section:
                continue
            new_section = _trim_section_content(section)
            if new_section != section:
                sections[tag] = new_section
                reduced = True
                break
        if not reduced:
            break

    compressed_xml = _compose_sections(sections)
    if (
        _estimate_tokens(trimmed_system_prompt + "\n" + compressed_xml, model_name)
        > token_budget
    ):
        compressed_xml = _trim_text_by_budget(
            compressed_xml,
            token_budget,
            model_name,
        )
    return trimmed_system_prompt, compressed_xml


@dataclass
class LifecyclePayload:
    user_id: str
    group_id: str | None
    message_text: str
    system_prompt: str
    context_xml: str
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    response_text: str | None = None


@dataclass
class ToolLifecyclePayload:
    session_id: str
    user_id: str
    group_id: str | None
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


LifecycleHook = Callable[[Any], Awaitable[None]]


class ChatInterLifecycleManager:
    def __init__(self) -> None:
        self._hooks: dict[LifecycleStage, list[LifecycleHook]] = {
            "pre_gate": [],
            "post_gate": [],
            "before_intent": [],
            "after_intent": [],
            "before_route": [],
            "after_route": [],
            "before_chat": [],
            "after_chat": [],
            "before_agent": [],
            "after_agent": [],
            "before_tool": [],
            "after_tool": [],
            "on_error": [],
        }
        self._lock = asyncio.Lock()

    async def register(self, stage: LifecycleStage, hook: LifecycleHook) -> None:
        async with self._lock:
            self._hooks.setdefault(stage, []).append(hook)

    async def dispatch(self, stage: LifecycleStage, payload: Any) -> None:
        hooks = list(self._hooks.get(stage, []))
        for hook in hooks:
            try:
                await hook(payload)
            except Exception as exc:
                logger.warning(
                    f"chatinter lifecycle hook failed: stage={stage}, error={exc}"
                )


async def _context_budget_hook(payload: LifecyclePayload) -> None:
    token_budget = _resolve_budget(payload.model_name)
    before_tokens = _estimate_tokens(
        f"{payload.system_prompt}\n{payload.context_xml}",
        payload.model_name,
    )
    compressed_prompt, compressed_context = _compress_context(
        payload.system_prompt,
        payload.context_xml,
        payload.model_name,
    )
    prompt_tokens = _estimate_tokens(compressed_prompt, payload.model_name)
    if prompt_tokens >= token_budget:
        compressed_prompt = _trim_text_by_budget(
            compressed_prompt,
            max(token_budget - 32, 1),
            payload.model_name,
        )
        prompt_tokens = _estimate_tokens(compressed_prompt, payload.model_name)

    context_budget = max(token_budget - prompt_tokens, 0)
    after_tokens = _estimate_tokens(
        f"{compressed_prompt}\n{compressed_context}",
        payload.model_name,
    )
    dropped_sections: list[str] = []
    if after_tokens > token_budget and context_budget > 0:
        compressed_context, dropped_sections = _drop_sections_to_budget(
            compressed_context,
            token_budget=context_budget,
            model_name=payload.model_name,
        )
    elif after_tokens > token_budget:
        compressed_context = ""

    after_tokens = _estimate_tokens(
        f"{compressed_prompt}\n{compressed_context}",
        payload.model_name,
    )
    if after_tokens > token_budget:
        context_budget = max(
            token_budget - _estimate_tokens(compressed_prompt, payload.model_name),
            0,
        )
        compressed_context = _trim_text_by_budget(
            compressed_context,
            context_budget,
            payload.model_name,
        )
        after_tokens = _estimate_tokens(
            f"{compressed_prompt}\n{compressed_context}",
            payload.model_name,
        )
    if after_tokens > token_budget:
        prompt_budget = max(
            token_budget - _estimate_tokens(compressed_context, payload.model_name),
            1,
        )
        compressed_prompt = _trim_text_by_budget(
            compressed_prompt,
            prompt_budget,
            payload.model_name,
        )
        after_tokens = _estimate_tokens(
            f"{compressed_prompt}\n{compressed_context}",
            payload.model_name,
        )

    payload.system_prompt = compressed_prompt
    payload.context_xml = compressed_context
    report = {
        "budget": token_budget,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "ratio": round(after_tokens / max(token_budget, 1), 3),
        "dropped_sections": dropped_sections,
        "phase": str(payload.metadata.get("phase") or "unknown"),
    }
    payload.metadata["budget_report"] = report
    if report["ratio"] >= _BUDGET_ALERT_RATIO:
        logger.debug(
            "chatinter context budget near limit: "
            f"phase={report['phase']} before={before_tokens} after={after_tokens} "
            f"budget={token_budget} dropped={','.join(dropped_sections) or '-'}"
        )


_lifecycle_manager = ChatInterLifecycleManager()
_startup_registered = False
_startup_lock = asyncio.Lock()


async def ensure_lifecycle_hooks_registered() -> None:
    global _startup_registered
    if _startup_registered:
        return
    async with _startup_lock:
        if _startup_registered:
            return
        await _lifecycle_manager.register("before_intent", _context_budget_hook)
        await _lifecycle_manager.register("before_route", _context_budget_hook)
        await _lifecycle_manager.register("before_chat", _context_budget_hook)
        await _lifecycle_manager.register("before_agent", _context_budget_hook)
        _startup_registered = True


def get_lifecycle_manager() -> ChatInterLifecycleManager:
    return _lifecycle_manager
