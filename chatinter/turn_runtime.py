from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import re
import time

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,8}", re.IGNORECASE)
_PROMPT_CACHE_TTL = 1800.0
_DEFAULT_PROMPT_BUDGET = 9000
_DEFAULT_TOOL_CALL_LIMIT = 12
_DEFAULT_TOOL_BATCH_LIMIT = 6
_DEFAULT_HOOK_LIMIT = 18
_DEFAULT_CLASSIFIER_LIMIT = 8
_SESSION_PROMPT_CACHE: dict[str, tuple[str, float]] = {}


def estimate_text_tokens(text: str) -> int:
    source = str(text or "")
    if not source:
        return 0
    token_hits = len(_TOKEN_PATTERN.findall(source))
    return max(1, int(token_hits * 0.9))


def trim_text_to_tokens(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    source = str(text or "")
    if not source:
        return ""
    current_tokens = estimate_text_tokens(source)
    if current_tokens <= token_budget:
        return source
    ratio = max(0.12, min(1.0, token_budget / max(current_tokens, 1)))
    cut = max(96, int(len(source) * ratio))
    return source[:cut].rstrip()


def fingerprint_prompt_section(*parts: str) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part or "").encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


def detect_prompt_cache_break(
    *,
    session_key: str,
    stage: str,
    fingerprint: str,
) -> bool:
    cache_key = f"{session_key}:{stage}"
    now = time.monotonic()
    cached = _SESSION_PROMPT_CACHE.get(cache_key)
    _SESSION_PROMPT_CACHE[cache_key] = (fingerprint, now + _PROMPT_CACHE_TTL)

    expired_keys = [
        key
        for key, (_value, deadline) in _SESSION_PROMPT_CACHE.items()
        if deadline <= now
    ]
    for key in expired_keys:
        _SESSION_PROMPT_CACHE.pop(key, None)

    if cached is None:
        return False
    return cached[0] != fingerprint


@dataclass
class TurnBudgetSnapshot:
    classifier_calls: int
    hook_calls: int
    tool_calls: int
    tool_batches: int
    prompt_tokens: int
    cache_breaks: tuple[str, ...]
    compacted_stages: tuple[str, ...]
    durations_ms: dict[str, float]


@dataclass
class TurnBudgetController:
    session_key: str
    prompt_budget_tokens: int = _DEFAULT_PROMPT_BUDGET
    max_classifier_calls: int = _DEFAULT_CLASSIFIER_LIMIT
    max_hook_calls: int = _DEFAULT_HOOK_LIMIT
    max_tool_calls: int = _DEFAULT_TOOL_CALL_LIMIT
    max_tool_batches: int = _DEFAULT_TOOL_BATCH_LIMIT
    classifier_calls: int = 0
    hook_calls: int = 0
    tool_calls: int = 0
    tool_batches: int = 0
    prompt_tokens: int = 0
    cache_breaks: set[str] = field(default_factory=set)
    compacted_stages: set[str] = field(default_factory=set)
    durations: defaultdict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    @classmethod
    def for_session(
        cls,
        session_key: str,
        *,
        prompt_budget_tokens: int | None = None,
    ) -> "TurnBudgetController":
        return cls(
            session_key=session_key,
            prompt_budget_tokens=max(
                int(prompt_budget_tokens or _DEFAULT_PROMPT_BUDGET),
                1200,
            ),
        )

    def allow_classifier(self, label: str) -> bool:
        if self.classifier_calls >= self.max_classifier_calls:
            self.durations[f"classifier_block:{label}"] += 0.0
            return False
        self.classifier_calls += 1
        return True

    def record_classifier(self, label: str, duration: float) -> None:
        self.durations[f"classifier:{label}"] += max(duration, 0.0)

    def allow_hook(self, stage: str) -> bool:
        if self.hook_calls >= self.max_hook_calls:
            self.durations[f"hook_block:{stage}"] += 0.0
            return False
        self.hook_calls += 1
        return True

    def record_hook(self, stage: str, duration: float) -> None:
        self.durations[f"hook:{stage}"] += max(duration, 0.0)

    def allow_tool_batch(self, *, call_count: int, batch_kind: str) -> bool:
        projected_calls = self.tool_calls + max(call_count, 0)
        projected_batches = self.tool_batches + 1
        if (
            projected_calls > self.max_tool_calls
            or projected_batches > self.max_tool_batches
        ):
            self.durations[f"tool_block:{batch_kind}"] += 0.0
            return False
        self.tool_calls = projected_calls
        self.tool_batches = projected_batches
        return True

    def record_tool_batch(self, *, batch_kind: str, duration: float) -> None:
        self.durations[f"tool:{batch_kind}"] += max(duration, 0.0)

    def record_prompt_use(
        self,
        *,
        stage: str,
        estimated_tokens: int,
        cache_break: bool,
        compacted: bool,
    ) -> None:
        self.prompt_tokens += max(int(estimated_tokens), 0)
        if cache_break:
            self.cache_breaks.add(stage)
        if compacted:
            self.compacted_stages.add(stage)

    def prompt_budget_remaining(self) -> int:
        return max(self.prompt_budget_tokens - self.prompt_tokens, 0)

    def snapshot(self) -> TurnBudgetSnapshot:
        return TurnBudgetSnapshot(
            classifier_calls=self.classifier_calls,
            hook_calls=self.hook_calls,
            tool_calls=self.tool_calls,
            tool_batches=self.tool_batches,
            prompt_tokens=self.prompt_tokens,
            cache_breaks=tuple(sorted(self.cache_breaks)),
            compacted_stages=tuple(sorted(self.compacted_stages)),
            durations_ms={
                key: round(value * 1000, 2)
                for key, value in sorted(self.durations.items())
            },
        )


__all__ = [
    "TurnBudgetController",
    "TurnBudgetSnapshot",
    "detect_prompt_cache_break",
    "estimate_text_tokens",
    "fingerprint_prompt_section",
    "trim_text_to_tokens",
]
