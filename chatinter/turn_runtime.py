from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,8}", re.IGNORECASE)
_DEFAULT_PROMPT_BUDGET = 9000
_DEFAULT_TOOL_CALL_LIMIT = 12
_DEFAULT_TOOL_BATCH_LIMIT = 6
_DEFAULT_HOOK_LIMIT = 18
_DEFAULT_CLASSIFIER_LIMIT = 8


def estimate_text_tokens(text: str) -> int:
    source = str(text or "")
    if not source:
        return 0
    token_hits = len(_TOKEN_PATTERN.findall(source))
    return max(1, int(token_hits * 0.9))


@dataclass
class TurnBudgetSnapshot:
    classifier_calls: int
    hook_calls: int
    tool_calls: int
    tool_batches: int
    prompt_tokens: int
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
        estimated_tokens: int,
    ) -> None:
        self.prompt_tokens += max(int(estimated_tokens), 0)

    def prompt_budget_remaining(self) -> int:
        return max(self.prompt_budget_tokens - self.prompt_tokens, 0)

    def snapshot(self) -> TurnBudgetSnapshot:
        return TurnBudgetSnapshot(
            classifier_calls=self.classifier_calls,
            hook_calls=self.hook_calls,
            tool_calls=self.tool_calls,
            tool_batches=self.tool_batches,
            prompt_tokens=self.prompt_tokens,
            durations_ms={
                key: round(value * 1000, 2)
                for key, value in sorted(self.durations.items())
            },
        )


__all__ = [
    "TurnBudgetController",
    "TurnBudgetSnapshot",
    "estimate_text_tokens",
]
