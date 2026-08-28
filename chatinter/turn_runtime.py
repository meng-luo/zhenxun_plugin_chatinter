from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .token_compat import estimate_text_tokens

_DEFAULT_PROMPT_BUDGET = 16000
_DEFAULT_TOOL_CALL_LIMIT = 12
_DEFAULT_TOOL_BATCH_LIMIT = 6
_DEFAULT_HOOK_LIMIT = 18
_DEFAULT_CLASSIFIER_LIMIT = 8


@dataclass
class TurnBudgetSnapshot:
    classifier_calls: int
    hook_calls: int
    tool_calls: int
    tool_batches: int
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    cache_observed_prompt_tokens: int
    cache_unknown_prompt_tokens: int
    cache_observed_model_calls: int
    cache_unknown_model_calls: int
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
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_observed_prompt_tokens: int = 0
    cache_unknown_prompt_tokens: int = 0
    cache_observed_model_calls: int = 0
    cache_unknown_model_calls: int = 0
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

    def record_stage(self, label: str, duration: float) -> None:
        self.durations[f"stage:{label}"] += max(duration, 0.0)

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

    def record_model_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_prompt_tokens: int = 0,
        cache_observed: bool = False,
    ) -> None:
        normalized_prompt = max(int(prompt_tokens), 0)
        self.prompt_tokens += normalized_prompt
        self.completion_tokens += max(int(completion_tokens), 0)
        self.cached_prompt_tokens += max(int(cached_prompt_tokens), 0)
        if cache_observed:
            self.cache_observed_prompt_tokens += normalized_prompt
            self.cache_observed_model_calls += 1
        else:
            self.cache_unknown_prompt_tokens += normalized_prompt
            self.cache_unknown_model_calls += 1

    def prompt_budget_remaining(self) -> int:
        return max(self.prompt_budget_tokens - self.prompt_tokens, 0)

    def snapshot(self) -> TurnBudgetSnapshot:
        return TurnBudgetSnapshot(
            classifier_calls=self.classifier_calls,
            hook_calls=self.hook_calls,
            tool_calls=self.tool_calls,
            tool_batches=self.tool_batches,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            cache_observed_prompt_tokens=self.cache_observed_prompt_tokens,
            cache_unknown_prompt_tokens=self.cache_unknown_prompt_tokens,
            cache_observed_model_calls=self.cache_observed_model_calls,
            cache_unknown_model_calls=self.cache_unknown_model_calls,
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
