"""Provider error classification and fallback chain.

    transient error  -> retry same model (bounded, with backoff)
    context overflow -> caller compresses context, retry once
    fallback-worthy  -> switch to next model in the configured chain
    content policy   -> fail fast (no model switch will help)

The classifier is intentionally pattern-based because provider adapters can
surface different concrete exception classes. Use type names and message text
instead of binding this layer to one adapter package.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
import inspect
import re
from typing import Any

from zhenxun.services.ai.core.exceptions import ContextLengthExceededException
from zhenxun.services.log import logger

_LOG_COMMAND = "ChatInterFailover"


class LLMFailureKind(str, Enum):
    TRANSIENT = "transient"
    CONTEXT_OVERFLOW = "context_overflow"
    CONTENT_POLICY = "content_policy"
    FALLBACK_ELIGIBLE = "fallback_eligible"
    UNKNOWN = "unknown"


_TRANSIENT_PATTERNS = re.compile(
    r"rate.?limit|too many request|429|overload|503|502|504|timeout|timed out"
    r"|connection|temporarily|server is busy|try again",
    re.IGNORECASE,
)
_CONTEXT_PATTERNS = re.compile(
    r"context.?length|context window|maximum context|too many tokens"
    r"|prompt is too long|input is too long|max_?tokens.*exceed|413",
    re.IGNORECASE,
)
_CONTENT_POLICY_PATTERNS = re.compile(
    r"content.?policy|content.?filter|safety|blocked by|prohibited"
    r"|refused due to|moderation",
    re.IGNORECASE,
)
_FALLBACK_PATTERNS = re.compile(
    r"401|403|invalid api key|unauthorized|authentication|quota|billing"
    r"|insufficient.?(balance|funds|quota)|model.+not (found|exist)"
    r"|unknown model|404|invalid model|500|internal server error",
    re.IGNORECASE,
)
_TRANSIENT_TYPE_HINTS = ("Timeout", "ConnectError", "ConnectionError", "ReadError")


def classify_llm_error(exc: BaseException) -> LLMFailureKind:
    chain = _exception_chain(exc)
    type_names = tuple(type(item).__name__ for item in chain)
    text = "\n".join(f"{type(item).__name__}: {item}" for item in chain)
    if any(isinstance(item, ContextLengthExceededException) for item in chain):
        return LLMFailureKind.CONTEXT_OVERFLOW
    if _CONTEXT_PATTERNS.search(text):
        return LLMFailureKind.CONTEXT_OVERFLOW
    if _CONTENT_POLICY_PATTERNS.search(text):
        return LLMFailureKind.CONTENT_POLICY
    if any(
        hint in type_name
        for type_name in type_names
        for hint in _TRANSIENT_TYPE_HINTS
    ):
        return LLMFailureKind.TRANSIENT
    if _TRANSIENT_PATTERNS.search(text):
        return LLMFailureKind.TRANSIENT
    if _FALLBACK_PATTERNS.search(text):
        return LLMFailureKind.FALLBACK_ELIGIBLE
    return LLMFailureKind.UNKNOWN


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    pending = [exc]
    chain: list[BaseException] = []
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        for attr in ("__cause__", "__context__", "cause"):
            nested = getattr(current, attr, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
    return tuple(chain)


@dataclass
class FailoverAttempt:
    model: str
    kind: str
    error: str


@dataclass
class FailoverOutcome:
    response: Any
    used_model: str | None
    attempts: list[FailoverAttempt]

    @property
    def switched(self) -> bool:
        return bool(self.attempts)


_MAX_TRANSIENT_RETRIES = 2
_BACKOFF_SECONDS = (1.0, 2.5)


async def request_with_failover(
    *,
    primary_model: str | None,
    fallback_models: tuple[str, ...],
    request_fn: Callable[[str | None], Awaitable[Any]],
    compress_fn: Callable[[], Awaitable[None] | None] | None = None,
    trace_id: str = "",
    transient_retries: int = _MAX_TRANSIENT_RETRIES,
) -> FailoverOutcome:
    """Run ``request_fn(model)`` through the recovery ladder.

    ``request_fn`` receives the model name to use (None = service default).
    ``compress_fn`` is invoked at most once per candidate model on
    context-overflow before retrying.
    Set ``transient_retries`` to zero when the provider layer owns same-model
    retries; candidate fallback and context recovery remain active.
    Raises the last error when the whole chain is exhausted; content-policy
    errors are re-raised immediately.
    """
    attempts: list[FailoverAttempt] = []
    chain: list[str | None] = [primary_model, *fallback_models]
    compressed_models: set[str] = set()
    max_transient_retries = max(int(transient_retries), 0)

    last_error: BaseException | None = None
    for chain_index, model in enumerate(chain):
        transient_retry_count = 0
        while True:
            try:
                response = await request_fn(model)
                return FailoverOutcome(
                    response=response,
                    used_model=model,
                    attempts=attempts,
                )
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                kind = classify_llm_error(exc)
                last_error = exc
                attempts.append(
                    FailoverAttempt(
                        model=str(model or "<default>"),
                        kind=kind.value,
                        error=str(exc)[:300],
                    )
                )
                logger.warning(
                    f"LLM 请求失败 model={model or '<default>'} kind={kind.value}"
                    f" trace={trace_id}: {str(exc)[:200]}",
                    _LOG_COMMAND,
                )
                if kind is LLMFailureKind.CONTENT_POLICY:
                    raise
                if kind is LLMFailureKind.TRANSIENT:
                    if transient_retry_count < max_transient_retries:
                        await asyncio.sleep(
                            _BACKOFF_SECONDS[
                                min(
                                    transient_retry_count,
                                    len(_BACKOFF_SECONDS) - 1,
                                )
                            ]
                        )
                        transient_retry_count += 1
                        continue
                elif kind is LLMFailureKind.CONTEXT_OVERFLOW:
                    model_key = str(model or "<default>")
                    if compress_fn is not None and model_key not in compressed_models:
                        compressed_models.add(model_key)
                        compressed = compress_fn()
                        if inspect.isawaitable(compressed):
                            await compressed
                        continue


                break
        if chain_index + 1 < len(chain):
            logger.info(
                f"切换降级模型: {chain[chain_index + 1]} trace={trace_id}",
                _LOG_COMMAND,
            )
    assert last_error is not None
    raise last_error


__all__ = [
    "FailoverAttempt",
    "FailoverOutcome",
    "LLMFailureKind",
    "classify_llm_error",
    "request_with_failover",
]
