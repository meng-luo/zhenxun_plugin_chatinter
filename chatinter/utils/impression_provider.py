"""User impression providers for ChatInter.

The chat pipeline depends on a generic `(impression, attitude)` contract rather
than a concrete plugin implementation.  The default provider reads the existing
SignUser model and optionally enriches attitude through any available attitude
resolver.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from zhenxun.models.sign_user import SignUser

AttitudeResolver = Callable[[float], str | Awaitable[str]]


class ImpressionProvider(Protocol):
    async def get_user_impression(self, user_id: str) -> tuple[float, str]:
        """Return `(impression, attitude)` for a user."""
        ...


class SignUserImpressionProvider:
    def __init__(self, attitude_resolver: AttitudeResolver | None = None) -> None:
        self._attitude_resolver = attitude_resolver

    async def get_user_impression(self, user_id: str) -> tuple[float, str]:
        sign_user = await SignUser.get_user(user_id)
        impression = float(sign_user.impression)
        attitude = await _resolve_attitude(impression, self._attitude_resolver)
        return impression, attitude


_provider: ImpressionProvider = SignUserImpressionProvider()


def set_impression_provider(provider: ImpressionProvider) -> None:
    global _provider
    _provider = provider


def get_impression_provider() -> ImpressionProvider:
    return _provider


async def _resolve_attitude(
    impression: float,
    resolver: AttitudeResolver | None,
) -> str:
    if resolver is not None:
        result = resolver(impression)
        if isinstance(result, Awaitable):
            result = await result
        return str(result or "未知")
    return _fallback_attitude(impression)


def _fallback_attitude(impression: float) -> str:
    if impression >= 90:
        return "亲密"
    if impression >= 60:
        return "友好"
    if impression >= 30:
        return "熟悉"
    if impression > 0:
        return "普通"
    return "未知"


__all__ = [
    "ImpressionProvider",
    "SignUserImpressionProvider",
    "get_impression_provider",
    "set_impression_provider",
]
