"""In-memory admission gate between mixed chat and Superuser Agent mode."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
import uuid

AgentActiveSource = bool | Callable[[], bool]


@dataclass
class _SessionGateState:
    agent_active: bool = False
    mixed_leases: set[str] = field(default_factory=set)
    switch_token: str | None = None


@dataclass(frozen=True, slots=True)
class MixedTurnAdmission:
    lease: MixedTurnLease | None
    blocked_by: str = ""

    @property
    def accepted(self) -> bool:
        return self.lease is not None


class MixedTurnLease:
    __slots__ = ("_gate", "_released", "_session_key", "_token")

    def __init__(self, gate: ModeGate, session_key: str, token: str) -> None:
        self._gate = gate
        self._session_key = session_key
        self._token = token
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        await self._gate._release_mixed_turn(self._session_key, self._token)
        self._released = True


class AgentModeTransition:
    __slots__ = ("_finished", "_gate", "_session_key", "_token")

    def __init__(self, gate: ModeGate, session_key: str, token: str) -> None:
        self._gate = gate
        self._session_key = session_key
        self._token = token
        self._finished = False

    async def finish(self, *, agent_active: AgentActiveSource) -> None:
        if self._finished:
            return
        await self._gate._finish_agent_transition(
            self._session_key,
            self._token,
            agent_active=agent_active,
        )
        self._finished = True


class ModeGate:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, _SessionGateState] = {}

    async def try_acquire_mixed_turn(
        self,
        session_key: str,
        *,
        agent_active: AgentActiveSource,
    ) -> MixedTurnAdmission:
        key = str(session_key)
        async with self._lock:
            state = self._sessions.setdefault(key, _SessionGateState())
            state.agent_active = _resolve_agent_active(agent_active)
            if state.switch_token is not None:
                return MixedTurnAdmission(None, "switching")
            if state.agent_active:
                return MixedTurnAdmission(None, "agent_active")
            token = uuid.uuid4().hex
            state.mixed_leases.add(token)
            return MixedTurnAdmission(MixedTurnLease(self, key, token))

    async def try_begin_agent_transition(
        self,
        session_key: str,
        *,
        agent_active: AgentActiveSource,
    ) -> tuple[AgentModeTransition | None, str]:
        key = str(session_key)
        async with self._lock:
            state = self._sessions.setdefault(key, _SessionGateState())
            state.agent_active = _resolve_agent_active(agent_active)
            if state.switch_token is not None:
                return None, "switching"
            if state.mixed_leases:
                return None, "mixed_busy"
            token = uuid.uuid4().hex
            state.switch_token = token
            return AgentModeTransition(self, key, token), ""

    async def sync_agent_active(
        self,
        session_key: str,
        *,
        active: AgentActiveSource,
    ) -> None:
        key = str(session_key)
        async with self._lock:
            state = self._sessions.setdefault(key, _SessionGateState())
            state.agent_active = _resolve_agent_active(active)
            self._prune_locked(key, state)

    async def _release_mixed_turn(self, session_key: str, token: str) -> None:
        async with self._lock:
            state = self._sessions.get(session_key)
            if state is None:
                return
            state.mixed_leases.discard(token)
            self._prune_locked(session_key, state)

    async def _finish_agent_transition(
        self,
        session_key: str,
        token: str,
        *,
        agent_active: AgentActiveSource,
    ) -> None:
        async with self._lock:
            state = self._sessions.get(session_key)
            if state is None or state.switch_token != token:
                return
            state.switch_token = None
            state.agent_active = _resolve_agent_active(agent_active)
            self._prune_locked(session_key, state)

    def _prune_locked(self, session_key: str, state: _SessionGateState) -> None:
        if (
            not state.agent_active
            and not state.mixed_leases
            and state.switch_token is None
        ):
            self._sessions.pop(session_key, None)


def _resolve_agent_active(source: AgentActiveSource) -> bool:
    return bool(source() if callable(source) else source)


_MODE_GATE = ModeGate()


def get_mode_gate() -> ModeGate:
    return _MODE_GATE


__all__ = [
    "AgentActiveSource",
    "AgentModeTransition",
    "MixedTurnAdmission",
    "MixedTurnLease",
    "ModeGate",
    "get_mode_gate",
]
