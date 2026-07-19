"""Superuser Agent boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..superuser_agent.runtime import run_superuser_agent_runtime
from ..superuser_agent.state import AgentRuntimeResult


@dataclass(slots=True)
class SuperuserRequest:
    """Compatibility request for the private superuser runtime."""

    message_text: str
    session_key: str | None
    progress_hook: Any | None = None


class SuperuserAgent:
    """Compatibility wrapper around the direct superuser runtime."""

    async def run(self, request: SuperuserRequest) -> AgentRuntimeResult:
        return await run_superuser_agent_runtime(
            message_text=request.message_text,
            session_key=request.session_key,
            progress_hook=request.progress_hook,
        )


__all__ = ["SuperuserAgent", "SuperuserRequest"]
