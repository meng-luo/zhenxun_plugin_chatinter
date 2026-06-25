"""Superuser agent boundary.

P0 keeps the current AgentRuntime behavior unchanged.  This wrapper gives the
superuser path an explicit home before approval, dynamic tool exposure and
artifact-first output are moved behind this boundary.
"""

from __future__ import annotations

import time

from ..native_route import NativeRouteReport
from ..route_text import is_usage_question, normalize_message_text
from .core import SUPERUSER_TOOL_SCOPE, AgentObservation, AgentRequest, AgentResult
from .superuser_agent_runtime import run_superuser_agent_runtime


class SuperuserAgent:
    """Boundary for private superuser engineering tasks."""

    async def run(self, request: AgentRequest) -> AgentResult:
        started = time.perf_counter()
        # Superuser Agent owns engineering tools.  Do not build the plugin
        # command index here; group plugin routing lives in PluginCommandAgent.
        report = request.report or NativeRouteReport(
            helper_mode=is_usage_question(normalize_message_text(request.message_text))
        )
        result = await run_superuser_agent_runtime(
            message_text=request.message_text,
            session_key=request.session_key,
            budget_controller=request.budget_controller,
            messages=request.messages,
            report=report,
            progress_hook=request.kwargs.get("progress_hook"),
        )
        return AgentResult(
            agent_kind="superuser",
            main_result=result,
            observations=(AgentObservation(kind="legacy_agent_runtime", status="ok"),),
            tool_scope=SUPERUSER_TOOL_SCOPE,
            elapsed_ms=max(int((time.perf_counter() - started) * 1000), 0),
        )


__all__ = ["SuperuserAgent"]
