"""Registry for superuser-only ChatInter agent tools."""

from __future__ import annotations

from collections.abc import Callable

from zhenxun.services.llm.types.protocols import ToolExecutable

ToolFactory = Callable[[], ToolExecutable]


class SuperuserToolRegistry:
    """Small explicit registry so superuser tools can be composed by toolset."""

    def __init__(self) -> None:
        self._factories: dict[str, ToolFactory] = {}

    def register(self, factory: ToolFactory) -> ToolFactory:
        tool = factory()
        name = tool_name(tool)
        if not name:
            raise ValueError("superuser agent tool must expose a non-empty name")
        self._factories[name] = factory
        return factory

    def build_tools(self) -> dict[str, ToolExecutable]:
        return {name: factory() for name, factory in self._factories.items()}

    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._factories)


def tool_name(tool: ToolExecutable) -> str:
    return str(getattr(tool, "name", "") or "").strip()


_REGISTRY = SuperuserToolRegistry()


def register_superuser_tool(factory: ToolFactory) -> ToolFactory:
    return _REGISTRY.register(factory)


def build_superuser_agent_tools() -> dict[str, ToolExecutable]:
    from . import toolsets as _toolsets  # noqa: F401  # import registers toolsets

    return _REGISTRY.build_tools()


def registered_superuser_tool_names() -> tuple[str, ...]:
    from . import toolsets as _toolsets  # noqa: F401  # import registers toolsets

    return _REGISTRY.tool_names()


__all__ = [
    "SuperuserToolRegistry",
    "build_superuser_agent_tools",
    "registered_superuser_tool_names",
    "register_superuser_tool",
]
