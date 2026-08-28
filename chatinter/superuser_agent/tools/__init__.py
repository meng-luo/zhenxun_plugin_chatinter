"""Fixed tools for the superuser Agent runtime."""

from ...llm_compat import ToolExecutable
from .active_task_tools import (
    ActiveTaskControlTool,
    ActiveTaskCreateTool,
    ActiveTaskListTool,
    ActiveTaskUpdateTool,
)
from .artifact_tools import ArtifactReadTool
from .file_tools import (
    ApplyPatchTool,
    ListDirTool,
    ReadFileTool,
    ReplaceInFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from .plan_tools import PlanTool
from .shell_tools import ShellCommandTool
from .subagent_tools import DelegateTasksTool
from .web_tools import WebFetchTool

SUPERUSER_CORE_TOOL_NAMES = (
    "read_file",
    "list_dir",
    "search_files",
    "write_file",
    "replace_in_file",
    "apply_patch",
    "shell_command",
    "artifact_read",
    "delegate_tasks",
    "plan",
    "active_task_create",
    "active_task_list",
    "active_task_update",
    "active_task_control",
)


def build_superuser_tools() -> dict[str, ToolExecutable]:
    from ...config import active_tasks_enabled
    from ...web_access import web_access_enabled

    tools: list[ToolExecutable] = [
        ReadFileTool(),
        ListDirTool(),
        SearchFilesTool(),
        WriteFileTool(),
        ReplaceInFileTool(),
        ApplyPatchTool(),
        ShellCommandTool(),
        ArtifactReadTool(),
        DelegateTasksTool(),
        PlanTool(),
    ]
    if active_tasks_enabled():
        tools.extend(
            (
                ActiveTaskCreateTool(),
                ActiveTaskListTool(),
                ActiveTaskUpdateTool(),
                ActiveTaskControlTool(),
            )
        )
    if web_access_enabled("superuser"):
        tools.append(WebFetchTool())
    return {tool.name: tool for tool in tools}


__all__ = ["SUPERUSER_CORE_TOOL_NAMES", "build_superuser_tools"]
