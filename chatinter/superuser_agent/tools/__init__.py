"""Fixed tools for the superuser Agent runtime."""

from ...llm_compat import ToolExecutable
from .artifact_tools import ArtifactReadTool
from .file_tools import (
    ListDirTool,
    ReadFileTool,
    ReplaceInFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from .shell_tools import ShellCommandTool

SUPERUSER_CORE_TOOL_NAMES = (
    "read_file",
    "list_dir",
    "search_files",
    "write_file",
    "replace_in_file",
    "shell_command",
    "artifact_read",
)


def build_superuser_tools() -> dict[str, ToolExecutable]:
    tools = (
        ReadFileTool(),
        ListDirTool(),
        SearchFilesTool(),
        WriteFileTool(),
        ReplaceInFileTool(),
        ShellCommandTool(),
        ArtifactReadTool(),
    )
    return {tool.name: tool for tool in tools}


__all__ = ["SUPERUSER_CORE_TOOL_NAMES", "build_superuser_tools"]
