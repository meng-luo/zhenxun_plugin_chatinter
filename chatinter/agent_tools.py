import ast
import asyncio
import math
from typing import Any

from zhenxun.services.llm.tools import RunContext, function_tool, tool_provider_manager

from .plugin_registry import PluginRegistry
from .sandbox import get_sandbox_manager

_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
}

_SAFE_NAMES = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}

_SAFE_SHELL_PREFIX = {"echo", "pwd", "dir", "ls", "whoami", "date"}


def _resolve_session_id(context: RunContext | None) -> str:
    if context and context.session_id:
        return str(context.session_id)
    return "global"


def _validate_expression(expr: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as exc:
        return False, f"表达式解析失败: {exc}"

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _ALLOWED_AST_NODES:
            return False, f"不允许的语法节点: {node_type.__name__}"
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False, "禁止访问内部名称"
            if node.id not in _SAFE_NAMES:
                return False, f"未授权名称: {node.id}"
    return True, ""


def _run_safe_eval(expr: str) -> Any:
    ok, reason = _validate_expression(expr)
    if not ok:
        raise ValueError(reason)
    return eval(
        compile(expr, "<chatinter-safe-eval>", "eval"),
        {"__builtins__": {}},
        dict(_SAFE_NAMES),
    )


@function_tool(
    name="chatinter_lookup_plugin",
    description="按关键词检索可用插件和命令",
)
async def chatinter_lookup_plugin(keyword: str, top_k: int = 5) -> dict[str, Any]:
    knowledge = await PluginRegistry.get_plugin_knowledge_base()
    query = keyword.strip().lower()
    if not query:
        return {"items": []}
    items = []
    for plugin in knowledge.plugins:
        text = (
            f"{plugin.name} {plugin.module} {plugin.description} "
            f"{' '.join(plugin.commands)} {(plugin.usage or '')}"
        ).lower()
        if query in text:
            items.append(
                {
                    "module": plugin.module,
                    "name": plugin.name,
                    "commands": plugin.commands[:5],
                    "description": plugin.description,
                }
            )
        if len(items) >= top_k:
            break
    return {"items": items}


@function_tool(
    name="chatinter_safe_eval",
    description="在受限沙箱中执行简单数学/逻辑表达式",
)
async def chatinter_safe_eval(expression: str) -> dict[str, Any]:
    expr = expression.strip()
    if not expr:
        return {"ok": False, "error": "表达式为空"}
    if len(expr) > 300:
        return {"ok": False, "error": "表达式过长"}
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_safe_eval, expr), timeout=2.5
        )
        return {"ok": True, "result": str(result)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@function_tool(
    name="chatinter_safe_shell",
    description="在白名单沙箱中执行只读 shell 命令",
)
async def chatinter_safe_shell(
    command: str,
    timeout: int = 3,
    context: RunContext | None = None,
) -> dict[str, Any]:
    command = command.strip()
    if not command:
        return {"ok": False, "error": "命令为空"}
    sandbox = await get_sandbox_manager()
    session_id = _resolve_session_id(context)
    return await sandbox.execute_shell(
        session_id=session_id,
        command=command,
        timeout=timeout,
        allowed_prefix=_SAFE_SHELL_PREFIX,
    )


async def get_chatinter_tools() -> dict[str, Any]:
    names = [
        "chatinter_lookup_plugin",
        "chatinter_safe_eval",
        "chatinter_safe_shell",
    ]
    return await tool_provider_manager.get_function_tools(names)
