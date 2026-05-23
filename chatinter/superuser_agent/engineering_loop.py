"""Fixed engineering Agent loop protocol.

This layer does not replace file/patch/eval tools.  It records a durable
protocol around them so code work follows a predictable sequence:
read code -> semantic patch -> checkpoint -> eval gate -> recovery or summary.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import time
from typing import Any, Literal
import uuid

from ..persistence import read_json, state_path, write_json
from .audit_log import record_audit_event
from .engineering_eval import get_engineering_eval
from .patch_operations import get_patch_operation

LoopStage = Literal[
    "created",
    "code_read",
    "patch_prepared",
    "patch_applied",
    "eval_planned",
    "eval_passed",
    "eval_failed",
    "diagnosed",
    "second_patch_prepared",
    "second_patch_applied",
    "rollback_prepared",
    "rolled_back",
    "completed",
    "failed",
]

_LOOPS_PATH = state_path("engineering_loops.json")
_LOOPS: dict[str, "EngineeringLoop"] = {}
_LOADED = False
_MAX_READ_BYTES = 256_000


@dataclass
class CodeSymbol:
    name: str
    kind: str
    path: str
    line: int
    column: int = 0
    signature: str = ""
    doc: str = ""
    references: int = 0

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineeringLoopEvent:
    kind: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def public_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timestamp": int(self.timestamp),
            "payload": dict(self.payload),
        }


@dataclass
class EngineeringLoop:
    loop_id: str
    user_id: str
    session_key: str
    task: str
    stage: LoopStage = "created"
    files: list[str] = field(default_factory=list)
    symbols: list[CodeSymbol] = field(default_factory=list)
    semantic_patch_plan: dict[str, Any] = field(default_factory=dict)
    patch_operation_id: str = ""
    eval_id: str = ""
    checkpoint_ids: dict[str, str] = field(default_factory=dict)
    rollback_operation_id: str = ""
    recovery_plan: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    attempt: int = 0
    failure_reason: str = ""
    events: list[EngineeringLoopEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def public_payload(self, *, include_events: bool = True) -> dict[str, Any]:
        payload = {
            "loop_id": self.loop_id,
            "user_id": self.user_id,
            "session_key": self.session_key,
            "task": self.task,
            "stage": self.stage,
            "files": list(self.files),
            "symbols": [item.public_payload() for item in self.symbols],
            "semantic_patch_plan": dict(self.semantic_patch_plan),
            "patch_operation_id": self.patch_operation_id,
            "eval_id": self.eval_id,
            "checkpoint_ids": dict(self.checkpoint_ids),
            "rollback_operation_id": self.rollback_operation_id,
            "recovery_plan": dict(self.recovery_plan),
            "diagnosis": dict(self.diagnosis),
            "attempt": self.attempt,
            "failure_reason": self.failure_reason,
            "created_at": int(self.created_at),
            "updated_at": int(self.updated_at),
        }
        if include_events:
            payload["events"] = [item.public_payload() for item in self.events[-24:]]
        return payload

    def to_record(self) -> dict[str, Any]:
        return {
            **self.public_payload(include_events=True),
            "events": [item.public_payload() for item in self.events],
        }

    def append_event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(EngineeringLoopEvent(kind=kind, payload=payload or {}))
        self.updated_at = time.time()


def create_engineering_loop(
    *,
    actor: dict[str, str],
    task: str,
    files: list[str] | None = None,
) -> EngineeringLoop:
    _ensure_loaded()
    loop = EngineeringLoop(
        loop_id=uuid.uuid4().hex[:12],
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        task=str(task or ""),
        files=_dedupe([str(item) for item in files or [] if str(item or "")]),
    )
    loop.append_event("loop_created", {"task": loop.task, "files": loop.files})
    _LOOPS[loop.loop_id] = loop
    _save_loops()
    record_audit_event(
        event="engineering_loop_created",
        user_id=loop.user_id,
        session_key=loop.session_key,
        action="engineering_loop_start",
        payload=loop.public_payload(include_events=False),
    )
    return loop


def get_engineering_loop(loop_id: str) -> EngineeringLoop | None:
    _ensure_loaded()
    return _LOOPS.get(str(loop_id or "").strip())


def list_engineering_loops(
    *,
    user_id: str,
    session_key: str,
    limit: int = 20,
) -> list[EngineeringLoop]:
    _ensure_loaded()
    rows = [
        item
        for item in _LOOPS.values()
        if item.user_id == str(user_id or "")
        and item.session_key == str(session_key or "")
    ]
    rows.sort(key=lambda item: item.updated_at, reverse=True)
    return rows[: max(1, min(int(limit or 20), 100))]


def record_lsp_read(
    *,
    loop_id: str,
    files: list[str],
    symbols: list[CodeSymbol],
    diagnostics: list[dict[str, Any]] | None = None,
) -> EngineeringLoop | None:
    loop = get_engineering_loop(loop_id)
    if loop is None:
        return None
    loop.files = _dedupe([*loop.files, *files])
    loop.symbols = _merge_symbols(loop.symbols, symbols)
    loop.stage = "code_read"
    loop.append_event(
        "lsp_code_read",
        {
            "files": files,
            "symbol_count": len(symbols),
            "diagnostics": diagnostics or [],
        },
    )
    _save_loops()
    return loop


def record_semantic_patch_plan(
    *,
    loop_id: str,
    plan: dict[str, Any],
) -> EngineeringLoop | None:
    loop = get_engineering_loop(loop_id)
    if loop is None:
        return None
    loop.semantic_patch_plan = dict(plan or {})
    loop.append_event("semantic_patch_planned", loop.semantic_patch_plan)
    _save_loops()
    return loop


def bind_patch_operation(
    *,
    loop_id: str,
    operation_id: str,
) -> EngineeringLoop | None:
    loop = get_engineering_loop(loop_id)
    if loop is None:
        return None
    operation = get_patch_operation(operation_id)
    previous_operation_id = loop.patch_operation_id
    loop.patch_operation_id = str(operation_id or "")
    if operation is not None:
        loop.files = _dedupe([*loop.files, *[item.path for item in operation.snapshots]])
        if (
            (
                loop.stage in {"eval_failed", "diagnosed", "second_patch_prepared"}
                or bool(loop.diagnosis)
            )
            and operation.status == "applied"
            and loop.stage != "second_patch_applied"
        ):
            loop.attempt += 1
        loop.checkpoint_ids.update(
            {
                key: value
                for key, value in {
                    "pre": operation.pre_checkpoint_id,
                    "post": operation.post_checkpoint_id,
                    "rollback": operation.rollback_checkpoint_id,
                    "failure": operation.failure_checkpoint_id,
                }.items()
                if value
            }
        )
        loop.eval_id = operation.bound_eval_id or loop.eval_id
        loop.rollback_operation_id = operation.operation_id
        if loop.diagnosis and operation.status != "applied":
            loop.stage = "second_patch_prepared"
        elif loop.attempt > 0 and operation.status == "applied":
            loop.stage = "second_patch_applied"
        elif loop.attempt > 0 or previous_operation_id != loop.patch_operation_id and loop.diagnosis:
            loop.stage = "second_patch_prepared"
        else:
            loop.stage = "patch_applied" if operation.status == "applied" else "patch_prepared"
    else:
        loop.stage = "patch_prepared"
    loop.append_event(
        "patch_operation_bound",
        {
            "operation_id": operation_id,
            "operation": operation.public_payload(include_content=False)
            if operation
            else None,
        },
    )
    _save_loops()
    return loop


def bind_eval(
    *,
    loop_id: str,
    eval_id: str,
) -> EngineeringLoop | None:
    loop = get_engineering_loop(loop_id)
    if loop is None:
        return None
    eval_run = get_engineering_eval(eval_id)
    loop.eval_id = str(eval_id or "")
    if eval_run is not None:
        loop.files = _dedupe([*loop.files, *eval_run.files])
        loop.checkpoint_ids.update(eval_run.patch_checkpoints)
        if eval_run.rollback_operation_id:
            loop.rollback_operation_id = eval_run.rollback_operation_id
        loop.stage = "eval_failed" if eval_run.status == "failed" else "eval_planned"
        if eval_run.status == "passed":
            loop.stage = "eval_passed"
        loop.recovery_plan = dict(eval_run.recovery_plan or {})
        loop.failure_reason = eval_run.error or loop.failure_reason
    else:
        loop.stage = "eval_planned"
    loop.append_event(
        "eval_bound",
        {"eval_id": eval_id, "eval": eval_run.public_payload() if eval_run else None},
    )
    _save_loops()
    return loop


def update_loop_from_eval(
    *,
    loop_id: str,
    eval_id: str,
) -> EngineeringLoop | None:
    loop = bind_eval(loop_id=loop_id, eval_id=eval_id)
    if loop is None:
        return None
    eval_run = get_engineering_eval(eval_id)
    if eval_run is None:
        return loop
    if eval_run.status == "passed":
        loop.stage = "eval_passed"
        loop.recovery_plan = {}
        loop.failure_reason = ""
    elif eval_run.status == "failed":
        loop.stage = "eval_failed"
        loop.recovery_plan = dict(eval_run.recovery_plan or {})
        loop.failure_reason = eval_run.last_recommended_action or eval_run.error
    loop.append_event(
        "eval_gate_updated",
        {
            "eval_id": eval_id,
            "status": eval_run.status,
            "recovery_plan": eval_run.recovery_plan,
        },
    )
    _save_loops()
    return loop


def diagnose_eval_failure(
    *,
    loop_id: str,
    eval_id: str = "",
    notes: str = "",
) -> EngineeringLoop | None:
    loop = get_engineering_loop(loop_id)
    if loop is None:
        return None
    eval_run = get_engineering_eval(eval_id or loop.eval_id)
    failure: dict[str, Any] = {}
    if eval_run is not None:
        failure = (
            eval_run.failure_observations[-1]
            if eval_run.failure_observations
            else {}
        )
        loop.recovery_plan = dict(
            eval_run.recovery_plan or failure.get("recovery_plan") or {}
        )
        loop.failure_reason = (
            eval_run.last_recommended_action
            or str(failure.get("reason", "") or "")
            or loop.failure_reason
        )
    loop.diagnosis = build_failure_diagnosis(
        loop=loop,
        eval_payload=eval_run.public_payload() if eval_run else {},
        failure_observation=failure,
        notes=notes,
    )
    loop.stage = "diagnosed"
    loop.append_event("failure_diagnosed", loop.diagnosis)
    _save_loops()
    return loop


def mark_loop_rolled_back(
    *,
    loop_id: str,
    operation_id: str,
) -> EngineeringLoop | None:
    loop = get_engineering_loop(loop_id)
    if loop is None:
        return None
    operation = get_patch_operation(operation_id)
    loop.rollback_operation_id = str(operation_id or "")
    loop.stage = "rolled_back"
    if operation is not None:
        loop.checkpoint_ids.update(
            {
                key: value
                for key, value in {
                    "rollback": operation.rollback_checkpoint_id,
                    "failure": operation.failure_checkpoint_id,
                }.items()
                if value
            }
        )
    loop.append_event(
        "rollback_completed",
        {
            "operation_id": operation_id,
            "operation": operation.public_payload(include_content=False)
            if operation
            else None,
        },
    )
    _save_loops()
    return loop


def complete_engineering_loop(*, loop_id: str, reason: str = "") -> EngineeringLoop | None:
    loop = get_engineering_loop(loop_id)
    if loop is None:
        return None
    gate = eval_gate(loop_id=loop_id)
    if not gate.get("ok"):
        loop.stage = "failed"
        loop.failure_reason = str(gate.get("reason", "") or "eval_gate_failed")
        loop.recovery_plan = dict(gate.get("recovery_plan") or {})
        loop.append_event("loop_completion_blocked", gate)
    else:
        loop.stage = "completed"
        loop.append_event("loop_completed", {"reason": reason or "eval_gate_passed"})
    _save_loops()
    return loop


def eval_gate(*, loop_id: str) -> dict[str, Any]:
    loop = get_engineering_loop(loop_id)
    if loop is None:
        return {
            "ok": False,
            "status": "engineering_loop_not_found",
            "reason": "loop_id not found",
        }
    if loop.eval_id:
        eval_run = get_engineering_eval(loop.eval_id)
        if eval_run is None:
            return {
                "ok": False,
                "status": "eval_missing",
                "loop": loop.public_payload(include_events=False),
                "reason": "bound eval_id not found",
                "next_tools": ["engineering_eval_status", "engineering_eval_plan"],
            }
        if eval_run.status == "passed":
            return {
                "ok": True,
                "status": "eval_gate_passed",
                "loop": loop.public_payload(include_events=False),
                "eval": eval_run.public_payload(),
            }
        return {
            "ok": False,
            "status": "eval_gate_blocked",
            "loop": loop.public_payload(include_events=False),
            "eval": eval_run.public_payload(),
            "reason": eval_run.last_recommended_action
            or eval_run.status
            or "eval_not_passed",
            "recovery_plan": eval_run.recovery_plan,
            "diagnosis": loop.diagnosis,
            "next_tools": _next_tools_from_recovery(
                eval_run.recovery_plan,
                diagnosed=bool(loop.diagnosis),
            ),
            "protocol_hint": (
                "先调用 engineering_failure_diagnose 固化失败诊断，再根据 diagnosis "
                "选择 engineering_lsp_read、semantic_patch_plan、patch_prepare 或 patch_rollback。"
                if not loop.diagnosis
                else "按 diagnosis.allowed_next_tools 执行，不要重复无变化测试。"
            ),
        }
    if loop.patch_operation_id:
        return {
            "ok": False,
            "status": "eval_required",
            "loop": loop.public_payload(include_events=False),
            "reason": "patch is bound but no eval is bound",
            "next_tools": ["engineering_eval_plan", "engineering_eval_run"],
        }
    return {
        "ok": False,
        "status": "patch_required",
        "loop": loop.public_payload(include_events=False),
        "reason": "no patch/eval evidence bound to loop",
        "next_tools": ["engineering_lsp_read", "semantic_patch_plan", "patch_prepare"],
    }


def read_code_symbols(
    *,
    files: list[str],
    query: str = "",
    max_symbols: int = 120,
) -> tuple[list[CodeSymbol], list[dict[str, Any]]]:
    symbols: list[CodeSymbol] = []
    diagnostics: list[dict[str, Any]] = []
    query_text = str(query or "").casefold()
    for path in files:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            diagnostics.append({"path": path, "level": "warning", "message": "missing"})
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            diagnostics.append({"path": path, "level": "error", "message": str(exc)})
            continue
        if len(content.encode("utf-8", errors="ignore")) > _MAX_READ_BYTES:
            diagnostics.append(
                {"path": path, "level": "warning", "message": "file_too_large"}
            )
            continue
        symbols.extend(_symbols_for_file(file_path, content, query=query_text))
        if len(symbols) >= max_symbols:
            break
    return symbols[: max(1, int(max_symbols or 120))], diagnostics


def build_semantic_patch_plan(
    *,
    task: str,
    files: list[str],
    symbols: list[CodeSymbol],
    instructions: str = "",
) -> dict[str, Any]:
    symbol_payload = [item.public_payload() for item in symbols[:40]]
    target_symbols = [
        item
        for item in symbol_payload
        if _text_overlap(str(task or "") + " " + str(instructions or ""), item)
    ][:12]
    if not target_symbols:
        target_symbols = symbol_payload[:12]
    return {
        "task": str(task or ""),
        "instructions": str(instructions or ""),
        "files": _dedupe(files),
        "target_symbols": target_symbols,
        "patch_protocol": [
            "engineering_lsp_read",
            "semantic_patch_plan",
            "patch_prepare",
            "patch_apply",
            "engineering_eval_run",
            "engineering_eval_gate",
        ],
        "failure_protocol": [
            "engineering_failure_diagnose",
            "engineering_lsp_read",
            "semantic_patch_plan",
            "patch_prepare",
            "patch_apply",
            "engineering_eval_run",
            "engineering_eval_gate",
        ],
        "constraints": [
            "Use exact replace patches when possible.",
            "Do not apply stale patch operations after dirty-lock failures.",
            "Bind patch operation and eval back to this loop.",
            "If eval fails, follow recovery_plan before rerunning unchanged tests.",
        ],
    }


def build_failure_diagnosis(
    *,
    loop: EngineeringLoop,
    eval_payload: dict[str, Any],
    failure_observation: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    recovery = dict(
        failure_observation.get("recovery_plan")
        or loop.recovery_plan
        or {}
    )
    recommended = str(
        recovery.get("recommended_action")
        or failure_observation.get("recommended_next_action")
        or loop.failure_reason
        or "inspect_output"
    )
    files_to_reread = _dedupe(
        [
            *[str(item) for item in recovery.get("files_to_reread", []) or []],
            *[
                str(item)
                for item in failure_observation.get("files_to_reread", []) or []
            ],
            *loop.files,
        ]
    )[:20]
    return {
        "status": "engineering_failure_diagnosis",
        "loop_id": loop.loop_id,
        "eval_id": str(eval_payload.get("eval_id", "") or loop.eval_id),
        "attempt": loop.attempt,
        "recommended_action": recommended,
        "allowed_next_tools": _diagnosis_next_tools(recommended, recovery),
        "files_to_reread": files_to_reread,
        "patch_operation_id": loop.patch_operation_id,
        "rollback_operation_id": loop.rollback_operation_id
        or str(recovery.get("rollback_operation_id", "") or ""),
        "checkpoint_ids": {
            **loop.checkpoint_ids,
            **dict(recovery.get("patch_checkpoints") or {}),
            **dict(recovery.get("checkpoint_ids") or {}),
        },
        "failed_command_signature": str(
            recovery.get("failed_command_signature")
            or failure_observation.get("failed_command_signature")
            or ""
        ),
        "do_not_repeat_failed_command": True,
        "notes": str(notes or ""),
        "instruction": (
            "这是固定工程闭环的失败诊断。下一步只能从 allowed_next_tools 里选择；"
            "二次 patch 前必须先读 files_to_reread 中的相关代码。"
        ),
    }


def _symbols_for_file(path: Path, content: str, *, query: str) -> list[CodeSymbol]:
    if path.suffix == ".py":
        return _python_symbols(path, content, query=query)
    return _generic_symbols(path, content, query=query)


def _python_symbols(path: Path, content: str, *, query: str) -> list[CodeSymbol]:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return [
            CodeSymbol(
                name="syntax_error",
                kind="diagnostic",
                path=str(path),
                line=int(exc.lineno or 1),
                column=int(exc.offset or 0),
                signature=str(exc.msg or "syntax error"),
            )
        ]
    rows: list[CodeSymbol] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            rows.append(_symbol_from_ast(path, node, kind="class", content=content))
        elif isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            rows.append(_symbol_from_ast(path, node, kind="function", content=content))
    return _filter_symbols(rows, query=query)


def _symbol_from_ast(
    path: Path,
    node: ast.ClassDef | ast.AsyncFunctionDef | ast.FunctionDef,
    *,
    kind: str,
    content: str,
) -> CodeSymbol:
    signature = node.name
    if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
        args = [arg.arg for arg in node.args.args]
        signature = f"{node.name}({', '.join(args)})"
    return CodeSymbol(
        name=node.name,
        kind=kind,
        path=str(path),
        line=int(getattr(node, "lineno", 1) or 1),
        column=int(getattr(node, "col_offset", 0) or 0),
        signature=signature,
        doc=ast.get_docstring(node) or "",
        references=_reference_count(content, node.name),
    )


def _generic_symbols(path: Path, content: str, *, query: str) -> list[CodeSymbol]:
    rows: list[CodeSymbol] = []
    patterns = (
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", "function"),
        (r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
        (r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", "variable"),
    )
    lines = content.splitlines()
    for index, line in enumerate(lines, 1):
        for pattern, kind in patterns:
            matched = re.search(pattern, line)
            if matched:
                name = matched.group(1)
                rows.append(
                    CodeSymbol(
                        name=name,
                        kind=kind,
                        path=str(path),
                        line=index,
                        column=max(line.find(name), 0),
                        signature=line.strip()[:240],
                        references=_reference_count(content, name),
                    )
                )
    return _filter_symbols(rows, query=query)


def _filter_symbols(symbols: list[CodeSymbol], *, query: str) -> list[CodeSymbol]:
    if not query:
        return symbols
    tokens = [token for token in re.split(r"\W+", query.casefold()) if token]
    if not tokens:
        return symbols
    ranked: list[tuple[int, CodeSymbol]] = []
    for symbol in symbols:
        haystack = " ".join(
            [symbol.name, symbol.kind, symbol.path, symbol.signature, symbol.doc]
        ).casefold()
        score = sum(1 for token in tokens if token in haystack)
        if score:
            ranked.append((score, symbol))
    ranked.sort(key=lambda item: (-item[0], item[1].path, item[1].line))
    return [item for _score, item in ranked] or symbols


def _merge_symbols(
    existing: list[CodeSymbol],
    incoming: list[CodeSymbol],
) -> list[CodeSymbol]:
    seen: set[tuple[str, str, int, str]] = set()
    result: list[CodeSymbol] = []
    for symbol in [*existing, *incoming]:
        key = (symbol.path, symbol.name, symbol.line, symbol.kind)
        if key in seen:
            continue
        seen.add(key)
        result.append(symbol)
    return result[:200]


def _reference_count(content: str, name: str) -> int:
    if not name:
        return 0
    return len(re.findall(rf"\b{re.escape(name)}\b", content))


def _text_overlap(query: str, payload: dict[str, Any]) -> bool:
    tokens = {token for token in re.split(r"\W+", query.casefold()) if len(token) >= 3}
    if not tokens:
        return False
    text = " ".join(str(value) for value in payload.values()).casefold()
    return any(token in text for token in tokens)


def _diagnosis_next_tools(
    recommended_action: str,
    recovery_plan: dict[str, Any],
) -> list[str]:
    action = str(recommended_action or "").casefold()
    next_tools = ["engineering_loop_status"]
    if "rollback" in action:
        next_tools.append("patch_rollback")
    if "reread" in action or "read" in action:
        next_tools.append("engineering_lsp_read")
    if "patch" in action or "reread" in action or "read" in action:
        next_tools.extend(["semantic_patch_plan", "patch_prepare", "patch_apply"])
    if "inspect" in action:
        next_tools.extend(["engineering_eval_status", "read_file", "patch_show"])
    raw = recovery_plan.get("next_tools") if isinstance(recovery_plan, dict) else None
    if isinstance(raw, list):
        for item in raw:
            text = str(item or "")
            if text in {"read_file", "search_files"}:
                text = "engineering_lsp_read"
            if text and text not in next_tools:
                next_tools.append(text)
    next_tools.extend(["engineering_eval_run", "engineering_eval_gate"])
    return _dedupe(next_tools)


def _next_tools_from_recovery(
    recovery_plan: dict[str, Any],
    *,
    diagnosed: bool = False,
) -> list[str]:
    if not diagnosed:
        return ["engineering_failure_diagnose"]
    raw = recovery_plan.get("next_tools") if isinstance(recovery_plan, dict) else None
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw if str(item or "")]
    return ["engineering_loop_status", "read_file", "patch_prepare"]


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    raw = read_json(_LOOPS_PATH, {})
    if not isinstance(raw, dict):
        return
    for loop_id, payload in raw.items():
        loop = _loop_from_payload(loop_id, payload)
        if loop is not None:
            _LOOPS[loop.loop_id] = loop


def _loop_from_payload(loop_id: object, payload: object) -> EngineeringLoop | None:
    if not isinstance(payload, dict):
        return None
    try:
        loop = EngineeringLoop(
            loop_id=str(payload.get("loop_id") or loop_id or ""),
            user_id=str(payload.get("user_id", "") or ""),
            session_key=str(payload.get("session_key", "") or ""),
            task=str(payload.get("task", "") or ""),
            stage=str(payload.get("stage", "") or "created"),  # type: ignore[arg-type]
            files=[str(item) for item in payload.get("files", []) or []],
            symbols=[
                CodeSymbol(
                    name=str(item.get("name", "") or ""),
                    kind=str(item.get("kind", "") or ""),
                    path=str(item.get("path", "") or ""),
                    line=int(item.get("line", 0) or 0),
                    column=int(item.get("column", 0) or 0),
                    signature=str(item.get("signature", "") or ""),
                    doc=str(item.get("doc", "") or ""),
                    references=int(item.get("references", 0) or 0),
                )
                for item in payload.get("symbols", []) or []
                if isinstance(item, dict)
            ],
            semantic_patch_plan=dict(payload.get("semantic_patch_plan") or {}),
            patch_operation_id=str(payload.get("patch_operation_id", "") or ""),
            eval_id=str(payload.get("eval_id", "") or ""),
            checkpoint_ids=dict(payload.get("checkpoint_ids") or {}),
            rollback_operation_id=str(payload.get("rollback_operation_id", "") or ""),
            recovery_plan=dict(payload.get("recovery_plan") or {}),
            diagnosis=dict(payload.get("diagnosis") or {}),
            attempt=int(payload.get("attempt", 0) or 0),
            failure_reason=str(payload.get("failure_reason", "") or ""),
            created_at=float(payload.get("created_at") or time.time()),
            updated_at=float(payload.get("updated_at") or time.time()),
        )
        loop.events = [
            EngineeringLoopEvent(
                kind=str(item.get("kind", "") or ""),
                timestamp=float(item.get("timestamp") or time.time()),
                payload=dict(item.get("payload") or {}),
            )
            for item in payload.get("events", []) or []
            if isinstance(item, dict)
        ]
        return loop
    except Exception:
        return None


def _save_loops() -> None:
    write_json(
        _LOOPS_PATH,
        {loop_id: loop.to_record() for loop_id, loop in sorted(_LOOPS.items())},
    )


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


__all__ = [
    "CodeSymbol",
    "EngineeringLoop",
    "EngineeringLoopEvent",
    "bind_eval",
    "bind_patch_operation",
    "build_semantic_patch_plan",
    "complete_engineering_loop",
    "create_engineering_loop",
    "diagnose_eval_failure",
    "eval_gate",
    "get_engineering_loop",
    "list_engineering_loops",
    "mark_loop_rolled_back",
    "read_code_symbols",
    "record_lsp_read",
    "record_semantic_patch_plan",
    "update_loop_from_eval",
]
