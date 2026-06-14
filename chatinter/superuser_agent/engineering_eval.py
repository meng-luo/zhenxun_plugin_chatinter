"""Engineering task eval plans for superuser Agent code work."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any
import uuid

from zhenxun.services.llm.types.models import ToolResult

from ..persistence import read_json, state_path, write_json
from .audit_log import record_audit_event

_EVALS_PATH = state_path("engineering_evals.json")
_EVALS: dict[str, "EngineeringEval"] = {}
_LOADED = False


@dataclass
class EvalStep:
    kind: str
    description: str
    command: str = ""
    cwd: str | None = None
    status: str = "pending"
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineeringEval:
    eval_id: str
    user_id: str
    session_key: str
    task: str
    files: list[str]
    tests: list[str]
    rollback_operation_id: str = ""
    patch_operation_id: str = ""
    suggested_tests: list[str] = field(default_factory=list)
    failure_observations: list[dict[str, Any]] = field(default_factory=list)
    recovery_plan: dict[str, Any] = field(default_factory=dict)
    patch_checkpoints: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "planned"
    steps: list[EvalStep] = field(default_factory=list)
    error: str = ""
    last_recommended_action: str = ""

    def public_payload(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "user_id": self.user_id,
            "session_key": self.session_key,
            "task": self.task,
            "files": list(self.files),
            "tests": list(self.tests),
            "rollback_operation_id": self.rollback_operation_id,
            "patch_operation_id": self.patch_operation_id,
            "suggested_tests": list(self.suggested_tests),
            "failure_observations": list(self.failure_observations[-8:]),
            "recovery_plan": dict(self.recovery_plan),
            "patch_checkpoints": dict(self.patch_checkpoints),
            "created_at": int(self.created_at),
            "updated_at": int(self.updated_at),
            "status": self.status,
            "error": self.error,
            "last_recommended_action": self.last_recommended_action,
            "steps": [asdict(step) for step in self.steps],
        }

    def to_record(self) -> dict[str, Any]:
        return self.public_payload()


def create_engineering_eval(
    *,
    actor: dict[str, str],
    task: str,
    files: list[str],
    tests: list[str],
    rollback_operation_id: str = "",
    patch_operation_id: str = "",
) -> EngineeringEval:
    _ensure_loaded()
    files = [str(item) for item in files if str(item or "")]
    suggested = suggest_test_commands(files=files, task=task)
    normalized_tests = _dedupe(
        [str(item) for item in tests if str(item or "")] + suggested
    )
    eval_run = EngineeringEval(
        eval_id=uuid.uuid4().hex[:12],
        user_id=actor["user_id"],
        session_key=actor["session_key"],
        task=str(task or ""),
        files=files,
        tests=normalized_tests,
        rollback_operation_id=str(rollback_operation_id or ""),
        patch_operation_id=str(patch_operation_id or rollback_operation_id or ""),
        suggested_tests=suggested,
        patch_checkpoints=_patch_checkpoints_for(
            patch_operation_id or rollback_operation_id
        ),
    )
    eval_run.steps = _default_steps(eval_run)
    _EVALS[eval_run.eval_id] = eval_run
    _save_evals()
    record_audit_event(
        event="engineering_eval_planned",
        user_id=eval_run.user_id,
        session_key=eval_run.session_key,
        action="engineering_eval_plan",
        payload=eval_run.public_payload(),
    )
    return eval_run


def get_engineering_eval(eval_id: str) -> EngineeringEval | None:
    _ensure_loaded()
    return _EVALS.get(str(eval_id or "").strip())


def list_engineering_evals(
    *,
    user_id: str,
    session_key: str,
    limit: int = 20,
) -> list[EngineeringEval]:
    _ensure_loaded()
    rows = [
        item
        for item in _EVALS.values()
        if item.user_id == str(user_id or "")
        and item.session_key == str(session_key or "")
    ]
    rows.sort(key=lambda item: item.updated_at, reverse=True)
    return rows[: max(1, min(int(limit or 20), 100))]


def mark_eval_step(
    *,
    eval_id: str,
    step_index: int,
    status: str,
    result: dict[str, Any],
) -> EngineeringEval | None:
    eval_run = get_engineering_eval(eval_id)
    if eval_run is None:
        return None
    if step_index < 0 or step_index >= len(eval_run.steps):
        return eval_run
    eval_run.steps[step_index].status = status
    eval_run.steps[step_index].result = dict(result or {})
    eval_run.updated_at = time.time()
    if status == "failed":
        eval_run.status = "failed"
        observation = build_failure_observation(eval_run, step_index=step_index)
        if observation:
            eval_run.failure_observations.append(observation)
            eval_run.recovery_plan = dict(observation.get("recovery_plan") or {})
            eval_run.last_recommended_action = str(
                observation.get("recommended_next_action", "") or ""
            )
    elif any(step.status == "failed" for step in eval_run.steps):
        eval_run.status = "failed"
        eval_run.last_recommended_action = (
            eval_run.last_recommended_action or "inspect_failure_observation"
        )
    elif _all_runnable_steps_passed(eval_run):
        eval_run.status = "passed"
        eval_run.last_recommended_action = "continue_or_summarize"
        eval_run.recovery_plan = {}
    else:
        eval_run.status = "running"
        eval_run.last_recommended_action = "run_next_eval_step"
    _save_evals()
    return eval_run


def suggest_test_commands(*, files: list[str], task: str = "") -> list[str]:
    normalized_files = [str(path or "").replace("\\", "/") for path in files if path]
    tests: list[str] = []
    py_files = [path for path in normalized_files if path.endswith(".py")]
    if py_files:
        tests.append(
            "py -3 -m compileall -q " + " ".join(_quote(path) for path in py_files[:20])
        )
    if any(
        path.endswith((".html", ".css", ".js", ".ts", ".vue"))
        for path in normalized_files
    ):
        tests.append("git diff --check")
    if any("plugins/chatinter" in path for path in normalized_files):
        tests.append("py -3 -m compileall -q zhenxun/plugins/chatinter")
    for path in normalized_files:
        if path.endswith(".json"):
            tests.append(f"py -3 -m json.tool {_quote(path)}")
    if not tests:
        tests.append("git diff --check")
    return _dedupe(tests)[:6]


def build_failure_observation(
    eval_run: EngineeringEval,
    *,
    step_index: int,
) -> dict[str, Any]:
    if step_index < 0 or step_index >= len(eval_run.steps):
        return {}
    step = eval_run.steps[step_index]
    result = step.result or {}
    stdout = str(result.get("stdout", "") or "")
    stderr = str(result.get("stderr", "") or "")
    error = str(result.get("error", "") or "")
    combined = "\n".join(part for part in [stderr, stdout, error] if part)
    lower = combined.lower()
    needs_reread = any(
        marker in lower
        for marker in [
            "no such file",
            "not found",
            "nameerror",
            "attributeerror",
            "importerror",
            "moduleNotFoundError".lower(),
            "syntaxerror",
        ]
    )
    needs_second_patch = any(
        marker in lower
        for marker in [
            "failed",
            "error",
            "assert",
            "syntaxerror",
            "typeerror",
            "nameerror",
            "attributeerror",
        ]
    )
    suggest_rollback = bool(
        eval_run.rollback_operation_id
        and any(
            marker in lower
            for marker in [
                "cannot import",
                "syntaxerror",
                "fatal",
                "corrupt",
                "permission denied",
            ]
        )
    )
    recommended = _recommended_action(
        needs_reread=needs_reread,
        needs_second_patch=needs_second_patch,
        suggest_rollback=suggest_rollback,
    )
    return {
        "ok": False,
        "status": "engineering_eval_failure_observation",
        "eval_id": eval_run.eval_id,
        "step_index": step_index,
        "command": step.command,
        "failed_step": asdict(step),
        "needs_reread": needs_reread,
        "needs_second_patch": needs_second_patch,
        "suggest_rollback": suggest_rollback,
        "recommended_next_action": recommended,
        "recovery_plan": _recovery_plan(
            eval_run=eval_run,
            recommended_action=recommended,
            needs_reread=needs_reread,
            needs_second_patch=needs_second_patch,
            suggest_rollback=suggest_rollback,
            combined_output=combined,
        ),
        "rollback_operation_id": eval_run.rollback_operation_id,
        "patch_operation_id": eval_run.patch_operation_id,
        "patch_checkpoints": dict(eval_run.patch_checkpoints),
        "files_to_reread": list(eval_run.files[:12]) if needs_reread else [],
        "do_not_repeat_failed_command": True,
        "failed_command_signature": _command_signature(step.command),
        "reason": _failure_reason(
            needs_reread=needs_reread,
            needs_second_patch=needs_second_patch,
            suggest_rollback=suggest_rollback,
        ),
    }


def tool_result(ok: bool, status: str, **payload: Any) -> ToolResult:
    return ToolResult(
        output={"ok": ok, "status": status, **payload}, display_content=status
    )


def _default_steps(eval_run: EngineeringEval) -> list[EvalStep]:
    steps = [
        EvalStep(
            kind="read_code",
            description="Read and understand target files before editing.",
            result={"files": eval_run.files},
        ),
        EvalStep(
            kind="modify_code",
            description=(
                "Apply changes through patch_prepare/patch_apply or "
                "transactional file tools."
            ),
        ),
    ]
    for command in eval_run.tests:
        steps.append(
            EvalStep(
                kind="run_test",
                description=f"Run validation command: {command}",
                command=command,
            )
        )
    if eval_run.rollback_operation_id:
        steps.append(
            EvalStep(
                kind="rollback_available",
                description="Rollback operation is available if validation fails.",
                result={"operation_id": eval_run.rollback_operation_id},
            )
        )
    return steps


def _all_runnable_steps_passed(eval_run: EngineeringEval) -> bool:
    runnable = [step for step in eval_run.steps if step.kind == "run_test"]
    if not runnable:
        return False
    return all(step.status in {"passed", "skipped"} for step in runnable)


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    raw = read_json(_EVALS_PATH, {})
    if not isinstance(raw, dict):
        return
    for eval_id, payload in raw.items():
        item = _eval_from_payload(eval_id, payload)
        if item is not None:
            _EVALS[item.eval_id] = item


def _eval_from_payload(eval_id: object, payload: object) -> EngineeringEval | None:
    if not isinstance(payload, dict):
        return None
    try:
        item = EngineeringEval(
            eval_id=str(payload.get("eval_id") or eval_id or ""),
            user_id=str(payload.get("user_id", "") or ""),
            session_key=str(payload.get("session_key", "") or ""),
            task=str(payload.get("task", "") or ""),
            files=[str(value) for value in payload.get("files", []) or []],
            tests=[str(value) for value in payload.get("tests", []) or []],
            rollback_operation_id=str(payload.get("rollback_operation_id", "") or ""),
            patch_operation_id=str(payload.get("patch_operation_id", "") or ""),
            suggested_tests=[
                str(value) for value in payload.get("suggested_tests", []) or []
            ],
            failure_observations=[
                dict(value)
                for value in payload.get("failure_observations", []) or []
                if isinstance(value, dict)
            ],
            patch_checkpoints=dict(payload.get("patch_checkpoints") or {}),
            created_at=float(payload.get("created_at") or time.time()),
            updated_at=float(payload.get("updated_at") or time.time()),
            status=str(payload.get("status", "") or "planned"),
            error=str(payload.get("error", "") or ""),
            last_recommended_action=str(
                payload.get("last_recommended_action", "") or ""
            ),
            recovery_plan=dict(payload.get("recovery_plan") or {}),
        )
        item.steps = [
            EvalStep(
                kind=str(step.get("kind", "") or ""),
                description=str(step.get("description", "") or ""),
                command=str(step.get("command", "") or ""),
                cwd=str(step.get("cwd", "") or "") or None,
                status=str(step.get("status", "") or "pending"),
                result=dict(step.get("result") or {}),
            )
            for step in payload.get("steps", [])
            if isinstance(step, dict)
        ]
        return item
    except Exception:
        return None


def _save_evals() -> None:
    write_json(
        _EVALS_PATH,
        {eval_id: item.to_record() for eval_id, item in sorted(_EVALS.items())},
    )


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _quote(path: str) -> str:
    return f'"{path}"' if " " in path else path


def _failure_reason(
    *,
    needs_reread: bool,
    needs_second_patch: bool,
    suggest_rollback: bool,
) -> str:
    if suggest_rollback:
        return "validation failed severely; rollback should be considered"
    if needs_reread and needs_second_patch:
        return "validation failed; reread relevant code then prepare a second patch"
    if needs_reread:
        return "validation failed; reread relevant code before changing again"
    if needs_second_patch:
        return "validation failed; a follow-up patch may be needed"
    return "validation failed; inspect command output"


def _recommended_action(
    *,
    needs_reread: bool,
    needs_second_patch: bool,
    suggest_rollback: bool,
) -> str:
    if suggest_rollback:
        return "rollback_or_reread_before_patch"
    if needs_reread:
        return "reread_code"
    if needs_second_patch:
        return "prepare_second_patch"
    return "inspect_output"


def _recovery_plan(
    *,
    eval_run: EngineeringEval,
    recommended_action: str,
    needs_reread: bool,
    needs_second_patch: bool,
    suggest_rollback: bool,
    combined_output: str,
) -> dict[str, Any]:
    next_tools: list[str] = []
    if needs_reread:
        next_tools.extend(["read_file", "search_files"])
    if needs_second_patch:
        next_tools.extend(["patch_prepare", "patch_apply"])
    if suggest_rollback and eval_run.rollback_operation_id:
        next_tools.append("patch_rollback")
    if not next_tools:
        next_tools.extend(["engineering_eval_status", "read_file"])
    return {
        "recommended_action": recommended_action,
        "next_tools": _dedupe(next_tools),
        "patch_operation_id": eval_run.patch_operation_id,
        "rollback_operation_id": eval_run.rollback_operation_id,
        "patch_checkpoints": dict(eval_run.patch_checkpoints),
        "files_to_reread": list(eval_run.files[:12]) if needs_reread else [],
        "failed_command_signature": _command_signature(
            next(
                (
                    step.command
                    for step in eval_run.steps
                    if step.status == "failed" and step.command
                ),
                "",
            )
        ),
        "do_not_repeat_failed_command": True,
        "suggested_patch_strategy": _patch_strategy(
            needs_reread=needs_reread,
            needs_second_patch=needs_second_patch,
            suggest_rollback=suggest_rollback,
        ),
        "evidence_tail": combined_output[-1200:],
        "instruction": (
            "Use this recovery_plan as the next engineering step. Do not keep "
            "rerunning the same failed command unchanged; reread or patch first "
            "unless the plan says rollback."
        ),
    }


def _patch_strategy(
    *,
    needs_reread: bool,
    needs_second_patch: bool,
    suggest_rollback: bool,
) -> str:
    if suggest_rollback:
        return "rollback first if the patch broke imports/syntax severely, then reread."
    if needs_reread and needs_second_patch:
        return "reread the failing files and prepare a focused second patch."
    if needs_reread:
        return "reread relevant code before deciding on changes."
    if needs_second_patch:
        return "prepare a small follow-up patch targeted at the failure."
    return "inspect output, then choose reread, second patch, or rollback."


def _patch_checkpoints_for(operation_id: str) -> dict[str, str]:
    if not operation_id:
        return {}
    try:
        from .patch_operations import get_patch_operation

        operation = get_patch_operation(operation_id)
    except Exception:
        operation = None
    if operation is None:
        return {}
    return {
        "pre_checkpoint_id": operation.pre_checkpoint_id,
        "post_checkpoint_id": operation.post_checkpoint_id,
        "rollback_checkpoint_id": operation.rollback_checkpoint_id,
        "failure_checkpoint_id": operation.failure_checkpoint_id,
    }


def _command_signature(command: str) -> str:
    text = " ".join(str(command or "").split())
    if not text:
        return ""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "EngineeringEval",
    "EvalStep",
    "build_failure_observation",
    "create_engineering_eval",
    "get_engineering_eval",
    "list_engineering_evals",
    "mark_eval_step",
    "suggest_test_commands",
]
