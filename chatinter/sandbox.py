import asyncio
import contextlib
from dataclasses import dataclass
import os
import shlex
import shutil
import time
from typing import Any

from zhenxun.services.log import logger

_SANDBOX_ROOT = os.path.join("data", "cache", "chatinter_sandbox")
_SANDBOX_IDLE_TTL = 900
_SANDBOX_CLEAN_INTERVAL = 120
_SANDBOX_MAX_CONCURRENCY = 4
_MAX_STDOUT = 2000
_MAX_STDERR = 1000
_SAFE_SHELL_PREFIX = {"echo", "pwd", "dir", "ls", "whoami", "date"}


@dataclass
class SandboxSession:
    session_id: str
    workdir: str
    created_at: float
    last_used: float


class SessionSandboxManager:
    def __init__(self):
        self._sessions: dict[str, SandboxSession] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(_SANDBOX_MAX_CONCURRENCY)
        self._cleanup_task: asyncio.Task | None = None

    async def startup(self) -> None:
        await asyncio.to_thread(os.makedirs, _SANDBOX_ROOT, exist_ok=True)
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def get_session(self, session_id: str) -> SandboxSession:
        if not session_id:
            session_id = "global"

        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_used = time.time()
                return session

            safe_id = "".join(
                ch for ch in session_id if ch.isalnum() or ch in {"-", "_"}
            )
            if not safe_id:
                safe_id = "global"
            workdir = os.path.join(_SANDBOX_ROOT, safe_id)
            await asyncio.to_thread(os.makedirs, workdir, exist_ok=True)
            now = time.time()
            session = SandboxSession(
                session_id=session_id,
                workdir=workdir,
                created_at=now,
                last_used=now,
            )
            self._sessions[session_id] = session
            return session

    async def execute_shell(
        self,
        session_id: str,
        command: str,
        timeout: int = 3,
        allowed_prefix: set[str] | None = None,
    ) -> dict[str, Any]:
        session = await self.get_session(session_id)
        allowed = allowed_prefix or _SAFE_SHELL_PREFIX
        command = command.strip()
        if not command:
            return {"ok": False, "error": "命令为空"}

        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except Exception as exc:
            return {"ok": False, "error": f"命令解析失败: {exc}"}
        if not argv:
            return {"ok": False, "error": "命令为空"}

        command_head = argv[0].lower()
        if command_head not in allowed:
            return {"ok": False, "error": f"命令不在白名单: {argv[0]}"}

        async with self._semaphore:
            shell = os.name == "nt"
            if shell:
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=session.workdir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=session.workdir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=max(1, timeout),
                )
            except asyncio.TimeoutError:
                with contextlib.suppress(Exception):
                    process.kill()
                return {"ok": False, "error": "命令执行超时"}

        session.last_used = time.time()
        text_out = (stdout or b"").decode("utf-8", errors="ignore")[:_MAX_STDOUT]
        text_err = (stderr or b"").decode("utf-8", errors="ignore")[:_MAX_STDERR]
        return {
            "ok": process.returncode == 0,
            "return_code": process.returncode,
            "stdout": text_out,
            "stderr": text_err,
            "workdir": session.workdir,
        }

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(_SANDBOX_CLEAN_INTERVAL)
            try:
                await self.cleanup_expired()
            except Exception as exc:
                logger.warning(f"chatinter sandbox cleanup failed: {exc}")

    async def cleanup_expired(self) -> None:
        now = time.time()
        expired: list[SandboxSession] = []

        async with self._lock:
            for key, session in list(self._sessions.items()):
                if now - session.last_used >= _SANDBOX_IDLE_TTL:
                    expired.append(session)
                    self._sessions.pop(key, None)

        for session in expired:
            await asyncio.to_thread(shutil.rmtree, session.workdir, True)


_SANDBOX_MANAGER: SessionSandboxManager | None = None


async def get_sandbox_manager() -> SessionSandboxManager:
    global _SANDBOX_MANAGER
    if _SANDBOX_MANAGER is None:
        _SANDBOX_MANAGER = SessionSandboxManager()
        await _SANDBOX_MANAGER.startup()
    return _SANDBOX_MANAGER
