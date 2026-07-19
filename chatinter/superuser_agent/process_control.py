"""Small cross-platform helpers for Agent subprocess trees."""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

_WINDOWS_JOBS: dict[int, Any] = {}


def subprocess_group_kwargs() -> dict[str, Any]:
    """Start Unix commands in a group that can be terminated as one unit."""

    return {} if os.name == "nt" else {"start_new_session": True}


def attach_process_tree(process: asyncio.subprocess.Process) -> None:
    """Bind a Windows subprocess tree to a kill-on-close Job Object."""

    if os.name != "nt":
        return
    import win32api
    import win32con
    import win32job

    job = win32job.CreateJobObject(None, "")
    info = win32job.QueryInformationJobObject(
        job,
        win32job.JobObjectExtendedLimitInformation,
    )
    info["BasicLimitInformation"]["LimitFlags"] |= (
        win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    win32job.SetInformationJobObject(
        job,
        win32job.JobObjectExtendedLimitInformation,
        info,
    )
    process_handle = win32api.OpenProcess(
        win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
        False,
        process.pid,
    )
    try:
        win32job.AssignProcessToJobObject(job, process_handle)
    except Exception:
        win32api.CloseHandle(job)
        raise
    finally:
        win32api.CloseHandle(process_handle)
    _WINDOWS_JOBS[process.pid] = job


def release_process_tree(process: asyncio.subprocess.Process) -> None:
    """Release process-group resources after normal completion."""

    if os.name != "nt":
        return
    import win32api

    job = _WINDOWS_JOBS.pop(process.pid, None)
    if job is not None:
        win32api.CloseHandle(job)


async def terminate_process_tree(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 2.0,
) -> None:
    """Terminate a subprocess tree and wait until the root process exits."""

    if os.name == "nt":
        await _terminate_windows_tree(process, grace_seconds=grace_seconds)
        return

    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=max(grace_seconds, 0.1))
    except asyncio.TimeoutError:
        pass


    _signal_process_group(process.pid, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()


async def _terminate_windows_tree(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float,
) -> None:
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(
                process.wait(), timeout=max(grace_seconds, 0.1)
            )
        except asyncio.TimeoutError:
            pass

    import win32api
    import win32job

    job = _WINDOWS_JOBS.pop(process.pid, None)
    if job is not None:
        try:
            win32job.TerminateJobObject(job, 1)
        finally:
            win32api.CloseHandle(job)
    elif process.returncode is None:
        process.kill()
    if process.returncode is None:
        await process.wait()


def _signal_process_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return


__all__ = [
    "attach_process_tree",
    "release_process_tree",
    "subprocess_group_kwargs",
    "terminate_process_tree",
]
