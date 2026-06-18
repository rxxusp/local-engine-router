"""Platform-dispatched system-memory and process-control helpers.

Replaces direct /proc/meminfo reads and pgrep/killpg calls so that
``_await_memory_settle`` and process termination work correctly on every OS
(Linux, macOS, Windows) rather than silently no-oping on non-Linux hosts.

Public API
----------
available_bytes() -> int
    Current MemAvailable (Linux: /proc/meminfo fast path; other: psutil).
    The internal reader is overridable via ``_mem_reader`` for tests.

terminate_process_tree(pid, term_timeout, kill_timeout)
    SIGTERM the whole process tree rooted at *pid*, wait *term_timeout*,
    then SIGKILL survivors. On POSIX, where a process group is available,
    os.killpg is the fast path (reaps forked workers in one call). psutil
    is the portable fallback (and the primary path on Windows).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Callable

import psutil

log = logging.getLogger("router.sysmem")

# ---------------------------------------------------------------------------
# Injectable hook — point tests at a fake reader without touching the module.
# ---------------------------------------------------------------------------
# The callable must return available memory in **bytes** (int), or raise
# OSError if the value is unavailable. Replace for tests:
#
#   import router.sysmem as sysmem
#   sysmem._mem_reader = lambda: some_bytes_value
#
_mem_reader: Callable[[], int] | None = None


# ---------------------------------------------------------------------------
# available_bytes()
# ---------------------------------------------------------------------------
def _linux_available_bytes() -> int:
    """Read MemAvailable from /proc/meminfo and return bytes.

    This is the same fast path the original _read_mem_available_kb used,
    kept separate so it can be tested independently.
    """
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb * 1024
    raise OSError("MemAvailable not found in /proc/meminfo")


def available_bytes() -> int:
    """Return available system memory in bytes.

    Priority:
    1. The injectable ``_mem_reader`` hook (for tests or custom probes).
    2. Linux /proc/meminfo (fast, no subprocess, same source as before).
    3. psutil.virtual_memory().available (macOS, Windows, any other OS).

    Raises ``OSError`` only when every path fails (callers treat that as
    "can't read memory — skip the wait").
    """
    if _mem_reader is not None:
        return _mem_reader()

    if sys.platform == "linux":
        try:
            return _linux_available_bytes()
        except (OSError, ValueError, StopIteration):
            pass  # fall through to psutil

    vm = psutil.virtual_memory()
    return int(vm.available)


# ---------------------------------------------------------------------------
# terminate_process_tree()
# ---------------------------------------------------------------------------
def _pgid_for(pid: int) -> int | None:
    """Return the process group id of *pid*, or None if unavailable."""
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        return None


def _send_signal_posix(pid: int, sig: int) -> None:
    """Send *sig* to the whole process group of *pid* (POSIX fast path),
    then fall back to signalling *pid* directly if the group fails."""
    pgid = _pgid_for(pid)
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    # Fallback: signal the process directly (handles strays not in our group).
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _children_of(pid: int) -> list[psutil.Process]:
    """Return all descendants of *pid* via psutil (empty list if not found)."""
    try:
        parent = psutil.Process(pid)
        return parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return []


def _terminate_psutil(pid: int) -> None:
    """Terminate the whole tree rooted at *pid* via psutil (portable path)."""
    children = _children_of(pid)
    # Collect all processes BEFORE terminating so we have the full picture.
    procs: list[psutil.Process] = []
    try:
        procs.append(psutil.Process(pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    procs.extend(children)

    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _kill_survivors(pid: int) -> None:
    """SIGKILL all remaining live processes in the tree rooted at *pid*."""
    procs: list[psutil.Process] = []
    try:
        procs.append(psutil.Process(pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    procs.extend(_children_of(pid))

    for p in procs:
        if p.is_running():
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


def _wait_all_gone(pid: int, timeout: float) -> bool:
    """Return True when all processes in the tree are gone within *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = False
        try:
            p = psutil.Process(pid)
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                alive = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if not alive:
            # Check children too.
            children = _children_of(pid)
            if not children:
                return True
            alive = any(
                c.is_running() and c.status() != psutil.STATUS_ZOMBIE
                for c in children
            )
            if not alive:
                return True
        time.sleep(0.05)
    return False


def terminate_process_tree(
    pid: int,
    term_timeout: float = 10.0,
    kill_timeout: float = 5.0,
) -> None:
    """Terminate the process tree rooted at *pid*.

    On POSIX systems where the process leads its own session/group (launched
    with ``start_new_session=True``), ``os.killpg`` is used as the fast path
    because it reaps forked workers in one call. ``psutil`` is the portable
    fallback (and the primary path on Windows / when a group is unavailable).

    Steps:
    1. SIGTERM the whole tree (group signal on POSIX, psutil on others).
    2. Wait up to *term_timeout* seconds for all processes to exit.
    3. SIGKILL any survivors; wait up to *kill_timeout* seconds.
    """
    if sys.platform != "win32":
        # POSIX fast path: signal the whole process group.
        _send_signal_posix(pid, signal.SIGTERM)
    else:
        _terminate_psutil(pid)

    if _wait_all_gone(pid, term_timeout):
        return

    log.warning("terminate_process_tree: pid %s did not exit in %.1fs; SIGKILL", pid, term_timeout)
    if sys.platform != "win32":
        _send_signal_posix(pid, signal.SIGKILL)
    else:
        _kill_survivors(pid)

    _wait_all_gone(pid, kill_timeout)
