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

import sys
from typing import Callable

import psutil

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
# signal_process_tree()
# ---------------------------------------------------------------------------
def _children_of(pid: int) -> list[psutil.Process]:
    """Return all descendants of *pid* via psutil (empty list if not found)."""
    try:
        parent = psutil.Process(pid)
        return parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return []


def signal_process_tree(pid: int, kill: bool = False) -> None:
    """Signal the whole process tree rooted at *pid* via psutil, without waiting.

    This is the portable counterpart to ``os.killpg`` for platforms that have no
    process groups (Windows). It SIGTERMs (``terminate()``) the root and every
    descendant, or SIGKILLs (``kill()``) them when *kill* is True.

    It does NOT block waiting for exit: callers use their own (async) poll loop
    to wait and to escalate from terminate to kill, so the event loop is never
    starved. The process list is captured once up front so children spawned
    after the snapshot are not chased, and already-gone processes are ignored.
    """
    procs: list[psutil.Process] = _children_of(pid)
    try:
        procs.insert(0, psutil.Process(pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for p in procs:
        try:
            p.kill() if kill else p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
