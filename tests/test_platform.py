"""MM1 — cross-platform memory and process-control unit tests.

Covers:
  * available_bytes() returns a positive int on this host.
  * _await_memory_settle plateau logic against a mocked rising-then-flat curve.
  * signal_process_tree terminate/kill against real short-lived subprocesses.
  * The engine's no-process-group (Windows) teardown branch.
  * macOS/Windows memory paths reachable via monkeypatching sys.platform / psutil.
  * Injectable _mem_reader hook is honoured.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import signal
import sys
import time

import psutil
import pytest

import router.sysmem as sysmem
from router.engines import EngineManager

from conftest import make_config, make_manager_with_fakes, FakeEngine


# ===========================================================================
# available_bytes — basic sanity
# ===========================================================================
def test_available_bytes_returns_positive_int():
    """available_bytes() must return a positive integer on this host."""
    # The autouse _instant_memory_settle fixture injects a raising reader;
    # we need to clear it here so we test the real implementation.
    saved = sysmem._mem_reader
    sysmem._mem_reader = None
    try:
        result = sysmem.available_bytes()
        assert isinstance(result, int), f"expected int, got {type(result)}"
        assert result > 0, f"expected positive value, got {result}"
    finally:
        sysmem._mem_reader = saved


def test_available_bytes_injectable_hook():
    """The _mem_reader hook overrides the OS path."""
    saved = sysmem._mem_reader
    sysmem._mem_reader = lambda: 42_000_000_000
    try:
        assert sysmem.available_bytes() == 42_000_000_000
    finally:
        sysmem._mem_reader = saved


def test_available_bytes_raises_oserror_when_hook_raises():
    """If the hook raises OSError, available_bytes() propagates it."""
    saved = sysmem._mem_reader
    sysmem._mem_reader = lambda: (_ for _ in ()).throw(OSError("nope"))
    try:
        with pytest.raises(OSError):
            sysmem.available_bytes()
    finally:
        sysmem._mem_reader = saved


def test_available_bytes_macos_path(monkeypatch):
    """On macOS (sys.platform='darwin'), psutil is used instead of /proc/meminfo."""
    monkeypatch.setattr(sysmem, "_mem_reader", None)
    monkeypatch.setattr(sys, "platform", "darwin")

    class _FakeVM:
        available = 8_000_000_000

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVM())
    result = sysmem.available_bytes()
    assert result == 8_000_000_000


def test_available_bytes_windows_path(monkeypatch):
    """On Windows (sys.platform='win32'), psutil is used."""
    monkeypatch.setattr(sysmem, "_mem_reader", None)
    monkeypatch.setattr(sys, "platform", "win32")

    class _FakeVM:
        available = 16_000_000_000

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVM())
    result = sysmem.available_bytes()
    assert result == 16_000_000_000


def test_available_bytes_linux_proc_meminfo_fallback_to_psutil(monkeypatch, tmp_path):
    """On Linux, if /proc/meminfo is unreadable, psutil is the fallback."""
    monkeypatch.setattr(sysmem, "_mem_reader", None)
    monkeypatch.setattr(sys, "platform", "linux")

    # Make _linux_available_bytes raise so we fall through to psutil.
    def _bad_read():
        raise OSError("simulated /proc unreadable")

    monkeypatch.setattr(sysmem, "_linux_available_bytes", _bad_read)

    class _FakeVM:
        available = 5_000_000_000

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVM())
    result = sysmem.available_bytes()
    assert result == 5_000_000_000


# ===========================================================================
# _await_memory_settle — plateau logic
# ===========================================================================
async def test_memory_settle_plateau_stops_early():
    """The settle loop must return as soon as two consecutive samples differ by
    less than 1 GiB, regardless of the timeout."""
    from router.config import ModelSpec

    # Build a simple two-fake manager so we can call _await_memory_settle.
    models = [
        ModelSpec(id="m-a", engine="a", display_name="a"),
    ]
    cfg = make_config(models=models, swap_memory_settle_timeout_s=10.0)
    fakes = {"a": FakeEngine("a")}
    mgr = make_manager_with_fakes(fakes, cfg=cfg)

    # Rising-then-plateau curve: first call returns a high value, then rises
    # a little, then plateaus (< 1 GiB change). We inject via sysmem._mem_reader
    # (available_bytes returns bytes; _read_mem_available_kb divides by 1024).
    # Rising-then-plateau, with a repeating tail so the loop never exits merely
    # because the readings ran out (StopIteration); it must exit on the plateau.
    readings_bytes = itertools.chain(
        iter([
            80 * 1024 ** 3,      # first sample: 80 GiB
            90 * 1024 ** 3,      # second: rose 10 GiB -> not stable
            90 * 1024 ** 3 + 1,  # third: rose < 1 GiB -> stable=1
            90 * 1024 ** 3 + 2,  # fourth: rose < 1 GiB -> stable=2 -> return
        ]),
        itertools.repeat(90 * 1024 ** 3 + 2),
    )

    saved = sysmem._mem_reader
    sysmem._mem_reader = lambda: next(readings_bytes)
    try:
        t0 = asyncio.get_running_loop().time()
        await mgr._await_memory_settle(10.0)
        dt = asyncio.get_running_loop().time() - t0
        # Must finish on the plateau, well before the 10s timeout.
        assert dt < 5.0, f"settle did not stop early on plateau: {dt:.2f}s"
    finally:
        sysmem._mem_reader = saved


async def test_memory_settle_none_returns_immediately():
    """When _read_mem_available_kb returns None, the settle returns immediately."""
    from router.config import ModelSpec
    models = [ModelSpec(id="m-a", engine="a", display_name="a")]
    cfg = make_config(models=models, swap_memory_settle_timeout_s=10.0)
    fakes = {"a": FakeEngine("a")}
    mgr = make_manager_with_fakes(fakes, cfg=cfg)

    # The autouse fixture already injects a raising hook so this is implicit,
    # but we verify the timing explicitly.
    t0 = asyncio.get_running_loop().time()
    await mgr._await_memory_settle(10.0)
    dt = asyncio.get_running_loop().time() - t0
    assert dt < 1.0, f"expected instant return, took {dt:.2f}s"


async def test_memory_settle_timeout_fires():
    """When memory never plateaus, the settle wait exits at timeout_s."""
    from router.config import ModelSpec
    models = [ModelSpec(id="m-a", engine="a", display_name="a")]
    cfg = make_config(models=models, swap_memory_settle_timeout_s=0.3)
    fakes = {"a": FakeEngine("a")}
    mgr = make_manager_with_fakes(fakes, cfg=cfg)

    # Always rising by more than 1 GiB per sample.
    counter = [0]
    def _always_rising():
        counter[0] += 1
        return counter[0] * 2 * 1024 ** 3  # 2 GiB more each time

    saved = sysmem._mem_reader
    sysmem._mem_reader = _always_rising
    try:
        t0 = asyncio.get_running_loop().time()
        await mgr._await_memory_settle(0.3)
        dt = asyncio.get_running_loop().time() - t0
        # Should have waited close to the timeout.
        assert 0.25 <= dt < 2.0, f"unexpected timeout duration: {dt:.2f}s"
    finally:
        sysmem._mem_reader = saved


# ===========================================================================
# signal_process_tree — portable, signal-only teardown (no internal wait)
# ===========================================================================
def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    """Poll until *pid* is gone (signal_process_tree does not wait itself)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.02)
    return not _alive(pid)


def test_signal_process_tree_terminates_subprocess():
    """signal_process_tree() must stop a real subprocess (SIGTERM path)."""
    import subprocess as sp

    proc = sp.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    pid = proc.pid
    try:
        sysmem.signal_process_tree(pid)
        assert _wait_dead(pid), f"process {pid} still alive after signal_process_tree"
    finally:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def test_signal_process_tree_kill_true_force_kills():
    """kill=True must SIGKILL even a process that ignores SIGTERM."""
    import subprocess as sp

    # Ignore SIGTERM so only SIGKILL (kill=True) can stop it.
    script = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(60)\n"
    )
    proc = sp.Popen([sys.executable, "-c", script], start_new_session=True)
    pid = proc.pid
    # Let the interpreter start and install the SIG_IGN handler before signaling,
    # otherwise SIGTERM can land during startup (default action) and kill it.
    time.sleep(0.7)
    try:
        sysmem.signal_process_tree(pid, kill=False)  # SIGTERM -> ignored
        time.sleep(0.3)
        assert _alive(pid), "process exited on SIGTERM but the script ignores it"
        sysmem.signal_process_tree(pid, kill=True)   # SIGKILL -> must die
        assert _wait_dead(pid), f"process {pid} survived kill=True"
    finally:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def test_signal_process_tree_reaps_child_process():
    """signal_process_tree must signal the parent AND its forked child."""
    import subprocess as sp

    script = (
        "import sys, time, subprocess\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(60)\n"
    )
    proc = sp.Popen([sys.executable, "-c", script], start_new_session=True)
    parent_pid = proc.pid

    # Poll until the child is actually forked, so the child assertion below can
    # never be vacuous (an empty snapshot would otherwise "pass" trivially).
    deadline = time.monotonic() + 5.0
    child_pids: list = []
    while time.monotonic() < deadline:
        child_pids = sysmem._children_of(parent_pid)
        if child_pids:
            break
        time.sleep(0.05)
    assert child_pids, "child process was never observed; test would be vacuous"

    try:
        sysmem.signal_process_tree(parent_pid, kill=True)
        assert _wait_dead(parent_pid), "parent still alive"
        for cp in child_pids:
            assert _wait_dead(cp.pid), f"child {cp.pid} still alive"
    finally:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def test_signal_process_tree_nonexistent_pid():
    """signal_process_tree on a nonexistent pid must not raise."""
    sysmem.signal_process_tree(999999)
    sysmem.signal_process_tree(999999, kill=True)


def test_engine_windows_branch_uses_signal_process_tree(monkeypatch):
    """On a platform without os.killpg AND without signal.SIGKILL (Windows),
    GenericProcessEngine._signal_pids routes teardown through
    sysmem.signal_process_tree without raising, and returns immediately (no
    blocking wait on the event loop)."""
    from router.config import GenericProcessConfig
    from router.engines import GenericProcessEngine
    import router.engines as engines_mod

    eng = GenericProcessEngine(
        GenericProcessConfig(base_url="http://127.0.0.1:9", start_cmd=["x"]),
        key="winproc",
    )
    # Simulate Windows: no process groups and no SIGKILL symbol. If the code
    # ever references signal.SIGKILL directly on this path it will AttributeError.
    monkeypatch.delattr(os, "killpg", raising=False)
    monkeypatch.delattr(signal, "SIGKILL", raising=False)
    calls: list = []
    monkeypatch.setattr(
        engines_mod.sysmem,
        "signal_process_tree",
        lambda pid, kill=False: calls.append((pid, kill)),
    )
    # Normal SIGTERM stop and a force-kill escalation (via the module's portable
    # _SIGKILL fallback) must both route to signal_process_tree without raising.
    eng._signal_pids([4321], signal.SIGTERM)
    eng._signal_pids([4321], engines_mod._SIGKILL)
    assert len(calls) == 2 and all(pid == 4321 for pid, _ in calls)


# ===========================================================================
# engines.py integration: _read_mem_available_kb delegates to sysmem
# ===========================================================================
def test_engines_read_mem_uses_sysmem_hook(monkeypatch):
    """EngineManager._read_mem_available_kb must read from sysmem, not /proc."""
    saved = sysmem._mem_reader
    sysmem._mem_reader = lambda: 100 * 1024 * 1024 * 1024  # 100 GiB in bytes
    try:
        # _read_mem_available_kb returns kB.
        result = EngineManager._read_mem_available_kb()
        assert result == 100 * 1024 * 1024  # 100 GiB in kB
    finally:
        sysmem._mem_reader = saved


def test_engines_read_mem_returns_none_on_oserror(monkeypatch):
    """_read_mem_available_kb returns None when sysmem.available_bytes raises."""
    saved = sysmem._mem_reader
    sysmem._mem_reader = lambda: (_ for _ in ()).throw(OSError("unavailable"))
    try:
        result = EngineManager._read_mem_available_kb()
        assert result is None
    finally:
        sysmem._mem_reader = saved


# ===========================================================================
# Helpers
# ===========================================================================
def _alive(pid: int) -> bool:
    """Return True if pid is still a running (non-zombie) process."""
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
