"""Engines and the swap state machine.

Only one *heavy* engine can hold the GB10's unified memory at a time, so the
router enforces strict mutual exclusion between engines: to bring one up it
first drains and frees whatever else currently holds the GPU.

Engine kinds
------------
* ``Engine``                base class; controls one backend's lifecycle.
* ``GenericProcessEngine``  launches + supervises a local server process
                            (llama.cpp/llama-server, llamafile, vLLM, SGLang,
                            Aphrodite). SIGTERM -> SIGKILL with port-close
                            verification (carried over from Ds4Engine because
                            llama.cpp has a confirmed SIGTERM-freeze bug).
* ``APISwapEngine``         an engine whose models load/unload over HTTP; the
                            router owns no process. ``free_vram`` calls a
                            configurable unload endpoint (covers TabbyAPI).
* ``OllamaEngine``          a thin preset of ``APISwapEngine`` keeping every
                            Ollama specific (/api/ps, /api/tags TTL cache,
                            keep_alive:0 + 'ollama stop' CLI fallback).
* ``Ds4Engine``             the bespoke escape hatch (systemctl --user / odd
                            lifecycle); kept exactly as before.

EngineManager.acquire(model_id) is the single entry point used by the HTTP
layer. It guarantees that, by the time it returns, the engine that owns
``model_id`` is the active one and has been counted as having one more in-flight
request. The caller MUST pair every successful acquire() with a release().

Concurrency model
-----------------
* ``_swap_lock`` (asyncio.Lock) serializes swap decisions and execution.
* ``_inflight`` counts active proxied requests per engine; mutated only under
  ``_inflight_cond`` (asyncio.Condition).
* Lock ordering is always _swap_lock -> _inflight_cond (acquire path and the
  drain inside a swap). release() takes only _inflight_cond. No cycle, so no
  deadlock. Draining waits on the condition without blocking the event loop, so
  the in-flight requests it's waiting on are free to finish and release.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import signal
import subprocess
import time
from typing import Any

import httpx
import psutil

from . import metrics
from . import sysmem
from .config import RouterConfig, build_model_index

log = logging.getLogger("router.engines")


class EngineError(RuntimeError):
    """Raised when an engine cannot be made ready (start/swap failure)."""


def _resolve_signal(name: str | int) -> int:
    """Resolve a signal name ("SIGTERM") or number to its int value."""
    if isinstance(name, int):
        return name
    try:
        return int(name)
    except (TypeError, ValueError):
        pass
    sig = getattr(signal, str(name).upper(), None)
    if sig is None:
        log.warning("unknown stop_signal %r; falling back to SIGTERM", name)
        return signal.SIGTERM
    return int(sig)


def _iter_objects(data: Any):
    """Yield every dict found at the top level of *data* (object, or list)."""
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def _collect_string_values(data: Any) -> set[str]:
    """Collect candidate model-id strings from a readiness/list response.

    Handles the common shapes: a bare list of strings, a list of objects with
    an id/model/name field, or an object with a ``data``/``models`` list of
    either. Best-effort — used only to assert a model id is present."""
    found: set[str] = set()

    def _from_entry(entry: Any) -> None:
        if isinstance(entry, str):
            found.add(entry)
        elif isinstance(entry, dict):
            for k in ("id", "model", "name"):
                v = entry.get(k)
                if isinstance(v, str):
                    found.add(v)

    if isinstance(data, list):
        for entry in data:
            _from_entry(entry)
    elif isinstance(data, dict):
        _from_entry(data)
        for k in ("data", "models"):
            seq = data.get(k)
            if isinstance(seq, list):
                for entry in seq:
                    _from_entry(entry)
    return found


def _ready_check_passes(ready_check: str, response) -> bool:
    """Evaluate an optional richer readiness assertion against *response*.

    Two forms (see config.GenericProcessConfig.ready_check):
      * ``"model:<id>"`` — <id> must appear in the response's model list.
      * ``"key==value"`` — some object in the JSON body must have key == value.
    Empty/None ``ready_check`` always passes (HTTP 200 already verified)."""
    spec = (ready_check or "").strip()
    if not spec:
        return True
    try:
        data = response.json()
    except (ValueError, TypeError):
        # Body isn't JSON but a body assertion was requested -> not ready.
        return False

    if spec.startswith("model:"):
        wanted = spec[len("model:"):].strip()
        return bool(wanted) and wanted in _collect_string_values(data)

    if "==" in spec:
        key, _, value = spec.partition("==")
        key = key.strip()
        value = value.strip()
        for obj in _iter_objects(data):
            if key in obj and str(obj[key]) == value:
                return True
        return False

    log.warning("unrecognised ready_check %r; treating server 200 as ready", spec)
    return True


def _render_body(tmpl: dict[str, Any] | None, name: str | None) -> dict[str, Any]:
    """Substitute ``{model}`` in every string value of *tmpl* with *name*.

    Shared by the load and unload request builders. With no *name* the template
    is returned unchanged (a copy)."""
    tmpl = tmpl or {}
    if not name:
        return dict(tmpl)
    out: dict[str, Any] = {}
    for k, v in tmpl.items():
        if isinstance(v, str):
            out[k] = v.replace("{model}", name)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Engine implementations
# --------------------------------------------------------------------------- #
class Engine:
    """Base class. Subclasses control one backend's lifecycle + readiness."""

    key: str
    base_url: str

    def __init__(self, control_headers: dict[str, str] | None = None) -> None:
        # Short-timeout client for control/health calls (never user traffic).
        # control_headers (optional) are sent on every such call by default —
        # e.g. an x-admin-key for a secured TabbyAPI or an Authorization bearer
        # for LM Studio/LocalAI. They are NOT added to the user-traffic proxy
        # client (see proxy.make_client); only this control client carries them.
        headers = {str(k): str(v) for k, v in (control_headers or {}).items()}
        self._ctl = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers=headers or None,
        )

    async def aclose(self) -> None:
        await self._ctl.aclose()

    async def is_ready(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    async def ensure_started(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def free_vram(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def wait_ready(self, timeout_s: float, interval_s: float = 1.5) -> bool:
        """Poll is_ready() until it returns True or *timeout_s* elapses."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if await self.is_ready():
                return True
            await asyncio.sleep(interval_s)
        return await self.is_ready()


class Ds4Engine(Engine):
    """The bespoke ds4-server.

    ds4 is normally managed by a `systemctl --user` unit (ds4.service) with
    Restart=always, so a plain SIGTERM is immediately respawned by systemd. The
    router therefore controls it via `systemctl --user start/stop <unit>` by
    default (control="systemd-user"): `stop` frees the ~81 GB of unified memory
    AND keeps ds4 down (an explicit stop does not trigger Restart=). A
    process-control fallback (control="process": launch serve.sh + SIGTERM) is
    kept for setups where ds4 is not a service.
    """

    key = "ds4"

    def __init__(self, cfg, *, key: str = "ds4") -> None:
        super().__init__(getattr(cfg, "control_headers", None))
        self.key = key
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self._proc: subprocess.Popen | None = None

    @property
    def _use_systemd(self) -> bool:
        return self.cfg.control == "systemd-user"

    # -- readiness ------------------------------------------------------- #
    async def is_ready(self) -> bool:
        try:
            r = await self._ctl.get(self.base_url + self.cfg.health_path)
            return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    # -- systemctl --user helpers --------------------------------------- #
    def _systemctl_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # The user bus lives here; ensure it's set even if we were launched
        # without a full login environment.
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        return env

    def _systemctl(self, *args: str, timeout: float = 30.0):
        """Run `systemctl --user <args>`; return CompletedProcess or None."""
        try:
            return subprocess.run(
                ["systemctl", "--user", *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._systemctl_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.error("ds4: 'systemctl --user %s' failed to run: %s", " ".join(args), exc)
            return None

    # -- process discovery ----------------------------------------------- #
    def _pids(self) -> list[int]:
        """All ds4-server pids, via pgrep -f (mode-independent)."""
        try:
            out = subprocess.run(
                ["pgrep", "-f", self.cfg.process_pattern],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        pids: list[int] = []
        for line in out.stdout.split():
            try:
                pid = int(line)
            except ValueError:
                continue
            if pid != os.getpid():  # never match our own process
                pids.append(pid)
        return pids

    def is_running(self) -> bool:
        if self._use_systemd:
            r = self._systemctl("is-active", self.cfg.systemd_user_unit, timeout=5)
            if r is not None:
                return r.stdout.strip() in ("active", "activating", "reloading")
        return bool(self._pids())

    # -- start ----------------------------------------------------------- #
    async def ensure_started(self) -> None:
        if await self.is_ready():
            return
        if self._use_systemd:
            unit = self.cfg.systemd_user_unit
            # Clear any leftover 'failed' state (e.g. from a prior forced kill)
            # so the start is clean and not refused by a start limit.
            self._systemctl("reset-failed", unit, timeout=5)
            log.info("ds4: starting user unit %s", unit)
            r = self._systemctl("start", unit, timeout=30.0)
            if r is None:
                raise EngineError(f"could not invoke systemctl to start {unit}")
            if r.returncode != 0:
                log.warning("ds4: systemctl start %s -> rc=%d: %s",
                            unit, r.returncode, r.stderr.strip())
        elif self.is_running():
            log.info("ds4: process already running, waiting for readiness")
        else:
            self._launch()
        ok = await self.wait_ready(self.cfg.start_timeout_s)
        if not ok:
            raise EngineError(
                f"ds4 did not become ready within {self.cfg.start_timeout_s}s"
            )

    def _launch(self) -> None:
        """Process-control fallback: launch serve.sh directly."""
        script = self.cfg.serve_script
        if not os.path.exists(script):
            raise EngineError(f"ds4 serve script not found: {script}")
        log.info("ds4: launching %s", script)
        try:
            os.makedirs(os.path.dirname(self.cfg.log_file), exist_ok=True)
            logfh = open(self.cfg.log_file, "ab", buffering=0)
        except OSError:
            logfh = subprocess.DEVNULL  # type: ignore[assignment]
        self._proc = subprocess.Popen(
            [script],
            stdout=logfh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    # -- stop / free VRAM ------------------------------------------------ #
    async def free_vram(self) -> None:
        if self._use_systemd:
            unit = self.cfg.systemd_user_unit
            log.info("ds4: stopping user unit %s", unit)
            # --no-block: don't wait on systemd's graceful stop timeout (a
            # lingering child can stretch a blocking `stop` to ~45s). systemd
            # sends the unit's SIGTERM right away; we let IT own that signal so
            # ds4-server exits cleanly (status 0 -> 'inactive', not 'failed'),
            # and only force-kill stragglers ourselves. An intentional stop
            # disables Restart=always, so ds4 won't respawn.
            self._systemctl("stop", unit, "--no-block", timeout=10)
            if not await self._wait_stopped(6.0):
                await self._sigkill_leftover()
        else:
            await self._terminate_pids(grace_s=6.0)

        self._proc = None
        if self._pids() or await self._port_open():
            raise EngineError("ds4 would not stop (still holding the GPU)")
        log.info("ds4: stopped, VRAM released")

    async def _sigkill_leftover(self) -> None:
        """Force-kill any ds4-server pids still present after a stop."""
        leftover = self._pids()
        if not leftover:
            return
        log.warning("ds4: %s still alive after stop; SIGKILL", leftover)
        for pid in leftover:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await self._wait_stopped(10.0)

    async def _terminate_pids(self, grace_s: float) -> None:
        """Process-control fallback (control=process): SIGTERM the ds4-server
        pids, then SIGKILL any that linger past grace_s."""
        pids = self._pids()
        if not pids:
            log.info("ds4: no ds4-server process to terminate")
            return
        log.info("ds4: SIGTERM %s", pids)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                log.error("ds4: not permitted to signal pid %s", pid)
        if not await self._wait_stopped(grace_s):
            await self._sigkill_leftover()

    async def _wait_stopped(self, timeout_s: float) -> bool:
        """Wait until no ds4-server pids remain and the port is closed."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if not self._pids() and not await self._port_open():
                return True
            await asyncio.sleep(0.5)
        return not self._pids()

    async def _port_open(self) -> bool:
        try:
            r = await self._ctl.get(
                self.base_url + self.cfg.health_path,
                timeout=httpx.Timeout(2.0, connect=1.0),
            )
            return r.status_code < 600  # any HTTP answer => port still open
        except (httpx.HTTPError, OSError):
            return False


class GenericProcessEngine(Engine):
    """A local server process the router launches and supervises.

    Configured entirely from YAML (see config.GenericProcessConfig): a
    ``start_cmd`` launched under its own session, polled at ``ready_path`` until
    HTTP 200 or ``start_timeout_s``. ``free_vram`` signals the process group
    with ``stop_signal``, escalates to SIGKILL after ``stop_timeout_s``, and
    VERIFIES the listening port is actually closed before returning — llama.cpp
    has a confirmed SIGTERM-freeze bug, so a plain SIGTERM is not trusted.

    Covers llama.cpp/llama-server, llamafile, vLLM, SGLang, Aphrodite.
    """

    def __init__(self, cfg, *, key: str) -> None:
        super().__init__(getattr(cfg, "control_headers", None))
        self.key = key
        self.cfg = cfg
        self.base_url = (cfg.base_url or "").rstrip("/")
        self._proc: subprocess.Popen | None = None

    # -- readiness ------------------------------------------------------- #
    async def is_ready(self) -> bool:
        try:
            r = await self._ctl.get(self.base_url + self.cfg.ready_path)
            if r.status_code != 200:
                return False
            return _ready_check_passes(getattr(self.cfg, "ready_check", ""), r)
        except (httpx.HTTPError, OSError):
            return False

    # -- process discovery ----------------------------------------------- #
    def _argv(self) -> list[str] | str:
        cmd = self.cfg.start_cmd
        return cmd

    def _pids(self) -> list[int]:
        """Find this engine's pids via the tracked Popen and/or process_pattern.

        On POSIX, ``pgrep -f`` is the fast path. On platforms without pgrep
        (macOS without Homebrew, Windows), psutil is the portable fallback."""
        pids: list[int] = []
        if self._proc is not None and self._proc.poll() is None:
            pids.append(self._proc.pid)
        pattern = self.cfg.process_pattern
        if pattern:
            found = self._pgrep_pids(pattern)
            for pid in found:
                if pid != os.getpid() and pid not in pids:
                    pids.append(pid)
        return pids

    @staticmethod
    def _pgrep_pids(pattern: str) -> list[int]:
        """Return pids whose command line matches *pattern*.

        Tries ``pgrep -f`` (POSIX fast path) then falls back to iterating
        psutil processes (portable: macOS, Windows, or when pgrep is absent)."""
        # POSIX fast path.
        try:
            out = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids: list[int] = []
            for line in out.stdout.split():
                try:
                    pids.append(int(line))
                except ValueError:
                    continue
            return pids
        except (OSError, subprocess.SubprocessError):
            pass  # pgrep not available — fall through to psutil

        # psutil fallback (macOS, Windows, or constrained environments).
        pids = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                if pattern in cmd_str:
                    pids.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    def is_running(self) -> bool:
        return bool(self._pids())

    # -- start ----------------------------------------------------------- #
    async def ensure_started(self) -> None:
        if await self.is_ready():
            return
        if self.is_running():
            log.info("%s: process already running, waiting for readiness", self.key)
        else:
            self._launch()
        ok = await self.wait_ready(self.cfg.start_timeout_s)
        if not ok:
            raise EngineError(
                f"{self.key} did not become ready within {self.cfg.start_timeout_s}s"
            )

    def _launch(self) -> None:
        cmd = self._argv()
        if not cmd:
            raise EngineError(f"{self.key}: no start_cmd configured")
        # Accept a shell string or an argv list. A string is split with shlex
        # (POSIX) so we still run without a shell and keep start_new_session
        # semantics (so free_vram can signal the whole process group).
        if isinstance(cmd, str):
            argv = shlex.split(cmd)
        else:
            argv = list(cmd)
        if not argv:
            raise EngineError(f"{self.key}: start_cmd is empty")

        env = None
        if self.cfg.env:
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in self.cfg.env.items()})

        logfh: Any = subprocess.DEVNULL
        if self.cfg.log_file:
            try:
                os.makedirs(os.path.dirname(self.cfg.log_file), exist_ok=True)
                logfh = open(self.cfg.log_file, "ab", buffering=0)
            except OSError as exc:
                log.warning("%s: could not open log_file %s: %s",
                            self.key, self.cfg.log_file, exc)
                logfh = subprocess.DEVNULL

        log.info("%s: launching %s", self.key, argv)
        self._proc = subprocess.Popen(
            argv,
            stdout=logfh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=env,
            cwd=self.cfg.cwd or None,
        )

    # -- stop / free VRAM ------------------------------------------------ #
    async def free_vram(self) -> None:
        sig = _resolve_signal(self.cfg.stop_signal)
        pids = self._pids()
        if not pids and not await self._port_open():
            log.info("%s: no process to stop", self.key)
            self._proc = None
            return

        try:
            sig_name = signal.Signals(sig).name
        except ValueError:
            sig_name = str(sig)
        log.info("%s: sending %s to process group(s) of %s", self.key, sig_name, pids)
        self._signal_pids(pids, sig)

        # SIGTERM->SIGKILL escalation + port-close verification (llama.cpp can
        # freeze on SIGTERM, so we never trust the signal alone).
        if not await self._wait_stopped(self.cfg.stop_timeout_s):
            await self._sigkill_leftover()

        self._proc = None
        if self._pids() or await self._port_open():
            raise EngineError(
                f"{self.key} would not stop (port still open / process alive)"
            )
        log.info("%s: stopped, VRAM released", self.key)

    def _signal_pids(self, pids: list[int], sig: int) -> None:
        """Signal every PID's whole PROCESS GROUP, then any PID directly.

        We launch with start_new_session=True so the launcher leads its own
        session/group; signalling the GROUP (os.killpg) reaps forked workers
        (vLLM/SGLang/Aphrodite/MAX spawn child processes) — not just the
        launcher. Process groups are de-duplicated so a shared group is signalled
        once. PIDs whose group can't be resolved (e.g. strays found via
        process_pattern that aren't in our session) are signalled directly as a
        fallback so the group signal never silently misses them.

        On platforms without process groups (Windows), psutil reaps the whole
        tree instead, so subprocess-managed engines (Ollama, LM Studio,
        KoboldCpp, llama.cpp) stop cleanly there too."""
        if not hasattr(os, "killpg"):
            for pid in pids:
                sysmem.terminate_process_tree(
                    pid,
                    term_timeout=self.cfg.stop_timeout_s,
                    kill_timeout=10.0,
                )
            return
        signalled_pgids: set[int] = set()
        for pid in pids:
            try:
                pgid = os.getpgid(pid)
            except (ProcessLookupError, PermissionError, OSError):
                pgid = None
            if pgid is not None:
                if pgid in signalled_pgids:
                    continue
                try:
                    os.killpg(pgid, sig)
                    signalled_pgids.add(pgid)
                    continue
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            # Fallback: signal the PID directly (reaps strays the group missed).
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                log.error("%s: not permitted to signal pid %s", self.key, pid)

    async def _sigkill_leftover(self) -> None:
        leftover = self._pids()
        if not leftover:
            return
        log.warning("%s: %s still alive after %s; SIGKILL",
                    self.key, leftover, self.cfg.stop_signal)
        self._signal_pids(leftover, signal.SIGKILL)
        await self._wait_stopped(10.0)

    async def _wait_stopped(self, timeout_s: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if not self._pids() and not await self._port_open():
                return True
            await asyncio.sleep(0.5)
        return not self._pids() and not await self._port_open()

    async def _port_open(self) -> bool:
        try:
            r = await self._ctl.get(
                self.base_url + self.cfg.ready_path,
                timeout=httpx.Timeout(2.0, connect=1.0),
            )
            return r.status_code < 600  # any HTTP answer => port still open
        except (httpx.HTTPError, OSError):
            return False


class APISwapEngine(Engine):
    """An engine whose models are loaded/unloaded over HTTP (no owned process).

    Generic, config-driven (see config.ApiSwapConfig): readiness is a GET on
    ``health_path``; ``free_vram`` issues the configured unload request and (if
    a ``loaded_path`` probe is configured) waits until no models remain. Covers
    TabbyAPI-style load/unload. ``OllamaEngine`` subclasses this to keep its
    Ollama-specific behaviour.
    """

    def __init__(self, cfg, *, key: str) -> None:
        super().__init__(getattr(cfg, "control_headers", None))
        self.key = key
        self.cfg = cfg
        self.base_url = (cfg.base_url or "").rstrip("/")
        self._tags_cache: tuple[float, set[str]] | None = None

    # -- readiness ------------------------------------------------------- #
    async def is_ready(self) -> bool:
        try:
            r = await self._ctl.get(self.base_url + self.cfg.health_path)
            if r.status_code != 200:
                return False
            return _ready_check_passes(getattr(self.cfg, "ready_check", ""), r)
        except (httpx.HTTPError, OSError):
            return False

    async def ensure_started(self) -> None:
        if await self.is_ready():
            return
        unit = getattr(self.cfg, "systemd_unit", None)
        if unit:
            log.info("%s: not answering, attempting to start %s", self.key, unit)
            for cmd in (["systemctl", "start", unit],
                        ["sudo", "-n", "systemctl", "start", unit]):
                try:
                    subprocess.run(cmd, capture_output=True, timeout=10)
                except (OSError, subprocess.SubprocessError):
                    continue
        if not await self.wait_ready(20.0):
            raise EngineError(f"{self.key} service is not reachable")

    # -- loaded models / unload ----------------------------------------- #
    async def loaded_models(self) -> list[str]:
        """List currently-loaded models via the configured loaded_path probe.

        Returns the UNLOAD identifiers: keyed by ``loaded_id_key`` when set
        (e.g. LM Studio's per-instance id), else by ``loaded_name_key`` (the
        display name). Entries are filtered by ``loaded_filter`` so only models
        that are ACTUALLY loaded are returned. These ids feed free_vram()'s
        {model} unload substitution."""
        data = await self._fetch_loaded()
        if data is None:
            return []
        return self._extract_loaded_names(data, by_id=True)

    async def loaded_model_names(self) -> list[str]:
        """Like loaded_models() but always keyed by the display name.

        Used to decide whether a requested model id is already loaded (the
        load_path skip check), independent of any distinct unload id."""
        data = await self._fetch_loaded()
        if data is None:
            return []
        return self._extract_loaded_names(data, by_id=False)

    async def _fetch_loaded(self) -> Any | None:
        """GET loaded_path and return parsed JSON, or None if unconfigured/down."""
        path = getattr(self.cfg, "loaded_path", None)
        if not path:
            return None
        try:
            r = await self._ctl.get(self.base_url + path)
            return r.json()
        except (httpx.HTTPError, OSError, ValueError):
            return None

    def _extract_loaded_names(self, data: Any, *, by_id: bool = True) -> list[str]:
        key = getattr(self.cfg, "loaded_models_key", "models")
        name_key = getattr(self.cfg, "loaded_name_key", "name")
        id_key = (getattr(self.cfg, "loaded_id_key", "") or "") if by_id else ""
        filter_spec = getattr(self.cfg, "loaded_filter", "") or ""
        fkey, fval = "", ""
        if "==" in filter_spec:
            fkey, _, fval = filter_spec.partition("==")
            fkey, fval = fkey.strip(), fval.strip()

        # Resolve the list of entries. A loaded_path may return a list, a
        # wrapper object with the list under loaded_models_key (Ollama /api/ps),
        # or a SINGLE loaded-model object (TabbyAPI /v1/model returns one object,
        # not a list) — treat that lone object as a one-element list.
        if isinstance(data, dict):
            if key in data and isinstance(data[key], list):
                entries: Any = data[key]
            else:
                entries = [data]
        else:
            entries = data

        names: list[str] = []
        for m in entries or []:
            if isinstance(m, str):
                # A bare string entry can't be filtered or id-keyed.
                names.append(m)
            elif isinstance(m, dict):
                if fkey and str(m.get(fkey)) != fval:
                    continue  # not actually loaded per loaded_filter
                if id_key:
                    val = m.get(id_key)
                else:
                    val = m.get(name_key) or m.get("model") or m.get("id")
                if val:
                    names.append(val)
        return names

    async def free_vram(self) -> None:
        loaded = await self.loaded_models()
        if getattr(self.cfg, "loaded_path", None) and not loaded:
            log.info("%s: no models loaded", self.key)
            return

        if loaded:
            log.info("%s: unloading models %s", self.key, loaded)
            for name in loaded:
                await self._unload(name)
        else:
            # No list probe: issue the unload request once (best effort).
            await self._unload(None)

        # If we can list loaded models, wait until empty (VRAM released).
        if getattr(self.cfg, "loaded_path", None):
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.cfg.unload_timeout_s
            while loop.time() < deadline:
                if not await self.loaded_models():
                    log.info("%s: all models unloaded, VRAM released", self.key)
                    return
                await asyncio.sleep(0.5)
            still = await self.loaded_models()
            if still:
                log.warning("%s: models still loaded after timeout: %s", self.key, still)

    async def _unload(self, name: str | None) -> None:
        """Issue the configured unload request (optionally for one model)."""
        url_path = getattr(self.cfg, "unload_path", "")
        if not url_path:
            log.debug("%s: no unload_path configured; free_vram is a no-op", self.key)
            return
        method = (getattr(self.cfg, "unload_method", "POST") or "POST").upper()
        body = self._render_unload_body(name)
        try:
            await self._ctl.request(
                method,
                self.base_url + url_path,
                json=body if body else None,
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
        except (httpx.HTTPError, OSError) as exc:
            log.warning("%s: unload request for %s failed: %s", self.key, name, exc)

    def _render_unload_body(self, name: str | None) -> dict[str, Any]:
        """Substitute {model} in the configured unload_body with *name*."""
        return _render_body(getattr(self.cfg, "unload_body", None), name)

    # -- explicit per-model load (load_path engines: TabbyAPI, TGW) ------- #
    async def load_model(self, model_id: str) -> None:
        """Load *model_id* via the configured load_path (no-op if unset).

        For engines that require an explicit load before they can serve a model
        (TabbyAPI, text-generation-webui). JIT engines (Ollama) leave load_path
        empty and load on first request, so this is a no-op for them."""
        url_path = getattr(self.cfg, "load_path", "") or ""
        if not url_path:
            return
        method = (getattr(self.cfg, "load_method", "POST") or "POST").upper()
        body = _render_body(getattr(self.cfg, "load_body", None), model_id)
        timeout_s = float(getattr(self.cfg, "load_timeout_s", 120.0) or 120.0)
        log.info("%s: loading model %s via %s", self.key, model_id, url_path)
        try:
            r = await self._ctl.request(
                method,
                self.base_url + url_path,
                json=body if body else None,
                timeout=httpx.Timeout(timeout_s, connect=5.0),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise EngineError(
                f"{self.key}: failed to load model {model_id!r}: {exc}"
            ) from exc
        if r.status_code >= 400:
            raise EngineError(
                f"{self.key}: load of model {model_id!r} returned "
                f"HTTP {r.status_code}"
            )
        # Loading another model frees VRAM/changes loaded set; drop tag cache.
        self._tags_cache = None

    # -- available tags (for routing fallback + /v1/models) -------------- #
    async def available_tags(self) -> set[str]:
        """Cached list of model ids this engine can serve (for routing).

        Generic base uses loaded_path; OllamaEngine overrides with /api/tags.
        """
        loop = asyncio.get_running_loop()
        now = loop.time()
        ttl = getattr(self.cfg, "tags_cache_ttl_s", 30.0)
        if self._tags_cache and now - self._tags_cache[0] < ttl:
            return self._tags_cache[1]
        # Routing matches on the model id clients send, so tags must be the
        # display names, NOT the unload identifiers (loaded_id_key, e.g. an LM
        # Studio instance_id) that loaded_models() returns.
        tags = set(await self.loaded_model_names())
        self._tags_cache = (now, tags)
        return tags


class OllamaEngine(APISwapEngine):
    """Ollama runs as a persistent systemd service. "Starting" it means making
    sure the service answers; freeing VRAM means unloading every loaded model
    (Ollama keeps them resident because OLLAMA_KEEP_ALIVE=-1).

    A thin preset of APISwapEngine that keeps every Ollama specific:
    loaded_models() via /api/ps, available_tags() via /api/tags with a TTL
    cache, and _unload() via keep_alive:0 with an 'ollama stop' CLI fallback.
    """

    key = "ollama"

    def __init__(self, cfg, *, key: str = "ollama") -> None:
        super().__init__(cfg, key=key)

    # -- loaded models / unload ----------------------------------------- #
    async def loaded_models(self) -> list[str]:
        try:
            r = await self._ctl.get(self.base_url + "/api/ps")
            data = r.json()
        except (httpx.HTTPError, OSError, ValueError):
            return []
        names = []
        for m in data.get("models", []) or []:
            name = m.get("name") or m.get("model")
            if name:
                names.append(name)
        return names

    async def free_vram(self) -> None:
        loaded = await self.loaded_models()
        if not loaded:
            log.info("ollama: no models loaded")
            return
        log.info("ollama: unloading models %s", loaded)
        for name in loaded:
            await self._unload(name)

        # Wait until /api/ps reports empty (VRAM actually released).
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.cfg.unload_timeout_s
        while loop.time() < deadline:
            if not await self.loaded_models():
                log.info("ollama: all models unloaded, VRAM released")
                return
            await asyncio.sleep(0.5)
        still = await self.loaded_models()
        if still:
            log.warning("ollama: models still loaded after timeout: %s", still)

    async def _unload(self, name: str | None) -> None:
        # keep_alive:0 with no prompt unloads immediately without generating.
        if not name:
            return
        try:
            await self._ctl.post(
                self.base_url + "/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
            return
        except (httpx.HTTPError, OSError) as exc:
            log.warning("ollama: API unload of %s failed (%s); trying CLI", name, exc)
        # Fallback to the CLI.
        try:
            subprocess.run(["ollama", "stop", name], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            log.error("ollama: CLI stop of %s failed: %s", name, exc)

    # -- available tags (for routing fallback + /v1/models) -------------- #
    async def available_tags(self) -> set[str]:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._tags_cache and now - self._tags_cache[0] < self.cfg.tags_cache_ttl_s:
            return self._tags_cache[1]
        tags: set[str] = set()
        try:
            r = await self._ctl.get(self.base_url + "/api/tags")
            for m in r.json().get("models", []) or []:
                name = m.get("name") or m.get("model")
                if name:
                    tags.add(name)
        except (httpx.HTTPError, OSError, ValueError):
            # Keep stale cache if we have one; otherwise empty.
            return self._tags_cache[1] if self._tags_cache else set()
        self._tags_cache = (now, tags)
        return tags

    async def free_vram_for_ds4(self) -> None:
        await self.free_vram()


# --------------------------------------------------------------------------- #
# Engine construction (generic, config-driven)
# --------------------------------------------------------------------------- #
def _build_engine(key: str, etype: str, params) -> Engine:
    """Instantiate one engine from its type + params dataclass."""
    if etype == "ds4":
        return Ds4Engine(params, key=key)
    if etype == "ollama":
        return OllamaEngine(params, key=key)
    if etype == "generic_process":
        return GenericProcessEngine(params, key=key)
    if etype == "api_swap":
        return APISwapEngine(params, key=key)
    raise EngineError(f"unknown engine type {etype!r} for engine {key!r}")


def build_engines(cfg: RouterConfig) -> dict[str, Engine]:
    """Build the engine table from config.

    If cfg.engines (the generic table) is present, build from it by type.
    Otherwise fall back to ds4 (from cfg.ds4) + ollama (from cfg.ollama),
    exactly as the router did before the generic table existed.
    """
    engines: dict[str, Engine] = {}
    if cfg.engines:
        for spec in cfg.engines:
            if not spec.enabled:
                continue
            engines[spec.key] = _build_engine(spec.key, spec.type, spec.params)
        return engines

    # Legacy path: identical behaviour to the original hardcoded construction.
    if cfg.ds4.enabled:
        engines["ds4"] = Ds4Engine(cfg.ds4)
    if cfg.ollama.enabled:
        engines["ollama"] = OllamaEngine(cfg.ollama)
    return engines


# --------------------------------------------------------------------------- #
# Engine manager: the swap state machine
# --------------------------------------------------------------------------- #
class EngineManager:
    def __init__(self, cfg: RouterConfig) -> None:
        self.cfg = cfg
        self.index = build_model_index(cfg)
        self.engines: dict[str, Engine] = build_engines(cfg)

        self.active_engine: str | None = None
        self._swap_lock = asyncio.Lock()
        self._inflight_cond = asyncio.Condition()
        self._inflight: dict[str, int] = {k: 0 for k in self.engines}
        self._last_swap: dict[str, Any] = {}

    # -- lifecycle ------------------------------------------------------- #
    async def startup(self) -> None:
        """Detect which engine currently holds the GPU by probing reality."""
        active: str | None = None
        # Prefer a process-style engine that is already serving.
        for key, engine in self.engines.items():
            if isinstance(engine, (Ds4Engine, GenericProcessEngine)):
                if await engine.is_ready():
                    active = key
                    break
        # Otherwise an API-swap engine with a model resident.
        if active is None:
            for key, engine in self.engines.items():
                if isinstance(engine, APISwapEngine):
                    if await engine.loaded_models():
                        active = key
                        break
        self.active_engine = active
        metrics.set_active_engine(active)
        log.info("startup: active engine detected as %s", self.active_engine)
        self._persist()

    async def aclose(self) -> None:
        for e in self.engines.values():
            await e.aclose()

    # -- routing --------------------------------------------------------- #
    def resolve_model_id(self, model_id: str | None) -> str | None:
        """Resolve a request alias to its real model id (single hop, no chains).

        Returns *model_id* unchanged if it is not a configured alias. Used by
        engine_for() and by the HTTP layer to rewrite the outgoing body so the
        upstream sees the REAL model id, never the alias."""
        if not model_id:
            return model_id
        return self.cfg.aliases.get(model_id, model_id)

    async def engine_for(self, model_id: str | None) -> Engine:
        """Resolve which engine owns *model_id*.

        Aliases are resolved first, then the static registry, then a live
        API-swap tag lookup (so models pulled after the router started still
        route correctly), then a best-effort guess (a process engine's fixed
        ids; otherwise an API-swap engine that can serve arbitrary tags)."""
        if not model_id:
            raise EngineError("request is missing a 'model' field")

        model_id = self.resolve_model_id(model_id)

        spec = self.index.get(model_id)
        if spec:
            engine = self.engines.get(spec.engine)
            if engine is None:
                raise EngineError(f"engine {spec.engine!r} is disabled")
            return engine

        # Unknown id: consult live tags from any API-swap engine (e.g. Ollama).
        for engine in self.engines.values():
            if isinstance(engine, APISwapEngine):
                tags = await engine.available_tags()
                if model_id in tags:
                    return engine

        # A process engine advertises a small, fixed set; if model_id is one of
        # those, use that engine.
        for key, engine in self.engines.items():
            if isinstance(engine, (Ds4Engine, GenericProcessEngine)) and any(
                s.engine == key and s.id == model_id for s in self.cfg.models
            ):
                return engine

        # Last resort: if only one engine is enabled, use it; else prefer an
        # API-swap engine (it can pull/serve arbitrary tags), otherwise error.
        if len(self.engines) == 1:
            return next(iter(self.engines.values()))
        for engine in self.engines.values():
            if isinstance(engine, APISwapEngine):
                log.warning("unknown model %r; defaulting to %s", model_id, engine.key)
                return engine
        raise EngineError(f"no engine can serve model {model_id!r}")

    # -- acquire / release ---------------------------------------------- #
    async def acquire(self, model_id: str | None) -> Engine:
        """Ensure the engine owning *model_id* is active, count one in-flight
        request against it, and return it. Pair with release()."""
        real_id = self.resolve_model_id(model_id)
        target = await self.engine_for(real_id)
        async with self._swap_lock:
            if self.active_engine != target.key:
                await self._swap_to(target)
            # Explicit-load engines (api_swap with a load_path) need the
            # requested model loaded into the now-active engine before we serve.
            # Done under _swap_lock (so it can't race a swap) and BEFORE the
            # in-flight increment (so a load failure leaks no in-flight count).
            await self._ensure_model_loaded(target, real_id)
            async with self._inflight_cond:
                self._inflight[target.key] += 1
        return target

    async def _ensure_model_loaded(self, target: Engine, model_id: str | None) -> None:
        """For an APISwapEngine with a load_path, load *model_id* unless it is
        already loaded. No-op for every other engine (JIT/process engines).

        Caller MUST hold _swap_lock and *target* must already be active."""
        if not isinstance(target, APISwapEngine):
            return
        if not getattr(target.cfg, "load_path", "") or not model_id:
            return
        try:
            already = await target.loaded_model_names()
        except Exception as exc:  # noqa: BLE001 - probe is best-effort
            log.warning("%s: could not check loaded models: %s", target.key, exc)
            already = []
        if model_id in already:
            log.debug("%s: model %s already loaded; skip load", target.key, model_id)
            return
        await target.load_model(model_id)

    async def release(self, engine_key: str) -> None:
        async with self._inflight_cond:
            if self._inflight.get(engine_key, 0) > 0:
                self._inflight[engine_key] -= 1
            self._inflight_cond.notify_all()

    # -- the swap -------------------------------------------------------- #
    async def _swap_to(self, target: Engine) -> None:
        """Make *target* the active engine. Caller must hold _swap_lock."""
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        prev = self.active_engine
        log.info("SWAP begin: %s -> %s", prev, target.key)

        # Record how many in-flight requests we are about to drain (the cost of
        # the swap to current traffic). Sum across the engines we will stop.
        in_flight = sum(
            self._inflight.get(key, 0) for key in self.engines if key != target.key
        )
        metrics.record_in_flight_at_swap_start(in_flight)

        # 1. Drain + free whatever currently holds the GPU.
        for key, engine in self.engines.items():
            if key == target.key:
                continue
            await self._drain(key)
            try:
                await engine.free_vram()
            except EngineError:
                # Re-raise: if we can't free the GPU we must not start target.
                dt = loop.time() - t0
                self._record_swap(prev, target.key, dt, ok=False)
                metrics.record_swap(prev, target.key, dt, ok=False)
                raise

        # 1b. If we just freed an active engine, wait for the kernel to reclaim
        # its (unified) memory before loading the next model — otherwise the
        # incoming model's pre-flight memory check sees the old model's pages
        # still resident and fails.
        if prev is not None and prev != target.key:
            await self._await_memory_settle(self.cfg.swap_memory_settle_timeout_s)

        # 2. Bring the target up and wait until it answers.
        try:
            await target.ensure_started()
        except EngineError:
            self.active_engine = None
            metrics.set_active_engine(None)
            dt = loop.time() - t0
            self._record_swap(prev, target.key, dt, ok=False)
            metrics.record_swap(prev, target.key, dt, ok=False)
            self._persist()
            raise

        self.active_engine = target.key
        metrics.set_active_engine(target.key)
        dt = loop.time() - t0
        self._record_swap(prev, target.key, dt, ok=True)
        metrics.record_swap(prev, target.key, dt, ok=True)
        self._persist()
        log.info("SWAP done: %s -> %s in %.1fs", prev, target.key, dt)

    @staticmethod
    def _read_mem_available_kb() -> int | None:
        """Return available memory in kB, or None if unreadable.

        Delegates to ``sysmem.available_bytes()`` so the settle logic works on
        every OS (Linux /proc/meminfo, macOS, Windows) rather than silently
        no-oping on non-Linux hosts. The injectable ``sysmem._mem_reader`` hook
        lets tests simulate any memory curve without touching real files."""
        try:
            return sysmem.available_bytes() // 1024
        except (OSError, ValueError):
            return None

    async def _await_memory_settle(self, timeout_s: float) -> None:
        """Block until system memory has been reclaimed after freeing an engine.

        The freed model's memory is released by the kernel over a couple of
        seconds; we poll available memory and return as soon as it stops rising
        (two consecutive samples within ~1 GiB), or after *timeout_s*."""
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        deadline = t0 + timeout_s
        prev = -1
        stable = 0
        while loop.time() < deadline:
            avail = self._read_mem_available_kb()
            if avail is None:
                metrics.record_memory_settle(loop.time() - t0)
                return  # can't read memory — don't block the swap
            if prev >= 0 and (avail - prev) < 1_000_000:  # rose < ~1 GiB
                stable += 1
                if stable >= 2:
                    log.info("memory settled: %.1f GiB available", avail / 1048576)
                    metrics.record_memory_settle(loop.time() - t0)
                    return
            else:
                stable = 0
            prev = avail
            await asyncio.sleep(0.5)
        metrics.record_memory_settle(loop.time() - t0)
        log.warning(
            "memory-settle wait hit %.0fs timeout (available=%.1f GiB)",
            timeout_s,
            (prev / 1048576) if prev > 0 else -1,
        )

    async def _drain(self, key: str) -> None:
        """Wait for in-flight requests on *key* to finish (bounded)."""
        async with self._inflight_cond:
            if self._inflight.get(key, 0) == 0:
                return
            log.info("draining %d in-flight request(s) on %s", self._inflight[key], key)
            try:
                await asyncio.wait_for(
                    self._inflight_cond.wait_for(lambda: self._inflight.get(key, 0) == 0),
                    timeout=self.cfg.drain_timeout_s,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "drain timeout on %s (%d still in-flight); proceeding to stop it",
                    key,
                    self._inflight.get(key, 0),
                )

    def _record_swap(self, frm, to, dt, ok) -> None:
        self._last_swap = {
            "from": frm,
            "to": to,
            "duration_s": round(dt, 2),
            "ok": ok,
            "at": int(time.time()),
        }

    # -- admin / observability ------------------------------------------ #
    async def force_swap(self, model_id: str | None = None, engine_key: str | None = None) -> Engine:
        """Proactively swap to an engine (by model id or engine key) without
        running a user request. Used by /admin/swap and routerctl."""
        if engine_key:
            target = self.engines.get(engine_key)
            if target is None:
                raise EngineError(f"unknown engine {engine_key!r}")
        else:
            target = await self.engine_for(model_id)
        async with self._swap_lock:
            if self.active_engine != target.key:
                await self._swap_to(target)
        return target

    async def status(self) -> dict[str, Any]:
        engines: dict[str, Any] = {}
        for key, engine in self.engines.items():
            entry: dict[str, Any] = {
                "ready": await engine.is_ready(),
                "in_flight": self._inflight.get(key, 0),
                "base_url": engine.base_url,
            }
            if isinstance(engine, APISwapEngine):
                # Display names (not unload ids) under the human-facing field.
                entry["loaded_models"] = await engine.loaded_model_names()
            if isinstance(engine, (Ds4Engine, GenericProcessEngine)):
                entry["process_running"] = engine.is_running()
            engines[key] = entry
        return {
            "active_engine": self.active_engine,
            "last_swap": self._last_swap or None,
            "engines": engines,
            "models": [
                {"id": s.id, "engine": s.engine, "name": s.display_name}
                for s in self.cfg.models
            ],
        }

    def _persist(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cfg.state_file), exist_ok=True)
            with open(self.cfg.state_file, "w") as fh:
                json.dump(
                    {"active_engine": self.active_engine, "last_swap": self._last_swap},
                    fh,
                )
        except OSError as exc:  # pragma: no cover - best effort
            log.debug("could not persist state: %s", exc)
