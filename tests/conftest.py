"""Shared fixtures + fakes for the llm-router test suite.

Everything here is hermetic: no GPU, no systemd, no real engine process, and no
network egress. The two big helpers are:

* ``FakeEngine`` — an in-memory, instant subclass of ``router.engines.Engine``.
  Its lifecycle methods (ensure_started / free_vram / is_ready / loaded_models)
  flip plain booleans/lists and never touch a socket, so the swap state machine
  can be exercised deterministically and fast.

* ``mock_upstream`` — a tiny FastAPI app served by a real ``uvicorn.Server`` on
  an ephemeral 127.0.0.1 port, in a background thread. It answers the handful of
  upstream routes the router proxies to (/v1/models, /v1/chat/completions,
  /api/tags, /api/chat, /api/ps). Used by the app-integration tests so a request
  genuinely flows create_app -> proxy -> a real HTTP upstream.

The process-global ``router.metrics`` registry is reset around every test via an
autouse fixture so swap/uptime assertions don't see leakage from other tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from router import metrics
from router.config import ModelSpec, RouterConfig
from router.engines import APISwapEngine, Engine, EngineManager


# --------------------------------------------------------------------------- #
# metrics isolation
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_metrics():
    """Reset the process-global metrics registry before and after each test."""
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture(autouse=True)
def _instant_memory_settle(monkeypatch):
    """Make the post-free memory-settle wait instant across the suite.

    EngineManager._await_memory_settle returns immediately when it can't read
    /proc/meminfo. Forcing _read_mem_available_kb -> None gives every swap an
    instant, deterministic settle (no real 0.5s polling sleeps), while still
    exercising the settle code path (it records the memory_settle metric). No
    test in this suite asserts on settle *timing*, so this is purely a speedup
    and removes machine-dependent flakiness."""
    monkeypatch.setattr(
        EngineManager, "_read_mem_available_kb", staticmethod(lambda: None)
    )


# --------------------------------------------------------------------------- #
# Fake engine (in-memory, instant, no sockets)
# --------------------------------------------------------------------------- #
class FakeEngine(Engine):
    """An Engine whose lifecycle is pure in-memory state.

    * ``ensure_started`` flips ``_ready`` True (optionally failing on demand and
      counting calls).
    * ``free_vram`` flips ``_ready`` False and clears loaded models.
    * ``is_ready`` returns the boolean.
    * ``loaded_models`` returns the configured list (so EngineManager.status and
      startup detection have something to read).

    Counters (``starts``, ``frees``) let tests assert exactly-once semantics.
    """

    def __init__(
        self,
        key: str,
        *,
        base_url: str = "http://fake.local",
        ready: bool = False,
        loaded: list[str] | None = None,
        fail_start: bool = False,
        start_delay_s: float = 0.0,
    ) -> None:
        # Deliberately do NOT call super().__init__(): that would open a real
        # httpx client we never use. We supply our own minimal attributes.
        self.key = key
        self.base_url = base_url.rstrip("/")
        self._ready = ready
        self._loaded = list(loaded or [])
        self.fail_start = fail_start
        self.start_delay_s = start_delay_s
        self.starts = 0
        self.frees = 0
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    async def is_ready(self) -> bool:
        return self._ready

    async def ensure_started(self) -> None:
        self.starts += 1
        if self.start_delay_s:
            await asyncio.sleep(self.start_delay_s)
        if self.fail_start:
            from router.engines import EngineError

            raise EngineError(f"{self.key}: forced start failure")
        self._ready = True

    async def free_vram(self) -> None:
        self.frees += 1
        self._ready = False
        self._loaded = []

    async def loaded_models(self) -> list[str]:
        return list(self._loaded)

    def is_running(self) -> bool:
        return self._ready


class FakeAPISwapEngine(APISwapEngine):
    """A FakeEngine that *is* an APISwapEngine subclass.

    Needed where routing/startup probes for ``isinstance(engine, APISwapEngine)``
    (e.g. the live-tag fallback). ``available_tags`` is overridable per instance.
    """

    def __init__(
        self,
        key: str,
        *,
        base_url: str = "http://fake-api.local",
        ready: bool = False,
        loaded: list[str] | None = None,
        tags: set[str] | None = None,
    ) -> None:
        self.key = key
        self.base_url = base_url.rstrip("/")
        self._ready = ready
        self._loaded = list(loaded or [])
        self._tags = set(tags or [])
        self.starts = 0
        self.frees = 0
        self.closed = False
        self._tags_cache = None

    async def aclose(self) -> None:
        self.closed = True

    async def is_ready(self) -> bool:
        return self._ready

    async def ensure_started(self) -> None:
        self.starts += 1
        self._ready = True

    async def free_vram(self) -> None:
        self.frees += 1
        self._ready = False
        self._loaded = []

    async def loaded_models(self) -> list[str]:
        return list(self._loaded)

    async def available_tags(self) -> set[str]:
        return set(self._tags)


# --------------------------------------------------------------------------- #
# Config + manager builders
# --------------------------------------------------------------------------- #
def make_config(
    *,
    models: list[ModelSpec] | None = None,
    api_keys: list[str] | None = None,
    **overrides: Any,
) -> RouterConfig:
    """Build a RouterConfig in code with fast timeouts (no slow defaults)."""
    cfg = RouterConfig(
        models=models or [],
        api_keys=api_keys or [],
        # Keep every wait short so the swap machine tests finish instantly.
        drain_timeout_s=overrides.pop("drain_timeout_s", 0.5),
        swap_memory_settle_timeout_s=overrides.pop("swap_memory_settle_timeout_s", 0.2),
        swap_keepalive_interval_s=overrides.pop("swap_keepalive_interval_s", 0.05),
        # Avoid writing into the real repo state.json during tests.
        state_file=overrides.pop("state_file", "/tmp/llm-router-test-state.json"),
        **overrides,
    )
    return cfg


def make_manager_with_fakes(
    fakes: dict[str, Engine],
    *,
    models: list[ModelSpec] | None = None,
    cfg: RouterConfig | None = None,
) -> EngineManager:
    """Construct an EngineManager, then swap in *fakes* as its engine table.

    Mirrors the recipe in the task: build a RouterConfig in code, construct the
    manager, then replace ``manager.engines`` and reset ``manager._inflight``.
    """
    if cfg is None:
        cfg = make_config(models=models or [])
    mgr = EngineManager(cfg)
    mgr.engines = dict(fakes)
    mgr._inflight = {k: 0 for k in mgr.engines}
    mgr.active_engine = None
    return mgr


# --------------------------------------------------------------------------- #
# Live mock upstream (real uvicorn server on an ephemeral port)
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _build_mock_app():
    """A minimal FastAPI app standing in for an upstream engine.

    Route functions are defined at this nesting level but their annotations
    (``Request``) are resolved by FastAPI against this module's globals — which
    is why FastAPI/Request are imported at module scope above (with
    ``from __future__ import annotations`` in effect, annotations are strings
    that FastAPI must look up in the function's ``__globals__``).
    """
    app = FastAPI()

    @app.get("/v1/models")
    async def v1_models():
        return {
            "object": "list",
            "data": [{"id": "mock-model", "object": "model"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("stream"):
            async def gen():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": body.get("model"),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "pong"}}
                ],
            }
        )

    @app.get("/api/tags")
    async def api_tags():
        return {"models": [{"name": "mock-ollama:latest", "model": "mock-ollama:latest"}]}

    @app.get("/api/ps")
    async def api_ps():
        return {"models": []}

    @app.post("/api/chat")
    async def api_chat(request: Request):
        body = await request.json()

        async def gen():
            yield b'{"model":"' + str(body.get("model")).encode() + b'","done":false}\n'
            yield b'{"model":"' + str(body.get("model")).encode() + b'","done":true}\n'

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app


class _BackgroundServer:
    """Run a uvicorn.Server in a daemon thread; expose its base_url."""

    def __init__(self, app, port: int) -> None:
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, timeout_s: float = 10.0) -> None:
        self._thread.start()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.server.started:
                return
            time.sleep(0.02)
        raise RuntimeError("mock upstream server did not start in time")

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5.0)


@pytest.fixture
def mock_upstream():
    """Start a real upstream HTTP server on an ephemeral port for the duration
    of one test. Yields the _BackgroundServer (use ``.base_url``)."""
    port = _free_port()
    srv = _BackgroundServer(_build_mock_app(), port)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@contextlib.contextmanager
def patch_engine_lifecycle(monkeypatch, *engines: Engine):
    """Make the given engines' start/stop/readiness instant and side-effect-free.

    Used by the app-integration tests so that an acquire() against a real engine
    object never touches systemd/GPU/processes; the engine still proxies to its
    (real, mock) base_url over HTTP.
    """
    async def _noop(self):  # pragma: no cover - trivial
        return None

    async def _ready(self):  # pragma: no cover - trivial
        return True

    for eng in engines:
        monkeypatch.setattr(eng, "ensure_started", _noop.__get__(eng))
        monkeypatch.setattr(eng, "free_vram", _noop.__get__(eng))
        monkeypatch.setattr(eng, "is_ready", _ready.__get__(eng))
    yield
