"""Section F — v0.3 engine-coverage features (EC1-EC5, MM4).

Hermetic: every control/health/load/unload call an engine makes goes through an
``httpx.MockTransport`` mounted onto the engine's real ``_ctl`` client, so we
assert exactly what the router sends (headers, method, path, body) and feed back
canned responses — no sockets, no real backend.

  EC1  control_headers are sent on control calls (and not required elsewhere).
  EC2  load_model issues the configured load request; acquire loads on swap-in
       for a load_path engine and skips when the model is already loaded.
  EC3  loaded_filter keeps only actually-loaded entries; loaded_id_key makes
       unload use the per-instance id; a single-object loaded_path is handled.
  EC4  ready_check asserts a JSON field / model-in-list beyond HTTP 200.
  EC5  free_vram signals the whole process GROUP (reaps a forked child).
  MM4  aliases resolve in routing AND the outgoing body's model is rewritten.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import httpx
import pytest

from router.config import (
    ApiSwapConfig,
    GenericProcessConfig,
    ModelSpec,
    RouterConfig,
)
from router.engines import (
    APISwapEngine,
    Engine,
    EngineError,
    GenericProcessEngine,
    _ready_check_passes,
)

from conftest import make_config, make_manager_with_fakes


# --------------------------------------------------------------------------- #
# MockTransport plumbing
# --------------------------------------------------------------------------- #
class _Recorder:
    """A MockTransport handler that records requests and serves routed replies.

    ``routes`` maps ``(METHOD, path)`` -> a callable(request) -> httpx.Response
    or a plain httpx.Response. Unmatched requests get a 404 (and are still
    recorded), which surfaces accidental calls in tests."""

    def __init__(self, routes=None) -> None:
        self.routes = routes or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method.upper(), request.url.path)
        handler = self.routes.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": "no route"})
        if callable(handler):
            return handler(request)
        return handler

    # Convenience queries -------------------------------------------------- #
    def calls_to(self, method: str, path: str) -> list[httpx.Request]:
        return [
            r
            for r in self.requests
            if r.method.upper() == method.upper() and r.url.path == path
        ]


def _mount(engine, recorder: _Recorder) -> _Recorder:
    """Mount *recorder* onto the engine's real _ctl client (keeps its headers)."""
    engine._ctl._transport = httpx.MockTransport(recorder)
    return recorder


# =========================================================================== #
# EC1 — control-call auth headers
# =========================================================================== #
async def test_ec1_control_headers_sent_on_api_swap_control_calls():
    cfg = ApiSwapConfig(
        base_url="http://tabby.local",
        health_path="/v1/model",
        loaded_path="/v1/model",
        unload_path="/v1/model/unload",
        control_headers={"x-admin-key": "s3cret"},
    )
    eng = APISwapEngine(cfg, key="tabby")
    rec = _mount(
        eng,
        _Recorder(
            {
                ("GET", "/v1/model"): httpx.Response(200, json={"id": "m"}),
                ("POST", "/v1/model/unload"): httpx.Response(200, json={}),
            }
        ),
    )
    try:
        await eng.is_ready()
        await eng.loaded_models()
        await eng._unload("m")
    finally:
        await eng.aclose()

    assert rec.requests, "no control calls were made"
    # Every control call carried the configured admin key.
    for r in rec.requests:
        assert r.headers.get("x-admin-key") == "s3cret"


async def test_ec1_generic_process_control_headers_sent():
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
        control_headers={"authorization": "Bearer tok"},
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    rec = _mount(eng, _Recorder({("GET", "/health"): httpx.Response(200)}))
    try:
        await eng.is_ready()
    finally:
        await eng.aclose()
    assert rec.calls_to("GET", "/health")
    assert rec.requests[0].headers.get("authorization") == "Bearer tok"


async def test_ec1_default_no_control_headers():
    """Default (no control_headers) sends no auth header — unchanged behavior."""
    cfg = ApiSwapConfig(base_url="http://x.local", health_path="/v1/models")
    eng = APISwapEngine(cfg, key="x")
    rec = _mount(eng, _Recorder({("GET", "/v1/models"): httpx.Response(200, json={})}))
    try:
        await eng.is_ready()
    finally:
        await eng.aclose()
    assert "x-admin-key" not in rec.requests[0].headers
    assert "authorization" not in rec.requests[0].headers


# =========================================================================== #
# EC2 — generic HTTP load_path
# =========================================================================== #
def _tabby_cfg(**extra) -> ApiSwapConfig:
    return ApiSwapConfig(
        base_url="http://tabby.local",
        health_path="/v1/model",
        loaded_path="/v1/model",
        loaded_models_key="data",
        loaded_name_key="id",
        load_path="/v1/model/load",
        load_body={"model_name": "{model}"},
        unload_path="/v1/model/unload",
        **extra,
    )


async def test_ec2_load_model_issues_configured_request():
    eng = APISwapEngine(_tabby_cfg(), key="tabby")
    rec = _mount(
        eng, _Recorder({("POST", "/v1/model/load"): httpx.Response(200, json={"ok": 1})})
    )
    try:
        await eng.load_model("my-exl2")
    finally:
        await eng.aclose()
    loads = rec.calls_to("POST", "/v1/model/load")
    assert len(loads) == 1
    import json as _json

    assert _json.loads(loads[0].content) == {"model_name": "my-exl2"}


async def test_ec2_load_model_noop_without_load_path():
    cfg = ApiSwapConfig(base_url="http://x.local", load_path="")
    eng = APISwapEngine(cfg, key="x")
    rec = _mount(eng, _Recorder())
    try:
        await eng.load_model("anything")
    finally:
        await eng.aclose()
    assert rec.requests == []  # nothing issued


async def test_ec2_load_model_raises_on_http_error():
    eng = APISwapEngine(_tabby_cfg(), key="tabby")
    _mount(eng, _Recorder({("POST", "/v1/model/load"): httpx.Response(500, json={})}))
    try:
        with pytest.raises(EngineError, match="load of model"):
            await eng.load_model("boom")
    finally:
        await eng.aclose()


async def test_ec2_acquire_loads_model_on_swap_in():
    """acquire() against an active load_path engine loads the requested model
    when it is not already loaded."""
    eng = APISwapEngine(_tabby_cfg(), key="tabby")
    # loaded_path returns NOTHING loaded -> acquire must issue a load.
    _mount(
        eng,
        _Recorder(
            {
                ("GET", "/v1/model"): httpx.Response(200, json={"data": []}),
                ("POST", "/v1/model/load"): httpx.Response(200, json={"ok": 1}),
            }
        ),
    )
    models = [ModelSpec(id="my-exl2", engine="tabby", display_name="exl2")]
    mgr = make_manager_with_fakes({"tabby": eng}, cfg=make_config(models=models))
    try:
        await mgr.acquire("my-exl2")
        assert mgr.active_engine == "tabby"
        rec = eng._ctl._transport.handler  # type: ignore[attr-defined]
        assert len(rec.calls_to("POST", "/v1/model/load")) == 1
        assert mgr._inflight["tabby"] == 1
    finally:
        await mgr.aclose()


async def test_ec2_acquire_skips_load_when_already_loaded():
    """If loaded_path already reports the requested model, acquire issues NO load."""
    eng = APISwapEngine(_tabby_cfg(), key="tabby")
    _mount(
        eng,
        _Recorder(
            {
                # The model is already loaded (keyed by loaded_name_key="id").
                ("GET", "/v1/model"): httpx.Response(
                    200, json={"data": [{"id": "my-exl2"}]}
                ),
                ("POST", "/v1/model/load"): httpx.Response(200, json={"ok": 1}),
            }
        ),
    )
    models = [ModelSpec(id="my-exl2", engine="tabby", display_name="exl2")]
    mgr = make_manager_with_fakes({"tabby": eng}, cfg=make_config(models=models))
    try:
        await mgr.acquire("my-exl2")
        rec = eng._ctl._transport.handler  # type: ignore[attr-defined]
        assert rec.calls_to("POST", "/v1/model/load") == []
    finally:
        await mgr.aclose()


async def test_ec2_load_failure_leaks_no_inflight():
    eng = APISwapEngine(_tabby_cfg(), key="tabby")
    _mount(
        eng,
        _Recorder(
            {
                ("GET", "/v1/model"): httpx.Response(200, json={"data": []}),
                ("POST", "/v1/model/load"): httpx.Response(500, json={}),
            }
        ),
    )
    models = [ModelSpec(id="my-exl2", engine="tabby", display_name="exl2")]
    mgr = make_manager_with_fakes({"tabby": eng}, cfg=make_config(models=models))
    try:
        with pytest.raises(EngineError):
            await mgr.acquire("my-exl2")
        # The failed load happened before the in-flight increment -> no leak.
        assert mgr._inflight["tabby"] == 0
    finally:
        await mgr.aclose()


# =========================================================================== #
# EC3 — loaded-state filtering + id keying + single object
# =========================================================================== #
async def test_ec3_loaded_filter_keeps_only_loaded_entries():
    cfg = ApiSwapConfig(
        base_url="http://lms.local",
        loaded_path="/api/v0/models",
        loaded_models_key="data",
        loaded_name_key="id",
        loaded_filter="state==loaded",
    )
    eng = APISwapEngine(cfg, key="lms")
    _mount(
        eng,
        _Recorder(
            {
                ("GET", "/api/v0/models"): httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "a", "state": "loaded"},
                            {"id": "b", "state": "not-loaded"},
                            {"id": "c", "state": "loaded"},
                        ]
                    },
                )
            }
        ),
    )
    try:
        loaded = await eng.loaded_models()
    finally:
        await eng.aclose()
    assert loaded == ["a", "c"]


async def test_ec3_loaded_id_key_used_for_names_and_unload():
    cfg = ApiSwapConfig(
        base_url="http://lms.local",
        loaded_path="/api/v0/models",
        loaded_models_key="data",
        loaded_name_key="id",
        loaded_id_key="instance_id",
        loaded_filter="state==loaded",
        unload_path="/api/v0/unload",
        unload_body={"instance": "{model}"},
        # Keep the post-unload "wait until empty" poll short: our mock keeps
        # reporting the model as loaded, so free_vram would otherwise block for
        # the full default unload_timeout_s. We only assert the unload REQUEST.
        unload_timeout_s=0.2,
    )
    eng = APISwapEngine(cfg, key="lms")
    rec = _mount(
        eng,
        _Recorder(
            {
                ("GET", "/api/v0/models"): httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "qwen", "instance_id": "inst-7", "state": "loaded"}
                        ]
                    },
                ),
                ("POST", "/api/v0/unload"): httpx.Response(200, json={}),
            }
        ),
    )
    try:
        # loaded_models() returns the UNLOAD id (instance_id), not the name.
        assert await eng.loaded_models() == ["inst-7"]
        # loaded_model_names() returns the DISPLAY name (used by the load skip).
        assert await eng.loaded_model_names() == ["qwen"]
        await eng.free_vram()
    finally:
        await eng.aclose()

    import json as _json

    unloads = rec.calls_to("POST", "/api/v0/unload")
    assert unloads, "no unload issued"
    # Unload body used the per-instance id, not the display name.
    assert _json.loads(unloads[0].content) == {"instance": "inst-7"}


async def test_ec3_available_tags_uses_display_names_not_unload_ids():
    """Routing matches the model id clients send, so available_tags() must be
    the display names — NOT the loaded_id_key unload identifiers."""
    cfg = ApiSwapConfig(
        base_url="http://lms.local",
        loaded_path="/api/v0/models",
        loaded_models_key="data",
        loaded_name_key="id",
        loaded_id_key="instance_id",
        loaded_filter="state==loaded",
        tags_cache_ttl_s=0.0,  # disable caching so the probe is hit
    )
    eng = APISwapEngine(cfg, key="lms")
    _mount(
        eng,
        _Recorder(
            {
                ("GET", "/api/v0/models"): httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "qwen", "instance_id": "inst-7", "state": "loaded"}
                        ]
                    },
                )
            }
        ),
    )
    try:
        tags = await eng.available_tags()
    finally:
        await eng.aclose()
    assert tags == {"qwen"}, "available_tags must use display names, not instance ids"


async def test_ec3_single_object_loaded_path():
    """TabbyAPI /v1/model returns a SINGLE loaded-model object, not a list."""
    cfg = ApiSwapConfig(
        base_url="http://tabby.local",
        loaded_path="/v1/model",
        loaded_name_key="id",
    )
    eng = APISwapEngine(cfg, key="tabby")
    _mount(
        eng,
        _Recorder(
            {("GET", "/v1/model"): httpx.Response(200, json={"id": "resident-exl2"})}
        ),
    )
    try:
        assert await eng.loaded_models() == ["resident-exl2"]
    finally:
        await eng.aclose()


async def test_ec3_default_name_keyed_list_unchanged():
    """No filter, name-keyed list (Ollama-style) is unaffected."""
    cfg = ApiSwapConfig(
        base_url="http://x.local",
        loaded_path="/api/ps",
        loaded_models_key="models",
        loaded_name_key="name",
    )
    eng = APISwapEngine(cfg, key="x")
    _mount(
        eng,
        _Recorder(
            {
                ("GET", "/api/ps"): httpx.Response(
                    200, json={"models": [{"name": "m1"}, {"name": "m2"}]}
                )
            }
        ),
    )
    try:
        assert await eng.loaded_models() == ["m1", "m2"]
    finally:
        await eng.aclose()


# =========================================================================== #
# EC4 — richer readiness probe
# =========================================================================== #
def test_ec4_ready_check_helper_field_equals():
    resp = httpx.Response(200, json={"status": "ok"})
    assert _ready_check_passes("status==ok", resp) is True
    resp2 = httpx.Response(200, json={"status": "loading"})
    assert _ready_check_passes("status==ok", resp2) is False


def test_ec4_ready_check_helper_model_in_list():
    resp = httpx.Response(
        200, json={"data": [{"id": "served-model"}, {"id": "other"}]}
    )
    assert _ready_check_passes("model:served-model", resp) is True
    assert _ready_check_passes("model:absent", resp) is False


def test_ec4_ready_check_empty_passes_on_200():
    assert _ready_check_passes("", httpx.Response(200, json={"anything": 1})) is True


def test_ec4_ready_check_non_json_body_fails_assertion():
    assert _ready_check_passes("status==ok", httpx.Response(200, text="not json")) is False


async def test_ec4_generic_is_ready_honors_ready_check():
    """vLLM /health returns 200 even before the model can serve; ready_check
    on a body field distinguishes truly-ready from 200-but-not-ready."""
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
        ready_check="status==ok",
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    # 200 but the wrong status -> NOT ready.
    rec = _Recorder({("GET", "/health"): httpx.Response(200, json={"status": "starting"})})
    _mount(eng, rec)
    try:
        assert await eng.is_ready() is False
        # Flip the canned response to the ready status.
        rec.routes[("GET", "/health")] = httpx.Response(200, json={"status": "ok"})
        assert await eng.is_ready() is True
    finally:
        await eng.aclose()


async def test_ec4_api_swap_is_ready_honors_ready_check():
    cfg = ApiSwapConfig(
        base_url="http://tabby.local",
        health_path="/v1/models",
        ready_check="model:loaded-one",
    )
    eng = APISwapEngine(cfg, key="tabby")
    rec = _Recorder(
        {("GET", "/v1/models"): httpx.Response(200, json={"data": [{"id": "other"}]})}
    )
    _mount(eng, rec)
    try:
        assert await eng.is_ready() is False
        rec.routes[("GET", "/v1/models")] = httpx.Response(
            200, json={"data": [{"id": "loaded-one"}]}
        )
        assert await eng.is_ready() is True
    finally:
        await eng.aclose()


# =========================================================================== #
# EC5 — process-group reaping
# =========================================================================== #
@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
async def test_ec5_free_vram_reaps_child_process_group():
    """A launched process that forks a child must have BOTH reaped: free_vram
    signals the whole process group (start_new_session=True), so the forked
    grandchild dies too — not just the launcher PID."""
    # A parent shell that forks a long-lived child (sleep), then itself waits.
    # Both share the session/group the launcher leads, so a group SIGTERM hits
    # the sleeping child as well.
    script = (
        "import os, sys, time, subprocess\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "sys.stdout.write(str(child.pid) + '\\n'); sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    cfg = GenericProcessConfig(
        base_url="http://127.0.0.1:1",  # unused port: _port_open() returns False
        start_cmd=[sys.executable, "-c", script],
        ready_path="/",
        stop_signal="SIGTERM",
        stop_timeout_s=5.0,
    )
    eng = GenericProcessEngine(cfg, key="grp")

    # Stop teardown from depending on a real socket: report the port closed so
    # _wait_stopped/free_vram decide "stopped" purely from the process group.
    async def _closed():
        return False

    eng._port_open = _closed  # type: ignore[assignment]

    leader: int | None = None
    child_pids: list[int] = []
    try:
        eng._launch()
        assert eng._proc is not None
        # The launcher's stdout goes to DEVNULL, so discover its forked child
        # by listing the process group the launcher leads.
        leader = eng._proc.pid
        pgid = os.getpgid(leader)
        deadline = time.time() + 5.0  # give the parent a moment to fork
        while time.time() < deadline:
            child_pids = _pids_in_group(pgid, exclude=leader)
            if child_pids:
                break
            time.sleep(0.05)
        assert child_pids, "expected a forked child in the process group"
        child = child_pids[0]

        await eng.free_vram()

        # Both the launcher and the forked child must be gone.
        assert not _alive(leader), "launcher still alive after free_vram"
        assert not _alive(child), "forked child survived (group not reaped)"
    finally:
        # Belt-and-suspenders cleanup if anything above raised mid-way.
        for pid in (leader, *child_pids):
            if pid:
                _force_kill(pid)
        await eng.aclose()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pids_in_group(pgid: int, *, exclude: int) -> list[int]:
    import subprocess

    out = subprocess.run(
        ["pgrep", "-g", str(pgid)], capture_output=True, text=True, timeout=5
    )
    pids = []
    for line in out.stdout.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != exclude and pid != os.getpid():
            pids.append(pid)
    return pids


def _force_kill(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# =========================================================================== #
# MM4 — alias / capability routing
# =========================================================================== #
async def test_mm4_alias_resolves_in_engine_for():
    models = [
        ModelSpec(id="deepseek-v4-flash", engine="ds4", display_name="d"),
    ]
    cfg = make_config(models=models, aliases={"gpt-4o": "deepseek-v4-flash"})
    from conftest import FakeAPISwapEngine, FakeEngine

    mgr = make_manager_with_fakes(
        {"ds4": FakeEngine("ds4"), "ollama": FakeAPISwapEngine("ollama")}, cfg=cfg
    )
    # The alias routes to the real model's engine.
    eng = await mgr.engine_for("gpt-4o")
    assert eng.key == "ds4"
    # resolve_model_id exposes the mapping; non-aliases pass through unchanged.
    assert mgr.resolve_model_id("gpt-4o") == "deepseek-v4-flash"
    assert mgr.resolve_model_id("deepseek-v4-flash") == "deepseek-v4-flash"
    assert mgr.resolve_model_id("unknown") == "unknown"


def test_mm4_alias_validation_rejects_chain(tmp_path):
    import textwrap

    from router.config import ConfigError, load_config

    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            models:
              - id: real
                engine: ds4
            aliases:
              a: b
              b: real
            """
        )
    )
    with pytest.raises(ConfigError, match="chain"):
        load_config(str(p))


def test_mm4_alias_validation_rejects_self(tmp_path):
    import textwrap

    from router.config import ConfigError, load_config

    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            models:
              - id: real
                engine: ds4
            aliases:
              loop: loop
            """
        )
    )
    with pytest.raises(ConfigError, match="itself"):
        load_config(str(p))


def test_mm4_alias_key_shadowing_model_id_raises(tmp_path):
    """An alias key that equals a configured model id would silently rewrite
    every request for that real model — reject it."""
    import textwrap

    from router.config import ConfigError, load_config

    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            models:
              - id: real
                engine: ds4
              - id: shadow
                engine: ds4
            aliases:
              shadow: real
            """
        )
    )
    with pytest.raises(ConfigError, match="collides with a configured model id"):
        load_config(str(p))


def test_mm4_alias_unknown_target_warns_not_fatal(tmp_path, caplog):
    import textwrap

    from router.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            models:
              - id: real
                engine: ds4
            aliases:
              fast: some-ollama-tag:latest
            """
        )
    )
    # Unknown target (could be a live Ollama tag) -> loads, with a warning.
    cfg = load_config(str(p))
    assert cfg.aliases["fast"] == "some-ollama-tag:latest"


# =========================================================================== #
# AM1 — available_models() capability
# =========================================================================== #
async def test_am1_base_engine_available_models_returns_empty():
    """Engine base class available_models() must return an empty set so that
    subclasses that do not override the method leave existing behavior intact."""

    class _Minimal(Engine):
        key = "minimal"
        base_url = "http://x.local"

        async def is_ready(self):
            return True

        async def ensure_started(self):
            pass

        async def free_vram(self):
            pass

    eng = _Minimal()
    try:
        result = await eng.available_models()
    finally:
        await eng.aclose()
    assert result == set()


async def test_am1_generic_process_running_parses_v1_models():
    """A running GenericProcessEngine parses the OpenAI /v1/models shape into
    the correct id set."""
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    rec = _mount(
        eng,
        _Recorder(
            {
                ("GET", "/health"): httpx.Response(200),
                ("GET", "/v1/models"): httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "data": [
                            {"id": "model-a", "object": "model"},
                            {"id": "model-b", "object": "model"},
                        ],
                    },
                ),
            }
        ),
    )
    try:
        result = await eng.available_models()
    finally:
        await eng.aclose()
    assert result == {"model-a", "model-b"}
    # The is_ready probe and the model list were both issued.
    assert rec.calls_to("GET", "/health")
    assert rec.calls_to("GET", "/v1/models")


async def test_am1_generic_process_not_ready_returns_empty():
    """A not-ready GenericProcessEngine must return empty without hitting the
    model list endpoint at all."""
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    rec = _mount(
        eng,
        _Recorder(
            {
                # Health endpoint returns non-200 -> engine is not ready.
                ("GET", "/health"): httpx.Response(503),
            }
        ),
    )
    try:
        result = await eng.available_models()
    finally:
        await eng.aclose()
    assert result == set()
    # /v1/models must NOT have been called.
    assert rec.calls_to("GET", "/v1/models") == []


async def test_am1_generic_process_http_error_returns_empty():
    """An HTTP error on /v1/models returns empty (best-effort; never raises)."""
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    _mount(
        eng,
        _Recorder(
            {
                ("GET", "/health"): httpx.Response(200),
                ("GET", "/v1/models"): httpx.Response(500, json={"error": "boom"}),
            }
        ),
    )
    try:
        result = await eng.available_models()
    finally:
        await eng.aclose()
    assert result == set()


async def test_am1_generic_process_ttl_cache_returns_cached_within_window():
    """Within the TTL window the cached set is returned without a second HTTP
    call to /v1/models."""
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    rec = _mount(
        eng,
        _Recorder(
            {
                ("GET", "/health"): httpx.Response(200),
                ("GET", "/v1/models"): httpx.Response(
                    200,
                    json={"object": "list", "data": [{"id": "cached-model"}]},
                ),
            }
        ),
    )
    try:
        first = await eng.available_models()
        second = await eng.available_models()
    finally:
        await eng.aclose()
    assert first == {"cached-model"}
    assert second == {"cached-model"}
    # /v1/models was only called once despite two available_models() calls.
    assert len(rec.calls_to("GET", "/v1/models")) == 1


async def test_am1_generic_process_ttl_cache_refetches_after_expiry():
    """After the TTL expires a fresh probe is issued."""
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    rec = _mount(
        eng,
        _Recorder(
            {
                ("GET", "/health"): httpx.Response(200),
                ("GET", "/v1/models"): httpx.Response(
                    200,
                    json={"object": "list", "data": [{"id": "stale-model"}]},
                ),
            }
        ),
    )
    try:
        await eng.available_models()
        # Expire the cache by back-dating its timestamp.
        assert eng._models_cache is not None
        ts, ids = eng._models_cache
        eng._models_cache = (ts - 999.0, ids)
        await eng.available_models()
    finally:
        await eng.aclose()
    # Two fetches must have gone to /v1/models (initial + post-expiry).
    assert len(rec.calls_to("GET", "/v1/models")) == 2


async def test_am1_generic_process_not_ready_result_is_cached():
    """A not-ready result must be cached so the engine is probed at most once
    per TTL rather than on every call when it is down."""
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    rec = _mount(
        eng,
        _Recorder(
            {
                # Engine is not ready.
                ("GET", "/health"): httpx.Response(503),
            }
        ),
    )
    try:
        first = await eng.available_models()
        second = await eng.available_models()
    finally:
        await eng.aclose()
    assert first == set()
    assert second == set()
    # The health probe must have been issued only once; the second call hit the
    # cache and never called is_ready() again.
    assert len(rec.calls_to("GET", "/health")) == 1


async def test_am1_generic_process_http_error_result_is_cached():
    """An HTTP error on /v1/models must cache the empty result so a broken
    engine is probed at most once per TTL."""
    cfg = GenericProcessConfig(
        base_url="http://vllm.local",
        start_cmd=["/bin/true"],
        ready_path="/health",
    )
    eng = GenericProcessEngine(cfg, key="vllm")
    rec = _mount(
        eng,
        _Recorder(
            {
                ("GET", "/health"): httpx.Response(200),
                ("GET", "/v1/models"): httpx.Response(500, json={"error": "oops"}),
            }
        ),
    )
    try:
        first = await eng.available_models()
        second = await eng.available_models()
    finally:
        await eng.aclose()
    assert first == set()
    assert second == set()
    # /v1/models was only queried once despite two available_models() calls.
    assert len(rec.calls_to("GET", "/v1/models")) == 1
