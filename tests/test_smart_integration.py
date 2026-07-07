"""App-level integration tests for smart routing mode.

Same harness as test_app_integration: the real create_app(cfg) driven through
an in-process ASGI transport, engines' base_urls pointing at the live mock
upstream, lifecycle patched instant. The default config here has TWO models on
two engines, with metadata that makes the pick deterministic.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid

import httpx
import pytest

from router import metrics
from router.app import create_app
from router.config import Ds4Config, ModelSpec, OllamaConfig, RouterConfig
from router.engines import EngineError

# Smart health/cooldowns and the benchmark cache persist via state_file and are
# re-loaded on manager construction — a SHARED temp path would leak cooldowns
# between tests. Every test gets a fresh unique path instead.
_STATE_DIR = "/tmp/local-engine-router-smart-it"


@pytest.fixture(autouse=True)
def _fresh_state_file():
    os.makedirs(_STATE_DIR, exist_ok=True)
    path = os.path.join(_STATE_DIR, f"state-{uuid.uuid4().hex}.json")
    global _STATE_FILE
    _STATE_FILE = path
    yield
    with contextlib.suppress(OSError):
        os.unlink(path)


def _smart_config(mock_base: str) -> RouterConfig:
    """ds4 hosts the 'strong' model, ollama the 'weak' one; both proxy to the
    mock upstream. quality tiers pin the ranking so tests are deterministic."""
    return RouterConfig(
        host="127.0.0.1",
        port=8077,
        state_file=_STATE_FILE,
        drain_timeout_s=0.5,
        swap_memory_settle_timeout_s=0.1,
        swap_keepalive_interval_s=0.05,
        ds4=Ds4Config(base_url=mock_base, health_path="/v1/models"),
        ollama=OllamaConfig(base_url=mock_base, health_path="/api/tags"),
        models=[
            ModelSpec(id="strong-70b", engine="ds4", display_name="Strong",
                      quality_tier=5, context_length=131072),
            ModelSpec(id="weak-1b", engine="ollama", display_name="Weak",
                      quality_tier=1, context_length=131072),
            # The mock upstream advertises this live Ollama tag; pin it out of
            # smart selection so picks/fallbacks stay deterministic (and the
            # smart_enabled flag gets exercised end-to-end).
            ModelSpec(id="mock-ollama:latest", engine="ollama",
                      display_name="Mock tag", smart_enabled=False),
        ],
    )


@contextlib.asynccontextmanager
async def _client_for(cfg: RouterConfig):
    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        manager = app.state.manager

        async def _noop(self):
            return None

        async def _ready(self):
            return True

        for eng in manager.engines.values():
            eng.ensure_started = _noop.__get__(eng)
            eng.free_vram = _noop.__get__(eng)
            eng.is_ready = _ready.__get__(eng)

        manager.active_engine = None
        metrics.set_active_engine(None)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://router.test"
        ) as client:
            yield client, manager


def _chat_body(model: str = "smart", **extra):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        **extra,
    }


# --------------------------------------------------------------------------- #
# Smart alias routing
# --------------------------------------------------------------------------- #
async def test_smart_alias_rewrites_and_proxies(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
        assert r.status_code == 200
        # The mock echoes the request body's model: it must be the REAL picked
        # id, never the alias.
        assert r.json()["model"] == "strong-70b"
        assert mgr.smart.last_pick["requested_model"] == "smart"


async def test_smart_headers_on_success(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        r = await client.post("/v1/chat/completions", json=_chat_body("auto"))
        assert r.status_code == 200
        assert r.headers["x-local-engine-router-mode"] == "smart"
        assert r.headers["x-local-engine-router-picked-model"] == "strong-70b"
        assert r.headers["x-local-engine-router-picked-engine"] == "ds4"
        conf = float(r.headers["x-local-engine-router-picker-confidence"])
        assert 0.0 <= conf <= 1.0


async def test_exact_model_id_has_no_smart_headers(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        r = await client.post("/v1/chat/completions", json=_chat_body("weak-1b"))
        assert r.status_code == 200
        assert r.json()["model"] == "weak-1b"
        assert "x-local-engine-router-mode" not in r.headers


async def test_cloud_model_name_routes_to_local(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        r = await client.post(
            "/v1/chat/completions", json=_chat_body("claude-3-5-sonnet-20241022")
        )
        assert r.status_code == 200
        assert r.json()["model"] == "strong-70b"
        assert r.headers["x-local-engine-router-mode"] == "smart"


async def test_smart_streaming_works_with_headers(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        async with client.stream(
            "POST", "/v1/chat/completions", json=_chat_body("smart", stream=True)
        ) as r:
            assert r.status_code == 200
            assert r.headers["x-local-engine-router-mode"] == "smart"
            assert r.headers["x-local-engine-router-picked-model"] == "strong-70b"
            text = (await r.aread()).decode()
        assert "data:" in text
        assert "[DONE]" in text


async def test_smart_api_chat_ndjson(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        r = await client.post(
            "/api/chat",
            json={"model": "smart",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.headers["x-local-engine-router-mode"] == "smart"
        lines = [json.loads(line) for line in r.text.strip().splitlines() if line]
        assert lines and lines[-1]["done"] is True
        assert lines[0]["model"] == "strong-70b"


async def test_manual_mode_disables_smart(mock_upstream):
    cfg = _smart_config(mock_upstream.base_url)
    cfg.routing_mode = "manual"
    async with _client_for(cfg) as (client, _):
        r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
        # Legacy behavior: unknown id falls back to the ollama engine with the
        # body untouched.
        assert r.status_code == 200
        assert r.json()["model"] == "smart"
        assert "x-local-engine-router-mode" not in r.headers


async def test_in_flight_released_after_smart_request(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
        assert r.status_code == 200
        assert all(n == 0 for n in mgr._inflight.values())


# --------------------------------------------------------------------------- #
# Retry / fall-forward
# --------------------------------------------------------------------------- #
async def test_engine_start_failure_falls_forward(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        async def _fail(self):
            raise EngineError("ds4: forced start failure")

        ds4 = mgr.engines["ds4"]
        ds4.ensure_started = _fail.__get__(ds4)

        r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
        assert r.status_code == 200
        # Fell forward to the weak model on the healthy engine.
        assert r.json()["model"] == "weak-1b"
        assert r.headers["x-local-engine-router-picked-model"] == "weak-1b"
        # Both the initial attempt and the same-model reload retry failed.
        assert mgr.smart.health["strong-70b"].consecutive_failures == 2
        assert all(n == 0 for n in mgr._inflight.values())


async def test_engine_start_failure_falls_forward_streaming(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        async def _fail(self):
            raise EngineError("ds4: forced start failure")

        ds4 = mgr.engines["ds4"]
        ds4.ensure_started = _fail.__get__(ds4)

        async with client.stream(
            "POST", "/v1/chat/completions", json=_chat_body("smart", stream=True)
        ) as r:
            text = (await r.aread()).decode()
        # No engine_error chunk: the fallback served real content.
        assert "engine_error" not in text
        assert '"content":"hi"' in text.replace(" ", "")
        assert all(n == 0 for n in mgr._inflight.values())


async def test_connect_failure_falls_forward(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        # ds4 comes up fine but its upstream socket is dead.
        mgr.engines["ds4"].base_url = "http://127.0.0.1:9"  # discard port; closed
        r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
        assert r.status_code == 200
        assert r.json()["model"] == "weak-1b"
        assert mgr.smart.health["strong-70b"].total_failures >= 1


async def test_manual_mode_failure_is_not_retried(mock_upstream):
    cfg = _smart_config(mock_upstream.base_url)
    cfg.routing_mode = "manual"
    async with _client_for(cfg) as (client, mgr):
        async def _fail(self):
            raise EngineError("ds4: forced start failure")

        ds4 = mgr.engines["ds4"]
        ds4.ensure_started = _fail.__get__(ds4)
        r = await client.post("/v1/chat/completions", json=_chat_body("strong-70b"))
        # Exactly the pre-smart behavior: a 5xx engine_error, no fallback.
        assert r.status_code == 503
        assert r.json()["error"]["type"] == "engine_error"


async def test_repeated_failures_enter_cooldown_and_next_pick_avoids(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        # The quality policy keeps re-picking the strong model despite its
        # failures (its quality gap dwarfs the reliability dent), so the
        # failure count actually reaches the cooldown threshold.
        mgr.smart.scfg.policy = "quality"

        async def _fail(self):
            raise EngineError("ds4: forced start failure")

        ds4 = mgr.engines["ds4"]
        ds4.ensure_started = _fail.__get__(ds4)

        # Two smart requests: 2 failures each (initial + reload retry) push the
        # strong model past failure_threshold=3 into cooldown.
        for _ in range(2):
            r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
            assert r.status_code == 200
        assert mgr.smart.in_cooldown("strong-70b")

        # The next pick must not even try the cooled-down model.
        r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
        assert r.status_code == 200
        assert r.json()["model"] == "weak-1b"
        assert r.headers["x-local-engine-router-picked-model"] == "weak-1b"

        # /status surfaces the cooldown.
        st = (await client.get("/status")).json()
        health = st["smart"]["health"]
        assert "strong-70b" in health
        assert health["strong-70b"]["cooldown_until"] > 0


# --------------------------------------------------------------------------- #
# Admin endpoints
# --------------------------------------------------------------------------- #
async def test_admin_smart_resolve_is_side_effect_free(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        before_active = mgr.active_engine
        before_pick = mgr.smart.last_pick
        r = await client.post(
            "/admin/smart/resolve",
            json={"model": "smart",
                  "messages": [{"role": "user", "content": "```py\nx=\n``` fix"}]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["smart_selection"] is True
        assert data["model"] == "strong-70b"
        assert data["engine"] == "ds4"
        assert 0 <= data["confidence"] <= 1
        assert data["policy"] == "balanced"
        assert data["candidates"]
        assert data["benchmarks"] is not None
        assert data["retry"]["never_after_streamed_bytes"] is True
        assert "swap_cost_s" in data
        # Side-effect free: no swap happened, no last-pick recorded.
        assert mgr.active_engine == before_active
        assert mgr.smart.last_pick == before_pick


async def test_admin_smart_resolve_exact_id_explains(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        r = await client.post("/admin/smart/resolve", json={"model": "weak-1b"})
        assert r.status_code == 200
        data = r.json()
        assert data["smart_selection"] is False
        assert "exact" in data["reason"]


async def test_admin_smart_mode_toggles_at_runtime(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        r = await client.post("/admin/smart/mode", json={"mode": "manual"})
        assert r.status_code == 200
        assert r.json() == {"routing_mode": "manual"}
        assert mgr.smart.mode == "manual"

        r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
        assert r.json()["model"] == "smart"  # legacy fallback, no rewrite

        r = await client.post("/admin/smart/mode", json={"mode": "smart"})
        assert r.status_code == 200
        r = await client.post("/v1/chat/completions", json=_chat_body("smart"))
        # Smart again: a real model is picked (which one depends on residency —
        # the manual request above parked the GPU on ollama, and staying
        # resident is legitimate swap-aware behaviour).
        assert r.json()["model"] != "smart"
        assert r.headers["x-local-engine-router-mode"] == "smart"

        r = await client.post("/admin/smart/mode", json={"mode": "clever"})
        assert r.status_code == 400


async def test_admin_benchmarks_endpoints(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        # A pick populates the cache.
        await client.post("/v1/chat/completions", json=_chat_body("smart"))
        r = await client.get("/admin/benchmarks")
        assert r.status_code == 200
        summary = r.json()
        assert summary["cached_models"] >= 1
        assert summary["providers"][0]["name"] == "builtin-priors"

        r = await client.post("/admin/benchmarks/refresh", json={"model": "weak-1b"})
        assert r.status_code == 200
        assert r.json()["count"] >= 1

        r = await client.post("/admin/benchmarks/clear", json={})
        assert r.status_code == 200
        assert r.json()["cleared"] >= 1
        assert (await client.get("/admin/benchmarks")).json()["cached_models"] == 0


async def test_admin_smart_calibrate(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, mgr):
        r = await client.post("/admin/smart/calibrate", json={"model": "weak-1b"})
        assert r.status_code == 200
        data = r.json()
        assert data["model"] == "weak-1b"
        assert "probes" in data and "scores" in data
        # The mock upstream answers "pong" to everything, so instruction/JSON
        # probes fail — what matters is that results were stored and released.
        assert "weak-1b" in mgr.smart.calibration
        assert all(n == 0 for n in mgr._inflight.values())


async def test_admin_smart_calibrate_gated(mock_upstream):
    cfg = _smart_config(mock_upstream.base_url)
    cfg.smart.calibration_enabled = False
    async with _client_for(cfg) as (client, _):
        r = await client.post("/admin/smart/calibrate", json={"model": "weak-1b"})
        assert r.status_code == 403


# --------------------------------------------------------------------------- #
# Status & metrics
# --------------------------------------------------------------------------- #
async def test_status_smart_section(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        await client.post("/v1/chat/completions", json=_chat_body("smart"))
        st = (await client.get("/status")).json()
        assert st["routing_mode"] == "smart"
        smart = st["smart"]
        assert smart["mode"] == "smart"
        assert smart["policy"] == "balanced"
        assert smart["last_pick"]["model"] == "strong-70b"
        assert "benchmark_cache" in smart


async def test_smart_metrics_exported(mock_upstream):
    async with _client_for(_smart_config(mock_upstream.base_url)) as (client, _):
        await client.post("/v1/chat/completions", json=_chat_body("smart"))
        text = (await client.get("/metrics")).text
        assert "# TYPE smart_pick_total counter" in text
        assert 'smart_pick_total{model="strong-70b"' in text
        assert "# TYPE smart_pick_confidence histogram" in text


async def test_smart_state_persisted_across_restart(mock_upstream):
    cfg = _smart_config(mock_upstream.base_url)
    async with _client_for(cfg) as (client, mgr):
        mgr.smart.record_failure("strong-70b", "synthetic")
        await client.post("/v1/chat/completions", json=_chat_body("smart"))
        mgr._persist()

    # New app instance, same state file: health + benchmark cache survive.
    async with _client_for(cfg) as (_, mgr2):
        assert mgr2.smart.health["strong-70b"].total_failures >= 1
        assert mgr2.smart.benchmarks.summary()["cached_models"] >= 1
