"""Section E — App integration via httpx.ASGITransport (in-process).

Builds the real ``create_app(cfg)`` and drives it through an in-process ASGI
transport (no real server for the router itself). The engines' base_urls point
at a *real* mock upstream (a uvicorn.Server on an ephemeral 127.0.0.1 port,
started by the ``mock_upstream`` fixture), so a request genuinely flows
create_app -> proxy -> HTTP upstream. Engine lifecycle (ensure_started /
free_vram / is_ready) is monkeypatched instant so a 'swap' never touches
systemd/GPU.

The router's FastAPI lifespan is entered manually (httpx.ASGITransport does not
run lifespan events), which is what populates app.state.manager / app.state.client.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import urllib.request

import httpx

from router import metrics
from router.app import create_app
from router.config import Ds4Config, DiscoverConfig, ModelSpec, OllamaConfig, RouterConfig
import router.cli as cli


def _app_config(mock_base: str, *, api_keys=None, aliases=None) -> RouterConfig:
    """Config whose ds4 + ollama both point at the mock upstream's base_url.

    The legacy ds4:/ollama: sections build a Ds4Engine + OllamaEngine; keeping
    the ollama key as an OllamaEngine is what makes /v1/models tag enrichment
    and the /api/* passthrough work (the app resolves Ollama by type).
    Discovery is OFF by default (the invariant: discover.enabled=False is
    byte-identical to origin/main behavior).
    """
    return RouterConfig(
        host="127.0.0.1",
        port=8077,
        api_keys=list(api_keys or []),
        aliases=dict(aliases or {}),
        state_file="/tmp/local-engine-router-test-state.json",
        drain_timeout_s=0.5,
        swap_memory_settle_timeout_s=0.1,
        swap_keepalive_interval_s=0.05,
        ds4=Ds4Config(base_url=mock_base, health_path="/v1/models"),
        ollama=OllamaConfig(base_url=mock_base, health_path="/api/tags"),
        models=[
            ModelSpec(id="deepseek-v4-flash", engine="ds4", display_name="DS4 Flash"),
            ModelSpec(id="qwen2.5:3b", engine="ollama", display_name="Qwen 3B"),
        ],
    )


def _app_config_discover_on(mock_base: str, *, api_keys=None, aliases=None) -> RouterConfig:
    """Like _app_config but with discover.enabled=True.

    Used by tests that verify the discovery-on code path (all-engine
    available_models() union + _discovered_index merge).
    """
    cfg = _app_config(mock_base, api_keys=api_keys, aliases=aliases)
    # RouterConfig is a plain dataclass (not frozen), so field assignment works.
    cfg.discover = DiscoverConfig(enabled=True)
    return cfg


@contextlib.asynccontextmanager
async def _client_for(cfg: RouterConfig):
    """Enter the app lifespan, patch engine lifecycle instant, yield a client."""
    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        manager = app.state.manager

        # Make every engine's lifecycle instant + side-effect-free. The base_url
        # still points at the live mock, so proxying is real HTTP.
        async def _noop(self):
            return None

        async def _ready(self):
            return True

        for eng in manager.engines.values():
            eng.ensure_started = _noop.__get__(eng)
            eng.free_vram = _noop.__get__(eng)
            eng.is_ready = _ready.__get__(eng)

        # startup() probed the live mock and may have detected an engine as
        # already active (the mock answers ds4's /v1/models health probe). Reset
        # to a known "nothing active" baseline so the first acquire deterministically
        # swaps — that's what the routing/metrics assertions below rely on.
        manager.active_engine = None
        metrics.set_active_engine(None)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://router.test"
        ) as client:
            yield client, manager


async def test_health(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


async def test_metrics_endpoint(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        r = await client.get("/metrics")
        assert r.status_code == 200
        # Prometheus exposition content-type (format version 0.0.4).
        assert r.headers["content-type"].startswith("text/plain")
        assert "version=0.0.4" in r.headers["content-type"]
        assert "# TYPE swap_duration_seconds histogram" in r.text


async def test_status_shape(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        r = await client.get("/status")
        assert r.status_code == 200
        st = r.json()
        assert "active_engine" in st
        assert "engines" in st
        assert set(st["engines"]) == {"ds4", "ollama"}
        assert "models" in st
        ids = {m["id"] for m in st["models"]}
        assert {"deepseek-v4-flash", "qwen2.5:3b"} <= ids


async def test_v1_models_union(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        r = await client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        ids = {m["id"] for m in data["data"]}
        # Static registry entries.
        assert "deepseek-v4-flash" in ids
        assert "qwen2.5:3b" in ids
        # Live ollama tag from the mock /api/tags (union with static).
        assert "mock-ollama:latest" in ids


async def test_chat_completions_non_stream_proxied_and_routed(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, mgr):
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        body = r.json()
        # The mock echoed our model + a pong message -> proof it reached upstream.
        assert body["model"] == "deepseek-v4-flash"
        assert body["choices"][0]["message"]["content"] == "pong"
        # Routed to ds4 and became the active engine via acquire().
        assert mgr.active_engine == "ds4"
        # And a swap to ds4 was recorded in metrics.
        assert 'to="ds4",result="ok"' in metrics.render()


async def test_chat_routes_to_ollama_for_ollama_model(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, mgr):
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.status_code == 200
        assert mgr.active_engine == "ollama"


async def test_chat_completions_streaming_yields_bytes(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        chunks: list[bytes] = []
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "deepseek-v4-flash", "stream": True, "messages": []},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
        joined = b"".join(chunks)
        assert b"data:" in joined
        assert b"[DONE]" in joined


async def test_api_chat_streaming_yields_ndjson(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        chunks: list[bytes] = []
        async with client.stream(
            "POST",
            "/api/chat",
            json={"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "y"}]},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/x-ndjson")
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
        joined = b"".join(chunks)
        # NDJSON lines from the mock — no SSE "data:" framing.
        assert b'"done":true' in joined
        assert b"data:" not in joined


async def test_streaming_aborts_on_client_disconnect(mock_upstream, monkeypatch):
    """Regression: a client that disconnects mid-stream must NOT leave the
    upstream generating to completion against a dead socket.

    This reproduces the orphaned-generation GPU leak: under ASGI>=2.4 Starlette
    only cancels a streaming generator when send() raises, which it may not for a
    half-closed client — so the router must poll request.is_disconnected() and
    stop pulling from upstream. We simulate the disconnect by flipping
    is_disconnected True after a couple of frames, then assert the stream is cut
    off early (well before the upstream's full run) and the engine's in-flight
    slot is released back to zero."""
    import time

    import starlette.requests

    # The router polls is_disconnected() once per upstream chunk (and during the
    # swap wait). Report "connected" for the first few polls so a *running* stream
    # is interrupted, then "disconnected".
    calls = {"n": 0}

    async def fake_is_disconnected(self):
        calls["n"] += 1
        return calls["n"] > 3

    monkeypatch.setattr(
        starlette.requests.Request, "is_disconnected", fake_is_disconnected
    )

    N = 200  # upstream would emit 200 frames (~1s) if drained to completion
    frames = 0
    started = time.monotonic()
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, mgr):
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "stream": True,
                "messages": [],
                "_test_stream_n": N,
            },
        ) as resp:
            assert resp.status_code == 200
            async for chunk in resp.aiter_bytes():
                frames += chunk.count(b'"i":')
        elapsed = time.monotonic() - started

        # Cut off after a few frames — not drained to all N...
        assert 0 < frames < N, f"expected an early abort, got {frames} frames"
        # ...and quickly, not after the full ~1s upstream stream...
        assert elapsed < 0.5, f"stream took {elapsed:.2f}s — upstream wasn't aborted"
        # ...with the engine's in-flight slot released (no leaked generation).
        assert all(v == 0 for v in mgr._inflight.values()), mgr._inflight


async def test_missing_model_field_is_400(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        r = await client.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "invalid_request_error"


async def test_admin_swap_by_engine_returns_status(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, mgr):
        r = await client.post("/admin/swap", json={"engine": "ollama"})
        assert r.status_code == 200
        st = r.json()
        assert st["active_engine"] == "ollama"
        assert mgr.active_engine == "ollama"


async def test_admin_swap_requires_model_or_engine(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        r = await client.post("/admin/swap", json={})
        assert r.status_code == 400


async def test_api_tags_passthrough(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        r = await client.get("/api/tags")
        assert r.status_code == 200
        names = {m["name"] for m in r.json()["models"]}
        assert "mock-ollama:latest" in names


# --------------------------------------------------------------------------- #
# Auth: when api_keys is set, requests need a key — /health and /metrics exempt.
# --------------------------------------------------------------------------- #
API_KEY = "test-secret-key-123"


async def test_auth_required_without_key(mock_upstream):
    cfg = _app_config(mock_upstream.base_url, api_keys=[API_KEY])
    async with _client_for(cfg) as (client, _):
        r = await client.get("/v1/models")
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"


async def test_auth_health_and_metrics_exempt(mock_upstream):
    cfg = _app_config(mock_upstream.base_url, api_keys=[API_KEY])
    async with _client_for(cfg) as (client, _):
        # No key, but these stay reachable.
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200


async def test_auth_accepts_bearer_key(mock_upstream):
    cfg = _app_config(mock_upstream.base_url, api_keys=[API_KEY])
    async with _client_for(cfg) as (client, _):
        r = await client.get(
            "/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}
        )
        assert r.status_code == 200


async def test_auth_accepts_x_api_key(mock_upstream):
    cfg = _app_config(mock_upstream.base_url, api_keys=[API_KEY])
    async with _client_for(cfg) as (client, _):
        r = await client.get("/v1/models", headers={"X-API-Key": API_KEY})
        assert r.status_code == 200


async def test_auth_rejects_wrong_key(mock_upstream):
    cfg = _app_config(mock_upstream.base_url, api_keys=[API_KEY])
    async with _client_for(cfg) as (client, _):
        r = await client.get(
            "/v1/models", headers={"Authorization": "Bearer wrong-key"}
        )
        assert r.status_code == 401


# --------------------------------------------------------------------------- #
# MM4: an alias routes to the real model AND the outgoing body's model is
# rewritten to the real id (the mock echoes body["model"], proving the rewrite).
# --------------------------------------------------------------------------- #
async def test_alias_routes_and_rewrites_body_model(mock_upstream):
    cfg = _app_config(mock_upstream.base_url, aliases={"gpt-4o": "deepseek-v4-flash"})
    async with _client_for(cfg) as (client, mgr):
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        # The upstream echoed the model it RECEIVED -> must be the real id, not
        # the alias, proving the body was rewritten before forwarding.
        assert r.json()["model"] == "deepseek-v4-flash"
        # And the alias routed to the real model's engine.
        assert mgr.active_engine == "ds4"


async def test_alias_rewrites_body_on_api_endpoint(mock_upstream):
    """The /api/* path (Ollama-native) rewrites the alias in the NDJSON body too."""
    cfg = _app_config(mock_upstream.base_url, aliases={"chat": "qwen2.5:3b"})
    async with _client_for(cfg) as (client, mgr):
        chunks: list[bytes] = []
        async with client.stream(
            "POST",
            "/api/chat",
            json={"model": "chat", "messages": [{"role": "user", "content": "y"}]},
        ) as resp:
            assert resp.status_code == 200
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
        joined = b"".join(chunks)
        # The mock /api/chat echoes the received model into each NDJSON line.
        assert b'"model":"qwen2.5:3b"' in joined
        assert b'"model":"chat"' not in joined
        assert mgr.active_engine == "ollama"


async def test_non_alias_request_body_unchanged(mock_upstream):
    """A non-aliased model id is forwarded unchanged (model echoed verbatim)."""
    cfg = _app_config(mock_upstream.base_url, aliases={"gpt-4o": "deepseek-v4-flash"})
    async with _client_for(cfg) as (client, _):
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4-flash", "messages": []},
        )
        assert r.status_code == 200
        assert r.json()["model"] == "deepseek-v4-flash"


# --------------------------------------------------------------------------- #
# Destructive Ollama management endpoints are refused by the catch-all unless
# allow_destructive_ollama_api is set (defense in depth: a firewall blocking
# direct Ollama access must not be bypassable through the router).
# --------------------------------------------------------------------------- #
async def test_destructive_api_blocked_by_default(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        for method, path in (
            ("DELETE", "/api/delete"),
            ("POST", "/api/create"),
            ("POST", "/api/copy"),
            ("POST", "/api/push"),
            ("POST", "/api/blobs/sha256:abc"),
        ):
            r = await client.request(method, path, json={"model": "x"})
            assert r.status_code == 403, path
            assert r.json()["error"]["code"] == "destructive_api_disabled"


async def test_destructive_api_allowed_when_enabled(mock_upstream):
    cfg = _app_config(mock_upstream.base_url)
    cfg.allow_destructive_ollama_api = True
    async with _client_for(cfg) as (client, _):
        # The mock upstream has no /api/delete route (404), but the request must
        # reach it rather than being refused by the router (403).
        r = await client.request("DELETE", "/api/delete", json={"model": "x"})
        assert r.status_code != 403


async def test_non_destructive_catchall_still_passes_through(mock_upstream):
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, _):
        r = await client.get("/api/some-future-endpoint")
        assert r.status_code != 403


# --------------------------------------------------------------------------- #
# Backward-compat: discover.enabled=False must NOT call available_models() on
# non-Ollama engines and must match the static+Ollama-only shape.
# --------------------------------------------------------------------------- #

async def test_v1_models_discover_off_does_not_call_non_ollama_available_models(mock_upstream):
    """With discover.enabled=False, /v1/models must not call available_models() on
    non-Ollama engines and must match the static config + Ollama-tags-only shape.

    This guards the hard invariant: when discovery is off the router behaves
    byte-identically to origin/main (static models + single OllamaEngine
    available_tags(); no other engine is queried).
    """
    calls: list[str] = []

    class _SpyEngine:
        """Records calls to available_models() so the test can assert it was not called."""

        def __init__(self, key: str) -> None:
            self.key = key
            self.base_url = f"http://spy-{key}.local"

        async def available_models(self) -> set[str]:
            calls.append(self.key)
            return {f"spy-model-from-{self.key}"}

        async def aclose(self) -> None:
            pass

    # discover.enabled defaults to False in _app_config.
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, manager):
        # Inject a non-Ollama spy alongside the real engines.
        manager.engines["spy"] = _SpyEngine("spy")
        manager._inflight["spy"] = 0

        r = await client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        ids = {m["id"] for m in data["data"]}

        # Static config entries must be present.
        assert "deepseek-v4-flash" in ids
        assert "qwen2.5:3b" in ids
        # Live Ollama tag from the mock /api/tags must be present.
        assert "mock-ollama:latest" in ids
        # The spy engine's model must NOT appear (discover is off).
        assert "spy-model-from-spy" not in ids
        # available_models() must NOT have been called on the non-Ollama spy.
        assert "spy" not in calls, (
            "available_models() was called on a non-Ollama engine with discover.enabled=False"
        )


# --------------------------------------------------------------------------- #
# Slice 4: /v1/models union from multiple engines' available_models()
# --------------------------------------------------------------------------- #

class _FakeDiscoverEngine:
    """Minimal fake engine that returns a fixed set from available_models().

    Defined locally so this file stays hermetic (does not edit conftest.py).
    Only the attributes and methods used by list_models / admin_discover are
    needed: key, base_url, available_models(), and aclose() so manager teardown
    works.
    """

    def __init__(self, key: str, models: set[str]) -> None:
        self.key = key
        self.base_url = f"http://fake-{key}.local"
        self._models = set(models)

    async def available_models(self) -> set[str]:
        return set(self._models)

    async def aclose(self) -> None:
        pass


async def test_v1_models_unions_multiple_engines(mock_upstream):
    """GET /v1/models with discover ON: static models first, then all engines' available_models()."""
    async with _client_for(_app_config_discover_on(mock_upstream.base_url)) as (client, manager):
        # Inject two fake engines alongside the real ones.
        manager.engines["fake-a"] = _FakeDiscoverEngine("fake-a", {"model-a1", "model-a2"})
        manager.engines["fake-b"] = _FakeDiscoverEngine("fake-b", {"model-b1"})
        manager._inflight["fake-a"] = 0
        manager._inflight["fake-b"] = 0

        r = await client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        ids = {m["id"] for m in data["data"]}

        # Static config entries must be present.
        assert "deepseek-v4-flash" in ids
        assert "qwen2.5:3b" in ids
        # The fake engines' discovered models must appear.
        assert "model-a1" in ids
        assert "model-a2" in ids
        assert "model-b1" in ids
        # Each discovered entry carries owned_by matching the engine key.
        for entry in data["data"]:
            if entry["id"] == "model-a1":
                assert entry["owned_by"] == "fake-a"
            if entry["id"] == "model-b1":
                assert entry["owned_by"] == "fake-b"


async def test_v1_models_deduplicates_across_engines(mock_upstream):
    """An id present in static config is NOT duplicated by an engine returning it (discover ON)."""
    async with _client_for(_app_config_discover_on(mock_upstream.base_url)) as (client, manager):
        # Return "deepseek-v4-flash" (already in static config) from a fake engine.
        manager.engines["fake-dup"] = _FakeDiscoverEngine(
            "fake-dup", {"deepseek-v4-flash", "extra-model"}
        )
        manager._inflight["fake-dup"] = 0

        r = await client.get("/v1/models")
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["data"]]
        # "deepseek-v4-flash" must appear exactly once.
        assert ids.count("deepseek-v4-flash") == 1
        # The non-duplicate still appears.
        assert "extra-model" in ids


async def test_v1_models_best_effort_skips_broken_engine(mock_upstream):
    """A single broken engine must not prevent the rest from being listed (discover ON)."""

    class _BrokenEngine:
        key = "broken"
        base_url = "http://broken.local"

        async def available_models(self) -> set[str]:
            raise RuntimeError("simulated failure")

        async def aclose(self) -> None:
            pass

    async with _client_for(_app_config_discover_on(mock_upstream.base_url)) as (client, manager):
        manager.engines["broken"] = _BrokenEngine()
        manager.engines["good"] = _FakeDiscoverEngine("good", {"good-model"})
        manager._inflight["broken"] = 0
        manager._inflight["good"] = 0

        r = await client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        # The good engine's model still appears.
        assert "good-model" in ids


async def test_v1_models_discovered_index_surfaced_when_present(mock_upstream):
    """When discover ON and manager has _discovered_index(), its stopped-engine models appear."""
    async with _client_for(_app_config_discover_on(mock_upstream.base_url)) as (client, manager):
        # Simulate the routing slice providing _discovered_index.
        manager._discovered_index = lambda: {"stopped-model": "some-engine"}

        r = await client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert "stopped-model" in ids

    # Restore: remove the injected method (no-op for other tests since each
    # test gets its own manager via _client_for).


# --------------------------------------------------------------------------- #
# Slice 4: POST /admin/discover
# --------------------------------------------------------------------------- #

async def test_admin_discover_returns_per_engine_summary(mock_upstream):
    """POST /admin/discover must return {"engines": {key: [model ids...]}}."""
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, manager):
        manager.engines["fake-x"] = _FakeDiscoverEngine("fake-x", {"mx1", "mx2"})
        manager._inflight["fake-x"] = 0

        r = await client.post("/admin/discover", json={})
        assert r.status_code == 200
        body = r.json()
        assert "engines" in body
        engines = body["engines"]
        # Our fake engine must be present with its models sorted.
        assert "fake-x" in engines
        assert sorted(engines["fake-x"]) == ["mx1", "mx2"]


async def test_admin_discover_best_effort_broken_engine(mock_upstream):
    """A broken engine must not cause /admin/discover to 500; its entry is []."""

    class _BrokenEngine:
        key = "broken-disc"
        base_url = "http://broken-disc.local"

        async def available_models(self) -> set[str]:
            raise OSError("boom")

        async def aclose(self) -> None:
            pass

    async with _client_for(_app_config(mock_upstream.base_url)) as (client, manager):
        manager.engines["broken-disc"] = _BrokenEngine()
        manager.engines["good-disc"] = _FakeDiscoverEngine("good-disc", {"gd1"})
        manager._inflight["broken-disc"] = 0
        manager._inflight["good-disc"] = 0

        r = await client.post("/admin/discover", json={})
        assert r.status_code == 200
        body = r.json()
        # Broken engine yields an empty list, not an error.
        assert body["engines"].get("broken-disc") == []
        # Good engine still appears correctly.
        assert body["engines"].get("good-disc") == ["gd1"]


async def test_admin_discover_merges_discovered_index(mock_upstream):
    """When _discovered_index is present, its entries appear in /admin/discover."""
    async with _client_for(_app_config(mock_upstream.base_url)) as (client, manager):
        manager._discovered_index = lambda: {"offline-model": "offline-engine"}

        r = await client.post("/admin/discover", json={})
        assert r.status_code == 200
        body = r.json()
        engines = body["engines"]
        assert "offline-model" in engines.get("offline-engine", [])


async def test_admin_discover_gated_by_auth(mock_upstream):
    """POST /admin/discover must require an API key when auth is enabled."""
    cfg = _app_config(mock_upstream.base_url, api_keys=[API_KEY])
    async with _client_for(cfg) as (client, _):
        # No key -> 401.
        r = await client.post("/admin/discover", json={})
        assert r.status_code == 401

        # Correct key -> 200.
        r = await client.post(
            "/admin/discover",
            json={},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Slice 4: routerctl discover subcommand
# --------------------------------------------------------------------------- #

class _FakeDiscoverResponse:
    """Minimal urllib-compatible response for /admin/discover."""

    def __init__(self, body: dict) -> None:
        self._data = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


_DISCOVER_PAYLOAD = {
    "engines": {
        "ds4": ["deepseek-v4-flash"],
        "ollama": ["qwen3:3b", "llama3:8b"],
    }
}


class TestCmdDiscover:
    def test_parser_recognises_discover(self):
        args = cli.build_parser().parse_args(["discover"])
        assert args.command == "discover"

    def test_cmd_discover_posts_to_admin_discover(self, monkeypatch):
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return _FakeDiscoverResponse(_DISCOVER_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        cli.cmd_discover(argparse.Namespace())

        assert len(captured) == 1
        assert captured[0].full_url.endswith("/admin/discover")
        assert captured[0].method == "POST"

    def test_cmd_discover_prints_engine_headers(self, monkeypatch, capsys):
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda req, timeout=None: _FakeDiscoverResponse(_DISCOVER_PAYLOAD),
        )
        cli.cmd_discover(argparse.Namespace())
        out = capsys.readouterr().out
        assert "[ds4]" in out
        assert "[ollama]" in out

    def test_cmd_discover_prints_model_ids(self, monkeypatch, capsys):
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda req, timeout=None: _FakeDiscoverResponse(_DISCOVER_PAYLOAD),
        )
        cli.cmd_discover(argparse.Namespace())
        out = capsys.readouterr().out
        assert "deepseek-v4-flash" in out
        assert "qwen3:3b" in out
        assert "llama3:8b" in out

    def test_main_discover_dispatches(self, monkeypatch, capsys):
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda req, timeout=None: _FakeDiscoverResponse(_DISCOVER_PAYLOAD),
        )
        import sys
        monkeypatch.setattr(sys, "argv", ["routerctl", "discover"])
        cli.main()
        out = capsys.readouterr().out
        assert "[ds4]" in out

    def test_cmd_discover_empty_result(self, monkeypatch, capsys):
        """An empty engines dict must produce a sensible message, not a crash."""
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda req, timeout=None: _FakeDiscoverResponse({"engines": {}}),
        )
        cli.cmd_discover(argparse.Namespace())
        out = capsys.readouterr().out
        assert "none" in out.lower() or out.strip() != ""
