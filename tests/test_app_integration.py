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

import contextlib

import httpx

from router import metrics
from router.app import create_app
from router.config import Ds4Config, ModelSpec, OllamaConfig, RouterConfig


def _app_config(mock_base: str, *, api_keys=None, aliases=None) -> RouterConfig:
    """Config whose ds4 + ollama both point at the mock upstream's base_url.

    The legacy ds4:/ollama: sections build a Ds4Engine + OllamaEngine; keeping
    the ollama key as an OllamaEngine is what makes /v1/models tag enrichment
    and the /api/* passthrough work (the app resolves Ollama by type).
    """
    return RouterConfig(
        host="127.0.0.1",
        port=8077,
        api_keys=list(api_keys or []),
        aliases=dict(aliases or {}),
        state_file="/tmp/llm-router-test-state.json",
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
