"""Regression coverage for routing, streaming, and interrupted engine swaps."""

import asyncio
import gzip

import httpx
import pytest

from conftest import FakeEngine, FakeAPISwapEngine, make_config, make_manager_with_fakes
from router.app import create_app
from router.config import ApiSwapConfig, DiscoverConfig, ModelSpec, OllamaConfig
from router.engines import APISwapEngine, EngineError, OllamaEngine
from router.catalog import CatalogEntry
from router.proxy import filter_request_headers, filter_response_headers


def manager_for(tmp_path, **kwargs):
    cfg = make_config(
        models=[ModelSpec(id="a", engine="a", display_name="A"),
                ModelSpec(id="b", engine="b", display_name="B")],
        state_file=str(tmp_path / "state.json"), **kwargs,
    )
    return make_manager_with_fakes({"a": FakeEngine("a"), "b": FakeEngine("b")}, cfg=cfg)


@pytest.mark.parametrize("phase", ["free", "settle", "start"])
async def test_cancelled_swap_invalidates_active_engine(tmp_path, phase):
    mgr = manager_for(tmp_path)
    await mgr.force_swap(engine_key="a")
    entered = asyncio.Event()

    async def block(*args):
        entered.set()
        await asyncio.Event().wait()

    if phase == "free":
        original = mgr.engines["a"].free_vram

        async def free():
            await original()
            await block()

        mgr.engines["a"].free_vram = free
    elif phase == "settle":
        mgr._await_memory_settle = block
    else:
        mgr.engines["b"].ensure_started = block

    pending = asyncio.create_task(mgr.acquire("b"))
    await asyncio.wait_for(entered.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert mgr.active_engine is None
    assert mgr._last_swap["ok"] is False
    assert mgr._inflight == {"a": 0, "b": 0}
    await mgr.aclose()


@pytest.mark.parametrize("admin", [False, True])
async def test_explicit_model_load_waits_for_existing_request(tmp_path, admin):
    mgr = manager_for(tmp_path, drain_timeout_s=5)
    engine = APISwapEngine(ApiSwapConfig(base_url="http://fake.local", load_path="/load"), key="a")
    mgr.engines["a"] = engine
    mgr.active_engine = "a"
    mgr._inflight["a"] = 1
    probed = asyncio.Event()
    loaded = []

    async def names():
        probed.set()
        return ["old-model"]

    async def load(model):
        loaded.append(model)

    engine.loaded_model_names = names
    engine.load_model = load
    operation = mgr.force_swap(model_id="a") if admin else mgr.acquire("a")
    pending = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(probed.wait(), 1)
        assert not loaded
        assert not pending.done()
        await mgr.release("a")
        await asyncio.wait_for(pending, 1)
        assert loaded == ["a"]
        assert mgr._inflight["a"] == (0 if admin else 1)
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await mgr.aclose()


async def test_catalog_probes_run_concurrently(tmp_path):
    mgr = manager_for(tmp_path, discover=DiscoverConfig(enabled=True))
    mgr.catalog.engines = mgr.engines
    entered = set()
    all_entered = asyncio.Event()

    def probe(key):
        async def available():
            entered.add(key)
            if len(entered) == 2:
                all_entered.set()
            await asyncio.wait_for(all_entered.wait(), 1)
            return {f"live-{key}"}
        return available

    for key, engine in mgr.engines.items():
        engine.available_models = probe(key)
    try:
        await mgr.refresh_catalog()
        assert mgr.catalog.owner_for("live-a").engine == "a"
        assert mgr.catalog.owner_for("live-b").engine == "b"
    finally:
        await mgr.aclose()


async def test_catalog_owner_matches_smart_and_actual_route(tmp_path):
    mgr = manager_for(tmp_path, discover=DiscoverConfig(enabled=True))
    mgr.engines["b"] = FakeAPISwapEngine("b", tags={"shared"})
    mgr.catalog.entries["shared"] = CatalogEntry(
        id="shared", engine="a", source="served_models", first_seen=1, last_seen=1,
    )
    try:
        assert (await mgr.explain_model("shared"))["engine"] == "a"
        assert (await mgr.smart._enumerate_candidates())["shared"] == "a"
        assert (await mgr.engine_for("shared")).key == "a"
    finally:
        await mgr.aclose()


@pytest.mark.parametrize("kind", ["api_swap", "ollama"])
async def test_unload_timeout_does_not_start_next_engine(tmp_path, kind):
    mgr = manager_for(tmp_path)
    if kind == "ollama":
        engine = OllamaEngine(OllamaConfig(unload_timeout_s=0), key="a")
    else:
        engine = APISwapEngine(ApiSwapConfig(
            base_url="http://fake.local", loaded_path="/loaded", unload_timeout_s=0,
        ), key="a")

    async def loaded():
        return ["stuck-model"]

    async def unload(name):
        pass

    engine.loaded_models = loaded
    engine._unload = unload
    mgr.engines["a"] = engine
    mgr.active_engine = "a"
    try:
        with pytest.raises(EngineError, match="still loaded"):
            await mgr.acquire("b")
        assert mgr.engines["b"].starts == 0
        assert mgr._inflight["b"] == 0
    finally:
        await mgr.aclose()


async def test_status_does_not_block_event_loop_on_process_probe(tmp_path, monkeypatch):
    import threading
    from router.engines import Ds4Engine
    from router.config import Ds4Config

    mgr = manager_for(tmp_path)
    engine = Ds4Engine(Ds4Config(), key="a")
    mgr.engines["a"] = engine
    probed = threading.Event()
    release = threading.Event()

    async def ready():
        return True

    def running():
        probed.set()
        return release.wait(2)

    monkeypatch.setattr(engine, "is_ready", ready)
    monkeypatch.setattr(engine, "is_running", running)
    pending = asyncio.create_task(mgr.status())
    try:
        for _ in range(100):
            if probed.is_set():
                break
            await asyncio.sleep(.005)
        assert probed.is_set()
        assert not pending.done()
        release.set()
        result = await asyncio.wait_for(pending, 1)
        assert result["engines"]["a"]["process_running"] is True
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await mgr.aclose()


@pytest.mark.parametrize("path", ["/api/embed", "/api/embeddings"])
async def test_ollama_embeddings_default_to_json(tmp_path, path):
    mgr = manager_for(tmp_path)
    app = create_app(mgr.cfg)
    app.state.manager = mgr
    transport = httpx.MockTransport(lambda req: httpx.Response(422, json={"error": "invalid input"}))
    async with httpx.AsyncClient(transport=transport) as upstream:
        app.state.client = upstream
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://router") as client:
            response = await client.post(path, json={"model": "a", "input": "hello"})
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"error": "invalid input"}
    await mgr.aclose()


async def test_shutdown_finishes_snapshot_tasks_before_closing_engines(tmp_path):
    mgr = manager_for(tmp_path)
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def snapshot():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert not mgr.engines["a"].closed
            cleaned.set()

    task = asyncio.create_task(snapshot())
    mgr._bg_tasks.add(task)
    await entered.wait()
    await mgr.aclose()
    assert task.done()
    assert cleaned.is_set()


@pytest.mark.parametrize("filter_headers", [filter_request_headers, filter_response_headers])
def test_connection_nominated_headers_are_removed(filter_headers):
    assert filter_headers({
        "Connection": "keep-alive, X-Internal, x-debug",
        "X-Internal": "private", "X-Debug": "private", "X-Request-ID": "keep",
    }) == {"X-Request-ID": "keep"}


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
@pytest.mark.parametrize("model", [["a"], {"id": "a"}, 42, True, "   "])
async def test_invalid_model_is_400_before_acquiring(tmp_path, path, model):
    mgr = manager_for(tmp_path)
    app = create_app(mgr.cfg)
    app.state.manager = mgr
    async with httpx.AsyncClient() as upstream:
        app.state.client = upstream
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://router") as client:
            response = await client.post(path, json={"model": model})
    assert response.status_code == 400
    assert mgr.active_engine is None
    await mgr.aclose()


class CompressedStream(httpx.AsyncByteStream):
    def __init__(self, payload):
        self.payload = gzip.compress(payload)

    async def __aiter__(self):
        yield self.payload


@pytest.mark.parametrize("path,payload", [
    ("/v1/chat/completions", b'data: {"choices":[]}\n\ndata: [DONE]\n\n'),
    ("/api/chat", b'{"message":{"content":"hello"},"done":true}\n'),
])
async def test_compressed_upstream_stream_is_decoded(tmp_path, path, payload):
    mgr = manager_for(tmp_path)
    app = create_app(mgr.cfg)
    app.state.manager = mgr
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, headers={"content-encoding": "gzip"}, stream=CompressedStream(payload),
    ))
    async with httpx.AsyncClient(transport=transport) as upstream:
        app.state.client = upstream
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://router") as client:
            response = await client.post(path, json={"model": "a", "stream": True})
    assert response.content == payload
    assert mgr._inflight["a"] == 0
    await mgr.aclose()
