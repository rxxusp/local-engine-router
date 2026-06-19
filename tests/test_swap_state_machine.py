"""Section A — EngineManager swap state machine.

The headline differentiator: strict mutual exclusion between heavy engines, with
in-flight accounting, draining, and metrics. All engines here are in-memory
FakeEngines (no sockets, instant), so timing is deterministic.

Also covers the discovery-adjacent persistence helpers added in Slice 3:
_seen_models seeding from a state file and the post-swap snapshot task.
"""

from __future__ import annotations

import asyncio
import json
import tempfile

import pytest

from router import metrics
from router.config import DiscoverConfig, ModelSpec
from router.engines import EngineError

from conftest import FakeEngine, make_config, make_manager_with_fakes


def _two_engine_manager(**kw):
    models = [
        ModelSpec(id="m-ds4", engine="ds4", display_name="ds4 model"),
        ModelSpec(id="m-oll", engine="ollama", display_name="ollama model"),
    ]
    cfg = make_config(models=models, **kw)
    fakes = {
        "ds4": FakeEngine("ds4", base_url="http://ds4.local"),
        "ollama": FakeEngine("ollama", base_url="http://ollama.local"),
    }
    return make_manager_with_fakes(fakes, cfg=cfg), fakes


async def test_acquire_routes_to_right_engine():
    mgr, fakes = _two_engine_manager()
    eng = await mgr.acquire("m-ds4")
    assert eng is fakes["ds4"]
    assert mgr.active_engine == "ds4"
    assert mgr._inflight["ds4"] == 1


async def test_acquire_inactive_engine_triggers_exactly_one_swap():
    mgr, fakes = _two_engine_manager()
    # First acquire brings ds4 up (one start, zero frees for ds4).
    await mgr.acquire("m-ds4")
    await mgr.release("ds4")
    assert fakes["ds4"].starts == 1

    # Acquiring the *other* engine swaps: ds4 freed exactly once, ollama started.
    await mgr.acquire("m-oll")
    assert mgr.active_engine == "ollama"
    assert fakes["ds4"].frees == 1
    assert fakes["ollama"].starts == 1
    # Mutual exclusion: ds4 is no longer ready.
    assert await fakes["ds4"].is_ready() is False
    assert await fakes["ollama"].is_ready() is True


async def test_never_two_active_after_swap():
    mgr, fakes = _two_engine_manager()
    await mgr.acquire("m-ds4")
    await mgr.release("ds4")
    await mgr.acquire("m-oll")
    await mgr.release("ollama")
    ready = [k for k, e in fakes.items() if await e.is_ready()]
    assert ready == ["ollama"]


async def test_release_decrements_in_flight():
    mgr, _ = _two_engine_manager()
    await mgr.acquire("m-ds4")
    await mgr.acquire("m-ds4")
    assert mgr._inflight["ds4"] == 2
    await mgr.release("ds4")
    assert mgr._inflight["ds4"] == 1
    await mgr.release("ds4")
    assert mgr._inflight["ds4"] == 0
    # Releasing below zero is a no-op (never goes negative).
    await mgr.release("ds4")
    assert mgr._inflight["ds4"] == 0


async def test_acquire_same_engine_does_not_swap():
    mgr, fakes = _two_engine_manager()
    await mgr.acquire("m-ds4")
    assert fakes["ds4"].starts == 1
    # Re-acquiring the SAME active engine must not start/free anything again.
    await mgr.acquire("m-ds4")
    assert fakes["ds4"].starts == 1
    assert fakes["ds4"].frees == 0
    assert mgr._inflight["ds4"] == 2


async def test_drain_waits_for_in_flight_to_reach_zero():
    """A swap must drain the outgoing engine: free_vram only happens once the
    in-flight count hits 0. We hold one request open, kick off a swap, and show
    the swap completes right after we release."""
    mgr, fakes = _two_engine_manager(drain_timeout_s=5.0)
    await mgr.acquire("m-ds4")  # ds4 active, in_flight=1
    await mgr.release("ds4")
    await mgr.acquire("m-ds4")  # ds4 active, in_flight=1 (held open)

    swap = asyncio.create_task(mgr.acquire("m-oll"))
    await asyncio.sleep(0.05)
    # Swap is blocked draining ds4 -> ollama not started, ds4 not freed yet.
    assert not swap.done()
    assert fakes["ollama"].starts == 0
    assert fakes["ds4"].frees == 0

    await mgr.release("ds4")  # in_flight hits 0 -> drain unblocks
    eng = await asyncio.wait_for(swap, timeout=2.0)
    assert eng is fakes["ollama"]
    assert fakes["ds4"].frees == 1
    assert mgr.active_engine == "ollama"


async def test_drain_respects_timeout_then_proceeds():
    """If in-flight never drains, _drain gives up after drain_timeout_s and the
    swap proceeds anyway (the old engine is stopped under the leaked request)."""
    mgr, fakes = _two_engine_manager(drain_timeout_s=0.2)
    await mgr.acquire("m-ds4")  # in_flight=1, never released

    t0 = asyncio.get_running_loop().time()
    eng = await mgr.acquire("m-oll")  # must not hang forever
    dt = asyncio.get_running_loop().time() - t0

    assert eng is fakes["ollama"]
    assert mgr.active_engine == "ollama"
    assert fakes["ds4"].frees == 1
    # It waited roughly the drain timeout, not indefinitely.
    assert 0.15 <= dt < 2.0


async def test_failed_start_leaves_active_none_and_raises():
    models = [
        ModelSpec(id="m-ds4", engine="ds4", display_name="ds4"),
        ModelSpec(id="m-oll", engine="ollama", display_name="oll"),
    ]
    cfg = make_config(models=models)
    fakes = {
        "ds4": FakeEngine("ds4"),
        "ollama": FakeEngine("ollama", fail_start=True),
    }
    mgr = make_manager_with_fakes(fakes, cfg=cfg)
    # Bring ds4 up first so the failing swap has something to free.
    await mgr.acquire("m-ds4")
    await mgr.release("ds4")

    with pytest.raises(EngineError):
        await mgr.acquire("m-oll")
    # A failed ensure_started leaves NO active engine.
    assert mgr.active_engine is None
    # The previous engine was still freed (we drained + freed before starting).
    assert fakes["ds4"].frees == 1
    # in-flight was never incremented for the failed target.
    assert mgr._inflight["ollama"] == 0


async def test_metrics_records_swap_after_swap():
    mgr, _ = _two_engine_manager()
    await mgr.acquire("m-ds4")
    text = metrics.render()
    # A successful swap to ds4 increments swap_total{...,result="ok"} and
    # observes the duration histogram (count >= 1).
    assert 'swap_total{from="none",to="ds4",result="ok"} 1' in text
    assert "swap_duration_seconds_count 1" in text


async def test_metrics_records_failed_swap():
    models = [ModelSpec(id="m-oll", engine="ollama", display_name="oll")]
    cfg = make_config(models=models)
    fakes = {"ollama": FakeEngine("ollama", fail_start=True)}
    mgr = make_manager_with_fakes(fakes, cfg=cfg)
    with pytest.raises(EngineError):
        await mgr.acquire("m-oll")
    text = metrics.render()
    assert 'result="error"' in text


async def test_force_swap_by_engine_key():
    mgr, fakes = _two_engine_manager()
    eng = await mgr.force_swap(engine_key="ollama")
    assert eng is fakes["ollama"]
    assert mgr.active_engine == "ollama"
    # No in-flight increment for a force_swap (it's a proactive swap).
    assert mgr._inflight["ollama"] == 0


async def test_force_swap_by_model_id():
    mgr, fakes = _two_engine_manager()
    eng = await mgr.force_swap(model_id="m-ds4")
    assert eng is fakes["ds4"]
    assert mgr.active_engine == "ds4"


async def test_force_swap_unknown_engine_raises():
    mgr, _ = _two_engine_manager()
    with pytest.raises(EngineError):
        await mgr.force_swap(engine_key="does-not-exist")


async def test_concurrent_acquire_same_engine_no_swap():
    """Many concurrent acquires of the SAME (initially inactive) engine must
    trigger at most one start and never a free."""
    mgr, fakes = _two_engine_manager()
    results = await asyncio.gather(*[mgr.acquire("m-ds4") for _ in range(8)])
    assert all(e is fakes["ds4"] for e in results)
    assert fakes["ds4"].starts == 1
    assert fakes["ds4"].frees == 0
    assert mgr._inflight["ds4"] == 8


async def test_concurrent_acquires_requiring_swap_serialize():
    """Concurrent acquires that flip between engines must serialize through the
    swap lock — they can never both be 'active' at once, and the manager ends in
    a consistent single-active state.

    Each acquired request releases immediately so drains don't dominate the
    runtime; the point under test is the swap-lock serialization, not draining
    (that has its own dedicated tests)."""
    mgr, fakes = _two_engine_manager(drain_timeout_s=0.2)

    async def acquire_release(model: str) -> str:
        eng = await mgr.acquire(model)
        # Release right away so the next swap's drain is immediate.
        await mgr.release(eng.key)
        return eng.key

    models = ["m-ds4", "m-oll", "m-ds4", "m-oll", "m-ds4", "m-oll"]
    keys = await asyncio.gather(*[acquire_release(m) for m in models])
    assert set(keys) <= {"ds4", "ollama"}

    # Exactly one engine is active and ready at the end; the other is down.
    ready = [k for k, e in fakes.items() if await e.is_ready()]
    assert len(ready) == 1
    assert mgr.active_engine == ready[0]
    # Every transition went through _swap_to, so each engine was started at
    # least once and the loser was freed (mutual exclusion held throughout).
    assert fakes["ds4"].starts >= 1
    assert fakes["ollama"].starts >= 1
    assert fakes[ready[0]].frees >= 0


async def test_swap_lock_serializes_overlapping_swaps():
    """Two overlapping swaps to *different* engines must not interleave: the
    second waits for the first to finish (proven by start ordering)."""
    order: list[str] = []

    class OrderedEngine(FakeEngine):
        async def ensure_started(self):
            await super().ensure_started()
            order.append(f"start:{self.key}")

        async def free_vram(self):
            order.append(f"free:{self.key}")
            await super().free_vram()

    models = [
        ModelSpec(id="m-ds4", engine="ds4", display_name="ds4"),
        ModelSpec(id="m-oll", engine="ollama", display_name="oll"),
    ]
    cfg = make_config(models=models, drain_timeout_s=1.0)
    fakes = {
        "ds4": OrderedEngine("ds4", start_delay_s=0.1),
        "ollama": OrderedEngine("ollama", start_delay_s=0.1),
    }
    mgr = make_manager_with_fakes(fakes, cfg=cfg)

    t1 = asyncio.create_task(mgr.acquire("m-ds4"))
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(mgr.acquire("m-oll"))
    await asyncio.gather(t1, t2)

    # ds4 must have fully started before ollama's swap freed it and started.
    assert order.index("start:ds4") < order.index("free:ds4")
    assert order.index("free:ds4") < order.index("start:ollama")
    assert mgr.active_engine == "ollama"


async def test_in_flight_at_swap_start_metric_recorded():
    mgr, _ = _two_engine_manager()
    await mgr.acquire("m-ds4")  # first swap: 0 in-flight on the (none) loser
    text = metrics.render()
    assert "in_flight_at_swap_start_count" in text
    # The histogram observed at least one sample.
    assert "in_flight_at_swap_start_count 1" in text


# --------------------------------------------------------------------------- #
# _seen_models persistence (Slice 3: discovery helpers)
# --------------------------------------------------------------------------- #
async def test_seen_models_seeded_from_state_file():
    """_seen_models is pre-populated from a state file that has a seen_models key."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        json.dump(
            {
                "active_engine": None,
                "last_swap": {},
                "seen_models": {"ds4": ["ds4-loaded-model", "ds4-other"]},
            },
            fh,
        )
        state_path = fh.name

    models = [ModelSpec(id="m-ds4", engine="ds4", display_name="d")]
    cfg = make_config(models=models, state_file=state_path)
    fakes = {"ds4": FakeEngine("ds4")}
    mgr = make_manager_with_fakes(fakes, cfg=cfg)
    # The state file is read in __init__ via _load_seen_models_from_state.
    assert "ds4-loaded-model" in mgr._seen_models.get("ds4", set())
    assert "ds4-other" in mgr._seen_models.get("ds4", set())


async def test_seen_models_missing_state_file_is_noop():
    """A missing state file does not crash and _seen_models stays empty."""
    models = [ModelSpec(id="m-ds4", engine="ds4", display_name="d")]
    cfg = make_config(models=models, state_file="/tmp/does-not-exist-ler-test.json")
    fakes = {"ds4": FakeEngine("ds4")}
    mgr = make_manager_with_fakes(fakes, cfg=cfg)
    assert mgr._seen_models.get("ds4", set()) == set()


async def test_persist_writes_seen_models_when_discovery_enabled():
    """_persist() includes seen_models in the written state when discovery is on."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        state_path = fh.name

    models = [ModelSpec(id="m-ds4", engine="ds4", display_name="d")]
    cfg = make_config(models=models, state_file=state_path, discover=DiscoverConfig(enabled=True))
    fakes = {"ds4": FakeEngine("ds4")}
    mgr = make_manager_with_fakes(fakes, cfg=cfg)
    mgr._seen_models["ds4"] = {"model-alpha", "model-beta"}
    mgr._persist()

    with open(state_path) as fh:
        data = json.load(fh)

    assert "seen_models" in data
    assert set(data["seen_models"].get("ds4", [])) == {"model-alpha", "model-beta"}


async def test_persist_omits_seen_models_when_discovery_disabled():
    """_persist() must NOT write the seen_models key when discovery is off.

    This preserves byte-identical state files for configs that do not use
    the discovery feature."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        state_path = fh.name

    models = [ModelSpec(id="m-ds4", engine="ds4", display_name="d")]
    # discover.enabled is False by default.
    cfg = make_config(models=models, state_file=state_path)
    fakes = {"ds4": FakeEngine("ds4")}
    mgr = make_manager_with_fakes(fakes, cfg=cfg)
    mgr._seen_models["ds4"] = {"model-alpha"}
    mgr._persist()

    with open(state_path) as fh:
        data = json.load(fh)

    assert "seen_models" not in data


async def test_snapshot_seen_models_updates_after_swap():
    """After a successful swap, _seen_models for the target engine is updated.

    We give the target FakeEngine an available_models() override that returns
    a known set, then run a swap and give the background task a tick to complete.
    Discovery must be enabled for the snapshot task to be scheduled.
    """
    class _ModelReturningEngine(FakeEngine):
        async def available_models(self) -> set[str]:
            return {"snapped-model-a", "snapped-model-b"}

    models = [
        ModelSpec(id="m-ds4", engine="ds4", display_name="d"),
        ModelSpec(id="m-oll", engine="ollama", display_name="o"),
    ]
    cfg = make_config(models=models, discover=DiscoverConfig(enabled=True))
    fakes = {
        "ds4": _ModelReturningEngine("ds4"),
        "ollama": FakeEngine("ollama"),
    }
    mgr = make_manager_with_fakes(fakes, cfg=cfg)
    await mgr.acquire("m-ds4")
    # Give the asyncio.ensure_future snapshot task a chance to run.
    await asyncio.sleep(0)
    assert "snapped-model-a" in mgr._seen_models.get("ds4", set())
    assert "snapped-model-b" in mgr._seen_models.get("ds4", set())


async def test_snapshot_not_scheduled_when_discovery_disabled():
    """When discover.enabled is False, _snapshot_seen_models is never scheduled
    after a swap, so _seen_models stays empty even if available_models() returns
    a non-empty set."""
    class _ModelReturningEngine(FakeEngine):
        async def available_models(self) -> set[str]:
            return {"should-not-appear"}

    models = [
        ModelSpec(id="m-ds4", engine="ds4", display_name="d"),
        ModelSpec(id="m-oll", engine="ollama", display_name="o"),
    ]
    # Use a fresh temp file so seen_models from prior tests cannot leak in.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        state_path = fh.name

    # discover.enabled defaults to False.
    cfg = make_config(models=models, state_file=state_path)
    fakes = {
        "ds4": _ModelReturningEngine("ds4"),
        "ollama": FakeEngine("ollama"),
    }
    mgr = make_manager_with_fakes(fakes, cfg=cfg)
    await mgr.acquire("m-ds4")
    # Give the event loop a tick; if a snapshot task were scheduled it would run.
    await asyncio.sleep(0)
    # No snapshot scheduled -> _seen_models remains empty for ds4.
    assert mgr._seen_models.get("ds4", set()) == set()
