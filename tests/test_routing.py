"""Section B — EngineManager.engine_for routing resolution.

Covers: the static registry; ds4's fixed ids; the live Ollama-tag fallback for
ids pulled after startup; single-engine fallback; and the generic ``engines:``
table (generic_process + api_swap) resolving by type.
"""

from __future__ import annotations

import pytest

from router.config import (
    ApiSwapConfig,
    Ds4Config,
    EngineSpec,
    GenericProcessConfig,
    ModelSpec,
    RouterConfig,
)
from router.engines import (
    APISwapEngine,
    Ds4Engine,
    EngineError,
    GenericProcessEngine,
)

from conftest import (
    FakeAPISwapEngine,
    FakeEngine,
    make_config,
    make_manager_with_fakes,
)


async def test_static_registry_resolves():
    models = [
        ModelSpec(id="m-ds4", engine="ds4", display_name="d"),
        ModelSpec(id="m-oll", engine="ollama", display_name="o"),
    ]
    mgr = make_manager_with_fakes(
        {"ds4": FakeEngine("ds4"), "ollama": FakeAPISwapEngine("ollama")},
        cfg=make_config(models=models),
    )
    assert (await mgr.engine_for("m-ds4")).key == "ds4"
    assert (await mgr.engine_for("m-oll")).key == "ollama"


async def test_missing_model_raises():
    mgr = make_manager_with_fakes({"ds4": FakeEngine("ds4")})
    with pytest.raises(EngineError, match="missing a 'model'"):
        await mgr.engine_for(None)


async def test_static_ref_to_disabled_engine_raises():
    """A model whose engine key isn't in the (built) engine table errors."""
    models = [ModelSpec(id="m-x", engine="ghost", display_name="x")]
    mgr = make_manager_with_fakes(
        {"ds4": FakeEngine("ds4")}, cfg=make_config(models=models)
    )
    with pytest.raises(EngineError, match="disabled"):
        await mgr.engine_for("m-x")


async def test_live_ollama_tag_fallback():
    """An id not in the static registry but present in an APISwapEngine's live
    tags routes to that engine."""
    mgr = make_manager_with_fakes(
        {
            "ds4": FakeEngine("ds4"),
            "ollama": FakeAPISwapEngine("ollama", tags={"pulled-later:latest"}),
        }
    )
    eng = await mgr.engine_for("pulled-later:latest")
    assert eng.key == "ollama"


async def test_ds4_fixed_id_when_not_in_index():
    """A process engine advertises a small fixed set; an id declared for it in
    cfg.models (but not built into the index) still routes there.

    The fixed-id fallback branch in engine_for() guards on
    ``isinstance(engine, (Ds4Engine, GenericProcessEngine))``, so this needs a
    real process-type engine (a Ds4Engine). engine_for() is pure routing and
    never touches its lifecycle, so the real engine is safe + instant here."""
    models = [ModelSpec(id="ds4-only", engine="ds4", display_name="d")]
    cfg = make_config(models=models)
    ds4 = Ds4Engine(Ds4Config(base_url="http://ds4.local"), key="ds4")
    mgr = make_manager_with_fakes(
        {"ds4": ds4, "ollama": FakeAPISwapEngine("ollama")}, cfg=cfg
    )
    try:
        # Wipe the static index so we exercise the *fixed-id* fallback branch,
        # not the static-registry branch. cfg.models still lists ds4-only -> ds4.
        mgr.index = {}
        eng = await mgr.engine_for("ds4-only")
        assert eng.key == "ds4"
    finally:
        await ds4.aclose()


async def test_single_engine_fallback_for_unknown_id():
    """If only one engine is enabled, an unknown id falls back to it."""
    mgr = make_manager_with_fakes({"solo": FakeEngine("solo")})
    eng = await mgr.engine_for("anything-at-all")
    assert eng.key == "solo"


async def test_unknown_id_prefers_apiswap_engine():
    """With multiple engines and an unknown id (not in any tag set), the manager
    defaults to an APISwapEngine (it can pull/serve arbitrary tags)."""
    mgr = make_manager_with_fakes(
        {
            "ds4": FakeEngine("ds4"),
            "ollama": FakeAPISwapEngine("ollama", tags=set()),
        }
    )
    eng = await mgr.engine_for("totally-unknown:1b")
    assert eng.key == "ollama"
    assert isinstance(eng, APISwapEngine)


async def test_unknown_id_no_apiswap_raises():
    """Two process-only engines + an unknown id => no engine can serve it."""
    models = [
        ModelSpec(id="a", engine="e1", display_name="a"),
        ModelSpec(id="b", engine="e2", display_name="b"),
    ]
    mgr = make_manager_with_fakes(
        {"e1": FakeEngine("e1"), "e2": FakeEngine("e2")},
        cfg=make_config(models=models),
    )
    with pytest.raises(EngineError, match="no engine can serve"):
        await mgr.engine_for("ghost-model")


# --------------------------------------------------------------------------- #
# Generic engines: table resolves real GenericProcessEngine / APISwapEngine
# --------------------------------------------------------------------------- #
def _generic_table_config() -> RouterConfig:
    """A RouterConfig using the generic engines: table (no legacy ds4/ollama)."""
    llamacpp = EngineSpec(
        key="llamacpp",
        type="generic_process",
        params=GenericProcessConfig(
            base_url="http://127.0.0.1:18080",
            start_cmd=["/bin/true"],
            ready_path="/health",
        ),
    )
    tabby = EngineSpec(
        key="tabby",
        type="api_swap",
        params=ApiSwapConfig(
            base_url="http://127.0.0.1:15000",
            unload_path="/v1/model/unload",
            loaded_path="/v1/model",
        ),
    )
    return RouterConfig(
        engines=[llamacpp, tabby],
        models=[
            ModelSpec(id="local-gguf", engine="llamacpp", display_name="gguf"),
            ModelSpec(id="tabby-exl2", engine="tabby", display_name="exl2"),
        ],
        state_file="/tmp/llm-router-test-state.json",
        drain_timeout_s=0.5,
    )


async def test_generic_table_builds_correct_engine_types():
    """build_engines via the generic table must yield real engine classes."""
    from router.engines import EngineManager

    mgr = EngineManager(_generic_table_config())
    try:
        assert isinstance(mgr.engines["llamacpp"], GenericProcessEngine)
        assert isinstance(mgr.engines["tabby"], APISwapEngine)
        # And they resolve by their static model ids.
        assert (await mgr.engine_for("local-gguf")).key == "llamacpp"
        assert (await mgr.engine_for("tabby-exl2")).key == "tabby"
    finally:
        await mgr.aclose()


async def test_generic_api_swap_is_apiswap_subclass_for_fallback():
    """The generic api_swap engine participates in the live-tag fallback path
    because it is an APISwapEngine."""
    from router.engines import EngineManager

    cfg = _generic_table_config()
    mgr = EngineManager(cfg)
    try:
        tabby = mgr.engines["tabby"]
        assert isinstance(tabby, APISwapEngine)
        # available_tags() on a real APISwapEngine reads loaded_path; with no
        # server it returns an empty set (cached), and an unknown id then
        # defaults to the api_swap engine (preferred over the process engine).
        eng = await mgr.engine_for("never-heard-of-it")
        assert eng.key == "tabby"
    finally:
        await mgr.aclose()
