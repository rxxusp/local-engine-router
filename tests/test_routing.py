"""Section B — EngineManager.engine_for routing resolution.

Covers: the static registry; ds4's fixed ids; the live Ollama-tag fallback for
ids pulled after startup; single-engine fallback; and the generic ``engines:``
table (generic_process + api_swap) resolving by type.

Also covers discovery routing (Slice 3): down-engine resolution via
served_ids_from_start_cmd, .gguf basenames, served_models hints, and the
_seen_models cache. All discovery tests are gated on cfg.discover.enabled so
that discover=False leaves routing byte-identical to the pre-discovery behaviour.
"""

from __future__ import annotations

import logging

import pytest

from router.config import (
    ApiSwapConfig,
    DiscoverConfig,
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
    served_ids_from_start_cmd,
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
        state_file="/tmp/local-engine-router-test-state.json",
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


# --------------------------------------------------------------------------- #
# served_ids_from_start_cmd unit tests
# --------------------------------------------------------------------------- #
def test_served_ids_from_start_cmd_served_model_name_single():
    ids = served_ids_from_start_cmd(
        ["vllm", "serve", "--served-model-name", "my-alias"]
    )
    assert ids == {"my-alias"}


def test_served_ids_from_start_cmd_served_model_name_multiple():
    """vLLM and SGLang allow multiple --served-model-name values."""
    ids = served_ids_from_start_cmd(
        ["vllm", "serve", "--served-model-name", "alias-a", "alias-b", "--port", "8080"]
    )
    assert ids == {"alias-a", "alias-b"}


def test_served_ids_from_start_cmd_model_flag():
    ids = served_ids_from_start_cmd(["/usr/bin/llama-server", "-m", "/models/foo.gguf"])
    assert "/models/foo.gguf" in ids
    assert "foo" in ids  # gguf basename without extension


def test_served_ids_from_start_cmd_model_path_flag():
    ids = served_ids_from_start_cmd(
        ["python3", "-m", "vllm.entrypoints.openai.api_server",
         "--model-path", "/storage/weights/gemma-9b"]
    )
    assert "/storage/weights/gemma-9b" in ids


def test_served_ids_from_start_cmd_gguf_basename_via_served_model_name():
    ids = served_ids_from_start_cmd(
        ["llama-server", "--served-model-name", "/path/to/model.gguf"]
    )
    assert "/path/to/model.gguf" in ids
    assert "model" in ids


def test_served_ids_from_start_cmd_string_input():
    ids = served_ids_from_start_cmd(
        "vllm serve --served-model-name chat-model --port 8080"
    )
    assert ids == {"chat-model"}


def test_served_ids_from_start_cmd_empty():
    assert served_ids_from_start_cmd([]) == set()
    assert served_ids_from_start_cmd("") == set()


def test_served_ids_from_start_cmd_no_relevant_flags():
    ids = served_ids_from_start_cmd(["python3", "server.py", "--port", "9000"])
    assert ids == set()


# --------------------------------------------------------------------------- #
# Local fake engine with a cfg that has discover_models
# --------------------------------------------------------------------------- #
class _FakeDiscoverEngine(FakeEngine):
    """FakeEngine with an attached cfg object for discovery tests.

    This lets _discovered_index() see .cfg.discover_models / .cfg.start_cmd /
    .cfg.served_models without us spawning a real GenericProcessEngine."""

    def __init__(
        self,
        key: str,
        *,
        discover_models: bool = True,
        start_cmd: list[str] | str | None = None,
        served_models: list[str] | None = None,
    ) -> None:
        super().__init__(key)

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.discover_models = discover_models
        cfg.start_cmd = start_cmd or []
        cfg.served_models = served_models or []
        self.cfg = cfg


def _discover_config(**discover_kw) -> RouterConfig:
    """RouterConfig with a DiscoverConfig built from keyword arguments."""
    return make_config(discover=DiscoverConfig(**discover_kw))


# --------------------------------------------------------------------------- #
# Discovery routing tests (all gated on discover.enabled)
# --------------------------------------------------------------------------- #
async def test_discovery_off_leaves_routing_unchanged():
    """When discover.enabled is False, engine_for does not use the discovery index.

    Two process-type engines are used (no APISwapEngine fallback) so that
    'no engine can serve' is raised for an unknown id -- proving that the
    discovery code path is completely bypassed when discover.enabled is False."""
    models = [
        ModelSpec(id="m-a", engine="eng-a", display_name="a"),
        ModelSpec(id="m-b", engine="eng-b", display_name="b"),
    ]
    engine_a = _FakeDiscoverEngine(
        "eng-a", discover_models=True,
        start_cmd=["--served-model-name", "secret-model"],
    )
    engine_b = _FakeDiscoverEngine("eng-b", discover_models=False)
    # make_config() leaves discover.enabled=False by default.
    cfg = make_config(models=models)
    mgr = make_manager_with_fakes({"eng-a": engine_a, "eng-b": engine_b}, cfg=cfg)
    # discovery is OFF: secret-model should NOT be found.
    with pytest.raises(EngineError, match="no engine can serve"):
        await mgr.engine_for("secret-model")


async def test_discovery_routes_via_served_model_name():
    """With discover.enabled, --served-model-name in start_cmd routes the down engine."""
    cfg = _discover_config(enabled=True)
    engine = _FakeDiscoverEngine(
        "proc",
        discover_models=True,
        start_cmd=["vllm", "serve", "--served-model-name", "vllm-model"],
    )
    mgr = make_manager_with_fakes({"proc": engine}, cfg=cfg)
    eng = await mgr.engine_for("vllm-model")
    assert eng.key == "proc"


async def test_discovery_routes_via_gguf_basename():
    """A .gguf model path in start_cmd also exposes the basename as a model id."""
    cfg = _discover_config(enabled=True)
    engine = _FakeDiscoverEngine(
        "llamacpp",
        discover_models=True,
        start_cmd=["/usr/local/bin/llama-server", "-m", "/models/my-llama.gguf"],
    )
    mgr = make_manager_with_fakes({"llamacpp": engine}, cfg=cfg)
    eng = await mgr.engine_for("my-llama")
    assert eng.key == "llamacpp"


async def test_discovery_routes_via_served_models_hint():
    """A model listed in served_models is discoverable even with no start_cmd."""
    cfg = _discover_config(enabled=True)
    engine = _FakeDiscoverEngine(
        "proc",
        discover_models=True,
        start_cmd=[],
        served_models=["explicit-hint-model"],
    )
    mgr = make_manager_with_fakes({"proc": engine}, cfg=cfg)
    eng = await mgr.engine_for("explicit-hint-model")
    assert eng.key == "proc"


async def test_discovery_routes_via_seen_cache():
    """A model id seeded into _seen_models is discoverable as if seen at runtime."""
    cfg = _discover_config(enabled=True)
    engine = _FakeDiscoverEngine("proc", discover_models=True)
    mgr = make_manager_with_fakes({"proc": engine}, cfg=cfg)
    # Manually seed the seen cache (simulates a previous uptime snapshot).
    mgr._seen_models["proc"] = {"previously-served-model"}
    eng = await mgr.engine_for("previously-served-model")
    assert eng.key == "proc"


async def test_discovery_static_index_wins_over_discovery():
    """The static index takes precedence over discovery: a model declared in
    models: for engine A cannot be hijacked by engine B's discovery."""
    models = [ModelSpec(id="static-model", engine="engine-a", display_name="s")]
    cfg = make_config(models=models, discover=DiscoverConfig(enabled=True))

    engine_a = _FakeDiscoverEngine("engine-a", discover_models=False)
    engine_b = _FakeDiscoverEngine(
        "engine-b",
        discover_models=True,
        served_models=["static-model"],  # tries to claim it
    )
    mgr = make_manager_with_fakes({"engine-a": engine_a, "engine-b": engine_b}, cfg=cfg)
    eng = await mgr.engine_for("static-model")
    assert eng.key == "engine-a"  # static index won


async def test_discovery_collision_config_order_wins_and_logs_warning(caplog):
    """When two engines both claim a model id, the first in cfg.engines order wins
    and a WARNING is emitted."""
    import router.engines as eng_mod

    # Clear the module-level collision warning tracker so the warning fires fresh.
    eng_mod._warned_collision.clear()

    cfg = make_config(
        discover=DiscoverConfig(enabled=True),
        engines=[
            EngineSpec(
                key="first",
                type="generic_process",
                enabled=True,
                params=GenericProcessConfig(
                    base_url="http://127.0.0.1:19001",
                    start_cmd=["--served-model-name", "shared-model"],
                    discover_models=True,
                ),
            ),
            EngineSpec(
                key="second",
                type="generic_process",
                enabled=True,
                params=GenericProcessConfig(
                    base_url="http://127.0.0.1:19002",
                    start_cmd=["--served-model-name", "shared-model"],
                    discover_models=True,
                ),
            ),
        ],
    )

    from router.engines import EngineManager

    mgr = EngineManager(cfg)
    try:
        with caplog.at_level(logging.WARNING, logger="router.engines"):
            eng = await mgr.engine_for("shared-model")

        assert eng.key == "first"
        assert any("shared-model" in r.message for r in caplog.records)
        assert any("config_order" in r.message for r in caplog.records)
    finally:
        await mgr.aclose()
