"""Unit tests for the durable runtime ModelCatalog."""

from __future__ import annotations

import json

import pytest

from router.catalog import ModelCatalog
from router.config import DiscoverConfig, EngineSpec, GenericProcessConfig, ModelSpec

from conftest import make_config


class _Engine:
    def __init__(self, key: str, *, cfg=None, models: set[str] | None = None) -> None:
        self.key = key
        self.base_url = f"http://{key}.test"
        self.cfg = cfg
        self._models = set(models or [])

    async def available_models(self) -> set[str]:
        return set(self._models)


def _cfg(tmp_path, *, models=None, aliases=None, ttl=2_592_000.0):
    return make_config(
        models=models or [],
        aliases=aliases or {},
        discover=DiscoverConfig(enabled=True, state_ttl_s=ttl),
        engines=[
            EngineSpec(
                key="first",
                type="generic_process",
                params=GenericProcessConfig(
                    base_url="http://first.test",
                    start_cmd=["llama-server", "-m", "/models/first.gguf"],
                    discover_models=True,
                    served_models=["served-first"],
                ),
            ),
            EngineSpec(
                key="second",
                type="generic_process",
                params=GenericProcessConfig(
                    base_url="http://second.test",
                    start_cmd=["vllm", "serve", "--served-model-name", "served-second"],
                    discover_models=True,
                ),
            ),
        ],
        state_file=str(tmp_path / "state.json"),
    )


def _engines(cfg):
    return {spec.key: _Engine(spec.key, cfg=spec.params) for spec in cfg.engines}


def test_static_wins_over_live_collision(tmp_path):
    cfg = _cfg(
        tmp_path,
        models=[ModelSpec(id="same", engine="first", display_name="Same")],
    )
    cat = ModelCatalog(cfg, _engines(cfg))
    summary = cat.rebuild({"second": {"same"}})

    entry = next(m for m in summary["models"] if m["id"] == "same")
    assert entry["engine"] == "first"
    assert entry["source"] == "static"
    assert entry["collisions"]


def test_config_order_wins_same_priority_collision(tmp_path):
    cfg = _cfg(tmp_path)
    cat = ModelCatalog(cfg, _engines(cfg))
    summary = cat.rebuild({"first": {"shared"}, "second": {"shared"}})

    entry = next(m for m in summary["models"] if m["id"] == "shared")
    assert entry["engine"] == "first"
    assert entry["source"] == "live"
    assert "config_order" in entry["collisions"][0]


def test_alias_entry_inherits_target_engine(tmp_path):
    cfg = _cfg(
        tmp_path,
        models=[ModelSpec(id="real-model", engine="second", display_name="Real")],
        aliases={"chat": "real-model"},
    )
    cat = ModelCatalog(cfg, _engines(cfg))

    entry = cat.owner_for("chat")
    assert entry is not None
    assert entry.engine == "second"
    assert entry.source == "alias"
    assert entry.resolved_model == "real-model"


def test_alias_wins_over_live_collision(tmp_path):
    cfg = _cfg(
        tmp_path,
        models=[ModelSpec(id="real-model", engine="first", display_name="Real")],
        aliases={"chat": "real-model"},
    )
    cat = ModelCatalog(cfg, _engines(cfg))
    summary = cat.rebuild({"second": {"chat"}})

    entry = next(m for m in summary["models"] if m["id"] == "chat")
    assert entry["engine"] == "first"
    assert entry["source"] == "alias"
    assert entry["collisions"]


def test_persisted_entry_routes_when_not_expired(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "catalog": {
                    "old-model": {
                        "engine": "second",
                        "source": "persisted",
                        "first_seen": 100,
                        "last_seen": 100,
                        "last_live_status": "stale",
                    }
                }
            }
        )
    )
    cfg = _cfg(tmp_path, ttl=0)
    cat = ModelCatalog(cfg, _engines(cfg))

    entry = cat.owner_for("old-model")
    assert entry is not None
    assert entry.engine == "second"
    assert entry.source == "persisted"
    assert entry.stale is True


def test_expired_persisted_entry_is_dropped(tmp_path, monkeypatch):
    import router.catalog as catalog_mod

    monkeypatch.setattr(catalog_mod, "_now", lambda: 10_000)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "catalog": {
                    "expired": {
                        "engine": "second",
                        "source": "persisted",
                        "first_seen": 1,
                        "last_seen": 1,
                    }
                }
            }
        )
    )
    cfg = _cfg(tmp_path, ttl=5)
    cat = ModelCatalog(cfg, _engines(cfg))

    assert cat.owner_for("expired") is None


@pytest.mark.asyncio
async def test_refresh_records_live_models(tmp_path):
    cfg = _cfg(tmp_path)
    engines = _engines(cfg)
    engines["second"]._models = {"live-model"}
    cat = ModelCatalog(cfg, engines)

    summary = await cat.refresh()

    entry = next(m for m in summary["models"] if m["id"] == "live-model")
    assert entry["engine"] == "second"
    assert entry["source"] == "live"
    assert entry["last_live_status"] == "live"
