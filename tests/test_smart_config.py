"""Config-loading tests for the smart picker: routing_mode, smart block,
model metadata, and the JSON schema additions."""

from __future__ import annotations

import pytest

from router.config import (
    ConfigError,
    RouterConfig,
    SmartConfig,
    config_json_schema,
    load_config,
)


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return str(path)


BASE = """
engines:
  ollama:
    type: ollama
    base_url: http://127.0.0.1:11434
models:
  - id: llama3.1:8b
    engine: ollama
"""


class TestRoutingMode:
    def test_missing_routing_mode_defaults_to_smart(self, tmp_path):
        cfg = load_config(_write(tmp_path, BASE))
        assert cfg.routing_mode == "smart"

    def test_manual_mode_loads(self, tmp_path):
        cfg = load_config(_write(tmp_path, BASE + "\nrouting_mode: manual\n"))
        assert cfg.routing_mode == "manual"

    def test_invalid_mode_fails_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="routing_mode"):
            load_config(_write(tmp_path, BASE + "\nrouting_mode: clever\n"))

    def test_code_default_is_smart(self):
        assert RouterConfig().routing_mode == "smart"


class TestSmartBlock:
    def test_absent_block_gives_defaults(self, tmp_path):
        cfg = load_config(_write(tmp_path, BASE))
        assert cfg.smart.aliases == ["smart", "auto", "default"]
        assert cfg.smart.policy == "balanced"
        assert cfg.smart.override_exact_model_ids is False
        assert cfg.smart.benchmarks.allow_network is False

    def test_full_block_parses(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                BASE
                + """
smart:
  aliases: [smart, best]
  policy: fast
  catch_cloud_models: false
  swap_margin: 0.2
  weights:
    quality: 0.5
    swap_cost: 0.3
  policies:
    mine:
      quality: 1.0
  retry:
    max_fallbacks: 1
    cooldown_s: 30
  benchmarks:
    allow_network: true
    cache_ttl_s: 3600
""",
            )
        )
        assert cfg.smart.aliases == ["smart", "best"]
        assert cfg.smart.policy == "fast"
        assert cfg.smart.catch_cloud_models is False
        assert cfg.smart.swap_margin == 0.2
        assert cfg.smart.weights == {"quality": 0.5, "swap_cost": 0.3}
        assert cfg.smart.policies == {"mine": {"quality": 1.0}}
        assert cfg.smart.retry.max_fallbacks == 1
        assert cfg.smart.retry.cooldown_s == 30.0
        assert cfg.smart.benchmarks.allow_network is True

    def test_custom_policy_name_usable(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                BASE + "\nsmart:\n  policy: mine\n  policies:\n    mine:\n      quality: 1.0\n",
            )
        )
        assert cfg.smart.policy == "mine"

    def test_unknown_smart_key_fails(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown key.*smart"):
            load_config(_write(tmp_path, BASE + "\nsmart:\n  bogus: true\n"))

    def test_unknown_weight_key_fails(self, tmp_path):
        with pytest.raises(ConfigError, match="weight"):
            load_config(
                _write(tmp_path, BASE + "\nsmart:\n  weights:\n    vibes: 1.0\n")
            )

    def test_negative_weight_fails(self, tmp_path):
        with pytest.raises(ConfigError, match="quality"):
            load_config(
                _write(tmp_path, BASE + "\nsmart:\n  weights:\n    quality: -1\n")
            )

    def test_unknown_policy_fails(self, tmp_path):
        with pytest.raises(ConfigError, match="policy"):
            load_config(_write(tmp_path, BASE + "\nsmart:\n  policy: warp\n"))

    def test_bad_aliases_fail(self, tmp_path):
        with pytest.raises(ConfigError, match="aliases"):
            load_config(_write(tmp_path, BASE + "\nsmart:\n  aliases: notalist\n"))

    def test_swap_margin_out_of_range_fails(self, tmp_path):
        with pytest.raises(ConfigError, match="swap_margin"):
            load_config(_write(tmp_path, BASE + "\nsmart:\n  swap_margin: 3\n"))

    def test_bad_retry_int_fails(self, tmp_path):
        with pytest.raises(ConfigError, match="max_fallbacks"):
            load_config(
                _write(tmp_path, BASE + "\nsmart:\n  retry:\n    max_fallbacks: -2\n")
            )

    def test_smart_alias_shadowing_model_warns_not_fails(self, tmp_path, caplog):
        text = BASE + "\nsmart:\n  aliases: ['llama3.1:8b']\n"
        with caplog.at_level("WARNING"):
            cfg = load_config(_write(tmp_path, text))
        assert cfg.smart.aliases == ["llama3.1:8b"]
        assert any("smart alias" in r.message for r in caplog.records)


class TestModelMetadata:
    def test_metadata_parses(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                """
engines:
  ollama: {type: ollama, base_url: "http://127.0.0.1:11434"}
models:
  - id: llama3.1:8b
    engine: ollama
    quality_tier: 4
    speed_tier: 3
    memory_gb: 6.5
    capabilities: [coding, tool_use]
    strengths: {coding: 0.9}
    smart_enabled: false
""",
            )
        )
        spec = cfg.models[0]
        assert spec.quality_tier == 4
        assert spec.speed_tier == 3
        assert spec.memory_gb == 6.5
        assert spec.capabilities == ["coding", "tool_use"]
        assert spec.strengths == {"coding": 0.9}
        assert spec.smart_enabled is False

    def test_metadata_defaults(self, tmp_path):
        cfg = load_config(_write(tmp_path, BASE))
        spec = cfg.models[0]
        assert spec.quality_tier is None
        assert spec.speed_tier is None
        assert spec.capabilities == []
        assert spec.strengths == {}
        assert spec.smart_enabled is True

    @pytest.mark.parametrize(
        "snippet,match",
        [
            ("quality_tier: 9", "quality_tier"),
            ("speed_tier: zero", "speed_tier"),
            ("memory_gb: -1", "memory_gb"),
            ("capabilities: [flying]", "capability"),
            ("strengths: {coding: 2}", "coding"),
            ("strengths: {vibes: 0.5}", "strength"),
        ],
    )
    def test_bad_metadata_fails_clearly(self, tmp_path, snippet, match):
        text = BASE.rstrip() + f"\n    {snippet}\n"
        with pytest.raises(ConfigError, match=match):
            load_config(_write(tmp_path, text))


class TestSmartSchema:
    def test_schema_has_routing_mode_enum(self):
        schema = config_json_schema()
        rm = schema["properties"]["routing_mode"]
        assert rm["enum"] == ["manual", "smart"]
        assert rm["default"] == "smart"

    def test_schema_has_smart_block(self):
        schema = config_json_schema()
        smart = schema["properties"]["smart"]
        assert smart["type"] == "object"
        for key in ("aliases", "policy", "weights", "retry", "benchmarks",
                    "override_exact_model_ids", "swap_margin"):
            assert key in smart["properties"]
        assert smart["additionalProperties"] is False

    def test_schema_model_metadata_fields(self):
        schema = config_json_schema()
        model_props = schema["properties"]["models"]["items"]["properties"]
        for key in ("quality_tier", "speed_tier", "memory_gb", "capabilities",
                    "strengths", "smart_enabled"):
            assert key in model_props

    def test_schema_round_trips_as_json(self):
        import json

        json.loads(json.dumps(config_json_schema()))


class TestSmartConfigDefaults:
    def test_default_dataclass_matches_loader_defaults(self, tmp_path):
        loaded = load_config(_write(tmp_path, BASE)).smart
        default = SmartConfig()
        assert loaded == default
