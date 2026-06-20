"""Section C — config loading, validation errors, and JSON schema.

Loads the real repo config.yaml, asserts ConfigError with an actionable message
for: a dangling model->engine ref, an unknown engine type, and a missing
required field; checks config_json_schema() returns a dict with $schema that
round-trips as JSON.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from router.config import (
    ConfigError,
    DiscoverConfig,
    config_json_schema,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Always load the shipped example config: a live deployment's config.yaml is
# gitignored and carries deployment-specific engine/model ids, so this test
# pins to config.example.yaml to stay deterministic on a fresh clone, in CI,
# and on a developer machine that has a customized config.yaml.
REAL_CONFIG = REPO_ROOT / "config.example.yaml"


def _write(tmp_path, text: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_load_real_config_yaml():
    """The shipped config.example.yaml loads and validates end to end."""
    cfg = load_config(str(REAL_CONFIG))
    assert cfg.port == 8077
    # The shipped example uses the generic engines: table (llama.cpp + Ollama).
    keys = set(cfg.engine_keys())
    assert {"llamacpp", "ollama"} <= keys
    # Every model references a configured engine (load_config validated it).
    for m in cfg.models:
        assert m.engine in keys
    # A couple of known ids from the shipped config.
    ids = {m.id for m in cfg.models}
    assert "qwen2.5-7b-instruct" in ids
    assert "llama3.1:8b" in ids


def test_dangling_model_engine_ref_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        models:
          - id: orphan
            engine: nonexistent-engine
            display_name: Orphan
        """,
    )
    with pytest.raises(ConfigError) as ei:
        load_config(path)
    msg = str(ei.value)
    assert "orphan" in msg
    assert "nonexistent-engine" in msg
    assert "unknown engine" in msg


def test_unknown_engine_type_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        engines:
          weird:
            type: not-a-real-type
            base_url: http://127.0.0.1:9000
        """,
    )
    with pytest.raises(ConfigError) as ei:
        load_config(path)
    msg = str(ei.value)
    assert "weird" in msg
    assert "unknown type" in msg
    assert "not-a-real-type" in msg


def test_missing_required_field_raises(tmp_path):
    # generic_process requires both base_url and start_cmd; omit start_cmd.
    path = _write(
        tmp_path,
        """
        engines:
          llamacpp:
            type: generic_process
            base_url: http://127.0.0.1:8080
        """,
    )
    with pytest.raises(ConfigError) as ei:
        load_config(path)
    msg = str(ei.value)
    assert "llamacpp" in msg
    assert "start_cmd" in msg
    assert "required" in msg


def test_missing_engine_type_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        engines:
          nope:
            base_url: http://127.0.0.1:8080
        """,
    )
    with pytest.raises(ConfigError, match="missing required 'type'"):
        load_config(path)


def test_model_without_id_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        models:
          - engine: ds4
            display_name: No ID
        """,
    )
    with pytest.raises(ConfigError, match="must have an 'id'"):
        load_config(path)


def test_model_without_engine_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        models:
          - id: floating
            display_name: Floating
        """,
    )
    with pytest.raises(ConfigError, match="must specify an 'engine'"):
        load_config(path)


def test_duplicate_engine_key_raises(tmp_path):
    # YAML can't have duplicate mapping keys cleanly, but the loader guards it
    # anyway; we trigger it by constructing the parse path. Use two engines that
    # are valid and confirm a *generic_process without base_url* still errors,
    # plus the api_swap branch warns (not raises) on a missing unload_path.
    path = _write(
        tmp_path,
        """
        engines:
          tabby:
            type: api_swap
            base_url: http://127.0.0.1:5000
        """,
    )
    # api_swap with no unload_path must LOAD (it only warns), not raise.
    cfg = load_config(path)
    assert cfg.engines[0].key == "tabby"
    assert cfg.engines[0].type == "api_swap"


def test_load_missing_file_uses_defaults():
    cfg = load_config("/nonexistent/path/to/config.yaml")
    # Falls back entirely to dataclass defaults.
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8077
    assert cfg.ds4.enabled is True


def test_generic_table_load_round_trip(tmp_path):
    path = _write(
        tmp_path,
        """
        engines:
          llamacpp:
            type: generic_process
            base_url: http://127.0.0.1:8080
            start_cmd: ["/usr/local/bin/llama-server", "-m", "/models/foo.gguf"]
            ready_path: /health
          tabby:
            type: api_swap
            base_url: http://127.0.0.1:5000
            unload_path: /v1/model/unload
            loaded_path: /v1/model
        models:
          - id: foo-gguf
            engine: llamacpp
            display_name: Foo
        """,
    )
    cfg = load_config(path)
    assert [e.key for e in cfg.engines] == ["llamacpp", "tabby"]
    assert cfg.engine_keys() == ["llamacpp", "tabby"]
    assert cfg.models[0].engine == "llamacpp"


# --------------------------------------------------------------------------- #
# JSON schema
# --------------------------------------------------------------------------- #
def test_config_json_schema_has_schema_key():
    schema = config_json_schema()
    assert isinstance(schema, dict)
    assert "$schema" in schema
    assert schema["$schema"].endswith("2020-12/schema")


def test_config_json_schema_round_trips_as_json():
    schema = config_json_schema()
    text = json.dumps(schema)
    again = json.loads(text)
    assert again == schema


def test_config_json_schema_describes_engines_and_models():
    schema = config_json_schema()
    props = schema["properties"]
    assert "engines" in props
    assert "models" in props
    assert "ds4" in props
    assert "ollama" in props
    # The engines table maps arbitrary keys to a oneOf of the four engine types.
    engine_entry = props["engines"]["additionalProperties"]
    consts = {
        arm["properties"]["type"]["const"] for arm in engine_entry["oneOf"]
    }
    assert consts == {"ds4", "ollama", "generic_process", "api_swap"}


# --------------------------------------------------------------------------- #
# New fields: GenericProcessConfig (discover_models, served_models,
#             tags_cache_ttl_s) and DiscoverConfig / RouterConfig.discover
# --------------------------------------------------------------------------- #

class TestGenericProcessNewFields:
    """GenericProcessConfig gets three new optional fields with safe defaults."""

    def test_defaults_when_absent(self, tmp_path):
        """Existing config without the new fields loads identically (back-compat)."""
        path = _write(
            tmp_path,
            """
            engines:
              vllm:
                type: generic_process
                base_url: http://127.0.0.1:8000
                start_cmd: ["/usr/local/bin/vllm", "serve"]
            """,
        )
        cfg = load_config(path)
        params = cfg.engines[0].params
        assert params.discover_models is False
        assert params.served_models == []
        assert params.tags_cache_ttl_s == 30.0

    def test_explicit_values_round_trip(self, tmp_path):
        path = _write(
            tmp_path,
            """
            engines:
              vllm:
                type: generic_process
                base_url: http://127.0.0.1:8000
                start_cmd: ["/usr/local/bin/vllm", "serve"]
                discover_models: true
                served_models: ["meta-llama/Llama-3.1-8B", "openai/gpt-4"]
                tags_cache_ttl_s: 60.0
            """,
        )
        cfg = load_config(path)
        params = cfg.engines[0].params
        assert params.discover_models is True
        assert params.served_models == ["meta-llama/Llama-3.1-8B", "openai/gpt-4"]
        assert params.tags_cache_ttl_s == 60.0

    def test_served_models_must_be_nonempty_strings(self, tmp_path):
        path = _write(
            tmp_path,
            """
            engines:
              vllm:
                type: generic_process
                base_url: http://127.0.0.1:8000
                start_cmd: ["/usr/local/bin/vllm", "serve"]
                served_models: ["valid-model", ""]
            """,
        )
        with pytest.raises(ConfigError, match="non-empty strings"):
            load_config(path)

    def test_tags_cache_ttl_s_must_be_nonnegative(self, tmp_path):
        path = _write(
            tmp_path,
            """
            engines:
              vllm:
                type: generic_process
                base_url: http://127.0.0.1:8000
                start_cmd: ["/usr/local/bin/vllm", "serve"]
                tags_cache_ttl_s: -1.0
            """,
        )
        with pytest.raises(ConfigError, match="tags_cache_ttl_s must be >= 0"):
            load_config(path)

    def test_tags_cache_ttl_s_zero_is_valid(self, tmp_path):
        """Zero is a valid TTL (disables caching)."""
        path = _write(
            tmp_path,
            """
            engines:
              vllm:
                type: generic_process
                base_url: http://127.0.0.1:8000
                start_cmd: ["/usr/local/bin/vllm", "serve"]
                tags_cache_ttl_s: 0.0
            """,
        )
        cfg = load_config(path)
        assert cfg.engines[0].params.tags_cache_ttl_s == 0.0

    def test_tags_cache_ttl_s_non_numeric_raises(self, tmp_path):
        """A non-numeric tags_cache_ttl_s must raise a clear ConfigError."""
        path = _write(
            tmp_path,
            """
            engines:
              vllm:
                type: generic_process
                base_url: http://127.0.0.1:8000
                start_cmd: ["/usr/local/bin/vllm", "serve"]
                tags_cache_ttl_s: "not-a-number"
            """,
        )
        with pytest.raises(ConfigError, match="tags_cache_ttl_s"):
            load_config(path)

    def test_served_models_bare_string_raises(self, tmp_path):
        """A bare string for served_models must raise (not silently split chars)."""
        path = _write(
            tmp_path,
            """
            engines:
              vllm:
                type: generic_process
                base_url: http://127.0.0.1:8000
                start_cmd: ["/usr/local/bin/vllm", "serve"]
                served_models: "single-string-not-a-list"
            """,
        )
        with pytest.raises(ConfigError, match="served_models"):
            load_config(path)


class TestDiscoverConfig:
    """DiscoverConfig + RouterConfig.discover field."""

    def test_default_when_absent(self):
        """No discover: block => all defaults (feature fully off)."""
        cfg = load_config("/nonexistent/config.yaml")
        d = cfg.discover
        assert isinstance(d, DiscoverConfig)
        assert d.enabled is False
        assert d.collision == "config_order"
        assert d.port_probe_enabled is False

    def test_discover_block_parsed(self, tmp_path):
        path = _write(
            tmp_path,
            """
            discover:
              enabled: true
              collision: config_order
              port_probe:
                enabled: true
            """,
        )
        cfg = load_config(path)
        d = cfg.discover
        assert d.enabled is True
        assert d.collision == "config_order"
        assert d.port_probe_enabled is True

    def test_discover_partial_override_uses_defaults_for_rest(self, tmp_path):
        path = _write(
            tmp_path,
            """
            discover:
              enabled: true
            """,
        )
        cfg = load_config(path)
        d = cfg.discover
        assert d.enabled is True
        # Unspecified keys fall back to DiscoverConfig defaults.
        assert d.collision == "config_order"
        assert d.port_probe_enabled is False

    def test_unknown_key_under_discover_raises(self, tmp_path):
        path = _write(
            tmp_path,
            """
            discover:
              enabled: true
              not_a_real_key: oops
            """,
        )
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(path)

    def test_unknown_key_under_port_probe_raises(self, tmp_path):
        path = _write(
            tmp_path,
            """
            discover:
              port_probe:
                enabled: false
                bogus_key: 42
            """,
        )
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(path)

    def test_invalid_collision_value_raises(self, tmp_path):
        path = _write(
            tmp_path,
            """
            discover:
              collision: not_a_valid_value
            """,
        )
        with pytest.raises(ConfigError, match="collision"):
            load_config(path)

    def test_config_order_collision_accepted(self, tmp_path):
        path = _write(
            tmp_path,
            """
            discover:
              collision: config_order
            """,
        )
        cfg = load_config(path)
        assert cfg.discover.collision == "config_order"

    def test_prefer_up_collision_rejected(self, tmp_path):
        """prefer_up is not implemented; the parser must reject it."""
        path = _write(
            tmp_path,
            """
            discover:
              collision: prefer_up
            """,
        )
        with pytest.raises(ConfigError, match="collision"):
            load_config(path)


class TestDiscoverRemovedKnobs:
    """Removed DiscoverConfig knobs must now be rejected, not silently no-op."""

    def test_augment_only_is_rejected_as_unknown_key(self, tmp_path):
        """augment_only was removed; the parser must reject it as an unknown key."""
        path = _write(
            tmp_path,
            """
            discover:
              enabled: true
              augment_only: true
            """,
        )
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(path)

    def test_discover_config_has_no_augment_only_attr(self):
        """DiscoverConfig must not expose augment_only at all."""
        d = DiscoverConfig()
        assert not hasattr(d, "augment_only")


class TestModelThinkingFloorValidation:
    """disable_thinking_below_max_tokens validation raises clear ConfigError."""

    def test_valid_floor_accepted(self, tmp_path):
        path = _write(
            tmp_path,
            """
            models:
              - id: gemma
                engine: ds4
                disable_thinking_below_max_tokens: 512
            """,
        )
        cfg = load_config(path)
        assert cfg.models[0].disable_thinking_below_max_tokens == 512

    def test_non_integer_floor_raises(self, tmp_path):
        """A non-integer value must raise a clear ConfigError, not a raw ValueError."""
        path = _write(
            tmp_path,
            """
            models:
              - id: gemma
                engine: ds4
                disable_thinking_below_max_tokens: "not-a-number"
            """,
        )
        with pytest.raises(ConfigError, match="disable_thinking_below_max_tokens"):
            load_config(path)

    def test_zero_floor_raises(self, tmp_path):
        """Value below 1 must raise ConfigError (existing check, just confirmed)."""
        path = _write(
            tmp_path,
            """
            models:
              - id: gemma
                engine: ds4
                disable_thinking_below_max_tokens: 0
            """,
        )
        with pytest.raises(ConfigError, match="disable_thinking_below_max_tokens"):
            load_config(path)


class TestSchemaNewFields:
    """JSON schema includes the new generic_process properties and discover block."""

    def test_generic_process_schema_has_new_fields(self):
        schema = config_json_schema()
        engine_entry = schema["properties"]["engines"]["additionalProperties"]
        gp_arm = next(
            a for a in engine_entry["oneOf"]
            if a["properties"]["type"]["const"] == "generic_process"
        )
        gp_props = gp_arm["properties"]
        assert "discover_models" in gp_props
        assert "served_models" in gp_props
        assert "tags_cache_ttl_s" in gp_props
        assert gp_props["discover_models"] == {"type": "boolean", "default": False}
        assert gp_props["tags_cache_ttl_s"] == {"type": "number", "default": 30.0}

    def test_discover_block_in_schema(self):
        schema = config_json_schema()
        assert "discover" in schema["properties"]
        d = schema["properties"]["discover"]
        assert d["type"] == "object"
        assert "enabled" in d["properties"]
        assert "collision" in d["properties"]
        assert "port_probe" in d["properties"]
        assert "augment_only" not in d["properties"]
        assert d["properties"]["collision"]["enum"] == ["config_order"]
        assert d["additionalProperties"] is False

    def test_schema_still_round_trips_as_json(self):
        schema = config_json_schema()
        text = __import__("json").dumps(schema)
        assert __import__("json").loads(text) == schema
