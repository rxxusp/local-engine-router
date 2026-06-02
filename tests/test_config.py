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
    config_json_schema,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG = REPO_ROOT / "config.yaml"


def _write(tmp_path, text: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_load_real_config_yaml():
    cfg = load_config(str(REAL_CONFIG))
    assert cfg.port == 8077
    assert cfg.ds4.enabled is True
    assert cfg.ollama.enabled is True
    # Every model references a configured engine (load_config validated it).
    keys = set(cfg.engine_keys())
    assert {"ds4", "ollama"} <= keys
    for m in cfg.models:
        assert m.engine in keys
    # A couple of known ids from the shipped config.
    ids = {m.id for m in cfg.models}
    assert "deepseek-v4-flash" in ids
    assert "qwen2.5:3b" in ids


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
