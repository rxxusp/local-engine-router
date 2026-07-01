"""Tests for router/wizard.py - the `init` setup wizard.

All tests are hermetic: no real sockets, no HTTP, no GPU. Port probing and HTTP
model-list fetching are injected as fakes, so every assertion is about the
wizard's logic (detection, config scaffolding, the suggest-and-confirm rule)
rather than a live environment.
"""

from __future__ import annotations

import io
import os

import pytest

from router import wizard
from router.config import load_config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
TARGETS = {t.engine_key: t for t in wizard.PROBE_TARGETS}


class _TTY(io.StringIO):
    """A StringIO that claims to be a terminal (so the wizard goes interactive)."""

    def isatty(self) -> bool:  # noqa: D401
        return True


def _load(path: str):
    return load_config(path)


def _validate_text(tmp_path, text: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return load_config(str(p))


def fake_probe_factory(open_ports):
    def probe(host, port, timeout=0.35):
        return port in open_ports
    return probe


def fake_http_factory(routes):
    """routes: dict mapping a URL substring -> payload (or None)."""

    def http_get(url, timeout=1.5):
        for needle, payload in routes.items():
            if needle in url:
                return payload
        return None

    return http_get


# --------------------------------------------------------------------------- #
# _extract_models
# --------------------------------------------------------------------------- #
def test_extract_models_ollama_tags():
    payload = {"models": [{"name": "llama3.1:8b"}, {"model": "qwen2.5:7b"}]}
    assert wizard._extract_models("ollama_tags", payload) == [
        "llama3.1:8b",
        "qwen2.5:7b",
    ]


def test_extract_models_openai():
    payload = {"data": [{"id": "a"}, {"id": "b"}, {"id": "a"}]}
    # de-duplicated, order preserved
    assert wizard._extract_models("openai_models", payload) == ["a", "b"]


def test_extract_models_garbage_returns_empty():
    assert wizard._extract_models("openai_models", None) == []
    assert wizard._extract_models("openai_models", {"nope": 1}) == []
    assert wizard._extract_models("ollama_tags", {"models": "notalist"}) == []


def test_extract_models_drops_control_char_ids():
    # A model id with a raw newline/tab/NUL (from a buggy or hostile local
    # engine) is dropped, never carried into the scaffolded YAML, so it cannot
    # corrupt the file or abort an otherwise-valid run.
    payload = {
        "data": [
            {"id": "good"},
            {"id": "ba\nd"},
            {"id": "ta\tb"},
            {"id": "nul\x00"},
            {"id": "alsogood"},
        ]
    }
    assert wizard._extract_models("openai_models", payload) == ["good", "alsogood"]
    tags = {"models": [{"name": "ok"}, {"name": "bad\nname"}]}
    assert wizard._extract_models("ollama_tags", tags) == ["ok"]


# --------------------------------------------------------------------------- #
# detect_engines
# --------------------------------------------------------------------------- #
def test_detect_nothing_open():
    dets = wizard.detect_engines(
        probe=fake_probe_factory(set()), http_get=fake_http_factory({})
    )
    assert all(not d.port_open for d in dets)
    assert all(not d.confirmed for d in dets)


def test_detect_ollama_confirmed_with_models():
    probe = fake_probe_factory({11434})
    http = fake_http_factory(
        {"11434/api/tags": {"models": [{"name": "llama3.1:8b"}]}}
    )
    dets = wizard.detect_engines(probe=probe, http_get=http)
    ollama = next(d for d in dets if d.target.engine_key == "ollama")
    assert ollama.port_open and ollama.confirmed
    assert ollama.models == ["llama3.1:8b"]


def test_detect_open_but_unconfirmed():
    # Port open, but the HTTP probe returns nothing parseable -> not confirmed.
    probe = fake_probe_factory({8080})
    http = fake_http_factory({})
    dets = wizard.detect_engines(probe=probe, http_get=http)
    llama = next(d for d in dets if d.target.engine_key == "llamacpp")
    assert llama.port_open and not llama.confirmed
    assert llama.models == []


def test_detect_openai_models_shape():
    probe = fake_probe_factory({8000})
    http = fake_http_factory({"8000/v1/models": {"data": [{"id": "Qwen/Qwen2.5"}]}})
    dets = wizard.detect_engines(probe=probe, http_get=http)
    vllm = next(d for d in dets if d.target.engine_key == "vllm")
    assert vllm.confirmed and vllm.models == ["Qwen/Qwen2.5"]


# --------------------------------------------------------------------------- #
# build_config_yaml - every generated config must pass the real loader
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(TARGETS))
def test_generated_config_validates_for_every_backend(tmp_path, key):
    sel = wizard.EngineSelection(target=TARGETS[key], models=[f"{key}-model"])
    text = wizard.build_config_yaml([sel])
    cfg = _validate_text(tmp_path, text)
    assert key in cfg.engine_keys()
    assert f"{key}-model" in [m.id for m in cfg.models]


def test_generated_config_with_api_key(tmp_path):
    sel = wizard.EngineSelection(target=TARGETS["ollama"], models=["llama3.1:8b"])
    text = wizard.build_config_yaml(
        [sel], host="0.0.0.0", api_keys=["a-secret-key"]
    )
    cfg = _validate_text(tmp_path, text)
    assert cfg.host == "0.0.0.0"
    assert cfg.api_keys == ["a-secret-key"]


def test_generated_multi_engine(tmp_path):
    sels = [
        wizard.EngineSelection(target=TARGETS["ollama"], models=["llama3.1:8b"]),
        wizard.EngineSelection(target=TARGETS["llamacpp"], models=["qwen2.5-7b"]),
    ]
    cfg = _validate_text(tmp_path, wizard.build_config_yaml(sels))
    assert cfg.engine_keys() == ["ollama", "llamacpp"]


def test_generic_process_emits_nonempty_start_cmd_placeholder(tmp_path):
    # generic_process REQUIRES a non-empty start_cmd; the placeholder must keep
    # the config valid while clearly signalling it needs editing.
    sel = wizard.EngineSelection(target=TARGETS["llamacpp"], models=["m"])
    text = wizard.build_config_yaml([sel])
    assert "start_cmd:" in text
    assert "<PATH_TO_LLAMACPP_SERVER>" in text
    cfg = _validate_text(tmp_path, text)
    spec = next(e for e in cfg.engines if e.key == "llamacpp")
    assert spec.params.start_cmd  # non-empty


def test_empty_selection_returns_valid_starter(tmp_path):
    text = wizard.build_config_yaml([])
    assert text == wizard.STARTER_CONFIG
    cfg = _validate_text(tmp_path, text)
    # Starter declares exactly one active engine (ollama), not the legacy
    # ds4+ollama fallback that an omitted engines: block would imply.
    assert cfg.engine_keys() == ["ollama"]


def test_starter_config_is_valid(tmp_path):
    cfg = _validate_text(tmp_path, wizard.STARTER_CONFIG)
    assert cfg.engine_keys() == ["ollama"]
    assert cfg.models == []


def test_yaml_str_quotes_risky_model_ids(tmp_path):
    # A model id with a colon (common in Ollama tags) must round-trip cleanly.
    sel = wizard.EngineSelection(target=TARGETS["ollama"], models=["llama3.1:8b"])
    cfg = _validate_text(tmp_path, wizard.build_config_yaml([sel]))
    assert "llama3.1:8b" in [m.id for m in cfg.models]


@pytest.mark.parametrize(
    "value",
    [
        "plain-id",
        "Qwen/Qwen2.5-7B-Instruct",
        "llama3.1:8b",
        "has space",
        "a\nb",
        "a\tb",
        "x\x00y",
        "trailing ",
        "",
        "true",
        "null",
        "- dashy",
        'quote"inside',
        "back\\slash",
        "#hash",
        "{braces}",
    ],
)
def test_yaml_str_roundtrips_any_scalar(value):
    import yaml

    # Whatever _yaml_str emits, parsing `v: <it>` must yield the exact original.
    rendered = wizard._yaml_str(value)
    parsed = yaml.safe_load(f"v: {rendered}\n")
    assert parsed == {"v": value}


def test_generic_api_swap_fallback_validates(tmp_path):
    # A synthetic api_swap target that is neither lmstudio nor tabbyapi exercises
    # the generic api_swap branch of _engine_block; its output must still load.
    target = wizard.ProbeTarget(
        label="Custom",
        port=7777,
        engine_key="customswap",
        engine_type="api_swap",
        models_path="/v1/models",
        models_kind="openai_models",
        base_url="http://127.0.0.1:7777",
    )
    sel = wizard.EngineSelection(target=target, models=["custom-model"])
    cfg = _validate_text(tmp_path, wizard.build_config_yaml([sel]))
    assert cfg.engine_keys() == ["customswap"]
    spec = next(e for e in cfg.engines if e.key == "customswap")
    assert spec.type == "api_swap"


# --------------------------------------------------------------------------- #
# run_init - non-interactive
# --------------------------------------------------------------------------- #
def test_run_init_yes_includes_only_confirmed(tmp_path):
    cfg_path = str(tmp_path / "config.yaml")
    probe = fake_probe_factory({11434, 8080})  # 8080 will be unconfirmed
    http = fake_http_factory(
        {"11434/api/tags": {"models": [{"name": "llama3.1:8b"}]}}
    )
    rc = wizard.run_init(
        ["--yes", "--config", cfg_path],
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
        probe=probe,
        http_get=http,
    )
    assert rc == 0
    cfg = _load(cfg_path)
    # Confirmed Ollama in; unconfirmed :8080 (llamacpp) NEVER auto-added.
    assert cfg.engine_keys() == ["ollama"]
    assert "llamacpp" not in cfg.engine_keys()


def test_run_init_never_auto_routes_unconfirmed(tmp_path):
    cfg_path = str(tmp_path / "config.yaml")
    probe = fake_probe_factory({8080})  # only an unconfirmed open port
    rc = wizard.run_init(
        ["--yes", "--config", cfg_path],
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
        probe=probe,
        http_get=fake_http_factory({}),
    )
    assert rc == 0
    cfg = _load(cfg_path)
    # No confirmed engines -> starter (ollama only), the unconfirmed port is gone.
    assert "llamacpp" not in cfg.engine_keys()


def test_run_init_overwrite_protection(tmp_path):
    cfg_path = str(tmp_path / "config.yaml")
    probe = fake_probe_factory({11434})
    http = fake_http_factory({"11434/api/tags": {"models": []}})
    args = ["--yes", "--config", cfg_path]
    assert wizard.run_init(args, stdin=io.StringIO(""), stdout=io.StringIO(),
                           probe=probe, http_get=http) == 0
    before = open(cfg_path).read()
    # Re-run without --force, non-interactive: refuses, leaves file untouched.
    rc = wizard.run_init(args, stdin=io.StringIO(""), stdout=io.StringIO(),
                         probe=probe, http_get=http)
    assert rc == 1
    assert open(cfg_path).read() == before
    # With --force it overwrites and succeeds.
    rc2 = wizard.run_init(args + ["--force"], stdin=io.StringIO(""),
                          stdout=io.StringIO(), probe=probe, http_get=http)
    assert rc2 == 0


def test_run_init_write_failure_returns_1(tmp_path):
    # Make the target config unwritable: its parent is a regular file, so
    # _write_config's makedirs raises. run_init must catch, report, and return 1.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    cfg_path = str(blocker / "config.yaml")
    out = io.StringIO()
    rc = wizard.run_init(
        ["--yes", "--config", cfg_path],
        stdin=io.StringIO(""),
        stdout=out,
        probe=fake_probe_factory(set()),
        http_get=fake_http_factory({}),
    )
    assert rc == 1
    assert "ERROR" in out.getvalue()


def test_run_init_example_does_not_probe(tmp_path):
    cfg_path = str(tmp_path / "config.yaml")

    def boom(*a, **k):
        raise AssertionError("--example must not probe ports or fetch over HTTP")

    rc = wizard.run_init(
        ["--example", "--config", cfg_path],
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
        probe=boom,
        http_get=boom,
    )
    assert rc == 0
    assert _load(cfg_path).engine_keys() == ["ollama"]


def test_run_init_example_respects_existing(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("host: 127.0.0.1\nport: 9999\n")
    rc = wizard.run_init(
        ["--example", "--config", str(cfg_path)],
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
        probe=fake_probe_factory(set()),
        http_get=fake_http_factory({}),
    )
    assert rc == 0
    # Untouched.
    assert "9999" in cfg_path.read_text()


def test_run_init_detect_only_writes_nothing(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    probe = fake_probe_factory({11434})
    http = fake_http_factory({"11434/api/tags": {"models": [{"name": "x"}]}})
    out = io.StringIO()
    rc = wizard.run_init(
        ["--detect-only", "--config", str(cfg_path)],
        stdin=io.StringIO(""),
        stdout=out,
        probe=probe,
        http_get=http,
    )
    assert rc == 0
    assert not cfg_path.exists()
    assert "Ollama" in out.getvalue()


# --------------------------------------------------------------------------- #
# run_init - interactive
# --------------------------------------------------------------------------- #
def test_run_init_interactive_include_decline_and_key(tmp_path):
    cfg_path = str(tmp_path / "config.yaml")
    probe = fake_probe_factory({11434, 8080})
    http = fake_http_factory(
        {"11434/api/tags": {"models": [{"name": "llama3.1:8b"}]}}
    )
    answers = "\n".join(
        [
            "y",  # include Ollama
            "n",  # decline the unconfirmed :8080
            "",  # bind host default
            "my-key",  # api key
        ]
    ) + "\n"
    rc = wizard.run_init(
        ["--config", cfg_path],
        stdin=_TTY(answers),
        stdout=io.StringIO(),
        probe=probe,
        http_get=http,
    )
    assert rc == 0
    cfg = _load(cfg_path)
    assert cfg.engine_keys() == ["ollama"]
    assert cfg.api_keys == ["my-key"]


def test_run_init_interactive_accept_unconfirmed_adds_it(tmp_path):
    # When the user explicitly says yes to an unconfirmed open port, it IS added
    # (suggest-and-confirm, not suggest-and-refuse).
    cfg_path = str(tmp_path / "config.yaml")
    probe = fake_probe_factory({8080})
    answers = "\n".join(["y", "", ""]) + "\n"  # add :8080, host default, no key
    rc = wizard.run_init(
        ["--config", cfg_path],
        stdin=_TTY(answers),
        stdout=io.StringIO(),
        probe=probe,
        http_get=fake_http_factory({}),
    )
    assert rc == 0
    assert "llamacpp" in _load(cfg_path).engine_keys()


# --------------------------------------------------------------------------- #
# Console-script dispatch: `routerctl init` and `local-engine-router init`
# --------------------------------------------------------------------------- #
def _neuter_network(monkeypatch):
    """Make the real probe/HTTP defaults explode, so any test that reaches them
    fails loudly instead of touching the network."""

    def boom(*a, **k):
        raise AssertionError("dispatch test must not touch the network")

    monkeypatch.setattr(wizard, "default_probe", boom)
    monkeypatch.setattr(wizard, "default_http_get_json", boom)


def test_routerctl_init_dispatches(monkeypatch, tmp_path):
    import router.cli as cli

    _neuter_network(monkeypatch)
    cfg = str(tmp_path / "config.yaml")
    monkeypatch.setattr("sys.argv", ["routerctl", "init", "--example", "--config", cfg])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert os.path.exists(cfg)


def test_router_main_init_dispatches(monkeypatch, tmp_path):
    import router.__main__ as rmain

    _neuter_network(monkeypatch)
    cfg = str(tmp_path / "config.yaml")
    monkeypatch.setattr("sys.argv", ["router", "init", "--example", "--config", cfg])
    with pytest.raises(SystemExit) as exc:
        rmain.main()
    assert exc.value.code == 0
    assert os.path.exists(cfg)


def test_routerctl_init_passthrough_help(monkeypatch, capsys):
    # `routerctl init --help` should reach the wizard's own parser, not the
    # routerctl subcommand parser.
    import router.cli as cli

    monkeypatch.setattr("sys.argv", ["routerctl", "init", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    # Assert on a flag that exists ONLY in the wizard's parser (not in
    # routerctl's top-level help), so this pins the wizard parser as the handler.
    assert "--detect-only" in capsys.readouterr().out
