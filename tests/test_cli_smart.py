"""Tests for the smart-mode routerctl commands: smart | manual, explain smart,
and benchmarks refresh|show|clear. Hermetic — urllib is monkeypatched."""

from __future__ import annotations

import argparse
import json
import urllib.request

import pytest

import router.cli as cli


class _FakeResponse:
    def __init__(self, body, status: int = 200) -> None:
        self._data = json.dumps(body).encode()
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _recording_urlopen(body, calls):
    def fake_urlopen(req, timeout=None):
        calls.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "data": json.loads(req.data.decode()) if req.data else None,
            }
        )
        return _FakeResponse(body)

    return fake_urlopen


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
class TestParser:
    def _p(self):
        return cli.build_parser()

    def test_smart_and_manual_subcommands(self):
        assert self._p().parse_args(["smart"]).command == "smart"
        assert self._p().parse_args(["manual"]).command == "manual"

    def test_benchmarks_subcommand(self):
        args = self._p().parse_args(["benchmarks", "refresh", "llama3.1:8b"])
        assert args.command == "benchmarks"
        assert args.action == "refresh"
        assert args.model == "llama3.1:8b"
        args = self._p().parse_args(["benchmarks", "show"])
        assert args.action == "show"
        assert args.model is None

    def test_benchmarks_rejects_bad_action(self):
        with pytest.raises(SystemExit):
            self._p().parse_args(["benchmarks", "sync"])

    def test_explain_takes_message(self):
        args = self._p().parse_args(["explain", "smart", "--message", "fix my code"])
        assert args.model == "smart"
        assert args.message == "fix my code"


# --------------------------------------------------------------------------- #
# routerctl smart / manual
# --------------------------------------------------------------------------- #
class TestSetMode:
    def _setup(self, monkeypatch, tmp_path, config_text: str | None):
        cfg_path = tmp_path / "config.yaml"
        if config_text is not None:
            cfg_path.write_text(config_text)
        monkeypatch.setenv("ROUTER_CONFIG", str(cfg_path))
        calls: list[dict] = []
        monkeypatch.setattr(
            urllib.request, "urlopen",
            _recording_urlopen({"routing_mode": "manual"}, calls),
        )
        return cfg_path, calls

    def test_manual_updates_config_and_router(self, monkeypatch, tmp_path, capsys):
        cfg_path, calls = self._setup(
            monkeypatch, tmp_path, "host: 127.0.0.1\nport: 8077\n"
        )
        cli.cmd_set_mode("manual")
        text = cfg_path.read_text()
        assert "routing_mode: manual" in text
        assert "host: 127.0.0.1" in text  # existing content preserved
        assert calls and calls[0]["url"].endswith("/admin/smart/mode")
        assert calls[0]["data"] == {"mode": "manual"}

    def test_smart_replaces_existing_mode_line(self, monkeypatch, tmp_path, capsys):
        cfg_path, _ = self._setup(
            monkeypatch, tmp_path, "routing_mode: manual\nport: 8077\n"
        )
        cli.cmd_set_mode("smart")
        text = cfg_path.read_text()
        assert "routing_mode: smart" in text
        assert "routing_mode: manual" not in text
        assert text.count("routing_mode") == 1

    def test_invalid_result_refuses_write(self, monkeypatch, tmp_path, capsys):
        # A config that is already invalid stays untouched (validation runs on
        # the WHOLE file before writing).
        bad = "models:\n  - id: x\n"  # missing engine -> ConfigError
        cfg_path, _ = self._setup(monkeypatch, tmp_path, bad)
        with pytest.raises(SystemExit):
            cli.cmd_set_mode("smart")
        assert cfg_path.read_text() == bad

    def test_missing_config_still_applies_live(self, monkeypatch, tmp_path, capsys):
        _, calls = self._setup(monkeypatch, tmp_path, None)
        cli.cmd_set_mode("manual")
        assert calls and calls[0]["data"] == {"mode": "manual"}
        assert "config file not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# routerctl explain smart
# --------------------------------------------------------------------------- #
_DECISION_PAYLOAD = {
    "smart_selection": True,
    "mode": "smart",
    "requested_model": "smart",
    "model": "qwen2.5-coder-14b",
    "engine": "llamacpp",
    "policy": "balanced",
    "confidence": 0.81,
    "job": {"coding": 0.6, "general": 0.4},
    "would_swap": True,
    "swap_cost_s": 23.5,
    "reasons": ["picked 'qwen2.5-coder-14b' on 'llamacpp'"],
    "candidates": [
        {"model": "qwen2.5-coder-14b", "engine": "llamacpp", "total": 0.7,
         "components": {"quality": 0.8}, "swap_cost_s": 23.5, "resident": False},
        {"model": "tiny", "engine": "ollama", "total": 0.0, "components": {},
         "swap_cost_s": 0, "resident": True, "excluded": "in failure cooldown"},
    ],
    "fallbacks": ["llama3.1:8b"],
    "benchmarks": [
        {"capability": "coding", "score": 0.61, "confidence": 0.75,
         "match": "exact", "benchmark": "LiveCodeBench", "source": "builtin-priors"},
    ],
    "retry": {"never_after_streamed_bytes": True},
}


class TestExplainSmart:
    def test_posts_message_and_prints_decision(self, monkeypatch, capsys):
        calls: list[dict] = []
        monkeypatch.setattr(
            urllib.request, "urlopen", _recording_urlopen(_DECISION_PAYLOAD, calls)
        )
        cli.cmd_explain_smart(
            argparse.Namespace(model="smart", message="fix my code", endpoint=None)
        )
        assert calls[0]["url"].endswith("/admin/smart/resolve")
        assert calls[0]["data"]["model"] == "smart"
        assert calls[0]["data"]["messages"][0]["content"] == "fix my code"
        out = capsys.readouterr().out
        assert "qwen2.5-coder-14b" in out
        assert "confidence: 0.81" in out
        assert "EXCLUDED" in out
        assert "fallbacks : llama3.1:8b" in out
        assert "LiveCodeBench" in out

    def test_non_smart_result_prints_reason(self, monkeypatch, capsys):
        payload = {"smart_selection": False, "mode": "manual",
                   "requested_model": "x", "reason": "routing_mode is 'manual'"}
        monkeypatch.setattr(
            urllib.request, "urlopen", _recording_urlopen(payload, [])
        )
        cli.cmd_explain_smart(
            argparse.Namespace(model="x", message=None, endpoint=None)
        )
        out = capsys.readouterr().out
        assert "smart pick: no" in out
        assert "manual" in out

    def test_main_dispatches_explain_smart(self, monkeypatch, capsys):
        calls: list[dict] = []
        monkeypatch.setattr(
            urllib.request, "urlopen", _recording_urlopen(_DECISION_PAYLOAD, calls)
        )
        monkeypatch.setattr(
            "sys.argv", ["routerctl", "explain", "smart", "--message", "hello"]
        )
        cli.main()
        assert calls[0]["url"].endswith("/admin/smart/resolve")

    def test_main_dispatches_plain_explain(self, monkeypatch, capsys):
        calls: list[dict] = []
        payload = {"requested_model": "chat", "resolved_model": "m",
                   "engine": "e", "source": "static", "would_swap": False,
                   "reasons": [], "collisions": []}
        monkeypatch.setattr(
            urllib.request, "urlopen", _recording_urlopen(payload, calls)
        )
        monkeypatch.setattr("sys.argv", ["routerctl", "explain", "chat"])
        cli.main()
        assert calls[0]["url"].endswith("/admin/resolve")


# --------------------------------------------------------------------------- #
# routerctl benchmarks
# --------------------------------------------------------------------------- #
class TestBenchmarksCommand:
    def test_refresh_all(self, monkeypatch, capsys):
        calls: list[dict] = []
        monkeypatch.setattr(
            urllib.request, "urlopen",
            _recording_urlopen({"refreshed": ["llama3.1-8b"], "count": 1}, calls),
        )
        cli.cmd_benchmarks(argparse.Namespace(action="refresh", model=None))
        assert calls[0]["url"].endswith("/admin/benchmarks/refresh")
        assert calls[0]["data"] == {}
        assert "refreshed 1 model(s)" in capsys.readouterr().out

    def test_refresh_one_model(self, monkeypatch, capsys):
        calls: list[dict] = []
        monkeypatch.setattr(
            urllib.request, "urlopen",
            _recording_urlopen({"refreshed": ["llama3.1-8b"], "count": 1}, calls),
        )
        cli.cmd_benchmarks(argparse.Namespace(action="refresh", model="llama3.1:8b"))
        assert calls[0]["data"] == {"model": "llama3.1:8b"}

    def test_clear(self, monkeypatch, capsys):
        calls: list[dict] = []
        monkeypatch.setattr(
            urllib.request, "urlopen", _recording_urlopen({"cleared": 3}, calls)
        )
        cli.cmd_benchmarks(argparse.Namespace(action="clear", model=None))
        assert calls[0]["url"].endswith("/admin/benchmarks/clear")
        assert "cleared 3" in capsys.readouterr().out

    def test_show(self, monkeypatch, capsys):
        payload = {
            "cached_models": 1,
            "providers": [{"name": "builtin-priors", "requires_network": False}],
            "models": {
                "llama3.1-8b": {
                    "fetched_at": 1751000000,
                    "raw_ids": ["llama3.1:8b"],
                    "records": [
                        {"capability": "general", "score": 0.55,
                         "confidence": 0.75, "match": "exact",
                         "source": "builtin-priors"},
                    ],
                }
            },
        }
        monkeypatch.setattr(
            urllib.request, "urlopen", _recording_urlopen(payload, [])
        )
        cli.cmd_benchmarks(argparse.Namespace(action="show", model=None))
        out = capsys.readouterr().out
        assert "builtin-priors" in out
        assert "llama3.1-8b" in out
        assert "general" in out
