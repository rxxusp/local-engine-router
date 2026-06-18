"""Tests for router/cli.py — routerctl argument parsing and HTTP dispatch.

All tests are hermetic: no real network, no systemd, no GPU.  The HTTP layer
(urllib.request.urlopen) is monkeypatched with a fake that records calls and
returns canned JSON, so every assertion is about *structure* (which path gets
called, what body is sent) rather than live responses.

The systemd service name is intentionally NOT hardcoded in these tests.
Service-control tests assert on command *structure*:
  ['systemctl', '--user', <action>, <any-name>]
so the tests remain correct after a rename in a parallel slice.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import types
import urllib.error
import urllib.request
from unittest.mock import MagicMock, call, patch

import pytest

import router.cli as cli


# --------------------------------------------------------------------------- #
# Fake urllib response + transport helpers
# --------------------------------------------------------------------------- #

class _FakeResponse:
    """Minimal file-like object that urllib.request.urlopen returns."""

    def __init__(self, body: dict | list, status: int = 200) -> None:
        self._data = json.dumps(body).encode()
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _make_urlopen(response_body: dict | list, status: int = 200):
    """Return a mock urlopen that always yields a canned _FakeResponse."""
    resp = _FakeResponse(response_body, status)

    def fake_urlopen(req, timeout=None):
        return resp

    return fake_urlopen


# --------------------------------------------------------------------------- #
# Minimal status payload
# --------------------------------------------------------------------------- #
_STATUS_PAYLOAD = {
    "active_engine": "ds4",
    "last_swap": {"from": "ollama", "to": "ds4", "duration_s": 2.1, "ok": True},
    "engines": {
        "ds4": {"ready": True, "in_flight": 0, "base_url": "http://ds4.local"},
        "ollama": {"ready": False, "in_flight": 0, "base_url": "http://ollama.local"},
    },
    "models": [
        {"id": "deepseek-v4-flash", "engine": "ds4", "name": "DS4 Flash"},
    ],
}

_MODELS_PAYLOAD = {
    "data": [
        {"id": "deepseek-v4-flash", "owned_by": "ds4", "context_length": 131072},
        {"id": "qwen3:3b", "owned_by": "ollama"},
    ]
}

_HEALTH_PAYLOAD = {"status": "ok"}


# --------------------------------------------------------------------------- #
# Parser tests — argument parsing only, no I/O
# --------------------------------------------------------------------------- #

class TestBuildParser:
    def _p(self):
        return cli.build_parser()

    def test_status_subcommand_parsed(self):
        args = self._p().parse_args(["status"])
        assert args.command == "status"

    def test_models_subcommand_parsed(self):
        args = self._p().parse_args(["models"])
        assert args.command == "models"

    def test_health_subcommand_parsed(self):
        args = self._p().parse_args(["health"])
        assert args.command == "health"

    def test_logs_subcommand_parsed(self):
        args = self._p().parse_args(["logs"])
        assert args.command == "logs"

    def test_use_subcommand_parses_target(self):
        args = self._p().parse_args(["use", "ds4"])
        assert args.command == "use"
        assert args.target == "ds4"

    def test_use_subcommand_parses_model_id(self):
        args = self._p().parse_args(["use", "qwen3:3b"])
        assert args.command == "use"
        assert args.target == "qwen3:3b"

    def test_ds4_shortcut_parsed(self):
        args = self._p().parse_args(["ds4"])
        assert args.command == "ds4"

    def test_ollama_shortcut_parsed(self):
        args = self._p().parse_args(["ollama"])
        assert args.command == "ollama"

    def test_start_subcommand_parsed(self):
        args = self._p().parse_args(["start"])
        assert args.command == "start"

    def test_stop_subcommand_parsed(self):
        args = self._p().parse_args(["stop"])
        assert args.command == "stop"

    def test_restart_subcommand_parsed(self):
        args = self._p().parse_args(["restart"])
        assert args.command == "restart"

    def test_use_requires_target(self):
        p = self._p()
        with pytest.raises(SystemExit):
            p.parse_args(["use"])

    def test_no_subcommand_exits(self):
        p = self._p()
        with pytest.raises(SystemExit):
            p.parse_args([])


# --------------------------------------------------------------------------- #
# cmd_status — GETs /status
# --------------------------------------------------------------------------- #

class TestCmdStatus:
    def test_gets_status_path(self, monkeypatch, capsys):
        captured_requests = []

        def fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        args = argparse.Namespace(command="status")
        cli.cmd_status(args)

        assert len(captured_requests) == 1
        assert captured_requests[0].full_url.endswith("/status")
        assert captured_requests[0].method == "GET"

    def test_prints_active_engine(self, monkeypatch, capsys):
        monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen(_STATUS_PAYLOAD))
        cli.cmd_status(argparse.Namespace())
        out = capsys.readouterr().out
        assert "ds4" in out


# --------------------------------------------------------------------------- #
# cmd_models — GETs /v1/models
# --------------------------------------------------------------------------- #

class TestCmdModels:
    def test_gets_models_path(self, monkeypatch):
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return _FakeResponse(_MODELS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        cli.cmd_models(argparse.Namespace())

        assert captured[0].full_url.endswith("/v1/models")
        assert captured[0].method == "GET"

    def test_prints_model_ids(self, monkeypatch, capsys):
        monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen(_MODELS_PAYLOAD))
        cli.cmd_models(argparse.Namespace())
        out = capsys.readouterr().out
        assert "deepseek-v4-flash" in out
        assert "qwen3:3b" in out


# --------------------------------------------------------------------------- #
# cmd_health — GETs /health
# --------------------------------------------------------------------------- #

class TestCmdHealth:
    def test_gets_health_path(self, monkeypatch):
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return _FakeResponse(_HEALTH_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        cli.cmd_health(argparse.Namespace())

        assert captured[0].full_url.endswith("/health")
        assert captured[0].method == "GET"

    def test_prints_json(self, monkeypatch, capsys):
        monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen(_HEALTH_PAYLOAD))
        cli.cmd_health(argparse.Namespace())
        out = capsys.readouterr().out
        parsed = json.loads(out.strip())
        assert parsed == _HEALTH_PAYLOAD


# --------------------------------------------------------------------------- #
# cmd_use — POSTs /admin/swap
# --------------------------------------------------------------------------- #

class TestCmdUse:
    def _fake_urlopen_two_stage(self, status_resp, swap_resp):
        """First call -> status_resp (for engine key discovery), second -> swap_resp."""
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req)
            if len(calls) == 1:
                return _FakeResponse(status_resp)
            return _FakeResponse(swap_resp)

        return fake_urlopen

    def test_swap_posts_to_admin_swap(self, monkeypatch):
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            if len(captured) == 1:
                return _FakeResponse(_STATUS_PAYLOAD)
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        args = argparse.Namespace(command="use", target="ds4")
        cli.cmd_use(args)

        # Second call is the swap POST
        post_req = captured[1]
        assert post_req.full_url.endswith("/admin/swap")
        assert post_req.method == "POST"

    def test_swap_engine_key_sends_engine_field(self, monkeypatch):
        """When the target is an engine key, body must have 'engine', not 'model'."""
        sent_bodies = []

        def fake_urlopen(req, timeout=None):
            if req.method == "POST":
                sent_bodies.append(json.loads(req.data.decode()))
                return _FakeResponse(_STATUS_PAYLOAD)
            # GET /status
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        args = argparse.Namespace(command="use", target="ds4")
        cli.cmd_use(args)

        assert len(sent_bodies) == 1
        assert "engine" in sent_bodies[0]
        assert "model" not in sent_bodies[0]
        assert sent_bodies[0]["engine"] == "ds4"

    def test_swap_model_id_sends_model_field(self, monkeypatch):
        """When the target is NOT an engine key, body must have 'model'."""
        sent_bodies = []

        def fake_urlopen(req, timeout=None):
            if req.method == "POST":
                sent_bodies.append(json.loads(req.data.decode()))
                return _FakeResponse(_STATUS_PAYLOAD)
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        args = argparse.Namespace(command="use", target="some-random-model-id")
        cli.cmd_use(args)

        assert sent_bodies[0].get("model") == "some-random-model-id"
        assert "engine" not in sent_bodies[0]

    def test_swap_post_timeout_is_generous(self, monkeypatch):
        """Swap POST must use a long timeout (>=60s) to wait for cold starts."""
        timeouts = []

        def fake_urlopen(req, timeout=None):
            timeouts.append(timeout)
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        args = argparse.Namespace(command="use", target="ds4")
        cli.cmd_use(args)

        # The swap POST is the one with a long timeout (300s in current impl).
        swap_timeout = max(t for t in timeouts if t is not None)
        assert swap_timeout >= 60.0

    def test_ds4_shortcut_routes_to_use(self, monkeypatch):
        """The 'ds4' shortcut should behave identically to 'use ds4'."""
        sent_bodies = []

        def fake_urlopen(req, timeout=None):
            if req.method == "POST":
                sent_bodies.append(json.loads(req.data.decode()))
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        args = argparse.Namespace(command="ds4", target="ds4")
        cli.cmd_use(args)

        assert sent_bodies[0].get("engine") == "ds4"

    def test_ollama_shortcut_routes_to_use(self, monkeypatch):
        sent_bodies = []

        def fake_urlopen(req, timeout=None):
            if req.method == "POST":
                sent_bodies.append(json.loads(req.data.decode()))
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        args = argparse.Namespace(command="ollama", target="ollama")
        cli.cmd_use(args)

        assert sent_bodies[0].get("engine") == "ollama"


# --------------------------------------------------------------------------- #
# cmd_service — systemctl --user <action> <service-name>
#
# Name-agnostic: assert on command STRUCTURE, not the exact service name string.
# The parallel rename slice may change "local-engine-router.service" to something
# else; these tests must still pass.
# --------------------------------------------------------------------------- #

class TestCmdService:
    def _run_and_capture(self, monkeypatch, action: str):
        """Run cmd_service with the given action; capture the subprocess args."""
        captured_cmds = []

        def fake_run(cmd, check=False, **kw):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        args = argparse.Namespace(action=action)
        cli.cmd_service(args)
        return captured_cmds

    def _assert_systemctl_structure(self, cmd: list[str], expected_action: str) -> None:
        """Assert the command is 'systemctl --user <action> <any-name>'."""
        assert cmd[0] == "systemctl"
        assert "--user" in cmd
        assert expected_action in cmd
        # There must be exactly 4 elements and the action must come before the name.
        assert len(cmd) == 4
        user_idx = cmd.index("--user")
        action_idx = cmd.index(expected_action)
        assert user_idx < action_idx  # --user before action
        # The last element is the service name (we don't assert its exact value).
        assert cmd[-1] != ""

    def test_start_calls_systemctl_start(self, monkeypatch):
        cmds = self._run_and_capture(monkeypatch, "start")
        self._assert_systemctl_structure(cmds[0], "start")

    def test_stop_calls_systemctl_stop(self, monkeypatch):
        cmds = self._run_and_capture(monkeypatch, "stop")
        self._assert_systemctl_structure(cmds[0], "stop")

    def test_restart_calls_systemctl_restart(self, monkeypatch):
        cmds = self._run_and_capture(monkeypatch, "restart")
        self._assert_systemctl_structure(cmds[0], "restart")

    def test_service_uses_user_flag(self, monkeypatch):
        """Confirm --user is always present (no sudo, user-unit semantics)."""
        cmds = self._run_and_capture(monkeypatch, "start")
        assert "--user" in cmds[0]

    def test_systemctl_error_exits_nonzero(self, monkeypatch, capsys):
        def fake_run(cmd, check=False, **kw):
            if check:
                exc = subprocess.CalledProcessError(1, cmd)
                exc.returncode = 1
                raise exc
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        args = argparse.Namespace(action="start")
        with pytest.raises(SystemExit) as ei:
            cli.cmd_service(args)
        assert ei.value.code != 0

    def test_systemctl_oserror_exits_nonzero(self, monkeypatch, capsys):
        def fake_run(cmd, **kw):
            raise OSError("no systemctl")

        monkeypatch.setattr(subprocess, "run", fake_run)
        args = argparse.Namespace(action="stop")
        with pytest.raises(SystemExit) as ei:
            cli.cmd_service(args)
        assert ei.value.code != 0


# --------------------------------------------------------------------------- #
# cmd_logs — journalctl first, file fallback
#
# Also name-agnostic: journalctl invocation is checked for structure.
# --------------------------------------------------------------------------- #

class TestCmdLogs:
    def test_logs_tries_journalctl_first(self, monkeypatch):
        captured = []

        def fake_run(cmd, **kw):
            captured.append(list(cmd))

        monkeypatch.setattr(subprocess, "run", fake_run)
        args = argparse.Namespace()
        cli.cmd_logs(args)

        assert len(captured) >= 1
        first = captured[0]
        assert first[0] == "journalctl"
        assert "--user" in first
        assert "-u" in first
        # There must be a service name argument after -u (we don't assert its value).
        u_idx = first.index("-u")
        assert u_idx + 1 < len(first) and first[u_idx + 1] != ""

    def test_logs_falls_back_to_tail_on_oserror(self, monkeypatch):
        """If journalctl raises OSError, fall back to 'tail -f <log_file>'."""
        captured = []

        def fake_run(cmd, **kw):
            if cmd[0] == "journalctl":
                raise OSError("no journalctl")
            captured.append(list(cmd))

        monkeypatch.setattr(subprocess, "run", fake_run)
        args = argparse.Namespace()
        cli.cmd_logs(args)

        assert len(captured) == 1
        assert captured[0][0] == "tail"
        assert "-f" in captured[0]


# --------------------------------------------------------------------------- #
# Connection error path
# --------------------------------------------------------------------------- #

class TestConnectionError:
    def test_get_conn_refused_exits(self, monkeypatch, capsys):
        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(SystemExit) as ei:
            cli.cmd_status(argparse.Namespace())
        assert ei.value.code != 0
        err = capsys.readouterr().err
        assert "not reachable" in err or "router" in err.lower()

    def test_post_http_error_exits(self, monkeypatch, capsys):
        call_count = []

        def fake_urlopen(req, timeout=None):
            call_count.append(1)
            if req.method == "POST":
                fp = io.BytesIO(b"bad request")
                exc = urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, fp)
                raise exc
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        args = argparse.Namespace(command="use", target="something-unknown-123xyz")
        with pytest.raises(SystemExit) as ei:
            cli.cmd_use(args)
        assert ei.value.code != 0


# --------------------------------------------------------------------------- #
# Authorization header
# --------------------------------------------------------------------------- #

class TestAuthHeaders:
    def test_api_key_env_var_sent_as_bearer(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_KEY", "test-secret-key")
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        cli.cmd_status(argparse.Namespace())

        headers = dict(captured[0].headers)
        # urllib capitalizes header names: "Authorization"
        auth = headers.get("Authorization") or headers.get("authorization", "")
        assert auth.startswith("Bearer ")
        assert "test-secret-key" in auth

    def test_no_api_key_sends_no_auth_header(self, monkeypatch):
        # Clear both the env var and the config path so _api_key() returns None.
        monkeypatch.delenv("ROUTER_API_KEY", raising=False)
        monkeypatch.setenv("ROUTER_CONFIG", "/nonexistent/config.yaml")
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        cli.cmd_status(argparse.Namespace())

        headers = dict(captured[0].headers)
        assert "Authorization" not in headers and "authorization" not in headers


# --------------------------------------------------------------------------- #
# main() dispatch — smoke test the top-level entry point
# --------------------------------------------------------------------------- #

class TestMainDispatch:
    def test_main_status(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["routerctl", "status"])
        monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen(_STATUS_PAYLOAD))
        cli.main()
        assert "ds4" in capsys.readouterr().out

    def test_main_models(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["routerctl", "models"])
        monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen(_MODELS_PAYLOAD))
        cli.main()
        out = capsys.readouterr().out
        assert "deepseek-v4-flash" in out

    def test_main_health(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["routerctl", "health"])
        monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen(_HEALTH_PAYLOAD))
        cli.main()

    def test_main_start_calls_systemctl(self, monkeypatch):
        captured = []

        def fake_run(cmd, check=False, **kw):
            captured.append(list(cmd))
            return MagicMock(returncode=0)

        monkeypatch.setattr(sys, "argv", ["routerctl", "start"])
        monkeypatch.setattr(subprocess, "run", fake_run)
        cli.main()

        assert any("systemctl" in c and "start" in c for c in captured)

    def test_main_ds4_shortcut(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["routerctl", "ds4"])
        sent_bodies = []

        def fake_urlopen(req, timeout=None):
            if req.method == "POST":
                sent_bodies.append(json.loads(req.data.decode()))
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        cli.main()
        assert any(b.get("engine") == "ds4" for b in sent_bodies)

    def test_main_ollama_shortcut(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["routerctl", "ollama"])
        sent_bodies = []

        def fake_urlopen(req, timeout=None):
            if req.method == "POST":
                sent_bodies.append(json.loads(req.data.decode()))
            return _FakeResponse(_STATUS_PAYLOAD)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        cli.main()
        assert any(b.get("engine") == "ollama" for b in sent_bodies)
