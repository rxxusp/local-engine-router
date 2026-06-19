"""Hermetic tests for the root install.sh bootstrap.

These never touch the network, pip, a venv, or systemd. They exercise only the
side-effect-free modes (--help, --dry-run, --print-unit) plus a bash syntax
check, and assert the generated systemd unit points at the venv (the new
bootstrap), not system python (the legacy deploy/ unit).
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL_SH = os.path.join(REPO_ROOT, "install.sh")

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="bash not available")


def _run(args, env_extra=None, **kw):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [bash, INSTALL_SH, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        **kw,
    )


def test_install_sh_exists_and_executable():
    assert os.path.exists(INSTALL_SH)
    assert os.access(INSTALL_SH, os.X_OK)


def test_bash_syntax_ok():
    # `bash -n` parses without executing; catches syntax errors.
    r = subprocess.run([bash, "-n", INSTALL_SH], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_help_exits_zero():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "Usage" in r.stdout
    assert "--no-service" in r.stdout
    assert "--uninstall" in r.stdout


def test_unknown_option_errors():
    r = _run(["--definitely-not-a-flag"])
    assert r.returncode == 2


def test_print_unit_targets_venv(tmp_path):
    venv = tmp_path / "venv"
    cfg = tmp_path / "sub" / "config.yaml"
    r = _run(
        ["--print-unit"],
        env_extra={"LER_VENV": str(venv), "LER_CONFIG": str(cfg)},
    )
    assert r.returncode == 0
    out = r.stdout
    # ExecStart uses the venv's console script, not /usr/bin/python3.
    assert f"ExecStart={venv}/bin/local-engine-router --config {cfg}" in out
    assert f"ROUTER_CONFIG={cfg}" in out
    # WorkingDirectory is the config's parent directory.
    assert f"WorkingDirectory={cfg.parent}" in out
    assert "/usr/bin/python3" not in out  # distinguishes from the legacy unit


def test_print_unit_is_deterministic(tmp_path):
    env = {"LER_VENV": str(tmp_path / "v"), "LER_CONFIG": str(tmp_path / "c.yaml")}
    a = _run(["--print-unit"], env_extra=env)
    b = _run(["--print-unit"], env_extra=env)
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_dry_run_makes_no_changes(tmp_path):
    venv = tmp_path / "venv"
    cfg = tmp_path / "cfgdir" / "config.yaml"
    bindir = tmp_path / "bin"
    r = _run(
        ["--dry-run"],
        env_extra={
            "LER_VENV": str(venv),
            "LER_CONFIG": str(cfg),
            "LER_BIN": str(bindir),
        },
    )
    assert r.returncode == 0
    assert "Dry run" in r.stdout
    assert str(venv) in r.stdout
    assert str(cfg) in r.stdout
    # No side effects: nothing was created.
    assert not venv.exists()
    assert not cfg.exists()
    assert not bindir.exists()


def test_dry_run_reports_no_service_when_skipped(tmp_path):
    r = _run(
        ["--dry-run", "--no-service"],
        env_extra={"LER_VENV": str(tmp_path / "v"), "LER_CONFIG": str(tmp_path / "c.yaml")},
    )
    assert r.returncode == 0
    assert "service     : skip" in r.stdout


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean():
    r = subprocess.run(
        ["shellcheck", "-S", "error", INSTALL_SH], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
