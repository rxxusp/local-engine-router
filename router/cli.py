"""routerctl — CLI control tool for local-engine-router.

Uses only stdlib: urllib.request, json, argparse, subprocess. (PyYAML — already
a router dependency — is used opportunistically to discover the API key from
the router config; its absence is tolerated.)
Base URL: $ROUTER_URL or http://127.0.0.1:8077
API key:  $ROUTER_API_KEY, else api_keys[0] from $ROUTER_CONFIG /
          <repo>/config.yaml (sent as Authorization: Bearer when present)

This is the importable home of the ``routerctl`` command. The top-level
``./routerctl`` script in a checkout is a thin shim that calls :func:`main`
here, and the ``routerctl`` console-script entry point installed by
``pyproject.toml`` does the same.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

BASE_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8077").rstrip("/")
# The systemd user unit name — single definition so a future rename stays in
# one place.
SERVICE_NAME = "local-engine-router"
# Repo root when running from a checkout (router/cli.py -> repo/). Defaults
# derive from it so a checkout needs no env vars; both are overridable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.environ.get(
    "ROUTER_LOG_FILE", os.path.join(_REPO_ROOT, "logs", "router.log")
)


def _api_key() -> str | None:
    """API key for the router, if it requires one.

    $ROUTER_API_KEY wins; otherwise a best-effort read of api_keys[0] from the
    router config ($ROUTER_CONFIG or <repo>/config.yaml)."""
    key = os.environ.get("ROUTER_API_KEY")
    if key:
        return key
    cfg_path = os.environ.get(
        "ROUTER_CONFIG", os.path.join(_REPO_ROOT, "config.yaml")
    )
    try:
        import yaml
        with open(cfg_path) as fh:
            keys = (yaml.safe_load(fh) or {}).get("api_keys") or []
        return str(keys[0]) if keys else None
    except Exception:
        return None


def _auth_headers() -> dict[str, str]:
    key = _api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def _get(path: str, timeout: float = 15.0) -> Any:
    """GET BASE_URL+path, return parsed JSON."""
    url = BASE_URL + path
    req = urllib.request.Request(url, headers=_auth_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionRefusedError, OSError) as exc:
        _conn_error(exc)


def _post(path: str, body: dict, timeout: float = 15.0) -> Any:
    """POST body as JSON to BASE_URL+path, return parsed JSON."""
    url = BASE_URL + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **_auth_headers()},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        print(f"HTTP {exc.code} from router: {body_text}", file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, ConnectionRefusedError, OSError) as exc:
        _conn_error(exc)


def _conn_error(exc: Exception) -> None:
    print(
        f"router not reachable at {BASE_URL} ({exc})\n"
        "  Is the service running?  Try: routerctl start",
        file=sys.stderr,
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Pretty printers
# --------------------------------------------------------------------------- #

def _print_status(st: dict) -> None:
    active = st.get("active_engine")
    print(f"active engine : {active or '(none)'}")
    last = st.get("last_swap")
    if last:
        ok_str = "OK" if last.get("ok") else "FAILED"
        print(
            f"last swap     : {last.get('from')} -> {last.get('to')} "
            f"in {last.get('duration_s')}s [{ok_str}]"
        )
    print()

    engines = st.get("engines") or {}
    for key, info in engines.items():
        marker = " *" if key == active else "  "
        ready = "ready" if info.get("ready") else "NOT READY"
        inflight = info.get("in_flight", 0)
        line = f"{marker} [{key}]  {ready}  in_flight={inflight}  {info.get('base_url', '')}"
        print(line)
        loaded = info.get("loaded_models")
        if loaded:
            for m in loaded:
                print(f"       loaded: {m}")
        running = info.get("process_running")
        if running is not None:
            print(f"       process_running: {running}")

    models = st.get("models") or []
    if models:
        print()
        print("models:")
        for m in models:
            print(f"  {m['id']}  ({m['engine']})  —  {m.get('name', '')}")


def _print_models(data: dict) -> None:
    for m in data.get("data") or []:
        mid = m.get("id", "")
        # Best-effort: owned_by carries the engine key when available.
        owned = m.get("owned_by", "")
        clen = m.get("context_length") or ""
        parts = [mid]
        if owned:
            parts.append(f"({owned})")
        if clen:
            parts.append(f"ctx={clen}")
        print("  " + "  ".join(parts))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_status(_args: argparse.Namespace) -> None:
    st = _get("/status")
    _print_status(st)


def cmd_models(_args: argparse.Namespace) -> None:
    data = _get("/v1/models")
    _print_models(data)


def cmd_use(args: argparse.Namespace) -> None:
    target = args.target
    # Decide whether the target is an engine KEY or a model id. Engine keys are
    # discovered live from /status (so generic engines:-table keys like
    # "llamacpp"/"tabby" work, not just the literal ds4/ollama), with a static
    # fallback if the router can't be reached for the lookup.
    engine_keys = {"ds4", "ollama"}
    try:
        st0 = _get("/status")
        engine_keys = set((st0.get("engines") or {}).keys()) or engine_keys
    except SystemExit:
        pass  # _get already printed a connection error; fall back to defaults
    if target in engine_keys:
        body: dict = {"engine": target}
    else:
        body = {"model": target}
    print(f"swapping to {target!r}...  (this may take a while for a cold engine swap)")
    st = _post("/admin/swap", body, timeout=300.0)
    _print_status(st)


def cmd_health(_args: argparse.Namespace) -> None:
    data = _get("/health")
    print(json.dumps(data))


def cmd_logs(_args: argparse.Namespace) -> None:
    # Try journalctl (user unit) first; fall back to tail -f of the log file.
    try:
        subprocess.run(["journalctl", "--user", "-u", SERVICE_NAME, "-f"])
    except (OSError, KeyboardInterrupt):
        try:
            subprocess.run(["tail", "-f", LOG_FILE])
        except KeyboardInterrupt:
            pass
        except OSError as exc:
            print(f"could not tail {LOG_FILE}: {exc}", file=sys.stderr)
            sys.exit(1)


def cmd_service(args: argparse.Namespace) -> None:
    action = args.action
    # The router is a *user* unit, so no sudo and the --user flag.
    try:
        subprocess.run(
            ["systemctl", "--user", action, f"{SERVICE_NAME}.service"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"systemctl --user {action} failed (exit {exc.returncode})", file=sys.stderr)
        sys.exit(exc.returncode)
    except OSError as exc:
        print(f"could not run systemctl --user: {exc}", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="routerctl",
        description="Control tool for local-engine-router. Base URL: $ROUTER_URL or http://127.0.0.1:8077",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show active engine and in-flight counts")
    sub.add_parser("models", help="list all known models (id, engine, name)")
    sub.add_parser("health", help="check router liveness (GET /health)")
    sub.add_parser("logs", help="tail the router log (journalctl or file fallback)")

    use_p = sub.add_parser("use", help="swap to an engine or model")
    use_p.add_argument("target", help="an engine key (e.g. ds4, ollama, or a generic engines: key) or a model id")

    # Convenience shortcuts for the two built-in engines.
    sub.add_parser("ds4", help="shortcut: swap to ds4 engine")
    sub.add_parser("ollama", help="shortcut: swap to ollama engine")

    for action in ("start", "stop", "restart"):
        sub.add_parser(action, help=f"systemctl --user {action} {SERVICE_NAME}.service")

    return p


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command
    if cmd == "status":
        cmd_status(args)
    elif cmd == "models":
        cmd_models(args)
    elif cmd == "use":
        cmd_use(args)
    elif cmd == "health":
        cmd_health(args)
    elif cmd == "logs":
        cmd_logs(args)
    elif cmd == "ds4":
        args.target = "ds4"
        cmd_use(args)
    elif cmd == "ollama":
        args.target = "ollama"
        cmd_use(args)
    elif cmd in ("start", "stop", "restart"):
        args.action = cmd
        cmd_service(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
