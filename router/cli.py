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
import re
import subprocess
import sys
import tempfile
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
        with open(cfg_path, encoding="utf-8-sig") as fh:
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
    except urllib.error.HTTPError as exc:
        # HTTPError subclasses URLError: without this branch a 401/500 from a
        # RUNNING router would be misreported as "router not reachable".
        body_text = exc.read().decode(errors="replace")
        print(f"HTTP {exc.code} from router: {body_text}", file=sys.stderr)
        sys.exit(1)
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
    mode = st.get("routing_mode")
    if mode:
        smart = st.get("smart") or {}
        policy = smart.get("policy")
        line = f"routing mode  : {mode}"
        if mode == "smart" and policy:
            line += f" (policy: {policy})"
        print(line)
        pick = smart.get("last_pick")
        if mode == "smart" and pick:
            print(
                f"last pick     : {pick.get('requested_model')} -> "
                f"{pick.get('model')} on {pick.get('engine')} "
                f"(conf {pick.get('confidence')})"
            )
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
    # Try journalctl (user unit) first; fall back to tail -f of the log file
    # when journalctl is missing or exits non-zero (unknown unit, no journal
    # access). Ctrl-C means "stop following" — exit, don't fall through to
    # tailing the file.
    try:
        proc = subprocess.run(["journalctl", "--user", "-u", SERVICE_NAME, "-f"])
        # 0 = clean exit; -2 = killed by the same Ctrl-C SIGINT when the
        # parent's KeyboardInterrupt loses the race to run() returning.
        if proc.returncode in (0, -2):
            return
    except KeyboardInterrupt:
        return
    except OSError:
        pass
    try:
        subprocess.run(["tail", "-f", LOG_FILE])
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(f"could not tail {LOG_FILE}: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_discover(_args: argparse.Namespace) -> None:
    """POST /admin/discover and print per-engine discovered model ids."""
    result = _post("/admin/discover", {})
    _print_catalog_engines(result)


def _print_catalog_engines(result: dict) -> None:
    engines = result.get("engines") or {}
    if not engines:
        print("(no engines reported)")
        return
    for engine_key, model_ids in sorted(engines.items()):
        print(f"[{engine_key}]")
        if model_ids:
            for mid in model_ids:
                print(f"  {mid}")
        else:
            print("  (none)")


def cmd_catalog(_args: argparse.Namespace) -> None:
    """GET /admin/catalog and print the merged catalog."""
    result = _get("/admin/catalog")
    models = result.get("models") or []
    if not models:
        print("(catalog is empty)")
        return
    for m in models:
        parts = [m.get("id", "")]
        if m.get("engine"):
            parts.append(f"({m['engine']})")
        if m.get("source"):
            parts.append(f"source={m['source']}")
        if m.get("resolved_model"):
            parts.append(f"-> {m['resolved_model']}")
        if m.get("stale"):
            parts.append("stale")
        print("  " + "  ".join(parts))
        for note in m.get("collisions") or []:
            print(f"    collision: {note}")


def cmd_refresh(_args: argparse.Namespace) -> None:
    """POST /admin/discover and print the refreshed catalog by engine."""
    result = _post("/admin/discover", {})
    _print_catalog_engines(result)


def cmd_explain(args: argparse.Namespace) -> None:
    """POST /admin/resolve for a model id and print routing diagnostics."""
    result = _post("/admin/resolve", {"model": args.model})
    print(f"requested : {result.get('requested_model')}")
    print(f"resolved  : {result.get('resolved_model')}")
    print(f"engine    : {result.get('engine') or '(none)'}")
    print(f"source    : {result.get('source') or '(none)'}")
    print(f"would swap: {str(bool(result.get('would_swap'))).lower()}")
    reasons = result.get("reasons") or []
    if reasons:
        print("reasons:")
        for reason in reasons:
            print(f"  {reason}")
    collisions = result.get("collisions") or []
    if collisions:
        print("collisions:")
        for note in collisions:
            print(f"  {note}")


def _config_path() -> str:
    return os.environ.get("ROUTER_CONFIG", os.path.join(_REPO_ROOT, "config.yaml"))


def _validate_config_text(text: str) -> None:
    """Raise ValueError if *text* is not a valid router config (real loader)."""
    from .config import load_config

    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="routerctl-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        load_config(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def cmd_set_mode(mode: str) -> None:
    """`routerctl smart` / `routerctl manual`: persist routing_mode in the
    config file (validated before writing) and apply it to the running router
    via POST /admin/smart/mode so no restart is needed."""
    path = _config_path()
    updated_file = False
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as fh:
            text = fh.read()
        if re.search(r"(?m)^routing_mode\s*:", text):
            new_text = re.sub(
                r"(?m)^routing_mode\s*:.*$", f"routing_mode: {mode}", text, count=1
            )
        else:
            new_text = text.rstrip("\n") + (
                f"\n\n# Routing mode: smart (picker chooses the best local model"
                f" for smart\n# aliases / cloud names / unknown ids) or manual"
                f" (exact ids only).\nrouting_mode: {mode}\n"
            )
        try:
            _validate_config_text(new_text)
        except ValueError as exc:
            print(f"refusing to write {path}: config would be invalid: {exc}",
                  file=sys.stderr)
            sys.exit(1)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        updated_file = True
        print(f"config updated: routing_mode: {mode}  ({path})")
    else:
        print(
            f"note: config file not found at {path}; changing the runtime mode "
            "only (a restart will revert to the config default)",
            file=sys.stderr,
        )
    # Best-effort live apply; the config edit above already covers restarts.
    try:
        result = _post("/admin/smart/mode", {"mode": mode})
        print(f"router routing mode is now: {result.get('routing_mode', mode)}")
    except SystemExit:
        if updated_file:
            print("router not reachable; the mode applies on next start "
                  "(routerctl restart)", file=sys.stderr)
        else:
            sys.exit(1)


def cmd_explain_smart(args: argparse.Namespace) -> None:
    """POST /admin/smart/resolve and print the full smart-picker diagnostics."""
    body: dict[str, Any] = {"model": args.model}
    if getattr(args, "message", None):
        body["messages"] = [{"role": "user", "content": args.message}]
    if getattr(args, "endpoint", None):
        body["endpoint"] = args.endpoint
    result = _post("/admin/smart/resolve", body)

    if not result.get("smart_selection"):
        print(f"mode      : {result.get('mode')}")
        print("smart pick: no")
        print(f"reason    : {result.get('reason')}")
        return

    print(f"requested : {result.get('requested_model')}")
    print(f"picked    : {result.get('model')}  on {result.get('engine')}")
    print(f"policy    : {result.get('policy')}")
    print(f"confidence: {result.get('confidence')}")
    print(f"would swap: {str(bool(result.get('would_swap'))).lower()}"
          f"  (est. cost {result.get('swap_cost_s')}s)")
    job = result.get("job") or {}
    if job:
        top = sorted(job.items(), key=lambda kv: -kv[1])[:4]
        print("job       : " + ", ".join(f"{k}={v:.2f}" for k, v in top))
    for reason in result.get("reasons") or []:
        print(f"  - {reason}")
    candidates = result.get("candidates") or []
    if candidates:
        print("candidates:")
        for c in candidates:
            if c.get("excluded"):
                print(f"  {c['model']}  ({c['engine']})  EXCLUDED: {c['excluded']}")
            else:
                comp = c.get("components") or {}
                print(
                    f"  {c['model']}  ({c['engine']})  total={c['total']}"
                    f"  quality={comp.get('quality')}  swap_cost={c.get('swap_cost_s')}s"
                    + ("  [resident]" if c.get("resident") else "")
                )
    fallbacks = result.get("fallbacks") or []
    if fallbacks:
        print("fallbacks : " + ", ".join(fallbacks))
    benches = result.get("benchmarks") or []
    if benches:
        print("benchmark provenance:")
        for b in benches:
            print(
                f"  {b.get('capability')}: {b.get('score')} "
                f"({b.get('match')}, conf {b.get('confidence')}) "
                f"— {b.get('benchmark')}"
            )


def cmd_benchmarks(args: argparse.Namespace) -> None:
    """`routerctl benchmarks refresh|show|clear [model]`."""
    action = args.action
    model = getattr(args, "model", None)
    if action == "refresh":
        body = {"model": model} if model else {}
        result = _post("/admin/benchmarks/refresh", body, timeout=60.0)
        refreshed = result.get("refreshed") or []
        print(f"refreshed {result.get('count', len(refreshed))} model(s)")
        for canonical in refreshed:
            print(f"  {canonical}")
    elif action == "clear":
        body = {"model": model} if model else {}
        result = _post("/admin/benchmarks/clear", body)
        print(f"cleared {result.get('cleared', 0)} cache entr(y/ies)")
    else:  # show
        result = _get("/admin/benchmarks")
        models = result.get("models") or {}
        print(
            f"benchmark cache: {result.get('cached_models', len(models))} model(s); "
            f"providers: "
            + ", ".join(p.get("name", "?") for p in result.get("providers") or [])
        )
        for canonical, entry in models.items():
            if model and model not in (canonical, *entry.get("raw_ids", [])):
                continue
            print(f"[{canonical}]  (seen as: {', '.join(entry.get('raw_ids') or ['-'])})")
            for rec in entry.get("records") or []:
                print(
                    f"  {rec.get('capability'):<16} {rec.get('score'):<7} "
                    f"conf={rec.get('confidence')} {rec.get('match')} "
                    f"src={rec.get('source')}"
                )


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

    sub.add_parser(
        "init",
        help="setup wizard: detect running engines and scaffold a config.yaml",
        add_help=False,
    )
    sub.add_parser("status", help="show active engine and in-flight counts")
    sub.add_parser("models", help="list all known models (id, engine, name)")
    sub.add_parser("discover", help="scan all engines for discoverable models (POST /admin/discover)")
    sub.add_parser("catalog", help="show the merged router model catalog")
    sub.add_parser("refresh", help="refresh the model catalog now")
    explain_p = sub.add_parser(
        "explain",
        help="explain how a model id would route; with 'smart' (or --message) "
             "shows the full smart-picker decision",
    )
    explain_p.add_argument("model", help="model id, alias, or 'smart'")
    explain_p.add_argument(
        "--message", help="sample user message to classify (smart explain)"
    )
    explain_p.add_argument(
        "--endpoint", help="endpoint to classify for (default /v1/chat/completions)"
    )

    sub.add_parser(
        "smart",
        help="enable smart routing mode (config + running router)",
    )
    sub.add_parser(
        "manual",
        help="disable smart routing: exact model-id routing only",
    )
    bench_p = sub.add_parser(
        "benchmarks", help="manage the benchmark cache: refresh | show | clear"
    )
    bench_p.add_argument("action", choices=["refresh", "show", "clear"])
    bench_p.add_argument("model", nargs="?", help="limit to one model id")
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
    # `init` is the setup wizard; it owns its own argument parser, so intercept
    # it before the subcommand parser runs (which lets `routerctl init --config
    # X --yes ...` pass straight through to the wizard).
    argv = sys.argv[1:]
    if argv and argv[0] == "init":
        from . import wizard
        sys.exit(wizard.run_init(argv[1:]))

    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command
    if cmd == "status":
        cmd_status(args)
    elif cmd == "models":
        cmd_models(args)
    elif cmd == "discover":
        cmd_discover(args)
    elif cmd == "catalog":
        cmd_catalog(args)
    elif cmd == "refresh":
        cmd_refresh(args)
    elif cmd == "explain":
        if args.model == "smart" or getattr(args, "message", None):
            cmd_explain_smart(args)
        else:
            cmd_explain(args)
    elif cmd == "smart":
        cmd_set_mode("smart")
    elif cmd == "manual":
        cmd_set_mode("manual")
    elif cmd == "benchmarks":
        cmd_benchmarks(args)
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
