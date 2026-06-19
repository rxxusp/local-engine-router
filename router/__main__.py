"""Entry point: ``python3 -m router --config config.yaml``.

Loads config, configures logging, builds the FastAPI app, and serves it with
uvicorn on the configured host/port.

Two non-serving utility modes are also available:
  --print-schema   print the config JSON Schema to stdout and exit
  --check-config   load + validate the config, print 'OK' or the error and exit
                   (non-zero on error)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import ConfigError, config_json_schema, configure_logging, load_config

# Default to the checkout's config.yaml (router/__main__.py -> repo/);
# $ROUTER_CONFIG or --config override.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.environ.get(
    "ROUTER_CONFIG", os.path.join(_REPO_ROOT, "config.yaml")
)


def main() -> None:
    # `init` is the interactive setup wizard (detect engines + scaffold a
    # config). It has its own argument parser, so intercept it before this
    # serve-oriented one runs: `local-engine-router init [--config X --yes ...]`.
    argv = sys.argv[1:]
    if argv and argv[0] == "init":
        from .wizard import run_init
        sys.exit(run_init(argv[1:]))

    ap = argparse.ArgumentParser(prog="router", description="local-engine LLM router")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    ap.add_argument(
        "--print-schema",
        action="store_true",
        help="print the config JSON Schema (draft 2020-12) to stdout and exit",
    )
    ap.add_argument(
        "--check-config",
        action="store_true",
        help="validate the config, print 'OK' or the error, and exit (non-zero on error)",
    )
    args = ap.parse_args()

    # --print-schema: emit schema and exit before touching logging/config.
    if args.print_schema:
        json.dump(config_json_schema(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # --check-config: load + validate, report, and exit with a status code.
    if args.check_config:
        try:
            cfg = load_config(args.config)
        except (ConfigError, ValueError) as exc:
            print(f"config error: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
            print(f"config error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"OK ({len(cfg.models)} model(s), engines: {cfg.engine_keys()})")
        return

    cfg = load_config(args.config)
    configure_logging(cfg)
    log = logging.getLogger("router")
    log.info("local-engine-router starting on %s:%s (config=%s)", cfg.host, cfg.port, args.config)

    # Imported here so logging is configured first.
    from .app import create_app

    import uvicorn

    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
