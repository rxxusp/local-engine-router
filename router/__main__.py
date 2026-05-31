"""Entry point: ``python3 -m router --config config.yaml``.

Loads config, configures logging, builds the FastAPI app, and serves it with
uvicorn on the configured host/port.
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .config import configure_logging, load_config

DEFAULT_CONFIG = os.environ.get(
    "ROUTER_CONFIG", "/home/grahamfm/llm-router/config.yaml"
)


def main() -> None:
    ap = argparse.ArgumentParser(prog="router", description="ds4 <-> Ollama LLM router")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg)
    log = logging.getLogger("router")
    log.info("llm-router starting on %s:%s (config=%s)", cfg.host, cfg.port, args.config)

    # Imported here so logging is configured first.
    from .app import create_app

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
