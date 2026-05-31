"""Configuration model and loader for llm-router.

The config is plain YAML (see config.yaml). Everything has a sensible default
baked in here so the YAML file can stay small. The dataclasses below are the
*contract* the rest of the package builds against:

  - RouterConfig.host / .port            where the router listens
  - RouterConfig.models                  list[ModelSpec], the static registry
  - RouterConfig.ds4 / .ollama           per-engine settings
  - build_model_index(cfg)               {model_id -> ModelSpec}

Routing is by the ``model`` field of each request. A model id is matched
against this static registry first; unknown ids fall back to a live Ollama
tag lookup at request time (see engines.EngineManager.engine_for).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

log = logging.getLogger("router.config")


# --------------------------------------------------------------------------- #
# Dataclasses (the interface contract)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """A single model the router knows how to route, keyed by the exact id
    clients send in the request ``model`` field."""

    id: str
    engine: str  # engine key: "ds4" | "ollama"
    display_name: str
    context_length: int = 131072


@dataclass
class Ds4Config:
    enabled: bool = True
    # Base URL the *router* uses to reach ds4 (host networking).
    base_url: str = "http://172.17.0.1:8099"
    # How the router controls ds4's lifecycle:
    #   "systemd-user" -> start/stop the user unit (default; ds4 is managed by a
    #                     `systemctl --user` service with Restart=always, so a
    #                     plain SIGTERM would just get respawned).
    #   "process"      -> launch serve_script + SIGTERM the process directly
    #                     (fallback for setups where ds4 is not a service).
    control: str = "systemd-user"
    # The user systemd unit that runs ds4 (control="systemd-user").
    systemd_user_unit: str = "ds4.service"
    # Script that launches ds4-server (control="process"; it `exec`s the binary).
    serve_script: str = "/home/grahamfm/ds4/serve.sh"
    # pgrep -f pattern used to find/stop the ds4-server process (control="process").
    process_pattern: str = "ds4/ds4-server"
    # Path used as a readiness probe (ds4 has no /health; /v1/models returns 200).
    health_path: str = "/v1/models"
    # Seconds to wait for ds4 to become ready after we (re)start it. The model
    # is ~81 GB so a cold start can take a while.
    start_timeout_s: float = 240.0
    # Seconds to wait for the process to exit + VRAM to free after stop.
    stop_timeout_s: float = 45.0
    # Where ds4-server stdout/stderr is appended when control="process".
    log_file: str = "/home/grahamfm/llm-router/logs/ds4-server.log"


@dataclass
class OllamaConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    # Native readiness probe (does not load any model).
    health_path: str = "/api/tags"
    # Wait for loaded models to unload (free VRAM) before we hand the GPU to ds4.
    unload_timeout_s: float = 60.0
    # systemd unit name; used only to (best-effort) start it if it's down.
    systemd_unit: str = "ollama.service"
    # TTL (seconds) for the cached /api/tags lookup used by routing fallback.
    tags_cache_ttl_s: float = 30.0


@dataclass
class RouterConfig:
    # Safe default: localhost only. Set to 0.0.0.0 explicitly to expose the
    # router off-localhost (e.g. to reach it from a Docker container via the
    # bridge gateway) — pair that with api_keys, or a host firewall.
    host: str = "127.0.0.1"
    port: int = 8077
    # Optional API keys. If non-empty, every request except GET /health must
    # present one via `Authorization: Bearer <key>` or `X-API-Key: <key>`.
    # Empty list = no authentication (fine for a localhost-only bind).
    api_keys: list[str] = field(default_factory=list)
    log_file: str = "/home/grahamfm/llm-router/logs/router.log"
    log_level: str = "INFO"
    # Persisted observability snapshot (active engine, last swap). Not trusted
    # as ground truth on startup — the manager re-probes reality.
    state_file: str = "/home/grahamfm/llm-router/state.json"
    # Cadence of SSE keep-alive comments emitted to streaming clients while a
    # swap is in progress, so they don't hit an idle/TTFB timeout.
    swap_keepalive_interval_s: float = 5.0
    # Whether to emit those keep-alive comments at all.
    swap_keepalive_enabled: bool = True
    # Wait for in-flight requests on an engine to drain before stopping it.
    drain_timeout_s: float = 30.0
    # After freeing one engine, wait for the kernel to reclaim its (unified)
    # memory before starting the next engine — otherwise the incoming model's
    # pre-flight memory check can fail (on a GB10 reclaiming ~81 GB takes ~2-3s).
    # The wait ends as soon as MemAvailable plateaus, capped at this timeout.
    swap_memory_settle_timeout_s: float = 25.0
    # Upstream connect timeout for user traffic. Read timeout is intentionally
    # unbounded (long generations / streaming).
    upstream_connect_timeout_s: float = 15.0
    ds4: Ds4Config = field(default_factory=Ds4Config)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    models: list[ModelSpec] = field(default_factory=list)

    # Convenience -------------------------------------------------------- #
    def engine_keys(self) -> list[str]:
        keys = []
        if self.ds4.enabled:
            keys.append("ds4")
        if self.ollama.enabled:
            keys.append("ollama")
        return keys


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _coerce_section(cls, data: dict[str, Any] | None):
    """Build a dataclass from a dict, ignoring unknown keys (forward-compat)."""
    if not data:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(data) - known
    if unknown:
        log.warning("ignoring unknown %s keys: %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str) -> RouterConfig:
    """Load YAML config from *path*, applying defaults for anything omitted."""
    raw: dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        log.warning("config file %s not found; using built-in defaults", path)

    ds4 = _coerce_section(Ds4Config, raw.get("ds4"))
    ollama = _coerce_section(OllamaConfig, raw.get("ollama"))

    models: list[ModelSpec] = []
    for m in raw.get("models", []) or []:
        models.append(
            ModelSpec(
                id=m["id"],
                engine=m["engine"],
                display_name=m.get("display_name", m["id"]),
                context_length=int(m.get("context_length", 131072)),
            )
        )

    top = {
        k: v
        for k, v in raw.items()
        if k in RouterConfig.__dataclass_fields__ and k not in ("ds4", "ollama", "models")
    }
    cfg = RouterConfig(ds4=ds4, ollama=ollama, models=models, **top)

    # Validate engine references.
    valid_engines = {"ds4", "ollama"}
    for spec in cfg.models:
        if spec.engine not in valid_engines:
            raise ValueError(
                f"model {spec.id!r} references unknown engine {spec.engine!r} "
                f"(must be one of {sorted(valid_engines)})"
            )
    return cfg


def build_model_index(cfg: RouterConfig) -> dict[str, ModelSpec]:
    """Return {model_id -> ModelSpec} from the static registry."""
    index: dict[str, ModelSpec] = {}
    for spec in cfg.models:
        if spec.id in index:
            log.warning("duplicate model id %r in config; later wins", spec.id)
        index[spec.id] = spec
    return index


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def configure_logging(cfg: RouterConfig) -> None:
    """Configure root logging to stdout (journald) and a rotating file."""
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Avoid duplicate handlers if called twice.
    for h in list(root.handlers):
        root.removeHandler(h)

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    try:
        os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
        fileh = logging.handlers.RotatingFileHandler(
            cfg.log_file, maxBytes=5_000_000, backupCount=3
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)
    except OSError as exc:  # pragma: no cover - best effort
        log.warning("could not open log file %s: %s", cfg.log_file, exc)

    # uvicorn access logs are noisy; keep them at WARNING.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
