"""Configuration model and loader for llm-router.

The config is plain YAML (see config.yaml). Everything has a sensible default
baked in here so the YAML file can stay small. The dataclasses below are the
*contract* the rest of the package builds against:

  - RouterConfig.host / .port            where the router listens
  - RouterConfig.models                  list[ModelSpec], the static registry
  - RouterConfig.ds4 / .ollama           per-engine settings (legacy presets)
  - RouterConfig.engines                 optional generic engine table (see below)
  - build_model_index(cfg)               {model_id -> ModelSpec}

Routing is by the ``model`` field of each request. A model id is matched
against this static registry first; unknown ids fall back to a live Ollama
tag lookup at request time (see engines.EngineManager.engine_for).

Engine configuration
--------------------
Historically the router hardcoded two engine keys, ``ds4`` and ``ollama``,
configured via the top-level ``ds4:`` and ``ollama:`` sections. That still works
unchanged. To add *new* engines without touching Python, use the optional
top-level ``engines:`` table instead::

    engines:
      llamacpp:
        type: generic_process
        enabled: true
        base_url: http://127.0.0.1:8080
        start_cmd: ["/usr/local/bin/llama-server", "-m", "/models/foo.gguf"]
        ready_path: /health
        start_timeout_s: 300
      tabby:
        type: api_swap
        enabled: true
        base_url: http://127.0.0.1:5000
        unload_path: /v1/model/unload
        loaded_path: /v1/model

When ``engines:`` is present it is the sole source of engines and the legacy
``ds4:``/``ollama:`` sections are ignored. When it is absent the router falls
back to building ``ds4`` (from ``ds4:``) + ``ollama`` (from ``ollama:``) exactly
as before.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

import yaml

log = logging.getLogger("router.config")


# Engine ``type`` discriminator values understood by the generic engine table.
ENGINE_TYPES: frozenset[str] = frozenset(
    {"ds4", "ollama", "generic_process", "api_swap"}
)


class ConfigError(ValueError):
    """Raised for structural configuration problems with an actionable message.

    Subclasses ``ValueError`` so existing callers that catch ``ValueError``
    (the historical behaviour of ``load_config``) keep working.
    """


# --------------------------------------------------------------------------- #
# Dataclasses (the interface contract)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """A single model the router knows how to route, keyed by the exact id
    clients send in the request ``model`` field."""

    id: str
    engine: str  # engine key: must match a configured engine
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
    # Optional headers sent on every control/health call the router makes to
    # this engine (NOT user traffic). Default {} = unchanged (no auth header).
    control_headers: dict[str, str] = field(default_factory=dict)


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
    # Optional headers on the control client (e.g. a Bearer key if Ollama is
    # fronted by an authenticating reverse proxy). Default {} = none.
    control_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class GenericProcessConfig:
    """A local server process the router launches and supervises.

    Covers llama.cpp/llama-server, llamafile, vLLM, SGLang, Aphrodite — anything
    that is a single long-running HTTP server we can spawn and signal.
    """

    enabled: bool = True
    # Base URL the router uses to reach this engine.
    base_url: str = ""
    # Command to launch the server. Either a list (argv, run without a shell)
    # or a string (run through the shell). Required.
    start_cmd: list[str] | str = field(default_factory=list)
    # Extra environment variables for the launched process (merged over os.environ).
    env: dict[str, str] = field(default_factory=dict)
    # Working directory for the launched process (optional).
    cwd: str | None = None
    # Readiness probe path appended to base_url (e.g. /health, /v1/models).
    ready_path: str = "/health"
    # Optional richer readiness assertion beyond "HTTP 200". Two forms:
    #   "key==value"  -> the JSON response (or any nested object/list-of-objects)
    #                    must contain that key set to that value
    #                    (e.g. "status==ok" for llama-server's /health body).
    #   "model:<id>"  -> the model id <id> must appear in the response's model
    #                    list (so /v1/models-style readiness waits for the model
    #                    to actually be servable, not just the server to answer).
    #                    Use this for vLLM, whose /health returns an EMPTY 200
    #                    body before the model can serve — set ready_path=/v1/models
    #                    and ready_check="model:<id>" to avoid a false-ready swap.
    # Default "" = current behaviour: HTTP 200 is sufficient.
    ready_check: str = ""
    # Seconds to wait for a cold start to answer ready_path with HTTP 200.
    # vLLM/SGLang can take minutes; default generously. NOTE: SGLang with
    # torch.compile enabled can take >=600s on a cold start — raise this
    # per-engine when running such backends.
    start_timeout_s: float = 300.0
    # Signal used to ask the process group to stop (name or number; default SIGTERM).
    stop_signal: str = "SIGTERM"
    # Seconds to wait after stop_signal before escalating to SIGKILL, and the
    # overall budget for the port to confirm closed.
    stop_timeout_s: float = 30.0
    # Optional pgrep -f pattern to find/kill stray processes we may not own
    # (e.g. left behind by a previous run). Falls back to this if we have no
    # tracked Popen handle.
    process_pattern: str | None = None
    # Where the launched process' stdout/stderr is appended (optional).
    log_file: str | None = None
    # Optional headers sent on every control/health/readiness call the router
    # makes to this engine (NOT user traffic). Default {} = unchanged.
    control_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ApiSwapConfig:
    """An engine whose models load/unload over HTTP; the router owns no process.

    Generalises Ollama and also covers TabbyAPI-style load/unload. ``free_vram``
    is performed by calling the configured unload endpoint.
    """

    enabled: bool = True
    base_url: str = ""
    # Readiness probe path (does not load a model).
    health_path: str = "/v1/models"
    # Optional richer readiness assertion beyond "HTTP 200" (same forms as
    # GenericProcessConfig.ready_check: "key==value" or "model:<id>"). Applied
    # to the health_path response. Default "" = HTTP 200 is sufficient.
    ready_check: str = ""
    # Endpoint + method + body used to unload / free VRAM.
    unload_path: str = ""
    unload_method: str = "POST"
    # JSON body sent to unload_path. {model} in any string value is substituted
    # with each currently-loaded model name (when a list-loaded probe exists);
    # otherwise the body is sent once as-is.
    unload_body: dict[str, Any] = field(default_factory=dict)
    # Optional explicit per-model load endpoint (for engines that require a model
    # to be loaded before it can serve, e.g. TabbyAPI / text-generation-webui).
    # When set, the router loads the requested model on acquire (after the engine
    # is active) if it is not already loaded. Default "" = no explicit load
    # (JIT engines like Ollama load on first request and need nothing here).
    load_path: str = ""
    load_method: str = "POST"
    # JSON body sent to load_path. {model} in any string value is substituted
    # with the requested model id (same templating as unload_body).
    load_body: dict[str, Any] = field(default_factory=dict)
    # Seconds to wait for a single explicit load to complete (cold loads of a
    # large model can be slow; default generously).
    load_timeout_s: float = 120.0
    # Optional probe that lists currently-loaded models, so we can unload each
    # and confirm VRAM is released. path + the JSON key holding the list of
    # entries + the per-entry key holding the model name.
    loaded_path: str | None = None
    loaded_models_key: str = "models"
    loaded_name_key: str = "name"
    # Optional "key==value" filter applied to each loaded_path entry so only
    # ACTUALLY-loaded models are returned (e.g. "state==loaded" for engines that
    # list known-but-unloaded models too). Default "" = no filter (every entry).
    loaded_filter: str = ""
    # Optional per-entry field whose value is the UNLOAD identifier (distinct
    # from the display name), e.g. "instance_id" for LM Studio. When set,
    # loaded_models() returns these ids and {model} unload substitution uses
    # them. Default "" = key by loaded_name_key (the display name).
    loaded_id_key: str = ""
    # Seconds to wait for loaded models to clear after issuing unloads.
    unload_timeout_s: float = 60.0
    # Optional systemd unit to (best-effort) start if the API is unreachable.
    systemd_unit: str | None = None
    # TTL (seconds) for any cached list lookups (e.g. /v1/models for routing).
    tags_cache_ttl_s: float = 30.0
    # Optional headers sent on every control/health/load/unload/loaded call the
    # router makes to this engine (NOT user traffic) — e.g. an x-admin-key for a
    # secured TabbyAPI, or an Authorization bearer for LM Studio/LocalAI.
    # Default {} = unchanged (no auth header on control calls).
    control_headers: dict[str, str] = field(default_factory=dict)


# Maps an engine ``type`` to the dataclass holding its parameters.
_ENGINE_PARAM_CLASSES: dict[str, type] = {
    "ds4": Ds4Config,
    "ollama": OllamaConfig,
    "generic_process": GenericProcessConfig,
    "api_swap": ApiSwapConfig,
}


@dataclass
class EngineSpec:
    """One entry of the optional generic ``engines:`` table.

    ``key`` is the engine key (the table's mapping key). ``type`` selects the
    engine implementation. ``params`` is the type-specific config dataclass
    (one of Ds4Config / OllamaConfig / GenericProcessConfig / ApiSwapConfig).
    """

    key: str
    type: str
    enabled: bool = True
    params: Any = None

    @property
    def base_url(self) -> str:
        return getattr(self.params, "base_url", "") or ""


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
    # The /api/* catch-all forwards unmatched Ollama management endpoints.
    # Destructive ones (delete, create, copy, push, blobs) are refused with 403
    # unless this is true — otherwise ANY client that can reach the router can
    # delete/overwrite/upload local models, even with api_keys unset. /api/pull
    # stays available (it has an explicit route) but is covered by api_keys.
    allow_destructive_ollama_api: bool = False
    ds4: Ds4Config = field(default_factory=Ds4Config)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    # Optional generic engine table. Empty == legacy ds4/ollama mode (built from
    # the ds4:/ollama: sections above). Non-empty == engines built from here by
    # type, and ds4:/ollama: are ignored.
    engines: list[EngineSpec] = field(default_factory=list)
    models: list[ModelSpec] = field(default_factory=list)
    # Optional alias map {alias -> real model id}. A request for an alias routes
    # to the real model's engine, and the outgoing body's "model" is rewritten
    # to the real id before forwarding. Targets must resolve to a known model id
    # or a configured engine's model (unknown live-Ollama targets are allowed
    # with a warning). Alias->alias chains and malformed entries are rejected.
    # Default {} = no aliases.
    aliases: dict[str, str] = field(default_factory=dict)

    # Convenience -------------------------------------------------------- #
    def engine_keys(self) -> list[str]:
        """Engine keys that are configured AND enabled, in declaration order."""
        if self.engines:
            return [e.key for e in self.engines if e.enabled]
        keys = []
        if self.ds4.enabled:
            keys.append("ds4")
        if self.ollama.enabled:
            keys.append("ollama")
        return keys


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _coerce_section(cls, data: dict[str, Any] | None, *, ctx: str | None = None):
    """Build a dataclass from a dict, ignoring unknown keys (forward-compat)."""
    if not data:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(data) - known
    if unknown:
        where = ctx or cls.__name__
        log.warning("ignoring unknown %s keys: %s", where, sorted(unknown))
    return cls(**{k: v for k, v in data.items() if k in known})


def _required_fields_present(params: Any, required: tuple[str, ...]) -> list[str]:
    """Return the subset of *required* attribute names that are empty/unset."""
    missing = []
    for name in required:
        val = getattr(params, name, None)
        if val in (None, "", [], {}):
            missing.append(name)
    return missing


def _build_engines_section(raw_engines: Any) -> list[EngineSpec]:
    """Parse the optional top-level ``engines:`` mapping into EngineSpecs.

    Validates the engine ``type`` discriminator, duplicate keys, and the
    required fields for each engine type. Raises ConfigError on structural
    problems; logs soft warnings for non-fatal issues (e.g. paths missing).
    """
    if not raw_engines:
        return []
    if not isinstance(raw_engines, dict):
        raise ConfigError(
            "'engines' must be a mapping of engine_key -> engine settings"
        )

    specs: list[EngineSpec] = []
    seen: set[str] = set()
    for key, body in raw_engines.items():
        if key in seen:
            raise ConfigError(f"duplicate engine key {key!r} in 'engines'")
        seen.add(key)
        if not isinstance(body, dict):
            raise ConfigError(
                f"engine {key!r}: settings must be a mapping, got {type(body).__name__}"
            )
        etype = body.get("type")
        if not etype:
            raise ConfigError(
                f"engine {key!r}: missing required 'type' "
                f"(one of {sorted(ENGINE_TYPES)})"
            )
        if etype not in ENGINE_TYPES:
            raise ConfigError(
                f"engine {key!r}: unknown type {etype!r} "
                f"(must be one of {sorted(ENGINE_TYPES)})"
            )

        enabled = bool(body.get("enabled", True))
        param_cls = _ENGINE_PARAM_CLASSES[etype]
        # Everything except the discriminator/enabled flag is engine params.
        param_data = {k: v for k, v in body.items() if k not in ("type", "enabled")}
        params = _coerce_section(
            param_cls, param_data, ctx=f"engine {key!r} ({etype})"
        )
        # Keep params.enabled in sync with the spec-level flag for consistency.
        if hasattr(params, "enabled"):
            params.enabled = enabled

        _validate_engine_params(key, etype, params)
        specs.append(EngineSpec(key=key, type=etype, enabled=enabled, params=params))

    return specs


def _validate_engine_params(key: str, etype: str, params: Any) -> None:
    """Validate required fields for a single engine; warn on soft issues."""
    if etype == "generic_process":
        missing = _required_fields_present(params, ("base_url", "start_cmd"))
        if missing:
            raise ConfigError(
                f"engine {key!r} (generic_process): missing required field(s): "
                f"{', '.join(missing)}"
            )
    elif etype == "api_swap":
        missing = _required_fields_present(params, ("base_url",))
        if missing:
            raise ConfigError(
                f"engine {key!r} (api_swap): missing required field(s): "
                f"{', '.join(missing)}"
            )
        if not getattr(params, "unload_path", ""):
            log.warning(
                "engine %r (api_swap): no 'unload_path' set; free_vram() will be "
                "a no-op (fine if this engine never needs to release the GPU)",
                key,
            )
    elif etype == "ds4":
        # ds4 has defaults for everything; only sanity-check serve_script when in
        # process-control mode.
        if getattr(params, "control", "") == "process":
            script = getattr(params, "serve_script", "")
            if script and not os.path.exists(script):
                log.warning(
                    "engine %r (ds4): serve_script %s does not exist", key, script
                )
    elif etype == "ollama":
        if not getattr(params, "base_url", ""):
            raise ConfigError(f"engine {key!r} (ollama): missing required 'base_url'")


def load_config(path: str) -> RouterConfig:
    """Load YAML config from *path*, applying defaults for anything omitted.

    Validates structural problems and raises ConfigError (a ValueError) with an
    actionable message: a model.engine that references no configured engine, an
    unknown engine type, a missing required field for an engine type, or a
    duplicate engine key. Non-fatal issues (e.g. a serve_script that does not
    exist) are logged as warnings.
    """
    raw: dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        log.warning("config file %s not found; using built-in defaults", path)

    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")

    ds4 = _coerce_section(Ds4Config, raw.get("ds4"))
    ollama = _coerce_section(OllamaConfig, raw.get("ollama"))
    engines = _build_engines_section(raw.get("engines"))

    models: list[ModelSpec] = []
    for m in raw.get("models", []) or []:
        if "id" not in m:
            raise ConfigError("every model entry must have an 'id'")
        if "engine" not in m:
            raise ConfigError(
                f"model {m['id']!r} must specify an 'engine'"
            )
        models.append(
            ModelSpec(
                id=m["id"],
                engine=m["engine"],
                display_name=m.get("display_name", m["id"]),
                context_length=int(m.get("context_length", 131072)),
            )
        )

    skip = {"ds4", "ollama", "engines", "models"}
    top = {
        k: v
        for k, v in raw.items()
        if k in RouterConfig.__dataclass_fields__ and k not in skip
    }
    cfg = RouterConfig(
        ds4=ds4, ollama=ollama, engines=engines, models=models, **top
    )

    # Validate model -> engine references against whatever engines are configured.
    if cfg.engines:
        valid_engines = {e.key for e in cfg.engines}
    else:
        valid_engines = {"ds4", "ollama"}
    for spec in cfg.models:
        if spec.engine not in valid_engines:
            raise ConfigError(
                f"model {spec.id!r} references unknown engine {spec.engine!r} "
                f"(configured engines: {sorted(valid_engines)})"
            )

    _validate_aliases(cfg)
    return cfg


def _validate_aliases(cfg: RouterConfig) -> None:
    """Validate cfg.aliases ({alias -> real model id}).

    Hard-fails (ConfigError) on a malformed entry or an alias whose target is
    itself another alias (no chains). Soft-warns when a target does not resolve
    to a known model id — live Ollama tags resolve at runtime, so an unknown
    target is not necessarily an error.
    """
    aliases = cfg.aliases or {}
    if not isinstance(aliases, dict):
        raise ConfigError("'aliases' must be a mapping of alias -> real model id")

    known_ids = {m.id for m in cfg.models}
    alias_keys = set(aliases.keys())
    for alias, target in aliases.items():
        if not isinstance(alias, str) or not alias:
            raise ConfigError(f"alias key {alias!r} must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise ConfigError(
                f"alias {alias!r} target must be a non-empty model id string"
            )
        if target == alias:
            raise ConfigError(f"alias {alias!r} points at itself")
        if alias in known_ids:
            raise ConfigError(
                f"alias {alias!r} collides with a configured model id; an alias "
                f"key must not shadow a real model (it would silently rewrite "
                f"every request for that model to {target!r})"
            )
        if target in alias_keys:
            raise ConfigError(
                f"alias {alias!r} -> {target!r} is a chain "
                f"(its target is itself an alias); aliases must point at a "
                f"real model id, not another alias"
            )
        if target not in known_ids:
            log.warning(
                "alias %r -> %r: target is not a known model id "
                "(ok if it resolves at runtime, e.g. a live Ollama tag)",
                alias,
                target,
            )


def build_model_index(cfg: RouterConfig) -> dict[str, ModelSpec]:
    """Return {model_id -> ModelSpec} from the static registry."""
    index: dict[str, ModelSpec] = {}
    for spec in cfg.models:
        if spec.id in index:
            log.warning("duplicate model id %r in config; later wins", spec.id)
        index[spec.id] = spec
    return index


# --------------------------------------------------------------------------- #
# JSON Schema (draft 2020-12), derived from the dataclasses
# --------------------------------------------------------------------------- #
def _json_type_for(anno: Any) -> dict[str, Any]:
    """Map a dataclass field annotation to a JSON Schema type fragment.

    Best-effort: handles the concrete annotations used by our dataclasses
    (str, int, float, bool, list[...], dict[...], optionals, unions).
    """
    # Annotations are stored as strings (``from __future__ import annotations``).
    text = anno if isinstance(anno, str) else getattr(anno, "__name__", str(anno))
    text = text.replace(" ", "")

    # Split top-level unions first (bracket-depth-aware) so "list[str]|str" is
    # treated as a union of two arms, not as a list with a malformed inner type.
    arms = _split_union(text)
    optional = "None" in arms
    arms = [a for a in arms if a != "None"]

    if not arms:  # was bare ``None``
        return {"type": "null"}
    if len(arms) > 1:
        frag: dict[str, Any] = {"anyOf": [_atom_type(a) for a in arms]}
    else:
        frag = _atom_type(arms[0])

    if optional and frag:
        # Permit null in addition to the declared shape.
        if "anyOf" in frag:
            frag["anyOf"].append({"type": "null"})
        else:
            frag = {"anyOf": [frag, {"type": "null"}]}
    return frag


def _split_union(text: str) -> list[str]:
    """Split a type string on top-level ``|`` (ignoring ``|`` inside brackets)."""
    arms: list[str] = []
    depth = 0
    cur = ""
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "|" and depth == 0:
            arms.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        arms.append(cur)
    return arms


def _atom_type(text: str) -> dict[str, Any]:
    """Map a single (non-union) type atom to a JSON Schema fragment."""
    scalars = {
        "str": {"type": "string"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "bool": {"type": "boolean"},
        "Any": {},
    }
    if text in scalars:
        return dict(scalars[text])
    if text.startswith("list["):
        inner = text[len("list[") : -1]
        return {"type": "array", "items": _json_type_for(inner)}
    if text.startswith("dict["):
        return {"type": "object", "additionalProperties": True}
    # Unknown / parameterised generic: accept anything.
    return {}


def _schema_for_dataclass(cls: type, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Build a JSON Schema object node from a dataclass' fields + defaults."""
    props: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in exclude:
            continue
        frag = _json_type_for(f.type)
        # Attach the default value as a documentation hint where it's a simple
        # scalar (skip factory defaults / dataclass instances).
        if f.default is not MISSING and isinstance(
            f.default, (str, int, float, bool)
        ):
            frag = {**frag, "default": f.default}
        props[f.name] = frag
    return {
        "type": "object",
        "properties": props,
        "additionalProperties": True,  # forward-compat: unknown keys warned, not rejected
    }


def config_json_schema() -> dict[str, Any]:
    """Return a JSON Schema (draft 2020-12) describing the full config.

    Derived from the dataclasses, including the generic engine types in the
    optional ``engines:`` table. ``additionalProperties`` is left open because
    the loader treats unknown keys as a soft warning (forward-compat).
    """
    ds4_schema = _schema_for_dataclass(Ds4Config)
    ollama_schema = _schema_for_dataclass(OllamaConfig)
    generic_schema = _schema_for_dataclass(GenericProcessConfig)
    apiswap_schema = _schema_for_dataclass(ApiSwapConfig)

    def _with_type(node: dict[str, Any], type_const: str) -> dict[str, Any]:
        node = {
            **node,
            "properties": {
                "type": {"const": type_const},
                **node["properties"],
            },
            "required": ["type"],
        }
        return node

    engine_entry = {
        "oneOf": [
            _with_type(ds4_schema, "ds4"),
            _with_type(ollama_schema, "ollama"),
            _with_type(generic_schema, "generic_process"),
            _with_type(apiswap_schema, "api_swap"),
        ],
    }

    model_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "engine": {"type": "string"},
            "display_name": {"type": "string"},
            "context_length": {"type": "integer", "default": 131072},
        },
        "required": ["id", "engine"],
        "additionalProperties": True,
    }

    root = _schema_for_dataclass(
        RouterConfig, exclude=("ds4", "ollama", "engines", "models")
    )
    root["properties"]["ds4"] = ds4_schema
    root["properties"]["ollama"] = ollama_schema
    root["properties"]["engines"] = {
        "type": "object",
        "description": (
            "Optional generic engine table: engine_key -> engine settings. "
            "When present it is the sole source of engines and the ds4:/ollama: "
            "sections are ignored."
        ),
        "additionalProperties": engine_entry,
    }
    root["properties"]["models"] = {"type": "array", "items": model_schema}

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/rxxusp/local-engine-router/config.schema.json",
        "title": "llm-router configuration",
        "description": (
            "Configuration schema for llm-router (local-engine-router). Unknown "
            "keys are accepted with a warning for forward compatibility."
        ),
        **root,
    }


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
