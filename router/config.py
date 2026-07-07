"""Configuration model and loader for local-engine-router.

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

# Valid routing modes. "smart" (the default) lets the router pick the best
# local model for smart aliases / cloud-model names / unknown ids; "manual"
# preserves exact model-id routing for everything (the pre-0.7 behaviour).
ROUTING_MODES: frozenset[str] = frozenset({"smart", "manual"})

# Scoring components the smart picker combines; the keys allowed in
# smart.weights and smart.policies.<name>.
SMART_WEIGHT_KEYS: frozenset[str] = frozenset(
    {"quality", "speed", "residency", "swap_cost", "reliability", "context"}
)

# Built-in smart policies (weight presets). A config may add its own under
# smart.policies; smart.policy must name one of these or a config-defined one.
SMART_BUILTIN_POLICIES: frozenset[str] = frozenset(
    {"balanced", "fast", "quality", "economy"}
)

# Capabilities a model may declare in its metadata. The benchmark capability
# axes plus request-shape capabilities that gate candidate eligibility.
MODEL_CAPABILITIES: frozenset[str] = frozenset(
    {
        "general", "coding", "code_editing", "math", "reasoning", "tool_use",
        "writing", "summarization", "long_context", "json_structured",
        "embedding", "vision",
    }
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
    # Reasoning/thinking-budget guard for models with thinking ON by default
    # (e.g. DiffusionGemma served with --reasoning-parser + enable_thinking).
    # Reasoning tokens count against the request's max_tokens, so a small budget
    # can be entirely consumed by the thought channel, leaving `content` empty
    # (finish_reason=length). When set, chat-completion requests routed here whose
    # max_tokens (or max_completion_tokens) is BELOW this value get
    # `chat_template_kwargs.enable_thinking=false` injected — unless the client
    # set enable_thinking itself. Generous/unset budgets keep thinking on (the
    # quality path). None = feature off. vLLM honors request-level
    # chat_template_kwargs over the server's --default-chat-template-kwargs.
    disable_thinking_below_max_tokens: int | None = None
    # ---- optional smart-picker metadata (all default to "unknown") -------- #
    # Coarse quality/speed tiers, 1 (worst) .. 5 (best). None = derive from
    # benchmark priors / model identity / observed runtime stats instead.
    quality_tier: int | None = None
    speed_tier: int | None = None
    # Approximate memory footprint when loaded, in GB (informational; used for
    # swap-cost estimation when set).
    memory_gb: float | None = None
    # What the model can do. Empty = a general chat model (everything except
    # "embedding"/"vision"). Include "embedding" for embedding models and
    # "vision" for multimodal ones so the picker matches endpoint/request shape.
    capabilities: list[str] = field(default_factory=list)
    # Per-capability score overrides (0..1) that beat benchmark priors, e.g.
    # {"coding": 0.9} for a model you know punches above its benchmarks.
    strengths: dict[str, float] = field(default_factory=dict)
    # Set false to keep this model out of smart selection entirely (it stays
    # routable by its exact id).
    smart_enabled: bool = True


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
    serve_script: str = ""
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
    log_file: str = "./logs/ds4-server.log"
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
    # When true, the engine is queried at runtime (via its /v1/models endpoint)
    # to discover models it is serving. Discovered models augment (never replace)
    # the static `models:` list; they are opt-in and off by default.
    discover_models: bool = False
    # Explicit list of model ids this engine is expected to serve. Used when
    # discover_models is false and you want to pre-declare the models without
    # a full static models: entry. Ignored when empty.
    served_models: list[str] = field(default_factory=list)
    # TTL (seconds) for the cached /v1/models lookup used by model discovery.
    # Mirrors the same field on OllamaConfig and ApiSwapConfig.
    tags_cache_ttl_s: float = 30.0


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


@dataclass
class DiscoverConfig:
    """Global model-discovery settings.

    Discovery is entirely opt-in. The default of ``enabled: false`` means
    behaviour is byte-identical to a config that omits the ``discover:`` block.
    When enabled, each engine that has ``discover_models: true`` is queried for
    its live model list. Discovered models *augment* (never replace) the static
    ``models:`` list; static entries always take precedence.
    """

    # Master switch. Discovery is a no-op unless this is true.
    enabled: bool = False
    # How to handle a model id found on multiple engines simultaneously.
    #   "config_order"  -> the engine that appears first in the engines: table wins
    collision: str = "config_order"
    # Background refresh cadence for the durable catalog when discovery is on.
    # Zero disables the periodic loop while still allowing startup/manual refresh.
    refresh_interval_s: float = 300.0
    # How long persisted catalog entries are trusted after last_seen. Zero means
    # never expire persisted entries.
    state_ttl_s: float = 2_592_000.0
    # Port probe sub-section (parsed from a nested ``port_probe:`` mapping).
    port_probe_enabled: bool = False


@dataclass
class SmartRetryConfig:
    """Retry/fall-forward policy for smart-picked requests.

    Applies only to requests the smart picker routed. The inviolable rule —
    never retry after response bytes have reached the client — is enforced in
    the app layer and is not configurable."""

    # Re-try the SAME model once when its engine fails to come up (a reload /
    # restart often clears a transient startup failure).
    same_model_reload: bool = True
    # After that, fall forward to the next-ranked compatible candidate(s).
    fall_forward: bool = True
    # How many fallback candidates a decision carries (bounds total attempts).
    max_fallbacks: int = 2
    # Consecutive failures before a model enters cooldown ...
    failure_threshold: int = 3
    # ... and how long the cooldown lasts.
    cooldown_s: float = 120.0


@dataclass
class SmartBenchmarksConfig:
    """Benchmark-intelligence settings for the smart picker."""

    # Use benchmark priors at all (off = metadata + calibration + runtime only).
    enabled: bool = True
    # Allow providers that need network egress. The builtin curated table is
    # offline, so routing works fully offline with this false (the default);
    # benchmark *sync* is an explicit, opt-in action.
    allow_network: bool = False
    # Cache TTL for fetched records; 0 = never expire (refresh is explicit via
    # `routerctl benchmarks refresh`).
    cache_ttl_s: float = 0.0


@dataclass
class SmartConfig:
    """Settings for the smart model picker (see router/smart.py).

    Active when ``routing_mode: smart`` (the default). All fields have
    sensible defaults so an empty/absent ``smart:`` block fully works."""

    # Request model ids that always trigger smart selection.
    aliases: list[str] = field(default_factory=lambda: ["smart", "auto", "default"])
    # Active scoring policy: balanced | fast | quality | economy, or a custom
    # name defined under `policies:` below.
    policy: str = "balanced"
    # Route well-known cloud model ids (gpt-*, claude-*, gemini-*, ...) through
    # the picker instead of the legacy unknown-model fallback.
    catch_cloud_models: bool = True
    # Route any OTHER unknown model id through the picker too (otherwise those
    # fall back to the legacy default-engine guess).
    catch_unknown_models: bool = True
    # When true, even exact configured/installed model ids go through the
    # picker. Default false: an exact id is an explicit user choice.
    override_exact_model_ids: bool = False
    # Minimum total-score advantage a non-resident model must have over the
    # best already-resident candidate to justify an engine swap.
    swap_margin: float = 0.08
    # Below this decision confidence, prefer the resident candidate.
    min_confidence: float = 0.25
    # Seconds of expected swap cost that count as "maximally expensive" when
    # normalizing the swap-cost score component.
    swap_cost_horizon_s: float = 180.0
    # Scoring-component weight overrides (merged over the policy's weights).
    # Keys: quality, speed, residency, swap_cost, reliability, context.
    weights: dict[str, float] = field(default_factory=dict)
    # Custom named policies: {name -> {weight key -> value}}.
    policies: dict[str, dict[str, float]] = field(default_factory=dict)
    retry: SmartRetryConfig = field(default_factory=SmartRetryConfig)
    benchmarks: SmartBenchmarksConfig = field(default_factory=SmartBenchmarksConfig)
    # Allow POST /admin/smart/calibrate to run local smoke probes (it acquires
    # engines, so it can trigger swaps; the probes themselves are tiny).
    calibration_enabled: bool = True


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
    log_file: str = "./logs/router.log"
    log_level: str = "INFO"
    # Persisted observability snapshot (active engine, last swap). Not trusted
    # as ground truth on startup — the manager re-probes reality.
    state_file: str = "./state.json"
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
    # Global model-discovery settings. Absent in config => all defaults (off).
    discover: DiscoverConfig = field(default_factory=DiscoverConfig)
    # Routing mode: "smart" (default; the picker chooses the best local model
    # for smart aliases / cloud names / unknown ids) or "manual" (exact
    # model-id routing only — the pre-0.7 behaviour). Switch with
    # `routerctl manual` / `routerctl smart`.
    routing_mode: str = "smart"
    # Smart-picker settings; ignored when routing_mode is "manual".
    smart: SmartConfig = field(default_factory=SmartConfig)

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
        _validate_generic_process_fields(key, params)
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


_DISCOVER_VALID_COLLISION: frozenset[str] = frozenset({"config_order"})
_DISCOVER_KNOWN_KEYS: frozenset[str] = frozenset(
    {"enabled", "collision", "refresh_interval_s", "state_ttl_s", "port_probe"}
)
_DISCOVER_PORT_PROBE_KNOWN_KEYS: frozenset[str] = frozenset({"enabled"})


def _parse_discover_section(raw_discover: Any) -> DiscoverConfig:
    """Parse the optional top-level ``discover:`` mapping into a DiscoverConfig.

    Absent or null ``discover:`` returns all defaults (fully off). Raises
    ConfigError on unknown keys or invalid values; mirrors the ConfigError style
    used elsewhere in load_config.
    """
    if not raw_discover:
        return DiscoverConfig()
    if not isinstance(raw_discover, dict):
        raise ConfigError("'discover' must be a mapping")

    unknown = set(raw_discover) - _DISCOVER_KNOWN_KEYS
    if unknown:
        raise ConfigError(
            f"unknown key(s) under 'discover': {sorted(unknown)} "
            f"(known: {sorted(_DISCOVER_KNOWN_KEYS)})"
        )

    collision = raw_discover.get("collision", "config_order")
    if collision not in _DISCOVER_VALID_COLLISION:
        raise ConfigError(
            f"discover.collision {collision!r} is not valid "
            f"(must be one of {sorted(_DISCOVER_VALID_COLLISION)})"
        )

    port_probe_raw = raw_discover.get("port_probe")
    port_probe_enabled = False
    if port_probe_raw is not None:
        if not isinstance(port_probe_raw, dict):
            raise ConfigError("'discover.port_probe' must be a mapping")
        unknown_pp = set(port_probe_raw) - _DISCOVER_PORT_PROBE_KNOWN_KEYS
        if unknown_pp:
            raise ConfigError(
                f"unknown key(s) under 'discover.port_probe': {sorted(unknown_pp)} "
                f"(known: {sorted(_DISCOVER_PORT_PROBE_KNOWN_KEYS)})"
            )
        port_probe_enabled = bool(port_probe_raw.get("enabled", False))

    return DiscoverConfig(
        enabled=bool(raw_discover.get("enabled", False)),
        collision=collision,
        refresh_interval_s=_parse_nonnegative_float(
            raw_discover.get("refresh_interval_s", DiscoverConfig.refresh_interval_s),
            "discover.refresh_interval_s",
        ),
        state_ttl_s=_parse_nonnegative_float(
            raw_discover.get("state_ttl_s", DiscoverConfig.state_ttl_s),
            "discover.state_ttl_s",
        ),
        port_probe_enabled=port_probe_enabled,
    )


def _parse_nonnegative_float(value: Any, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{field_name} must be a number (got {value!r})")
    if out < 0:
        raise ConfigError(f"{field_name} must be >= 0 (got {value!r})")
    return out


# --------------------------------------------------------------------------- #
# Smart-picker section parsing
# --------------------------------------------------------------------------- #
_SMART_KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "aliases", "policy", "policies", "weights", "catch_cloud_models",
        "catch_unknown_models", "override_exact_model_ids", "swap_margin",
        "min_confidence", "swap_cost_horizon_s", "retry", "benchmarks",
        "calibration_enabled",
    }
)
_SMART_RETRY_KNOWN_KEYS: frozenset[str] = frozenset(
    {"same_model_reload", "fall_forward", "max_fallbacks",
     "failure_threshold", "cooldown_s"}
)
_SMART_BENCHMARKS_KNOWN_KEYS: frozenset[str] = frozenset(
    {"enabled", "allow_network", "cache_ttl_s"}
)


def _parse_weight_table(raw: Any, ctx: str) -> dict[str, float]:
    """Validate a {weight key -> number} mapping for smart.weights / policies."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{ctx} must be a mapping of weight -> number")
    unknown = set(raw) - SMART_WEIGHT_KEYS
    if unknown:
        raise ConfigError(
            f"unknown weight key(s) under {ctx}: {sorted(unknown)} "
            f"(known: {sorted(SMART_WEIGHT_KEYS)})"
        )
    out: dict[str, float] = {}
    for key, val in raw.items():
        out[key] = _parse_nonnegative_float(val, f"{ctx}.{key}")
    if out and not any(v > 0 for v in out.values()):
        raise ConfigError(f"{ctx}: at least one weight must be > 0")
    return out


def _parse_fraction(value: Any, field_name: str) -> float:
    out = _parse_nonnegative_float(value, field_name)
    if out > 1:
        raise ConfigError(f"{field_name} must be between 0 and 1 (got {value!r})")
    return out


def _parse_smart_section(raw_smart: Any) -> SmartConfig:
    """Parse the optional top-level ``smart:`` mapping into a SmartConfig.

    Absent/null returns all defaults. Raises ConfigError on unknown keys or
    invalid values (mirrors the strictness of the ``discover:`` parser)."""
    if not raw_smart:
        return SmartConfig()
    if not isinstance(raw_smart, dict):
        raise ConfigError("'smart' must be a mapping")

    unknown = set(raw_smart) - _SMART_KNOWN_KEYS
    if unknown:
        raise ConfigError(
            f"unknown key(s) under 'smart': {sorted(unknown)} "
            f"(known: {sorted(_SMART_KNOWN_KEYS)})"
        )

    defaults = SmartConfig()

    aliases = raw_smart.get("aliases", defaults.aliases)
    if not isinstance(aliases, list) or not all(
        isinstance(a, str) and a for a in aliases
    ):
        raise ConfigError("smart.aliases must be a list of non-empty strings")

    policies_raw = raw_smart.get("policies") or {}
    if not isinstance(policies_raw, dict):
        raise ConfigError("smart.policies must be a mapping of name -> weights")
    policies: dict[str, dict[str, float]] = {}
    for name, table in policies_raw.items():
        if not isinstance(name, str) or not name:
            raise ConfigError("smart.policies keys must be non-empty strings")
        policies[name] = _parse_weight_table(table, f"smart.policies.{name}")

    policy = raw_smart.get("policy", defaults.policy)
    valid_policies = SMART_BUILTIN_POLICIES | set(policies)
    if policy not in valid_policies:
        raise ConfigError(
            f"smart.policy {policy!r} is not a built-in policy "
            f"({sorted(SMART_BUILTIN_POLICIES)}) or defined under smart.policies"
        )

    weights = _parse_weight_table(raw_smart.get("weights") or {}, "smart.weights")

    retry_raw = raw_smart.get("retry") or {}
    if not isinstance(retry_raw, dict):
        raise ConfigError("'smart.retry' must be a mapping")
    unknown_r = set(retry_raw) - _SMART_RETRY_KNOWN_KEYS
    if unknown_r:
        raise ConfigError(
            f"unknown key(s) under 'smart.retry': {sorted(unknown_r)} "
            f"(known: {sorted(_SMART_RETRY_KNOWN_KEYS)})"
        )
    retry_defaults = SmartRetryConfig()
    max_fallbacks = retry_raw.get("max_fallbacks", retry_defaults.max_fallbacks)
    failure_threshold = retry_raw.get(
        "failure_threshold", retry_defaults.failure_threshold
    )
    for name, val in (("max_fallbacks", max_fallbacks),
                      ("failure_threshold", failure_threshold)):
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise ConfigError(f"smart.retry.{name} must be a non-negative integer")
    retry = SmartRetryConfig(
        same_model_reload=bool(
            retry_raw.get("same_model_reload", retry_defaults.same_model_reload)
        ),
        fall_forward=bool(retry_raw.get("fall_forward", retry_defaults.fall_forward)),
        max_fallbacks=max_fallbacks,
        failure_threshold=failure_threshold,
        cooldown_s=_parse_nonnegative_float(
            retry_raw.get("cooldown_s", retry_defaults.cooldown_s),
            "smart.retry.cooldown_s",
        ),
    )

    bench_raw = raw_smart.get("benchmarks") or {}
    if not isinstance(bench_raw, dict):
        raise ConfigError("'smart.benchmarks' must be a mapping")
    unknown_b = set(bench_raw) - _SMART_BENCHMARKS_KNOWN_KEYS
    if unknown_b:
        raise ConfigError(
            f"unknown key(s) under 'smart.benchmarks': {sorted(unknown_b)} "
            f"(known: {sorted(_SMART_BENCHMARKS_KNOWN_KEYS)})"
        )
    bench_defaults = SmartBenchmarksConfig()
    benchmarks = SmartBenchmarksConfig(
        enabled=bool(bench_raw.get("enabled", bench_defaults.enabled)),
        allow_network=bool(bench_raw.get("allow_network", bench_defaults.allow_network)),
        cache_ttl_s=_parse_nonnegative_float(
            bench_raw.get("cache_ttl_s", bench_defaults.cache_ttl_s),
            "smart.benchmarks.cache_ttl_s",
        ),
    )

    return SmartConfig(
        aliases=list(aliases),
        policy=str(policy),
        catch_cloud_models=bool(
            raw_smart.get("catch_cloud_models", defaults.catch_cloud_models)
        ),
        catch_unknown_models=bool(
            raw_smart.get("catch_unknown_models", defaults.catch_unknown_models)
        ),
        override_exact_model_ids=bool(
            raw_smart.get("override_exact_model_ids", defaults.override_exact_model_ids)
        ),
        swap_margin=_parse_fraction(
            raw_smart.get("swap_margin", defaults.swap_margin), "smart.swap_margin"
        ),
        min_confidence=_parse_fraction(
            raw_smart.get("min_confidence", defaults.min_confidence),
            "smart.min_confidence",
        ),
        swap_cost_horizon_s=_parse_nonnegative_float(
            raw_smart.get("swap_cost_horizon_s", defaults.swap_cost_horizon_s),
            "smart.swap_cost_horizon_s",
        ),
        weights=weights,
        policies=policies,
        retry=retry,
        benchmarks=benchmarks,
        calibration_enabled=bool(
            raw_smart.get("calibration_enabled", defaults.calibration_enabled)
        ),
    )


def _parse_model_smart_metadata(m: dict[str, Any]) -> dict[str, Any]:
    """Validate + normalize the optional smart-picker fields of a model entry."""
    mid = m.get("id", "?")
    out: dict[str, Any] = {}

    for tier_name in ("quality_tier", "speed_tier"):
        tier = m.get(tier_name)
        if tier is not None:
            if not isinstance(tier, int) or isinstance(tier, bool) or not (
                1 <= tier <= 5
            ):
                raise ConfigError(
                    f"model {mid!r}: {tier_name} must be an integer 1..5 "
                    f"(got {tier!r}); omit it to derive from benchmarks"
                )
        out[tier_name] = tier

    memory_gb = m.get("memory_gb")
    if memory_gb is not None:
        try:
            memory_gb = float(memory_gb)
        except (TypeError, ValueError):
            raise ConfigError(f"model {mid!r}: memory_gb must be a number")
        if memory_gb <= 0:
            raise ConfigError(f"model {mid!r}: memory_gb must be > 0")
    out["memory_gb"] = memory_gb

    caps = m.get("capabilities") or []
    if not isinstance(caps, list) or not all(isinstance(c, str) for c in caps):
        raise ConfigError(f"model {mid!r}: capabilities must be a list of strings")
    unknown_caps = set(caps) - MODEL_CAPABILITIES
    if unknown_caps:
        raise ConfigError(
            f"model {mid!r}: unknown capability(ies) {sorted(unknown_caps)} "
            f"(known: {sorted(MODEL_CAPABILITIES)})"
        )
    out["capabilities"] = list(caps)

    strengths = m.get("strengths") or {}
    if not isinstance(strengths, dict):
        raise ConfigError(
            f"model {mid!r}: strengths must be a mapping of capability -> 0..1"
        )
    unknown_str = set(strengths) - MODEL_CAPABILITIES
    if unknown_str:
        raise ConfigError(
            f"model {mid!r}: unknown strength key(s) {sorted(unknown_str)} "
            f"(known: {sorted(MODEL_CAPABILITIES)})"
        )
    for cap, score in strengths.items():
        out.setdefault("strengths", {})[cap] = _parse_fraction(
            score, f"model {mid!r} strengths.{cap}"
        )
    out.setdefault("strengths", {})

    out["smart_enabled"] = bool(m.get("smart_enabled", True))
    return out


def _validate_generic_process_fields(key: str, params: GenericProcessConfig) -> None:
    """Validate the new discovery-related fields on a GenericProcessConfig.

    Called from _validate_engine_params after the base checks pass.
    """
    try:
        ttl = float(params.tags_cache_ttl_s)
    except (TypeError, ValueError):
        raise ConfigError(
            f"engine {key!r} (generic_process): tags_cache_ttl_s must be a number "
            f"(got {params.tags_cache_ttl_s!r})"
        )
    if ttl < 0:
        raise ConfigError(
            f"engine {key!r} (generic_process): tags_cache_ttl_s must be >= 0 "
            f"(got {params.tags_cache_ttl_s})"
        )
    if not isinstance(params.served_models, list):
        raise ConfigError(
            f"engine {key!r} (generic_process): served_models must be a list "
            f"of non-empty strings (got {type(params.served_models).__name__!r})"
        )
    for mid in params.served_models:
        if not isinstance(mid, str) or not mid:
            raise ConfigError(
                f"engine {key!r} (generic_process): served_models must be a list "
                f"of non-empty strings; got an empty or non-string entry"
            )


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
        _thinking_floor = m.get("disable_thinking_below_max_tokens")
        if _thinking_floor is not None:
            try:
                _thinking_floor = int(_thinking_floor)
            except (TypeError, ValueError):
                raise ConfigError(
                    f"model {m['id']!r}: disable_thinking_below_max_tokens must be "
                    f"an integer (got {_thinking_floor!r})"
                )
            if _thinking_floor < 1:
                raise ConfigError(
                    f"model {m['id']!r}: disable_thinking_below_max_tokens must be "
                    f">= 1 (got {_thinking_floor}); omit it to disable the guard"
                )
        models.append(
            ModelSpec(
                id=m["id"],
                engine=m["engine"],
                display_name=m.get("display_name", m["id"]),
                context_length=int(m.get("context_length", 131072)),
                disable_thinking_below_max_tokens=_thinking_floor,
                **_parse_model_smart_metadata(m),
            )
        )

    discover = _parse_discover_section(raw.get("discover"))
    smart = _parse_smart_section(raw.get("smart"))

    routing_mode = raw.get("routing_mode", "smart")
    if routing_mode not in ROUTING_MODES:
        raise ConfigError(
            f"routing_mode {routing_mode!r} is not valid "
            f"(must be one of {sorted(ROUTING_MODES)})"
        )

    skip = {"ds4", "ollama", "engines", "models", "discover", "smart",
            "routing_mode"}
    top = {
        k: v
        for k, v in raw.items()
        if k in RouterConfig.__dataclass_fields__ and k not in skip
    }
    # An explicit empty `aliases:` / `api_keys:` key is YAML null; normalize to
    # the empty container so runtime code can rely on the declared types
    # (engines.py calls cfg.aliases.get() on every request).
    if top.get("aliases", ...) is None:
        top["aliases"] = {}
    if top.get("api_keys", ...) is None:
        top["api_keys"] = []
    cfg = RouterConfig(
        ds4=ds4, ollama=ollama, engines=engines, models=models,
        discover=discover, smart=smart, routing_mode=routing_mode, **top
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
    _validate_smart(cfg)
    return cfg


def _validate_smart(cfg: RouterConfig) -> None:
    """Cross-field checks for the smart picker settings.

    A smart alias that collides with a real model id (or a configured alias)
    is only a warning: exact ids always win over smart aliases by design, so
    the model stays reachable — but the user probably didn't intend it."""
    if cfg.routing_mode != "smart":
        return
    known_ids = {m.id for m in cfg.models}
    for alias in cfg.smart.aliases:
        if alias in known_ids:
            log.warning(
                "smart alias %r is also a configured model id; the exact model "
                "wins, so this alias will never trigger smart selection",
                alias,
            )
        if alias in (cfg.aliases or {}):
            log.warning(
                "smart alias %r is also a configured alias (-> %r); the "
                "configured alias wins, so this alias will never trigger "
                "smart selection",
                alias,
                cfg.aliases[alias],
            )


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

    _tier_schema = {
        "anyOf": [
            {"type": "integer", "minimum": 1, "maximum": 5},
            {"type": "null"},
        ],
        "default": None,
    }
    model_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "engine": {"type": "string"},
            "display_name": {"type": "string"},
            "context_length": {"type": "integer", "default": 131072},
            "disable_thinking_below_max_tokens": {
                "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
                "default": None,
                "description": (
                    "Inject chat_template_kwargs.enable_thinking=false on chat "
                    "requests whose max_tokens is below this, so a small budget "
                    "isn't eaten by the reasoning channel. null/omitted = off."
                ),
            },
            "quality_tier": {
                **_tier_schema,
                "description": (
                    "Optional coarse quality tier 1 (worst) .. 5 (best) for the "
                    "smart picker; omitted = derive from benchmark priors."
                ),
            },
            "speed_tier": {
                **_tier_schema,
                "description": (
                    "Optional coarse speed tier 1 (slowest) .. 5 (fastest) for "
                    "the smart picker; omitted = derive from model size."
                ),
            },
            "memory_gb": {
                "anyOf": [
                    {"type": "number", "exclusiveMinimum": 0},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Approximate loaded memory footprint in GB.",
            },
            "capabilities": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(MODEL_CAPABILITIES)},
                "description": (
                    "What the model can do; empty = general chat model. Include "
                    "'embedding' / 'vision' so the smart picker matches request "
                    "shape."
                ),
            },
            "strengths": {
                "type": "object",
                "additionalProperties": {
                    "type": "number", "minimum": 0, "maximum": 1,
                },
                "description": (
                    "Per-capability score overrides (0..1) that beat benchmark "
                    "priors for the smart picker."
                ),
            },
            "smart_enabled": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Set false to exclude this model from smart selection (it "
                    "stays routable by its exact id)."
                ),
            },
        },
        "required": ["id", "engine"],
        "additionalProperties": True,
    }

    discover_schema = {
        "type": "object",
        "description": (
            "Optional global model-discovery settings. Absent or omitted = all "
            "defaults (discovery off). Discovery augments the static models: list "
            "and is entirely opt-in per engine via discover_models: true."
        ),
        "properties": {
            "enabled": {"type": "boolean", "default": False},
            "collision": {
                "type": "string",
                "enum": ["config_order"],
                "default": "config_order",
            },
            "refresh_interval_s": {"type": "number", "default": 300.0},
            "state_ttl_s": {"type": "number", "default": 2_592_000.0},
            "port_probe": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }

    _weight_table_schema = {
        "type": "object",
        "properties": {
            key: {"type": "number", "minimum": 0} for key in sorted(SMART_WEIGHT_KEYS)
        },
        "additionalProperties": False,
    }
    smart_schema = {
        "type": "object",
        "description": (
            "Smart model-picker settings; active when routing_mode is 'smart' "
            "(the default). Absent = all defaults."
        ),
        "properties": {
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["smart", "auto", "default"],
                "description": "Request model ids that always trigger smart selection.",
            },
            "policy": {
                "type": "string",
                "default": "balanced",
                "description": (
                    "Active scoring policy: balanced | fast | quality | economy, "
                    "or a custom name defined under policies."
                ),
            },
            "policies": {
                "type": "object",
                "additionalProperties": _weight_table_schema,
                "description": "Custom named policies (weight presets).",
            },
            "weights": {
                **_weight_table_schema,
                "description": (
                    "Scoring-component weight overrides, merged over the "
                    "active policy's weights."
                ),
            },
            "catch_cloud_models": {"type": "boolean", "default": True},
            "catch_unknown_models": {"type": "boolean", "default": True},
            "override_exact_model_ids": {"type": "boolean", "default": False},
            "swap_margin": {
                "type": "number", "minimum": 0, "maximum": 1, "default": 0.08,
                "description": (
                    "Score advantage a non-resident model needs over the best "
                    "resident candidate to justify an engine swap."
                ),
            },
            "min_confidence": {
                "type": "number", "minimum": 0, "maximum": 1, "default": 0.25,
            },
            "swap_cost_horizon_s": {"type": "number", "minimum": 0, "default": 180.0},
            "retry": {
                "type": "object",
                "properties": {
                    "same_model_reload": {"type": "boolean", "default": True},
                    "fall_forward": {"type": "boolean", "default": True},
                    "max_fallbacks": {"type": "integer", "minimum": 0, "default": 2},
                    "failure_threshold": {"type": "integer", "minimum": 0, "default": 3},
                    "cooldown_s": {"type": "number", "minimum": 0, "default": 120.0},
                },
                "additionalProperties": False,
            },
            "benchmarks": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "allow_network": {"type": "boolean", "default": False},
                    "cache_ttl_s": {"type": "number", "minimum": 0, "default": 0.0},
                },
                "additionalProperties": False,
            },
            "calibration_enabled": {"type": "boolean", "default": True},
        },
        "additionalProperties": False,
    }

    root = _schema_for_dataclass(
        RouterConfig,
        exclude=("ds4", "ollama", "engines", "models", "discover", "smart"),
    )
    root["properties"]["routing_mode"] = {
        "type": "string",
        "enum": sorted(ROUTING_MODES),
        "default": "smart",
        "description": (
            "'smart' (default): the picker chooses the best local model for "
            "smart aliases / cloud names / unknown ids. 'manual': exact "
            "model-id routing only."
        ),
    }
    root["properties"]["smart"] = smart_schema
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
    root["properties"]["discover"] = discover_schema

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/rxxusp/local-engine-router/config.schema.json",
        "title": "local-engine-router configuration",
        "description": (
            "Configuration schema for local-engine-router (local-engine-router). Unknown "
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
        log_dir = os.path.dirname(cfg.log_file)
        if log_dir:  # bare filename => cwd; makedirs("") raises FileNotFoundError
            os.makedirs(log_dir, exist_ok=True)
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
