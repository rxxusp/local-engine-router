# Changelog

All notable changes to this project are documented here. The project aims to
follow [Semantic Versioning](https://semver.org/) once it reaches a stable API;
until then it is in a `0.x` channel where minor versions may break.

## [0.4.0] - 2026-06-19

### Added
- **Opt-in model auto-discovery** (`discover:` block). A new top-level
  `discover:` config section enables live model discovery. Discovery is fully
  opt-in (`enabled: false` by default) and **augments** the static `models:`
  list -- static entries always win over discovered ones. When
  `discover.enabled` is false (or the block is absent), router behaviour is
  byte-identical to previous versions. Collision resolution between engines is
  controlled by `collision: config_order` (first engine in declaration order
  wins; the only supported mode). A `port_probe.enabled` sub-key is accepted
  and validated but reserved for future use.
- **Per-engine discovery fields on `generic_process` engines.** Set
  `discover_models: true` on a `generic_process` engine to opt it into the
  discovery index. Optional supporting fields: `served_models: [..]` (extra
  hint ids registered even when the engine is stopped) and `tags_cache_ttl_s`
  (TTL in seconds for the cached `/v1/models` response used during discovery,
  default 30 s).
- **Stopped-engine model resolution.** When a `generic_process` engine has
  `discover_models: true` but is not currently running, the router resolves its
  candidate model ids from three sources in union: the `start_cmd` argv
  (parsing `--served-model-name`, `--model`/`-m`, and `.gguf` basenames), a
  self-healing last-seen cache populated during past uptime and persisted in
  `state.json`, and the explicit `served_models` list. Down-engine models
  appear in `GET /v1/models` and route correctly, starting the engine on demand.
- **All-engine `GET /v1/models` union** (gated on `discover.enabled`). When
  discovery is enabled, the model list response includes discovered ids from
  stopped engines in addition to the static registry and live engine tags. When
  discovery is disabled the endpoint is unchanged.
- **`POST /admin/discover`** admin endpoint. Calls `available_models()` on
  every engine (best-effort, one try/except per engine) and merges in
  stopped-engine entries from the discovery index. Returns a per-engine mapping
  of sorted model id lists. Auth-gated identically to `POST /admin/swap`.
- **`routerctl discover`** CLI command. Calls `POST /admin/discover` and prints
  the per-engine model id summary.
- **Per-model `disable_thinking_below_max_tokens` guard.** A new integer field
  on any `models:` entry. When set, the router intercepts
  `POST /v1/chat/completions` for that model and injects
  `enable_thinking: false` into the request body if the request's `max_tokens`
  is below the threshold and the client has not explicitly set `enable_thinking`.
  This prevents models from allocating a thinking budget that exceeds the
  available token budget, which can produce empty responses. Explicit client
  values are never overridden.

## [0.3.0] - 2026-06-18

### Fixed
- **Orphaned upstream generations on client disconnect** — streaming `/v1/*` and
  `/api/*` handlers now poll `request.is_disconnected()` and stop pulling from the
  upstream engine the moment the client goes away, closing `client.stream()` on a
  normal control-flow path so the engine aborts generation. Previously a client
  that disconnected mid-stream left the engine generating to completion against a
  dead socket (relying on Starlette's cancellation-based disconnect handling,
  whose upstream close can be interrupted under `CancelledError`) — observed
  pinning a GPU at 96% for hours with multiple stuck requests. Also covers the
  swap-wait keepalive loop. Regression test added.

### Added — v0.3 engine-coverage core (EC1–EC5 + MM4)
- **Control-call auth headers** — optional `control_headers` on each engine
  config, applied to the control client only (not user traffic). Unblocks
  secured TabbyAPI (`x-admin-key`) and LM Studio/LocalAI (`Authorization: Bearer`).
- **Generic HTTP `load_path`** on `api_swap` engines — explicitly load a model
  into a running engine on acquire (`load_path`/`load_method`/`load_body`/
  `load_timeout_s`), skipped when the model is already loaded. Enables
  explicit-load engines (TabbyAPI, text-generation-webui) config-only.
- **Loaded-state filtering + id keying** — `loaded_filter` (e.g. `state==loaded`)
  and `loaded_id_key` (e.g. LM Studio `instance_id`), plus handling of a
  single-object `loaded_path` response (TabbyAPI `/v1/model`).
- **Richer readiness probe** — optional `ready_check` (`key==value` or
  `model:<id>`) beyond HTTP 200, so a false-ready engine (e.g. vLLM `/health`)
  isn't marked ready before it can serve.
- **Process-group reaping** — `generic_process` teardown signals the whole
  process group (`os.killpg`) so forked workers (vLLM/SGLang/Aphrodite/MAX) are
  reaped, with SIGTERM→SIGKILL escalation + port-close verification retained.
- **Alias / capability routing** — top-level `aliases` map (`{alias → real id}`);
  resolved before routing, with the outgoing request body's `model` rewritten to
  the real id so a client asking for e.g. `gpt-4o-mini` reaches the chosen model.

### Added -- v0.3 public-release polish
- **Cross-platform memory-settle and process control.** The post-free
  memory-settle wait and engine teardown now work on Linux, macOS, and Windows
  via a new `router.sysmem` module: a `/proc/meminfo` fast path on Linux and
  `psutil` everywhere else, with psutil-based process-tree reaping where process
  groups are unavailable (Windows). `psutil` is now a runtime dependency.
- **Backend presets.** Copy-paste `engines:` config fragments under `presets/`
  for llama.cpp, vLLM, SGLang, KoboldCpp, MLX, MAX, ramalama, LocalAI, TabbyAPI,
  LM Studio, and Ollama, plus a schema validator and an index.
- **README centered on the GPU-swap mechanic**, with documented sharp edges
  (non-streaming requests block for the whole swap, the single-active-engine
  invariant, and the in-container engine limitation).
- **Multi-arch Docker** images (`linux/amd64` + `linux/arm64`) and a container
  `HEALTHCHECK`.
- **CI matrix** across Python 3.10 / 3.11 / 3.12 and a release-time check that
  the git tag matches the package version.
- **Test coverage** for `routerctl` and the proxy header/forwarding utilities.

### Changed
- The systemd unit and CLI service name are now `local-engine-router` (via a
  single `SERVICE_NAME` constant), and the unit uses the `%h` specifier so it is
  no longer tied to one user's home directory.

## [0.2.0] — 2026-06-02
Second build-out wave: a generic engine layer, keep-alive on every streaming
path, packaging + a published Docker image, a hermetic test suite + CI, config
validation, and Prometheus metrics. The router is still pure-Python and uses no
GPU; the default ds4 + Ollama behaviour is unchanged when you don't opt in to
the new pieces.

### Added
- **Generic engine layer.** New optional top-level `engines:` table
  (`engine_key -> {type, ...}`) lets you add engines with config only, no
  Python. Four engine types:
  - `generic_process` — `GenericProcessEngine`: launches and supervises a local
    server (llama.cpp/llama-server, llamafile, vLLM, SGLang, Aphrodite) from a
    `start_cmd`, polls `ready_path` until HTTP 200, and stops it with
    `stop_signal` → SIGKILL escalation plus listening-port-close verification
    (llama.cpp has a confirmed SIGTERM-freeze bug, so the signal is never
    trusted alone). Per-engine `start_timeout_s` covers slow cold starts.
  - `api_swap` — `APISwapEngine`: an engine whose models load/unload over HTTP
    (the router owns no process). Configurable `unload_path`/`unload_method`/
    `unload_body` (with `{model}` substitution) and an optional `loaded_path`
    probe to confirm VRAM is released. Covers TabbyAPI-style load/unload.
  - `ollama` — now an `APISwapEngine` preset (`/api/ps`, `/api/tags` TTL cache,
    `keep_alive:0` with an `ollama stop` CLI fallback).
  - `ds4` — the bespoke `systemctl --user` escape hatch, unchanged.
  When `engines:` is absent the router builds `ds4` + `ollama` from the legacy
  `ds4:`/`ollama:` sections exactly as before (full back-compat).
- **Config validation** with actionable errors: a new `ConfigError`
  (subclasses `ValueError`) is raised for an unknown engine `type`, a missing
  required field for an engine type, a duplicate engine key, or a `model.engine`
  that references no configured engine. Soft issues (e.g. a `serve_script` that
  does not exist) are logged as warnings.
- **JSON Schema** for the config (`config_json_schema()`, shipped as
  `config.schema.json`) for editor autocomplete, plus two non-serving CLI modes:
  `python3 -m router --print-schema` and `python3 -m router --check-config`
  (loads + validates, prints `OK …` or the error, non-zero exit on failure).
- **Prometheus `/metrics` endpoint** (unauthenticated, exempt from the API-key
  middleware so scrapers reach it keyless). Series: `swap_duration_seconds`,
  `memory_settle_seconds`, `in_flight_at_swap_start` (histograms),
  `swap_total{from,to,result}` (counter), and `engine_uptime_seconds` (gauge).
  No new dependency — the text exposition is hand-rolled.
- **Packaging**: `pyproject.toml` with console scripts `local-engine-router`
  (`router.__main__:main`) and `routerctl` (`router.cli:main`), and a `[dev]`
  extra (pytest, pytest-asyncio). A pure-Python `Dockerfile`
  (`python:3.12-slim`, no CUDA) and a `docker-publish` GitHub Actions workflow
  that pushes images to `ghcr.io/<owner>/local-engine-router` on `v*` tags and
  releases.
- **Tests + CI**: a 69-test hermetic `pytest` suite (no GPU, no network —
  config/validation, routing, the swap state machine, metrics, and app
  integration against a mock backend) plus a `ci.yml` GitHub Actions workflow
  and a README CI badge.

### Changed
- **Keep-alive now covers every streaming path.** During a swap, `/v1/*` SSE
  streams still get `": keepalive"` comment lines, and Ollama-native `/api/*`
  NDJSON streams now get a bare-newline holding frame that NDJSON readers skip
  (SSE comment syntax is deliberately *not* emitted on `/api/*`, as it would
  corrupt the stream). Streaming responses start immediately and acquire the
  engine inside the generator, so a long cold start no longer blocks with zero
  bytes sent.
- `routerctl` logic moved into the importable `router/cli.py`; the top-level
  `./routerctl` script is now a thin shim, and the installed `routerctl`
  console script calls the same entry point.

### Known limitations
- **Non-streaming** requests (`stream:false` on `/v1/*` or `/api/*`) cannot
  carry keep-alive frames — a single JSON body has nowhere to put them — so they
  block for the duration of a swap (up to ~240 s for a cold ds4 start). Such
  callers MUST raise their client read-timeout above the worst-case swap. The
  streaming keep-alive limitation from 0.1.0 is otherwise resolved.
- Process-control engines (`ds4` `systemd-user`/`process`, and
  `generic_process`) reach into the host's process/service tree and therefore
  do **not** work from inside the published container — run the router on the
  host for those. The container is appropriate for fronting `api_swap`/remote
  engines it only talks to over HTTP.

## [0.1.0] — 2026-05-30
Initial release.

### Added
- Single-port OpenAI- and Ollama-compatible reverse proxy that swaps the GPU
  between engines based on the requested `model`.
- Swap state machine: in-flight drain (asyncio lock + condition), free the
  other engine, wait for kernel memory to settle (`/proc/meminfo`), start target.
- Engine backends: `ds4` (managed via `systemctl --user`) and Ollama
  (managed via `keep_alive:0` unload).
- SSE keep-alive heartbeats during a swap on `/v1/*` streaming requests.
- Endpoints: `/v1/*`, Ollama `/api/*`, `/v1/models` (union), `/status`,
  `/admin/swap`, `/health`.
- `routerctl` CLI and a systemd **user** service.
- **MIT license** (attribution to `rxxusp`).
- **API-key authentication** (`Authorization: Bearer` / `X-API-Key`, constant-time
  check, `/health` exempt) and a **safe default bind** (`127.0.0.1`), with a
  startup warning when bound off-localhost without keys.

### Known limitations
- SSE keep-alive covers `/v1/*` streaming only; non-streaming and `/api/*`
  NDJSON requests block for the duration of a swap (see roadmap item 2).
- Engine layer is specific to `ds4` + Ollama; generic engines (llama.cpp, vLLM,
  …) are planned (roadmap item 1).
