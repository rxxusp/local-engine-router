# Changelog

All notable changes to this project are documented here. The project aims to
follow [Semantic Versioning](https://semver.org/) once it reaches a stable API;
until then it is in a `0.x` channel where minor versions may break.

## [0.6.0] - 2026-07-01

A bug-fix release: a full review of the codebase found and fixed 16 defects.
No new features and no config changes; everything below makes existing
behaviour correct.

### Fixed — request handling
- **Streaming requests no longer mask upstream errors as empty 200 streams.**
  When the engine returns 4xx/5xx on a streaming request (e.g. context length
  exceeded), the error is now relayed as a framed SSE `data:` error chunk on
  `/v1/*` and as an Ollama-style `{"error": ...}` NDJSON line on `/api/*`,
  instead of raw unframed bytes (or silence) on a 200 response.
- **Query strings are preserved on proxied requests.** Previously every
  forwarded and passthrough request silently dropped `?query=params`, breaking
  any upstream endpoint that takes them (notably via the `/api/*` catch-all).
- **`/api/pull` now streams NDJSON progress live.** It was buffered through the
  non-streaming forwarder, so clients saw zero bytes until the entire model
  download finished and typically timed out.
- **Valid-but-non-object JSON bodies (`"hi"`, `[1,2]`, `42`) return 400** with a
  proper `invalid_request_error` instead of crashing with a 500 on `/v1/*`,
  `/api/*` model-bearing posts, and `/admin/swap`.
- **Non-ASCII API keys no longer crash auth with a 500.** `hmac.compare_digest`
  raises `TypeError` on non-ASCII str input; comparison is now done on bytes,
  so a stray `Authorization: Bearer café` gets a clean 401.

### Fixed — engine lifecycle
- **Windows: graceful stop no longer escalates straight to a hard kill.** On
  platforms without `SIGKILL`/`killpg` the force-kill fallback aliased SIGTERM,
  so the initial graceful stop was misread as a kill request and the
  SIGTERM→wait→SIGKILL escalation never got its graceful phase.
- **Post-swap model snapshots can no longer be lost to garbage collection.**
  The fire-and-forget `asyncio.ensure_future` task now holds a strong
  reference until it completes (the event loop only keeps weak refs).

### Fixed — config, CLI, wizard
- **`log_file: router.log` (no directory) now works.** `os.makedirs("")` raised
  and the handler-install was silently skipped, disabling file logging.
- **An explicit empty `aliases:` / `api_keys:` key (YAML null) no longer
  crashes every request** with `AttributeError` at routing time; both
  normalize to their empty container at load.
- **The wizard quotes numeric-looking model ids** (`1.5`, `123`, `1_000`,
  `0x1F`) so a scaffolded config round-trips them as strings; bare, they
  re-parsed as numbers and could never match request routing.
- **`routerctl logs`: Ctrl-C exits** instead of falling through to `tail -f`
  (requiring a second Ctrl-C), and a journalctl that exits non-zero (unknown
  unit, no journal access) now correctly falls back to tailing the log file.
- **`routerctl status/models/health` report HTTP errors as HTTP errors.** A
  401/500 from a *running* router was misreported as "router not reachable"
  because `HTTPError` subclasses `URLError`.

### Fixed — packaging, deploy, CI
- **Docker healthcheck honors the configured port.** It read only
  `$ROUTER_PORT` (which nothing sets from the config), so a non-default
  `port:` in the mounted config marked a healthy container unhealthy. It now
  reads the port from the config itself, falling back to the env var.
- **docker-compose quickstart documents the required `host: "0.0.0.0"` bind** —
  with the example config's `127.0.0.1` the published port was unreachable
  from the host while the container still reported healthy.
- **PyPI publish workflow no longer fails on release-created duplicate runs.**
  Pushing a tag and then publishing a GitHub Release fired both triggers; the
  second upload died with "File already exists". Now `skip-existing: true`.
- **smoke_test.sh step 3 sends one request, not two.** It previously issued
  the same real (GPU-swapping) completion twice and paired the body of the
  first with the HTTP status of the second, which could diverge; temp files
  now use `mktemp` throughout.

## [0.5.0] - 2026-06-19

This release is about getting from zero to a running router in a couple of
copy-paste steps. Everything is additive and opt-in; existing installs and
configs keep working unchanged.

### Added
- **One-command bootstrap installer** (`install.sh` at the repo root). Run it
  with `curl -fsSL .../install.sh | bash` and it creates an isolated virtualenv,
  installs the package and its dependencies, puts `local-engine-router` and
  `routerctl` on your `PATH`, writes a starter config if none exists, and offers
  to install and enable the systemd `--user` service. It is idempotent and safe
  to re-run, prints what it will do, and is fully parameterised by environment
  variables (`LER_VENV`, `LER_CONFIG`, `LER_BIN`, `LER_UNIT_DIR`, `LER_SOURCE`)
  with `--yes`, `--no-service`, `--dry-run`, `--print-unit`, and `--uninstall`
  flags.
- **Interactive setup wizard** (`local-engine-router init`, also `routerctl
  init`). It probes the well-known localhost ports of every supported backend
  (Ollama 11434, llama.cpp/LocalAI 8080, vLLM/MAX 8000, SGLang 30000, LM Studio
  1234, TabbyAPI 5000, KoboldCpp 5001), confirms what is actually listening,
  fetches each engine's live model list, asks the few things it cannot infer
  (bind host, API key, which detected engines to include), and scaffolds a
  working `config.yaml` from the matching presets. It is suggest-and-confirm:
  an open port that does not confirm as a known backend is never added without
  an explicit yes. Modes: `--yes` (non-interactive), `--example` (write a
  commented starter without probing), `--detect-only` (report and write
  nothing), `--force`, `--host`, `--port`, `--config`, `--probe-host`. The
  generated config is validated through the real loader before it is written,
  so the wizard never leaves an invalid config behind. The wizard does its own
  well-known-port probing; the `discover.port_probe` config flag reserved in
  0.4.0 stays parse-only and is not yet consumed at runtime.
- **Docker quickstart** (`docker-compose.yml` at the repo root) using the
  published multi-arch `ghcr.io/rxxusp/local-engine-router` image, for people
  who prefer containers.
- **Tag-triggered PyPI publish workflow** (`.github/workflows/pypi-publish.yml`).
  On a `v*` tag it asserts the tag matches the package version, builds an sdist
  and wheel, and publishes to PyPI via Trusted Publishing (OIDC, no stored
  token). It is gated behind the `ENABLE_PYPI_PUBLISH` repository variable, which
  a maintainer sets to enable publishing. This workflow published 0.5.0 to PyPI,
  so `pip install local-engine-router` works.

### Changed
- **README rewritten around a friendly Install / Quickstart** near the top with
  three pick-your-path options (pip/pipx, one-line script, Docker), the `init`
  wizard, and a first curl request, aiming for under five minutes to a working
  router. The deeper docs (swap mechanic, engine types, presets, auto-discovery,
  sharp edges) are unchanged below it.

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
