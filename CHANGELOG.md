# Changelog

All notable changes to this project are documented here. The project aims to
follow [Semantic Versioning](https://semver.org/) once it reaches a stable API;
until then it is in a `0.x` channel where minor versions may break.

## [Unreleased]

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
- **Packaging**: `pyproject.toml` with console scripts `llm-router`
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
Initial private release.

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
