# Changelog

All notable changes to this project are documented here. The project aims to
follow [Semantic Versioning](https://semver.org/) once it reaches a stable API;
until then it is in a `0.x` channel where minor versions may break.

## [Unreleased]

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
