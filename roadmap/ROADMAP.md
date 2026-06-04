# Roadmap — local-engine-router

> **Status: provisional / not set in stone.** This is a working plan, not a
> commitment. Priorities and scope will change. Informed by a competitive
> survey (May 2026) of llama-swap, LocalAI, GPUStack, llama.cpp router mode,
> LiteLLM, vLLM/SGLang/TabbyAPI, and others.
>
> **This is the v0.2 roadmap (delivered as v0.2.0).** For the next milestone —
> config-only support for all the major local engines + the "make it better"
> work — see [`ROADMAP-v0.3-engine-coverage.md`](ROADMAP-v0.3-engine-coverage.md).
> For the at-a-glance done/next picture and branch/PR map, see
> [`../STATUS.md`](../STATUS.md).

## What this is

A single OpenAI- and Ollama-compatible reverse proxy that reads a request's
`model` field, figures out which local engine owns it, and transparently
**swaps engines** (only one heavy model fits a memory-constrained GPU at once):
drain in-flight → stop the other engine → wait for memory to be reclaimed →
start the target → proxy, with SSE keep-alive so clients don't time out.

Today it ships two engine backends — a bespoke `ds4` server (managed via
`systemctl --user`) and Ollama (managed via `keep_alive:0` unload) — and is
verified end-to-end on a DGX Spark (GB10, 128 GB unified memory).

## Positioning (why this exists next to llama-swap)

`llama-swap` is the closest tool and does ~80% of the general job (engine-
agnostic process swap, web UI, metrics, auth, Docker, concurrent-model groups).
We are **not** trying to be a better general swap proxy. The defensible niche is
**memory-constrained, unified-memory, heterogeneous-lifecycle** hardware
(DGX Spark / GB10, Apple Silicon) where the cluster tools are overkill and
llama-swap's VRAM handling is weak (its VRAM telemetry is reported broken on
GB10). Four things no maintained tool does today:

1. **Explicit kernel memory-settle wait** — poll `MemAvailable` until it
   plateaus before loading the next engine. Everyone else relies on implicit OS
   reclaim or unreliable VRAM estimates; on unified memory reclaim lags seconds.
2. **Managing engines you did not spawn** — e.g. `systemctl --user` units with
   `Restart=always`. Structurally impossible with llama-swap's `cmd:` model.
3. **Native Ollama `/api/*` on a swap proxy** (llama-swap needs a fork for this).
4. **Upstream-independent SSE keep-alive** during long cold starts.

---

## Done

> Big rocks 1–7 below were delivered across two build-out waves and are shipped
> as of **v0.2.0** on the `roadmap-buildout` branch.

- [x] Core swap state machine (asyncio lock + in-flight drain), mutual exclusion
- [x] Explicit `/proc/meminfo` memory-settle wait before starting the next engine
- [x] Dual API surface: OpenAI `/v1/*` + Ollama-native `/api/*`
- [x] `ds4` (systemctl --user) + Ollama (keep_alive:0) engine backends
- [x] `/status`, `/admin/swap`, `routerctl` CLI, systemd user service, persisted state
- [x] **LICENSE** — MIT (attribution to `rxxusp`). *[blocker — done]*
- [x] **API-key auth + safe default bind** — `Authorization: Bearer` / `X-API-Key`,
      constant-time check, `/health` exempt; default bind `127.0.0.1`; startup
      warning when exposed without keys. *[blocker — done]*
- [x] **1. Generalized engine layer** *(v0.2.0)* — optional top-level `engines:`
      table; `GenericProcessEngine` (llama.cpp/llamafile/vLLM/SGLang/Aphrodite,
      with per-engine cold-start timeouts + SIGTERM→SIGKILL + port-close
      verification), `APISwapEngine` (HTTP load/unload, covers TabbyAPI),
      `OllamaEngine` as an `APISwapEngine` preset, `Ds4Engine` as the bespoke
      escape hatch. Engines now need **config only, no Python**; legacy
      `ds4:`/`ollama:` mode is unchanged when `engines:` is absent.
- [x] **2. Keep-alive on all streaming paths** *(v0.2.0)* — SSE comments on
      `/v1/*` AND bare-newline holding frames on `/api/*` NDJSON. Remaining
      caveat (documented): non-stream callers must raise their read-timeout.
- [x] **3. Packaging** *(v0.2.0)* — `pyproject.toml` (console scripts
      `local-engine-router` + `routerctl`), a pure-Python `Dockerfile` (`python:3.12-slim`,
      no CUDA — a pinned-CUDA base was rejected as needless bloat), and a
      `docker-publish` workflow pushing to `ghcr.io`.
- [x] **4. Tests + CI** *(v0.2.0)* — 69 hermetic pytest tests (no GPU/network,
      mock backend) + a `ci.yml` GitHub Actions workflow + badge.
- [x] **5. Config validation + JSON Schema** *(v0.2.0)* — `ConfigError` with
      actionable messages, `config_json_schema()` / `config.schema.json`, and
      `--print-schema` / `--check-config` CLI modes.
- [x] **6. Prometheus `/metrics`** *(v0.2.0)* — `swap_duration_seconds`,
      `memory_settle_seconds`, `in_flight_at_swap_start` (histograms),
      `swap_total` (counter), `engine_uptime_seconds` (gauge); zero new deps.
- [x] **7. README hero + semver/CHANGELOG** *(v0.2.0)* — problem-led README with
      the four differentiators, `0.x` semver channel, `CHANGELOG.md` kept current.

---

## Differentiation opportunities (vs llama-swap & the field)
- Purpose-built + validated for unified-memory hardware (GB10 / Apple Silicon).
- "Engine abstraction, not a process manager" — wrap engines you didn't spawn.
- In-flight drain directly fixes a named llama-swap pain point (their issue #588).
- Composable: LiteLLM can sit in front (budget/cloud fallback); Harbor can
  provision the engine containers underneath. We own only the request-time swap.

## Ideas / later
- Priority-based preemption (pin a high-priority engine; queue/preempt others) —
  borrowed concept from the SwapServeLLM research prototype.
- Optional rename of the internal package/service (`router` / `local-engine-router`) to
  match the repo name (`local-engine-router`).
- Optional concurrent-model support (run >1 small model when memory allows).

## Open decisions
- **Publish vs upstream?** A viable alternative to maintaining a competitor is
  contributing the two genuinely novel pieces (meminfo-settle wait;
  external/systemd engine lifecycle) **upstream to llama-swap** as PRs. Decide
  before investing in the full publishability checklist.
- Apache-2.0 vs MIT (currently MIT) if a patent grant becomes desirable.

## Honest risks
- llama-swap is far ahead operationally; a generic publish risks being "a
  less-featured llama-swap in Python." Stay tightly scoped to the niche.
- Modest TAM today (DGX Spark / GB10 owners), though growing with Apple Silicon.
- A swap state machine across many engines/kernels is a real maintenance burden
  (KoboldCpp shipped deadlock/segfault bugs in its router; llama.cpp has a
  SIGTERM-freeze bug). Bug reports will span hardware you don't run.

---
*Competitor facts (star counts, versions, issue numbers, TGI archival) are from
a May 2026 survey and this space moves weekly — re-verify before quoting them
in any public comparison.*
