# Roadmap — local-engine-router

> **Status: provisional / not set in stone.** This is a working plan, not a
> commitment. Priorities and scope will change. Informed by a competitive
> survey (May 2026) of llama-swap, LocalAI, GPUStack, llama.cpp router mode,
> LiteLLM, vLLM/SGLang/TabbyAPI, and others.

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

- [x] Core swap state machine (asyncio lock + in-flight drain), mutual exclusion
- [x] Explicit `/proc/meminfo` memory-settle wait before starting the next engine
- [x] Dual API surface: OpenAI `/v1/*` + Ollama-native `/api/*`
- [x] SSE keep-alive heartbeats during swap (⚠️ currently `/v1/*` streaming only)
- [x] `ds4` (systemctl --user) + Ollama (keep_alive:0) engine backends
- [x] `/status`, `/admin/swap`, `routerctl` CLI, systemd user service, persisted state
- [x] **LICENSE** — MIT (attribution to `rxxusp`). *[blocker — done]*
- [x] **API-key auth + safe default bind** — `Authorization: Bearer` / `X-API-Key`,
      constant-time check, `/health` exempt; default bind `127.0.0.1`; startup
      warning when exposed without keys. *[blocker — done]*

---

## Big rocks (provisional, ~1–3 weeks total)

### 1. Generalize the engine layer — **[L]**  *(gates the "engine-agnostic" claim)*
Replace the hardcoded `ds4`/`ollama` typed classes with reusable abstractions so
new engines need **config only, no Python**:
- `GenericProcessEngine` — `{ start_cmd, env, ready_url, ready_timeout_s, stop_signal }`.
  Covers **llama.cpp/llama-server, llamafile, vLLM, SGLang, Aphrodite**.
- `APISwapEngine` — load/unload by HTTP. Generalizes the current `OllamaEngine`
  and also covers **TabbyAPI**.
- Keep `Ds4Engine` as the bespoke escape hatch (systemd / odd lifecycle).
- Carry over the two things that already make us robust: per-engine cold-start
  timeouts (vLLM/SGLang take minutes) and SIGTERM→SIGKILL + port-close
  verification (llama.cpp has a confirmed SIGTERM-freeze bug).
- Do **not** target TGI (HuggingFace archived it, March 2026).

### 2. Extend SSE keep-alive to non-stream + Ollama NDJSON — **[M]**  *(headline-feature integrity)*
⚠️ Load-bearing caveat: today the keep-alive heartbeat fires **only on `/v1/*`
streaming**. Non-streaming calls and `/api/*` NDJSON streams still **block for
the entire swap** (up to 240 s for ds4) and will time clients out. Either emit
periodic holding frames on those paths too, or clearly document that non-stream
callers must raise their client timeout. Can't headline keep-alive until this is
fixed or scoped.

### 3. Packaging — **[M]**
`pyproject.toml` (pipx install of `routerctl` + the service) and a pinned-CUDA
Docker image published to `ghcr.io`. Discovery dead-ends at `git clone` without it.

### 4. Tests + CI — **[M]**
`pytest` + an async httpx client + a mock backend HTTP server can exercise
acquire/release/drain/swap **without a GPU**. Add a GitHub Actions workflow +
badge. (llama-swap has no visible test suite — a differentiator.)

### 5. Config validation + JSON Schema — **[M]**
Config loading is dataclass-based and silently ignores unknown keys. Add
startup validation with human-readable errors (e.g. "unit ds4.service not
found") and export a JSON Schema for editor autocomplete. (Note: README implied
pydantic but config is plain dataclasses — this is real work, not a one-liner.)

### 6. Prometheus `/metrics` — **[S, ~20 lines]**
Expose metrics no other proxy can: `swap_duration_seconds` (histogram),
`memory_settle_seconds`, `in_flight_at_swap_start`, `engine_uptime_seconds`.
They quantify the cost of GPU time-sharing.

### 7. README hero + semver/CHANGELOG — **[S–M]**
Tight problem statement + install + minimal config, leading with the
memory-settle + keep-alive differentiators. Adopt semver with a `v0.x` channel
and keep `CHANGELOG.md` current.

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
- Optional rename of the internal package/service (`router` / `llm-router`) to
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
