# Project status

> Living snapshot of what's shipped and what's next. Last updated **2026-06-02**.
> Detailed plans live in `roadmap/ROADMAP.md` (v0.2) and
> `roadmap/ROADMAP-v0.3-engine-coverage.md` (v0.3). Changelog: `CHANGELOG.md`.

## TL;DR

`local-engine-router` is a pure-Python FastAPI reverse proxy that fronts local
LLM engines on memory-constrained unified-memory hardware (DGX Spark / GB10,
Apple Silicon) and **swaps engines on demand** because only one heavy model fits
at a time. Core (v0.1) shipped; a generic engine layer + packaging + tests +
metrics (v0.2.0) shipped; the v0.3 **engine-coverage core** (config-only support
for the major engines) shipped. Remaining v0.3 work is queued below.

## Done

### v0.1 — core (on `main`'s lineage)
Swap state machine (drain → free other engine → `/proc/meminfo` memory-settle →
start target), dual OpenAI `/v1/*` + Ollama `/api/*` surface, SSE keep-alive on
`/v1` streaming, ds4 + Ollama engines, `/status` · `/admin/swap` · `routerctl` ·
systemd user service, API-key auth, safe default bind, MIT license.

### v0.2.0 — build-out (branch `roadmap-buildout`, **PR #1 open**)
- **Generic engine layer**: `GenericProcessEngine` (llama.cpp/vLLM/SGLang/
  llamafile/Aphrodite), `APISwapEngine` (TabbyAPI etc.), `OllamaEngine` (preset),
  `Ds4Engine` (escape hatch); optional top-level `engines:` table; legacy
  ds4+ollama path unchanged when absent.
- **Keep-alive on every streaming path** (`/v1` SSE + `/api` NDJSON).
- **Config validation** (`ConfigError`, `--check-config`) + **JSON Schema**
  (`config_json_schema()`, `config.schema.json`, `--print-schema`).
- **Prometheus `/metrics`** (zero-dep): swap_duration / memory_settle /
  in_flight_at_swap_start / swap_total / engine_uptime.
- **Packaging**: `pyproject.toml` + console scripts, `router/cli.py`, pure-Python
  Dockerfile, ghcr publish workflow. **Tests + CI**: hermetic pytest suite + GH Actions.

### v0.3 — engine-coverage core, EC1–EC5 + MM4 (branch `engine-coverage-core`)
Makes the major local engines onboard **config-only**. 98 tests green.
- **EC1** `control_headers` (control client only) → secured TabbyAPI / LM Studio / LocalAI.
- **EC2** generic HTTP `load_path` + per-model load on acquire → explicit-load engines.
- **EC3** `loaded_filter` + `loaded_id_key` + single-object `loaded_path` (LM Studio / TabbyAPI).
- **EC4** richer `ready_check` (`key==value` / `model:<id>`) → fixes vLLM false-ready.
- **EC5** process-group reaping (`os.killpg`) → reaps vLLM/SGLang forked workers.
- **MM4** alias / capability routing + outgoing-body rewrite (`gpt-4o-mini` → real model).

## Next up (v0.3 remainder, in recommended order)

See `roadmap/ROADMAP-v0.3-engine-coverage.md` §5 for full detail + effort tags.

1. **MM1 — cross-platform memory + process portability `[L]` (do first).** The
   memory-settle wait reads Linux `/proc/meminfo` and the process control uses
   `pgrep`/`killpg`/`systemd`; on macOS the settle wait silently no-ops. Gates
   the Apple-Silicon claim the README already makes, and everything below it.
2. **EC8 — ready-made config presets + docs `[S]`** for llama.cpp/vLLM/SGLang/
   KoboldCpp/MLX/TabbyAPI/LM Studio/LocalAI/ramalama/MAX (proves the new knobs).
3. **MM3 — rerank + audio passthrough & surface audit `[S]`** (`/v1/rerank`, `/v1/audio/*`).
4. **MM2 — VRAM/RAM fit-checking `[M]`** (`estimate_footprint` + pre-load gate). Needs MM1.
5. **EC6 — `self_manages_memory` / `defer_unload` flag `[S/M]`** (LM Studio / LocalAI / Jan self-swap).
6. **EC7 — `stop_cmd` for container/CLI engines `[M]`** (ramalama / Podman).
7. **MM5 — hot config reload `[S/M]`**, **MM6 — read-only web dashboard `[M/L]`**.
8. **EC9 — model auto-discovery `[M]`** (scan GGUF dir / HF cache / Ollama / LM Studio).
9. **MM7 — request queue, pinning & preemption `[M/L]`**.
10. **MM8 — concurrent / co-resident models `[L]` (the big one)** — needs MM1+MM2.
11. **MM9 — cold-start mitigation `[S/M]`**, **MM10 — multi-tenant auth tiers `[M]`**.

### Deferred / out of scope (intentional)
TGI (archived), GPT4All (no headless load API), PowerInfer (niche), genuine
Anthropic↔OpenAI body translation (LiteLLM does it better in front), and all
cluster/multi-node tooling (GPUStack / NIM / Triton / exo — opposite niche).
Also unbuilt from v0.2's roadmap: priority preemption (now MM7), package rename,
and the publish-vs-upstream-to-llama-swap decision.

### Strategic note
The v0.3 survey found llama.cpp's own `llama-server` now ships a **router mode**
(and Jan/LM Studio/LocalAI self-swap), so "swap models behind one port" is
increasingly built into individual engines. The defensible edge is **cross-engine**
swapping (llama.cpp + vLLM + Ollama + MLX + TabbyAPI under one OpenAI/Ollama port)
**with the only correct unified-memory settle wait in the field** — which is why
MM1 (portability) is the highest-ROI next bet.

## Branch / PR map

| Branch | Contains | State |
|---|---|---|
| `main` | initial commit only | base |
| `roadmap-buildout` | v0.2.0 build-out | **PR #1 open** (base `main`) |
| `planning-v0.3-engine-coverage` | the v0.3 research roadmap doc (superseded by the copy on `engine-coverage-core`) | pushed |
| `engine-coverage-core` | v0.2.0 **+** v0.3 core (EC1–EC5 + MM4) **+** this doc | pushed, no PR yet (stacks on `roadmap-buildout`) |

Suggested merge order: PR #1 (`roadmap-buildout` → `main`) first, then open a PR
for `engine-coverage-core` (rebased onto `main`).
