# Roadmap v0.3 — Engine coverage & "make it better"

> **Status: provisional / not set in stone.** This is a working plan, not a
> commitment. Priorities and scope will change. Informed by a primary-source
> survey (2026-06-02) of llama.cpp router mode, vLLM, SGLang, llamafile,
> Aphrodite, KoboldCpp, text-generation-webui, MLX, Ollama, LM Studio, TabbyAPI,
> LocalAI, Jan/cortex, GPT4All, ramalama, Modular MAX, and the closest competitor
> llama-swap. Builds on the shipped **v0.2.0** engine layer (see `ROADMAP.md`).

## Progress (updated 2026-06-02)

The **do-next core — EC1–EC5 + MM4 — is DONE** (branch `engine-coverage-core`):
control-call auth headers, generic HTTP `load_path`, loaded-state filtering + id
keying, richer readiness probe, process-group reaping, and alias/capability
routing. 98 tests green. See `../STATUS.md` for the at-a-glance done/next picture
and the branch/PR map. Shipped items in §5 are marked **✅ SHIPPED**; everything
else is still **next up**. The next recommended item is **MM1 (cross-platform
memory + process portability)** — it gates the Apple-Silicon claim and the later
fit-checking/concurrency work.

## 1. Goal

Make the router support **all (or the most-used) local engines config-only** —
no new Python to onboard llama.cpp, vLLM, SGLang, llamafile, Aphrodite,
KoboldCpp, text-generation-webui, MLX, LM Studio, TabbyAPI, LocalAI, ramalama,
or MAX — by closing the small set of adapter-knob gaps the survey exposed
(generic HTTP load, control-call auth headers, richer readiness probes,
loaded-state filtering, process-group reaping, container-style stop commands).
Alongside that, land the targeted "make it better" improvements that the field
now treats as table stakes — **cross-platform memory portability first**
(because it gates the Apple-Silicon claim the README already makes), then
VRAM-aware fit-checking and optional concurrent models, OpenAI/Anthropic surface
completeness, alias/capability routing, a web dashboard, hot reload, and a
request queue with pinning/priority. All of this **without losing the
unified-memory niche** — the explicit kernel memory-settle wait, managing
engines we didn't spawn, native dual OpenAI+Ollama surface, and keep-alive
through cold starts remain the differentiators that no maintained tool combines.

## 2. Engine support matrix

Verified against primary sources on 2026-06-02. "Maps to" is the adapter the
engine fits **once the v0.3 adapter knobs in §4 land**; "today" gaps are what is
missing right now in v0.2.0.

| Engine | Category | Popularity | API surface | Load/Unload mechanism | Maps to | What's missing today |
|---|---|---|---|---|---|---|
| **llama.cpp / llama-server** | process; *also* native router/multi-model | Dominant GGUF engine (underlies Ollama/LM Studio/llamafile/Kobold) | OpenAI `/v1/*` incl. `/v1/rerank`, native Anthropic `/v1/messages`, `/health` (503 loading → 200 ok) | Single-model: restart to swap. **Router mode**: `--models-dir`/`--models-preset`/`--models-max N` (default **4**)/`--no-models-autoload`; `POST /models/load`, `POST /models/unload` (body `{"model":...}`); `GET /v1/models` with status `unloaded/loading/loaded/sleeping`; LRU evict only when > `--models-max`; built-in web UI | `generic_process` (single-model) ✓ today | Nothing for single-model. Router mode overlaps our core feature for GGUFs — see §3. Optional: drive router-mode load/unload via the new generic `load_path` |
| **vLLM** | process (one model/proc) | De-facto production OSS engine; ~50k★ | OpenAI `/v1/*`, `/v1/responses`, Anthropic `/v1/messages`, `/v1/audio/transcriptions`, rerank `/rerank`+`/v1/rerank`+`/v2/rerank`+`/score`+`/v1/score`, `/health`, `/v1/models` (all in released **v0.22.0**, pin min-version per endpoint) | Restart to swap. `/health` is **engine-alive only** — 200 ~16s after spawn, stays 200 before model-ready and even on GPU page-fault (#36960 open; no `/ready` or `/health/ready` shipped) | `generic_process` (`frees_own_memory=False`) ✓ | Readiness stronger than bare 200 (assert model in `/v1/models`); `process_pattern` to reap forked workers; larger `start_timeout_s` for CUDA-graph capture |
| **SGLang** | process (one model/proc) | Top-tier production; ~15k★ | OpenAI `/v1/*`; native `/generate`, `/health`, `/health_generate`, `/get_model_info` | Restart to swap | `generic_process` ✓ | **Cold start up to ~6 min with `torch.compile`** (~1:30 without) — exceeds 300s default `start_timeout_s`; raise to ≥600s. Prefer `/get_model_info` readiness |
| **llamafile** | process (single-file binary) | Well-known Mozilla project | OpenAI `/v1/*`, Anthropic Messages, `/health` (llama.cpp core) | Restart to swap | `generic_process` ✓ | Nothing — clean fit. Inherits llama.cpp SIGTERM-freeze caution (escalation already handled) |
| **Aphrodite** | process (vLLM fork) | PygmalionAI RP/creative community | OpenAI `/v1/*`, KoboldAI-compat; default port 2242; vLLM-derived `/health` | Restart to swap | `generic_process` ✓ | Same hardening as vLLM (bigger timeout, `process_pattern` reap, model-present readiness). Verify exact `/health` path on the build |
| **KoboldCpp** | process (single binary, llama.cpp core) | Very popular on r/LocalLLaMA (RP/story) | OpenAI `/v1/*`, native KoboldAI `/api/*`, Ollama shim; **no `/health`** | Restart to swap; `--singleinstance` lets a new same-port launch shut down the old one via localhost-gated `/api/extra/shutdown` | `generic_process` ✓ | `ready_path` must be `/api/v1/model` (returns `{"result":"<model>"}`); don't rely on `--singleinstance` for lifecycle. SIGTERM-freeze caution (llama.cpp core) |
| **text-generation-webui** | process **with HTTP load/list** | Long-standing, very popular | OpenAI+Anthropic `/v1/*`; **internal** `/v1/internal/model/load`, `/v1/internal/model/list`, `/v1/internal/logits` (default API port 5000) | Load model into persistent process via HTTP `POST /v1/internal/model/load`; switch by loading a different one. **Unload route UNCONFIRMED — re-verify** at `:5000/docs` | `api_swap` **once `load_path` lands**; else `generic_process` (restart-to-swap) | Generic HTTP `load_path` knob (does not exist today); confirm unload endpoint on the running build |
| **MLX / mlx_lm.server** | process (Apple-Silicon only) | THE Apple-Silicon server — defines our claimed niche | OpenAI `/v1/chat/completions`, `/v1/models`; **no `/health`**; default `localhost:8080` | Per-request model switch via `model` field (local path relative to start-dir, or HF repo id); **no HTTP unload** — GC on switch; stop process to free | `generic_process` (`ready_path=/v1/models`) ✓ for routing | **macOS portability**: no `/proc/meminfo`, no pgrep/systemd — biggest gap (§3, §5 Phase A). Document `ready_path=/v1/models` (no `/health`) |
| **Ollama** | api-load-unload | Dominant runtime; >150k★ | Native `/api/*` + OpenAI `/v1/*` + Anthropic `/v1/messages` (≥0.14.0); default 11434 | JIT load on first request; unload `POST /api/generate\|/api/chat {"model":n,"keep_alive":0}` or `ollama stop`; `keep_alive` 0=now/-1=never/default 5m; `GET /api/ps` loaded, `GET /api/tags` catalog; `OLLAMA_MAX_LOADED_MODELS` (default 3×GPUs) | **already supported** (`OllamaEngine`) ✓ | None. **No native rerank** (`/api/rerank` 404; #3368 open) — router cannot route rerank to Ollama. Can self-manage multiple loaded models → our single-active rule is conservative here |
| **LM Studio** | api-load-unload / desktop | Among most popular desktop apps; first-class Apple MLX | OpenAI `/v1/*`, native `/api/v0/*` and **`/api/v1/*` (v1 REST since 0.4.0)**, Anthropic `/v1/messages` (≥0.4.1); default 1234; `Authorization: Bearer` when key set | `lms server start`; `POST /api/v1/models/load` (body: model/context_length/eval_batch_size/flash_attention/num_experts/offload_kv_cache_to_gpu/echo_load_config — **no `ttl`/`identifier`**); `POST /api/v1/models/unload {"instance_id":...}` (**keyed by instance_id, not model**; no REST unload-all — only CLI `lms unload --all`); `/api/v0/models` lists all downloaded each with `state: loaded\|not-loaded`; JIT + Auto-Evict + idle TTL self-manage | `api_swap` + new knobs, **or** `defer_unload` self-manager | Bearer auth header; `loaded_filter` (`state==loaded`); `loaded_id_key`/unload body keyed to `instance_id`; optional pre-load; `self_manages_memory` flag |
| **TabbyAPI** | api-load-unload | Standard EXL2/EXL3 server (SillyTavern crowd); Linux/Windows, no Apple | OpenAI `/v1/*`; `/v1/model` (single loaded), `/v1/model/load`, `/v1/model/unload`, `/v1/models`, `/v1/model/list`; `GET /health` → `{"status":"healthy","issues":[...]}` | `POST /v1/model/load {"model_name":...}` (+optional max_seq_len/tensor_parallel/gpu_split/cache_mode/draft_model); **`POST /v1/model/unload` takes NO body, returns null**; both require **`x-admin-key`** header; auto-unloads current on a new load; no JIT | `api_swap` (near-perfect) + admin-key | **Admin-key header on control calls (#1 blocker)**; parse single-object `/v1/model`; optional `load_path` (+ SSE load-progress / longer load_timeout) to pre-warm |
| **LocalAI** | api-load-unload / proxy | Popular self-hosted multi-backend gateway; Linux/macOS/Windows | OpenAI `/v1/*` + images/audio; `GET /v1/models`, `POST /models/apply`, **`POST /backend/shutdown {"model":...}`** to unload; `GET /readyz`/`/healthz`; default 8080 | Implicit load; explicit unload via `/backend/shutdown`. **`--max-active-backends=1`** (env `LOCALAI_MAX_ACTIVE_BACKENDS`; supersedes deprecated `--single-active-backend`) auto-unloads-before-load; idle watchdog `LOCALAI_WATCHDOG_IDLE` + `_TIMEOUT` (Go-duration, default **15m**) | `api_swap` (`unload_path=/backend/shutdown`, `health_path=/readyz`) **or** `defer_unload` | **No list-loaded-backends endpoint** → can't confirm freed via `loaded_path`, rely on memory-settle; Bearer auth; `self_manages_memory` flag (with max-active-backends=1 it self-swaps) |
| **Jan / cortex** | desktop / proxy | Popular desktop app; cross-platform incl. Apple | OpenAI `/v1/chat/completions`, `/v1/models`; default `127.0.0.1:1337`; optional Bearer; **no public load/unload** | Jan **v0.8.0 (2026-05-22)** runs a single unified **llama.cpp router process**; load/unload live **inside** llama-server (`POST /models/load`, `/models/unload`), **not on Jan's public API**. cortex.cpp **archived 2025-07-04** (moved to menloresearch/llama.cpp) | `api_swap` **proxy-only** (no `unload_path`) / self-managing | Router can only proxy; cannot reclaim on demand → reinforces `self_manages_memory`. Headless spawn brittle (GUI-first) |
| **GPT4All** | desktop | Long-standing consumer app; CPU-first | Subset OpenAI on `:4891/v1` (`/v1/models`, `/v1/chat/completions`, `/v1/completions`); localhost only; no auth | UI-selected model only; **no API load/unload**; memory freed only via GUI; one model at a time | `api_swap` **proxy-only** (no unload) — effectively out of scope for router-driven swapping | No headless server, no load/unload API → router cannot manage lifecycle. Low priority (CPU-first, no agentic focus) |
| **ramalama** | process (container) | Fast-growing Red Hat/Podman project; Linux/macOS | OpenAI `/v1/*` served by in-container llama.cpp/vLLM; default **a free port in 8080–8180** | `ramalama serve [-d] MODEL` (detached prints container id; default containerized); **stop via `ramalama stop <name>`** (stops/removes container); **no in-server unload** — memory freed by stopping the container | **needs `stop_cmd` knob** (closest to `generic_process` but stop is a CLI/container op, not a signal) | Configurable **stop command** (`ramalama stop {name}`) instead of SIGTERM-by-pid; longer start_timeout for image/model pulls |
| **Modular MAX** | process (one model/proc) | Modular's production platform; growing | OpenAI `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, **`/v1/health`** (note the `/v1/` prefix); default port **8000** | Restart to swap; `--device-memory-utilization` for KVCache fraction | `generic_process` ✓ | Raise `start_timeout_s` for graph/kernel compile; `ready_path=/v1/health`; `process_pattern` to reap workers |
| **PowerInfer** | process (llama.cpp fork) | Niche research engine (~8k★) | Inherits llama.cpp native server; **OpenAI `/v1` UNCONFIRMED** on current build | Restart to swap | `generic_process` *if* it serves `/v1` | Lowest priority; staleness/divergence risk; requires PowerInfer-format models |
| **TGI (text-generation-inference)** | — | **ARCHIVED 2026-03-21** (read-only/maintenance) | — | — | **SKIP** | README steers users to vLLM/SGLang/llama.cpp/MLX. Do not integrate |

## 3. Gap analysis (router vs the field, and vs what these engines need)

- **Concurrency / co-residency.** The router enforces **exactly one active
  engine**, always, even when memory could hold two (a 27 GiB chat model + a
  small embeddings/reranker fit easily in ~121 GiB). This is the biggest visible
  delta vs llama-swap, whose shipped **"matrix" solver** (issue #643 closed;
  DSL `&`/`|`/`()`/`+ref` + admin-assigned `evict_costs`, lowest-cost eviction;
  *not* real VRAM) and the now-legacy `groups` model both run >1 model
  concurrently. llama.cpp router mode holds up to `--models-max` (default 4) and
  Ollama up to `OLLAMA_MAX_LOADED_MODELS`. The chat+embeddings and chat+reranker
  patterns are impossible today.
- **VRAM-aware fit-checking.** Memory accounting is only a `MemAvailable`
  plateau poll *after* a stop (`engines.py:_await_memory_settle`); there is **no
  per-model footprint estimate and no pre-flight fit check**, so the router can
  start a model that then OOMs and cannot reason about co-residency. *Evidence:*
  GGUF metadata + ctx + KV-dtype gives a usable estimate (gpustack/gguf-parser-go,
  oobabooga's GGUF-VRAM formula). llama-swap has **zero** memory accounting
  (TTL-only) — this is our linchpin differentiator to *extend*, not abandon.
- **Cross-platform memory (Linux-only today, but we claim Apple Silicon).** The
  settle wait reads `/proc/meminfo` and **returns early when it's unreadable**
  (`engines.py:961-964`) — so on macOS the swap proceeds *without* waiting for
  reclaim. Process control uses `pgrep -f` + `os.killpg` + systemd, none of which
  behave the same on macOS/Windows. *Good news:* on the flagship the system-RAM
  signal is the **correct** one — GB10 `nvidia-smi`/`nvmlDeviceGetMemoryInfo`
  report "Not Supported", so `MemAvailable` (not NVML) is right. On Apple
  Silicon the analogue is `psutil.virtual_memory().available` (wraps Mach
  `host_statistics64`, includes reclaimable inactive/purgeable pages; 16 KiB
  pages; "free" alone mis-fires under lazy reclaim). Metal counters are
  unsuitable (`currentAllocatedSize` is **per-process**; `recommendedMaxWorking‑
  SetSize` ≈ 75% of RAM is a cap, not free-VRAM). This is the single largest
  portability gap and it gates the headline positioning.
- **OpenAI/Anthropic API completeness.** The router is a **byte-level
  pass-through** (`app.py _handle_v1_post` / `_handle_api_post_with_model`) — it
  does not transform bodies, so surface "completeness" is inherited from the
  resolved upstream. Tool calling / structured outputs / vision ride inside the
  chat body and pass through fine *if the engine supports them*. But there is
  **no `/v1/rerank` and no `/v1/audio/*` route at all** (silent 404s; Ollama has
  no native rerank anyway), and `/v1/messages`/`/v1/responses` only work for
  engines that natively speak them (Ollama ≥0.14, LM Studio ≥0.4.1, llama.cpp,
  vLLM v0.22.0 do; ds4/legacy may not — no translation shim).
- **Discovery.** Models are a hand-written static `models:` list (plus live
  Ollama `/api/tags` fallback for routing). No scan of a GGUF dir / HF cache /
  LM Studio catalog. llama.cpp `--models-dir` auto-discovers and auto-loads; this
  is the main config-maintenance friction and a prerequisite for a good UI picker.
- **Web UI.** Only a minimal HTML home page + `/status` JSON + Prometheus
  `/metrics`. llama-swap ships `/ui` (live `/logs/stream`, running-model view) and
  even llama.cpp's own server now has a web UI. A read-only dashboard is now
  table-stakes credibility.
- **Hot reload.** Config is loaded once at startup (`config.load_config`); adding
  a model/alias/engine needs a full restart that drops streams and cold-reloads
  the active model. (llama-swap's `-watch-config` *restarts* the proxy; a
  graceful reload is still desired — **re-verify** the current state of
  llama-swap #160/#547 before quoting them.)
- **Queue / priority / preemption.** Just an asyncio lock + in-flight drain. No
  model pinning, no priority lanes, no protection against a background embedding
  call evicting the chat model mid-conversation. (Ollama queues and only evicts
  *idle* models; the explicit "don't swap while running" knob does **not** exist
  in llama-swap either — #588 was closed via *request collation* in PR #790,
  merged 2026-05-29, not a drain knob, so our in-flight drain is "matched in
  spirit, not by feature" — **re-verify exact semantics**.)
- **Capability/alias routing.** Routing is by **exact** model id only. A client
  asking for `gpt-4o-mini` or `claude-3-5-sonnet` 404s unless a local model is
  named identically — the most common wiring blocker (Claude Code, OpenAI SDKs).
- **Auth control headers.** The internal control client sends **no configurable
  auth header**, so loaded/unload/health calls **401** against a secured
  TabbyAPI (`x-admin-key`) or LM Studio/LocalAI (`Bearer`) — a hard blocker for
  that whole lane.

## 4. The minimal engine-adapter interface

The four existing classes already imply a clean contract; v0.3 promotes two
buried decisions into **declared capabilities** so a new engine becomes pure YAML.
Every adapter must provide:

| Capability | What it does | Already satisfied by |
|---|---|---|
| **`ensure_started()`** | Bring the engine up — launch a process *or* confirm a persistent service answers | All four (`Ds4Engine`, `GenericProcessEngine`, `APISwapEngine`, `OllamaEngine`) |
| **`stop()` / `free_memory(model=None)`** | Release the engine's memory and **verify it** — kill-process+port-closed, OR unload-over-HTTP+poll-until-empty | All four (`free_vram`); `GenericProcessEngine`/`Ds4Engine` free by exit, `APISwapEngine`/`OllamaEngine` free per-model |
| **`is_ready()`** | HTTP probe of a ready/health path → 200 (the **authoritative** signal; keep the port/HTTP probe over `net_connections()`) | All four (`is_ready`) |
| **`list_models()`** | Ids this engine can serve — static set (process) or live catalog (`/api/tags`, `/v1/models`) | `APISwapEngine`/`OllamaEngine` (`loaded_models`); process engines = static config set |
| **`load(model)`** *(new knob)* | Optional explicit HTTP load into a running process (mirror of `unload_path`): `load_path`/`load_method`/`load_body` (`{model}`) | **None today** — needed for text-generation-webui, TabbyAPI pre-warm, router-mode llama-server |
| **`estimate_footprint(model)`** *(new)* | Bytes the model will occupy (weights + KV + overhead), for fit-checking | **None** — start as a static `footprint_bytes` config field; GGUF-header parse later |
| **`frees_own_memory`** *(new flag)* | True if a per-model unload frees memory without the process dying (api_swap) vs only-by-exit (process) | Implicit in `isinstance` today: `api_swap`=True, process=False — promote to a declared flag |
| **`serves_concurrently`** *(new flag)* | Can hold >1 model / co-exist with others when memory allows | Universally **false** today (single active engine) — declare it to unlock co-residency |

Promoting `frees_own_memory`/`serves_concurrently`/`estimate_footprint` removes
the `isinstance` branching in `EngineManager._swap_to` and is the prerequisite
for both fit-checking and concurrency — neither needs a behavior change until a
feature consumes the flags (defaults preserve today's behavior). Supporting knobs
that ride alongside: **control auth headers**, **loaded_filter** (`key==value`,
e.g. `state==loaded`), **loaded_id_key** (unload by `instance_id`), and a
**`stop_cmd`** (container/CLI lifecycle, e.g. ramalama).

## 5. Roadmap — prioritized phases / big-rocks

Effort tags `[S]` ≈ hours, `[M]` ≈ a day or two, `[L]` ≈ multi-day/architectural.
Sequenced by dependency and value.

### Engine coverage (config-only support for the major engines)

- ✅ **EC1. Control-call auth headers `[S]`** — thread an optional `control_headers`
  dict into the internal `_ctl` client. *Rationale:* the #1 hard blocker;
  unlocks secured TabbyAPI (`x-admin-key`), LM Studio/LocalAI (`Bearer`). Do first.
- ✅ **EC2. Generic HTTP `load_path` (+method/body/timeout) `[S]`** — symmetric
  mirror of the existing `unload_*` machinery, called in `ensure_started`.
  *Rationale:* unlocks text-generation-webui and TabbyAPI/LM-Studio pre-warm
  without restart; tiny, reuses `{model}` renderer.
- ✅ **EC3. Loaded-state filtering + id keying `[S/M]`** — `loaded_filter`
  (`state==loaded`) + `loaded_id_key` (unload by `instance_id`) + parse a
  single-object loaded response (TabbyAPI `/v1/model`). *Rationale:* without it
  the router treats every *downloaded* LM Studio model as loaded and unloads by
  the wrong key.
- ✅ **EC4. Richer readiness probe `[M]`** — optional JSON assertion (`status==ok`
  / model present in `/v1/models`) or a secondary probe path. *Rationale:* vLLM
  `/health` is false-ready (200 before model-ready, even on GPU fault); MLX/Kobold
  have no `/health`. Makes `generic_process` robust across the lane.
- ✅ **EC5. Process-group reaping + per-engine timeouts `[M]`** — `setsid`+`killpg`
  by default for `generic_process`, document `process_pattern` as effectively
  required for vLLM/SGLang/Aphrodite/MAX; ship sane `start_timeout_s` defaults
  (SGLang+compile ≥600s; vLLM/MAX generous). *Rationale:* orphaned forked workers
  pin unified memory and defeat the settle wait — the exact OOM class we exist to
  prevent.
- **EC6. `self_manages_memory` / `defer_unload` flag `[S/M]`** — mark an engine
  as never-needs-router-freeing; skip `free_vram` and the settle wait while
  staying within it; only swap across *different* engines. *Rationale:* LM Studio
  (Auto-Evict+TTL), LocalAI (`max-active-backends=1`), Jan (router mode) already
  self-swap — fighting them is wrong; deferring gets correct behavior for free.
- **EC7. `stop_cmd` for container/CLI engines `[M]`** — allow a full stop command
  (`ramalama stop {name}`) instead of SIGTERM-by-pid, plus longer start timeout
  for image/model pulls. *Rationale:* generalizes beyond signal-based control to
  Podman/Docker-managed engines.
- **EC8. Ready-made config presets + docs `[S]`** — `config.example` snippets for
  llama.cpp, vLLM, SGLang, llamafile, KoboldCpp, MLX, TabbyAPI, LM Studio,
  LocalAI, ramalama, MAX. *Rationale:* turns the lane into copy-paste and proves
  the new knobs cover real engines.
- **EC9. Model auto-discovery `[M]`** — scan a GGUF dir / HF cache / live Ollama
  `/api/tags` / LM Studio catalog and self-populate the registry (config wins on
  conflict). *Rationale:* removes the main config-maintenance friction; a UI-picker
  prerequisite. Depends loosely on the footprint parse (MM2).

### Make it better

- **MM1. Cross-platform memory + process portability `[L]` — DO FIRST.** A
  `mem.available_bytes()` provider dispatched on `sys.platform` (Linux
  `MemAvailable`; macOS/Windows `psutil.virtual_memory().available`) feeding
  `_await_memory_settle`; swap `pgrep`/`killpg` for `psutil`
  `process_iter`/`terminate`/`wait_procs`/`kill` (systemd stays a Linux opt-in);
  keep the HTTP port probe as the readiness oracle. *Rationale:* **gates the
  Apple-Silicon claim the README already makes** — today the settle wait silently
  no-ops on macOS. On GB10 the existing signal is already correct (document that
  NVML is intentionally bypassed).
- **MM2. VRAM/RAM fit-checking `[M]`** — `estimate_footprint` (static config
  field first, GGUF-header parse next) + a pre-load `available ≥ footprint +
  headroom` gate; hardware-class dispatch (NVML *only* for discrete GPUs, system
  RAM for unified, DXGI for Windows dGPU). *Rationale:* refuse/queue instead of
  OOMing; prerequisite for fit-by-memory concurrency. Depends on MM1's mem layer.
- **MM3. Rerank + audio passthrough & surface audit `[S]`** — add `/v1/rerank`
  (+`/rerank`) and `/v1/audio/*` as model-bearing routes that swap+forward like
  chat; document the engine-by-surface support matrix honestly. *Rationale:*
  rerank is a core RAG primitive vLLM/llama.cpp already serve; cheap win.
- ✅ **MM4. Alias / capability routing `[S]`** — an `{alias → model_id}` map resolved
  before the existing index lookup (+ optional downstream rename). *Rationale:*
  highest-leverage tiny change; unlocks Claude Code / `gpt-4o-mini` wiring today.
- **MM5. Hot config reload `[S/M]`** — re-read YAML on SIGHUP / `/admin/reload`,
  diff, apply additively (add models/aliases cheap; changing the active engine
  drains first); schema validation already gates bad reloads. *Rationale:* a
  one-line edit shouldn't drop streams and cold-reload the model.
- **MM6. Web dashboard `[M/L]`** — read-only single-page `/ui`: active engine,
  in-flight count, last/in-progress swap duration, MemAvailable gauge, log tail;
  `POST /admin/swap` button. *Rationale:* table-stakes credibility; data already
  exists in `/status`+`/metrics`. Keep read-only first to dodge auth/CSRF scope.
- **MM7. Request queue, pinning & preemption `[M/L]`** — `pinned` flag (skip
  eviction), "don't swap while in-flight unless `preempt=true`" (queue the swap),
  then priority lanes. *Rationale:* makes a swapping router safe under concurrent
  agent+chat+RAG load. The drain lock is the foundation.
- **MM8. Concurrent / co-resident models `[L]` — the big one.** Generalize
  `EngineManager` from a single `active_engine` to an active-set with
  admission control: if `available ≥ Σ footprints`, run both; else fall back to
  stop-one-start-one. Start incrementally with a declared always-resident small
  engine (embeddings) alongside the swapped heavy one. *Rationale:* closes the
  largest gap to llama-swap; the unified-memory settle/measure machinery (MM1/MM2)
  is the reusable foundation no competitor has. Depends on MM1+MM2 and the §4 flags.
- **MM9. Cold-start mitigation `[S/M]`** — startup `preload` list + keep-warm/TTL
  semantics (bounded to one warm engine until MM8). *Rationale:* cold start is the
  dominant UX cost; removes the first-hit penalty for the common path.
- **MM10. Multi-tenant auth tiers + usage accounting `[M]`** — named/virtual keys,
  per-key RPM/TPM + budgets, token accounting via response `usage`. *Rationale:*
  standard once shared; lower priority than the above for a single-user box.

### Recommended sequencing

**Do NEXT (the unblocking core):** EC1 → EC2 → EC3 → MM4 (alias) → EC4 → EC5,
then **MM1** (portability) before anything depending on memory, then EC8 presets
and MM3 rerank as quick credibility wins. **Then** MM2 → MM5 → MM6 → MM7, with
**MM8 (concurrency)** as the headline once MM1+MM2 land.

**Keep OUT of scope:** TGI (archived), GPT4All (no headless/no load API — proxy
only at best), PowerInfer (niche, llama.cpp covers the hardware better), genuine
Anthropic↔OpenAI↔Ollama *body translation* (LiteLLM does it better in front —
only worth a shim for ds4/legacy), and all cluster/multi-node tooling
(GPUStack / vLLM production-stack / NIM / Triton / exo are the opposite niche —
add one "use GPUStack instead if you have a cluster" line and move on). Compose
**with** LiteLLM (front) and Harbor (provisioning underneath); don't rebuild them.

## 6. Honest risks

- **The niche narrowed under us.** llama.cpp's own `llama-server` now ships
  **router mode** (auto-discovery, `--models-max` LRU, `/models/load|unload`,
  built-in web UI) and Jan v0.8.0 wraps it — so "swap models behind one port" is
  built into llama.cpp itself for GGUFs. Our defensible edge collapses to
  **cross-engine** swap (llama.cpp + vLLM + Ollama + MLX + TabbyAPI under one
  OpenAI/Ollama port) **with correct unified-memory settle** — real, but narrow.
  Consider a "delegate GGUF swapping to router-mode llama-server, own only
  cross-engine swaps" integration mode.
- **"A less-featured llama-swap in Python."** llama-swap is far ahead
  operationally (Go, ~4.4k★, `/ui`, GPU+system telemetry, matrix concurrency,
  Homebrew/multi-arch). The standing **open decision** survives: seriously cost
  out **upstreaming** the meminfo-settle wait + external/systemd lifecycle as PRs
  before investing in standalone publishability. Our two clean wins it cannot do
  (manage engines it didn't spawn; native Ollama `/api/*`) plus the *only*
  memory-settle wait in the field are the case for shipping separately.
- **Concurrency breaks the core invariant.** MM8 is genuinely architectural —
  single-active is woven through the swap state machine. Phase it (always-resident
  small engine first) and gate strictly on MM2 fit-checks or it will OOM the box.
- **Maintenance burden across hardware we don't run.** A swap state machine over
  many engines/kernels means bug reports on macOS/Windows/discrete GPUs we can't
  reproduce; llama.cpp's SIGTERM-freeze and KoboldCpp router bugs are a preview.
- **Footprint estimation is engine-specific.** GGUF parsing won't cover
  ds4/vLLM/EXL formats; those need configured/measured fallbacks, and a wrong
  estimate either OOMs or wastes memory.
- **Modest TAM today** (GB10 / DGX Spark owners), growing with Apple Silicon —
  which makes MM1 portability the highest-ROI bet, not a side quest.

---
*Facts above are from a **2026-06-02** primary-source survey; this space moves
weekly (star counts, versions, issue states, exact endpoint paths/flags) —
**re-verify before quoting**. Specifically flagged to re-verify: text-generation-
webui's unload route; llama-swap #160/#547 (hot reload) and the in-flight-drain
semantics of PR #790/#588; the anecdotal GB10 "30s-to-minutes / ~40% load
failure" figures (the qualitative reclaim lag is well-supported, the percentages
are third-party, not an NVIDIA spec).*
