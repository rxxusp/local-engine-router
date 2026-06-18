# local-engine-router

[![CI](https://github.com/rxxusp/local-engine-router/actions/workflows/ci.yml/badge.svg)](https://github.com/rxxusp/local-engine-router/actions/workflows/ci.yml)

**On memory-constrained, unified-memory hardware, only one heavy LLM engine can
hold the GPU at a time.** local-engine-router is a single-port, OpenAI- and
Ollama-compatible reverse proxy that reads each request's `model` field, figures
out which local engine owns it, and **swaps engines on demand** so your clients
never have to know which backend is currently active. The proxy itself is **pure
Python and uses no GPU**.

Built and verified on a DGX Spark (GB10, 128 GB unified CPU+GPU memory), where
DeepSeek-V4-Flash alone uses ~81 GB and running two heavy engines simultaneously
causes OOM failures.

---

## How the GPU swap works

This is the core mechanic that distinguishes local-engine-router from simpler
proxies. When a request arrives for a model that belongs to a different engine
than the one currently holding the GPU, the router performs a full swap before
forwarding the request:

```
  client request (model: qwen3.6-uncensored:27b)
        │
        ▼
  resolve model → engine: ollama
        │
        ├── already active? ──yes──► forward immediately
        │
       no
        │
        ▼
  acquire _swap_lock
        │
        ▼
  [1] DRAIN in-flight requests on every non-target engine
      (wait up to drain_timeout_s, default 30 s)
        │
        ▼
  [2] FREE VRAM on every non-target engine
      (systemctl stop / keep_alive:0 / SIGTERM, etc.)
        │
        ▼
  [3] WAIT FOR OS MEMORY RECLAIM  ← THE DIFFERENTIATOR
      poll MemAvailable until it plateaus (~1 GiB between samples)
      or swap_memory_settle_timeout_s (default 25 s) elapses
        │
        │   WHY: after a heavy engine (e.g. ~81 GB ds4) stops,
        │   the kernel may not reclaim those pages for several
        │   seconds. If the next model starts while they are still
        │   resident, its pre-flight memory check sees less free
        │   memory than actually exists and fails with OOM.
        │   The memory-settle wait eliminates that race.
        │
        ▼
  [4] ensure_started() on the target engine
      poll readiness until HTTP 200 (up to start_timeout_s, default 240 s)
        │
        │   ┌────────────────────────────────────────────┐
        │   │  while waiting (streaming clients only):   │
        │   │  /v1/* SSE streams:  ": keepalive\n\n"    │
        │   │  /api/* NDJSON:      "\n" (bare newline)  │
        │   │  emitted every swap_keepalive_interval_s  │
        │   └────────────────────────────────────────────┘
        │
        ▼
  [5] active_engine = target; release _swap_lock
        │
        ▼
  increment in-flight counter; forward request to target engine
        │
        ▼
  response complete → release() decrements in-flight counter
```

**Keep-alive and disconnect safety.** Streaming responses start immediately; the
engine acquire runs inside the response generator, so a long cold start never
blocks with zero bytes sent to the client. A shielded asyncio task guarantees
`release()` is called even if the client disconnects mid-swap, preventing GPU
leaks. On client disconnect during a swap, the pending acquire is cancelled
cleanly on a normal control-flow path (not under `CancelledError`).

**Non-streaming requests** cannot carry keep-alive frames and block for the
entire swap. See [Sharp edges](#sharp-edges).


## Quickstart

```bash
# 1. Clone and install (Python >= 3.10; no GPU required)
git clone https://github.com/rxxusp/local-engine-router.git
cd local-engine-router
pip install .

# 2. Write a minimal config
cat > config.yaml << 'EOF'
host: 127.0.0.1
port: 8077

engines:
  llamacpp:
    type: generic_process
    base_url: http://127.0.0.1:8080
    start_cmd:
      - /usr/local/bin/llama-server
      - -m
      - /models/my-model.gguf
      - --port
      - "8080"
    ready_path: /health

models:
  - id: my-model
    engine: llamacpp
EOF

# 3. Start the router
local-engine-router --config config.yaml

# 4. Hit it (same interface as OpenAI)
curl http://127.0.0.1:8077/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hello"}],"stream":false}'
```

The router logs `SWAP begin:` / `SWAP done:` lines as it works. Check
`GET /health` for liveness and `GET /status` for full engine state.

### systemd (Linux only)

```bash
# One-shot idempotent installer: copies the unit, enables lingering, starts it.
bash deploy/install.sh

# Or manually:
mkdir -p ~/.config/systemd/user
cp deploy/local-engine-router.service ~/.config/systemd/user/
sudo loginctl enable-linger "$USER"    # boot-start without a login session
systemctl --user daemon-reload
systemctl --user enable --now local-engine-router
```

The unit file is `deploy/local-engine-router.service` and uses `%h` (systemd's
home-directory placeholder) for all paths, so it works for any user without
editing.

### Docker

```bash
docker run --rm -p 8077:8077 \
  -v "$PWD/config.yaml:/app/config.yaml" \
  ghcr.io/rxxusp/local-engine-router:latest
```

The image (`python:3.12-slim`, no CUDA) is published on every `v*` tag. See
[Container limitation](#container-limitation) under Sharp edges below.


## Platform support

| Platform | Router runs | Memory-settle | systemd unit |
|----------|------------|---------------|--------------|
| Linux    | yes        | yes (reads `/proc/meminfo`) | yes (`deploy/local-engine-router.service`) |
| macOS    | yes        | yes (via `psutil`, requires `pip install psutil`) | no |
| Windows  | yes        | yes (via `psutil`, requires `pip install psutil`) | no |

`psutil` is optional. When present the router uses it for cross-platform memory
polling; when absent it falls back to `/proc/meminfo` (Linux only). Install it
with `pip install psutil` on macOS or Windows.


## What makes this different

The general space (llama-swap, LocalAI, GPUStack, ...) is crowded; this targets
the **memory-constrained unified-memory** niche (GB10 / Apple Silicon) and does
four things no maintained tool does today:

1. **Explicit kernel memory-settle wait.** After freeing an engine the router
   polls `MemAvailable` until it plateaus *before* starting the next engine, so
   the incoming model's pre-flight memory check doesn't fail on pages the kernel
   hasn't reclaimed yet. On a GB10 with an ~81 GB model, this takes a few
   seconds and is the difference between a clean swap and an OOM failure.
2. **Manages engines it didn't spawn.** It can drive a `systemctl --user` unit
   with `Restart=always` (a plain SIGTERM would just respawn) -- structurally
   impossible with a pure `cmd:`-launches-the-process model.
3. **Native Ollama `/api/*` on a swap proxy.** Both the OpenAI `/v1/*` surface
   and Ollama-native `/api/*` are first-class and trigger swaps.
4. **Upstream-independent keep-alive** during long cold starts -- on both
   `/v1/*` SSE streams and `/api/*` NDJSON streams.


## Sharp edges

### Non-streaming requests block for the entire swap

A single JSON response body has nowhere to embed a keep-alive frame.
Non-streaming requests (`stream: false`) block from the moment they arrive until
the swap completes and the upstream returns. On a cold ds4 start that is up to
`start_timeout_s` (default 240 s). **Set your client read-timeout above the
worst-case swap** -- 300 s is a safe ceiling for most setups.

Streaming clients (`stream: true`) are not affected: they receive keep-alive
frames and never see a multi-second silence.

### Only one engine holds the GPU at a time

This is by design. The single-active invariant is the whole point of the router
on unified-memory hardware. There is no "run two engines in parallel" mode.

### Container limitation

The Docker image runs the router process only -- it cannot reach into the host's
process tree or `systemctl --user` namespace. Engines of type `generic_process`
(the router launches the server) and `ds4` (systemd-user lifecycle) do **not**
work inside the container. Only `api_swap` and `ollama` engines work from the
container, because they communicate over HTTP to servers already running on the
host.

Run the router on the host directly (pip install / systemd) if you need
process-control engines.

### Memory-settle reads /proc/meminfo (Linux only without psutil)

By default the router reads `/proc/meminfo`. On macOS and Windows, install
`psutil` or the memory-settle wait is skipped (logged as a warning). A skipped
wait may cause the incoming model's pre-flight memory check to fail with OOM on
heavily loaded systems.


## Authentication and binding

By default the router binds `127.0.0.1` (localhost only).

- **Bind:** set `host: 0.0.0.0` only if you need off-localhost access (e.g.
  Open WebUI in Docker reaching the host via `host.docker.internal`).
- **Auth:** set `api_keys` in `config.yaml`. When set, every request except
  `GET /health` must present a key via `Authorization: Bearer <key>` or
  `X-API-Key: <key>`. Keys are compared in constant time.
- If the router is bound off-localhost with no `api_keys`, it logs a security
  warning on startup.


## Backend presets

Ready-to-paste `engines:` + `models:` blocks for common backends live in
[`presets/`](presets/). Copy the relevant file into your `config.yaml` and fill
in the `<ANGLE_BRACKET>` placeholders.

| Backend | Preset file | Engine type |
|---------|-------------|-------------|
| llama.cpp (llama-server) | [`presets/llamacpp.yaml`](presets/llamacpp.yaml) | `generic_process` |
| vLLM | [`presets/vllm.yaml`](presets/vllm.yaml) | `generic_process` |
| SGLang | [`presets/sglang.yaml`](presets/sglang.yaml) | `generic_process` |
| KoboldCpp | [`presets/koboldcpp.yaml`](presets/koboldcpp.yaml) | `generic_process` |
| MLX (mlx-lm) | [`presets/mlx.yaml`](presets/mlx.yaml) | `generic_process` |
| TabbyAPI | [`presets/tabbyapi.yaml`](presets/tabbyapi.yaml) | `api_swap` |
| LM Studio | [`presets/lmstudio.yaml`](presets/lmstudio.yaml) | `api_swap` |
| LocalAI | [`presets/localai.yaml`](presets/localai.yaml) | `api_swap` |
| ramalama | [`presets/ramalama.yaml`](presets/ramalama.yaml) | `generic_process` |
| MAX (Modular) | [`presets/max.yaml`](presets/max.yaml) | `generic_process` |

See [`presets/README.md`](presets/README.md) for gotchas per backend (e.g. vLLM
reports a false `/health` 200 before the model is actually servable; the preset
uses `ready_path: /v1/models` + `ready_check: "model:<id>"` to work around it).


## Integrations

- **Open WebUI**: [`deploy/openwebui-wiring.md`](deploy/openwebui-wiring.md) --
  route all Open WebUI model requests through the router via Admin Panel
  (recommended, zero risk) or a `docker run` recreate.
- **OpenCode**: [`deploy/opencode.snippet.md`](deploy/opencode.snippet.md) --
  point both OpenCode providers at the router by changing two `baseURL` values
  in `~/.config/opencode/opencode.json`.


## Architecture

```
  OpenCode / curl / any OpenAI client
        │
        │  POST /v1/chat/completions (model: "qwen3.6-uncensored:27b")
        ▼
  ┌──────────────────────────────────────┐
  │        local-engine-router :8077     │
  │                                      │
  │  /v1/chat /v1/completions           │
  │  /v1/embeddings /v1/messages        │  ← OpenAI-compatible
  │  /v1/responses                      │
  │                                      │
  │  /api/chat /api/generate            │  ← Ollama-native
  │  /api/embeddings /api/embed         │
  │                                      │
  │  reads "model", resolves engine,    │
  │  swaps if needed, proxies           │
  └──────────┬───────────────┬──────────┘
             │               │
   ┌─────────▼────┐   ┌──────▼──────────┐
   │  ds4 :8099   │   │  Ollama :11434   │
   │  (~81 GB)    │   │  (various sizes) │
   └──────────────┘   └─────────────────┘
         ← only ONE active at a time →
         (single-GPU unified memory pool)

  swap: drain → free VRAM → wait for OS memory reclaim → start target
  streaming clients stay alive: SSE comments on /v1/*, bare newlines on /api/*
```


## Config reference

Copy `config.example.yaml` to `config.yaml` and edit for your machine. A
[JSON Schema (`config.schema.json`)](config.schema.json) ships in the repo;
point your editor's YAML language server at it for inline validation.

Validate without starting:

```bash
python3 -m router --check-config --config config.yaml
python3 -m router --print-schema    # print the JSON Schema
```

### Top-level keys

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to expose off-localhost (pair with `api_keys`). |
| `port` | `8077` | Listen port. |
| `api_keys` | `[]` | When non-empty, require a key on all requests except `GET /health`. |
| `allow_destructive_ollama_api` | `false` | Allow `/api/delete`, `/api/create`, `/api/copy`, `/api/push`, `/api/blobs` (refused with 403 when false). |
| `log_level` | `INFO` | Python log level. |
| `log_file` | `logs/router.log` | Rotating log (5 MB x 3 backups). |
| `state_file` | `state.json` | Persisted active-engine snapshot (re-probed on startup). |
| `swap_keepalive_enabled` | `true` | Emit keep-alive frames to streaming clients during swaps. |
| `swap_keepalive_interval_s` | `5.0` | Seconds between keep-alive frames. |
| `drain_timeout_s` | `30.0` | Max wait for in-flight requests before stopping an engine. |
| `swap_memory_settle_timeout_s` | `25.0` | Max wait for freed memory to plateau before starting the next engine. |
| `upstream_connect_timeout_s` | `15.0` | Connect timeout to backends (read timeout is unbounded). |

### Generic `engines:` table

Use this to add any number of engines with config only -- no Python needed.
`type` is one of `ds4`, `ollama`, `generic_process`, `api_swap`. When
`engines:` is present it is the **sole** source of engines (the `ds4:`/`ollama:`
legacy sections are ignored).

```yaml
engines:
  # Local server the router launches + supervises (llama.cpp, vLLM, SGLang, ...)
  llamacpp:
    type: generic_process
    base_url: http://127.0.0.1:8080
    start_cmd: ["/usr/local/bin/llama-server", "-m", "/models/foo.gguf", "--port", "8080"]
    ready_path: /health
    start_timeout_s: 300

  # Engine whose models load/unload over HTTP (TabbyAPI, LM Studio, ...)
  tabby:
    type: api_swap
    base_url: http://127.0.0.1:5000
    health_path: /v1/model
    unload_path: /v1/model/unload
    loaded_path: /v1/model
    loaded_models_key: data
    loaded_name_key: id

models:
  - { id: qwen2.5-7b-instruct, engine: llamacpp }
  - { id: my-tabby-model,       engine: tabby }
```

See `config.example.yaml` and `config.schema.json` for the full key reference
on `generic_process`, `api_swap`, `ds4`, and `ollama` engine types.

### Model aliases

Map a fixed client-side name to a real model id. The router rewrites the
request body so the upstream always sees the real id.

```yaml
aliases:
  gpt-4o-mini: qwen3.6-uncensored:27b
  claude-3-5-sonnet: deepseek-v4-flash
```


## Endpoint reference

| Method | Path | Behaviour |
|--------|------|-----------|
| GET | `/` | HTML status page |
| GET | `/health` | `{"status":"ok"}` -- liveness; never triggers a swap |
| GET | `/metrics` | Prometheus text exposition. Unauthenticated even when `api_keys` are set. |
| GET | `/status` | Full status: active engine, last swap, per-engine state, model list |
| GET | `/v1/models` | OpenAI model list: static registry + live Ollama tags, deduplicated. No swap. |
| POST | `/v1/chat/completions` | OpenAI chat; routed by `body.model` |
| POST | `/v1/completions` | OpenAI legacy completions; routed by `body.model` |
| POST | `/v1/embeddings` | OpenAI embeddings; routed by `body.model` |
| POST | `/v1/messages` | Anthropic messages format; routed by `body.model` |
| POST | `/v1/responses` | Responses API; routed by `body.model` |
| POST | `/api/chat` | Ollama-native chat; routed by `body.model` |
| POST | `/api/generate` | Ollama-native generate; routed by `body.model` |
| POST | `/api/embeddings` | Ollama-native embeddings; routed by `body.model` |
| POST | `/api/embed` | Ollama-native embed; routed by `body.model` |
| GET/POST | `/api/tags`, `/api/ps`, `/api/version`, `/api/show`, `/api/pull`, `/api/*` | Passthrough to Ollama, no swap. Destructive endpoints refused with 403 unless `allow_destructive_ollama_api: true`. |
| POST | `/admin/swap` | Body: `{"model":"<id>"}` or `{"engine":"<key>"}`. Proactive swap without a user request. |


## Metrics

`GET /metrics` exposes Prometheus text (format v0.0.4). No `prometheus_client`
dependency -- the exposition is hand-rolled.

| Series | Type | Meaning |
|--------|------|---------|
| `swap_duration_seconds` | histogram | Wall-clock duration of a full engine swap |
| `memory_settle_seconds` | histogram | Time spent waiting for memory to plateau |
| `in_flight_at_swap_start` | histogram | In-flight requests being drained at swap start |
| `swap_total{from,to,result}` | counter | Count of swaps by transition and result (`ok`/`error`) |
| `engine_uptime_seconds{engine}` | gauge | Seconds the active engine has been active |


## routerctl

`routerctl` is a thin CLI for inspecting and controlling the running router.

```bash
routerctl status                    # active engine, in-flight, last swap
routerctl models                    # list all known models
routerctl ds4                       # swap to ds4 now
routerctl ollama                    # swap to ollama now
routerctl use qwen3.6-uncensored:27b  # swap to whatever engine owns this model
routerctl logs                      # tail the service journal
routerctl restart                   # restart the service
```


## Operations

### Start / stop / restart

```bash
systemctl --user start   local-engine-router
systemctl --user stop    local-engine-router
systemctl --user restart local-engine-router
systemctl --user status  local-engine-router
```

### Logs

```bash
# Rotating file
tail -f ~/local-engine-router/logs/router.log

# systemd journal
journalctl --user -u local-engine-router -f
journalctl --user -u local-engine-router --since "1 hour ago"
```

### State file

`state.json` is written after every swap. It is a snapshot only; the router
re-probes reality on startup rather than trusting it.

```json
{"active_engine": "ollama", "last_swap": {"from": "ds4", "to": "ollama", "duration_s": 52.3, "ok": true}}
```


## Development and tests

The test suite is hermetic -- no GPU and no network (engines are replaced by a
mock backend), so it runs anywhere CI does:

```bash
pip install '.[dev]'      # adds pytest and pytest-asyncio
python3 -m pytest -q
```

CI runs the same suite on every push (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)).


## Troubleshooting

### Non-streaming request timed out

The client read-timeout is shorter than the swap. See [Sharp edges: non-streaming
requests block for the entire swap](#non-streaming-requests-block-for-the-entire-swap).
Set the client timeout above `start_timeout_s` (default 240 s); 300 s is a safe
ceiling.

### Swap to next engine fails with "more system memory than is available"

The outgoing model's memory hasn't been reclaimed yet. The router waits up to
`swap_memory_settle_timeout_s` (default 25 s). If you still hit this, the model
genuinely doesn't fit in available memory -- check `free -g` against the model
size. On macOS or Windows without `psutil`, the wait is skipped and this race is
more likely; install `psutil`.

### ds4 won't stop during a swap

```bash
systemctl --user stop ds4.service
pgrep -f ds4/ds4-server             # any leftover process?
kill -9 <pid>                       # last resort
```

Then retry via `routerctl ollama` or restart the router.

### Open WebUI model picker is empty

The container is missing `--add-host=host.docker.internal:host-gateway`. Without
it the container cannot resolve `host.docker.internal` and the picker shows
nothing. See [`deploy/openwebui-wiring.md`](deploy/openwebui-wiring.md) for the
safe recreate command.

### Port 8077 busy

```bash
ss -tlnp | grep 8077
systemctl --user stop local-engine-router
# kill the offending pid, then restart
```

### Ollama won't unload a model

The router sends `keep_alive: 0` then falls back to `ollama stop <name>`. If
models remain after `unload_timeout_s` (60 s), the router logs a warning and
proceeds. Unload manually:

```bash
ollama list
ollama stop <name>
```


## License

MIT. Attribution to `rxxusp`. See [`LICENSE`](LICENSE).

> The Python package is `router`; the console scripts are `local-engine-router`
> and `routerctl`; the project/repo is **local-engine-router**.
> See [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md) for where this is headed.
