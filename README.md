# local-engine-router

A lightweight OpenAI- and Ollama-compatible reverse proxy that sits in front of
`ds4` and `Ollama` on a DGX Spark (GB10, 128 GB unified memory). Because the
two engines share one physical memory pool, only one heavy engine can hold the
GPU at a time. The router enforces mutual exclusion, swapping automatically when
the requested model lives on the other engine.

> The Python package is `router` and the systemd unit is `llm-router`; the
> project/repo is **local-engine-router**. Licensed **MIT** (attribution to
> `rxxusp`). See [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md) for where this is headed.

## Quick start

```bash
cp config.example.yaml config.yaml      # then edit for your machine
python3 -m router --config config.yaml   # or: bash deploy/install.sh
```

By default the router binds **127.0.0.1** (localhost only). To reach it from a
Docker container (e.g. Open WebUI via the bridge gateway) set `host: 0.0.0.0`
in `config.yaml` — and then set `api_keys` (see below) or rely on a host firewall.

## Authentication & binding

- **Bind:** `host` defaults to `127.0.0.1`. Only widen to `0.0.0.0` if you need
  off-localhost access; the router logs a security warning if it is bound
  off-localhost with no API keys.
- **Auth:** set one or more `api_keys` in `config.yaml`. When set, every request
  except `GET /health` must present a key via either header:
  - `Authorization: Bearer <key>`
  - `X-API-Key: <key>`

  Keys are compared in constant time. Configure the same key in OpenCode
  (provider `apiKey`) and Open WebUI (connection API key). Leave `api_keys`
  empty only on a localhost-only bind.


## Architecture

```
                        ┌─────────────────────────────────────────┐
  OpenCode              │              llm-router :8077            │
  (localhost)  ──────►  │                                         │
                        │  POST /v1/chat/completions               │
  Open WebUI            │       /v1/completions                    │
  (docker)     ──────►  │       /v1/embeddings                     │
  host.docker.          │       /v1/messages, /v1/responses        │
  internal:8077         │       /api/chat, /api/generate, ...      │
                        │                                         │
                        │  reads "model" field, resolves engine   │
                        │  swaps if needed, then proxies          │
                        └────────────┬──────────────┬────────────┘
                                     │              │
                     ┌───────────────▼──┐    ┌──────▼───────────────┐
                     │  ds4 :8099       │    │  Ollama :11434        │
                     │                  │    │                       │
                     │  deepseek-v4-    │    │  qwen3.6-*, qwen35-*, │
                     │  flash / pro     │    │  nemotron-*, ...      │
                     │  (~81 GB)        │    │  (various sizes)      │
                     └──────────────────┘    └───────────────────────┘
                     ← only ONE active ──────────────────────────── →
                       at a time (GB10 unified memory pool)

  swap: drain in-flight → stop other engine → wait for memory to settle → start target
  streaming clients stay alive via SSE ": keepalive" comments during the wait
```


## Why only one engine at a time

The GB10 SoC has 128 GB of unified CPU+GPU memory. DeepSeek-V4-Flash uses ~81 GB
of that pool, and large Ollama models can consume most of the remainder. Running
both simultaneously causes OOM failures. The router enforces mutual exclusion:

- To activate **ds4**: unload every loaded Ollama model (freeing their VRAM), then
  start the `ds4.service` user unit.
- To activate **Ollama**: stop the `ds4.service` user unit (it releases ~81 GB),
  then let Ollama load the requested model on demand.

Because ds4 runs under a `systemctl --user` unit with `Restart=always`, the
router controls it via `systemctl --user start/stop` — a plain SIGTERM would just
get respawned. The freed memory takes ~2-3 s for the kernel to reclaim, so the
router waits for it to settle before loading the incoming model.


## How a swap works (step by step)

1. A request arrives for model X. The router calls `acquire(model_id)`.
2. `acquire` resolves the owning engine via the static registry (or a live Ollama
   tag lookup for models pulled after router start).
3. If the target engine is already active, the request is counted in-flight and
   forwarded immediately — no swap needed.
4. If the target engine is **not** active, the router acquires `_swap_lock` and
   begins the swap:
   a. **Drain**: wait up to `drain_timeout_s` (default 30 s) for in-flight
      requests on the current engine to complete naturally.
   b. **Free VRAM**: call `free_vram()` on every non-target engine:
      - ds4: `systemctl --user stop ds4.service --no-block` (disables its
        `Restart=always`), then SIGTERM/SIGKILL any process and confirm the
        port is closed.
      - Ollama: POST `keep_alive: 0` to each loaded model; fall back to
        `ollama stop <name>` if that fails.
   c. **Settle**: poll `MemAvailable` until the freed model's memory has been
      reclaimed (`swap_memory_settle_timeout_s`), so the incoming model's
      pre-flight memory check doesn't fail.
   d. **Start target**: call `ensure_started()`:
      - ds4: `systemctl --user start ds4.service`, then poll `GET /v1/models`
        for up to `start_timeout_s` (default 240 s).
      - Ollama: if not answering, try `systemctl start ollama.service` and poll
        `GET /api/tags` for up to 20 s.
   e. Mark `active_engine`, persist `state.json`, release the swap lock.
5. While the swap is in progress, streaming clients receive SSE `": keepalive"`
   comment lines every `swap_keepalive_interval_s` (default 5 s) so they do not
   hit a first-token timeout.
6. The request is counted in-flight on the new engine and forwarded. On response
   completion (or error), `release()` decrements the counter.


## Endpoint reference

| Method | Path | Behaviour |
|--------|------|-----------|
| GET | `/` | Small HTML status page |
| GET | `/health` | `{"status":"ok"}` — router liveness; never triggers a swap |
| GET | `/status` | Full status dict: active engine, last swap, per-engine state, model list |
| GET | `/v1/models` | OpenAI model list: union of static registry + live Ollama tags, deduped. No swap; works even if engines are down. |
| POST | `/v1/chat/completions` | OpenAI chat; routed by `body.model` |
| POST | `/v1/completions` | OpenAI legacy completions; routed by `body.model` |
| POST | `/v1/embeddings` | OpenAI embeddings; routed by `body.model` |
| POST | `/v1/messages` | Anthropic messages format; routed by `body.model` |
| POST | `/v1/responses` | Responses API; routed by `body.model` |
| POST | `/api/chat` | Ollama-native chat; routed by `body.model` |
| POST | `/api/generate` | Ollama-native generate; routed by `body.model` |
| POST | `/api/embeddings` | Ollama-native embeddings; routed by `body.model` |
| POST | `/api/embed` | Ollama-native embed; routed by `body.model` |
| GET/POST | `/api/tags`, `/api/ps`, `/api/version`, `/api/show`, `/api/pull`, `/api/*` | Passthrough to Ollama, no swap (management/catalog) |
| POST | `/admin/swap` | Body: `{"model":"<id>"}` or `{"engine":"ds4"\|"ollama"}`. Proactive swap; returns `status()`. 400 if neither field. |


## Config reference (`config.yaml`)

Top-level keys:

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to expose off-localhost (pair with `api_keys` or a firewall). |
| `port` | `8077` | Listen port |
| `api_keys` | `[]` | If non-empty, require a key (`Authorization: Bearer` / `X-API-Key`) on all requests except `GET /health` |
| `log_level` | `INFO` | Python log level |
| `log_file` | `logs/router.log` | Rotating log (5 MB × 3 backups) |
| `state_file` | `state.json` | Persisted active-engine snapshot |
| `swap_keepalive_enabled` | `true` | Emit SSE keepalive during swaps |
| `swap_keepalive_interval_s` | `5.0` | Seconds between keepalive comments |
| `drain_timeout_s` | `30.0` | Max wait for in-flight requests before stopping an engine |
| `swap_memory_settle_timeout_s` | `25.0` | Max wait for freed memory to be reclaimed before starting the next engine (ends early once it plateaus) |
| `upstream_connect_timeout_s` | `15.0` | Connect timeout to backends (read is unbounded) |

`ds4` section:

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Whether ds4 is usable |
| `base_url` | `http://172.17.0.1:8099` | How the router reaches ds4 |
| `control` | `systemd-user` | How the router controls ds4: `systemd-user` (start/stop the user unit) or `process` (launch serve.sh + SIGTERM) |
| `systemd_user_unit` | `ds4.service` | The `systemctl --user` unit to start/stop (control=systemd-user) |
| `serve_script` | `/home/grahamfm/ds4/serve.sh` | Script to launch ds4-server (control=process fallback) |
| `process_pattern` | `ds4/ds4-server` | `pgrep -f` pattern used to confirm the process is gone after stop |
| `health_path` | `/v1/models` | Readiness probe path (returns 200 when ready) |
| `start_timeout_s` | `240.0` | Max wait for ds4 to become ready (~81 GB model) |
| `stop_timeout_s` | `45.0` | Max wait for the process to exit + port to close after stop |
| `log_file` | `logs/ds4-server.log` | Where ds4 stdout/stderr is appended (control=process) |

`ollama` section:

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Whether Ollama is usable |
| `base_url` | `http://127.0.0.1:11434` | How the router reaches Ollama |
| `health_path` | `/api/tags` | Readiness probe path |
| `unload_timeout_s` | `60.0` | Max wait for loaded models to unload |
| `systemd_unit` | `ollama.service` | Unit to start if Ollama is not answering |
| `tags_cache_ttl_s` | `30.0` | TTL for the live Ollama tag cache used in routing |

`models` list entries:

| Key | Required | Description |
|-----|----------|-------------|
| `id` | yes | Exact string sent in the `model` field by clients |
| `engine` | yes | `"ds4"` or `"ollama"` |
| `display_name` | no | Human-readable name (defaults to `id`) |
| `context_length` | no | Context window in tokens (default 131072) |

Models not listed in `config.yaml` are still routed correctly if they exist as
live Ollama tags (the router caches `/api/tags` for `tags_cache_ttl_s` seconds).


## Client wiring

- **OpenCode**: see [`deploy/opencode.snippet.md`](deploy/opencode.snippet.md)
- **Open WebUI**: see [`deploy/openwebui-wiring.md`](deploy/openwebui-wiring.md)


## routerctl

`routerctl` is a thin CLI for inspecting and controlling the running router.
Common commands:

```bash
# Show current status (active engine, in-flight, last swap)
./routerctl status

# Proactively swap to ds4 (shortcut, or: ./routerctl use ds4)
./routerctl ds4

# Proactively swap to ollama (shortcut, or: ./routerctl use ollama)
./routerctl ollama

# Swap to whichever engine owns a given model
./routerctl use qwen3.6-uncensored:27b
```


## Operations

### Install / enable as a systemd *user* service

The router runs as a **user** unit (not a system unit) so it shares the same
`systemctl --user` manager as `ds4.service` and can start/stop it. Lingering is
enabled so it starts at boot without a login session.

```bash
# Recommended: run the installer (no full root needed; only sudo for linger)
bash deploy/install.sh

# Equivalent manual steps:
mkdir -p ~/.config/systemd/user
cp deploy/llm-router.service ~/.config/systemd/user/
sudo loginctl enable-linger "$USER"      # boot-start without login (one-time)
systemctl --user daemon-reload
systemctl --user enable --now llm-router
```

### Start / stop / restart

```bash
systemctl --user start   llm-router
systemctl --user stop     llm-router
systemctl --user restart llm-router
systemctl --user status  llm-router
# or: routerctl start | stop | restart | status
```

Note: the router controls ds4 by starting/stopping the **`ds4.service`** user
unit (which has `Restart=always`). An explicit stop disables that auto-restart,
so ds4 stays down while Ollama holds the GPU; stopping the router itself does
not touch ds4.

### Logs

```bash
# Rotating file (5 MB × 3 backups)
tail -f /home/grahamfm/llm-router/logs/router.log

# systemd journal (user unit)
journalctl --user -u llm-router -f
journalctl --user -u llm-router --since "1 hour ago"

# ds4-server output goes to its own unit's journal:
journalctl --user -u ds4 -f
```

### State file

`/home/grahamfm/llm-router/state.json` is written after every swap and on
startup. It holds the last known active engine and last swap details. It is a
snapshot only — the router re-probes reality on startup rather than trusting it.

```json
{"active_engine": "ollama", "last_swap": {"from": "ds4", "to": "ollama", "duration_s": 52.3, "ok": true, "at": 1748000000}}
```


## Troubleshooting

### ds4 won't stop during a swap

The router issues `systemctl --user stop ds4.service --no-block` (so it doesn't
block on systemd's graceful timeout), then SIGTERMs the process and escalates to
SIGKILL after a short grace. If ds4 still won't stop:

```bash
systemctl --user stop ds4.service   # authoritative stop
pgrep -f ds4/ds4-server             # any leftover pid?
kill -9 <pid>                        # last resort
```

Then retry the swap via `routerctl ollama` or restart the router.

### Swap to Ollama fails with "more system memory than is available"

The outgoing ds4 model's ~81 GB takes a couple of seconds to be reclaimed by
the kernel; the router waits for memory to settle (`swap_memory_settle_timeout_s`)
before starting the next engine. If you still hit this, the model genuinely
doesn't fit — check `free -g` and the model size.

### Ollama won't unload a model

The router sends `keep_alive: 0` via the API, then falls back to `ollama stop
<name>`. If models remain loaded after `unload_timeout_s` (60 s), the router
logs a warning and proceeds anyway (the next acquire call may still fail if
there is not enough VRAM). To unload manually:

```bash
ollama list        # see what's loaded
ollama stop <name>
```

### Port 8077 busy

```bash
ss -tlnp | grep 8077
# If another process has it:
systemctl --user stop llm-router
# or kill the offending pid and restart
```

### Open WebUI model picker is empty

This is caused by the container being recreated without the
`--add-host=host.docker.internal:host-gateway` flag. Without that flag the
container cannot reach the host network and the model picker shows nothing.
Always preserve that flag on any `docker run` or `docker create` invocation.
See [`deploy/openwebui-wiring.md`](deploy/openwebui-wiring.md) for the full
safe recreate command.

### First request is slow (model cold-load)

The first request after a swap to ds4 starts the `ds4.service` unit and waits
up to 240 s for the ~81 GB model to load (typically ~12 s when warm). Subsequent
requests on the same engine are fast. The streaming keepalive ensures the client
does not time out while waiting.
