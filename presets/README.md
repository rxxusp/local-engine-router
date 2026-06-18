# Backend presets

Copy-paste config fragments for common inference backends. Each file is a
ready-to-edit `engines:` block (plus a matching `models:` entry) for one
backend, with `<ANGLE_BRACKET>` placeholders for the paths, ports, and model
ids you fill in, and inline notes on the gotchas specific to that engine.

These exist so you do not have to reverse-engineer the engine schema to wire up
a new backend. Read [`../config.example.yaml`](../config.example.yaml) and
[`../config.schema.json`](../config.schema.json) for the full field reference.

## How to use a preset

1. Open the preset for your backend and copy the `engines:` block into your
   `config.yaml` under the top-level `engines:` map (merge it with any engines
   you already have).
2. Copy the matching `models:` entry into your top-level `models:` list. The
   `id` is what clients send in the `model` field, and it is what the router
   uses to route a request to this engine.
3. Replace every `<ANGLE_BRACKET>` placeholder (binary paths, model files,
   ports, admin keys) with your real values.
4. Start the router and send a request with the model `id` you chose. The
   router brings the engine up on first use and swaps the GPU to it.

Validate your edits against the schema with
[`validate_presets.py`](validate_presets.py), or just start the router (it
validates the whole config at load time and reports actionable errors).

## Engine types

The presets use the three lifecycle models the router supports:

- **`generic_process`** the router launches the server as a child process and
  stops it (SIGTERM, escalating to SIGKILL) when it swaps away. Use this when
  the engine is a plain server binary you want the router to own.
- **`api_swap`** you run the server yourself; the router frees and reloads VRAM
  by calling the engine's explicit HTTP load/unload endpoints. Use this for
  engines that own their own process lifecycle but expose load/unload control.
- **`ollama`** a preset of `api_swap` tuned for Ollama's JIT model loading and
  its `/api/ps` and `/api/tags` endpoints.

## Included backends

| Preset | Engine type | Notes |
|---|---|---|
| [`llamacpp.yaml`](llamacpp.yaml) | `generic_process` | `llama-server`; SIGTERM-freeze handled by SIGKILL escalation |
| [`vllm.yaml`](vllm.yaml) | `generic_process` | OpenAI server; spawns worker children, reaped by process group |
| [`sglang.yaml`](sglang.yaml) | `generic_process` | `sglang.launch_server` |
| [`koboldcpp.yaml`](koboldcpp.yaml) | `generic_process` | single-binary GGUF server |
| [`mlx.yaml`](mlx.yaml) | `generic_process` | `mlx_lm.server`, Apple Silicon |
| [`max.yaml`](max.yaml) | `generic_process` | Modular MAX serve |
| [`ramalama.yaml`](ramalama.yaml) | `generic_process` | OCI-packaged models |
| [`localai.yaml`](localai.yaml) | `generic_process` | LocalAI server |
| [`tabbyapi.yaml`](tabbyapi.yaml) | `api_swap` | ExLlamaV2; explicit load/unload, needs `x-admin-key` |
| [`lmstudio.yaml`](lmstudio.yaml) | `api_swap` | LM Studio local server |
| [`ollama.yaml`](ollama.yaml) | `ollama` | JIT loading; no explicit load path |

If your backend is not listed, the closest preset of the same engine type is
usually a one or two field edit away.
