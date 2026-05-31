# OpenCode wiring: route both providers through llm-router

After the router is running on `:8077`, update `~/.config/opencode/opencode.json`
so both provider `options.baseURL` values point at the router instead of the
backends directly. **Nothing else changes** — the model ids, per-model limits,
`tools`/`reasoning` flags, and the top-level `"model"` default all stay as-is.
The router reads the `model` field from each request and dispatches to ds4 or
Ollama automatically.

## Before / After

### BEFORE (direct to backends)

```json
{
  "provider": {
    "ds4": {
      "name": "DeepSeek (local ds4)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://172.17.0.1:8099/v1"
      },
      ...
    },
    "ollama": {
      "name": "Ollama",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:11434/v1"
      },
      ...
    }
  }
}
```

### AFTER (both through the router)

```json
{
  "provider": {
    "ds4": {
      "name": "DeepSeek (local ds4)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8077/v1"
      },
      ...
    },
    "ollama": {
      "name": "Ollama",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8077/v1"
      },
      ...
    }
  }
}
```

Only two values change:
- `provider.ds4.options.baseURL`: `http://172.17.0.1:8099/v1` → `http://127.0.0.1:8077/v1`
- `provider.ollama.options.baseURL`: `http://127.0.0.1:11434/v1` → `http://127.0.0.1:8077/v1`

## One-liner to apply it

Using `python3` (no `jq` required):

```bash
python3 - <<'EOF'
import json, pathlib

cfg_path = pathlib.Path.home() / ".config/opencode/opencode.json"
cfg = json.loads(cfg_path.read_text())

router_url = "http://127.0.0.1:8077/v1"
for key in ("ds4", "ollama"):
    cfg.setdefault("provider", {}).setdefault(key, {}).setdefault("options", {})["baseURL"] = router_url

cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
print("Done. Restart OpenCode for the change to take effect.")
EOF
```

Or with `jq`:

```bash
jq '
  .provider.ds4.options.baseURL = "http://127.0.0.1:8077/v1" |
  .provider.ollama.options.baseURL = "http://127.0.0.1:8077/v1"
' ~/.config/opencode/opencode.json > /tmp/opencode.json.tmp \
  && mv /tmp/opencode.json.tmp ~/.config/opencode/opencode.json
echo "Done. Restart OpenCode for the change to take effect."
```

After applying, restart OpenCode (or reload its config) so it picks up the new
`baseURL` values.
