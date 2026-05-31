# Open WebUI wiring: route through llm-router

The router is reachable from inside the `open-webui` container at
`http://host.docker.internal:8077` (the bridge gateway is `172.17.0.1`, mapped
to `host.docker.internal` via `--add-host=host.docker.internal:host-gateway`).

---

## Recommended path: Admin Panel (zero risk, no container recreate)

This method requires no Docker commands and carries no risk of losing data or
breaking the container.

1. Open Open WebUI at `http://localhost:8080`.
2. Go to **Admin Panel** > **Settings** > **Connections**.
3. Under **OpenAI API**, click **+** to add a new connection:
   - **URL**: `http://host.docker.internal:8077/v1`
   - **API Key**: leave empty or enter any non-empty string (the router ignores it)
   - Save.
4. **Disable** the direct Ollama connection (toggle it off or delete it).
   This is important: if the direct Ollama connection remains enabled, models
   will appear twice in the picker (once from Ollama, once from the router), and
   direct requests to Ollama bypass the router — preventing automatic engine swaps.
5. Verify: open the model picker and confirm you see the router's model list
   (deepseek-v4-flash, qwen3.6-uncensored:27b, etc.).

**RAG / embeddings are unaffected.** Embeddings run inside the container via
`sentence-transformers` (the `USE_EMBEDDING_MODEL_DOCKER` env var) and do not
call any external API.

---

## Optional / advanced: docker run recreate

> **Warning**: recreating the container is the riskier path. The known gotcha is
> that dropping `--add-host=host.docker.internal:host-gateway` from the `docker
> run` command causes the model picker to be empty because the container can no
> longer resolve `host.docker.internal`. Always include it.
>
> The named volume `open-webui` persists all user data (conversations, settings,
> uploaded files) across a recreate. Do not use `--rm` or a different volume
> name.

This command was derived from `docker inspect open-webui` and faithfully
reproduces the current container, adding only
`OPENAI_API_BASE_URL=http://host.docker.internal:8077/v1`:

```bash
docker stop open-webui
docker rm open-webui

docker run -d \
  --name open-webui \
  --restart unless-stopped \
  --network bridge \
  --add-host=host.docker.internal:host-gateway \
  -p 8080:8080 \
  -v open-webui:/app/backend/data \
  -v /home/grahamfm/models:/models \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8077/v1 \
  -e OPENAI_API_KEY="" \
  -e WEBUI_SECRET_KEY="" \
  -e SCARF_NO_ANALYTICS=true \
  -e DO_NOT_TRACK=true \
  -e ANONYMIZED_TELEMETRY=false \
  -e PYTHONUNBUFFERED=1 \
  -e ENV=prod \
  -e PORT=8080 \
  -e USE_OLLAMA_DOCKER=false \
  -e USE_CUDA_DOCKER=false \
  -e USE_SLIM_DOCKER=false \
  -e USE_CUDA_DOCKER_VER=cu128 \
  -e USE_EMBEDDING_MODEL_DOCKER=sentence-transformers/all-MiniLM-L6-v2 \
  -e USE_RERANKING_MODEL_DOCKER="" \
  -e USE_AUXILIARY_EMBEDDING_MODEL_DOCKER=TaylorAI/bge-micro-v2 \
  -e RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  -e RAG_RERANKING_MODEL="" \
  -e AUXILIARY_EMBEDDING_MODEL=TaylorAI/bge-micro-v2 \
  -e SENTENCE_TRANSFORMERS_HOME=/app/backend/data/cache/embedding/models \
  -e TIKTOKEN_ENCODING_NAME=cl100k_base \
  -e TIKTOKEN_CACHE_DIR=/app/backend/data/cache/tiktoken \
  -e HF_HOME=/app/backend/data/cache/embedding/models \
  -e WHISPER_MODEL=base \
  -e WHISPER_MODEL_DIR=/app/backend/data/cache/whisper/models \
  ghcr.io/open-webui/open-webui:main
```

After the container starts, go to **Admin Panel** > **Settings** > **Connections**
and disable the direct Ollama connection so all requests flow through the router.

### What this adds / changes vs the current container

| | Current | After recreate |
|-|---------|---------------|
| `OPENAI_API_BASE_URL` | _(empty)_ | `http://host.docker.internal:8077/v1` |
| Everything else | unchanged | unchanged |

### Checklist before recreating

- [ ] The router is running: `curl http://127.0.0.1:8077/health`
- [ ] `--add-host=host.docker.internal:host-gateway` is in the command above (it is)
- [ ] The named volume is `open-webui` (not a path bind) — user data is safe
- [ ] No active Open WebUI sessions you don't want to interrupt
