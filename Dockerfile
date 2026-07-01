# syntax=docker/dockerfile:1
#
# local-engine-router — container image for the proxy.
#
# Base is python:3.12-slim, NOT a CUDA image: the router is PURE PYTHON and
# needs NO GPU. The heavy LLM engines (ds4, Ollama, vLLM, ...) run on the host;
# the router only reads the request's `model`, picks an engine, and proxies the
# bytes through. A pinned-CUDA base (as an older roadmap line suggested) would
# bloat this image ~8x for zero benefit, so we deliberately avoid it.
#
# IMPORTANT — process control does NOT work inside this container:
#   The `systemd-user` and `process` (pgrep/serve_script) engine controls reach
#   into the host's process/service tree, which a container cannot see. This
#   image is appropriate for fronting `api_swap` / remote engines (engines the
#   router only talks to over HTTP and never starts/stops itself). For
#   systemd-user ds4 control, run the router on the host instead (see deploy/).

# --- build stage: produce a wheel ------------------------------------------
FROM python:3.12-slim AS build

WORKDIR /src
RUN pip install --no-cache-dir build

# Copy only what's needed to build the wheel (the rest is .dockerignore'd).
COPY pyproject.toml README.md LICENSE ./
COPY router ./router

RUN python -m build --wheel --outdir /dist

# --- runtime stage ----------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="local-engine-router" \
      org.opencontainers.image.description="OpenAI/Ollama-compatible swap proxy for local LLM engines (pure-Python, no GPU)." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/rxxusp/local-engine-router"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ROUTER_CONFIG=/app/config.yaml \
    ROUTER_PORT=8077

WORKDIR /app

# Install the wheel built above (pulls fastapi/uvicorn/httpx/pyyaml).
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Ship the example config so a fresh container has something to copy/mount.
# Bind-mount or COPY your real config to /app/config.yaml at run time.
COPY config.example.yaml /app/config.example.yaml

# Run as a non-root user.
RUN useradd --create-home --uid 10001 router
USER router

EXPOSE 8077

# Liveness probe using Python's built-in urllib so no extra tool (curl/wget) is
# needed. /health returns {"status":"ok"} and never triggers a swap. The port is
# read from the mounted config (the same place the app gets it), falling back to
# $ROUTER_PORT, so it stays honest if you run on a non-default port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c \
        "import os, urllib.request, sys, yaml; p=os.environ.get('ROUTER_PORT','8077'); \
        cfg=os.environ.get('ROUTER_CONFIG','/app/config.yaml'); \
        p=(yaml.safe_load(open(cfg)) or {}).get('port', p) if os.path.exists(cfg) else p; \
        r=urllib.request.urlopen(f'http://localhost:{p}/health',timeout=4); sys.exit(0 if r.status==200 else 1)"

# `python -m router` reads $ROUTER_CONFIG (defaults to /app/config.yaml).
# Provide one, e.g.:
#   docker run -p 8077:8077 -v $PWD/config.yaml:/app/config.yaml ghcr.io/rxxusp/local-engine-router
CMD ["python", "-m", "router"]
