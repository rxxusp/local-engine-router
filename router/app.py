"""FastAPI application for llm-router.

Implements the full shared endpoint contract:

  GET  /                          tiny HTML status page
  GET  /health                    router liveness (no swap)
  GET  /status                    EngineManager.status()
  GET  /v1/models                 union of static config + live ollama tags
  POST /v1/chat/completions       }
  POST /v1/completions            } OpenAI-compatible; route by body["model"]
  POST /v1/embeddings             }
  POST /v1/messages               } Anthropic-compat forwarded to ds4
  POST /v1/responses              }
  POST /api/chat                  } Ollama-native; route by body["model"]
  POST /api/generate              }
  POST /api/embeddings            }
  POST /api/embed                 }
  GET|POST /api/tags, /api/ps,    } Ollama management passthrough (no swap)
            /api/version,         }
            /api/show, /api/pull  }
  POST /admin/swap                proactive engine swap

SSE keep-alive comments are injected for /v1/* streaming endpoints only.
/api/* streams use application/x-ndjson and MUST NOT get SSE comment lines.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import RouterConfig
from .engines import EngineError, EngineManager, OllamaEngine
from .proxy import (
    filter_request_headers,
    forward,
    make_client,
    upstream_url,
)

log = logging.getLogger("router.app")

# Fixed "created" timestamp used in /v1/models responses (2026-01-01 00:00:00 UTC).
_MODELS_CREATED_TS = 1767225600


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _openai_error(message: str, error_type: str, code: str | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"message": message, "type": error_type}
    if code is not None:
        err["code"] = code
    return {"error": err}


def sse_error_chunk(exc: Exception) -> bytes:
    """Format an engine/HTTP error as an SSE data line."""
    payload = json.dumps({"error": {"message": str(exc), "type": "engine_error"}})
    return b"data: " + payload.encode() + b"\n\n"


# Paths reachable without an API key even when auth is enabled (liveness probes).
_AUTH_EXEMPT_PATHS = frozenset({"/health"})


def _extract_api_key(headers) -> str | None:
    """Pull the API key from Authorization: Bearer <key> or X-API-Key."""
    auth = headers.get("authorization")  # Starlette Headers are case-insensitive
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key")


def _api_key_valid(provided: str, keys: list[str]) -> bool:
    """Constant-time membership check against the configured keys."""
    return any(hmac.compare_digest(provided, k) for k in keys)


def _error_status_for(exc: Exception) -> int:
    """Map EngineError / httpx errors to HTTP status codes."""
    if isinstance(exc, EngineError):
        msg = str(exc).lower()
        if "no engine can serve" in msg or "unknown" in msg or "disabled" in msg:
            return 404
        # Missing model field is caught before calling acquire; anything else
        # from an engine that wouldn't come up → 503.
        return 503
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        return 502
    return 502


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cfg: RouterConfig) -> FastAPI:
    """Build and return the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = EngineManager(cfg)
        await manager.startup()
        client = make_client(cfg)
        app.state.manager = manager
        app.state.client = client
        log.info("router: started on %s:%d", cfg.host, cfg.port)
        if cfg.api_keys:
            log.info("auth: API-key authentication ENABLED (%d key(s))", len(cfg.api_keys))
        elif cfg.host not in ("127.0.0.1", "localhost", "::1"):
            log.warning(
                "SECURITY: bound to %s with NO api_keys set — the router is "
                "reachable off-localhost with no authentication. Set 'api_keys' "
                "in config, or bind 127.0.0.1, or ensure a host firewall restricts access.",
                cfg.host,
            )
        try:
            yield
        finally:
            await client.aclose()
            await manager.aclose()
            log.info("router: shutdown complete")

    app = FastAPI(title="llm-router", lifespan=lifespan)

    # -----------------------------------------------------------------------
    # API-key auth (only installed when api_keys are configured)
    # -----------------------------------------------------------------------
    if cfg.api_keys:
        keys = list(cfg.api_keys)

        @app.middleware("http")
        async def _auth_middleware(request: Request, call_next):
            if (
                request.method != "OPTIONS"
                and request.url.path not in _AUTH_EXEMPT_PATHS
            ):
                provided = _extract_api_key(request.headers)
                if not provided or not _api_key_valid(provided, keys):
                    return JSONResponse(
                        _openai_error(
                            "missing or invalid API key",
                            "authentication_error",
                            "invalid_api_key",
                        ),
                        status_code=401,
                    )
            return await call_next(request)

    # -----------------------------------------------------------------------
    # Liveness / status
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Router liveness probe. Never triggers a swap."""
        return {"status": "ok"}

    @app.get("/status")
    async def status(request: Request) -> JSONResponse:
        manager: EngineManager = request.app.state.manager
        return JSONResponse(await manager.status())

    # -----------------------------------------------------------------------
    # HTML home page
    # -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        manager: EngineManager = request.app.state.manager
        st = await manager.status()
        active = st.get("active_engine") or "none"
        models_html = "".join(
            f"<li><code>{m['id']}</code> → {m['engine']} <em>{m['name']}</em></li>"
            for m in st.get("models", [])
        )
        html = f"""<!DOCTYPE html>
<html>
<head><title>llm-router</title></head>
<body>
<h1>llm-router</h1>
<p><strong>Active engine:</strong> {active}</p>
<h2>Models</h2>
<ul>{models_html}</ul>
<p>
  <a href="/status">JSON status</a> &nbsp;|&nbsp;
  <a href="/v1/models">OpenAI model list</a>
</p>
</body>
</html>"""
        return HTMLResponse(html)

    # -----------------------------------------------------------------------
    # /v1/models — union of static config + live ollama tags (no swap)
    # -----------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        manager: EngineManager = request.app.state.manager
        seen: set[str] = set()
        data: list[dict[str, Any]] = []

        # Static registry first.
        for spec in cfg.models:
            seen.add(spec.id)
            data.append(
                {
                    "id": spec.id,
                    "object": "model",
                    "created": _MODELS_CREATED_TS,
                    "owned_by": spec.engine,
                    "name": spec.display_name,
                    "context_length": spec.context_length,
                }
            )

        # Live Ollama tags (best-effort; don't fail if ollama is down).
        ollama_engine = manager.engines.get("ollama")
        if isinstance(ollama_engine, OllamaEngine):
            try:
                tags = await ollama_engine.available_tags()
                for tag in sorted(tags):
                    if tag not in seen:
                        seen.add(tag)
                        data.append(
                            {
                                "id": tag,
                                "object": "model",
                                "created": _MODELS_CREATED_TS,
                                "owned_by": "ollama",
                                "name": tag,
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                log.warning("could not fetch ollama tags for /v1/models: %s", exc)

        return JSONResponse({"object": "list", "data": data})

    # -----------------------------------------------------------------------
    # Admin: force swap
    # -----------------------------------------------------------------------

    @app.post("/admin/swap")
    async def admin_swap(request: Request) -> JSONResponse:
        manager: EngineManager = request.app.state.manager
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                _openai_error("request body must be JSON", "invalid_request_error"),
                status_code=400,
            )

        model_id: str | None = body.get("model")
        engine_key: str | None = body.get("engine")

        if not model_id and not engine_key:
            return JSONResponse(
                _openai_error(
                    "body must contain 'model' or 'engine'", "invalid_request_error"
                ),
                status_code=400,
            )

        try:
            await manager.force_swap(model_id=model_id, engine_key=engine_key)
        except EngineError as exc:
            return JSONResponse(
                _openai_error(str(exc), "engine_error"),
                status_code=_error_status_for(exc),
            )

        return JSONResponse(await manager.status())

    # -----------------------------------------------------------------------
    # Internal helpers used by all proxied routes
    # -----------------------------------------------------------------------

    def _build_fwd_headers(request: Request) -> dict[str, str]:
        """Strip hop-by-hop from the incoming request headers."""
        return filter_request_headers(dict(request.headers))

    # -----------------------------------------------------------------------
    # OpenAI-style /v1 endpoints (chat completions, completions, embeddings,
    # messages, responses)
    # -----------------------------------------------------------------------

    async def _handle_v1_post(request: Request, path: str) -> Response:
        """Common handler for all POST /v1/* endpoints."""
        manager: EngineManager = request.app.state.manager
        client: httpx.AsyncClient = request.app.state.client

        raw_body = await request.body()
        try:
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            return JSONResponse(
                _openai_error("request body is not valid JSON", "invalid_request_error"),
                status_code=400,
            )

        model: str | None = body.get("model")
        if not model:
            return JSONResponse(
                _openai_error("missing required field: 'model'", "invalid_request_error"),
                status_code=400,
            )

        is_stream: bool = bool(body.get("stream", False))
        fwd_headers = _build_fwd_headers(request)

        if is_stream:
            # SSE streaming with keep-alive comment injection during swaps.
            async def gen() -> "AsyncGenerator[bytes, None]":  # type: ignore[name-defined]
                acq = asyncio.create_task(manager.acquire(model))
                engine = None
                try:
                    # Wait for the engine to be acquired (a swap may be in
                    # progress), emitting SSE keep-alive comments so the client
                    # doesn't hit a TTFB/idle timeout. acq is shielded, so a
                    # keepalive timeout never cancels the in-progress swap.
                    try:
                        while not acq.done():
                            try:
                                await asyncio.wait_for(
                                    asyncio.shield(acq),
                                    timeout=cfg.swap_keepalive_interval_s,
                                )
                            except asyncio.TimeoutError:
                                if cfg.swap_keepalive_enabled:
                                    log.debug(
                                        "keepalive: waiting for swap (model=%s)", model
                                    )
                                    yield b": keepalive (swapping engines)\n\n"
                        engine = acq.result()  # raises EngineError on failure
                    except EngineError as exc:
                        log.error("acquire failed for %s: %s", model, exc)
                        yield sse_error_chunk(exc)
                        yield b"data: [DONE]\n\n"
                        return

                    log.info("stream %s -> %s", model, engine.key)
                    url = upstream_url(engine.base_url, path)
                    try:
                        async with client.stream(
                            request.method, url, content=raw_body, headers=fwd_headers
                        ) as up:
                            async for chunk in up.aiter_raw():
                                yield chunk
                    except httpx.HTTPError as exc:
                        log.error("upstream stream error on %s: %s", url, exc)
                        yield sse_error_chunk(exc)
                        yield b"data: [DONE]\n\n"
                finally:
                    # Guarantee the in-flight count is released however the
                    # generator exits: normal end, EngineError, upstream error,
                    # or client disconnect (CancelledError / GeneratorExit) at
                    # any point — including mid-swap while emitting keepalives.
                    if not acq.done():
                        # Still pending => acquire hasn't incremented in-flight
                        # yet (the increment is the last, await-free step), so
                        # cancelling here is leak-free.
                        acq.cancel()
                    elif engine is None and not acq.cancelled():
                        # acquire completed (and incremented) but we were
                        # cancelled before binding `engine`. Recover it so the
                        # increment is paired with a release.
                        try:
                            engine = acq.result()
                        except BaseException:
                            engine = None
                    if engine is not None:
                        # shield so a cancellation in flight can't skip release.
                        await asyncio.shield(manager.release(engine.key))

            return StreamingResponse(gen(), media_type="text/event-stream")
        else:
            # Non-streaming: acquire -> proxy -> release.
            engine = None
            try:
                engine = await manager.acquire(model)
            except EngineError as exc:
                status = _error_status_for(exc)
                return JSONResponse(_openai_error(str(exc), "engine_error"), status_code=status)

            log.info("request %s -> %s", model, engine.key)
            url = upstream_url(engine.base_url, path)
            try:
                status, resp_headers, body_bytes = await forward(
                    client, request.method, url, fwd_headers, raw_body
                )
            except httpx.HTTPError as exc:
                log.error("upstream error on %s: %s", url, exc)
                return JSONResponse(
                    _openai_error(str(exc), "upstream_error"), status_code=502
                )
            finally:
                await manager.release(engine.key)

            return Response(content=body_bytes, status_code=status, headers=resp_headers)

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request) -> Response:
        return await _handle_v1_post(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def v1_completions(request: Request) -> Response:
        return await _handle_v1_post(request, "/v1/completions")

    @app.post("/v1/embeddings")
    async def v1_embeddings(request: Request) -> Response:
        return await _handle_v1_post(request, "/v1/embeddings")

    @app.post("/v1/messages")
    async def v1_messages(request: Request) -> Response:
        return await _handle_v1_post(request, "/v1/messages")

    @app.post("/v1/responses")
    async def v1_responses(request: Request) -> Response:
        return await _handle_v1_post(request, "/v1/responses")

    # -----------------------------------------------------------------------
    # Ollama-native /api endpoints that carry a model and trigger a swap
    # -----------------------------------------------------------------------

    async def _handle_api_post_with_model(request: Request, path: str) -> Response:
        """Handle Ollama-native model-bearing POSTs (/api/chat, /api/generate,
        /api/embeddings, /api/embed).

        Ollama streams by default; treat stream=True unless body has stream==False.
        NDJSON streams must NOT get SSE comment lines.
        """
        manager: EngineManager = request.app.state.manager
        client: httpx.AsyncClient = request.app.state.client

        raw_body = await request.body()
        try:
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            return JSONResponse(
                _openai_error("request body is not valid JSON", "invalid_request_error"),
                status_code=400,
            )

        model: str | None = body.get("model")
        if not model:
            return JSONResponse(
                _openai_error("missing required field: 'model'", "invalid_request_error"),
                status_code=400,
            )

        # Ollama streams by default; only non-stream if explicitly false.
        is_stream: bool = body.get("stream", True) is not False

        fwd_headers = _build_fwd_headers(request)

        engine = None
        try:
            engine = await manager.acquire(model)
        except EngineError as exc:
            status = _error_status_for(exc)
            return JSONResponse(_openai_error(str(exc), "engine_error"), status_code=status)

        log.info("api request %s -> %s (stream=%s)", model, engine.key, is_stream)
        url = upstream_url(engine.base_url, path)

        if is_stream:
            # NDJSON stream — no SSE comment injection.
            async def ndjson_gen() -> "AsyncGenerator[bytes, None]":  # type: ignore[name-defined]
                try:
                    async with client.stream(
                        request.method, url, content=raw_body, headers=fwd_headers
                    ) as up:
                        async for chunk in up.aiter_raw():
                            yield chunk
                except httpx.HTTPError as exc:
                    log.error("upstream stream error on %s: %s", url, exc)
                    # Can't inject SSE; just end the stream.
                finally:
                    await manager.release(engine.key)

            return StreamingResponse(ndjson_gen(), media_type="application/x-ndjson")
        else:
            try:
                status, resp_headers, body_bytes = await forward(
                    client, request.method, url, fwd_headers, raw_body
                )
            except httpx.HTTPError as exc:
                log.error("upstream error on %s: %s", url, exc)
                return JSONResponse(
                    _openai_error(str(exc), "upstream_error"), status_code=502
                )
            finally:
                await manager.release(engine.key)

            return Response(content=body_bytes, status_code=status, headers=resp_headers)

    @app.post("/api/chat")
    async def api_chat(request: Request) -> Response:
        return await _handle_api_post_with_model(request, "/api/chat")

    @app.post("/api/generate")
    async def api_generate(request: Request) -> Response:
        return await _handle_api_post_with_model(request, "/api/generate")

    @app.post("/api/embeddings")
    async def api_embeddings(request: Request) -> Response:
        return await _handle_api_post_with_model(request, "/api/embeddings")

    @app.post("/api/embed")
    async def api_embed(request: Request) -> Response:
        return await _handle_api_post_with_model(request, "/api/embed")

    # -----------------------------------------------------------------------
    # Ollama management/catalog passthrough (no swap, no model routing)
    # These forward directly to ollama, whatever its state.
    # -----------------------------------------------------------------------

    async def _passthrough_to_ollama(request: Request, path: str) -> Response:
        """Forward a request directly to ollama without any swap logic."""
        manager: EngineManager = request.app.state.manager
        client: httpx.AsyncClient = request.app.state.client

        ollama_engine = manager.engines.get("ollama")
        if ollama_engine is None:
            return JSONResponse(
                _openai_error("ollama engine is disabled", "engine_error"),
                status_code=503,
            )

        raw_body = await request.body()
        fwd_headers = _build_fwd_headers(request)
        url = upstream_url(ollama_engine.base_url, path)

        try:
            status, resp_headers, body_bytes = await forward(
                client, request.method, url, fwd_headers, raw_body
            )
        except httpx.HTTPError as exc:
            log.warning("ollama passthrough error on %s: %s", url, exc)
            return JSONResponse(
                _openai_error(str(exc), "upstream_error"), status_code=502
            )
        return Response(content=body_bytes, status_code=status, headers=resp_headers)

    @app.get("/api/tags")
    @app.post("/api/tags")
    async def api_tags(request: Request) -> Response:
        return await _passthrough_to_ollama(request, "/api/tags")

    @app.get("/api/ps")
    @app.post("/api/ps")
    async def api_ps(request: Request) -> Response:
        return await _passthrough_to_ollama(request, "/api/ps")

    @app.get("/api/version")
    @app.post("/api/version")
    async def api_version(request: Request) -> Response:
        return await _passthrough_to_ollama(request, "/api/version")

    @app.post("/api/show")
    @app.get("/api/show")
    async def api_show(request: Request) -> Response:
        return await _passthrough_to_ollama(request, "/api/show")

    @app.post("/api/pull")
    async def api_pull(request: Request) -> Response:
        return await _passthrough_to_ollama(request, "/api/pull")

    # Catch-all for any other /api/* paths not handled above — passthrough
    # to ollama without swap (management endpoints).
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def api_catchall(request: Request, path: str) -> Response:
        return await _passthrough_to_ollama(request, f"/api/{path}")

    return app
