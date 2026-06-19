"""FastAPI application for local-engine-router.

Implements the full shared endpoint contract:

  GET  /                          tiny HTML status page
  GET  /health                    router liveness (no swap)
  GET  /metrics                   Prometheus exposition (unauthenticated)
  GET  /status                    EngineManager.status()
  GET  /v1/models                 union of static config + live ollama tags
  POST /v1/chat/completions       }
  POST /v1/completions            } OpenAI-compatible; route by body["model"]
  POST /v1/embeddings             }
  POST /v1/messages               } Anthropic-compat; route by body["model"]
  POST /v1/responses              }
  POST /api/chat                  } Ollama-native; route by body["model"]
  POST /api/generate              }
  POST /api/embeddings            }
  POST /api/embed                 }
  GET|POST /api/tags, /api/ps,    } Ollama management passthrough (no swap)
            /api/version,         }
            /api/show, /api/pull  }
  POST /admin/swap                proactive engine swap

During an engine swap, streaming responses emit periodic keep-alive frames so
clients don't hit a TTFB/idle timeout: /v1/* streams (text/event-stream) get SSE
comment lines (": ...\\n\\n"), while /api/* streams (application/x-ndjson) get a
bare newline ("\\n") that NDJSON readers skip — /api/* MUST NOT get SSE comment
lines as they would corrupt the NDJSON stream. Non-streaming requests cannot
carry keep-alive frames, so their callers must raise the client read-timeout
above the worst-case swap.
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

from . import metrics
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


# Paths reachable without an API key even when auth is enabled: liveness probes
# and the Prometheus scrape endpoint (scrapers must reach /metrics keyless).
_AUTH_EXEMPT_PATHS = frozenset({"/health", "/metrics"})

# First path segment of /api/* endpoints that mutate the engine's model store.
# Refused by the catch-all passthrough unless cfg.allow_destructive_ollama_api.
_DESTRUCTIVE_API_SEGMENTS = frozenset({"delete", "create", "copy", "push", "blobs"})


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


def _resolve_alias_and_rewrite(
    manager: EngineManager, model: str, body: dict[str, Any], raw_body: bytes
) -> tuple[str, bytes]:
    """Resolve *model* through the alias map; if it is an alias, rewrite the
    request body's "model" to the REAL id and re-serialize.

    Returns ``(real_model_id, raw_body)``. When *model* is not an alias the
    inputs are returned unchanged (no re-serialization), so non-aliased
    requests are byte-for-byte identical to before. Routing always uses the
    real id, and the upstream engine must see the real id in the body — never
    the alias."""
    real = manager.resolve_model_id(model)
    if real == model:
        return model, raw_body
    log.debug("alias %s -> %s; rewriting body model", model, real)
    body["model"] = real
    return real, json.dumps(body).encode()


def _apply_thinking_policy(
    manager: EngineManager, model: str, path: str, body: dict[str, Any], raw_body: bytes
) -> bytes:
    """Disable the reasoning/thinking channel on small-budget chat-completion
    requests for models configured with ``disable_thinking_below_max_tokens``.

    Reasoning tokens count against ``max_tokens``; with thinking on a small
    budget can be wholly consumed by the thought channel, leaving ``content``
    empty (finish_reason=length). vLLM honors a per-request
    ``chat_template_kwargs.enable_thinking`` that overrides the server's
    ``--default-chat-template-kwargs``, so for sub-threshold budgets we inject
    ``enable_thinking: false`` — unless the client set it explicitly. A
    generous or unset budget is left untouched (thinking on = the quality path),
    and the request bytes are returned unchanged unless the policy actually
    fires."""
    spec = manager.index.get(model)
    threshold = getattr(spec, "disable_thinking_below_max_tokens", None) if spec else None
    if not threshold:
        return raw_body
    # chat_template_kwargs / enable_thinking only apply to chat completions.
    # /v1/messages (Anthropic) and /v1/responses don't carry this knob, so the
    # guarantee deliberately does not extend there — leave them untouched.
    if not path.rstrip("/").endswith("chat/completions"):
        return raw_body
    budget = body.get("max_completion_tokens")
    if budget is None:
        budget = body.get("max_tokens")
    # bool is an int subclass — reject it explicitly. Accept int OR float, since
    # JSON/JS clients may send the budget as 500.0. A non-number means no budget
    # was set (full context available) → leave thinking on, as does a generous
    # budget at/above the threshold.
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        return raw_body
    if budget >= threshold:
        return raw_body
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        return raw_body  # client made an explicit choice → respect it
    ctk = dict(ctk) if isinstance(ctk, dict) else {}
    ctk["enable_thinking"] = False
    body["chat_template_kwargs"] = ctk
    log.info(
        "thinking policy: %s max_tokens=%s < %s -> enable_thinking=false",
        model, budget, threshold,
    )
    return json.dumps(body).encode()


def _find_ollama_engine(manager: EngineManager) -> OllamaEngine | None:
    """Return the first OllamaEngine in the manager's table, or None.

    The Ollama-capable engine used to be looked up by the literal key
    "ollama", but with the generic ``engines:`` table a user can key it under
    any name. Resolve it by type instead so /v1/models tag enrichment and the
    /api/* passthrough keep working regardless of the configured key. If more
    than one OllamaEngine is configured, the first (by insertion order) is used
    — the /api/* passthrough and tag enrichment bind to that one.
    """
    for engine in manager.engines.values():
        if isinstance(engine, OllamaEngine):
            return engine
    return None


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

    app = FastAPI(title="local-engine-router", lifespan=lifespan)

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

    @app.get("/metrics")
    async def get_metrics() -> Response:
        """Prometheus exposition. Unauthenticated (exempt from the api-key
        middleware) so scrapers can reach it without a key."""
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)

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
<head><title>local-engine-router</title></head>
<body>
<h1>local-engine-router</h1>
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
    # /v1/models — union of static config + live tags from all engines (no swap)
    # -----------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        manager: EngineManager = request.app.state.manager
        seen: set[str] = set()
        data: list[dict[str, Any]] = []

        # Static registry first (with context_length, which only the config knows).
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

        # Live tags from every engine — best-effort, one try/except per engine
        # so a single slow or broken engine cannot fail the whole listing.
        for engine_key, engine in manager.engines.items():
            try:
                tags = await engine.available_models()
            except Exception as exc:  # noqa: BLE001
                log.warning("could not fetch models from engine %r for /v1/models: %s",
                            engine_key, exc)
                continue
            for tag in sorted(tags):
                if tag not in seen:
                    seen.add(tag)
                    data.append(
                        {
                            "id": tag,
                            "object": "model",
                            "created": _MODELS_CREATED_TS,
                            "owned_by": engine_key,
                            "name": tag,
                        }
                    )

        # Surface any stopped-engine discovered ids when the routing slice has
        # populated a _discovered_index on the manager (hasattr guard so this
        # slice works standalone today and picks up the index after integration).
        disc = manager._discovered_index() if hasattr(manager, "_discovered_index") else {}
        for model_id, engine_key in disc.items():
            if model_id not in seen:
                seen.add(model_id)
                data.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": _MODELS_CREATED_TS,
                        "owned_by": engine_key,
                        "name": model_id,
                    }
                )

        return JSONResponse({"object": "list", "data": data})

    # -----------------------------------------------------------------------
    # Admin: trigger discovery scan — returns per-engine model lists
    # -----------------------------------------------------------------------

    @app.post("/admin/discover")
    async def admin_discover(request: Request) -> JSONResponse:
        """Return a per-engine summary of discoverable model ids.

        Calls available_models() on every engine (best-effort, one try/except
        per engine). Also merges in the stopped-engine map from
        manager._discovered_index() when the routing slice has populated it.
        Auth-gated identically to /admin/swap.
        """
        manager: EngineManager = request.app.state.manager
        engines_out: dict[str, list[str]] = {}

        for engine_key, engine in manager.engines.items():
            try:
                ids = await engine.available_models()
                engines_out[engine_key] = sorted(ids)
            except Exception as exc:  # noqa: BLE001
                log.warning("discover: engine %r available_models failed: %s",
                            engine_key, exc)
                engines_out[engine_key] = []

        # Merge in any stopped-engine entries from the routing slice's index.
        disc = manager._discovered_index() if hasattr(manager, "_discovered_index") else {}
        for model_id, engine_key in disc.items():
            bucket = engines_out.setdefault(engine_key, [])
            if model_id not in bucket:
                bucket.append(model_id)
                bucket.sort()

        return JSONResponse({"engines": engines_out})

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

        # Resolve a capability/alias to the real model id and rewrite the body
        # so the upstream sees the real id (no-op + unchanged bytes if not an alias).
        model, raw_body = _resolve_alias_and_rewrite(manager, model, body, raw_body)

        # Reasoning/thinking-budget guard (e.g. DiffusionGemma): on small-budget
        # chat requests, turn thinking off so the answer channel isn't starved to
        # empty. No-op + unchanged bytes for models without the policy configured.
        raw_body = _apply_thinking_policy(manager, model, path, body, raw_body)

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
                                # If the client vanished while we were waiting on
                                # a swap, stop here — returning runs the finally
                                # block which cancels the still-pending acquire.
                                # Polling explicitly (rather than leaning on
                                # Starlette's cancellation-based disconnect path)
                                # keeps swap teardown on a normal control-flow path.
                                if await request.is_disconnected():
                                    log.info(
                                        "client gone during swap wait; aborting %s", model
                                    )
                                    return
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
                                # Stop pulling from upstream the moment the client
                                # disconnects. Breaking exits the client.stream()
                                # context on a NORMAL control-flow path, so its
                                # __aexit__ deterministically closes the upstream
                                # connection and the engine aborts generation.
                                # We can't rely on Starlette cancelling this
                                # generator on disconnect: that cleanup runs under
                                # CancelledError, where the upstream close can be
                                # interrupted before the engine is told to stop —
                                # leaving a generation running to completion against
                                # a dead socket (the orphaned-generation GPU leak
                                # observed in production).
                                if await request.is_disconnected():
                                    log.info(
                                        "client disconnected; aborting upstream stream %s",
                                        model,
                                    )
                                    break
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
                        # yet (the increment is the final step, after any
                        # explicit model-load), so cancelling here is leak-free.
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
            # Non-streaming: acquire -> proxy -> release. A single JSON body
            # cannot carry keep-alive frames, so a long swap blocks until it
            # completes; non-stream callers MUST set their client read-timeout
            # above the worst-case swap (~240s for a cold ds4 start).
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

        # Resolve a capability/alias to the real model id and rewrite the body
        # so the upstream sees the real id (no-op + unchanged bytes if not an alias).
        model, raw_body = _resolve_alias_and_rewrite(manager, model, body, raw_body)

        # Ollama streams by default; only non-stream if explicitly false.
        is_stream: bool = body.get("stream", True) is not False

        fwd_headers = _build_fwd_headers(request)

        if is_stream:
            # NDJSON stream with keep-alive during swaps. The StreamingResponse
            # starts IMMEDIATELY and the acquire happens inside the generator so
            # a long swap (up to ~240s) doesn't block with zero bytes sent and
            # time the client out. While the swap is in progress we emit a bare
            # newline ("\n") as a holding frame: newline-delimited JSON readers
            # (Ollama/OpenAI-NDJSON clients iterate non-empty lines) simply skip
            # it. We MUST NOT emit SSE "data:"/comment syntax here — that would
            # corrupt the NDJSON stream.
            #
            # This mirrors the shielded-acquire + finally-release pattern proven
            # in _handle_v1_post's gen(), including its cancellation/leak-safety.
            async def ndjson_gen() -> "AsyncGenerator[bytes, None]":  # type: ignore[name-defined]
                acq = asyncio.create_task(manager.acquire(model))
                engine = None
                try:
                    # Wait for the engine to be acquired (a swap may be in
                    # progress), emitting NDJSON-safe holding frames so the
                    # client doesn't hit a TTFB/idle timeout. acq is shielded,
                    # so a keepalive timeout never cancels the in-progress swap.
                    try:
                        while not acq.done():
                            try:
                                await asyncio.wait_for(
                                    asyncio.shield(acq),
                                    timeout=cfg.swap_keepalive_interval_s,
                                )
                            except asyncio.TimeoutError:
                                # Client gone while waiting on a swap: stop and let
                                # the finally block cancel the pending acquire.
                                # (See the SSE handler for why we poll explicitly
                                # rather than rely on Starlette's disconnect path.)
                                if await request.is_disconnected():
                                    log.info(
                                        "client gone during swap wait; aborting %s", model
                                    )
                                    return
                                if cfg.swap_keepalive_enabled:
                                    log.debug(
                                        "keepalive: waiting for swap (model=%s)", model
                                    )
                                    # Bare newline: skipped by NDJSON readers.
                                    yield b"\n"
                        engine = acq.result()  # raises EngineError on failure
                    except EngineError as exc:
                        # Can't inject a JSON error into a half-started NDJSON
                        # stream without risking client confusion; log and end
                        # the stream (mirrors the upstream-error handling below).
                        log.error("acquire failed for %s: %s", model, exc)
                        return

                    log.info("api stream %s -> %s", model, engine.key)
                    url = upstream_url(engine.base_url, path)
                    try:
                        async with client.stream(
                            request.method, url, content=raw_body, headers=fwd_headers
                        ) as up:
                            async for chunk in up.aiter_raw():
                                # Abort the upstream pull when the client
                                # disconnects so the engine stops generating into
                                # a dead socket. See the SSE handler above for the
                                # full rationale (a normal-control-flow break closes
                                # the upstream deterministically; cancellation-based
                                # cleanup may not).
                                if await request.is_disconnected():
                                    log.info(
                                        "client disconnected; aborting upstream stream %s",
                                        model,
                                    )
                                    break
                                yield chunk
                    except httpx.HTTPError as exc:
                        log.error("upstream stream error on %s: %s", url, exc)
                        # Can't inject SSE; just end the stream.
                finally:
                    # Guarantee the in-flight count is released however the
                    # generator exits: normal end, EngineError, upstream error,
                    # or client disconnect (CancelledError / GeneratorExit) at
                    # any point — including mid-swap while emitting keepalives.
                    if not acq.done():
                        # Still pending => acquire hasn't incremented in-flight
                        # yet (the increment is the final step, after any
                        # explicit model-load), so cancelling here is leak-free.
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

            return StreamingResponse(ndjson_gen(), media_type="application/x-ndjson")
        else:
            # Non-streaming: acquire -> proxy -> release. A single JSON body
            # cannot carry holding frames, so a long swap blocks until it
            # completes; non-stream callers MUST set their client read-timeout
            # above the worst-case swap (~240s for a cold ds4 start).
            engine = None
            try:
                engine = await manager.acquire(model)
            except EngineError as exc:
                status = _error_status_for(exc)
                return JSONResponse(_openai_error(str(exc), "engine_error"), status_code=status)

            log.info("api request %s -> %s (stream=False)", model, engine.key)
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

        ollama_engine = _find_ollama_engine(manager)
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
    # to ollama without swap (management endpoints). Destructive endpoints are
    # refused unless explicitly enabled in config: /api/delete removes local
    # models, /api/create + /api/blobs write new ones, /api/copy clones,
    # /api/push uploads to a registry — none of which a chat client needs, and
    # all of which would otherwise be exposed to anyone who can reach the
    # router (defeating a firewall that blocks direct access to Ollama).
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def api_catchall(request: Request, path: str) -> Response:
        head = path.split("/", 1)[0].lower()
        if head in _DESTRUCTIVE_API_SEGMENTS and not cfg.allow_destructive_ollama_api:
            return JSONResponse(
                _openai_error(
                    f"/api/{head} is disabled through the router; set "
                    "'allow_destructive_ollama_api: true' in the router config "
                    "to enable it, or call the engine directly from its host",
                    "permission_error",
                    "destructive_api_disabled",
                ),
                status_code=403,
            )
        return await _passthrough_to_ollama(request, f"/api/{path}")

    return app
