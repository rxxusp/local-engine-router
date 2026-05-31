"""HTTP proxy plumbing for llm-router.

Provides:
  - make_client(cfg)       — factory for the long-lived httpx.AsyncClient used
                             for user traffic (unbounded read timeout).
  - upstream_url(engine, path) — build the full URL from engine.base_url + path.
  - filter_request_headers(headers) — strip hop-by-hop / problematic headers
                                       before forwarding to upstream.
  - filter_response_headers(headers) — strip hop-by-hop + length/encoding
                                        headers from upstream responses.
  - forward(client, method, url, headers, body) — non-streaming forward;
                                                    returns (status, headers, bytes).

Streaming is done by callers via ``client.stream(...)`` directly (see
router/app.py), so they can interleave SSE keep-alive comments with the
upstream byte stream during an engine swap.
"""

from __future__ import annotations

import logging
from typing import Mapping

import httpx

from .config import RouterConfig

log = logging.getLogger("router.proxy")

# ---------------------------------------------------------------------------
# Hop-by-hop and problematic headers to strip from *request* before forwarding
# ---------------------------------------------------------------------------
_DROP_REQUEST_HEADERS: frozenset[str] = frozenset(
    [
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "upgrade",
        "accept-encoding",  # we want raw bytes, not gzip/deflate
    ]
)

# Headers to strip from *upstream response* before returning to clients.
# Starlette (FastAPI) recomputes content-length and handles transfer-encoding.
_DROP_RESPONSE_HEADERS: frozenset[str] = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",   # Starlette will set this from the body
        "content-encoding", # we pass raw bytes; Starlette handles encoding
    ]
)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_client(cfg: RouterConfig) -> httpx.AsyncClient:
    """Return a long-lived AsyncClient for user traffic.

    The connect timeout respects the configured value; read/write/pool
    timeouts are unbounded so long-running generations and streaming
    responses don't time out mid-flight.
    """
    timeout = httpx.Timeout(
        connect=cfg.upstream_connect_timeout_s,
        read=None,
        write=None,
        pool=None,
    )
    return httpx.AsyncClient(timeout=timeout)


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

def upstream_url(base_url: str, path: str) -> str:
    """Concatenate *base_url* (no trailing slash) with *path* (leading slash)."""
    return base_url.rstrip("/") + path


# ---------------------------------------------------------------------------
# Header filters
# ---------------------------------------------------------------------------

def filter_request_headers(
    headers: Mapping[str, str],
) -> dict[str, str]:
    """Return a copy of *headers* with hop-by-hop / problematic entries removed."""
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _DROP_REQUEST_HEADERS
    }


def filter_response_headers(
    headers: Mapping[str, str],
) -> dict[str, str]:
    """Return upstream response headers safe to pass back to the client."""
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _DROP_RESPONSE_HEADERS
    }


# ---------------------------------------------------------------------------
# Non-streaming forward
# ---------------------------------------------------------------------------

async def forward(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> tuple[int, dict[str, str], bytes]:
    """Send *method* request to *url* and return (status_code, headers, body).

    Headers are filtered for both the outgoing request and the returned
    response.  The caller is responsible for error handling.
    """
    resp = await client.request(
        method,
        url,
        content=body,
        headers=headers,
    )
    return resp.status_code, filter_response_headers(dict(resp.headers)), resp.content
