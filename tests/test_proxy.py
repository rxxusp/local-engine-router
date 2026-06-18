"""Tests for router/proxy.py utilities.

Covers:
  - filter_request_headers  — hop-by-hop and problematic headers are stripped,
                               application headers are passed through unchanged.
  - filter_response_headers — hop-by-hop + length/encoding headers stripped,
                               application headers pass through.
  - upstream_url             — base_url + path construction edge cases.
  - forward()                — mocked httpx client; asserts status, headers, body
                               are proxied correctly and response headers filtered.

No network, no GPU, no real httpx transport.
"""

from __future__ import annotations

import pytest
import httpx

from router.proxy import (
    _DROP_REQUEST_HEADERS,
    _DROP_RESPONSE_HEADERS,
    filter_request_headers,
    filter_response_headers,
    upstream_url,
    forward,
)


# --------------------------------------------------------------------------- #
# upstream_url
# --------------------------------------------------------------------------- #

class TestUpstreamUrl:
    def test_basic_concatenation(self):
        assert upstream_url("http://127.0.0.1:8080", "/v1/chat/completions") == \
            "http://127.0.0.1:8080/v1/chat/completions"

    def test_trailing_slash_on_base_stripped(self):
        assert upstream_url("http://127.0.0.1:8080/", "/v1/models") == \
            "http://127.0.0.1:8080/v1/models"

    def test_multiple_trailing_slashes_stripped(self):
        assert upstream_url("http://host:9000///", "/health") == \
            "http://host:9000/health"

    def test_path_without_leading_slash(self):
        # The function simply concatenates; callers that omit the leading slash
        # get a joined result.  Assert we don't crash and the result is deterministic.
        result = upstream_url("http://host:8080", "no-slash")
        assert result == "http://host:8080no-slash"

    def test_empty_path(self):
        result = upstream_url("http://host:8080", "")
        assert result == "http://host:8080"

    def test_path_with_query_string(self):
        result = upstream_url("http://host:8080", "/v1/models?limit=10")
        assert result == "http://host:8080/v1/models?limit=10"


# --------------------------------------------------------------------------- #
# filter_request_headers
# --------------------------------------------------------------------------- #

class TestFilterRequestHeaders:
    def test_strips_host(self):
        result = filter_request_headers({"host": "original.host", "authorization": "Bearer x"})
        assert "host" not in result
        assert "Host" not in result

    def test_strips_content_length(self):
        result = filter_request_headers({"content-length": "42", "content-type": "application/json"})
        assert "content-length" not in result
        assert "Content-length" not in result

    def test_strips_connection(self):
        result = filter_request_headers({"connection": "keep-alive"})
        assert "connection" not in result

    def test_strips_transfer_encoding(self):
        result = filter_request_headers({"transfer-encoding": "chunked"})
        assert "transfer-encoding" not in result

    def test_strips_keep_alive(self):
        result = filter_request_headers({"keep-alive": "timeout=5"})
        assert "keep-alive" not in result

    def test_strips_accept_encoding(self):
        # We want raw bytes, not gzip/deflate from the upstream.
        result = filter_request_headers({"accept-encoding": "gzip, deflate"})
        assert "accept-encoding" not in result

    def test_strips_proxy_headers(self):
        headers = {
            "proxy-authenticate": "Basic",
            "proxy-authorization": "Basic abc",
            "proxy-connection": "keep-alive",
        }
        result = filter_request_headers(headers)
        assert not any(k in result for k in headers)

    def test_strips_te_trailer_upgrade(self):
        headers = {"te": "trailers", "trailer": "Expires", "upgrade": "websocket"}
        result = filter_request_headers(headers)
        assert not any(k in result for k in headers)

    def test_preserves_authorization(self):
        result = filter_request_headers({"authorization": "Bearer token123"})
        assert result.get("authorization") == "Bearer token123"

    def test_preserves_content_type(self):
        result = filter_request_headers({"content-type": "application/json"})
        assert result.get("content-type") == "application/json"

    def test_preserves_custom_header(self):
        result = filter_request_headers({"x-custom-header": "value"})
        assert result.get("x-custom-header") == "value"

    def test_case_insensitive_stripping(self):
        """Headers stored with mixed case should still be dropped."""
        # filter_request_headers does k.lower() comparison.
        result = filter_request_headers({
            "Host": "example.com",
            "Content-Length": "10",
            "Transfer-Encoding": "chunked",
        })
        assert "Host" not in result
        assert "Content-Length" not in result
        assert "Transfer-Encoding" not in result

    def test_empty_headers_returns_empty(self):
        assert filter_request_headers({}) == {}

    def test_only_hop_by_hop_returns_empty(self):
        headers = {h: "v" for h in ["host", "connection", "transfer-encoding"]}
        assert filter_request_headers(headers) == {}

    def test_returns_dict(self):
        result = filter_request_headers({"content-type": "text/plain"})
        assert isinstance(result, dict)

    def test_all_drop_request_headers_stripped(self):
        """Every header in _DROP_REQUEST_HEADERS must be removed."""
        headers = {h: "x" for h in _DROP_REQUEST_HEADERS}
        result = filter_request_headers(headers)
        assert result == {}

    def test_mixed_drop_and_keep(self):
        headers = {
            "host": "should-go",
            "content-length": "should-go",
            "authorization": "keep-me",
            "content-type": "keep-me-too",
        }
        result = filter_request_headers(headers)
        assert "host" not in result
        assert "content-length" not in result
        assert result["authorization"] == "keep-me"
        assert result["content-type"] == "keep-me-too"


# --------------------------------------------------------------------------- #
# filter_response_headers
# --------------------------------------------------------------------------- #

class TestFilterResponseHeaders:
    def test_strips_connection(self):
        result = filter_response_headers({"connection": "close"})
        assert "connection" not in result

    def test_strips_content_length(self):
        # Starlette recomputes content-length from the body.
        result = filter_response_headers({"content-length": "99"})
        assert "content-length" not in result

    def test_strips_content_encoding(self):
        # We pass raw bytes; Starlette handles re-encoding.
        result = filter_response_headers({"content-encoding": "gzip"})
        assert "content-encoding" not in result

    def test_strips_transfer_encoding(self):
        result = filter_response_headers({"transfer-encoding": "chunked"})
        assert "transfer-encoding" not in result

    def test_strips_keep_alive(self):
        result = filter_response_headers({"keep-alive": "timeout=60"})
        assert "keep-alive" not in result

    def test_strips_proxy_headers(self):
        headers = {
            "proxy-authenticate": "Basic",
            "proxy-authorization": "Basic abc",
            "proxy-connection": "keep-alive",
        }
        result = filter_response_headers(headers)
        assert not any(k in result for k in headers)

    def test_strips_te_trailer_upgrade(self):
        headers = {"te": "trailers", "trailer": "Expires", "upgrade": "websocket"}
        result = filter_response_headers(headers)
        assert not any(k in result for k in headers)

    def test_preserves_content_type(self):
        result = filter_response_headers({"content-type": "application/json"})
        assert result.get("content-type") == "application/json"

    def test_preserves_x_request_id(self):
        result = filter_response_headers({"x-request-id": "abc-123"})
        assert result.get("x-request-id") == "abc-123"

    def test_case_insensitive_stripping(self):
        result = filter_response_headers({
            "Content-Length": "500",
            "Transfer-Encoding": "chunked",
            "Content-Encoding": "gzip",
        })
        assert "Content-Length" not in result
        assert "Transfer-Encoding" not in result
        assert "Content-Encoding" not in result

    def test_empty_headers_returns_empty(self):
        assert filter_response_headers({}) == {}

    def test_all_drop_response_headers_stripped(self):
        """Every header in _DROP_RESPONSE_HEADERS must be removed."""
        headers = {h: "x" for h in _DROP_RESPONSE_HEADERS}
        result = filter_response_headers(headers)
        assert result == {}

    def test_returns_dict(self):
        assert isinstance(filter_response_headers({}), dict)

    def test_mixed_drop_and_keep_response(self):
        headers = {
            "content-length": "should-go",
            "content-type": "application/json",
            "x-custom": "stays",
        }
        result = filter_response_headers(headers)
        assert "content-length" not in result
        assert result["content-type"] == "application/json"
        assert result["x-custom"] == "stays"


# --------------------------------------------------------------------------- #
# forward() — mocked httpx.AsyncClient
# --------------------------------------------------------------------------- #

class _FakeAsyncClient:
    """Minimal stub that replaces httpx.AsyncClient.request() for forward() tests."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._headers = headers or {}
        self._content = content
        self.last_request: dict | None = None

    async def request(
        self,
        method: str,
        url: str,
        content: bytes = b"",
        headers: dict | None = None,
    ) -> httpx.Response:
        self.last_request = {
            "method": method,
            "url": url,
            "content": content,
            "headers": dict(headers or {}),
        }
        # Build a minimal httpx.Response so filter_response_headers has real .headers.
        return httpx.Response(
            status_code=self.status_code,
            headers=self._headers,
            content=self._content,
        )


@pytest.mark.asyncio
async def test_forward_returns_status_code():
    client = _FakeAsyncClient(status_code=201, content=b"{}")
    status, _, _ = await forward(client, "POST", "http://up/path", {}, b"body")
    assert status == 201


@pytest.mark.asyncio
async def test_forward_returns_body():
    body = b'{"choices":[]}'
    client = _FakeAsyncClient(content=body)
    _, _, returned_body = await forward(client, "POST", "http://up/v1/chat", {}, b"")
    assert returned_body == body


@pytest.mark.asyncio
async def test_forward_passes_method_and_url():
    client = _FakeAsyncClient()
    await forward(client, "GET", "http://upstream:8080/v1/models", {}, b"")
    assert client.last_request["method"] == "GET"
    assert client.last_request["url"] == "http://upstream:8080/v1/models"


@pytest.mark.asyncio
async def test_forward_passes_body_to_upstream():
    client = _FakeAsyncClient()
    await forward(client, "POST", "http://up/chat", {}, b"hello-body")
    assert client.last_request["content"] == b"hello-body"


@pytest.mark.asyncio
async def test_forward_strips_hop_by_hop_from_response():
    """forward() must strip hop-by-hop headers from the upstream response."""
    client = _FakeAsyncClient(
        headers={
            "content-type": "application/json",
            "content-length": "14",         # must be stripped
            "transfer-encoding": "chunked",  # must be stripped
            "x-keep": "this",
        },
        content=b"{}",
    )
    _, headers, _ = await forward(client, "GET", "http://up/", {}, b"")
    assert "content-length" not in headers
    assert "transfer-encoding" not in headers
    assert headers.get("content-type") == "application/json"
    assert headers.get("x-keep") == "this"


@pytest.mark.asyncio
async def test_forward_returns_filtered_headers_dict():
    client = _FakeAsyncClient(
        headers={"content-type": "text/plain", "connection": "keep-alive"},
        content=b"hi",
    )
    _, headers, _ = await forward(client, "GET", "http://up/", {}, b"")
    assert isinstance(headers, dict)
    assert "connection" not in headers
    assert "content-type" in headers


@pytest.mark.asyncio
async def test_forward_passes_request_headers_to_upstream():
    """Headers supplied to forward() are forwarded verbatim to the upstream."""
    client = _FakeAsyncClient()
    req_headers = {"authorization": "Bearer tok", "content-type": "application/json"}
    await forward(client, "POST", "http://up/", req_headers, b"")
    sent = client.last_request["headers"]
    assert sent.get("authorization") == "Bearer tok"
    assert sent.get("content-type") == "application/json"


@pytest.mark.asyncio
async def test_forward_empty_body():
    client = _FakeAsyncClient(content=b"")
    status, _, body = await forward(client, "GET", "http://up/health", {}, b"")
    assert status == 200
    assert body == b""
