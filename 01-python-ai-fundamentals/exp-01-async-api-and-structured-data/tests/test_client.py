"""Unit tests for AsyncJsonClient using httpx.MockTransport (offline)."""

import httpx
import pytest

from async_api_lab.client import AsyncJsonClient
from async_api_lab.models import NonRetryableApiError, RetryableApiError


def _make_client(
    transport: httpx.MockTransport,
    *,
    max_retries: int = 2,
) -> AsyncJsonClient:
    return AsyncJsonClient(
        max_retries=max_retries,
        base_backoff_seconds=0.0,
        transport=transport,
    )


async def test_get_json_success() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"ok": True, "path": request.url.path})
    )
    async with _make_client(transport) as client:
        result = await client.get_json("https://example.test/v1/status")

    assert result.status_code == 200
    assert result.attempts == 1
    assert result.latency_ms >= 0.0
    assert result.payload == {"ok": True, "path": "/v1/status"}


@pytest.mark.parametrize("transient_status", [429, 503])
async def test_get_json_retries_transient_status_and_recovers(transient_status: int) -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(transient_status, json={"error": "busy"})
        return httpx.Response(200, json={"recovered": True})

    async with _make_client(httpx.MockTransport(handler)) as client:
        result = await client.get_json("https://example.test/retry")

    assert result.attempts == 2
    assert result.payload == {"recovered": True}


async def test_get_json_exhausts_retries_and_raises() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(502, json={"error": "bad gateway"}))

    async with _make_client(transport) as client:
        with pytest.raises(RetryableApiError) as exc_info:
            await client.get_json("https://example.test/down")

    assert exc_info.value.attempts == 3
    assert exc_info.value.status_code == 502


async def test_get_json_does_not_retry_client_errors() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    async with _make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(NonRetryableApiError) as exc_info:
            await client.get_json("https://example.test/auth")

    assert calls == 1
    assert exc_info.value.status_code == 401


async def test_get_json_rejects_malformed_or_non_object_json() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text="not-json"))

    async with _make_client(transport, max_retries=1) as client:
        with pytest.raises(NonRetryableApiError, match="Invalid JSON"):
            await client.get_json("https://example.test/bad-json")


async def test_get_json_retries_timeouts() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    async with _make_client(httpx.MockTransport(handler), max_retries=1) as client:
        with pytest.raises(RetryableApiError) as exc_info:
            await client.get_json("https://example.test/slow")

    assert calls == 2
    assert exc_info.value.attempts == 2
