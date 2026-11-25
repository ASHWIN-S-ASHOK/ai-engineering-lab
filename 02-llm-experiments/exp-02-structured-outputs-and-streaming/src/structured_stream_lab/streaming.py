"""Shared SSE / NDJSON framing parsers and resilient async HTTP transport."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

import httpx

from structured_stream_lab.models import (
    NonRetryableStreamError,
    PreparedHttpRequest,
    RetryableStreamError,
    StreamFraming,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


async def iter_sse_json(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    """Parse Server-Sent Events (SSE) ``data:`` lines into JSON dictionaries."""
    data_buffer: list[str] = []

    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if line == "":
            if data_buffer:
                payload = _decode_sse_buffer("\n".join(data_buffer))
                data_buffer.clear()
                if payload is not None:
                    yield payload
            continue

        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_buffer.append(line[5:].lstrip())

    if data_buffer:
        payload = _decode_sse_buffer("\n".join(data_buffer))
        if payload is not None:
            yield payload


async def iter_ndjson(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    """Parse Newline-Delimited JSON (NDJSON) lines into JSON dictionaries."""
    async for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        yield _parse_json_dict(stripped, source="NDJSON")


def _decode_sse_buffer(raw_data: str) -> dict[str, Any] | None:
    stripped = raw_data.strip()
    if not stripped or stripped == "[DONE]":
        return None
    return _parse_json_dict(stripped, source="SSE")


def _parse_json_dict(text: str, *, source: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise NonRetryableStreamError(f"Malformed {source} JSON chunk: {exc}") from exc
    if not isinstance(parsed, dict):
        raise NonRetryableStreamError(
            f"Expected JSON object in {source} chunk, got {type(parsed).__name__}."
        )
    return parsed


class ResilientHttpTransport:
    """Async HTTP transport supporting retried JSON POSTs and SSE/NDJSON streams."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        base_backoff_seconds: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def __aenter__(self) -> "ResilientHttpTransport":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close underlying httpx connection pool."""
        await self._client.aclose()

    async def post_json(
        self,
        prepared: PreparedHttpRequest,
    ) -> tuple[dict[str, Any], float, int]:
        """Send a non-streaming JSON POST request with bounded retries."""
        started = time.perf_counter()
        total_budget = self._max_retries + 1

        for attempt in range(1, total_budget + 1):
            should_retry, payload, error = await self._attempt_post(prepared, attempt)
            if payload is not None:
                latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
                return payload, latency_ms, attempt
            if not should_retry or attempt == total_budget:
                assert error is not None
                raise error

            delay = self._base_backoff_seconds * (2 ** (attempt - 1))
            logger.info(
                "Retrying %s in %.2fs (attempt %d/%d)",
                prepared.url,
                delay,
                attempt,
                total_budget,
            )
            await asyncio.sleep(delay)

        raise AssertionError("Unreachable retry loop termination")

    async def stream_events(
        self,
        prepared: PreparedHttpRequest,
        *,
        framing: StreamFraming,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open a streaming POST request and yield decoded SSE or NDJSON objects."""
        try:
            async with self._client.stream(
                "POST",
                prepared.url,
                headers=prepared.headers,
                json=prepared.json_body,
            ) as response:
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    body = (await response.aread()).decode("utf-8", errors="replace")[:240]
                    raise RetryableStreamError(
                        f"HTTP {response.status_code} from {prepared.url}: {body}",
                        attempts=1,
                        status_code=response.status_code,
                    )
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")[:240]
                    raise NonRetryableStreamError(
                        f"HTTP {response.status_code} from {prepared.url}: {body}",
                        attempts=1,
                        status_code=response.status_code,
                    )

                parser = iter_sse_json if framing == "sse" else iter_ndjson
                async for event in parser(response.aiter_lines()):
                    yield event
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableStreamError(
                f"Streaming transport error calling {prepared.url}: {exc}",
                attempts=1,
            ) from exc

    async def _attempt_post(
        self,
        prepared: PreparedHttpRequest,
        attempt: int,
    ) -> tuple[bool, dict[str, Any] | None, RetryableStreamError | NonRetryableStreamError | None]:
        try:
            response = await self._client.post(
                prepared.url,
                headers=prepared.headers,
                json=prepared.json_body,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return (
                True,
                None,
                RetryableStreamError(
                    f"Transport error calling {prepared.url}: {exc}",
                    attempts=attempt,
                ),
            )

        status = response.status_code
        detail = response.text[:240].strip() or "empty error body"
        if status in _RETRYABLE_STATUS_CODES:
            return (
                True,
                None,
                RetryableStreamError(
                    f"HTTP {status} from {prepared.url}: {detail}",
                    attempts=attempt,
                    status_code=status,
                ),
            )
        if status >= 400:
            return (
                False,
                None,
                NonRetryableStreamError(
                    f"HTTP {status} from {prepared.url}: {detail}",
                    attempts=attempt,
                    status_code=status,
                ),
            )

        try:
            data = response.json()
        except ValueError as exc:
            return (
                False,
                None,
                NonRetryableStreamError(
                    f"Invalid JSON from {prepared.url}: {exc}",
                    attempts=attempt,
                    status_code=status,
                ),
            )
        if not isinstance(data, dict):
            return (
                False,
                None,
                NonRetryableStreamError(
                    f"Expected JSON object from {prepared.url}, got {type(data).__name__}",
                    attempts=attempt,
                    status_code=status,
                ),
            )
        return False, data, None
