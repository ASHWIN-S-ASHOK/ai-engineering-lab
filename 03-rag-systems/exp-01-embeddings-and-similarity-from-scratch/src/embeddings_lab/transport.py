"""Shared resilient async HTTP POST transport with bounded retries."""

import asyncio
import logging
import time
from types import TracebackType
from typing import Any

import httpx

from embeddings_lab.models import (
    NonRetryableEmbeddingError,
    PreparedHttpRequest,
    RetryableEmbeddingError,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class ResilientHttpTransport:
    """Async HTTP POST client with exponential backoff on 429/5xx/timeouts."""

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
        """Execute JSON POST with retries and return ``(payload, latency_ms, attempts)``."""
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

    async def _attempt_post(
        self,
        prepared: PreparedHttpRequest,
        attempt: int,
    ) -> tuple[
        bool,
        dict[str, Any] | None,
        RetryableEmbeddingError | NonRetryableEmbeddingError | None,
    ]:
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
                RetryableEmbeddingError(
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
                RetryableEmbeddingError(
                    f"HTTP {status} from {prepared.url}: {detail}",
                    attempts=attempt,
                    status_code=status,
                ),
            )
        if status >= 400:
            return (
                False,
                None,
                NonRetryableEmbeddingError(
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
                NonRetryableEmbeddingError(
                    f"Invalid JSON from {prepared.url}: {exc}",
                    attempts=attempt,
                    status_code=status,
                ),
            )
        if not isinstance(data, dict):
            return (
                False,
                None,
                NonRetryableEmbeddingError(
                    f"Expected JSON object from {prepared.url}, got {type(data).__name__}",
                    attempts=attempt,
                    status_code=status,
                ),
            )
        return False, data, None
