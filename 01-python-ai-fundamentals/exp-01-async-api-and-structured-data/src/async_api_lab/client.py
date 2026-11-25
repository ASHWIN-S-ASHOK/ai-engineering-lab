"""Resilient async HTTP JSON client with bounded retries and latency measurement."""

import asyncio
import logging
import time
from types import TracebackType
from typing import Any

import httpx

from async_api_lab.models import FetchResult, NonRetryableApiError, RetryableApiError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class AsyncJsonClient:
    """Async context-managed HTTP client with exponential backoff for transient errors."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        base_backoff_seconds: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def __aenter__(self) -> "AsyncJsonClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying httpx connection pool."""
        await self._client.aclose()

    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> FetchResult:
        """Fetch a JSON object from ``url``, retrying transient HTTP/network failures."""
        started = time.perf_counter()
        total_budget = self._max_retries + 1

        for attempt in range(1, total_budget + 1):
            should_retry, result, error = await self._attempt_get(
                url=url,
                headers=headers,
                attempt=attempt,
                started=started,
            )
            if result is not None:
                return result
            if not should_retry or attempt == total_budget:
                assert error is not None
                raise error

            delay = self._base_backoff_seconds * (2 ** (attempt - 1))
            logger.info("Retrying %s in %.2fs (attempt %d/%d)", url, delay, attempt, total_budget)
            await asyncio.sleep(delay)

        raise AssertionError("Unreachable retry loop termination")

    async def _attempt_get(
        self,
        *,
        url: str,
        headers: dict[str, str] | None,
        attempt: int,
        started: float,
    ) -> tuple[bool, FetchResult | None, RetryableApiError | NonRetryableApiError | None]:
        """Execute one HTTP GET attempt and classify its outcome."""
        try:
            response = await self._client.get(url, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.warning(
                "Transient transport failure for %s on attempt %d: %s", url, attempt, exc
            )
            return (
                True,
                None,
                RetryableApiError(
                    f"Transport error requesting {url}: {exc}",
                    attempts=attempt,
                ),
            )

        status = response.status_code
        if status in _RETRYABLE_STATUS_CODES:
            logger.warning("Retryable HTTP %d from %s on attempt %d", status, url, attempt)
            return (
                True,
                None,
                RetryableApiError(
                    f"HTTP {status} from {url}",
                    attempts=attempt,
                    status_code=status,
                ),
            )

        if status >= 400:
            return (
                False,
                None,
                NonRetryableApiError(
                    f"Non-retryable HTTP {status} from {url}",
                    attempts=attempt,
                    status_code=status,
                ),
            )

        payload = _parse_json_object(response, url=url, attempt=attempt)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        logger.info(
            "Fetched %s status=%d attempts=%d latency_ms=%.2f",
            url,
            status,
            attempt,
            latency_ms,
        )
        return (
            False,
            FetchResult(
                url=url,
                status_code=status,
                latency_ms=latency_ms,
                attempts=attempt,
                payload=payload,
            ),
            None,
        )


def _parse_json_object(response: httpx.Response, *, url: str, attempt: int) -> dict[str, Any]:
    """Parse response body and enforce a JSON object root."""
    try:
        data = response.json()
    except ValueError as exc:
        raise NonRetryableApiError(
            f"Invalid JSON payload from {url}: {exc}",
            attempts=attempt,
            status_code=response.status_code,
        ) from exc

    if not isinstance(data, dict):
        raise NonRetryableApiError(
            f"Expected JSON object from {url}, got {type(data).__name__}",
            attempts=attempt,
            status_code=response.status_code,
        )
    return data
