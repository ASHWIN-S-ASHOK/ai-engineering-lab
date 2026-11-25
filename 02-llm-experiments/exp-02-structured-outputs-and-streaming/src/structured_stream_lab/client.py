"""High-level client for structured JSON-schema extraction and normalized streaming."""

import logging
import time
from collections.abc import AsyncIterator
from types import TracebackType

import httpx
from pydantic import ValidationError

from structured_stream_lab.adapters import ADAPTER_REGISTRY, StructuredStreamProvider
from structured_stream_lab.config import AppSettings
from structured_stream_lab.models import (
    GenerationRequest,
    ProviderName,
    SchemaValidationError,
    StreamEvent,
    StructuredResult,
    TModel,
    TokenUsage,
    schema_from_model,
)
from structured_stream_lab.streaming import ResilientHttpTransport

logger = logging.getLogger(__name__)


class StructuredStreamingClient:
    """Facade providing typed structured outputs and normalized SSE/NDJSON streaming."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        http_transport: ResilientHttpTransport | None = None,
        mock_transport: httpx.AsyncBaseTransport | None = None,
        adapters: dict[ProviderName, StructuredStreamProvider] | None = None,
    ) -> None:
        self._settings = settings
        self._adapters = adapters or ADAPTER_REGISTRY
        self._owns_transport = http_transport is None
        self._transport = http_transport or ResilientHttpTransport(
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.http_max_retries,
            transport=mock_transport,
        )

    async def __aenter__(self) -> "StructuredStreamingClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close owned HTTP transport resources."""
        if self._owns_transport:
            await self._transport.aclose()

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[TModel],
        system_prompt: str | None = None,
        provider: ProviderName | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
    ) -> StructuredResult[TModel]:
        """Generate JSON constrained by ``response_model`` and validate it with Pydantic."""
        runtime, api_key = self._settings.resolve_provider(
            provider=provider,
            require_credentials=True,
        )
        adapter = self._adapters[runtime.provider]
        request = GenerationRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            schema_spec=schema_from_model(response_model),
            stream=False,
        )
        prepared = adapter.build_request(request, runtime, api_key)
        payload, latency_ms, attempts = await self._transport.post_json(prepared)
        raw = adapter.parse_response(
            payload,
            requested_model=runtime.model,
            latency_ms=latency_ms,
            attempts=attempts,
        )

        try:
            validated = response_model.model_validate_json(raw.text)
        except ValidationError as exc:
            raise SchemaValidationError(
                f"Provider '{raw.provider}' output failed {response_model.__name__} validation "
                f"(finish_reason={raw.finish_reason}): {exc}",
                raw_text=raw.text,
                finish_reason=raw.finish_reason,
            ) from exc

        logger.info(
            "Structured call provider=%s model=%s schema=%s latency_ms=%.2f",
            raw.provider,
            raw.model,
            response_model.__name__,
            raw.latency_ms,
        )
        return StructuredResult[TModel](
            provider=raw.provider,
            model=raw.model,
            data=validated,
            raw_text=raw.text,
            finish_reason=raw.finish_reason,
            usage=raw.usage,
            latency_ms=raw.latency_ms,
            attempts=raw.attempts,
        )

    async def stream_generate(
        self,
        request: GenerationRequest,
        *,
        provider: ProviderName | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream normalized text deltas and final TTFT/total latency metrics."""
        runtime, api_key = self._settings.resolve_provider(
            provider=provider,
            require_credentials=True,
        )
        adapter = self._adapters[runtime.provider]
        stream_req = request.model_copy(update={"stream": True})
        prepared = adapter.build_request(stream_req, runtime, api_key)

        started = time.perf_counter()
        ttft_ms: float | None = None
        input_tokens = 0
        output_tokens = 0
        total_tokens: int | None = None
        last_finish_reason: str | None = None
        emitted_done = False

        async for raw_event in self._transport.stream_events(
            prepared,
            framing=adapter.stream_framing,
        ):
            chunk = adapter.parse_stream_event(raw_event)
            if chunk.input_tokens is not None:
                input_tokens = chunk.input_tokens
            if chunk.output_tokens is not None:
                output_tokens = chunk.output_tokens
            if chunk.total_tokens is not None:
                total_tokens = chunk.total_tokens
            if chunk.finish_reason is not None:
                last_finish_reason = chunk.finish_reason

            if chunk.delta_text:
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - started) * 1000.0, 2)
                yield StreamEvent(
                    provider=runtime.provider,
                    delta_text=chunk.delta_text,
                    done=False,
                    ttft_ms=ttft_ms,
                )

            if chunk.done:
                emitted_done = True
                total_ms = round((time.perf_counter() - started) * 1000.0, 2)
                usage = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=(
                        total_tokens if total_tokens is not None else input_tokens + output_tokens
                    ),
                )
                yield StreamEvent(
                    provider=runtime.provider,
                    done=True,
                    finish_reason=last_finish_reason,
                    usage=usage,
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                )

        if not emitted_done:
            total_ms = round((time.perf_counter() - started) * 1000.0, 2)
            yield StreamEvent(
                provider=runtime.provider,
                done=True,
                finish_reason=last_finish_reason,
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=(
                        total_tokens if total_tokens is not None else input_tokens + output_tokens
                    ),
                ),
                ttft_ms=ttft_ms,
                total_ms=total_ms,
            )


def create_client(
    settings: AppSettings | None = None,
    *,
    mock_transport: httpx.AsyncBaseTransport | None = None,
) -> StructuredStreamingClient:
    """Factory creating a StructuredStreamingClient from settings."""
    return StructuredStreamingClient(
        settings=settings or AppSettings(),
        mock_transport=mock_transport,
    )
