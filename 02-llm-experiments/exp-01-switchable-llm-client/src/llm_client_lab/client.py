"""High-level switchable LLM client orchestrating config, adapters, and transport."""

import logging
from types import TracebackType

import httpx

from llm_client_lab.adapters import ADAPTER_REGISTRY, LLMProvider
from llm_client_lab.config import AppSettings
from llm_client_lab.models import LLMRequest, LLMResponse, ProviderName
from llm_client_lab.transport import ResilientHttpTransport

logger = logging.getLogger(__name__)


class SwitchableLLMClient:
    """Facade executing normalized LLMRequest calls against any configured provider."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        http_transport: ResilientHttpTransport | None = None,
        mock_transport: httpx.AsyncBaseTransport | None = None,
        adapters: dict[ProviderName, LLMProvider] | None = None,
    ) -> None:
        self._settings = settings
        self._adapters = adapters or ADAPTER_REGISTRY
        self._owns_transport = http_transport is None
        self._transport = http_transport or ResilientHttpTransport(
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.http_max_retries,
            transport=mock_transport,
        )

    async def __aenter__(self) -> "SwitchableLLMClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close owned HTTP resources."""
        if self._owns_transport:
            await self._transport.aclose()

    async def generate(
        self,
        request: LLMRequest,
        *,
        provider: ProviderName | None = None,
    ) -> LLMResponse:
        """Send ``request`` to the configured or overridden provider and return LLMResponse."""
        runtime, api_key = self._settings.resolve_provider(
            provider=provider,
            require_credentials=True,
        )
        adapter = self._adapters[runtime.provider]
        prepared = adapter.build_request(request, runtime, api_key)

        payload, latency_ms, attempts = await self._transport.post_json(prepared)
        response = adapter.parse_response(
            payload,
            requested_model=runtime.model,
            latency_ms=latency_ms,
            attempts=attempts,
        )
        logger.info(
            "LLM call provider=%s model=%s in_tokens=%d out_tokens=%d latency_ms=%.2f attempts=%d",
            response.provider,
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.latency_ms,
            response.attempts,
        )
        return response


def create_llm_client(
    settings: AppSettings | None = None,
    *,
    mock_transport: httpx.AsyncBaseTransport | None = None,
) -> SwitchableLLMClient:
    """Factory creating a SwitchableLLMClient from environment settings."""
    return SwitchableLLMClient(
        settings=settings or AppSettings(),
        mock_transport=mock_transport,
    )
