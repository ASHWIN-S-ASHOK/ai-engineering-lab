"""Switchable REST embedding adapters for Gemini, OpenAI, and Ollama."""

import asyncio
from typing import Protocol, runtime_checkable

from pydantic import SecretStr

from embeddings_lab.models import (
    EmbeddingProviderName,
    EmbeddingTask,
    NonRetryableEmbeddingError,
    PreparedHttpRequest,
    ProviderRuntimeConfig,
)
from embeddings_lab.transport import ResilientHttpTransport

_GEMINI_TASK_TYPES: dict[EmbeddingTask, str] = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Uniform async embedding interface across cloud and local backends."""

    @property
    def provider_name(self) -> EmbeddingProviderName:
        """Identifier of the embedding provider."""
        ...

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
        transport: ResilientHttpTransport,
    ) -> list[list[float]]:
        """Generate one embedding vector per text in ``texts``."""
        ...


def _require_key(api_key: SecretStr | None, provider: EmbeddingProviderName) -> str:
    if api_key is None or not api_key.get_secret_value().strip():
        raise NonRetryableEmbeddingError(f"Missing API key for provider '{provider}'.")
    return api_key.get_secret_value()


class GeminiEmbedder:
    """Google Gemini embedding adapter (``POST /v1beta/models/{model}:embedContent``)."""

    provider_name: EmbeddingProviderName = "gemini"

    def build_single_request(
        self,
        text: str,
        *,
        task: EmbeddingTask,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        """Build a single ``embedContent`` request with explicit ``taskType``."""
        key = _require_key(api_key, self.provider_name)
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/v1beta/models/{runtime.model}:embedContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json_body={
                "model": f"models/{runtime.model}",
                "content": {"parts": [{"text": text}]},
                "taskType": _GEMINI_TASK_TYPES[task],
            },
        )

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
        transport: ResilientHttpTransport,
    ) -> list[list[float]]:
        """Embed texts concurrently using ``embedContent``."""
        if not texts:
            return []
        requests = [
            self.build_single_request(text, task=task, runtime=runtime, api_key=api_key)
            for text in texts
        ]
        responses = await asyncio.gather(*(transport.post_json(req) for req in requests))
        vectors: list[list[float]] = []
        for payload, _, attempts in responses:
            values = (payload.get("embedding") or {}).get("values")
            if not isinstance(values, list):
                raise NonRetryableEmbeddingError(
                    "Gemini embedContent response missing 'embedding.values'.",
                    attempts=attempts,
                )
            vectors.append([float(v) for v in values])
        return vectors


class OpenAIEmbedder:
    """OpenAI embedding adapter (``POST /v1/embeddings``)."""

    provider_name: EmbeddingProviderName = "openai"

    def build_batch_request(
        self,
        texts: list[str],
        *,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        """Build a batched ``/v1/embeddings`` request."""
        key = _require_key(api_key, self.provider_name)
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/embeddings",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json_body={
                "model": runtime.model,
                "input": texts,
                "encoding_format": "float",
            },
        )

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
        transport: ResilientHttpTransport,
    ) -> list[list[float]]:
        """Embed all texts in one OpenAI batch call."""
        del task
        if not texts:
            return []
        prepared = self.build_batch_request(texts, runtime=runtime, api_key=api_key)
        payload, _, attempts = await transport.post_json(prepared)
        items = payload.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise NonRetryableEmbeddingError(
                "OpenAI embeddings response missing expected 'data' items.",
                attempts=attempts,
            )
        ordered = sorted(items, key=lambda entry: int(entry.get("index", 0)))
        return [[float(v) for v in entry["embedding"]] for entry in ordered]


class OllamaEmbedder:
    """Local Ollama embedding adapter (``POST /api/embed``)."""

    provider_name: EmbeddingProviderName = "ollama"

    def build_batch_request(
        self,
        texts: list[str],
        *,
        runtime: ProviderRuntimeConfig,
    ) -> PreparedHttpRequest:
        """Build a batched ``/api/embed`` request."""
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/api/embed",
            headers={"Content-Type": "application/json"},
            json_body={
                "model": runtime.model,
                "input": texts,
            },
        )

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
        transport: ResilientHttpTransport,
    ) -> list[list[float]]:
        """Embed all texts in one local Ollama ``/api/embed`` request."""
        del task, api_key
        if not texts:
            return []
        prepared = self.build_batch_request(texts, runtime=runtime)
        payload, _, attempts = await transport.post_json(prepared)
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise NonRetryableEmbeddingError(
                "Ollama /api/embed response missing expected 'embeddings' list.",
                attempts=attempts,
            )
        return [[float(v) for v in vec] for vec in embeddings]


EMBEDDER_REGISTRY: dict[EmbeddingProviderName, EmbeddingProvider] = {
    "gemini": GeminiEmbedder(),
    "openai": OpenAIEmbedder(),
    "ollama": OllamaEmbedder(),
}
