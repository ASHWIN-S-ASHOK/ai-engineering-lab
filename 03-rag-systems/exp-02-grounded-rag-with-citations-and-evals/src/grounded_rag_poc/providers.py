"""Switchable structured LLM adapters and embedding adapters."""

import asyncio
from typing import Any, Protocol, runtime_checkable

from pydantic import SecretStr

from grounded_rag_poc.models import (
    EmbeddingTask,
    EmbedProviderName,
    LlmProviderName,
    NonRetryableApiError,
    PreparedHttpRequest,
    ProviderRuntimeConfig,
    TokenUsage,
)
from grounded_rag_poc.transport import ResilientHttpTransport

_GEMINI_TASK_TYPES: dict[EmbeddingTask, str] = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


def _require_key(api_key: SecretStr | None, provider: str) -> str:
    if api_key is None or not api_key.get_secret_value().strip():
        raise NonRetryableApiError(f"Missing API key for provider '{provider}'.")
    return api_key.get_secret_value()


@runtime_checkable
class StructuredLlmAdapter(Protocol):
    """Protocol for structured JSON generation across LLM vendors."""

    @property
    def provider_name(self) -> LlmProviderName:
        """Identifier of the generation provider."""
        ...

    def build_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        """Build structured generation HTTP request."""
        ...

    def parse_response(self, payload: dict[str, Any]) -> tuple[str, TokenUsage]:
        """Return ``(raw_json_text, TokenUsage)`` from provider response."""
        ...


class GeminiLlmAdapter:
    """Gemini structured output adapter."""

    provider_name: LlmProviderName = "gemini"

    def build_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        del schema_name
        key = _require_key(api_key, self.provider_name)
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/v1beta/models/{runtime.model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json_body={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 512,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": json_schema,
                },
            },
        )

    def parse_response(self, payload: dict[str, Any]) -> tuple[str, TokenUsage]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise NonRetryableApiError("Gemini response missing 'candidates'.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(p["text"]) for p in parts if "text" in p)
        usage_raw = payload.get("usageMetadata", {})
        in_tok = int(usage_raw.get("promptTokenCount", 0))
        out_tok = int(usage_raw.get("candidatesTokenCount", 0))
        tot_tok = int(usage_raw.get("totalTokenCount", in_tok + out_tok))
        return text, TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok)


class OpenAILlmAdapter:
    """OpenAI Responses API structured output adapter."""

    provider_name: LlmProviderName = "openai"

    def build_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/responses",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json_body={
                "model": runtime.model,
                "instructions": system_prompt,
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
                ],
                "temperature": 0.0,
                "max_output_tokens": 512,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": json_schema,
                    }
                },
            },
        )

    def parse_response(self, payload: dict[str, Any]) -> tuple[str, TokenUsage]:
        output = payload.get("output")
        if not isinstance(output, list):
            raise NonRetryableApiError("OpenAI response missing 'output'.")
        text = "".join(
            str(part["text"])
            for item in output
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text" and "text" in part
        )
        usage_raw = payload.get("usage", {})
        in_tok = int(usage_raw.get("input_tokens", 0))
        out_tok = int(usage_raw.get("output_tokens", 0))
        tot_tok = int(usage_raw.get("total_tokens", in_tok + out_tok))
        return text, TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok)


class AnthropicLlmAdapter:
    """Anthropic Messages API structured output adapter."""

    provider_name: LlmProviderName = "anthropic"

    def build_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        del schema_name
        key = _require_key(api_key, self.provider_name)
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": runtime.anthropic_version,
            },
            json_body={
                "model": runtime.model,
                "system": system_prompt,
                "max_tokens": 512,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": user_prompt}],
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema,
                    }
                },
            },
        )

    def parse_response(self, payload: dict[str, Any]) -> tuple[str, TokenUsage]:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise NonRetryableApiError("Anthropic response missing 'content'.")
        text = "".join(
            str(b["text"]) for b in blocks if b.get("type") == "text" and "text" in b
        )
        usage_raw = payload.get("usage", {})
        in_tok = int(usage_raw.get("input_tokens", 0))
        out_tok = int(usage_raw.get("output_tokens", 0))
        return text, TokenUsage(
            input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok
        )


class OllamaLlmAdapter:
    """Local Ollama structured JSON output adapter."""

    provider_name: LlmProviderName = "ollama"

    def build_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        del schema_name, api_key
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            json_body={
                "model": runtime.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": json_schema,
                "options": {"temperature": 0.0, "num_predict": 512},
            },
        )

    def parse_response(self, payload: dict[str, Any]) -> tuple[str, TokenUsage]:
        message = payload.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise NonRetryableApiError("Ollama response missing 'message.content'.")
        in_tok = int(payload.get("prompt_eval_count", 0))
        out_tok = int(payload.get("eval_count", 0))
        return str(message["content"]), TokenUsage(
            input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok
        )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding backends."""

    @property
    def provider_name(self) -> EmbedProviderName:
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
        """Embed a batch of texts into float vectors."""
        ...


class GeminiEmbedder:
    """Google Gemini ``embedContent`` adapter."""

    provider_name: EmbedProviderName = "gemini"

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
        transport: ResilientHttpTransport,
    ) -> list[list[float]]:
        if not texts:
            return []
        key = _require_key(api_key, self.provider_name)
        requests = [
            PreparedHttpRequest(
                url=f"{runtime.base_url}/v1beta/models/{runtime.model}:embedContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                json_body={
                    "model": f"models/{runtime.model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": _GEMINI_TASK_TYPES[task],
                },
            )
            for t in texts
        ]
        responses = await asyncio.gather(*(transport.post_json(req) for req in requests))
        return [[float(v) for v in payload["embedding"]["values"]] for payload, _, _ in responses]


class OpenAIEmbedder:
    """OpenAI ``/v1/embeddings`` adapter."""

    provider_name: EmbedProviderName = "openai"

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
        transport: ResilientHttpTransport,
    ) -> list[list[float]]:
        del task
        if not texts:
            return []
        key = _require_key(api_key, self.provider_name)
        prepared = PreparedHttpRequest(
            url=f"{runtime.base_url}/embeddings",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json_body={"model": runtime.model, "input": texts, "encoding_format": "float"},
        )
        payload, _, _ = await transport.post_json(prepared)
        ordered = sorted(payload["data"], key=lambda entry: int(entry.get("index", 0)))
        return [[float(v) for v in entry["embedding"]] for entry in ordered]


class OllamaEmbedder:
    """Local Ollama ``/api/embed`` adapter."""

    provider_name: EmbedProviderName = "ollama"

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
        transport: ResilientHttpTransport,
    ) -> list[list[float]]:
        del task, api_key
        if not texts:
            return []
        prepared = PreparedHttpRequest(
            url=f"{runtime.base_url}/api/embed",
            headers={"Content-Type": "application/json"},
            json_body={"model": runtime.model, "input": texts},
        )
        payload, _, _ = await transport.post_json(prepared)
        return [[float(v) for v in vec] for vec in payload["embeddings"]]


LLM_REGISTRY: dict[LlmProviderName, StructuredLlmAdapter] = {
    "gemini": GeminiLlmAdapter(),
    "openai": OpenAILlmAdapter(),
    "anthropic": AnthropicLlmAdapter(),
    "ollama": OllamaLlmAdapter(),
}

EMBEDDER_REGISTRY: dict[EmbedProviderName, EmbeddingProvider] = {
    "gemini": GeminiEmbedder(),
    "openai": OpenAIEmbedder(),
    "ollama": OllamaEmbedder(),
}
