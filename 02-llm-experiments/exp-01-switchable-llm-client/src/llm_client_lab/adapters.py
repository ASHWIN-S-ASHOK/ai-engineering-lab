"""Provider-specific REST wire adapters behind a single Protocol."""

from typing import Any, Protocol, runtime_checkable

from pydantic import SecretStr

from llm_client_lab.models import (
    LLMRequest,
    LLMResponse,
    NonRetryableLLMError,
    PreparedHttpRequest,
    ProviderName,
    ProviderRuntimeConfig,
    TokenUsage,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Wire-protocol translation contract for a model backend."""

    @property
    def provider_name(self) -> ProviderName:
        """Identifier of the backend provider."""
        ...

    def build_request(
        self,
        request: LLMRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        """Serialize a normalized LLMRequest into provider-specific HTTP fields."""
        ...

    def parse_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> LLMResponse:
        """Normalize a provider JSON response into LLMResponse."""
        ...


def _require_key_value(api_key: SecretStr | None, provider: ProviderName) -> str:
    if api_key is None or not api_key.get_secret_value().strip():
        raise NonRetryableLLMError(f"Missing API key for provider '{provider}'.")
    return api_key.get_secret_value()


def _join_text_parts(items: list[dict[str, Any]], *, text_key: str = "text") -> str:
    return "".join(str(item[text_key]) for item in items if text_key in item)


class GeminiAdapter:
    """Google Gemini API adapter (``POST /v1beta/models/{model}:generateContent``)."""

    provider_name: ProviderName = "gemini"

    def build_request(
        self,
        request: LLMRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key_value(api_key, self.provider_name)
        gen_config: dict[str, Any] = {"maxOutputTokens": request.max_output_tokens}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": gen_config,
        }
        if request.system_prompt:
            body["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/v1beta/models/{runtime.model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json_body=body,
        )

    def parse_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> LLMResponse:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise NonRetryableLLMError(
                "Gemini response missing 'candidates'.", attempts=attempts
            )

        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])
        text = _join_text_parts(parts)
        usage_raw = payload.get("usageMetadata", {})
        input_tokens = int(usage_raw.get("promptTokenCount", 0))
        output_tokens = int(usage_raw.get("candidatesTokenCount", 0))
        total_tokens = int(usage_raw.get("totalTokenCount", input_tokens + output_tokens))

        return LLMResponse(
            provider=self.provider_name,
            model=str(payload.get("modelVersion", requested_model)),
            text=text,
            finish_reason=first.get("finishReason"),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
            attempts=attempts,
        )


class OpenAIAdapter:
    """OpenAI Responses API adapter (``POST /v1/responses``)."""

    provider_name: ProviderName = "openai"

    def build_request(
        self,
        request: LLMRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key_value(api_key, self.provider_name)
        body: dict[str, Any] = {
            "model": runtime.model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request.prompt}],
                }
            ],
            "max_output_tokens": request.max_output_tokens,
        }
        if request.system_prompt:
            body["instructions"] = request.system_prompt
        if request.temperature is not None:
            body["temperature"] = request.temperature

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/responses",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json_body=body,
        )

    def parse_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> LLMResponse:
        output_items = payload.get("output")
        if not isinstance(output_items, list):
            raise NonRetryableLLMError("OpenAI response missing 'output' list.", attempts=attempts)

        chunks: list[str] = []
        for item in output_items:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text" and "text" in part:
                    chunks.append(str(part["text"]))

        usage_raw = payload.get("usage", {})
        input_tokens = int(usage_raw.get("input_tokens", 0))
        output_tokens = int(usage_raw.get("output_tokens", 0))
        total_tokens = int(usage_raw.get("total_tokens", input_tokens + output_tokens))

        incomplete = payload.get("incomplete_details") or {}
        finish_reason = incomplete.get("reason") or payload.get("status")

        return LLMResponse(
            provider=self.provider_name,
            model=str(payload.get("model", requested_model)),
            text="".join(chunks),
            finish_reason=finish_reason,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
            attempts=attempts,
        )


class AnthropicAdapter:
    """Anthropic Messages API adapter (``POST /v1/messages``)."""

    provider_name: ProviderName = "anthropic"

    def build_request(
        self,
        request: LLMRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key_value(api_key, self.provider_name)
        body: dict[str, Any] = {
            "model": runtime.model,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
        }
        if request.system_prompt:
            body["system"] = request.system_prompt
        if request.temperature is not None:
            body["temperature"] = request.temperature

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": runtime.anthropic_version,
            },
            json_body=body,
        )

    def parse_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> LLMResponse:
        content_blocks = payload.get("content")
        if not isinstance(content_blocks, list):
            raise NonRetryableLLMError(
                "Anthropic response missing 'content' list.", attempts=attempts
            )

        text = "".join(
            str(block["text"])
            for block in content_blocks
            if block.get("type") == "text" and "text" in block
        )
        usage_raw = payload.get("usage", {})
        input_tokens = int(usage_raw.get("input_tokens", 0))
        output_tokens = int(usage_raw.get("output_tokens", 0))

        return LLMResponse(
            provider=self.provider_name,
            model=str(payload.get("model", requested_model)),
            text=text,
            finish_reason=payload.get("stop_reason"),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            latency_ms=latency_ms,
            attempts=attempts,
        )


class OllamaAdapter:
    """Local Ollama Chat API adapter (``POST /api/chat``)."""

    provider_name: ProviderName = "ollama"

    def build_request(
        self,
        request: LLMRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        del api_key  # Local Ollama does not require an API key.
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        options: dict[str, Any] = {"num_predict": request.max_output_tokens}
        if request.temperature is not None:
            options["temperature"] = request.temperature

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            json_body={
                "model": runtime.model,
                "messages": messages,
                "stream": False,
                "options": options,
            },
        )

    def parse_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> LLMResponse:
        message = payload.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise NonRetryableLLMError(
                "Ollama response missing 'message.content'.", attempts=attempts
            )

        input_tokens = int(payload.get("prompt_eval_count", 0))
        output_tokens = int(payload.get("eval_count", 0))

        return LLMResponse(
            provider=self.provider_name,
            model=str(payload.get("model", requested_model)),
            text=str(message["content"]),
            finish_reason=payload.get("done_reason"),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            latency_ms=latency_ms,
            attempts=attempts,
        )


ADAPTER_REGISTRY: dict[ProviderName, LLMProvider] = {
    "gemini": GeminiAdapter(),
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "ollama": OllamaAdapter(),
}
