"""Switchable adapters for generation and preflight token counting."""

from typing import Any, Protocol, runtime_checkable

from pydantic import SecretStr

from context_compare_lab.models import (
    ModelRequest,
    ModelResponse,
    NonRetryableApiError,
    PreparedHttpRequest,
    ProviderName,
    ProviderRuntimeConfig,
    TokenUsage,
)


@runtime_checkable
class ContextModelAdapter(Protocol):
    """Protocol for generation + optional preflight token counting per provider."""

    @property
    def provider_name(self) -> ProviderName:
        """Identifier of the backend provider."""
        ...

    def build_generate_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        """Serialize a generation request."""
        ...

    def parse_generate_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> ModelResponse:
        """Parse provider response into normalized ModelResponse."""
        ...

    def build_count_tokens_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest | None:
        """Return a preflight token-count HTTP request, or None if unsupported."""
        ...

    def parse_count_tokens_response(self, payload: dict[str, Any]) -> int:
        """Extract exact input token count from preflight response."""
        ...


def _require_key(api_key: SecretStr | None, provider: ProviderName) -> str:
    if api_key is None or not api_key.get_secret_value().strip():
        raise NonRetryableApiError(f"Missing API key for provider '{provider}'.")
    return api_key.get_secret_value()


class GeminiAdapter:
    """Gemini adapter supporting ``:generateContent`` and ``:countTokens``."""

    provider_name: ProviderName = "gemini"

    def build_generate_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        gen_cfg: dict[str, Any] = {"maxOutputTokens": request.max_output_tokens}
        if request.temperature is not None:
            gen_cfg["temperature"] = request.temperature
        body = self._base_payload(request)
        body["generationConfig"] = gen_cfg
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/v1beta/models/{runtime.model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json_body=body,
        )

    def parse_generate_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> ModelResponse:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise NonRetryableApiError("Gemini response missing 'candidates'.", attempts=attempts)
        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])
        text = "".join(str(p["text"]) for p in parts if "text" in p)
        usage_raw = payload.get("usageMetadata", {})
        in_tok = int(usage_raw.get("promptTokenCount", 0))
        out_tok = int(usage_raw.get("candidatesTokenCount", 0))
        tot_tok = int(usage_raw.get("totalTokenCount", in_tok + out_tok))
        return ModelResponse(
            provider=self.provider_name,
            model=str(payload.get("modelVersion", requested_model)),
            text=text,
            finish_reason=first.get("finishReason"),
            usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok),
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def build_count_tokens_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/v1beta/models/{runtime.model}:countTokens",
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json_body=self._base_payload(request),
        )

    def parse_count_tokens_response(self, payload: dict[str, Any]) -> int:
        if "totalTokens" not in payload:
            raise NonRetryableApiError("Gemini countTokens response missing 'totalTokens'.")
        return int(payload["totalTokens"])

    @staticmethod
    def _base_payload(request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": request.prompt}]}]}
        if request.system_prompt:
            body["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        return body


class OpenAIAdapter:
    """OpenAI Responses API adapter."""

    provider_name: ProviderName = "openai"

    def build_generate_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        body: dict[str, Any] = {
            "model": runtime.model,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": request.prompt}]}
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

    def parse_generate_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> ModelResponse:
        output_items = payload.get("output")
        if not isinstance(output_items, list):
            raise NonRetryableApiError("OpenAI response missing 'output' list.", attempts=attempts)
        text = "".join(
            str(part["text"])
            for item in output_items
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text" and "text" in part
        )
        usage_raw = payload.get("usage", {})
        in_tok = int(usage_raw.get("input_tokens", 0))
        out_tok = int(usage_raw.get("output_tokens", 0))
        tot_tok = int(usage_raw.get("total_tokens", in_tok + out_tok))
        incomplete = payload.get("incomplete_details") or {}
        return ModelResponse(
            provider=self.provider_name,
            model=str(payload.get("model", requested_model)),
            text=text,
            finish_reason=incomplete.get("reason") or payload.get("status"),
            usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok),
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def build_count_tokens_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest | None:
        del request, runtime, api_key
        return None

    def parse_count_tokens_response(self, payload: dict[str, Any]) -> int:
        del payload
        raise NonRetryableApiError("Preflight token counting not configured for OpenAI.")


class AnthropicAdapter:
    """Anthropic Messages adapter supporting ``/v1/messages`` and ``/v1/messages/count_tokens``."""

    provider_name: ProviderName = "anthropic"

    def build_generate_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
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

    def parse_generate_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> ModelResponse:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise NonRetryableApiError(
                "Anthropic response missing 'content' list.", attempts=attempts
            )
        text = "".join(
            str(b["text"]) for b in blocks if b.get("type") == "text" and "text" in b
        )
        usage_raw = payload.get("usage", {})
        in_tok = int(usage_raw.get("input_tokens", 0))
        out_tok = int(usage_raw.get("output_tokens", 0))
        return ModelResponse(
            provider=self.provider_name,
            model=str(payload.get("model", requested_model)),
            text=text,
            finish_reason=payload.get("stop_reason"),
            usage=TokenUsage(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            ),
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def build_count_tokens_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        body: dict[str, Any] = {
            "model": runtime.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            body["system"] = request.system_prompt
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/messages/count_tokens",
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": runtime.anthropic_version,
            },
            json_body=body,
        )

    def parse_count_tokens_response(self, payload: dict[str, Any]) -> int:
        if "input_tokens" not in payload:
            raise NonRetryableApiError("Anthropic count_tokens missing 'input_tokens'.")
        return int(payload["input_tokens"])


class OllamaAdapter:
    """Local Ollama adapter configuring ``options.num_ctx`` explicitly."""

    provider_name: ProviderName = "ollama"

    def build_generate_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        del api_key
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        options: dict[str, Any] = {
            "num_ctx": runtime.ollama_num_ctx,
            "num_predict": request.max_output_tokens,
        }
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

    def parse_generate_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> ModelResponse:
        message = payload.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise NonRetryableApiError(
                "Ollama response missing 'message.content'.", attempts=attempts
            )
        in_tok = int(payload.get("prompt_eval_count", 0))
        out_tok = int(payload.get("eval_count", 0))
        return ModelResponse(
            provider=self.provider_name,
            model=str(payload.get("model", requested_model)),
            text=str(message["content"]),
            finish_reason=payload.get("done_reason"),
            usage=TokenUsage(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            ),
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def build_count_tokens_request(
        self,
        request: ModelRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest | None:
        del request, runtime, api_key
        return None

    def parse_count_tokens_response(self, payload: dict[str, Any]) -> int:
        del payload
        raise NonRetryableApiError("Preflight token counting not configured for Ollama.")


ADAPTER_REGISTRY: dict[ProviderName, ContextModelAdapter] = {
    "gemini": GeminiAdapter(),
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "ollama": OllamaAdapter(),
}
