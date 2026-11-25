"""Provider-specific structured output and streaming wire adapters."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import SecretStr

from structured_stream_lab.models import (
    GenerationRequest,
    NonRetryableStreamError,
    PreparedHttpRequest,
    ProviderName,
    ProviderRuntimeConfig,
    RawGenerationResponse,
    StreamFraming,
    TokenUsage,
)


@dataclass(frozen=True)
class ParsedStreamChunk:
    """Normalized incremental chunk extracted from one SSE or NDJSON event."""

    delta_text: str = ""
    done: bool = False
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@runtime_checkable
class StructuredStreamProvider(Protocol):
    """Protocol for adapters supporting JSON Schema output and streaming."""

    @property
    def provider_name(self) -> ProviderName:
        """Identifier of the backend provider."""
        ...

    @property
    def stream_framing(self) -> StreamFraming:
        """Wire framing used by this provider (``sse`` or ``ndjson``)."""
        ...

    def build_request(
        self,
        request: GenerationRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        """Serialize request into provider-specific URL, headers, and JSON body."""
        ...

    def parse_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> RawGenerationResponse:
        """Parse a non-streaming response payload into RawGenerationResponse."""
        ...

    def parse_stream_event(self, event: dict[str, Any]) -> ParsedStreamChunk:
        """Parse one decoded SSE or NDJSON object into a normalized chunk."""
        ...


def _require_key(api_key: SecretStr | None, provider: ProviderName) -> str:
    if api_key is None or not api_key.get_secret_value().strip():
        raise NonRetryableStreamError(f"Missing API key for provider '{provider}'.")
    return api_key.get_secret_value()


def _join_gemini_parts(parts: list[dict[str, Any]]) -> str:
    return "".join(str(part["text"]) for part in parts if "text" in part)


class GeminiAdapter:
    """Gemini REST adapter supporting ``responseJsonSchema`` and ``streamGenerateContent``."""

    provider_name: ProviderName = "gemini"
    stream_framing: StreamFraming = "sse"

    def build_request(
        self,
        request: GenerationRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        gen_config: dict[str, Any] = {"maxOutputTokens": request.max_output_tokens}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.schema_spec is not None:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseJsonSchema"] = request.schema_spec.json_schema

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": gen_config,
        }
        if request.system_prompt:
            body["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}

        action = "streamGenerateContent?alt=sse" if request.stream else "generateContent"
        return PreparedHttpRequest(
            url=f"{runtime.base_url}/v1beta/models/{runtime.model}:{action}",
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
    ) -> RawGenerationResponse:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise NonRetryableStreamError(
                "Gemini response missing 'candidates'.", attempts=attempts
            )

        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])
        usage_raw = payload.get("usageMetadata", {})
        in_tok = int(usage_raw.get("promptTokenCount", 0))
        out_tok = int(usage_raw.get("candidatesTokenCount", 0))
        tot_tok = int(usage_raw.get("totalTokenCount", in_tok + out_tok))

        return RawGenerationResponse(
            provider=self.provider_name,
            model=str(payload.get("modelVersion", requested_model)),
            text=_join_gemini_parts(parts),
            finish_reason=first.get("finishReason"),
            usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok),
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def parse_stream_event(self, event: dict[str, Any]) -> ParsedStreamChunk:
        candidates = event.get("candidates") or []
        delta_text = ""
        finish_reason: str | None = None
        if candidates:
            first = candidates[0]
            parts = first.get("content", {}).get("parts", [])
            delta_text = _join_gemini_parts(parts)
            finish_reason = first.get("finishReason")

        usage_raw = event.get("usageMetadata") or {}
        in_tok = usage_raw.get("promptTokenCount")
        out_tok = usage_raw.get("candidatesTokenCount")
        tot_tok = usage_raw.get("totalTokenCount")

        return ParsedStreamChunk(
            delta_text=delta_text,
            done=finish_reason is not None,
            finish_reason=finish_reason,
            input_tokens=int(in_tok) if in_tok is not None else None,
            output_tokens=int(out_tok) if out_tok is not None else None,
            total_tokens=int(tot_tok) if tot_tok is not None else None,
        )


class OpenAIAdapter:
    """OpenAI Responses API adapter supporting ``text.format`` strict JSON Schema & SSE."""

    provider_name: ProviderName = "openai"
    stream_framing: StreamFraming = "sse"

    def build_request(
        self,
        request: GenerationRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        body: dict[str, Any] = {
            "model": runtime.model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request.prompt}],
                }
            ],
            "max_output_tokens": request.max_output_tokens,
            "stream": request.stream,
        }
        if request.system_prompt:
            body["instructions"] = request.system_prompt
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.schema_spec is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_spec.name,
                    "strict": True,
                    "schema": request.schema_spec.json_schema,
                }
            }

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
    ) -> RawGenerationResponse:
        output_items = payload.get("output")
        if not isinstance(output_items, list):
            raise NonRetryableStreamError(
                "OpenAI response missing 'output' list.", attempts=attempts
            )

        chunks = [
            str(part["text"])
            for item in output_items
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text" and "text" in part
        ]
        usage_raw = payload.get("usage", {})
        in_tok = int(usage_raw.get("input_tokens", 0))
        out_tok = int(usage_raw.get("output_tokens", 0))
        tot_tok = int(usage_raw.get("total_tokens", in_tok + out_tok))
        incomplete = payload.get("incomplete_details") or {}

        return RawGenerationResponse(
            provider=self.provider_name,
            model=str(payload.get("model", requested_model)),
            text="".join(chunks),
            finish_reason=incomplete.get("reason") or payload.get("status"),
            usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok),
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def parse_stream_event(self, event: dict[str, Any]) -> ParsedStreamChunk:
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            return ParsedStreamChunk(delta_text=str(event.get("delta", "")))

        if event_type in {"response.completed", "response.incomplete"}:
            resp = event.get("response") or {}
            usage = resp.get("usage") or {}
            incomplete = resp.get("incomplete_details") or {}
            in_tok = usage.get("input_tokens")
            out_tok = usage.get("output_tokens")
            tot_tok = usage.get("total_tokens")
            return ParsedStreamChunk(
                done=True,
                finish_reason=incomplete.get("reason") or resp.get("status") or event_type,
                input_tokens=int(in_tok) if in_tok is not None else None,
                output_tokens=int(out_tok) if out_tok is not None else None,
                total_tokens=int(tot_tok) if tot_tok is not None else None,
            )

        return ParsedStreamChunk()


class AnthropicAdapter:
    """Anthropic Messages API adapter supporting ``output_config.format`` & SSE lifecycle."""

    provider_name: ProviderName = "anthropic"
    stream_framing: StreamFraming = "sse"

    def build_request(
        self,
        request: GenerationRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        body: dict[str, Any] = {
            "model": runtime.model,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": request.stream,
        }
        if request.system_prompt:
            body["system"] = request.system_prompt
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.schema_spec is not None:
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": request.schema_spec.json_schema,
                }
            }

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
    ) -> RawGenerationResponse:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise NonRetryableStreamError(
                "Anthropic response missing 'content' list.", attempts=attempts
            )

        text = "".join(
            str(block["text"])
            for block in blocks
            if block.get("type") == "text" and "text" in block
        )
        usage_raw = payload.get("usage", {})
        in_tok = int(usage_raw.get("input_tokens", 0))
        out_tok = int(usage_raw.get("output_tokens", 0))

        return RawGenerationResponse(
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

    def parse_stream_event(self, event: dict[str, Any]) -> ParsedStreamChunk:
        event_type = event.get("type")
        if event_type == "message_start":
            msg_usage = (event.get("message") or {}).get("usage") or {}
            in_tok = msg_usage.get("input_tokens")
            return ParsedStreamChunk(input_tokens=int(in_tok) if in_tok is not None else None)

        if event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                return ParsedStreamChunk(delta_text=str(delta.get("text", "")))

        if event_type == "message_delta":
            delta = event.get("delta") or {}
            usage = event.get("usage") or {}
            out_tok = usage.get("output_tokens")
            return ParsedStreamChunk(
                finish_reason=delta.get("stop_reason"),
                output_tokens=int(out_tok) if out_tok is not None else None,
            )

        if event_type == "message_stop":
            return ParsedStreamChunk(done=True)

        return ParsedStreamChunk()


class OllamaAdapter:
    """Local Ollama adapter supporting ``format`` JSON Schema and NDJSON streaming."""

    provider_name: ProviderName = "ollama"
    stream_framing: StreamFraming = "ndjson"

    def build_request(
        self,
        request: GenerationRequest,
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        del api_key
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        options: dict[str, Any] = {"num_predict": request.max_output_tokens}
        if request.temperature is not None:
            options["temperature"] = request.temperature

        body: dict[str, Any] = {
            "model": runtime.model,
            "messages": messages,
            "stream": request.stream,
            "options": options,
        }
        if request.schema_spec is not None:
            body["format"] = request.schema_spec.json_schema

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            json_body=body,
        )

    def parse_response(
        self,
        payload: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
        attempts: int,
    ) -> RawGenerationResponse:
        message = payload.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise NonRetryableStreamError(
                "Ollama response missing 'message.content'.", attempts=attempts
            )

        in_tok = int(payload.get("prompt_eval_count", 0))
        out_tok = int(payload.get("eval_count", 0))
        return RawGenerationResponse(
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

    def parse_stream_event(self, event: dict[str, Any]) -> ParsedStreamChunk:
        content = str((event.get("message") or {}).get("content", ""))
        done = bool(event.get("done", False))
        in_tok = event.get("prompt_eval_count")
        out_tok = event.get("eval_count")
        return ParsedStreamChunk(
            delta_text=content,
            done=done,
            finish_reason=event.get("done_reason"),
            input_tokens=int(in_tok) if in_tok is not None else None,
            output_tokens=int(out_tok) if out_tok is not None else None,
        )


ADAPTER_REGISTRY: dict[ProviderName, StructuredStreamProvider] = {
    "gemini": GeminiAdapter(),
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "ollama": OllamaAdapter(),
}
