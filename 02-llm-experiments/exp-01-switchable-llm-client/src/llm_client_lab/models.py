"""Normalized request, response, token usage, and error contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]


class LLMRequest(BaseModel):
    """Provider-agnostic prompt and generation parameters."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    temperature: float | None = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=256, gt=0, le=8192)


class TokenUsage(BaseModel):
    """Normalized token counts across different provider metering schemas."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class LLMResponse(BaseModel):
    """Normalized text generation output and telemetry."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    text: str
    finish_reason: str | None = None
    usage: TokenUsage
    latency_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)


class ProviderRuntimeConfig(BaseModel):
    """Safe, non-secret runtime metadata for an active provider."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    base_url: str
    has_api_key: bool
    anthropic_version: str = "2023-06-01"


class PreparedHttpRequest(BaseModel):
    """Wire-level HTTP POST request prepared by a provider adapter."""

    model_config = ConfigDict(frozen=True)

    url: str
    headers: dict[str, str]
    json_body: dict[str, Any]


class LLMClientError(Exception):
    """Base exception for LLM client failures."""


class RetryableLLMError(LLMClientError):
    """Transient HTTP/network error (timeout, 429, 5xx) after exhausting retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class NonRetryableLLMError(LLMClientError):
    """Deterministic failure (non-429 4xx, missing credentials, or unexpected payload)."""

    def __init__(self, message: str, *, attempts: int = 1, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
