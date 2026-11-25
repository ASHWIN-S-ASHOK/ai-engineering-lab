"""Domain models and explicit error hierarchy for the async API lab."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]


class ProviderSpec(BaseModel):
    """Static metadata describing how a supported LLM provider authenticates."""

    model_config = ConfigDict(frozen=True)

    name: ProviderName
    requires_api_key: bool
    auth_header: str | None = None
    auth_prefix: str = ""
    extra_headers: dict[str, str] = Field(default_factory=dict)


class ResolvedProviderConfig(BaseModel):
    """Non-sensitive view of the active provider configuration."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    base_url: str
    has_api_key: bool


class FetchResult(BaseModel):
    """Normalized HTTP JSON fetch response with latency and retry telemetry."""

    model_config = ConfigDict(frozen=True)

    url: str
    status_code: int
    latency_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)
    payload: dict[str, Any]


class ApiClientError(Exception):
    """Base exception for HTTP client failures."""


class RetryableApiError(ApiClientError):
    """Raised when a transient error (timeout, 429, 5xx) persists past all retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class NonRetryableApiError(ApiClientError):
    """Raised on deterministic failures (non-429 4xx or malformed JSON payloads)."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
