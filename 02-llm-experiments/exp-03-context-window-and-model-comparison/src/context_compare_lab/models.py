"""Domain models for context budgeting, preflight counting, and comparison benchmarks."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]
OverflowPolicy = Literal["fail_fast", "drop_oldest", "drop_middle"]
NeedlePosition = Literal["start", "middle", "end"]


class ContextChunk(BaseModel):
    """Single document or conversation chunk competing for context window budget."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class PackedContext(BaseModel):
    """Result of fitting system prompt, user query, and chunks into a token budget."""

    model_config = ConfigDict(frozen=True)

    kept_chunks: list[ContextChunk]
    dropped_chunk_ids: list[str]
    estimated_input_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(ge=1)
    budget_limit_tokens: int = Field(ge=1)
    policy: OverflowPolicy


class TokenUsage(BaseModel):
    """Normalized token usage metrics."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class TokenPricing(BaseModel):
    """Estimated USD per 1M tokens for comparative cost modeling."""

    model_config = ConfigDict(frozen=True)

    input_per_million_usd: float = Field(default=0.0, ge=0.0)
    output_per_million_usd: float = Field(default=0.0, ge=0.0)


class ModelRequest(BaseModel):
    """Normalized model prompt payload."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=128, gt=0, le=8192)


class ModelResponse(BaseModel):
    """Normalized provider generation response."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    text: str
    finish_reason: str | None = None
    usage: TokenUsage
    latency_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)


class BenchmarkCaseResult(BaseModel):
    """Single needle-in-a-haystack evaluation outcome."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    position: NeedlePosition
    expected_answer: str
    response_text: str
    matched: bool
    latency_ms: float = Field(ge=0.0)
    usage: TokenUsage
    estimated_cost_usd: float = Field(ge=0.0)


class ProviderSummary(BaseModel):
    """Aggregated accuracy, latency, token, and cost summary for one provider."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    cases_run: int = Field(ge=1)
    cases_passed: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_estimated_cost_usd: float = Field(ge=0.0)


class ComparisonReport(BaseModel):
    """Complete multi-provider comparison output."""

    model_config = ConfigDict(frozen=True)

    summaries: list[ProviderSummary]
    results: list[BenchmarkCaseResult]


class ProviderRuntimeConfig(BaseModel):
    """Safe non-secret runtime settings for a resolved provider."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    base_url: str
    has_api_key: bool
    anthropic_version: str = "2023-06-01"
    ollama_num_ctx: int = 4096


class PreparedHttpRequest(BaseModel):
    """Wire-level HTTP POST request."""

    model_config = ConfigDict(frozen=True)

    url: str
    headers: dict[str, str]
    json_body: dict[str, Any]


class ContextCompareError(Exception):
    """Base error for Experiment 4."""


class ContextBudgetExceededError(ContextCompareError):
    """Raised under ``fail_fast`` policy when prompt + chunks exceed token budget."""


class RetryableApiError(ContextCompareError):
    """Transient HTTP/network error after exhausting retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class NonRetryableApiError(ContextCompareError):
    """Deterministic HTTP, credential, or payload failure."""

    def __init__(self, message: str, *, attempts: int = 1, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
