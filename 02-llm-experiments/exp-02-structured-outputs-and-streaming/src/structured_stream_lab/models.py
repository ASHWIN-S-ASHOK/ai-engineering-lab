"""Normalized models, JSON schema helper, and error hierarchy."""

from copy import deepcopy
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]
StreamFraming = Literal["sse", "ndjson"]
TModel = TypeVar("TModel", bound=BaseModel)


def _enforce_strict_objects(node: Any) -> Any:
    """Recursively ensure object schemas forbid additional properties for strict mode."""
    if isinstance(node, dict):
        updated = {key: _enforce_strict_objects(value) for key, value in node.items()}
        if updated.get("type") == "object" and "additionalProperties" not in updated:
            updated["additionalProperties"] = False
        return updated
    if isinstance(node, list):
        return [_enforce_strict_objects(item) for item in node]
    return node


def schema_from_model(
    model_cls: type[BaseModel],
    *,
    name: str | None = None,
) -> "StructuredSchemaSpec":
    """Build a strict JSON Schema specification from a Pydantic model class."""
    raw_schema = deepcopy(model_cls.model_json_schema())
    strict_schema = _enforce_strict_objects(raw_schema)
    return StructuredSchemaSpec(
        name=name or model_cls.__name__,
        json_schema=strict_schema,
    )


class StructuredSchemaSpec(BaseModel):
    """Named JSON Schema contract enforced during model decoding."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    json_schema: dict[str, Any]


class GenerationRequest(BaseModel):
    """Provider-agnostic request supporting optional structured output and streaming."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, gt=0, le=8192)
    schema_spec: StructuredSchemaSpec | None = None
    stream: bool = False


class TokenUsage(BaseModel):
    """Normalized token usage counters."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class RawGenerationResponse(BaseModel):
    """Normalized raw text generation output prior to Pydantic schema parsing."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    text: str
    finish_reason: str | None = None
    usage: TokenUsage
    latency_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)


class StructuredResult[TModel: BaseModel](BaseModel):
    """Validated Pydantic model result alongside raw wire telemetry."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    data: TModel
    raw_text: str
    finish_reason: str | None = None
    usage: TokenUsage
    latency_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)


class StreamEvent(BaseModel):
    """Normalized incremental chunk or terminal summary from a streaming call."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    delta_text: str = ""
    done: bool = False
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    ttft_ms: float | None = Field(default=None, ge=0.0)
    total_ms: float | None = Field(default=None, ge=0.0)


class ProviderRuntimeConfig(BaseModel):
    """Non-secret runtime settings for a resolved provider."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    base_url: str
    has_api_key: bool
    anthropic_version: str = "2023-06-01"


class PreparedHttpRequest(BaseModel):
    """Wire-level HTTP POST request prepared by an adapter."""

    model_config = ConfigDict(frozen=True)

    url: str
    headers: dict[str, str]
    json_body: dict[str, Any]


class StructuredStreamError(Exception):
    """Base exception for structured generation and streaming failures."""


class RetryableStreamError(StructuredStreamError):
    """Transient HTTP/network error (429, 5xx, timeout) after retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class NonRetryableStreamError(StructuredStreamError):
    """Deterministic HTTP, credential, or framing failure."""

    def __init__(self, message: str, *, attempts: int = 1, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class SchemaValidationError(StructuredStreamError):
    """Raised when model output fails JSON decoding or Pydantic validation."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.finish_reason = finish_reason
