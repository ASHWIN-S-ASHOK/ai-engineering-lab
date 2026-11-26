"""Normalized conversation turns, tool contracts, traces, and errors."""

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]


def _enforce_strict_objects(node: Any) -> Any:
    if isinstance(node, dict):
        updated = {k: _enforce_strict_objects(v) for k, v in node.items()}
        if updated.get("type") == "object" and "additionalProperties" not in updated:
            updated["additionalProperties"] = False
        return updated
    if isinstance(node, list):
        return [_enforce_strict_objects(item) for item in node]
    return node


def build_strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Generate a strict JSON Schema dict from a Pydantic tool-argument model."""
    return _enforce_strict_objects(deepcopy(model_cls.model_json_schema()))


class ToolSpec(BaseModel):
    """Serializable tool contract exposed to model providers."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    json_schema: dict[str, Any]


class ToolCallRequest(BaseModel):
    """Normalized model request to invoke a named tool."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    """Normalized outcome of executing one tool inside the local allow-list."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    output: dict[str, Any]
    is_error: bool = False


class UserTurn(BaseModel):
    """User message turn."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user"] = "user"
    text: str = Field(min_length=1)


class AssistantToolTurn(BaseModel):
    """Assistant turn containing requested tool calls."""

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant_tools"] = "assistant_tools"
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(min_length=1)


class ToolResultTurn(BaseModel):
    """Turn carrying tool execution outputs back to the model."""

    model_config = ConfigDict(frozen=True)

    role: Literal["tool_results"] = "tool_results"
    results: list[ToolExecutionResult] = Field(min_length=1)


class AssistantTextTurn(BaseModel):
    """Final assistant text turn."""

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant_text"] = "assistant_text"
    text: str


ConversationTurn = UserTurn | AssistantToolTurn | ToolResultTurn | AssistantTextTurn


class TokenUsage(BaseModel):
    """Normalized token counts."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ModelStepOutput(BaseModel):
    """Single LLM step output: either tool calls or final text."""

    model_config = ConfigDict(frozen=True)

    model: str
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage


class RouterTraceStep(BaseModel):
    """Auditable trace record for one step in the bounded router loop."""

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=1)
    requested_tools: list[str]
    tool_results: list[ToolExecutionResult]
    latency_ms: float = Field(ge=0.0)
    usage: TokenUsage


class RouterResult(BaseModel):
    """Final answer plus step-by-step audit trace and cumulative usage."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    answer: str
    steps_taken: int = Field(ge=1)
    trace: list[RouterTraceStep]
    total_usage: TokenUsage
    total_latency_ms: float = Field(ge=0.0)


class ProviderRuntimeConfig(BaseModel):
    """Safe non-secret runtime metadata for a resolved provider."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    base_url: str
    has_api_key: bool
    anthropic_version: str = "2023-06-01"


class PreparedHttpRequest(BaseModel):
    """Wire-level HTTP POST request."""

    model_config = ConfigDict(frozen=True)

    url: str
    headers: dict[str, str]
    json_body: dict[str, Any]


class ToolRouterError(Exception):
    """Base error for tool-calling router failures."""


class MaxStepsExceededError(ToolRouterError):
    """Raised when the model fails to reach a final answer within ``max_steps``."""

    def __init__(self, message: str, *, trace: list[RouterTraceStep]) -> None:
        super().__init__(message)
        self.trace = trace


class RetryableApiError(ToolRouterError):
    """Transient HTTP/network error after exhausting retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class NonRetryableApiError(ToolRouterError):
    """Deterministic HTTP, credential, or payload failure."""

    def __init__(self, message: str, *, attempts: int = 1, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
