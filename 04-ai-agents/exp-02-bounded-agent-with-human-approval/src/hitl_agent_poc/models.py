"""Serializable state machine, approval contracts, audit log, and error hierarchy."""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]
RiskTier = Literal["read_only", "mutating"]
RunStatus = Literal["RUNNING", "PENDING_APPROVAL", "COMPLETED", "FAILED"]


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
    """Generate a strict JSON Schema dict from a Pydantic model."""
    return _enforce_strict_objects(deepcopy(model_cls.model_json_schema()))


class ToolSpec(BaseModel):
    """Serializable tool definition exposed to the LLM."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    risk_tier: RiskTier
    json_schema: dict[str, Any]


class ToolCallRequest(BaseModel):
    """Normalized model intent to invoke a tool."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    """Normalized outcome of validating/executing or rejecting a tool call."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    output: dict[str, Any]
    is_error: bool = False


class UserTurn(BaseModel):
    """Initial user prompt turn."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user"] = "user"
    text: str = Field(min_length=1)


class AssistantToolTurn(BaseModel):
    """Assistant turn requesting one or more tool calls."""

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant_tools"] = "assistant_tools"
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(min_length=1)


class ToolResultTurn(BaseModel):
    """Turn returning results for all tool calls in the preceding AssistantToolTurn."""

    model_config = ConfigDict(frozen=True)

    role: Literal["tool_results"] = "tool_results"
    results: list[ToolExecutionResult] = Field(min_length=1)


class AssistantTextTurn(BaseModel):
    """Assistant text turn."""

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant_text"] = "assistant_text"
    text: str


ConversationTurn = UserTurn | AssistantToolTurn | ToolResultTurn | AssistantTextTurn


class TokenUsage(BaseModel):
    """Normalized token usage metrics."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ModelStepOutput(BaseModel):
    """Single LLM step output."""

    model_config = ConfigDict(frozen=True)

    model: str
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage


class PendingApproval(BaseModel):
    """Paused mutating tool call awaiting human authorization."""

    model_config = ConfigDict(frozen=True)

    action_id: str = Field(min_length=1)
    tool_call: ToolCallRequest
    risk_tier: RiskTier = "mutating"
    summary: str = Field(min_length=1)


class ApprovalDecision(BaseModel):
    """Explicit human authorization or denial for a pending mutating action."""

    model_config = ConfigDict(frozen=True)

    action_id: str = Field(min_length=1)
    approved: bool
    reviewer: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class AuditEntry(BaseModel):
    """Immutable audit log record for every execution, pause, approval, or rejection."""

    model_config = ConfigDict(frozen=True)

    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    step: int = Field(ge=1)
    event: str = Field(min_length=1)
    detail: dict[str, Any] = Field(default_factory=dict)


class AgentRunState(BaseModel):
    """Serializable state snapshot enabling pause/resume across HTTP or mobile workflows."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    provider: ProviderName
    model: str
    status: RunStatus
    step_count: int = Field(default=0, ge=0)
    max_steps: int = Field(ge=1)
    system_prompt: str
    turns: list[ConversationTurn]
    pending_approval: PendingApproval | None = None
    queued_calls: list[ToolCallRequest] = Field(default_factory=list)
    completed_turn_results: list[ToolExecutionResult] = Field(default_factory=list)
    final_answer: str | None = None
    audit_log: list[AuditEntry] = Field(default_factory=list)
    total_usage: TokenUsage = Field(default_factory=TokenUsage)


class ProviderRuntimeConfig(BaseModel):
    """Safe non-secret runtime metadata."""

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


class HitlAgentError(Exception):
    """Base error for HITL agent failures."""


class ApprovalMismatchError(HitlAgentError):
    """Raised when resuming a run with an invalid state or mismatched ``action_id``."""


class MaxStepsExceededError(HitlAgentError):
    """Raised when an agent exceeds ``max_steps`` without completing."""

    def __init__(self, message: str, *, state: AgentRunState) -> None:
        super().__init__(message)
        self.state = state


class RetryableApiError(HitlAgentError):
    """Transient HTTP/network error after exhausting retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class NonRetryableApiError(HitlAgentError):
    """Deterministic HTTP, credential, or payload failure."""

    def __init__(self, message: str, *, attempts: int = 1, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
