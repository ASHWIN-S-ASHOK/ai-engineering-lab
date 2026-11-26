"""Domain models, strict JSON Schema helpers, and RAG error hierarchy."""

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LlmProviderName = Literal["gemini", "openai", "anthropic", "ollama"]
EmbedProviderName = Literal["gemini", "openai", "ollama"]
EmbeddingTask = Literal["document", "query"]
RetrievalMode = Literal["dense", "bm25", "hybrid"]


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


class Document(BaseModel):
    """Input source document to ingest and chunk."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    title: str = ""
    text: str = Field(min_length=1)


class ChunkRecord(BaseModel):
    """Persisted document chunk with provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    title: str = ""
    text: str = Field(min_length=1)


class SearchHit(BaseModel):
    """Retrieved chunk with dense, BM25, and fused RRF scores."""

    model_config = ConfigDict(frozen=True)

    chunk: ChunkRecord
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


class Citation(BaseModel):
    """Grounding reference linking a claim to an exact retrieved chunk and quote."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class GroundedAnswer(BaseModel):
    """Structured JSON output contract enforced on the generation LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str = Field(min_length=1)
    abstained: bool
    citations: list[Citation] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Normalized generation token counts."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class RagResponse(BaseModel):
    """End-to-end RAG answer with verified citations and telemetry."""

    model_config = ConfigDict(frozen=True)

    query: str
    retrieval_mode: RetrievalMode
    llm_provider: LlmProviderName
    model: str
    answer: str
    abstained: bool
    citations: list[Citation]
    invalid_citations: list[str] = Field(default_factory=list)
    retrieved_hits: list[SearchHit]
    usage: TokenUsage
    latency_ms: float = Field(ge=0.0)


class RagEvalCase(BaseModel):
    """Labeled RAG evaluation case covering retrieval, citation, and abstention."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    category: str = "semantic"
    query: str = Field(min_length=1)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    expect_abstain: bool = False


class RagEvalCaseResult(BaseModel):
    """Per-case retrieval and faithfulness outcome."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    category: str
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    citations_valid: bool
    abstention_correct: bool


class RagEvalReport(BaseModel):
    """Aggregated retrieval + faithfulness evaluation report."""

    model_config = ConfigDict(frozen=True)

    retrieval_mode: RetrievalMode
    k: int = Field(ge=1)
    cases_run: int = Field(ge=1)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    citation_validity_rate: float = Field(ge=0.0, le=1.0)
    abstention_accuracy: float = Field(ge=0.0, le=1.0)
    results: list[RagEvalCaseResult]


class ProviderRuntimeConfig(BaseModel):
    """Safe non-secret runtime metadata for an LLM or embedding provider."""

    model_config = ConfigDict(frozen=True)

    provider: str
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


class RagPocError(Exception):
    """Base error for the Grounded RAG POC."""


class FabricatedCitationError(RagPocError):
    """Raised in strict mode when an LLM cites a missing chunk or unverified quote."""


class RetryableApiError(RagPocError):
    """Transient HTTP/network error after exhausting retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class NonRetryableApiError(RagPocError):
    """Deterministic HTTP, credential, schema, or payload failure."""

    def __init__(self, message: str, *, attempts: int = 1, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
