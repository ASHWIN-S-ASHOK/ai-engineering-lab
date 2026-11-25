"""Domain models and error hierarchy for chunking, embeddings, and retrieval evaluation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EmbeddingProviderName = Literal["gemini", "openai", "ollama"]
EmbeddingTask = Literal["document", "query"]


class Document(BaseModel):
    """Source document prior to chunking."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    title: str = ""
    text: str = Field(min_length=1)


class TextChunk(BaseModel):
    """Chunked segment preserving provenance back to the source document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str = Field(min_length=1)


class SearchHit(BaseModel):
    """Single top-k similarity search match."""

    model_config = ConfigDict(frozen=True)

    chunk: TextChunk
    score: float


class EvalCase(BaseModel):
    """Labeled retrieval evaluation query with expected relevant document IDs."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_doc_ids: list[str] = Field(min_length=1)
    category: str = "semantic"


class EvalCaseResult(BaseModel):
    """Observed evaluation outcome for a single retrieval query."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    category: str
    query: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    top_score: float | None = None
    recall_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)


class RetrievalEvalReport(BaseModel):
    """Aggregated retrieval metrics across a benchmark set."""

    model_config = ConfigDict(frozen=True)

    k: int = Field(ge=1)
    cases_run: int = Field(ge=1)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    results: list[EvalCaseResult]


class ProviderRuntimeConfig(BaseModel):
    """Safe non-secret runtime metadata for an embedding provider."""

    model_config = ConfigDict(frozen=True)

    provider: EmbeddingProviderName
    model: str
    base_url: str
    has_api_key: bool


class PreparedHttpRequest(BaseModel):
    """Wire-level HTTP POST request."""

    model_config = ConfigDict(frozen=True)

    url: str
    headers: dict[str, str]
    json_body: dict[str, Any]


class EmbeddingError(Exception):
    """Base exception for Track 03 embedding experiments."""


class UnsupportedEmbeddingProviderError(EmbeddingError):
    """Raised when an unsupported provider (e.g. Anthropic) is requested for embeddings."""


class RetryableEmbeddingError(EmbeddingError):
    """Transient HTTP/network error after exhausting retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class NonRetryableEmbeddingError(EmbeddingError):
    """Deterministic HTTP, credential, or payload failure."""

    def __init__(self, message: str, *, attempts: int = 1, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
