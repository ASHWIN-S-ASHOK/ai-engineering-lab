"""Embeddings, chunking, pure NumPy cosine similarity, and retrieval evaluation lab."""

from embeddings_lab.chunking import chunk_fixed_overlap, chunk_paragraphs
from embeddings_lab.config import AppSettings
from embeddings_lab.embedders import EMBEDDER_REGISTRY, EmbeddingProvider
from embeddings_lab.evaluation import evaluate_retrieval, recall_at_k, reciprocal_rank
from embeddings_lab.models import (
    Document,
    EmbeddingError,
    EmbeddingProviderName,
    EmbeddingTask,
    EvalCase,
    EvalCaseResult,
    NonRetryableEmbeddingError,
    RetrievalEvalReport,
    RetryableEmbeddingError,
    SearchHit,
    TextChunk,
    UnsupportedEmbeddingProviderError,
)
from embeddings_lab.similarity import InMemoryVectorIndex, cosine_similarity_scores, l2_normalize

__all__ = [
    "AppSettings",
    "Document",
    "EMBEDDER_REGISTRY",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderName",
    "EmbeddingTask",
    "EvalCase",
    "EvalCaseResult",
    "InMemoryVectorIndex",
    "NonRetryableEmbeddingError",
    "RetrievalEvalReport",
    "RetryableEmbeddingError",
    "SearchHit",
    "TextChunk",
    "UnsupportedEmbeddingProviderError",
    "chunk_fixed_overlap",
    "chunk_paragraphs",
    "cosine_similarity_scores",
    "evaluate_retrieval",
    "l2_normalize",
    "recall_at_k",
    "reciprocal_rank",
]
