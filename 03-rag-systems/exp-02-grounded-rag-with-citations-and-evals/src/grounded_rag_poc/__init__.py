"""Grounded RAG Proof of Concept with hybrid retrieval, citations, and evaluations."""

from grounded_rag_poc.config import AppSettings
from grounded_rag_poc.evaluation import evaluate_rag_pipeline, recall_at_k, reciprocal_rank
from grounded_rag_poc.models import (
    ChunkRecord,
    Citation,
    Document,
    FabricatedCitationError,
    GroundedAnswer,
    RagEvalCase,
    RagEvalCaseResult,
    RagEvalReport,
    RagPocError,
    RagResponse,
    RetrievalMode,
    SearchHit,
)
from grounded_rag_poc.providers import EMBEDDER_REGISTRY, LLM_REGISTRY
from grounded_rag_poc.rag import GroundedRagPipeline, verify_citations
from grounded_rag_poc.store import Retriever, SqliteHybridStore

__all__ = [
    "AppSettings",
    "ChunkRecord",
    "Citation",
    "Document",
    "EMBEDDER_REGISTRY",
    "FabricatedCitationError",
    "GroundedAnswer",
    "GroundedRagPipeline",
    "LLM_REGISTRY",
    "RagEvalCase",
    "RagEvalCaseResult",
    "RagEvalReport",
    "RagPocError",
    "RagResponse",
    "RetrievalMode",
    "Retriever",
    "SearchHit",
    "SqliteHybridStore",
    "evaluate_rag_pipeline",
    "recall_at_k",
    "reciprocal_rank",
    "verify_citations",
]
