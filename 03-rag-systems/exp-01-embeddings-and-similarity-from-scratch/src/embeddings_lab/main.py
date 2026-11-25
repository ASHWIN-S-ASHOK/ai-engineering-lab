"""Offline semantic-vs-SKU demo and optional live embedding retrieval CLI."""

import argparse
import asyncio
import logging

import numpy as np

from embeddings_lab.chunking import chunk_fixed_overlap, chunk_paragraphs
from embeddings_lab.config import AppSettings
from embeddings_lab.embedders import EMBEDDER_REGISTRY
from embeddings_lab.evaluation import evaluate_retrieval
from embeddings_lab.models import Document, EmbeddingProviderName, EvalCase, TextChunk
from embeddings_lab.similarity import InMemoryVectorIndex
from embeddings_lab.transport import ResilientHttpTransport

_CORPUS: list[Document] = [
    Document(
        doc_id="policy-returns",
        title="Damaged Parcel Return Policy",
        text=(
            "Customers who receive a broken or damaged parcel may initiate a refund within "
            "14 days of delivery by uploading two photos of the packaging."
        ),
    ),
    Document(
        doc_id="policy-coldchain",
        title="Perishable Cold-Chain Handling",
        text=(
            "Refrigerated groceries must stay below 4 degrees Celsius during last-mile courier "
            "handoff. Insulated liners are mandatory for routes exceeding 45 minutes."
        ),
    ),
    Document(
        doc_id="catalog-sku-4812",
        title="Scanner Replacement Battery Spec",
        text=(
            "Hardware SKU-4812-X is a 5000mAh lithium battery pack compatible with Zebra TC52 "
            "handheld warehouse barcode terminals."
        ),
    ),
]

_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="q-paraphrase-refund",
        category="semantic_paraphrase",
        query="How do I send back a broken delivery box and get money back?",
        relevant_doc_ids=["policy-returns"],
    ),
    EvalCase(
        case_id="q-paraphrase-cooling",
        category="semantic_paraphrase",
        query="Temperature rules for fresh food delivery vans",
        relevant_doc_ids=["policy-coldchain"],
    ),
    EvalCase(
        case_id="q-exact-sku",
        category="exact_identifier",
        query="SKU-4812-X",
        relevant_doc_ids=["catalog-sku-4812"],
    ),
]


def _synthetic_concept_vector(text: str) -> list[float]:
    """Deterministic 4D toy concept vector [returns, coldchain, hardware, generic].

    Simulates why semantic embeddings cluster paraphrases well while losing precision on
    bare alphanumeric identifiers like ``SKU-4812-X`` (no semantic keyword overlap).
    """
    lower = text.lower()
    returns_sig = sum(
        w in lower for w in ("refund", "damaged", "broken", "parcel", "send back", "money back")
    )
    cold_sig = sum(
        w in lower for w in ("cold", "celsius", "refrigerated", "temperature", "fresh food")
    )
    hw_sig = sum(w in lower for w in ("battery", "warehouse", "barcode", "scanner", "lithium"))
    if returns_sig == 0 and cold_sig == 0 and hw_sig == 0:
        return [0.6, 0.2, 0.1, 0.5]
    return [float(returns_sig), float(cold_sig), float(hw_sig), 0.1]


def run_offline_demo() -> None:
    """Demonstrate chunking overlap, NumPy cosine search, and semantic-vs-SKU behavior."""
    print("=== 1. Chunking Strategy Comparison ===")
    long_doc = Document(
        doc_id="ops-manual",
        text=(
            "Section 1: Scan inbound pallets at Dock A before staging.\n\n"
            "Section 2: Fragile electronics require corner guards and shock labels.\n\n"
            "Section 3: Hazmat manifests must be signed by the shift supervisor."
        ),
    )
    fixed = chunk_fixed_overlap(long_doc, chunk_size=75, overlap=15)
    para = chunk_paragraphs(long_doc, max_chunk_chars=90)
    print(f"Fixed-overlap chunks: {len(fixed)} | Paragraph chunks: {len(para)}")

    print("\n=== 2. Pure NumPy Cosine Retrieval (Offline Toy Concept Space) ===")
    chunks: list[TextChunk] = []
    for doc in _CORPUS:
        chunks.extend(chunk_fixed_overlap(doc, chunk_size=300, overlap=0))

    doc_vectors = np.array([_synthetic_concept_vector(c.text) for c in chunks], dtype=np.float64)
    query_vectors = np.array(
        [_synthetic_concept_vector(c.query) for c in _EVAL_CASES], dtype=np.float64
    )

    index = InMemoryVectorIndex()
    index.add(chunks, doc_vectors)
    report = evaluate_retrieval(index=index, cases=_EVAL_CASES, query_vectors=query_vectors, k=1)
    print(report.model_dump_json(indent=2))


async def run_live_demo(provider: EmbeddingProviderName | None) -> None:
    """Embed corpus and queries via a live provider (Gemini, OpenAI, or Ollama)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = AppSettings()
    runtime, api_key = settings.resolve_provider(provider=provider, require_credentials=True)
    embedder = EMBEDDER_REGISTRY[runtime.provider]

    chunks = [
        chunk
        for doc in _CORPUS
        for chunk in chunk_fixed_overlap(doc, chunk_size=300, overlap=0)
    ]
    async with ResilientHttpTransport(
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
    ) as transport:
        doc_vecs = await embedder.embed_batch(
            [c.text for c in chunks],
            task="document",
            runtime=runtime,
            api_key=api_key,
            transport=transport,
        )
        query_vecs = await embedder.embed_batch(
            [c.query for c in _EVAL_CASES],
            task="query",
            runtime=runtime,
            api_key=api_key,
            transport=transport,
        )

    index = InMemoryVectorIndex()
    index.add(chunks, doc_vecs)
    report = evaluate_retrieval(index=index, cases=_EVAL_CASES, query_vectors=query_vecs, k=1)
    print(report.model_dump_json(indent=2))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Embeddings and cosine similarity from scratch.")
    parser.add_argument("--live", action="store_true", help="Call live embedding API from .env.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "ollama"],
        default=None,
        help="Override EMBEDDING_PROVIDER.",
    )
    args = parser.parse_args()
    if args.live:
        asyncio.run(run_live_demo(args.provider))
    else:
        run_offline_demo()


if __name__ == "__main__":
    main()
