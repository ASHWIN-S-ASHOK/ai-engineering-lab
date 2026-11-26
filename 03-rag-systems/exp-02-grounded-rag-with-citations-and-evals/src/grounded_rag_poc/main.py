"""Offline & optional live Grounded RAG POC demo comparing dense vs. hybrid RRF."""

import argparse
import asyncio
import json
import logging

import httpx

from grounded_rag_poc.config import AppSettings
from grounded_rag_poc.evaluation import evaluate_rag_pipeline
from grounded_rag_poc.models import ChunkRecord, LlmProviderName, RagEvalCase
from grounded_rag_poc.providers import EMBEDDER_REGISTRY
from grounded_rag_poc.rag import GroundedRagPipeline
from grounded_rag_poc.store import SqliteHybridStore
from grounded_rag_poc.transport import ResilientHttpTransport

SAMPLE_CHUNKS: list[ChunkRecord] = [
    ChunkRecord(
        chunk_id="policy-returns#0",
        doc_id="policy-returns",
        chunk_index=0,
        title="Damaged Parcel Return Policy",
        text=(
            "Customers who receive a broken or damaged parcel may initiate a refund within "
            "14 days of delivery by uploading two photos of the outer packaging."
        ),
    ),
    ChunkRecord(
        chunk_id="policy-coldchain#0",
        doc_id="policy-coldchain",
        chunk_index=0,
        title="Cold-Chain Delivery Standard",
        text=(
            "Refrigerated groceries must stay below 4 degrees Celsius during last-mile courier "
            "handoff. Insulated liners are mandatory for routes exceeding 45 minutes."
        ),
    ),
    ChunkRecord(
        chunk_id="catalog-sku-4812#0",
        doc_id="catalog-sku-4812",
        chunk_index=0,
        title="Scanner Replacement Battery",
        text=(
            "Hardware SKU-4812-X is a 5000mAh lithium battery pack compatible with Zebra TC52 "
            "handheld warehouse barcode terminals."
        ),
    ),
]

EVAL_CASES: list[RagEvalCase] = [
    RagEvalCase(
        case_id="q-paraphrase-return",
        category="semantic_paraphrase",
        query="How do I send back a broken parcel for a refund?",
        relevant_doc_ids=["policy-returns"],
        expect_abstain=False,
    ),
    RagEvalCase(
        case_id="q-exact-sku",
        category="exact_identifier",
        query="SKU-4812-X",
        relevant_doc_ids=["catalog-sku-4812"],
        expect_abstain=False,
    ),
    RagEvalCase(
        case_id="q-unanswerable",
        category="out_of_scope",
        query="What is the maximum weight limit for drone deliveries?",
        relevant_doc_ids=[],
        expect_abstain=True,
    ),
]


def _toy_embed(text: str) -> list[float]:
    lower = text.lower()
    ret = sum(w in lower for w in ("refund", "damaged", "broken", "parcel", "send back"))
    cold = sum(w in lower for w in ("cold", "celsius", "refrigerated", "groceries"))
    hw = sum(w in lower for w in ("battery", "warehouse", "barcode", "scanner", "lithium"))
    if ret == 0 and cold == 0 and hw == 0:
        return [0.6, 0.2, 0.1, 0.5]
    return [float(ret), float(cold), float(hw), 0.1]


def _build_offline_transport() -> httpx.MockTransport:
    """Simulate Gemini embeddings + strictly grounded Gemini LLM JSON responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))

        if ":embedContent" in url:
            text = body["content"]["parts"][0]["text"]
            return httpx.Response(200, json={"embedding": {"values": _toy_embed(text)}})

        prompt = body["contents"][0]["parts"][0]["text"]
        if "Question: How do I send back" in prompt and "policy-returns#0" in prompt:
            answer = {
                "answer": "Damaged parcels qualify for a refund within 14 days with two photos.",
                "abstained": False,
                "citations": [
                    {
                        "chunk_id": "policy-returns#0",
                        "quote": "initiate a refund within 14 days of delivery",
                    }
                ],
            }
        elif "Question: SKU-4812-X" in prompt and "catalog-sku-4812#0" in prompt:
            answer = {
                "answer": "SKU-4812-X is a 5000mAh lithium battery pack for Zebra TC52 scanners.",
                "abstained": False,
                "citations": [
                    {
                        "chunk_id": "catalog-sku-4812#0",
                        "quote": "5000mAh lithium battery pack compatible with Zebra TC52",
                    }
                ],
            }
        else:
            answer = {
                "answer": "Insufficient evidence in the provided context to answer this question.",
                "abstained": True,
                "citations": [],
            }

        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(answer)}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {
                    "promptTokenCount": 95,
                    "candidatesTokenCount": 24,
                    "totalTokenCount": 119,
                },
            },
        )

    return httpx.MockTransport(handler)


async def run_poc(
    *,
    live: bool,
    llm_provider: LlmProviderName | None,
    embed_provider: str | None,
) -> None:
    """Ingest sample chunks, compare dense vs. hybrid RRF, and print grounded eval report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = (
        AppSettings()
        if live
        else AppSettings(_env_file=None, gemini_api_key="demo-gemini-key")
    )
    mock = None if live else _build_offline_transport()
    store = SqliteHybridStore(db_path=settings.sqlite_db_path, rrf_k=settings.rrf_k)

    try:
        async with ResilientHttpTransport(
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.http_max_retries,
            transport=mock,
        ) as transport:
            embed_rt, embed_key = settings.resolve_embedder(
                provider=embed_provider, require_credentials=True
            )
            embedder = EMBEDDER_REGISTRY[embed_rt.provider]  # type: ignore[index]
            doc_vecs = await embedder.embed_batch(
                [c.text for c in SAMPLE_CHUNKS],
                task="document",
                runtime=embed_rt,
                api_key=embed_key,
                transport=transport,
            )
            store.add_chunks(SAMPLE_CHUNKS, doc_vecs)

            pipeline = GroundedRagPipeline(
                settings=settings,
                store=store,
                transport=transport,
                strict_citations=False,
            )
            dense_report = await evaluate_rag_pipeline(
                pipeline=pipeline,
                cases=EVAL_CASES,
                k=1,
                mode="dense",
                llm_provider=llm_provider,
                embed_provider=embed_provider,
            )
            hybrid_report = await evaluate_rag_pipeline(
                pipeline=pipeline,
                cases=EVAL_CASES,
                k=1,
                mode="hybrid",
                llm_provider=llm_provider,
                embed_provider=embed_provider,
            )

        print("=== Dense-Only vs. Hybrid (FTS5 BM25 + Cosine RRF) Eval ===")
        print("DENSE REPORT:")
        print(dense_report.model_dump_json(indent=2))
        print("\nHYBRID REPORT:")
        print(hybrid_report.model_dump_json(indent=2))
    finally:
        store.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Grounded RAG POC with hybrid search & evals.")
    parser.add_argument("--live", action="store_true", help="Use live .env credentials.")
    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "openai", "anthropic", "ollama"],
        default=None,
    )
    parser.add_argument(
        "--embed-provider",
        choices=["gemini", "openai", "ollama"],
        default=None,
    )
    args = parser.parse_args()
    asyncio.run(
        run_poc(
            live=args.live,
            llm_provider=args.llm_provider,
            embed_provider=args.embed_provider,
        )
    )


if __name__ == "__main__":
    main()
