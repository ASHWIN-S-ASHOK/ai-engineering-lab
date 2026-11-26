"""Unit tests for GroundedRagPipeline citation verification and abstention."""

import json

import httpx
import pytest

from grounded_rag_poc.config import AppSettings
from grounded_rag_poc.models import ChunkRecord, Citation, FabricatedCitationError
from grounded_rag_poc.rag import GroundedRagPipeline, verify_citations
from grounded_rag_poc.store import SqliteHybridStore
from grounded_rag_poc.transport import ResilientHttpTransport


def _store_with_sample() -> SqliteHybridStore:
    store = SqliteHybridStore(":memory:")
    store.add_chunks(
        [
            ChunkRecord(
                chunk_id="c1",
                doc_id="doc-1",
                chunk_index=0,
                title="Policy",
                text="Refunds require two photos of damaged packaging within 14 days.",
            )
        ],
        [[1.0, 0.0]],
    )
    return store


def test_verify_citations_rejects_unknown_chunk_and_bad_quote() -> None:
    store = _store_with_sample()
    try:
        hits = store.search(query_text="refund", query_vector=[1.0, 0.0], top_k=1)
        valid, invalid = verify_citations(
            citations=[
                Citation(chunk_id="c1", quote="two photos of damaged packaging"),
                Citation(chunk_id="missing", quote="anything"),
                Citation(chunk_id="c1", quote="nonexistent quote text"),
            ],
            hits=hits,
        )
        assert len(valid) == 1
        assert len(invalid) == 2
    finally:
        store.close()


async def test_pipeline_raises_on_fabricated_citation_in_strict_mode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if ":embedContent" in str(request.url):
            return httpx.Response(200, json={"embedding": {"values": [1.0, 0.0]}})
        bad_payload = {
            "answer": "Refunds take 30 days.",
            "abstained": False,
            "citations": [{"chunk_id": "fake-chunk-99", "quote": "Refunds take 30 days."}],
        }
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": json.dumps(bad_payload)}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10},
            },
        )

    store = _store_with_sample()
    settings = AppSettings(_env_file=None, gemini_api_key="gem-key")
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        async with transport:
            pipeline = GroundedRagPipeline(
                settings=settings,
                store=store,
                transport=transport,
                strict_citations=True,
            )
            with pytest.raises(FabricatedCitationError, match="fake-chunk-99"):
                await pipeline.answer("How long for refunds?")
    finally:
        store.close()


async def test_pipeline_accepts_verified_citations_and_abstention() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if ":embedContent" in str(request.url):
            return httpx.Response(200, json={"embedding": {"values": [1.0, 0.0]}})
        body = json.loads(request.content.decode("utf-8"))
        prompt = body["contents"][0]["parts"][0]["text"]
        if "drone" in prompt:
            payload = {
                "answer": "Insufficient evidence in context.",
                "abstained": True,
                "citations": [],
            }
        else:
            payload = {
                "answer": "Upload two photos within 14 days.",
                "abstained": False,
                "citations": [{"chunk_id": "c1", "quote": "two photos of damaged packaging"}],
            }
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(payload)}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 12},
            },
        )

    store = _store_with_sample()
    settings = AppSettings(_env_file=None, gemini_api_key="gem-key")
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        async with transport:
            pipeline = GroundedRagPipeline(
                settings=settings,
                store=store,
                transport=transport,
                strict_citations=True,
            )
            grounded = await pipeline.answer("How do refunds work?")
            abstained = await pipeline.answer("What is the drone weight limit?")

        assert grounded.abstained is False
        assert len(grounded.citations) == 1
        assert abstained.abstained is True
        assert abstained.citations == []
    finally:
        store.close()
