"""End-to-end offline evaluation tests comparing dense-only vs. hybrid RRF retrieval."""

from grounded_rag_poc.config import AppSettings
from grounded_rag_poc.evaluation import evaluate_rag_pipeline
from grounded_rag_poc.main import EVAL_CASES, SAMPLE_CHUNKS, _build_offline_transport
from grounded_rag_poc.providers import EMBEDDER_REGISTRY
from grounded_rag_poc.rag import GroundedRagPipeline
from grounded_rag_poc.store import SqliteHybridStore
from grounded_rag_poc.transport import ResilientHttpTransport


async def test_hybrid_outperforms_dense_on_sku_while_keeping_faithfulness() -> None:
    settings = AppSettings(_env_file=None, gemini_api_key="gem-key")
    store = SqliteHybridStore(":memory:", rrf_k=60)
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=_build_offline_transport(),
    )
    try:
        async with transport:
            emb_rt, emb_key = settings.resolve_embedder(require_credentials=True)
            doc_vecs = await EMBEDDER_REGISTRY["gemini"].embed_batch(
                [c.text for c in SAMPLE_CHUNKS],
                task="document",
                runtime=emb_rt,
                api_key=emb_key,
                transport=transport,
            )
            store.add_chunks(SAMPLE_CHUNKS, doc_vecs)

            pipeline = GroundedRagPipeline(
                settings=settings,
                store=store,
                transport=transport,
                strict_citations=False,
            )
            dense = await evaluate_rag_pipeline(
                pipeline=pipeline, cases=EVAL_CASES, k=1, mode="dense"
            )
            hybrid = await evaluate_rag_pipeline(
                pipeline=pipeline, cases=EVAL_CASES, k=1, mode="hybrid"
            )

        assert dense.mean_recall_at_k == 0.5
        assert hybrid.mean_recall_at_k == 1.0
        assert hybrid.citation_validity_rate == 1.0
        assert hybrid.abstention_accuracy == 1.0
    finally:
        store.close()
