"""End-to-end RAG evaluation harness (recall@k, MRR, citation validity, abstention)."""

from statistics import mean

from grounded_rag_poc.models import (
    LlmProviderName,
    RagEvalCase,
    RagEvalCaseResult,
    RagEvalReport,
    RetrievalMode,
)
from grounded_rag_poc.rag import GroundedRagPipeline


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], *, k: int) -> float:
    """Fraction of relevant docs appearing in top-k unique retrieved doc IDs."""
    if not relevant_doc_ids:
        return 0.0
    top_unique = list(dict.fromkeys(retrieved_doc_ids))[:k]
    hits = sum(1 for doc_id in relevant_doc_ids if doc_id in top_unique)
    return round(hits / len(relevant_doc_ids), 4)


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """Reciprocal rank (1 / rank) of first relevant doc, or 0.0 if absent."""
    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(dict.fromkeys(retrieved_doc_ids), start=1):
        if doc_id in relevant:
            return round(1.0 / rank, 4)
    return 0.0


async def evaluate_rag_pipeline(
    *,
    pipeline: GroundedRagPipeline,
    cases: list[RagEvalCase],
    k: int = 2,
    mode: RetrievalMode = "hybrid",
    llm_provider: LlmProviderName | None = None,
    embed_provider: str | None = None,
) -> RagEvalReport:
    """Run labeled cases through ``pipeline`` and score retrieval + faithfulness."""
    results: list[RagEvalCaseResult] = []
    recalls: list[float] = []
    rrs: list[float] = []

    for case in cases:
        resp = await pipeline.answer(
            case.query,
            top_k=k,
            mode=mode,
            llm_provider=llm_provider,
            embed_provider=embed_provider,
        )
        retrieved_docs = [hit.chunk.doc_id for hit in resp.retrieved_hits]

        case_recall: float | None = None
        case_rr: float | None = None
        if case.relevant_doc_ids:
            case_recall = recall_at_k(retrieved_docs, case.relevant_doc_ids, k=k)
            case_rr = reciprocal_rank(retrieved_docs, case.relevant_doc_ids)
            recalls.append(case_recall)
            rrs.append(case_rr)

        citations_valid = not resp.invalid_citations and (
            resp.abstained or len(resp.citations) > 0
        )
        abstention_correct = resp.abstained == case.expect_abstain

        results.append(
            RagEvalCaseResult(
                case_id=case.case_id,
                category=case.category,
                recall_at_k=case_recall,
                reciprocal_rank=case_rr,
                citations_valid=citations_valid,
                abstention_correct=abstention_correct,
            )
        )

    return RagEvalReport(
        retrieval_mode=mode,
        k=k,
        cases_run=len(results),
        mean_recall_at_k=round(mean(recalls), 4) if recalls else 0.0,
        mrr=round(mean(rrs), 4) if rrs else 0.0,
        citation_validity_rate=round(mean(1.0 if r.citations_valid else 0.0 for r in results), 4),
        abstention_accuracy=round(mean(1.0 if r.abstention_correct else 0.0 for r in results), 4),
        results=results,
    )
