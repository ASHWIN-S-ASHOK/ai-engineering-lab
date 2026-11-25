"""Retrieval evaluation metrics (recall@k and MRR) over InMemoryVectorIndex."""

from statistics import mean

import numpy as np

from embeddings_lab.models import EvalCase, EvalCaseResult, RetrievalEvalReport
from embeddings_lab.similarity import InMemoryVectorIndex


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], *, k: int) -> float:
    """Fraction of relevant documents found in the top-k unique retrieved document IDs."""
    if not relevant_doc_ids:
        return 0.0
    top_unique = list(dict.fromkeys(retrieved_doc_ids))[:k]
    hits = sum(1 for doc_id in relevant_doc_ids if doc_id in top_unique)
    return round(hits / len(relevant_doc_ids), 4)


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """Reciprocal rank (1 / rank) of the first relevant document, or 0.0 if absent."""
    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(dict.fromkeys(retrieved_doc_ids), start=1):
        if doc_id in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def evaluate_retrieval(
    *,
    index: InMemoryVectorIndex,
    cases: list[EvalCase],
    query_vectors: list[list[float]] | np.ndarray,
    k: int = 2,
) -> RetrievalEvalReport:
    """Score top-k retrieval quality across a labeled set of evaluation queries."""
    if len(cases) != len(query_vectors):
        raise ValueError("Number of eval cases must match number of query vectors.")

    results: list[EvalCaseResult] = []
    for case, q_vec in zip(cases, query_vectors, strict=True):
        hits = index.search(q_vec, top_k=k)
        retrieved_docs = [hit.chunk.doc_id for hit in hits]
        results.append(
            EvalCaseResult(
                case_id=case.case_id,
                category=case.category,
                query=case.query,
                relevant_doc_ids=case.relevant_doc_ids,
                retrieved_doc_ids=retrieved_docs,
                top_score=hits[0].score if hits else None,
                recall_at_k=recall_at_k(retrieved_docs, case.relevant_doc_ids, k=k),
                reciprocal_rank=reciprocal_rank(retrieved_docs, case.relevant_doc_ids),
            )
        )

    return RetrievalEvalReport(
        k=k,
        cases_run=len(results),
        mean_recall_at_k=round(mean(r.recall_at_k for r in results), 4),
        mrr=round(mean(r.reciprocal_rank for r in results), 4),
        results=results,
    )
