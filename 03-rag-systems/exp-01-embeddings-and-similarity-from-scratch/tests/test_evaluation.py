"""Unit tests for recall@k, MRR, and semantic vs exact-identifier diagnostics."""

import numpy as np

from embeddings_lab.chunking import chunk_fixed_overlap
from embeddings_lab.evaluation import evaluate_retrieval, recall_at_k, reciprocal_rank
from embeddings_lab.main import _CORPUS, _EVAL_CASES, _synthetic_concept_vector
from embeddings_lab.similarity import InMemoryVectorIndex


def test_recall_and_mrr_metrics() -> None:
    retrieved = ["doc-b", "doc-a", "doc-c"]
    assert recall_at_k(retrieved, ["doc-a", "doc-d"], k=2) == 0.5
    assert reciprocal_rank(retrieved, ["doc-a"]) == 0.5
    assert reciprocal_rank(retrieved, ["missing"]) == 0.0


def test_offline_eval_surfaces_semantic_win_vs_sku_miss() -> None:
    chunks = [
        chunk
        for doc in _CORPUS
        for chunk in chunk_fixed_overlap(doc, chunk_size=300, overlap=0)
    ]
    doc_vecs = np.array([_synthetic_concept_vector(c.text) for c in chunks], dtype=np.float64)
    query_vecs = np.array(
        [_synthetic_concept_vector(c.query) for c in _EVAL_CASES], dtype=np.float64
    )

    index = InMemoryVectorIndex()
    index.add(chunks, doc_vecs)
    report = evaluate_retrieval(index=index, cases=_EVAL_CASES, query_vectors=query_vecs, k=1)

    by_id = {r.case_id: r for r in report.results}
    assert by_id["q-paraphrase-refund"].recall_at_k == 1.0
    assert by_id["q-paraphrase-cooling"].recall_at_k == 1.0
    assert by_id["q-exact-sku"].recall_at_k == 0.0
    assert report.mean_recall_at_k == round(2 / 3, 4)
