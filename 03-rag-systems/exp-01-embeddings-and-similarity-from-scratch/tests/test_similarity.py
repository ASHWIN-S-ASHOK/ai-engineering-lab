"""Unit tests for pure NumPy normalization, cosine similarity, and vector index."""

import numpy as np
import pytest

from embeddings_lab.models import TextChunk
from embeddings_lab.similarity import InMemoryVectorIndex, cosine_similarity_scores, l2_normalize


def _chunk(idx: int) -> TextChunk:
    return TextChunk(
        chunk_id=f"d#{idx}",
        doc_id=f"d-{idx}",
        chunk_index=idx,
        start_char=0,
        end_char=5,
        text=f"chunk {idx}",
    )


def test_l2_normalize_handles_zero_vector_without_nan() -> None:
    vec = l2_normalize(np.array([0.0, 0.0, 0.0]))
    assert np.all(vec == 0.0)
    assert not np.isnan(vec).any()


def test_cosine_similarity_is_scale_invariant() -> None:
    matrix = np.array([[1.0, 0.0], [10.0, 10.0], [-1.0, 0.0]])
    scores = cosine_similarity_scores(matrix, np.array([5.0, 0.0]))
    assert pytest.approx(scores[0], abs=1e-6) == 1.0
    assert pytest.approx(scores[1], abs=1e-6) == np.sqrt(0.5)
    assert pytest.approx(scores[2], abs=1e-6) == -1.0


def test_vector_index_ranks_top_k_and_checks_dimensions() -> None:
    index = InMemoryVectorIndex()
    index.add([_chunk(0), _chunk(1), _chunk(2)], [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])

    hits = index.search([1.0, 0.1], top_k=2)
    assert [h.chunk.doc_id for h in hits] == ["d-0", "d-1"]
    assert hits[0].score > hits[1].score

    with pytest.raises(ValueError, match="Dimension mismatch"):
        index.search([1.0, 0.0, 0.0])
