"""Pure NumPy vector normalization, cosine similarity, and top-k index."""

import numpy as np

from embeddings_lab.models import SearchHit, TextChunk


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of a 1D or 2D float array without NaNs on zero vectors."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return np.zeros_like(arr) if norm == 0.0 else arr / norm
    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        normalized = arr / safe_norms
        return np.where(norms == 0.0, 0.0, normalized)
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")


def cosine_similarity_scores(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row in ``matrix`` and ``query_vector``."""
    mat = np.asarray(matrix, dtype=np.float64)
    query = np.asarray(query_vector, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {mat.ndim}D.")
    if query.ndim != 1:
        raise ValueError(f"Expected 1D query vector, got {query.ndim}D.")
    if mat.shape[1] != query.shape[0]:
        raise ValueError(
            f"Dimension mismatch: matrix has {mat.shape[1]}, query has {query.shape[0]}."
        )
    return l2_normalize(mat) @ l2_normalize(query)


class InMemoryVectorIndex:
    """Minimal NumPy-backed dense vector store for understanding retrieval mechanics."""

    def __init__(self) -> None:
        self._chunks: list[TextChunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[TextChunk], vectors: list[list[float]] | np.ndarray) -> None:
        """Append ``chunks`` and their corresponding dense ``vectors`` to the index."""
        if not chunks:
            return
        mat = np.asarray(vectors, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != len(chunks):
            raise ValueError(
                f"Expected ({len(chunks)}, D) vectors, got shape {mat.shape}."
            )
        normalized = l2_normalize(mat)
        if self._matrix is None:
            self._matrix = normalized
        else:
            if normalized.shape[1] != self._matrix.shape[1]:
                raise ValueError(
                    f"Dimension mismatch: index uses {self._matrix.shape[1]}, "
                    f"got {normalized.shape[1]}."
                )
            self._matrix = np.vstack([self._matrix, normalized])
        self._chunks.extend(chunks)

    def search(self, query_vector: list[float] | np.ndarray, *, top_k: int = 3) -> list[SearchHit]:
        """Return top-k chunks ordered by descending cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self._matrix is None or not self._chunks:
            return []

        query = np.asarray(query_vector, dtype=np.float64)
        scores = cosine_similarity_scores(self._matrix, query)
        limit = min(top_k, len(self._chunks))
        ranked_indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            SearchHit(chunk=self._chunks[int(idx)], score=round(float(scores[int(idx)]), 6))
            for idx in ranked_indices
        ]
