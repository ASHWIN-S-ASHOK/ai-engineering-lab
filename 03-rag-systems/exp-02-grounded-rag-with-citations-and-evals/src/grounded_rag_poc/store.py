"""SQLite FTS5 + NumPy dense cosine hybrid retriever using Reciprocal Rank Fusion."""

import json
import re
import sqlite3
from typing import Protocol, runtime_checkable

import numpy as np

from grounded_rag_poc.models import ChunkRecord, RetrievalMode, SearchHit


@runtime_checkable
class Retriever(Protocol):
    """Persistence and retrieval contract isolating storage details from RAG orchestration."""

    def add_chunks(
        self,
        chunks: list[ChunkRecord],
        vectors: list[list[float]] | np.ndarray,
    ) -> None:
        """Persist chunks and their dense embeddings."""
        ...

    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float] | np.ndarray,
        top_k: int = 3,
        mode: RetrievalMode = "hybrid",
    ) -> list[SearchHit]:
        """Return top-k ranked chunks using ``dense``, ``bm25``, or ``hybrid`` (RRF)."""
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of a 1D or 2D array, guarding zero-norm vectors."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return np.zeros_like(arr) if norm == 0.0 else arr / norm
    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe = np.where(norms == 0.0, 1.0, norms)
        return np.where(norms == 0.0, 0.0, arr / safe)
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")


def sanitize_fts5_query(query_text: str) -> str | None:
    """Convert raw user text into a safe FTS5 OR query preventing syntax errors on hyphens."""
    tokens = re.findall(r"[A-Za-z0-9_]+", query_text)
    if not tokens:
        return None
    return " OR ".join(f'"{tok}"' for tok in tokens)


class SqliteHybridStore:
    """Local hybrid store combining SQLite FTS5 (BM25) and NumPy cosine similarity via RRF."""

    def __init__(self, db_path: str = ":memory:", *, rrf_k: int = 60) -> None:
        self._rrf_k = rrf_k
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(chunk_id UNINDEXED, title, text)
                """
            )

    def add_chunks(
        self,
        chunks: list[ChunkRecord],
        vectors: list[list[float]] | np.ndarray,
    ) -> None:
        """Upsert chunks and unit-normalized embeddings atomically into SQLite + FTS5."""
        if not chunks:
            return
        mat = np.asarray(vectors, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != len(chunks):
            raise ValueError(f"Expected ({len(chunks)}, D) vectors, got {mat.shape}.")
        normed = l2_normalize(mat)

        with self._conn:
            for chunk, vec in zip(chunks, normed, strict=True):
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, doc_id, chunk_index, title, text, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.chunk_index,
                        chunk.title,
                        chunk.text,
                        json.dumps(vec.tolist()),
                    ),
                )
                self._conn.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    (chunk.chunk_id,),
                )
                self._conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, title, text) VALUES (?, ?, ?)",
                    (chunk.chunk_id, chunk.title, chunk.text),
                )

    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float] | np.ndarray,
        top_k: int = 3,
        mode: RetrievalMode = "hybrid",
    ) -> list[SearchHit]:
        """Search chunks using ``dense``, ``bm25``, or ``hybrid`` (Reciprocal Rank Fusion)."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        records, dense_ranked = self._dense_rank(query_vector)
        if not records:
            return []

        bm25_ranked = self._bm25_rank(query_text) if mode in {"bm25", "hybrid"} else []

        dense_pos = {cid: rank for rank, (cid, _) in enumerate(dense_ranked, start=1)}
        bm25_pos = {cid: rank for rank, cid in enumerate(bm25_ranked, start=1)}

        if mode == "dense":
            ordered_ids = [cid for cid, _ in dense_ranked[:top_k]]
            score_map = {cid: round(score, 6) for cid, score in dense_ranked}
        elif mode == "bm25":
            ordered_ids = bm25_ranked[:top_k]
            score_map = {cid: round(1.0 / rank, 6) for rank, cid in enumerate(bm25_ranked, start=1)}
        else:
            score_map = self._compute_rrf_scores(dense_pos, bm25_pos)
            ordered_ids = sorted(
                score_map,
                key=lambda cid: (-score_map[cid], dense_pos.get(cid, 10_000)),
            )[:top_k]

        return [
            SearchHit(
                chunk=records[cid],
                score=score_map[cid],
                dense_rank=dense_pos.get(cid),
                bm25_rank=bm25_pos.get(cid),
            )
            for cid in ordered_ids
        ]

    def _dense_rank(
        self,
        query_vector: list[float] | np.ndarray,
    ) -> tuple[dict[str, ChunkRecord], list[tuple[str, float]]]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, chunk_index, title, text, embedding_json FROM chunks"
        ).fetchall()
        if not rows:
            return {}, []

        records: dict[str, ChunkRecord] = {}
        vectors: list[list[float]] = []
        ids: list[str] = []
        for row in rows:
            cid = str(row["chunk_id"])
            records[cid] = ChunkRecord(
                chunk_id=cid,
                doc_id=str(row["doc_id"]),
                chunk_index=int(row["chunk_index"]),
                title=str(row["title"]),
                text=str(row["text"]),
            )
            ids.append(cid)
            vectors.append(json.loads(str(row["embedding_json"])))

        mat = np.asarray(vectors, dtype=np.float64)
        q_norm = l2_normalize(np.asarray(query_vector, dtype=np.float64))
        if mat.shape[1] != q_norm.shape[0]:
            raise ValueError(
                f"Dimension mismatch: stored {mat.shape[1]}, query {q_norm.shape[0]}."
            )
        scores = mat @ q_norm
        order = np.argsort(-scores, kind="stable")
        ranked = [(ids[int(i)], float(scores[int(i)])) for i in order]
        return records, ranked

    def _bm25_rank(self, query_text: str) -> list[str]:
        fts_query = sanitize_fts5_query(query_text)
        if not fts_query:
            return []
        rows = self._conn.execute(
            """
            SELECT chunk_id
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            """,
            (fts_query,),
        ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def _compute_rrf_scores(
        self,
        dense_pos: dict[str, int],
        bm25_pos: dict[str, int],
    ) -> dict[str, float]:
        candidate_ids = set(dense_pos) | set(bm25_pos)
        scores: dict[str, float] = {}
        for cid in candidate_ids:
            rrf = 0.0
            if cid in dense_pos:
                rrf += 1.0 / (self._rrf_k + dense_pos[cid])
            if cid in bm25_pos:
                rrf += 1.0 / (self._rrf_k + bm25_pos[cid])
            scores[cid] = round(rrf, 6)
        return scores
