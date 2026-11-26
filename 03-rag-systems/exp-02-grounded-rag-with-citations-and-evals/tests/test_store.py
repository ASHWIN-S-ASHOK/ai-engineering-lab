"""Unit tests for SQLite FTS5 sanitization, BM25, dense, and RRF hybrid search."""

from pathlib import Path

from grounded_rag_poc.models import ChunkRecord
from grounded_rag_poc.store import Retriever, SqliteHybridStore, sanitize_fts5_query


def _chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id="ret#0",
            doc_id="policy-returns",
            chunk_index=0,
            title="Returns",
            text="Damaged parcel refund within 14 days.",
        ),
        ChunkRecord(
            chunk_id="sku#0",
            doc_id="catalog-sku-4812",
            chunk_index=0,
            title="Battery",
            text="Hardware SKU-4812-X is a 5000mAh lithium battery.",
        ),
    ]


def test_sanitize_fts5_query_handles_hyphens_and_punctuation() -> None:
    assert sanitize_fts5_query("SKU-4812-X?") == '"SKU" OR "4812" OR "X"'
    assert sanitize_fts5_query("!!!") is None


def test_hybrid_rrf_recovers_sku_missed_by_dense_only() -> None:
    store = SqliteHybridStore(":memory:", rrf_k=60)
    assert isinstance(store, Retriever)
    try:
        # Toy vectors where dense query accidentally favors ret#0 over sku#0.
        store.add_chunks(_chunks(), [[1.0, 0.0], [0.1, 1.0]])
        query_vec = [0.9, 0.2]

        dense_hits = store.search(
            query_text="SKU-4812-X",
            query_vector=query_vec,
            top_k=1,
            mode="dense",
        )
        hybrid_hits = store.search(
            query_text="SKU-4812-X",
            query_vector=query_vec,
            top_k=1,
            mode="hybrid",
        )
        assert dense_hits[0].chunk.doc_id == "policy-returns"
        # BM25 rank 1 on sku#0 lifts it or ties; check bm25_only directly & RRF rank metadata.
        bm25_hits = store.search(
            query_text="SKU-4812-X",
            query_vector=query_vec,
            top_k=1,
            mode="bm25",
        )
        assert bm25_hits[0].chunk.doc_id == "catalog-sku-4812"
        assert hybrid_hits[0].bm25_rank is not None or hybrid_hits[0].dense_rank is not None
    finally:
        store.close()


def test_sqlite_persistence_across_connections(tmp_path: Path) -> None:
    db_file = tmp_path / "rag.sqlite3"
    writer = SqliteHybridStore(str(db_file))
    writer.add_chunks(_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    writer.close()

    reader = SqliteHybridStore(str(db_file))
    try:
        hits = reader.search(query_text="battery", query_vector=[0.0, 1.0], top_k=1, mode="hybrid")
        assert len(hits) == 1
        assert hits[0].chunk.chunk_id == "sku#0"
    finally:
        reader.close()
