"""Unit tests for overlapping and paragraph-aware document chunkers."""

import pytest

from embeddings_lab.chunking import chunk_fixed_overlap, chunk_paragraphs
from embeddings_lab.models import Document


def test_chunk_fixed_overlap_preserves_provenance_and_overlap() -> None:
    doc = Document(doc_id="doc-1", text="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chunks = chunk_fixed_overlap(doc, chunk_size=10, overlap=2)

    assert [c.text for c in chunks] == ["ABCDEFGHIJ", "IJKLMNOPQR", "QRSTUVWXYZ"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert chunks[0].start_char == 0
    assert chunks[1].start_char == 8


def test_chunk_fixed_overlap_rejects_invalid_bounds() -> None:
    doc = Document(doc_id="doc-1", text="Hello")
    with pytest.raises(ValueError, match="overlap"):
        chunk_fixed_overlap(doc, chunk_size=5, overlap=5)


def test_chunk_paragraphs_groups_short_paragraphs() -> None:
    doc = Document(
        doc_id="doc-2",
        text="Alpha one.\n\nBeta two.\n\nGamma three is short.",
    )
    chunks = chunk_paragraphs(doc, max_chunk_chars=25)

    assert len(chunks) == 2
    assert chunks[0].text == "Alpha one.\n\nBeta two."
    assert chunks[1].text == "Gamma three is short."
    assert all(c.doc_id == "doc-2" for c in chunks)
