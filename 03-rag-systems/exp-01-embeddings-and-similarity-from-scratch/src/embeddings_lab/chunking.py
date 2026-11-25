"""Document chunking strategies preserving source provenance."""

from embeddings_lab.models import Document, TextChunk


def chunk_fixed_overlap(
    document: Document,
    *,
    chunk_size: int = 300,
    overlap: int = 60,
) -> list[TextChunk]:
    """Split ``document.text`` into fixed character windows with sliding overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    text = document.text.strip()
    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        segment = text[start:end].strip()
        if segment:
            chunks.append(
                TextChunk(
                    chunk_id=f"{document.doc_id}#c{idx}",
                    doc_id=document.doc_id,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    text=segment,
                )
            )
            idx += 1
        if end == len(text):
            break
        start += step

    return chunks


def chunk_paragraphs(
    document: Document,
    *,
    max_chunk_chars: int = 400,
) -> list[TextChunk]:
    """Group paragraphs up to ``max_chunk_chars`` without splitting mid-paragraph when avoidable."""
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive.")

    paragraphs = [p.strip() for p in document.text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_len = 0
    cursor = 0
    chunk_start = 0

    def flush(end_pos: int) -> None:
        if not buffer:
            return
        idx = len(chunks)
        chunks.append(
            TextChunk(
                chunk_id=f"{document.doc_id}#p{idx}",
                doc_id=document.doc_id,
                chunk_index=idx,
                start_char=chunk_start,
                end_char=end_pos,
                text="\n\n".join(buffer),
            )
        )

    for para in paragraphs:
        para_start = document.text.find(para, cursor)
        if para_start == -1:
            para_start = cursor
        para_end = para_start + len(para)

        if len(para) > max_chunk_chars:
            flush(cursor)
            buffer.clear()
            buffer_len = 0
            sub_doc = Document(doc_id=document.doc_id, text=para)
            for sub in chunk_fixed_overlap(sub_doc, chunk_size=max_chunk_chars, overlap=0):
                idx = len(chunks)
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.doc_id}#p{idx}",
                        doc_id=document.doc_id,
                        chunk_index=idx,
                        start_char=para_start + sub.start_char,
                        end_char=para_start + sub.end_char,
                        text=sub.text,
                    )
                )
            cursor = para_end
            chunk_start = para_end
            continue

        projected = buffer_len + (2 if buffer else 0) + len(para)
        if buffer and projected > max_chunk_chars:
            flush(cursor)
            buffer.clear()
            buffer_len = 0
            chunk_start = para_start
        elif not buffer:
            chunk_start = para_start

        buffer.append(para)
        buffer_len += (2 if len(buffer) > 1 else 0) + len(para)
        cursor = para_end

    flush(cursor)
    return chunks
