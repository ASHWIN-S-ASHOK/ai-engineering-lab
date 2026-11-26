"""Grounded RAG orchestration with programmatic citation verification and abstention."""

import re
import time

from pydantic import ValidationError

from grounded_rag_poc.config import AppSettings
from grounded_rag_poc.models import (
    Citation,
    FabricatedCitationError,
    GroundedAnswer,
    LlmProviderName,
    NonRetryableApiError,
    RagResponse,
    RetrievalMode,
    SearchHit,
    build_strict_schema,
)
from grounded_rag_poc.providers import EMBEDDER_REGISTRY, LLM_REGISTRY
from grounded_rag_poc.store import Retriever
from grounded_rag_poc.transport import ResilientHttpTransport

_SYSTEM_PROMPT = (
    "You are a strictly grounded technical assistant. Answer ONLY using the provided context "
    "chunks. Every factual claim must cite the exact chunk_id and an exact verbatim quote from "
    "that chunk. If the context does not contain enough evidence to answer, set abstained=true, "
    "leave citations empty, and state clearly that insufficient evidence was found."
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_citations(
    *,
    citations: list[Citation],
    hits: list[SearchHit],
) -> tuple[list[Citation], list[str]]:
    """Verify every cited chunk_id exists in ``hits`` and quote appears in chunk text."""
    by_id = {hit.chunk.chunk_id: _normalize_whitespace(hit.chunk.text) for hit in hits}
    valid: list[Citation] = []
    invalid_reasons: list[str] = []

    for cite in citations:
        chunk_norm = by_id.get(cite.chunk_id)
        if chunk_norm is None:
            invalid_reasons.append(f"Unknown chunk_id '{cite.chunk_id}'")
            continue
        quote_norm = _normalize_whitespace(cite.quote)
        if not quote_norm or quote_norm not in chunk_norm:
            invalid_reasons.append(f"Quote mismatch for '{cite.chunk_id}': '{cite.quote}'")
            continue
        valid.append(cite)

    return valid, invalid_reasons


def format_context_block(hits: list[SearchHit]) -> str:
    """Format retrieved chunks with explicit chunk IDs for grounded LLM prompting."""
    lines = [
        f"[{hit.chunk.chunk_id}] ({hit.chunk.title or hit.chunk.doc_id}): {hit.chunk.text}"
        for hit in hits
    ]
    return "\n\n".join(lines)


class GroundedRagPipeline:
    """End-to-end hybrid retrieval + grounded JSON generation + citation validation."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        store: Retriever,
        transport: ResilientHttpTransport,
        strict_citations: bool = True,
    ) -> None:
        self._settings = settings
        self._store = store
        self._transport = transport
        self._strict_citations = strict_citations

    async def answer(
        self,
        query: str,
        *,
        top_k: int = 3,
        mode: RetrievalMode = "hybrid",
        llm_provider: LlmProviderName | None = None,
        embed_provider: str | None = None,
    ) -> RagResponse:
        """Retrieve relevant context, generate grounded JSON, and verify citations."""
        started = time.perf_counter()
        embed_rt, embed_key = self._settings.resolve_embedder(
            provider=embed_provider, require_credentials=True
        )
        embedder = EMBEDDER_REGISTRY[embed_rt.provider]  # type: ignore[index]
        query_vecs = await embedder.embed_batch(
            [query],
            task="query",
            runtime=embed_rt,
            api_key=embed_key,
            transport=self._transport,
        )

        hits = self._store.search(
            query_text=query,
            query_vector=query_vecs[0],
            top_k=top_k,
            mode=mode,
        )

        llm_rt, llm_key = self._settings.resolve_llm(
            provider=llm_provider, require_credentials=True
        )
        llm = LLM_REGISTRY[llm_rt.provider]  # type: ignore[index]
        user_prompt = (
            f"Context:\n{format_context_block(hits)}\n\n"
            f"Question: {query}\n\n"
            "Respond strictly in JSON matching GroundedAnswer."
        )
        prepared = llm.build_request(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=build_strict_schema(GroundedAnswer),
            schema_name="GroundedAnswer",
            runtime=llm_rt,
            api_key=llm_key,
        )
        payload, _, _ = await self._transport.post_json(prepared)
        raw_text, usage = llm.parse_response(payload)

        try:
            parsed = GroundedAnswer.model_validate_json(raw_text)
        except ValidationError as exc:
            raise NonRetryableApiError(f"GroundedAnswer validation failed: {exc}") from exc

        valid_cites, invalid_reasons = verify_citations(citations=parsed.citations, hits=hits)
        if not parsed.abstained and not valid_cites:
            invalid_reasons.append("Non-abstained answer produced zero verified citations.")
        if self._strict_citations and invalid_reasons:
            raise FabricatedCitationError("; ".join(invalid_reasons))

        total_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return RagResponse(
            query=query,
            retrieval_mode=mode,
            llm_provider=llm_rt.provider,  # type: ignore[arg-type]
            model=llm_rt.model,
            answer=parsed.answer,
            abstained=parsed.abstained,
            citations=valid_cites,
            invalid_citations=invalid_reasons,
            retrieved_hits=hits,
            usage=usage,
            latency_ms=total_ms,
        )
