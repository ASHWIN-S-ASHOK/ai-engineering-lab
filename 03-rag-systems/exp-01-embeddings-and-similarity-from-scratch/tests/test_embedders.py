"""Unit tests for switchable Gemini, OpenAI, and Ollama embedding adapters."""

import json

import httpx
import pytest

from embeddings_lab.config import AppSettings
from embeddings_lab.embedders import EMBEDDER_REGISTRY, EmbeddingProvider
from embeddings_lab.models import UnsupportedEmbeddingProviderError
from embeddings_lab.transport import ResilientHttpTransport


def test_all_embedders_conform_to_protocol() -> None:
    assert set(EMBEDDER_REGISTRY.keys()) == {"gemini", "openai", "ollama"}
    for embedder in EMBEDDER_REGISTRY.values():
        assert isinstance(embedder, EmbeddingProvider)


def test_anthropic_raises_guidance_error_for_embeddings() -> None:
    settings = AppSettings(_env_file=None)
    with pytest.raises(UnsupportedEmbeddingProviderError, match="Anthropic"):
        settings.resolve_provider(provider="anthropic")


async def test_gemini_embed_batch_uses_task_type() -> None:
    seen_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"embedding": {"values": [0.1, 0.9]}})

    settings = AppSettings(_env_file=None, gemini_api_key="gem-key")
    runtime, key = settings.resolve_provider(provider="gemini", require_credentials=True)
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    async with transport:
        vectors = await EMBEDDER_REGISTRY["gemini"].embed_batch(
            ["doc one", "doc two"],
            task="document",
            runtime=runtime,
            api_key=key,
            transport=transport,
        )

    assert len(vectors) == 2
    assert seen_bodies[0]["taskType"] == "RETRIEVAL_DOCUMENT"


async def test_openai_and_ollama_batch_embeddings() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/embeddings" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )
        return httpx.Response(200, json={"embeddings": [[0.5, 0.5], [0.2, 0.8]]})

    settings = AppSettings(_env_file=None, openai_api_key="oai-key")
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    async with transport:
        oai_rt, oai_key = settings.resolve_provider(provider="openai", require_credentials=True)
        oai_vecs = await EMBEDDER_REGISTRY["openai"].embed_batch(
            ["first", "second"],
            task="query",
            runtime=oai_rt,
            api_key=oai_key,
            transport=transport,
        )
        oll_rt, oll_key = settings.resolve_provider(provider="ollama", require_credentials=True)
        oll_vecs = await EMBEDDER_REGISTRY["ollama"].embed_batch(
            ["first", "second"],
            task="document",
            runtime=oll_rt,
            api_key=oll_key,
            transport=transport,
        )

    assert oai_vecs == [[1.0, 0.0], [0.0, 1.0]]
    assert oll_vecs == [[0.5, 0.5], [0.2, 0.8]]
