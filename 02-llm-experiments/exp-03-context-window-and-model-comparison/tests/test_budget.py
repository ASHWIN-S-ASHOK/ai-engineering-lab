"""Unit tests for ContextBudgetManager and preflight token counting."""

import httpx
import pytest

from context_compare_lab.budget import ContextBudgetManager, preflight_count_tokens
from context_compare_lab.config import AppSettings
from context_compare_lab.models import ContextBudgetExceededError, ContextChunk, ModelRequest
from context_compare_lab.transport import ResilientHttpTransport


def _word_counter(text: str) -> int:
    return len(text.split())


def test_pack_fail_fast_raises_when_budget_exceeded() -> None:
    manager = ContextBudgetManager(token_counter=_word_counter)
    chunks = [ContextChunk(chunk_id="c1", text="one two three four five")]

    with pytest.raises(ContextBudgetExceededError):
        manager.pack(
            system_prompt="sys",
            user_query="ask",
            chunks=chunks,
            context_limit_tokens=10,
            reserved_output_tokens=5,
            policy="fail_fast",
        )


def test_pack_drop_oldest_removes_leading_chunks() -> None:
    manager = ContextBudgetManager(token_counter=_word_counter)
    chunks = [
        ContextChunk(chunk_id="c1", text="a b c"),
        ContextChunk(chunk_id="c2", text="d e f"),
        ContextChunk(chunk_id="c3", text="g h i"),
    ]
    packed = manager.pack(
        system_prompt="sys",
        user_query="ask",
        chunks=chunks,
        context_limit_tokens=12,
        reserved_output_tokens=4,
        policy="drop_oldest",
    )
    assert packed.dropped_chunk_ids == ["c1"]
    assert [c.chunk_id for c in packed.kept_chunks] == ["c2", "c3"]
    assert packed.estimated_input_tokens <= 8


def test_pack_drop_middle_preserves_edges() -> None:
    manager = ContextBudgetManager(token_counter=_word_counter)
    chunks = [
        ContextChunk(chunk_id="c1", text="a b c"),
        ContextChunk(chunk_id="c2", text="d e f"),
        ContextChunk(chunk_id="c3", text="g h i"),
    ]
    packed = manager.pack(
        system_prompt="sys",
        user_query="ask",
        chunks=chunks,
        context_limit_tokens=12,
        reserved_output_tokens=4,
        policy="drop_middle",
    )
    assert packed.dropped_chunk_ids == ["c2"]
    assert [c.chunk_id for c in packed.kept_chunks] == ["c1", "c3"]


async def test_preflight_count_tokens_for_gemini_and_anthropic() -> None:
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if ":countTokens" in str(request.url):
            return httpx.Response(200, json={"totalTokens": 27})
        return httpx.Response(200, json={"input_tokens": 31})

    settings = AppSettings(
        _env_file=None,
        gemini_api_key="gem-key",
        anthropic_api_key="ant-key",
    )
    req = ModelRequest(prompt="Count me", system_prompt="Brief")
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    async with transport:
        gem_tokens = await preflight_count_tokens(
            settings=settings, transport=transport, request=req, provider="gemini"
        )
        ant_tokens = await preflight_count_tokens(
            settings=settings, transport=transport, request=req, provider="anthropic"
        )
        ollama_tokens = await preflight_count_tokens(
            settings=settings, transport=transport, request=req, provider="ollama"
        )

    assert gem_tokens == 27
    assert ant_tokens == 31
    assert ollama_tokens is None
    assert any(":countTokens" in u for u in seen_urls)
    assert any("/messages/count_tokens" in u for u in seen_urls)
