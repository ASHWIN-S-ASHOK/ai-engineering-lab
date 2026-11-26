"""Unit tests for switchable LLM + embedding provider combinations."""

import pytest

from grounded_rag_poc.config import AppSettings
from grounded_rag_poc.models import GroundedAnswer, NonRetryableApiError, build_strict_schema
from grounded_rag_poc.providers import EMBEDDER_REGISTRY, LLM_REGISTRY


def test_anthropic_llm_can_pair_with_gemini_embeddings() -> None:
    settings = AppSettings(
        _env_file=None,
        llm_provider="anthropic",
        embedding_provider="gemini",
        anthropic_api_key="ant-key",
        gemini_api_key="gem-key",
    )
    llm_rt, llm_key = settings.resolve_llm()
    emb_rt, emb_key = settings.resolve_embedder()

    assert llm_rt.provider == "anthropic"
    assert emb_rt.provider == "gemini"
    req = LLM_REGISTRY["anthropic"].build_request(
        system_prompt="Sys",
        user_prompt="Ask",
        json_schema=build_strict_schema(GroundedAnswer),
        schema_name="GroundedAnswer",
        runtime=llm_rt,
        api_key=llm_key,
    )
    assert req.json_body["output_config"]["format"]["type"] == "json_schema"
    assert emb_key is not None
    assert set(EMBEDDER_REGISTRY) == {"gemini", "openai", "ollama"}


def test_anthropic_rejected_as_embedding_provider() -> None:
    settings = AppSettings(_env_file=None)
    with pytest.raises(NonRetryableApiError, match="Anthropic"):
        settings.resolve_embedder(provider="anthropic")
