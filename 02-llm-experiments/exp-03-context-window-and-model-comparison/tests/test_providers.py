"""Unit tests for switchable adapters, Ollama num_ctx, and preflight payloads."""

from pydantic import SecretStr

from context_compare_lab.models import ModelRequest, ProviderRuntimeConfig
from context_compare_lab.providers import ADAPTER_REGISTRY, ContextModelAdapter


def test_all_adapters_implement_protocol() -> None:
    assert set(ADAPTER_REGISTRY.keys()) == {"gemini", "openai", "anthropic", "ollama"}
    for adapter in ADAPTER_REGISTRY.values():
        assert isinstance(adapter, ContextModelAdapter)


def test_ollama_forwards_configured_num_ctx() -> None:
    adapter = ADAPTER_REGISTRY["ollama"]
    runtime = ProviderRuntimeConfig(
        provider="ollama",
        model="llama3.1",
        base_url="http://localhost:11434",
        has_api_key=False,
        ollama_num_ctx=16384,
    )
    prepared = adapter.build_generate_request(ModelRequest(prompt="Hi"), runtime, None)
    assert prepared.json_body["options"]["num_ctx"] == 16384


def test_gemini_and_anthropic_count_tokens_requests() -> None:
    req = ModelRequest(prompt="Hello", system_prompt="System")
    gem_rt = ProviderRuntimeConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com",
        has_api_key=True,
    )
    gem_prep = ADAPTER_REGISTRY["gemini"].build_count_tokens_request(
        req, gem_rt, SecretStr("g-key")
    )
    assert gem_prep is not None
    assert gem_prep.url.endswith(":countTokens")
    assert "systemInstruction" in gem_prep.json_body

    ant_rt = ProviderRuntimeConfig(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        base_url="https://api.anthropic.com/v1",
        has_api_key=True,
    )
    ant_prep = ADAPTER_REGISTRY["anthropic"].build_count_tokens_request(
        req, ant_rt, SecretStr("a-key")
    )
    assert ant_prep is not None
    assert ant_prep.url.endswith("/messages/count_tokens")
    assert ant_prep.json_body["system"] == "System"
