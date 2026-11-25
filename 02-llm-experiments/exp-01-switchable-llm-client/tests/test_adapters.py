"""Unit tests verifying wire-level serialization and normalization across all 4 adapters."""

import pytest
from pydantic import SecretStr

from llm_client_lab.adapters import ADAPTER_REGISTRY, LLMProvider
from llm_client_lab.models import LLMRequest, NonRetryableLLMError, ProviderRuntimeConfig


def test_all_adapters_implement_protocol() -> None:
    assert set(ADAPTER_REGISTRY.keys()) == {"gemini", "openai", "anthropic", "ollama"}
    for adapter in ADAPTER_REGISTRY.values():
        assert isinstance(adapter, LLMProvider)


def test_gemini_build_and_parse() -> None:
    adapter = ADAPTER_REGISTRY["gemini"]
    runtime = ProviderRuntimeConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com",
        has_api_key=True,
    )
    req = LLMRequest(
        prompt="Explain tokens.",
        system_prompt="Be brief.",
        temperature=0.1,
        max_output_tokens=100,
    )
    prepared = adapter.build_request(req, runtime, SecretStr("gem-secret"))

    assert prepared.url.endswith("/v1beta/models/gemini-2.5-flash:generateContent")
    assert prepared.headers["x-goog-api-key"] == "gem-secret"
    assert prepared.json_body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert prepared.json_body["generationConfig"] == {"maxOutputTokens": 100, "temperature": 0.1}

    parsed = adapter.parse_response(
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Tokens are sub-word pieces."}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
            "modelVersion": "gemini-2.5-flash",
        },
        requested_model="gemini-2.5-flash",
        latency_ms=42.5,
        attempts=1,
    )
    assert parsed.text == "Tokens are sub-word pieces."
    assert parsed.finish_reason == "STOP"
    assert parsed.usage.total_tokens == 15


def test_openai_responses_build_and_parse() -> None:
    adapter = ADAPTER_REGISTRY["openai"]
    runtime = ProviderRuntimeConfig(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        has_api_key=True,
    )
    req = LLMRequest(prompt="Hi", system_prompt="Be concise.", max_output_tokens=64)
    prepared = adapter.build_request(req, runtime, SecretStr("oai-secret"))

    assert prepared.url == "https://api.openai.com/v1/responses"
    assert prepared.headers["Authorization"] == "Bearer oai-secret"
    assert prepared.json_body["instructions"] == "Be concise."
    assert prepared.json_body["max_output_tokens"] == 64

    parsed = adapter.parse_response(
        {
            "model": "gpt-4o-mini",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Partial"}]}
            ],
            "usage": {"input_tokens": 8, "output_tokens": 64, "total_tokens": 72},
        },
        requested_model="gpt-4o-mini",
        latency_ms=55.0,
        attempts=1,
    )
    assert parsed.text == "Partial"
    assert parsed.finish_reason == "max_output_tokens"
    assert parsed.usage.output_tokens == 64


def test_anthropic_messages_build_and_parse() -> None:
    adapter = ADAPTER_REGISTRY["anthropic"]
    runtime = ProviderRuntimeConfig(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        base_url="https://api.anthropic.com/v1",
        has_api_key=True,
        anthropic_version="2023-06-01",
    )
    req = LLMRequest(prompt="Hi", system_prompt="One line.", max_output_tokens=50)
    prepared = adapter.build_request(req, runtime, SecretStr("ant-secret"))

    assert prepared.url == "https://api.anthropic.com/v1/messages"
    assert prepared.headers["x-api-key"] == "ant-secret"
    assert prepared.headers["anthropic-version"] == "2023-06-01"
    assert prepared.json_body["system"] == "One line."

    parsed = adapter.parse_response(
        {
            "model": "claude-3-5-haiku-latest",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 9, "output_tokens": 2},
        },
        requested_model="claude-3-5-haiku-latest",
        latency_ms=33.0,
        attempts=1,
    )
    assert parsed.text == "Hello!"
    assert parsed.finish_reason == "end_turn"
    assert parsed.usage.total_tokens == 11


def test_ollama_chat_build_and_parse() -> None:
    adapter = ADAPTER_REGISTRY["ollama"]
    runtime = ProviderRuntimeConfig(
        provider="ollama",
        model="llama3.1",
        base_url="http://localhost:11434",
        has_api_key=False,
    )
    req = LLMRequest(prompt="Hi", system_prompt="Brief.", max_output_tokens=40)
    prepared = adapter.build_request(req, runtime, None)

    assert prepared.url == "http://localhost:11434/api/chat"
    assert prepared.json_body["stream"] is False
    assert prepared.json_body["messages"][0] == {"role": "system", "content": "Brief."}

    parsed = adapter.parse_response(
        {
            "model": "llama3.1",
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Local reply"},
            "prompt_eval_count": 7,
            "eval_count": 3,
        },
        requested_model="llama3.1",
        latency_ms=19.2,
        attempts=1,
    )
    assert parsed.text == "Local reply"
    assert parsed.usage.total_tokens == 10


def test_missing_candidates_raises_non_retryable() -> None:
    with pytest.raises(NonRetryableLLMError, match="candidates"):
        ADAPTER_REGISTRY["gemini"].parse_response(
            {}, requested_model="gemini-2.5-flash", latency_ms=1.0, attempts=1
        )
