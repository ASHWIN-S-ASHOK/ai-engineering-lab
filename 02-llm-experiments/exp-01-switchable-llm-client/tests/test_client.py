"""End-to-end offline tests for SwitchableLLMClient switching and error resilience."""

import httpx
import pytest

from llm_client_lab.client import SwitchableLLMClient
from llm_client_lab.config import AppSettings
from llm_client_lab.models import LLMRequest, NonRetryableLLMError, RetryableLLMError
from llm_client_lab.transport import ResilientHttpTransport


def _settings() -> AppSettings:
    return AppSettings(
        _env_file=None,
        llm_provider="gemini",
        gemini_api_key="gem-key",
        openai_api_key="oai-key",
        anthropic_api_key="ant-key",
        http_max_retries=2,
    )


def _client_with_transport(
    settings: AppSettings,
    transport: httpx.MockTransport,
) -> SwitchableLLMClient:
    resilient = ResilientHttpTransport(
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        base_backoff_seconds=0.0,
        transport=transport,
    )
    return SwitchableLLMClient(settings=settings, http_transport=resilient)


async def test_switchable_client_routes_across_providers() -> None:
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if "generateContent" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "from-gemini"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 7,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "from-ollama"},
                "prompt_eval_count": 5,
                "eval_count": 2,
            },
        )

    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    async with transport:
        client = SwitchableLLMClient(settings=_settings(), http_transport=transport)
        req = LLMRequest(prompt="Ping")
        res_gemini = await client.generate(req)
        res_ollama = await client.generate(req, provider="ollama")

    assert res_gemini.provider == "gemini"
    assert res_gemini.text == "from-gemini"
    assert res_ollama.provider == "ollama"
    assert res_ollama.text == "from-ollama"
    assert len(seen_urls) == 2


async def test_client_retries_429_and_recovers() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "ok after retry"},
                "prompt_eval_count": 4,
                "eval_count": 3,
            },
        )

    transport = ResilientHttpTransport(
        max_retries=2,
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    async with transport:
        client = SwitchableLLMClient(settings=_settings(), http_transport=transport)
        res = await client.generate(LLMRequest(prompt="Hi"), provider="ollama")

    assert res.attempts == 2
    assert res.text == "ok after retry"


async def test_client_fails_fast_on_401_without_retry() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "invalid key"})

    transport = ResilientHttpTransport(
        max_retries=2,
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    async with transport:
        client = SwitchableLLMClient(settings=_settings(), http_transport=transport)
        with pytest.raises(NonRetryableLLMError) as exc_info:
            await client.generate(LLMRequest(prompt="Hi"), provider="openai")

    assert calls == 1
    assert exc_info.value.status_code == 401


async def test_client_raises_retryable_after_503_exhaustion() -> None:
    transport = ResilientHttpTransport(
        max_retries=1,
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable")),
    )
    async with transport:
        client = SwitchableLLMClient(settings=_settings(), http_transport=transport)
        with pytest.raises(RetryableLLMError) as exc_info:
            await client.generate(LLMRequest(prompt="Hi"), provider="ollama")

    assert exc_info.value.attempts == 2
    assert exc_info.value.status_code == 503
