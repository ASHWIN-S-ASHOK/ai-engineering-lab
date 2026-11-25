"""Unit tests for switchable LLM provider configuration."""

import pytest

from llm_client_lab.config import AppSettings
from llm_client_lab.models import NonRetryableLLMError, ProviderName


def test_default_provider_is_gemini() -> None:
    settings = AppSettings(_env_file=None)
    runtime, key = settings.resolve_provider(require_credentials=False)

    assert runtime.provider == "gemini"
    assert runtime.model == "gemini-2.5-flash"
    assert runtime.has_api_key is False
    assert key is None


@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic"])
def test_cloud_providers_require_api_key_when_strict(provider: ProviderName) -> None:
    settings = AppSettings(_env_file=None, llm_provider=provider)
    with pytest.raises(NonRetryableLLMError, match=f"{provider.upper()}_API_KEY"):
        settings.resolve_provider(require_credentials=True)


def test_ollama_allows_keyless_strict_resolution() -> None:
    settings = AppSettings(_env_file=None, llm_provider="ollama")
    runtime, key = settings.resolve_provider(require_credentials=True)

    assert runtime.provider == "ollama"
    assert runtime.base_url == "http://localhost:11434"
    assert key is None


def test_secret_key_masked_in_repr() -> None:
    secret = "secret-test-token-123"
    settings = AppSettings(_env_file=None, gemini_api_key=secret)
    runtime, _ = settings.resolve_provider(provider="gemini", require_credentials=True)

    assert secret not in repr(settings)
    assert secret not in repr(runtime)
