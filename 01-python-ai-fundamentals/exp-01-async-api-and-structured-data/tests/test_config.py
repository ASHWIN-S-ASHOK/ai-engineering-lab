"""Unit tests for typed multi-provider configuration."""

import pytest
from pydantic import ValidationError

from async_api_lab.config import AppSettings
from async_api_lab.models import ProviderName


def test_default_provider_is_gemini() -> None:
    settings = AppSettings(_env_file=None)
    resolved, key = settings.resolve_provider(require_credentials=False)

    assert resolved.provider == "gemini"
    assert resolved.model == "gemini-2.5-flash"
    assert resolved.has_api_key is False
    assert key is None


@pytest.mark.parametrize(
    ("provider", "expected_base_url"),
    [
        ("gemini", "https://generativelanguage.googleapis.com"),
        ("openai", "https://api.openai.com/v1"),
        ("anthropic", "https://api.anthropic.com/v1"),
        ("ollama", "http://localhost:11434"),
    ],
)
def test_provider_switching_resolves_defaults(
    provider: ProviderName,
    expected_base_url: str,
) -> None:
    settings = AppSettings(_env_file=None, llm_provider=provider)
    resolved, _ = settings.resolve_provider(require_credentials=False)

    assert resolved.provider == provider
    assert resolved.base_url == expected_base_url


def test_cloud_provider_requires_key_when_strict() -> None:
    settings = AppSettings(_env_file=None, llm_provider="openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.resolve_provider(require_credentials=True)


def test_ollama_does_not_require_api_key() -> None:
    settings = AppSettings(_env_file=None, llm_provider="ollama")
    resolved, key = settings.resolve_provider(require_credentials=True)

    assert resolved.provider == "ollama"
    assert resolved.has_api_key is False
    assert key is None


def test_secret_api_key_is_masked_in_repr() -> None:
    raw_secret = "super-secret-test-key"
    settings = AppSettings(_env_file=None, gemini_api_key=raw_secret)
    resolved, key = settings.resolve_provider(provider="gemini", require_credentials=True)

    assert resolved.has_api_key is True
    assert raw_secret not in repr(settings)
    assert raw_secret not in repr(resolved)
    assert key is not None
    assert key.get_secret_value() == raw_secret


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, http_timeout_seconds=0.0)
