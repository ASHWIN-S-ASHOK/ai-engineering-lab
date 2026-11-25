"""Unit tests for switchable provider registry and header generation."""

from async_api_lab.config import AppSettings
from async_api_lab.providers import PROVIDER_SPECS, ProviderAdapter, create_provider


def test_all_four_providers_registered() -> None:
    assert set(PROVIDER_SPECS.keys()) == {"gemini", "openai", "anthropic", "ollama"}


def test_provider_implements_protocol_and_builds_expected_headers() -> None:
    settings = AppSettings(
        _env_file=None,
        gemini_api_key="gem-key",
        openai_api_key="oai-key",
        anthropic_api_key="ant-key",
    )

    gemini = create_provider(settings, provider="gemini", require_credentials=True)
    openai = create_provider(settings, provider="openai", require_credentials=True)
    anthropic = create_provider(settings, provider="anthropic", require_credentials=True)
    ollama = create_provider(settings, provider="ollama", require_credentials=True)

    for adapter in (gemini, openai, anthropic, ollama):
        assert isinstance(adapter, ProviderAdapter)

    assert gemini.build_headers()["x-goog-api-key"] == "gem-key"
    assert openai.build_headers()["Authorization"] == "Bearer oai-key"
    assert anthropic.build_headers()["x-api-key"] == "ant-key"
    assert anthropic.build_headers()["anthropic-version"] == "2023-06-01"
    assert ollama.build_headers() == {"Content-Type": "application/json"}
