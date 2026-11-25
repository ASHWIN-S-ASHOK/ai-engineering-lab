"""Typed environment configuration for switchable embedding backends."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from embeddings_lab.models import (
    EmbeddingProviderName,
    NonRetryableEmbeddingError,
    ProviderRuntimeConfig,
    UnsupportedEmbeddingProviderError,
)

_PROVIDER_REQUIRES_KEY: dict[EmbeddingProviderName, bool] = {
    "gemini": True,
    "openai": True,
    "ollama": False,
}


class AppSettings(BaseSettings):
    """Validated runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_provider: EmbeddingProviderName = "gemini"

    gemini_api_key: SecretStr | None = None
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"

    openai_api_key: SecretStr | None = None
    openai_embed_model: str = "text-embedding-3-small"
    openai_base_url: str = "https://api.openai.com/v1"

    ollama_embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    http_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    http_max_retries: int = Field(default=2, ge=0, le=5)

    def resolve_provider(
        self,
        provider: str | None = None,
        *,
        require_credentials: bool = True,
    ) -> tuple[ProviderRuntimeConfig, SecretStr | None]:
        """Resolve runtime config for Gemini, OpenAI, or Ollama embeddings."""
        selected = provider or self.embedding_provider
        if selected == "anthropic":
            raise UnsupportedEmbeddingProviderError(
                "Anthropic does not expose a native embedding API; pair Claude generation "
                "with 'gemini', 'openai', or local 'ollama' embeddings."
            )
        if selected not in _PROVIDER_REQUIRES_KEY:
            raise UnsupportedEmbeddingProviderError(
                f"Unsupported embedding provider '{selected}'."
            )

        typed_provider: EmbeddingProviderName = selected  # type: ignore[assignment]
        model: str = getattr(self, f"{typed_provider}_embed_model")
        base_url: str = getattr(self, f"{typed_provider}_base_url").rstrip("/")
        api_key: SecretStr | None = getattr(self, f"{typed_provider}_api_key", None)

        has_key = api_key is not None and bool(api_key.get_secret_value().strip())
        if require_credentials and _PROVIDER_REQUIRES_KEY[typed_provider] and not has_key:
            raise NonRetryableEmbeddingError(
                f"Provider '{typed_provider}' requires {typed_provider.upper()}_API_KEY."
            )

        runtime = ProviderRuntimeConfig(
            provider=typed_provider,
            model=model,
            base_url=base_url,
            has_api_key=has_key,
        )
        return runtime, (api_key if has_key else None)
