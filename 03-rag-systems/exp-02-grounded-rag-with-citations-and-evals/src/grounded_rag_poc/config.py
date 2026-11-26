"""Typed configuration supporting independent LLM and embedding provider switching."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from grounded_rag_poc.models import (
    EmbedProviderName,
    LlmProviderName,
    NonRetryableApiError,
    ProviderRuntimeConfig,
)

_REQUIRES_KEY: dict[str, bool] = {
    "gemini": True,
    "openai": True,
    "anthropic": True,
    "ollama": False,
}


class AppSettings(BaseSettings):
    """Validated settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LlmProviderName = "gemini"
    embedding_provider: EmbedProviderName = "gemini"

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    openai_base_url: str = "https://api.openai.com/v1"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-3-5-haiku-latest"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_version: str = "2023-06-01"

    ollama_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    sqlite_db_path: str = ":memory:"
    rrf_k: int = Field(default=60, ge=1, le=500)
    http_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    http_max_retries: int = Field(default=2, ge=0, le=5)

    def resolve_llm(
        self,
        provider: LlmProviderName | None = None,
        *,
        require_credentials: bool = True,
    ) -> tuple[ProviderRuntimeConfig, SecretStr | None]:
        """Resolve generation provider metadata and secret key."""
        selected = provider or self.llm_provider
        return self._resolve(
            selected=selected,
            model=getattr(self, f"{selected}_model"),
            require_credentials=require_credentials,
        )

    def resolve_embedder(
        self,
        provider: str | None = None,
        *,
        require_credentials: bool = True,
    ) -> tuple[ProviderRuntimeConfig, SecretStr | None]:
        """Resolve embedding provider metadata and secret key."""
        selected = provider or self.embedding_provider
        if selected == "anthropic":
            raise NonRetryableApiError(
                "Anthropic has no native embedding endpoint; set EMBEDDING_PROVIDER to "
                "'gemini', 'openai', or 'ollama'."
            )
        if selected not in {"gemini", "openai", "ollama"}:
            raise NonRetryableApiError(f"Unsupported embedding provider '{selected}'.")
        return self._resolve(
            selected=selected,
            model=getattr(self, f"{selected}_embed_model"),
            require_credentials=require_credentials,
        )

    def _resolve(
        self,
        *,
        selected: str,
        model: str,
        require_credentials: bool,
    ) -> tuple[ProviderRuntimeConfig, SecretStr | None]:
        base_url: str = getattr(self, f"{selected}_base_url").rstrip("/")
        api_key: SecretStr | None = getattr(self, f"{selected}_api_key", None)
        has_key = api_key is not None and bool(api_key.get_secret_value().strip())
        if require_credentials and _REQUIRES_KEY[selected] and not has_key:
            raise NonRetryableApiError(
                f"Provider '{selected}' requires {selected.upper()}_API_KEY."
            )
        runtime = ProviderRuntimeConfig(
            provider=selected,
            model=model,
            base_url=base_url,
            has_api_key=has_key,
            anthropic_version=self.anthropic_version,
        )
        return runtime, (api_key if has_key else None)
