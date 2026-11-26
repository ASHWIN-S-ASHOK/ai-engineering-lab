"""Typed environment configuration for the switchable tool-calling router."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from tool_router_lab.models import NonRetryableApiError, ProviderName, ProviderRuntimeConfig

_PROVIDER_REQUIRES_KEY: dict[ProviderName, bool] = {
    "gemini": True,
    "openai": True,
    "anthropic": True,
    "ollama": False,
}


class AppSettings(BaseSettings):
    """Validated runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: ProviderName = "gemini"

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-3-5-haiku-latest"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_version: str = "2023-06-01"

    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"

    router_max_steps: int = Field(default=4, ge=1, le=12)
    http_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    http_max_retries: int = Field(default=2, ge=0, le=5)

    def resolve_provider(
        self,
        provider: ProviderName | None = None,
        *,
        require_credentials: bool = True,
    ) -> tuple[ProviderRuntimeConfig, SecretStr | None]:
        """Resolve normalized runtime metadata and optional secret key."""
        selected: ProviderName = provider or self.llm_provider
        model: str = getattr(self, f"{selected}_model")
        base_url: str = getattr(self, f"{selected}_base_url").rstrip("/")
        api_key: SecretStr | None = getattr(self, f"{selected}_api_key", None)

        has_key = api_key is not None and bool(api_key.get_secret_value().strip())
        if require_credentials and _PROVIDER_REQUIRES_KEY[selected] and not has_key:
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
