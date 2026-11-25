"""Switchable provider contract and registry mapping."""

from typing import Protocol, runtime_checkable

from pydantic import SecretStr

from async_api_lab.config import AppSettings
from async_api_lab.models import ProviderName, ProviderSpec, ResolvedProviderConfig

PROVIDER_SPECS: dict[ProviderName, ProviderSpec] = {
    "gemini": ProviderSpec(
        name="gemini",
        requires_api_key=True,
        auth_header="x-goog-api-key",
        auth_prefix="",
    ),
    "openai": ProviderSpec(
        name="openai",
        requires_api_key=True,
        auth_header="Authorization",
        auth_prefix="Bearer ",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        requires_api_key=True,
        auth_header="x-api-key",
        auth_prefix="",
        extra_headers={"anthropic-version": "2023-06-01"},
    ),
    "ollama": ProviderSpec(
        name="ollama",
        requires_api_key=False,
        auth_header=None,
    ),
}


@runtime_checkable
class ProviderAdapter(Protocol):
    """Structural contract for switchable model providers (similar to a Kotlin interface)."""

    @property
    def provider_name(self) -> ProviderName:
        """Identifier of the underlying model provider."""
        ...

    def describe(self) -> ResolvedProviderConfig:
        """Return safe, non-secret metadata for logging and diagnostics."""
        ...

    def build_headers(self) -> dict[str, str]:
        """Build HTTP headers required by the provider."""
        ...


class ConfiguredProvider:
    """Concrete provider descriptor driven uniformly by ``PROVIDER_SPECS``."""

    def __init__(
        self,
        spec: ProviderSpec,
        config: ResolvedProviderConfig,
        api_key: SecretStr | None,
    ) -> None:
        self._spec = spec
        self._config = config
        self._api_key = api_key

    @property
    def provider_name(self) -> ProviderName:
        """Identifier of the configured provider."""
        return self._spec.name

    def describe(self) -> ResolvedProviderConfig:
        """Return immutable non-secret provider metadata."""
        return self._config

    def build_headers(self) -> dict[str, str]:
        """Build provider-specific headers without mutating shared spec dictionaries."""
        headers = {"Content-Type": "application/json", **self._spec.extra_headers}
        if self._spec.auth_header and self._api_key is not None:
            headers[self._spec.auth_header] = (
                f"{self._spec.auth_prefix}{self._api_key.get_secret_value()}"
            )
        return headers


def create_provider(
    settings: AppSettings,
    provider: ProviderName | None = None,
    *,
    require_credentials: bool = False,
) -> ConfiguredProvider:
    """Instantiate a configured provider adapter from typed settings."""
    resolved, api_key = settings.resolve_provider(
        provider=provider,
        require_credentials=require_credentials,
    )
    spec = PROVIDER_SPECS[resolved.provider]
    return ConfiguredProvider(spec=spec, config=resolved, api_key=api_key)
