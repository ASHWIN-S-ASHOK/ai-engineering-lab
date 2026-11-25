"""Runnable offline demonstration of switchable provider configs and resilient HTTP fetch."""

import asyncio
import logging

import httpx

from async_api_lab.client import AsyncJsonClient
from async_api_lab.config import AppSettings
from async_api_lab.models import ProviderName
from async_api_lab.providers import create_provider

_ALL_PROVIDERS: tuple[ProviderName, ...] = ("gemini", "openai", "anthropic", "ollama")


def _build_demo_transport() -> httpx.MockTransport:
    """Simulate a transient 503 followed by a 200 JSON response without hitting the network."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(503, json={"error": "simulated transient overload"})
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "endpoint": str(request.url),
                "note": "Simulated recovery on retry attempt 2",
            },
        )

    return httpx.MockTransport(handler)


async def run_demo() -> None:
    """Display configured switchable providers and run a resilient async fetch demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = AppSettings()

    print("=== Configured Switchable Providers ===")
    print(f"Active default provider: {settings.llm_provider}")
    for name in _ALL_PROVIDERS:
        adapter = create_provider(settings, provider=name, require_credentials=False)
        info = adapter.describe()
        print(
            f"- {info.provider:9s} | model={info.model:24s} "
            f"| base_url={info.base_url} | api_key_set={info.has_api_key}"
        )

    print("\n=== Resilient Async HTTP Demo (Simulated 503 -> 200 Recovery) ===")
    active = create_provider(settings, require_credentials=False)
    demo_url = f"{active.describe().base_url}/health"
    async with AsyncJsonClient(
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        base_backoff_seconds=0.05,
        transport=_build_demo_transport(),
    ) as client:
        result = await client.get_json(demo_url, headers=active.build_headers())

    print(result.model_dump_json(indent=2))


def main() -> None:
    """CLI entry point."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
