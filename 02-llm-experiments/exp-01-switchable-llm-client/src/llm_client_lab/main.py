"""CLI demo showing offline simulated multi-provider switching or optional live call."""

import argparse
import asyncio
import json
import logging

import httpx

from llm_client_lab.client import create_llm_client
from llm_client_lab.config import AppSettings
from llm_client_lab.models import LLMRequest, ProviderName

_ALL_PROVIDERS: tuple[ProviderName, ...] = ("gemini", "openai", "anthropic", "ollama")


def _build_offline_mock_transport() -> httpx.MockTransport:
    """Return canned wire-accurate responses for all 4 providers for offline demos."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "generativelanguage.googleapis.com" in url:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "text": (
                                            "Clean Architecture isolates domain rules "
                                            "from model vendor HTTP details."
                                        )
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 18,
                        "candidatesTokenCount": 11,
                        "totalTokenCount": 29,
                    },
                    "modelVersion": "gemini-2.5-flash",
                },
            )
        if "/responses" in url:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": (
                                        "Adapters decouple prompt contracts "
                                        "from OpenAI's Responses wire schema."
                                    ),
                                }
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 18, "output_tokens": 12, "total_tokens": 30},
                },
            )
        if "/messages" in url:
            return httpx.Response(
                200,
                json={
                    "model": "claude-3-5-haiku-latest",
                    "stop_reason": "end_turn",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Anthropic messages normalize into the same "
                                "LLMResponse contract."
                            ),
                        }
                    ],
                    "usage": {"input_tokens": 18, "output_tokens": 11},
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Local Ollama runs offline with zero API cost.",
                },
                "prompt_eval_count": 18,
                "eval_count": 9,
            },
        )

    return httpx.MockTransport(handler)


async def run_demo(*, live: bool, provider: ProviderName | None, prompt: str) -> None:
    """Run either a live call against the chosen provider or an offline 4-provider demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    request = LLMRequest(
        system_prompt="Answer concisely in one engineering sentence.",
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=128,
    )

    if live:
        settings = AppSettings()
        async with create_llm_client(settings=settings) as client:
            response = await client.generate(request, provider=provider)
        print(response.model_dump_json(indent=2))
        return

    demo_settings = AppSettings(
        _env_file=None,
        gemini_api_key="demo-gemini-key",
        openai_api_key="demo-openai-key",
        anthropic_api_key="demo-anthropic-key",
    )
    async with create_llm_client(
        settings=demo_settings,
        mock_transport=_build_offline_mock_transport(),
    ) as client:
        results = []
        for target in (provider,) if provider else _ALL_PROVIDERS:
            res = await client.generate(request, provider=target)
            results.append(res.model_dump())

    print(json.dumps(results, indent=2))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Switchable multi-provider LLM client demo.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the real provider configured in .env instead of offline mock transport.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic", "ollama"],
        default=None,
        help="Override LLM_PROVIDER for this invocation.",
    )
    parser.add_argument(
        "--prompt",
        default="Why should an AI client decouple provider wire formats behind a Protocol?",
        help="User prompt to send.",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(live=args.live, provider=args.provider, prompt=args.prompt))


if __name__ == "__main__":
    main()
