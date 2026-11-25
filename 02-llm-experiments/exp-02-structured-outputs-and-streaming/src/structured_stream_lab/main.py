"""Offline & optional live demo of structured JSON extraction and SSE/NDJSON streaming."""

import argparse
import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from structured_stream_lab.client import create_client
from structured_stream_lab.config import AppSettings
from structured_stream_lab.models import GenerationRequest, ProviderName


class ArchitectureFinding(BaseModel):
    """Example Pydantic target schema for structured LLM extraction."""

    summary: str = Field(description="One-sentence engineering takeaway.")
    risk_level: Literal["low", "medium", "high"]
    recommended_actions: list[str] = Field(min_length=1, max_length=3)


class _ChunkedAsyncStream(httpx.AsyncByteStream):
    """Helper streaming mock bytes in small chunks to simulate incremental network delivery."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(0.01)
            yield chunk


def _build_offline_transport() -> httpx.MockTransport:
    """Mock both structured JSON output and SSE/NDJSON streaming across providers."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "streamGenerateContent" in url:
            sse_chunks = [
                (
                    b'data: {"candidates":[{"content":{"parts":'
                    b'[{"text":"Streaming separates "}]}}]}\n\n'
                ),
                (
                    b'data: {"candidates":[{"content":{"parts":'
                    b'[{"text":"TTFT from total latency."}]},"finishReason":"STOP"}],'
                    b'"usageMetadata":{"promptTokenCount":14,"candidatesTokenCount":9,'
                    b'"totalTokenCount":23}}\n\n'
                ),
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_ChunkedAsyncStream(sse_chunks),
            )

        payload_text = json.dumps(
            {
                "summary": "Enforcing JSON Schema at decoding prevents prose wrapping.",
                "risk_level": "medium",
                "recommended_actions": [
                    "Validate model output with Pydantic",
                    "Inspect finish_reason for max_output_tokens truncation",
                ],
            }
        )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": payload_text}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {
                    "promptTokenCount": 42,
                    "candidatesTokenCount": 31,
                    "totalTokenCount": 73,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        )

    return httpx.MockTransport(handler)


async def run_demo(*, live: bool, provider: ProviderName | None) -> None:
    """Demonstrate schema-enforced extraction followed by incremental streaming."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = (
        AppSettings()
        if live
        else AppSettings(_env_file=None, gemini_api_key="demo-gemini-key")
    )
    transport = None if live else _build_offline_transport()

    async with create_client(settings=settings, mock_transport=transport) as client:
        print("=== 1. Structured JSON Extraction (Pydantic-validated) ===")
        structured = await client.generate_structured(
            system_prompt="Return concise architectural review JSON.",
            prompt="Summarize why LLM APIs need schema validation and truncation checks.",
            response_model=ArchitectureFinding,
            provider=provider,
        )
        print(structured.model_dump_json(indent=2))

        print("\n=== 2. Streaming Generation (TTFT vs Total Latency) ===")
        stream_req = GenerationRequest(
            prompt="Explain TTFT vs total latency in one sentence.",
            stream=True,
            max_output_tokens=128,
        )
        async for event in client.stream_generate(stream_req, provider=provider):
            if event.delta_text:
                print(event.delta_text, end="", flush=True)
            if event.done:
                print(f"\n[done finish_reason={event.finish_reason} ttft_ms={event.ttft_ms} "
                      f"total_ms={event.total_ms} usage={event.usage}]")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Structured output & streaming LLM demo.")
    parser.add_argument("--live", action="store_true", help="Use live .env provider credentials.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic", "ollama"],
        default=None,
        help="Override LLM_PROVIDER.",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(live=args.live, provider=args.provider))


if __name__ == "__main__":
    main()
