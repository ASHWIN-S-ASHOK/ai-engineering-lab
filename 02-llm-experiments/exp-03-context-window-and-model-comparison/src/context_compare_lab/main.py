"""CLI demonstration of context window packing and multi-provider comparison."""

import argparse
import asyncio
import json
import logging

import httpx

from context_compare_lab.budget import ContextBudgetManager
from context_compare_lab.comparison import ModelComparisonRunner
from context_compare_lab.config import AppSettings
from context_compare_lab.models import ContextChunk, ProviderName
from context_compare_lab.transport import ResilientHttpTransport


def _build_offline_transport() -> httpx.MockTransport:
    """Simulate Gemini (3/3 recall) vs Ollama (misses middle needle under small context)."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))

        if "generativelanguage.googleapis.com" in url:
            prompt_text = body["contents"][0]["parts"][0]["text"]
            answer = "RET-7492" if "RET-7492" in prompt_text else "UNKNOWN"
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": answer}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 210,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 214,
                    },
                    "modelVersion": "gemini-2.5-flash",
                },
            )

        # Simulated local Ollama demonstrating a "lost in the middle" miss when needle is centered.
        user_msg = body["messages"][-1]["content"]
        lines = [
            line
            for line in user_msg.splitlines()
            if line.startswith("Note ") or "CRITICAL" in line
        ]
        mid_idx = len(lines) // 2
        middle_only = mid_idx > 1 and "CRITICAL" in lines[mid_idx]
        answer = "I cannot verify the code from those notes." if middle_only else "RET-7492"
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "done_reason": "stop",
                "message": {"role": "assistant", "content": answer},
                "prompt_eval_count": 210,
                "eval_count": 6,
            },
        )

    return httpx.MockTransport(handler)


async def run_demo(*, live: bool, providers: list[ProviderName]) -> None:
    """Run context budget demonstration + needle-in-a-haystack model comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("=== 1. Context Window Budget Packing (drop_middle policy) ===")
    manager = ContextBudgetManager()
    chunks = [
        ContextChunk(chunk_id=f"doc-{i}", text=f"Chunk {i}: " + ("retail telemetry " * 12))
        for i in range(1, 7)
    ]
    packed = manager.pack(
        system_prompt="Answer concisely using provided context.",
        user_query="Summarize telemetry anomalies.",
        chunks=chunks,
        context_limit_tokens=260,
        reserved_output_tokens=64,
        policy="drop_middle",
    )
    print(packed.model_dump_json(indent=2))

    print("\n=== 2. Needle-in-a-Haystack Model Comparison ===")
    settings = (
        AppSettings()
        if live
        else AppSettings(_env_file=None, gemini_api_key="demo-gemini-key")
    )
    mock = None if live else _build_offline_transport()
    async with ResilientHttpTransport(
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        transport=mock,
    ) as transport:
        runner = ModelComparisonRunner(settings=settings, transport=transport)
        report = await runner.run_needle_benchmark(providers=providers, distractor_count=8)

    print(report.model_dump_json(indent=2))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Context budgeting and model comparison lab.")
    parser.add_argument("--live", action="store_true", help="Call live providers from .env.")
    parser.add_argument(
        "--providers",
        default="gemini,ollama",
        help="Comma-separated providers to compare (gemini,openai,anthropic,ollama).",
    )
    args = parser.parse_args()
    selected: list[ProviderName] = [p.strip() for p in args.providers.split(",") if p.strip()]  # type: ignore[assignment]
    asyncio.run(run_demo(live=args.live, providers=selected))


if __name__ == "__main__":
    main()
