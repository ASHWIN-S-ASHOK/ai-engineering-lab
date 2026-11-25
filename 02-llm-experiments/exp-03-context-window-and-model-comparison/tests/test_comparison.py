"""Unit tests for needle-in-a-haystack positioning and multi-provider comparison."""

import json

import httpx

from context_compare_lab.comparison import (
    ModelComparisonRunner,
    build_needle_haystack,
    estimate_cost_usd,
)
from context_compare_lab.config import AppSettings
from context_compare_lab.models import TokenPricing, TokenUsage
from context_compare_lab.transport import ResilientHttpTransport


def test_build_needle_haystack_positions() -> None:
    needle = "NEEDLE: SECRET-99"
    start_lines = build_needle_haystack(
        needle_fact=needle, position="start", distractor_count=4
    ).splitlines()
    mid_lines = build_needle_haystack(
        needle_fact=needle, position="middle", distractor_count=4
    ).splitlines()
    end_lines = build_needle_haystack(
        needle_fact=needle, position="end", distractor_count=4
    ).splitlines()

    assert start_lines[0] == needle
    assert mid_lines[2] == needle
    assert end_lines[-1] == needle


def test_estimate_cost_usd() -> None:
    usage = TokenUsage(input_tokens=100_000, output_tokens=50_000, total_tokens=150_000)
    pricing = TokenPricing(input_per_million_usd=0.20, output_per_million_usd=0.60)
    assert estimate_cost_usd(usage, pricing) == 0.05


async def test_model_comparison_runner_scores_accuracy_and_cost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        if "/responses" in url:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "RET-7492"}],
                        }
                    ],
                    "usage": {"input_tokens": 1000, "output_tokens": 10, "total_tokens": 1010},
                },
            )

        user_msg = body["messages"][-1]["content"]
        lines = [ln for ln in user_msg.splitlines() if ln.startswith("Note ") or "CRITICAL" in ln]
        mid_idx = len(lines) // 2
        missed_middle = "CRITICAL" in lines[mid_idx]
        reply = "Not found" if missed_middle else "RET-7492"
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "done_reason": "stop",
                "message": {"role": "assistant", "content": reply},
                "prompt_eval_count": 1000,
                "eval_count": 10,
            },
        )

    settings = AppSettings(_env_file=None, openai_api_key="oai-key")
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    async with transport:
        runner = ModelComparisonRunner(settings=settings, transport=transport)
        report = await runner.run_needle_benchmark(
            providers=["openai", "ollama"],
            distractor_count=6,
        )

    assert len(report.summaries) == 2
    openai_summary, ollama_summary = report.summaries
    assert openai_summary.accuracy == 1.0
    assert openai_summary.total_estimated_cost_usd > 0.0
    assert ollama_summary.cases_passed == 2
    assert ollama_summary.accuracy == round(2 / 3, 4)
    assert ollama_summary.total_estimated_cost_usd == 0.0
