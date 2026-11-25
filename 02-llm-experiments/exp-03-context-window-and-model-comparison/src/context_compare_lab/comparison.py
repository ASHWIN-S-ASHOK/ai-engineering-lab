"""Needle-in-a-haystack position probe and multi-provider comparison runner."""

from statistics import mean

from context_compare_lab.config import AppSettings
from context_compare_lab.models import (
    BenchmarkCaseResult,
    ComparisonReport,
    ModelRequest,
    NeedlePosition,
    ProviderName,
    ProviderSummary,
    TokenPricing,
    TokenUsage,
)
from context_compare_lab.providers import ADAPTER_REGISTRY
from context_compare_lab.transport import ResilientHttpTransport

# Reference illustrative rates (USD per 1M tokens) configurable per experiment run.
# Ollama is local ($0.00); Gemini free-tier has $0 billing cost.
DEFAULT_PRICING: dict[ProviderName, TokenPricing] = {
    "gemini": TokenPricing(input_per_million_usd=0.0, output_per_million_usd=0.0),
    "openai": TokenPricing(input_per_million_usd=0.15, output_per_million_usd=0.60),
    "anthropic": TokenPricing(input_per_million_usd=0.80, output_per_million_usd=4.00),
    "ollama": TokenPricing(input_per_million_usd=0.0, output_per_million_usd=0.0),
}

_NEEDLE_FRACTIONS: dict[NeedlePosition, float] = {
    "start": 0.0,
    "middle": 0.5,
    "end": 1.0,
}


def estimate_cost_usd(usage: TokenUsage, pricing: TokenPricing) -> float:
    """Compute estimated request cost in USD from token counts and 1M-token rates."""
    input_cost = (usage.input_tokens / 1_000_000.0) * pricing.input_per_million_usd
    output_cost = (usage.output_tokens / 1_000_000.0) * pricing.output_per_million_usd
    return round(input_cost + output_cost, 8)


def build_needle_haystack(
    *,
    needle_fact: str,
    position: NeedlePosition,
    distractor_count: int = 8,
) -> str:
    """Build synthetic engineering notes with ``needle_fact`` placed at start, middle, or end."""
    distractors = [
        f"Note {idx:02d}: Warehouse scanner batch sync interval is configured for region {idx % 4}."
        for idx in range(1, distractor_count + 1)
    ]
    insert_at = round(_NEEDLE_FRACTIONS[position] * len(distractors))
    lines = [*distractors[:insert_at], needle_fact, *distractors[insert_at:]]
    return "\n".join(lines)


class ModelComparisonRunner:
    """Runs identical position probes across switchable providers and aggregates metrics."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        transport: ResilientHttpTransport,
        pricing: dict[ProviderName, TokenPricing] | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._pricing = pricing or DEFAULT_PRICING

    async def run_needle_benchmark(
        self,
        *,
        providers: list[ProviderName],
        positions: tuple[NeedlePosition, ...] = ("start", "middle", "end"),
        distractor_count: int = 8,
        expected_code: str = "RET-7492",
    ) -> ComparisonReport:
        """Evaluate exact-match needle recall across providers and positions."""
        needle_fact = (
            f"CRITICAL OVERRIDE: The emergency dispatch authorization code is {expected_code}."
        )
        all_results: list[BenchmarkCaseResult] = []
        summaries: list[ProviderSummary] = []

        for provider in providers:
            provider_results: list[BenchmarkCaseResult] = []
            for position in positions:
                context = build_needle_haystack(
                    needle_fact=needle_fact,
                    position=position,
                    distractor_count=distractor_count,
                )
                result = await self._run_single_case(
                    provider=provider,
                    position=position,
                    context=context,
                    expected_answer=expected_code,
                )
                provider_results.append(result)
                all_results.append(result)

            summaries.append(self._summarize(provider=provider, results=provider_results))

        return ComparisonReport(summaries=summaries, results=all_results)

    async def _run_single_case(
        self,
        *,
        provider: ProviderName,
        position: NeedlePosition,
        context: str,
        expected_answer: str,
    ) -> BenchmarkCaseResult:
        runtime, api_key = self._settings.resolve_provider(
            provider=provider,
            require_credentials=True,
        )
        adapter = ADAPTER_REGISTRY[provider]
        req = ModelRequest(
            system_prompt=(
                "Answer using only the provided notes. Return only the authorization code."
            ),
            prompt=(
                f"Notes:\n{context}\n\n"
                "Question: What is the emergency dispatch authorization code?"
            ),
            temperature=0.0,
            max_output_tokens=32,
        )
        prepared = adapter.build_generate_request(req, runtime, api_key)
        payload, latency_ms, attempts = await self._transport.post_json(prepared)
        response = adapter.parse_generate_response(
            payload,
            requested_model=runtime.model,
            latency_ms=latency_ms,
            attempts=attempts,
        )
        matched = expected_answer in response.text
        cost = estimate_cost_usd(response.usage, self._pricing[provider])
        return BenchmarkCaseResult(
            provider=provider,
            model=response.model,
            position=position,
            expected_answer=expected_answer,
            response_text=response.text.strip(),
            matched=matched,
            latency_ms=response.latency_ms,
            usage=response.usage,
            estimated_cost_usd=cost,
        )

    @staticmethod
    def _summarize(
        *,
        provider: ProviderName,
        results: list[BenchmarkCaseResult],
    ) -> ProviderSummary:
        passed = sum(1 for r in results if r.matched)
        return ProviderSummary(
            provider=provider,
            model=results[0].model,
            cases_run=len(results),
            cases_passed=passed,
            accuracy=round(passed / len(results), 4),
            avg_latency_ms=round(mean(r.latency_ms for r in results), 2),
            total_input_tokens=sum(r.usage.input_tokens for r in results),
            total_output_tokens=sum(r.usage.output_tokens for r in results),
            total_estimated_cost_usd=round(sum(r.estimated_cost_usd for r in results), 8),
        )
