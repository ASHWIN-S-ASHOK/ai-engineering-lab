"""Context window budgeting, preflight token counting, and model comparison lab."""

from context_compare_lab.budget import ContextBudgetManager, estimate_tokens
from context_compare_lab.comparison import (
    DEFAULT_PRICING,
    ModelComparisonRunner,
    build_needle_haystack,
    estimate_cost_usd,
)
from context_compare_lab.config import AppSettings
from context_compare_lab.models import (
    BenchmarkCaseResult,
    ComparisonReport,
    ContextBudgetExceededError,
    ContextChunk,
    ContextCompareError,
    NeedlePosition,
    NonRetryableApiError,
    OverflowPolicy,
    PackedContext,
    ProviderName,
    ProviderSummary,
    RetryableApiError,
    TokenPricing,
    TokenUsage,
)
from context_compare_lab.providers import ADAPTER_REGISTRY, ContextModelAdapter

__all__ = [
    "ADAPTER_REGISTRY",
    "AppSettings",
    "BenchmarkCaseResult",
    "ComparisonReport",
    "ContextBudgetExceededError",
    "ContextBudgetManager",
    "ContextChunk",
    "ContextCompareError",
    "ContextModelAdapter",
    "DEFAULT_PRICING",
    "ModelComparisonRunner",
    "NeedlePosition",
    "NonRetryableApiError",
    "OverflowPolicy",
    "PackedContext",
    "ProviderName",
    "ProviderSummary",
    "RetryableApiError",
    "TokenPricing",
    "TokenUsage",
    "build_needle_haystack",
    "estimate_cost_usd",
    "estimate_tokens",
]
