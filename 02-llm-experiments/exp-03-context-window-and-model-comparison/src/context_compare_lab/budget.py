"""Context window budget packing and preflight token counting."""

from collections.abc import Callable
from math import ceil

from context_compare_lab.config import AppSettings
from context_compare_lab.models import (
    ContextBudgetExceededError,
    ContextChunk,
    ModelRequest,
    OverflowPolicy,
    PackedContext,
    ProviderName,
)
from context_compare_lab.providers import ADAPTER_REGISTRY
from context_compare_lab.transport import ResilientHttpTransport


def estimate_tokens(text: str) -> int:
    """Estimate token count conservatively (~3.5 chars/token + framing overhead)."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, ceil(len(stripped) / 3.5))


def _drop_oldest_index(kept: list[ContextChunk]) -> int:
    del kept
    return 0


def _drop_middle_index(kept: list[ContextChunk]) -> int:
    return len(kept) // 2


_DROP_INDEX_SELECTOR: dict[str, Callable[[list[ContextChunk]], int]] = {
    "drop_oldest": _drop_oldest_index,
    "drop_middle": _drop_middle_index,
}


class ContextBudgetManager:
    """Packs ordered chunks into a token budget using explicit overflow policies."""

    def __init__(self, *, token_counter: Callable[[str], int] = estimate_tokens) -> None:
        self._count = token_counter

    def pack(
        self,
        *,
        system_prompt: str,
        user_query: str,
        chunks: list[ContextChunk],
        context_limit_tokens: int,
        reserved_output_tokens: int,
        policy: OverflowPolicy = "drop_middle",
    ) -> PackedContext:
        """Fit ``chunks`` into ``context_limit_tokens - reserved_output_tokens``."""
        available_input = context_limit_tokens - reserved_output_tokens
        base_tokens = self._count(system_prompt) + self._count(user_query)
        if base_tokens > available_input:
            raise ContextBudgetExceededError(
                f"Base prompt ({base_tokens} tokens) exceeds available input budget "
                f"({available_input} tokens)."
            )

        kept = list(chunks)
        dropped_ids: list[str] = []

        while self._total_tokens(base_tokens, kept) > available_input:
            if policy == "fail_fast" or not kept:
                raise ContextBudgetExceededError(
                    f"Prompt + {len(kept)} chunks ({self._total_tokens(base_tokens, kept)} tokens) "
                    f"exceed available input budget ({available_input} tokens)."
                )
            idx = _DROP_INDEX_SELECTOR[policy](kept)
            removed = kept.pop(idx)
            dropped_ids.append(removed.chunk_id)

        return PackedContext(
            kept_chunks=kept,
            dropped_chunk_ids=dropped_ids,
            estimated_input_tokens=self._total_tokens(base_tokens, kept),
            reserved_output_tokens=reserved_output_tokens,
            budget_limit_tokens=context_limit_tokens,
            policy=policy,
        )

    def _total_tokens(self, base_tokens: int, chunks: list[ContextChunk]) -> int:
        return base_tokens + sum(self._count(chunk.text) for chunk in chunks)


async def preflight_count_tokens(
    *,
    settings: AppSettings,
    transport: ResilientHttpTransport,
    request: ModelRequest,
    provider: ProviderName | None = None,
) -> int | None:
    """Query provider preflight token counter (Gemini/Anthropic) or return None."""
    runtime, api_key = settings.resolve_provider(provider=provider, require_credentials=True)
    adapter = ADAPTER_REGISTRY[runtime.provider]
    prepared = adapter.build_count_tokens_request(request, runtime, api_key)
    if prepared is None:
        return None
    payload, _, _ = await transport.post_json(prepared)
    return adapter.parse_count_tokens_response(payload)
