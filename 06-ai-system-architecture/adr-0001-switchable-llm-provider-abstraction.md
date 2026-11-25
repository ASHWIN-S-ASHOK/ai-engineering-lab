# ADR-0001: Switchable Multi-Provider LLM Client Abstraction

**Status:** `Accepted`  
**Date:** `2026-11-25`  
**Scope:** `02-llm-experiments/exp-01-switchable-llm-client` (and early Track 02–03 experiments)  
**Related ADRs:** None (foundational LLM transport decision)

---

## 1. Context & Problem Statement

Our AI Engineering Lab requires a single client interface that can switch via `.env` configuration among four model backends:
- **Google Gemini API** (default free-tier cloud API via Google AI Studio)
- **OpenAI**
- **Anthropic**
- **Ollama** (zero-cost local inference on Apple Silicon)

In Android/enterprise architecture, we separate domain logic from network transport using interfaces and repository adapters. We need to choose **how** our Python experiments talk to these four LLM backends without locking our learning code into one vendor or hiding the underlying HTTP mechanics too early.

---

## 2. Decision Drivers

1. **Conceptual clarity:** Understand the actual HTTP request/response shapes, token usage fields, and error behaviors of each provider before relying on high-level abstractions.
2. **Minimal dependencies:** Keep early Track `02-llm-experiments` packages lightweight and fast to install with `uv`.
3. **Provider switchability:** Allow changing `LLM_PROVIDER=gemini|openai|anthropic|ollama` in `.env` with zero changes to calling code.
4. **Deterministic offline testing:** Support 100% offline unit testing via `httpx.MockTransport` without requiring 4 live API keys in CI.
5. **Reversibility:** Make it trivial to swap an adapter to an official SDK later if advanced features (e.g., complex multimodal streaming or provider-managed batch APIs) demand it.

---

## 3. Considered Options

### Option A: Thin Direct `httpx` REST Adapters Behind a `Protocol` *(Recommended)*
Each provider (`gemini`, `openai`, `anthropic`, `ollama`) is implemented as a small adapter (`~40–60` lines) using our existing `httpx` transport and `pydantic` models, normalizing requests/responses into one `LLMRequest` / `LLMResponse` contract.

- **Pros:**
  - Zero new SDK dependencies beyond `httpx` and `pydantic`.
  - Exposes real wire-level differences (auth headers, `systemInstruction` vs. `system` vs. `instructions`, `usageMetadata` vs. `usage`, `eval_count`).
  - Reuses the exact `httpx.MockTransport` offline testing pattern from Experiment 1.
- **Cons:**
  - Requires reading and keeping up with each provider's REST payload schema manually.
  - Complex provider-specific features (e.g., OpenAI Realtime WebSockets or provider-specific file upload helpers) take more manual code later.

### Option B: Four Official Provider SDKs (`google-genai`, `openai`, `anthropic`, `ollama`) Behind a `Protocol`
Install all four official Python SDKs and wrap each inside a thin adapter class.

- **Pros:**
  - Official SDKs maintain request/response types, SSE helpers, and retry logic per vendor.
  - Less manual JSON parsing code.
- **Cons:**
  - Pulls four separate SDK dependency trees into small Track 02 experiments.
  - Each SDK uses different mocking patterns and exception hierarchies, making uniform offline unit tests harder.
  - Hides the raw HTTP payloads that are valuable to learn early on.

### Option C: Unified Third-Party Wrapper (`LiteLLM` or `LangChain`)
Adopt a multi-provider abstraction library immediately.

- **Pros:**
  - Single API call across dozens of providers out of the box.
- **Cons:**
  - Adds a large dependency footprint and hides provider differences, token accounting nuances, and failure modes—working directly against our "understand before abstracting" learning goal.

---

## 4. Decision Outcome

**Chosen Option:** **Option A — Thin Direct `httpx` REST Adapters Behind a `Protocol`** (for Track `02-llm-experiments` foundational labs).

**Rationale:**  
Because our goal is to build deep architectural intuition as an AI-enabled Solution Architect, seeing the exact HTTP wire differences across Gemini (`POST /v1beta/models/{model}:generateContent`), OpenAI (`POST /v1/responses`), Anthropic (`POST /v1/messages`), and Ollama (`POST /api/chat`) pays off immediately when debugging latency, token discrepancies, and schema failures. Keeping the `LLMProvider` `Protocol` boundary clean means we can swap any individual adapter to Option B (an official SDK) later without changing callers.

---

## 5. Consequences

### Positive
- Direct continuity from `01-python-ai-fundamentals/exp-01-async-api-and-structured-data`.
- Consistent `httpx.MockTransport` test suite across all four providers.
- Lightweight `uv` environments.

### Negative / Technical Debt Accepted
- Adapter payload builders and response parsers must explicitly match each provider's current REST schema and be verified against official API docs.

### Follow-up Actions
- Implement `02-llm-experiments/exp-01-switchable-llm-client` using this `Protocol` + adapter layout.
- Document any provider-specific quirks observed during live Gemini Free Tier and local Ollama testing.

---

## 6. Architecture Quality Attributes

- **Security & Privacy:** API keys remain wrapped in `pydantic.SecretStr` and injected only into per-request headers (`x-goog-api-key`, `Authorization: Bearer ...`, `x-api-key`); never logged. Free-tier Gemini usage is restricted to synthetic/public prompts.
- **Scalability & Reliability:** Reuses bounded timeouts and exponential backoff retries on `429` (rate limit) and `5xx` errors; non-retryable `4xx` errors fail fast.
- **Cost & Performance:** Normalizes `input_tokens` and `output_tokens` across providers (`usageMetadata` vs. `usage` vs. `prompt_eval_count`/`eval_count`) alongside wall-clock `latency_ms` so cost/latency comparisons are apples-to-apples.

---

## 7. Validation & Revisit Criteria

- **Validation:** Offline tests verify request serialization, response parsing, token normalization, and error classification for all 4 providers; optional live checks verify at least Gemini Free Tier and/or local Ollama end-to-end.
- **Revisit Trigger:** Upgrade a specific adapter to its official SDK (Option B) when an experiment requires complex multimodal file uploads, tool-streaming state machines, or provider-native caching APIs where raw `httpx` becomes error-prone.
