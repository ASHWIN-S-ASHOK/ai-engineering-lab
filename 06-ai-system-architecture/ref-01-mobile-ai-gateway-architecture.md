# Reference Architecture: Mobile-to-AI Gateway (BFF) for Enterprise Field Operations

**Classification:** `Architecture Study`  
**Track:** `06-ai-system-architecture`  
**Status:** `Completed`  
**Last Updated:** `2026-11-26`

---

## 1. Problem Statement

Enterprise logistics and retail field apps cannot connect mobile clients directly to third-party LLM APIs:
1. Shipping API keys inside an APK exposes credentials to extraction.
2. Client apps need a unified orchestration boundary for hybrid RAG, tool authorization, audit logging, and PII controls.
3. Field devices operate across mixed hardware tiers and intermittent connectivity—requiring coordinated **on-device** and **cloud** routing.

This reference architecture unifies the components built across Tracks `01`–`05` into a single, vendor-portable system design.

---

## 2. Requirements

### Functional Requirements
- **Hybrid Mobile Routing:** Run privacy-sensitive or offline short summaries on-device via Android AICore (`05`) when `FeatureStatus.AVAILABLE`, with graceful cloud or extractive fallback.
- **Grounded Enterprise Knowledge Q&A:** Support hybrid retrieval (`BM25 + dense cosine` via RRF) with mandatory citation verification and abstention (`03`).
- **Human-Approved Operations:** Pause mutating agent actions (`issue_customer_refund`, `dispatch_replacement_shipment`) in `PENDING_APPROVAL` until an authorized supervisor resumes them (`04`).
- **Vendor Portability:** Route across Gemini, OpenAI, Anthropic, or private Ollama via configuration without changing mobile client contracts (`02`).

### Non-Functional Requirements
- **Latency:** Sub-second local edge response on warm Gemini Nano; streaming SSE TTFT `< 1.0s` for cloud responses (subject to network/provider conditions).
- **Security:** Zero provider API keys on mobile devices; tenant isolation at retrieval layer; deterministic policy caps on financial actions.
- **Reliability:** Bounded timeouts, exponential backoff retries with jitter on `429`/`5xx`, and circuit-breaker fallback to secondary providers.

---

## 3. High-Level Architecture & Diagrams

- **System Context & Component Topology:** [system-mobile-ai-gateway.md](../architecture/system-mobile-ai-gateway.md)
- **Sequence & Data Flows (Hybrid RAG + HITL Pause/Resume):** [dataflow-rag-and-hitl-approval.md](../architecture/dataflow-rag-and-hitl-approval.md)
- **Deployment Topology:** [deployment-edge-cloud-topology.md](../architecture/deployment-edge-cloud-topology.md)

### Component Responsibilities
1. **Android Client (`HybridSummarizerUseCase` + `MlKitGenAiPromptAdapter`):** Inspects `FeatureStatus`, manages AICore lifecycle (`.close()`), enforces local privacy flags, and talks solely to the BFF over HTTPS.
2. **AI Gateway / BFF (Python):** Authenticates mobile OIDC tokens, redacts/tags PII, enforces context token budgets (`drop_middle` / `fail_fast`), and orchestrates RAG + HITL state transitions.
3. **Hybrid Retriever (`Retriever` Protocol):** Combines lexical `BM25` and dense cosine similarity via RRF (`k=60`), scoped strictly by `tenant_id` / user permissions.
4. **HITL State Store (`AgentRunState`):** Persists paused runs and immutable `AuditEntry` records across asynchronous mobile approval interactions.
5. **Switchable Model Router (`Protocol` + Adapter Registry):** Normalizes requests, structured JSON outputs, streaming, and token usage across Gemini, OpenAI, Anthropic, and Ollama.

---

## 4. Related Architecture Decision Records

- [ADR-0001](./adr-0001-switchable-llm-provider-abstraction.md) — Switchable multi-provider LLM abstraction.
- [ADR-0002](./adr-0002-hybrid-retrieval-and-local-vector-store.md) — Hybrid retrieval (BM25 + dense vectors + RRF) and migration path to `pgvector` / Qdrant.
- [ADR-0003](./adr-0003-hitl-approval-state-machine.md) — Pause-and-resume state machine for HITL tool authorization.
- [ADR-0004](./adr-0004-android-hybrid-edge-cloud-routing.md) — Hybrid edge–cloud AI routing for Android fleets.

---

## 5. Scalability, Reliability & Failure Modes

| Failure Mode | Mitigation Pattern | Reference Lab |
| --- | --- | --- |
| Cloud provider `429` / `5xx` spike | Bounded exponential backoff + jitter + secondary provider failover | `01-python.../exp-01`, `02-llm.../exp-01` |
| Gemini Nano `UNAVAILABLE` on older phones | Clean fallback to BFF or deterministic offline extractive summary | `05-android.../exp-01` |
| Hallucinated RAG citations | Strict JSON Schema + programmatic `verify_citations()` + abstention | `03-rag.../exp-02` |
| Indirect prompt injection triggering refund | `<untrusted_tool_data>` isolation + `MAX_REFUND_USD` + `PENDING_APPROVAL` | `04-ai-agents/exp-02` |
| Context overflow / silent truncation | Reserved output budget + preflight `:countTokens` / `count_tokens` | `02-llm.../exp-03` |

---

## 6. Limitations & Migration Path

- **Local POC persistence vs. production:** Our POCs use SQLite (`FTS5` + NumPy) and in-memory `AgentRunState`; production scale replaces `SqliteHybridStore` behind the `Retriever` `Protocol` with `pgvector` or Qdrant and stores `AgentRunState` in Postgres/Redis with TTLs.
- **Next step:** Connect a minimal runnable Python BFF + Android client contract in the **Month 6 Capstone** (`README.md` / `ROADMAP.md`).
