# 6-Month AI Engineering & Solution Architecture Roadmap

**Target Pace:** ~6–8 hours/week over 24 weeks (6 months)  
**Goal:** Build deep, practical competence across Python, LLMs, RAG, AI agents, Android on-device AI, and AI system architecture—documented through tested experiments, POCs, and Architecture Decision Records (ADRs).

---

## Guiding Principles

1. **Understand before coding:** Learn the problem, mechanics, and alternatives before writing an integration.
2. **Avoid premature frameworks:** Build raw API and math/retrieval baselines first so framework abstractions make sense later.
3. **Bridge existing strengths:** Connect Android, Clean Architecture, API design, and distributed systems instincts to AI architecture.
4. **Test and measure:** Every experiment includes automated tests, explicit inputs/outputs, and honest limitations.
5. **Document decisions:** Record *why* a technology or pattern was chosen using ADRs.

---

## Phase Overview

| Month | Weeks | Primary Track(s) | Core Theme | Key Deliverable |
| --- | --- | --- | --- | --- |
| **1** | 1–4 | `01-python-ai-fundamentals`, `02-llm-experiments` | Pythonic AI foundations & raw LLM mechanics | Typed async Python lab + Raw LLM client, structured output & streaming labs |
| **2** | 5–8 | `03-rag-systems` | Embeddings, retrieval & grounded generation | Similarity from scratch + Grounded RAG POC with citations & evals |
| **3** | 9–12 | `04-ai-agents` | Tool calling, state machines & guardrails | Deterministic tool-calling agent POC with human-approval gate |
| **4** | 13–16 | `05-android-on-device-ai` | Mobile edge AI & hybrid routing | Android on-device AI experiment + Edge vs. Cloud ADR |
| **5** | 17–20 | `06-ai-system-architecture`, `architecture/` | Security, reliability, cost & observability | Reference architecture, threat model & latency/cost study |
| **6** | 21–24 | Cross-track Capstone | Mobile + AI BFF + Eval integration | End-to-end Capstone POC + full portfolio review |

---

## Month-by-Month Plan

### Month 1 (Weeks 1–4): Python AI Foundations & LLM API Mechanics
**Tracks:** `01-python-ai-fundamentals/`, `02-llm-experiments/`

- **Concepts:**
  - Python type hints, `pydantic` models, `asyncio` + `httpx`, structured logging, `pytest` fixtures/mocking.
  - Tokens, context windows, temperature/sampling, system vs. user roles, stateless LLM APIs.
  - Structured JSON outputs (schema enforcement + validation) and Server-Sent Events (SSE) streaming.
- **Experiments:**
  - `exp-01-async-api-and-structured-data` *(Experiment)*
  - `exp-01-switchable-llm-client` *(Experiment)*
  - `exp-02-structured-outputs-and-streaming` *(Experiment)*
  - `exp-03-context-window-and-model-comparison` *(Experiment)*
- **Checkpoint:** Can explain token costs, latency (TTFT vs. total time), schema validation failures, and retry strategies without relying on wrapper frameworks.

### Month 2 (Weeks 5–8): Retrieval-Augmented Generation (RAG) & Evaluation
**Track:** `03-rag-systems/`

- **Concepts:**
  - Dense embeddings, cosine similarity vs. dot product, semantic vs. lexical (BM25) search.
  - Document ingestion, chunking strategies (fixed, recursive, semantic/structural), metadata filtering.
  - Vector databases (trade-offs: in-memory/local vs. managed) and retrieval evaluation (`precision@k`, `recall@k`, groundedness/faithfulness).
- **Experiments:**
  - `exp-01-embeddings-and-similarity-from-scratch` *(Experiment)*
  - `exp-02-grounded-rag-with-citations-and-evals` *(Proof of Concept)*
  - `exp-03-chunking-and-hybrid-search-comparison` *(Experiment)*
- **Checkpoint:** ADR comparing chunking strategies and vector store choices; reproducible eval harness showing retrieval improvements quantitatively.

### Month 3 (Weeks 9–12): AI Agents, Tool Calling & Safety Boundaries
**Track:** `04-ai-agents/`

- **Concepts:**
  - Function/tool calling protocol, JSON schema tool contracts, tool-execution loops.
  - Workflow state machines vs. autonomous loops (when *not* to use an open-ended agent).
  - Human-in-the-loop (HITL) approval gates for side-effecting actions, max-step budgets, prompt-injection defense boundaries.
- **Experiments:**
  - `exp-01-tool-calling-router` *(Experiment)*
  - `exp-02-bounded-agent-with-human-approval` *(Proof of Concept)*
- **Checkpoint:** Working multi-step tool workflow with explicit step budgets, audit logs, and mocked failure/injection test cases.

### Month 4 (Weeks 13–16): Android On-Device AI & Edge–Cloud Trade-offs
**Track:** `05-android-on-device-ai/`

- **Concepts:**
  - Current Android on-device GenAI capabilities (e.g., Gemini Nano / AICore / ML Kit GenAI APIs — verified against official Android documentation at implementation time).
  - Device constraints: memory, thermal throttling, model availability, cold-start latency, APK/model download impact.
  - Hybrid routing architecture: privacy-sensitive/offline tasks on-device vs. complex reasoning/RAG in cloud.
- **Experiments:**
  - `exp-01-android-on-device-summarizer` *(Experiment — Kotlin/Jetpack Compose)*
  - `exp-02-hybrid-edge-cloud-decision-study` *(Architecture Study + ADR)*
- **Checkpoint:** Clear decision matrix and ADR tailored to enterprise mobile apps (e.g., logistics/retail field scenarios).

### Month 5 (Weeks 17–20): AI System Architecture, Security & FinOps
**Track:** `06-ai-system-architecture/`, `architecture/`

- **Concepts:**
  - AI Gateway / Backend-for-Frontend (BFF) pattern for mobile + AI clients.
  - Security & privacy: OWASP Top 10 for LLM Applications, PII redaction, tenant isolation in RAG, output validation.
  - Reliability & FinOps: semantic/exact caching, fallback models, circuit breakers, token budgeting, observability/tracing.
- **Deliverables:**
  - `ref-01-mobile-ai-gateway-architecture` *(Architecture Study + Mermaid diagrams)*
  - Security & Privacy Threat Model *(Architecture Study)*
  - Cost, Latency & Scalability Model *(Architecture Study)*
- **Checkpoint:** Recruiter- and architect-grade system design package with ADRs, Mermaid diagrams, and failure-mode tables.

### Month 6 (Weeks 21–24): Capstone POC & Portfolio Audit
**Track:** Cross-track Reference Implementation / POC

- **Deliverable:**
  - One cohesive **Capstone Proof of Concept** (e.g., *Field Operations / Retail Knowledge Assistant*: lightweight Android/API client -> Python AI BFF -> grounded RAG + bounded tool calling + structured logging + automated evals).
- **Portfolio Audit (Phase 6):**
  - End-to-end review from the perspective of a Principal Engineer / Solution Architect interviewer: verify every README, test suite, ADR, diagram, and security boundary.
  - Trim noise, refine root README navigation, and ensure every claim is backed by observed code and test output.
