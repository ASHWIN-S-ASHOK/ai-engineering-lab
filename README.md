# Ashwin's AI Engineering Labs

> **Repository:** `ai-engineering-lab`  
> A personal engineering laboratory exploring Python, Large Language Models (LLMs), Generative AI, Retrieval-Augmented Generation (RAG), AI agents, Android on-device AI, and AI system architecture through working experiments, proofs of concept (POCs), and documented architecture studies.

---

## Disclaimer & Scope

All AI/GenAI work in this repository reflects **personal learning, hands-on experimentation, proofs of concept, and architecture studies**. None of the projects, benchmarks, or designs here represent production client deliveries or enterprise deployments.

---

## About This Repository

This repository is structured as a disciplined engineering notebook and portfolio rather than a collection of throwaway scripts. Every experiment starts with a clear problem statement, architectural trade-offs, working typed Python (or Kotlin/Android) code, automated tests, and honest documentation of limitations and lessons learned.

Each entry is explicitly tagged by scope:

- **Experiment** — Focused investigation of a single concept, API, or trade-off.
- **Proof of Concept (POC)** — Working multi-component prototype validating feasibility.
- **Reference Implementation** — Clean, reusable baseline pattern with tests and observability.
- **Architecture Study** — System design, threat model, cost/latency analysis, and Architecture Decision Records (ADRs).

---

## Engineering Background & Learning Direction

I am a **Technical Architect** with ~14 years of software engineering experience, primarily in enterprise **Android and mobile architecture** (large-scale logistics and retail applications).

- **Core Strengths:** Kotlin, Java, Android Jetpack & Jetpack Compose, MVVM, Clean Architecture, SOLID principles, REST/GraphQL/real-time APIs, CI/CD, and technical design & architecture reviews.
- **Active Expansion Areas:** Python engineering, ML/AI fundamentals, LLM APIs & prompt contracts, RAG & vector retrieval, AI agents with safety boundaries, on-device Android AI, and cloud-native AI system architecture.
- **Long-Term Objective:** Grow into an **AI-enabled Solution Architect** who bridges mobile edge clients, backend/cloud services, and practical AI capabilities with strong architectural rigor.

---

## Technology Areas Explored

Folders are created progressively as experiments begin (no empty placeholder directories are committed).

| Area | Directory | Focus | Status |
| --- | --- | --- | --- |
| Python & AI Fundamentals | [`01-python-ai-fundamentals/`](./01-python-ai-fundamentals/) | Typed Python, Pydantic, async APIs, data handling, core ML concepts | Active |
| LLM Experiments | [`02-llm-experiments/`](./02-llm-experiments/) | Direct LLM APIs, prompt engineering, structured outputs, streaming, context limits | Active |
| RAG Systems | [`03-rag-systems/`](./03-rag-systems/) | Ingestion, chunking, embeddings, vector search, grounding, citations, retrieval evals | Active |
| AI Agents | [`04-ai-agents/`](./04-ai-agents/) | Tool/function calling, bounded workflows, human-in-the-loop, failure & safety controls | Active |
| Android On-Device AI | `05-android-on-device-ai/` | Gemini Nano / AICore, ML Kit GenAI, on-device vs. cloud routing, offline & privacy | Planned |
| AI System Architecture | [`06-ai-system-architecture/`](./06-ai-system-architecture/) | Reference architectures, ADRs, OWASP LLM security, cost/latency/reliability analysis | Active |
| System Diagrams | `architecture/` | Cross-project system, data-flow, and deployment diagrams (Mermaid) | Planned |
| Notes & Templates | `docs/` | Learning notes, technical investigations, and standard templates | Active |
| Visual Assets | `assets/` | Architecture exports and demo captures (added only when valuable) | Planned |

---

## Repository Navigation

- [6-Month Learning & Implementation Roadmap](./ROADMAP.md) — Phased plan, milestones, and review checkpoints.
- [Engineering & Project Conventions](./docs/conventions.md) — Python style, project layout, testing, security, and Git standards.
- [Experiment / Project README Template](./docs/templates/project-readme-template.md) — Standard 12-point structure used for every lab entry.
- [Architecture Decision Record (ADR) Template](./docs/templates/adr-template.md) — Template for documenting technology selections and trade-offs.

---

## Featured Experiments & Architecture Case Studies

- **Experiment — [Typed Async API Client & Switchable Provider Configuration](./01-python-ai-fundamentals/exp-01-async-api-and-structured-data/README.md):** `pydantic-settings` + `SecretStr` multi-provider config (`gemini`, `openai`, `anthropic`, `ollama`), `Protocol`-based provider registry, and resilient `httpx` client with 18 offline `pytest` cases.
- **Experiment — [Switchable Multi-Provider LLM Client](./02-llm-experiments/exp-01-switchable-llm-client/README.md):** Thin direct `httpx` REST adapters behind an `LLMProvider` `Protocol` normalizing Gemini, OpenAI Responses, Anthropic Messages, and local Ollama (`16` offline `pytest` cases + optional `--live` mode).
- **Experiment — [Structured Outputs & Streaming Responses](./02-llm-experiments/exp-02-structured-outputs-and-streaming/README.md):** Strict JSON Schema extraction (`Pydantic` validation + truncation handling) and unified SSE/NDJSON streaming with TTFT metrics (`13` offline `pytest` cases + optional `--live` mode).
- **Experiment — [Context Window Budgeting & Model Comparison](./02-llm-experiments/exp-03-context-window-and-model-comparison/README.md):** Context overflow policies (`fail_fast`, `drop_oldest`, `drop_middle`), preflight token counting (`:countTokens`, `/messages/count_tokens`), Ollama `num_ctx`, and needle-in-a-haystack comparison (`10` offline `pytest` cases + optional `--live` mode).
- **Experiment — [Embeddings & Cosine Similarity from Scratch](./03-rag-systems/exp-01-embeddings-and-similarity-from-scratch/README.md):** Overlapping & paragraph chunkers, switchable Gemini/OpenAI/Ollama embedders, pure NumPy L2-norm + cosine search, and `recall@k` / `MRR` diagnostics (`12` offline `pytest` cases + optional `--live` mode).
- **Proof of Concept — [Grounded RAG with Hybrid Retrieval, Citations & Evals](./03-rag-systems/exp-02-grounded-rag-with-citations-and-evals/README.md):** SQLite `FTS5` BM25 + NumPy cosine via RRF, programmatic citation verification, abstention on missing evidence, and dense-vs-hybrid evals (`9` offline `pytest` cases + optional `--live` mode).
- **Experiment — [Bounded Multi-Provider Tool-Calling Router](./04-ai-agents/exp-01-tool-calling-router/README.md):** Allow-listed `ToolRegistry` with Pydantic validation, multi-turn Gemini/OpenAI/Anthropic/Ollama tool adapters, `max_steps` cap, and audit traces (`11` offline `pytest` cases + optional `--live` mode).
- **Proof of Concept — [Bounded Agent with Human-in-the-Loop Approval Gate](./04-ai-agents/exp-02-bounded-agent-with-human-approval/README.md):** Risk-tiered tools, serializable pause/resume state machine, `MAX_REFUND_USD` policy checks, and prompt-injection defense (`8` offline `pytest` cases + interactive `--live` mode).
- **Architecture Decision — [ADR-0001: Switchable Multi-Provider LLM Client Abstraction](./06-ai-system-architecture/adr-0001-switchable-llm-provider-abstraction.md):** Evaluates thin direct `httpx` REST adapters vs. four official SDKs vs. wrapper libraries.
- **Architecture Decision — [ADR-0002: Hybrid Retrieval & Local Vector Store](./06-ai-system-architecture/adr-0002-hybrid-retrieval-and-local-vector-store.md):** Evaluates SQLite `FTS5` + NumPy + RRF vs. embedded vector DBs vs. managed cloud vector stores for the Grounded RAG POC.
- **Architecture Decision — [ADR-0003: Pause-and-Resume HITL Approval State Machine](./06-ai-system-architecture/adr-0003-hitl-approval-state-machine.md):** Evaluates serializable `AgentRunState` pause/resume transitions vs. blocking callbacks vs. post-hoc undo.
- **Upcoming Next Milestone:** Month 4 — Track `05-android-on-device-ai`: Android On-Device GenAI (Gemini Nano / ML Kit GenAI) & Hybrid Edge–Cloud Architecture. See [ROADMAP.md](./ROADMAP.md) for the full sequence.

---

## Current Learning Roadmap

The portfolio follows a structured **6-month (~24-week)** progression at a sustainable pace:

1. **Month 1:** Python for AI Engineering & Direct LLM API Mechanics
2. **Month 2:** Embeddings, Vector Search & Grounded RAG with Evaluations
3. **Month 3:** Tool Calling, Bounded AI Agents & Safety Controls
4. **Month 4:** On-Device Android AI & Hybrid Edge–Cloud Routing
5. **Month 5:** AI System Architecture, Security, Observability & Cost Modeling
6. **Month 6:** End-to-End Mobile + AI Capstone POC & Portfolio Architecture Review

Full details, deliverables, and checkpoints are tracked in [ROADMAP.md](./ROADMAP.md).

---

## Setup Instructions

### Prerequisites

- **Python:** `3.13+`
- **Package & Environment Manager:** [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `venv` + `pip`
- **Android Track (Month 4+):** Android Studio (current stable release) and Kotlin SDK

### General Workflow

Each experiment manages its own isolated dependencies and `.env.example` so early labs remain lightweight and independent.

```bash
# Clone the repository
git clone https://github.com/ASHWIN-S-ASHOK/ai-engineering-lab.git
cd ai-engineering-lab

# Navigate to an experiment directory and sync its isolated environment
cd <track>/<experiment-dir>
uv sync

# Copy environment template and provide local keys (never commit .env)
cp .env.example .env

# Run tests
uv run pytest
```

### Security & Hygiene Standards

- Never commit `.env` files, API keys, tokens, personal data, or proprietary/client material.
- External API integrations must support mockable or offline unit tests alongside optional live integration tests.

---

## Engineering & Documentation Standards

Every experiment in this repository aims to follow consistent engineering practices:

- Explicit type hints, schema validation, structured logging, and clear error handling.
- Concise unit and integration tests using `pytest`.
- Reproducible setup with minimal dependencies and documented inputs/outputs.
- Written reflection covering limitations, trade-offs, what was learned, and follow-up experiments.
