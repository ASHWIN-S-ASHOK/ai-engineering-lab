# System Architecture Diagram: Mobile-to-AI Gateway (BFF)

**Classification:** `Architecture Study`  
**Related Study:** [ref-01-mobile-ai-gateway-architecture.md](../06-ai-system-architecture/ref-01-mobile-ai-gateway-architecture.md)

```mermaid
flowchart TD
    subgraph MobileEdge["Android Client Edge (Kotlin / Compose)"]
        UI["Field Operations UI"]
        Router["HybridSummarizerUseCase"]
        Nano["AICore / Gemini Nano (Offline & PII-Safe)"]
        Fallback["Deterministic Extractive Fallback"]
    end

    subgraph BffLayer["AI Gateway / Backend-for-Frontend (Python)"]
        Auth["AuthN / AuthZ & Rate Limiting"]
        Pii["PII Redaction & Context Budgeter"]
        Rag["Hybrid RAG Engine (BM25 + Vector RRF)"]
        Agent["HITL Bounded Agent State Machine"]
        Verify["Citation & Schema Verifier"]
        ModelRouter["Switchable Provider Router"]
    end

    subgraph DataStores["Persistence & Telemetry"]
        VectorDb["Vector + FTS Index (Tenant-Scoped)"]
        StateDb["AgentRunState & Audit Ledger"]
        Obs["Structured Traces & Eval Metrics"]
    end

    subgraph ModelProviders["Model Providers"]
        Gemini["Google Gemini API"]
        OpenAI["OpenAI Responses API"]
        Anthropic["Anthropic Messages API"]
        Ollama["Local / Private Ollama"]
    end

    UI --> Router
    Router -->|"AVAILABLE"| Nano
    Router -->|"Offline & Cloud Disallowed"| Fallback
    Router -->|"HTTPS + Bearer Token"| Auth

    Auth --> Pii
    Pii --> Rag
    Pii --> Agent
    Rag --> VectorDb
    Agent --> StateDb
    Rag --> Verify
    Agent --> Verify
    Verify --> ModelRouter

    ModelRouter --> Gemini
    ModelRouter --> OpenAI
    ModelRouter --> Anthropic
    ModelRouter --> Ollama

    BffLayer --> Obs
```
