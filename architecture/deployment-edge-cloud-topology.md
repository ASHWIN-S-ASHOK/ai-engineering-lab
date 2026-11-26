# Deployment Topology: Edge–Cloud Hybrid AI System

**Classification:** `Architecture Study`  
**Related Study:** [ref-01-mobile-ai-gateway-architecture.md](../06-ai-system-architecture/ref-01-mobile-ai-gateway-architecture.md)

```mermaid
flowchart LR
    subgraph Devices["Field Fleet (Android minSdk 26+)"]
        D1["Flagship / Supported Device (AICore + Gemini Nano)"]
        D2["Standard Scanner (Cloud Fallback / Offline Extractive)"]
    end

    subgraph EdgeNet["Ingress & Edge Security"]
        TLS["API Gateway (TLS, OAuth2/OIDC, Rate Limits)"]
    end

    subgraph AppTier["Stateless BFF Containers (Cloud Run / K8s)"]
        API["FastAPI / Python AI BFF"]
        Worker["Async Ingestion & Eval Worker"]
    end

    subgraph StateTier["Managed Data & Secrets"]
        PG["Postgres + pgvector (Tenant-Scoped RAG + AgentRunState)"]
        Cache["Redis (Response Cache + Rate-Limit Counters)"]
        Secrets["Secret Manager (Provider API Keys)"]
    end

    subgraph ExternalAI["External & Private Inference"]
        CloudLLMs["Cloud LLM APIs (Gemini / OpenAI / Anthropic)"]
        PrivateLLM["Private VPC / Self-Hosted Inference (Ollama / vLLM)"]
    end

    D1 -->|"On-device when AVAILABLE"| D1
    D1 -->|"HTTPS"| TLS
    D2 -->|"HTTPS"| TLS
    TLS --> API
    API --> PG
    API --> Cache
    API --> Secrets
    API --> CloudLLMs
    API --> PrivateLLM
    Worker --> PG
```
