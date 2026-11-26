# Architecture Study 01: Security & Privacy Threat Model (OWASP LLM Top 10 — 2026)

**Classification:** `Architecture Study`  
**Track:** `06-ai-system-architecture`  
**Status:** `Completed`  
**Last Updated:** `2026-11-26`

---

## 1. Scope & Trust Boundaries

This threat model evaluates the **Mobile-to-AI Gateway Architecture** ([ref-01-mobile-ai-gateway-architecture.md](./ref-01-mobile-ai-gateway-architecture.md)) across four trust boundaries:
1. **Mobile Edge Device** (untrusted client environment; potential offline PII).
2. **AI Gateway / BFF** (trusted backend holding vendor API keys, policy caps, and tool handlers).
3. **Retrieval Store** (corpus chunks and embeddings; risk of cross-tenant leakage or poisoned documents).
4. **External Model Providers** (third-party cloud endpoints; free-tier vs. enterprise privacy terms).

---

## 2. OWASP Top 10 for LLM Applications (2026 Edition) Mapping

| OWASP ID (2026) | Risk Category | Concrete Attack Vector in Our System | Controls Proven in Labs | Remaining Production Hardening Needed |
| --- | --- | --- | --- | --- |
| **LLM01:2026** | Prompt Injection | Delivery note (`SHP-6660`) instructs LLM to invoke `issue_customer_refund` | `<untrusted_tool_data>` wrapping + deterministic `PENDING_APPROVAL` gate (`04-ai-agents/exp-02`) | Add dedicated input/output classifiers & strict role separation |
| **LLM02:2026** | Sensitive Information Disclosure | Customer PII sent to free-tier cloud APIs or leaked in logs | `SecretStr` key masking, zero keys on mobile, `allowCloudFallback=false` edge routing (`05-android/exp-01`) | Add BFF PII redaction/tokenization (Presidio/regex) before cloud calls & enterprise DPA tiers |
| **LLM03:2026** | Excessive Agency | Compromised agent loops or issues unauthorized refunds/dispatches | Explicit allow-list `ToolRegistry`, `risk_tier` gating, `MAX_REFUND_USD` cap, `ApprovalMismatchError` (`04-ai-agents/exp-02`) | Enforce RBAC verifying supervisor identity on `/resume` JWT |
| **LLM04:2026** | Supply Chain | Compromised PyPI/Gradle packages or unpinned dependencies | Minimal dependencies, committed `uv.lock`, Gradle version catalog (`libs.versions.toml`), CI checks | Add SBOM generation, Dependabot, and container image signing |
| **LLM05:2026** | Data & Model Poisoning | Malicious documents ingested into RAG corpus skewing answers | Provenance tracking (`doc_id`, `chunk_id`) and verifiable quotes (`03-rag/exp-02`) | Require authenticated ingestion pipelines & content-hash audit logs |
| **LLM06:2026** | Unbounded Consumption | Recursive tool loops or oversized prompts exhausting quotas/budget | `ROUTER_MAX_STEPS` / `AGENT_MAX_STEPS`, `ContextBudgetManager`, `max_output_tokens`, timeouts (`02` & `04`) | Add per-user/per-tenant Redis token-bucket rate limits & daily spend caps |
| **LLM07:2026** | Misinformation | Model fabricates shipment facts or invents nonexistent citations | `verify_citations()` checking `chunk_id` + verbatim quote; forced `abstained=true` (`03-rag/exp-02`) | Add NLI/LLM-as-judge faithfulness checks in offline CI eval suites |
| **LLM08:2026** | Hidden Context Exposure | User prompts extract system instructions or internal tool schemas | Never place secrets, SQL creds, or internal endpoints inside system prompts | Strip debug trace fields (`raw_text`, `retrieved_hits`) before returning to untrusted clients |
| **LLM09:2026** | Vector & Embedding Weaknesses | Cross-tenant chunk retrieval or FTS5 operator injection (`-`, `OR`) | `sanitize_fts5_query()` neutralizing FTS5 operators (`03-rag/exp-02`) | Enforce mandatory `WHERE tenant_id = ?` / vector metadata ACL filters pre-ranking |
| **LLM10:2026** | Improper Output Handling | Model emits malformed JSON or unsafe strings consumed downstream | Pydantic `model_validate_json()` + `SchemaValidationError` + typed Android UI state (`02` & `05`) | Strict output encoding if rendered in HTML/WebViews |

---

## 3. Free-Tier vs. Enterprise Privacy Rule

A critical architectural distinction documented across this portfolio:
- **Free-tier cloud APIs** (e.g., Google AI Studio free tier) may use submitted prompts/outputs for product improvement—acceptable **only** for synthetic/public learning data.
- **Enterprise deployments** must either route PII-sensitive text **on-device** (`Gemini Nano` / local `Ollama` in private VPC) or use **paid enterprise agreements with zero-data-retention** guarantees.
