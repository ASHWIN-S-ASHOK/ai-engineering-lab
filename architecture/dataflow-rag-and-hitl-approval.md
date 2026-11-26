# Data-Flow Diagrams: Grounded Hybrid RAG & Async HITL Approval

**Classification:** `Architecture Study`  
**Related Study:** [ref-01-mobile-ai-gateway-architecture.md](../06-ai-system-architecture/ref-01-mobile-ai-gateway-architecture.md)

## 1. Grounded Hybrid RAG Request Flow

```mermaid
sequenceDiagram
    autonumber
    participant App as Android Field App
    participant BFF as AI Gateway BFF
    participant Store as Hybrid Store (BM25 + Dense)
    participant LLM as Switchable LLM Adapter

    App->>BFF: POST /v1/rag/query (query, tenant_id)
    BFF->>BFF: Sanitize query & embed query vector
    BFF->>Store: Search BM25 + Cosine (top_k, RRF k=60)
    Store-->>BFF: Ranked SearchHits with chunk_id
    BFF->>BFF: Pack context within token budget
    BFF->>LLM: Generate GroundedAnswer (strict JSON Schema)
    LLM-->>BFF: answer, abstained, citations[]
    BFF->>BFF: verify_citations() against retrieved chunk_ids
    alt Valid citations or abstained=true
        BFF-->>App: 200 OK (RagResponse + verified citations)
    else Fabricated chunk_id or quote mismatch
        BFF-->>App: 422 / Safe Abstention Fallback
    end
```

## 2. Asynchronous HITL Mutating Action Flow

```mermaid
sequenceDiagram
    autonumber
    participant DriverApp as Field App
    participant BFF as HITL Agent Service
    participant StateDB as AgentRunState Store
    participant SupApp as Supervisor Android App
    participant Ledger as Operations Tool Registry

    DriverApp->>BFF: POST /v1/agent/runs (query)
    BFF->>Ledger: Execute read_only lookup_shipment_status
    Ledger-->>BFF: Shipment data (<untrusted_tool_data> wrapped)
    Note over BFF: LLM requests mutating issue_customer_refund
    BFF->>BFF: Validate Pydantic schema & MAX_REFUND_USD cap
    BFF->>StateDB: Save snapshot (status=PENDING_APPROVAL, action_id)
    BFF-->>DriverApp: 202 Accepted (run_id, PENDING_APPROVAL)

    SupApp->>BFF: GET /v1/agent/approvals/pending
    BFF-->>SupApp: PendingApproval (action_id, tool, args)
    SupApp->>BFF: POST /v1/agent/runs/{run_id}/resume (action_id, approved=true)
    BFF->>StateDB: Verify status=PENDING_APPROVAL & action_id match
    BFF->>Ledger: Execute validated issue_customer_refund
    Ledger-->>BFF: Refund confirmation (REF-0001)
    BFF->>StateDB: Save COMPLETED state & immutable AuditEntry
    BFF-->>SupApp: 200 OK (COMPLETED + final_answer)
```
