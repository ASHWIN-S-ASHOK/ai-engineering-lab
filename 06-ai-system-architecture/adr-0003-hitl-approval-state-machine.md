# ADR-0003: Pause-and-Resume State Machine for Human-in-the-Loop Tool Approval

**Status:** `Accepted`  
**Date:** `2026-11-26`  
**Scope:** `04-ai-agents/exp-02-bounded-agent-with-human-approval`  
**Related ADRs:** [ADR-0001](./adr-0001-switchable-llm-provider-abstraction.md)

---

## 1. Context & Problem Statement

In `04-ai-agents/exp-01-tool-calling-router`, tools were restricted to read-only lookups and pure calculations. Once an AI workflow gains **mutating, side-effecting tools** (issuing financial refunds, dispatching replacement shipments, modifying records), autonomous execution introduces unacceptable risk from:
- Model hallucinations or misread order amounts.
- **Indirect prompt injection** embedded in external data (e.g., delivery instructions or support notes).

We need an approval mechanism that works cleanly in asynchronous backend/mobile architectures (e.g., an Android operations app approving a pending exception) rather than only in blocking terminal scripts.

---

## 2. Decision Drivers

1. **No blocking I/O on human input:** Async services and mobile BFFs cannot hold open HTTP requests or threads waiting minutes/hours for a human reviewer.
2. **Deterministic risk enforcement:** Side-effecting tools must never execute without explicit human authorization, regardless of what the prompt or external tool data says.
3. **Wire-protocol integrity:** Multi-turn tool APIs (Gemini, OpenAI Responses, Anthropic Messages) require every requested tool call ID to receive a corresponding tool result before continuing.
4. **Full auditability:** Every tool request, approval/rejection decision, reviewer identity, and side effect must be recorded immutably.

---

## 3. Considered Options

### Option A: Blocking In-Process Callback (`input()` / Prompt Callback)
Pause inside the `while` loop by awaiting a local CLI `input()` or synchronous prompt callback.

- **Pros:**
  - Simple for single-user terminal scripts.
- **Cons:**
  - Ties up server memory and request lifecycle if used behind an API.
  - Cannot survive process restarts or hand off approval to a separate UI (such as a mobile app).

### Option B: Serializable Pause-and-Resume State Machine (`AgentRunState`) *(Recommended)*
Split execution into `start()` and `resume(state, decision)` transitions over an explicit `AgentRunState` Pydantic model (`RUNNING` -> `PENDING_APPROVAL` -> `COMPLETED` / `FAILED`).

- **Pros:**
  - State snapshots serialize directly to JSON (easy to persist in SQLite/Redis/Postgres later).
  - Cleanly decouples agent orchestration from the approval UI (CLI today, REST/Android app tomorrow).
  - Enforces `action_id` matching so stale or mismatched approvals are rejected deterministically.
- **Cons:**
  - Slightly more state bookkeeping when a single LLM turn emits both `read_only` and `mutating` tool calls.

### Option C: Autonomous Execution + Post-Hoc Undo
Let the agent execute mutating tools immediately and rely on compensating rollback transactions if a reviewer objects later.

- **Pros:**
  - Lowest latency for happy paths.
- **Cons:**
  - Some external side effects (emails, refunds, carrier dispatches) are expensive or impossible to undo cleanly.

---

## 4. Decision Outcome

**Chosen Option:** **Option B — Serializable Pause-and-Resume State Machine (`AgentRunState`)**.

**Rationale:**  
Representing an agent run as an explicit Pydantic state machine keeps safety enforcement deterministic in Python, supports asynchronous mobile/web approval workflows naturally, and produces an auditable trace for compliance and debugging.

---

## 5. Consequences

### Positive
- Zero mutating side effects can execute before explicit human `ApprovalDecision`.
- Indirect prompt injections in external tool outputs cannot self-approve actions.
- `AgentRunState` can be persisted and resumed across requests or processes.

### Negative / Technical Debt Accepted
- This POC holds `AgentRunState` in memory between `start()` and `resume()` (serialization to SQLite/Postgres is straightforward via `model_dump_json()`).

---

## 6. Architecture Quality Attributes

- **Security & Privacy:** External tool data is wrapped in `<untrusted_tool_data>` markers; Pydantic schemas enforce hard business caps (e.g., `MAX_REFUND_USD`) before an approval request is even surfaced; `action_id` prevents replay/mismatch approvals.
- **Scalability & Reliability:** Hard `AGENT_MAX_STEPS` cap prevents infinite loops; multi-tool turns finish `read_only` items first and preserve 1:1 `call_id` pairing across pause/resume boundaries.
- **Cost & Performance:** Invalid tool arguments fail validation locally before hitting another LLM round trip or wasting reviewer time.

---

## 7. Validation & Revisit Criteria

- **Validation:** Offline tests verify pause at `PENDING_APPROVAL`, approval execution, rejection feedback without side effects, mismatched `action_id` rejection, and indirect prompt-injection blocking.
- **Revisit Trigger:** Persist `AgentRunState` in SQLite/Postgres with TTL expiration and RBAC reviewer verification when integrating into Track `06-ai-system-architecture` mobile/BFF services.
