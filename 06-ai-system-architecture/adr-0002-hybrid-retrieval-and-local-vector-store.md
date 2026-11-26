# ADR-0002: Hybrid Retrieval (BM25 + Dense Vectors) and Local Persistence for Grounded RAG POC

**Status:** `Proposed`  
**Date:** `2026-11-26`  
**Scope:** `03-rag-systems/exp-02-grounded-rag-with-citations-and-evals`  
**Related ADRs:** [ADR-0001](./adr-0001-switchable-llm-provider-abstraction.md)

---

## 1. Context & Problem Statement

In `03-rag-systems/exp-01-embeddings-and-similarity-from-scratch`, pure dense cosine similarity handled paraphrased policy queries well (`recall@1 = 1.0`) but failed on bare alphanumeric identifiers such as `SKU-4812-X` (`recall@1 = 0.0`). For our first end-to-end **Grounded RAG Proof of Concept**, we need:
1. A retrieval layer that handles **both** semantic paraphrases and exact identifiers/codes.
2. Persistent local storage for chunks, metadata, lexical index, and embedding vectors without requiring Docker or external cloud accounts.
3. Strict citation verification back to persisted chunk IDs and explicit abstention when evidence is missing.

---

## 2. Decision Drivers

1. **Hybrid accuracy:** Combine lexical matching (BM25) with dense semantic similarity so exact SKUs, policy IDs, and paraphrased questions all retrieve reliably.
2. **Zero infrastructure friction:** Must run locally out of the box on macOS (`Python 3.13`, `sqlite 3.45.3` with `FTS5` verified) and in GitHub Actions CI.
3. **Transparency over magic:** Keep Reciprocal Rank Fusion (RRF), chunk provenance, and citation validation inspectable in plain Python before adopting managed vector databases.
4. **Reversibility:** Isolate storage/search behind a `Retriever` `Protocol` so swapping to Qdrant, Chroma, or `pgvector` later requires no changes to RAG orchestration or evaluation code.

---

## 3. Considered Options

### Option A: Local SQLite (`FTS5` BM25) + NumPy Dense Cosine + RRF *(Recommended)*
Store document/chunk metadata and embeddings in SQLite (with `FTS5` virtual table for built-in BM25 lexical scoring), compute dense cosine similarity in NumPy, and merge ranked lists using **Reciprocal Rank Fusion (RRF)**.

- **Pros:**
  - Zero new runtime dependencies (`sqlite3` + `FTS5` are built into Python 3.13; `numpy` is already in Track 03).
  - Single-file ACID persistence (`*.sqlite3` ignored by `.gitignore`) with deterministic offline testing via `:memory:`.
  - Exposes exact BM25 rank, dense rank, and RRF score breakdown for debugging and evals.
- **Cons:**
  - Dense vector scan is $O(N)$ in memory; not suitable once corpus size reaches tens/hundreds of thousands of chunks.
  - No distributed replication or multi-tenant cloud auth.

### Option B: Embedded Vector Store (`chromadb` or local `qdrant-client`)
Use an embedded vector database with built-in HNSW approximate nearest neighbor indexing.

- **Pros:**
  - Scales cleanly to larger local corpora with fast ANN search and rich metadata filtering.
- **Cons:**
  - Adds heavier C++/Rust/protobuf dependency trees into CI and early experiments.
  - Some embedded setups require separate configuration or manual wiring for BM25 hybrid fusion.

### Option C: Managed Cloud Vector DB (`pgvector` / Qdrant Cloud / Pinecone)
Provision a hosted database for the POC.

- **Pros:**
  - Production-ready scaling, backups, and managed HNSW indexes.
- **Cons:**
  - Requires external credentials, network calls, and cloud cost/setup—unnecessary overhead for a local learning POC.

---

## 4. Decision Outcome

**Chosen Option:** **Option A — Local SQLite (`FTS5` BM25) + NumPy Dense Cosine + RRF** behind a `Retriever` `Protocol`.

**Rationale:**  
Using SQLite `FTS5` + NumPy + RRF solves the exact SKU retrieval gap observed in Experiment 5 while keeping every piece of the hybrid retrieval pipeline transparent, persistent, and zero-infrastructure. Placing this behind a `Retriever` `Protocol` preserves clean Clean Architecture boundaries for future migration to `pgvector` or Qdrant.

---

## 5. Consequences

### Positive
- Fast offline CI execution using SQLite `:memory:`.
- Full observability into lexical vs. dense vs. fused RRF scores per query.
- Zero external server processes or Docker dependencies.

### Negative / Technical Debt Accepted
- Brute-force dense scan is intentionally bounded to small/medium POC corpora (< ~20,000 chunks).

### Follow-up Actions
- Implement `03-rag-systems/exp-02-grounded-rag-with-citations-and-evals` using this hybrid retriever, structured citation validation, and an end-to-end eval harness.
- Compare dense-only vs. BM25-only vs. RRF hybrid scores quantitatively in the POC's evaluation suite.

---

## 6. Architecture Quality Attributes

- **Security & Privacy:** All corpus and index files remain local (`*.sqlite3` gitignored). Cloud LLM/embedding calls use synthetic sample docs only. Generated citations are verified against retrieved `chunk_id`s to prevent hallucinated references.
- **Scalability & Reliability:** SQLite transactions guarantee atomic chunk ingestion; FTS5 query syntax is sanitized to avoid query-parser exceptions on punctuation/hyphens (e.g., `SKU-4812-X`).
- **Cost & Performance:** RRF requires no score calibration or extra reranker LLM calls, keeping token cost and latency minimal.

---

## 7. Validation & Revisit Criteria

- **Validation:** Offline tests verify FTS5 hyphen/punctuation sanitization, RRF ranking improvements on `SKU-4812-X` + semantic paraphrases, fabricated-citation rejection, and abstention on out-of-scope queries.
- **Revisit Trigger:** Migrate the `Retriever` adapter to Option B/C (`qdrant-client` or `pgvector`) when corpus size exceeds ~20,000 chunks, concurrent multi-user writes are needed, or Track `06-ai-system-architecture` moves to multi-tenant cloud deployment.
