# Problem Statement

## 1. Background and Motivation

Financial compliance is a critical function in the Indian securities market. The Securities and Exchange Board of India (SEBI) issues Master Circulars that consolidate regulatory directives covering capital issuance, disclosure requirements, and market surveillance. These documents are dense, technical, and constantly evolving, making it challenging for compliance officers, legal professionals, and market participants to quickly find accurate, contextual answers to their regulatory questions.

### The Core Problem

**Compliance professionals need rapid, accurate, and contextual answers to questions about SEBI regulations, but the existing approach of manually searching through hundreds of pages of Master Circulars is time-consuming, error-prone, and does not support follow-up or conversational queries.**

### Specific Challenges

1. **Volume and Complexity**: SEBI Master Circulars span hundreds of pages of dense legal and regulatory text. A single circular on capital issuance and disclosure requirements alone contains extensive provisions covering eligibility criteria, pricing guidelines, lock-in periods, promoter obligations, and more.

2. **Cross-Referencing**: Answers to compliance questions often require synthesizing information scattered across multiple sections or even multiple circulars (e.g., capital issuance, market surveillance, and compliance provisions).

3. **Contextual Understanding**: Follow-up questions are common in compliance workflows. A user might ask "What are the eligibility criteria for an IPO?" followed by "What about for a rights issue?" — the system must maintain conversational context.

4. **Accuracy Requirements**: In financial compliance, incorrect or incomplete answers can lead to regulatory violations, penalties, and legal liability. The system must be grounded in the source documents and must clearly indicate when it cannot answer a question.

5. **Terminology Specificity**: Financial compliance uses domain-specific terminology (e.g., "promoter holding," "lock-in period," "book building," "anchor investor allocation") that general-purpose search tools handle poorly.

---

## 2. Problem Definition

### Primary Objective

Design and implement a **Retrieval-Augmented Generation (RAG) pipeline** that:

- Ingests SEBI Master Circulars (PDF documents)
- Enables natural language question-answering over the regulatory corpus
- Supports multi-turn conversations with session-based context
- Returns source documents alongside generated answers for verification
- Serves the system as a production-ready REST API

### Secondary Objectives

1. **Evaluate and compare** multiple embedding strategies (Sentence Transformers, OpenAI Small, OpenAI Large) to select the optimal retrieval model
2. **Evaluate and compare** multiple LLM providers (OpenAI gpt-4o-mini, gpt-5-mini, finetuned Qwen-0.5B) for generation quality
3. **Implement hybrid search** combining dense (semantic) and sparse (lexical) embeddings for superior retrieval performance
4. **Build a pluggable architecture** that allows swapping LLM providers, embedding models, and vector databases without code changes
5. **Containerize** the application for reproducible deployment

---

## 3. Scope

### In Scope

| Component | Description |
|-----------|-------------|
| **Document Ingestion** | PDF loading, text chunking (RecursiveCharacterTextSplitter), embedding generation, and vector database upsert |
| **Retrieval Pipeline** | Hybrid search combining dense (OpenAI text-embedding-3-small) and sparse (SPLADE) embeddings via Pinecone |
| **Generation Pipeline** | LLM-based response generation with query rewriting, context summarization, and source attribution |
| **Conversation Management** | Session-based multi-turn conversation persistence in SQLite |
| **API Layer** | FastAPI REST endpoint (`POST /ask`) for integration |
| **Frontend** | Streamlit chat interface for interactive use |
| **Evaluation** | Retriever evaluation (MRR, nDCG@10, Recall@K, Precision@K) and generation evaluation (RAGAS Faithfulness, Answer Relevancy) |
| **Deployment** | Docker containerization for API and UI |
| **Load Testing** | Locust-based concurrent user simulation |

### Out of Scope

- Real-time regulatory update ingestion (document updates are manual)
- User authentication and authorization
- Multi-language support (English only)
- Fine-grained access control per document or section
- Production-grade monitoring and alerting infrastructure
- Horizontal scaling and distributed deployment

---

## 4. Source Data

The system operates on three SEBI Master Circulars:

1. **Master Circular for Issue of Capital and Disclosure Requirements** — Covers IPO eligibility, pricing, disclosure norms, book-building, and promoter obligations
2. **Master Circular on Surveillance of Securities Market** — Covers market surveillance mechanisms, trading controls, and monitoring frameworks
3. **Master Circular for Compliance with the provisions of SEBI** — Covers general compliance obligations, reporting requirements, and regulatory provisions

These documents are authoritative regulatory texts published by SEBI and form the knowledge base for the RAG system.

---

## 5. Success Criteria

### Retrieval Quality

| Metric | Target | Achieved |
|--------|--------|----------|
| Recall@10 | > 0.85 | **0.9369** (OpenAI Small) |
| nDCG@10 | > 0.60 | **0.6933** (OpenAI Small) |
| MRR | > 0.55 | **0.6158** (OpenAI Small) |

### Generation Quality

| Metric | Target | Achieved |
|--------|--------|----------|
| Faithfulness | > 0.80 | **0.8987** (gpt-5-mini) |
| Answer Relevancy | > 0.90 | **0.9579** (gpt-5-mini) |

### System Quality

| Requirement | Status |
|-------------|--------|
| REST API with < 30s response time | Achieved |
| Multi-turn conversation support | Achieved |
| Source document attribution | Achieved |
| Pluggable LLM/embedding architecture | Achieved |
| Docker deployment | Achieved |

---

## 6. Stakeholders

| Stakeholder | Interest |
|-------------|----------|
| **Compliance Officers** | Quick, accurate answers to regulatory questions |
| **Legal Professionals** | Source-grounded responses for legal research |
| **Market Participants** | Understanding SEBI requirements for capital market activities |
| **Regulators** | Improved accessibility of regulatory information |
| **Development Team** | Maintainable, extensible system architecture |

---

## 7. Constraints and Assumptions

### Constraints

1. **API Rate Limits**: OpenAI API has rate limits; concurrency is controlled via semaphore (max 10 concurrent LLM calls)
2. **Pinecone Free Tier**: Vector database capacity may be limited on free/starter plans
3. **Model Context Windows**: LLM context windows limit the amount of retrieved context that can be included in a single prompt
4. **SQLite Limitations**: Single-writer limitation of SQLite (mitigated by WAL mode)

### Assumptions

1. Source documents (SEBI Master Circulars) are authoritative and accurate
2. Users query in English
3. The regulatory corpus is relatively stable (not updated in real-time)
4. Users provide session IDs for multi-turn conversation tracking
5. The system runs in a single-node deployment environment

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph over simple chain** | Enables parallel execution, explicit state management, and visual workflow debugging |
| **Hybrid search over dense-only** | SPLADE sparse embeddings capture keyword matches that dense embeddings miss, improving recall |
| **OpenAI text-embedding-3-small** | Best Recall@10 (0.9369) and nDCG@10 (0.6933) among evaluated options |
| **gpt-4o-mini as primary LLM** | Balances cost, speed, and quality; gpt-5-mini is more faithful but more expensive |
| **Query rewriting** | Improves retrieval by transforming conversational queries into retrieval-optimized forms |
| **Context summarization** | Compresses conversation history to fit within LLM context windows |
| **Protocol-based abstraction** | Enables easy swapping of LLM, embedding, and database providers |
| **SQLite with WAL** | Simple persistence with concurrent read support, appropriate for single-node deployment |
