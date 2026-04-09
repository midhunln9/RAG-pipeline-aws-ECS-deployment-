# Data Documentation

## 1. Overview

This document describes the source data, data processing pipeline, storage strategy, and data flow within the RAG pipeline for SEBI Financial Compliance. The system ingests regulatory PDF documents, processes them into vector embeddings, and stores them in a Pinecone vector database for retrieval. Conversation data is persisted in SQLite for multi-turn context management.

---

## 2. Source Data

### 2.1 Document Corpus

The knowledge base consists of three SEBI (Securities and Exchange Board of India) Master Circulars in PDF format:

| Document | Description | Location |
|----------|-------------|----------|
| **Master Circular for Issue of Capital and Disclosure Requirements** | Covers IPO eligibility, pricing guidelines, book-building processes, promoter obligations, lock-in periods, disclosure norms, and capital issuance procedures | `Ingestion_plus_Retriever_eval/Documents/` |
| **Master Circular on Surveillance of Securities Market** | Covers market surveillance mechanisms, trading controls, price band monitoring, and market integrity frameworks | `Ingestion_plus_Retriever_eval/Documents/` |
| **Master Circular for Compliance with the provisions of SEBI** | Covers general compliance obligations, reporting requirements, regulatory deadlines, and enforcement provisions | `Ingestion_plus_Retriever_eval/Documents/` |

### 2.2 Data Characteristics

- **Format**: PDF documents
- **Language**: English
- **Domain**: Indian financial regulatory compliance (SEBI)
- **Content Type**: Legal/regulatory text with tables, numbered provisions, cross-references, and definitions
- **Authority**: Official SEBI regulatory publications
- **Total Chunks After Processing**: ~1,108 chunks (based on evaluation dataset)

### 2.3 Data Quality Considerations

| Aspect | Details |
|--------|---------|
| **Completeness** | Master Circulars are comprehensive consolidations of all relevant SEBI directives |
| **Accuracy** | Official SEBI publications; authoritative source |
| **Structure** | Hierarchical with chapters, sections, sub-sections, and numbered provisions |
| **Challenges** | PDF parsing may lose formatting; tables may not parse cleanly; cross-references between sections |
| **Freshness** | Static corpus; manual re-ingestion required for updates |

---

## 3. Data Processing Pipeline

### 3.1 Ingestion Architecture

The ingestion pipeline is implemented in the `Ingestion_plus_Retriever_eval/` module and consists of four stages:

```
PDF Files → [Document Loader] → [Text Splitter] → [Embedding Generator] → [Vector DB Upsert]
```

### 3.2 Stage 1: Document Loading

- **Tool**: PyPDFLoader (from LangChain)
- **Mode**: Page-level loading
- **Implementation**: `Ingestion_plus_Retriever_eval/src/chunker_service.py`
- **Source Directory**: `Ingestion_plus_Retriever_eval/Documents/`
- **Repository**: `FileRepository` loads all PDFs from the configured directory

```python
# Each PDF is loaded page-by-page, producing Document objects with:
# - page_content: extracted text
# - metadata: {source: filename, page: page_number}
```

### 3.3 Stage 2: Text Chunking

- **Strategy**: RecursiveCharacterTextSplitter (LangChain)
- **Implementation**: `Ingestion_plus_Retriever_eval/src/recursive_character_text_splitting.py`
- **Configuration**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 1000 characters | Balances context completeness with embedding quality |
| `chunk_overlap` | 200 characters | Preserves context at chunk boundaries |
| `separators` | Default recursive (`\n\n`, `\n`, ` `, `""`) | Respects document structure hierarchy |

- **Output**: ~1,108 document chunks with preserved metadata (source file, page number)
- **Config Class**: `RecursiveTextSplitterConfig` in `Ingestion_plus_Retriever_eval/configs/`

### 3.4 Stage 3: Embedding Generation

Two types of embeddings are generated for each chunk to enable hybrid search:

#### Dense Embeddings

| Model | Dimensions | Purpose |
|-------|-----------|---------|
| OpenAI `text-embedding-3-small` | 1536 | Semantic similarity matching |
| Sentence Transformers `all-MiniLM-L6-v2` | 384 | Alternative (evaluated, not selected) |

- **Selected Model**: OpenAI `text-embedding-3-small` based on evaluation results (best Recall@10: 0.9369)
- **Implementation**: `Ingestion_plus_Retriever_eval/src/openai_embedding.py`

#### Sparse Embeddings

| Model | Type | Purpose |
|-------|------|---------|
| SPLADE (`naver/splade-cocondenser-ensembledistil`) | Sparse lexical | Keyword/term matching |

- **Implementation**: `Ingestion_plus_Retriever_eval/src/sparse_embedding.py`
- **Output Format**: Dictionary with `indices` (token IDs) and `values` (weights)
- **Benefit**: Captures exact keyword matches that dense embeddings may miss (e.g., specific regulatory terms like "Regulation 27(2)")

### 3.5 Stage 4: Vector Database Upsert

- **Target**: Pinecone cloud vector database
- **Index**: `final-rag-index-openai-small`
- **Region**: AWS us-east-1
- **Metric**: Dot product
- **Batch Size**: 200 vectors per upsert operation
- **Implementation**: `Ingestion_plus_Retriever_eval/Repositories/pinecone_repository.py`

Each vector record contains:
```json
{
  "id": "unique-chunk-id",
  "values": [0.012, -0.034, ...],       // Dense embedding (1536 dims)
  "sparse_values": {
    "indices": [102, 4521, ...],          // SPLADE token indices
    "values": [1.23, 0.87, ...]           // SPLADE token weights
  },
  "metadata": {
    "text": "chunk content...",
    "source": "Master Circular for Issue of Capital...",
    "page": 5
  }
}
```

### 3.6 Pipeline Orchestration

The `Pipeline` class (`Ingestion_plus_Retriever_eval/src/pipeline.py`) orchestrates:
1. `ChunkerService.chunk_documents()` — loads PDFs and splits into chunks
2. `UpsertService.upsert_chunks()` — generates embeddings and upserts to Pinecone

Entry point: `Ingestion_plus_Retriever_eval/main.py`

---

## 4. Evaluation Datasets

### 4.1 Retriever Evaluation Dataset

Generated synthetically for hard evaluation of retriever performance:

- **Location**: `notebooks/datasets/`
- **Subdirectories**: `openai_small/`, `openai_large/`, `sentence_transformers/`
- **Contents**: `all_chunks.csv` — contains all 1,108 chunks with metadata
- **Query Generation**: Ollama `llama3` used to generate synthetic queries from chunks
- **Ground Truth**: Each generated query is mapped to its source chunk (creating qrels)
- **Sample Size**: 10% of chunks (111 chunks) sampled for evaluation

### 4.2 RAGAS Evaluation Dataset

Generated by running the full RAG workflow on 30 domain-specific SEBI regulation questions:

- **Location**: `rag_pipeline/ragas_eval/`
- **Questions**: 30 manually curated questions covering various aspects of SEBI regulations
- **Format**: RAGAS `EvaluationDataset` with fields:
  - `user_input`: Original question
  - `response`: LLM-generated answer
  - `retrieved_contexts`: List of retrieved document contents

---

## 5. Conversation Data (Runtime)

### 5.1 Database Schema

Conversations are stored in SQLite (`sample.db`) using SQLAlchemy ORM:

```sql
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  VARCHAR NOT NULL,    -- Indexed for fast lookup
    messages    TEXT NOT NULL,        -- JSON-serialized conversation
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_conversations_session_id ON conversations(session_id);
```

### 5.2 Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer (PK) | Auto-incrementing primary key |
| `session_id` | String (Indexed) | User session identifier for grouping conversations |
| `messages` | Text (JSON) | Serialized list of messages (query + response pairs) |
| `created_at` | DateTime | Timestamp of conversation creation |

### 5.3 Conversation Retrieval

- **Query**: Last 10 conversations for a given `session_id`, ordered by `created_at` descending
- **Purpose**: Provides context for the Context Summarizer node
- **Summarization Trigger**: If more than 5 conversations exist, LLM summarizes them; otherwise returns a default message

### 5.4 Database Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| **Engine** | SQLite | Lightweight, file-based persistence |
| **WAL Mode** | Enabled | Write-Ahead Logging for concurrent reads |
| **Busy Timeout** | 5000ms | Wait time for locked database |
| **check_same_thread** | False | Allow multi-threaded access (FastAPI) |
| **Connection Pool** | SQLAlchemy default | Managed sessions with auto-commit/rollback |

---

## 6. Data Flow at Runtime

```
1. User submits query + session_id
   ↓
2. AgentState initialized with query and session_id
   ↓
3. Query Rewriter:
   - Input: original query
   - Output: rewritten query optimized for retrieval
   ↓
4. Parallel Execution:
   a. Document Fetcher:
      - Generates dense embedding (OpenAI) of rewritten query
      - Generates sparse embedding (SPLADE) of rewritten query
      - Queries Pinecone with hybrid search
      - Returns top-10 documents with metadata
   
   b. Context Summarizer:
      - Fetches last 10 conversations from SQLite by session_id
      - If > 5 conversations: LLM summarizes history
      - If <= 5 conversations: returns default summary
   ↓
5. Response Generator:
   - Combines: rewritten query + retrieved documents + context summary
   - LLM generates response with system prompt (Financial Compliance domain)
   ↓
6. DB Writer:
   - Serializes messages (query + response) as JSON
   - Inserts new conversation record with session_id
   ↓
7. API returns: response text + source documents (content + metadata)
```

---

## 7. Data Privacy and Security Considerations

| Concern | Mitigation |
|---------|-----------|
| **API Keys** | Stored in environment variables, not in code |
| **User Queries** | Stored in local SQLite database; not transmitted beyond LLM API calls |
| **Source Documents** | Public SEBI regulatory documents; no PII |
| **Session Data** | Session IDs are user-provided identifiers; no authentication layer |
| **LLM Data** | Queries are sent to OpenAI API; subject to OpenAI's data handling policies |
| **Environment File** | `ingestion.env` contains sensitive keys — must be excluded from version control |

---

## 8. Data Volume Estimates

| Data Type | Approximate Size | Growth Rate |
|-----------|-----------------|-------------|
| Source PDFs | ~3 documents, hundreds of pages | Static (manual updates) |
| Chunks in Pinecone | ~1,108 vectors | Per re-ingestion |
| Dense Embeddings | 1,108 x 1,536 floats = ~6.8 MB | Per re-ingestion |
| Sparse Embeddings | Variable (SPLADE output) | Per re-ingestion |
| SQLite Conversations | ~1 KB per conversation | Per user query |
| Evaluation Datasets | ~111 query-document pairs | Per evaluation run |
