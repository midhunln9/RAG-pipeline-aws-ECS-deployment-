# System Design

## 1. Overview

This document describes the complete system architecture of the RAG pipeline for SEBI Financial Compliance. The system is designed as a **layered, protocol-based architecture** orchestrated by a **LangGraph state machine**, served through a **FastAPI REST API**, and backed by **Pinecone** (vector search) and **SQLite** (conversation persistence).

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                             │
│                                                                     │
│   ┌──────────────────┐         ┌──────────────────────────┐        │
│   │  Streamlit UI    │         │  Any HTTP Client          │        │
│   │  (Port 8001)     │────────▶│  (curl, Postman, etc.)   │        │
│   └──────────────────┘         └──────────────────────────┘        │
│                                         │                           │
└─────────────────────────────────────────┼───────────────────────────┘
                                          │ HTTP POST /ask
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API LAYER                                   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  FastAPI Server (Port 8000)                                  │  │
│   │  ├── Lifespan: Dependency initialization & validation       │  │
│   │  ├── POST /ask: AskRequest → AskResponse                   │  │
│   │  └── GET /: Health check                                    │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │ Invoke Workflow
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                              │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  LangGraph State Machine (RAGWorkflow)                       │  │
│   │                                                               │  │
│   │  START → [Query Rewriter]                                    │  │
│   │              │                                                │  │
│   │              ├──→ [Document Fetcher]     ─┐ (Parallel)       │  │
│   │              └──→ [Context Summarizer]   ─┘                  │  │
│   │                           │                                   │  │
│   │                    [Response Generator]                       │  │
│   │                           │                                   │  │
│   │                    [DB Writer] → END                          │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Node Orchestrator (5 node implementations)                  │  │
│   │  RAGService (business logic utility methods)                 │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────┬──────────┬──────────────┬──────────────┬───────────────────┘
         │          │              │              │
         ▼          ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SERVICE LAYER                                  │
│                                                                     │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ LLM      │  │ Vector DB  │  │ Conversation │  │ Embedding   │  │
│  │ Protocol  │  │ Protocol   │  │ Repository   │  │ Strategies  │  │
│  └─────┬────┘  └──────┬─────┘  └──────┬───────┘  └──────┬──────┘  │
│        │              │               │                  │          │
└────────┼──────────────┼───────────────┼──────────────────┼──────────┘
         │              │               │                  │
         ▼              ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                               │
│                                                                     │
│  ┌──────────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ OpenAI API   │  │ Pinecone   │  │ SQLite   │  │ SPLADE      │  │
│  │ gpt-4o-mini  │  │ Cloud      │  │ WAL Mode │  │ Local Model │  │
│  ├──────────────┤  │ AWS us-e-1 │  │ sample.db│  └─────────────┘  │
│  │ Ollama       │  └────────────┘  └──────────┘                    │
│  │ llama3.2     │                                                   │
│  ├──────────────┤  ┌──────────────────────────┐                    │
│  │ Finetuned    │  │ OpenAI Embeddings        │                    │
│  │ Qwen-0.5B   │  │ text-embedding-3-small   │                    │
│  └──────────────┘  └──────────────────────────┘                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layered Architecture

### 3.1 Presentation Layer

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Streamlit UI | `streamlit` (Port 8001) | Interactive chat interface with session management, source display, and custom CSS styling |
| HTTP Clients | curl, Postman, etc. | Direct API integration for downstream systems |

**Streamlit UI Features**:
- UUID-based session management
- Chat history display with user/assistant messages
- Source document cards with metadata (page numbers, file names)
- Error handling for connection timeouts and HTTP errors
- Configurable `RAG_API_URL` for backend connection

### 3.2 API Layer

**Framework**: FastAPI (async, with Pydantic validation)

**Entry Point**: `api/main.py`

**Endpoints**:

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/ask` | POST | `AskRequest(query, session_id)` | `AskResponse(response, session_id, source_documents)` |
| `/` | GET | None | Health check message |

**Lifespan Management**:
The FastAPI application uses an async lifespan context manager for:
1. Loading environment variables from `ingestion.env`
2. Validating required credentials (OpenAI key, Pinecone key, database URL)
3. Initializing all dependencies in order:
   - Database engine and session factory
   - Dense embedding model (OpenAI)
   - Sparse embedding model (SPLADE)
   - Pinecone repository
   - LLM instance
   - Conversation repository
   - RAGService (aggregates all dependencies)
   - RAGWorkflow (LangGraph state machine)
4. Storing initialized components in `app.state` for request access

**Request Handling**:
- Incoming requests are processed via `asyncio.to_thread()` to avoid blocking the event loop
- The LangGraph workflow is invoked synchronously within the thread
- Retrieved documents are extracted from workflow state and returned as `SourceDocument` objects

### 3.3 Orchestration Layer

**Core**: LangGraph State Machine (`workflow/graph.py`)

The workflow is defined as a directed acyclic graph (DAG) with 5 nodes:

```python
# Graph topology
START → query_rewriter
query_rewriter → [fetch_documents, generate_summary_last_5_messages]  # Parallel
[fetch_documents, generate_summary_last_5_messages] → llm_call        # Fan-in
llm_call → add_conversation
add_conversation → END
```

**State Schema** (`workflow/state.py`):

```python
class AgentState(TypedDict, total=False):
    query: str                                    # Original user query
    rewritten_query: str                          # LLM-optimized query
    retrieved_documents: list[Document]            # Top-10 from Pinecone
    session_id: str                               # User session identifier
    conversation_history: list[BaseMessage]        # Conversation messages
    response: str                                  # Final LLM response
    summary_before_last_five_messages: str         # Context summary
```

**Node Implementations** (`workflow/node_orchestrator.py`):

| Node | Input State Fields | Output State Fields | Dependencies |
|------|-------------------|---------------------|-------------|
| `query_rewriter` | query | rewritten_query, conversation_history | LLM |
| `fetch_documents` | rewritten_query | retrieved_documents | Vector DB |
| `generate_summary_last_5_messages` | session_id | summary_before_last_five_messages | Database, LLM |
| `llm_call` | query, retrieved_documents, summary | response, conversation_history | LLM |
| `add_conversation_to_db` | session_id, conversation_history, response | (persisted) | Database |

**RAGService** (`workflow/service.py`):
A utility class that encapsulates business logic called by nodes:
- `rewrite_query()`: LLM-based query optimization
- `retrieve_documents()`: Vector DB hybrid search
- `generate_context_summary()`: Conversation history summarization
- `generate_response()`: Prompt construction and LLM invocation
- `save_conversation()`: Database persistence

### 3.4 Service Layer (Protocols)

The service layer is defined through Python `Protocol` classes, enabling dependency inversion and pluggability:

#### LLM Protocol (`workflow/protocols/llm_protocol.py`)
```python
class LLMProtocol(Protocol):
    def invoke(self, messages: list[BaseMessage]) -> BaseMessage: ...
```

**Implementations**:
| Class | Model | Features |
|-------|-------|----------|
| `OpenAILLM` | gpt-4o-mini / gpt-5-mini | Semaphore (max 10), 30s timeout, 2 retries |
| `OllamaLLM` | llama3.2 | Local inference, no rate limiting |
| `FinetunedLLM` | Qwen-0.5B | HuggingFace transformers, 512 max tokens |

#### Vector DB Protocol (`workflow/protocols/vector_db_protocol.py`)
```python
class VectorDBProtocol(Protocol):
    def query(self, query: str) -> list[Document]: ...
```

**Implementation**: `PineconeRepository`
- Accepts query string
- Generates dense embedding (OpenAI) and sparse embedding (SPLADE)
- Performs hybrid search on Pinecone index
- Returns top-K LangChain `Document` objects with metadata

#### Database Repository Protocol (`workflow/protocols/database_repo_protocol.py`)
```python
class DatabaseRepositoryProtocol(Protocol):
    def add_conversation(self, session, session_id, messages): ...
    def get_conversations_by_session_id(self, session, session_id): ...
```

**Implementation**: `ConversationRepository`
- SQLAlchemy ORM-based CRUD operations
- Session-scoped transactions with auto-commit/rollback
- Indexed lookups by `session_id`

### 3.5 Infrastructure Layer

| Component | Technology | Configuration |
|-----------|-----------|---------------|
| **LLM API** | OpenAI | `gpt-4o-mini`, API key via env var |
| **LLM Local** | Ollama | `llama3.2`, local server |
| **LLM Finetuned** | HuggingFace | Qwen-0.5B, local weights |
| **Vector DB** | Pinecone Cloud | AWS us-east-1, dot product, `final-rag-index-openai-small` |
| **Dense Embedding** | OpenAI | `text-embedding-3-small` (1536 dims) |
| **Sparse Embedding** | SPLADE | `naver/splade-cocondenser-ensembledistil` (local) |
| **Database** | SQLite | WAL mode, `sample.db`, 5000ms busy timeout |

---

## 4. Key Design Patterns

### 4.1 Fan-Out / Fan-In (Parallel Execution)

```
Query Rewriter (sequential)
        │
        ├──→ Document Fetcher      ─┐
        │                            │  Execute in parallel
        └──→ Context Summarizer    ─┘
                    │
             Response Generator (sequential, waits for both)
```

Document retrieval and context summarization are independent operations that execute simultaneously, reducing total latency by running I/O-bound operations concurrently.

### 4.2 Protocol-Based Dependency Inversion

All major components are defined by abstract `Protocol` interfaces:
- **LLMProtocol**: Any LLM that accepts messages and returns a message
- **VectorDBProtocol**: Any vector store that accepts a query string and returns documents
- **DatabaseRepositoryProtocol**: Any store that can save/retrieve conversations

This enables:
- Swapping OpenAI for Ollama without changing workflow code
- Replacing Pinecone with Weaviate/Qdrant by implementing `VectorDBProtocol`
- Moving from SQLite to PostgreSQL by implementing `DatabaseRepositoryProtocol`

### 4.3 State Machine Orchestration

LangGraph provides:
- **Explicit state management**: All data flows through a typed `AgentState` dictionary
- **Visual debugging**: The workflow graph can be rendered for inspection
- **Declarative topology**: Nodes and edges are defined declaratively, not imperatively
- **Automatic parallelization**: Multiple edges from one node trigger parallel execution

### 4.4 Async API with Sync Workflow

```python
# In ask_endpoint.py
result = await asyncio.to_thread(workflow.invoke, initial_state)
```

The FastAPI server is async (for concurrent request handling), but the LangGraph workflow is synchronous. `asyncio.to_thread()` bridges this gap, running the workflow in a thread pool without blocking the event loop.

### 4.5 Configuration Cascade

```
Environment Variables (highest priority)
    ↓
.env file (ingestion.env)
    ↓
Pydantic Settings defaults (lowest priority)
```

Configuration is managed via Pydantic `BaseSettings` with environment variable binding, service-specific `@dataclass` configs, and a `from_settings()` factory pattern.

---

## 5. Data Models

### 5.1 API Models

```python
class AskRequest(BaseModel):
    query: str                  # User's question
    session_id: str             # Session identifier

class SourceDocument(BaseModel):
    content: str                # Document text
    metadata: dict              # Source file, page number, etc.

class AskResponse(BaseModel):
    response: str               # LLM-generated answer
    session_id: str             # Echo back session ID
    source_documents: list[SourceDocument]  # Retrieved sources
```

### 5.2 Workflow State

```python
class AgentState(TypedDict, total=False):
    query: str
    rewritten_query: str
    retrieved_documents: list[Document]
    session_id: str
    conversation_history: list[BaseMessage]
    response: str
    summary_before_last_five_messages: str
```

### 5.3 Database Model

```python
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    messages = Column(String, nullable=False)  # JSON-serialized
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## 6. Prompt Architecture

### 6.1 Query Rewriter Prompt

**Purpose**: Transform conversational queries into retrieval-optimized forms

**Strategy**: Domain-aware rewriting that preserves intent while using Financial Compliance terminology

**Constraints**: No added facts, concise output, only the rewritten query returned

### 6.2 RAG Response Prompt

**System Prompt**: Establishes the LLM as a Financial Compliance assistant that:
- Uses only provided documents and conversation summary
- States "I don't know" when information is unavailable
- Grounds all responses in source material

**User Prompt Template**:
```
Query: {query}
Summary of conversation so far: {summary}
Documents: {documents}
```

### 6.3 Context Summary Prompt

**Purpose**: Compress conversation history into a concise summary

**Trigger**: Only when > 5 conversations exist for the session

**Output**: Single summary paragraph capturing the conversation trajectory

---

## 7. Concurrency and Resource Management

| Resource | Control Mechanism | Limit |
|----------|------------------|-------|
| LLM API Calls | Threading Semaphore | Max 10 concurrent |
| FastAPI Requests | Async event loop + thread pool | Limited by server workers |
| Database Writes | SQLAlchemy session scope + WAL | Single writer, concurrent reads |
| Pinecone Queries | No explicit limit | Subject to Pinecone plan limits |
| Embedding Generation | Per-request | Sequential within request |

---

## 8. Error Handling

| Layer | Strategy |
|-------|----------|
| **API** | FastAPI exception handlers, HTTP status codes, dependency validation at startup |
| **Workflow** | Try/except in each node, logging, graceful degradation |
| **Database** | Transaction rollback on exception, context manager pattern |
| **LLM** | Timeout (30s), retries (2), semaphore for rate limiting |
| **Vector DB** | Connection validation at startup, error logging |

---

## 9. Scalability Considerations

### Current Design (Single-Node)
- SQLite WAL mode supports concurrent reads but single writes
- Thread-based concurrency via FastAPI's thread pool
- Semaphore-controlled LLM rate limiting

### Scaling Path
1. **Vertical**: Increase thread pool size, LLM semaphore limit
2. **Database**: Migrate from SQLite to PostgreSQL for multi-writer support
3. **API**: Deploy multiple Uvicorn workers behind a load balancer
4. **Caching**: Add Redis for embedding cache and response cache
5. **Vector DB**: Pinecone cloud scales automatically with plan upgrades

---

## 10. Security Considerations

| Aspect | Implementation |
|--------|---------------|
| **API Keys** | Environment variables, never hardcoded |
| **Input Validation** | Pydantic models enforce request schema |
| **SQL Injection** | SQLAlchemy ORM parameterized queries |
| **CORS** | Configurable via FastAPI middleware |
| **Authentication** | Not implemented (future enhancement) |
| **Rate Limiting** | LLM semaphore; no API-level rate limiting (future) |

---

## 11. Directory-to-Layer Mapping

```
PRESENTATION:    UI/app.py
API:             api/main.py, api/routes/ask_endpoint.py
ORCHESTRATION:   workflow/graph.py, workflow/node_orchestrator.py
SERVICE:         workflow/service.py, workflow/protocols/
DATA ACCESS:     workflow/repositories/, workflow/database/
INFRASTRUCTURE:  workflow/llms/, workflow/embeddings/
CONFIGURATION:   workflow/config.py, workflow/configs/
PROMPTS:         workflow/prompts/
```
