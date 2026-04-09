# RAG Pipeline - Quick Reference Guide

## System Overview
Your RAG pipeline is a **LangGraph-based workflow** served as a **FastAPI application** that retrieves documents from Pinecone and generates responses using LLMs with full conversation context.

## 5-Node Workflow

```
Query Rewriter → Parallel Execution → Response Generator → DB Writer
                ├─ Document Fetcher
                └─ Context Summarizer
```

| Node | Purpose | Input | Output |
|------|---------|-------|--------|
| **Query Rewriter** | Optimizes query for better retrieval | Original query | Rewritten query |
| **Document Fetcher** | Hybrid search (dense + sparse) | Rewritten query | Top-10 documents |
| **Context Summarizer** | Compresses conversation history | Last 10 conversations | Summary string |
| **Response Generator** | Synthesizes LLM response | All above + query | Final response |
| **DB Writer** | Persists interaction | Response + messages | Saved to SQLite |

## API Endpoint

```bash
POST /ask
Content-Type: application/json

Request:
{
  "query": "Your question here",
  "session_id": "user-identifier"
}

Response:
{
  "response": "Answer here...",
  "session_id": "user-identifier",
  "source_documents": [
    {
      "content": "Document text...",
      "metadata": {...}
    }
  ]
}
```

## Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| **Web Server** | FastAPI | Async, validation, dependency injection |
| **Workflow** | LangGraph | State machine with 5 nodes |
| **LLM Orchestration** | LangChain | Message-based abstraction |
| **Vector DB** | Pinecone | Cloud, hybrid search (dense + sparse) |
| **Database** | SQLite | WAL mode, conversation persistence |
| **Embeddings** | OpenAI (primary) + SPLADE (sparse) | Hybrid search strategy |
| **LLM (Primary)** | OpenAI gpt-4o-mini | Can switch to Ollama or Finetuned |

## Key Files & Locations

### Core Workflow
```
api/main.py                          → FastAPI app entry point
api/routes/ask_endpoint.py           → /ask endpoint definition
workflow/graph.py                    → LangGraph workflow definition
workflow/node_orchestrator.py        → 5 node implementations
workflow/service.py                  → RAGService utility methods
workflow/state.py                    → AgentState schema
```

### External Services
```
workflow/repositories/pinecone_repository.py → Vector DB queries
workflow/database/db_repositories/          → SQLite operations
workflow/llms/                              → LLM implementations (openai.py, ollama.py, finetuned_llm.py)
workflow/embeddings/                        → Embedding strategies (dense & sparse)
```

### Configuration
```
workflow/config.py                   → Main settings
workflow/configs/                    → Service-specific configs (LLM, Pinecone, DB)
```

### Prompts
```
workflow/prompts/query_rewriter.py   → Query rewriting prompt
workflow/prompts/augment_query_rag.py → RAG system + user prompts
workflow/prompts/summary_so_far.py   → Context summarization prompt
```

## Data Flow at a Glance

```
1. User sends query + session_id
   ↓
2. Rewrite query for better retrieval
   ↓
3. Execute in parallel:
   - Retrieve top-10 documents from Pinecone
   - Summarize last 5 conversation turns from SQLite
   ↓
4. LLM generates response using: query + documents + context
   ↓
5. Save to SQLite for future context
   ↓
6. Return response + source documents to user
```

## Configuration Variables

```bash
# Critical (Required)
OPENAI_API_KEY=sk_...
PINECONE_API_KEY=...

# Optional (Has defaults)
DATABASE_URL=sqlite:///sample.db
OPENAI_MODEL=gpt-4o-mini
PINECONE_INDEX=final-rag-index-openai-small
PINECONE_METRIC=dotproduct
```

## Running the Pipeline

```bash
# Start API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Test endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is financial compliance?",
    "session_id": "test-user-1"
  }'
```

## Key Design Patterns

### Parallelization
- Document retrieval and context summarization run **simultaneously** (fan-out/fan-in)
- API handles multiple requests asynchronously

### Pluggable Components
- **LLM**: Switch between OpenAI, Ollama, or Finetuned models
- **Embeddings**: OpenAI dense + SPLADE sparse for hybrid search
- **Vector DB**: Abstract protocol allows swapping Pinecone

### Session Management
- Each user has a `session_id`
- Conversations stored with session tracking
- Context from last 5 turns automatically summarized

### Protocol-Based Design
- LLM interface: `invoke(messages) → message`
- Vector DB interface: `query(text) → list[Document]`
- Easy to implement alternative providers

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **LLM Concurrency** | Max 10 | Controlled via Semaphore |
| **Vector Batch Size** | 200 | Pinecone bulk operations |
| **Conversation History** | Last 10 | Retrieved for summarization |
| **Document Retrieval** | Top 10 | Hybrid rank fusion |
| **Database Mode** | SQLite WAL | Allows concurrent reads |

## Extension Points

### Add a New LLM Provider
1. Create class in `workflow/llms/new_provider.py`
2. Implement `LLMProtocol` interface
3. Update `workflow/configs/llm_config.py`
4. Initialize in `api/main.py`

### Add a New Embedding Strategy
1. Create class in `workflow/embeddings/new_strategy.py`
2. Implement dense/sparse embedding interface
3. Update Pinecone configuration
4. Initialize in `api/main.py`

### Modify Workflow
1. Edit `workflow/graph.py` to add/remove nodes
2. Update `workflow/state.py` if state schema changes
3. Add node implementation in `workflow/node_orchestrator.py`
4. Update service calls in `workflow/service.py`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY not found` | Set env var: `export OPENAI_API_KEY=sk_...` |
| `Pinecone connection error` | Check `PINECONE_API_KEY` and network access |
| `SQLite database locked` | Wait for WAL checkpoint, or restart app |
| `Empty documents returned` | Check query rewriting, embeddings, or document index |
| `Slow response time` | Check LLM concurrency limits, Pinecone latency |

## Directory Structure

```
rag_pipeline/
├── api/
│   ├── main.py                 (FastAPI app)
│   └── routes/
│       └── ask_endpoint.py     (/ask route)
├── workflow/
│   ├── graph.py                (LangGraph definition)
│   ├── node_orchestrator.py    (5 node implementations)
│   ├── service.py              (RAGService)
│   ├── state.py                (AgentState)
│   ├── llms/                   (LLM implementations)
│   ├── embeddings/             (Embedding strategies)
│   ├── repositories/           (Pinecone access)
│   ├── database/               (SQLite models & access)
│   ├── configs/                (Configuration)
│   ├── protocols/              (Abstract interfaces)
│   └── prompts/                (System prompts)
├── ARCHITECTURE.md             (Full Mermaid diagrams)
├── ARCHITECTURE_ASCII.txt      (ASCII diagrams)
├── pyproject.toml              (Dependencies)
└── sample.db                   (SQLite database)
```

## Resources

- **Mermaid Diagrams**: See `ARCHITECTURE.md` for visual representations
- **ASCII Diagrams**: See `ARCHITECTURE_ASCII.txt` for text-based layout
- **Code Structure**: Follow imports in `api/main.py` to understand initialization order
- **Prompts**: Check `workflow/prompts/` for domain-specific instructions

---

**Last Updated**: 2026-04-09
**Domain**: Financial Compliance
**Primary LLM**: OpenAI gpt-4o-mini
**Vector DB**: Pinecone (AWS us-east-1)
