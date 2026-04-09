# RAG Pipeline for Financial Compliance

A production-grade Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **FastAPI**, and **Pinecone**, specializing in SEBI (Securities and Exchange Board of India) financial compliance regulations. The system employs hybrid search (dense + sparse embeddings), multi-turn conversation context, and a pluggable LLM architecture served as a REST API.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Evaluation Results](#evaluation-results)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## Overview

This project implements an end-to-end RAG pipeline that answers questions about Indian financial compliance regulations (SEBI Master Circulars). The system:

1. **Ingests** PDF documents, chunks them, and stores embeddings in Pinecone
2. **Retrieves** relevant documents using hybrid search (OpenAI dense + SPLADE sparse embeddings)
3. **Generates** context-aware responses using LLMs with conversation history
4. **Persists** all interactions in SQLite for multi-turn conversation support
5. **Serves** everything through a FastAPI REST API with a Streamlit UI

## Architecture

```
User Query + Session ID
        |
        v
  [Query Rewriter] -- LLM rewrites for better retrieval
        |
        +---> [Document Fetcher]       (Hybrid search: Pinecone)
        |           |                         |
        +---> [Context Summarizer]     (Last 10 conversations from SQLite)
                    |                         |
                    +--------+--------+
                             |
                             v
                    [Response Generator] -- LLM synthesizes answer
                             |
                             v
                    [DB Writer] -- Persists to SQLite
                             |
                             v
                    Response + Source Documents
```

The workflow is orchestrated as a **LangGraph state machine** with 5 nodes, featuring parallel execution of document retrieval and context summarization (fan-out/fan-in pattern).

## Features

- **Hybrid Search**: Combines dense embeddings (OpenAI `text-embedding-3-small`) with sparse embeddings (SPLADE) for superior retrieval
- **Multi-Turn Conversations**: Session-based context management with automatic history summarization
- **Pluggable LLMs**: Switch between OpenAI (gpt-4o-mini, gpt-5-mini), Ollama (llama3.2), or a finetuned Qwen-0.5B model
- **Parallel Processing**: Document retrieval and context summarization execute simultaneously
- **Protocol-Based Design**: Abstract interfaces for LLM, Vector DB, and Database — easy to extend
- **Comprehensive Evaluation**: Both retriever-level (MRR, nDCG, Recall) and generation-level (RAGAS Faithfulness, Answer Relevancy) metrics
- **Production-Ready**: Docker support, load testing (Locust), async API, concurrency controls

## Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| Web Framework | FastAPI | Async, validation, dependency injection |
| Workflow Engine | LangGraph | State machine with 5 nodes, parallel execution |
| LLM Orchestration | LangChain | Message-based abstraction layer |
| Vector Database | Pinecone | Cloud-hosted, hybrid search (AWS us-east-1) |
| Embeddings (Dense) | OpenAI `text-embedding-3-small` | 1536 dimensions |
| Embeddings (Sparse) | SPLADE (`naver/splade-cocondenser-ensembledistil`) | Lexical matching |
| Database | SQLite (WAL mode) | Conversation persistence |
| Primary LLM | OpenAI `gpt-4o-mini` | Production model |
| Frontend | Streamlit | Chat interface |
| Load Testing | Locust | Concurrent user simulation |
| Evaluation | RAGAS + ranx | Faithfulness, relevancy, retriever metrics |
| Containerization | Docker | Separate API and UI containers |

## Project Structure

```
rag_pipeline/
├── api/
│   ├── main.py                          # FastAPI app entry point & dependency initialization
│   └── routes/
│       └── ask_endpoint.py              # POST /ask endpoint definition
├── workflow/
│   ├── graph.py                         # LangGraph workflow definition (state machine)
│   ├── node_orchestrator.py             # 5 node implementations
│   ├── service.py                       # RAGService utility methods
│   ├── state.py                         # AgentState TypedDict schema
│   ├── config.py                        # Pydantic Settings configuration
│   ├── configs/                         # Service-specific configs
│   │   ├── llm_config.py
│   │   ├── pinecone_config.py
│   │   └── db_config.py
│   ├── llms/                            # Pluggable LLM implementations
│   │   ├── openai.py                    # OpenAI ChatGPT (primary)
│   │   ├── ollama_llama.py              # Ollama local inference
│   │   └── finetuned_llm.py             # HuggingFace finetuned model
│   ├── embeddings/                      # Embedding strategies
│   │   ├── openai_embedding.py          # Dense: OpenAI text-embedding-3-small
│   │   ├── sentence_transformer_embedding.py  # Dense: all-MiniLM-L6-v2
│   │   └── sparse_embedding.py          # Sparse: SPLADE
│   ├── repositories/
│   │   └── pinecone_repository.py       # Vector DB queries (hybrid search)
│   ├── database/
│   │   ├── sessions.py                  # SQLAlchemy engine & session management
│   │   ├── base.py                      # Declarative base
│   │   ├── models/conversations.py      # Conversation ORM model
│   │   └── db_repositories/
│   │       └── conversation_repository.py  # CRUD operations
│   ├── protocols/                       # Abstract interfaces
│   │   ├── llm_protocol.py
│   │   ├── vector_db_protocol.py
│   │   └── database_repo_protocol.py
│   ├── strategies/                      # Embedding abstract base classes
│   │   ├── dense_embedding_strategy.py
│   │   └── sparse_embedding_strategy.py
│   └── prompts/                         # Domain-specific system prompts
│       ├── query_rewriter.py            # Query optimization prompt
│       ├── augment_query_rag.py         # RAG response generation prompt
│       └── summary_so_far.py            # Context summarization prompt
├── ragas_eval/                          # RAGAS evaluation notebooks
│   ├── ragas_gpt_4o_mini.ipynb
│   ├── ragas_gpt_5_mini.ipynb
│   └── finetuned_llm.ipynb
├── reference/                           # Architecture documentation
│   ├── ARCHITECTURE.md
│   ├── ARCHITECTURE_ASCII.txt
│   └── QUICK_REFERENCE.md
├── docs/                                # Project documentation
└── sample.db                            # SQLite database
```

## Quick Start

### Prerequisites

- Python >= 3.12
- OpenAI API key
- Pinecone API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag20march_with_eval

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .

# Set environment variables
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=...
```

### Run the API Server

```bash
uvicorn rag_pipeline.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run the Streamlit UI

```bash
cd UI
streamlit run app.py --server.port 8001
```

### Run with Docker

```bash
# Build and run API
docker build -f Dockerfiles/Dockerfile.rag.api -t rag-api .
docker run -p 8000:8000 --env-file ingestion.env rag-api

# Build and run UI
docker build -f Dockerfiles/Dockerfile.rag.ui -t rag-ui .
docker run -p 8001:8001 rag-ui
```

## API Usage

```bash
# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the disclosure requirements under SEBI regulations?",
    "session_id": "user-123"
  }'

# Response
{
  "response": "According to SEBI regulations...",
  "session_id": "user-123",
  "source_documents": [
    {
      "content": "Document text...",
      "metadata": {"page": 5, "source": "Master Circular..."}
    }
  ]
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API authentication |
| `PINECONE_API_KEY` | Required | Pinecone vector DB authentication |
| `DATABASE_URL` | `sqlite:///sample.db` | SQLite connection string |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |
| `PINECONE_INDEX` | `final-rag-index-openai-small` | Pinecone index name |
| `PINECONE_METRIC` | `dotproduct` | Similarity metric |

## Evaluation Results

### Retriever Evaluation (Hybrid Search, No Reranker)

| Metric | Sentence Transformers | OpenAI Small | OpenAI Large |
|--------|----------------------|--------------|--------------|
| MRR | 0.6078 | **0.6158** | 0.6012 |
| nDCG@10 | 0.6645 | **0.6933** | 0.6547 |
| Recall@5 | 0.8108 | **0.8378** | 0.7568 |
| Recall@10 | 0.8378 | **0.9369** | 0.8198 |
| Precision@5 | 0.1622 | **0.1676** | 0.1514 |

**Selected**: OpenAI `text-embedding-3-small` (best Recall@10 and nDCG@10)

### Generation Evaluation (RAGAS Metrics)

| Model | Faithfulness | Answer Relevancy |
|-------|-------------|-----------------|
| gpt-5-mini | **0.8987** | **0.9579** |
| gpt-4o-mini | 0.8427 | 0.9256 |
| finetuned-qwen-0.5B | 0.2333 | 0.9561 |

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Problem Statement](docs/problem_statement.md) - Business context and objectives
- [Data Documentation](docs/data_documentation.md) - Source data and processing pipeline
- [Modeling Report](docs/modeling_report.md) - Model selection and architecture decisions
- [Evaluation Report](docs/evaluation_report.md) - Comprehensive evaluation results
- [System Design](docs/system_design.md) - Architecture and design patterns
- [API Documentation](docs/api_docs.md) - Endpoint specifications
- [Deployment Guide](docs/deployment_guide.md) - Setup and deployment instructions
- [Model Card](docs/model_card.md) - Model specifications and limitations

---

**Domain**: SEBI Financial Compliance Regulations  
**Primary LLM**: OpenAI gpt-4o-mini  
**Vector DB**: Pinecone (AWS us-east-1)  
**Python**: >= 3.12

---

## Env requirements

- AWS_ACCESS_KEY_ID=access_key
- AWS_SECRET_ACCESS_KEY=secret_key
- AWS_REGION=region
- PINECONE_API_KEY=pinecone_api_key
- HF_HOME=storage
- HF_HUB_DISABLE_TELEMETRY=1
- TRANSFORMERS_OFFLINE=0
- OPENAI_API_KEY=open_ai_api_key
- HF_KEY=hf_key


