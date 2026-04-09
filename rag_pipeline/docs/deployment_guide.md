# Deployment Guide

## 1. Overview

This guide covers setting up, configuring, and deploying the RAG Pipeline for SEBI Financial Compliance. The system consists of two deployable components: a **FastAPI backend** (API server) and a **Streamlit frontend** (chat UI). Both can be deployed locally, via Docker, or on cloud infrastructure.

---

## 2. Prerequisites

### 2.1 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.12+ | 3.12+ |
| **RAM** | 4 GB | 8 GB (16 GB if using local models) |
| **Disk** | 2 GB | 5 GB (includes model weights) |
| **OS** | Linux, macOS, Windows | Linux or macOS |
| **Docker** | 20.10+ (optional) | Latest stable |

### 2.2 External Service Accounts

| Service | Required | Purpose | Sign-up |
|---------|----------|---------|---------|
| **OpenAI** | Yes | LLM (gpt-4o-mini) + Dense Embeddings | https://platform.openai.com |
| **Pinecone** | Yes | Vector database hosting | https://www.pinecone.io |
| **Ollama** | Optional | Local LLM alternative | https://ollama.com |

### 2.3 API Keys

```bash
# Required
OPENAI_API_KEY=sk-...           # OpenAI API key
PINECONE_API_KEY=...            # Pinecone API key

# Optional
HF_KEY=...                      # HuggingFace token (for finetuned model)
```

---

## 3. Local Development Setup

### 3.1 Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd rag20march_with_eval

# Option A: Install with uv (recommended, faster)
pip install uv
uv sync

# Option B: Install with pip
pip install -e .
```

### 3.2 Environment Configuration

Create or update the environment file:

```bash
# Create environment file
cat > ingestion.env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key-here
DATABASE_URL=sqlite:///sample.db
EOF
```

Or export directly:

```bash
export OPENAI_API_KEY=sk-your-key-here
export PINECONE_API_KEY=your-pinecone-key-here
export DATABASE_URL=sqlite:///sample.db
```

### 3.3 Data Ingestion (First-Time Setup)

Before running the API, you need to ingest documents into Pinecone:

```bash
# Ensure your PDF documents are in the Documents directory
ls Ingestion_plus_Retriever_eval/Documents/

# Run the ingestion pipeline
cd Ingestion_plus_Retriever_eval
python main.py
```

This will:
1. Load all PDFs from `Documents/`
2. Split them into ~1,108 chunks (1000 chars, 200 overlap)
3. Generate dense embeddings (OpenAI `text-embedding-3-small`)
4. Generate sparse embeddings (SPLADE)
5. Upsert all vectors to Pinecone index `final-rag-index-openai-small`

### 3.4 Start the API Server

```bash
# From the project root
uvicorn rag_pipeline.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 3.5 Start the Streamlit UI

```bash
# In a separate terminal
cd UI
streamlit run app.py --server.port 8001
```

### 3.6 Verify Installation

```bash
# Health check
curl http://localhost:8000/

# Test query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is financial compliance?",
    "session_id": "test-user-1"
  }'
```

---

## 4. Docker Deployment

### 4.1 Dockerfile.rag.api

The API Dockerfile uses Python 3.12-slim with `uv` for fast dependency resolution:

```dockerfile
# Key details:
# - Base: python:3.12-slim
# - Package manager: uv
# - Port: 8000
# - Entry: uvicorn rag_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

### 4.2 Dockerfile.rag.ui

The UI Dockerfile runs the Streamlit application:

```dockerfile
# Key details:
# - Base: python:3.12-slim
# - Port: 8001
# - Entry: streamlit run app.py --server.port 8001
```

### 4.3 Build and Run

```bash
# Build API container
docker build -f Dockerfiles/Dockerfile.rag.api -t rag-api .

# Build UI container
docker build -f Dockerfiles/Dockerfile.rag.ui -t rag-ui .

# Run API with environment variables
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e PINECONE_API_KEY=... \
  -e DATABASE_URL=sqlite:///sample.db \
  rag-api

# Run UI
docker run -d \
  --name rag-ui \
  -p 8001:8001 \
  -e RAG_API_URL=http://rag-api:8000 \
  --link rag-api \
  rag-ui
```

### 4.4 Docker Compose (Example)

```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfiles/Dockerfile.rag.api
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - DATABASE_URL=sqlite:///sample.db
    volumes:
      - ./sample.db:/app/sample.db

  ui:
    build:
      context: .
      dockerfile: Dockerfiles/Dockerfile.rag.ui
    ports:
      - "8001:8001"
    environment:
      - RAG_API_URL=http://api:8000
    depends_on:
      - api
```

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

---

## 5. Configuration Reference

### 5.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for LLM and embeddings |
| `PINECONE_API_KEY` | Yes | — | Pinecone vector database key |
| `DATABASE_URL` | No | `sqlite:///sample.db` | SQLAlchemy connection string |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model for generation |
| `PINECONE_INDEX` | No | `final-rag-index-openai-small` | Pinecone index name |
| `PINECONE_METRIC` | No | `dotproduct` | Similarity metric |
| `RAG_API_URL` | No | `http://localhost:8000` | API URL for Streamlit UI |

### 5.2 Application Settings

Managed via `workflow/config.py` (Pydantic BaseSettings):

| Setting | Default | Description |
|---------|---------|-------------|
| `database_url` | `sqlite:///sample.db` | Database connection |
| `llm_model_name` | `llama3.2` | Ollama model name |
| `openai_model_name_4o_mini` | `gpt-4o-mini` | OpenAI primary model |
| `openai_model_name_5_mini` | `gpt-5-mini` | OpenAI alternative model |
| `pinecone_index_name` | `final-rag-index-openai-small` | Vector index |
| `pinecone_metric` | `dotproduct` | Similarity metric |
| `pinecone_batch_size` | `200` | Upsert batch size |
| `pinecone_cloud` | `aws` | Cloud provider |
| `pinecone_region` | `us-east-1` | Cloud region |
| `log_level` | `INFO` | Logging level |
| `environment` | `development` | App environment |

### 5.3 Pinecone Index Configuration

If creating a new Pinecone index, use these settings:

| Parameter | Value |
|-----------|-------|
| **Index Name** | `final-rag-index-openai-small` |
| **Metric** | Dot Product |
| **Dimensions** | 1536 (for OpenAI `text-embedding-3-small`) |
| **Cloud** | AWS |
| **Region** | us-east-1 |
| **Pod Type** | Starter (free tier) or s1/p1 (production) |

---

## 6. Database Management

### 6.1 SQLite Database

The application uses SQLite with WAL (Write-Ahead Logging) mode:

```bash
# Database file location
ls sample.db

# View database schema (using sqlite3 CLI)
sqlite3 sample.db ".schema"

# View conversation count
sqlite3 sample.db "SELECT COUNT(*) FROM conversations;"

# View conversations for a session
sqlite3 sample.db "SELECT * FROM conversations WHERE session_id='user-123' ORDER BY created_at DESC LIMIT 5;"

# Clear all conversations
sqlite3 sample.db "DELETE FROM conversations;"
```

### 6.2 Database Initialization

The database and tables are created automatically on application startup via SQLAlchemy's `create_all()`. No manual migration is needed.

---

## 7. Running with Alternative LLMs

### 7.1 Ollama (Local)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2

# Start Ollama server
ollama serve

# Update the application to use OllamaLLM in api/main.py
```

### 7.2 Finetuned Model (Local)

The finetuned Qwen-0.5B model weights are located at `finetuned_model/`:

```bash
ls finetuned_model/
# config.json  generation_config.json  model.safetensors  tokenizer.json  tokenizer_config.json  chat_template.jinja
```

To use the finetuned model, update the LLM initialization in `api/main.py` to use `FinetunedLLM` instead of `OpenAILLM`.

> **Warning**: The finetuned Qwen-0.5B model has low faithfulness (0.2333) and is not recommended for production compliance use.

---

## 8. Load Testing

### 8.1 Setup

```bash
# Install Locust (included in dependencies)
pip install locust

# Run load tests
cd load_testing
locust -f locustfile.py --host http://localhost:8000
```

### 8.2 Usage

1. Open Locust UI at `http://localhost:8089`
2. Configure number of users and spawn rate
3. Start the test
4. Monitor response times, failure rates, and throughput

### 8.3 Test Configuration

The Locust test file includes:
- 8 predefined SEBI compliance queries
- Unique session IDs per simulated user
- 1-3 second think time between requests
- Tests the `POST /ask` endpoint

---

## 9. Monitoring and Logging

### 9.1 Application Logs

The application uses Python's `logging` module with configurable level:

```bash
# Set log level via environment
export LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR

# View logs during development
uvicorn rag_pipeline.api.main:app --reload --log-level debug
```

### 9.2 Key Log Points

| Component | Log Level | What It Logs |
|-----------|-----------|-------------|
| API Startup | INFO | Dependency initialization, validation results |
| Query Rewriter | DEBUG | Original and rewritten queries |
| Document Fetcher | INFO | Number of documents retrieved |
| Context Summarizer | DEBUG | Summary content |
| LLM Call | INFO | Response generation time |
| DB Writer | DEBUG | Conversation save confirmation |

---

## 10. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|---------|
| `OPENAI_API_KEY not found` | Environment variable not set | `export OPENAI_API_KEY=sk-...` |
| `Pinecone connection error` | Invalid API key or network issue | Verify `PINECONE_API_KEY` and internet access |
| `SQLite database locked` | Concurrent write attempts | Wait for WAL checkpoint; restart app |
| `Empty documents returned` | Index empty or query mismatch | Run ingestion pipeline; check embeddings |
| `Slow response time` | LLM latency or rate limiting | Check semaphore limits; verify API key quota |
| `Module not found` | Dependencies not installed | Run `uv sync` or `pip install -e .` |
| `Port already in use` | Another process on port 8000 | `lsof -i :8000` and kill the process |
| `Docker build fails` | Missing build dependencies | Ensure `build-essential` is in Dockerfile |
| `Streamlit can't connect to API` | Wrong API URL | Set `RAG_API_URL=http://localhost:8000` |

---

## 11. Production Checklist

- [ ] Set all required environment variables (`OPENAI_API_KEY`, `PINECONE_API_KEY`)
- [ ] Run data ingestion pipeline to populate Pinecone index
- [ ] Verify API health check returns 200
- [ ] Test with sample queries to confirm end-to-end functionality
- [ ] Configure appropriate log level (INFO or WARNING for production)
- [ ] Set up Docker containers for reproducible deployment
- [ ] Run load tests to verify performance under expected load
- [ ] Exclude `ingestion.env` and `sample.db` from version control
- [ ] Back up Pinecone index configuration
- [ ] Document the `session_id` scheme for downstream integrations

---

## 12. Dependencies

### 12.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >= 0.135.2 | Web framework |
| `uvicorn` | >= 0.42.0 | ASGI server |
| `langgraph` | >= 1.1.3 | Workflow engine |
| `langchain` | >= 1.2.13 | LLM orchestration |
| `langchain-core` | >= 1.2.20 | Core abstractions |
| `langchain-community` | >= 0.4.1 | Community integrations |
| `langchain-huggingface` | >= 1.2.1 | HuggingFace LLM support |
| `langchain-ollama` | >= 1.0.1 | Ollama LLM support |
| `langchain-text-splitters` | >= 1.1.1 | Text chunking |
| `pinecone` | >= 8.1.0 | Vector database client |
| `sentence-transformers` | >= 5.3.0 | Embedding models |

### 12.2 Evaluation Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ragas` | >= 0.4.3 | RAG evaluation framework |
| `ranx` | >= 0.3.21 | IR ranking metrics |

### 12.3 Additional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | >= 1.40.0 | Chat UI |
| `locust` | >= 2.43.3 | Load testing |
| `pandas` | >= 2.0.0 | Data manipulation |
| `pypdf` | >= 5.0.0 | PDF parsing |
| `accelerate` | >= 1.0.0 | Model inference acceleration |
| `litellm` | >= 1.0.0 | LLM abstraction |
