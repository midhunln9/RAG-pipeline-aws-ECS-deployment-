# API Documentation

## 1. Overview

The RAG Pipeline API is a REST API built with **FastAPI** that provides a question-answering interface over SEBI Financial Compliance regulations. The API accepts natural language queries with session tracking and returns LLM-generated responses grounded in regulatory source documents.

**Base URL**: `http://localhost:8000`  
**Server**: Uvicorn ASGI server  
**Framework**: FastAPI with Pydantic validation  

---

## 2. Endpoints

### 2.1 Ask Question

Submits a compliance question and returns an AI-generated response with source documents.

```
POST /ask
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | `string` | Yes | The user's natural language question about financial compliance |
| `session_id` | `string` | Yes | Unique identifier for the user's conversation session. Used for multi-turn context management. |

**Example Request**:
```json
{
  "query": "What are the eligibility criteria for companies planning an IPO under SEBI guidelines?",
  "session_id": "user-abc-123"
}
```

#### Response Body

| Field | Type | Description |
|-------|------|-------------|
| `response` | `string` | The AI-generated answer grounded in SEBI regulatory documents |
| `session_id` | `string` | Echo of the session identifier from the request |
| `source_documents` | `array[SourceDocument]` | List of source documents used to generate the response |

**SourceDocument Object**:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `string` | The text content of the retrieved document chunk |
| `metadata` | `object` | Metadata about the source (file name, page number, etc.) |

**Example Response**:
```json
{
  "response": "According to SEBI regulations, the eligibility criteria for companies planning an IPO include: (1) The company must have a net tangible asset of at least Rs. 3 crore in each of the preceding three full years, (2) The company must have a minimum average pre-tax operating profit of Rs. 15 crore during the three most profitable years out of the immediately preceding five years...",
  "session_id": "user-abc-123",
  "source_documents": [
    {
      "content": "Eligibility requirements for Initial Public Offering: A company can make an initial public offering only if it meets the following conditions...",
      "metadata": {
        "source": "Master Circular for Issue of Capital and Disclosure Requirements.pdf",
        "page": 12
      }
    },
    {
      "content": "The issuer company shall also satisfy the following conditions for making an IPO...",
      "metadata": {
        "source": "Master Circular for Issue of Capital and Disclosure Requirements.pdf",
        "page": 13
      }
    }
  ]
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| `200 OK` | Successful response |
| `422 Unprocessable Entity` | Invalid request body (missing or malformed fields) |
| `500 Internal Server Error` | Workflow execution failure, LLM error, or database error |

#### cURL Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the disclosure requirements for preferential allotment?",
    "session_id": "user-123"
  }'
```

#### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "query": "What are the disclosure requirements for preferential allotment?",
        "session_id": "user-123"
    }
)

data = response.json()
print(data["response"])
for doc in data["source_documents"]:
    print(f"Source: {doc['metadata']['source']}, Page: {doc['metadata']['page']}")
```

---

### 2.2 Health Check

Verifies the API server is running and responsive.

```
GET /
```

#### Response

```json
{
  "message": "RAG Pipeline API is running"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| `200 OK` | Server is healthy |

---

## 3. Data Models

### 3.1 AskRequest

```python
class AskRequest(BaseModel):
    query: str          # The user's natural language question
    session_id: str     # Unique session identifier for context tracking
```

**Validation Rules**:
- Both fields are required (non-optional)
- Both must be strings
- No length constraints enforced (relies on downstream handling)

### 3.2 AskResponse

```python
class AskResponse(BaseModel):
    response: str                           # Generated answer
    session_id: str                         # Echo of request session_id
    source_documents: list[SourceDocument]  # Retrieved source documents
```

### 3.3 SourceDocument

```python
class SourceDocument(BaseModel):
    content: str      # Document chunk text
    metadata: dict    # Source metadata (file, page, etc.)
```

**Common Metadata Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `source` | `string` | Original PDF filename |
| `page` | `integer` | Page number in the source PDF |

---

## 4. Session Management

### 4.1 How Sessions Work

The `session_id` field enables multi-turn conversations:

1. **First Query**: User provides a `session_id` (e.g., UUID). No prior context exists.
2. **Subsequent Queries**: Same `session_id` triggers retrieval of conversation history.
3. **Context Summarization**: Last 10 conversations are fetched; if > 5 exist, they are summarized by the LLM.
4. **Response Generation**: The LLM uses both retrieved documents AND conversation context to generate responses.

### 4.2 Session Lifecycle

```
Query 1 (session_id: "abc"):
  → No history → Fresh response → Saved to DB

Query 2 (session_id: "abc"):
  → 1 prior conversation found → Used as context → Saved to DB

Query 6 (session_id: "abc"):
  → 5 prior conversations found → LLM summarizes history → Used as context → Saved to DB
```

### 4.3 Session Best Practices

- Use UUIDs or unique identifiers for `session_id`
- Different users should have different session IDs
- To start a fresh conversation, use a new `session_id`
- Session data persists in SQLite until manually cleared

---

## 5. Internal Processing Pipeline

When a request hits the `/ask` endpoint, the following pipeline executes:

```
1. Request Validation (FastAPI/Pydantic)
   ↓
2. AgentState Initialization
   {query, session_id, empty lists}
   ↓
3. Query Rewriting (Node 1)
   LLM optimizes query for retrieval
   ↓
4. Parallel Execution (Nodes 2 & 3)
   ├─ Document Fetcher: Hybrid search on Pinecone (top-10)
   └─ Context Summarizer: Last 10 conversations from SQLite
   ↓
5. Response Generation (Node 4)
   LLM synthesizes answer from documents + context + query
   ↓
6. Database Persistence (Node 5)
   Save conversation to SQLite
   ↓
7. Response Construction
   Extract response + source_documents from workflow state
   ↓
8. HTTP Response (200 OK)
```

---

## 6. Error Handling

### 6.1 Validation Errors (422)

Returned when the request body doesn't match the expected schema:

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 6.2 Server Errors (500)

Returned when an internal error occurs during processing. Common causes:
- OpenAI API key invalid or rate-limited
- Pinecone connection failure
- SQLite database locked
- LLM timeout (> 30 seconds)

### 6.3 Startup Validation

The API validates all dependencies at startup. If any are missing, the server fails to start with a descriptive error:
- `OPENAI_API_KEY` not set
- `PINECONE_API_KEY` not set
- `DATABASE_URL` not configured

---

## 7. Configuration

### 7.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API authentication key |
| `PINECONE_API_KEY` | Yes | — | Pinecone vector database key |
| `DATABASE_URL` | No | `sqlite:///sample.db` | SQLite connection string |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | LLM model identifier |
| `PINECONE_INDEX` | No | `final-rag-index-openai-small` | Pinecone index name |
| `PINECONE_METRIC` | No | `dotproduct` | Similarity metric |

### 7.2 Server Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Host | `0.0.0.0` | Bind address |
| Port | `8000` | Listening port |
| Workers | 1 (dev) | Uvicorn worker count |
| Reload | `--reload` (dev) | Auto-reload on code changes |

---

## 8. Running the API

### 8.1 Development

```bash
# Set required environment variables
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=...

# Start with auto-reload
uvicorn rag_pipeline.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 8.2 Production (Docker)

```bash
# Build
docker build -f Dockerfiles/Dockerfile.rag.api -t rag-api .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e PINECONE_API_KEY=... \
  rag-api
```

### 8.3 Interactive Documentation

FastAPI automatically generates interactive API documentation:

| URL | Description |
|-----|-------------|
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc (read-only) |
| `http://localhost:8000/openapi.json` | OpenAPI schema (JSON) |

---

## 9. Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Max Concurrent LLM Calls | 10 | Controlled via semaphore |
| LLM Timeout | 30 seconds | Per invocation |
| LLM Retries | 2 | On transient failures |
| Documents Retrieved | Top 10 | Per query, hybrid search |
| Conversation History | Last 10 | Retrieved for context |
| Database Mode | SQLite WAL | Concurrent reads supported |
| Request Handling | Async | Via `asyncio.to_thread()` |

---

## 10. Load Testing

The project includes a Locust load testing configuration:

```bash
# Run load tests
cd load_testing
locust -f locustfile.py --host http://localhost:8000
```

**Test Configuration**:
- 8 predefined compliance queries covering various SEBI topics
- Each simulated user gets a unique `session_id`
- 1-3 second wait time between requests
- Configurable concurrent user count via Locust UI (http://localhost:8089)

---

## 11. Integration Examples

### 11.1 Streamlit Frontend

The included Streamlit UI (`UI/app.py`) demonstrates frontend integration:
- Chat-style interface with message history
- Source document display with metadata
- Session management with UUID generation
- Error handling for API connectivity issues

### 11.2 Custom Integration

```python
import requests
import uuid

class RAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())
    
    def ask(self, query: str) -> dict:
        response = requests.post(
            f"{self.base_url}/ask",
            json={"query": query, "session_id": self.session_id},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def new_session(self):
        self.session_id = str(uuid.uuid4())

# Usage
client = RAGClient()
result = client.ask("What is the lock-in period for promoter holdings?")
print(result["response"])

# Follow-up question (same session)
result = client.ask("What about for anchor investors?")
print(result["response"])

# Start new conversation
client.new_session()
```
