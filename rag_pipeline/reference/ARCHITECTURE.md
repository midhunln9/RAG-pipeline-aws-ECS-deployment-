# RAG Pipeline System Architecture

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        Client["User/Client"]
    end
    
    subgraph "API Layer"
        FastAPI["FastAPI Server<br/>Port 8000"]
        AskEndpoint["POST /ask<br/>Endpoint"]
    end
    
    subgraph "Application Layer"
        LGWorkflow["LangGraph Workflow<br/>State Machine"]
        RAGService["RAGService<br/>Orchestrator"]
        NodeOrch["Node Orchestrator<br/>5 Nodes"]
    end
    
    subgraph "Processing Layer"
        QueryRewriter["Query Rewriter<br/>Node"]
        DocRetriever["Document Retriever<br/>Node"]
        ContextSummarizer["Context Summarizer<br/>Node"]
        LLMGenerator["LLM Response<br/>Generator Node"]
        DBWriter["DB Writer<br/>Node"]
    end
    
    subgraph "Infrastructure Layer"
        subgraph "Vector Store"
            Pinecone["Pinecone<br/>Vector Database"]
            DenseEmbed["Dense Embeddings<br/>OpenAI text-embedding-3-small"]
            SparseEmbed["Sparse Embeddings<br/>SPLADE Model"]
        end
        
        subgraph "LLM Providers"
            OpenAILLM["OpenAI<br/>gpt-4o-mini"]
            OllamaLLM["Ollama<br/>llama3.2"]
            FinetuneLL["Finetuned LLM<br/>Local Model"]
        end
        
        SQLiteDB["SQLite Database<br/>Conversations"]
    end
    
    subgraph "Configuration"
        Config["Settings & Config<br/>Environment Variables"]
    end
    
    Client -->|Query Request| FastAPI
    FastAPI --> AskEndpoint
    AskEndpoint -->|Invoke Workflow| LGWorkflow
    LGWorkflow --> RAGService
    RAGService --> NodeOrch
    NodeOrch --> QueryRewriter
    NodeOrch --> DocRetriever
    NodeOrch --> ContextSummarizer
    NodeOrch --> LLMGenerator
    NodeOrch --> DBWriter
    
    QueryRewriter -->|Uses| OpenAILLM
    ContextSummarizer -->|Uses| OpenAILLM
    LLMGenerator -->|Uses| OpenAILLM
    
    DocRetriever -->|Queries| Pinecone
    Pinecone -->|Uses| DenseEmbed
    Pinecone -->|Uses| SparseEmbed
    DenseEmbed -->|Calls| OpenAILLM
    
    DBWriter -->|Stores| SQLiteDB
    
    Config -.->|Configures| FastAPI
    Config -.->|Configures| OpenAILLM
    Config -.->|Configures| Pinecone
    
    FastAPI -->|Response| Client
```

## 2. LangGraph Workflow State Machine

```mermaid
graph LR
    Start([START]) --> QR["⚙️ Query Rewriter<br/>Rewrites query for<br/>better retrieval"]
    
    QR --> Fork{"Parallel<br/>Execution"}
    
    Fork -->|Branch 1| DR["📚 Fetch Documents<br/>Hybrid Search<br/>Dense + Sparse"]
    Fork -->|Branch 2| CS["💬 Context Summarizer<br/>Summarizes Last<br/>5 Messages"]
    
    DR --> Join{"Join<br/>Results"}
    CS --> Join
    
    Join --> LG["🔬 LLM Call<br/>Generates Response<br/>from Documents +<br/>Context"]
    
    LG --> DB["💾 Add to Database<br/>Persists Conversation<br/>to SQLite"]
    
    DB --> End([END])
    
    style Start fill:#90EE90
    style End fill:#FFB6C6
    style Fork fill:#87CEEB
    style Join fill:#87CEEB
    style QR fill:#FFE4B5
    style DR fill:#FFE4B5
    style CS fill:#FFE4B5
    style LG fill:#FFE4B5
    style DB fill:#FFE4B5
```

## 3. Request/Response Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI<br/>Server
    participant WF as LangGraph<br/>Workflow
    participant LLM as LLM Services
    participant VS as Pinecone<br/>Vector Store
    participant DB as SQLite<br/>Database
    
    Client->>API: POST /ask<br/>{query, session_id}
    API->>WF: Execute workflow
    
    WF->>LLM: Rewrite query
    LLM-->>WF: Rewritten query
    
    par
        WF->>VS: Retrieve documents<br/>(rewritten query)
        VS-->>WF: Top-k documents
    and
        WF->>DB: Get conversation history
        DB-->>WF: Last 10 conversations
        WF->>LLM: Summarize context
        LLM-->>WF: Context summary
    end
    
    WF->>LLM: Generate response<br/>(query + docs + context)
    LLM-->>WF: LLM response
    
    WF->>DB: Save conversation<br/>(query, response, messages)
    DB-->>WF: Saved
    
    WF-->>API: {response, documents}
    API-->>Client: 200 OK<br/>{response, session_id,<br/>source_documents}
```

## 4. Component Architecture Layers

```mermaid
graph TB
    subgraph L1 ["🌐 Presentation Layer"]
        L1A["User/Client Applications"]
    end
    
    subgraph L2 ["🔌 API Layer"]
        L2A["FastAPI Server"]
        L2B["POST /ask Endpoint"]
        L2C["Request/Response Models"]
    end
    
    subgraph L3 ["⚙️ Orchestration Layer"]
        L3A["LangGraph Workflow Manager"]
        L3B["State Machine Executor"]
        L3C["Node Orchestrator"]
    end
    
    subgraph L4 ["🧠 Service Layer"]
        L4A["RAGService"]
        L4B["Query Rewriting Service"]
        L4C["Document Retrieval Service"]
        L4D["Context Summarization Service"]
        L4E["Response Generation Service"]
        L4F["Persistence Service"]
    end
    
    subgraph L5 ["📦 Data Access Layer"]
        L5A["Pinecone Repository"]
        L5B["Conversation Repository"]
        L5C["Embedding Strategies"]
    end
    
    subgraph L6 ["🗄️ Infrastructure Layer"]
        L6A["Pinecone Vector DB"]
        L6B["SQLite Database"]
        L6C["OpenAI API"]
        L6D["Ollama Local Server"]
        L6E["Finetuned Model"]
    end
    
    L1A --> L2A
    L2B --> L3A
    L3A --> L3B
    L3B --> L3C
    L3C --> L4A
    L4A --> L4B
    L4A --> L4C
    L4A --> L4D
    L4A --> L4E
    L4A --> L4F
    L4B --> L5C
    L4C --> L5A
    L4D --> L5B
    L4E --> L5C
    L4F --> L5B
    L5A --> L6A
    L5A --> L6C
    L5B --> L6B
    L6C --> L6C
    L4B --> L6C
    L4D --> L6C
    L4E --> L6C
```

## 5. Data Models & State Management

```mermaid
graph TB
    subgraph "Input"
        AskReq["AskRequest<br/>- query: str<br/>- session_id: str"]
    end
    
    subgraph "Workflow State"
        AgentState["AgentState<br/>- query: str<br/>- rewritten_query: str<br/>- retrieved_documents: List<br/>- session_id: str<br/>- conversation_history: List<br/>- response: str<br/>- summary_before_last_five"]
    end
    
    subgraph "Database Models"
        ConvModel["Conversation<br/>- id: int<br/>- session_id: str<br/>- messages: str<br/>- created_at: datetime"]
    end
    
    subgraph "Output"
        AskResp["AskResponse<br/>- response: str<br/>- session_id: str<br/>- source_documents"]
    end
    
    AskReq -->|Initialize| AgentState
    AgentState -->|Transform| ConvModel
    ConvModel -->|Return| AskResp
```

## 6. External Dependencies & Integrations

```mermaid
graph TB
    subgraph "RAG Pipeline"
        Core["RAG Pipeline Core"]
    end
    
    subgraph "LLM Providers (Pluggable)"
        direction LR
        OAI["OpenAI<br/>gpt-4o-mini<br/>Active"]
        OLL["Ollama<br/>llama3.2<br/>Local Alternative"]
        FT["Finetuned Model<br/>Local Alternative"]
    end
    
    subgraph "Embedding Models"
        direction LR
        DE["Dense: OpenAI<br/>text-embedding-3-small<br/>Active"]
        SE["Sparse: SPLADE<br/>naver/splade-cocondenser"]
    end
    
    subgraph "Vector Database"
        PV["Pinecone<br/>Cloud - AWS us-east-1<br/>Index: final-rag-index<br/>Metric: dotproduct"]
    end
    
    subgraph "Relational Database"
        SQL["SQLite<br/>Local File: sample.db<br/>Mode: WAL"]
    end
    
    subgraph "Configuration"
        ENV["Environment Variables<br/>OPENAI_API_KEY<br/>PINECONE_API_KEY"]
    end
    
    Core --> OAI
    Core --> OLL
    Core --> FT
    Core --> DE
    Core --> SE
    Core --> PV
    Core --> SQL
    ENV -.->|Configures| Core
```

## 7. Processing Pipeline Flow

```mermaid
graph LR
    Input["User Query<br/>+ Session ID"]
    
    Step1["1️⃣ Query Rewriting<br/>Optimize for retrieval"]
    Step2a["2️⃣ Document Retrieval<br/>Hybrid search"]
    Step2b["2️⃣ Context Summarization<br/>Last 5 turns"]
    Step3["3️⃣ Response Generation<br/>LLM synthesis"]
    Step4["4️⃣ Persistence<br/>Save to DB"]
    Output["Response + Sources"]
    
    Input --> Step1
    Step1 --> Step2a
    Step1 --> Step2b
    Step2a -->|Parallel| Step3
    Step2b -->|Parallel| Step3
    Step3 --> Step4
    Step4 --> Output
    
    style Input fill:#E8F5E9
    style Output fill:#E8F5E9
    style Step3 fill:#FFF3E0
    style Step4 fill:#E3F2FD
```

## 8. Key Features & Patterns

### Parallel Processing
- **Fan-out/Fan-in Pattern**: Query rewriter outputs are distributed to document retrieval and context summarization simultaneously
- **Async Execution**: All API calls are non-blocking using `asyncio.to_thread()`

### Pluggable Components
- **LLM Abstraction**: Easy switching between OpenAI, Ollama, or Finetuned models
- **Embedding Strategy**: Support for both dense (OpenAI) and sparse (SPLADE) embeddings
- **Vector DB Protocol**: Abstract interface for vector database operations

### Data Persistence
- **SQLite with WAL**: Write-Ahead Logging for concurrent reads
- **Session-based Conversations**: All interactions stored with session tracking
- **Indexed Queries**: Fast lookup by session_id

### Domain-Specific
- **Financial Compliance Domain**: Prompts and instructions tailored for compliance queries
- **Context-Aware Responses**: Uses conversation history for coherent multi-turn interactions

## 9. Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| **Web Framework** | FastAPI | 0.135.2+ |
| **Workflow Engine** | LangGraph | 1.1.3+ |
| **LLM Orchestration** | LangChain | 1.2.13+ |
| **Vector Database** | Pinecone | 8.1.0+ |
| **Embeddings** | Sentence Transformers | 5.3.0+ |
| **LLM APIs** | OpenAI | Latest |
| **Database** | SQLAlchemy + SQLite | Latest |
| **ASGI Server** | Uvicorn | 0.42.0+ |
| **Evaluation** | Ragas | 0.4.3+ |

## 10. Deployment Architecture

```mermaid
graph TB
    subgraph "Development/Production"
        direction LR
        FastAPI["FastAPI<br/>App<br/>uvicorn"]
    end
    
    subgraph "Cloud Services"
        direction LR
        OpenAI["OpenAI API<br/>gpt-4o-mini"]
        Pinecone["Pinecone Cloud<br/>Vector DB<br/>AWS us-east-1"]
    end
    
    subgraph "Local/Edge"
        direction LR
        SQLite["SQLite<br/>sample.db"]
        OllamaOpt["Ollama<br/>Optional"]
        FinetuneOpt["Finetuned Model<br/>Optional"]
    end
    
    FastAPI -->|API Calls| OpenAI
    FastAPI -->|Query/Insert| Pinecone
    FastAPI -->|Read/Write| SQLite
    FastAPI -.->|Optional| OllamaOpt
    FastAPI -.->|Optional| FinetuneOpt
    
    style FastAPI fill:#FFF3E0
    style OpenAI fill:#E3F2FD
    style Pinecone fill:#E3F2FD
    style SQLite fill:#F3E5F5
```

## Configuration & Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=sk_...          # Required for OpenAI
OPENAI_MODEL=gpt-4o-mini       # Active model

# Vector Database
PINECONE_API_KEY=...            # Required
PINECONE_INDEX=final-rag-index-openai-small
PINECONE_METRIC=dotproduct

# Database
DATABASE_URL=sqlite:///sample.db

# Embedding Models
DENSE_EMBEDDING=openai          # or sentence-transformers
SPARSE_EMBEDDING=splade         # SPLADE for sparse embeddings

# API Server
API_HOST=0.0.0.0
API_PORT=8000
```

## How to Read These Diagrams

1. **High-Level Architecture**: Start here to understand all components and how they connect
2. **Workflow State Machine**: See how data flows through the 5 processing nodes
3. **Data Flow Sequence**: Understand the step-by-step request/response cycle
4. **Layer Architecture**: Understand the separation of concerns and abstraction levels
5. **Dependencies**: See what external services are required and what's pluggable
