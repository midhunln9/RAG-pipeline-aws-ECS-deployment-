# Evaluation Report

## 1. Executive Summary

This report presents the comprehensive evaluation of the RAG pipeline for SEBI Financial Compliance, covering both **retrieval quality** (how well the system finds relevant documents) and **generation quality** (how well the LLM synthesizes accurate, relevant responses). Evaluation was conducted using industry-standard frameworks: **ranx** for information retrieval metrics and **RAGAS** for generation quality assessment.

**Key Findings**:
- **Best Retriever**: OpenAI `text-embedding-3-small` with hybrid search achieves **0.9369 Recall@10** and **0.6933 nDCG@10**
- **Best Generator**: OpenAI `gpt-5-mini` achieves **0.8987 Faithfulness** and **0.9579 Answer Relevancy**
- **Production Model**: OpenAI `gpt-4o-mini` selected for optimal cost-quality balance (**0.8427 Faithfulness**, **0.9256 Answer Relevancy**)
- **Finetuned Model**: Qwen-0.5B shows critical faithfulness issues (**0.2333**) — unsuitable for compliance domain

---

## 2. Evaluation Framework

### 2.1 Two-Stage Evaluation

The RAG pipeline is evaluated in two independent stages:

```
Stage 1: Retriever Evaluation (Information Retrieval Metrics)
├── Does the system find the right documents?
├── How well are they ranked?
└── Tools: ranx library, synthetic queries

Stage 2: Generation Evaluation (RAGAS Metrics)
├── Are responses grounded in retrieved documents?
├── Do responses address the user's question?
└── Tools: RAGAS framework, 30 curated questions
```

### 2.2 Why Two-Stage?

Evaluating retrieval and generation separately allows:
1. **Isolating failure points**: Poor answers could stem from bad retrieval OR bad generation
2. **Independent optimization**: Embedding models and LLMs can be tuned separately
3. **Fair comparison**: Same retrieved context when comparing LLMs; same LLM when comparing embeddings

---

## 3. Retriever Evaluation

### 3.1 Methodology

#### Dataset Construction

1. **Corpus**: All 1,108 document chunks from the ingested SEBI Master Circulars
2. **Sampling**: 10% random sample (111 chunks) for evaluation efficiency
3. **Query Generation**: Ollama `llama3` generated one synthetic query per sampled chunk
4. **Ground Truth (qrels)**: Each generated query is mapped to its source chunk as the relevant document
5. **Search Mode**: Hybrid search (dense + sparse embeddings) via Pinecone

#### Evaluation Process

For each embedding strategy:
1. Embed all chunks and store in Pinecone
2. For each synthetic query, run hybrid search and retrieve top-K results
3. Compare retrieved results against ground truth relevance judgments
4. Compute metrics using the `ranx` library

### 3.2 Metrics Definitions

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **MRR** (Mean Reciprocal Rank) | Average of 1/rank of the first relevant document across all queries | How quickly users find relevant information |
| **nDCG@10** (Normalized Discounted Cumulative Gain at 10) | Measures ranking quality of top-10 results, penalizing relevant docs at lower ranks | Overall retrieval quality |
| **Recall@5** | Fraction of relevant documents found in top-5 results | Coverage in a short list |
| **Recall@10** | Fraction of relevant documents found in top-10 results | Coverage in the context window passed to LLM |
| **Precision@5** | Fraction of top-5 results that are relevant | Signal-to-noise ratio |

### 3.3 Results

#### Hard Retriever Evaluation (Hybrid Search, No Reranker)

| Metric | Sentence Transformers | OpenAI Small | OpenAI Large |
|--------|----------------------|--------------|--------------|
| **MRR** | 0.6078 | **0.6158** | 0.6012 |
| **nDCG@10** | 0.6645 | **0.6933** | 0.6547 |
| **Recall@5** | 0.8108 | **0.8378** | 0.7568 |
| **Recall@10** | 0.8378 | **0.9369** | 0.8198 |
| **Precision@5** | 0.1622 | **0.1676** | 0.1514 |

### 3.4 Retriever Analysis

#### OpenAI `text-embedding-3-small` (Winner)
- **Recall@10 = 0.9369**: The most critical metric for RAG — 93.7% of relevant documents are found in the top-10 results passed to the LLM. This means the LLM almost always has access to the information needed to answer correctly.
- **nDCG@10 = 0.6933**: Best ranking quality — relevant documents tend to appear higher in the result list.
- **MRR = 0.6158**: On average, the first relevant document appears around rank 1.6 (1/0.6158), meaning users typically see relevant content immediately.

#### Sentence Transformers `all-MiniLM-L6-v2`
- **Recall@10 = 0.8378**: Decent but 10 percentage points below OpenAI Small.
- **Advantage**: Free, local inference — no API calls needed.
- **Trade-off**: Lower retrieval quality vs. zero marginal cost.

#### OpenAI `text-embedding-3-large`
- **Recall@10 = 0.8198**: Surprisingly the weakest performer.
- **Analysis**: Higher dimensionality (3072 vs 1536) likely introduces noise for this corpus size (~1,108 chunks). The "curse of dimensionality" — with a relatively small corpus, the additional dimensions don't capture meaningful distinctions and instead spread relevant documents further apart in the embedding space.
- **Conclusion**: Bigger is not always better; model size should match corpus characteristics.

#### Why Precision@5 Is Low (~0.16)
- Each query has exactly 1 relevant document in ground truth (the source chunk)
- Precision@5 = 1/5 = 0.20 is the theoretical maximum
- Achieving 0.1676 means the relevant document appears in top-5 for ~84% of queries

---

## 4. Generation Evaluation

### 4.1 Methodology

#### Evaluation Framework: RAGAS

RAGAS (Retrieval Augmented Generation Assessment) uses LLM-as-judge to evaluate RAG system outputs.

#### Question Set

30 manually curated domain-specific questions covering:
- IPO eligibility and listing requirements
- Disclosure norms and compliance obligations
- Book-building processes and pricing
- Promoter obligations and lock-in periods
- Market surveillance mechanisms
- SEBI regulatory procedures

Example questions:
- "What are the eligibility criteria for companies planning an IPO under SEBI guidelines?"
- "Explain the lock-in requirements for promoter holdings"
- "What are the disclosure requirements for preferential allotment?"

#### Evaluation Process

For each LLM model:
1. Run the full RAG workflow (query rewriting → retrieval → context summary → generation) for all 30 questions
2. Collect: user query, generated response, retrieved contexts
3. Construct RAGAS `EvaluationDataset`
4. Evaluate using Faithfulness and Answer Relevancy metrics

### 4.2 Metrics Definitions

| Metric | Definition | Scale | What It Measures |
|--------|-----------|-------|-----------------|
| **Faithfulness** | Fraction of claims in the response that are supported by the retrieved context | 0.0 - 1.0 | Hallucination resistance; grounding in source documents |
| **Answer Relevancy** | How well the response addresses the user's original question | 0.0 - 1.0 | Response utility and directness |

### 4.3 Results

#### RAGAS Metrics (OpenAI Small Embeddings + Various LLMs)

| Model | Faithfulness | Answer Relevancy |
|-------|-------------|-----------------|
| **gpt-5-mini** | **0.8987** | **0.9579** |
| **gpt-4o-mini** | 0.8427 | 0.9256 |
| **finetuned-qwen-0.5B** | 0.2333 | 0.9561 |

### 4.4 Generation Analysis

#### gpt-5-mini (Highest Quality)

- **Faithfulness = 0.8987**: ~90% of claims are grounded in source documents. The remaining ~10% may include reasonable inferences or minor extrapolations.
- **Answer Relevancy = 0.9579**: Nearly all responses directly address the question asked.
- **Best For**: High-stakes compliance queries where accuracy is paramount.
- **Trade-off**: Higher API cost per token.

#### gpt-4o-mini (Production Choice)

- **Faithfulness = 0.8427**: ~84% of claims grounded in sources. Slightly more prone to adding contextual information beyond the retrieved documents.
- **Answer Relevancy = 0.9256**: Consistently relevant and useful responses.
- **Best For**: Day-to-day compliance queries with good cost efficiency.
- **Why Selected**: Best cost-to-quality ratio. The ~5.6% faithfulness gap vs gpt-5-mini is an acceptable trade-off for significantly lower cost and faster inference.

#### finetuned-qwen-0.5B (Not Recommended)

- **Faithfulness = 0.2333**: Critical failure — only ~23% of claims are grounded in source documents. The model hallucinations extensively.
- **Answer Relevancy = 0.9561**: Paradoxically high — the model produces fluent, on-topic text that sounds authoritative but fabricates regulatory details.
- **Root Cause**: 0.5B parameters is insufficient for the complex task of grounded generation in a specialized legal/regulatory domain. The model has learned to generate compliance-sounding text but cannot reliably constrain itself to the provided context.
- **Risk**: In a compliance domain, this level of hallucination is dangerous — users might act on fabricated regulatory information.
- **Conclusion**: **Do not use for production.** Serves as a baseline demonstrating that local finetuned models require significantly more parameters for this task.

---

## 5. End-to-End System Performance

### 5.1 Selected Configuration

| Component | Selection | Key Metric |
|-----------|-----------|-----------|
| Dense Embedding | OpenAI `text-embedding-3-small` | Recall@10: 0.9369 |
| Sparse Embedding | SPLADE | Hybrid search complement |
| Primary LLM | OpenAI `gpt-4o-mini` | Faithfulness: 0.8427 |
| Search Mode | Hybrid (dense + sparse) | Combined ranking |
| Top-K Documents | 10 | Passed to LLM context |

### 5.2 System Characteristics

| Characteristic | Value |
|---------------|-------|
| Max Concurrent LLM Calls | 10 (semaphore) |
| Vector Batch Size | 200 |
| Conversation History | Last 10 conversations |
| Document Retrieval | Top 10 per query |
| Database Mode | SQLite WAL |

### 5.3 Load Testing

Load testing was conducted using Locust with:
- 8 predefined compliance queries
- Configurable concurrent users
- 1-3 second think time between requests
- Unique session IDs per simulated user

---

## 6. Evaluation Notebooks

| Notebook | Location | Purpose |
|----------|----------|---------|
| `hard_evals1.ipynb` | `notebooks/` | Sentence Transformers retriever evaluation |
| `hard_evals2.ipynb` | `notebooks/` | OpenAI Small retriever evaluation |
| `hard_evals3.ipynb` | `notebooks/` | OpenAI Large retriever evaluation |
| `ragas_gpt_4o_mini.ipynb` | `rag_pipeline/ragas_eval/` | RAGAS evaluation with gpt-4o-mini |
| `ragas_gpt_5_mini.ipynb` | `rag_pipeline/ragas_eval/` | RAGAS evaluation with gpt-5-mini |
| `finetuned_llm.ipynb` | `rag_pipeline/ragas_eval/` | RAGAS evaluation with finetuned Qwen-0.5B |

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Synthetic Query Evaluation**: Retriever evaluation uses synthetically generated queries, which may not perfectly represent real user queries
2. **Single-Annotator Ground Truth**: Each query has exactly one relevant document; real queries may have multiple relevant chunks
3. **No Human Evaluation**: All metrics are automated; human judgment may differ
4. **Fixed Question Set**: RAGAS evaluation uses 30 curated questions; a larger, more diverse set would increase confidence
5. **No Latency Metrics**: Evaluation focuses on quality, not response time

### 7.2 Future Improvements

1. **Reranking**: Add a cross-encoder reranker after initial retrieval to improve precision
2. **Human Evaluation**: Conduct user studies with compliance professionals
3. **Larger Evaluation Sets**: Expand to 100+ questions covering edge cases
4. **Context Relevancy**: Add RAGAS `ContextRelevancy` metric to evaluate retrieval quality within the RAGAS framework
5. **Answer Correctness**: Add ground-truth reference answers for exact correctness measurement
6. **A/B Testing**: Compare system versions in production with real user queries
7. **Continuous Evaluation**: Automated evaluation pipeline triggered on model or data updates

---

## 8. Reproducibility

### 8.1 Environment

- Python >= 3.12
- Dependencies: See `pyproject.toml`
- Evaluation frameworks: `ragas>=0.4.3`, `ranx>=0.3.21`

### 8.2 Data

- Retriever evaluation datasets: `notebooks/datasets/{openai_small,openai_large,sentence_transformers}/all_chunks.csv`
- RAGAS questions: Embedded in evaluation notebooks

### 8.3 Running Evaluations

```bash
# Retriever evaluations (Jupyter notebooks)
jupyter notebook notebooks/hard_evals1.ipynb   # Sentence Transformers
jupyter notebook notebooks/hard_evals2.ipynb   # OpenAI Small
jupyter notebook notebooks/hard_evals3.ipynb   # OpenAI Large

# RAGAS evaluations
jupyter notebook rag_pipeline/ragas_eval/ragas_gpt_4o_mini.ipynb
jupyter notebook rag_pipeline/ragas_eval/ragas_gpt_5_mini.ipynb
jupyter notebook rag_pipeline/ragas_eval/finetuned_llm.ipynb
```

### 8.4 Required API Keys

```bash
OPENAI_API_KEY=...     # For OpenAI embeddings and LLMs
PINECONE_API_KEY=...   # For vector database queries
```
