# Modeling Report

## 1. Executive Summary

This report details the modeling decisions, experiments, and results for the RAG pipeline targeting SEBI Financial Compliance. The system involves two categories of models: **retrieval models** (embedding strategies for document search) and **generation models** (LLMs for response synthesis). Through systematic evaluation, we selected **OpenAI `text-embedding-3-small`** for dense embeddings and **OpenAI `gpt-4o-mini`** as the primary generation LLM, with **SPLADE** for sparse embeddings in a hybrid search configuration.

---

## 2. Retrieval Modeling

### 2.1 Embedding Strategy Selection

The retrieval component converts both documents and queries into vector representations for similarity search. We evaluated three dense embedding models and one sparse model.

#### Dense Embedding Candidates

| Model | Dimensions | Type | Source |
|-------|-----------|------|--------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Open-source, local | HuggingFace |
| OpenAI `text-embedding-3-small` | 1536 | Cloud API | OpenAI |
| OpenAI `text-embedding-3-large` | 3072 | Cloud API | OpenAI |

#### Sparse Embedding Model

| Model | Type | Purpose |
|-------|------|---------|
| `naver/splade-cocondenser-ensembledistil` (SPLADE) | Sparse lexical | Keyword/term matching to complement dense retrieval |

### 2.2 Retriever Evaluation Methodology

**Evaluation Framework**: Hard retriever evaluation using the `ranx` library

**Process**:
1. Loaded all 1,108 document chunks from the ingested corpus
2. Sampled 10% (111 chunks) for evaluation
3. Used Ollama `llama3` to generate synthetic queries from each sampled chunk
4. Created ground truth relevance judgments (qrels): each query mapped to its source chunk
5. Ran hybrid search (dense + sparse) for each query
6. Computed standard IR metrics

**Metrics**:
- **MRR (Mean Reciprocal Rank)**: Average reciprocal of the rank of the first relevant document
- **nDCG@10 (Normalized Discounted Cumulative Gain)**: Measures ranking quality of top-10 results
- **Recall@5 / Recall@10**: Fraction of relevant documents found in top-5 / top-10 results
- **Precision@5**: Fraction of top-5 results that are relevant

### 2.3 Retriever Evaluation Results

| Metric | Sentence Transformers | OpenAI Small | OpenAI Large |
|--------|----------------------|--------------|--------------|
| **MRR** | 0.6078 | **0.6158** | 0.6012 |
| **nDCG@10** | 0.6645 | **0.6933** | 0.6547 |
| **Recall@5** | 0.8108 | **0.8378** | 0.7568 |
| **Recall@10** | 0.8378 | **0.9369** | 0.8198 |
| **Precision@5** | 0.1622 | **0.1676** | 0.1514 |

### 2.4 Retriever Analysis and Selection

**Selected Model: OpenAI `text-embedding-3-small`**

**Rationale**:
1. **Best Recall@10 (0.9369)**: Significantly outperforms both alternatives. This is the most critical metric for RAG — ensuring relevant documents are retrieved in the top-10 results that will be passed to the LLM.
2. **Best nDCG@10 (0.6933)**: Not only retrieves relevant documents but ranks them higher.
3. **Best MRR (0.6158)**: The first relevant document appears at a higher rank on average.
4. **Surprising**: OpenAI Large (3072 dims) performed worse than OpenAI Small (1536 dims), likely due to the relatively small corpus size where higher dimensionality introduces noise rather than capturing additional semantic nuance.
5. **Cost-Effective**: Cheaper per token than OpenAI Large while delivering better performance.

### 2.5 Hybrid Search Strategy

The selected retrieval approach combines dense and sparse embeddings:

```
Query → [Dense Embedding (OpenAI Small)] → Semantic similarity scores
      → [Sparse Embedding (SPLADE)]      → Lexical match scores
      → [Rank Fusion]                     → Combined top-10 results
```

**Why Hybrid Search?**
- **Dense embeddings** excel at semantic understanding (e.g., "IPO requirements" matches "Initial Public Offering eligibility criteria")
- **Sparse embeddings (SPLADE)** excel at exact term matching (e.g., "Regulation 27(2)" matches documents containing that exact reference)
- Combined, they achieve the best retrieval performance across both semantic and lexical queries

**Pinecone Configuration**:
- Metric: Dot product (compatible with both dense and sparse vectors)
- Index: `final-rag-index-openai-small`
- Cloud: AWS us-east-1

---

## 3. Generation Modeling

### 3.1 LLM Candidates

Three LLM providers were evaluated for response generation:

| Model | Type | Parameters | Context Window | Provider |
|-------|------|-----------|---------------|----------|
| OpenAI `gpt-4o-mini` | Cloud API | Proprietary | 128K tokens | OpenAI |
| OpenAI `gpt-5-mini` | Cloud API | Proprietary | 128K tokens | OpenAI |
| Finetuned `Qwen-0.5B` | Local | 0.5B | Limited | HuggingFace (finetuned) |
| Ollama `llama3.2` | Local | ~8B | 128K tokens | Ollama (not formally evaluated) |

### 3.2 Generation Evaluation Methodology

**Evaluation Framework**: RAGAS (Retrieval Augmented Generation Assessment)

**Process**:
1. Curated 30 domain-specific questions covering various SEBI regulatory topics
2. Ran the full RAG workflow for each question using each LLM
3. Collected responses and retrieved contexts
4. Evaluated using RAGAS metrics with an LLM-as-judge approach

**Metrics**:
- **Faithfulness**: Measures whether the generated response is grounded in the retrieved documents (no hallucination). Score 0-1.
- **Answer Relevancy**: Measures whether the generated response actually addresses the user's question. Score 0-1.

### 3.3 Generation Evaluation Results

| Model | Faithfulness | Answer Relevancy |
|-------|-------------|-----------------|
| **gpt-5-mini** | **0.8987** | **0.9579** |
| **gpt-4o-mini** | 0.8427 | 0.9256 |
| **finetuned-qwen-0.5B** | 0.2333 | 0.9561 |

### 3.4 Generation Analysis

#### gpt-5-mini
- **Highest Faithfulness (0.8987)**: Most grounded in source documents; least prone to hallucination
- **Highest Answer Relevancy (0.9579)**: Most directly addresses user questions
- **Trade-off**: Higher cost per token compared to gpt-4o-mini

#### gpt-4o-mini (Selected as Primary)
- **Strong Faithfulness (0.8427)**: Acceptably grounded in source documents
- **Strong Answer Relevancy (0.9256)**: Consistently relevant responses
- **Advantage**: Lower cost, faster inference, good balance of quality and efficiency
- **Selected Reason**: Best cost-to-quality ratio for production deployment

#### finetuned-qwen-0.5B
- **Low Faithfulness (0.2333)**: Severe hallucination problem — generates plausible but ungrounded answers
- **High Answer Relevancy (0.9561)**: Paradoxically, responses sound relevant but are not faithful to sources
- **Analysis**: The 0.5B parameter count is insufficient for grounded generation in a specialized domain. The model generates fluent text that appears relevant but fabricates regulatory details not present in the retrieved documents.
- **Conclusion**: Not suitable for production use in compliance domain where accuracy is critical

#### Ollama llama3.2
- Not formally evaluated with RAGAS but available as a local fallback option
- Used for synthetic query generation during retriever evaluation

### 3.5 LLM Architecture in the Pipeline

The selected LLM (gpt-4o-mini) is used in three distinct roles within the pipeline:

| Role | Node | Prompt | Purpose |
|------|------|--------|---------|
| **Query Rewriter** | Node 1 | `QUERY_REWRITER_PROMPT` | Optimizes user queries for better retrieval |
| **Context Summarizer** | Node 3 | `SUMMARY_SO_FAR` | Compresses conversation history into concise context |
| **Response Generator** | Node 4 | `RAG_SYSTEM_PROMPT` + `RAG_USER_PROMPT` | Synthesizes final response from documents and context |

### 3.6 Prompt Engineering

#### Query Rewriter Prompt
```
Rewrites queries to be:
- Specific and concise
- Accurate to original intent
- Aligned with Financial Compliance terminology
- Without adding new facts
```

#### RAG System Prompt
```
You are a helpful Financial Compliance assistant.
- Use only the provided documents and conversation summary
- If the answer is not in the documents, say "I don't know"
- Ground all responses in source material
```

#### Context Summary Prompt
```
Summarize the conversation history so far.
- Output only the summary
- Condense multiple turns into a coherent narrative
```

### 3.7 Concurrency and Rate Limiting

| Parameter | Value | Purpose |
|-----------|-------|---------|
| LLM Semaphore | Max 10 concurrent | Prevents API rate limit errors |
| Timeout | 30 seconds | Per LLM call |
| Retries | 2 | For transient failures |

---

## 4. Model Selection Summary

| Component | Selected Model | Rationale |
|-----------|---------------|-----------|
| **Dense Embedding** | OpenAI `text-embedding-3-small` | Best Recall@10 (0.9369), best nDCG@10 (0.6933) |
| **Sparse Embedding** | SPLADE (`naver/splade-cocondenser-ensembledistil`) | Lexical matching complement to dense embeddings |
| **Primary LLM** | OpenAI `gpt-4o-mini` | Best cost-to-quality ratio (Faithfulness: 0.8427, Relevancy: 0.9256) |
| **Alternative LLM** | OpenAI `gpt-5-mini` | Higher quality (Faithfulness: 0.8987) for use cases requiring maximum accuracy |
| **Local Fallback** | Ollama `llama3.2` | Offline capability, no API dependency |

---

## 5. Text Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Strategy** | RecursiveCharacterTextSplitter | Respects document structure (paragraphs, sentences) |
| **Chunk Size** | 1000 characters | Sufficient context per chunk for regulatory text |
| **Chunk Overlap** | 200 characters | Preserves context at boundaries |
| **Separators** | `\n\n`, `\n`, ` `, `""` (recursive) | Hierarchical splitting respects document formatting |
| **Total Chunks** | ~1,108 | From 3 SEBI Master Circulars |

---

## 6. Future Modeling Considerations

1. **Reranking**: Adding a cross-encoder reranker (e.g., Cohere Rerank, BGE Reranker) after initial retrieval could further improve precision
2. **Larger Finetuned Models**: A 7B+ parameter finetuned model may achieve better faithfulness than the 0.5B Qwen model
3. **Embedding Fine-tuning**: Domain-specific fine-tuning of embedding models on financial compliance text could improve retrieval
4. **Multi-vector Retrieval**: ColBERT-style late interaction models for more nuanced matching
5. **Chunk Size Optimization**: Systematic evaluation of different chunk sizes (500, 750, 1000, 1500) to find the optimal balance
