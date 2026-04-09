# Model Card

## 1. Model Overview

This model card describes the complete set of models used in the RAG Pipeline for SEBI Financial Compliance. The system employs multiple models across three categories: **embedding models** (for document retrieval), **generation models** (for response synthesis), and a **sparse lexical model** (for keyword matching). Each model serves a specific role in the pipeline.

---

## 2. System Model Summary

| Role | Selected Model | Type | Provider |
|------|---------------|------|----------|
| **Dense Embedding** | OpenAI `text-embedding-3-small` | Cloud API | OpenAI |
| **Sparse Embedding** | SPLADE (`naver/splade-cocondenser-ensembledistil`) | Local | HuggingFace / Naver |
| **Primary LLM** | OpenAI `gpt-4o-mini` | Cloud API | OpenAI |
| **Alternative LLM** | OpenAI `gpt-5-mini` | Cloud API | OpenAI |
| **Local LLM Fallback** | Ollama `llama3.2` | Local | Meta / Ollama |
| **Finetuned LLM** | Qwen-0.5B (finetuned) | Local | Alibaba / Custom |

---

## 3. Dense Embedding Model

### OpenAI `text-embedding-3-small` (Selected)

| Attribute | Value |
|-----------|-------|
| **Model Name** | `text-embedding-3-small` |
| **Provider** | OpenAI |
| **Type** | Dense embedding |
| **Dimensions** | 1536 |
| **Max Tokens** | 8191 |
| **Use Case** | Semantic similarity search for document retrieval |
| **Integration** | `workflow/embeddings/openai_embedding.py` |

#### Performance in This System

| Metric | Score |
|--------|-------|
| MRR | 0.6158 |
| nDCG@10 | 0.6933 |
| Recall@5 | 0.8378 |
| **Recall@10** | **0.9369** |
| Precision@5 | 0.1676 |

#### Why Selected
- Highest Recall@10 (0.9369) among all evaluated embedding models
- Best nDCG@10 (0.6933) — superior ranking quality
- Cost-effective compared to `text-embedding-3-large`
- 1536 dimensions provide good balance of expressiveness and efficiency

### Evaluated Alternatives

#### Sentence Transformers `all-MiniLM-L6-v2`

| Attribute | Value |
|-----------|-------|
| **Dimensions** | 384 |
| **Type** | Open-source, local inference |
| **Recall@10** | 0.8378 |
| **Trade-off** | Free but lower retrieval quality |
| **Implementation** | `workflow/embeddings/sentence_transformer_embedding.py` |

#### OpenAI `text-embedding-3-large`

| Attribute | Value |
|-----------|-------|
| **Dimensions** | 3072 |
| **Recall@10** | 0.8198 |
| **Observation** | Worse than `text-embedding-3-small` for this corpus size (~1,108 chunks) |
| **Analysis** | Higher dimensionality introduces noise in small corpora |

---

## 4. Sparse Embedding Model

### SPLADE (`naver/splade-cocondenser-ensembledistil`)

| Attribute | Value |
|-----------|-------|
| **Model Name** | `naver/splade-cocondenser-ensembledistil` |
| **Provider** | Naver (via HuggingFace) |
| **Type** | Sparse lexical embedding |
| **Output Format** | `{indices: [...], values: [...]}` (sparse vector) |
| **Use Case** | Keyword/term matching to complement dense embeddings |
| **Integration** | `workflow/embeddings/sparse_embedding.py` |

#### Role in Hybrid Search
- Captures exact term matches (e.g., "Regulation 27(2)", "SEBI Act 1992")
- Complements dense embeddings which focus on semantic similarity
- Combined with dense embeddings via Pinecone's hybrid search with dot product scoring

#### Technical Details
- Uses PyTorch for inference
- Outputs sparse tensors converted to Pinecone-compatible format
- Runs locally (no API calls)
- Provides lexical coverage for domain-specific regulatory terminology

---

## 5. Primary Generation Model

### OpenAI `gpt-4o-mini` (Production)

| Attribute | Value |
|-----------|-------|
| **Model Name** | `gpt-4o-mini` |
| **Provider** | OpenAI |
| **Type** | Large Language Model (Cloud API) |
| **Context Window** | 128K tokens |
| **Max Output** | 16K tokens |
| **Use Cases** | Query rewriting, context summarization, response generation |
| **Integration** | `workflow/llms/openai.py` |
| **Concurrency** | Semaphore (max 10 concurrent) |
| **Timeout** | 30 seconds |
| **Retries** | 2 |

#### Performance in This System (RAGAS)

| Metric | Score |
|--------|-------|
| **Faithfulness** | 0.8427 |
| **Answer Relevancy** | 0.9256 |

#### Three Roles in Pipeline

1. **Query Rewriter** (Node 1): Rewrites user queries for optimal retrieval using Financial Compliance terminology
2. **Context Summarizer** (Node 3): Compresses conversation history into concise summaries
3. **Response Generator** (Node 4): Synthesizes final answers from retrieved documents and conversation context

#### Why Selected as Primary
- Best cost-to-quality ratio among evaluated models
- Faithfulness (0.8427) is adequate for most compliance queries
- Faster and cheaper than gpt-5-mini
- Large context window accommodates extensive document passages

#### Limitations
- ~16% of generated claims may not be directly supported by source documents
- Cloud dependency — requires internet connectivity and API key
- Subject to OpenAI rate limits and pricing changes

---

## 6. Alternative Generation Model

### OpenAI `gpt-5-mini` (High-Accuracy Option)

| Attribute | Value |
|-----------|-------|
| **Model Name** | `gpt-5-mini` |
| **Provider** | OpenAI |
| **Type** | Large Language Model (Cloud API) |
| **Context Window** | 128K tokens |

#### Performance in This System (RAGAS)

| Metric | Score |
|--------|-------|
| **Faithfulness** | **0.8987** |
| **Answer Relevancy** | **0.9579** |

#### When to Use
- High-stakes compliance queries requiring maximum accuracy
- Cases where the ~5.6% faithfulness improvement over gpt-4o-mini justifies increased cost
- Quality-critical deployments where cost is secondary

---

## 7. Local LLM Fallback

### Ollama `llama3.2`

| Attribute | Value |
|-----------|-------|
| **Model Name** | `llama3.2` |
| **Provider** | Meta (via Ollama) |
| **Type** | Large Language Model (Local) |
| **Parameters** | ~8B |
| **Context Window** | 128K tokens |
| **Integration** | `workflow/llms/ollama_llama.py` |
| **Requirements** | Ollama server running locally |

#### Use Cases
- Offline environments without internet access
- Development and testing without API costs
- Synthetic query generation for evaluation (used in retriever evaluation)

#### Limitations
- Not formally evaluated with RAGAS in this system
- Requires local GPU for acceptable inference speed
- May have lower faithfulness than cloud models for domain-specific queries

---

## 8. Finetuned Model

### Finetuned Qwen-0.5B

| Attribute | Value |
|-----------|-------|
| **Base Model** | Qwen-0.5B |
| **Type** | Causal Language Model (Local, finetuned) |
| **Parameters** | 0.5 billion |
| **Max Tokens** | 512 |
| **Temperature** | 0.7 |
| **Weights Location** | `finetuned_model/` |
| **Integration** | `workflow/llms/finetuned_llm.py` |
| **Framework** | HuggingFace Transformers + `text-generation` pipeline |

#### Performance in This System (RAGAS)

| Metric | Score |
|--------|-------|
| **Faithfulness** | **0.2333** (Critical failure) |
| **Answer Relevancy** | 0.9561 |

#### Analysis
- **Severe hallucination problem**: Only 23.3% of claims are grounded in source documents
- **Paradoxically high relevancy**: Generates fluent, on-topic text that sounds authoritative but fabricates details
- **Root cause**: 0.5B parameters is insufficient for grounded generation in a specialized legal/regulatory domain

#### Model Files

```
finetuned_model/
├── config.json               # Model architecture configuration
├── generation_config.json    # Generation parameters
├── model.safetensors         # Model weights (safetensors format)
├── tokenizer.json            # Tokenizer vocabulary
├── tokenizer_config.json     # Tokenizer settings
└── chat_template.jinja       # Chat formatting template
```

#### Recommendation
**Do NOT use for production compliance applications.** The faithfulness score of 0.2333 means the model fabricates regulatory information in approximately 77% of its claims. This is dangerous in a compliance domain where users may act on fabricated information. Included as a baseline and for research purposes only.

---

## 9. Ethical Considerations

### Intended Use
- **Primary**: Assisting compliance professionals in finding information within SEBI regulations
- **Not Intended**: Replacing legal advice, making compliance determinations, or serving as a sole source of regulatory truth

### Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| **Hallucination** | System prompt instructs LLM to say "I don't know" when information isn't in documents; source documents returned for verification |
| **Outdated Information** | System operates on a static corpus; users should verify against current SEBI publications |
| **Over-reliance** | Source attribution encourages users to verify claims against original documents |
| **Bias** | Models may reflect biases in training data; compliance decisions should involve human judgment |
| **Confidentiality** | User queries are sent to OpenAI API; sensitive queries may be exposed |

### Transparency
- All responses include source document references (content + metadata)
- The system explicitly states "I don't know" when information is not found in the retrieved documents
- Users can verify any claim against the cited source document and page number

---

## 10. Model Comparison Summary

### Embedding Models

| Model | Dims | Recall@10 | nDCG@10 | Cost | Inference |
|-------|------|-----------|---------|------|-----------|
| **OpenAI Small** | 1536 | **0.9369** | **0.6933** | Per-token | Cloud API |
| Sentence Transformers | 384 | 0.8378 | 0.6645 | Free | Local |
| OpenAI Large | 3072 | 0.8198 | 0.6547 | Per-token | Cloud API |

### Generation Models

| Model | Faithfulness | Answer Relevancy | Cost | Inference |
|-------|-------------|-----------------|------|-----------|
| **gpt-5-mini** | **0.8987** | **0.9579** | Higher | Cloud API |
| **gpt-4o-mini** | 0.8427 | 0.9256 | Moderate | Cloud API |
| llama3.2 | Not evaluated | Not evaluated | Free | Local |
| Finetuned Qwen-0.5B | 0.2333 | 0.9561 | Free | Local |

---

## 11. Reproducibility

### Environment
- Python >= 3.12
- CUDA-compatible GPU recommended for local models (SPLADE, Finetuned)
- Dependencies specified in `pyproject.toml`

### Model Access
- OpenAI models: Via API key
- SPLADE: Auto-downloaded from HuggingFace on first use
- Sentence Transformers: Auto-downloaded from HuggingFace on first use
- Finetuned Qwen-0.5B: Included in `finetuned_model/` directory
- Ollama llama3.2: Pulled via `ollama pull llama3.2`
