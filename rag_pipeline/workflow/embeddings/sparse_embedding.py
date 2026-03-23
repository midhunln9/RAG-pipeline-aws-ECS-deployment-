from rag_pipeline.workflow.strategies.sparse_embedding_strategy import SparseEmbeddingStrategy
from sentence_transformers import SparseEncoder
from typing import List
import torch
from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig


class SentenceTransformerSparseEmbedding(SparseEmbeddingStrategy):
    def __init__(self, pinecone_config : PineconeConfig):
        self.model = SparseEncoder(pinecone_config.sparse_embedding_model_name)

    def _sparse_tensor_to_pinecone_dict(self, sparse_tensor: torch.Tensor) -> dict:
        dense = sparse_tensor.to_dense().cpu()
        non_zero = torch.nonzero(dense).squeeze(1)
        return {
            "indices": non_zero.tolist(),
            "values": dense[non_zero].tolist()
        }

    def embed_documents(self, documents : List[str]) -> List[dict]:
        embeddings = self.model.encode(documents)
        return [self._sparse_tensor_to_pinecone_dict(embedding) for embedding in embeddings]

    def embed_query(self, query : str) -> dict:
        embedding = self.model.encode([query])[0]
        return self._sparse_tensor_to_pinecone_dict(embedding)