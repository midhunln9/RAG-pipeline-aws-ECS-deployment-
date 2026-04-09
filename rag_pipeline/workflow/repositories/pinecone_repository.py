"""
Pinecone vector database repository.

Provides integration with Pinecone for hybrid semantic and lexical search.
"""

from typing import Optional

from langchain_core.documents import Document
from pinecone import Pinecone

from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig
from rag_pipeline.workflow.protocols.vector_db_protocol import VectorDBProtocol
from rag_pipeline.workflow.strategies.dense_embedding_strategy import (
    DenseEmbeddingStrategy,
)
from rag_pipeline.workflow.strategies.sparse_embedding_strategy import (
    SparseEmbeddingStrategy,
)


class PineconeRepository(VectorDBProtocol):
    """
    Vector database repository using Pinecone.
    
    Implements hybrid search combining dense semantic embeddings
    with sparse lexical embeddings.
    """

    def __init__(
        self,
        api_key: str,
        pinecone_config: PineconeConfig,
        dense_embedding_strategy: DenseEmbeddingStrategy,
        sparse_embedding_strategy: SparseEmbeddingStrategy,
        environment: Optional[str] = None,
    ):
        """
        Initialize Pinecone repository.
        
        Args:
            api_key: Pinecone API key.
            pinecone_config: Pinecone configuration.
            dense_embedding_strategy: Strategy for dense embeddings.
            sparse_embedding_strategy: Strategy for sparse embeddings.
            environment: Optional environment name (kept for backward compatibility).
        """
        self.api_key = api_key
        self.pinecone_config = pinecone_config
        self.dense_embedding_strategy = dense_embedding_strategy
        self.sparse_embedding_strategy = sparse_embedding_strategy
        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(pinecone_config.index_name)

    def query(self, query: str, top_k: int = 10) -> list[Document]:
        """
        Query the vector database for documents.

        Performs hybrid search using both dense and sparse embeddings.

        Args:
            query: Query text.
            top_k: Number of results to retrieve (default: 10).

        Returns:
            List of relevant documents.
        """

        # Generate embeddings
        query_vector = self.dense_embedding_strategy.embed_query(query)
        sparse_embedding = self.sparse_embedding_strategy.embed_query(query)

        # Query Pinecone with hybrid search
        results = self.index.query(
            vector=query_vector,
            sparse_vector=sparse_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        # Convert results to Document objects
        documents = [
            Document(page_content=match.metadata.get("text", match.id), metadata=match.metadata or {})
            for match in results.matches
        ]
        return documents
