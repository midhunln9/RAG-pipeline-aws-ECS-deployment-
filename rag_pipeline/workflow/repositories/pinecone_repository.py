from typing import List
from langchain_core.documents import Document
from pinecone import Pinecone
from rag_pipeline.workflow.protocols.vector_db_protocol import VectorDBProtocol
from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig
from rag_pipeline.workflow.strategies.dense_embedding_strategy import DenseEmbeddingStrategy
from rag_pipeline.workflow.strategies.sparse_embedding_strategy import SparseEmbeddingStrategy

class PineconeRepository(VectorDBProtocol):
    def __init__(self, api_key: str, environment: str, pinecone_config: PineconeConfig, 
    dense_embedding_strategy: DenseEmbeddingStrategy, sparse_embedding_strategy: SparseEmbeddingStrategy):
        self.api_key = api_key
        self.environment = environment
        self.client = Pinecone(api_key=api_key)
        self.pinecone_config = pinecone_config
        self.dense_embedding_strategy = dense_embedding_strategy
        self.sparse_embedding_strategy = sparse_embedding_strategy

    def query(self, query: str) -> List[Document]:
        index = self.client.Index(self.pinecone_config.index_name)

        # Generate embeddings
        query_vector = self.dense_embedding_strategy.embed_query(query)
        sparse_embedding = self.sparse_embedding_strategy.embed_query(query)

        # Query Pinecone
        results = index.query(
            vector=query_vector,
            sparse_vector=sparse_embedding,
            top_k=10,
            include_metadata=False  # not needed for RankX
        )
        return [Document(page_content=match.id, metadata=match.metadata) for match in results.matches]
