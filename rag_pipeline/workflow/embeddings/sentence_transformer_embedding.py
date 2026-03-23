from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline.workflow.strategies.dense_embedding_strategy import DenseEmbeddingStrategy
from typing import List
from langchain_core.documents import Document
from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig

class SentenceTransformerEmbedding(DenseEmbeddingStrategy):
    def __init__(self, pinecone_config : PineconeConfig):
        self.model = HuggingFaceEmbeddings(model_name = pinecone_config.dense_embedding_model_name)
    def get_sentence_embedding_dimension(self) -> int:
        embedding_dim = len(self.model.embed_query("This is a test query to get embedding dimension"))
        return embedding_dim
    def get_embeddings(self, documents : List[Document]) -> List[List[float]]:
        return self.model.embed_documents([doc.page_content for doc in documents])
    def embed_query(self, query : str) -> List[float]:
        return self.model.embed_query(query)