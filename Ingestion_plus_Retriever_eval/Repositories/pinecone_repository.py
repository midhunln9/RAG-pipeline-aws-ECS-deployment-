from typing import List
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from strategy.dense_embedding_strategy import DenseEmbeddingStrategy
from strategy.sparse_embedding_strategy import SparseEmbeddingStrategy
from configs.pinecone_config import PineconeConfig
from Protocols.vector_db_protocol import VectorDBProtocol
import pandas as pd
import logging
from typing import List, Dict


class PineconeRepository(VectorDBProtocol):
    def __init__(self, api_key: str, environment: str, dense_embedding_strategy: DenseEmbeddingStrategy,
    sparse_embedding_strategy: SparseEmbeddingStrategy, pinecone_config: PineconeConfig):
        self.api_key = api_key
        self.environment = environment
        self.client = Pinecone(api_key=self.api_key)
        self.dense_embedding_strategy = dense_embedding_strategy
        self.sparse_embedding_strategy = sparse_embedding_strategy
        self.pinecone_config = pinecone_config
        self.logger = logging.getLogger(__name__)
        self.index_created = False

    def check_index_exists(self) -> bool:
        try:
            indexes = self.client.list_indexes().names()
            self.logger.debug(f"Available Pinecone indexes: {indexes}")
            if self.pinecone_config.index_name in indexes:
                self.logger.info(f"Index '{self.pinecone_config.index_name}' already exists")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking if index exists: {e}")
            return False

    def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists, creating it if necessary. Called once before processing batches."""
        if self.index_created:
            return
        
        if not self.check_index_exists():
            try:
                self.logger.info(f"Creating Pinecone index with name: '{self.pinecone_config.index_name}'")
                self.logger.info(f"Index dimensions: {self.dense_embedding_strategy.get_sentence_embedding_dimension()}, metric: {self.pinecone_config.metric}")
                
                self.client.create_index(
                    name = self.pinecone_config.index_name,
                    dimension = self.dense_embedding_strategy.get_sentence_embedding_dimension(),
                    metric = self.pinecone_config.metric,
                    spec = ServerlessSpec(cloud = self.pinecone_config.cloud, region = self.pinecone_config.region)
                )
                self.logger.info(f"Successfully created index '{self.pinecone_config.index_name}'")
            except Exception as e:
                self.logger.error(f"Error creating index: {e}")
                raise
        
        self.index_created = True

    def upsert_chunks(self, chunks: List[Document]):
        """
        Ingest document chunks into Pinecone index in batches.
        
        Process:
        1. Ensure index exists (one-time check before all batches)
        2. Get index reference (reused for all batches)
        3. For each batch:
           - Generate dense and sparse embeddings
           - Upsert vectors with metadata
           - Collect records for evaluation dataset
        """
        dfs = []
        
        # Ensure index exists exactly once before processing any batches
        self._ensure_index_exists()
        # Get index reference once and reuse for all batch upserts
        index = self.client.Index(self.pinecone_config.index_name)
        
        for i in range(0, len(chunks), self.pinecone_config.batch_size):
            # Calculate batch number correctly using actual batch_size, not hardcoded value
            batch_number = i // self.pinecone_config.batch_size + 1
            index_end = i + self.pinecone_config.batch_size
            if index_end > len(chunks):
                index_end = len(chunks)
            chunk_batch = chunks[i:index_end]
            # Create unique IDs combining source, page, and position in batch
            ids = [f"{chunk.metadata['source']}_{chunk.metadata['page']}_{j}" for j, chunk in enumerate(chunk_batch)]
            
            self.logger.info(f"Generating dense embeddings for batch {batch_number}")
            dense_embeddings = self.dense_embedding_strategy.get_embeddings(chunk_batch)
            
            self.logger.info(f"Generating sparse embeddings for batch {batch_number}")
            sparse_embeddings = self.sparse_embedding_strategy.embed_documents([chunk.page_content for chunk in chunk_batch])
            
            # Prepare metadata with source, page number, and original text for retrieval
            metadata = [{"source" : chunk.metadata["source"], "page" : chunk.metadata["page"], "text" : chunk.page_content} for chunk in chunk_batch]

            # Combine dense and sparse embeddings with metadata for upsert
            dict_vector_chunks = [{"id" : vec_id, "values" : dense_embedding, "sparse_values" : sparse_embedding, "metadata" : metadata_chunk} for vec_id, dense_embedding, sparse_embedding, metadata_chunk in zip(ids, dense_embeddings, sparse_embeddings, metadata)]

            self.logger.info(f"Upserting {len(dict_vector_chunks)} vectors to index")
            index.upsert(
                vectors=dict_vector_chunks
            )

            # Accumulate records for evaluation dataset (single write at end)
            for chunk, chunk_id in zip(chunk_batch, ids):
                row = {"id": chunk_id, "text": chunk.page_content}
                dfs.append(row)
        
        # Write accumulated evaluation dataset after all batches are processed
        self.logger.info(f"Creating evaluation dataset with {len(dfs)} records")
        final_eval_df = pd.DataFrame(dfs)
        final_eval_df.to_csv("evaluation_dataset.csv", index = False)
        self.logger.info("Evaluation dataset saved to evaluation_dataset.csv")

    def query_vector_store_for_rankx(self, query: str) -> List[Dict]:
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

        # Return RankX-compatible format
        return [
            {
                "id": match.id,
                "score": float(match.score)  # ensure JSON serializable
            }
            for match in results.matches
        ]
                
            
            
            

            








